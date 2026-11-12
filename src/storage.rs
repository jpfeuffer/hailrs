use crate::{invalid, Result};
use bytes::Bytes;
use object_store::{
    aws::AmazonS3Builder, gcp::GoogleCloudStorageBuilder, local::LocalFileSystem, path::Path,
    ClientOptions, ObjectStore,
};
use serde::{Deserialize, Serialize};
use std::{
    ops::Range,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use url::Url;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageOptions {
    /// Do not sign requests (public buckets only).
    pub anonymous: bool,
    pub region: Option<String>,
    /// GCS billing project for requester-pays buckets.
    pub requester_pays_project: Option<String>,
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ReadStats {
    pub requests: u64,
    pub bytes: u64,
}

#[derive(Clone)]
pub struct Storage {
    inner: Arc<dyn ObjectStore>,
    root: String,
    scope: String,
    requests: Arc<AtomicU64>,
    bytes: Arc<AtomicU64>,
}

impl Storage {
    pub fn open(uri: &str, options: &StorageOptions) -> Result<Self> {
        let (inner, root): (Arc<dyn ObjectStore>, String) =
            if uri.starts_with("s3://") || uri.starts_with("gs://") || uri.starts_with("gcs://") {
                let url = Url::parse(uri)?;
                let bucket = url.host_str().ok_or_else(|| invalid("missing bucket"))?;
                let store: Arc<dyn ObjectStore> = if url.scheme() == "s3" {
                    let mut builder = AmazonS3Builder::from_env()
                        .with_bucket_name(bucket)
                        .with_skip_signature(options.anonymous);
                    if let Some(region) = &options.region {
                        builder = builder.with_region(region);
                    }
                    if let Some(endpoint) = &options.endpoint {
                        builder = builder
                            .with_endpoint(endpoint)
                            .with_allow_http(endpoint.starts_with("http://"));
                    }

                    Arc::new(builder.build()?)
                } else {
                    let mut builder = GoogleCloudStorageBuilder::from_env()
                        .with_bucket_name(bucket)
                        .with_skip_signature(options.anonymous);
                    if let Some(project) = &options.requester_pays_project {
                        builder = builder.with_client_options(
                            ClientOptions::new().with_default_headers(
                                [(
                                    "x-goog-user-project"
                                        .parse()
                                        .map_err(|_| invalid("header name"))?,
                                    project
                                        .parse()
                                        .map_err(|_| invalid("invalid billing project header"))?,
                                )]
                                .into_iter()
                                .collect(),
                            ),
                        );
                    }
                    Arc::new(builder.build()?)
                };
                let root = Path::from_url_path(url.path()).map_err(|e| invalid(e.to_string()))?;
                (store, root.to_string())
            } else {
                let path = if uri.starts_with("file://") {
                    Url::parse(uri)?
                        .to_file_path()
                        .map_err(|_| invalid("invalid file URL"))?
                } else {
                    std::path::PathBuf::from(uri)
                };
                let absolute = std::fs::canonicalize(path)?;
                (
                    Arc::new(LocalFileSystem::new()),
                    absolute
                        .to_string_lossy()
                        .trim_start_matches('/')
                        .to_owned(),
                )
            };
        Ok(Self {
            inner,
            scope: root.clone(),
            root,
            requests: Arc::default(),
            bytes: Arc::default(),
        })
    }

    pub fn child(&self, relative: &str) -> Result<Self> {
        Ok(Self {
            root: self.path(relative)?.to_string(),
            ..self.clone()
        })
    }

    fn path(&self, relative: &str) -> Result<Path> {
        if relative.starts_with('/') || relative.contains("://") {
            return Err(invalid("component path must be relative"));
        }
        let joined = format!("{}/{}", self.root, relative);
        let mut parts = Vec::new();
        for part in joined.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    parts
                        .pop()
                        .ok_or_else(|| invalid("component path escapes storage root"))?;
                }
                _ => parts.push(part),
            }
        }
        let path = parts.join("/");
        if path != self.scope && !path.starts_with(&format!("{}/", self.scope)) {
            return Err(invalid("component path escapes dataset directory"));
        }
        Path::parse(path).map_err(|e| invalid(e.to_string()))
    }

    pub async fn get(&self, relative: &str) -> Result<Bytes> {
        let data = self.inner.get(&self.path(relative)?).await?.bytes().await?;
        self.record(data.len());
        Ok(data)
    }

    pub async fn get_range(&self, relative: &str, range: Range<u64>) -> Result<Bytes> {
        let data = self.inner.get_range(&self.path(relative)?, range).await?;
        self.record(data.len());
        Ok(data)
    }

    pub async fn size(&self, relative: &str) -> Result<u64> {
        let size = self.inner.head(&self.path(relative)?).await?.size;
        self.requests.fetch_add(1, Ordering::Relaxed);
        Ok(size)
    }

    fn record(&self, bytes: usize) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub fn stats(&self) -> ReadStats {
        ReadStats {
            requests: self.requests.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
        }
    }

    pub(crate) async fn metadata(&self, relative: &str) -> Result<serde_json::Value> {
        use std::io::Read;
        let bytes = self.get(relative).await?;
        if relative.ends_with(".gz") {
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(bytes.as_ref())
                .take(256 * 1024 * 1024)
                .read_to_end(&mut out)?;
            Ok(serde_json::from_slice(&out)?)
        } else {
            Ok(serde_json::from_slice(&bytes)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn local_ranges_and_dataset_path_boundary() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("data"), b"0123456789").unwrap();
        let storage = Storage::open(
            directory.path().to_str().unwrap(),
            &StorageOptions::default(),
        )
        .unwrap();
        assert_eq!(
            storage.get_range("data", 2..5).await.unwrap().as_ref(),
            b"234"
        );
        assert_eq!(storage.stats().bytes, 3);
        assert!(storage.get("../outside").await.is_err());
        assert!(storage.child("/absolute").is_err());
        assert!(storage.child("s3://other/path").is_err());
    }

    #[test]
    fn gcs_anonymous_and_requester_pays_configuration() {
        let opts = StorageOptions {
            anonymous: true,
            requester_pays_project: Some("billing-project".into()),
            ..Default::default()
        };
        assert!(Storage::open("gs://example-bucket/dataset.ht", &opts).is_ok());
        let bad = StorageOptions {
            requester_pays_project: Some("invalid\nheader".into()),
            ..opts
        };
        assert!(Storage::open("gs://example-bucket/dataset.ht", &bad).is_err());
    }
}
