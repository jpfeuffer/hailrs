use crate::{
    buffer::{BufferSpec, Cursor},
    invalid, Error, HailType, ReadStats, Result, Storage, StorageOptions, Value,
};
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocusInterval {
    pub contig: String,
    /// One-based, inclusive.
    pub start: i64,
    /// One-based, exclusive.
    pub end: i64,
    #[serde(default = "locus_name")]
    pub field: String,
}
fn locus_name() -> String {
    "locus".into()
}

impl LocusInterval {
    fn validate(&self) -> Result<()> {
        if self.contig.is_empty() || self.start < 1 || self.end < self.start {
            return Err(invalid(
                "locus interval requires a contig and 1 <= start <= end",
            ));
        }
        Ok(())
    }

    fn contains(&self, row: &Value) -> bool {
        row.field(&self.field).is_some_and(|locus| {
            locus.field("contig").and_then(Value::as_str) == Some(self.contig.as_str())
                && locus
                    .field("position")
                    .and_then(Value::as_i64)
                    .is_some_and(|p| p >= self.start && p < self.end)
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReadOptions {
    pub partitions: Option<Vec<usize>>,
    pub interval: Option<LocusInterval>,
    pub limit: Option<usize>,
    pub concurrency: usize,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            partitions: None,
            interval: None,
            limit: None,
            concurrency: 8,
        }
    }
}

#[derive(Clone)]
struct Rvd {
    storage: Storage,
    metadata: Json,
    encoded: HailType,
    logical: HailType,
    buffer: BufferSpec,
    files: Vec<String>,
}

impl Rvd {
    async fn partition_range(
        &self,
        index: usize,
        start: usize,
        count: usize,
    ) -> Result<Vec<Value>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let file = self
            .files
            .get(index)
            .ok_or_else(|| invalid("partition out of bounds"))?;
        let index_spec = &self.metadata["_indexSpec"];
        if index_spec.is_null() {
            return Ok(self
                .partition(index)
                .await?
                .into_iter()
                .skip(start)
                .take(count)
                .collect());
        }
        let virtual_offset =
            crate::index::row_offset(&self.storage, index_spec, file, start).await?;
        let path = format!("parts/{file}");
        let size = self.storage.size(&path).await?;
        let mut frames = crate::index::Frames {
            storage: &self.storage,
            file: &path,
            spec: &self.buffer,
            offset: virtual_offset >> 16,
            size,
        };
        let mut data = frames.next().await?;
        let mut position = (virtual_offset & 0xffff) as usize;
        if position > data.len() {
            return Err(invalid("virtual offset exceeds frame"));
        }
        let mut rows = Vec::new();
        while rows.len() < count {
            let mut cursor = Cursor::new(&data[position..], self.buffer.leb128);
            let decoded = (|| {
                if cursor.byte()? != 1 {
                    return Err(invalid("indexed row missing marker"));
                }
                self.encoded.decode(&mut cursor, Some(&self.logical))
            })();
            match decoded {
                Ok(row) => {
                    position += cursor.position;
                    rows.push(row);
                }
                Err(Error::Truncated { .. }) => {
                    data.drain(..position);
                    position = 0;
                    data.extend(frames.next().await?);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(rows)
    }

    async fn open(storage: Storage) -> Result<Self> {
        let metadata = storage.metadata("metadata.json.gz").await?;
        let codec = &metadata["_codecSpec"];
        if codec["name"] != "TypedCodecSpec" {
            return Err(Error::Unsupported("pre-TypedCodecSpec RVD metadata".into()));
        }
        let encoded = HailType::parse(
            codec["_eType"]
                .as_str()
                .ok_or_else(|| invalid("missing _eType"))?,
        )?;
        let logical = HailType::parse(
            codec["_vType"]
                .as_str()
                .ok_or_else(|| invalid("missing _vType"))?,
        )?;
        let buffer = BufferSpec::parse(&codec["_bufferSpec"])?;
        let files = serde_json::from_value(metadata["_partFiles"].clone())?;
        Ok(Self {
            storage,
            metadata,
            encoded,
            logical,
            buffer,
            files,
        })
    }

    async fn partition(&self, index: usize) -> Result<Vec<Value>> {
        let file = self
            .files
            .get(index)
            .ok_or_else(|| invalid(format!("partition {index} out of bounds")))?;
        let data = self.storage.get(&format!("parts/{file}")).await?;
        let decoded = self.buffer.decode(&data)?;
        let mut cursor = Cursor::new(&decoded, self.buffer.leb128);
        let mut rows = Vec::new();
        loop {
            match cursor.byte()? {
                0 => break,
                1 => rows.push(self.encoded.decode(&mut cursor, Some(&self.logical))?),
                n => {
                    return Err(invalid(format!(
                        "invalid row marker {n} in partition {index}"
                    )))
                }
            }
        }
        if cursor.remaining() != 0 {
            return Err(invalid(format!("trailing bytes in partition {index}")));
        }
        Ok(rows)
    }

    fn overlaps(&self, index: usize, interval: &LocusInterval) -> bool {
        if self.metadata["_key"]
            .as_array()
            .and_then(|k| k.first())
            .and_then(Json::as_str)
            != Some(interval.field.as_str())
        {
            return true;
        }
        let bound = &self.metadata["_jRangeBounds"][index];
        let start = &bound["start"][&interval.field];
        let end = &bound["end"][&interval.field];
        let (Some(sc), Some(ec), Some(sp), Some(ep)) = (
            start["contig"].as_str(),
            end["contig"].as_str(),
            start["position"].as_i64(),
            end["position"].as_i64(),
        ) else {
            return true;
        };
        // Only exclude bounds we can prove disjoint. Cross-contig bounds on
        // custom references remain candidates rather than guessing contig order.
        if sc == ec && sc != interval.contig {
            return false;
        }
        let reference = self
            .logical
            .fields
            .iter()
            .find(|(name, _)| name == &interval.field)
            .and_then(|(_, typ)| typ.args.first())
            .map(|t| t.name.as_str());
        if let Some(reference) = reference {
            if let (Some(start), Some(end), Some(target)) = (
                crate::reference::contig_rank(reference, sc),
                crate::reference::contig_rank(reference, ec),
                crate::reference::contig_rank(reference, &interval.contig),
            ) {
                if target < start || target > end {
                    return false;
                }
            }
        }
        if sc == interval.contig && sp >= interval.end {
            return false;
        }
        if ec == interval.contig && ep < interval.start {
            return false;
        }
        true
    }
}

#[derive(Clone)]
pub struct Table {
    storage: Storage,
    pub metadata: Json,
    rows: Rvd,
    globals: Rvd,
}

impl Table {
    pub async fn open(uri: &str, options: StorageOptions) -> Result<Self> {
        Self::open_storage(Storage::open(uri, &options)?).await
    }

    async fn open_storage(storage: Storage) -> Result<Self> {
        storage.get("_SUCCESS").await?;
        let metadata = storage.metadata("metadata.json.gz").await?;
        validate_version(&metadata)?;
        if metadata["table_type"].as_str().is_none() {
            return Err(invalid(
                "not a Hail Table; use MatrixTable for .mt datasets",
            ));
        }
        let (rows, globals) = tokio::try_join!(
            Rvd::open(storage.child(component(&metadata, "rows")?)?),
            Rvd::open(storage.child(component(&metadata, "globals")?)?)
        )?;
        if let Some(counts) = metadata["components"]["partition_counts"]["counts"].as_array() {
            if counts.len() != rows.files.len() {
                return Err(invalid("partition count metadata length mismatch"));
            }
        }
        Ok(Self {
            storage,
            metadata,
            rows,
            globals,
        })
    }

    pub fn row_type(&self) -> &HailType {
        &self.rows.logical
    }
    pub fn encoded_type(&self) -> &HailType {
        &self.rows.encoded
    }
    pub fn global_type(&self) -> &HailType {
        &self.globals.logical
    }
    pub fn n_partitions(&self) -> usize {
        self.rows.files.len()
    }
    pub fn stats(&self) -> ReadStats {
        self.storage.stats()
    }
    pub fn partition_bounds(&self) -> &Json {
        &self.rows.metadata["_jRangeBounds"]
    }

    pub async fn globals(&self) -> Result<Value> {
        let mut rows = self.globals.partition(0).await?;
        if rows.len() != 1 {
            return Err(invalid("globals must contain exactly one row"));
        }
        Ok(rows.remove(0))
    }

    pub async fn read_partition(&self, index: usize) -> Result<Vec<Value>> {
        self.rows.partition(index).await
    }

    /// Global ordinal row range, using index B-trees and compressed range GETs
    /// for indexed partitions. Unindexed partitions are scanned in full.
    pub async fn read_row_range(
        &self,
        range: std::ops::Range<usize>,
        concurrency: usize,
    ) -> Result<Vec<Value>> {
        if range.start > range.end || concurrency == 0 {
            return Err(invalid("invalid row range/concurrency"));
        }
        let counts: Vec<usize> = serde_json::from_value(
            self.metadata["components"]["partition_counts"]["counts"].clone(),
        )?;
        let total = counts
            .iter()
            .try_fold(0usize, |a, b| a.checked_add(*b))
            .ok_or_else(|| invalid("row count overflow"))?;
        if range.end > total {
            return Err(invalid("row range exceeds table length"));
        }
        let mut base = 0;
        let mut selected = Vec::new();
        for (partition, count) in counts.into_iter().enumerate() {
            let start = range.start.max(base);
            let end = range.end.min(base + count);
            if start < end {
                selected.push((partition, start - base, end - start));
            }
            base += count;
        }
        let mut partitions = stream::iter(selected)
            .map(|(p, start, count)| self.rows.partition_range(p, start, count))
            .buffered(concurrency);
        let mut out = Vec::new();
        while let Some(rows) = partitions.next().await {
            out.extend(rows?);
        }
        Ok(out)
    }

    pub fn selected_partitions(&self, options: &ReadOptions) -> Result<Vec<usize>> {
        if options.concurrency == 0 {
            return Err(invalid("concurrency must be positive"));
        }
        if let Some(interval) = &options.interval {
            interval.validate()?;
            let field = self
                .row_type()
                .fields
                .iter()
                .find(|(name, _)| name == &interval.field);
            if !field.is_some_and(|(_, typ)| typ.name == "Locus") {
                return Err(invalid(format!("{} is not a locus field", interval.field)));
            }
        }
        let parts = options
            .partitions
            .clone()
            .unwrap_or_else(|| (0..self.n_partitions()).collect());
        if parts.iter().any(|p| *p >= self.n_partitions()) {
            return Err(invalid("partition out of bounds"));
        }
        Ok(parts
            .into_iter()
            .filter(|&p| {
                options.interval.as_ref().is_none_or(|interval| {
                    interval.start != interval.end && self.rows.overlaps(p, interval)
                })
            })
            .collect())
    }

    pub async fn read(&self, options: ReadOptions) -> Result<Vec<Value>> {
        let parts = self.selected_partitions(&options)?;
        let mut out = Vec::new();
        if options.limit == Some(0) {
            return Ok(out);
        }
        let mut partitions = stream::iter(parts)
            .map(|p| self.rows.partition(p))
            .buffered(options.concurrency);
        while let Some(rows) = partitions.next().await {
            for row in rows? {
                if options.interval.as_ref().is_some_and(|i| !i.contains(&row)) {
                    continue;
                }
                out.push(row);
                if options.limit == Some(out.len()) {
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }
}

fn component<'a>(metadata: &'a Json, name: &str) -> Result<&'a str> {
    metadata["components"][name]["rel_path"]
        .as_str()
        .ok_or_else(|| invalid(format!("missing {name} component")))
}

fn validate_version(metadata: &Json) -> Result<()> {
    let version = metadata["file_version"]
        .as_u64()
        .ok_or_else(|| invalid("missing Hail file_version (0.1 files are not supported)"))?;
    if version >> 16 != 1 {
        return Err(Error::Unsupported(format!("file format version {version}")));
    }
    Ok(())
}

/// MatrixTable is four native Table components. Entry rows contain an entry
/// array in column order; row and entry partition indices are aligned.
pub struct MatrixTable {
    pub metadata: Json,
    pub rows: Table,
    pub cols: Table,
    pub entries: Table,
    pub globals: Table,
}

impl MatrixTable {
    pub async fn open(uri: &str, options: StorageOptions) -> Result<Self> {
        let storage = Storage::open(uri, &options)?;
        storage.get("_SUCCESS").await?;
        let metadata = storage.metadata("metadata.json.gz").await?;
        validate_version(&metadata)?;
        if metadata["matrix_type"].as_str().is_none() {
            return Err(invalid("not a MatrixTable"));
        }
        let (rows, cols, entries, globals) = tokio::try_join!(
            Table::open_storage(storage.child("rows")?),
            Table::open_storage(storage.child("cols")?),
            Table::open_storage(storage.child("entries")?),
            Table::open_storage(storage.child("globals")?)
        )?;
        if rows.n_partitions() != entries.n_partitions() {
            return Err(invalid("MatrixTable rows and entries partition mismatch"));
        }
        Ok(Self {
            metadata,
            rows,
            cols,
            entries,
            globals,
        })
    }
}
