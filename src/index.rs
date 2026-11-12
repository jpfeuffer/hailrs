use crate::{
    buffer::{BufferSpec, Cursor},
    invalid, Error, HailType, Result, Storage, Value,
};
use serde_json::Value as Json;

pub(crate) struct Frames<'a> {
    pub storage: &'a Storage,
    pub file: &'a str,
    pub spec: &'a BufferSpec,
    pub offset: u64,
    pub size: u64,
}

impl Frames<'_> {
    pub async fn next(&mut self) -> Result<Vec<u8>> {
        if self.offset >= self.size {
            return Err(invalid("indexed offset past end of file"));
        }
        let header = self
            .storage
            .get_range(self.file, self.offset..(self.offset + 5).min(self.size))
            .await?;
        let length = self.spec.frame_size(&header)? as u64;
        let end = self
            .offset
            .checked_add(length)
            .filter(|e| *e <= self.size)
            .ok_or_else(|| invalid("truncated indexed frame"))?;
        let bytes = self.storage.get_range(self.file, self.offset..end).await?;
        let (decoded, _) = self.spec.frame(&bytes)?;
        self.offset = end;
        Ok(decoded)
    }
}

/// Read a row's virtual data offset through the on-disk B-tree. Both legacy
/// index directories and newer single files with inline metadata are supported.
pub(crate) async fn row_offset(
    storage: &Storage,
    spec: &Json,
    part: &str,
    row: usize,
) -> Result<u64> {
    let relative = spec["_relPath"]
        .as_str()
        .ok_or_else(|| invalid("index _relPath"))?;
    let index = storage.child(&format!("{relative}/{part}.idx"))?;
    let (metadata, file, size) = match index.metadata("metadata.json.gz").await {
        Ok(metadata) => {
            let file = metadata["indexPath"]
                .as_str()
                .ok_or_else(|| invalid("indexPath"))?
                .to_owned();
            let size = index.size(&file).await?;
            (metadata, file, size)
        }
        Err(error) if missing_directory_metadata(&error) => {
            let size = index.size("").await?;
            if size < 8 {
                return Err(invalid("index footer missing"));
            }
            let footer = index.get_range("", size - 8..size).await?;
            let offset = u64::from_le_bytes(footer.as_ref().try_into().unwrap());
            if offset > size - 8 {
                return Err(invalid("index metadata offset out of bounds"));
            }
            let bytes = index.get_range("", offset..size - 8).await?;
            let metadata: Json = serde_json::from_slice(&bytes)?;
            (metadata, String::new(), offset)
        }
        Err(error) => return Err(error),
    };
    if row as u64
        >= metadata["nKeys"]
            .as_u64()
            .ok_or_else(|| invalid("index nKeys"))?
    {
        return Err(invalid("row outside index"));
    }
    let height = metadata["height"]
        .as_u64()
        .filter(|h| (1..=64).contains(h))
        .ok_or_else(|| invalid("index height"))?;
    let mut offset = metadata["rootOffset"]
        .as_u64()
        .ok_or_else(|| invalid("index rootOffset"))?;
    for level in (0..height).rev() {
        let leaf = level == 0;
        let codec = &spec[if leaf {
            "_leafCodec"
        } else {
            "_internalNodeCodec"
        }];
        let encoded = HailType::parse(
            codec["_eType"]
                .as_str()
                .ok_or_else(|| invalid("index encoded type"))?,
        )?;
        let logical = HailType::parse(
            codec["_vType"]
                .as_str()
                .ok_or_else(|| invalid("index logical type"))?,
        )?;
        let buffer = BufferSpec::parse(&codec["_bufferSpec"])?;
        let mut frames = Frames {
            storage: &index,
            file: &file,
            spec: &buffer,
            offset,
            size,
        };
        let mut data = Vec::new();
        let node = loop {
            data.extend(frames.next().await?);
            let mut cursor = Cursor::new(&data, buffer.leb128);
            if cursor.byte()? != if leaf { 0 } else { 1 } {
                return Err(invalid("index node marker"));
            }
            match encoded.decode(&mut cursor, Some(&logical)) {
                Ok(node) => break node,
                Err(Error::Truncated { .. }) => {}
                Err(error) => return Err(error),
            }
        };
        let children = match node.field(if leaf { "keys" } else { "children" }) {
            Some(Value::Array(children)) => children,
            _ => return Err(invalid("index children missing")),
        };
        if leaf {
            let first = integer(&node, "first_idx")?;
            let child = children
                .get(
                    row.checked_sub(first as usize)
                        .ok_or_else(|| invalid("index leaf first_idx"))?,
                )
                .ok_or_else(|| invalid("index leaf does not contain row"))?;
            let offset = if let Some(field) = spec["_offsetField"].as_str() {
                integer(
                    child
                        .field("annotation")
                        .ok_or_else(|| invalid("index annotation"))?,
                    field,
                )?
            } else {
                integer(child, "offset")?
            };
            return Ok(offset);
        }
        let child = children
            .iter()
            .rev()
            .find(|child| {
                child
                    .field("first_idx")
                    .and_then(Value::as_i64)
                    .is_some_and(|i| i >= 0 && i as usize <= row)
            })
            .ok_or_else(|| invalid("index internal node does not contain row"))?;
        offset = integer(child, "index_file_offset")?;
    }
    Err(invalid("index traversal did not reach a leaf"))
}

fn integer(value: &Value, field: &str) -> Result<u64> {
    value
        .field(field)
        .and_then(Value::as_i64)
        .filter(|n| *n >= 0)
        .map(|n| n as u64)
        .ok_or_else(|| invalid(format!("invalid index {field}")))
}

fn missing_directory_metadata(error: &Error) -> bool {
    if matches!(error, Error::Storage(object_store::Error::NotFound { .. })) {
        return true;
    }
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if source
            .downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::NotADirectory)
        {
            return true;
        }
        match source.source() {
            Some(next) => source = next,
            None => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StorageOptions;

    #[tokio::test]
    async fn inline_footer_matches_legacy_index_directory() {
        let original = Storage::open(
            &format!("{}/tests/fixtures/all_types.ht", env!("CARGO_MANIFEST_DIR")),
            &StorageOptions::default(),
        )
        .unwrap()
        .child("rows")
        .unwrap();
        let metadata = original.metadata("metadata.json.gz").await.unwrap();
        let spec = &metadata["_indexSpec"];
        let part = metadata["_partFiles"][0].as_str().unwrap();
        let expected = row_offset(&original, spec, part, 1).await.unwrap();
        let index = original.child(&format!("../index/{part}.idx")).unwrap();
        let metadata = index.metadata("metadata.json.gz").await.unwrap();
        let mut bytes = index.get("index").await.unwrap().to_vec();
        let offset = bytes.len() as u64;
        bytes.extend(serde_json::to_vec(&metadata).unwrap());
        bytes.extend(offset.to_le_bytes());
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("index")).unwrap();
        std::fs::create_dir(directory.path().join("rows")).unwrap();
        std::fs::write(directory.path().join(format!("index/{part}.idx")), bytes).unwrap();
        let converted = Storage::open(
            directory.path().to_str().unwrap(),
            &StorageOptions::default(),
        )
        .unwrap()
        .child("rows")
        .unwrap();
        assert_eq!(
            row_offset(&converted, spec, part, 1).await.unwrap(),
            expected
        );
    }
}
