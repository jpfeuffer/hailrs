use crate::{
    buffer::{BufferSpec, Cursor},
    invalid, ReadStats, Result, Storage, StorageOptions,
};
use futures::{stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockMatrixMetadata {
    pub block_size: usize,
    pub n_rows: usize,
    pub n_cols: usize,
    pub maybe_filtered: Option<Vec<u64>>,
    pub part_files: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DenseMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    /// Row-major values, preserving the requested index order and duplicates.
    pub data: Vec<f64>,
}

#[derive(Clone)]
pub struct BlockMatrix {
    storage: Storage,
    pub metadata: BlockMatrixMetadata,
    parts: HashMap<u64, usize>,
}

impl BlockMatrix {
    pub async fn open(uri: &str, options: StorageOptions) -> Result<Self> {
        let storage = Storage::open(uri, &options)?;
        storage.get("_SUCCESS").await?;
        let metadata: BlockMatrixMetadata =
            serde_json::from_value(storage.metadata("metadata.json").await?)?;
        if metadata.block_size == 0 || metadata.n_rows == 0 || metadata.n_cols == 0 {
            return Err(invalid("matrix dimensions/blockSize must be positive"));
        }
        let count = metadata
            .n_rows
            .div_ceil(metadata.block_size)
            .checked_mul(metadata.n_cols.div_ceil(metadata.block_size))
            .ok_or_else(|| invalid("block grid overflow"))?;
        let ids = match &metadata.maybe_filtered {
            Some(ids) => ids.clone(),
            None => {
                if count != metadata.part_files.len() {
                    return Err(invalid("dense matrix part list does not match block grid"));
                }
                (0..count as u64).collect()
            }
        };
        if ids.len() != metadata.part_files.len() || ids.iter().any(|id| *id >= count as u64) {
            return Err(invalid("matrix part list does not match block grid"));
        }
        let parts: HashMap<_, _> = ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        if parts.len() != ids.len() {
            return Err(invalid("duplicate block id"));
        }
        Ok(Self {
            storage,
            metadata,
            parts,
        })
    }

    pub fn stats(&self) -> ReadStats {
        self.storage.stats()
    }

    pub async fn read(
        &self,
        rows: &[usize],
        cols: &[usize],
        concurrency: usize,
    ) -> Result<DenseMatrix> {
        self.read_impl(rows, cols, concurrency, false).await
    }

    /// Interpret the stored upper triangle as symmetric; never applied implicitly.
    pub async fn read_symmetric_upper(
        &self,
        rows: &[usize],
        cols: &[usize],
        concurrency: usize,
    ) -> Result<DenseMatrix> {
        if self.metadata.n_rows != self.metadata.n_cols {
            return Err(invalid("symmetric reads require a square matrix"));
        }
        self.read_impl(rows, cols, concurrency, true).await
    }

    async fn read_impl(
        &self,
        rows: &[usize],
        cols: &[usize],
        concurrency: usize,
        symmetric: bool,
    ) -> Result<DenseMatrix> {
        if concurrency == 0 {
            return Err(invalid("concurrency must be positive"));
        }
        if rows.iter().any(|r| *r >= self.metadata.n_rows)
            || cols.iter().any(|c| *c >= self.metadata.n_cols)
        {
            return Err(invalid("matrix index out of bounds"));
        }
        let length = rows
            .len()
            .checked_mul(cols.len())
            .ok_or_else(|| invalid("result size overflow"))?;
        let size = self.metadata.block_size;
        let n_block_rows = self.metadata.n_rows.div_ceil(size);
        let mut needed = HashSet::new();
        for &r in rows {
            for &c in cols {
                let (r, c) = if symmetric && r > c { (c, r) } else { (r, c) };
                let id = (c / size * n_block_rows + r / size) as u64;
                if self.parts.contains_key(&id) {
                    needed.insert(id);
                }
            }
        }
        // Extract only requested cells while each block is resident, bounding memory
        // by concurrency * block size rather than the number of selected blocks.
        let cells: Vec<BTreeMap<usize, f64>> = stream::iter(needed)
            .map(|id| async move {
                let file = &self.metadata.part_files[self.parts[&id]];
                let compressed = self.storage.get(&format!("parts/{file}")).await?;
                let decoded = BufferSpec::matrix().decode(&compressed)?;
                let mut cursor = Cursor::new(&decoded, false);
                let nr = cursor.int()?;
                let nc = cursor.int()?;
                let row_major = match cursor.byte()? {
                    0 => false,
                    1 => true,
                    _ => return Err(invalid("matrix transpose flag")),
                };
                let br = id as usize % n_block_rows;
                let bc = id as usize / n_block_rows;
                let expected_rows = size.min(self.metadata.n_rows - br * size);
                let expected_cols = size.min(self.metadata.n_cols - bc * size);
                if nr <= 0
                    || nc <= 0
                    || nr as usize != expected_rows
                    || nc as usize != expected_cols
                {
                    return Err(invalid("block shape disagrees with matrix metadata"));
                }
                let byte_length = expected_rows
                    .checked_mul(expected_cols)
                    .and_then(|n| n.checked_mul(8))
                    .ok_or_else(|| invalid("block byte length overflow"))?;
                let raw = cursor.take(byte_length)?;
                if cursor.remaining() != 0 {
                    return Err(invalid("trailing matrix bytes"));
                }
                let mut selected = BTreeMap::new();
                for (i, &r) in rows.iter().enumerate() {
                    for (j, &c) in cols.iter().enumerate() {
                        let (r, c) = if symmetric && r > c { (c, r) } else { (r, c) };
                        if r / size != br || c / size != bc {
                            continue;
                        }
                        let offset = if row_major {
                            (r % size * expected_cols + c % size) * 8
                        } else {
                            (c % size * expected_rows + r % size) * 8
                        };
                        let value = f64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap());
                        selected.insert(i * cols.len() + j, value);
                    }
                }
                Ok::<_, crate::Error>(selected)
            })
            .buffer_unordered(concurrency)
            .try_collect()
            .await?;
        let mut data = vec![0.0; length];
        for block in cells {
            for (index, value) in block {
                data[index] = value;
            }
        }
        Ok(DenseMatrix {
            n_rows: rows.len(),
            n_cols: cols.len(),
            data,
        })
    }
}
