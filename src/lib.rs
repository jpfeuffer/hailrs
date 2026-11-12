//! JVM-free readers for native Hail datasets.
//!
//! Storage is asynchronous. Opening a dataset reads metadata only; data reads
//! select partitions or blocks explicitly. Unsupported codecs fail explicitly.
mod buffer;
mod index;
mod matrix;
mod reference;
mod schema;
mod storage;
mod table;

#[cfg(feature = "python")]
mod python;

pub use matrix::{BlockMatrix, BlockMatrixMetadata, DenseMatrix};
pub use schema::{HailType, Value};
pub use storage::{ReadStats, Storage, StorageOptions};
pub use table::{LocusInterval, MatrixTable, ReadOptions, Table};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid Hail format: {0}")]
    Format(String),
    #[error("truncated stream at byte {position}, need {needed} bytes")]
    Truncated { position: usize, needed: usize },
    #[error("unsupported Hail feature: {0}")]
    Unsupported(String),
    #[error(transparent)]
    Storage(#[from] object_store::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Url(#[from] url::ParseError),
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Format(message.into())
}
