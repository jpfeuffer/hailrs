# hailrs
Spark-free reader for Hail Tables and BlockMatrices in Rust with Python bindings (S3/GCS)

Read native Hail **Tables (`.ht`), BlockMatrices (`.bm`), and MatrixTables
(`.mt`) without Java, Spark, or the Hail Python package**. Rust handles
metadata, schema-driven decoding, compression, and asynchronous object storage;
Python exposes NumPy arrays and typed PyArrow tables.

## Install

**[Version 0.1.0 is available as prebuilt GitHub release wheels](https://github.com/jpfeuffer/hailrs/releases/tag/v0.1.0).**
No Rust, Java, or Spark is needed to install a wheel. This package is not yet
published to PyPI or crates.io; `pip install hailrs` is not the installation
method for this release.

For **macOS Apple Silicon with Python 3.12**, inside a virtual environment:

```sh
python -m pip install https://github.com/jpfeuffer/hailrs/releases/download/v0.1.0/hailrs-0.1.0-cp312-cp312-macosx_11_0_arm64.whl
```

Or, with uv in a new directory:

```sh
uv venv --python 3.12
uv pip install https://github.com/jpfeuffer/hailrs/releases/download/v0.1.0/hailrs-0.1.0-cp312-cp312-macosx_11_0_arm64.whl
```

Choose the matching wheel from the release for other environments:

| Platform | Wheel platform tag | Python versions |
|---|---|---|
| macOS 11+, Apple Silicon | `macosx_11_0_arm64` | CPython 3.10-3.13 |
| macOS 11+, Intel | `macosx_11_0_x86_64` | CPython 3.10-3.13 |
| Linux glibc 2.28+, ARM64 | `manylinux_2_28_aarch64` | CPython 3.10-3.13 |
| Linux glibc 2.28+, x86_64 | `manylinux_2_28_x86_64` | CPython 3.10-3.13 |

`cp310`, `cp311`, `cp312`, and `cp313` identify the Python version. NumPy and
PyArrow dependencies are installed automatically. Windows, musl-based Linux,
PyPy, free-threaded Python, and Python 3.14+ wheels are not included.
All 16 wheels were built in CI and each passed the 175-test real-Hail fixture
suite after installation on its native platform. The release includes the
source distribution and `SHA256SUMS`.

### Build from source

Requires a Rust toolchain and Python 3.10-3.13.

```sh
git clone https://github.com/jpfeuffer/hailrs.git
cd hailrs
uv sync --python 3.12
```

For development after changing Rust code:

```sh
uv run maturin develop --release
```

Alternatively, install from a checkout with `python -m pip install .` inside
your own virtual environment. No Hail/JVM dependencies are installed.

### Maintainer publishing

`.github/workflows/publish.yml` builds and tests the wheel matrix and source
distribution. Dispatch it with `publish=false` for artifacts only. PyPI
publication requires a pending GitHub Trusted Publisher for project `hailrs`,
owner/repository `jpfeuffer/hailrs`, workflow `publish.yml`, environment
`pypi`; no long-lived upload token is needed. With that account-side setup,
dispatch from `main` with `publish=true` and the matching package version.
The publish job runs only after every distribution job succeeds.

## Python

### Tables and Arrow

```python
from hailrs import LocusInterval, StorageOptions, Table

storage = StorageOptions(anonymous=True, region="us-east-1")
variants = Table(
    "s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.variant.b38.ht",
    storage=storage,
)

# One-based positions; end is exclusive. Only candidate partitions are fetched.
region = variants.read(interval=LocusInterval("chr22", 20012563, 20013428))
assert region["idx"].to_pylist() == list(range(22893433, 22893443))
print(region.to_pandas())
print(variants.globals)
print(variants.stats)  # Successful requests and transferred bytes, cumulatively.

# Explicit partition reads, streaming scans, and indexed ordinal row ranges:
first = variants.read(partitions=[0], limit=10)
for batch in variants.iter_batches(partitions=[0, 1], batch_size=1024):
    pass
ten_rows = variants.read_row_range(100, 110)

# Polars interoperability, without an additional hailrs dependency:
# import polars as pl
# frame = pl.from_arrow(region)
```

`read_rows()` returns ordinary Python dictionaries instead of Arrow. Empty
results retain the Arrow schema. Partition order, row order, and explicit
partition-list order are preserved. `limit` limits returned rows, although
bounded concurrent prefetches can fetch additional candidate partitions.
`iter_batches()` fetches one entire partition at a time; it does not hold an
entire table in memory.

### Selective BlockMatrix reads

```python
import numpy as np
from hailrs import BlockMatrix, StorageOptions

matrix = BlockMatrix(
    "s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.bm",
    storage=StorageOptions(anonymous=True, region="us-east-1"),
)
indices = list(range(22893433, 22893443))
ld = matrix.read(indices, indices, symmetric_upper=True)
assert ld.shape == (10, 10)

# Optional Gentropy convention, NOT applied by the native reader:
np.fill_diagonal(ld, 1.0)
```

Indices may be unordered or repeated. Slices work too, including negative
slice steps. Individual indices must be nonnegative and in bounds. Missing
sparse blocks are zeros; failed downloads are errors, never zeros.
Only intersecting stored block files are fetched, concurrently and once per
read. Each selected block file is downloaded and decompressed in full: there is
no byte-level index within a BlockMatrix block. Avoid calling `read()` without
indices on a large matrix.

By default the reader returns **exactly the stored matrix**, not a symmetric
interpretation. `symmetric_upper=True` explicitly reads the upper triangle for
both halves of a square result. It also works for rectangular selections from
a square source. The LD-adjusted Pan-UKBB diagonal is slightly below one;
setting it to one is a separate application-level transformation.

### MatrixTables

```python
from hailrs import MatrixTable

mt = MatrixTable("data.mt")
rows = mt.rows.read()
columns = mt.cols.read()
entries = mt.entries.read()
globals_ = mt.globals
```

MatrixTables expose their four native Table components. Entry rows contain
an array in column order; filtered entries are null array elements. Hail's
synthetic entry-field name is retained. Row/entry partition indices align:
select the same partitions from both and apply the same row mask when filtering.
This API does not silently flatten entries or change their sparsity.
Open components through `MatrixTable`, rather than treating an internal
`data.mt/entries` directory as an independent dataset.

## Rust

Depend on this Git repository until a crates.io release is available:

```toml
[dependencies]
hailrs = { git = "https://github.com/jpfeuffer/hailrs" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use hailrs::{BlockMatrix, LocusInterval, ReadOptions, StorageOptions, Table};

#[tokio::main]
async fn main() -> hailrs::Result<()> {
    let storage = StorageOptions {
        anonymous: true,
        region: Some("us-east-1".into()),
        ..Default::default()
    };
    let table = Table::open(
        "s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.variant.b38.ht",
        storage.clone(),
    ).await?;
    let rows = table.read(ReadOptions {
        interval: Some(LocusInterval {
            contig: "chr22".into(), start: 20012563, end: 20013428,
            field: "locus".into(),
        }),
        ..Default::default()
    }).await?;
    println!("{:?}", rows[0]);
    let matrix = BlockMatrix::open(
        "s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.bm", storage,
    ).await?;
    let indices: Vec<usize> = (22893433..22893443).collect();
    let result = matrix.read_symmetric_upper(&indices, &indices, 4).await?;
    println!("{:?}", result.data); // Row-major f64 values.
    Ok(())
}
```

Rust table rows use the public `Value` enum, with distinct integers, floats,
strings, binary, arrays, structs, and nulls; `row_type()` supplies the Hail
logical schema. Floats retain NaN, infinities, and signed zero in memory.
`MatrixTable` also exposes its four Table components in Rust.

## Storage and credentials

Local paths and `file://`, `s3://`, `gs://`, and `gcs://` URIs are supported.
Readers require the native `_SUCCESS` marker and reject truncated data.
Metadata component paths cannot escape the opened dataset directory.

- **S3:** `StorageOptions(anonymous=True, region="us-east-1")` for public
  unsigned access. Otherwise credentials are resolved by `object_store`, e.g.
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`.
  Region can also be supplied through `AWS_REGION`. `endpoint` configures an
  S3-compatible endpoint; an explicitly HTTP endpoint enables plain HTTP.
- **GCS:** use application-default credentials or
  `GOOGLE_APPLICATION_CREDENTIALS`. Set
  `StorageOptions(requester_pays_project="billing-project")` to send
  `x-goog-user-project` on requests. The credential must have the necessary
  storage and billing permissions. Anonymous access is also supported.

Local and anonymous S3 reads are exercised end-to-end. GCS configuration and
requester-pays header validation have unit coverage, but **authenticated GCS
and real requester-pays billing have not been tested against a live bucket**.

## Format support and boundaries

| Surface | Support |
|---|---|
| Tables | Hail 0.2 native `TypedCodecSpec` RVD components, globals, keys, partition bounds/counts |
| Scalars | Boolean, Int32/64, Float32/64, UTF-8 string, binary, packed integer Call |
| Nested types | Arrays, structs, tuples, sets, dictionaries, loci, intervals, missing values |
| NDArray | Column-major native encoding, exposed as `{shape, data}` with column-major flattened data |
| Compression | LZ4 fast/HC/legacy, zstd, size-based/raw variants, blocked/unblocked streams |
| Integers | Legacy LEB128 wrapper, little-endian fixed width, modern `EVarint` and v2 collection lengths |
| Block framing | Fixed-width and varint frame lengths; dense matrices in either stored orientation |
| Matrix sparsity | Column-major block IDs, `maybeFiltered`, absent blocks as zero |
| Indexes | Legacy `.idx` directories and inline-footer `.idx` files; B-tree ordinal row lookup |
| MatrixTable | Rows, columns, globals, and entry arrays including filtered/missing entries |

Locus-interval reads prune partitions using key bounds, then filter decoded
rows. They **do not yet seek within a selected partition by key**.
`read_row_range(start, end)` does use index B-trees and compressed range GETs
for ordinal reads; unindexed partitions are scanned in full.
GRCh37/GRCh38 contig order is embedded, including alternate contigs. For custom
references, conventional struct-encoded loci are decoded, and uncertain
cross-contig bounds are conservatively retained. Experimental packed loci
require an embedded GRCh37/GRCh38 reference.

Arrow mappings preserve Hail type metadata: loci/intervals/tuples are structs,
sets are lists, dictionaries are lists of `{key, value}` pairs (including
missing keys), and calls are their Hail packed Int32 representation rather
than genotype strings. Logical Hail fields are nullable in Arrow.

This is a reader, not the Hail expression or computation engine. Hail 0.1,
pre-`TypedCodecSpec` metadata, experimental RNG-state encodings, encrypted
formats, writers, and arbitrary future codecs are not supported. Unknown
encodings fail explicitly. Decoding bounds nesting to 128 and individual
collection lengths/NDArray element counts to 16,777,216.

## Verification and performance

```sh
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
uv run pytest -q
HAILRS_NETWORK=1 uv run pytest -q -s tests/test_panukb.py
```

The checked-in tiny fixtures were written and read back by real Hail 0.2.137.
They cover zstd/LEB128 and LZ4/fixed-width tables, nested types, missing values,
integer extremes, unicode, nonfinite/signed-zero floats, dense/transposed/sparse
BlockMatrices, and MatrixTable components. Rust also tests both physical dense
matrix orientations, modern scalar/frame encodings, inline index conversion,
path containment, and corrupt/truncated streams. No Java is needed to run tests.
`tests/fixtures/generate_fixtures.py` can regenerate fixtures in an environment
with Hail, Java 17, and Spark; it is not a runtime dependency.

The public S3 integration checks all 10 target variant indices and compares all
100 LD values to a real-Hail, Gentropy-style reference (with explicit unit
diagonal normalization). It also exercises the Hail 0.2.64 GRCh37 table's
legacy index directory; the GRCh38 table was written by Hail 0.2.95.

One macOS/Apple Silicon run with an optimized extension took **1.70 s for the
10 variants + 10.99 s for the 10x10 LD matrix = 12.69 s** including metadata
reads. It fetched one variant partition (747,548 bytes including metadata) and
one LD block (76,497,818 bytes including metadata), not the multi-TB matrix.
The supplied Hail comparison was approximately **41 s**, or about 3.2x slower
than this sample; this is a network-dependent observation, not a controlled
cross-engine benchmark. Successful request/byte counters make the transfer
scope observable.

## License and format references

MIT. The embedded GRCh37/GRCh38 reference definitions come from
[Hail](https://github.com/hail-is/hail/tree/main/hail/hail/resources/reference)
and retain Hail's MIT notice in `src/reference/LICENSE`. The reader follows
Hail's upstream `BufferSpecs`, `InputBuffers`, `EType`, `EBaseStruct`, `EArray`,
`DenseMatrix`, `BlockMatrix`, and `IndexReader` implementations.
