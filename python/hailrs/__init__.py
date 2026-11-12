"""JVM-free readers for native Hail .ht, .mt and .bm datasets."""

from __future__ import annotations

import json
import operator
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from ._native import NativeBlockMatrix, NativeMatrixTable, NativeTable

__all__ = ["BlockMatrix", "Table", "MatrixTable", "LocusInterval", "StorageOptions"]


@dataclass(frozen=True)
class StorageOptions:
    anonymous: bool = False
    region: str | None = None
    requester_pays_project: str | None = None
    endpoint: str | None = None

    def _json(self) -> str:
        return json.dumps(vars(self))


@dataclass(frozen=True)
class LocusInterval:
    """One-based half-open locus interval [start, end)."""

    contig: str
    start: int
    end: int
    field: str = "locus"


def _arrow_type(typ: dict) -> pa.DataType:
    name, args, fields = typ["name"], typ["args"], typ["fields"]
    primitives = {
        "Int32": pa.int32(), "Int64": pa.int64(),
        "Float32": pa.float32(), "Float64": pa.float64(),
        "Boolean": pa.bool_(), "String": pa.string(), "Binary": pa.binary(),
        "Call": pa.int32(),
    }
    if name in primitives:
        return primitives[name]
    if name == "Struct":
        return pa.struct([pa.field(n, _arrow_type(t), nullable=not t["required"]) for n, t in fields])
    if name in ("Array", "Set"):
        return pa.list_(_arrow_type(args[0]))
    if name == "Dict":
        # Hail permits missing keys; Arrow maps do not. Keep ordered pairs.
        return pa.list_(pa.struct([("key", _arrow_type(args[0])), ("value", _arrow_type(args[1]))]))
    if name == "Locus":
        return pa.struct([("contig", pa.string()), ("position", pa.int32())])
    if name == "Interval":
        return pa.struct([
            ("start", _arrow_type(args[0])), ("end", _arrow_type(args[0])),
            ("includesStart", pa.bool_()), ("includesEnd", pa.bool_()),
        ])
    if name == "Tuple":
        return pa.struct([(str(i), _arrow_type(t)) for i, t in enumerate(args)])
    if name == "NDArray":
        return pa.struct([("shape", pa.list_(pa.int64())), ("data", pa.list_(_arrow_type(args[0])))])
    raise NotImplementedError(f"Arrow export for Hail type {name}")


class Table:
    """A native Hail Table; opening loads metadata, not data partitions."""

    def __init__(self, uri: str, *, storage: StorageOptions | None = None):
        self._native = NativeTable(str(uri), (storage or StorageOptions())._json())
        self._initialize_schema()

    @classmethod
    def _from_native(cls, native: NativeTable):
        table = cls.__new__(cls)
        table._native = native
        table._initialize_schema()
        return table

    def _initialize_schema(self):
        self.metadata = json.loads(self._native.metadata_json)
        self.hail_type = json.loads(self._native.schema_json)
        self.schema = pa.schema(
            list(_arrow_type(self.hail_type)),
            metadata={b"hail_type": self._native.schema_json.encode()},
        )

    @property
    def n_partitions(self) -> int:
        return self._native.n_partitions

    @property
    def partition_bounds(self) -> list:
        return json.loads(self._native.bounds_json)

    @property
    def stats(self) -> dict:
        """Cumulative successful request/byte counts, including metadata."""
        return json.loads(self._native.stats_json())

    @property
    def globals(self) -> dict:
        return self._native.globals()

    @staticmethod
    def _read_options(partitions, interval, limit, concurrency) -> str:
        return json.dumps({
            "partitions": None if partitions is None else list(partitions),
            "interval": None if interval is None else vars(interval),
            "limit": limit,
            "concurrency": concurrency,
        })

    def read_rows(
        self, *, partitions: Iterable[int] | None = None,
        interval: LocusInterval | None = None, limit: int | None = None,
        concurrency: int = 8,
    ) -> list[dict]:
        """Read rows, pruning partitions by locus bounds before fetching."""
        return self._native.read(self._read_options(partitions, interval, limit, concurrency))

    def read(self, **kwargs) -> pa.Table:
        """Read a typed Arrow Table. Use iter_batches for bounded-memory scans."""
        return pa.Table.from_pylist(self.read_rows(**kwargs), schema=self.schema)

    def read_partition(self, partition: int) -> pa.Table:
        return pa.Table.from_pylist(self._native.read_partition(partition), schema=self.schema)

    def read_row_range(self, start: int, end: int, *, concurrency: int = 8) -> pa.Table:
        """Read global ordinal rows [start, end), seeking via indexes if present."""
        return pa.Table.from_pylist(
            self._native.read_row_range(start, end, concurrency), schema=self.schema,
        )

    def iter_batches(
        self, *, partitions: Iterable[int] | None = None,
        interval: LocusInterval | None = None, batch_size: int = 65536,
    ) -> Iterator[pa.RecordBatch]:
        """Fetch one partition at a time; split its Arrow result into batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected = self._native.selected_partitions(self._read_options(partitions, interval, None, 1))
        for partition in selected:
            rows = self.read_rows(partitions=[partition], interval=interval, concurrency=1)
            yield from pa.Table.from_pylist(rows, schema=self.schema).to_batches(max_chunksize=batch_size)


class BlockMatrix:
    """Selective native BlockMatrix reads, fetching only selected block files."""

    def __init__(self, uri: str, *, storage: StorageOptions | None = None):
        self._native = NativeBlockMatrix(str(uri), (storage or StorageOptions())._json())
        self.metadata = json.loads(self._native.metadata_json)
        self.shape = (self.metadata["nRows"], self.metadata["nCols"])
        self.block_size = self.metadata["blockSize"]

    @property
    def stats(self) -> dict:
        return json.loads(self._native.stats_json())

    @staticmethod
    def _indices(selection, size) -> list[int]:
        if selection is None:
            return list(range(size))
        if isinstance(selection, slice):
            return list(range(*selection.indices(size)))
        return [operator.index(i) for i in selection]

    def read(
        self, rows: Iterable[int] | slice | None = None,
        cols: Iterable[int] | slice | None = None, *,
        concurrency: int = 4, symmetric_upper: bool = False,
    ) -> np.ndarray:
        """Return float64 data in requested order, with absent sparse blocks zero.

        symmetric_upper explicitly reconstructs a symmetric matrix from stored
        upper-triangle values, as used in the Pan-UKBB LD release.
        """
        return self._native.read(
            self._indices(rows, self.shape[0]), self._indices(cols, self.shape[1]),
            concurrency, symmetric_upper,
        )


class MatrixTable:
    """Native MatrixTable components, without flattening entry arrays.

    Entries are arrays in column order. To subset by row, select row partitions
    and read the matching entry partitions, then apply the same row mask.
    """

    def __init__(self, uri: str, *, storage: StorageOptions | None = None):
        matrix = NativeMatrixTable(str(uri), (storage or StorageOptions())._json())
        self.rows = Table._from_native(matrix.rows)
        self.cols = Table._from_native(matrix.cols)
        self.entries = Table._from_native(matrix.entries)
        self._globals = Table._from_native(matrix.globals)

    @property
    def globals(self) -> dict:
        return self._globals.read_rows()[0]
