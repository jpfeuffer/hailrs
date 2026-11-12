"""Opt-in real anonymous S3 integration; never fetches the complete LD matrix."""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from hailrs import BlockMatrix, LocusInterval, StorageOptions, Table

pytestmark = pytest.mark.skipif(
    os.environ.get("HAILRS_NETWORK") != "1", reason="set HAILRS_NETWORK=1 for public S3 integration",
)


def test_panukb_variants_and_ld():
    storage = StorageOptions(anonymous=True, region="us-east-1")
    root = "s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj"
    started = time.perf_counter()
    table = Table(root + ".variant.b38.ht", storage=storage)
    rows = table.read(interval=LocusInterval("chr22", 20012563, 20013428)).to_pylist()
    assert len(rows) == 10
    indices = [row["idx"] for row in rows]
    assert indices == list(range(22893433, 22893443))
    assert rows[0]["alleles"] == ["T", "C"]
    assert rows[0]["original_locus"]["position"] == 20000086
    assert rows[1]["locus"]["position"] == 20012623
    assert rows[1]["alleles"] == ["G", "A"]
    assert rows[-1]["locus"]["position"] == 20013427
    assert rows[-1]["alleles"] == ["T", "G"]
    variant_seconds = time.perf_counter() - started

    started = time.perf_counter()
    matrix = BlockMatrix(root + ".bm", storage=storage)
    actual = matrix.read(indices, indices, symmetric_upper=True)
    matrix_seconds = time.perf_counter() - started
    with (Path(__file__).parent / "fixtures/panukb_ld.csv").open() as handle:
        reference = list(csv.reader(handle))
    expected_ids = [
        f"22_{row['locus']['position']}_{row['alleles'][0]}_{row['alleles'][1]}" for row in rows
    ]
    assert reference[0][1:] == expected_ids
    assert [row[0] for row in reference[1:]] == expected_ids
    expected = np.array([[float(value) for value in row[1:]] for row in reference[1:]])
    # The supplied Gentropy-style reference sets the diagonal to one. The
    # native reader deliberately preserves the original LD-adjusted diagonal.
    assert np.all((actual.diagonal() > 0.999) & (actual.diagonal() <= 1.0))
    np.fill_diagonal(actual, 1.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert matrix.stats["requests"] == 3  # Success marker, metadata, precisely one block.
    assert matrix.stats["bytes"] < 100_000_000
    assert table.stats["requests"] == 5  # Success marker, three metadata files, one data partition.
    print(json.dumps({
        "variant_seconds": variant_seconds, "matrix_seconds": matrix_seconds,
        "total_seconds": variant_seconds + matrix_seconds,
        "variant_io": table.stats, "matrix_io": matrix.stats,
    }, indent=2))


def test_legacy_panukb_table_and_index():
    storage = StorageOptions(anonymous=True, region="us-east-1")
    table = Table("s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldadj.variant.ht", storage=storage)
    rows = table.read_rows(interval=LocusInterval("22", 20000086, 20001000))
    assert rows[0]["idx"] == 22893433
    assert rows[0]["alleles"] == ["T", "C"]
    # This file's original idx is its ordinal row index; exercises legacy idx directories.
    direct = table.read_row_range(22893433, 22893443).to_pylist()
    assert [row["idx"] for row in direct] == list(range(22893433, 22893443))
    assert direct[0] == rows[0]
