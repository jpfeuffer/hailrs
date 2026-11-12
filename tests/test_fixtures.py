"""Verify hailrs against real Hail-written fixtures in tests/fixtures/.

Expected values come from Hail itself (see tests/fixtures/generate_fixtures.py):
``hl.json`` exports for tables/matrix tables and ``BlockMatrix.to_numpy`` CSVs.
Hail's JSON export differs from Hail's physical representation in a few places;
``normalize`` maps the export form onto the physical form the reader returns:

* tuples:    JSON array ``[a, b]``             -> struct ``{"0": a, "1": b}``
* intervals: ``includeStart``/``includeEnd``     -> ``includesStart``/``includesEnd``
* calls:     string ``"0/1"``, ``"1|1"``, ``"2"`` -> packed int32 (see ``call_to_int``)
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import hailrs
from hailrs import BlockMatrix, LocusInterval, MatrixTable, Table

FIXTURES = Path(__file__).parent / "fixtures"
SCRATCH = Path(__file__).parent / ".scratch"
ENTRIES = "the entries! [877f12a8827e18f61222c6c8c5fb04a8]"


# ---------------------------------------------------------------------------
# Call encoding, ported from Hail 0.2.137 is.hail.variant.{Call, Call1, Call2}
# and Genotype.diploidGtIndex. Cross-checked against the Hail JVM:
# Call2(0,1,false)=12, Call2(1,1,true)=37, Call1(2,false)=18,
# Call2(0,0,false)=4, Call2(1,1,false)=20, Call2(1,0,false)=12.
#
# This is a short Python re-expression of the bit layout and genotype-index
# formulas (no Scala code copied verbatim), pinned to the Hail commit that
# wrote the fixtures (hail_version 0.2.137-733ac4ccd943):
#   https://github.com/hail-is/hail/blob/733ac4ccd943526c8a5144bdce48c47302c56a9f/hail/hail/src/is/hail/variant/Call.scala
#   https://github.com/hail-is/hail/blob/733ac4ccd943526c8a5144bdce48c47302c56a9f/hail/hail/src/is/hail/variant/Genotype.scala
# Hail is MIT-licensed (not Apache-2.0). Since the logic is derived from it:
#   Copyright (c) 2015-2023, Hail Authors. All rights reserved.
#   Licensed under the MIT License:
#   https://github.com/hail-is/hail/blob/733ac4ccd943526c8a5144bdce48c47302c56a9f/LICENSE
# ---------------------------------------------------------------------------
def _diploid_gt_index(j: int, k: int) -> int:
    assert 0 <= j <= k
    return k * (k + 1) // 2 + j


def call_to_int(s: str) -> int:
    phased = "|" in s
    alleles = [int(a) for a in s.replace("|", "/").split("/") if a != "" and a != "-"]
    ploidy = len(alleles)
    if ploidy == 0:
        ar = 0
    elif ploidy == 1:
        ar = alleles[0]
    elif ploidy == 2:
        j, k = alleles
        ar = _diploid_gt_index(j, j + k) if phased else _diploid_gt_index(min(j, k), max(j, k))
    else:
        raise NotImplementedError(s)
    return (ar << 3) | (ploidy << 1) | int(phased)


@pytest.mark.parametrize(
    "s,expected", [("0/1", 12), ("1|1", 37), ("2", 18), ("0/0", 4), ("1/1", 20), ("1/0", 12)]
)
def test_call_encoding_matches_hail_jvm(s, expected):
    assert call_to_int(s) == expected


def normalize(value, typ: str):
    """Convert one ``hl.json`` value of Hail type string ``typ`` to reader form."""
    if value is None:
        return None
    typ = typ.strip()
    if typ == "call":
        return call_to_int(value)
    if typ.startswith("tuple("):
        elems = _split_args(typ[len("tuple("):-1])
        return {str(i): normalize(v, t) for i, (v, t) in enumerate(zip(value, elems))}
    if typ.startswith("interval<"):
        point = typ[len("interval<"):-1]
        return {
            "start": normalize(value["start"], point),
            "end": normalize(value["end"], point),
            "includesStart": value["includeStart"],
            "includesEnd": value["includeEnd"],
        }
    if typ.startswith(("array<", "set<")):
        elem = typ[typ.index("<") + 1:-1]
        return [normalize(v, elem) for v in value]
    if typ.startswith("dict<"):
        k, v = _split_args(typ[len("dict<"):-1])
        return [{"key": normalize(e["key"], k), "value": normalize(e["value"], v)} for e in value]
    if typ.startswith("struct{"):
        fields = _struct_fields(typ)
        return {name: normalize(value[name], t) for name, t in fields}
    return value


def _split_args(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<{(":
            depth += 1
        elif ch in ">})":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _struct_fields(typ: str) -> list[tuple[str, str]]:
    fields = []
    for part in _split_args(typ[len("struct{"):-1]):
        name, t = part.split(":", 1)
        fields.append((name.strip().strip("`"), t.strip()))
    return fields


def assert_same(actual, expected, path="$"):
    """Deep equality; floats compared bit-faithfully (NaN, inf, signed zero)."""
    if isinstance(expected, float) or (isinstance(actual, float) and isinstance(expected, int)
                                       and not isinstance(expected, bool)):
        assert isinstance(actual, (float, int)) and not isinstance(actual, bool), f"{path}: {actual!r}"
        a, e = float(actual), float(expected)
        if math.isnan(e):
            assert math.isnan(a), f"{path}: {a!r} != nan"
        else:
            assert a == e, f"{path}: {a!r} != {e!r}"
            assert math.copysign(1.0, a) == math.copysign(1.0, e), f"{path}: sign of zero {a!r} != {e!r}"
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: {actual!r} is not a struct"
        assert set(actual) == set(expected), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for k in expected:
            assert_same(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: {actual!r} is not a list"
        assert len(actual) == len(expected), f"{path}: len {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_same(a, e, f"{path}[{i}]")
    else:
        assert type(actual) is type(expected) or expected is None, f"{path}: {actual!r} vs {expected!r}"
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def assert_matrix(actual: np.ndarray, expected: np.ndarray):
    assert actual.dtype == np.float64
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)  # treats NaN == NaN
    assert np.array_equal(np.signbit(actual), np.signbit(expected))


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
TABLES = ["all_types.ht", "all_types_nonleb.ht"]
SCHEMA = json.loads((FIXTURES / "all_types.schema.json").read_text())
ROW_TYPE = SCHEMA["row"]
EXPECTED_ROWS = [
    normalize(json.loads(line), ROW_TYPE)
    for line in (FIXTURES / "all_types.expected.jsonl").read_text(encoding="utf-8").splitlines()
]
EXPECTED_GLOBALS = normalize(
    json.loads((FIXTURES / "all_types.globals.json").read_text(encoding="utf-8")), SCHEMA["globals"]
)

LOCUS = pa.struct([("contig", pa.string()), ("position", pa.int32())])
EXPECTED_ARROW = pa.schema([
    ("idx", pa.int32()),
    ("i32", pa.int32()),
    ("i64", pa.int64()),
    ("f32", pa.float32()),
    ("f64", pa.float64()),
    ("b", pa.bool_()),
    ("s", pa.string()),
    ("arr_i32", pa.list_(pa.int32())),
    ("arr_str", pa.list_(pa.string())),
    ("st", pa.struct([
        ("x", pa.float64()),
        ("inner", pa.struct([("a", pa.int32()), ("b", pa.string())])),
        ("tags", pa.list_(pa.string())),
    ])),
    ("locus", LOCUS),
    ("interval", pa.struct([
        ("start", LOCUS), ("end", LOCUS), ("includesStart", pa.bool_()), ("includesEnd", pa.bool_()),
    ])),
    ("int_interval", pa.struct([
        ("start", pa.int32()), ("end", pa.int32()),
        ("includesStart", pa.bool_()), ("includesEnd", pa.bool_()),
    ])),
    ("set_i32", pa.list_(pa.int32())),
    ("dict_str_i64", pa.list_(pa.struct([("key", pa.string()), ("value", pa.int64())]))),
    ("tup", pa.struct([("0", pa.int32()), ("1", pa.string()), ("2", pa.float64())])),
    ("call", pa.int32()),
])


@pytest.fixture(params=TABLES)
def table(request) -> Table:
    return Table(str(FIXTURES / request.param))


def test_table_metadata(table):
    assert table.n_partitions == 2
    assert [b["start"] for b in table.partition_bounds] == [{"idx": 0}, {"idx": 2}]
    assert [b["end"] for b in table.partition_bounds] == [{"idx": 1}, {"idx": 3}]


def test_table_hail_codec_is_as_expected():
    import gzip
    specs = {
        t: json.loads(gzip.decompress((FIXTURES / t / "rows" / "metadata.json.gz").read_bytes()))
        for t in TABLES
    }
    assert specs["all_types.ht"]["_codecSpec"]["_bufferSpec"]["name"] == "LEB128BufferSpec"
    assert specs["all_types_nonleb.ht"]["_codecSpec"]["_bufferSpec"]["name"] == "BlockingBufferSpec"


def test_table_arrow_schema(table):
    assert table.schema.remove_metadata().equals(EXPECTED_ARROW), table.schema
    assert b"hail_type" in table.schema.metadata


def test_table_rows(table):
    assert_same(table.read_rows(), EXPECTED_ROWS)


def test_table_globals(table):
    assert_same(table.globals, EXPECTED_GLOBALS)


def test_table_arrow_roundtrip(table):
    arrow = table.read()
    assert arrow.schema.remove_metadata().equals(EXPECTED_ARROW)
    assert arrow.num_rows == len(EXPECTED_ROWS)
    assert_same(arrow.to_pylist(), EXPECTED_ROWS)


def test_table_specific_values(table):
    rows = {r["idx"]: r for r in table.read_rows()}
    assert rows[0]["call"] == 12 and rows[1]["call"] == 37 and rows[3]["call"] == 18
    assert rows[0]["tup"] == {"0": 7, "1": "seven", "2": None}
    assert rows[1]["s"] == "unicodé ✓ 漢字 🧬"
    assert rows[0]["i64"] == -(2**63) and rows[1]["i64"] == 2**63 - 1
    assert math.copysign(1.0, rows[3]["f64"]) == -1.0
    assert rows[2]["f32"] == float(np.float32(3.0e38))


@pytest.mark.parametrize("parts,idxs", [([0], [0, 1]), ([1], [2, 3]), ([1, 0], None), ([], [])])
def test_table_read_partitions(table, parts, idxs):
    got = [r["idx"] for r in table.read_rows(partitions=parts)]
    if idxs is None:  # order of requested partitions: either preserved or sorted, rows must be complete
        assert sorted(got) == [0, 1, 2, 3]
    else:
        assert got == idxs


def test_table_read_partition(table):
    assert_same(table.read_partition(0).to_pylist(), EXPECTED_ROWS[:2])
    assert_same(table.read_partition(1).to_pylist(), EXPECTED_ROWS[2:])


def test_table_limit(table):
    assert_same(table.read_rows(limit=3), EXPECTED_ROWS[:3])
    assert table.read_rows(limit=0) == []


@pytest.mark.parametrize("concurrency", [1, 2, 8, 64])
def test_table_concurrency_invariant(table, concurrency):
    assert_same(table.read_rows(concurrency=concurrency), EXPECTED_ROWS)


@pytest.mark.parametrize("batch_size,n_batches", [(1, 4), (2, 2), (65536, 2)])
def test_table_iter_batches(table, batch_size, n_batches):
    batches = list(table.iter_batches(batch_size=batch_size))
    assert len(batches) == n_batches
    assert all(b.num_rows <= batch_size for b in batches)
    assert all(b.schema.remove_metadata().equals(EXPECTED_ARROW) for b in batches)
    assert_same(pa.Table.from_batches(batches).to_pylist(), EXPECTED_ROWS)


def test_table_iter_batches_partition_subset(table):
    batches = list(table.iter_batches(partitions=[1]))
    assert_same(pa.Table.from_batches(batches, schema=table.schema).to_pylist(), EXPECTED_ROWS[2:])


def test_table_errors(table):
    with pytest.raises(Exception):
        table.read_rows(partitions=[2])
    with pytest.raises(Exception):
        table.read_partition(2)
    with pytest.raises(Exception):
        table.read_rows(concurrency=0)
    with pytest.raises(ValueError):
        next(table.iter_batches(batch_size=0))


def test_missing_table_errors():
    with pytest.raises(Exception):
        Table(str(FIXTURES / "does_not_exist.ht"))
    with pytest.raises(Exception):
        Table(str(FIXTURES / "bm_5x7_bs3.bm"))
    with pytest.raises(Exception):
        BlockMatrix(str(FIXTURES / "all_types.ht"))


# ---------------------------------------------------------------------------
# BlockMatrix
# ---------------------------------------------------------------------------
def load_csv(name: str) -> np.ndarray:
    return np.loadtxt(FIXTURES / name, delimiter=",", dtype=np.float64, ndmin=2)


BMS = {
    "bm_5x7_bs3.bm": load_csv("bm_5x7_bs3.expected.csv"),
    "bm_5x7_bs3_T.bm": load_csv("bm_5x7_bs3_T.expected.csv"),
    "bm_sparse_band.bm": load_csv("bm_sparse_band.expected.csv"),
}


@pytest.mark.parametrize("name", BMS)
def test_bm_full(name):
    bm = BlockMatrix(str(FIXTURES / name))
    expected = BMS[name]
    assert bm.shape == expected.shape
    assert bm.block_size == 3
    assert_matrix(bm.read(), expected)


@pytest.mark.parametrize("name", BMS)
@pytest.mark.parametrize("rows,cols", [
    ([4, 0, 2], [4, 0, 3]),            # reversed / unordered across blocks
    ([1, 1, 1], [2, 2]),               # duplicates (includes the NaN cell)
    ([0], [0]),
    ([], [0, 1]),
    (slice(None, None, -1), slice(1, None, 2)),
])
def test_bm_subset(name, rows, cols):
    bm = BlockMatrix(str(FIXTURES / name))
    expected = BMS[name]
    r = bm._indices(rows, bm.shape[0])
    c = bm._indices(cols, bm.shape[1])
    want = expected[np.ix_(r, c)] if r and c else np.empty((len(r), len(c)))
    assert_matrix(bm.read(rows, cols), want)


def test_bm_transpose_matches():
    assert_matrix(BlockMatrix(str(FIXTURES / "bm_5x7_bs3_T.bm")).read(), BMS["bm_5x7_bs3.bm"].T)


def test_bm_nonfinite():
    m = BlockMatrix(str(FIXTURES / "bm_5x7_bs3.bm")).read([1, 4], [2, 6])
    assert math.isnan(m[0, 0]) and m[1, 1] == -math.inf


def test_bm_sparse_absent_blocks_are_zero_without_fetch():
    meta = json.loads((FIXTURES / "bm_sparse_band.expected.json").read_text())
    bm = BlockMatrix(str(FIXTURES / "bm_sparse_band.bm"))
    assert bm.metadata["maybeFiltered"] == meta["present_block_indices_col_major"]
    before = bm.stats["requests"]
    out = bm.read([0, 1, 2], [6])  # block (0, 2) is absent
    assert_matrix(out, np.zeros((3, 1)))
    assert bm.stats["requests"] == before
    # Present block (1, 2) still fetched and read correctly.
    assert_matrix(bm.read([3, 4], [6]), BMS["bm_sparse_band.bm"][[3, 4]][:, [6]])


def test_bm_reads_only_needed_blocks():
    bm = BlockMatrix(str(FIXTURES / "bm_5x7_bs3.bm"))
    before = bm.stats["requests"]
    bm.read([0, 2, 1], [0, 1])
    assert bm.stats["requests"] - before == 1


@pytest.mark.parametrize("concurrency", [1, 2, 16])
def test_bm_concurrency_invariant(concurrency):
    bm = BlockMatrix(str(FIXTURES / "bm_5x7_bs3.bm"))
    assert_matrix(bm.read(concurrency=concurrency), BMS["bm_5x7_bs3.bm"])


def test_bm_errors():
    bm = BlockMatrix(str(FIXTURES / "bm_5x7_bs3.bm"))
    with pytest.raises(Exception):
        bm.read([5], [0])
    with pytest.raises(Exception):
        bm.read([0], [7])
    with pytest.raises(Exception):
        bm.read([-1], [0])
    with pytest.raises(Exception):
        bm.read(concurrency=0)
    with pytest.raises(Exception):
        bm.read(symmetric_upper=True)  # 5x7 is not square


# ---------------------------------------------------------------------------
# MatrixTable
# ---------------------------------------------------------------------------
MT_TRUTH = json.loads((FIXTURES / "mt_3x2.expected.json").read_text(encoding="utf-8"))
MT_SCHEMA = MT_TRUTH["schema"]


@pytest.fixture
def mt() -> MatrixTable:
    return MatrixTable(str(FIXTURES / "mt_3x2.mt"))


def test_mt_globals(mt):
    assert_same(mt.globals, normalize(MT_TRUTH["globals"], MT_SCHEMA["globals"]))


def test_mt_rows(mt):
    assert_same(mt.rows.read_rows(), [normalize(r, MT_SCHEMA["row"]) for r in MT_TRUTH["rows"]])
    assert mt.rows.n_partitions == mt.entries.n_partitions == 2


def test_mt_cols(mt):
    assert_same(mt.cols.read_rows(), [normalize(c, MT_SCHEMA["col"]) for c in MT_TRUTH["cols"]])


def test_mt_entries(mt):
    expected = [
        {ENTRIES: [normalize(e, MT_SCHEMA["entry"]) for e in row]} for row in MT_TRUTH["entries"]
    ]
    got = mt.entries.read_rows()
    assert_same(got, expected)
    assert got[2][ENTRIES][1] is None  # filtered entry
    assert got[0][ENTRIES][0]["GT"] == 4 and got[1][ENTRIES][1]["GT"] == 20


def test_mt_entries_arrow(mt):
    entry = pa.struct([("GT", pa.int32()), ("DP", pa.int32()), ("GQ", pa.float32())])
    assert mt.entries.schema.remove_metadata().equals(pa.schema([(ENTRIES, pa.list_(entry))]))
    arrow = mt.entries.read()
    assert arrow.num_rows == MT_TRUTH["n_rows"]
    assert all(len(r) == MT_TRUTH["n_cols"] for r in arrow.column(ENTRIES).to_pylist())


def test_mt_entries_partitions_align_with_rows(mt):
    for p in range(mt.rows.n_partitions):
        rows = mt.rows.read_rows(partitions=[p])
        ents = mt.entries.read_rows(partitions=[p])
        assert len(rows) == len(ents)
        for r, e in zip(rows, ents):
            exp = MT_TRUTH["entries"][r["row_idx"]]
            assert_same(e[ENTRIES], [normalize(x, MT_SCHEMA["entry"]) for x in exp])


@pytest.mark.parametrize("start,end,positions", [
    (1, 1_000_000, [1000, 2000, 3000]),
    (1000, 3000, [1000, 2000]),       # half-open: 3000 excluded
    (1500, 2500, [2000]),
    (3000, 3001, [3000]),
    (3001, 9999, []),
])
def test_mt_rows_locus_interval(mt, start, end, positions):
    rows = mt.rows.read_rows(interval=LocusInterval("chr1", start, end))
    assert [r["locus"]["position"] for r in rows] == positions


def test_mt_rows_other_contig_empty(mt):
    assert mt.rows.read_rows(interval=LocusInterval("chr2", 1, 10_000)) == []


def test_mt_interval_prunes_partitions(mt):
    before = mt.rows.stats["requests"]
    mt.rows.read_rows(interval=LocusInterval("chr1", 3000, 3001))
    assert mt.rows.stats["requests"] - before == 1  # only partition 1 fetched


def test_mt_iter_batches_interval(mt):
    batches = list(mt.rows.iter_batches(interval=LocusInterval("chr1", 1500, 3001), batch_size=1))
    got = pa.Table.from_batches(batches, schema=mt.rows.schema).to_pylist()
    assert [r["locus"]["position"] for r in got] == [2000, 3000]


# ---------------------------------------------------------------------------
# Corruption / truncation
# ---------------------------------------------------------------------------
@pytest.fixture
def scratch(request):
    path = SCRATCH / request.node.name.replace("[", "_").replace("]", "")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
    if SCRATCH.exists() and not any(SCRATCH.iterdir()):
        SCRATCH.rmdir()


def _truncate(path: Path, keep: float = 0.5):
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * keep)])


@pytest.mark.parametrize("name", TABLES)
def test_truncated_table_partition_errors(scratch, name):
    dst = scratch / name
    shutil.copytree(FIXTURES / name, dst)
    part = sorted((dst / "rows" / "parts").iterdir())[1]
    _truncate(part)
    t = Table(str(dst))
    assert_same(t.read_rows(partitions=[0]), EXPECTED_ROWS[:2])  # intact partition still readable
    with pytest.raises(Exception):
        t.read_rows(partitions=[1])


def test_truncated_bm_block_errors(scratch):
    dst = scratch / "bm.bm"
    shutil.copytree(FIXTURES / "bm_5x7_bs3.bm", dst)
    _truncate(sorted((dst / "parts").iterdir())[0])
    bm = BlockMatrix(str(dst))
    assert_matrix(bm.read([3], [3]), BMS["bm_5x7_bs3.bm"][[3]][:, [3]])
    with pytest.raises(Exception):
        bm.read([0], [0])


# ---------------------------------------------------------------------------
# Indexed row-range reads (seek via index metadata + B-tree, ranged GETs)
# ---------------------------------------------------------------------------
def _all_ranges(n: int):
    return [(a, b) for a in range(n + 1) for b in range(a, n + 1)]


@pytest.mark.parametrize("name", TABLES)
@pytest.mark.parametrize("start,end", _all_ranges(4))
def test_table_read_row_range(name, start, end):
    t = Table(str(FIXTURES / name))
    got = t.read_row_range(start, end)
    assert isinstance(got, pa.Table)
    assert got.schema.remove_metadata().equals(EXPECTED_ARROW)
    assert_same(got.to_pylist(), EXPECTED_ROWS[start:end])
    assert_same(got.to_pylist(), t.read_rows()[start:end])


@pytest.mark.parametrize("name", TABLES)
@pytest.mark.parametrize("concurrency", [1, 2, 8, 64])
def test_table_read_row_range_concurrency(name, concurrency):
    t = Table(str(FIXTURES / name))
    assert_same(t.read_row_range(1, 3, concurrency=concurrency).to_pylist(), EXPECTED_ROWS[1:3])


def test_table_read_row_range_within_partition_fetches_one_part():
    t = Table(str(FIXTURES / "all_types.ht"))
    before = t.stats["requests"]
    assert_same(t.read_row_range(3, 4).to_pylist(), EXPECTED_ROWS[3:4])
    after_one = t.stats["requests"] - before
    before = t.stats["requests"]
    t.read_row_range(0, 4)
    after_all = t.stats["requests"] - before
    assert after_one < after_all


MT_ENTRY_ROWS = [
    {ENTRIES: [normalize(e, MT_SCHEMA["entry"]) for e in row]} for row in MT_TRUTH["entries"]
]


@pytest.mark.parametrize("component", ["rows", "entries", "cols", "_globals"])
def test_mt_component_read_row_range_all(component):
    mt = MatrixTable(str(FIXTURES / "mt_3x2.mt"))
    t = getattr(mt, component)
    full = t.read_rows()
    for start, end in _all_ranges(len(full)):
        assert_same(t.read_row_range(start, end).to_pylist(), full[start:end], f"{component}[{start}:{end}]")


@pytest.mark.parametrize("start,end", _all_ranges(3))
def test_mt_entries_read_row_range(start, end):
    entries = MatrixTable(str(FIXTURES / "mt_3x2.mt")).entries
    got = entries.read_row_range(start, end).to_pylist()
    assert_same(got, MT_ENTRY_ROWS[start:end])
    if start <= 2 < end:
        assert got[2 - start][ENTRIES][1] is None  # filtered entry across partition boundary (2|1)


def _open_range_target(name: str) -> Table:
    if name.startswith("mt:"):
        return getattr(MatrixTable(str(FIXTURES / "mt_3x2.mt")), name[3:])
    return Table(str(FIXTURES / name))


@pytest.mark.parametrize("name", TABLES + ["mt:entries", "mt:rows"])
@pytest.mark.parametrize("start,end", [(0, 99), (5, 5), (5, 6), (3, 2), (-1, 2), (0, -1)])
def test_read_row_range_out_of_range_errors(name, start, end):
    t = _open_range_target(name)
    with pytest.raises((RuntimeError, ValueError, IndexError, OverflowError)):
        t.read_row_range(start, end)


@pytest.mark.parametrize("name", TABLES)
def test_read_row_range_zero_concurrency_errors(name):
    with pytest.raises(Exception):
        Table(str(FIXTURES / name)).read_row_range(0, 1, concurrency=0)


@pytest.mark.parametrize("name", TABLES)
def test_read_row_range_truncated_part_errors(scratch, name):
    dst = scratch / name
    shutil.copytree(FIXTURES / name, dst)
    _truncate(sorted((dst / "rows" / "parts").iterdir())[1], keep=0.3)
    t = Table(str(dst))
    assert_same(t.read_row_range(0, 2).to_pylist(), EXPECTED_ROWS[:2])
    with pytest.raises(Exception):
        t.read_row_range(2, 4)


def test_public_api_exports():
    assert set(hailrs.__all__) >= {"BlockMatrix", "Table", "MatrixTable", "LocusInterval"}
