#!/usr/bin/env python3
"""Regenerate the Hail-written verification fixtures in tests/fixtures/.

These fixtures are written by real Hail (0.2.x, JVM/Spark backend) and are used
only to verify the JVM-free Rust reader. They are NOT runtime dependencies.

Usage (from the repository root, with a Python env that has Hail installed):

    export JAVA_HOME=/path/to/jdk17   # Hail 0.2.137 runs on Java 11/17
    export SPARK_LOCAL_IP=127.0.0.1
    python tests/fixtures/generate_fixtures.py

Outputs (all under tests/fixtures/):
    all_types.ht              table, default codec (LEB128 + blocking + zstd/lz4)
    all_types_nonleb.ht       same table, non-LEB128 codec (fixed-width ints)
    all_types.schema.json     row/global/key types as Hail type strings
    all_types.expected.jsonl  one hl.json(row) per line, in key order
    all_types.globals.json    hl.json(globals)
    bm_5x7_bs3.bm             dense 5x7 BlockMatrix, block_size 3
    bm_5x7_bs3_T.bm           its transpose (7x5); Hail 0.2.137 materializes the
                              transpose, so blocks are byte-identical with or
                              without force_row_major=True
    bm_sparse_band.bm         5x7 bs3 sparsified (sparsify_band), only some blocks
    bm_*.expected.csv         dense values (row-major, %.17g; absent blocks = 0)
    bm_sparse_band.expected.json  shape/block_size/present block indices
    mt_3x2.mt                 3 rows x 2 cols MatrixTable with missing entries
    mt_3x2.expected.json      globals, rows, cols, entries (hl.json), schema
"""

import json
import os
import shutil
import sys

import numpy as np

import hail as hl

HERE = os.path.dirname(os.path.abspath(__file__))

# Non-LEB128 buffer spec: blocking + LZ4 compression, fixed-width integer encoding.
NON_LEB_CODEC = json.dumps(
    {
        "name": "BlockingBufferSpec",
        "blockSize": 65536,
        "child": {
            "name": "LZ4FastBlockBufferSpec",
            "blockSize": 65536,
            "child": {"name": "StreamBlockBufferSpec"},
        },
    }
)


def out(name):
    return os.path.join(HERE, name)


def rm(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def init_hail():
    conf = {
        "spark.driver.host": "127.0.0.1",
        "spark.driver.bindAddress": "127.0.0.1",
        "spark.ui.enabled": "false",
    }
    hl.init(
        master="local[1]",
        quiet=True,
        spark_conf=conf,
        log=os.devnull,
    )


def make_table():
    na = hl.missing
    s_inner = hl.tstruct(a=hl.tint32, b=hl.tstr)
    s_outer = hl.tstruct(x=hl.tfloat64, inner=s_inner, tags=hl.tarray(hl.tstr))
    rows = [
        dict(
            idx=0,
            i32=-2147483648,
            i64=-9223372036854775808,
            f32=-1.5,
            f64=-0.1,
            b=True,
            s="plain ascii",
            arr_i32=[1, None, -3],
            arr_str=["a", None, ""],
            st=dict(x=1.25, inner=dict(a=-7, b="nested"), tags=["t1", None]),
            locus=hl.Locus("chr1", 12345, "GRCh38"),
            interval=hl.Interval(
                hl.Locus("chr1", 100, "GRCh38"), hl.Locus("chr1", 200, "GRCh38"), True, False
            ),
            int_interval=hl.Interval(-5, 10, includes_start=True, includes_end=True),
            set_i32={3, 1, 2},
            dict_str_i64={"one": 1, "minus": -1},
            tup=(7, "seven", None),
            call=hl.Call([0, 1], phased=False),
        ),
        dict(
            idx=1,
            i32=2147483647,
            i64=9223372036854775807,
            f32=None,
            f64=None,
            b=False,
            s="unicodé ✓ 漢字 🧬",
            arr_i32=[],
            arr_str=None,
            st=None,
            locus=hl.Locus("chrX", 1, "GRCh38"),
            interval=None,
            int_interval=hl.Interval(0, 0, includes_start=False, includes_end=False),
            set_i32=set(),
            dict_str_i64={},
            tup=None,
            call=hl.Call([1, 1], phased=True),
        ),
        dict(
            idx=2,
            i32=0,
            i64=-1,
            f32=3.0e38,
            f64=1.0e-300,
            b=None,
            s=None,
            arr_i32=None,
            arr_str=["x"],
            st=dict(x=None, inner=None, tags=None),
            locus=None,
            interval=hl.Interval(
                hl.Locus("chrM", 1, "GRCh38"), hl.Locus("chrM", 16569, "GRCh38"), False, True
            ),
            int_interval=None,
            set_i32=None,
            dict_str_i64={"null": None},
            tup=(None, None, 2.5),
            call=None,
        ),
        dict(
            idx=3,
            i32=-1,
            i64=4294967296,
            f32=0.0,
            f64=-0.0,
            b=True,
            s="",
            arr_i32=[None],
            arr_str=[],
            st=dict(x=-2.5, inner=dict(a=None, b=None), tags=[]),
            locus=hl.Locus("chr22", 50818468, "GRCh38"),
            interval=None,
            int_interval=hl.Interval(-2147483648, 2147483647, True, False),
            set_i32={-1},
            dict_str_i64={"z": 9223372036854775807, "a": -9223372036854775808},
            tup=(-1, "", 0.0),
            call=hl.Call([2], phased=False),
        ),
    ]
    schema = hl.tstruct(
        idx=hl.tint32,
        i32=hl.tint32,
        i64=hl.tint64,
        f32=hl.tfloat32,
        f64=hl.tfloat64,
        b=hl.tbool,
        s=hl.tstr,
        arr_i32=hl.tarray(hl.tint32),
        arr_str=hl.tarray(hl.tstr),
        st=s_outer,
        locus=hl.tlocus("GRCh38"),
        interval=hl.tinterval(hl.tlocus("GRCh38")),
        int_interval=hl.tinterval(hl.tint32),
        set_i32=hl.tset(hl.tint32),
        dict_str_i64=hl.tdict(hl.tstr, hl.tint64),
        tup=hl.ttuple(hl.tint32, hl.tstr, hl.tfloat64),
        call=hl.tcall,
    )
    ht = hl.Table.parallelize(rows, schema, key="idx", n_partitions=2)
    ht = ht.annotate_globals(
        g_name="fixture ✓",
        g_ver=hl.int64(42),
        g_arr=hl.array([hl.float64(1.5), na(hl.tfloat64)]),
        g_missing=na(hl.tint32),
    )
    return ht


def table_truth(ht, prefix):
    rows = ht.select(_json=hl.json(ht.row)).collect()
    with open(out(f"{prefix}.expected.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r._json + "\n")
    with open(out(f"{prefix}.globals.json"), "w", encoding="utf-8") as f:
        f.write(hl.eval(hl.json(ht.globals)) + "\n")
    with open(out(f"{prefix}.schema.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "row": str(ht.row.dtype),
                "globals": str(ht.globals.dtype),
                "key": list(ht.key),
                "n_rows": len(rows),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    return [json.loads(r._json) for r in rows]


def write_csv(path, arr):
    np.savetxt(path, arr, delimiter=",", fmt="%.17g")


def make_block_matrices():
    a = np.arange(35, dtype=np.float64).reshape(5, 7)
    a = (a - 17.0) / 4.0  # negative + fractional values
    a[1, 2] = np.nan
    a[4, 6] = -np.inf

    bm = hl.linalg.BlockMatrix.from_numpy(a, block_size=3)
    bm.write(out("bm_5x7_bs3.bm"), overwrite=True)
    write_csv(out("bm_5x7_bs3.expected.csv"), a)

    bm.T.write(out("bm_5x7_bs3_T.bm"), overwrite=True)
    write_csv(out("bm_5x7_bs3_T.expected.csv"), a.T)

    sp = bm.sparsify_band(lower=-1, upper=1, blocks_only=True)
    sp.write(out("bm_sparse_band.bm"), overwrite=True)
    sp_read = hl.linalg.BlockMatrix.read(out("bm_sparse_band.bm"))
    dense = sp_read.to_numpy()
    write_csv(out("bm_sparse_band.expected.csv"), dense)
    with open(out("bm_sparse_band.expected.json"), "w") as f:
        meta = json.load(open(os.path.join(out("bm_sparse_band.bm"), "metadata.json")))
        json.dump(
            {
                "shape": [5, 7],
                "block_size": 3,
                "sparsify": "sparsify_band(lower=-1, upper=1, blocks_only=True)",
                "present_block_indices_col_major": meta.get("maybeFiltered"),
                "absent_blocks_read_as": 0.0,
            },
            f,
            indent=2,
        )
        f.write("\n")

    for name, exp in [
        ("bm_5x7_bs3.bm", a),
        ("bm_5x7_bs3_T.bm", a.T),
        ("bm_sparse_band.bm", dense),
    ]:
        got = hl.linalg.BlockMatrix.read(out(name)).to_numpy()
        assert np.array_equal(got, exp, equal_nan=True), name


def make_matrix_table():
    mt = hl.utils.range_matrix_table(3, 2, n_partitions=2)
    mt = mt.annotate_globals(g_str="mt global", g_int=hl.int32(-3))
    mt = mt.annotate_rows(
        locus=hl.locus("chr1", (mt.row_idx + 1) * 1000, "GRCh38"),
        alleles=hl.array(["A", "T"]),
        rsid=hl.or_missing(mt.row_idx != 1, hl.str("rs") + hl.str(mt.row_idx)),
        af=hl.or_missing(mt.row_idx != 2, hl.float64(mt.row_idx) / 10.0),
    )
    mt = mt.annotate_cols(
        s=hl.str("sample_") + hl.str(mt.col_idx),
        pheno=hl.or_missing(mt.col_idx == 0, hl.struct(age=hl.int32(40), sex=hl.str("F"))),
    )
    mt = mt.annotate_entries(
        GT=hl.or_missing(
            (mt.row_idx + mt.col_idx) % 3 != 1,
            hl.call(mt.row_idx % 2, mt.col_idx % 2),
        ),
        DP=hl.or_missing(mt.row_idx != mt.col_idx, mt.row_idx * 10 + mt.col_idx),
        GQ=hl.float32(mt.row_idx) - hl.float32(mt.col_idx) / 2,
    )
    mt = mt.key_rows_by("locus", "alleles").key_cols_by("s")
    mt = mt.filter_entries((mt.row_idx == 2) & (mt.col_idx == 1), keep=False)
    mt.write(out("mt_3x2.mt"), overwrite=True)

    mt2 = hl.read_matrix_table(out("mt_3x2.mt"))
    rt = mt2.rows()
    rows = [json.loads(r._j) for r in rt.select(_j=hl.json(rt.row)).collect()]
    ct = mt2.cols()
    cols = [json.loads(r._j) for r in ct.select(_j=hl.json(ct.row)).collect()]
    ent = mt2.select_entries(_j=hl.json(mt2.entry))
    # Dense entry matrix (rows x cols); filtered entries are null.
    ent_rows = ent.localize_entries("_e").select("_e").collect()
    entries = [[None if e is None else json.loads(e._j) for e in r._e] for r in ent_rows]
    truth = {
        "schema": {
            "globals": str(mt2.globals.dtype),
            "row": str(mt2.row.dtype),
            "col": str(mt2.col.dtype),
            "entry": str(mt2.entry.dtype),
            "row_key": list(mt2.row_key),
            "col_key": list(mt2.col_key),
        },
        "n_rows": mt2.count_rows(),
        "n_cols": mt2.count_cols(),
        "globals": json.loads(hl.eval(hl.json(mt2.globals))),
        "rows": rows,
        "cols": cols,
        "entries": entries,
        "note": "entries[i][j] is null when the entry was filtered (row_idx=2,col_idx=1)",
    }
    with open(out("mt_3x2.expected.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    for name in [
        "all_types.ht",
        "all_types_nonleb.ht",
        "bm_5x7_bs3.bm",
        "bm_5x7_bs3_T.bm",
        "bm_sparse_band.bm",
        "mt_3x2.mt",
    ]:
        rm(out(name))
    init_hail()

    ht = make_table()
    ht.write(out("all_types.ht"), overwrite=True)
    ht.write(out("all_types_nonleb.ht"), overwrite=True, _codec_spec=NON_LEB_CODEC)
    expected = table_truth(hl.read_table(out("all_types.ht")), "all_types")
    nonleb = hl.read_table(out("all_types_nonleb.ht"))
    got = [json.loads(r._j) for r in nonleb.select(_j=hl.json(nonleb.row)).collect()]
    assert got == expected, "non-LEB table differs"

    make_block_matrices()
    make_matrix_table()

    # Drop Spark/Hadoop checksum side files to keep fixtures minimal.
    for root, _, files in os.walk(HERE):
        for fn in files:
            if fn.endswith(".crc"):
                os.remove(os.path.join(root, fn))
    hl.stop()
    print("fixtures written to", HERE, file=sys.stderr)


if __name__ == "__main__":
    main()
