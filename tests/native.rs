use hailrs::{BlockMatrix, MatrixTable, ReadOptions, StorageOptions, Table, Value};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

#[tokio::test]
async fn real_hail_tables_and_indexes() {
    for name in ["all_types.ht", "all_types_nonleb.ht"] {
        let table = Table::open(&fixture(name), StorageOptions::default())
            .await
            .unwrap();
        let rows = table.read(ReadOptions::default()).await.unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].field("i32"), Some(&Value::Int(i32::MIN as i64)));
        assert_eq!(rows[1].field("i64"), Some(&Value::Int(i64::MAX)));
        assert_eq!(
            table.globals().await.unwrap().field("g_ver"),
            Some(&Value::Int(42))
        );
        for start in 0..=4 {
            for end in start..=4 {
                assert_eq!(
                    table.read_row_range(start..end, 2).await.unwrap(),
                    rows[start..end]
                );
            }
        }
    }
}

#[tokio::test]
async fn real_hail_block_matrices() {
    for name in ["bm_5x7_bs3", "bm_5x7_bs3_T", "bm_sparse_band"] {
        let matrix = BlockMatrix::open(&fixture(&format!("{name}.bm")), StorageOptions::default())
            .await
            .unwrap();
        let expected = std::fs::read_to_string(fixture(&format!("{name}.expected.csv"))).unwrap();
        let expected: Vec<f64> = expected
            .lines()
            .flat_map(|line| line.split(',').map(|x| x.parse::<f64>().unwrap()))
            .collect();
        let rows: Vec<_> = (0..matrix.metadata.n_rows).collect();
        let cols: Vec<_> = (0..matrix.metadata.n_cols).collect();
        let result = matrix.read(&rows, &cols, 2).await.unwrap();
        assert_eq!(result.data.len(), expected.len());
        for (actual, expected) in result.data.iter().zip(expected) {
            assert!(actual == &expected || (actual.is_nan() && expected.is_nan()));
        }
    }
}

#[tokio::test]
async fn real_hail_matrix_table_components() {
    let mt = MatrixTable::open(&fixture("mt_3x2.mt"), StorageOptions::default())
        .await
        .unwrap();
    assert_eq!(mt.rows.read(ReadOptions::default()).await.unwrap().len(), 3);
    assert_eq!(mt.cols.read(ReadOptions::default()).await.unwrap().len(), 2);
    assert_eq!(
        mt.entries.read(ReadOptions::default()).await.unwrap().len(),
        3
    );
}

#[tokio::test]
async fn physical_matrix_orientations() {
    for row_major in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("parts")).unwrap();
        std::fs::write(dir.path().join("_SUCCESS"), []).unwrap();
        std::fs::write(
            dir.path().join("metadata.json"),
            r#"{"blockSize":3,"nRows":2,"nCols":3,"maybeFiltered":null,"partFiles":["part-0"]}"#,
        )
        .unwrap();
        let values: Vec<f64> = if row_major {
            vec![0., 1., 2., 3., 4., 5.]
        } else {
            vec![0., 3., 1., 4., 2., 5.]
        };
        let mut raw = Vec::new();
        raw.extend(2i32.to_le_bytes());
        raw.extend(3i32.to_le_bytes());
        raw.push(u8::from(row_major));
        for value in values {
            raw.extend(value.to_le_bytes());
        }
        let compressed = lz4_flex::block::compress(&raw);
        let mut part = ((compressed.len() + 4) as u32).to_le_bytes().to_vec();
        part.extend((raw.len() as u32).to_le_bytes());
        part.extend(compressed);
        std::fs::write(dir.path().join("parts/part-0"), part).unwrap();
        let matrix = BlockMatrix::open(dir.path().to_str().unwrap(), StorageOptions::default())
            .await
            .unwrap();
        assert_eq!(
            matrix.read(&[1, 0, 1], &[2, 0], 1).await.unwrap().data,
            vec![5., 3., 2., 0., 5., 3.]
        );
    }
}
