use crate::{BlockMatrix, DenseMatrix, MatrixTable, ReadOptions, StorageOptions, Table, Value};
use numpy::{ndarray::Array2, IntoPyArray, PyArray2};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict, PyList},
};
use std::sync::Arc;
use tokio::runtime::Runtime;

fn error(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
fn runtime() -> PyResult<Arc<Runtime>> {
    Ok(Arc::new(Runtime::new().map_err(error)?))
}
fn options(json: &str) -> PyResult<StorageOptions> {
    serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn object(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    Ok(match value {
        Value::Null => py.None(),
        Value::Bool(v) => v.into_pyobject(py)?.to_owned().into_any().unbind(),
        Value::Int(v) => v.into_pyobject(py)?.into_any().unbind(),
        Value::Float(v) => v.into_pyobject(py)?.into_any().unbind(),
        Value::String(v) => v.into_pyobject(py)?.into_any().unbind(),
        Value::Binary(v) => PyBytes::new(py, v).into_any().unbind(),
        Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                list.append(object(py, value)?)?;
            }
            list.into_any().unbind()
        }
        Value::Struct(fields) => {
            let dict = PyDict::new(py);
            for (name, value) in fields {
                dict.set_item(name, object(py, value)?)?;
            }
            dict.into_any().unbind()
        }
    })
}

#[pyclass(name = "NativeTable")]
struct PyTable {
    table: Table,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyTable {
    #[new]
    #[pyo3(signature=(uri, storage_options="{}"))]
    fn new(py: Python<'_>, uri: &str, storage_options: &str) -> PyResult<Self> {
        let runtime = runtime()?;
        let opts = options(storage_options)?;
        let table = py
            .allow_threads(|| runtime.block_on(Table::open(uri, opts)))
            .map_err(error)?;
        Ok(Self { table, runtime })
    }
    #[getter]
    fn schema_json(&self) -> PyResult<String> {
        serde_json::to_string(self.table.row_type()).map_err(error)
    }
    #[getter]
    fn global_schema_json(&self) -> PyResult<String> {
        serde_json::to_string(self.table.global_type()).map_err(error)
    }
    #[getter]
    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.table.metadata).map_err(error)
    }
    #[getter]
    fn bounds_json(&self) -> PyResult<String> {
        serde_json::to_string(self.table.partition_bounds()).map_err(error)
    }
    #[getter]
    fn n_partitions(&self) -> usize {
        self.table.n_partitions()
    }
    fn stats_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.table.stats()).map_err(error)
    }
    fn globals(&self, py: Python<'_>) -> PyResult<PyObject> {
        let value = py
            .allow_threads(|| self.runtime.block_on(self.table.globals()))
            .map_err(error)?;
        object(py, &value)
    }
    fn selected_partitions(&self, read_options: &str) -> PyResult<Vec<usize>> {
        let opts: ReadOptions = serde_json::from_str(read_options).map_err(error)?;
        self.table.selected_partitions(&opts).map_err(error)
    }
    #[pyo3(signature=(read_options="{}"))]
    fn read(&self, py: Python<'_>, read_options: &str) -> PyResult<PyObject> {
        let opts: ReadOptions = serde_json::from_str(read_options).map_err(error)?;
        let rows = py
            .allow_threads(|| self.runtime.block_on(self.table.read(opts)))
            .map_err(error)?;
        object(py, &Value::Array(rows))
    }
    fn read_partition(&self, py: Python<'_>, partition: usize) -> PyResult<PyObject> {
        let rows = py
            .allow_threads(|| self.runtime.block_on(self.table.read_partition(partition)))
            .map_err(error)?;
        object(py, &Value::Array(rows))
    }
    #[pyo3(signature=(start, end, concurrency=8))]
    fn read_row_range(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
        concurrency: usize,
    ) -> PyResult<PyObject> {
        let rows = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.table.read_row_range(start..end, concurrency))
            })
            .map_err(error)?;
        object(py, &Value::Array(rows))
    }
}

#[pyclass(name = "NativeBlockMatrix")]
struct PyBlockMatrix {
    matrix: BlockMatrix,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyBlockMatrix {
    #[new]
    #[pyo3(signature=(uri, storage_options="{}"))]
    fn new(py: Python<'_>, uri: &str, storage_options: &str) -> PyResult<Self> {
        let runtime = runtime()?;
        let opts = options(storage_options)?;
        let matrix = py
            .allow_threads(|| runtime.block_on(BlockMatrix::open(uri, opts)))
            .map_err(error)?;
        Ok(Self { matrix, runtime })
    }
    #[getter]
    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.matrix.metadata).map_err(error)
    }
    fn stats_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.matrix.stats()).map_err(error)
    }
    #[pyo3(signature=(rows, cols, concurrency=4, symmetric_upper=false))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<usize>,
        cols: Vec<usize>,
        concurrency: usize,
        symmetric_upper: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let DenseMatrix {
            n_rows,
            n_cols,
            data,
        } = py
            .allow_threads(|| {
                self.runtime.block_on(async {
                    if symmetric_upper {
                        self.matrix
                            .read_symmetric_upper(&rows, &cols, concurrency)
                            .await
                    } else {
                        self.matrix.read(&rows, &cols, concurrency).await
                    }
                })
            })
            .map_err(error)?;
        Ok(Array2::from_shape_vec((n_rows, n_cols), data)
            .map_err(error)?
            .into_pyarray(py))
    }
}

#[pyclass(name = "NativeMatrixTable")]
struct PyMatrixTable {
    matrix: MatrixTable,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyMatrixTable {
    #[new]
    #[pyo3(signature=(uri, storage_options="{}"))]
    fn new(py: Python<'_>, uri: &str, storage_options: &str) -> PyResult<Self> {
        let runtime = runtime()?;
        let opts = options(storage_options)?;
        let matrix = py
            .allow_threads(|| runtime.block_on(MatrixTable::open(uri, opts)))
            .map_err(error)?;
        Ok(Self { matrix, runtime })
    }
    #[getter]
    fn rows(&self) -> PyTable {
        PyTable {
            table: self.matrix.rows.clone(),
            runtime: self.runtime.clone(),
        }
    }
    #[getter]
    fn cols(&self) -> PyTable {
        PyTable {
            table: self.matrix.cols.clone(),
            runtime: self.runtime.clone(),
        }
    }
    #[getter]
    fn entries(&self) -> PyTable {
        PyTable {
            table: self.matrix.entries.clone(),
            runtime: self.runtime.clone(),
        }
    }
    #[getter]
    fn globals(&self) -> PyTable {
        PyTable {
            table: self.matrix.globals.clone(),
            runtime: self.runtime.clone(),
        }
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTable>()?;
    m.add_class::<PyBlockMatrix>()?;
    m.add_class::<PyMatrixTable>()?;
    Ok(())
}
