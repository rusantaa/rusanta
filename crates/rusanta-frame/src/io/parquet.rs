use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;

use arrow::array::{ArrayRef, Float64Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;
use arrow::error::ArrowError;

use parquet::arrow::{ArrowWriter, ParquetFileArrowReader};
use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::dataframe::DataFrame;
use crate::series::Series;
use crate::index::Index;
use crate::error::{Error, Result};

/// Read a Parquet file into a DataFrame.
///
/// Assumptions:
/// - numeric columns only
/// - one row group or many (handled)
pub fn read_parquet<P>(path: P) -> Result<DataFrame<f64>>
where
P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)
    .map_err(|e| Error::ParseError(e.to_string()))?;

    let mut arrow_reader = ParquetFileArrowReader::new(Arc::new(reader));
    let record_reader = arrow_reader
    .get_record_reader(1024)
    .map_err(|e| Error::ParseError(e.to_string()))?;

    let mut columns: HashMap<String, Vec<f64>> = HashMap::new();
    let mut index_len = 0usize;

    for batch in record_reader {
        let batch = batch.map_err(|e| Error::ParseError(e.to_string()))?;
        index_len += batch.num_rows();

        for (i, field) in batch.schema().fields().iter().enumerate() {
            let name = field.name().clone();
            let array = batch.column(i);

            let float_array = array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                Error::InvalidValue(format!(
                    "column '{}' is not Float64",
                    name
                ))
            })?;

            let col = columns.entry(name).or_insert_with(Vec::new);
            for j in 0..float_array.len() {
                col.push(float_array.value(j));
            }
        }
    }

    let index = Index::range(index_len);

    let mut data = HashMap::new();
    for (name, vec) in columns {
        let s = Series::new(vec, index.clone(), Some(name.clone()))?;
        data.insert(name, s);
    }

    DataFrame::new(data)
}

/// Write a DataFrame to Parquet.
///
/// Notes:
/// - writes Float64 columns
/// - row-group size = full frame
pub fn write_parquet<P>(df: &DataFrame<f64>, path: P) -> Result<()>
where
P: AsRef<Path>,
{
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for col in df.columns() {
        fields.push(Field::new(col, DataType::Float64, false));

        let series = df.column(col)?;
        let values = series.to_vec();
        let array = Float64Array::from(values);
        arrays.push(Arc::new(array));
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)
    .map_err(|e| Error::ParseError(e.to_string()))?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
    .map_err(|e| Error::ParseError(e.to_string()))?;

    writer.write(&batch)
    .map_err(|e| Error::ParseError(e.to_string()))?;
    writer.close()
    .map_err(|e| Error::ParseError(e.to_string()))?;

    Ok(())
}
