use std::collections::HashMap;
use std::path::Path;

use calamine::{open_workbook_auto, DataType, Reader};

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::index::Index;
use crate::series::Series;
use rusanta_core::traits::numeric::Numeric;

/// XLSX read options.
#[derive(Debug, Clone)]
pub struct XlsxReadOptions {
    /// Use the first row as header.
    pub has_header: bool,
    /// Sheet name (None = first sheet).
    pub sheet: Option<String>,
    /// Column index to use as index.
    pub index_col: Option<usize>,
}

impl Default for XlsxReadOptions {
    fn default() -> Self {
        Self {
            has_header: true,
            sheet: None,
            index_col: None,
        }
    }
}

/// Read an XLSX file into a DataFrame.
///
/// Notes:
/// - Only numeric cells are supported
/// - Empty cells are an error
/// - Date cells must already be converted to numbers
pub fn read_xlsx<T, P>(
    path: P,
    options: XlsxReadOptions,
) -> Result<DataFrame<T>>
where
T: Numeric + std::str::FromStr,
<T as std::str::FromStr>::Err: std::fmt::Display,
P: AsRef<Path>,
{
    let mut workbook = open_workbook_auto(path)
    .map_err(|e| Error::ParseError(e.to_string()))?;

    let sheet_name = match options.sheet {
        Some(ref s) => s.clone(),
        None => workbook
        .sheet_names()
        .get(0)
        .cloned()
        .ok_or_else(|| Error::InvalidValue("no sheets found".into()))?,
    };

    let range = workbook
    .worksheet_range(&sheet_name)
    .ok_or_else(|| Error::KeyNotFound(sheet_name.clone()))?
    .map_err(|e| Error::ParseError(e.to_string()))?;

    let mut rows = range.rows();

    // Header
    let headers: Vec<String> = if options.has_header {
        let row = rows
        .next()
        .ok_or_else(|| Error::InvalidValue("empty sheet".into()))?;

        row.iter()
        .map(|c| match c {
            DataType::String(s) => s.clone(),
             _ => Err(Error::InvalidValue(
                 "header row must be strings".into(),
             ))?,
        })
        .collect()
    } else {
        Vec::new()
    };

    let mut columns: Vec<Vec<T>> = Vec::new();
    let mut index_labels: Vec<String> = Vec::new();
    let mut initialized = false;

    for (row_idx, row) in rows.enumerate() {
        if !initialized {
            columns = vec![Vec::new(); row.len()];
            initialized = true;
        }

        for (i, cell) in row.iter().enumerate() {
            if Some(i) == options.index_col {
                index_labels.push(cell.to_string());
                continue;
            }

            let value = match cell {
                DataType::Float(f) => f.to_string(),
                DataType::Int(i) => i.to_string(),
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "non-numeric cell at row {}, col {}",
                        row_idx, i
                    )))
                }
            };

            let parsed = value
            .parse::<T>()
            .map_err(|e| Error::ParseError(e.to_string()))?;

            columns[i].push(parsed);
        }
    }

    if !initialized {
        return Err(Error::InvalidValue("no data rows found".into()));
    }

    let index = if options.index_col.is_some() {
        Index::new(index_labels)?
    } else {
        Index::range(columns[0].len())
    };

    let mut data = HashMap::new();

    for (i, col) in columns.into_iter().enumerate() {
        if Some(i) == options.index_col {
            continue;
        }

        let name = if options.has_header {
            headers
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("col{}", i))
        } else {
            format!("col{}", i)
        };

        let series = Series::new(col, index.clone(), Some(name.clone()))?;
        data.insert(name, series);
    }

    DataFrame::new(data)
}
