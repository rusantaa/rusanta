use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::index::Index;
use crate::series::Series;
use rusanta_core::traits::numeric::Numeric;

/// TSV read options.
#[derive(Debug, Clone)]
pub struct TsvReadOptions {
    pub has_header: bool,
    pub index_col: Option<usize>,
}

impl Default for TsvReadOptions {
    fn default() -> Self {
        Self {
            has_header: true,
            index_col: None,
        }
    }
}

/// TSV write options.
#[derive(Debug, Clone)]
pub struct TsvWriteOptions {
    pub include_header: bool,
}

impl Default for TsvWriteOptions {
    fn default() -> Self {
        Self {
            include_header: true,
        }
    }
}

/// Read a TSV file into a DataFrame.
///
/// Notes:
/// - Uses `\t` as delimiter
/// - All non-index columns must be numeric
pub fn read_tsv<T, P>(path: P, options: TsvReadOptions) -> Result<DataFrame<T>>
where
T: Numeric + std::str::FromStr,
<T as std::str::FromStr>::Err: std::fmt::Display,
P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Header
    let header = if options.has_header {
        let line = lines
        .next()
        .ok_or_else(|| Error::InvalidValue("empty TSV file".into()))??;

        line.split('\t')
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let mut columns: Vec<Vec<T>> = Vec::new();
    let mut index_labels: Vec<String> = Vec::new();
    let mut initialized = false;

    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();

        if !initialized {
            columns = vec![Vec::new(); fields.len()];
            initialized = true;
        }

        for (i, field) in fields.iter().enumerate() {
            if Some(i) == options.index_col {
                index_labels.push(field.to_string());
                continue;
            }

            let value = field
            .trim()
            .parse::<T>()
            .map_err(|e| Error::ParseError(e.to_string()))?;

            columns[i].push(value);
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

    let mut data = std::collections::HashMap::new();

    for (i, col) in columns.into_iter().enumerate() {
        if Some(i) == options.index_col {
            continue;
        }

        let name = if options.has_header {
            header
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

/// Write a DataFrame to TSV.
pub fn write_tsv<T, P>(
    df: &DataFrame<T>,
    path: P,
    options: TsvWriteOptions,
) -> Result<()>
where
T: Numeric + ToString,
P: AsRef<Path>,
{
    let mut file = File::create(path)?;

    let columns = df.columns();
    let nrows = df.nrows();

    // Header
    if options.include_header {
        let header = columns
        .iter()
        .map(|c| c.as_str())
        .collect::<Vec<_>>()
        .join("\t");

        writeln!(file, "{}", header)?;
    }

    // Rows
    for row in 0..nrows {
        let mut values = Vec::with_capacity(columns.len());
        for col in &columns {
            let series = df.column(col)?;
            values.push(series.get(row)?.to_string());
        }

        writeln!(file, "{}", values.join("\t"))?;
    }

    Ok(())
}
