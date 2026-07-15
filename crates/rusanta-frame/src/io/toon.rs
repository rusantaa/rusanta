use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::collections::HashMap;

use crate::dataframe::DataFrame;
use crate::series::Series;
use crate::index::Index;
use crate::error::{Error, Result};

const MAGIC: &[u8; 4] = b"TOON";
const VERSION: u16 = 1;

/// Write a DataFrame to TOON binary format.
///
/// Notes:
/// - numeric only (f64)
/// - columnar layout
pub fn write_toon<P>(df: &DataFrame<f64>, path: P) -> Result<()>
where
P: AsRef<Path>,
{
    let mut file = File::create(path)?;

    // Header
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;

    let ncols = df.ncols() as u32;
    let nrows = df.nrows() as u64;

    file.write_all(&ncols.to_le_bytes())?;
    file.write_all(&nrows.to_le_bytes())?;

    // Columns
    for col in df.columns() {
        let name_bytes = col.as_bytes();
        let name_len = name_bytes.len() as u16;

        file.write_all(&name_len.to_le_bytes())?;
        file.write_all(name_bytes)?;

        let series = df.column(col)?;
        for v in series.iter() {
            file.write_all(&v.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Read a TOON file into a DataFrame.
pub fn read_toon<P>(path: P) -> Result<DataFrame<f64>>
where
P: AsRef<Path>,
{
    let mut file = File::open(path)?;

    // Magic
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::InvalidValue("invalid TOON magic".into()));
    }

    // Version
    let mut vbuf = [0u8; 2];
    file.read_exact(&mut vbuf)?;
    let version = u16::from_le_bytes(vbuf);

    if version != VERSION {
        return Err(Error::InvalidValue(format!(
            "unsupported TOON version {}",
            version
        )));
    }

    // Shape
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    file.read_exact(&mut buf4)?;
    let ncols = u32::from_le_bytes(buf4) as usize;

    file.read_exact(&mut buf8)?;
    let nrows = u64::from_le_bytes(buf8) as usize;

    let index = Index::range(nrows);
    let mut data: HashMap<String, Series<f64>> = HashMap::new();

    for _ in 0..ncols {
        // Column name
        let mut len_buf = [0u8; 2];
        file.read_exact(&mut len_buf)?;
        let name_len = u16::from_le_bytes(len_buf) as usize;

        let mut name_buf = vec![0u8; name_len];
        file.read_exact(&mut name_buf)?;
        let name = String::from_utf8(name_buf)
        .map_err(|_| Error::InvalidValue("invalid UTF-8 column name".into()))?;

        // Column data
        let mut values = Vec::with_capacity(nrows);
        for _ in 0..nrows {
            let mut vbuf = [0u8; 8];
            file.read_exact(&mut vbuf)?;
            values.push(f64::from_le_bytes(vbuf));
        }

        let series = Series::new(values, index.clone(), Some(name.clone()))?;
        data.insert(name, series);
    }

    DataFrame::new(data)
}
