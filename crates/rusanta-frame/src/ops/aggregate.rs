use std::collections::HashMap;
use std::hash::Hash;

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::index::Index;
use crate::series::Series;

/// Join types supported by Rusanta.
#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    Left,
    Right,
}

/// Perform a join between two DataFrames.
pub fn join<T>(
    left: &DataFrame<T>,
    right: &DataFrame<T>,
    key: &str,
    how: JoinType,
) -> Result<DataFrame<T>>
where
T: Copy + Eq + Hash,
{
    if !left.has_column(key) {
        return Err(Error::KeyNotFound(format!(
            "left key '{}' not found",
            key
        )));
    }
    if !right.has_column(key) {
        return Err(Error::KeyNotFound(format!(
            "right key '{}' not found",
            key
        )));
    }

    let left_key = left.column(key)?;
    let right_key = right.column(key)?;

    // Build hash map for right side
    let mut right_map: HashMap<T, Vec<usize>> = HashMap::new();
    for i in 0..right.nrows() {
        right_map.entry(right_key.get(i)?).or_default().push(i);
    }

    let mut result_cols: HashMap<String, Vec<T>> = HashMap::new();

    let mut push_row = |lidx: Option<usize>, ridx: Option<usize>| -> Result<()> {
        // left columns
        for col in left.columns() {
            if col == key {
                continue;
            }
            let v = match lidx {
                Some(i) => left.column(col)?.get(i)?,
                None => return Err(Error::InvalidValue("null values not supported yet".into())),
            };
            result_cols.entry(col.clone()).or_default().push(v);
        }

        // right columns
        for col in right.columns() {
            if col == key {
                continue;
            }
            let name = if left.has_column(col) {
                format!("{}_right", col)
            } else {
                col.clone()
            };

            let v = match ridx {
                Some(i) => right.column(col)?.get(i)?,
                None => return Err(Error::InvalidValue("null values not supported yet".into())),
            };
            result_cols.entry(name).or_default().push(v);
        }

        Ok(())
    };

    let mut row_count = 0usize;

    match how {
        JoinType::Inner | JoinType::Left => {
            for i in 0..left.nrows() {
                let key_val = left_key.get(i)?;
                if let Some(ridxs) = right_map.get(&key_val) {
                    for &r in ridxs {
                        push_row(Some(i), Some(r))?;
                        row_count += 1;
                    }
                } else if matches!(how, JoinType::Left) {
                    push_row(Some(i), None)?;
                    row_count += 1;
                }
            }
        }
        JoinType::Right => {
            for i in 0..right.nrows() {
                let key_val = right_key.get(i)?;
                let mut matched = false;
                for j in 0..left.nrows() {
                    if left_key.get(j)? == key_val {
                        push_row(Some(j), Some(i))?;
                        row_count += 1;
                        matched = true;
                    }
                }
                if !matched {
                    push_row(None, Some(i))?;
                    row_count += 1;
                }
            }
        }
    }

    let index = Index::range(row_count);
    let mut data = HashMap::new();

    for (name, vec) in result_cols {
        let s = Series::new(vec, index.clone(), Some(name.clone()))?;
        data.insert(name, s);
    }

    DataFrame::new(data)
}

/// Convenience methods on DataFrame.
impl<T> DataFrame<T>
where
T: Copy + Eq + Hash,
{
    pub fn join(
        &self,
        other: &DataFrame<T>,
        key: &str,
        how: JoinType,
    ) -> Result<DataFrame<T>> {
        join(self, other, key, how)
    }
}
