use std::collections::HashMap;
use std::hash::Hash;

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::index::Index;
use crate::series::Series;

/// Supported aggregation operations.
#[derive(Debug, Clone, Copy)]
pub enum Agg {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

/// GroupBy object (lazy until aggregate is called).
pub struct GroupBy<'a, T>
where
T: Copy + PartialOrd + std::ops::Add<Output = T> + From<f64>,
{
    df: &'a DataFrame<T>,
    key: String,
}

impl<'a, T> GroupBy<'a, T>
where
T: Copy + PartialOrd + std::ops::Add<Output = T> + From<f64>,
{
    /// Create a new GroupBy from a DataFrame.
    pub fn new(df: &'a DataFrame<T>, key: &str) -> Result<Self> {
        if !df.has_column(key) {
            return Err(Error::KeyNotFound(key.into()));
        }
        Ok(Self {
            df,
            key: key.into(),
        })
    }

    /// Aggregate using a single aggregation op.
    pub fn agg(&self, op: Agg) -> Result<DataFrame<T>>
    where
    T: Hash + Eq,
    {
        let key_series = self.df.column(&self.key)?;
        let nrows = self.df.nrows();

        // group -> row indices
        let mut groups: HashMap<T, Vec<usize>> = HashMap::new();
        for i in 0..nrows {
            let v = key_series.get(i)?;
            groups.entry(v).or_default().push(i);
        }

        let mut result_cols: HashMap<String, Vec<T>> = HashMap::new();
        let mut index_labels: Vec<String> = Vec::new();

        for (group_key, indices) in groups.iter() {
            index_labels.push(group_key.to_string());

            for col in self.df.columns() {
                if col == &self.key {
                    continue;
                }

                let series = self.df.column(col)?;
                let values: Vec<T> = indices
                .iter()
                .map(|&i| series.get(i).unwrap())
                .collect();

                let agg_value = match op {
                    Agg::Sum => values.iter().copied().reduce(|a, b| a + b)
                    .ok_or_else(|| Error::InvalidValue("empty group".into()))?,
                    Agg::Mean => {
                        let sum = values.iter().copied().reduce(|a, b| a + b)
                        .ok_or_else(|| Error::InvalidValue("empty group".into()))?;
                        let denom = T::from(values.len() as f64);
                        sum + T::from(0.0) / denom
                    }
                    Agg::Min => values.iter().copied().reduce(|a, b| if a < b { a } else { b })
                    .ok_or_else(|| Error::InvalidValue("empty group".into()))?,
                    Agg::Max => values.iter().copied().reduce(|a, b| if a > b { a } else { b })
                    .ok_or_else(|| Error::InvalidValue("empty group".into()))?,
                    Agg::Count => T::from(values.len() as f64),
                };

                result_cols
                .entry(col.clone())
                .or_insert_with(Vec::new)
                .push(agg_value);
            }
        }

        let index = Index::new(index_labels)?;

        let mut data = HashMap::new();
        for (name, vec) in result_cols {
            let s = Series::new(vec, index.clone(), Some(name.clone()))?;
            data.insert(name, s);
        }

        DataFrame::new(data)
    }
}

/// Convenience method on DataFrame.
impl<T> DataFrame<T>
where
T: Copy + PartialOrd + std::ops::Add<Output = T> + From<f64> + Hash + Eq,
{
    /// Group the DataFrame by a column.
    pub fn groupby(&self, key: &str) -> Result<GroupBy<'_, T>> {
        GroupBy::new(self, key)
    }
}
