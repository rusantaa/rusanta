use std::collections::HashMap;
use std::hash::Hash;

use rusanta_viz::{
    plot::bar::BarPlot,
    style::Color,
};

/// Aggregation function for categorical plots.
#[derive(Debug, Clone, Copy)]
pub enum CatAgg {
    Count,
    Sum,
    Mean,
}

/// Count occurrences of categorical values.
///
/// Equivalent to seaborn `countplot`.
pub fn count_plot<K>(data: &[K]) -> BarPlot
where
K: Eq + Hash + Clone,
{
    let mut counts: HashMap<K, usize> = HashMap::new();

    for k in data {
        *counts.entry(k.clone()).or_insert(0) += 1;
    }

    let mut keys: Vec<_> = counts.keys().cloned().collect();
    keys.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));

    let x: Vec<f64> = (0..keys.len()).map(|i| i as f64).collect();
    let heights: Vec<f64> = keys
    .iter()
    .map(|k| counts.get(k).copied().unwrap_or(0) as f64)
    .collect();

    BarPlot::new(x, heights)
    .color(Color::BLUE)
    .label("count")
}

/// Aggregate numeric values grouped by categories.
///
/// Equivalent to seaborn `barplot`.
pub fn categorical_aggregate<K>(
    categories: &[K],
    values: &[f64],
    agg: CatAgg,
) -> BarPlot
where
K: Eq + Hash + Clone,
{
    assert!(
        categories.len() == values.len(),
            "categories and values must have same length"
    );

    let mut groups: HashMap<K, Vec<f64>> = HashMap::new();

    for (k, v) in categories.iter().cloned().zip(values.iter().copied()) {
        groups.entry(k).or_default().push(v);
    }

    let mut keys: Vec<_> = groups.keys().cloned().collect();
    keys.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));

    let x: Vec<f64> = (0..keys.len()).map(|i| i as f64).collect();

    let heights: Vec<f64> = keys
    .iter()
    .map(|k| {
        let vals = &groups[k];
        match agg {
            CatAgg::Count => vals.len() as f64,
         CatAgg::Sum => vals.iter().sum(),
         CatAgg::Mean => {
             if vals.is_empty() {
                 f64::NAN
             } else {
                 vals.iter().sum::<f64>() / vals.len() as f64
             }
         }
        }
    })
    .collect();

    BarPlot::new(x, heights)
    .color(Color::GREEN)
    .label(match agg {
        CatAgg::Count => "count",
        CatAgg::Sum => "sum",
        CatAgg::Mean => "mean",
    })
}
