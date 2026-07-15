// rusanta-ml/src/metrics.rs

use std::fmt::Debug;

/// Regression metrics.
pub mod regression {
    /// Mean Squared Error.
    pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
        assert_eq!(y_true.len(), y_pred.len());
        let n = y_true.len() as f64;
        y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n
    }

    /// Root Mean Squared Error.
    pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
        mse(y_true, y_pred).sqrt()
    }

    /// Mean Absolute Error.
    pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
        assert_eq!(y_true.len(), y_pred.len());
        let n = y_true.len() as f64;
        y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / n
    }

    /// R-squared (coefficient of determination).
    pub fn r2(y_true: &[f64], y_pred: &[f64]) -> f64 {
        assert_eq!(y_true.len(), y_pred.len());
        let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;

        let ss_tot = y_true.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
        let ss_res = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();

        if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }
}

/// Classification metrics.
pub mod classification {
    /// Accuracy score.
    pub fn accuracy<T: PartialEq>(y_true: &[T], y_pred: &[T]) -> f64 {
        assert_eq!(y_true.len(), y_pred.len());
        let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();
        correct as f64 / y_true.len() as f64
    }

    /// Confusion matrix for binary classification.
    ///
    /// Returns (tp, fp, fn, tn).
    pub fn confusion_matrix<T: PartialEq + Debug>(
        y_true: &[T],
        y_pred: &[T],
        positive: &T,
    ) -> (usize, usize, usize, usize) {
        assert_eq!(y_true.len(), y_pred.len());

        let mut tp = 0;
        let mut fp = 0;
        let mut fn_ = 0;
        let mut tn = 0;

        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            match (t == positive, p == positive) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        (tp, fp, fn_, tn)
    }

    /// Precision score.
    pub fn precision(tp: usize, fp: usize) -> f64 {
        if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        }
    }

    /// Recall score.
    pub fn recall(tp: usize, fn_: usize) -> f64 {
        if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        }
    }

    /// F1 score.
    pub fn f1(tp: usize, fp: usize, fn_: usize) -> f64 {
        let p = precision(tp, fp);
        let r = recall(tp, fn_);
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}
