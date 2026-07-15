use std::fmt;
use std::error::Error as StdError;

/// A unified Result type for all Rusanta crates.
pub type Result<T> = std::result::Result<T, RusantaError>;

/// The central error type for the Rusanta ecosystem.
///
/// Design goals:
/// - One error type across all crates
/// - Human-readable messages
/// - Machine-matchable variants
/// - Zero-cost when unused
#[derive(Debug)]
pub enum RusantaError {
    // ========================
    // Core / Internal
    // ========================
    Internal {
        message: String,
    },

    NotImplemented {
        feature: &'static str,
    },

    // ========================
    // Data & Shape Errors
    // ========================
    ShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    DimensionMismatch {
        lhs: usize,
        rhs: usize,
    },

    EmptyData {
        context: &'static str,
    },

    // ========================
    // Indexing & Access
    // ========================
    IndexOutOfBounds {
        index: usize,
        len: usize,
    },

    ColumnNotFound {
        name: String,
    },

    // ========================
    // Type & Value Errors
    // ========================
    InvalidType {
        expected: &'static str,
        found: &'static str,
    },

    InvalidValue {
        message: String,
    },

    // ========================
    // Machine Learning
    // ========================
    ModelNotFitted,

    InvalidHyperparameter {
        name: &'static str,
        reason: String,
    },

    // ========================
    // IO & External
    // ========================
    Io(std::io::Error),

    Parse {
        message: String,
    },
}

impl fmt::Display for RusantaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RusantaError::Internal { message } => {
                write!(f, "Internal Rusanta error: {}", message)
            }

            RusantaError::NotImplemented { feature } => {
                write!(f, "Feature not implemented: {}", feature)
            }

            RusantaError::ShapeMismatch { expected, found } => {
                write!(
                    f,
                    "Shape mismatch: expected {:?}, found {:?}",
                    expected, found
                )
            }

            RusantaError::DimensionMismatch { lhs, rhs } => {
                write!(
                    f,
                    "Dimension mismatch: left has {}, right has {}",
                    lhs, rhs
                )
            }

            RusantaError::EmptyData { context } => {
                write!(f, "Empty data encountered ({})", context)
            }

            RusantaError::IndexOutOfBounds { index, len } => {
                write!(
                    f,
                    "Index out of bounds: index {}, length {}",
                    index, len
                )
            }

            RusantaError::ColumnNotFound { name } => {
                write!(f, "Column not found: '{}'", name)
            }

            RusantaError::InvalidType { expected, found } => {
                write!(
                    f,
                    "Invalid type: expected {}, found {}",
                    expected, found
                )
            }

            RusantaError::InvalidValue { message } => {
                write!(f, "Invalid value: {}", message)
            }

            RusantaError::ModelNotFitted => {
                write!(f, "Model has not been fitted")
            }

            RusantaError::InvalidHyperparameter { name, reason } => {
                write!(
                    f,
                    "Invalid hyperparameter '{}': {}",
                    name, reason
                )
            }

            RusantaError::Io(err) => {
                write!(f, "IO error: {}", err)
            }

            RusantaError::Parse { message } => {
                write!(f, "Parse error: {}", message)
            }
        }
    }
}

impl StdError for RusantaError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            RusantaError::Io(err) => Some(err),
            _ => None,
        }
    }
}

// ========================
// Conversions
// ========================

impl From<std::io::Error> for RusantaError {
    fn from(err: std::io::Error) -> Self {
        RusantaError::Io(err)
    }
}

impl From<&str> for RusantaError {
    fn from(message: &str) -> Self {
        RusantaError::Internal {
            message: message.to_string(),
        }
    }
}

impl From<String> for RusantaError {
    fn from(message: String) -> Self {
        RusantaError::Internal { message }
    }
}
