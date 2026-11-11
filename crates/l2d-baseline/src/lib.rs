//! Baseline scorer: character n-gram hashing and logistic regression.

pub mod hashing;
pub mod logreg;

pub use hashing::{CharHasher, HashingError, HashingSettings, RoutingMode, SparseRow};
pub use logreg::{FitReport, LogReg, LogRegError, LogRegSettings, SparseMatrix, Termination};
