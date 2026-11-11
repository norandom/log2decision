//! Baseline scorer: character n-gram hashing and logistic regression.

pub mod hashing;
pub mod logreg;
pub mod scorer;

pub use hashing::{CharHasher, HashingError, HashingSettings, RoutingMode, SparseRow};
pub use logreg::{FitReport, LogReg, LogRegError, LogRegSettings, SparseMatrix, Termination};
pub use scorer::{BaselineFitReport, BaselineScorer, BaselineSettings, FitError, FittingInput};
