//! Baseline scorer: character n-gram hashing and logistic regression.

pub mod hashing;

pub use hashing::{CharHasher, HashingError, HashingSettings, RoutingMode, SparseRow};
