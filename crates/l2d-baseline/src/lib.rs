//! Baseline scorer: character n-gram hashing and logistic regression.

pub mod hashing;
pub mod logreg;
pub mod markscore;
pub mod redact;
pub mod scorer;

pub use hashing::{CharHasher, HashingError, HashingSettings, RoutingMode, SparseRow};
pub use logreg::{FitReport, LogReg, LogRegError, LogRegSettings, SparseMatrix, Termination};
pub use markscore::{
    BatchRoom, GroupKeyRule, MarkFit, MarkFitCounts, MarkFitSettings, MarkSetOrigin, OtherClass,
    fit_from_marks,
};
pub use redact::{
    RedactionInput, RedactionProposal, RedactionProposals, RedactionRules, RedactionSettings,
    propose_redactions,
};
pub use scorer::{BaselineFitReport, BaselineScorer, BaselineSettings, FitError, FittingInput};
