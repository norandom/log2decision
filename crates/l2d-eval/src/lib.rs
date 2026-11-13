//! Evaluation: grouped splits, quality metrics, cost, reports.
//!
//! - [`split`]: folds of group keys. Every log of a group key is on one side of every
//!   split (requirement 9.2).
//! - [`metrics`]: ROC AUC, average precision, recall at a review budget and log loss,
//!   computed exactly, with ties handled (requirement 9.1).
//! - [`evaluate`]: the grouped evaluation of the baseline scorer: per fold the fit, the
//!   choice of the L2 penalty inside the fitting side, the calibration on a part of the
//!   fitting side the scorer did not see, and the figures on the held-out side
//!   (requirements 9.2, 9.8, 11.8).
//! - [`report`]: the evaluation as JSON and as a readable summary (requirement 9.6).
//!
//! The pieces are those task 3.7 generalizes: [`split`] and [`metrics`] know nothing
//! about a scorer, and [`evaluate::summarize`] works on held-out ratios whatever
//! produced them.

pub mod evaluate;
pub mod metrics;
pub mod report;
pub mod split;
