//! Deterministic L2-penalized logistic regression over sparse or dense rows.
//!
//! # Objective
//!
//! For weights `w`, bias `b`, rows `x_i` and signs `s_i` (`+1` for a positive row, `-1`
//! for a negative one) the fit minimizes
//!
//! ```text
//! F(w, b) = sum_i ln(1 + exp(-s_i * (x_i . w + b))) + (l2 / 2) * ||w||^2
//! ```
//!
//! - The loss is the **sum** over rows, not the mean: with twice the rows, the same `l2`
//!   penalizes half as strongly. `l2` is the `alpha` of `linfa-logistic` and the inverse
//!   of the `C` of scikit-learn.
//! - The **bias is not penalized**. A model always has a bias (intercept).
//! - A positive row is one whose label is `true`; [`LogReg::probability_sparse`] and its
//!   siblings return the probability of that class.
//!
//! # Solver
//!
//! The optimizer is the L-BFGS solver of the `argmin` crate (0.11) with a history of ten
//! steps and its Moré–Thuente line search, started from the zero vector. This is the
//! solver, the history length, the line search and the start that `linfa-logistic` 0.8
//! uses; its objective is the one above. `linfa-logistic` itself is not called because it
//! fits dense matrices only, and a dense matrix of one training fold of the baseline
//! (20,000 rows by 65,536 columns) would need more than 10 GB. The optimizer's vector
//! arithmetic runs on `ndarray` vectors through `argmin-math`.
//!
//! The fit stops when the Euclidean norm of the gradient of `F` falls below
//! [`LogRegSettings::tolerance`], when `F` changes by less than the machine epsilon
//! between two iterations (both reported as [`Termination::Converged`]), or after
//! [`LogRegSettings::max_iter`] iterations ([`Termination::IterationCap`]). A capped fit
//! is not an error: it returns the best parameters seen, and [`FitReport`] says so.
//!
//! # Determinism
//!
//! Equal rows in equal order with equal labels and settings give bit-equal weights, in
//! every run and process of the same build on the same machine:
//!
//! - no random number is drawn; the start is the zero vector,
//! - nothing runs in parallel: neither `argmin` (its `rayon` feature is off) nor `ndarray`
//!   (no thread pool, no BLAS backend) nor this module starts a thread,
//! - every sum is taken in one fixed order: rows in input order, entries of a row in
//!   ascending column order, and the fixed order of `ndarray`'s dot product,
//! - no time, address, hash-map order or environment value enters the arithmetic.
//!
//! A sparse matrix and the dense matrix with the same entries give bit-equal models as
//! well, because the dense kernels add the same terms in the same order plus zeros.
//!
//! Row order is part of the input: other orders reach the same optimum only up to
//! rounding. Across machines or toolchains the bits may differ, because `exp` and `ln_1p`
//! come from the platform's mathematics library.
//!
//! # Scale
//!
//! Memory is the stored entries (12 bytes each) plus a few dozen vectors of the model's
//! length; one iteration costs a handful of passes over the entries.

use std::cell::RefCell;

use argmin::core::{CostFunction, Error as SolverError, Executor, Gradient, State};
use argmin::core::{TerminationReason, TerminationStatus};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::Array1;

use crate::hashing::SparseRow;

/// Number of past steps the L-BFGS solver keeps, as in `linfa-logistic`.
const HISTORY: usize = 10;

/// Settings of a fit. The defaults are those of `linfa-logistic`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogRegSettings {
    /// Strength of the L2 penalty (see the module documentation), finite and not
    /// negative. With `0.0` separable rows have no finite optimum; the fit then stops at
    /// the iteration cap or when the loss no longer changes.
    pub l2: f64,
    /// Largest number of L-BFGS iterations, at least 1.
    pub max_iter: u32,
    /// The fit has converged when the Euclidean norm of the gradient of the objective is
    /// below this value. Finite and not negative. It is an absolute bound on the gradient
    /// of the summed loss, so it is stricter the more rows there are.
    pub tolerance: f64,
}

impl Default for LogRegSettings {
    fn default() -> Self {
        Self {
            l2: 1.0,
            max_iter: 100,
            tolerance: 1e-4,
        }
    }
}

impl LogRegSettings {
    fn validate(&self) -> Result<(), LogRegError> {
        if !self.l2.is_finite() || self.l2 < 0.0 {
            return Err(LogRegError::InvalidSettings {
                name: "l2",
                value: self.l2,
            });
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(LogRegError::InvalidSettings {
                name: "tolerance",
                value: self.tolerance,
            });
        }
        if self.max_iter == 0 {
            return Err(LogRegError::InvalidSettings {
                name: "max_iter",
                value: 0.0,
            });
        }
        Ok(())
    }
}

/// Why a fit or a score was refused. Row numbers count from 0; a single scored row is
/// row 0.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LogRegError {
    /// A setting is outside its documented range.
    #[error("setting `{name}` has the unusable value {value}")]
    InvalidSettings { name: &'static str, value: f64 },
    /// There is nothing to fit.
    #[error("no rows were supplied")]
    NoRows,
    /// Rows and labels differ in number.
    #[error("{rows} rows but {labels} labels")]
    LabelCount { rows: usize, labels: usize },
    /// All labels are equal.
    #[error("only one class is present: {positive} positive and {negative} negative rows")]
    SingleClass { positive: usize, negative: usize },
    /// A dense row is not as long as the first row or as the model.
    #[error("row {row} has {found} values where {expected} are expected")]
    RowLength {
        row: usize,
        expected: usize,
        found: usize,
    },
    /// The index and value lists of a sparse row differ in length.
    #[error("row {row} has {indices} indices but {values} values")]
    EntryCount {
        row: usize,
        indices: usize,
        values: usize,
    },
    /// A sparse row holds a column at or above the dimension.
    #[error("row {row} holds index {index}, outside the dimension {dimension}")]
    IndexOutOfRange {
        row: usize,
        index: u32,
        dimension: usize,
    },
    /// The columns of a sparse row are not strictly ascending.
    #[error("the indices of row {row} are not strictly ascending")]
    UnorderedIndices { row: usize },
    /// A row holds NaN or an infinity.
    #[error("row {row} holds a value that is not finite")]
    NonFinite { row: usize },
    /// A weight or the bias is NaN or an infinity.
    #[error("a model parameter is not finite")]
    NonFiniteModel,
    /// The solver reported an error of its own.
    #[error("the solver failed: {0}")]
    Solver(String),
}

/// Why a fit ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// The gradient norm fell below the tolerance, or the objective stopped changing.
    Converged,
    /// [`LogRegSettings::max_iter`] iterations were used up.
    IterationCap,
    /// The line search could not improve on the current point (for example at the limit
    /// of floating-point resolution). The best parameters seen are returned.
    LineSearchStopped,
}

/// How a fit ended. `objective` and `gradient_norm` are evaluated at the returned model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitReport {
    /// L-BFGS iterations performed.
    pub iterations: u64,
    /// Why the fit ended.
    pub termination: Termination,
    /// Value of the objective at the returned model.
    pub objective: f64,
    /// Euclidean norm of the gradient of the objective at the returned model.
    pub gradient_norm: f64,
}

impl FitReport {
    /// Whether the fit ended by convergence rather than at the cap or in the line search.
    pub fn converged(&self) -> bool {
        self.termination == Termination::Converged
    }
}

/// Checks one sparse row: as many indices as values, strictly ascending indices below
/// `dimension`, finite values.
fn check_entries(
    row: usize,
    indices: &[u32],
    values: &[f64],
    dimension: usize,
) -> Result<(), LogRegError> {
    if indices.len() != values.len() {
        return Err(LogRegError::EntryCount {
            row,
            indices: indices.len(),
            values: values.len(),
        });
    }
    if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(LogRegError::UnorderedIndices { row });
    }
    if let Some(&index) = indices.last()
        && index as usize >= dimension
    {
        return Err(LogRegError::IndexOutOfRange {
            row,
            index,
            dimension,
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(LogRegError::NonFinite { row });
    }
    Ok(())
}

/// Sparse rows in compressed row form: the training input of [`LogReg::fit_matrix`].
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    dimension: u32,
    /// `starts[i]..starts[i + 1]` is the entry range of row `i`.
    starts: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<f64>,
}

impl SparseMatrix {
    /// A matrix without rows whose columns are `0..dimension`.
    pub fn new(dimension: u32) -> Self {
        Self {
            dimension,
            starts: vec![0],
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// The matrix of `rows`, in their order. Size `dimension` from
    /// [`crate::hashing::CharHasher::dimension`].
    pub fn from_rows(dimension: u32, rows: &[SparseRow]) -> Result<Self, LogRegError> {
        let stored = rows.iter().map(SparseRow::len).sum();
        let mut matrix = Self::new(dimension);
        matrix.starts.reserve(rows.len());
        matrix.indices.reserve(stored);
        matrix.values.reserve(stored);
        for row in rows {
            matrix.push_row(row.indices(), row.values())?;
        }
        Ok(matrix)
    }

    /// Appends a row given as strictly ascending column indices and their finite values.
    /// A refused row leaves the matrix unchanged.
    pub fn push_row(&mut self, indices: &[u32], values: &[f64]) -> Result<(), LogRegError> {
        check_entries(self.rows(), indices, values, self.dimension as usize)?;
        self.indices.extend_from_slice(indices);
        self.values.extend_from_slice(values);
        self.starts.push(self.indices.len());
        Ok(())
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.starts.len() - 1
    }

    /// Number of columns.
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Number of stored entries.
    pub fn stored(&self) -> usize {
        self.indices.len()
    }
}

/// Dense rows of equal length, one after the other.
struct DenseMatrix {
    rows: usize,
    columns: usize,
    values: Vec<f64>,
}

/// The two products of a row matrix the objective needs. Both implementations add the
/// terms of a sum in the same order (rows in order, columns ascending within a row), and
/// a dense matrix only adds further zeros, so equal matrices give equal bits.
trait Design {
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;
    /// `out[i] = x_i . weights + bias`.
    fn margins(&self, weights: &[f64], bias: f64, out: &mut [f64]);
    /// `sums[j] += sum_i x_ij * factors[i]`.
    fn add_weighted_rows(&self, factors: &[f64], sums: &mut [f64]);
}

impl Design for SparseMatrix {
    fn rows(&self) -> usize {
        SparseMatrix::rows(self)
    }

    fn columns(&self) -> usize {
        self.dimension as usize
    }

    fn margins(&self, weights: &[f64], bias: f64, out: &mut [f64]) {
        for (row, slot) in out.iter_mut().enumerate() {
            let range = self.starts[row]..self.starts[row + 1];
            *slot = dot_entries(&self.indices[range.clone()], &self.values[range], weights) + bias;
        }
    }

    fn add_weighted_rows(&self, factors: &[f64], sums: &mut [f64]) {
        // Plain index loops here and in `dot_entries`: these two loops are the whole cost
        // of a fit, and iterator chains make an unoptimized build several times slower.
        for (row, factor) in factors.iter().enumerate() {
            let mut entry = self.starts[row];
            let end = self.starts[row + 1];
            while entry < end {
                sums[self.indices[entry] as usize] += self.values[entry] * factor;
                entry += 1;
            }
        }
    }
}

impl Design for DenseMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn margins(&self, weights: &[f64], bias: f64, out: &mut [f64]) {
        for (row, slot) in out.iter_mut().enumerate() {
            let values = &self.values[row * self.columns..(row + 1) * self.columns];
            *slot = dot_dense(values, weights) + bias;
        }
    }

    fn add_weighted_rows(&self, factors: &[f64], sums: &mut [f64]) {
        for (row, factor) in factors.iter().enumerate() {
            let values = &self.values[row * self.columns..(row + 1) * self.columns];
            for (sum, value) in sums.iter_mut().zip(values) {
                *sum += value * factor;
            }
        }
    }
}

/// `sum_k values[k] * weights[indices[k]]`, added in the order of the entries.
fn dot_entries(indices: &[u32], values: &[f64], weights: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut entry = 0;
    while entry < indices.len() {
        sum += values[entry] * weights[indices[entry] as usize];
        entry += 1;
    }
    sum
}

/// `sum_j values[j] * weights[j]`, added in ascending order of `j`.
fn dot_dense(values: &[f64], weights: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (value, weight) in values.iter().zip(weights) {
        sum += value * weight;
    }
    sum
}

/// The logistic function, evaluated without overflow for any finite `z`.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// `ln(1 + exp(t))`, evaluated without overflow for any finite `t`.
fn softplus(t: f64) -> f64 {
    t.max(0.0) + (-t.abs()).exp().ln_1p()
}

/// The objective of the module documentation over a row matrix. The parameter vector
/// holds the weights first and the bias last.
///
/// The solver asks for the value and then the gradient at one point, and for the gradient
/// of an accepted point a second time. The margins and the gradient of the last point are
/// therefore kept. Both are pure functions of the point, so reusing them changes no bit
/// of the result; it only saves passes over the rows.
struct Objective<'a, D> {
    design: &'a D,
    positive: &'a [bool],
    l2: f64,
    last: RefCell<Option<Evaluation>>,
}

/// What is known about the objective at one parameter vector.
struct Evaluation {
    parameters: Vec<f64>,
    /// `x_i . w + b` for every row.
    margins: Vec<f64>,
    gradient: Option<Vec<f64>>,
}

impl<'a, D: Design> Objective<'a, D> {
    fn new(design: &'a D, positive: &'a [bool], l2: f64) -> Self {
        Self {
            design,
            positive,
            l2,
            last: RefCell::new(None),
        }
    }

    /// Runs `use_evaluation` on the evaluation at `parameters`, computing the margins
    /// first unless the last point was the same, bit for bit.
    fn at<R>(&self, parameters: &[f64], use_evaluation: impl FnOnce(&mut Evaluation) -> R) -> R {
        let mut last = self.last.borrow_mut();
        let known = last.as_ref().is_some_and(|evaluation| {
            evaluation.parameters.len() == parameters.len()
                && evaluation
                    .parameters
                    .iter()
                    .zip(parameters)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        });
        if !known {
            let columns = self.design.columns();
            let mut margins = vec![0.0; self.design.rows()];
            self.design
                .margins(&parameters[..columns], parameters[columns], &mut margins);
            *last = Some(Evaluation {
                parameters: parameters.to_vec(),
                margins,
                gradient: None,
            });
        }
        use_evaluation(last.as_mut().expect("the evaluation was just stored"))
    }

    fn value(&self, parameters: &[f64]) -> f64 {
        let loss = self.at(parameters, |evaluation| {
            let mut loss = 0.0;
            for (margin, positive) in evaluation.margins.iter().zip(self.positive) {
                loss += softplus(if *positive { -margin } else { *margin });
            }
            loss
        });
        let mut squares = 0.0;
        for weight in &parameters[..self.design.columns()] {
            squares += weight * weight;
        }
        loss + 0.5 * self.l2 * squares
    }

    fn gradient(&self, parameters: &[f64]) -> Vec<f64> {
        self.at(parameters, |evaluation| {
            if evaluation.gradient.is_none() {
                evaluation.gradient = Some(self.gradient_from(parameters, &evaluation.margins));
            }
            evaluation.gradient.clone().unwrap_or_default()
        })
    }

    fn gradient_from(&self, parameters: &[f64], margins: &[f64]) -> Vec<f64> {
        let columns = self.design.columns();
        // Residual of a row: probability minus target, written so that neither class
        // loses digits near a probability of 0 or 1.
        let residuals: Vec<f64> = margins
            .iter()
            .zip(self.positive)
            .map(|(margin, positive)| {
                if *positive {
                    -sigmoid(-margin)
                } else {
                    sigmoid(*margin)
                }
            })
            .collect();
        let mut gradient = vec![0.0; columns + 1];
        self.design
            .add_weighted_rows(&residuals, &mut gradient[..columns]);
        for (slot, weight) in gradient.iter_mut().zip(&parameters[..columns]) {
            *slot += self.l2 * weight;
        }
        let mut bias_slope = 0.0;
        for residual in &residuals {
            bias_slope += residual;
        }
        gradient[columns] = bias_slope;
        gradient
    }
}

fn contiguous(parameters: &Array1<f64>) -> Result<&[f64], SolverError> {
    parameters
        .as_slice()
        .ok_or_else(|| SolverError::msg("the parameter vector is not contiguous"))
}

impl<D: Design> CostFunction for Objective<'_, D> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, parameters: &Self::Param) -> Result<Self::Output, SolverError> {
        Ok(self.value(contiguous(parameters)?))
    }
}

impl<D: Design> Gradient for Objective<'_, D> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, parameters: &Self::Param) -> Result<Self::Gradient, SolverError> {
        Ok(Array1::from_vec(Objective::gradient(
            self,
            contiguous(parameters)?,
        )))
    }
}

fn solver_error(error: SolverError) -> LogRegError {
    LogRegError::Solver(error.to_string())
}

/// Fits the model on a checked row matrix.
fn fit<D: Design>(
    design: &D,
    labels: &[bool],
    settings: &LogRegSettings,
) -> Result<(LogReg, FitReport), LogRegError> {
    settings.validate()?;
    if design.rows() == 0 {
        return Err(LogRegError::NoRows);
    }
    if labels.len() != design.rows() {
        return Err(LogRegError::LabelCount {
            rows: design.rows(),
            labels: labels.len(),
        });
    }
    let positive = labels.iter().filter(|label| **label).count();
    let negative = labels.len() - positive;
    if positive == 0 || negative == 0 {
        return Err(LogRegError::SingleClass { positive, negative });
    }

    let columns = design.columns();
    let line_search: MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64> =
        MoreThuenteLineSearch::new();
    let solver = LBFGS::new(line_search, HISTORY)
        .with_tolerance_grad(settings.tolerance)
        .map_err(solver_error)?;
    let start = Array1::<f64>::zeros(columns + 1);
    let result = Executor::new(Objective::new(design, labels, settings.l2), solver)
        .configure(|state| state.param(start).max_iters(u64::from(settings.max_iter)))
        .run()
        .map_err(solver_error)?;

    let mut state = result.state;
    let termination = match state.get_termination_status() {
        TerminationStatus::Terminated(TerminationReason::SolverConverged) => Termination::Converged,
        TerminationStatus::Terminated(TerminationReason::MaxItersReached) => {
            Termination::IterationCap
        }
        TerminationStatus::Terminated(TerminationReason::SolverExit(_)) => {
            Termination::LineSearchStopped
        }
        other => return Err(LogRegError::Solver(format!("unexpected end: {other:?}"))),
    };
    let iterations = state.get_iter();
    let parameters = match state.take_best_param() {
        Some(best) => best.to_vec(),
        None => vec![0.0; columns + 1],
    };
    if parameters.iter().any(|value| !value.is_finite()) {
        return Err(LogRegError::NonFiniteModel);
    }

    let objective = Objective::new(design, labels, settings.l2);
    let gradient = objective.gradient(&parameters);
    let mut squares = 0.0;
    for slope in &gradient {
        squares += slope * slope;
    }
    let report = FitReport {
        iterations,
        termination,
        objective: objective.value(&parameters),
        gradient_norm: squares.sqrt(),
    };
    let model = LogReg {
        bias: parameters[columns],
        weights: {
            let mut weights = parameters;
            weights.truncate(columns);
            weights
        },
    };
    Ok((model, report))
}

/// A fitted logistic regression: one weight per column and a bias.
#[derive(Debug, Clone, PartialEq)]
pub struct LogReg {
    weights: Vec<f64>,
    bias: f64,
}

impl LogReg {
    /// Fits hashed rows whose columns are `0..dimension`; `labels[i]` is `true` when row
    /// `i` is positive. Size `dimension` from [`crate::hashing::CharHasher::dimension`].
    pub fn fit_sparse(
        rows: &[SparseRow],
        dimension: u32,
        labels: &[bool],
        settings: &LogRegSettings,
    ) -> Result<(Self, FitReport), LogRegError> {
        fit(&SparseMatrix::from_rows(dimension, rows)?, labels, settings)
    }

    /// Fits the rows of a sparse matrix; `labels[i]` is `true` when row `i` is positive.
    pub fn fit_matrix(
        matrix: &SparseMatrix,
        labels: &[bool],
        settings: &LogRegSettings,
    ) -> Result<(Self, FitReport), LogRegError> {
        fit(matrix, labels, settings)
    }

    /// Fits dense rows of equal length (`f64` or `f32` values, as vectors, arrays or
    /// slices); `labels[i]` is `true` when row `i` is positive. The dimension is the
    /// length of the first row.
    pub fn fit_dense<R: AsRef<[T]>, T: Copy + Into<f64>>(
        rows: &[R],
        labels: &[bool],
        settings: &LogRegSettings,
    ) -> Result<(Self, FitReport), LogRegError> {
        let columns = rows.first().map_or(0, |row| row.as_ref().len());
        let mut values = Vec::with_capacity(rows.len() * columns);
        for (row, cells) in rows.iter().enumerate() {
            let cells = cells.as_ref();
            if cells.len() != columns {
                return Err(LogRegError::RowLength {
                    row,
                    expected: columns,
                    found: cells.len(),
                });
            }
            for cell in cells {
                let value: f64 = (*cell).into();
                if !value.is_finite() {
                    return Err(LogRegError::NonFinite { row });
                }
                values.push(value);
            }
        }
        let matrix = DenseMatrix {
            rows: rows.len(),
            columns,
            values,
        };
        fit(&matrix, labels, settings)
    }

    /// A model from stored parameters, which must be finite.
    pub fn from_parts(weights: Vec<f64>, bias: f64) -> Result<Self, LogRegError> {
        if !bias.is_finite() || weights.iter().any(|weight| !weight.is_finite()) {
            return Err(LogRegError::NonFiniteModel);
        }
        Ok(Self { weights, bias })
    }

    /// The weight of every column.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The bias (intercept).
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Number of columns the model was fitted on.
    pub fn dimension(&self) -> usize {
        self.weights.len()
    }

    /// Probability, in `0.0..=1.0`, that a hashed row is positive. An empty row scores the
    /// logistic function of the bias.
    pub fn probability_sparse(&self, row: &SparseRow) -> Result<f64, LogRegError> {
        self.probability_entries(row.indices(), row.values())
    }

    /// Probability that a sparse row, given as strictly ascending column indices and
    /// their values, is positive.
    pub fn probability_entries(&self, indices: &[u32], values: &[f64]) -> Result<f64, LogRegError> {
        check_entries(0, indices, values, self.weights.len())?;
        Ok(sigmoid(
            dot_entries(indices, values, &self.weights) + self.bias,
        ))
    }

    /// Probability that a dense row of the model's dimension is positive.
    pub fn probability_dense<T: Copy + Into<f64>>(&self, row: &[T]) -> Result<f64, LogRegError> {
        if row.len() != self.weights.len() {
            return Err(LogRegError::RowLength {
                row: 0,
                expected: self.weights.len(),
                found: row.len(),
            });
        }
        let mut sum = 0.0;
        for (cell, weight) in row.iter().zip(&self.weights) {
            let value: f64 = (*cell).into();
            if !value.is_finite() {
                return Err(LogRegError::NonFinite { row: 0 });
            }
            sum += value * weight;
        }
        Ok(sigmoid(sum + self.bias))
    }
}
