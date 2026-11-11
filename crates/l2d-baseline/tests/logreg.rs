//! Deterministic L2-penalized logistic regression (task 1.6; requirements 2.1, 2.4, 2.7).
//!
//! The objective pinned by these tests, for weights `w`, bias `b`, rows `x_i` and signs
//! `s_i` (`+1` for a positive row, `-1` for a negative one):
//!
//! ```text
//! F(w, b) = sum_i ln(1 + exp(-s_i * (x_i . w + b))) + (l2 / 2) * ||w||^2
//! ```
//!
//! The loss is a sum over rows, not a mean, and the bias is not penalized.

use std::collections::BTreeSet;
use std::process::Command;
use std::time::Instant;

use l2d_baseline::hashing::{CharHasher, HashingSettings, RoutingMode, SparseRow};
use l2d_baseline::logreg::{
    FitReport, LogReg, LogRegError, LogRegSettings, SparseMatrix, Termination,
};

// ---------------------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------------------

/// A fixed linear congruential sequence: varied numbers without randomness.
struct Sequence(u64);

impl Sequence {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// A value in `[-1, 1)`.
    fn next_signed(&mut self) -> f64 {
        f64::from(self.next_u32() % 2_000_001) / 1_000_000.0 - 1.0
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn settings(l2: f64, max_iter: u32, tolerance: f64) -> LogRegSettings {
    LogRegSettings {
        l2,
        max_iter,
        tolerance,
    }
}

fn bits(model: &LogReg) -> Vec<u64> {
    let mut bits: Vec<u64> = model.weights().iter().map(|w| w.to_bits()).collect();
    bits.push(model.bias().to_bits());
    bits
}

type Entries = Vec<(u32, f64)>;

/// A noisy two-class problem: `rows` sparse rows with `per_row` entries each. The labels
/// follow a fixed weight vector, with a tenth of them flipped, so the classes overlap and
/// the optimum is finite and not trivial.
fn noisy_problem(
    rows: usize,
    dimension: u32,
    per_row: usize,
    seed: u64,
) -> (Vec<Entries>, Vec<bool>) {
    let mut sequence = Sequence(seed);
    let truth: Vec<f64> = (0..dimension).map(|_| sequence.next_signed()).collect();
    let mut data = Vec::with_capacity(rows);
    let mut labels = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut indices = BTreeSet::new();
        while indices.len() < per_row {
            indices.insert(sequence.next_u32() % dimension);
        }
        let entries: Entries = indices
            .into_iter()
            .map(|index| {
                let value = sequence.next_signed();
                (index, if value == 0.0 { 0.5 } else { value })
            })
            .collect();
        let margin: f64 = entries
            .iter()
            .map(|(index, value)| value * truth[*index as usize])
            .sum();
        let flipped = row % 10 == 3;
        labels.push((margin > 0.0) != flipped);
        data.push(entries);
    }
    assert!(labels.iter().any(|label| *label) && labels.iter().any(|label| !*label));
    (data, labels)
}

fn matrix(dimension: u32, data: &[Entries]) -> SparseMatrix {
    let mut matrix = SparseMatrix::new(dimension);
    for entries in data {
        let indices: Vec<u32> = entries.iter().map(|(index, _)| *index).collect();
        let values: Vec<f64> = entries.iter().map(|(_, value)| *value).collect();
        matrix.push_row(&indices, &values).expect("a valid row");
    }
    matrix
}

fn densified(dimension: u32, data: &[Entries]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|entries| {
            let mut row = vec![0.0; dimension as usize];
            for (index, value) in entries {
                row[*index as usize] = *value;
            }
            row
        })
        .collect()
}

/// The documented objective and its gradient, written a second time in the plainest form
/// on dense rows. The gradient holds the weights first and the bias last.
fn reference_objective(
    rows: &[Vec<f64>],
    labels: &[bool],
    l2: f64,
    weights: &[f64],
    bias: f64,
) -> (f64, Vec<f64>) {
    let mut objective = 0.0;
    let mut gradient = vec![0.0; weights.len() + 1];
    for (row, label) in rows.iter().zip(labels) {
        let margin: f64 = row.iter().zip(weights).map(|(x, w)| x * w).sum::<f64>() + bias;
        let sign = if *label { 1.0 } else { -1.0 };
        objective += (1.0 + (-sign * margin).exp()).ln();
        let residual = sigmoid(margin) - if *label { 1.0 } else { 0.0 };
        for (slot, x) in gradient.iter_mut().zip(row) {
            *slot += residual * x;
        }
        gradient[weights.len()] += residual;
    }
    for (slot, w) in gradient.iter_mut().zip(weights) {
        *slot += l2 * w;
        objective += 0.5 * l2 * w * w;
    }
    (objective, gradient)
}

fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn max_difference(a: &LogReg, b: &LogReg) -> f64 {
    assert_eq!(a.dimension(), b.dimension());
    a.weights()
        .iter()
        .zip(b.weights())
        .map(|(x, y)| (x - y).abs())
        .fold((a.bias() - b.bias()).abs(), f64::max)
}

// ---------------------------------------------------------------------------------------
// Separable toy data is classified without error.
// ---------------------------------------------------------------------------------------

#[test]
fn separable_sparse_rows_are_classified_without_error() {
    // Positive rows use the features 0 to 2, negative rows the features 3 to 5.
    let mut data: Vec<Entries> = Vec::new();
    let mut labels = Vec::new();
    for step in 0..12u32 {
        let positive = step % 2 == 0;
        let base = if positive { 0 } else { 3 };
        let a = base + step % 3;
        let b = base + (step + 1) % 3;
        let (low, high) = (a.min(b), a.max(b));
        data.push(vec![(low, 0.6), (high, 0.8)]);
        labels.push(positive);
    }
    let matrix = matrix(6, &data);
    let (model, report) =
        LogReg::fit_matrix(&matrix, &labels, &settings(0.01, 200, 1e-6)).expect("the fit runs");
    assert!(report.converged(), "{report:?}");
    assert_eq!(model.dimension(), 6);
    for (entries, label) in data.iter().zip(&labels) {
        let indices: Vec<u32> = entries.iter().map(|(index, _)| *index).collect();
        let values: Vec<f64> = entries.iter().map(|(_, value)| *value).collect();
        let probability = model.probability_entries(&indices, &values).unwrap();
        assert!((0.0..=1.0).contains(&probability));
        assert_eq!(probability > 0.5, *label, "row {entries:?}: {probability}");
    }
}

#[test]
fn separable_hashed_texts_are_classified_without_error() {
    let hasher = CharHasher::default();
    let good = [
        "GET /shop/list.jsp?page=1",
        "GET /shop/list.jsp?page=2",
        "GET /shop/item.jsp?id=17",
        "POST /shop/cart.jsp\nid=17&count=2",
        "GET /shop/item.jsp?id=204",
        "POST /shop/login.jsp\nuser=anna&pwd=secret",
    ];
    let bad = [
        "GET /shop/item.jsp?id=17' OR '1'='1",
        "GET /shop/list.jsp?page=<script>alert(1)</script>",
        "POST /shop/login.jsp\nuser=admin'--&pwd=x",
        "GET /shop/item.jsp?id=../../../../etc/passwd",
        "POST /shop/cart.jsp\nid=17;DROP TABLE users&count=2",
        "GET /shop/list.jsp?page=1 UNION SELECT pwd FROM users",
    ];
    let rows: Vec<SparseRow> = good.iter().chain(&bad).map(|t| hasher.row(t)).collect();
    let labels: Vec<bool> = (0..rows.len()).map(|i| i >= good.len()).collect();
    let (model, _) = LogReg::fit_sparse(
        &rows,
        hasher.dimension(),
        &labels,
        &settings(0.01, 200, 1e-6),
    )
    .expect("the fit runs");
    assert_eq!(model.dimension(), 65_536);
    for (row, label) in rows.iter().zip(&labels) {
        let probability = model.probability_sparse(row).unwrap();
        assert_eq!(probability > 0.5, *label);
    }
}

#[test]
fn separable_dense_rows_are_classified_without_error() {
    // Positive exactly when x0 + x1 > 3: needs the intercept.
    let mut rows: Vec<[f64; 2]> = Vec::new();
    let mut labels = Vec::new();
    for a in 0..6 {
        for b in 0..6 {
            if a + b == 3 {
                continue;
            }
            rows.push([f64::from(a), f64::from(b)]);
            labels.push(a + b > 3);
        }
    }
    let (model, report) =
        LogReg::fit_dense(&rows, &labels, &settings(0.001, 500, 1e-6)).expect("the fit runs");
    assert!(report.converged(), "{report:?}");
    assert!(model.bias() < 0.0, "the boundary is off the origin");
    for (row, label) in rows.iter().zip(&labels) {
        assert_eq!(model.probability_dense(row).unwrap() > 0.5, *label);
    }
}

#[test]
fn separable_dense_single_precision_vectors_are_classified_without_error() {
    // Two shifted clouds of 128-dimension vectors, as the separability measure passes them.
    let mut sequence = Sequence(77);
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut labels = Vec::new();
    for row in 0..120 {
        let positive = row % 3 == 0;
        let shift = if positive { 0.75 } else { -0.75 };
        rows.push(
            (0..128)
                .map(|column| {
                    let centre = if column % 2 == 0 { shift } else { -shift };
                    (centre + 0.5 * sequence.next_signed()) as f32
                })
                .collect(),
        );
        labels.push(positive);
    }
    let (model, _) =
        LogReg::fit_dense(&rows, &labels, &LogRegSettings::default()).expect("the fit runs");
    assert_eq!(model.dimension(), 128);
    for (row, label) in rows.iter().zip(&labels) {
        assert_eq!(model.probability_dense(row).unwrap() > 0.5, *label);
    }
    // Slices of rows are accepted as well.
    let slices: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    let (again, _) = LogReg::fit_dense(&slices, &labels, &LogRegSettings::default()).unwrap();
    assert_eq!(bits(&again), bits(&model));
}

// ---------------------------------------------------------------------------------------
// Determinism.
// ---------------------------------------------------------------------------------------

#[test]
fn a_second_fit_gives_bit_equal_weights() {
    let (data, labels) = noisy_problem(300, 500, 20, 1);
    let sparse = matrix(500, &data);
    let chosen = settings(0.5, 100, 1e-6);
    let (first, first_report) = LogReg::fit_matrix(&sparse, &labels, &chosen).unwrap();
    let (second, second_report) = LogReg::fit_matrix(&sparse, &labels, &chosen).unwrap();
    assert!(first.weights().iter().any(|w| *w != 0.0));
    assert_eq!(bits(&first), bits(&second));
    assert_eq!(first_report, second_report);

    let dense = densified(500, &data);
    let (third, _) = LogReg::fit_dense(&dense, &labels, &chosen).unwrap();
    let (fourth, _) = LogReg::fit_dense(&dense, &labels, &chosen).unwrap();
    assert_eq!(bits(&third), bits(&fourth));
}

#[test]
fn sparse_and_dense_rows_give_bit_equal_models() {
    for (seed, l2) in [(2, 1.0), (3, 0.05), (4, 10.0)] {
        let (data, labels) = noisy_problem(200, 64, 9, seed);
        let chosen = settings(l2, 100, 1e-6);
        let (sparse, sparse_report) =
            LogReg::fit_matrix(&matrix(64, &data), &labels, &chosen).unwrap();
        let (dense, dense_report) =
            LogReg::fit_dense(&densified(64, &data), &labels, &chosen).unwrap();
        assert_eq!(bits(&sparse), bits(&dense), "seed {seed}");
        assert_eq!(sparse_report, dense_report, "seed {seed}");
        // And both score a row alike.
        let row = &densified(64, &data)[0];
        let indices: Vec<u32> = data[0].iter().map(|(index, _)| *index).collect();
        let values: Vec<f64> = data[0].iter().map(|(_, value)| *value).collect();
        assert_eq!(
            sparse
                .probability_entries(&indices, &values)
                .unwrap()
                .to_bits(),
            dense.probability_dense(row).unwrap().to_bits()
        );
    }
}

#[test]
fn row_order_is_part_of_the_input() {
    // Not a promise of order independence: the same rows in the same order give the same
    // bits, and that is all that is claimed. Reversed rows still reach the same optimum.
    let (data, labels) = noisy_problem(150, 40, 6, 5);
    let chosen = settings(1.0, 300, 1e-8);
    let (forward, _) = LogReg::fit_matrix(&matrix(40, &data), &labels, &chosen).unwrap();
    let reversed_data: Vec<Entries> = data.iter().rev().cloned().collect();
    let reversed_labels: Vec<bool> = labels.iter().rev().copied().collect();
    let (backward, _) =
        LogReg::fit_matrix(&matrix(40, &reversed_data), &reversed_labels, &chosen).unwrap();
    assert!(max_difference(&forward, &backward) < 1e-6);
}

// ---------------------------------------------------------------------------------------
// The optimum, verified independently.
// ---------------------------------------------------------------------------------------

#[test]
fn rows_without_features_give_the_closed_form_bias() {
    // With no features the optimum is bias = ln(positive / negative), whatever the
    // penalty: the bias is not penalized.
    let rows = vec![SparseRow::default(); 40];
    let labels: Vec<bool> = (0..40).map(|i| i < 30).collect();
    for l2 in [0.0, 1.0, 1e6] {
        let (model, report) =
            LogReg::fit_sparse(&rows, 16, &labels, &settings(l2, 100, 1e-9)).unwrap();
        assert!(report.converged(), "{report:?}");
        assert!(model.weights().iter().all(|w| *w == 0.0));
        assert!(
            (model.bias() - 3.0f64.ln()).abs() < 1e-8,
            "l2 {l2}: bias {}",
            model.bias()
        );
        let probability = model.probability_sparse(&SparseRow::default()).unwrap();
        assert!((probability - 0.75).abs() < 1e-8);
    }
}

#[test]
fn the_gradient_of_the_documented_objective_vanishes_at_the_solution() {
    let (data, labels) = noisy_problem(250, 30, 5, 6);
    let dense = densified(30, &data);
    for l2 in [0.1, 1.0, 25.0] {
        let (model, report) =
            LogReg::fit_matrix(&matrix(30, &data), &labels, &settings(l2, 500, 1e-7)).unwrap();
        assert!(report.converged(), "{report:?}");
        let (objective, gradient) =
            reference_objective(&dense, &labels, l2, model.weights(), model.bias());
        assert!(
            norm(&gradient) < 1e-6,
            "l2 {l2}: gradient norm {}",
            norm(&gradient)
        );
        assert!((report.gradient_norm - norm(&gradient)).abs() < 1e-9);
        assert!((report.objective - objective).abs() < 1e-9 * objective.max(1.0));

        // The objective is convex, so no nearby point is better.
        let mut sequence = Sequence(9);
        for _ in 0..20 {
            let weights: Vec<f64> = model
                .weights()
                .iter()
                .map(|w| w + 0.01 * sequence.next_signed())
                .collect();
            let bias = model.bias() + 0.01 * sequence.next_signed();
            let (moved, _) = reference_objective(&dense, &labels, l2, &weights, bias);
            assert!(moved > objective);
        }
    }
}

#[test]
fn the_loss_is_a_sum_over_rows() {
    // Every row twice with twice the penalty is the same problem times two.
    let (data, labels) = noisy_problem(120, 20, 4, 7);
    let mut twice = data.clone();
    twice.extend(data.iter().cloned());
    let mut twice_labels = labels.clone();
    twice_labels.extend(&labels);
    let (once, _) =
        LogReg::fit_matrix(&matrix(20, &data), &labels, &settings(0.5, 500, 1e-8)).unwrap();
    let (doubled, _) = LogReg::fit_matrix(
        &matrix(20, &twice),
        &twice_labels,
        &settings(1.0, 500, 1e-8),
    )
    .unwrap();
    assert!(max_difference(&once, &doubled) < 1e-6);
    // The same rows with the same penalty are a different problem.
    let (other, _) = LogReg::fit_matrix(
        &matrix(20, &twice),
        &twice_labels,
        &settings(0.5, 500, 1e-8),
    )
    .unwrap();
    assert!(max_difference(&once, &other) > 1e-3);
}

#[test]
fn a_stronger_penalty_gives_smaller_weights() {
    let (data, labels) = noisy_problem(200, 30, 5, 8);
    let sparse = matrix(30, &data);
    let mut previous = f64::INFINITY;
    for l2 in [0.01, 1.0, 100.0] {
        let (model, _) = LogReg::fit_matrix(&sparse, &labels, &settings(l2, 500, 1e-7)).unwrap();
        let size = norm(model.weights());
        assert!(size < previous, "l2 {l2}: {size} is not below {previous}");
        previous = size;
    }
}

#[test]
fn the_dense_fit_agrees_with_linfa_logistic() {
    use linfa::prelude::*;
    use linfa_logistic::LogisticRegression;
    use ndarray::{Array1, Array2};

    let (data, labels) = noisy_problem(180, 12, 4, 10);
    let dense = densified(12, &data);
    let (model, report) = LogReg::fit_dense(&dense, &labels, &settings(0.7, 500, 1e-8)).unwrap();
    assert!(report.converged(), "{report:?}");

    let flat: Vec<f64> = dense.iter().flatten().copied().collect();
    let records = Array2::from_shape_vec((dense.len(), 12), flat).unwrap();
    let targets = Array1::from_vec(labels.clone());
    let reference = LogisticRegression::default()
        .alpha(0.7)
        .max_iterations(500)
        .gradient_tolerance(1e-8)
        .fit(&Dataset::new(records, targets))
        .expect("linfa-logistic fits the same data");

    // linfa-logistic chooses which class counts as positive from the order and the counts
    // of the labels, so its model may be the mirror image.
    let sign = if (reference.intercept() - model.bias()).abs()
        <= (reference.intercept() + model.bias()).abs()
    {
        1.0
    } else {
        -1.0
    };
    assert!(model.bias().abs() > 1e-3 && norm(model.weights()) > 0.1);
    assert!((sign * reference.intercept() - model.bias()).abs() < 1e-6);
    for (theirs, ours) in reference.params().iter().zip(model.weights()) {
        assert!(
            (sign * theirs - ours).abs() < 1e-6,
            "linfa {theirs}, own {ours}"
        );
    }
}

// ---------------------------------------------------------------------------------------
// The iteration cap.
// ---------------------------------------------------------------------------------------

#[test]
fn the_iteration_cap_stops_the_fit_and_is_reported() {
    let (data, labels) = noisy_problem(300, 200, 12, 11);
    let sparse = matrix(200, &data);

    let (_, full_report) =
        LogReg::fit_matrix(&sparse, &labels, &settings(0.01, 2000, 1e-6)).unwrap();
    assert_eq!(full_report.termination, Termination::Converged);
    assert!(full_report.iterations > 3 && full_report.iterations < 2000);
    assert!(full_report.gradient_norm < 1e-6);

    let capped_settings = settings(0.01, 3, 1e-6);
    let (capped, capped_report) = LogReg::fit_matrix(&sparse, &labels, &capped_settings).unwrap();
    assert_eq!(capped_report.termination, Termination::IterationCap);
    assert!(!capped_report.converged());
    assert_eq!(capped_report.iterations, 3);
    assert!(capped_report.gradient_norm > 1e-6);
    assert!(capped_report.objective > full_report.objective);
    // A capped fit is still a usable, finite model, better than the zero model ...
    assert!(capped.weights().iter().all(|w| w.is_finite()) && capped.bias().is_finite());
    let zero_objective = labels.len() as f64 * 2.0f64.ln();
    assert!(capped_report.objective < zero_objective);
    // ... and as reproducible as a converged one.
    let (again, again_report) = LogReg::fit_matrix(&sparse, &labels, &capped_settings).unwrap();
    assert_eq!(bits(&capped), bits(&again));
    assert_eq!(capped_report, again_report);
}

#[test]
fn a_problem_solved_by_the_zero_model_needs_no_iteration() {
    // The same row once with each label: the zero model is the optimum.
    let rows = [[1.0, -2.0], [1.0, -2.0]];
    let (model, report) =
        LogReg::fit_dense(&rows, &[true, false], &LogRegSettings::default()).unwrap();
    assert_eq!(
        report,
        FitReport {
            iterations: 0,
            termination: Termination::Converged,
            objective: 2.0 * 2.0f64.ln(),
            gradient_norm: 0.0,
        }
    );
    assert_eq!(model.weights(), &[0.0, 0.0]);
    assert_eq!(model.bias(), 0.0);
    assert_eq!(model.probability_dense(&rows[0]).unwrap(), 0.5);
}

// ---------------------------------------------------------------------------------------
// Scoring.
// ---------------------------------------------------------------------------------------

#[test]
fn an_empty_sparse_row_scores_the_sigmoid_of_the_bias() {
    let model = LogReg::from_parts(vec![2.0, -1.0, 0.5], 0.3).unwrap();
    assert_eq!(model.weights(), &[2.0, -1.0, 0.5]);
    assert_eq!(model.bias(), 0.3);
    assert_eq!(model.dimension(), 3);
    let empty = model.probability_sparse(&SparseRow::default()).unwrap();
    assert!((empty - sigmoid(0.3)).abs() < 1e-15);
    assert_eq!(
        model.probability_entries(&[], &[]).unwrap().to_bits(),
        empty.to_bits()
    );
    assert_eq!(
        model.probability_dense(&[0.0, 0.0, 0.0]).unwrap().to_bits(),
        empty.to_bits()
    );
}

#[test]
fn probabilities_follow_the_weights_and_stay_in_range() {
    let model = LogReg::from_parts(vec![2.0, -1.0, 0.5], 0.3).unwrap();
    let expected = sigmoid(2.0 * 0.5 - 4.0 + 0.3);
    let sparse = model.probability_entries(&[0, 1], &[0.5, 4.0]).unwrap();
    let dense = model.probability_dense(&[0.5f32, 4.0, 0.0]).unwrap();
    assert!((sparse - expected).abs() < 1e-15);
    assert_eq!(sparse.to_bits(), dense.to_bits());

    // Far out the probability saturates without leaving the range or becoming NaN.
    let steep = LogReg::from_parts(vec![1.0], 0.0).unwrap();
    assert_eq!(steep.probability_dense(&[1e6]).unwrap(), 1.0);
    assert_eq!(steep.probability_dense(&[-1e6]).unwrap(), 0.0);
    assert!(steep.probability_dense(&[-700.0]).unwrap() > 0.0);
}

#[test]
fn rows_that_do_not_fit_the_model_are_refused_when_scoring() {
    let model = LogReg::from_parts(vec![2.0, -1.0, 0.5], 0.3).unwrap();
    assert_eq!(
        model.probability_dense(&[1.0, 2.0]),
        Err(LogRegError::RowLength {
            row: 0,
            expected: 3,
            found: 2
        })
    );
    assert_eq!(
        model.probability_dense(&[1.0, f64::NAN, 2.0]),
        Err(LogRegError::NonFinite { row: 0 })
    );
    assert_eq!(
        model.probability_entries(&[1, 3], &[1.0, 1.0]),
        Err(LogRegError::IndexOutOfRange {
            row: 0,
            index: 3,
            dimension: 3
        })
    );
    assert_eq!(
        model.probability_entries(&[2, 1], &[1.0, 1.0]),
        Err(LogRegError::UnorderedIndices { row: 0 })
    );
    assert_eq!(
        model.probability_entries(&[1], &[1.0, 1.0]),
        Err(LogRegError::EntryCount {
            row: 0,
            indices: 1,
            values: 2
        })
    );
    assert_eq!(
        model.probability_entries(&[1], &[f64::INFINITY]),
        Err(LogRegError::NonFinite { row: 0 })
    );
    // A hashed row of the default feature space does not fit a three-column model.
    let row = CharHasher::default().row("GET /index.jsp");
    assert!(matches!(
        model.probability_sparse(&row),
        Err(LogRegError::IndexOutOfRange { dimension: 3, .. })
    ));

    assert_eq!(
        LogReg::from_parts(vec![1.0, f64::NAN], 0.0),
        Err(LogRegError::NonFiniteModel)
    );
    assert_eq!(
        LogReg::from_parts(vec![1.0], f64::NEG_INFINITY),
        Err(LogRegError::NonFiniteModel)
    );
}

// ---------------------------------------------------------------------------------------
// Degenerate input is refused.
// ---------------------------------------------------------------------------------------

#[test]
fn degenerate_training_input_is_refused() {
    let chosen = LogRegSettings::default();
    let rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    // No rows.
    let none: [[f64; 2]; 0] = [];
    assert_eq!(
        LogReg::fit_dense(&none, &[], &chosen).unwrap_err(),
        LogRegError::NoRows
    );
    assert_eq!(
        LogReg::fit_sparse(&[], 8, &[], &chosen).unwrap_err(),
        LogRegError::NoRows
    );
    assert_eq!(
        LogReg::fit_matrix(&SparseMatrix::new(8), &[], &chosen).unwrap_err(),
        LogRegError::NoRows
    );

    // As many labels as rows.
    assert_eq!(
        LogReg::fit_dense(&rows, &[true, false], &chosen).unwrap_err(),
        LogRegError::LabelCount { rows: 3, labels: 2 }
    );
    assert_eq!(
        LogReg::fit_sparse(&[SparseRow::default()], 8, &[true, false], &chosen).unwrap_err(),
        LogRegError::LabelCount { rows: 1, labels: 2 }
    );

    // Both classes.
    assert_eq!(
        LogReg::fit_dense(&rows, &[true, true, true], &chosen).unwrap_err(),
        LogRegError::SingleClass {
            positive: 3,
            negative: 0
        }
    );
    assert_eq!(
        LogReg::fit_dense(&rows, &[false, false, false], &chosen).unwrap_err(),
        LogRegError::SingleClass {
            positive: 0,
            negative: 3
        }
    );

    // Rows of one length.
    let ragged = vec![vec![1.0, 2.0], vec![3.0], vec![5.0, 6.0]];
    assert_eq!(
        LogReg::fit_dense(&ragged, &[true, false, true], &chosen).unwrap_err(),
        LogRegError::RowLength {
            row: 1,
            expected: 2,
            found: 1
        }
    );

    // Finite values only.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let rows = [[1.0, 2.0], [3.0, 4.0], [bad, 6.0]];
        assert_eq!(
            LogReg::fit_dense(&rows, &[true, false, true], &chosen).unwrap_err(),
            LogRegError::NonFinite { row: 2 }
        );
    }
    let single = [[1.0f32, f32::NAN], [3.0, 4.0]];
    assert_eq!(
        LogReg::fit_dense(&single, &[true, false], &chosen).unwrap_err(),
        LogRegError::NonFinite { row: 0 }
    );

    // Hashed rows inside the stated dimension.
    let hasher = CharHasher::default();
    let hashed = [hasher.row("GET /a?x=1"), hasher.row("GET /a?x=<script>")];
    assert!(matches!(
        LogReg::fit_sparse(&hashed, 16, &[false, true], &chosen).unwrap_err(),
        LogRegError::IndexOutOfRange {
            row: 0,
            dimension: 16,
            ..
        }
    ));
    // The isolate mode doubles the dimension; sizing by the bucket count is caught.
    let isolating = CharHasher::new(HashingSettings {
        routing: RoutingMode::Isolate,
        ..HashingSettings::default()
    })
    .unwrap();
    let isolated = [
        isolating.row("GET /a?x=1"),
        isolating.row("GET /a?x=<script>"),
    ];
    assert!(matches!(
        LogReg::fit_sparse(&isolated, 65_536, &[false, true], &chosen).unwrap_err(),
        LogRegError::IndexOutOfRange { .. }
    ));
    assert!(LogReg::fit_sparse(&isolated, isolating.dimension(), &[false, true], &chosen).is_ok());
}

#[test]
fn unusable_matrix_rows_are_refused() {
    let mut matrix = SparseMatrix::new(4);
    assert_eq!(matrix.dimension(), 4);
    matrix.push_row(&[0, 3], &[1.0, -1.0]).unwrap();
    matrix.push_row(&[], &[]).unwrap();
    assert_eq!(
        matrix.push_row(&[0, 4], &[1.0, 1.0]),
        Err(LogRegError::IndexOutOfRange {
            row: 2,
            index: 4,
            dimension: 4
        })
    );
    assert_eq!(
        matrix.push_row(&[2, 2], &[1.0, 1.0]),
        Err(LogRegError::UnorderedIndices { row: 2 })
    );
    assert_eq!(
        matrix.push_row(&[3, 1], &[1.0, 1.0]),
        Err(LogRegError::UnorderedIndices { row: 2 })
    );
    assert_eq!(
        matrix.push_row(&[1, 2], &[1.0]),
        Err(LogRegError::EntryCount {
            row: 2,
            indices: 2,
            values: 1
        })
    );
    assert_eq!(
        matrix.push_row(&[1], &[f64::NAN]),
        Err(LogRegError::NonFinite { row: 2 })
    );
    // A refused row leaves the matrix as it was.
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.stored(), 2);
    matrix.push_row(&[1], &[0.5]).unwrap();
    assert_eq!(matrix.rows(), 3);
    assert_eq!(matrix.stored(), 3);

    let hasher = CharHasher::default();
    let rows = [hasher.row("abc"), SparseRow::default(), hasher.row("abd")];
    let built = SparseMatrix::from_rows(hasher.dimension(), &rows).unwrap();
    assert_eq!(built.rows(), 3);
    assert_eq!(built.stored(), rows[0].len() + rows[2].len());
}

#[test]
fn unusable_settings_are_refused() {
    let rows = [[1.0], [-1.0]];
    let labels = [true, false];
    let refused = |chosen: LogRegSettings| LogReg::fit_dense(&rows, &labels, &chosen).unwrap_err();
    for l2 in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            refused(settings(l2, 100, 1e-4)),
            LogRegError::InvalidSettings { name: "l2", .. }
        ));
    }
    for tolerance in [-1e-9, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            refused(settings(1.0, 100, tolerance)),
            LogRegError::InvalidSettings {
                name: "tolerance",
                ..
            }
        ));
    }
    assert_eq!(
        refused(settings(1.0, 0, 1e-4)),
        LogRegError::InvalidSettings {
            name: "max_iter",
            value: 0.0
        }
    );
    // The defaults are usable and are the documented ones.
    let defaults = LogRegSettings::default();
    assert_eq!(
        (defaults.l2, defaults.max_iter, defaults.tolerance),
        (1.0, 100, 1e-4)
    );
    assert!(LogReg::fit_dense(&rows, &labels, &defaults).is_ok());
    // Error messages name what was found.
    assert_eq!(
        LogRegError::SingleClass {
            positive: 7,
            negative: 0
        }
        .to_string(),
        "only one class is present: 7 positive and 0 negative rows"
    );
}

// ---------------------------------------------------------------------------------------
// Bit-identical weights across processes.
// ---------------------------------------------------------------------------------------

const PROBE_VARIABLE: &str = "L2D_BASELINE_LOGREG_PROBE";
const PROBE_MARKER: &str = "L2D_LOGREG_MODEL ";

fn probe_texts() -> (Vec<String>, Vec<bool>) {
    let mut sequence = Sequence(21);
    let mut texts = Vec::new();
    let mut labels = Vec::new();
    for row in 0..160 {
        let bad = row % 4 == 1;
        let mut text = String::from("GET http://localhost:8080/shop/item.jsp?id=");
        for _ in 0..40 {
            text.push(char::from(b'a' + (sequence.next_u32() % 26) as u8));
        }
        if bad {
            text.push_str(["' OR '1'='1", "<script>", "../../etc"][row % 3]);
        }
        // Some good rows carry a fragment too, so the classes overlap.
        if row % 16 == 0 {
            text.push_str("<script>");
        }
        texts.push(text);
        labels.push(bad);
    }
    (texts, labels)
}

/// Fitted models as text: the bit patterns of every non-zero weight and of the bias.
fn probe_lines() -> Vec<String> {
    let (texts, labels) = probe_texts();
    let mut lines = Vec::new();
    for routing in [RoutingMode::Keep, RoutingMode::Isolate] {
        let hasher = CharHasher::new(HashingSettings {
            routing,
            ..HashingSettings::default()
        })
        .unwrap();
        let rows: Vec<SparseRow> = texts.iter().map(|text| hasher.row(text)).collect();
        let (model, report) = LogReg::fit_sparse(
            &rows,
            hasher.dimension(),
            &labels,
            &LogRegSettings::default(),
        )
        .unwrap();
        let weights: Vec<String> = model
            .weights()
            .iter()
            .enumerate()
            .filter(|(_, w)| **w != 0.0)
            .map(|(index, w)| format!("{index:x}:{:016x}", w.to_bits()))
            .collect();
        lines.push(format!(
            "{routing:?} iterations={} bias={:016x} objective={:016x} [{}]",
            report.iterations,
            model.bias().to_bits(),
            report.objective.to_bits(),
            weights.join(",")
        ));
    }
    let (data, labels) = noisy_problem(150, 128, 128, 22);
    let (model, report) =
        LogReg::fit_dense(&densified(128, &data), &labels, &settings(2.0, 50, 1e-6)).unwrap();
    let weights: Vec<String> = bits(&model).iter().map(|b| format!("{b:016x}")).collect();
    lines.push(format!(
        "dense iterations={} [{}]",
        report.iterations,
        weights.join(",")
    ));
    lines
}

/// Prints the probe models when started as a child of
/// [`weights_are_bit_identical_across_processes`]; does nothing in a normal test run.
#[test]
fn process_probe() {
    if std::env::var_os(PROBE_VARIABLE).is_none() {
        return;
    }
    println!();
    for line in probe_lines() {
        println!("{PROBE_MARKER}{line}");
    }
}

fn models_of_a_child_process() -> Vec<String> {
    let binary = std::env::current_exe().expect("the test binary has a path");
    let output = Command::new(binary)
        .args([
            "process_probe",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(PROBE_VARIABLE, "1")
        .output()
        .expect("the test binary starts a second time");
    assert!(output.status.success(), "child failed: {output:?}");
    String::from_utf8(output.stdout)
        .expect("probe output is text")
        .lines()
        .filter_map(|line| {
            line.find(PROBE_MARKER)
                .map(|at| line[at + PROBE_MARKER.len()..].trim_end().to_string())
        })
        .collect()
}

#[test]
fn weights_are_bit_identical_across_processes() {
    let own = probe_lines();
    assert_eq!(own.len(), 3);
    // The models carry content: many non-zero weights each.
    assert!(
        own.iter()
            .all(|line| line.matches(':').count() >= 100 || line.starts_with("dense"))
    );
    for run in ["first", "second"] {
        let child = models_of_a_child_process();
        assert_eq!(child.len(), own.len(), "{run} child process: line count");
        for (position, (theirs, ours)) in child.iter().zip(&own).enumerate() {
            assert!(
                theirs == ours,
                "{run} child process differs from this process in model {position}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------
// Scale: the size of one training fold of the CSIC baseline (task 1.12).
// ---------------------------------------------------------------------------------------

const SCALE_ROWS: usize = 20_000;

/// Synthetic request texts of about 300 distinct n-grams each. A bad text carries one of
/// a few attack fragments; one good text in fifty carries one too.
fn scale_texts() -> (Vec<String>, Vec<bool>) {
    const FRAGMENTS: [&str; 6] = [
        "' OR '1'='1",
        "<script>alert(1)</script>",
        "../../../etc/passwd",
        "; DROP TABLE users--",
        "%00%27%22",
        " UNION SELECT ",
    ];
    let mut sequence = Sequence(31);
    let mut texts = Vec::with_capacity(SCALE_ROWS);
    let mut labels = Vec::with_capacity(SCALE_ROWS);
    for row in 0..SCALE_ROWS {
        let bad = sequence.next_u32() % 5 < 2;
        let mut text = String::with_capacity(160);
        for _ in 0..110 {
            let byte = 0x20 + (sequence.next_u32() % 95) as u8;
            text.push(if byte == b'?' { '_' } else { char::from(byte) });
        }
        if bad || row % 50 == 7 {
            let fragment = FRAGMENTS[sequence.next_u32() as usize % FRAGMENTS.len()];
            let at = sequence.next_u32() as usize % text.len();
            text.insert_str(at, fragment);
        }
        texts.push(text);
        labels.push(bad);
    }
    (texts, labels)
}

/// Runs on request only, in release mode:
/// `cargo test --release -p l2d-baseline --test logreg -- --ignored --nocapture`
#[test]
#[ignore = "scale test; run in release mode on request"]
fn a_training_fold_of_csic_size_fits_in_reasonable_time_and_memory() {
    let hasher = CharHasher::default();
    let (texts, labels) = scale_texts();
    let rows: Vec<SparseRow> = texts.iter().map(|text| hasher.row(text)).collect();
    let stored: usize = rows.iter().map(SparseRow::len).sum();
    let per_row = stored as f64 / rows.len() as f64;
    assert_eq!(hasher.dimension(), 65_536);
    assert!(
        (250.0..400.0).contains(&per_row),
        "{per_row} entries per row"
    );

    let chosen = LogRegSettings::default();
    let started = Instant::now();
    let (model, report) =
        LogReg::fit_sparse(&rows, hasher.dimension(), &labels, &chosen).expect("the fit runs");
    let elapsed = started.elapsed();

    // What the fit holds besides the caller's rows: the row matrix (4 + 8 bytes per entry,
    // 8 per row) and the solver's vectors (parameters, gradients and a history of ten
    // pairs, here counted generously as forty vectors) plus two per-row vectors.
    let matrix_bytes = stored * 12 + rows.len() * 8;
    let solver_bytes = 40 * (model.dimension() + 1) * 8 + 2 * rows.len() * 8;
    let dense_bytes = rows.len() * model.dimension() * 8;
    println!(
        "L2D_LOGREG_SCALE rows={} dimension={} entries={stored} ({per_row:.1} per row) \
         fit_seconds={:.2} iterations={} termination={:?} objective={:.3} \
         gradient_norm={:.3e} fit_memory_megabytes={:.1} (dense matrix would be {:.1})",
        rows.len(),
        model.dimension(),
        elapsed.as_secs_f64(),
        report.iterations,
        report.termination,
        report.objective,
        report.gradient_norm,
        (matrix_bytes + solver_bytes) as f64 / 1e6,
        dense_bytes as f64 / 1e6,
    );

    assert!(model.weights().iter().all(|w| w.is_finite()) && model.bias().is_finite());
    assert!(report.iterations >= 1 && report.iterations <= u64::from(chosen.max_iter));
    let correct = rows
        .iter()
        .zip(&labels)
        .filter(|(row, label)| (model.probability_sparse(row).unwrap() > 0.5) == **label)
        .count();
    let accuracy = correct as f64 / rows.len() as f64;
    println!("L2D_LOGREG_SCALE training_accuracy={accuracy:.4}");
    assert!(accuracy > 0.9, "training accuracy {accuracy}");
    assert!(
        elapsed.as_secs() < 300,
        "the fit took {elapsed:?}; a fold of this size must stay usable"
    );
}
