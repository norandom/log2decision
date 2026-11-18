use l2d_core::{MemoryBudget, MemorySettings};
use l2d_decide::shift::{DEFAULT_PERMUTATIONS, ShiftError, shift_test, shift_test_with};

#[test]
fn unequal_groups_match_independent_ordered_pair_and_permutation_reference() {
    let a: Vec<_> = (0..30)
        .map(|i| vec![(i * i) as f32, (i * 7 % 13) as f32])
        .collect();
    let b: Vec<_> = (0..34)
        .map(|i| vec![(i * i + 120) as f32, (i * 5 % 19) as f32])
        .collect();
    // Independent ordered-pair sums, sorted Euclidean distances (median
    // 296.01689123644746), and 3 of 99 seeded label shuffles at least as extreme.
    let r = shift_test(&a, &b, 99, 23).unwrap();
    assert!((r.statistic - 0.0436587116760494).abs() < 1e-12, "{r:?}");
    assert_eq!(r.p_value, 0.04);
    assert_eq!((r.n_a, r.n_b, r.permutations, r.seed), (30, 34, 99, 23));
}

fn gaussian_sample(mut state: u32, mean: f64) -> Vec<Vec<f32>> {
    let mut uniform = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (f64::from(state) + 0.5) / 4294967296.0
    };
    (0..60)
        .map(|_| {
            let radius = (-2.0 * uniform().ln()).sqrt();
            let angle = std::f64::consts::TAU * uniform();
            vec![
                (radius * angle.cos() + mean) as f32,
                (radius * angle.sin()) as f32,
            ]
        })
        .collect()
}

#[test]
fn identical_and_shifted_gaussian_groups() {
    let a = gaussian_sample(17, 0.);
    let b = gaussian_sample(812, 3.);
    let same = shift_test(&a, &a, DEFAULT_PERMUTATIONS, 55).unwrap();
    assert!(same.statistic < 0. && same.p_value > 0.05, "{same:?}");
    let shifted = shift_test(&a, &b, DEFAULT_PERMUTATIONS, 55).unwrap();
    assert_eq!(DEFAULT_PERMUTATIONS, 199);
    assert_eq!(shifted.p_value, 0.005);
    assert_eq!(
        shifted,
        shift_test(&a, &b, DEFAULT_PERMUTATIONS, 55).unwrap()
    );
}

#[test]
fn zero_bandwidth_ties_and_zero_permutations() {
    let a = vec![vec![1.]; 30];
    let mut b = a.clone();
    let equal = shift_test(&a, &b, 73, 11).unwrap();
    assert_eq!((equal.statistic, equal.p_value), (0., 1.));
    b[..5].fill(vec![2.]);
    let sparse = shift_test(&a, &b, 0, 11).unwrap();
    // Zero-bandwidth Gaussian limit is equality. Five rows differ.
    assert!((sparse.statistic - 2. * 5. * 4. / (30. * 29.)).abs() < 1e-12);
    assert_eq!((sparse.p_value, sparse.permutations), (1., 0));
}

#[test]
fn small_groups_report_both_original_sizes() {
    let v = vec![vec![0.]; 30];
    for (n_a, n_b) in [(29, 30), (30, 29), (0, 0)] {
        assert!(matches!(shift_test(&v[..n_a],&v[..n_b],5,4),
            Err(ShiftError::GroupTooSmall{n_a:a,n_b:b,minimum:30}) if (a,b)==(n_a,n_b)));
    }
}

#[test]
fn malformed_inputs_refused_even_outside_subsample() {
    let a = vec![vec![1.]; 30];
    let mut b = vec![vec![1.]; 2100];
    for bad in [vec![], vec![1., 1.], vec![f32::NAN], vec![f32::INFINITY]] {
        b[2099] = bad;
        assert!(matches!(
            shift_test(&a, &b, 0, 3),
            Err(ShiftError::InvalidVector {
                group: "b",
                row: 2099
            })
        ));
    }
}

#[test]
fn whole_pairwise_workspace_refused_with_budget_figures() {
    let mut budget =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(1 << 30)).unwrap();
    budget.hold(budget.left_bytes() - 4096);
    let v = vec![vec![1.]; 30];
    let e = shift_test_with(&v, &v, 19, 0, &budget)
        .unwrap_err()
        .to_string();
    assert!(e.contains("estimated") && e.contains("left"), "{e}");
    assert_eq!(budget.left_bytes(), 4096);
}

#[test]
fn oversized_group_is_sampled_reproducibly_beyond_its_prefix() {
    let a: Vec<_> = (0..2400).map(|i| vec![i as f32]).collect();
    let b = vec![vec![600.]; 30];
    let r = shift_test(&b, &a, 0, 1).unwrap();
    assert_eq!((r.n_a, r.n_b), (30, 2000));
    assert_eq!(r, shift_test(&b, &a, 0, 1).unwrap());
    assert_ne!(r.statistic, shift_test(&b, &a, 0, 2).unwrap().statistic);
    assert_ne!(
        r.statistic,
        shift_test(&b, &a[..2000], 0, 1).unwrap().statistic
    );
}
