use l2d_assess::{VectorSet, neighbour_overlap_with, separability_with};
use l2d_core::{Label, MemoryBudget, MemorySettings};

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(1 << 32)).unwrap()
}

fn rows(vectors: Vec<Vec<f32>>) -> VectorSet {
    VectorSet {
        vectors,
        labels: None,
        groups: None,
        name: "fixture".into(),
    }
}

fn overlap(a: &VectorSet, b: &VectorSet, k: usize) -> f32 {
    neighbour_overlap_with(a, b, k, 91, &budget())
        .unwrap()
        .overlap
}

#[test]
fn original_alignment_and_negative_control() {
    // Four disjoint nearest pairs AB, CD, EF, GH. The second representation
    // retains AB and CD but pairs EG and FH. Four of eight neighbour IDs match.
    let a = rows(
        [0., 1., 10., 11., 30., 31., 80., 81.]
            .map(|v| vec![v])
            .to_vec(),
    );
    let b = rows(
        [0., 1., 10., 11., 30., 80., 31., 81.]
            .map(|v| vec![v, 0.])
            .to_vec(),
    );
    assert_eq!(overlap(&a, &a, 1), 1.);
    assert_eq!(overlap(&a, &b, 1), 0.5);
    let reorder = |s: &VectorSet| {
        rows(
            [4, 0, 7, 1, 6, 2, 3, 5]
                .map(|i| s.vectors[i].clone())
                .to_vec(),
        )
    };
    assert_eq!(overlap(&reorder(&a), &reorder(&b), 1), 0.5);

    // Every original's true neighbour is the other member of its planted pair.
    // A seeded permutation changes vector assignments at fixed original IDs.
    let n = 256;
    let a = rows(
        (0..n)
            .map(|i| vec![(10 * (i / 2) + i % 2) as f32])
            .collect(),
    );
    let mut permutation: Vec<_> = (0..n).collect();
    let mut state = 7531u32;
    for i in (1..n).rev() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        permutation.swap(i, ((u64::from(state) * (i + 1) as u64) >> 32) as usize);
    }
    let shuffled = rows(permutation.iter().map(|&i| a.vectors[i].clone()).collect());
    let expected = (0..n)
        .filter(|&i| permutation[i] ^ 1 == permutation[i ^ 1])
        .count() as f32
        / n as f32;
    assert_eq!(overlap(&a, &shuffled, 1), expected);
    // Random k=1 neighbourhoods share about 1/(n-1) of their members.
    assert!(expected < 0.04);
}

#[test]
fn k_is_intersection_fraction_with_self_excluded_and_ties_by_original_index() {
    let a = rows(vec![vec![0.], vec![0.], vec![1.], vec![3.]]);
    let b = rows(vec![vec![0.], vec![3.], vec![1.], vec![0.]]);
    // Independent k=2 lists: [12,02,01,20] and [32,20,03,02].
    assert_eq!(overlap(&a, &b, 2), 0.75);
}

#[test]
fn held_out_separability_ignores_unknown_and_cannot_memorize_identity() {
    let n = 40;
    let labels: Vec<_> = (0..n)
        .map(|i| if i % 2 == 0 { Label::Bad } else { Label::Good })
        .collect();
    let mut set = rows(
        (0..n)
            .map(|i| vec![if i % 2 == 0 { 3. } else { -3. }])
            .collect(),
    );
    set.vectors.push(vec![999.]);
    set.labels = Some(labels.iter().copied().chain([Label::Unknown]).collect());
    set.groups = Some(vec!["shared".into(); n + 1]);
    let measured = separability_with(&set, 71, &budget()).unwrap().unwrap();
    assert_eq!(
        (measured.auc, measured.labelled, measured.folds),
        (1., 40, 5)
    );
    // With one-hot identities every held-out log has all training coordinates
    // zero. Each held-out pair ties even though an in-sample fit separates them.
    let mut ids = rows(
        (0..n)
            .map(|i| {
                let mut x = vec![0.; n];
                x[i] = 1.;
                x
            })
            .collect(),
    );
    ids.labels = Some(labels);
    assert_eq!(
        separability_with(&ids, 2, &budget()).unwrap().unwrap().auc,
        0.5
    );
}

#[test]
fn measures_refuse_invalid_metadata_shape_sizes_or_budget() {
    let mut a = rows(vec![vec![1.], vec![2.], vec![3.]]);
    assert!(separability_with(&a, 1, &budget()).unwrap().is_none());
    a.labels = Some(vec![Label::Good, Label::Good, Label::Bad]);
    assert!(separability_with(&a, 1, &budget()).unwrap().is_none());
    assert!(neighbour_overlap_with(&a, &a, 3, 1, &budget()).is_err());
    assert!(neighbour_overlap_with(&a, &a, 0, 1, &budget()).is_err());
    let mut tiny = budget();
    tiny.hold(tiny.left_bytes() - 1);
    assert!(
        neighbour_overlap_with(&a, &a, 1, 1, &tiny)
            .unwrap_err()
            .to_string()
            .contains("estimated")
    );
    a.groups = Some(vec![]);
    assert!(separability_with(&a, 1, &budget()).is_err());
    a.groups = None;
    a.vectors[0][0] = f32::NAN;
    assert!(neighbour_overlap_with(&a, &a, 1, 1, &budget()).is_err());
}

#[test]
fn public_entry_points_report_sizes_and_reject_misaligned_shapes() {
    use l2d_assess::{neighbour_overlap, separability};
    let mut a = rows(vec![vec![0.], vec![1.], vec![10.], vec![11.]]);
    let report = neighbour_overlap(&a, &a, 2).unwrap();
    assert_eq!(
        (report.sample_size, report.k, report.seed, report.overlap),
        (4, 2, 0, 1.)
    );
    assert!(separability(&a, 4).unwrap().is_none());
    let b = rows(vec![vec![0.], vec![1.]]);
    assert!(neighbour_overlap(&a, &b, 1).is_err());
    a.labels = Some(vec![Label::Bad]);
    assert!(a.validate().is_err());
    a.labels = None;
    a.vectors[1].clear();
    assert!(a.validate().is_err());
    a.vectors[1] = vec![1., 2.];
    assert!(a.validate().is_err());
}

#[test]
fn dense_fit_refusal_accounts_for_the_existing_inputs_and_solver() {
    let mut a = rows(vec![vec![0.; 2048]; 20]);
    a.labels = Some([vec![Label::Good; 10], vec![Label::Bad; 10]].concat());
    let mut limit = budget();
    limit.hold(limit.left_bytes() - 300_000);
    let reason = separability_with(&a, 1, &limit).unwrap_err().to_string();
    assert!(reason.contains("estimated") && reason.contains("left"));
}
