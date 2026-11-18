use l2d_assess::{VectorSet, assess_loss, assess_loss_with};
use l2d_core::{L2dError, Label, MemoryBudget, MemorySettings};
use rand::{Rng, SeedableRng, rngs::SmallRng};

fn set(vectors: Vec<Vec<f32>>) -> VectorSet {
    VectorSet {
        vectors,
        labels: None,
        groups: None,
        name: "synthetic".into(),
    }
}

#[test]
fn translated_rotated_plane_has_known_pca_spectrum() {
    // Orthogonal loadings (1,1,0) and (0,0,1) carry variance 8 and 1.
    let mut rows = vec![];
    for a in [-2., 2.] {
        for b in [-1., 1.] {
            rows.push(vec![100. + a, -20. + a, b]);
        }
    }
    let r = assess_loss(&set(rows), &[], 3).unwrap();
    assert_eq!(r.explained_variance.len(), 3);
    for (v, e) in r.explained_variance.iter().zip([8. / 9., 1. / 9., 0.]) {
        assert!((v - e).abs() < 1e-6, "{r:?}");
    }
}

#[test]
fn twonn_matches_distance_ratio_likelihood() {
    // At 0,1,3,7 the ratios r2/r1 are 3,2,3/2,6/4.
    let r = assess_loss(&set([0., 1., 3., 7.].map(|x| vec![x]).to_vec()), &[], 12).unwrap();
    assert!((r.intrinsic_dimension.unwrap() - 3. / 13.5_f64.ln()).abs() < 1e-12);
    assert_eq!((r.sample_size, r.seed), (4, 12));
}

#[test]
fn iid_plane_dimension_and_coordinate_prefix_effects() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut rows = vec![];
    let mut labels = vec![];
    for _ in 0..650 {
        let x = rng.random::<f32>();
        let y = rng.random::<f32>();
        rows.push(vec![x, y, 0.]);
        labels.push(if y > 0.5 { Label::Bad } else { Label::Good });
    }
    let mut plane = set(rows);
    plane.labels = Some(labels);
    let r = assess_loss(&plane, &[2, 1], 42).unwrap();
    println!(
        "plane dimension {:?}, baseline {:?}, shortened {:?}",
        r.intrinsic_dimension, r.separability, r.shortened
    );
    assert!((1.7..2.3).contains(&r.intrinsic_dimension.unwrap()));
    let a = &r.shortened[0];
    let b = &r.shortened[1];
    assert_eq!((a.dimension, b.dimension), (2, 1));
    assert_eq!(a.neighbour_overlap, 1.);
    assert!(a.separability_auc.unwrap() > 0.98);
    assert_eq!(a.separability_auc.unwrap(), r.separability.unwrap().auc);
    assert!(b.neighbour_overlap < 0.3);
    assert!((0.35..0.65).contains(&b.separability_auc.unwrap()));
}

#[test]
fn degeneracies_invalid_lengths_and_caller_memory() {
    let s = set(vec![vec![1.; 3]; 4]);
    let r = assess_loss(&s, &[1], 1).unwrap();
    assert!(r.explained_variance.is_empty());
    assert!(r.intrinsic_dimension.is_none());
    assert!(r.shortened[0].separability_auc.is_none());
    for d in [0, 3, 4] {
        assert!(matches!(
            assess_loss(&s, &[d], 1),
            Err(L2dError::InvalidInput { .. })
        ));
    }
    let cfg = MemorySettings {
        memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 128),
        ..Default::default()
    };
    let mut budget = MemoryBudget::observe_with(&cfg, || Ok(1 << 30)).unwrap();
    budget.hold(17);
    assert!(matches!(
        assess_loss_with(&s, &[1], 1, &mut budget),
        Err(L2dError::Refused { .. })
    ));
    assert_eq!(budget.held_bytes(), 17);
}

#[test]
fn curved_line_has_two_pca_components_but_one_local_dimension() {
    let mut rng = SmallRng::seed_from_u64(19);
    let circle = set((0..800)
        .map(|_| {
            let t = rng.random::<f64>() * std::f64::consts::TAU;
            vec![t.cos() as f32, t.sin() as f32]
        })
        .collect());
    let r = assess_loss(&circle, &[], 19).unwrap();
    println!(
        "sampled circle PCA {:?}, TwoNN {:?}",
        r.explained_variance, r.intrinsic_dimension
    );
    assert!(r.explained_variance.iter().all(|v| (0.4..0.6).contains(v)));
    assert!((0.8..1.2).contains(&r.intrinsic_dimension.unwrap()));
}

#[test]
fn observation_space_pca_ties_and_pairwise_sample_cap() {
    let wide = set(vec![vec![1.; 9], vec![3.; 9]]);
    let r = assess_loss(&wide, &[1], 2).unwrap();
    assert_eq!(r.explained_variance.len(), 2);
    assert!((r.explained_variance[0] - 1.).abs() < 1e-6 && r.explained_variance[1] < 1e-6);
    assert!(r.intrinsic_dimension.is_none());
    let lattice = set(vec![vec![0.], vec![1.], vec![2.]]);
    assert!(
        assess_loss(&lattice, &[], 2)
            .unwrap()
            .intrinsic_dimension
            .is_none()
    );
    let duplicates = set(vec![vec![0.]; 6000]);
    let r = assess_loss(&duplicates, &[], 2).unwrap();
    assert_eq!(r.sample_size, 5000);
}

#[test]
fn pca_succeeds_before_dense_fit_refuses_and_restores_accounting() {
    let mut data = set((0..200).map(|i| vec![i as f32; 20]).collect());
    data.labels = Some(
        (0..200)
            .map(|i| if i % 2 == 0 { Label::Good } else { Label::Bad })
            .collect(),
    );
    let cfg = MemorySettings {
        memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 65_000),
        ..Default::default()
    };
    let mut budget = MemoryBudget::observe_with(&cfg, || Ok(1 << 30)).unwrap();
    budget.hold(257);
    let error = assess_loss_with(&data, &[1], 1, &mut budget).unwrap_err();
    assert!(error.to_string().contains("separability"), "{error}");
    assert_eq!(budget.held_bytes(), 257);
}
