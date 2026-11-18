use l2d_assess::{VectorSet, assess_clusters, assess_clusters_with, separability};
use l2d_core::{Label, MemoryBudget, MemorySettings, memory::PROCESS_BASELINE_BYTES};

fn data(xs: &[f32]) -> VectorSet {
    VectorSet {
        vectors: xs.iter().map(|&x| vec![x]).collect(),
        labels: None,
        groups: None,
        name: String::new(),
    }
}

#[test]
fn euclidean_silhouette_matches_each_original_and_requested_k() {
    let set = data(&[0., 2., 10., 12.]);
    let r = assess_clusters(&set, 2, 17).unwrap();
    let (silhouette, overall, within): (f32, Option<f32>, Option<f32>) =
        (r.silhouette, r.separability_auc, r.within_group_auc);
    let expected = (79.0_f64 / 99.0) as f32;
    eprintln!(
        "two clusters: silhouette={} expected={expected}",
        r.silhouette
    );
    assert_eq!(silhouette, expected);
    assert_eq!((r.k, r.seed, r.sample_size), (2, 17, 4));
    assert_eq!((overall, within), (None, None));
    // Two singletons contribute zero to the three-cluster mean.
    let three = assess_clusters(&set, 3, 17).unwrap();
    assert_eq!(three.silhouette, (31.0_f64 / 80.0) as f32);
}

#[test]
fn keyed_auc_is_weighted_by_evaluated_labelled_rows() {
    let mut xs = vec![];
    let mut labels = vec![];
    let mut keys = vec![];
    for (key, n) in [
        ("signal", 20),
        ("tied", 40),
        ("small", 19),
        ("single-class", 24),
        ("one-bad", 26),
        ("unknown", 80),
    ] {
        for i in 0..n {
            let label = match key {
                "unknown" => Label::Unknown,
                "single-class" => Label::Good,
                "one-bad" if i > 0 => Label::Good,
                _ if i % 2 == 0 => Label::Bad,
                _ => Label::Good,
            };
            labels.push(label);
            keys.push(key.to_string());
            xs.push(if key == "signal" && label == Label::Bad {
                9.
            } else {
                1.
            });
        }
    }
    for _ in 0..21 {
        keys.push("signal".into());
        labels.push(Label::Unknown);
        xs.push(1.);
    }
    let mut set = data(&xs);
    set.labels = Some(labels);
    set.groups = Some(keys);
    let r = assess_clusters(&set, 2, 31).unwrap();
    eprintln!(
        "group-count={} weighted-AUC={:?} overall-AUC={:?}",
        r.evaluated_groups, r.within_group_auc, r.separability_auc
    );
    assert_eq!(r.evaluated_groups, 2);
    assert_eq!(r.within_group_auc, Some((2.0_f64 / 3.0) as f32));
    assert_eq!(
        r.separability_auc,
        Some(separability(&set, 31).unwrap().unwrap().auc as f32)
    );
    set.groups = None;
    assert_eq!(assess_clusters(&set, 2, 31).unwrap().within_group_auc, None);
}

#[test]
fn invalid_cluster_counts_or_too_few_distinct_points_refuse() {
    for k in [0, 1, 4, 5] {
        assert!(assess_clusters(&data(&[1., 2., 3., 4.]), k, 3).is_err());
    }
    assert!(assess_clusters(&data(&[1., 1., 1., 2.]), 3, 3).is_err());
    let mut set = data(&[1., 2., 3., 4.]);
    set.labels = Some(vec![Label::Bad]);
    assert!(assess_clusters(&set, 2, 3).is_err());
}

#[test]
fn capped_sample_is_seeded_and_reproducible() {
    let set = data(
        &(0..5100)
            .map(|i| {
                if i % 2 == 0 {
                    0.
                } else {
                    100. + (i % 13) as f32
                }
            })
            .collect::<Vec<_>>(),
    );
    let a = assess_clusters(&set, 2, 42).unwrap();
    assert_eq!(a.sample_size, 5000);
    assert_eq!(a, assess_clusters(&set, 2, 42).unwrap());
    assert_ne!(
        a.silhouette,
        assess_clusters(&set, 2, 8).unwrap().silhouette
    );
}

#[test]
fn low_budget_refuses_before_cluster_workspace() {
    let cfg = MemorySettings {
        memory_limit_bytes: Some(PROCESS_BASELINE_BYTES + 1024),
        ..Default::default()
    };
    let budget = MemoryBudget::observe_with(&cfg, || Ok(1 << 30)).unwrap();
    let error = assess_clusters_with(&data(&[0., 1., 2., 3.]), 2, 3, &budget).unwrap_err();
    assert!(error.to_string().contains("workspace"), "{error}");
}

#[test]
fn empty_group_denominator_and_sparse_labels_are_absent() {
    let mut set = data(&(0..24).map(|i| i as f32).collect::<Vec<_>>());
    let mut labels = vec![Label::Good; 24];
    labels[0] = Label::Bad;
    set.labels = Some(labels);
    set.groups = Some(vec!["one endpoint".into(); 24]);
    let r = assess_clusters(&set, 2, 1).unwrap();
    assert_eq!(
        (r.evaluated_groups, r.within_group_auc, r.separability_auc),
        (0, None, None)
    );
}
