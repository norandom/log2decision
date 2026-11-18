use l2d_assess::{
    VectorSet, assess_clusters, assess_loss, assess_properties, compare, compare_with,
    neighbour_overlap_with,
};
use l2d_core::{L2dError, Label, MemoryBudget, MemorySettings, memory::PROCESS_BASELINE_BYTES};

fn fixture(width: usize, name: &str) -> VectorSet {
    let n = 24;
    VectorSet {
        vectors: (0..n)
            .map(|i| {
                vec![
                    1. + i as f32,
                    (i % 2) as f32,
                    100. * (i % 3) as f32,
                    (i % 5) as f32,
                ][..width]
                    .to_vec()
            })
            .collect(),
        labels: Some(
            (0..n)
                .map(|i| if i % 2 == 0 { Label::Bad } else { Label::Good })
                .collect(),
        ),
        groups: Some(vec!["one group".to_owned(); n]),
        name: name.to_owned(),
    }
}

fn budget(available: u64) -> MemoryBudget {
    MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(PROCESS_BASELINE_BYTES + available),
            ..Default::default()
        },
        || Ok(1 << 30),
    )
    .unwrap()
}

#[test]
fn shortened_originals_show_both_columns_and_reduced_overlap() {
    let a = fixture(4, "full");
    let b = fixture(2, "prefix");
    let seed = 37;
    let r = compare(&a, &b, seed).unwrap();
    for (set, col) in [(&a, &r.left), (&b, &r.right)] {
        assert_eq!(col.name, set.name);
        assert_eq!(col.originals, 24);
        assert_eq!(col.properties, assess_properties(set, seed).unwrap());
        assert_eq!(col.clusters, assess_clusters(set, 2, seed).unwrap());
        let loss = assess_loss(set, &[1], seed).unwrap();
        assert_eq!(col.loss.explained_variance, loss.explained_variance);
        assert_eq!(col.loss.intrinsic_dimension, loss.intrinsic_dimension);
        assert_eq!(col.loss.separability, loss.separability);
        assert_eq!(col.loss.shortened[0].dimension, 1);
        assert_eq!(
            col.loss.shortened[0].neighbour_overlap,
            loss.shortened[0].neighbour_overlap
        );
        assert_eq!(col.loss.sample_size, 24);
        assert_eq!(col.loss.seed, seed);
    }
    assert_eq!(
        r.neighbour_overlap,
        neighbour_overlap_with(&a, &b, 10, seed, &budget(1 << 20)).unwrap()
    );
    assert!(r.neighbour_overlap.overlap < 0.8);
    let table = r.to_string();
    assert!(table.contains("Measurement\tfull\tprefix"));
    assert!(table.contains("Dimension\t4\t2"));
    assert!(table.contains(&format!(
        "Silhouette\t{}\t{}",
        r.left.clusters.silhouette, r.right.clusters.silhouette
    )));
    assert!(table.contains(&format!(
        "Explained variance\t{:?}\t{:?}",
        r.left.loss.explained_variance, r.right.loss.explained_variance
    )));
    assert!(table.contains(&format!(
        "Neighbour overlap vs first\t1\t{}",
        r.neighbour_overlap.overlap
    )));
    println!("{table}");
}

#[test]
fn self_comparison_and_single_coordinate_need_no_prefix() {
    let mut a = fixture(1, "single");
    a.groups = None;
    a.labels = None;
    let r = compare(&a, &a, 91).unwrap();
    assert_eq!(r.neighbour_overlap.overlap, 1.);
    assert!(r.left.loss.shortened.is_empty());
    assert!(r.right.clusters.separability_auc.is_none());
    assert!(r.to_string().contains("Within-group AUC\tN/A\tN/A"));
}

#[test]
fn unequal_counts_or_conflicting_metadata_fail() {
    let a = fixture(4, "full");
    for kind in 0..4 {
        let mut b = fixture(2, "prefix");
        match kind {
            0 => {
                b.vectors.pop();
                b.labels.as_mut().unwrap().pop();
                b.groups.as_mut().unwrap().pop();
            }
            1 => b.labels.as_mut().unwrap().swap(0, 1),
            2 => b.groups.as_mut().unwrap()[1] = "different".into(),
            _ => b.labels = None,
        }
        assert!(matches!(
            compare(&a, &b, 1),
            Err(L2dError::InvalidInput { .. })
        ));
    }
}

#[test]
fn errors_and_success_restore_preexisting_held_bytes() {
    let a = fixture(4, "full");
    let mut b = fixture(2, "prefix");
    let mut ample = budget(1 << 20);
    ample.hold(457);
    compare_with(&a, &b, 0, &mut ample).unwrap();
    assert_eq!(ample.held_bytes(), 457);
    // Spare capacity is live even while only the other representation is assessed.
    b.name.reserve(1 << 16);
    let mut constrained = budget(1 << 15);
    constrained.hold(457);
    assert!(matches!(
        compare_with(&a, &b, 0, &mut constrained),
        Err(L2dError::Refused { .. })
    ));
    assert_eq!(constrained.held_bytes(), 457);
    for row in &mut b.vectors {
        row.fill(0.);
    }
    assert!(matches!(
        compare_with(&a, &b, 0, &mut ample),
        Err(L2dError::InvalidInput { .. })
    ));
    assert_eq!(ample.held_bytes(), 457);
}

#[test]
fn small_neighbourhoods_can_distinguish_encoders_and_names_stay_in_two_columns() {
    let mut a = fixture(4, "tab\tname");
    let mut b = fixture(2, "line\nbreak");
    for set in [&mut a, &mut b] {
        set.vectors.truncate(4);
        set.labels = None;
        set.groups = None;
    }
    let r = compare(&a, &b, 1).unwrap();
    assert!(r.neighbour_overlap.overlap < 1.);
    let table = r.to_string();
    assert!(table.starts_with("Measurement\ttab\\tname\tline\\nbreak\n"));
    assert!(table.lines().all(|row| row.split('\t').count() == 3));
}

#[test]
fn second_properties_pass_charges_both_retained_loss_reports() {
    // With sufficiently many rows, the properties pair buffer is the peak.
    // Reusing the same input isolates the two retained outputs from input costs.
    let set = VectorSet {
        vectors: (1..=180).map(|i| vec![i as f32, (i % 19) as f32]).collect(),
        labels: None,
        groups: None,
        name: "shared".into(),
    };
    let mut ample = budget(1 << 20);
    let loss = assess_loss(&set, &[1], 9).unwrap();
    let retained = loss.explained_variance.capacity() * size_of::<f32>()
        + loss.shortened.capacity() * size_of::<l2d_assess::ShortenedResult>();
    l2d_assess::assess_properties_with(&set, 9, &ample).unwrap();
    let properties_peak = ample.usage().report().counted_peak_bytes - PROCESS_BASELINE_BYTES;
    let names_and_report = 2 * set.name.len() + size_of::<l2d_assess::ComparisonReport>();
    let required = properties_peak + (names_and_report + 2 * retained) as u64;
    let mut tight = budget(required - 1);
    assert!(
        matches!(compare_with(&set, &set, 9, &mut tight), Err(L2dError::Refused { reason }) if reason.contains("properties"))
    );
    assert_eq!(tight.held_bytes(), 0);
    compare_with(&set, &set, 9, &mut ample).unwrap();
    assert_eq!(
        ample.usage().report().counted_peak_bytes - PROCESS_BASELINE_BYTES,
        required
    );
}
