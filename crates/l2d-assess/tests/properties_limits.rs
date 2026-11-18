use l2d_assess::{
    VectorSet, assess_properties, assess_properties_with, project_to_csv, project_to_csv_with,
};
use l2d_core::{L2dError, MemoryBudget, MemorySettings, memory::PROCESS_BASELINE_BYTES};
use rand::{Rng, SeedableRng, rngs::SmallRng};

fn set(vectors: Vec<Vec<f32>>) -> VectorSet {
    VectorSet {
        vectors,
        labels: None,
        groups: None,
        name: String::new(),
    }
}

#[test]
fn isotropic_unit_vectors_have_unit_norms_and_near_zero_mean_cosine() {
    let mut rng = SmallRng::seed_from_u64(123);
    let data = set((0..256)
        .map(|_| {
            let row: Vec<f64> = (0..8).map(|_| rng.random_range(-1.0..1.0)).collect();
            let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            row.iter().map(|x| (x / norm) as f32).collect()
        })
        .collect());
    let r = assess_properties(&data, 77).unwrap();
    assert!(r.norm_quantiles.iter().all(|x| (x - 1.).abs() < 1e-6));
    assert!(r.anisotropy.abs() < 0.01);
    assert_eq!(r.cosine_pairs, 256 * 255 / 2);
}

#[test]
fn pair_sample_is_capped_while_norms_and_duplicates_use_every_original() {
    let data = set((0..5200)
        .map(|i| vec![if i < 24 { 1. } else { 0. }])
        .collect());
    let r = assess_properties(&data, 904).unwrap();
    assert_eq!(r.sample_size, 5000);
    assert!(r.cosine_pairs > 0 && r.cosine_pairs <= 24 * 23 / 2);
    assert_eq!(r.norm_quantiles, [0., 0., 0., 0., 1.]);
    assert_eq!(r.cosine_quantiles, [1.; 5]);
    assert_eq!(r.duplicate_rate, 5198. / 5200.);
}

#[test]
fn invalid_and_unrepresentable_properties_return_errors() {
    for rows in [
        vec![vec![0.], vec![1.]],
        vec![vec![f32::MAX; 2]; 2],
        vec![vec![1.; 65536]; 2],
        vec![vec![f32::NAN]; 2],
    ] {
        assert!(matches!(
            assess_properties(&set(rows), 0),
            Err(L2dError::InvalidInput { .. })
        ));
    }
}

#[test]
fn refusal_or_invalid_metadata_preserves_output_and_no_partial_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.csv");
    std::fs::write(&path, "previous").unwrap();
    let cfg = MemorySettings {
        memory_limit_bytes: Some(PROCESS_BASELINE_BYTES + 128),
        ..Default::default()
    };
    let budget = MemoryBudget::observe_with(&cfg, || Ok(1 << 30)).unwrap();
    let mut data = set((0..8).map(|i| vec![i as f32, 0.]).collect());
    assert!(matches!(
        assess_properties_with(&data, 0, &budget),
        Err(L2dError::Refused { .. })
    ));
    assert!(matches!(
        project_to_csv_with(&data, &path, 0, &budget),
        Err(L2dError::Refused { .. })
    ));
    data.groups = Some(vec![]);
    assert!(matches!(
        project_to_csv(&data, &path, 0),
        Err(L2dError::InvalidInput { .. })
    ));
    assert_eq!(std::fs::read(&path).unwrap(), b"previous");
    data.groups = None;
    assert!(matches!(
        project_to_csv(&data, dir.path(), 0),
        Err(L2dError::Runtime { .. })
    ));
    assert_eq!(dir.path().read_dir().unwrap().count(), 1);
}

#[test]
fn small_supported_projection_writes_blank_optional_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.csv");
    assert!(project_to_csv(&set(vec![vec![0.]; 3]), &path, 0).is_err());
    assert!(!path.exists());
    project_to_csv(&set(vec![vec![0.]; 4]), &path, 0).unwrap();
    let rows = csv::Reader::from_path(path)
        .unwrap()
        .records()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rows.len(), 4);
    for row in rows {
        assert_eq!((&row[2], &row[3]), ("", ""));
    }
}
