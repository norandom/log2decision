use l2d_core::{
    EncoderIdentity, Label, LogRecord, LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};
use l2d_decide::outlier::{OutlierQuery, rank_outliers};
use l2d_store::{LogFlags, Store, VectorOrigin, rows_frame};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 << 30)).unwrap()
}
fn fixture() -> (Store, EncoderIdentity) {
    let paths = ProjectPaths::new(std::env::temp_dir().join(format!(
        "l2d-outlier-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )));
    let identity = EncoderIdentity {
        name: "outlier-fixture".into(),
        revision: "a".repeat(64),
        dimension: 2,
    };
    let origin = VectorOrigin::without_release(&identity);
    let mut store = Store::create(&paths, "test", origin).unwrap();
    let logs = LogsFrame::from_records(
        &(0..7)
            .map(|i| LogRecord {
                text: format!("log {i}"),
                label: if i < 2 {
                    Label::Bad
                } else if i < 5 {
                    Label::Good
                } else {
                    Label::Unknown
                },
                group: None,
                first_position: i,
                occurrences: if i == 0 { 900 } else { 1 },
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let vectors = [1., 1., 2., 1., 3., 1., 4., 1., 5., 1., 6., 1., 100., 1.];
    store
        .add_with(
            &rows_frame(&logs, "fixture", &vectors, 2, &[LogFlags::default(); 7]).unwrap(),
            origin,
            &mut budget(),
        )
        .unwrap();
    (store, identity)
}
#[test]
fn planted_far_stored_log_ranks_first_at_default_fifth_other_distance() {
    let (store, identity) = fixture();
    let queries = [
        OutlierQuery {
            source: "fixture",
            text: "log 0",
            vector: &[1., 1.],
        },
        OutlierQuery {
            source: "fixture",
            text: "log 6",
            vector: &[100., 1.],
        },
    ];
    let ranking = rank_outliers(
        &store,
        VectorOrigin::without_release(&identity),
        &queries,
        None,
        &mut budget(),
    )
    .unwrap();
    assert_eq!(ranking.k, 5);
    assert_eq!(ranking.distance_metric, "squared_l2_distance");
    assert_eq!(
        ranking
            .rows
            .iter()
            .map(|r| (r.query, r.distance, r.self_excluded))
            .collect::<Vec<_>>(),
        [(1, 9604., true), (0, 25., true)]
    );
}

#[test]
fn known_neighbourhood_counts_rows_not_occurrences_and_means_squared_distances() {
    let (store, id) = fixture();
    let queries = [OutlierQuery {
        source: "external",
        text: "new",
        vector: &[1., 1.],
    }];
    let out = rank_outliers(
        &store,
        VectorOrigin::without_release(&id),
        &queries,
        Some(6),
        &mut budget(),
    )
    .unwrap();
    let row = &out.rows[0];
    assert!(!row.self_excluded);
    assert_eq!(row.distance, 25.);
    assert_eq!(
        (row.evidence.bad.count, row.evidence.bad.mean_distance),
        (2, Some(0.5))
    );
    assert_eq!(
        (row.evidence.good.count, row.evidence.good.mean_distance),
        (3, Some(29. / 3.))
    );
    assert_eq!(
        (
            row.evidence.unknown.count,
            row.evidence.unknown.mean_distance
        ),
        (1, Some(25.))
    );
}

#[test]
fn refuses_small_store_including_self_and_mismatched_stored_vector() {
    let (store, id) = fixture();
    let origin = VectorOrigin::without_release(&id);
    let mut b = budget();
    b.hold(123);
    let q = [OutlierQuery {
        source: "fixture",
        text: "log 0",
        vector: &[1., 1.],
    }];
    let error = rank_outliers(&store, origin, &q, Some(7), &mut b)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("7 stored") && error.contains("8 needed"),
        "{error}"
    );
    assert_eq!(b.held_bytes(), 123);
    let q = [OutlierQuery {
        source: "fixture",
        text: "log 0",
        vector: &[2., 1.],
    }];
    assert!(
        rank_outliers(&store, origin, &q, Some(2), &mut b)
            .unwrap_err()
            .to_string()
            .contains("stored vector")
    );
    assert_eq!(b.held_bytes(), 123);
}

#[test]
fn invalid_queries_origin_and_budget_fail_without_retained_reservation() {
    let (store, id) = fixture();
    for vector in [&[0., 0.][..], &[f32::NAN, 1.], &[f32::INFINITY, 1.], &[1.]] {
        let mut b = budget();
        let q = [OutlierQuery {
            source: "x",
            text: "y",
            vector,
        }];
        assert!(
            rank_outliers(&store, VectorOrigin::without_release(&id), &q, None, &mut b).is_err()
        );
        assert_eq!(b.held_bytes(), 0);
    }
    let q = [OutlierQuery {
        source: "x",
        text: "y",
        vector: &[1., 1.],
    }];
    let mut wrong = id.clone();
    wrong.revision = "other".into();
    assert!(
        rank_outliers(
            &store,
            VectorOrigin::without_release(&wrong),
            &q,
            None,
            &mut budget()
        )
        .unwrap_err()
        .to_string()
        .contains("other")
    );
    assert!(
        rank_outliers(
            &store,
            VectorOrigin::made_under(&id, "different-release"),
            &q,
            None,
            &mut budget()
        )
        .is_err()
    );
    assert!(
        rank_outliers(
            &store,
            VectorOrigin::without_release(&id),
            &q,
            Some(0),
            &mut budget()
        )
        .is_err()
    );
    let mut b = budget();
    let held = b.usable_bytes() - 16;
    b.hold(held);
    let error = rank_outliers(&store, VectorOrigin::without_release(&id), &q, None, &mut b)
        .unwrap_err()
        .to_string();
    assert!(error.contains("memory"), "{error}");
    assert_eq!(b.held_bytes(), held);
}

#[test]
fn batch_size_does_not_change_distance_or_evidence_and_output_stays_reserved() {
    let (store, id) = fixture();
    let q = [OutlierQuery {
        source: "fixture",
        text: "log 0",
        vector: &[1., 1.],
    }];
    let mut serializations = vec![];
    for batch in [1, 2, 100] {
        let mut b = MemoryBudget::observe_with(
            &MemorySettings {
                max_batch_rows: Some(batch),
                ..Default::default()
            },
            || Ok(8 << 30),
        )
        .unwrap();
        b.hold(123);
        let result =
            rank_outliers(&store, VectorOrigin::without_release(&id), &q, None, &mut b).unwrap();
        assert!(b.held_bytes() > 123);
        assert_eq!(b.held_bytes(), 123 + result.retained_bytes);
        serializations.push(serde_json::to_string(&result.rows).unwrap());
    }
    assert!(serializations.windows(2).all(|x| x[0] == x[1]));
}

fn add_one(store: &mut Store, source: &str, text: &str, vector: &[f32], flags: LogFlags) {
    let identity = store.identity().clone();
    let logs = LogsFrame::from_records(&[LogRecord {
        text: text.into(),
        label: Label::Unknown,
        group: None,
        first_position: 1,
        occurrences: 1,
    }])
    .unwrap();
    store
        .add_with(
            &rows_frame(&logs, source, vector, 2, &[flags]).unwrap(),
            VectorOrigin::without_release(&identity),
            &mut budget(),
        )
        .unwrap();
}

#[test]
fn invalid_stored_vectors_are_refused_without_changing_general_store_search() {
    for (vector, flags) in [
        ([0., 0.], LogFlags::default()),
        (
            [1., 1.],
            LogFlags {
                truncated: true,
                ..Default::default()
            },
        ),
        (
            [1., 1.],
            LogFlags {
                text_longer_than_input: true,
                ..Default::default()
            },
        ),
    ] {
        let (mut store, id) = fixture();
        add_one(&mut store, "x", "unsupported", &vector, flags);
        let q = [OutlierQuery {
            source: "x",
            text: "new",
            vector: &[1., 1.],
        }];
        let mut b = budget();
        assert!(
            rank_outliers(&store, VectorOrigin::without_release(&id), &q, None, &mut b)
                .unwrap_err()
                .to_string()
                .contains("unsupported")
        );
        assert_eq!(b.held_bytes(), 0);
        assert!(
            store
                .nearest(
                    &[1., 1.],
                    VectorOrigin::without_release(&id),
                    5,
                    &Default::default()
                )
                .is_ok()
        );
    }
}

#[test]
fn equal_vector_other_record_is_not_self_and_empty_label_mean_is_absent() {
    let (mut store, id) = fixture();
    add_one(
        &mut store,
        "other-source",
        "same-vector",
        &[1., 1.],
        LogFlags::default(),
    );
    let q = [OutlierQuery {
        source: "fixture",
        text: "log 0",
        vector: &[1., 1.],
    }];
    let out = rank_outliers(
        &store,
        VectorOrigin::without_release(&id),
        &q,
        Some(1),
        &mut budget(),
    )
    .unwrap();
    assert_eq!(out.rows[0].distance, 0.);
    assert_eq!(out.rows[0].evidence.unknown.count, 1);
    assert_eq!(out.rows[0].evidence.bad.mean_distance, None);
    assert_eq!(out.rows[0].evidence.good.mean_distance, None);
}

#[test]
fn another_connection_write_during_scan_invalidates_entire_ranking() {
    use std::sync::{Arc, atomic::AtomicBool};
    let (store, id) = fixture();
    let path = store.path().parent().unwrap().parent().unwrap().to_owned();
    rusqlite::Connection::open(store.path())
        .unwrap()
        .execute_batch("PRAGMA journal_mode=WAL")
        .unwrap();
    let armed = Arc::new(AtomicBool::new(false));
    let signal = armed.clone();
    let mut b = MemoryBudget::observe_with(
        &MemorySettings {
            max_batch_rows: Some(1),
            ..Default::default()
        },
        move || {
            if signal.swap(false, Ordering::SeqCst) {
                let mut writer = Store::open(&ProjectPaths::new(&path), "test").unwrap();
                add_one(
                    &mut writer,
                    "concurrent",
                    "new record",
                    &[7., 1.],
                    LogFlags::default(),
                );
            }
            Ok(8 << 30)
        },
    )
    .unwrap();
    let generation = store.read_generation().unwrap();
    armed.store(true, Ordering::SeqCst);
    let q = [OutlierQuery {
        source: "fixture",
        text: "log 0",
        vector: &[1., 1.],
    }];
    assert!(
        rank_outliers(&store, VectorOrigin::without_release(&id), &q, None, &mut b)
            .unwrap_err()
            .to_string()
            .contains("store changed")
    );
    assert_ne!(generation, store.read_generation().unwrap());
    assert_eq!(b.held_bytes(), 0);
}
