//! Explicit replay over the frozen actual-NPU development panel, no encoder run.
use l2d_core::{
    EncoderIdentity, Label, LogRecord, LogsFrame, MemoryBudget, MemorySettings, ProjectPaths,
};
use l2d_decide::outlier::{OutlierQuery, rank_outliers};
use l2d_store::{LogFlags, Store, VectorOrigin, rows_frame};
use serde::Deserialize;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::PathBuf,
};
#[derive(Deserialize)]
struct Row {
    i: u64,
    text: String,
    group: String,
    proxy: String,
    vector: Vec<f32>,
}
#[test]
#[ignore = "requires frozen task 2.11 CSIC actual-NPU panel; L2D_OUTLIER_PANEL"]
fn frozen_csic_k_comparison() {
    let root = PathBuf::from(std::env::var_os("L2D_OUTLIER_PANEL").expect("L2D_OUTLIER_PANEL"));
    let file = root.join("panel.jsonl");
    let bytes = fs::metadata(&file).unwrap().len();
    assert!(bytes < 32 * 1024 * 1024);
    let mut budget = MemoryBudget::observe(&MemorySettings {
        memory_limit_bytes: Some(1024 * 1024 * 1024),
        max_batch_rows: Some(64),
        ..Default::default()
    })
    .unwrap();
    let held = bytes * 32 + 32 * 1024 * 1024;
    budget
        .require_whole(
            held,
            "bounded scientific panel JSON, rows, vectors, and serialization",
        )
        .unwrap();
    budget.hold(held);
    let plan: serde_json::Value =
        serde_json::from_reader(fs::File::open(root.join("plan.json")).unwrap()).unwrap();
    let identity = EncoderIdentity {
        name: plan["origin"]["name"].as_str().unwrap().into(),
        revision: plan["origin"]["revision"].as_str().unwrap().into(),
        dimension: 128,
    };
    let origin = VectorOrigin::made_under(
        &identity,
        plan["origin"]["runtime_release"].as_str().unwrap(),
    );
    let rows: Vec<Row> = BufReader::new(fs::File::open(file).unwrap())
        .lines()
        .map(|x| serde_json::from_str(&x.unwrap()).unwrap())
        .collect();
    assert_eq!(rows.len(), 2144);
    assert!(rows.iter().all(|r| r.vector.len() == 128));
    // Replay creates a fresh local store; original datasets/models remain untouched.
    let paths = ProjectPaths::new(root.join(format!("store-{}", std::process::id())));
    let mut store = Store::create(&paths, "reference", origin).unwrap();
    for batch in rows[..2048].chunks(64) {
        assert!(batch.iter().all(|r| r.proxy == "reference"));
        let logs = LogsFrame::from_records(
            &batch
                .iter()
                .map(|r| LogRecord {
                    text: r.text.clone(),
                    label: Label::Unknown,
                    group: Some(r.group.clone()),
                    first_position: r.i,
                    occurrences: 1,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let vectors: Vec<_> = batch
            .iter()
            .flat_map(|r| r.vector.iter().copied())
            .collect();
        let frame = rows_frame(
            &logs,
            "csic-reference",
            &vectors,
            128,
            &vec![LogFlags::default(); batch.len()],
        )
        .unwrap();
        store.add_with(&frame, origin, &mut budget).unwrap();
    }
    let queries: Vec<_> = rows[2048..]
        .iter()
        .map(|r| OutlierQuery {
            source: "csic-query",
            text: &r.text,
            vector: &r.vector,
        })
        .collect();
    for k in [1, 5, 11, 21] {
        let out = rank_outliers(&store, origin, &queries, Some(k), &mut budget).unwrap();
        assert_eq!(out.rows.len(), 96);
        assert!(
            out.rows
                .iter()
                .all(|r| !r.self_excluded && r.evidence.unknown.count == k)
        );
        serde_json::to_writer_pretty(
            fs::File::create(root.join(format!("k{k}.json"))).unwrap(),
            &out,
        )
        .unwrap();
        println!(
            "k={k} queries={} reference={} counted_peak={}",
            out.rows.len(),
            store.len().unwrap(),
            budget.usage().report().counted_peak_bytes
        );
        drop(out);
        budget.hold(held);
    }
}
