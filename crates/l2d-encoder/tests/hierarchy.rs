use l2d_core::{MemoryBudget, MemorySettings};
use l2d_encoder::hierarchy::{HierarchicalSession, SESSION_BYTES, verify_profile};
#[test]
fn a_legacy_or_incomplete_profile_refuses_before_opening_the_runtime() {
    let b = MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
        .unwrap();
    let p = std::env::temp_dir().join(format!("hierarchy-invalid-{}", std::process::id()));
    std::fs::create_dir_all(&p).unwrap();
    std::fs::write(p.join("byte.int8.onnx"), b"old model").unwrap();
    let e = verify_profile(&p, &b).unwrap_err().to_string();
    assert!(e.contains("incompatible") && e.contains("no CPU encoder fallback"));
    std::fs::remove_dir_all(p).unwrap();
}
#[test]
#[ignore = "requires accepted gate artifacts and reference NPU/runtime; explicit hardware evidence"]
fn stable_profile_matches_gate_vector_and_refuses_empty_or_long_originals() {
    use std::io::BufRead;
    let fixture =
        std::path::PathBuf::from(std::env::var_os("L2D_HG_FIXTURE").expect("L2D_HG_FIXTURE"));
    let runtime =
        std::path::PathBuf::from(std::env::var_os("L2D_RUNTIME_DIR").expect("L2D_RUNTIME_DIR"));
    let work =
        std::path::PathBuf::from(std::env::var_os("L2D_NPU_WORK_DIR").expect("L2D_NPU_WORK_DIR"));
    let b = MemoryBudget::observe(&MemorySettings {
        memory_limit_bytes: Some(SESSION_BYTES * 2),
        ..Default::default()
    })
    .unwrap();
    let mut session =
        HierarchicalSession::open(&fixture, &runtime, &work.join("cache"), &b).unwrap();
    let line = std::io::BufReader::new(std::fs::File::open(fixture.join("texts.jsonl")).unwrap())
        .lines()
        .next()
        .unwrap()
        .unwrap();
    let text: String = serde_json::from_str(&line).unwrap();
    assert!(text.len() <= 1024);
    let expected = std::fs::read(fixture.join("npu-raw.f32")).unwrap()[..128 * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    let actual = session.encode(&text).unwrap().unwrap();
    assert_eq!(actual, expected);
    assert!(session.encode("").unwrap().is_none());
    assert!(session.encode(&"x".repeat(1025)).unwrap().is_none());
    assert!(session.encode(&"é".repeat(513)).unwrap().is_none());
    assert_eq!(session.encode(&text).unwrap().unwrap(), actual);
}
