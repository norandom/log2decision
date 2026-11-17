#[path = "npu_support/resources.rs"]
mod resources;

#[test]
fn speed_measurements_count_known_allocations_and_refuse_before_holding() {
    let measurement =
        resources::Measurement::start("lightweight throughput accounting test", false);
    let before = measurement.report()["memory"]["counted_peak_bytes"]
        .as_u64()
        .unwrap();
    resources::reserve(65536, "known fixture buffer");
    let admitted = measurement.report();
    assert_eq!(
        admitted["memory"]["counted_peak_bytes"].as_u64(),
        Some(before + 65536)
    );
    assert_eq!(admitted["native_memory_unaccounted"], false);
    assert!(
        admitted["accounting_scope"]
            .as_str()
            .unwrap()
            .contains("counted conservative")
    );
    assert!(admitted["gate_held"].as_bool().unwrap());
    let refused = std::panic::catch_unwind(|| resources::reserve(u64::MAX, "impossible output"));
    assert!(refused.is_err());
    assert_eq!(measurement.report()["memory"], admitted["memory"]);
}
