//! Triager task 3.2: `l2d tui --demo` and `--report` refuse the options that belong to
//! the workbench over the service (`--trace` and the host options of `l2d mcp`), so
//! no option is accepted and then silently ignored.

use std::process::Command;

#[test]
fn the_retained_workbench_refuses_the_options_of_the_workbench_over_the_service() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("tui-options");
    std::fs::create_dir_all(&dir).unwrap();
    let trace = dir.join("trace.jsonl");
    let _ = std::fs::remove_file(&trace);
    let trace_arg = trace.to_string_lossy().into_owned();
    let cases: [&[&str]; 8] = [
        &["--trace", &trace_arg],
        &["--source", "logs=archive.zip"],
        &["--membership", "c=c.jsonl"],
        &["--inline-wait-ms", "5"],
        &["--receipts", "receipts"],
        &["--disclosure", "private-local"],
        &["--max-patterns", "256"],
        &["--tables-on-disk"],
    ];
    for extra in cases {
        let output = Command::new(env!("CARGO_BIN_EXE_l2d"))
            .current_dir(&dir)
            .args(["--json", "tui", "--demo"])
            .args(extra)
            .output()
            .unwrap();
        let shown = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.status.success(), "{extra:?} was accepted: {shown}");
        assert!(shown.contains(extra[0]), "{extra:?}: {shown}");
        assert!(
            shown.contains("--requests or --receipt"),
            "{extra:?}: {shown}"
        );
    }
    assert!(!trace.exists(), "a refused --trace writes nothing");
    // The workbench over the service waits 50 ms inline and polls: it refuses
    // --inline-wait-ms rather than ignore it, before anything is opened.
    let output = Command::new(env!("CARGO_BIN_EXE_l2d"))
        .current_dir(&dir)
        .args([
            "--json",
            "tui",
            "--requests",
            "absent.jsonl",
            "--inline-wait-ms",
            "5",
        ])
        .output()
        .unwrap();
    let shown = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.status.success(), "{shown}");
    assert!(shown.contains("--inline-wait-ms"), "{shown}");
    assert!(
        !shown.contains("absent.jsonl"),
        "refused before the recipe is read: {shown}"
    );
    // Without them the demo still opens.
    let output = Command::new(env!("CARGO_BIN_EXE_l2d"))
        .current_dir(&dir)
        .args(["--json", "tui", "--demo"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
}
