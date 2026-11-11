//! Command-line skeleton tests (requirements 10.6, 10.7; design "Command line / CLI").
//!
//! Every operation of the design's command table is reachable, accepts `--json`, and, until
//! its task wires it, fails with a non-zero exit code and one line stating what failed.

use std::process::{Command, Output};

/// Every operation of the design's command table, as the words typed after `l2d`.
const OPERATIONS: &[&[&str]] = &[
    &["ingest"],
    &["baseline", "fit"],
    &["baseline", "score"],
    &["encoder", "train"],
    &["encoder", "encode"],
    &["encoder", "info"],
    &["store", "add"],
    &["store", "search"],
    &["triage"],
    &["shift"],
    &["assess"],
    &["eval"],
    &["runtime", "check"],
];

fn l2d(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_l2d"))
        .args(args)
        .output()
        .expect("the l2d binary runs")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is UTF-8")
}

/// The single line of a stream, failing when the stream holds no line or more than one.
fn single_line(stream: &str) -> &str {
    let lines: Vec<&str> = stream.lines().collect();
    assert_eq!(lines.len(), 1, "expected exactly one line, got: {stream:?}");
    lines[0]
}

/// Whether a help text lists `name` as a subcommand (an indented line starting with it).
fn lists_subcommand(help: &str, name: &str) -> bool {
    help.lines()
        .any(|line| line.starts_with("  ") && line.split_whitespace().next() == Some(name))
}

// --- 10.7: every operation is offered from the command line -------------------------------

#[test]
fn help_lists_every_top_level_subcommand_and_the_json_flag() {
    let output = l2d(&["--help"]);
    assert_eq!(output.status.code(), Some(0));
    let help = stdout(&output);
    for name in [
        "ingest", "baseline", "encoder", "store", "triage", "shift", "assess", "eval", "runtime",
    ] {
        assert!(
            lists_subcommand(&help, name),
            "`{name}` missing in:\n{help}"
        );
    }
    assert!(help.contains("--json"), "`--json` missing in:\n{help}");
}

#[test]
fn group_help_lists_the_nested_subcommands() {
    for (group, nested) in [
        ("baseline", &["fit", "score"][..]),
        ("encoder", &["train", "encode", "info"][..]),
        ("store", &["add", "search"][..]),
        ("runtime", &["check"][..]),
    ] {
        let output = l2d(&[group, "--help"]);
        assert_eq!(output.status.code(), Some(0), "l2d {group} --help");
        let help = stdout(&output);
        for name in nested {
            assert!(
                lists_subcommand(&help, name),
                "`{group} {name}` missing in:\n{help}"
            );
        }
    }
}

#[test]
fn every_operation_accepts_the_json_flag_in_its_own_help() {
    for operation in OPERATIONS {
        let mut args = operation.to_vec();
        args.push("--help");
        let output = l2d(&args);
        assert_eq!(output.status.code(), Some(0), "l2d {}", args.join(" "));
        assert!(
            stdout(&output).contains("--json"),
            "`--json` missing in the help of l2d {}",
            operation.join(" ")
        );
    }
}

// --- 10.6: a failing operation exits non-zero and states what failed ----------------------

#[test]
fn every_stub_exits_non_zero_with_one_line_on_stderr() {
    for operation in OPERATIONS {
        let name = operation.join(" ");
        let output = l2d(operation);
        assert_eq!(output.status.code(), Some(4), "exit code of l2d {name}");
        assert_eq!(stdout(&output), "", "l2d {name} must not print to stdout");
        let err = stderr(&output);
        let line = single_line(&err);
        assert!(line.starts_with("error: "), "l2d {name}: {line}");
        assert!(line.contains("not implemented"), "l2d {name}: {line}");
        assert!(
            line.contains(&format!("l2d {name}")),
            "the message names the command: {line}"
        );
    }
}

#[test]
fn json_flag_puts_a_machine_readable_error_on_stdout() {
    // The flag is global: it is accepted before and after the subcommand.
    for args in [
        &["--json", "ingest"][..],
        &["ingest", "--json"][..],
        &["store", "search", "--json"][..],
        &["store", "--json", "search"][..],
    ] {
        let output = l2d(args);
        assert_eq!(output.status.code(), Some(4), "l2d {}", args.join(" "));
        let out = stdout(&output);
        let value: serde_json::Value =
            serde_json::from_str(single_line(&out)).expect("stdout is one JSON document");
        assert_eq!(value["error"]["kind"], "refused");
        assert_eq!(value["error"]["exit_code"], 4);
        let message = value["error"]["message"].as_str().expect("message text");
        assert!(message.contains("not implemented"), "{message}");
        // The diagnostic still goes to stderr, as one line.
        let err = stderr(&output);
        assert!(single_line(&err).contains("not implemented"));
    }
}

#[test]
fn unknown_subcommand_is_an_analyst_error_with_one_line() {
    let output = l2d(&["frobnicate"]);
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(stdout(&output), "");
    let err = stderr(&output);
    let line = single_line(&err);
    assert!(line.starts_with("error: invalid input: "), "{line}");
    assert!(line.contains("frobnicate"), "{line}");
}

#[test]
fn unknown_option_with_json_reports_invalid_input_as_json() {
    let output = l2d(&["--json", "triage", "--no-such-option"]);
    assert_eq!(output.status.code(), Some(2));
    let out = stdout(&output);
    let value: serde_json::Value =
        serde_json::from_str(single_line(&out)).expect("stdout is one JSON document");
    assert_eq!(value["error"]["kind"], "invalid_input");
    assert_eq!(value["error"]["exit_code"], 2);
    let message = value["error"]["message"].as_str().expect("message text");
    assert!(message.contains("--no-such-option"), "{message}");
    let err = stderr(&output);
    assert!(single_line(&err).contains("--no-such-option"));
}

#[test]
fn missing_subcommand_exits_non_zero_and_shows_usage_on_stderr() {
    let output = l2d(&[]);
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(stdout(&output), "");
    assert!(stderr(&output).contains("Usage: l2d"));
}
