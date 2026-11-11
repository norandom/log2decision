//! Human and JSON rendering.
//!
//! Results and the JSON error envelope go to standard output; diagnostics go to standard
//! error. A failure always prints one line that states what failed.

use l2d_core::L2dError;

/// The `kind` of an error in JSON output; lower case, like the other enums in the JSON form.
fn error_kind(error: &L2dError) -> &'static str {
    match error {
        L2dError::Missing { .. } => "missing",
        L2dError::InvalidInput { .. } => "invalid_input",
        L2dError::Refused { .. } => "refused",
        L2dError::Runtime { .. } => "runtime",
    }
}

/// The error message on one line, whatever the underlying message contains.
fn error_message(error: &L2dError) -> String {
    error
        .to_string()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// The one-line diagnostic for standard error.
pub fn error_line(error: &L2dError) -> String {
    format!("error: {}", error_message(error))
}

/// The machine-readable form of a failure: `{"error":{"kind":…,"message":…,"exit_code":…}}`.
pub fn error_json(error: &L2dError) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "kind": error_kind(error),
            "message": error_message(error),
            "exit_code": error.exit_code(),
        }
    })
}

/// Reports a failure: always one line on standard error and, when `json` is set, the error
/// envelope as one JSON document on standard output.
pub fn report_error(error: &L2dError, json: bool) {
    eprintln!("{}", error_line(error));
    if json {
        println!("{}", error_json(error));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn error_line_stays_on_one_line() {
        let error = L2dError::Runtime {
            detail: "database error:\n  disk full\r\n".into(),
        };
        assert_eq!(
            error_line(&error),
            "error: runtime failure: database error: disk full"
        );
    }

    #[test]
    fn error_json_names_kind_message_and_exit_code_for_every_variant() {
        let cases = [
            (
                L2dError::Missing {
                    item: "store".into(),
                    expected_at: PathBuf::from("stores/x.db"),
                },
                "missing",
                3,
            ),
            (
                L2dError::InvalidInput {
                    detail: "unknown adapter".into(),
                },
                "invalid_input",
                2,
            ),
            (
                L2dError::Refused {
                    reason: "too few labels".into(),
                },
                "refused",
                4,
            ),
            (
                L2dError::Runtime {
                    detail: "provider failed".into(),
                },
                "runtime",
                5,
            ),
        ];
        for (error, kind, exit_code) in cases {
            let value = error_json(&error);
            assert_eq!(value["error"]["kind"], kind);
            assert_eq!(value["error"]["exit_code"], exit_code);
            assert_eq!(value["error"]["message"], error.to_string());
        }
    }
}
