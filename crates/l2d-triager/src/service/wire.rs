//! The transport limits of the interface contract, the error shape, and the small
//! projections every result goes through: record IDs, cursors, safe display text.

use l2d_core::L2dError;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// The version every request carries and every result states.
pub const VERSION: u64 = 1;
/// Rows of a page when the request names no limit.
pub const DEFAULT_ROWS: usize = 25;
/// The most rows of a page.
pub const MAX_ROWS: usize = 100;
/// The most bytes of one serialized MCP tool result, its text form included.
pub const MAX_RESULT_BYTES: usize = 32 * 1024;
/// The most bytes of one projected excerpt.
pub const MAX_EXCERPT_BYTES: usize = 1024;
/// The most bytes of an error message, a reason or a displayed entry path.
pub const MAX_MESSAGE_BYTES: usize = 1024;
pub(crate) const MAX_PATH_BYTES: usize = 320;
/// The most named sets of one request.
pub(crate) const MAX_SETS: usize = 8;
/// The most named sets of a region overview: the eight regions of A, B and C.
pub(crate) const MAX_REGION_SETS: usize = 3;

/// What the MCP result object adds around a payload: `content` with one text item,
/// `structuredContent` and `isError`.
const RESULT_ENVELOPE_BYTES: usize = 96;
/// What a job status adds around the result of its job, in both forms.
const JOB_WRAPPER_BYTES: usize = 2 * 1024;
/// The most a tool payload may take as an MCP result, so that it still fits when a
/// job status carries it.
pub(crate) const PAYLOAD_TRANSPORT_BYTES: usize = MAX_RESULT_BYTES - JOB_WRAPPER_BYTES;

/// What `payload` (serialized JSON) adds to its own length when it is written again
/// as a JSON string: one byte for every quote and backslash. The payloads here come
/// from `serde_json`, which writes control characters as escapes already.
pub(crate) fn escape_bytes(payload: &str) -> usize {
    payload
        .bytes()
        .filter(|byte| matches!(byte, b'"' | b'\\'))
        .count()
}

/// The bytes of the MCP tool result that carries `payload`: the structured form, the
/// text form (the same JSON as a string) and the envelope.
pub fn transport_bytes(payload: &str) -> usize {
    RESULT_ENVELOPE_BYTES + 2 * payload.len() + escape_bytes(payload)
}

/// Serializes `result` with `cost.response_bytes` set to [`transport_bytes`] of the
/// serialized result itself: what the client is returned, counted the way the limit
/// is. The figure is part of what it counts, so it is settled in a few rounds.
pub(crate) fn with_response_bytes(mut result: Value) -> String {
    settle_response_bytes(&mut result, &|written| written)
}

/// [`with_response_bytes`] for a result that `finish` completes once it is
/// serialized, for what is spliced into it unparsed.
pub(crate) fn settle_response_bytes(
    result: &mut Value,
    finish: &dyn Fn(String) -> String,
) -> String {
    let mut bytes = 0;
    let mut written = String::new();
    for _ in 0..8 {
        result["cost"]["response_bytes"] = json!(bytes);
        written = finish(result.to_string());
        let counted = transport_bytes(&written);
        if counted == bytes {
            break;
        }
        bytes = counted;
    }
    written
}

/// Why a call ends without a result. `code` is one of a fixed list a client can
/// branch on; `message` says what was observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Fault {
    pub code: &'static str,
    pub message: String,
    /// Whether the message was cut when the fault was kept ([`Fault::bounded`]).
    pub clipped: bool,
}

pub(crate) const CANCELLED: &str = "cancelled";
/// The reason of the refusal that ends a read when its job was cancelled.
const CANCELLED_REASON: &str = "the job was cancelled by its client";

impl Fault {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            clipped: false,
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new("invalid_arguments", message)
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new("unsupported_operation", message)
    }

    pub fn refused(message: impl Into<String>) -> Self {
        Self::new("refused", message)
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::new("runtime_failure", message)
    }

    /// The fault as the job table keeps it: its message in the display projection and
    /// at most [`MAX_MESSAGE_BYTES`], so that a kept fault takes a known share.
    pub fn bounded(self) -> Self {
        let (message, clipped) = display(&self.message, MAX_MESSAGE_BYTES);
        Self {
            code: self.code,
            clipped: clipped || self.clipped,
            message,
        }
    }

    /// The error payload: bounded, with clipping stated. A message can quote a path
    /// or an entry name, so it goes through the display projection as evidence does.
    pub fn payload(&self) -> String {
        let (message, clipped) = display(&self.message, MAX_MESSAGE_BYTES);
        json!({
            "v": VERSION,
            "status": "error",
            "error": {"code": self.code, "message": message, "clipped": clipped || self.clipped},
        })
        .to_string()
    }
}

/// The error a read is ended with when its job was cancelled.
pub(crate) fn cancelled() -> L2dError {
    L2dError::Refused {
        reason: CANCELLED_REASON.into(),
    }
}

impl From<L2dError> for Fault {
    fn from(error: L2dError) -> Self {
        match error {
            L2dError::Refused { reason } if reason == CANCELLED_REASON => {
                Fault::new(CANCELLED, reason)
            }
            L2dError::InvalidInput { detail } => Fault::invalid(detail),
            L2dError::Missing { .. } => Fault::new("missing", error.to_string()),
            L2dError::Refused { reason } => Fault::refused(reason),
            L2dError::Runtime { detail } => Fault::runtime(detail),
        }
    }
}

impl From<Fault> for L2dError {
    fn from(fault: Fault) -> Self {
        match fault.code {
            "invalid_arguments" | "unsupported_version" | "unsupported_operation" => {
                L2dError::InvalidInput {
                    detail: fault.message,
                }
            }
            "runtime_failure" => L2dError::Runtime {
                detail: fault.message,
            },
            _ => L2dError::Refused {
                reason: format!("{}: {}", fault.code, fault.message),
            },
        }
    }
}

/// `text` cut to at most `limit` bytes on a character boundary, and whether it was cut.
pub(crate) fn clip(text: &str, limit: usize) -> (&str, bool) {
    if text.len() <= limit {
        return (text, false);
    }
    let mut end = limit;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (&text[..end], true)
}

/// A safe display projection of untrusted text: terminal controls, invisible and
/// direction-changing characters are written as escapes, never interpreted. At most
/// `limit` bytes are written; a character that no longer fits is left out whole.
/// Returns the projection and whether anything of `text` was left out.
pub(crate) fn display(text: &str, limit: usize) -> (String, bool) {
    let mut shown = String::with_capacity(text.len().min(limit));
    let mut escaped = [0u8; 16];
    for character in text.chars() {
        let piece: &str = if l2d_core::terminal_text::is_invisible_or_control(character) {
            let mut length = 0;
            for part in character.escape_default() {
                length += part.encode_utf8(&mut escaped[length..]).len();
            }
            std::str::from_utf8(&escaped[..length]).unwrap_or("?")
        } else {
            character.encode_utf8(&mut escaped)
        };
        if shown.len() + piece.len() > limit {
            return (shown, true);
        }
        shown.push_str(piece);
    }
    (shown, false)
}

/// The record handle of the original first read at `position`.
pub(crate) fn record_id(position: u64) -> String {
    format!("r-{position:016x}")
}

/// The position a record handle names; `None` when it is no record handle.
pub(crate) fn record_position(id: &str) -> Option<u64> {
    let digits = id.strip_prefix("r-")?;
    if digits.len() != 16
        || !digits
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    u64::from_str_radix(digits, 16).ok()
}

/// A handle of `kind` for `parts`, bound to the project: the same content under
/// another project is another handle, and a handle of another project resolves to
/// nothing.
pub(crate) fn handle(kind: &str, project: &str, parts: &[&str]) -> String {
    let mut hash = Sha256::new();
    for part in [kind, project].iter().chain(parts) {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part.as_bytes());
    }
    let digest = format!("{:x}", hash.finalize());
    format!("{kind}-{}", &digest[..32])
}

/// What a cursor is bound to beside the position it continues after: the project, the
/// handle (and so its revision), the view, the membership columns, the order and the
/// projection.
pub(crate) fn cursor_binding(project: &str, parts: &[&str]) -> String {
    handle("cursor", project, parts)[7..23].to_string()
}

pub(crate) fn cursor(after: u64, binding: &str) -> String {
    format!("c1.{after:016x}.{binding}")
}

/// The position a cursor continues after, when it was issued for `binding`.
pub(crate) fn cursor_position(cursor: &str, binding: &str) -> Option<u64> {
    let mut parts = cursor.split('.');
    let (version, after, bound) = (parts.next()?, parts.next()?, parts.next()?);
    if version != "c1" || parts.next().is_some() || bound != binding || after.len() != 16 {
        return None;
    }
    u64::from_str_radix(after, 16).ok()
}

/// The limits every inspect result states.
pub(crate) fn applied_limits(rows: usize) -> Value {
    json!({
        "rows": rows,
        "default_rows": DEFAULT_ROWS,
        "max_rows": MAX_ROWS,
        "max_result_bytes": MAX_RESULT_BYTES,
        "max_excerpt_bytes": MAX_EXCERPT_BYTES,
    })
}

/// The retained share of a parent; undefined for an empty parent.
pub(crate) fn share(part: u64, parent: u64) -> Value {
    if parent == 0 {
        Value::Null
    } else {
        json!(part as f64 / parent as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_transport_size_is_the_size_of_the_result_object() {
        for payload in [
            json!({"v": 1, "status": "ready"}),
            json!({"v": 1, "text": "a \"quoted\" \\ path\n\u{1b}[0m", "rows": [1, 2, {"x": "é"}]}),
        ] {
            let payload = payload.to_string();
            let structured: Value = serde_json::from_str(&payload).unwrap();
            let result = json!({
                "content": [{"type": "text", "text": payload}],
                "structuredContent": structured,
                "isError": false,
            });
            let written = serde_json::to_string(&result).unwrap().len();
            assert!(written <= transport_bytes(&payload), "{written}");
            assert!(transport_bytes(&payload) - written <= RESULT_ENVELOPE_BYTES);
        }
    }

    #[test]
    fn the_response_bytes_of_a_result_count_the_result_that_states_them() {
        // Sizes on both sides of a change in the number of digits.
        for filler in [0, 4_380, 4_420, 49_000] {
            let written = with_response_bytes(
                json!({"v": 1, "pad": "p".repeat(filler), "cost": {"archive_scans": 0}}),
            );
            let value: Value = serde_json::from_str(&written).unwrap();
            assert_eq!(
                value["cost"]["response_bytes"],
                transport_bytes(&written) as u64,
                "{filler}"
            );
        }
        let spliced = settle_response_bytes(&mut json!({"cost": {}}), &|mut written| {
            written.pop();
            written.push_str(",\"artifact\":{\"x\":\"y\"}}");
            written
        });
        let value: Value = serde_json::from_str(&spliced).unwrap();
        assert_eq!(value["artifact"]["x"], "y");
        assert_eq!(
            value["cost"]["response_bytes"],
            transport_bytes(&spliced) as u64
        );
    }

    #[test]
    fn display_escapes_controls_and_cuts_on_a_character() {
        let (shown, cut) = display("a\u{1b}[31m\u{202e}é\tz", 100);
        assert_eq!(shown, "a\\u{1b}[31m\\u{202e}é\\tz");
        assert!(!cut);
        let (shown, cut) = display("ééé", 5);
        assert_eq!((shown.as_str(), cut), ("éé", true));
        let (shown, cut) = display("ab\u{7}", 4);
        assert_eq!((shown.as_str(), cut), ("ab", true));
        assert_eq!(clip("ééé", 3), ("é", true));
    }

    #[test]
    fn record_handles_and_cursors_round_trip_and_reject_others() {
        assert_eq!(record_position(&record_id(u64::MAX)), Some(u64::MAX));
        assert_eq!(record_position(&record_id(7)), Some(7));
        for other in ["R1", "r-7", "r-000000000000000G", "r-00000000000000071"] {
            assert_eq!(record_position(other), None, "{other}");
        }
        let binding = cursor_binding("p", &["set-1", "records"]);
        assert_eq!(cursor_position(&cursor(9, &binding), &binding), Some(9));
        let other = cursor_binding("q", &["set-1", "records"]);
        assert_eq!(cursor_position(&cursor(9, &binding), &other), None);
        assert_eq!(cursor_position("zz", &binding), None);
        assert_ne!(
            handle("set", "p", &["ab", "c"]),
            handle("set", "p", &["a", "bc"])
        );
    }
}
