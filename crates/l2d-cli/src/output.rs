//! Human and JSON rendering.
//!
//! Results and the JSON error envelope go to standard output; diagnostics go to standard
//! error. A failure always prints one line that states what failed.
//!
//! # A failure after output has begun
//!
//! `l2d baseline score` and `l2d triage` write their rows as they go, so that a result
//! of any size is never in memory as a whole. Everything that can be checked is checked
//! before the first byte is written (options, the models, the input, the memory for
//! scoring and printing); what can still fail afterwards is the scoring of a later
//! batch when the free memory of the host has fallen (`baseline score`), and writing
//! itself (a closed pipe, a full disk). A consumer must therefore check the exit code:
//! on a failed run standard output is not a valid document. With `--json` it is then
//! the beginning of the object, cut off, and after a line feed the error envelope on
//! the last line of its own, so the last line of standard output is always one whole
//! JSON document: the result, or the envelope. When standard output itself is closed
//! the envelope cannot be written and only the exit code and standard error say so.
//! Nothing here panics on a closed stream: every write to standard output and standard
//! error is a write whose failure is handled or ignored.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use l2d_core::L2dError;
use serde::Serialize;

/// The failure of writing a result to standard output (a closed pipe, a full disk).
pub fn output_failed(error: io::Error) -> L2dError {
    L2dError::Runtime {
        detail: format!("writing the output failed: {error}"),
    }
}

/// `bytes` in the largest binary unit that keeps it at or above one, with one decimal.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

/// Whether a character changes how a terminal lays out or interprets what follows
/// without being visible itself: control characters (C0, C1), and the format
/// characters that hide text or reverse its direction (soft hyphen, the Arabic letter
/// mark, zero-width characters and direction marks U+200B to U+200F, line and paragraph
/// separators and the bidirectional embeddings and overrides U+2028 to U+202E, the word
/// joiner, invisible operators and bidirectional isolates U+2060 to U+206F, the byte
/// order mark, the interlinear annotation marks and the tag characters).
fn is_invisible_or_control(character: char) -> bool {
    character.is_control()
        || matches!(
            character,
            '\u{ad}'
                | '\u{61c}'
                | '\u{180e}'
                | '\u{200b}'..='\u{200f}'
                | '\u{2028}'..='\u{202e}'
                | '\u{2060}'..='\u{206f}'
                | '\u{feff}'
                | '\u{fff9}'..='\u{fffb}'
                | '\u{e0000}'..='\u{e007f}'
        )
}

/// A log text for the text output, on one line: control characters and invisible
/// format characters are shown escaped (log text is data, often an attacker's, and must
/// not reach the terminal as control sequences or reorder what is displayed), and a
/// text of more than `max_chars` characters is cut and ends in `...`.
pub fn shown_text(text: &str, max_chars: usize) -> String {
    let mut shown = String::with_capacity(text.len().min(max_chars * 2) + 3);
    for (index, character) in text.chars().enumerate() {
        if index == max_chars {
            shown.push_str("...");
            break;
        }
        if is_invisible_or_control(character) {
            shown.extend(character.escape_default());
        } else {
            shown.push(character);
        }
    }
    shown
}

/// One JSON object written piece by piece, so that a list of any length is never in
/// memory as a whole: fields in the order they are given, lists element by element.
/// The document is one line and ends in a line feed.
///
/// While a document is begun and not ended, [`report_error`] puts a line feed before
/// the error envelope (module documentation, "A failure after output has begun").
pub struct JsonOut<W: Write> {
    out: W,
    fields: usize,
    elements: usize,
}

impl<W: Write> JsonOut<W> {
    /// Starts the object.
    pub fn begin(mut out: W) -> io::Result<Self> {
        DOCUMENT_OPEN.store(true, Ordering::Relaxed);
        out.write_all(b"{")?;
        Ok(Self {
            out,
            fields: 0,
            elements: 0,
        })
    }

    fn key(&mut self, key: &str) -> io::Result<()> {
        if self.fields > 0 {
            self.out.write_all(b",")?;
        }
        self.fields += 1;
        serde_json::to_writer(&mut self.out, key)?;
        self.out.write_all(b":")
    }

    /// Writes one field.
    pub fn field<T: Serialize + ?Sized>(&mut self, key: &str, value: &T) -> io::Result<()> {
        self.key(key)?;
        serde_json::to_writer(&mut self.out, value)?;
        Ok(())
    }

    /// Starts a field that is a list; follow with [`JsonOut::element`] and
    /// [`JsonOut::end_list`].
    pub fn begin_list(&mut self, key: &str) -> io::Result<()> {
        self.key(key)?;
        self.elements = 0;
        self.out.write_all(b"[")
    }

    /// Writes one element of the list that was started.
    pub fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> io::Result<()> {
        if self.elements > 0 {
            self.out.write_all(b",")?;
        }
        self.elements += 1;
        serde_json::to_writer(&mut self.out, value)?;
        Ok(())
    }

    /// Ends the list that was started.
    pub fn end_list(&mut self) -> io::Result<()> {
        self.out.write_all(b"]")
    }

    /// Ends the object and the line, and flushes.
    pub fn end(mut self) -> io::Result<()> {
        self.out.write_all(b"}\n")?;
        self.out.flush()?;
        DOCUMENT_OPEN.store(false, Ordering::Relaxed);
        Ok(())
    }
}

/// Whether a JSON document was begun on standard output and not ended.
static DOCUMENT_OPEN: AtomicBool = AtomicBool::new(false);

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

/// What a failure puts on standard output with `--json`: the envelope on a line of its
/// own. After a document that was begun and not ended, a line feed comes first.
fn error_document(error: &L2dError, document_open: bool) -> String {
    format!(
        "{}{}\n",
        if document_open { "\n" } else { "" },
        error_json(error)
    )
}

/// Reports a failure: always one line on standard error and, when `json` is set, the error
/// envelope as one JSON document on standard output.
///
/// A stream that cannot be written to (a closed pipe) is not a second failure: the
/// exit code still says what happened.
pub fn report_error(error: &L2dError, json: bool) {
    let _ = writeln!(io::stderr(), "{}", error_line(error));
    if json {
        let document = error_document(error, DOCUMENT_OPEN.load(Ordering::Relaxed));
        let mut out = io::stdout();
        let _ = out
            .write_all(document.as_bytes())
            .and_then(|()| out.flush());
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

    #[test]
    fn a_json_object_is_written_piece_by_piece_as_one_line() {
        let mut bytes = Vec::new();
        let mut json = JsonOut::begin(&mut bytes).unwrap();
        // This is the one test that begins a document, so the flag is its own.
        assert!(DOCUMENT_OPEN.load(Ordering::Relaxed));
        json.field("command", "x").unwrap();
        json.begin_list("rows").unwrap();
        json.element(&serde_json::json!({"ratio": 0.5_f32, "text": "a\n\"b\""}))
            .unwrap();
        json.element(&2).unwrap();
        json.end_list().unwrap();
        json.begin_list("none").unwrap();
        json.end_list().unwrap();
        json.field("count", &2).unwrap();
        json.end().unwrap();
        assert!(!DOCUMENT_OPEN.load(Ordering::Relaxed));
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(
            text,
            "{\"command\":\"x\",\"rows\":[{\"ratio\":0.5,\"text\":\"a\\n\\\"b\\\"\"},2],\"none\":[],\"count\":2}\n"
        );
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["rows"][0]["text"], "a\n\"b\"");
    }

    #[test]
    fn log_text_is_shown_on_one_line_without_control_characters() {
        assert_eq!(shown_text("GET /a?b=c", 100), "GET /a?b=c");
        assert_eq!(shown_text("a\nb\tc\u{1b}[31m", 100), "a\\nb\\tc\\u{1b}[31m");
        assert_eq!(shown_text("abcdef", 3), "abc...");
        assert_eq!(shown_text("abc", 3), "abc");
        assert_eq!(shown_text("\u{e9}t\u{e9}", 2), "\u{e9}t...");
    }

    #[test]
    fn invisible_format_characters_are_shown_escaped() {
        // Bidirectional overrides, embeddings and isolates, zero-width characters, the
        // byte order mark, separators, tags: none reaches the terminal.
        let hidden = [
            '\u{ad}',
            '\u{61c}',
            '\u{180e}',
            '\u{200b}',
            '\u{200c}',
            '\u{200d}',
            '\u{200e}',
            '\u{200f}',
            '\u{2028}',
            '\u{2029}',
            '\u{202a}',
            '\u{202b}',
            '\u{202c}',
            '\u{202d}',
            '\u{202e}',
            '\u{2060}',
            '\u{2066}',
            '\u{2067}',
            '\u{2068}',
            '\u{2069}',
            '\u{feff}',
            '\u{fff9}',
            '\u{e0001}',
            '\u{e007f}',
            '\u{85}',
            '\u{7f}',
        ];
        for character in hidden {
            let shown = shown_text(&format!("a{character}b"), 100);
            assert_eq!(
                shown,
                format!("a\\u{{{:x}}}b", character as u32),
                "U+{:04X}",
                character as u32
            );
            assert!(shown.is_ascii());
        }
        assert_eq!(
            shown_text("file=\u{202e}gpj.exe", 100),
            "file=\\u{202e}gpj.exe"
        );
        // Visible text in any script stays as it is.
        let visible = "\u{e9}t\u{e9} \u{4e2d}\u{6587} \u{5d0}\u{5d1} \u{1f600} a-b_c";
        assert_eq!(shown_text(visible, 100), visible);
    }

    #[test]
    fn the_error_envelope_is_always_a_line_of_its_own() {
        let error = L2dError::Refused {
            reason: "the free memory\nfell".into(),
        };
        let alone = error_document(&error, false);
        assert_eq!(alone.lines().count(), 1);
        assert!(alone.ends_with("}\n"));
        // After a document that was cut off: a line feed first, so the last line of
        // standard output is the envelope and nothing else.
        let after = format!(
            "{{\"command\":\"x\",\"scores\":[{{\"row\":0}}{}",
            error_document(&error, true)
        );
        let last = after.lines().last().unwrap();
        let value: serde_json::Value = serde_json::from_str(last).unwrap();
        assert_eq!(value["error"]["kind"], "refused");
        assert_eq!(value["error"]["exit_code"], 4);
        assert_eq!(after.lines().count(), 2);
        assert!(serde_json::from_str::<serde_json::Value>(&after).is_err());
    }

    #[test]
    fn bytes_are_shown_in_binary_units() {
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(64 * 1024 * 1024), "64.0 MiB");
        assert_eq!(human_bytes(3 * 1024 * 1024 * 1024 / 2), "1.5 GiB");
    }
}
