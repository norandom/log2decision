//! HTTP CSIC 2010 (`csic_database.csv`), the first evaluation dataset.
//!
//! # Source layout
//!
//! A CSV file with a header row. The columns used are `Method`, `URL`, `content` (the
//! request body, often empty) and `classification` (`0` normal, `1` anomalous). The first
//! column has an empty header name and repeats the label as `Normal` / `Anomalous`; when
//! it is present it must agree with `classification`. The `URL` column holds the request
//! target followed by the protocol version, for example
//! `http://localhost:8080/tienda1/publico/anadir.jsp?id=2 HTTP/1.1`.
//!
//! # Text layout
//!
//! ```text
//! METHOD SP URL [LF BODY]
//! ```
//!
//! - `URL` is the request URL as the dataset gives it — scheme, host, path and query
//!   string — without the trailing protocol version (` HTTP/1.1`), which is not part of
//!   the URL.
//! - The line feed and `BODY` are present only when the body is not empty.
//!
//! The session cookie and the other request headers are not part of the text, so the same
//! request sent in two sessions is one distinct text.
//!
//! # Routing and payload
//!
//! The *routing* part of a text is everything before the first `?` or line feed: the
//! method and the URL up to the end of the path. The *payload* part is everything after
//! that delimiter: the query string and the body. The adapter skips a row whose method
//! contains white space or `?`, or whose URL contains a line break, so the split is
//! unambiguous for every text it returns. [`split_routing`] performs the split and
//! [`group_key`] takes the group key out of the routing part.
//!
//! # Group key
//!
//! The URL path alone: no method, scheme, host, query string, body or protocol version,
//! for example `/tienda1/publico/anadir.jsp`. Paths are not normalized or case-folded.
//! All requests to one path share one key, whatever their method, payload and label,
//! which is what grouped splits need. The method is left out because the dataset repeats
//! the same payload as a GET query string and as a POST body; with the method in the key
//! the two copies could fall on different sides of a split.
//!
//! # Reading
//!
//! A validation pass over the raw rows finds the unreadable ones; Polars parses the
//! accepted rows. One function (`text_and_label`) both judges a raw row in the
//! validation pass and builds the text and label from the fields Polars parsed.
//!
//! # Untrusted data
//!
//! The rows are attack traffic. They are data only: nothing here executes them or uses
//! them to build paths or commands.

use std::path::{Path, PathBuf};

use polars::prelude::Column;

use crate::error::L2dError;
use crate::input::{
    CsvSource, Dedup, InputAdapter, csv_field_name, parse_csv_records, read_source, row_digest,
    string_column, strip_bom,
};
use crate::paths::ProjectPaths;
use crate::types::{InputSummary, Label, LogsFrame};

/// The dataset's directory below the project's data directory.
pub const CSIC2010_DIR_NAME: &str = "csic-2010";

/// The dataset's file name.
pub const CSIC2010_FILE_NAME: &str = "csic_database.csv";

const ITEM: &str = "HTTP CSIC 2010 dataset (csic_database.csv)";

/// How many readable rows carried each label, counted before deduplication.
///
/// The logs frame cannot give these numbers: a text seen with both labels is one row
/// with the label `unknown`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Csic2010Counts {
    pub good: u64,
    pub bad: u64,
}

/// A CSIC-layout text split into its routing and payload parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingSplit<'a> {
    /// The method and the URL up to the end of the path.
    pub routing: &'a str,
    /// The query string and the body, without the delimiter that followed the routing
    /// part; empty when the request has neither.
    pub payload: &'a str,
}

/// Splits a text in the layout of this module at its first `?` or line feed.
///
/// The delimiter itself belongs to neither part. A text without a delimiter is routing
/// only. This is meant for texts in the CSIC layout; other log lines have no such
/// structure.
pub fn split_routing(text: &str) -> RoutingSplit<'_> {
    match text.find(['?', '\n']) {
        Some(at) => RoutingSplit {
            routing: &text[..at],
            payload: &text[at + 1..],
        },
        None => RoutingSplit {
            routing: text,
            payload: "",
        },
    }
}

/// The group key of a text in the layout of this module, or of its routing part alone:
/// the URL path, without the method.
///
/// Scheme and host are dropped; a URL without a path has the path `/`. A request target
/// that is not an absolute URL is kept as it is. The path is not case-folded.
pub fn group_key(text: &str) -> String {
    let routing = split_routing(text).routing;
    let target = routing
        .split_once(' ')
        .map_or(routing, |(_method, target)| target);
    path_of(target).to_string()
}

/// The path of `target`, which holds no query string.
fn path_of(target: &str) -> &str {
    let Some((scheme, rest)) = target.split_once("://") else {
        return target;
    };
    let is_scheme = scheme.starts_with(|first: char| first.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    if !is_scheme {
        return target;
    }
    match rest.find('/') {
        Some(at) => &rest[at..],
        None => "/",
    }
}

/// `field` without a trailing protocol version such as ` HTTP/1.1`.
fn without_protocol_version(field: &str) -> &str {
    match field.rsplit_once(' ') {
        Some((url, version)) if version.starts_with("HTTP/") => url,
        _ => field,
    }
}

/// Reads `csic_database.csv` of HTTP CSIC 2010.
///
/// A row is skipped and reported when its field count differs from the header, when a
/// used field is not valid UTF-8, when the method is empty or would break the text
/// layout, when the URL is empty or contains a line break, when `classification` is
/// neither `0` nor `1`, or when the first column disagrees with `classification`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Csic2010Adapter;

impl Csic2010Adapter {
    pub fn new() -> Self {
        Self
    }

    /// Locates the dataset file: `data/csic-2010/csic_database.csv` below the project
    /// root, or the path the analyst supplied. A supplied directory is expected to hold
    /// `csic_database.csv`.
    ///
    /// A file that does not exist is [`L2dError::Missing`] with its expected location.
    pub fn locate(paths: &ProjectPaths, supplied: Option<&Path>) -> Result<PathBuf, L2dError> {
        let location = match supplied {
            Some(path) if path.is_dir() => path.join(CSIC2010_FILE_NAME),
            Some(path) => path.to_path_buf(),
            None => paths
                .data_dir()
                .join(CSIC2010_DIR_NAME)
                .join(CSIC2010_FILE_NAME),
        };
        paths.require(ITEM, location)
    }

    /// Reads the dataset like [`InputAdapter::read`] and also returns the label counts
    /// before deduplication.
    pub fn read_counted(
        &self,
        path: &Path,
    ) -> Result<(LogsFrame, InputSummary, Csic2010Counts), L2dError> {
        let bytes = read_source(path)?;
        let mut dedup = Dedup::new();
        read_rows(strip_bom(&bytes), &mut dedup)?;
        let (good, bad) = dedup.label_counts();
        let (logs, summary) = dedup.finish(path, &bytes)?;
        Ok((logs, summary, Csic2010Counts { good, bad }))
    }
}

impl InputAdapter for Csic2010Adapter {
    fn read(&self, path: &Path) -> Result<(LogsFrame, InputSummary), L2dError> {
        self.read_counted(path)
            .map(|(logs, summary, _)| (logs, summary))
    }
}

fn read_rows(content: &[u8], dedup: &mut Dedup) -> Result<(), L2dError> {
    let source = CsvSource::open(content)?;
    let headers = source.headers().clone();
    let column = |name: &str| {
        headers
            .iter()
            .position(|header| header == name.as_bytes())
            .ok_or_else(|| L2dError::InvalidInput {
                detail: format!(
                    "column '{name}' is not in the CSV header ({} columns); \
                     expected the HTTP CSIC 2010 file {CSIC2010_FILE_NAME}",
                    headers.len()
                ),
            })
    };
    let method_at = column("Method")?;
    let url_at = column("URL")?;
    let body_at = column("content")?;
    let class_at = column("classification")?;
    // The unnamed first column, `Normal` / `Anomalous`.
    let status_at = headers.get(0).filter(|name| name.is_empty()).map(|_| 0);

    // Validation pass: every raw row is judged by `text_and_label`.
    let accepted = source.validate(false, dedup, |row| {
        let (text, label) = text_and_label(|field| {
            let (at, name) = match field {
                Field::Method => (method_at, "method"),
                Field::Url => (url_at, "URL"),
                Field::Class => (class_at, "classification"),
                Field::Body => (body_at, "body"),
                Field::Status => match status_at {
                    Some(at) => (at, "first column"),
                    None => return Ok(None),
                },
            };
            std::str::from_utf8(&row[at])
                .map(Some)
                .map_err(|_| format!("{name} is not valid UTF-8"))
        })?;
        Ok((row_digest(&text, label, Some(&group_key(&text))), label))
    })?;
    if accepted.is_empty() {
        return Ok(());
    }

    // Polars parses the accepted rows; the same function builds text and label from
    // its fields. The first column was checked by the validation pass and is not read
    // again.
    let parsed = parse_csv_records(
        accepted.raw(),
        headers.len(),
        &[method_at, url_at, body_at, class_at],
        b',',
        Some(b'"'),
    )?;
    let field = |at: usize| string_column(&parsed, &csv_field_name(at));
    let (methods, urls, bodies, classes) = (
        field(method_at)?,
        field(url_at)?,
        field(body_at)?,
        field(class_at)?,
    );
    let rows = parsed.height();
    let mut texts: Vec<String> = Vec::with_capacity(rows);
    let mut labels: Vec<&str> = Vec::with_capacity(rows);
    let mut groups: Vec<String> = Vec::with_capacity(rows);
    for (((method, url), body), class) in methods
        .iter()
        .zip(urls.iter())
        .zip(bodies.iter())
        .zip(classes.iter())
    {
        let built = text_and_label(|field| {
            Ok(match field {
                Field::Method => Some(method.unwrap_or_default()),
                Field::Url => Some(url.unwrap_or_default()),
                Field::Class => Some(class.unwrap_or_default()),
                Field::Body => Some(body.unwrap_or_default()),
                Field::Status => None,
            })
        });
        let (text, label) = built.map_err(|reason| L2dError::Runtime {
            detail: format!(
                "Polars and the validation pass disagree: a row the validation pass \
                 accepted reads as unusable after parsing ({reason})"
            ),
        })?;
        groups.push(group_key(&text));
        labels.push(label.as_str());
        texts.push(text);
    }
    dedup.push(
        accepted,
        Column::new("text".into(), texts),
        Column::new("label".into(), labels),
        Column::new("group".into(), groups),
    )
}

/// The fields of a row that [`text_and_label`] asks for.
#[derive(Debug, Clone, Copy)]
enum Field {
    Method,
    Url,
    Class,
    /// The unnamed first column; absent in files without it.
    Status,
    Body,
}

/// The text and label of one row, or the reason it cannot be read.
///
/// `field` gives the value of a field, `Ok(None)` when the row has no such field, or
/// the reason the field cannot be read.
fn text_and_label<'r>(
    field: impl Fn(Field) -> Result<Option<&'r str>, String>,
) -> Result<(String, Label), String> {
    let required = |which: Field| field(which).map(Option::unwrap_or_default);
    let method = required(Field::Method)?.trim();
    if method.is_empty() {
        return Err("empty method".into());
    }
    if method.contains(|c: char| c.is_whitespace() || c == '?') {
        return Err("method contains white space or '?'".into());
    }
    let url = without_protocol_version(required(Field::Url)?.trim());
    if url.is_empty() {
        return Err("empty URL".into());
    }
    if url.contains(['\n', '\r']) {
        return Err("URL contains a line break".into());
    }
    let class = required(Field::Class)?.trim();
    let (label, status_name) = match class {
        "0" => (Label::Good, "Normal"),
        "1" => (Label::Bad, "Anomalous"),
        other => return Err(format!("classification '{other}' is neither 0 nor 1")),
    };
    if let Some(status) = field(Field::Status)? {
        let status = status.trim();
        if status != status_name {
            return Err(format!(
                "the first column '{status}' and the classification column '{class}' disagree"
            ));
        }
    }
    let body = required(Field::Body)?;
    let text = if body.is_empty() {
        format!("{method} {url}")
    } else {
        format!("{method} {url}\n{body}")
    };
    Ok((text, label))
}
