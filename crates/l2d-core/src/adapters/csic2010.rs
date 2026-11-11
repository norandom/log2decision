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
//! [`group_key`] turns the routing part into the group key.
//!
//! # Group key
//!
//! `METHOD /path`: no scheme, host, query string, body or protocol version, for example
//! `GET /tienda1/publico/anadir.jsp`. All requests to one endpoint share one key, whatever
//! their payload and label, which is what grouped splits need.
//!
//! # Untrusted data
//!
//! The rows are attack traffic. They are data only: nothing here executes them or uses
//! them to build paths or commands.

use std::path::{Path, PathBuf};

use crate::error::L2dError;
use crate::input::{Dedup, InputAdapter, read_source, strip_bom};
use crate::paths::ProjectPaths;
use crate::types::{InputSummary, Label, LogRecord};

/// The dataset's directory below the project's data directory.
pub const CSIC2010_DIR_NAME: &str = "csic-2010";

/// The dataset's file name.
pub const CSIC2010_FILE_NAME: &str = "csic_database.csv";

const ITEM: &str = "HTTP CSIC 2010 dataset (csic_database.csv)";

/// How many readable rows carried each label, counted before deduplication.
///
/// The deduplicated records cannot give these numbers: a text seen with both labels is
/// one record with [`Label::Unknown`].
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

/// The group key `METHOD /path` of a text in the layout of this module, or of its
/// routing part alone.
///
/// Scheme and host are dropped; a URL without a path has the path `/`. A request target
/// that is not an absolute URL is kept as it is.
pub fn group_key(text: &str) -> String {
    let routing = split_routing(text).routing;
    let Some((method, target)) = routing.split_once(' ') else {
        return routing.to_string();
    };
    format!("{method} {}", path_of(target))
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
    ) -> Result<(Vec<LogRecord>, InputSummary, Csic2010Counts), L2dError> {
        let bytes = read_source(path)?;
        let mut dedup = Dedup::new();
        let counts = read_rows(strip_bom(&bytes), &mut dedup)?;
        let (records, summary) = dedup.finish(path, &bytes);
        Ok((records, summary, counts))
    }
}

impl InputAdapter for Csic2010Adapter {
    fn read(&self, path: &Path) -> Result<(Vec<LogRecord>, InputSummary), L2dError> {
        self.read_counted(path)
            .map(|(records, summary, _)| (records, summary))
    }
}

fn read_rows(bytes: &[u8], dedup: &mut Dedup) -> Result<Csic2010Counts, L2dError> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        // Rows of any length are read so that a wrong field count is reported as a
        // skipped record with its position instead of ending the read.
        .flexible(true)
        .from_reader(bytes);
    let headers = reader
        .byte_headers()
        .map_err(|error| L2dError::InvalidInput {
            detail: format!("unreadable CSV header: {error}"),
        })?
        .clone();
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

    let mut counts = Csic2010Counts::default();
    let mut row = csv::ByteRecord::new();
    let mut position = 0u64;
    loop {
        let more = reader
            .read_byte_record(&mut row)
            .map_err(|error| L2dError::Runtime {
                detail: format!("reading CSV data row {}: {error}", position + 1),
            })?;
        if !more {
            return Ok(counts);
        }
        position += 1;
        if row.len() != headers.len() {
            dedup.skip(
                position,
                format!("expected {} fields, found {}", headers.len(), row.len()),
            );
            continue;
        }
        match parse_row(&row, method_at, url_at, body_at, class_at, status_at) {
            Ok((text, label)) => {
                match label {
                    Label::Good => counts.good += 1,
                    Label::Bad => counts.bad += 1,
                    Label::Unknown => {}
                }
                let group = group_key(&text);
                dedup.accept(position, text, label, Some(group));
            }
            Err(reason) => dedup.skip(position, reason),
        }
    }
}

/// The text and label of one row, or the reason it cannot be read.
fn parse_row(
    row: &csv::ByteRecord,
    method_at: usize,
    url_at: usize,
    body_at: usize,
    class_at: usize,
    status_at: Option<usize>,
) -> Result<(String, Label), String> {
    let field = |at: usize, name: &str| {
        std::str::from_utf8(&row[at]).map_err(|_| format!("{name} is not valid UTF-8"))
    };
    let method = field(method_at, "method")?.trim();
    if method.is_empty() {
        return Err("empty method".into());
    }
    if method.contains(|c: char| c.is_whitespace() || c == '?') {
        return Err("method contains white space or '?'".into());
    }
    let url = without_protocol_version(field(url_at, "URL")?.trim());
    if url.is_empty() {
        return Err("empty URL".into());
    }
    if url.contains(['\n', '\r']) {
        return Err("URL contains a line break".into());
    }
    let class = field(class_at, "classification")?.trim();
    let (label, status_name) = match class {
        "0" => (Label::Good, "Normal"),
        "1" => (Label::Bad, "Anomalous"),
        other => return Err(format!("classification '{other}' is neither 0 nor 1")),
    };
    if let Some(at) = status_at {
        let status = field(at, "first column")?.trim();
        if status != status_name {
            return Err(format!(
                "the first column '{status}' and the classification column '{class}' disagree"
            ));
        }
    }
    let body = field(body_at, "body")?;
    let text = if body.is_empty() {
        format!("{method} {url}")
    } else {
        format!("{method} {url}\n{body}")
    };
    Ok((text, label))
}
