//! The source a score run records in its `manifest.json`, as a claim a scope can
//! verify before the run's rows are joined to its originals (`Scope::join_keys`).
//!
//! `l2d baseline score` on an archive writes its rows as result parts and, last, a
//! manifest naming the archive, its SHA-256, the entry patterns and the maximum record
//! length. Those are what a scope binds too, so the manifest alone says whether the
//! rows were made on the scope's source and reading.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use l2d_core::marks::InputInterpretation;
use l2d_core::{L2dError, MarkedInput, MarkedSource, MemoryBudget, ZipInputSettings};
use serde::Deserialize;

/// What reading a manifest holds beside the strings it keeps: the read buffer and the
/// parser's own state.
const READ_FIXED_BYTES: u64 = 256 * 1024;
const FORMAT: &str = "l2d-result-parts";
const COMMAND: &str = "baseline score";

/// The scorer a score run was made with.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ManifestScorer {
    pub kind: String,
    pub name: String,
    pub revision: String,
}

/// What a score run's manifest says about its source, its scorer and its rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreManifest {
    /// The archive, its revision, the entry patterns and the reader interpretation.
    pub source: MarkedSource,
    pub scorer: ManifestScorer,
    /// The rows of the scores table: the distinct logs of the archive.
    pub rows: u64,
}

#[derive(Deserialize)]
struct Settings {
    entries: Vec<String>,
    max_record_bytes: u64,
}

#[derive(Deserialize)]
struct Input {
    source_path: String,
    content_sha256: String,
}

/// The fields read; the entry reports and the tables are passed over unread.
#[derive(Deserialize)]
struct Document {
    format: String,
    version: u32,
    command: String,
    scorer: ManifestScorer,
    settings: Settings,
    input: Input,
    rows: u64,
}

fn refused(path: &Path, reason: &str) -> L2dError {
    L2dError::Refused {
        reason: format!("the score manifest {} {reason}", path.display()),
    }
}

impl ScoreManifest {
    /// Reads the manifest at `path`.
    ///
    /// The file is read as a stream; what is kept of it is at most its own size, which
    /// is asked of `budget` first. [`L2dError::Missing`] when there is no file (a run
    /// that did not end writes none), [`L2dError::Refused`] for another format,
    /// version or command, for a hash that is no SHA-256, and for a source path that
    /// is not absolute: such a path names another file from every other directory, so
    /// the claim could not be verified.
    pub fn load(path: &Path, budget: &MemoryBudget) -> Result<Self, L2dError> {
        let file = File::open(path).map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => L2dError::Missing {
                item: "score manifest".into(),
                expected_at: path.to_path_buf(),
            },
            _ => L2dError::Runtime {
                detail: format!("opening {} failed: {error}", path.display()),
            },
        })?;
        let size = file
            .metadata()
            .map_err(|error| L2dError::Runtime {
                detail: format!("reading the size of {} failed: {error}", path.display()),
            })?
            .len();
        budget.require_whole(
            READ_FIXED_BYTES.saturating_add(size.saturating_mul(2)),
            "reading a score manifest",
        )?;
        let document: Document = serde_json::from_reader(BufReader::new(file))
            .map_err(|error| refused(path, &format!("cannot be read: {error}")))?;
        if document.format != FORMAT || document.version != 1 || document.command != COMMAND {
            return Err(refused(
                path,
                "is not the manifest of a `baseline score` run in format 1",
            ));
        }
        let hash = &document.input.content_sha256;
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(refused(path, "records a content hash that is no SHA-256"));
        }
        let source_path = Path::new(&document.input.source_path);
        if !source_path.is_absolute() {
            return Err(refused(
                path,
                "records a relative source path, which names another file from another \
                 directory; score the archive again with its absolute path",
            ));
        }
        let settings = ZipInputSettings {
            entries: document.settings.entries,
            max_record_bytes: document.settings.max_record_bytes,
            entries_at_once: None,
        };
        Ok(Self {
            source: MarkedSource::Input(MarkedInput {
                path: MarkedInput::source_location(source_path)?,
                content_sha256: document.input.content_sha256,
                interpretation: Some(InputInterpretation::archive(&settings)),
                archive_entries: Some(settings.entries),
            }),
            scorer: document.scorer,
            rows: document.rows,
        })
    }
}
