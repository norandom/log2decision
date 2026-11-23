//! One bounded local investigation: an immutable scope over a file, selected archive
//! entries or a store, and the relation between its distinct originals and their
//! occurrences (triager requirements 1.1, 1.2, 1.3 and 6.2).
//!
//! A scope is opened by reading its source once with the existing readers, under the
//! existing host lease and memory budget. What it binds is in [`ScopeBinding`]: the
//! source identity and revision, the entry filter, the reader interpretation, the path
//! that policy query IDs are derived from, and a digest of the relation itself. The
//! handle is the SHA-256 of that binding, so another revision, filter or
//! interpretation is another scope.
//!
//! Every accepted distinct original is one row of the relation (`index.rs`): its first
//! native position, its native identity and its exact occurrence count. Existing
//! selection, score and policy keys are resolved to those rows in `join.rs`, after
//! the artifact's own source was compared with the binding.
//!
//! A scope-local record ID ([`OriginalId`]) is the first native position of the
//! original: the record number of a file, the packed entry and line of an archive, the
//! `seq` of a stored log. It does not depend on batch sizes, on how many entries are
//! read at once, or on whether deduplication or the relation went to disk.
//!
//! The existing readers keep the first position and the total of an original, not the
//! position of each repeat. A scope therefore locates an original by its first
//! occurrence and states its exact count; listing every repeat takes another admitted
//! pass over the source (design.md, "What the foundation already supplies").

mod index;
mod join;
mod manifest;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use l2d_core::input::content_sha256;
use l2d_core::marks::InputInterpretation;
use l2d_core::patterns::PatternStore;
use l2d_core::polars::prelude::{Column, DataFrame, PolarsError};
use l2d_core::resource::{ResourceGuard, busy_choice};
use l2d_core::{
    ArchivePosition, ArchiveSummary, EntrySkip, FreezeDiskCheck, InputAdapter, InputSummary,
    L2dError, LogsFrame, MarkedInput, MarkedSource, MarkedStore, MemoryBudget, ZipInput,
    ZipInputSettings,
};
use l2d_store::Store;
use serde::Serialize;
use sha2::{Digest, Sha256};

use index::{ID_BYTES, Index, Spool, Work};

pub use join::{JOIN_FIXED_BYTES, JoinError, JoinReport, NativeKey};
pub use manifest::{ManifestScorer, ScoreManifest};

/// What opening a scope holds beside its reader: the two write buffers of the
/// relation, the rows its tables are built from at a time, the buffer the archive is
/// hashed with again, and the identities and reasons in flight.
pub const SCOPE_FIXED_BYTES: u64 = 1024 * 1024;

const BINDING_VERSION: u32 = 1;
/// The most distinct reasons for skipped records a scope keeps apart.
const MAX_REASONS: usize = 16;
const OTHER_REASONS: &str = "other reasons";

/// Where the relation of a scope is kept once its source is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexPlacement {
    /// In memory when the budget admits all of it, on disk otherwise.
    #[default]
    MemoryWhenAdmitted,
    /// In the work directory.
    Disk,
}

/// How a scope is opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeSettings {
    /// The directory the scope writes into: absent or empty, and removed with the
    /// scope as far as the scope made it.
    pub work_dir: PathBuf,
    pub index: IndexPlacement,
    /// For an archive: the most distinct texts deduplication keeps in memory before it
    /// goes on on disk; `None` leaves that to the memory limit.
    pub dedup_set_texts: Option<u64>,
    /// The free disk space to compare with in place of the operating system's figure.
    pub free_disk_bytes: Option<u64>,
}

impl ScopeSettings {
    pub fn new(work_dir: impl Into<PathBuf>) -> Self {
        Self {
            work_dir: work_dir.into(),
            index: IndexPlacement::default(),
            dedup_set_texts: None,
            free_disk_bytes: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    File,
    Archive,
    Store,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceKind::File => "file",
            SourceKind::Archive => "archive",
            SourceKind::Store => "store",
        }
    }
}

/// The scope-local record ID of a distinct original: its first native position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OriginalId(pub u64);

/// Where an original was first read, in its source's own terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locator {
    /// The logical record of a file as its adapter numbers it; not a physical line.
    FileRecord { position: u64 },
    /// The index of the entry in the archive's directory and the 1-based line in it.
    ArchiveLine { entry: u32, line: u64 },
    /// The row of a stored log.
    StoreRow { seq: i64 },
}

/// The native identity of an original: `l2d_store::log_id` of the source and the
/// text. For a file or an archive that is the ID a policy gives the query; for a store
/// it is the stored log's ID.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativeId(pub(crate) [u8; ID_BYTES]);

impl NativeId {
    /// The 64 lower-case hexadecimal characters the producers write.
    pub fn to_hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut text = String::with_capacity(2 * ID_BYTES);
        for byte in self.0 {
            text.push(char::from(HEX[usize::from(byte >> 4)]));
            text.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        text
    }

    /// `None` unless `text` is 64 lower-case hexadecimal characters.
    pub(crate) fn from_hex(text: &str) -> Option<Self> {
        let text = text.as_bytes();
        if text.len() != 2 * ID_BYTES {
            return None;
        }
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        };
        let mut id = [0u8; ID_BYTES];
        for (byte, [high, low]) in id.iter_mut().zip(text.as_chunks::<2>().0) {
            *byte = digit(*high)? << 4 | digit(*low)?;
        }
        Some(Self(id))
    }
}

impl std::fmt::Debug for NativeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// One distinct original of a scope with what locates and counts it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Original {
    pub id: OriginalId,
    /// The first occurrence.
    pub first: Locator,
    /// How often the text occurred in the scope, the first occurrence included.
    pub occurrences: u64,
    pub native_id: NativeId,
}

/// What a count of [`ScopeCompleteness`] counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    Records,
    Entries,
}

/// How many records or entries were left out for one reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CountReason {
    pub unit: Unit,
    pub count: u64,
    pub reason: String,
}

/// What a scope covers of its source. A record or entry that was left out has no
/// original and so no record ID; it is counted here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScopeCompleteness {
    /// Nothing selected was skipped, unsupported or left unread.
    pub complete: bool,
    /// Records read from the source, the skipped ones included.
    pub records_scanned: u64,
    /// Records accepted: the occurrences of the originals.
    pub occurrences: u64,
    /// Distinct originals.
    pub originals: u64,
    /// Records that could not be read.
    pub skipped_records: u64,
    /// Archive entries read to their end.
    pub entries_read: u64,
    /// Archive entries that are damaged or were not reached.
    pub entries_incomplete: u64,
    /// Archive entries the reader cannot read: not text, encrypted, another
    /// compression method, a nested archive, a link.
    pub entries_unsupported: u64,
    /// Archive entries outside the entry filter, and directories.
    pub entries_excluded: u64,
    /// The counts of what was left out, by unit and reason. At most 16 reasons for
    /// records are kept apart; the rest are counted together.
    pub reasons: Vec<CountReason>,
}

/// What a scope handle binds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScopeBinding {
    pub version: u32,
    /// The source identity, its revision, the entry filter and the reader
    /// interpretation, as the existing artifacts record them.
    pub source: MarkedSource,
    /// The path spelling policy query IDs are derived from; `None` for a store.
    pub policy_id_source: Option<String>,
    /// A digest over every original's position, identity and count.
    pub relation: String,
    pub originals: u64,
    pub occurrences: u64,
}

/// An immutable scope: see the module description.
pub struct Scope {
    handle: String,
    binding: ScopeBinding,
    kind: SourceKind,
    completeness: ScopeCompleteness,
    dedup_on_disk: bool,
    archive_readers: Option<usize>,
    retained: u64,
    index: Index,
}

impl std::fmt::Debug for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scope")
            .field("handle", &self.handle)
            .field("binding", &self.binding)
            .field("completeness", &self.completeness)
            .field("index_on_disk", &self.index.on_disk())
            .finish()
    }
}

fn frame_failed(error: PolarsError) -> L2dError {
    L2dError::Runtime {
        detail: format!("reading a batch for a triager scope failed: {error}"),
    }
}

fn refused(reason: impl Into<String>) -> L2dError {
    L2dError::Refused {
        reason: reason.into(),
    }
}

fn native_id(source: &str, text: &str) -> Result<[u8; ID_BYTES], L2dError> {
    NativeId::from_hex(&l2d_store::log_id(source, text))
        .map(|id| id.0)
        .ok_or_else(|| L2dError::Runtime {
            detail: "a log ID is not 64 hexadecimal characters".into(),
        })
}

/// Writes the originals of one batch; their counts too when the batch states the
/// final ones.
fn spool_logs(
    spool: &mut Spool,
    logs: &LogsFrame,
    id_source: &str,
    counts: bool,
) -> Result<(), L2dError> {
    let frame = logs.frame();
    let texts = frame
        .column(LogsFrame::TEXT)
        .and_then(Column::str)
        .map_err(frame_failed)?;
    let positions = frame
        .column(LogsFrame::FIRST_POSITION)
        .and_then(Column::u64)
        .map_err(frame_failed)?;
    let occurrences = frame
        .column(LogsFrame::OCCURRENCES)
        .and_then(Column::u32)
        .map_err(frame_failed)?;
    for ((text, position), count) in texts.iter().zip(positions.iter()).zip(occurrences.iter()) {
        let (Some(text), Some(position), Some(count)) = (text, position, count) else {
            return Err(L2dError::Runtime {
                detail: "a batch for a triager scope holds a null".into(),
            });
        };
        spool.push(position, &native_id(id_source, text)?)?;
        if counts {
            spool.count(count)?;
        }
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<String, L2dError> {
    let failed = |error: std::io::Error| L2dError::Runtime {
        detail: format!("reading {} again failed: {error}", path.display()),
    };
    let mut file = File::open(path).map_err(failed)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(failed)?;
        if read == 0 {
            return Ok(format!("{:x}", hash.finalize()));
        }
        hash.update(&buffer[..read]);
    }
}

/// The final resource record is written also when the operation failed; a failure to
/// write it is reported only when the operation itself succeeded.
fn finish<T>(guard: ResourceGuard, result: Result<T, L2dError>) -> Result<T, L2dError> {
    let recorded = guard.finish();
    let value = result?;
    recorded?;
    Ok(value)
}

/// What the summary of a file read takes in memory: its skipped records with the room
/// their list and their reasons have, its path and its hash.
pub fn input_summary_bytes(summary: &InputSummary) -> u64 {
    let skipped = summary.skipped.capacity() * std::mem::size_of::<l2d_core::SkippedRecord>()
        + summary
            .skipped
            .iter()
            .map(|record| record.reason.capacity())
            .sum::<usize>();
    (std::mem::size_of::<InputSummary>()
        + summary.source_path.capacity()
        + summary.content_sha256.capacity()
        + skipped) as u64
}

/// What the summary of an archive read takes in memory: [`input_summary_bytes`] and
/// its entry reports with their paths and reasons.
pub fn archive_summary_bytes(summary: &ArchiveSummary) -> u64 {
    let entries = summary.entries.capacity() * std::mem::size_of::<l2d_core::EntryReport>()
        + summary
            .entries
            .iter()
            .map(|entry| {
                entry.path.capacity()
                    + match &entry.skipped {
                        Some(EntrySkip::Damaged(reason)) => reason.capacity(),
                        _ => 0,
                    }
            })
            .sum::<usize>();
    input_summary_bytes(&summary.input) + entries as u64
}

/// Counts what a read left out, by unit and reason.
fn completeness(
    index: &Index,
    summary: &InputSummary,
    entries: &[l2d_core::EntryReport],
) -> Result<ScopeCompleteness, L2dError> {
    let skipped_records = summary.skipped.len() as u64;
    if summary.distinct_texts != index.rows()
        || index.occurrences().saturating_add(skipped_records) != summary.records
    {
        return Err(L2dError::Runtime {
            detail: format!(
                "the scope's relation holds {} originals with {} occurrences, and the reader \
                 reports {} distinct texts in {} records of which {skipped_records} were skipped",
                index.rows(),
                index.occurrences(),
                summary.distinct_texts,
                summary.records
            ),
        });
    }
    let mut reasons: BTreeMap<(Unit, &str), u64> = BTreeMap::new();
    let mut kept = 0;
    for skipped in &summary.skipped {
        let known = reasons.contains_key(&(Unit::Records, skipped.reason.as_str()));
        let reason = if known || kept < MAX_REASONS {
            kept += usize::from(!known);
            skipped.reason.as_str()
        } else {
            OTHER_REASONS
        };
        *reasons.entry((Unit::Records, reason)).or_default() += 1;
    }
    let (mut read, mut incomplete, mut unsupported, mut excluded) = (0u64, 0u64, 0u64, 0u64);
    let mut methods: BTreeMap<u16, u64> = BTreeMap::new();
    for entry in entries {
        let reason = match &entry.skipped {
            None => {
                read += 1;
                continue;
            }
            Some(EntrySkip::NoPatternMatch) => {
                excluded += 1;
                "matches no entry pattern"
            }
            Some(EntrySkip::Directory) => {
                excluded += 1;
                "directory"
            }
            Some(EntrySkip::Damaged(_)) => {
                incomplete += 1;
                "damaged"
            }
            Some(EntrySkip::NotReached) => {
                incomplete += 1;
                "not reached"
            }
            Some(EntrySkip::UnsupportedMethod(method)) => {
                unsupported += 1;
                *methods.entry(*method).or_default() += 1;
                continue;
            }
            Some(EntrySkip::NotText) => {
                unsupported += 1;
                "not text"
            }
            Some(EntrySkip::Encrypted) => {
                unsupported += 1;
                "encrypted"
            }
            Some(EntrySkip::NestedArchive) => {
                unsupported += 1;
                "nested archive"
            }
            Some(EntrySkip::Link) => {
                unsupported += 1;
                "symbolic link"
            }
        };
        *reasons.entry((Unit::Entries, reason)).or_default() += 1;
    }
    let mut reasons: Vec<CountReason> = reasons
        .into_iter()
        .map(|((unit, reason), count)| CountReason {
            unit,
            count,
            reason: reason.to_string(),
        })
        .collect();
    reasons.extend(methods.into_iter().map(|(method, count)| CountReason {
        unit: Unit::Entries,
        count,
        reason: EntrySkip::UnsupportedMethod(method).to_string(),
    }));
    Ok(ScopeCompleteness {
        complete: skipped_records == 0 && incomplete == 0 && unsupported == 0,
        records_scanned: summary.records,
        occurrences: index.occurrences(),
        originals: index.rows(),
        skipped_records,
        entries_read: read,
        entries_incomplete: incomplete,
        entries_unsupported: unsupported,
        entries_excluded: excluded,
        reasons,
    })
}

impl Scope {
    fn new(
        kind: SourceKind,
        source: MarkedSource,
        policy_id_source: Option<String>,
        completeness: ScopeCompleteness,
        index: Index,
    ) -> Result<Self, L2dError> {
        let binding = ScopeBinding {
            version: BINDING_VERSION,
            source,
            policy_id_source,
            relation: index.digest().to_string(),
            originals: index.rows(),
            occurrences: index.occurrences(),
        };
        let failed = |error: serde_json::Error| L2dError::Runtime {
            detail: format!("describing a triager scope failed: {error}"),
        };
        let bytes = serde_json::to_vec(&binding).map_err(failed)?;
        let described = serde_json::to_vec(&completeness).map_err(failed)?;
        // The strings of the binding and of the reasons, with the room strings and
        // lists may have beyond their content.
        let retained = 4 * (bytes.len() + described.len()) as u64 + 1024;
        Ok(Self {
            handle: format!("scope-{}", content_sha256(&bytes)),
            binding,
            kind,
            completeness,
            dedup_on_disk: false,
            archive_readers: None,
            retained,
            index,
        })
    }

    /// Opens a scope over the file at `path` as `adapter` reads it.
    ///
    /// The file is read as every plain-file read is: its deduplicated rows are in
    /// memory as a whole while the relation is written, counted in `budget`, and let
    /// go before the relation is placed. The interpretation bound is the digest of
    /// those rows, so other column choices are another scope.
    ///
    /// Also returns the reader's summary with every skipped record. It is counted in
    /// `budget` while the scope is opened, so the relation is placed in the memory
    /// that is left beside it. From the return on it is the caller's to hold, beside
    /// [`Scope::held_bytes`]: [`input_summary_bytes`] says what it takes.
    ///
    /// [`L2dError::Refused`] when the read, the fixed share of the scope or the disk
    /// space does not fit, with the figures; nothing is left in the work directory.
    pub fn open_file(
        path: &Path,
        adapter: &impl InputAdapter,
        settings: &ScopeSettings,
        budget: &mut MemoryBudget,
    ) -> Result<(Self, InputSummary), L2dError> {
        let guard =
            ResourceGuard::tracked("triager scope over a file", busy_choice(), budget.usage())?;
        let before = budget.held_bytes();
        let result = (|| {
            let (logs, summary, _) = adapter.read_with(path, budget)?;
            let rows = logs.frame().estimated_size() as u64;
            // The summary lives on beside the scope: it stays counted from here on,
            // also while the relation is placed.
            let beside = SCOPE_FIXED_BYTES.saturating_add(input_summary_bytes(&summary));
            budget.hold(before.saturating_add(rows));
            budget.require_whole(
                beside,
                "opening a triager scope over a file, with the read's summary of skipped records,",
            )?;
            budget.hold(before.saturating_add(rows).saturating_add(beside));
            let interpretation = InputInterpretation::file(&logs, budget)?;
            let id_source = summary.source_path.to_string_lossy().into_owned();
            let work = Work::create(&settings.work_dir, settings.free_disk_bytes)?;
            let mut spool = Spool::create(work)?;
            spool_logs(&mut spool, &logs, &id_source, true)?;
            drop(logs);
            budget.hold(before.saturating_add(beside));
            let index = spool.finish(settings.index, budget)?;
            let completeness = completeness(&index, &summary, &[])?;
            let source = MarkedSource::Input(MarkedInput {
                path: MarkedInput::source_location(path)?,
                content_sha256: summary.content_sha256.clone(),
                archive_entries: None,
                interpretation: Some(interpretation),
            });
            let scope = Self::new(
                SourceKind::File,
                source,
                Some(id_source),
                completeness,
                index,
            )?;
            Ok((scope, summary))
        })();
        budget.hold(before);
        finish(guard, result)
    }

    /// Opens a scope over the entries of the archive at `path` that `zip` selects.
    ///
    /// The archive is streamed with the existing reader in its batches; the scope
    /// holds only its fixed share beside it, so no batch is kept. Deduplication goes
    /// on on disk in the work directory when its set no longer fits the memory, and
    /// the exact occurrence counts over the whole archive are read from there. The
    /// archive is hashed again afterwards: one that changed meanwhile is refused.
    ///
    /// Also returns the reader's summary with every entry and skipped record. It is
    /// counted in `budget` from the end of the read on, so the relation is placed in
    /// the memory that is left beside it. From the return on it is the caller's to
    /// hold, beside [`Scope::held_bytes`]: [`archive_summary_bytes`] says what it
    /// takes.
    ///
    /// [`L2dError::Refused`] as the archive read, and when the fixed share or the disk
    /// space does not fit, with the figures; nothing is left in the work directory.
    pub fn open_archive(
        path: &Path,
        zip: &ZipInputSettings,
        settings: &ScopeSettings,
        budget: &mut MemoryBudget,
    ) -> Result<(Self, ArchiveSummary), L2dError> {
        Self::open_archive_watched(path, zip, settings, budget, &mut |_| Ok(()))
    }

    /// [`Scope::open_archive`] with `watch` called after every batch with the rows
    /// read so far: for a caller that reports progress or ends the read early. An
    /// error of `watch` ends the open as any other error does, with nothing left in
    /// the work directory.
    pub fn open_archive_watched(
        path: &Path,
        zip: &ZipInputSettings,
        settings: &ScopeSettings,
        budget: &mut MemoryBudget,
        watch: &mut dyn FnMut(u64) -> Result<(), L2dError>,
    ) -> Result<(Self, ArchiveSummary), L2dError> {
        let guard = ResourceGuard::tracked(
            "triager scope over an archive",
            busy_choice(),
            budget.usage(),
        )?;
        let before = budget.held_bytes();
        let result = (|| {
            budget.require_whole(SCOPE_FIXED_BYTES, "opening a triager scope over an archive")?;
            let held = before.saturating_add(SCOPE_FIXED_BYTES);
            budget.hold(held);
            // Before the archive, so that the archive's own directory is removed first.
            let work = Work::create(&settings.work_dir, settings.free_disk_bytes)?;
            let mut spool = Spool::create(work)?;
            let mut input = ZipInput::open(path, zip, budget)?;
            // The share is constant, so no batch is sized without it.
            input.hold_beside(held);
            input.spill_into(&spool.work().join("dedup"), settings.dedup_set_texts)?;
            input.check_disk_at_freeze(FreezeDiskCheck {
                beside: Vec::new(),
                free_bytes: settings.free_disk_bytes,
            });
            let id_source = input.path().to_string_lossy().into_owned();
            let mut rows = 0u64;
            for batch in input.batches(budget) {
                let batch = batch?;
                spool_logs(&mut spool, &batch, &id_source, false)?;
                rows += batch.len() as u64;
                watch(rows)?;
            }
            for chunk in input.occurrence_counts() {
                for count in chunk? {
                    spool.count(count)?;
                }
            }
            let dedup_on_disk = input.disk_dedup().is_some();
            let readers = input.entries_at_once();
            let summary = input.finish();
            // The reader's own count of the summary ended with the reader, and the
            // summary lives on beside the scope: it stays counted from here on, also
            // while the relation is placed.
            budget.hold(held);
            let kept = archive_summary_bytes(&summary);
            budget.require_whole(
                kept,
                "keeping the summary of an archive read, its skipped records and entries, \
                 beside a triager scope",
            )?;
            budget.hold(held.saturating_add(kept));
            if hash_file(path)? != summary.input.content_sha256 {
                return Err(refused(format!(
                    "the archive {} changed while the scope was opened; open it again",
                    path.display()
                )));
            }
            if summary.input.source_path.to_string_lossy() != id_source {
                return Err(L2dError::Runtime {
                    detail: "the archive reports another path than it was opened at".into(),
                });
            }
            let index = spool.finish(settings.index, budget)?;
            let completeness = completeness(&index, &summary.input, &summary.entries)?;
            let source = MarkedSource::Input(MarkedInput {
                path: MarkedInput::source_location(path)?,
                content_sha256: summary.input.content_sha256.clone(),
                archive_entries: Some(zip.entries.clone()),
                interpretation: Some(InputInterpretation::archive(zip)),
            });
            let mut scope = Self::new(
                SourceKind::Archive,
                source,
                Some(id_source),
                completeness,
                index,
            )?;
            scope.dedup_on_disk = dedup_on_disk;
            scope.archive_readers = Some(readers);
            Ok((scope, summary))
        })();
        budget.hold(before);
        finish(guard, result)
    }

    /// Opens a scope over the logs of `store` as they are now.
    ///
    /// The store is read in its batches of row, ID and occurrence count; nothing is
    /// kept per row. The read generation is compared before and after: a store another
    /// connection wrote to meanwhile is refused, never mixed. The stored labels are
    /// not read: they are no substitute for a saved mark set.
    ///
    /// [`L2dError::Refused`] as the store read, and when the fixed share or the disk
    /// space does not fit, with the figures; nothing is left in the work directory.
    pub fn open_store(
        store: &Store,
        settings: &ScopeSettings,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        let guard =
            ResourceGuard::tracked("triager scope over a store", busy_choice(), budget.usage())?;
        let before = budget.held_bytes();
        let result = (|| {
            budget.require_whole(SCOPE_FIXED_BYTES, "opening a triager scope over a store")?;
            let held = before.saturating_add(SCOPE_FIXED_BYTES);
            budget.hold(held);
            let generation = store.read_generation()?;
            let content = store.content_sha256_with(budget)?;
            let work = Work::create(&settings.work_dir, settings.free_disk_bytes)?;
            let mut spool = Spool::create(work)?;
            // The fixed share is held in the budget, and nothing is taken per row.
            store.pattern_batches(0, 0, budget, &mut |frame| {
                spool_stored(&mut spool, frame)?;
                Ok(0)
            })?;
            budget.hold(held);
            if store.read_generation()? != generation {
                return Err(refused(format!(
                    "the store {} changed while the scope was opened; no scope over mixed \
                     generations is kept, open it again",
                    store.name()
                )));
            }
            let index = spool.finish(settings.index, budget)?;
            let completeness = ScopeCompleteness {
                complete: true,
                records_scanned: index.occurrences(),
                occurrences: index.occurrences(),
                originals: index.rows(),
                skipped_records: 0,
                entries_read: 0,
                entries_incomplete: 0,
                entries_unsupported: 0,
                entries_excluded: 0,
                reasons: Vec::new(),
            };
            let source = MarkedSource::Store(MarkedStore {
                name: store.name().to_string(),
                content_sha256: content,
            });
            Self::new(SourceKind::Store, source, None, completeness, index)
        })();
        budget.hold(before);
        finish(guard, result)
    }

    /// The immutable handle: the SHA-256 of [`Scope::binding`].
    pub fn handle(&self) -> &str {
        &self.handle
    }

    pub fn binding(&self) -> &ScopeBinding {
        &self.binding
    }

    pub fn source(&self) -> &MarkedSource {
        &self.binding.source
    }

    pub fn kind(&self) -> SourceKind {
        self.kind
    }

    pub fn completeness(&self) -> &ScopeCompleteness {
        &self.completeness
    }

    /// The distinct originals of the scope.
    pub fn originals(&self) -> u64 {
        self.index.rows()
    }

    /// The accepted records of the scope: the occurrences of its originals.
    pub fn occurrences(&self) -> u64 {
        self.index.occurrences()
    }

    /// Whether the relation is read from the work directory.
    pub fn index_on_disk(&self) -> bool {
        self.index.on_disk()
    }

    /// Whether the archive's deduplication went on on disk while the scope was opened.
    /// Part of what the open cost, not of what the scope binds.
    pub fn dedup_on_disk(&self) -> bool {
        self.dedup_on_disk
    }

    /// How many entries of the archive were read at the same time while the scope was
    /// opened; `None` for a file or a store. Part of what the open cost, not of what
    /// the scope binds.
    pub fn archive_readers(&self) -> Option<usize> {
        self.archive_readers
    }

    /// What the scope keeps in memory. Its owner holds this in the budget for as long
    /// as the scope lives.
    pub fn held_bytes(&self) -> u64 {
        self.index.held_bytes() + self.retained
    }

    /// What the scope keeps in its work directory.
    pub fn disk_bytes(&self) -> u64 {
        self.index.disk_bytes()
    }

    /// The original a record ID of this scope names; `None` when it names none.
    pub fn original(&self, id: OriginalId) -> Result<Option<Original>, L2dError> {
        match self.index.find_position(id.0)? {
            Some(row) => self.original_at(row).map(Some),
            None => Ok(None),
        }
    }

    /// The row of the relation that holds the original first read at `position`.
    ///
    /// Rows are in the order the source handed its originals out. That order is fixed
    /// for one open scope and differs between batch sizes and spill, so a row is never
    /// part of a result: results name originals by their record ID.
    pub(crate) fn row_of(&self, position: u64) -> Result<Option<u64>, L2dError> {
        self.index.find_position(position)
    }

    /// The original in `row` of the relation; see [`Scope::row_of`].
    pub(crate) fn row(&self, row: u64) -> Result<Original, L2dError> {
        self.original_at(row)
    }

    fn original_at(&self, row: u64) -> Result<Original, L2dError> {
        let entry = self.index.entry(row)?;
        let first = match self.kind {
            SourceKind::File => Locator::FileRecord {
                position: entry.position,
            },
            SourceKind::Archive => {
                let at = ArchivePosition::unpack(entry.position);
                Locator::ArchiveLine {
                    entry: at.entry,
                    line: at.line,
                }
            }
            SourceKind::Store => Locator::StoreRow {
                seq: i64::try_from(entry.position).map_err(|_| L2dError::Runtime {
                    detail: "a stored row lies beyond the rows of a store".into(),
                })?,
            },
        };
        Ok(Original {
            id: OriginalId(entry.position),
            first,
            occurrences: u64::from(entry.count),
            native_id: NativeId(entry.id),
        })
    }
}

/// Writes the stored logs of one batch: row, ID and occurrence count.
fn spool_stored(spool: &mut Spool, frame: &DataFrame) -> Result<(), L2dError> {
    let rows = frame
        .column(l2d_store::SEQ)
        .and_then(Column::i64)
        .map_err(frame_failed)?;
    let ids = frame
        .column(l2d_store::query::ID)
        .and_then(Column::str)
        .map_err(frame_failed)?;
    let occurrences = frame
        .column(l2d_store::OCCURRENCES)
        .and_then(Column::u32)
        .map_err(frame_failed)?;
    for ((row, id), count) in rows.iter().zip(ids.iter()).zip(occurrences.iter()) {
        let stored = row
            .and_then(|row| u64::try_from(row).ok())
            .zip(id.and_then(NativeId::from_hex))
            .zip(count);
        let Some(((row, id), count)) = stored else {
            return Err(L2dError::Runtime {
                detail: "a stored log has no usable row, ID or occurrence count".into(),
            });
        };
        spool.push(row, &id.0)?;
        spool.count(count)?;
    }
    Ok(())
}
