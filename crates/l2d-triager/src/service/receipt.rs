//! Saved receipts (`triager_save`, and `triager_open` with `receipt`; requirements
//! 3.3, 4.1, 4.2 and 6.3).
//!
//! A receipt retains a candidate view or a final work set so that a second client, in
//! another server process, reopens it without the source: the source identity, the
//! saved set and every set it was made from (their producers, revisions, counts and
//! coverage), and for each retained original its record handle, native identity,
//! occurrences, its state in every one of those sets, the values a set keeps beside
//! a state (scores, Venn routes) and its slot in the work set. A candidate retains
//! the members of the view; a final work set every original eligible for its
//! allocation. A receipt holds no source text and no host path (except, under the
//! private-local profile, the path of a source opened by its path), and states what it
//! contains. Reopening reads the receipt only: no archive is inflated, nothing is
//! scored, encoded or issued, and the reopened sets can be composed, allocated,
//! inspected and saved again.
//!
//! # Format
//!
//! JSON lines, in this order: a preamble (`triager_receipt`, the counts of what
//! follows and the bytes of the header), the header, the recipe (the tool calls that
//! make the saved set again from the source, as `l2d mcp --requests` takes them), the
//! entry names of the retained originals, one line per retained original by record ID
//! ascending, and a trailer with the digests. The receipt's handle is the SHA-256 of
//! every byte before the trailer; `content_sha256` leaves out the preamble and the
//! lineage, so a replay of the recipe on the same input has the content of the receipt
//! it replays.
//!
//! # Atomic publication
//!
//! A receipt is written to a staging file in the receipt directory, synced, and
//! published by a hard link to its final name, which never replaces an existing file:
//! the name is the content's digest, so a receipt that exists is the same receipt. A
//! failure or a crash before the link leaves no receipt under the name; a staging
//! file left by a crash is never read.
//!
//! # Memory
//!
//! Saving holds a block of every retained set, the header, the buffer the file is
//! written through and a bit per archive entry. Reopening holds the header and its
//! parse, the buffer the file is read through, the relation (as an opened source's)
//! and the tables of every set, in memory when the budget admits them and in the work
//! directory otherwise.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use l2d_core::zip_input::EntryReport;
use l2d_core::{
    ArchiveSummary, InputSummary, L2dError, MarkedInput, MarkedSource, ZipInputSettings,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::allocate::{self, Allocation, OrderKey, Pool, PoolOutcome, Slot};
use super::membership::{Expr, State, States, Tally, code_in, put_code};
use super::ops::{self, ARTIFACT_FIXED_BYTES, Commit, Cost, Done, TALLY_BYTES, Worker, registry};
use super::state::{Judged, Origin, Population, Producer, Retained, Run, Set, SetKind, Snapshot};
use super::tables::{BLOCK_ROWS, Scores, require_disk, score_in};
use super::wire::{self, Fault, VERSION};
use super::{DisclosureProfile, HostConfig};
use crate::scope::{NativeId, RetainedPush, RowBlock, Scope, ScopeSettings, SourceKind};

/// The format a receipt states of itself, and its version.
pub(crate) const FORMAT: &str = "triager-receipt";
pub(crate) const FORMAT_VERSION: u64 = 1;
/// The most sets one receipt retains: the saved set and what it was made from.
pub(crate) const MAX_SETS: usize = 16;
/// The most originals one receipt retains.
pub(crate) const MAX_ORIGINALS: u64 = 1_000_000;
/// The most bytes of the header line, of the preamble, of an entry line and of a row
/// line, their line ends not counted.
pub(crate) const MAX_HEADER_BYTES: usize = 128 * 1024;
const MAX_PREAMBLE_BYTES: usize = 1024;
const MAX_ENTRY_LINE_BYTES: usize = 2048;
const MAX_ROW_BYTES: usize = 512;
/// The most judged records a receipt retains of one issued-batch set (the most a run
/// issues: 64 batches of 100), and the most bytes of one judged line.
pub(crate) const MAX_JUDGED: u64 = 64 * 100;
const MAX_JUDGED_LINE_BYTES: usize = 8 * 1024;
/// A recipe step is a request line `l2d mcp --requests` takes.
const MAX_STEP_BYTES: usize = crate::mcp::MAX_REQUEST_BYTES;
/// The most recipe bytes a reopen result lists; a longer recipe is replayed with
/// `l2d mcp --replay`.
const RECIPE_SHOWN_BYTES: usize = 8 * 1024;
/// The buffer a receipt is written and read through.
const IO_BYTES: u64 = 64 * 1024;
/// What a parsed JSON value takes at most for each byte of its text.
const PARSE_FACTOR: u64 = 16;
/// The prefix of a receipt handle and the extension of its file.
const PREFIX: &str = "rcpt-";
const EXTENSION: &str = "receipt";

/// What a receipt contains and what it leaves out, as it states of itself.
const CONTAINS: [&str; 10] = [
    "model judgments of issued batches: outcome, reason, caller-reported model and prompt \
     revision, batch revisions and input digests, the submissions covered",
    "record handles and their first locators",
    "native identities (SHA-256 of the source path and the text, as the policy keys them)",
    "archive entry names of the retained originals",
    "occurrence counts",
    "membership states of the saved set and every set it was made from",
    "values kept beside a state: scores, Venn route codes, distances",
    "allocation slots, pool counts and deferral counts",
    "producer, expression and budget descriptions, revisions and counts",
    "the recipe of tool calls that makes the saved set again",
];
const EXCLUDES: [&str; 4] = [
    "source text and excerpts",
    "host paths (except the path of a source opened by path under private-local)",
    "the prompt text of model judgments (its caller-reported revision is kept)",
    "timings and other cost figures",
];

/// Whether `text` is a receipt handle: the prefix and 32 lower-case hex digits.
pub(crate) fn is_receipt(text: &str) -> bool {
    text.strip_prefix(PREFIX).is_some_and(|digits| {
        digits.len() == 32
            && digits
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

/// The file of the receipt `receipt` in the receipt directory of `config`.
pub(crate) fn receipt_path(config: &HostConfig, receipt: &str) -> Result<PathBuf, Fault> {
    let root = config.receipts.as_ref().ok_or_else(|| {
        Fault::new(
            "receipts_not_configured",
            "the host configured no receipt directory; receipts are neither saved nor reopened",
        )
    })?;
    if !is_receipt(receipt) {
        return Err(Fault::invalid(
            "a receipt handle is `rcpt-` and 32 lower-case hexadecimal digits, as \
             `triager_save` returned it",
        ));
    }
    Ok(root.join(format!("{receipt}.{EXTENSION}")))
}

/// The refusal for a receipt the directory does not hold.
fn unknown_receipt(receipt: &str) -> Fault {
    Fault::new(
        "unknown_receipt",
        format!("the receipt directory holds no receipt `{receipt}`"),
    )
}

fn damaged(what: &str) -> L2dError {
    L2dError::Refused {
        reason: format!("the receipt is damaged: {what}; nothing was reopened"),
    }
}

fn io(doing: &str, error: &std::io::Error) -> L2dError {
    // A message never names where the host keeps its receipts.
    L2dError::Runtime {
        detail: format!("{doing} a receipt failed ({:?})", error.kind()),
    }
}

// --- What a receipt retains ------------------------------------------------------------

/// The saved set, the sets the request names beside it and every set they were made
/// from, each after what it was made from: its parent population and its inputs.
/// Refused when they are more than a receipt retains.
pub(crate) fn lineage(set: &Arc<Set>, beside: &[Arc<Set>]) -> Result<Vec<Arc<Set>>, Fault> {
    fn visit(set: &Arc<Set>, found: &mut Vec<Arc<Set>>, depth: usize) -> Result<(), Fault> {
        if found.iter().any(|known| Arc::ptr_eq(known, set)) {
            return Ok(());
        }
        if depth > 32 {
            return Err(Fault::refused(
                "the saved set was made from too deep a lineage",
            ));
        }
        if let Some(parent) = &set.parent {
            visit(parent, found, depth + 1)?;
        }
        for input in &set.inputs {
            visit(input, found, depth + 1)?;
        }
        if !found.iter().any(|known| Arc::ptr_eq(known, set)) {
            found.push(set.clone());
        }
        Ok(())
    }
    let mut found = Vec::new();
    for named in beside {
        visit(named, &mut found, 0)?;
    }
    visit(set, &mut found, 0)?;
    if found.len() > MAX_SETS {
        return Err(Fault::refused(format!(
            "the saved set was made from {} sets, and a receipt retains at most {MAX_SETS}; \
             compose a view of fewer operands and save it",
            found.len()
        )));
    }
    if let Some(other) = found
        .iter()
        .find(|known| known.kind == SetKind::Final && !Arc::ptr_eq(known, set))
    {
        return Err(Fault::refused(format!(
            "a receipt retains one allocation, that of the saved set, and `{}` is another \
             final work set it was made from; save that final work set itself",
            other.handle
        )));
    }
    Ok(found)
}

/// The tool calls that make `lineage` again from the source, with names bound in
/// place of handles, or why that cannot be done.
fn recipe(
    snapshot: &Snapshot,
    lineage: &[Arc<Set>],
    saved_at: usize,
    named: &[(String, String)],
    private_local: bool,
) -> Result<Vec<Value>, String> {
    let source: Value = serde_json::from_str(&snapshot.result)
        .ok()
        .and_then(|result: Value| result.get("source").cloned())
        .ok_or("the snapshot states no source")?;
    let mut opened = Map::new();
    if let Some(name) = source.get("ref").and_then(Value::as_str) {
        opened.insert("ref".into(), json!(name));
    } else if source.get("path").is_some() && private_local && snapshot.retained.is_none() {
        opened.insert("path".into(), json!(snapshot.path.to_string_lossy()));
    } else {
        return Err(
            "the source was opened by its path, and a recipe names a host path only \
                    under the private-local profile"
                .into(),
        );
    }
    if let Some(entries) = source.get("entries")
        && entries
            .as_array()
            .is_some_and(|entries| !entries.is_empty())
    {
        opened.insert("entries".into(), entries.clone());
    }
    if let Some(bytes) = source.get("max_record_bytes")
        && *bytes != json!(l2d_core::zip_input::MAX_RECORD_BYTES)
        && *bytes != json!(ZipInputSettings::default().max_record_bytes)
    {
        opened.insert("max_record_bytes".into(), bytes.clone());
    }
    let mut steps = vec![json!({
        "tool": "triager_open",
        "args": {"v": 1, "source": opened},
        "bind": {"S": "snapshot"},
    })];
    let name_of = |set: &Arc<Set>| -> String {
        let at = lineage
            .iter()
            .position(|known| Arc::ptr_eq(known, set))
            .expect("every set of the lineage is in it");
        format!("X{}", at + 1)
    };
    let scope_of = |set: &Set| -> String {
        set.parent.as_ref().map_or_else(
            || "$S".to_string(),
            |parent| format!("${}", name_of(parent)),
        )
    };
    for (at, set) in lineage.iter().enumerate() {
        let bind = format!("X{}", at + 1);
        let described = set.producer.describe();
        let origin = Origin::of(&set.producer);
        let step = match (set.kind, &origin) {
            (SetKind::Selection, Origin::Score { .. }) => {
                let mut producer = json!({"kind": "score", "model": described["model"]});
                for field in ["top_k", "ties", "threshold"] {
                    if let Some(value) = described.get(field) {
                        producer[field] = value.clone();
                    }
                }
                json!({"tool": "triager_select",
                       "args": {"v": 1, "scope": scope_of(set), "producer": producer},
                       "bind": {bind: "set"}})
            }
            (SetKind::Selection, Origin::Filter) => {
                let query = match &set.producer {
                    Producer::Filter { query, .. } => json!(query),
                    _ if described["query_clipped"] == false => described["query"].clone(),
                    _ => {
                        return Err(format!(
                            "the filter of `{}` is retained only as a clipped query and its hash",
                            set.handle
                        ));
                    }
                };
                json!({"tool": "triager_select",
                       "args": {"v": 1, "scope": scope_of(set), "producer": {
                           "kind": "filter", "syntax": described["syntax"], "query": query,
                           "case_insensitive": described["case_insensitive"]}},
                       "bind": {bind: "set"}})
            }
            (SetKind::Selection, Origin::Membership) => json!({
                "tool": "triager_select",
                "args": {"v": 1, "scope": scope_of(set),
                         "producer": {"kind": "membership", "ref": described["ref"]}},
                "bind": {bind: "set"}}),
            (SetKind::Selection, Origin::Venn { .. }) => json!({
                "tool": "triager_select",
                "args": {"v": 1, "scope": scope_of(set), "producer": {
                    "kind": "recommend", "method": "venn", "policy": described["policy"],
                    "membership": described["membership"]}},
                "bind": {bind: "set"}}),
            (SetKind::Selection, other) => {
                return Err(format!(
                    "`{}` was made by the producer `{}`, whose result depends on a refinement \
                     or a model client's judgments; replaying it asks for that work again",
                    set.handle,
                    other.name()
                ));
            }
            (SetKind::Final, _) => {
                // The operands under the names the request gave them, so that the
                // replay states the same allocation; a score set that was no operand
                // was found by `score_desc` itself and is found again.
                let allocation = set
                    .allocation
                    .as_ref()
                    .ok_or("a final work set without its allocation")?;
                let names: Vec<(String, String)> = allocation
                    .operands
                    .iter()
                    .zip(&allocation.names)
                    .map(|(operand, name)| (operand.handle.clone(), name.clone()))
                    .collect();
                let sets: Map<String, Value> = allocation
                    .operands
                    .iter()
                    .zip(&allocation.names)
                    .map(|(operand, name)| (name.clone(), json!(format!("${}", name_of(operand)))))
                    .collect();
                let mut budget = rename_handles(&described["budget"], &names);
                for pool in budget["pools"].as_array_mut().into_iter().flatten() {
                    for key in pool["order"].as_array_mut().into_iter().flatten() {
                        // A name has no '-'; a handle that was not renamed has.
                        if let Some(handle) = key.get("score_desc").and_then(Value::as_str)
                            && handle.contains('-')
                        {
                            *key = json!("score_desc");
                        }
                    }
                }
                let mut args = json!({"v": 1, "mode": "allocate", "snapshot": "$S", "sets": sets,
                                      "expr": rename_handles(&described["expr"], &names),
                                      "budget": budget});
                if set.parent.is_some() {
                    args["scope"] = json!(scope_of(set));
                }
                json!({"tool": "triager_compose", "args": args, "bind": {bind: "set"}})
            }
            (SetKind::View, Origin::Occurrence) => {
                return Err(format!(
                    "`{}` is a field or time view: replaying it needs an occurrence projection \
                     of the source, which is another read of it; open the source with its \
                     `projections` and drill again",
                    set.handle
                ));
            }
            (SetKind::View, _) => {
                let names: Vec<(String, String)> = set
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(index, input)| (input.handle.clone(), format!("s{}", index + 1)))
                    .collect();
                let rename = |value: &Value| rename_handles(value, &names);
                let sets: Map<String, Value> = set
                    .inputs
                    .iter()
                    .zip(&names)
                    .map(|(input, (_, name))| (name.clone(), json!(format!("${}", name_of(input)))))
                    .collect();
                let mut args = json!({"v": 1, "snapshot": "$S", "sets": sets});
                if set.parent.is_some() {
                    args["scope"] = json!(scope_of(set));
                }
                args["mode"] = json!("view");
                args["expr"] = rename(&described["expr"]);
                json!({"tool": "triager_compose", "args": args, "bind": {bind: "view"}})
            }
        };
        if step.to_string().len() > MAX_STEP_BYTES {
            return Err(format!(
                "the call that makes `{}` is longer than a request line",
                set.handle
            ));
        }
        steps.push(step);
    }
    let mut save = json!({"v": 1, "handle": format!("$X{}", saved_at + 1)});
    if !named.is_empty() {
        let beside: Map<String, Value> = named
            .iter()
            .map(|(name, handle)| {
                let at = lineage
                    .iter()
                    .position(|known| known.handle == *handle)
                    .expect("a named set is in the lineage");
                (name.clone(), json!(format!("$X{}", at + 1)))
            })
            .collect();
        save["sets"] = Value::Object(beside);
    }
    steps.push(json!({"tool": "triager_save", "args": save}));
    Ok(steps)
}

/// `value` with every string that is one of the handles of `names` replaced by its
/// name.
fn rename_handles(value: &Value, names: &[(String, String)]) -> Value {
    match value {
        Value::String(text) => names
            .iter()
            .find(|(handle, _)| handle == text)
            .map_or_else(|| value.clone(), |(_, name)| json!(name)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| rename_handles(item, names))
                .collect(),
        ),
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(key, item)| (key.clone(), rename_handles(item, names)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// What a header describes of one retained set.
fn describe_set(set: &Arc<Set>, lineage: &[Arc<Set>]) -> Value {
    let result: Value = serde_json::from_str(&set.result).unwrap_or_default();
    let mut described = json!({
        "handle": set.handle,
        "kind": set.kind.name(),
        "producer": set.producer.describe(),
        "origin": Origin::of(&set.producer).describe(),
        "revision": set.revision,
        "parent": set.parent.as_ref().map(|parent| parent.handle.clone()),
        "inputs": set.inputs.iter().map(|input| input.handle.clone()).collect::<Vec<_>>(),
        "values": set.scores.is_some(),
        "counts": set.tally.counts(),
        "occurrence_counts": set.tally.occurrence_counts(),
        "coverage": result.get("coverage").cloned().unwrap_or(Value::Null),
    });
    debug_assert!(
        set.inputs
            .iter()
            .chain(&set.parent)
            .all(|known| lineage.iter().any(|set| Arc::ptr_eq(set, known)))
    );
    if let Some(retained) = result.get("retained_from") {
        described["retained_from"] = retained.clone();
    }
    described
}

// --- Saving ----------------------------------------------------------------------------

pub(crate) struct SaveTask {
    pub set: Arc<Set>,
    /// Every set after what it was made from, the saved one among them.
    pub lineage: Vec<Arc<Set>>,
    /// The sets the request named beside the saved one, by name and handle.
    pub named: Vec<(String, String)>,
    pub parent_receipt: Option<String>,
}

/// A writer that hashes what it writes for the receipt's handle and its content.
struct Hashed<W: Write> {
    out: W,
    receipt: Sha256,
    bytes: u64,
}

impl<W: Write> Hashed<W> {
    fn line(&mut self, line: &[u8], also: &mut [&mut Sha256]) -> Result<(), L2dError> {
        let write = |out: &mut W| -> std::io::Result<()> {
            out.write_all(line)?;
            out.write_all(b"\n")
        };
        write(&mut self.out).map_err(|error| io("writing", &error))?;
        self.receipt.update(line);
        self.receipt.update(b"\n");
        for hash in also {
            hash.update(line);
            hash.update(b"\n");
        }
        self.bytes += line.len() as u64 + 1;
        Ok(())
    }
}

/// A staging file in the receipt directory, removed unless it was published.
struct Staged(PathBuf);

impl Drop for Staged {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

static STAGED: AtomicU64 = AtomicU64::new(0);

/// Writes a receipt with `write` into a staging file of `root` and publishes it
/// under the name the returned handle gives: atomically, never over another file. A
/// receipt of the same handle that exists is the same receipt and is left as it is.
/// Returns the handle and whether the file was published by this call.
pub(crate) fn publish(
    root: &Path,
    write: &mut dyn FnMut(&mut BufWriter<File>) -> Result<String, L2dError>,
) -> Result<(String, bool), L2dError> {
    fs::create_dir_all(root).map_err(|error| io("preparing the directory of", &error))?;
    let staged = Staged(root.join(format!(
        ".saving-{}-{}.tmp",
        std::process::id(),
        STAGED.fetch_add(1, Ordering::Relaxed)
    )));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged.0)
        .map_err(|error| io("staging", &error))?;
    let mut out = BufWriter::with_capacity(IO_BYTES as usize, file);
    let receipt = write(&mut out)?;
    out.flush().map_err(|error| io("writing", &error))?;
    out.get_ref()
        .sync_all()
        .map_err(|error| io("syncing", &error))?;
    drop(out);
    let path = root.join(format!("{receipt}.{EXTENSION}"));
    let published = match fs::hard_link(&staged.0, &path) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
        Err(error) => return Err(io("publishing", &error)),
    };
    drop(staged);
    Ok((receipt, published))
}

/// Hex of `bytes` into `out` without an allocation.
fn hex_into<'a>(bytes: &[u8], out: &'a mut [u8]) -> &'a str {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (at, byte) in bytes.iter().enumerate() {
        out[2 * at] = HEX[usize::from(byte >> 4)];
        out[2 * at + 1] = HEX[usize::from(byte & 0x0f)];
    }
    std::str::from_utf8(&out[..2 * bytes.len()]).expect("ASCII")
}

/// Writes the row line of one retained original into `line`, without an allocation
/// beyond what `line` holds: at most [`MAX_ROW_BYTES`].
fn row_line(
    line: &mut Vec<u8>,
    position: u64,
    native: &[u8; 32],
    occurrences: u64,
    codes: &[u8],
    values: &[u32],
    slot: Option<(usize, &Slot)>,
) -> Result<(), L2dError> {
    line.clear();
    let mut id = [0u8; 18];
    let mut native_text = [0u8; 64];
    let mut codes_text = [0u8; MAX_SETS];
    for (at, code) in codes.iter().enumerate() {
        codes_text[at] = b'0' + code;
    }
    let mut values_text = [0u8; MAX_SETS * 8];
    for (at, bits) in values.iter().enumerate() {
        hex_into(&bits.to_be_bytes(), &mut values_text[at * 8..at * 8 + 8]);
    }
    let failed = |error: std::io::Error| io("writing", &error);
    write!(
        line,
        "{{\"r\":[\"{}\",\"{}\",{},\"{}\",\"{}\",",
        allocate::record_id_into(position, &mut id),
        hex_into(native, &mut native_text),
        occurrences,
        std::str::from_utf8(&codes_text[..codes.len()]).expect("ASCII"),
        std::str::from_utf8(&values_text[..values.len() * 8]).expect("ASCII"),
    )
    .map_err(failed)?;
    match slot {
        Some((index, slot)) => write!(
            line,
            "[{},{},{},{}]]}}",
            index,
            slot.pool,
            slot.rank,
            u8::from(slot.outside_candidate)
        ),
        None => write!(line, "null]}}"),
    }
    .map_err(failed)?;
    if line.len() > MAX_ROW_BYTES {
        return Err(L2dError::Runtime {
            detail: "a row of a receipt is longer than a row line".into(),
        });
    }
    Ok(())
}

/// Where the judgments of an issued-batch set come from when it is saved: its live run,
/// or the judgments a receipt retained when it was reopened.
enum JudgedSource<'a> {
    Live {
        run: &'a Arc<Run>,
        upto: usize,
        batches: usize,
    },
    Kept(&'a Arc<Judged>),
}

impl JudgedSource<'_> {
    fn of(set: &Set) -> Option<JudgedSource<'_>> {
        match &set.producer {
            Producer::Issued {
                run, upto, batches, ..
            } => Some(JudgedSource::Live {
                run,
                upto: *upto,
                batches: *batches,
            }),
            Producer::Retained {
                judged: Some(judged),
                ..
            } => Some(JudgedSource::Kept(judged)),
            _ => None,
        }
    }

    /// The run as the receipt states it: name, caller-reported model and prompt
    /// revision, the batches covered with their revisions and input digests.
    fn run_block(&self) -> Value {
        match self {
            JudgedSource::Kept(judged) => judged.run.clone(),
            JudgedSource::Live { run, upto, batches } => {
                let state = run.state();
                json!({
                    "run": run.handle,
                    "name": run.name,
                    "model": state.model.as_ref().map(|(model, _)| model.clone()),
                    "prompt_revision": state.model.as_ref().map(|(_, prompt)| prompt.clone()),
                    "model_identity": "caller-reported",
                    "excerpts": run.excerpts,
                    "submissions": upto,
                    "batches": state.batches.iter().take(*batches).map(|batch| json!({
                        "batch": batch.handle,
                        "revision": batch.revision,
                        "input_sha256": batch.input_sha256,
                        "records": batch.positions.len(),
                    })).collect::<Vec<_>>(),
                })
            }
        }
    }

    /// What the set says of the record `position`, as JSON text; `None` for a record
    /// the run did not issue.
    fn detail(&self, position: u64, value: f32) -> Option<String> {
        match self {
            JudgedSource::Kept(judged) => judged.detail(position).map(str::to_string),
            JudgedSource::Live { run, upto, batches } => {
                let issued = {
                    let state = run.state();
                    state
                        .batches
                        .iter()
                        .take(*batches)
                        .any(|batch| batch.positions.binary_search(&position).is_ok())
                };
                issued.then(|| {
                    super::recommend::issued_detail(run, (*upto, *batches), position, value)
                        .to_string()
                })
            }
        }
    }
}

/// Whether row `at` of the current blocks is retained, by the saved set: a member of
/// a candidate view, or for a final work set an original its allocation was eligible
/// for or slotted.
fn retained_at(set: &Set, operand_codes: &[&[u8]], saved: &[u8], at: usize) -> bool {
    match &set.allocation {
        None => code_in(saved, at) == State::In as u8,
        Some(allocation) => {
            if code_in(saved, at) == State::In as u8 {
                return true;
            }
            let state = |index: usize| State::of(code_in(operand_codes[index], at));
            allocation.expr.eval(&state) == State::In
                || allocation
                    .pools
                    .iter()
                    .any(|pool| pool.from.eval(&state) == State::In)
        }
    }
}

/// Saves the set of `task` as a receipt in the host's receipt directory.
pub(crate) fn save(worker: &mut Worker<'_>, task: &SaveTask) -> Result<Done, Fault> {
    let started = Instant::now();
    let shared = worker.shared;
    let config = &shared.config;
    let root = config.receipts.clone().ok_or_else(|| {
        Fault::new(
            "receipts_not_configured",
            "the host configured no receipt directory",
        )
    })?;
    let set = &task.set;
    let snapshot = set.snapshot.clone();
    let rows = snapshot.scope.originals();
    let lineage = &task.lineage;
    let saved_at = lineage
        .iter()
        .position(|known| Arc::ptr_eq(known, set))
        .ok_or_else(|| Fault::runtime("the saved set is not in its own lineage"))?;
    let valued: Vec<usize> = (0..lineage.len())
        .filter(|at| lineage[*at].scores.is_some())
        .collect();
    // The operands of an allocation, by their place in the lineage.
    let operand_at: Vec<usize> = set.allocation.as_ref().map_or_else(Vec::new, |allocation| {
        allocation
            .operands
            .iter()
            .map(|operand| {
                lineage
                    .iter()
                    .position(|known| Arc::ptr_eq(known, operand))
                    .expect("an operand of the allocation is in the lineage")
            })
            .collect()
    });
    // The issued-batch sets, whose judgments the receipt keeps beside their states:
    // their place in the lineage, where the judgments come from and where their values
    // are among the blocks.
    let judged_sets: Vec<(usize, JudgedSource<'_>, Option<usize>)> = lineage
        .iter()
        .enumerate()
        .filter_map(|(at, set)| {
            JudgedSource::of(set)
                .map(|source| (at, source, valued.iter().position(|valued| *valued == at)))
        })
        .collect();
    // Everything the save holds, asked for before any of it is allocated.
    let entries = snapshot.summary.entries.len() as u64;
    let header_estimate: u64 = 64 * 1024
        + lineage
            .iter()
            .map(|set| set.producer.held_bytes() + 2 * set.result.len() as u64 + 8 * 1024)
            .sum::<u64>();
    let fixed = (lineage.len() as u64 + 1) * States::BLOCK_BYTES as u64
        + valued.len() as u64 * Scores::BLOCK_BYTES as u64
        + RowBlock::BYTES
        + IO_BYTES
        + 2 * MAX_ROW_BYTES as u64
        + entries.div_ceil(8)
        + PARSE_FACTOR * header_estimate
        // The detail of one judged record as it is made and written.
        + if judged_sets.is_empty() { 0 } else { PARSE_FACTOR * MAX_JUDGED_LINE_BYTES as u64 }
        + ARTIFACT_FIXED_BYTES;
    worker.budget.hold(worker.base);
    worker.budget.require_whole(
        fixed,
        "saving a receipt: a block of every retained set, the header and the buffer the \
         receipt is written through,",
    )?;
    worker.budget.hold(worker.base.saturating_add(fixed));

    // The blocks of every pass: every set of the lineage, the values of the sets that
    // keep them, and a run of rows of the relation.
    let mut codes: Vec<Vec<u8>> = lineage
        .iter()
        .map(|_| vec![0u8; States::BLOCK_BYTES])
        .collect();
    let mut values: Vec<Vec<u8>> = valued
        .iter()
        .map(|_| vec![0u8; Scores::BLOCK_BYTES])
        .collect();
    let mut run = RowBlock::new();
    let load = |first: u64,
                codes: &mut Vec<Vec<u8>>,
                values: &mut Vec<Vec<u8>>|
     -> Result<usize, L2dError> {
        let mut rows_in = 0;
        for (set, block) in lineage.iter().zip(codes.iter_mut()) {
            rows_in = set.states.read_block(first, block)?;
        }
        for (at, block) in valued.iter().zip(values.iter_mut()) {
            lineage[*at]
                .scores
                .as_ref()
                .expect("a valued set has values")
                .read_block(first, block)?;
        }
        Ok(rows_in)
    };
    // One pass to count what is retained and which entries it names.
    let mut named_entries = vec![0u8; entries.div_ceil(8) as usize];
    let (mut retained, mut retained_occurrences) = (0u64, 0u64);
    // The judged records of every issued-batch set and the bytes of their details.
    let mut judged = vec![0u64; judged_sets.len()];
    let mut judged_bytes = 0u64;
    let mut first = 0;
    let (mut run_start, mut run_end) = (0u64, 0u64);
    while first < rows {
        worker.check_cancel()?;
        let in_block = load(first, &mut codes, &mut values)?;
        let operand_codes: Vec<&[u8]> = operand_at.iter().map(|at| codes[*at].as_slice()).collect();
        for at in 0..in_block {
            if !retained_at(set, &operand_codes, &codes[saved_at], at) {
                continue;
            }
            let row = first + at as u64;
            if !(run_start..run_end).contains(&row) {
                run_start = row;
                run_end = row + snapshot.scope.rows(row, true, &mut run)?;
            }
            retained += 1;
            retained_occurrences += run.occurrences(row);
            let position = run.position(row);
            let entry = l2d_core::ArchivePosition::unpack(position).entry as usize;
            if let Some(byte) = named_entries.get_mut(entry / 8) {
                *byte |= 1 << (entry % 8);
            }
            for (count, (_, source, value_at)) in judged.iter_mut().zip(&judged_sets) {
                let value = value_at.map_or(f32::NAN, |value_at| score_in(&values[value_at], at));
                if let Some(detail) = source.detail(position, value) {
                    if detail.len() > MAX_JUDGED_LINE_BYTES - 64 {
                        return Err(Fault::runtime(
                            "the judgments of one record are longer than a receipt line",
                        ));
                    }
                    *count += 1;
                    judged_bytes += detail.len() as u64;
                }
            }
        }
        first += BLOCK_ROWS;
    }
    if judged.iter().any(|count| *count > MAX_JUDGED) {
        return Err(Fault::refused(format!(
            "an issued-batch set judged more than {MAX_JUDGED} of the retained originals, \
             more than a run issues; nothing was saved"
        )));
    }
    let judged_total: u64 = judged.iter().sum();
    if retained > MAX_ORIGINALS {
        return Err(Fault::refused(format!(
            "the saved set retains {retained} originals, and a receipt retains at most \
             {MAX_ORIGINALS}; reduce it further and save the reduction"
        )));
    }
    let entry_count: u64 = named_entries
        .iter()
        .map(|byte| u64::from(byte.count_ones()))
        .sum();
    fs::create_dir_all(&root).map_err(|error| io("preparing the directory of", &error))?;
    let needed = retained * MAX_ROW_BYTES as u64
        + entry_count * MAX_ENTRY_LINE_BYTES as u64
        + judged_bytes
        + judged_total * 64
        + header_estimate;
    match require_disk(&root, needed, config.free_disk_bytes, "writing a receipt") {
        Ok(()) => {}
        // The figures without the directory: where the host keeps receipts is its own.
        Err(L2dError::Refused { .. }) => {
            return Err(Fault::refused(format!(
                "writing a receipt of {retained} originals needs at most {needed} bytes in the \
                 receipt directory, and its disk has less free; nothing was saved"
            )));
        }
        Err(error) => return Err(error.into()),
    }

    // The header.
    let private_local = config.profile == DisclosureProfile::PrivateLocal;
    let snapshot_result: Value = serde_json::from_str(&snapshot.result).unwrap_or_default();
    let mut source = snapshot_result["source"].clone();
    if source.get("path").is_some() && !private_local {
        source["path"] = json!("<not retained>");
    }
    let (origin_snapshot, origin_counts) = match &snapshot.retained {
        Some(retained) => (
            retained.origin.clone(),
            (retained.origin_originals, retained.origin_occurrences),
        ),
        None => (
            snapshot.handle.clone(),
            (snapshot.scope.originals(), snapshot.scope.occurrences()),
        ),
    };
    let steps = recipe(&snapshot, lineage, saved_at, &task.named, private_local);
    let (recipe_block, steps) = match steps {
        Ok(steps) => (
            json!({"replayable": true, "steps": steps.len(), "reason": Value::Null,
                   "replay": "l2d mcp --replay RECEIPT, or these calls in order with \
                              `parent_receipt` on the save"}),
            steps,
        ),
        Err(reason) => (
            json!({"replayable": false, "steps": 0, "reason": reason}),
            Vec::new(),
        ),
    };
    let mut contains: Vec<&str> = CONTAINS.to_vec();
    if source
        .get("path")
        .is_some_and(|path| path != "<not retained>")
    {
        contains.push("the host path of the source, opened by its path under private-local");
    }
    let kind = if set.kind == SetKind::Final {
        "final"
    } else {
        "candidate"
    };
    let header = json!({
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "contains": contains,
        "excludes": EXCLUDES,
        "row_format": ["record handle", "native identity (64 hex digits)", "occurrences",
                       "state codes of the sets in order (0 out, 1 in, 2 not evaluated, 3 abstained)",
                       "values of the sets that keep them, 8 hex digits of f32 bits each",
                       "slot: [index in the work set, pool, rank in the pool, outside the candidate] or null"],
        "project_sha256": l2d_core::input::content_sha256(config.project.as_bytes()),
        "disclosure": {"profile_at_save": config.profile.name(),
                       "source_class": if snapshot.public { "public_fixture" } else { "private" }},
        "source": source,
        "snapshot": {
            "handle": origin_snapshot,
            "revision": snapshot_result["origin"]["revision"].as_str()
                .unwrap_or(&snapshot.scope.binding().relation),
            "distinct_originals": origin_counts.0,
            "occurrences": origin_counts.1,
            "completeness": snapshot_result.get("origin").and_then(|origin| origin.get("completeness"))
                .or_else(|| snapshot_result.get("completeness")).cloned().unwrap_or(Value::Null),
        },
        "saved": {"handle": set.handle, "kind": set.kind.name(), "revision": set.revision},
        "named": task.named.iter().map(|(name, handle)| (name.clone(), json!(handle))).collect::<Map<String, Value>>(),
        "sets": lineage.iter().map(|set| describe_set(set, lineage)).collect::<Vec<_>>(),
        "allocation": set.allocation.as_ref().map(|allocation| {
            let mut block = allocation.describe();
            block["spec"] = allocation.spec.clone();
            block["operands"] = json!(allocation.operands.iter().map(|set| set.handle.clone()).collect::<Vec<_>>());
            block["names"] = json!(allocation.names);
            block
        }),
        "retained": {
            "population": if set.kind == SetKind::Final {
                "every original eligible for the allocation: in its candidate expression or in a pool's population"
            } else {
                "the members of the saved view"
            },
            "distinct_originals": retained,
            "occurrences": retained_occurrences,
            "entries": entry_count,
        },
        "recipe": recipe_block,
        "issued": judged_sets.iter().zip(&judged).map(|((at, source, _), count)| {
            let mut block = source.run_block();
            block["set"] = json!(at);
            block["judged_records"] = json!(count);
            block
        }).collect::<Vec<_>>(),
        "lineage": {"parent_receipt": task.parent_receipt},
    });
    let header_text = header.to_string();
    if header_text.len() > MAX_HEADER_BYTES {
        return Err(Fault::refused(format!(
            "the header of this receipt takes {} bytes and a receipt takes at most \
             {MAX_HEADER_BYTES}; save a set of fewer operands",
            header_text.len()
        )));
    }
    // The content: the header without its lineage, what follows it and the rows.
    let mut header = header;
    let parent = header["lineage"].take();
    let unlinked = header.to_string();
    header["lineage"] = parent;
    drop(header);
    let step_lines: Vec<String> = steps.iter().map(Value::to_string).collect();
    drop(steps);
    let entry_slots = named_entries
        .iter()
        .enumerate()
        .rev()
        .find(|(_, byte)| **byte != 0)
        .map_or(0, |(at, byte)| {
            at as u64 * 8 + 8 - u64::from(byte.leading_zeros())
        });
    let preamble = json!({
        "triager_receipt": FORMAT_VERSION,
        "header_bytes": header_text.len(),
        "recipe_steps": step_lines.len(),
        "entries": entry_count,
        "entry_slots": entry_slots,
        "rows": retained,
        "sets": lineage.len(),
        "valued": valued.len(),
        "slots": set.allocation.as_ref().map_or(0, |allocation| allocation.slots.len()),
        "judged": judged_total,
        "judged_bytes": judged_bytes,
    })
    .to_string();

    let mut rows_hash = Sha256::new();
    let mut content = Sha256::new();
    let mut line: Vec<u8> = Vec::with_capacity(2 * MAX_ROW_BYTES);
    let (receipt, published) = publish(&root, &mut |out| {
        let mut writer = Hashed {
            out,
            receipt: Sha256::new(),
            bytes: 0,
        };
        writer.line(preamble.as_bytes(), &mut [])?;
        writer.line(header_text.as_bytes(), &mut [])?;
        content.update(unlinked.as_bytes());
        content.update(b"\n");
        for step in &step_lines {
            writer.line(step.as_bytes(), &mut [&mut content])?;
        }
        for (entry, report) in snapshot.summary.entries.iter().enumerate() {
            if named_entries[entry / 8] & (1 << (entry % 8)) == 0 {
                continue;
            }
            let shown = wire::display(&report.path, wire::MAX_PATH_BYTES).0;
            let entry_line = json!({"entry": [entry, shown]}).to_string();
            writer.line(entry_line.as_bytes(), &mut [&mut content])?;
        }
        // The judgments, by record ID ascending and set: `[set, record, detail]`.
        if !judged_sets.is_empty() {
            let mut first = 0;
            let mut written = 0u64;
            let mut id = [0u8; 18];
            while first < rows {
                worker.check_cancel()?;
                let in_block = load(first, &mut codes, &mut values)?;
                let operand_codes: Vec<&[u8]> =
                    operand_at.iter().map(|at| codes[*at].as_slice()).collect();
                for at in 0..in_block {
                    if !retained_at(set, &operand_codes, &codes[saved_at], at) {
                        continue;
                    }
                    let position = snapshot.scope.row(first + at as u64)?.id.0;
                    for (set_at, source, value_at) in &judged_sets {
                        let value =
                            value_at.map_or(f32::NAN, |value_at| score_in(&values[value_at], at));
                        if let Some(detail) = source.detail(position, value) {
                            let judged_line = format!(
                                "{{\"judged\":[{set_at},\"{}\",{detail}]}}",
                                allocate::record_id_into(position, &mut id)
                            );
                            writer.line(judged_line.as_bytes(), &mut [&mut content])?;
                            written += 1;
                        }
                    }
                }
                first += BLOCK_ROWS;
            }
            if written != judged_total {
                return Err(L2dError::Runtime {
                    detail: "the judgments changed between the passes of a save".into(),
                });
            }
        }
        // The rows, by record ID ascending.
        let mut first = 0;
        let mut written = 0u64;
        while first < rows {
            worker.check_cancel()?;
            let in_block = load(first, &mut codes, &mut values)?;
            let operand_codes: Vec<&[u8]> =
                operand_at.iter().map(|at| codes[*at].as_slice()).collect();
            for at in 0..in_block {
                if !retained_at(set, &operand_codes, &codes[saved_at], at) {
                    continue;
                }
                let row = first + at as u64;
                let original = snapshot.scope.row(row)?;
                let mut row_codes = [0u8; MAX_SETS];
                for (index, block) in codes.iter().enumerate() {
                    row_codes[index] = code_in(block, at);
                }
                let mut row_values = [0u32; MAX_SETS];
                for (index, block) in values.iter().enumerate() {
                    row_values[index] = score_in(block, at).to_bits();
                }
                let slot = set
                    .allocation
                    .as_ref()
                    .and_then(|allocation| allocation.slot_of(row));
                row_line(
                    &mut line,
                    original.id.0,
                    &original.native_id.0,
                    original.occurrences,
                    &row_codes[..codes.len()],
                    &row_values[..values.len()],
                    slot,
                )?;
                writer.line(&line, &mut [&mut content, &mut rows_hash])?;
                written += 1;
            }
            first += BLOCK_ROWS;
        }
        if written != retained {
            return Err(L2dError::Runtime {
                detail: "the retained originals changed between the passes of a save".into(),
            });
        }
        let rows_sha256 = format!("{:x}", rows_hash.clone().finalize());
        let content_sha256 = format!("{:x}", content.clone().finalize());
        let digest = format!("{:x}", writer.receipt.clone().finalize());
        let receipt = format!("{PREFIX}{}", &digest[..32]);
        let trailer = json!({"end": {"rows": retained, "rows_sha256": rows_sha256,
                                     "content_sha256": content_sha256, "receipt": receipt}})
        .to_string();
        let mut also: [&mut Sha256; 0] = [];
        writer.line(trailer.as_bytes(), &mut also)?;
        Ok(receipt)
    })?;
    shared
        .activity
        .receipts_saved
        .fetch_add(1, Ordering::Relaxed);
    let rows_sha256 = format!("{:x}", rows_hash.finalize());
    let content_sha256 = format!("{:x}", content.finalize());
    let mut cost = Cost::new(started);
    cost.set("index_rows_read", 2 * rows)
        .set("output", retained)
        .set("receipt_published", published);
    let result = wire::with_response_bytes(json!({
        "v": VERSION,
        "status": "ready",
        "receipt": receipt,
        "kind": kind,
        "snapshot": snapshot.handle,
        "set": set.handle,
        "content_sha256": content_sha256,
        "rows_sha256": rows_sha256,
        "retained": {"distinct_originals": retained, "occurrences": retained_occurrences,
                     "sets": lineage.len(), "entries": entry_count},
        "recipe": recipe_block,
        "lineage": {"parent_receipt": task.parent_receipt},
        "contains": contains,
        "excludes": EXCLUDES,
        "reopen": {"tool": "triager_open", "args": {"v": 1, "receipt": receipt}},
        "cost": cost.finish(worker),
    }));
    // The receipt is published: nothing that can fail comes after it. The result is
    // bounded by what it states (handles, digests, counts, a reason of a few hundred
    // bytes and the fixed lists), far within the response limit.
    debug_assert!(wire::transport_bytes(&result) <= wire::PAYLOAD_TRANSPORT_BYTES);
    worker.budget.hold(worker.base);
    Ok(Done {
        result,
        commit: None,
    })
}

/// The recipe of a receipt: a reader at its first tool call, and how many there are.
/// The preamble is read (at most [`MAX_PREAMBLE_BYTES`]), the header passed over a
/// buffer at a time; a receipt whose recipe is not replayable has no calls and is
/// refused.
pub(crate) fn recipe_of(
    config: &HostConfig,
    receipt: &str,
) -> Result<(BufReader<File>, usize), Fault> {
    let path = receipt_path(config, receipt)?;
    let open = || match File::open(&path) {
        Ok(file) => Ok(file),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Err(unknown_receipt(receipt)),
        Err(error) => Err(Fault::from(io("opening", &error))),
    };
    // Nothing of a receipt is run before every byte of it is checked against its
    // handle: one pass that hashes each line, two line buffers of at most a header.
    {
        let mut reader = BufReader::with_capacity(8 * 1024, open()?);
        let mut hash = Sha256::new();
        let (mut line, mut pending) = (Vec::new(), Vec::new());
        let mut any = false;
        while read_line(&mut reader, &mut line, MAX_HEADER_BYTES, &mut [])? {
            if any {
                hash.update(&pending);
                hash.update(b"\n");
            }
            std::mem::swap(&mut line, &mut pending);
            any = true;
        }
        let trailer: Value = serde_json::from_slice(&pending)
            .map_err(|_| damaged("the trailer is no JSON object"))?;
        let expected = format!("{PREFIX}{}", &format!("{:x}", hash.finalize())[..32]);
        if !any || trailer["end"]["receipt"] != json!(expected) || expected != receipt {
            return Err(damaged("its digests are not those of its content").into());
        }
    }
    let mut reader = BufReader::with_capacity(8 * 1024, open()?);
    let mut line = Vec::with_capacity(MAX_PREAMBLE_BYTES + 2);
    if !read_line(&mut reader, &mut line, MAX_PREAMBLE_BYTES, &mut [])? {
        return Err(damaged("it is empty").into());
    }
    let preamble: Value =
        serde_json::from_slice(&line).map_err(|_| damaged("the preamble is no JSON object"))?;
    if preamble["triager_receipt"] != json!(FORMAT_VERSION) {
        return Err(Fault::refused(
            "this is no triager receipt of a version this server reads",
        ));
    }
    let header_bytes = number(&preamble, "header_bytes", MAX_HEADER_BYTES as u64)?;
    let steps = number(&preamble, "recipe_steps", 64)? as usize;
    // The header, passed over without being kept.
    let skipped = std::io::copy(
        &mut reader.by_ref().take(header_bytes + 1),
        &mut std::io::sink(),
    )
    .map_err(|error| io("reading", &error))?;
    if skipped != header_bytes + 1 {
        return Err(damaged("it ends within its header").into());
    }
    if steps == 0 {
        return Err(Fault::refused(format!(
            "the receipt `{receipt}` has no replayable recipe; its header states why"
        )));
    }
    Ok((reader, steps))
}

// --- Reopening -------------------------------------------------------------------------

pub(crate) struct ReopenTask {
    pub receipt: String,
    pub path: PathBuf,
}

/// A row line as a receipt writes it; every string is borrowed from the line.
type RowLine<'a> = (
    &'a str,
    &'a str,
    u64,
    &'a str,
    &'a str,
    Option<(u32, u32, u32, u8)>,
);

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Row<'a> {
    #[serde(borrow)]
    r: RowLine<'a>,
}

/// Reads one line of at most `most` bytes into `line`, its line end taken off, and
/// feeds it with its line end to `hashes`. `false` at the end of the file.
fn read_line(
    reader: &mut BufReader<File>,
    line: &mut Vec<u8>,
    most: usize,
    hashes: &mut [&mut Sha256],
) -> Result<bool, L2dError> {
    line.clear();
    let read = reader
        .by_ref()
        .take(most as u64 + 1)
        .read_until(b'\n', line)
        .map_err(|error| io("reading", &error))?;
    if read == 0 {
        return Ok(false);
    }
    if line.pop() != Some(b'\n') {
        return Err(damaged(
            "a line is longer than its section allows or the receipt ends early",
        ));
    }
    for hash in hashes {
        hash.update(&line[..]);
        hash.update(b"\n");
    }
    Ok(true)
}

fn number(value: &Value, field: &str, most: u64) -> Result<u64, L2dError> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .filter(|number| *number <= most)
        .ok_or_else(|| damaged(&format!("the preamble's `{field}` is missing or too large")))
}

/// One set as the header describes it, with what reopening makes of it.
struct Described {
    handle: String,
    kind: SetKind,
    described: Value,
    origin: Origin,
    parent: Option<usize>,
    inputs: Vec<usize>,
    values: Option<usize>,
    states: Option<States>,
    scores: Option<Scores>,
    code_block: Vec<u8>,
    value_block: Vec<u8>,
    /// For an issued-batch set: its run block and its judged records.
    judged: Option<Judged>,
}

/// Reopens the receipt of `task` as a retained snapshot with every set it retained.
pub(crate) fn reopen(worker: &mut Worker<'_>, task: &ReopenTask) -> Result<Done, Fault> {
    let started = Instant::now();
    let shared = worker.shared;
    let config = &shared.config;
    let project = &config.project;
    let file = match File::open(&task.path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(unknown_receipt(&task.receipt));
        }
        Err(error) => return Err(io("opening", &error).into()),
    };
    worker.budget.hold(worker.base);
    let reading = IO_BYTES + MAX_PREAMBLE_BYTES as u64 + 2 * MAX_STEP_BYTES as u64;
    worker
        .budget
        .require_whole(reading, "reading a receipt: its buffer and a line,")?;
    let mut held = worker.base.saturating_add(reading);
    worker.budget.hold(held);
    let mut reader = BufReader::with_capacity(IO_BYTES as usize, file);
    let mut receipt_hash = Sha256::new();
    let mut content = Sha256::new();
    let mut rows_hash = Sha256::new();
    let mut line: Vec<u8> = Vec::with_capacity(MAX_PREAMBLE_BYTES + 2);
    if !read_line(
        &mut reader,
        &mut line,
        MAX_PREAMBLE_BYTES,
        &mut [&mut receipt_hash],
    )? {
        return Err(damaged("it is empty").into());
    }
    let preamble: Value =
        serde_json::from_slice(&line).map_err(|_| damaged("the preamble is no JSON object"))?;
    if preamble["triager_receipt"] != json!(FORMAT_VERSION) {
        return Err(Fault::refused(
            "this is no triager receipt of a version this server reads",
        ));
    }
    let header_bytes = number(&preamble, "header_bytes", MAX_HEADER_BYTES as u64)? as usize;
    let steps = number(&preamble, "recipe_steps", 64)? as usize;
    let entries = number(&preamble, "entries", u64::from(u32::MAX))?;
    let entry_slots = number(&preamble, "entry_slots", u64::from(u32::MAX) + 1)?;
    let rows = number(&preamble, "rows", MAX_ORIGINALS)?;
    let set_count = number(&preamble, "sets", MAX_SETS as u64)? as usize;
    let valued = number(&preamble, "valued", MAX_SETS as u64)? as usize;
    let slot_count = number(&preamble, "slots", allocate::MAX_SLOTS)? as usize;
    let judged_total = number(&preamble, "judged", MAX_JUDGED * MAX_SETS as u64)?;
    let judged_bytes = number(
        &preamble,
        "judged_bytes",
        judged_total * MAX_JUDGED_LINE_BYTES as u64,
    )?;
    if entries > entry_slots || set_count == 0 {
        return Err(damaged("the preamble's counts do not agree").into());
    }
    // The header and its parse, the copy its content digest is taken from, and the
    // parse of the recipe a result lists.
    let header_share =
        (header_bytes as u64 + 2) * (PARSE_FACTOR + 2) + RECIPE_SHOWN_BYTES as u64 * PARSE_FACTOR;
    worker.budget.require_whole(
        header_share,
        "reading the header of a receipt and its parse,",
    )?;
    held = held.saturating_add(header_share);
    worker.budget.hold(held);
    let mut text: Vec<u8> = Vec::with_capacity(header_bytes + 2);
    if !read_line(
        &mut reader,
        &mut text,
        header_bytes,
        &mut [&mut receipt_hash],
    )? || text.len() != header_bytes
    {
        return Err(damaged("the header is not as long as the preamble says").into());
    }
    let mut header: Value =
        serde_json::from_slice(&text).map_err(|_| damaged("the header is no JSON object"))?;
    drop(text);
    {
        // The content without the lineage, taken out and put back: no copy is made.
        let lineage = header["lineage"].take();
        content.update(header.to_string().as_bytes());
        content.update(b"\n");
        header["lineage"] = lineage;
    }
    if header["format"] != FORMAT || header["format_version"] != json!(FORMAT_VERSION) {
        return Err(Fault::refused(
            "the receipt states another format than this server reads",
        ));
    }
    if header["project_sha256"] != json!(l2d_core::input::content_sha256(project.as_bytes())) {
        return Err(Fault::new(
            "wrong_project",
            "the receipt was saved by another project; a receipt reopens in its own project",
        ));
    }
    let public = header["disclosure"]["source_class"] == "public_fixture";
    if config.profile == DisclosureProfile::PublicDevelopment && !public {
        return Err(Fault::new(
            "disclosure_refused",
            "the host runs the public-development profile, and the receipt retains a private \
             source; it is not reopened",
        ));
    }
    let listed = header["sets"]
        .as_array()
        .filter(|sets| sets.len() == set_count)
        .ok_or_else(|| damaged("the header lists another number of sets than the preamble"))?;
    // The sets, each after what it was made from.
    let mut sets: Vec<Described> = Vec::with_capacity(set_count);
    let mut next_value = 0;
    for listed_set in listed {
        let handle = listed_set["handle"]
            .as_str()
            .ok_or_else(|| damaged("a set has no handle"))?
            .to_string();
        let kind = listed_set["kind"]
            .as_str()
            .and_then(SetKind::parse)
            .ok_or_else(|| damaged("a set has no kind"))?;
        let origin = Origin::parse(&listed_set["origin"])
            .ok_or_else(|| damaged("a set names a producer this server does not know"))?;
        let find = |known: &str| sets.iter().position(|set| set.handle == known);
        let parent = match &listed_set["parent"] {
            Value::Null => None,
            Value::String(parent) => Some(
                find(parent).ok_or_else(|| damaged("a set's parent is not listed before it"))?,
            ),
            _ => return Err(damaged("a set's parent is no handle").into()),
        };
        let inputs = listed_set["inputs"]
            .as_array()
            .ok_or_else(|| damaged("a set lists no inputs"))?
            .iter()
            .map(|input| input.as_str().and_then(find))
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| damaged("a set's input is not listed before it"))?;
        let values = (listed_set["values"] == true).then(|| {
            next_value += 1;
            next_value - 1
        });
        sets.push(Described {
            handle,
            kind,
            described: listed_set["producer"].clone(),
            origin,
            parent,
            inputs,
            values,
            states: None,
            scores: None,
            code_block: Vec::new(),
            value_block: Vec::new(),
            judged: None,
        });
    }
    // The run blocks of the issued-batch sets; their records follow the entry names.
    for block in header["issued"].as_array().into_iter().flatten() {
        let at = block["set"]
            .as_u64()
            .map(|at| at as usize)
            .filter(|at| *at < sets.len() && sets[*at].origin == Origin::Issued)
            .ok_or_else(|| damaged("judgments name no issued-batch set"))?;
        if sets[at].judged.is_some() {
            return Err(damaged("an issued-batch set has two run blocks").into());
        }
        sets[at].judged = Some(Judged {
            run: block.clone(),
            records: Vec::new(),
        });
    }
    let saved_at = sets
        .iter()
        .position(|set| header["saved"]["handle"] == json!(set.handle));
    let Some(saved_at) = saved_at.filter(|_| next_value == valued) else {
        return Err(damaged("the header and the preamble disagree on the sets").into());
    };
    let saved_kind = sets[saved_at].kind;
    if (saved_kind == SetKind::Final) != (header["kind"] == "final")
        || (saved_kind == SetKind::Final) != header["allocation"].is_object()
    {
        return Err(damaged("the saved kind and the allocation disagree").into());
    }

    // The recipe: hashed, and listed when it is short.
    let mut recipe_shown: Vec<Value> = Vec::new();
    let mut recipe_bytes = 0usize;
    let mut step_line: Vec<u8> = Vec::with_capacity(MAX_STEP_BYTES + 2);
    for _ in 0..steps {
        if !read_line(
            &mut reader,
            &mut step_line,
            MAX_STEP_BYTES,
            &mut [&mut receipt_hash, &mut content],
        )? {
            return Err(damaged("it ends within its recipe").into());
        }
        recipe_bytes += step_line.len();
        if recipe_bytes <= RECIPE_SHOWN_BYTES {
            recipe_shown.push(
                serde_json::from_slice(&step_line)
                    .map_err(|_| damaged("a recipe step is no JSON"))?,
            );
        }
    }
    drop(step_line);
    if recipe_bytes > RECIPE_SHOWN_BYTES {
        recipe_shown.clear();
    }

    // The entry names: what the locators of the retained originals name.
    let entry_share = entry_slots * std::mem::size_of::<EntryReport>() as u64
        + entries * (wire::MAX_PATH_BYTES as u64 + 64);
    worker
        .budget
        .require_whole(entry_share, "keeping the entry names a receipt retained,")?;
    held = held.saturating_add(entry_share);
    worker.budget.hold(held);
    let mut reports: Vec<EntryReport> = Vec::with_capacity(entry_slots as usize);
    for _ in 0..entries {
        if !read_line(
            &mut reader,
            &mut line,
            MAX_ENTRY_LINE_BYTES,
            &mut [&mut receipt_hash, &mut content],
        )? {
            return Err(damaged("it ends within its entry names").into());
        }
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct EntryLine {
            entry: (u64, String),
        }
        let EntryLine {
            entry: (index, name),
        } = serde_json::from_slice(&line).map_err(|_| damaged("an entry line is malformed"))?;
        if index < reports.len() as u64 || index >= entry_slots {
            return Err(damaged("the entry names are not in order").into());
        }
        while (reports.len() as u64) < index {
            reports.push(EntryReport {
                path: String::new(),
                records: 0,
                skipped: None,
            });
        }
        reports.push(EntryReport {
            path: name,
            records: 0,
            skipped: None,
        });
    }

    // The judgments of the issued-batch sets: what every judged record's explanation
    // says, kept as text by record ID, asked for before it is read. A line is parsed
    // into a value of at most sixteen times its bytes on its way.
    let judged_share = judged_bytes
        + judged_total * std::mem::size_of::<(u64, Box<str>)>() as u64
        + if judged_total > 0 {
            (PARSE_FACTOR + 2) * MAX_JUDGED_LINE_BYTES as u64
        } else {
            0
        };
    worker.budget.require_whole(
        judged_share,
        "keeping the model judgments a receipt retained,",
    )?;
    held = held.saturating_add(judged_share);
    worker.budget.hold(held);
    let counted: u64 = sets
        .iter()
        .filter_map(|set| set.judged.as_ref())
        .map(|judged| judged.run["judged_records"].as_u64().unwrap_or(u64::MAX))
        .fold(0u64, u64::saturating_add);
    if counted != judged_total {
        return Err(damaged("the run blocks count other judgments than the preamble").into());
    }
    for set in &mut sets {
        if let Some(judged) = &mut set.judged {
            let count = judged.run["judged_records"].as_u64().unwrap_or(0);
            judged.records.reserve_exact(count as usize);
        }
    }
    let mut judged_line: Vec<u8> = Vec::new();
    let mut read_bytes = 0u64;
    for _ in 0..judged_total {
        if !read_line(
            &mut reader,
            &mut judged_line,
            MAX_JUDGED_LINE_BYTES,
            &mut [&mut receipt_hash, &mut content],
        )? {
            return Err(damaged("it ends within its judgments").into());
        }
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct JudgedLine {
            judged: (usize, String, Value),
        }
        let JudgedLine {
            judged: (at, id, detail),
        } = serde_json::from_slice(&judged_line)
            .map_err(|_| damaged("a judgment line is malformed"))?;
        let position =
            wire::record_position(&id).ok_or_else(|| damaged("a judgment has no record handle"))?;
        let detail = detail.to_string().into_boxed_str();
        read_bytes += detail.len() as u64;
        let records = &mut sets
            .get_mut(at)
            .and_then(|set| set.judged.as_mut())
            .ok_or_else(|| damaged("a judgment names no issued-batch set"))?
            .records;
        if records.last().is_some_and(|(last, _)| *last >= position)
            || records.len() == records.capacity()
            || read_bytes > judged_bytes
        {
            return Err(damaged("the judgments are not in order or not as counted").into());
        }
        records.push((position, detail));
    }
    drop(judged_line);

    // The tables of every set, in memory when the budget admits them all.
    let in_memory =
        set_count as u64 * States::bytes_for(rows) + valued as u64 * Scores::bytes_for(rows);
    let blocks = set_count as u64 * States::BLOCK_BYTES as u64
        + valued as u64 * Scores::BLOCK_BYTES as u64
        + MAX_ROW_BYTES as u64 * 2
        + (slot_count * std::mem::size_of::<(usize, Slot)>()) as u64;
    worker.budget.require_whole(
        blocks,
        "reopening a receipt: a block of every retained set and the slots of its work set,",
    )?;
    held = held.saturating_add(blocks);
    worker.budget.hold(held);
    let on_disk = ops::tables_on_disk(worker, in_memory);
    if !on_disk {
        held = held.saturating_add(in_memory);
        worker.budget.hold(held);
    }
    for set in &mut sets {
        let (states, scores) = if on_disk {
            let dir = ops::set_dir(
                worker,
                States::bytes_for(rows)
                    + if set.values.is_some() {
                        Scores::bytes_for(rows)
                    } else {
                        0
                    },
                "placing the tables of a reopened set on disk",
            )?;
            (
                States::on_disk(&dir, rows)?,
                match set.values {
                    Some(_) => Some(Scores::on_disk(&dir, rows)?),
                    None => None,
                },
            )
        } else {
            (
                States::in_memory(rows, State::Out),
                set.values.map(|_| Scores::in_memory(rows)),
            )
        };
        set.states = Some(states);
        set.scores = scores;
        set.code_block = vec![0u8; States::BLOCK_BYTES];
        if set.values.is_some() {
            set.value_block = vec![0u8; Scores::BLOCK_BYTES];
        }
    }
    let mut slots: Vec<(usize, Slot)> = Vec::with_capacity(slot_count);

    // The rows: the relation and the tables, one block at a time.
    let settings = ScopeSettings {
        index: config.index,
        free_disk_bytes: config.free_disk_bytes,
        ..ScopeSettings::new(worker.fresh_dir("snapshot"))
    };
    let entry_patterns: Vec<String> = header["source"]["entries"]
        .as_array()
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let content_sha256 = header["source"]["content_sha256"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let source = MarkedSource::Input(MarkedInput {
        path: format!("receipt:{}", task.receipt),
        content_sha256: content_sha256.clone(),
        archive_entries: Some(entry_patterns.clone()),
        interpretation: None,
    });
    let complete = header["snapshot"]["completeness"]["complete"] == true;
    let cancel = worker.cancel;
    let mut total_occurrences = 0u64;
    let mut read_rows = 0u64;
    let mut row_line: Vec<u8> = Vec::with_capacity(MAX_ROW_BYTES + 2);
    let base_for_scope = held;
    worker.budget.hold(base_for_scope);
    let scope = {
        let sets = &mut sets;
        let slots = &mut slots;
        let reader = &mut reader;
        let receipt_hash = &mut receipt_hash;
        let content = &mut content;
        let rows_hash = &mut rows_hash;
        let mut feed = |push: RetainedPush<'_>| -> Result<(), L2dError> {
            let flush =
                |sets: &mut Vec<Described>, first: u64, in_block: usize| -> Result<(), L2dError> {
                    for set in sets.iter_mut() {
                        set.states.as_mut().expect("made above").write_block(
                            first,
                            &set.code_block,
                            in_block,
                        )?;
                        if let Some(scores) = &mut set.scores {
                            scores.write_block(first, &set.value_block, in_block)?;
                        }
                    }
                    Ok(())
                };
            let mut first = 0u64;
            for row in 0..rows {
                if row % BLOCK_ROWS == 0 && cancel.load(Ordering::Relaxed) {
                    return Err(super::wire::cancelled());
                }
                if !read_line(
                    reader,
                    &mut row_line,
                    MAX_ROW_BYTES,
                    &mut [&mut *receipt_hash, &mut *content, &mut *rows_hash],
                )? {
                    return Err(damaged("it ends within its rows"));
                }
                let Row {
                    r: (id, native, occurrences, codes, values, slot),
                } = serde_json::from_slice(&row_line).map_err(|_| damaged("a row is malformed"))?;
                let position = wire::record_position(id)
                    .ok_or_else(|| damaged("a row has no record handle"))?;
                let native = NativeId::from_hex(native)
                    .ok_or_else(|| damaged("a row has no native identity"))?;
                let occurrences = u32::try_from(occurrences)
                    .ok()
                    .filter(|count| *count > 0 && *count < u32::MAX)
                    .ok_or_else(|| damaged("a row has no usable occurrence count"))?;
                if codes.len() != sets.len() || values.len() != 8 * valued {
                    return Err(damaged(
                        "a row has another number of states or values than sets",
                    ));
                }
                push(position, &native.0, occurrences)?;
                total_occurrences += u64::from(occurrences);
                let at = (row % BLOCK_ROWS) as usize;
                if at == 0 && row > 0 {
                    flush(sets, first, BLOCK_ROWS as usize)?;
                    first = row;
                }
                for (set, code) in sets.iter_mut().zip(codes.bytes()) {
                    let code = code
                        .checked_sub(b'0')
                        .filter(|code| *code <= 3)
                        .ok_or_else(|| damaged("a state code is not 0 to 3"))?;
                    put_code(&mut set.code_block, at, code);
                    if let Some(index) = set.values {
                        let hex = &values[8 * index..8 * index + 8];
                        let bits = u32::from_str_radix(hex, 16)
                            .map_err(|_| damaged("a value is not 8 hex digits"))?;
                        set.value_block[at * 4..at * 4 + 4].copy_from_slice(&bits.to_le_bytes());
                    }
                }
                if let Some((index, pool, rank, outside)) = slot {
                    if slots.len() >= slot_count || outside > 1 {
                        return Err(damaged("a row holds a slot beyond the work set"));
                    }
                    slots.push((
                        index as usize,
                        Slot {
                            row,
                            pool,
                            rank,
                            outside_candidate: outside == 1,
                        },
                    ));
                }
                read_rows += 1;
            }
            if rows > 0 {
                flush(sets, first, (rows - first) as usize)?;
            }
            Ok(())
        };
        Scope::from_retained(
            SourceKind::Archive,
            source,
            complete,
            &settings,
            worker.budget,
            &mut feed,
        )?
    };
    drop(row_line);
    // The relation is the snapshot's to hold from here on.
    held = held.saturating_add(scope.held_bytes());
    worker.budget.hold(held);
    // The trailer: every digest must be the one recomputed.
    let receipt_digest = format!("{:x}", receipt_hash.finalize());
    if !read_line(&mut reader, &mut line, MAX_PREAMBLE_BYTES, &mut [])? {
        return Err(damaged("it has no trailer").into());
    }
    let trailer: Value =
        serde_json::from_slice(&line).map_err(|_| damaged("the trailer is no JSON object"))?;
    let rows_sha256 = format!("{:x}", rows_hash.finalize());
    let content_sha256_of = format!("{:x}", content.finalize());
    let expected = format!("{PREFIX}{}", &receipt_digest[..32]);
    if trailer["end"]["rows"] != json!(read_rows)
        || trailer["end"]["rows_sha256"] != json!(rows_sha256)
        || trailer["end"]["content_sha256"] != json!(content_sha256_of)
        || trailer["end"]["receipt"] != json!(expected)
        || expected != task.receipt
    {
        return Err(damaged("its digests are not those of its content").into());
    }
    if read_line(&mut reader, &mut line, 1, &mut [])? || !line.is_empty() {
        return Err(damaged("it goes on after its trailer").into());
    }
    drop(reader);
    if scope.occurrences() != total_occurrences || scope.originals() != rows {
        return Err(damaged("the retained relation does not add up").into());
    }

    // The snapshot.
    let summary = ArchiveSummary {
        input: InputSummary {
            source_path: PathBuf::new(),
            records: total_occurrences,
            distinct_texts: rows,
            content_sha256: content_sha256.clone(),
            skipped: Vec::new(),
            label_conflicts: 0,
        },
        entries: reports,
    };
    let snapshot_handle = wire::handle("snap", project, &[scope.handle(), &task.receipt]);
    if let Some(known) = registry(shared).snapshot(&snapshot_handle) {
        return Ok(Done {
            result: known.result.clone(),
            commit: None,
        });
    }
    let origin = &header["snapshot"];
    let retained = Retained {
        receipt: task.receipt.clone(),
        origin: origin["handle"].as_str().unwrap_or_default().to_string(),
        origin_originals: origin["distinct_originals"].as_u64().unwrap_or(0),
        origin_occurrences: origin["occurrences"].as_u64().unwrap_or(0),
    };
    let max_record_bytes = header["source"]["max_record_bytes"]
        .as_u64()
        .unwrap_or(ZipInputSettings::default().max_record_bytes);
    let mut snapshot = Snapshot {
        handle: snapshot_handle,
        scope,
        summary,
        path: PathBuf::new(),
        zip: ZipInputSettings {
            entries: entry_patterns,
            max_record_bytes,
            ..ZipInputSettings::default()
        },
        public,
        result: String::new(),
        held: 0,
        retained: Some(retained),
    };
    // The sets, each after what it was made from, as the receipt retained them.
    worker.budget.require_whole(
        TALLY_BYTES + ARTIFACT_FIXED_BYTES * set_count as u64,
        "counting the reopened sets,",
    )?;
    let snapshot_arc;
    let mut made: Vec<Arc<Set>> = Vec::with_capacity(set_count);
    let mut listed_sets: Vec<Value> = Vec::with_capacity(set_count);
    {
        // The result of the snapshot names every set; it is made first with the
        // handles, which do not depend on it.
        let handles: Vec<String> = sets
            .iter()
            .map(|set| {
                let prefix = match set.kind {
                    SetKind::Selection => "set",
                    SetKind::View => "view",
                    SetKind::Final => "final",
                };
                wire::handle(prefix, project, &[&snapshot.handle, &set.handle])
            })
            .collect();
        for (set, handle) in sets.iter().zip(&handles) {
            let name = header["named"]
                .as_object()
                .and_then(|named| named.iter().find(|(_, saved)| **saved == json!(set.handle)))
                .map(|(name, _)| json!(name));
            listed_sets.push(json!({
                "handle": handle,
                "saved_as": set.handle,
                "kind": set.kind.name(),
                "producer": set.origin.name(),
                "name": name,
            }));
        }
        let reopened = handles[saved_at].clone();
        let mut cost = Cost::new(started);
        cost.set("receipts_read", 1)
            .set("output", rows)
            .set("tables_on_disk", on_disk)
            .set("index_on_disk", snapshot.scope.index_on_disk());
        let source_block = header["source"].clone();
        let result = wire::with_response_bytes(json!({
            "v": VERSION,
            "status": "ready",
            "snapshot": snapshot.handle,
            "revision": snapshot.scope.binding().relation,
            "receipt": task.receipt,
            "kind": header["kind"],
            "set": reopened,
            "saved_as": header["saved"]["handle"],
            "sets": listed_sets,
            "source": source_block,
            "counts": {"distinct_originals": rows, "occurrences": total_occurrences},
            "retained": header["retained"],
            "origin": {
                "snapshot": origin["handle"],
                "revision": origin["revision"],
                "distinct_originals": origin["distinct_originals"],
                "occurrences": origin["occurrences"],
                "completeness": origin["completeness"],
            },
            "completeness": {"complete": complete, "retained_from_receipt": true},
            "lineage": header["lineage"],
            "content_sha256": content_sha256_of,
            "contains": header["contains"],
            "excludes": header["excludes"],
            "recipe": {
                "replayable": header["recipe"]["replayable"],
                "reason": header["recipe"]["reason"],
                "steps": header["recipe"]["steps"],
                "calls": if recipe_shown.is_empty() { Value::Null } else { json!(recipe_shown) },
                "listed": !recipe_shown.is_empty(),
                "replay": "l2d mcp --replay RECEIPT runs it against the registered source",
            },
            "capabilities": {
                "compose_modes": ["view", "allocate"],
                "operators": ["union", "intersection", "difference", "complement", "state"],
                "inspect_views": ["overview", "regions", "records", "evidence"],
                "save": true,
                "not_supported": ["triager_select", "triager_refine", "evidence excerpt"],
                "why": "a reopened receipt retains no source text; open the source and replay \
                        the recipe to select, refine or read an excerpt",
            },
            "disclosure": shared.disclosure(&snapshot),
            "cost": cost.finish(worker),
        }));
        ops::require_fit(&result)?;
        snapshot.result = result;
        snapshot.held = snapshot
            .scope
            .held_bytes()
            .saturating_add(entry_share)
            .saturating_add(2 * snapshot.result.capacity() as u64)
            .saturating_add(ARTIFACT_FIXED_BYTES);
        snapshot_arc = Arc::new(snapshot);
        for (index, set) in sets.drain(..).enumerate() {
            let parent = set.parent.map(|at| made[at].clone());
            let population = match &parent {
                Some(parent) => Population::Of(parent.clone()),
                None => Population::All(snapshot_arc.clone()),
            };
            let states = set.states.expect("made above");
            let (tally, revision, _) = ops::tally(&snapshot_arc, &population, &states)?;
            let inputs: Vec<Arc<Set>> = set.inputs.iter().map(|at| made[*at].clone()).collect();
            let listed_set = &listed[index];
            let saved_from = json!({
                "receipt": task.receipt,
                "handle": set.handle,
                "revision": listed_set["revision"],
                "counts": listed_set["counts"],
                "occurrence_counts": listed_set["occurrence_counts"],
            });
            let producer = Producer::Retained {
                described: set.described,
                origin: set.origin.clone(),
                judged: set.judged.map(Arc::new),
            };
            let handle = handles[index].clone();
            let allocation = if set.kind == SetKind::Final {
                Some(Arc::new(rebuild_allocation(
                    &header["allocation"],
                    &inputs,
                    std::mem::take(&mut slots),
                    slot_count,
                )?))
            } else {
                None
            };
            let result = match &allocation {
                Some(allocation) => {
                    let named: Vec<(String, Arc<Set>)> = allocation
                        .names
                        .iter()
                        .cloned()
                        .zip(allocation.operands.iter().cloned())
                        .collect();
                    allocate::final_result(
                        &handle,
                        &snapshot_arc,
                        &population,
                        &named,
                        allocation,
                        &tally,
                        &revision,
                        json!({"archive_scans": 0, "rows_scored": 0, "response_bytes": 0}),
                        Some(saved_from),
                    )
                }
                None => reopened_result(
                    &handle,
                    set.kind,
                    &snapshot_arc,
                    &population,
                    &producer,
                    &tally,
                    &revision,
                    &listed_set["coverage"],
                    saved_from,
                ),
            };
            ops::require_fit(&result)?;
            let scores = set.scores.map(Arc::new);
            let kept = states
                .held_bytes()
                .saturating_add(scores.as_ref().map_or(0, |scores| scores.held_bytes()))
                .saturating_add(producer.held_bytes())
                .saturating_add(
                    allocation
                        .as_ref()
                        .map_or(0, |allocation| allocation.held_bytes()),
                )
                .saturating_add(2 * result.len() as u64 + handle.capacity() as u64)
                .saturating_add(ARTIFACT_FIXED_BYTES);
            made.push(Arc::new(Set {
                handle,
                kind: set.kind,
                snapshot: snapshot_arc.clone(),
                parent,
                states,
                owns_scores: scores.is_some(),
                scores,
                producer,
                tally,
                revision,
                result,
                held: kept,
                inputs,
                allocation,
            }));
        }
    }
    shared
        .activity
        .receipts_opened
        .fetch_add(1, Ordering::Relaxed);
    let result = snapshot_arc.result.clone();
    let commit = Commit::Reopened {
        snapshot: snapshot_arc,
        sets: made,
    };
    worker
        .budget
        .hold(worker.base.saturating_add(commit.held()));
    Ok(Done {
        result,
        commit: Some(commit),
    })
}

/// The result of a reopened selection or view.
#[allow(clippy::too_many_arguments)]
fn reopened_result(
    handle: &str,
    kind: SetKind,
    snapshot: &Snapshot,
    population: &Population,
    producer: &Producer,
    tally: &Tally,
    revision: &str,
    coverage: &Value,
    saved_from: Value,
) -> String {
    let mut result = json!({
        "v": VERSION,
        "status": "ready",
        "kind": kind.name(),
        "snapshot": snapshot.handle,
        "scope": population.handle(),
        "producer": producer.describe(),
        "revision": revision,
        "counts": tally.counts(),
        "occurrence_counts": tally.occurrence_counts(),
        "coverage": ops::coverage(tally, "not_retained_or_not_evaluated"),
        "saved_coverage": coverage,
        "retained_from": saved_from,
        "evidence": {"tool": "triager_inspect", "handle": handle, "views": ["records", "evidence"]},
        "cost": {"archive_scans": 0, "rows_scored": 0, "response_bytes": 0},
    });
    match kind {
        SetKind::View => {
            result["view"] = json!(handle);
            result["parent"] = json!(population.handle());
        }
        _ => result["set"] = json!(handle),
    }
    wire::with_response_bytes(result)
}

/// The allocation of a reopened final work set: the request, the pools and their
/// counts from the header, the slots from the rows.
fn rebuild_allocation(
    block: &Value,
    inputs: &[Arc<Set>],
    mut slots: Vec<(usize, Slot)>,
    slot_count: usize,
) -> Result<Allocation, L2dError> {
    let spec = block["spec"].clone();
    let operand_handles: Vec<&str> = block["operands"]
        .as_array()
        .map(|handles| handles.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let names: Vec<String> = block["names"]
        .as_array()
        .map(|names| {
            names
                .iter()
                .filter_map(|name| name.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if operand_handles.len() != names.len() || operand_handles.len() > inputs.len() {
        return Err(damaged(
            "the allocation's operands are not those of its sets",
        ));
    }
    // The operands are the first inputs, in the order the expressions index them.
    let operands: Vec<Arc<Set>> = inputs[..operand_handles.len()].to_vec();
    let expr = Expr::parse(&spec["expr"], &operand_handles)
        .map_err(|_| damaged("the allocation's expression does not read"))?;
    let listed = spec["budget"]["pools"]
        .as_array()
        .ok_or_else(|| damaged("the allocation lists no pools"))?;
    let counted = block["pools"]
        .as_array()
        .filter(|pools| pools.len() == listed.len())
        .ok_or_else(|| damaged("the allocation's pools and their counts disagree"))?;
    let mut pools = Vec::with_capacity(listed.len());
    let mut outcomes = Vec::with_capacity(listed.len());
    for (pool, counts) in listed.iter().zip(counted) {
        let order = pool["order"]
            .as_array()
            .ok_or_else(|| damaged("a pool has no order"))?
            .iter()
            .map(|key| match key.as_str() {
                Some("seeded_hash") => Some(OrderKey::SeededHash),
                Some("record_id_ascending") => Some(OrderKey::RecordIdAscending),
                _ => key
                    .get("score_desc")
                    .and_then(Value::as_str)
                    .map(|_| OrderKey::ScoreDesc(0)),
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| damaged("a pool's order does not read"))?;
        let pair = |value: &Value| -> (u64, u64) {
            (
                value["distinct_originals"].as_u64().unwrap_or(0),
                value["occurrences"].as_u64().unwrap_or(0),
            )
        };
        let name = pool["name"].as_str().unwrap_or_default().to_string();
        pools.push(Pool {
            name: name.clone(),
            limit: pool["limit"].as_u64().unwrap_or(0),
            from: Expr::parse(&pool["from"], &operand_handles)
                .map_err(|_| damaged("a pool's population does not read"))?,
            order,
            seed: pool["seed"].as_u64(),
        });
        outcomes.push(PoolOutcome {
            name,
            limit: counts["limit"].as_u64().unwrap_or(0),
            selected: counts["selected"].as_u64().unwrap_or(0),
            eligible: pair(&counts["eligible"]),
            taken_elsewhere: counts["taken_by_other_pools"].as_u64().unwrap_or(0),
            deferred: pair(&counts["deferred"]),
            outside_candidate: counts["outside_candidate"].as_u64().unwrap_or(0),
        });
    }
    if slots.len() != slot_count {
        return Err(damaged(
            "the rows hold another number of slots than the work set",
        ));
    }
    slots.sort_unstable_by_key(|(index, _)| *index);
    if slots
        .iter()
        .enumerate()
        .any(|(at, (index, slot))| *index != at || slot.pool as usize >= pools.len())
    {
        return Err(damaged("the slots of the work set are not one each"));
    }
    let slots: Vec<Slot> = slots.into_iter().map(|(_, slot)| slot).collect();
    let mut by_row: Vec<(u64, u32)> = slots
        .iter()
        .enumerate()
        .map(|(index, slot)| (slot.row, index as u32))
        .collect();
    by_row.sort_unstable();
    let pair = |value: &Value| -> (u64, u64) {
        (
            value["distinct_originals"].as_u64().unwrap_or(0),
            value["occurrences"].as_u64().unwrap_or(0),
        )
    };
    Ok(Allocation {
        total: spec["budget"]["total"].as_u64().unwrap_or(0),
        spec,
        names,
        operands,
        expr,
        pools,
        outcomes,
        slots,
        by_row,
        candidate: pair(&block["candidate"]),
        candidate_unknown: pair(&block["candidate"]["not_evaluated"]),
        deferred: pair(&block["deferred"]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A receipt is published whole or not at all: a write that fails leaves no
    /// receipt and no staging file, and one that exists is never replaced.
    #[test]
    fn a_receipt_is_published_whole_or_not_at_all() {
        let root = tempfile::tempdir().unwrap();
        let receipts = root.path().join("receipts");
        let handle = format!("{PREFIX}{}", "a".repeat(32));
        let (published, first) = publish(&receipts, &mut |out| {
            out.write_all(b"first\n").unwrap();
            Ok(handle.clone())
        })
        .unwrap();
        assert_eq!(published, handle);
        assert!(first);
        let path = receipts.join(format!("{handle}.{EXTENSION}"));
        assert_eq!(fs::read(&path).unwrap(), b"first\n");
        // A write that fails half-way: nothing under any name, no staging file left.
        let other = format!("{PREFIX}{}", "b".repeat(32));
        let failed = publish(&receipts, &mut |out| {
            out.write_all(&[7u8; 300_000]).unwrap();
            Err(L2dError::Runtime {
                detail: "the disk filled up".into(),
            })
        });
        assert!(failed.is_err());
        assert!(!receipts.join(format!("{other}.{EXTENSION}")).exists());
        // The same handle again: the receipt that exists stays as it is.
        let (_, again) = publish(&receipts, &mut |out| {
            out.write_all(b"second\n").unwrap();
            Ok(handle.clone())
        })
        .unwrap();
        assert!(!again);
        assert_eq!(fs::read(&path).unwrap(), b"first\n");
        let left: Vec<String> = fs::read_dir(&receipts)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(left, [format!("{handle}.{EXTENSION}")]);
    }

    #[test]
    fn receipt_handles_are_checked_before_a_file_is_named() {
        assert!(is_receipt(&format!("{PREFIX}{}", "0f".repeat(16))));
        for bad in [
            "rcpt-",
            "rcpt-0F0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f",
            "rcpt-../../x",
            "H1",
        ] {
            assert!(!is_receipt(bad), "{bad}");
        }
    }

    /// The longest row a receipt writes fits its line and its buffer, and is written
    /// without an allocation; a header of the most sets parses within the share a
    /// reopen asks for it.
    #[test]
    fn the_longest_row_fits_and_a_header_parses_within_its_share() {
        let mut line = Vec::with_capacity(2 * MAX_ROW_BYTES);
        let slot = Slot {
            row: u64::MAX,
            pool: allocate::MAX_POOLS as u32,
            rank: allocate::MAX_SLOTS as u32,
            outside_candidate: true,
        };
        let ((), peak) = crate::test_heap::measure(|| {
            row_line(
                &mut line,
                u64::MAX,
                &[0xab; 32],
                u64::from(u32::MAX) - 1,
                &[3; MAX_SETS],
                &[u32::MAX; MAX_SETS],
                Some((allocate::MAX_SLOTS as usize - 1, &slot)),
            )
            .unwrap();
        });
        assert_eq!(peak, 0, "writing a row allocated {peak} bytes");
        assert!(line.len() <= MAX_ROW_BYTES, "{} bytes", line.len());
        let row: Row<'_> = serde_json::from_slice(&line).unwrap();
        assert_eq!(row.r.3, "3".repeat(MAX_SETS));
        // A header of sixteen sets, each with a long producer description.
        let set = json!({
            "handle": "set-0123456789abcdef0123456789abcdef", "kind": "selection",
            "producer": {"kind": "filter", "query": "q".repeat(512), "query_sha256": "0".repeat(64),
                         "syntax": "regex", "version": 1, "query_bytes": 4096,
                         "query_clipped": true, "case_insensitive": false},
            "origin": {"kind": "filter"}, "revision": "0".repeat(32), "parent": Value::Null,
            "inputs": vec!["view-0123456789abcdef0123456789abcdef"; 8], "values": true,
            "counts": {"unit": "distinct_originals", "parent": 1, "in": 1, "out": 0, "unknown": 0,
                       "retained_share": 1.0},
            "coverage": {"unit": "distinct_originals", "evaluated": 1, "not_evaluated": 0,
                         "reasons": [{"reason": "abstained", "distinct_originals": 1,
                                      "occurrences": 1}]},
        });
        let header = json!({"format": FORMAT, "sets": vec![set; MAX_SETS],
                            "contains": CONTAINS, "excludes": EXCLUDES});
        let text = header.to_string();
        let (parsed, peak) =
            crate::test_heap::measure(|| serde_json::from_slice::<Value>(text.as_bytes()).unwrap());
        assert_eq!(parsed, header);
        assert!(
            peak <= PARSE_FACTOR * text.len() as u64,
            "a header of {} bytes parsed into {peak} bytes",
            text.len()
        );
    }

    #[test]
    fn a_row_line_is_read_without_an_allocation() {
        let line = format!(
            "{{\"r\":[\"r-0000000100000002\",\"{}\",3,\"0123\",\"3f800000\",[0,1,2,1]]}}",
            "ab".repeat(32)
        );
        let (row, peak) = crate::test_heap::measure(|| {
            let row: Row<'_> = serde_json::from_slice(line.as_bytes()).unwrap();
            row.r.2 + u64::from(row.r.5.unwrap().2)
        });
        assert_eq!(row, 5);
        assert_eq!(peak, 0, "reading a row allocated {peak} bytes");
        let mut out = [0u8; 8];
        assert_eq!(
            hex_into(&1.0f32.to_bits().to_be_bytes(), &mut out),
            "3f800000"
        );
    }
}
