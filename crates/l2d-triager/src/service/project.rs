//! Occurrence projections (triager task 3.1; requirements 1.3, 2.4, 4.3, 5.2, 6.1, 6.3
//! and 6.4): the declared timestamp and field values of every occurrence of a snapshot,
//! repeats included, kept beside the relation so that field and time drill-down reads
//! committed tables only.
//!
//! A scope keeps the first position and the exact count of an original. A projection
//! is one more admitted pass over the source (design.md, "What the foundation already
//! supplies"): the archive's reader hands every accepted occurrence to an observer
//! before identical texts are collapsed (`ZipBatches::observe`), and each occurrence is
//! attributed to its original by its native identity. The pass runs when a client
//! declares projections at `triager_open`, or as an explicit re-scan of an open
//! snapshot (`triager_open` with `snapshot`). Nothing else reads the source for it.
//!
//! # What an occurrence carries
//!
//! - `source`: the archive entry it was read from, always;
//! - each declared field: the value of Splunk-style automatic `key=value` extraction
//!   from the record's text (`from: "kv"`, the semantics of the terminal search,
//!   `flow::query::field::kv_value`), or the first capture group of a regular
//!   expression over the entry path (`from: "entry"`);
//! - the declared timestamp: the record's leading ISO 8601 timestamp (`from: "text"`),
//!   the value of a `key=value` field (`from: "kv"`) or a capture of the entry path
//!   (`from: "entry"`), read as ISO 8601 with `Z` or an offset (`parse_iso`, as the
//!   terminal's `_time`) or, with `format: "date"`, as a `YYYY-MM-DD` day at 00:00 UTC.
//!   A timestamp without a zone is not trusted: it is counted as invalid.
//!
//! Text-derived values are the same for every occurrence of an original; entry-derived
//! values differ when one text was read from several entries. Missing values (no key,
//! no capture), invalid values (a timestamp that does not read, a value longer than
//! [`MAX_VALUE_BYTES`]) and values beyond the retained limit of a field (`max_values`,
//! the first values seen in read order) are counted apart and never guessed.
//!
//! # The table
//!
//! One fixed-width record per occurrence: the row of its original, its packed position,
//! its time and the value IDs of its fields, sorted by row and position
//! (`crate::extsort`). A table is in memory when the budget admits it and the host did
//! not place tables on disk, and in the work directory otherwise; both are read in
//! blocks of [`READER_RECORDS`]. The order does not depend on the batch size, on how
//! many entries were read at once or on where deduplication went, so neither does
//! anything computed from it.
//!
//! # One projection a snapshot
//!
//! A re-scan keeps every declaration of the snapshot's latest projection and adds the
//! new ones; a declaration under a name that is declared otherwise is refused. Field
//! and time views are evaluated against the latest projection, which declares all a
//! view's filter names. A projection is bound to its snapshot: another source,
//! revision, entry filter or reader setting is another snapshot with no projection.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::sync::Arc;
use std::time::Instant;

use l2d_core::marks::{REGEX_DFA_SIZE_LIMIT, REGEX_HELD_BYTES, REGEX_SIZE_LIMIT};
use l2d_core::{ArchivePosition, L2dError};
use regex::{Regex, RegexBuilder};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::ops::{
    ARTIFACT_FIXED_BYTES, Commit, Cost, Done, OpenTask, SCAN_FIXED_BYTES, Worker, registry,
    scan_with, tables_on_disk,
};
use super::state::{Population, Producer, Shared, Snapshot, source_not_retained};
use super::tables::{SetDir, Table, require_disk};
use super::wire::{self, Fault, VERSION};
use crate::extsort::{sort_file, sort_memory};
use crate::flow::query::field::{kv_value, parse_iso};

/// The most declared fields of a projection, beside `source`.
pub(crate) const MAX_FIELDS: usize = 8;
/// The most regular expressions over entry paths a projection declares.
pub(crate) const MAX_CAPTURES: usize = 4;
/// The most bytes of one capture expression.
pub(crate) const MAX_CAPTURE_BYTES: usize = 256;
/// Values a field retains when the request names no limit, and the most it may name.
pub(crate) const DEFAULT_MAX_VALUES: u64 = 10_000;
pub(crate) const MAX_MAX_VALUES: u64 = 100_000;
/// The most bytes of one field value; a longer one is counted as invalid.
pub(crate) const MAX_VALUE_BYTES: usize = 256;
/// The built-in field of every occurrence: the archive entry it was read from.
pub(crate) const SOURCE_FIELD: &str = "source";
/// The revision of the extraction rules: part of what a projection binds.
pub(crate) const EXTRACTOR_REVISION: &str = "triager-projection-v1 (kv: flow::query::field::kv_value; time: iso8601 with zone, date as 00:00 UTC)";
/// The most clauses of a field or time view's occurrence filter: one a drill.
pub(crate) const MAX_CLAUSES: usize = 16;

/// The records a table is read through at a time.
pub(crate) const READER_RECORDS: usize = 4_096;
/// What a record takes beside its value IDs: row, position and time.
const RECORD_FIXED_BYTES: usize = 24;
/// The widest record.
pub(crate) const MAX_RECORD_BYTES: usize = RECORD_FIXED_BYTES + 4 * MAX_FIELDS;
/// The buffer a table is read through.
pub(crate) const READER_BYTES: u64 = (READER_RECORDS * MAX_RECORD_BYTES) as u64;
/// The buffer the spool of a projection is written through.
const SPOOL_BUFFER_BYTES: u64 = 64 * 1024;
/// What a build holds beside its shares: the value IDs of the current entry, the record
/// on its way, the digest and the result.
const BUILD_FIXED_BYTES: u64 = 256 * 1024;

pub(crate) const TIME_MISSING: i64 = i64::MIN;
pub(crate) const TIME_INVALID: i64 = i64::MIN + 1;
pub(crate) const VALUE_MISSING: u32 = u32::MAX;
pub(crate) const VALUE_OVERFLOW: u32 = u32::MAX - 1;
pub(crate) const VALUE_INVALID: u32 = u32::MAX - 2;

// --- Declarations ----------------------------------------------------------------------

/// Where a field's value comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FieldFrom {
    /// Automatic `key=value` extraction from the record's text, the key being the name.
    Kv,
    /// The first capture group (or the whole match) of a regular expression over the
    /// entry path.
    Entry { capture: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FieldSpec {
    pub name: String,
    pub from: FieldFrom,
}

impl FieldSpec {
    pub fn describe(&self) -> Value {
        match &self.from {
            FieldFrom::Kv => json!({"name": self.name, "from": "kv"}),
            FieldFrom::Entry { capture } => {
                json!({"name": self.name, "from": "entry", "capture": capture})
            }
        }
    }

    /// Whether its value is the same for every occurrence of an original.
    pub fn per_original(&self) -> bool {
        matches!(self.from, FieldFrom::Kv)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TimeFrom {
    /// The record's leading timestamp.
    Text,
    Kv {
        key: String,
    },
    Entry {
        capture: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimeFormat {
    /// ISO 8601 with `Z` or an offset, as the terminal's `_time`.
    Iso8601,
    /// `YYYY-MM-DD`, read as 00:00 UTC.
    Date,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TimeSpec {
    pub from: TimeFrom,
    pub format: TimeFormat,
}

impl TimeSpec {
    pub fn describe(&self) -> Value {
        let format = match self.format {
            TimeFormat::Iso8601 => "iso8601",
            TimeFormat::Date => "date",
        };
        let mut described = match &self.from {
            TimeFrom::Text => json!({"from": "text"}),
            TimeFrom::Kv { key } => json!({"from": "kv", "key": key}),
            TimeFrom::Entry { capture } => json!({"from": "entry", "capture": capture}),
        };
        described["format"] = json!(format);
        described["timezone"] = json!("UTC");
        described["field"] = json!("_time");
        described
    }
}

/// What a projection declares: fields in the order of their names, the timestamp and
/// the values a field retains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Spec {
    pub fields: Vec<FieldSpec>,
    pub time: Option<TimeSpec>,
    pub max_values: u64,
}

fn usable_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
}

fn only(
    object: &serde_json::Map<String, Value>,
    known: &[&str],
    within: &str,
) -> Result<(), Fault> {
    match object.keys().find(|key| !known.contains(&key.as_str())) {
        None => Ok(()),
        Some(key) => Err(Fault::invalid(format!(
            "`{within}.{}` is no argument here; the arguments are {}",
            wire::clip(key, 64).0,
            known.join(", ")
        ))),
    }
}

fn capture_of(object: &serde_json::Map<String, Value>, within: &str) -> Result<String, Fault> {
    match object.get("capture").and_then(Value::as_str) {
        Some(capture) if !capture.is_empty() && capture.len() <= MAX_CAPTURE_BYTES => {
            Ok(capture.to_string())
        }
        _ => Err(Fault::invalid(format!(
            "`{within}.capture` is a regular expression over the entry path of 1 to \
             {MAX_CAPTURE_BYTES} bytes; its first group, or the whole match, is the value"
        ))),
    }
}

impl Spec {
    /// The declarations of a request: `{"fields": [...], "time": {...}, "max_values": N}`.
    /// Regular expressions are compiled at the start of the job, where the budget is.
    pub fn parse(value: &Value) -> Result<Self, Fault> {
        let object = value.as_object().ok_or_else(|| {
            Fault::invalid("`projections` is an object with `fields`, `time` and `max_values`")
        })?;
        only(object, &["fields", "time", "max_values"], "projections")?;
        let mut fields = Vec::new();
        if let Some(listed) = object.get("fields") {
            let listed = listed
                .as_array()
                .filter(|listed| listed.len() <= MAX_FIELDS)
                .ok_or_else(|| {
                    Fault::invalid(format!(
                        "`projections.fields` is a list of at most {MAX_FIELDS} fields"
                    ))
                })?;
            for field in listed {
                let field = field.as_object().ok_or_else(|| {
                    Fault::invalid(
                        "a field of `projections.fields` is an object with `name` and `from`",
                    )
                })?;
                only(field, &["name", "from", "capture"], "projections.fields[]")?;
                let name = field
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| usable_name(name))
                    .ok_or_else(|| {
                        Fault::invalid("a field name is 1 to 64 letters, digits, '_', '.' or '-'")
                    })?;
                if [SOURCE_FIELD, "_time", "_raw", "id"].contains(&name) {
                    return Err(Fault::invalid(format!(
                        "`{name}` is a built-in field and is not declared: `source` is the \
                         entry of every occurrence, and the timestamp is declared as `time`"
                    )));
                }
                let from = match field.get("from").and_then(Value::as_str) {
                    Some("kv") if !field.contains_key("capture") => FieldFrom::Kv,
                    Some("entry") => FieldFrom::Entry {
                        capture: capture_of(field, "projections.fields[]")?,
                    },
                    _ => {
                        return Err(Fault::invalid(
                            "a field's `from` is `kv` (the record's `key=value`, keyed by the \
                             name) or `entry` (with `capture`, over the entry path)",
                        ));
                    }
                };
                if fields.iter().any(|known: &FieldSpec| known.name == name) {
                    return Err(Fault::invalid(format!(
                        "the field `{name}` is declared twice"
                    )));
                }
                fields.push(FieldSpec {
                    name: name.to_string(),
                    from,
                });
            }
        }
        let time = match object.get("time") {
            None => None,
            Some(time) => {
                let time = time
                    .as_object()
                    .ok_or_else(|| Fault::invalid("`projections.time` is an object with `from`"))?;
                only(
                    time,
                    &["from", "key", "capture", "format", "timezone"],
                    "projections.time",
                )?;
                let from = match time.get("from").and_then(Value::as_str) {
                    Some("text") if !time.contains_key("key") && !time.contains_key("capture") => {
                        TimeFrom::Text
                    }
                    Some("kv") if !time.contains_key("capture") => TimeFrom::Kv {
                        key: time
                            .get("key")
                            .and_then(Value::as_str)
                            .filter(|key| usable_name(key))
                            .ok_or_else(|| {
                                Fault::invalid(
                                    "`projections.time.key` is the key of a `key=value` field",
                                )
                            })?
                            .to_string(),
                    },
                    Some("entry") if !time.contains_key("key") => TimeFrom::Entry {
                        capture: capture_of(time, "projections.time")?,
                    },
                    _ => {
                        return Err(Fault::invalid(
                            "`projections.time.from` is `text` (the leading timestamp), `kv` \
                             (with `key`) or `entry` (with `capture`)",
                        ));
                    }
                };
                let format = match time.get("format").and_then(Value::as_str) {
                    None | Some("iso8601") => TimeFormat::Iso8601,
                    Some("date") => TimeFormat::Date,
                    Some(_) => {
                        return Err(Fault::invalid(
                            "`projections.time.format` is `iso8601` (with `Z` or an offset) or \
                             `date` (`YYYY-MM-DD`, read as 00:00 UTC)",
                        ));
                    }
                };
                match time.get("timezone") {
                    None => {}
                    Some(zone) if zone == "UTC" => {}
                    Some(_) => {
                        return Err(Fault::invalid(
                            "`projections.time.timezone` is `UTC`: timestamps are read with \
                             their own zone and bucketed in UTC",
                        ));
                    }
                }
                Some(TimeSpec { from, format })
            }
        };
        let max_values = match object.get("max_values") {
            None => DEFAULT_MAX_VALUES,
            Some(limit) => limit
                .as_u64()
                .filter(|limit| (1..=MAX_MAX_VALUES).contains(limit))
                .ok_or_else(|| {
                    Fault::invalid(format!(
                        "`projections.max_values` is a whole number from 1 to {MAX_MAX_VALUES}"
                    ))
                })?,
        };
        if fields.is_empty() && time.is_none() {
            return Err(Fault::invalid(
                "`projections` declares at least one field or the timestamp",
            ));
        }
        let spec = Self::ordered(fields, time, max_values);
        spec.check()?;
        Ok(spec)
    }

    fn ordered(mut fields: Vec<FieldSpec>, time: Option<TimeSpec>, max_values: u64) -> Self {
        fields.sort_by(|a, b| a.name.cmp(&b.name));
        Self {
            fields,
            time,
            max_values,
        }
    }

    fn captures(&self) -> usize {
        self.fields
            .iter()
            .filter(|field| matches!(field.from, FieldFrom::Entry { .. }))
            .count()
            + usize::from(matches!(
                self.time,
                Some(TimeSpec {
                    from: TimeFrom::Entry { .. },
                    ..
                })
            ))
    }

    fn check(&self) -> Result<(), Fault> {
        if self.fields.len() > MAX_FIELDS {
            return Err(Fault::invalid(format!(
                "a projection declares at most {MAX_FIELDS} fields, those of the snapshot's \
                 projection included; this one would declare {}",
                self.fields.len()
            )));
        }
        if self.captures() > MAX_CAPTURES {
            return Err(Fault::invalid(format!(
                "a projection declares at most {MAX_CAPTURES} expressions over entry paths, \
                 those of the snapshot's projection included; this one would declare {}",
                self.captures()
            )));
        }
        Ok(())
    }

    /// This request with every declaration of `current` kept: the projection of a
    /// re-scan declares all the snapshot's latest one did. A name declared otherwise is
    /// refused with both declarations.
    pub fn merged(&self, current: Option<&Spec>) -> Result<Self, Fault> {
        let Some(current) = current else {
            return Ok(self.clone());
        };
        let mut fields = current.fields.clone();
        for field in &self.fields {
            match fields.iter().find(|known| known.name == field.name) {
                Some(known) if known == field => {}
                Some(known) => {
                    return Err(Fault::new(
                        "field_defined",
                        format!(
                            "the snapshot's projection declares `{}` as {}; this request \
                             declares it as {}. A projection keeps every declaration of the one \
                             before it: give the field another name",
                            field.name,
                            known.describe(),
                            field.describe()
                        ),
                    ));
                }
                None => fields.push(field.clone()),
            }
        }
        let time = match (&current.time, &self.time) {
            (Some(known), Some(asked)) if known != asked => {
                return Err(Fault::new(
                    "time_defined",
                    format!(
                        "the snapshot's projection declares the timestamp as {}; this request \
                         declares {}. A snapshot has one timestamp declaration",
                        known.describe(),
                        asked.describe()
                    ),
                ));
            }
            (Some(known), _) => Some(known.clone()),
            (None, asked) => asked.clone(),
        };
        let merged = Self::ordered(fields, time, current.max_values.max(self.max_values));
        merged.check()?;
        Ok(merged)
    }

    /// The declarations as results state them and as the handle binds them.
    pub fn describe(&self) -> Value {
        json!({
            "fields": self.fields.iter().map(FieldSpec::describe).collect::<Vec<_>>(),
            "time": self.time.as_ref().map(TimeSpec::describe),
            "max_values": self.max_values,
            "built_in": [SOURCE_FIELD],
            "extractor": EXTRACTOR_REVISION,
        })
    }

    /// The bytes of one record of a table with these fields.
    pub fn width(&self) -> usize {
        RECORD_FIXED_BYTES + 4 * self.fields.len()
    }

    pub fn field(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|field| field.name == name)
    }

    /// Whether this declares all `other` declares.
    pub fn covers(&self, other: &Spec) -> bool {
        other.fields.iter().all(|field| self.fields.contains(field))
            && (other.time.is_none() || other.time == self.time)
    }

    fn held_bytes(&self) -> u64 {
        16 * self.describe().to_string().len() as u64
    }
}

/// What a field's values may take in memory while a projection is built: the map that
/// interns them, sized for `max_values` before the read, the values themselves at their
/// longest, the two lists they are put in order through (16 bytes a value each) and
/// whether each reads as a number.
pub(crate) fn dictionary_bytes(max_values: u64) -> u64 {
    // The map's table: a power of two of buckets of 24 bytes and a control byte, at
    // least 8/7 of the entries it holds, and its group of control bytes.
    let buckets = (max_values.max(4) * 8 / 7 + 1).next_power_of_two();
    buckets * 25 + 64 + max_values * (MAX_VALUE_BYTES as u64 + 2 * 16 + 1)
}

// --- The projection --------------------------------------------------------------------

/// The values a field retained, by ID, and whether each reads as a number.
pub(crate) struct Dictionary {
    pub values: Vec<Box<str>>,
    pub numeric: Vec<bool>,
    /// Whether an occurrence had a value beyond the retained limit.
    pub overflowed: bool,
}

impl Dictionary {
    pub fn held_bytes(&self) -> u64 {
        (self.values.capacity() * std::mem::size_of::<Box<str>>()
            + self.values.iter().map(|value| value.len()).sum::<usize>()
            + self.numeric.capacity()) as u64
    }

    pub fn find(&self, value: &str) -> Option<u32> {
        self.values
            .iter()
            .position(|known| &**known == value)
            .map(|at| at as u32)
    }
}

/// One immutable occurrence projection of a snapshot.
pub(crate) struct Projection {
    pub handle: String,
    pub snapshot: Arc<Snapshot>,
    pub spec: Spec,
    /// A digest of the table's records: the same for the same occurrences, whatever
    /// the batch size or the placement.
    pub revision: String,
    pub dictionaries: Vec<Dictionary>,
    pub table: Table,
    pub records: u64,
    pub width: usize,
    pub result: String,
    pub held: u64,
}

impl Projection {
    /// The value of field `field` with ID `id`, or the class it stands for.
    pub fn value(&self, field: usize, id: u32) -> Option<&str> {
        self.dictionaries[field]
            .values
            .get(id as usize)
            .map(|value| &**value)
    }

    /// The path of the archive entry with index `entry`.
    pub fn entry_path(&self, entry: u32) -> &str {
        self.snapshot
            .summary
            .entries
            .get(entry as usize)
            .map_or("", |report| report.path.as_str())
    }

    pub fn entries(&self) -> usize {
        self.snapshot.summary.entries.len()
    }

    /// The first record of the occurrences of `row`, or where they would be.
    pub fn first_of_row(&self, row: u64) -> Result<u64, L2dError> {
        let (mut low, mut high) = (0u64, self.records);
        let mut cell = [0u8; 8];
        while low < high {
            let middle = low + (high - low) / 2;
            self.table.read(middle * self.width as u64, &mut cell)?;
            if u64::from_le_bytes(cell) < row {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        Ok(low)
    }

    /// What the result names of the table.
    pub fn describe(&self) -> Value {
        json!({
            "projection": self.handle,
            "revision": self.revision,
            "declared": self.spec.describe(),
            "occurrences": self.records,
            "table": {
                "on_disk": self.table.is_on_disk(),
                "bytes": self.table.byte_len(),
                "record_bytes": self.width,
            },
        })
    }
}

/// One occurrence as a table record holds it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Occurrence {
    pub row: u64,
    pub position: u64,
    pub time: i64,
    pub ids: [u32; MAX_FIELDS],
}

impl Occurrence {
    pub fn entry(&self) -> u32 {
        ArchivePosition::unpack(self.position).entry
    }

    pub fn line(&self) -> u64 {
        ArchivePosition::unpack(self.position).line
    }

    pub fn has_time(&self) -> bool {
        self.time > TIME_INVALID
    }

    pub(crate) fn decode(record: &[u8], fields: usize) -> Self {
        let word = |at: usize| u64::from_le_bytes(record[at..at + 8].try_into().expect("8 bytes"));
        let mut ids = [VALUE_MISSING; MAX_FIELDS];
        for (field, id) in ids.iter_mut().enumerate().take(fields) {
            let at = RECORD_FIXED_BYTES + 4 * field;
            *id = u32::from_le_bytes(record[at..at + 4].try_into().expect("4 bytes"));
        }
        Self {
            row: word(0),
            position: word(8),
            time: word(16) as i64,
            ids,
        }
    }

    fn encode(&self, fields: usize, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.row.to_le_bytes());
        out[8..16].copy_from_slice(&self.position.to_le_bytes());
        out[16..24].copy_from_slice(&self.time.to_le_bytes());
        for field in 0..fields {
            let at = RECORD_FIXED_BYTES + 4 * field;
            out[at..at + 4].copy_from_slice(&self.ids[field].to_le_bytes());
        }
    }
}

/// Reads the records of a table in order, [`READER_RECORDS`] at a time through a
/// buffer of at most [`READER_BYTES`], which the caller holds in the budget.
pub(crate) struct Reader<'p> {
    projection: &'p Projection,
    buffer: Vec<u8>,
    next: u64,
    at: usize,
    filled: usize,
    /// The records read so far.
    pub read: u64,
}

impl<'p> Reader<'p> {
    pub fn new(projection: &'p Projection, from: u64) -> Self {
        Self {
            projection,
            buffer: vec![0u8; READER_RECORDS * projection.width],
            next: from,
            at: 0,
            filled: 0,
            read: 0,
        }
    }

    pub fn next(&mut self) -> Result<Option<Occurrence>, L2dError> {
        let width = self.projection.width;
        if self.at == self.filled {
            let left = self.projection.records.saturating_sub(self.next);
            if left == 0 {
                return Ok(None);
            }
            let records = left.min(READER_RECORDS as u64) as usize;
            self.projection.table.read(
                self.next * width as u64,
                &mut self.buffer[..records * width],
            )?;
            self.next += records as u64;
            self.at = 0;
            self.filled = records * width;
        }
        let occurrence = Occurrence::decode(
            &self.buffer[self.at..self.at + width],
            self.projection.spec.fields.len(),
        );
        self.at += width;
        self.read += 1;
        Ok(Some(occurrence))
    }
}

// --- Occurrence filters ----------------------------------------------------------------

/// One drill of a field or time view: what its occurrences must be.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Clause {
    /// A declared field with this value, or with none.
    Field {
        field: FieldSpec,
        value: Option<String>,
    },
    /// Read from this archive entry.
    Source { value: String },
    /// A valid timestamp in `[start, end)`, in seconds since 1970 UTC.
    Time {
        time: TimeSpec,
        start: i64,
        end: i64,
    },
    /// No valid timestamp: missing or unreadable.
    NoTime { time: TimeSpec },
}

impl Clause {
    pub fn describe(&self) -> Value {
        match self {
            Clause::Field { field, value } => match value {
                Some(value) => json!({"kind": "field", "field": field.name, "value": value,
                                      "definition": field.describe()}),
                None => json!({"kind": "field", "field": field.name, "missing": true,
                               "definition": field.describe()}),
            },
            Clause::Source { value } => {
                json!({"kind": "field", "field": SOURCE_FIELD, "value": value})
            }
            Clause::Time { time, start, end } => json!({
                "kind": "time", "start": iso(*start), "end": iso(*end),
                "definition": time.describe()}),
            Clause::NoTime { time } => {
                json!({"kind": "time", "no_time": true, "definition": time.describe()})
            }
        }
    }

    fn held_bytes(&self) -> u64 {
        16 * self.describe().to_string().len() as u64
    }

    /// What a projection must declare to evaluate it.
    fn needs(&self) -> Spec {
        let (fields, time) = match self {
            Clause::Field { field, .. } => (vec![field.clone()], None),
            Clause::Source { .. } => (Vec::new(), None),
            Clause::Time { time, .. } | Clause::NoTime { time } => (Vec::new(), Some(time.clone())),
        };
        Spec {
            fields,
            time,
            max_values: 1,
        }
    }
}

/// The occurrences a population counts: all of its members', or, for a field or time
/// view, those that pass every clause of the drills that made it.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Filter {
    pub clauses: Vec<Clause>,
}

impl Filter {
    /// The filter of a population.
    pub fn of(population: &Population) -> Filter {
        match population.as_set().map(|set| &set.producer) {
            Some(Producer::Occurrence { filter, .. }) => (**filter).clone(),
            _ => Filter::default(),
        }
    }

    pub fn describe(&self) -> Value {
        Value::Array(self.clauses.iter().map(Clause::describe).collect())
    }

    pub fn held_bytes(&self) -> u64 {
        self.clauses.iter().map(Clause::held_bytes).sum::<u64>()
            + (self.clauses.capacity() * std::mem::size_of::<Clause>()) as u64
    }

    /// What a projection must declare to evaluate the filter.
    pub fn needs(&self) -> Spec {
        let mut needs = Spec {
            fields: Vec::new(),
            time: None,
            max_values: 1,
        };
        for clause in &self.clauses {
            let one = clause.needs();
            for field in one.fields {
                if !needs.fields.contains(&field) {
                    needs.fields.push(field);
                }
            }
            if one.time.is_some() {
                needs.time = one.time;
            }
        }
        needs
    }
}

enum Test {
    Value { field: usize, id: u32 },
    Entry(u32),
    Never,
    Time { start: i64, end: i64 },
    NoTime,
}

/// A filter resolved against one projection.
pub(crate) struct Resolved {
    tests: Vec<Test>,
}

impl Resolved {
    pub fn passes(&self, occurrence: &Occurrence) -> bool {
        self.tests.iter().all(|test| match test {
            Test::Value { field, id } => occurrence.ids[*field] == *id,
            Test::Entry(entry) => occurrence.entry() == *entry,
            Test::Never => false,
            Test::Time { start, end } => {
                occurrence.has_time() && occurrence.time >= *start && occurrence.time < *end
            }
            Test::NoTime => !occurrence.has_time(),
        })
    }
}

/// The refusal of a value a field did not retain: it may lie among the values beyond
/// the limit, which are counted but not told apart.
fn not_retained(field: &str, value: &str, max_values: u64) -> Fault {
    Fault::new(
        "value_not_retained",
        format!(
            "the field `{field}` retained its first {max_values} values in read order and \
             counted the rest as `overflow`; `{}` is not among the retained ones, so which \
             occurrences hold it is not known. Re-scan with a larger `max_values` (at most \
             {MAX_MAX_VALUES}) to retain it",
            wire::display(value, 128).0
        ),
    )
}

/// `filter` as tests over the records of `projection`, which declares all it needs.
pub(crate) fn resolve(filter: &Filter, projection: &Projection) -> Result<Resolved, Fault> {
    let mut tests = Vec::with_capacity(filter.clauses.len());
    for clause in &filter.clauses {
        tests.push(match clause {
            Clause::Field { field, value } => {
                let index = projection
                    .spec
                    .fields
                    .iter()
                    .position(|known| known == field)
                    .ok_or_else(|| {
                        Fault::runtime("a field view names a field its projection does not declare")
                    })?;
                match value {
                    None => Test::Value {
                        field: index,
                        id: VALUE_MISSING,
                    },
                    Some(value) => match projection.dictionaries[index].find(value) {
                        Some(id) => Test::Value { field: index, id },
                        None if projection.dictionaries[index].overflowed => {
                            return Err(not_retained(
                                &field.name,
                                value,
                                projection.spec.max_values,
                            ));
                        }
                        None => Test::Never,
                    },
                }
            }
            Clause::Source { value } => (0..projection.entries() as u32)
                .find(|entry| projection.entry_path(*entry) == value)
                .map_or(Test::Never, Test::Entry),
            Clause::Time { time, start, end } => {
                if projection.spec.time.as_ref() != Some(time) {
                    return Err(Fault::runtime(
                        "a time view names a timestamp its projection does not declare",
                    ));
                }
                Test::Time {
                    start: *start,
                    end: *end,
                }
            }
            Clause::NoTime { time } => {
                if projection.spec.time.as_ref() != Some(time) {
                    return Err(Fault::runtime(
                        "a time view names a timestamp its projection does not declare",
                    ));
                }
                Test::NoTime
            }
        });
    }
    Ok(Resolved { tests })
}

/// The latest projection of `snapshot`, if it declares all `needs`.
pub(crate) fn covering(
    shared: &Shared,
    snapshot: &Snapshot,
    needs: &Spec,
) -> Option<Arc<Projection>> {
    registry(shared)
        .latest_projection(&snapshot.handle)
        .filter(|projection| projection.spec.covers(needs))
}

// --- Times ------------------------------------------------------------------------------

fn days_to_civil(days: i64) -> (i64, i64, i64) {
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Seconds since 1970 as ISO 8601 in UTC.
pub(crate) fn iso(seconds: i64) -> String {
    let days = seconds.div_euclid(86_400);
    let within = seconds.rem_euclid(86_400);
    let (year, month, day) = days_to_civil(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        within / 3_600,
        within % 3_600 / 60,
        within % 60
    )
}

/// A timestamp of a request (ISO 8601 with a zone), in whole seconds.
pub(crate) fn parse_instant(text: &str) -> Option<i64> {
    parse_iso(text).map(|seconds| seconds.floor() as i64)
}

fn read_time(token: &str, format: TimeFormat) -> Option<i64> {
    match format {
        TimeFormat::Iso8601 => parse_iso(token).map(|seconds| seconds.floor() as i64),
        TimeFormat::Date => {
            let bytes = token.as_bytes();
            if bytes.len() != 10 {
                return None;
            }
            let mut composed = [0u8; 20];
            composed[..10].copy_from_slice(bytes);
            composed[10..].copy_from_slice(b"T00:00:00Z");
            parse_iso(std::str::from_utf8(&composed).ok()?).map(|seconds| seconds as i64)
        }
    }
}

/// Whether `text` begins like a date: four digits and a '-'.
fn looks_dated(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() >= 5 && bytes[..4].iter().all(u8::is_ascii_digit) && bytes[4] == b'-'
}

/// The leading timestamp of a record: its first token, or its first two when a date
/// and a time are written apart, optionally in brackets. Missing when the record does
/// not begin like a date, invalid when it does and does not read.
pub(crate) fn leading_time(text: &str, format: TimeFormat) -> i64 {
    let text = text.strip_prefix('[').unwrap_or(text);
    if !looks_dated(text) {
        return TIME_MISSING;
    }
    let stop = |from: usize| {
        text[from..]
            .find([' ', '\t', ']'])
            .map_or(text.len(), |at| from + at)
    };
    let first = stop(0);
    if let Some(time) = read_time(&text[..first], format) {
        return time;
    }
    if text.as_bytes().get(first) == Some(&b' ') {
        let second = stop(first + 1);
        if let Some(time) = read_time(&text[..second], format) {
            return time;
        }
    }
    TIME_INVALID
}

fn value_time(value: Option<&str>, format: TimeFormat) -> i64 {
    match value {
        None => TIME_MISSING,
        Some(value) => read_time(value, format).unwrap_or(TIME_INVALID),
    }
}

// --- Building a projection --------------------------------------------------------------

/// Where a projection's snapshot comes from: an open in the same job, or a snapshot
/// that is open already (an explicit re-scan).
pub(crate) enum ProjectSource {
    Open(OpenTask),
    Snapshot(Arc<Snapshot>),
}

pub(crate) struct ProjectTask {
    pub source: ProjectSource,
    pub spec: Spec,
}

/// Interns the values of one field during a build.
struct Interner {
    map: HashMap<Box<str>, u32>,
    order: Vec<Box<str>>,
    most: usize,
    overflowed: bool,
}

impl Interner {
    fn new(most: u64) -> Self {
        Self {
            map: HashMap::with_capacity(most as usize),
            order: Vec::new(),
            most: most as usize,
            overflowed: false,
        }
    }

    fn id(&mut self, value: Option<&str>) -> u32 {
        let Some(value) = value else {
            return VALUE_MISSING;
        };
        if value.len() > MAX_VALUE_BYTES {
            return VALUE_INVALID;
        }
        if let Some(id) = self.map.get(value) {
            return *id;
        }
        if self.map.len() == self.most {
            self.overflowed = true;
            return VALUE_OVERFLOW;
        }
        let id = self.map.len() as u32;
        self.map.insert(Box::from(value), id);
        id
    }

    /// The values by ID; the map is let go as they are taken out of it.
    fn finish(mut self) -> Dictionary {
        let mut values: Vec<Option<Box<str>>> = Vec::new();
        values.resize_with(self.map.len(), || None);
        for (value, id) in self.map.drain() {
            values[id as usize] = Some(value);
        }
        drop(self.map);
        self.order = values.into_iter().flatten().collect();
        let numeric = self
            .order
            .iter()
            .map(|value| crate::flow::query::field::parse_number(value).is_some())
            .collect();
        Dictionary {
            values: self.order,
            numeric,
            overflowed: self.overflowed,
        }
    }
}

fn compile(capture: &str) -> Result<Regex, Fault> {
    RegexBuilder::new(capture)
        .size_limit(REGEX_SIZE_LIMIT)
        .dfa_size_limit(REGEX_DFA_SIZE_LIMIT)
        .build()
        .map_err(|error| {
            Fault::invalid(format!(
                "the capture `{}` cannot be used: {}",
                wire::display(capture, 120).0,
                wire::display(&error.to_string(), 300).0
            ))
        })
}

fn captured<'t>(regex: &Regex, text: &'t str) -> Option<&'t str> {
    let found = regex.captures(text)?;
    found
        .get(1)
        .or_else(|| found.get(0))
        .map(|part| part.as_str())
}

fn io_failed(doing: &str, path: &std::path::Path, error: &dyn std::fmt::Display) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} {} failed: {error}", path.display()),
    }
}

/// What an extraction is, once its expressions are compiled.
struct Extractor<'s> {
    spec: &'s Spec,
    fields: Vec<(Option<Regex>, Interner)>,
    time: Option<(TimeFormat, Option<Regex>)>,
    /// The entry whose entry-derived values are below, and those values.
    entry: Option<u32>,
    entry_ids: [u32; MAX_FIELDS],
    entry_time: i64,
}

impl<'s> Extractor<'s> {
    fn new(spec: &'s Spec) -> Result<Self, Fault> {
        let mut fields = Vec::with_capacity(spec.fields.len());
        for field in &spec.fields {
            let regex = match &field.from {
                FieldFrom::Kv => None,
                FieldFrom::Entry { capture } => Some(compile(capture)?),
            };
            fields.push((regex, Interner::new(spec.max_values)));
        }
        let time = match &spec.time {
            None => None,
            Some(TimeSpec {
                from: TimeFrom::Entry { capture },
                format,
            }) => Some((*format, Some(compile(capture)?))),
            Some(TimeSpec { format, .. }) => Some((*format, None)),
        };
        Ok(Self {
            spec,
            fields,
            time,
            entry: None,
            entry_ids: [VALUE_MISSING; MAX_FIELDS],
            entry_time: TIME_MISSING,
        })
    }

    fn occurrence(
        &mut self,
        row: u64,
        position: u64,
        text: &str,
        path: &dyn Fn(u32) -> String,
    ) -> Occurrence {
        let entry = ArchivePosition::unpack(position).entry;
        if self.entry != Some(entry) {
            // Entry-derived values are found once an entry.
            let path = path(entry);
            for (at, (regex, interner)) in self.fields.iter_mut().enumerate() {
                if let Some(regex) = regex {
                    self.entry_ids[at] = interner.id(captured(regex, &path));
                }
            }
            if let Some((format, Some(regex))) = &self.time {
                self.entry_time = value_time(captured(regex, &path), *format);
            }
            self.entry = Some(entry);
        }
        let mut ids = [VALUE_MISSING; MAX_FIELDS];
        for (at, ((regex, interner), field)) in
            self.fields.iter_mut().zip(&self.spec.fields).enumerate()
        {
            ids[at] = match regex {
                Some(_) => self.entry_ids[at],
                None => interner.id(kv_value(text, &field.name)),
            };
        }
        let time = match (&self.spec.time, &self.time) {
            (None, _) | (_, None) => TIME_MISSING,
            (Some(spec), Some((format, _))) => match &spec.from {
                TimeFrom::Text => leading_time(text, *format),
                TimeFrom::Kv { key } => value_time(kv_value(text, key), *format),
                TimeFrom::Entry { .. } => self.entry_time,
            },
        };
        Occurrence {
            row,
            position,
            time,
            ids,
        }
    }

    fn finish(self) -> Vec<Dictionary> {
        self.fields
            .into_iter()
            .map(|(_, interner)| interner.finish())
            .collect()
    }
}

/// What reading the source for a projection holds beside the dictionaries.
pub(crate) fn build_fixed_bytes(spec: &Spec) -> u64 {
    SCAN_FIXED_BYTES
        + spec.captures() as u64 * REGEX_HELD_BYTES
        + SPOOL_BUFFER_BYTES
        + BUILD_FIXED_BYTES
}

/// What the dictionaries of a build may take.
pub(crate) fn dictionaries_bytes(spec: &Spec) -> u64 {
    spec.fields.len() as u64 * dictionary_bytes(spec.max_values)
}

/// Builds the projection `spec` of `snapshot`, which the operation holds `beside` the
/// service's base for; `Ok(None)` when it was built before (`known`).
fn build(
    worker: &mut Worker<'_>,
    snapshot: &Arc<Snapshot>,
    requested: &Spec,
    beside: u64,
) -> Result<(Arc<Projection>, bool), Fault> {
    let started = Instant::now();
    source_not_retained(snapshot, "an occurrence projection")?;
    let current = registry(worker.shared).latest_projection(&snapshot.handle);
    let spec = requested.merged(current.as_ref().map(|projection| &projection.spec))?;
    let handle = wire::handle(
        "proj",
        &worker.shared.config.project,
        &[&snapshot.handle, &spec.describe().to_string()],
    );
    if let Some(known) = registry(worker.shared).projection(&handle) {
        return Ok((known, false));
    }
    let base = worker.base.saturating_add(beside);
    let width = spec.width();
    let records = snapshot.scope.occurrences();
    let table_bytes = records.saturating_mul(width as u64);
    let dictionaries = dictionaries_bytes(&spec);
    let fixed = build_fixed_bytes(&spec);
    worker.budget.hold(base);
    worker.budget.require_whole(
        dictionaries.saturating_add(fixed),
        "projecting occurrences: the values of every declared field at the retained limit, \
         the expressions over entry paths and the read of the source,",
    )?;
    let held = base.saturating_add(dictionaries).saturating_add(fixed);
    worker.budget.hold(held);
    // The spool, its runs and the ordered table: at most three times the table.
    let config = &worker.shared.config;
    std::fs::create_dir_all(&config.work_root)
        .map_err(|error| io_failed("creating", &config.work_root, &error))?;
    require_disk(
        &config.work_root,
        table_bytes.saturating_mul(3),
        config.free_disk_bytes,
        "projecting occurrences: the spool, its runs and the ordered table",
    )?;
    let dir = SetDir::create(worker.fresh_dir("projection"))?;
    let spool_path = dir.path().join("spool.bin");
    let table_name = "occurrences.bin";
    let table_path = dir.path().join(table_name);
    // Removed when the build ends, however it ends; the table is adopted below.
    let _spool = Removed(spool_path.clone());
    // The ordered table too, until a table adopts it.
    let mut unadopted = Removed(table_path.clone());
    let mut extractor = Extractor::new(&spec)?;
    let mut writer = File::create(&spool_path)
        .map(|file| BufWriter::with_capacity(SPOOL_BUFFER_BYTES as usize, file))
        .map_err(|error| io_failed("creating", &spool_path, &error))?;
    let mut written = 0u64;
    let mut record = [0u8; MAX_RECORD_BYTES];
    let fields = spec.fields.len();
    let summary = &snapshot.summary;
    let path = |entry: u32| -> String {
        summary
            .entries
            .get(entry as usize)
            .map_or_else(String::new, |report| report.path.clone())
    };
    let scope = &snapshot.scope;
    let mut observe = |position: u64, text: &str| -> Result<(), L2dError> {
        let row = scope.row_of_text(text)?.ok_or_else(|| L2dError::Refused {
            reason: "an occurrence of the source names no original of the snapshot; the source \
                     no longer reads as the snapshot bound it, open it again"
                .into(),
        })?;
        let occurrence = extractor.occurrence(row, position, text, &path);
        occurrence.encode(fields, &mut record);
        writer
            .write_all(&record[..width])
            .map_err(|error| io_failed("writing", &spool_path, &error))?;
        written += 1;
        Ok(())
    };
    let scanned = scan_with(
        worker,
        snapshot,
        held,
        &mut |_| Ok(true),
        Some(&mut observe),
    )?
    .ok_or_else(|| Fault::runtime("the read for a projection ended early"))?;
    writer
        .into_inner()
        .map_err(|error| io_failed("writing", &spool_path, &error.into_error()))?;
    if written != records {
        return Err(Fault::refused(format!(
            "the read for a projection gave {written} occurrences and the snapshot holds \
             {records}; the source no longer reads as the snapshot bound it, open it again"
        )));
    }
    let dictionaries = extractor.finish();
    let dictionary_held: u64 = dictionaries.iter().map(Dictionary::held_bytes).sum();
    // The read is over: the dictionaries are what stays of it.
    let held = base.saturating_add(dictionary_held);
    worker.budget.hold(held);
    worker.check_cancel()?;
    let sort = sort_memory(records, width, worker.budget.left_bytes());
    worker
        .budget
        .require_whole(sort, "putting the occurrences of a projection in order")?;
    worker.budget.hold(held.saturating_add(sort));
    let sorted = sort_file(&spool_path, &table_path, dir.path(), width, records, sort)?;
    worker.budget.hold(held);
    worker.check_cancel()?;
    // The table goes where the tables of sets go.
    let on_disk = tables_on_disk(worker, table_bytes);
    let (table, revision) = if on_disk {
        let table = Table::adopt(&dir, table_name, table_bytes)?;
        // The table removes its file from here on.
        unadopted.0 = std::path::PathBuf::new();
        let revision = digest_table(&table, width, records, worker, held)?;
        (table, revision)
    } else {
        worker.budget.hold(held.saturating_add(table_bytes));
        let mut bytes = vec![0u8; table_bytes as usize];
        File::open(&table_path)
            .and_then(|mut file| file.read_exact(&mut bytes))
            .map_err(|error| io_failed("reading", &table_path, &error))?;
        let _ = std::fs::remove_file(&table_path);
        let table = Table::from_bytes(bytes);
        let revision = digest_table(&table, width, records, worker, held + table_bytes)?;
        (table, revision)
    };
    let table_held = table.held_bytes();
    let held = held.saturating_add(table_held);
    worker.budget.hold(held);
    let mut projection = Projection {
        handle,
        snapshot: snapshot.clone(),
        spec,
        revision,
        dictionaries,
        table,
        records,
        width,
        result: String::new(),
        held: 0,
    };
    // What the snapshot's occurrences hold, as the fields and histogram views count
    // them: one more pass over the table, within what a view takes.
    let fields_summary = super::facets::summary_for(
        worker,
        &projection,
        &Population::All(snapshot.clone()),
        held,
    )?;
    let time_summary = super::facets::time_for(
        worker,
        &projection,
        &Population::All(snapshot.clone()),
        held,
    )?;
    let mut cost = Cost::new(started);
    cost.scanned(&scanned);
    cost.set("candidates", records)
        .set("output", records)
        .set("tables_on_disk", on_disk)
        .set("index_rows_read", records)
        .set("sort_runs", sorted.runs)
        .set("disk_bytes", projection.table.disk_bytes())
        .set(
            "disk_peak_bytes",
            table_bytes
                .saturating_mul(2)
                .saturating_add(sorted.run_bytes),
        );
    let mut result = projection.describe();
    result["v"] = json!(VERSION);
    result["status"] = json!("ready");
    result["snapshot"] = json!(snapshot.handle);
    result["distinct_originals"] = json!(snapshot.scope.originals());
    result["fields"] = fields_summary;
    result["time"] = time_summary;
    result["previous"] = json!(current.as_ref().map(|projection| projection.handle.clone()));
    result["cost"] = cost.finish(worker);
    let mut result = wire::with_response_bytes(result);
    super::ops::require_fit(&result)?;
    result.shrink_to_fit();
    projection.result = result;
    projection.held = dictionary_held
        .saturating_add(table_held)
        .saturating_add(2 * projection.result.capacity() as u64)
        .saturating_add(projection.spec.held_bytes())
        .saturating_add(projection.handle.capacity() as u64 + projection.revision.capacity() as u64)
        .saturating_add(ARTIFACT_FIXED_BYTES);
    worker.budget.hold(base.saturating_add(projection.held));
    Ok((Arc::new(projection), true))
}

/// A digest of the records of `table`, read in blocks.
fn digest_table(
    table: &Table,
    width: usize,
    records: u64,
    worker: &mut Worker<'_>,
    held: u64,
) -> Result<String, Fault> {
    worker
        .budget
        .require_whole(READER_BYTES, "reading the table of a projection in blocks")?;
    worker.budget.hold(held.saturating_add(READER_BYTES));
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; READER_RECORDS * width];
    let mut at = 0u64;
    while at < records {
        let now = (records - at).min(READER_RECORDS as u64) as usize;
        table.read(at * width as u64, &mut buffer[..now * width])?;
        hash.update(&buffer[..now * width]);
        at += now as u64;
    }
    drop(buffer);
    worker.budget.hold(held);
    Ok(format!("{:x}", hash.finalize())[..32].to_string())
}

/// A removed file when the operation that made it ends, however it ends.
struct Removed(std::path::PathBuf);

impl Drop for Removed {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Opens a source and projects it in the same job, or projects an open snapshot again.
pub(crate) fn run(worker: &mut Worker<'_>, task: &ProjectTask) -> Result<Done, Fault> {
    match &task.source {
        ProjectSource::Snapshot(snapshot) => {
            let (projection, made) = build(worker, snapshot, &task.spec, 0)?;
            Ok(Done {
                result: projection.result.clone(),
                commit: made.then(|| Commit::Projection {
                    snapshot: None,
                    projection: projection.clone(),
                }),
            })
        }
        ProjectSource::Open(open) => {
            let opened = super::ops::open(worker, open)?;
            let (snapshot, new) = match opened.commit {
                Some(Commit::Snapshot(snapshot)) => (snapshot, true),
                _ => {
                    let result: Value = serde_json::from_str(&opened.result)
                        .map_err(|_| Fault::runtime("an open's result does not read"))?;
                    let handle = result["snapshot"].as_str().unwrap_or_default();
                    let known = registry(worker.shared).snapshot(handle).ok_or_else(|| {
                        Fault::runtime("the snapshot of an open is not registered")
                    })?;
                    (known, false)
                }
            };
            let beside = if new { snapshot.held } else { 0 };
            let (projection, made) = build(worker, &snapshot, &task.spec, beside)?;
            let mut result: Value = serde_json::from_str(&snapshot.result)
                .map_err(|_| Fault::runtime("an open's result does not read"))?;
            let projected: Value = serde_json::from_str(&projection.result)
                .map_err(|_| Fault::runtime("a projection's result does not read"))?;
            // The open's result as it was, and what the projection is and cost: its
            // fields and times are read with `triager_inspect`.
            let mut compact = projection.describe();
            compact["cost"] = projected["cost"].clone();
            result["projection"] = compact;
            let mut result = wire::with_response_bytes(result);
            super::ops::require_fit(&result)?;
            result.shrink_to_fit();
            worker
                .budget
                .hold(worker.base.saturating_add(beside).saturating_add(if made {
                    projection.held
                } else {
                    0
                }));
            // A new snapshot has no projection before this one: a projection is bound to
            // its snapshot's handle, so `made` holds for it.
            let commit = (new || made).then(|| Commit::Projection {
                snapshot: new.then_some(snapshot),
                projection: projection.clone(),
            });
            Ok(Done { result, commit })
        }
    }
}

// --- What a projection that is not there would take -------------------------------------

/// The result that names what a request needs and the snapshot's projection does not
/// declare, with the scan that would provide it and the request that runs it
/// (performance.md, "Explicit rescan and expiry"). Nothing is scheduled.
pub(crate) fn missing(
    shared: &Shared,
    population: &Population,
    view: &str,
    needs: &Spec,
) -> Result<String, Fault> {
    let snapshot = population.snapshot();
    source_not_retained(snapshot, "an occurrence projection")?;
    let current = registry(shared).latest_projection(&snapshot.handle);
    let request = Spec {
        fields: needs.fields.clone(),
        time: needs.time.clone(),
        max_values: current
            .as_ref()
            .map_or(DEFAULT_MAX_VALUES, |projection| projection.spec.max_values),
    };
    let merged = request.merged(current.as_ref().map(|projection| &projection.spec))?;
    let missing_fields: Vec<&str> = needs
        .fields
        .iter()
        .filter(|field| {
            current
                .as_ref()
                .is_none_or(|projection| !projection.spec.fields.contains(field))
        })
        .map(|field| field.name.as_str())
        .collect();
    let missing_time = needs.time.is_some()
        && current
            .as_ref()
            .is_none_or(|projection| projection.spec.time != needs.time);
    let (originals, occurrences) = population.size();
    let completeness = snapshot.scope.completeness();
    let selected = snapshot.scope.selected_bytes().unwrap_or_default();
    let width = merged.width() as u64;
    let mut asked = json!({"v": VERSION, "snapshot": snapshot.handle});
    let mut declared = json!({
        "fields": merged.fields.iter().map(FieldSpec::describe).collect::<Vec<_>>(),
        "max_values": merged.max_values,
    });
    if let Some(time) = &merged.time {
        let mut time = time.describe();
        if let Some(object) = time.as_object_mut() {
            object.remove("field");
        }
        declared["time"] = time;
    }
    asked["projections"] = declared;
    let mut head = super::inspect::head(view, population);
    head.insert("status".into(), json!("projection_missing"));
    head.insert(
        "missing".into(),
        json!({
            "fields": missing_fields,
            "time": missing_time,
            "coverage": {
                "unit": "occurrences",
                "projected": 0,
                "not_projected": occurrences,
                "distinct_originals": originals,
                "reason": if current.is_some() {
                    "the snapshot's projection does not declare it"
                } else {
                    "the snapshot has no occurrence projection"
                },
            },
        }),
    );
    head.insert(
        "projections".into(),
        json!(current.as_ref().map(|projection| projection.describe())),
    );
    head.insert(
        "estimate".into(),
        json!({
            "requires": "one admitted read of the verified source, as a job",
            "archive_scans": 1,
            "records_to_scan": completeness.records_scanned,
            "entries_to_read": completeness.entries_read,
            "bytes_packed": selected.packed,
            "bytes_unpacked": selected.unpacked,
            "occurrences_to_project": snapshot.scope.occurrences(),
            "affected": {
                "snapshot": {"distinct_originals": snapshot.scope.originals(),
                             "occurrences": snapshot.scope.occurrences()},
                "population": {"distinct_originals": originals, "occurrences": occurrences},
            },
            "table_bytes": snapshot.scope.occurrences().saturating_mul(width),
            "memory_bytes_at_least": dictionaries_bytes(&merged) + build_fixed_bytes(&merged),
            "rows_scored": 0,
            "rows_encoded": 0,
            "genai_rows_submitted": 0,
        }),
    );
    head.insert(
        "rescan".into(),
        json!({"tool": "triager_open", "args": asked,
               "note": "keeps every declaration of the snapshot's projection; nothing runs \
                        until it is called"}),
    );
    head.insert(
        "cost".into(),
        json!({"archive_scans": 0, "rows_scored": 0,
                                       "index_rows_read": 0, "response_bytes": 0}),
    );
    let mut payload = wire::with_response_bytes(Value::Object(head));
    payload.shrink_to_fit();
    if wire::transport_bytes(&payload) > wire::PAYLOAD_TRANSPORT_BYTES {
        return Err(Fault::runtime(
            "an inspect result does not fit the response limit",
        ));
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declarations_are_ordered_merged_and_refused_by_name() {
        let spec = Spec::parse(&json!({"fields": [
            {"name": "user", "from": "kv"},
            {"name": "host", "from": "entry", "capture": "^([^/]+)/"}],
            "time": {"from": "text"}}))
        .unwrap();
        assert_eq!(spec.fields[0].name, "host");
        assert_eq!(spec.width(), 32);
        assert_eq!(spec.max_values, DEFAULT_MAX_VALUES);
        let more = Spec::parse(&json!({"fields": [{"name": "status", "from": "kv"}]})).unwrap();
        let merged = more.merged(Some(&spec)).unwrap();
        assert_eq!(
            merged
                .fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            ["host", "status", "user"]
        );
        assert_eq!(merged.time, spec.time);
        assert!(merged.covers(&spec));
        let other =
            Spec::parse(&json!({"fields": [{"name": "user", "from": "entry", "capture": "u"}]}))
                .unwrap();
        assert_eq!(other.merged(Some(&spec)).unwrap_err().code, "field_defined");
        let time = Spec::parse(&json!({"time": {"from": "kv", "key": "ts"}})).unwrap();
        assert_eq!(time.merged(Some(&spec)).unwrap_err().code, "time_defined");
        for refused in [
            json!({}),
            json!({"fields": [{"name": "source", "from": "kv"}]}),
            json!({"fields": [{"name": "a", "from": "kv"}, {"name": "a", "from": "kv"}]}),
            json!({"fields": [{"name": "a", "from": "entry"}]}),
            json!({"fields": [{"name": "a", "from": "kv", "capture": "x"}]}),
            json!({"time": {"from": "text", "timezone": "Europe/Vienna"}}),
            json!({"time": {"from": "kv"}}),
            json!({"fields": [], "max_values": 0}),
            json!({"fields": [{"name": "a", "from": "kv"}], "max_values": 100_001}),
            json!({"fields": [{"name": "a", "from": "kv"}], "other": 1}),
        ] {
            assert!(Spec::parse(&refused).is_err(), "{refused}");
        }
        let captures: Vec<Value> = (0..5)
            .map(|at| json!({"name": format!("e{at}"), "from": "entry", "capture": "x"}))
            .collect();
        assert!(Spec::parse(&json!({"fields": captures})).is_err());
    }

    #[test]
    fn timestamps_need_a_zone_and_missing_is_told_from_invalid() {
        let zoned = TimeFormat::Iso8601;
        assert_eq!(
            leading_time("2026-01-02T03:04:05Z login", zoned),
            1_767_323_045
        );
        assert_eq!(
            leading_time("[2026-01-02 03:04:05+01:00] x", zoned),
            1_767_319_445
        );
        assert_eq!(
            leading_time("2026-01-02T03:04:05.9Z x", zoned),
            1_767_323_045
        );
        assert_eq!(
            leading_time("2026-01-02 03:04:05 no zone", zoned),
            TIME_INVALID
        );
        assert_eq!(leading_time("2026-13-02T03:04:05Z", zoned), TIME_INVALID);
        assert_eq!(leading_time("login ok", zoned), TIME_MISSING);
        assert_eq!(leading_time("", zoned), TIME_MISSING);
        assert_eq!(
            value_time(Some("2026-01-02"), TimeFormat::Date),
            1_767_312_000
        );
        assert_eq!(
            value_time(Some("2026-02-30"), TimeFormat::Date),
            TIME_INVALID
        );
        assert_eq!(value_time(None, TimeFormat::Date), TIME_MISSING);
        assert_eq!(iso(1_767_323_045), "2026-01-02T03:04:05Z");
        assert_eq!(iso(-1), "1969-12-31T23:59:59Z");
        assert_eq!(parse_instant("2026-01-02T03:04:05Z"), Some(1_767_323_045));
        assert_eq!(parse_instant("2026-01-02T03:04:05"), None);
    }

    #[test]
    fn an_interner_retains_its_first_values_and_counts_the_rest_apart() {
        let mut interner = Interner::new(2);
        assert_eq!(interner.id(Some("b")), 0);
        assert_eq!(interner.id(Some("a")), 1);
        assert_eq!(interner.id(Some("b")), 0);
        assert_eq!(interner.id(Some("c")), VALUE_OVERFLOW);
        assert_eq!(interner.id(None), VALUE_MISSING);
        assert_eq!(
            interner.id(Some(&"x".repeat(MAX_VALUE_BYTES + 1))),
            VALUE_INVALID
        );
        let dictionary = interner.finish();
        assert_eq!(dictionary.values, vec![Box::from("b"), Box::from("a")]);
        assert!(dictionary.overflowed);
        assert_eq!(dictionary.find("a"), Some(1));
        assert_eq!(dictionary.find("c"), None);
    }

    #[test]
    fn the_dictionaries_of_a_build_stay_within_their_charge() {
        // Every value at the longest length, the limit reached and passed: what the
        // build allocates for one field against what it asks the budget for.
        for most in [1u64, 7, 100, 5_000] {
            let ((), peak) = crate::test_heap::measure(|| {
                let mut interner = Interner::new(most);
                for at in 0..most + 10 {
                    let mut value = format!("{at:08}");
                    value.push_str(&"v".repeat(MAX_VALUE_BYTES - value.len()));
                    interner.id(Some(&value));
                }
                let dictionary = interner.finish();
                assert_eq!(dictionary.values.len() as u64, most);
                assert!(dictionary.held_bytes() <= dictionary_bytes(most));
            });
            assert!(
                peak <= dictionary_bytes(most) + 2 * MAX_VALUE_BYTES as u64,
                "{most} values: the heap peaked at {peak} bytes of {}",
                dictionary_bytes(most)
            );
        }
    }

    #[test]
    fn records_round_trip_and_a_filter_tests_them() {
        let spec = Spec::parse(
            &json!({"fields": [{"name": "a", "from": "kv"}, {"name": "b", "from": "kv"}]}),
        )
        .unwrap();
        let occurrence = Occurrence {
            row: 7,
            position: ArchivePosition { entry: 3, line: 9 }.pack(),
            time: -5,
            ids: [1, VALUE_MISSING, 0, 0, 0, 0, 0, 0],
        };
        let mut record = [0u8; MAX_RECORD_BYTES];
        occurrence.encode(spec.fields.len(), &mut record);
        let back = Occurrence::decode(&record[..spec.width()], spec.fields.len());
        assert_eq!(
            (back.row, back.entry(), back.line(), back.time),
            (7, 3, 9, -5)
        );
        assert_eq!(back.ids[..2], [1, VALUE_MISSING]);
        let resolved = Resolved {
            tests: vec![
                Test::Value { field: 0, id: 1 },
                Test::Value {
                    field: 1,
                    id: VALUE_MISSING,
                },
                Test::Entry(3),
                Test::Time { start: -10, end: 0 },
            ],
        };
        assert!(resolved.passes(&back));
        let missing_time = Occurrence {
            time: TIME_MISSING,
            ..back
        };
        assert!(!resolved.passes(&missing_time));
        assert!(
            Resolved {
                tests: vec![Test::NoTime]
            }
            .passes(&missing_time)
        );
        assert!(
            !Resolved {
                tests: vec![Test::Never]
            }
            .passes(&back)
        );
    }
}
