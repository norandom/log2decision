//! Reading committed artifacts: the overview, the region counts, a page of records
//! and the evidence of one record.
//!
//! These read the membership a selection or view keeps and the snapshot's relation;
//! none of them reads the source, except an excerpt, which is one admitted read of
//! the source on the worker. Totals come from the committed counts, never from a page.
//! What a call builds stays within the adapter's fixed share
//! ([`super::ADAPTER_FIXED_BYTES`]): at most [`MAX_ROWS`] rows and one response.

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde_json::{Map, Value, json};

use super::DisclosureProfile;
use super::membership::State;
use super::ops::{SCAN_FIXED_BYTES, Worker, rows_of, scan};
use super::state::{Population, Producer, Set, Shared, Snapshot};
use super::wire::{
    self, Fault, MAX_EXCERPT_BYTES, MAX_PATH_BYTES, MAX_REGION_SETS, PAYLOAD_TRANSPORT_BYTES,
    VERSION, applied_limits,
};
use crate::scope::{Locator, Original};

/// The named sets of an inspect request, resolved.
pub(crate) type Named = Vec<(String, Arc<Set>)>;

fn locator(snapshot: &Snapshot, original: &Original) -> (Value, bool) {
    match original.first {
        Locator::ArchiveLine { line, .. } => {
            // An entry name is data from the archive: shown escaped and bounded.
            let (path, _) = snapshot.summary.locate(original.id.0).unwrap_or(("", line));
            let (shown, clipped) = wire::display(path, MAX_PATH_BYTES);
            (json!({"entry": shown, "line": line}), clipped)
        }
        Locator::FileRecord { position } => (json!({"record": position}), false),
        Locator::StoreRow { seq } => (json!({"seq": seq}), false),
    }
}

fn coverage(population: &Population) -> Value {
    match population {
        Population::All(snapshot) => {
            let completeness = snapshot.scope.completeness();
            json!({
                "unit": "distinct_originals",
                "evaluated": completeness.originals,
                "not_evaluated": 0,
                "source_complete": completeness.complete,
            })
        }
        Population::Of(set) => {
            let (unknown, _) = set.tally.of(State::Unknown);
            json!({
                "unit": "distinct_originals",
                "evaluated": set.tally.parent().0 - unknown,
                "not_evaluated": unknown,
                "source_complete": set.snapshot.scope.completeness().complete,
            })
        }
    }
}

/// What every inspect result states first.
fn head(view: &str, population: &Population) -> Map<String, Value> {
    let mut head = Map::new();
    head.insert("v".into(), json!(VERSION));
    head.insert("status".into(), json!("ready"));
    head.insert("view".into(), json!(view));
    head.insert("handle".into(), json!(population.handle()));
    head.insert("kind".into(), json!(population.kind()));
    head.insert("snapshot".into(), json!(population.snapshot().handle));
    head.insert("revision".into(), json!(population.revision()));
    head.insert("coverage".into(), coverage(population));
    head
}

fn fits(payload: &str) -> Result<(), Fault> {
    if wire::transport_bytes(payload) > PAYLOAD_TRANSPORT_BYTES {
        return Err(Fault::runtime(
            "an inspect result does not fit the response limit",
        ));
    }
    Ok(())
}

/// The artifact as the call that made it returned it, and what the server did so far.
pub(crate) fn overview(shared: &Shared, population: &Population) -> Result<String, Fault> {
    let mut result = head("overview", population);
    let (originals, occurrences) = population.size();
    result.insert(
        "members".into(),
        json!({"distinct_originals": originals, "occurrences": occurrences}),
    );
    result.insert("activity".into(), shared.activity());
    result.insert(
        "disclosure".into(),
        shared.disclosure(population.snapshot()),
    );
    result.insert("applied_limits".into(), applied_limits(0));
    result.insert("clipping".into(), json!({"clipped": false}));
    result.insert("page".into(), Value::Null);
    let mut payload = Value::Object(result).to_string();
    // The creating result is spliced in as it was returned, not parsed again.
    let artifact = match population {
        Population::All(snapshot) => &snapshot.result,
        Population::Of(set) => &set.result,
    };
    payload.pop();
    payload.push_str(",\"artifact\":");
    payload.push_str(artifact);
    payload.push('}');
    fits(&payload)?;
    Ok(payload)
}

/// The regions of up to three named sets within a population: every fully evaluated
/// combination, "none" included, and apart from them the combinations that hold a
/// set that was not evaluated. Both units; the counts add up to the population.
pub(crate) fn regions(population: &Population, sets: &Named) -> Result<String, Fault> {
    if sets.is_empty() || sets.len() > MAX_REGION_SETS {
        return Err(Fault::invalid(format!(
            "the regions view takes 1 to {MAX_REGION_SETS} named `sets`"
        )));
    }
    let snapshot = population.snapshot();
    let combinations = 3usize.pow(sets.len() as u32);
    let mut counts = vec![(0u64, 0u64); combinations];
    let rows = snapshot.scope.originals();
    for row in 0..rows {
        if population.state(row) != State::In {
            continue;
        }
        let mut at = 0;
        for (_, set) in sets.iter().rev() {
            at = at * 3 + set.states.get(row) as usize;
        }
        let original = snapshot.scope.row(row)?;
        counts[at].0 += 1;
        counts[at].1 += original.occurrences;
    }
    let (mut known, mut unknown) = (Vec::new(), Vec::new());
    let mut not_evaluated = (0u64, 0u64);
    let mut total = (0u64, 0u64);
    // In before out, the first set varying slowest: in/in, in/out, out/in, out/out.
    let order = [State::In, State::Out, State::Unknown];
    let mut digits = vec![0usize; sets.len()];
    loop {
        let states: Vec<State> = digits.iter().map(|digit| order[*digit]).collect();
        let at = states
            .iter()
            .rev()
            .fold(0, |at, state| at * 3 + *state as usize);
        let membership: Map<String, Value> = sets
            .iter()
            .zip(&states)
            .map(|((name, _), state)| (name.clone(), json!(state.name())))
            .collect();
        let (originals, occurrences) = counts[at];
        total.0 += originals;
        total.1 += occurrences;
        let region = json!({
            "membership": membership,
            "distinct_originals": originals,
            "occurrences": occurrences,
        });
        if states.contains(&State::Unknown) {
            not_evaluated.0 += originals;
            not_evaluated.1 += occurrences;
            if originals > 0 {
                unknown.push(region);
            }
        } else {
            known.push(region);
        }
        let mut place = sets.len();
        loop {
            if place == 0 {
                break;
            }
            place -= 1;
            digits[place] += 1;
            if digits[place] < 3 {
                break;
            }
            digits[place] = 0;
        }
        if digits.iter().all(|digit| *digit == 0) {
            break;
        }
    }
    if total != population.size() {
        return Err(Fault::runtime(
            "the regions do not add up to their parent population",
        ));
    }
    let named: Map<String, Value> = sets
        .iter()
        .map(|(name, set)| (name.clone(), json!(set.handle)))
        .collect();
    let mut result = head("regions", population);
    result.insert("sets".into(), Value::Object(named));
    result.insert(
        "parent".into(),
        json!({"handle": population.handle(), "distinct_originals": total.0, "occurrences": total.1}),
    );
    result.insert("regions".into(), Value::Array(known));
    result.insert("unknown_regions".into(), Value::Array(unknown));
    result.insert(
        "unknown".into(),
        json!({"distinct_originals": not_evaluated.0, "occurrences": not_evaluated.1}),
    );
    result.insert(
        "complete".into(),
        json!(not_evaluated.0 == 0 && snapshot.scope.completeness().complete),
    );
    result.insert(
        "cost".into(),
        json!({"archive_scans": 0, "index_rows_read": total.0}),
    );
    result.insert("applied_limits".into(), applied_limits(0));
    result.insert("clipping".into(), json!({"clipped": false}));
    result.insert("page".into(), Value::Null);
    let payload = Value::Object(result).to_string();
    fits(&payload)?;
    Ok(payload)
}

/// The columns of a records page.
pub(crate) const RECORD_FIELDS: [&str; 4] = ["id", "locator", "occurrences", "membership"];

pub(crate) struct RecordsRequest<'a> {
    pub population: &'a Population,
    pub sets: &'a Named,
    /// The fields asked for, in the order of [`RECORD_FIELDS`].
    pub projection: Vec<&'static str>,
    pub limit: usize,
    pub cursor: Option<&'a str>,
}

/// One page of the members of a population, by record ID ascending.
///
/// The members are found in the membership kept in memory and located in the
/// relation; the page is the `limit` lowest record IDs after the cursor. Rows are
/// added while the serialized result stays within the byte limit, and a page cut
/// short by bytes says so and continues from its cursor.
pub(crate) fn records(shared: &Shared, request: &RecordsRequest<'_>) -> Result<String, Fault> {
    let population = request.population;
    let snapshot = population.snapshot();
    let mut bound: Vec<&str> = vec![population.handle(), population.revision(), "records"];
    for (name, set) in request.sets {
        bound.push(name);
        bound.push(&set.handle);
    }
    bound.push("record_id_ascending");
    bound.extend(request.projection.iter());
    let binding = wire::cursor_binding(&shared.config.project, &bound);
    let after = match request.cursor {
        None => None,
        Some(cursor) => Some(wire::cursor_position(cursor, &binding).ok_or_else(|| {
            Fault::new(
                "stale_cursor",
                "the cursor was not issued for this handle, its revision, these sets, this \
                 order and this projection; start again from the first page",
            )
        })?),
    };
    // The `limit + 1` lowest record IDs after the cursor: a bounded heap whose top is
    // the highest of them.
    let wanted = request.limit + 1;
    let mut lowest: BinaryHeap<(u64, u64)> = BinaryHeap::with_capacity(wanted + 1);
    let rows = snapshot.scope.originals();
    let mut read = 0u64;
    for row in 0..rows {
        if population.state(row) != State::In {
            continue;
        }
        let position = snapshot.scope.row(row)?.id.0;
        read += 1;
        if after.is_some_and(|after| position <= after) {
            continue;
        }
        if lowest.len() < wanted {
            lowest.push((position, row));
        } else if lowest.peek().is_some_and(|highest| position < highest.0) {
            lowest.pop();
            lowest.push((position, row));
        }
    }
    let mut page: Vec<(u64, u64)> = lowest.into_vec();
    page.sort_unstable();
    let more_rows = page.len() > request.limit;
    page.truncate(request.limit);

    let (total_originals, total_occurrences) = population.size();
    let mut result = head("records", population);
    result.insert("sort".into(), json!(["record_id_ascending"]));
    result.insert("projection".into(), json!(request.projection));
    result.insert(
        "cost".into(),
        json!({"archive_scans": 0, "index_rows_read": read}),
    );
    result.insert("applied_limits".into(), applied_limits(request.limit));
    // The frame of the result with the longest values its page fields can take, so
    // that the rows are added against what is left of the byte limit.
    let frame = {
        let mut frame = result.clone();
        frame.insert("rows".into(), json!([]));
        frame.insert(
            "page".into(),
            json!({
                "limit": request.limit,
                "returned": request.limit,
                "total": {"distinct_originals": total_originals, "occurrences": total_occurrences},
                "next_cursor": wire::cursor(u64::MAX, &binding),
            }),
        );
        frame.insert(
            "clipping".into(),
            json!({"clipped": true, "rows_by_bytes": true, "fields": ["locator.entry"]}),
        );
        Value::Object(frame).to_string()
    };
    let mut used = wire::transport_bytes(&frame);
    let mut shown: Vec<Value> = Vec::with_capacity(page.len());
    let mut last = None;
    let mut path_clipped = false;
    let mut by_bytes = false;
    for (position, row) in &page {
        let original = snapshot.scope.row(*row)?;
        let mut fields = Map::new();
        for field in &request.projection {
            match *field {
                "id" => {
                    fields.insert("id".into(), json!(wire::record_id(*position)));
                }
                "locator" => {
                    let (shown, clipped) = locator(snapshot, &original);
                    path_clipped |= clipped;
                    fields.insert("locator".into(), shown);
                }
                "occurrences" => {
                    fields.insert("occurrences".into(), json!(original.occurrences));
                }
                _ => {
                    let membership: Map<String, Value> = request
                        .sets
                        .iter()
                        .map(|(name, set)| (name.clone(), json!(set.states.get(*row).name())))
                        .collect();
                    fields.insert("membership".into(), Value::Object(membership));
                }
            }
        }
        let fields = Value::Object(fields);
        let text = fields.to_string();
        // A row, its comma, and both again in the text form.
        let bytes = 2 * (text.len() + 1) + wire::escape_bytes(&text);
        if used + bytes > PAYLOAD_TRANSPORT_BYTES {
            by_bytes = true;
            break;
        }
        used += bytes;
        last = Some(*position);
        shown.push(fields);
    }
    let more = more_rows || by_bytes;
    if shown.is_empty() && !page.is_empty() {
        return Err(Fault::runtime(
            "one row of this projection does not fit the response limit",
        ));
    }
    result.insert(
        "page".into(),
        json!({
            "limit": request.limit,
            "returned": shown.len(),
            "total": {"distinct_originals": total_originals, "occurrences": total_occurrences},
            "next_cursor": match (more, last) {
                (true, Some(last)) => json!(wire::cursor(last, &binding)),
                _ => Value::Null,
            },
        }),
    );
    let mut clipped_fields = Vec::new();
    if path_clipped {
        clipped_fields.push("locator.entry");
    }
    result.insert(
        "clipping".into(),
        json!({
            "clipped": by_bytes || path_clipped,
            "rows_by_bytes": by_bytes,
            "fields": clipped_fields,
        }),
    );
    result.insert("rows".into(), Value::Array(shown));
    let payload = Value::Object(result).to_string();
    fits(&payload)?;
    Ok(payload)
}

/// The fields of an evidence result.
pub(crate) const EVIDENCE_FIELDS: [&str; 6] = [
    "id",
    "locator",
    "occurrences",
    "membership",
    "explanation",
    "excerpt",
];
/// What an evidence result holds when the request names no projection: what the
/// service keeps, without a read of the source.
pub(crate) const EVIDENCE_DEFAULT: [&str; 5] =
    ["id", "locator", "occurrences", "membership", "explanation"];

/// The evidence of one record, with every handle resolved and the record found in
/// the scope it was asked in.
pub(crate) struct EvidenceTask {
    pub population: Population,
    pub sets: Named,
    pub projection: Vec<&'static str>,
    pub row: u64,
}

impl EvidenceTask {
    pub fn wants_excerpt(&self) -> bool {
        self.projection.contains(&"excerpt")
    }
}

/// Finds the record a request names in the population it names.
pub(crate) fn find_record(population: &Population, record: &str) -> Result<u64, Fault> {
    let position = wire::record_position(record).ok_or_else(|| {
        Fault::invalid(
            "`record` is a record handle as the records view returns it in `id` \
             (`r-` and 16 hexadecimal digits)",
        )
    })?;
    let row = population
        .snapshot()
        .scope
        .row_of(position)?
        .ok_or_else(|| {
            Fault::new(
                "record_not_in_scope",
                format!("`{record}` names no original of the snapshot"),
            )
        })?;
    let state = population.state(row);
    if state != State::In {
        return Err(Fault::new(
            "record_not_in_scope",
            format!(
                "`{record}` is {} in `{}`; evidence is read under a scope that holds the \
                 record",
                state.name(),
                population.handle()
            ),
        ));
    }
    Ok(row)
}

/// The rank of `row` among the scored originals of a score set's parent: one more
/// than those that come before it, by score descending and record ID ascending.
fn rank(set: &Set, scores: &[f32], row: u64, position: u64) -> Result<u64, Fault> {
    let score = scores[row as usize];
    let mut before = 0u64;
    for (other, other_score) in scores.iter().enumerate() {
        if !other_score.is_finite() || other as u64 == row {
            continue;
        }
        match other_score.total_cmp(&score) {
            std::cmp::Ordering::Greater => before += 1,
            std::cmp::Ordering::Equal => {
                if set.snapshot.scope.row(other as u64)?.id.0 < position {
                    before += 1;
                }
            }
            std::cmp::Ordering::Less => {}
        }
    }
    Ok(before + 1)
}

/// Why `row` has the state it has in `set`.
fn explain(name: &str, set: &Set, row: u64, position: u64) -> Result<Value, Fault> {
    let state = set.states.get(row);
    let mut explained = json!({
        "set": name,
        "handle": set.handle,
        "producer": set.producer.describe(),
        "state": state.name(),
    });
    let outside = set
        .parent
        .as_ref()
        .is_some_and(|parent| parent.states.get(row) != State::In);
    if state == State::Unknown {
        explained["reason"] = json!(match (&set.producer, outside) {
            (Producer::Expression { .. }, _) => "operand_not_evaluated",
            (_, true) => "outside_producer_scope",
            (_, false) => "missing_model_output",
        });
    }
    match (&set.producer, &set.scores) {
        (Producer::Score { .. }, Some(scores)) if !outside => {
            let score = scores[row as usize];
            if score.is_finite() {
                explained["score"] = json!(score);
                explained["rank"] = json!(rank(set, scores, row, position)?);
                explained["ranked_among"] =
                    json!(set.tally.parent().0 - set.tally.of(State::Unknown).0);
            }
        }
        (Producer::Filter { .. }, _) if state != State::Unknown => {
            explained["matched"] = json!(state == State::In);
        }
        _ => {}
    }
    Ok(explained)
}

/// The most an excerpt adds to a tool result in both its forms: 1,024 bytes of text
/// whose escapes are written twice over, and the fields and the cost beside it.
pub(crate) const EXCERPT_TRANSPORT_BYTES: usize = 8 * 1024;

/// An excerpt read from the source for an evidence result.
pub(crate) struct Excerpt {
    text: String,
    clipped: bool,
    source_bytes: u64,
    cost: Value,
}

/// The evidence result of `task`, with `excerpt` when one was read.
pub(crate) fn evidence(
    shared: &Shared,
    task: &EvidenceTask,
    excerpt: Option<Excerpt>,
) -> Result<String, Fault> {
    let population = &task.population;
    let snapshot = population.snapshot();
    let original = snapshot.scope.row(task.row)?;
    let position = original.id.0;
    let wants = |field: &str| task.projection.contains(&field);
    let mut result = head("evidence", population);
    let mut record = Map::new();
    let mut clipped_fields = Vec::new();
    if wants("id") {
        record.insert("id".into(), json!(wire::record_id(position)));
        record.insert("native_id".into(), json!(original.native_id.to_hex()));
    }
    if wants("locator") {
        let (shown, clipped) = locator(snapshot, &original);
        if clipped {
            clipped_fields.push("record.locator.entry");
        }
        record.insert("locator".into(), shown);
    }
    if wants("occurrences") {
        record.insert("occurrences".into(), json!(original.occurrences));
        // The scope keeps the first occurrence and the exact count of an original.
        record.insert("occurrence_locators".into(), json!("first_only"));
    }
    result.insert("record".into(), Value::Object(record));
    if wants("membership") {
        let membership: Map<String, Value> = task
            .sets
            .iter()
            .map(|(name, set)| {
                (
                    name.clone(),
                    json!({"state": set.states.get(task.row).name()}),
                )
            })
            .collect();
        result.insert("membership".into(), Value::Object(membership));
    }
    if wants("explanation") {
        let mut explained = Vec::with_capacity(task.sets.len() + 1);
        if let Population::Of(set) = population {
            explained.push(explain("handle", set, task.row, position)?);
        }
        for (name, set) in &task.sets {
            explained.push(explain(name, set, task.row, position)?);
        }
        result.insert("explanation".into(), Value::Array(explained));
    }
    let mut disclosed = 0;
    let mut cost = json!({"archive_scans": 0});
    if let Some(excerpt) = excerpt {
        disclosed = excerpt.text.len();
        if excerpt.clipped {
            clipped_fields.push("excerpt.text");
        }
        result.insert(
            "excerpt".into(),
            json!({
                "text": excerpt.text,
                "bytes": disclosed,
                "clipped": excerpt.clipped,
                "source_bytes": excerpt.source_bytes,
                "projection": "terminal controls and invisible characters escaped",
            }),
        );
        cost = excerpt.cost;
    }
    let mut disclosure = shared.disclosure(snapshot);
    disclosure["excerpt_bytes"] = json!(disclosed);
    result.insert("disclosure".into(), disclosure);
    result.insert("cost".into(), cost);
    result.insert("applied_limits".into(), applied_limits(1));
    result.insert(
        "clipping".into(),
        json!({"clipped": !clipped_fields.is_empty(), "fields": clipped_fields}),
    );
    result.insert("page".into(), Value::Null);
    let payload = Value::Object(result).to_string();
    if wire::transport_bytes(&payload) > PAYLOAD_TRANSPORT_BYTES {
        // What the result grows with is the request's: the sets it names.
        return Err(Fault::invalid(
            "the evidence of this record with these `sets` does not fit the response \
             limit; name fewer sets or project fewer fields",
        ));
    }
    Ok(payload)
}

/// Checks, before anything is read, that this session may be shown an excerpt of the
/// snapshot, and returns the most bytes it may be shown.
pub(crate) fn excerpt_limit(shared: &Shared, snapshot: &Snapshot) -> Result<usize, Fault> {
    match shared.config.profile {
        DisclosureProfile::PrivateLocal => Ok(MAX_EXCERPT_BYTES),
        DisclosureProfile::PublicDevelopment if !snapshot.public => Err(Fault::new(
            "disclosure_refused",
            "the host runs the public-development profile and this source is not declared \
             a public fixture; no excerpt of it is shown",
        )),
        DisclosureProfile::PublicDevelopment => match shared.remaining_evidence() {
            Some(0) | None => Err(Fault::new(
                "disclosure_refused",
                "the evidence allowance of this session is used up; counts, handles and \
                 memberships stay available",
            )),
            Some(left) => Ok(MAX_EXCERPT_BYTES.min(usize::try_from(left).unwrap_or(usize::MAX))),
        },
    }
}

/// Reads the excerpt of the record of `task` from the source and returns its
/// evidence: one admitted read of the source, ended at the record.
pub(crate) fn evidence_with_excerpt(
    worker: &mut Worker<'_>,
    task: &EvidenceTask,
) -> Result<String, Fault> {
    let shared = worker.shared;
    let snapshot = task.population.snapshot().clone();
    let limit = excerpt_limit(shared, &snapshot)?;
    let position = snapshot.scope.row(task.row)?.id.0;
    worker.budget.hold(worker.base);
    worker.budget.require_whole(
        SCAN_FIXED_BYTES,
        "reading the excerpt of one record from its source",
    )?;
    let held = worker.base.saturating_add(SCAN_FIXED_BYTES);
    worker.budget.hold(held);
    let started = std::time::Instant::now();
    let mut found: Option<(String, bool, u64)> = None;
    let mut scanned = 0u64;
    let ended = scan(worker, &snapshot, held, &mut |logs| {
        for (text, at) in rows_of(logs)? {
            scanned += 1;
            if at == position {
                let (shown, clipped) = wire::display(text, limit);
                found = Some((shown, clipped, text.len() as u64));
                return Ok(false);
            }
        }
        Ok(true)
    });
    worker.budget.hold(worker.base);
    ended?;
    let (text, clipped, source_bytes) = found.ok_or_else(|| {
        Fault::refused(
            "the record was not found when its source was read again; the source no \
             longer reads as the snapshot bound it, open it again",
        )
    })?;
    if shared.config.profile == DisclosureProfile::PublicDevelopment {
        let mut left = shared
            .allowance
            .lock()
            .unwrap_or_else(|held| held.into_inner());
        *left = left.saturating_sub(text.len() as u64);
    }
    shared
        .activity
        .evidence_bytes
        .fetch_add(text.len() as u64, Ordering::Relaxed);
    let report = shared.usage.report();
    let excerpt = Excerpt {
        text,
        clipped,
        source_bytes,
        cost: json!({
            "archive_scans": 1,
            "distinct_originals_read": scanned,
            "read_ended_at_record": true,
            "elapsed_ms": started.elapsed().as_millis() as u64,
            "memory_limit_bytes": report.memory_limit_bytes,
            "counted_peak_bytes": report.counted_peak_bytes,
            "process_peak_bytes": null,
        }),
    };
    evidence(shared, task, Some(excerpt))
}
