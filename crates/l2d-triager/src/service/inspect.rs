//! Reading committed artifacts: the overview, the region counts, a page of records
//! and the evidence of one record.
//!
//! These read the membership a selection or view keeps, in memory or on disk, and the
//! snapshot's relation; none of them reads the source, except an excerpt, which is one
//! admitted read of the source on the worker. Totals come from the committed counts,
//! never from a page. What a call builds stays within the adapter's fixed share
//! ([`super::ADAPTER_FIXED_BYTES`]): at most [`MAX_ROWS`] rows, one response, and the
//! blocks a table is read through ([`READ_BLOCK_BYTES`]).
//!
//! The rows of a relation are in the order of the record IDs, so a page is the next
//! members after its cursor: it reads the codes of its population from the cursor on
//! and the rows of the members it returns, not the relation.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use l2d_core::L2dError;
use serde_json::{Map, Value, json};

use super::DisclosureProfile;
use super::allocate::Allocation;
use super::membership::{ABSTAINED, State, States, code_in};
use super::ops::{SCAN_FIXED_BYTES, Worker, rows_of, scan};
use super::state::{Origin, Population, Producer, Set, Shared, Snapshot};
use super::tables::{BLOCK_ROWS, Scores, score_in};
use super::wire::{
    self, Fault, MAX_EXCERPT_BYTES, MAX_PATH_BYTES, MAX_REGION_SETS, PAYLOAD_TRANSPORT_BYTES,
    VERSION, applied_limits,
};
use crate::scope::{Locator, Original, RowBlock};

/// The named sets of an inspect request, resolved.
pub(crate) type Named = Vec<(String, Arc<Set>)>;

/// The most a read of committed tables holds at a time on a calling thread: a block
/// of the population and of three sets and a run of rows for the regions, or a block
/// of scores and of a parent for a rank.
pub(crate) const READ_BLOCK_BYTES: u64 =
    4 * States::BLOCK_BYTES as u64 + RowBlock::BYTES + Scores::BLOCK_BYTES as u64;

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

/// The named sets that are not independent of each other: a Venn recommendation and
/// every score set made by the scorer revision its policy scored with (requirement
/// 2.2). Empty when no such pair is named.
fn dependencies(sets: &Named) -> Vec<Value> {
    let mut found = Vec::new();
    for (name, set) in sets {
        // Also of sets a receipt retained.
        let Origin::Venn { scorer_revision } = Origin::of(&set.producer) else {
            continue;
        };
        let scorer_revision = &scorer_revision;
        for (other, scored) in sets {
            if matches!(Origin::of(&scored.producer), Origin::Score { scorer_revision: revision }
                if revision == *scorer_revision)
            {
                found.push(json!({
                    "set": name,
                    "depends_on": other,
                    "shared": "scorer_revision",
                    "scorer_revision": wire::display(scorer_revision, 160).0,
                    "meaning": "the policy scored with the scorer of this set: their \
                                agreement is not independent evidence",
                }));
            }
        }
    }
    found
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

/// What a read of committed tables cost: no source, no score, and the rows of the
/// index it read.
fn warm_cost(index_rows_read: u64) -> Value {
    json!({"archive_scans": 0, "rows_scored": 0, "index_rows_read": index_rows_read,
           "response_bytes": 0})
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
    if let Population::Of(set) = population {
        // Where the set keeps its membership and scores, and what they take.
        result.insert(
            "tables".into(),
            json!({
                "on_disk": set.states.is_on_disk(),
                "disk_bytes": set.disk_bytes(),
                "memory_bytes": set.states.held_bytes(),
                "shares_scores": set.scores.is_some() && !set.owns_scores,
            }),
        );
    }
    let (mut dependencies_total, mut dependencies_listed) = (0u64, 0u64);
    if let Population::Of(set) = population
        && let Producer::Venn {
            scorer_revision, ..
        } = &set.producer
    {
        // Found now, not when the set was made: the score sets of this snapshot by
        // the scorer revision the policy scored with. C is not independent of them
        // (requirement 2.2), whichever was selected first.
        // Nothing is collected: the sets are counted as the registry is walked,
        // and the eight with the lowest handles are kept by reference, so the
        // list is the same however the registry orders its sets.
        let registry = super::ops::registry(shared);
        let mut lowest: Vec<&Arc<Set>> = Vec::with_capacity(wire::MAX_SETS + 1);
        for other in registry.sets() {
            let depends = other.snapshot.handle == set.snapshot.handle
                && matches!(&other.producer, Producer::Score { scorer, .. }
                    if scorer.revision == *scorer_revision);
            if !depends {
                continue;
            }
            dependencies_total += 1;
            let at = lowest.partition_point(|kept| kept.handle < other.handle);
            if at < wire::MAX_SETS {
                lowest.insert(at, other);
                lowest.truncate(wire::MAX_SETS);
            }
        }
        dependencies_listed = lowest.len() as u64;
        let revision = wire::display(scorer_revision, 160).0;
        let found: Vec<Value> = lowest
            .iter()
            .map(|other| {
                let model = match &other.producer {
                    Producer::Score { model, .. } => wire::display(model, 128).0,
                    _ => String::new(),
                };
                json!({"set": other.handle, "model": model, "shared": "scorer_revision",
                       "scorer_revision": revision})
            })
            .collect();
        result.insert("dependencies_total".into(), json!(dependencies_total));
        result.insert("dependencies".into(), Value::Array(found));
    }
    result.insert("activity".into(), shared.activity());
    result.insert("session".into(), shared.session());
    result.insert(
        "disclosure".into(),
        shared.disclosure(population.snapshot()),
    );
    result.insert("applied_limits".into(), applied_limits(0));
    // A list of dependencies that is shorter than their number says so.
    result.insert(
        "clipping".into(),
        if dependencies_total > dependencies_listed {
            json!({"clipped": true, "fields": ["dependencies"]})
        } else {
            json!({"clipped": false})
        },
    );
    result.insert("page".into(), Value::Null);
    result.insert("cost".into(), warm_cost(0));
    // The creating result is spliced in as it was returned, not parsed again.
    let artifact = match population {
        Population::All(snapshot) => &snapshot.result,
        Population::Of(set) => &set.result,
    };
    let payload = wire::settle_response_bytes(&mut Value::Object(result), &|mut payload| {
        payload.pop();
        payload.push_str(",\"artifact\":");
        payload.push_str(artifact);
        payload.push('}');
        payload
    });
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
    let mut within = vec![0u8; States::BLOCK_BYTES];
    let mut codes: Vec<Vec<u8>> = sets
        .iter()
        .map(|_| vec![0u8; States::BLOCK_BYTES])
        .collect();
    let mut block = RowBlock::new();
    let mut first = 0;
    while first < rows {
        let in_block = population.read_block(first, &mut within)?;
        for ((_, set), codes) in sets.iter().zip(&mut codes) {
            set.states.read_block(first, codes)?;
        }
        let mut at = 0;
        while at < in_block {
            let run = snapshot.scope.rows(first + at as u64, false, &mut block)? as usize;
            for at in at..(at + run).min(in_block) {
                if code_in(&within, at) != State::In as u8 {
                    continue;
                }
                let mut combination = 0;
                for codes in codes.iter().rev() {
                    combination = combination * 3 + State::of(code_in(codes, at)) as usize;
                }
                counts[combination].0 += 1;
                counts[combination].1 += block.occurrences(first + at as u64);
            }
            at += run;
        }
        first += BLOCK_ROWS;
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
    result.insert("dependencies".into(), Value::Array(dependencies(sets)));
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
    result.insert("cost".into(), warm_cost(rows));
    result.insert("applied_limits".into(), applied_limits(0));
    result.insert("clipping".into(), json!({"clipped": false}));
    result.insert("page".into(), Value::Null);
    let payload = wire::with_response_bytes(Value::Object(result));
    fits(&payload)?;
    Ok(payload)
}

/// The columns of a records page.
pub(crate) const RECORD_FIELDS: [&str; 4] = ["id", "locator", "occurrences", "membership"];
/// The columns of a records page of a final work set: also the slot of each member.
pub(crate) const FINAL_FIELDS: [&str; 5] =
    ["id", "locator", "occurrences", "membership", "allocation"];

/// The projected fields of one record of a page.
fn row_fields(
    snapshot: &Snapshot,
    request: &RecordsRequest<'_>,
    allocation: Option<&Allocation>,
    row: u64,
    original: &Original,
    path_clipped: &mut bool,
) -> Result<Value, Fault> {
    let mut fields = Map::new();
    for field in &request.projection {
        match *field {
            "id" => {
                fields.insert("id".into(), json!(wire::record_id(original.id.0)));
            }
            "locator" => {
                let (shown, clipped) = locator(snapshot, original);
                *path_clipped |= clipped;
                fields.insert("locator".into(), shown);
            }
            "occurrences" => {
                fields.insert("occurrences".into(), json!(original.occurrences));
            }
            "allocation" => {
                let slot = allocation.and_then(|allocation| {
                    allocation.slot_of(row).map(|(index, slot)| {
                        json!({
                            "pool": allocation.pools[slot.pool as usize].name,
                            "slot": index + 1,
                            "pool_rank": slot.rank,
                            "outside_candidate": slot.outside_candidate,
                        })
                    })
                });
                fields.insert("allocation".into(), slot.unwrap_or(Value::Null));
            }
            _ => {
                let mut membership = Map::new();
                for (name, set) in request.sets {
                    membership.insert(name.clone(), json!(set.states.get(row)?.name()));
                }
                fields.insert("membership".into(), Value::Object(membership));
            }
        }
    }
    Ok(Value::Object(fields))
}

/// One page of the members of a final work set in the order of its slots: pool by
/// pool, each in its order. The cursor continues after a slot.
pub(crate) fn final_records(
    shared: &Shared,
    request: &RecordsRequest<'_>,
    allocation: &Allocation,
) -> Result<String, Fault> {
    let population = request.population;
    let snapshot = population.snapshot();
    let mut bound: Vec<&str> = vec![population.handle(), population.revision(), "records"];
    for (name, set) in request.sets {
        bound.push(name);
        bound.push(&set.handle);
    }
    bound.push("allocation");
    bound.extend(request.projection.iter());
    let binding = wire::cursor_binding(&shared.config.project, &bound);
    let start = match request.cursor {
        None => 0,
        Some(cursor) => {
            let after = wire::cursor_position(cursor, &binding)
                .filter(|after| *after < allocation.slots.len() as u64)
                .ok_or_else(|| {
                    Fault::new(
                        "stale_cursor",
                        "the cursor was not issued for this handle, its revision, these sets, \
                         this order and this projection; start again from the first page",
                    )
                })?;
            after as usize + 1
        }
    };
    let (total_originals, total_occurrences) = population.size();
    let mut result = head("records", population);
    result.insert("sort".into(), json!(["allocation"]));
    result.insert("projection".into(), json!(request.projection));
    result.insert("applied_limits".into(), applied_limits(request.limit));
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
        frame.insert(
            "cost".into(),
            json!({"archive_scans": 0, "rows_scored": 0, "index_rows_read": u64::MAX,
                   "response_bytes": PAYLOAD_TRANSPORT_BYTES}),
        );
        Value::Object(frame).to_string()
    };
    let mut used = wire::transport_bytes(&frame);
    let mut shown: Vec<Value> = Vec::with_capacity(request.limit);
    let mut path_clipped = false;
    let mut by_bytes = false;
    let mut last = None;
    let mut read = 0u64;
    let end = allocation.slots.len().min(start + request.limit);
    for (index, slot) in allocation.slots[start.min(end)..end].iter().enumerate() {
        let original = snapshot.scope.row(slot.row)?;
        read += 1;
        let fields = row_fields(
            snapshot,
            request,
            Some(allocation),
            slot.row,
            &original,
            &mut path_clipped,
        )?;
        let text = fields.to_string();
        let bytes = 2 * (text.len() + 1) + wire::escape_bytes(&text);
        if used + bytes > PAYLOAD_TRANSPORT_BYTES {
            by_bytes = true;
            break;
        }
        used += bytes;
        last = Some((start + index) as u64);
        shown.push(fields);
    }
    if shown.is_empty() && start < end {
        return Err(Fault::runtime(
            "one row of this projection does not fit the response limit",
        ));
    }
    let more = by_bytes || end < allocation.slots.len();
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
    let clipped_fields: Vec<&str> = if path_clipped {
        vec!["locator.entry"]
    } else {
        Vec::new()
    };
    result.insert(
        "clipping".into(),
        json!({"clipped": by_bytes || path_clipped, "rows_by_bytes": by_bytes,
               "fields": clipped_fields}),
    );
    result.insert("cost".into(), warm_cost(read));
    result.insert("rows".into(), Value::Array(shown));
    let payload = wire::with_response_bytes(Value::Object(result));
    fits(&payload)?;
    Ok(payload)
}

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
/// The rows of the relation are in that order, so the page is the next `limit`
/// members after the row of the cursor: the codes of the population are read from
/// there on, a block at a time, until the page is full. Rows are added while the
/// serialized result stays within the byte limit, and a page cut short by bytes says
/// so and continues from its cursor.
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
    let stale = || {
        Fault::new(
            "stale_cursor",
            "the cursor was not issued for this handle, its revision, these sets, this \
             order and this projection; start again from the first page",
        )
    };
    // The row after the one the cursor ended on.
    let start = match request.cursor {
        None => 0,
        Some(cursor) => {
            let after = wire::cursor_position(cursor, &binding).ok_or_else(stale)?;
            snapshot.scope.row_of(after)?.ok_or_else(stale)? + 1
        }
    };
    // The next `limit + 1` members: one more than the page tells whether another
    // page follows.
    let wanted = request.limit + 1;
    let mut page: Vec<u64> = Vec::with_capacity(wanted);
    let rows = snapshot.scope.originals();
    let mut read = 0u64;
    let mut within = vec![0u8; States::BLOCK_BYTES];
    let mut first = start - start % BLOCK_ROWS;
    'blocks: while first < rows {
        let in_block = population.read_block(first, &mut within)?;
        for at in (start.saturating_sub(first) as usize)..in_block {
            read += 1;
            if code_in(&within, at) == State::In as u8 {
                page.push(first + at as u64);
                if page.len() == wanted {
                    break 'blocks;
                }
            }
        }
        first += BLOCK_ROWS;
    }
    let more_rows = page.len() > request.limit;
    page.truncate(request.limit);

    let (total_originals, total_occurrences) = population.size();
    let mut result = head("records", population);
    result.insert("sort".into(), json!(["record_id_ascending"]));
    result.insert("projection".into(), json!(request.projection));
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
        frame.insert(
            "cost".into(),
            json!({"archive_scans": 0, "rows_scored": 0, "index_rows_read": u64::MAX,
                   "response_bytes": PAYLOAD_TRANSPORT_BYTES}),
        );
        Value::Object(frame).to_string()
    };
    let mut used = wire::transport_bytes(&frame);
    let mut shown: Vec<Value> = Vec::with_capacity(page.len());
    let mut last = None;
    let mut path_clipped = false;
    let mut by_bytes = false;
    let allocation = population
        .as_set()
        .and_then(|set| set.allocation.as_deref());
    for row in &page {
        let original = snapshot.scope.row(*row)?;
        read += 1;
        let position = original.id.0;
        let fields = row_fields(
            snapshot,
            request,
            allocation,
            *row,
            &original,
            &mut path_clipped,
        )?;
        let text = fields.to_string();
        // A row, its comma, and both again in the text form.
        let bytes = 2 * (text.len() + 1) + wire::escape_bytes(&text);
        if used + bytes > PAYLOAD_TRANSPORT_BYTES {
            by_bytes = true;
            break;
        }
        used += bytes;
        last = Some(position);
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
    result.insert("cost".into(), warm_cost(read));
    result.insert("rows".into(), Value::Array(shown));
    let payload = wire::with_response_bytes(Value::Object(result));
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
    let state = population.state(row)?;
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
/// than those that come before it, by score descending and record ID ascending. The
/// rows are in the order of the record IDs, and a table of scores a set shares may
/// hold scores outside its parent, which do not count.
fn rank(set: &Set, scores: &Scores, row: u64) -> Result<u64, L2dError> {
    let score = scores.get(row)?;
    let mut before = 0u64;
    let mut values = vec![0u8; Scores::BLOCK_BYTES];
    let mut within = vec![0u8; States::BLOCK_BYTES];
    let mut first = 0;
    while first < set.states.rows() {
        let in_block = scores.read_block(first, &mut values)?;
        if let Some(parent) = &set.parent {
            parent.states.read_block(first, &mut within)?;
        }
        for at in 0..in_block {
            let other = first + at as u64;
            let other_score = score_in(&values, at);
            if !other_score.is_finite()
                || other == row
                || (set.parent.is_some() && code_in(&within, at) != State::In as u8)
            {
                continue;
            }
            match other_score.total_cmp(&score) {
                std::cmp::Ordering::Greater => before += 1,
                std::cmp::Ordering::Equal if other < row => before += 1,
                _ => {}
            }
        }
        first += BLOCK_ROWS;
    }
    Ok(before + 1)
}

/// Why `row` has the state it has in `set`. `read` grows by the rows of the index
/// that were read for it: a rank reads the score of every original.
fn explain(name: &str, set: &Set, row: u64, read: &mut u64) -> Result<Value, Fault> {
    let code = set.states.code(row)?;
    let state = State::of(code);
    let mut explained = json!({
        "set": name,
        "handle": set.handle,
        "producer": set.producer.describe(),
        "state": state.name(),
    });
    let outside = match &set.parent {
        Some(parent) => parent.states.get(row)? != State::In,
        None => false,
    };
    // What kind of producer made the set, also for one a receipt retained.
    let origin = Origin::of(&set.producer);
    if state == State::Unknown {
        explained["reason"] = json!(match (&origin, outside) {
            (Origin::Expression | Origin::Allocation, _) => "operand_not_evaluated",
            (_, true) => "outside_producer_scope",
            (Origin::Neighbours, false) if code == ABSTAINED => "not_represented",
            (_, false) if code == ABSTAINED => "abstained",
            (
                Origin::Membership
                | Origin::Pattern { .. }
                | Origin::Neighbours
                | Origin::Venn { .. }
                | Origin::Issued,
                false,
            ) => "outside_producer_coverage",
            (_, false) => "missing_model_output",
        });
    }
    if let Some(allocation) = &set.allocation
        && !outside
    {
        // Why a slot holds the original, or why none does.
        explained["allocation"] = allocation.explain(row)?;
        return Ok(explained);
    }
    if let Producer::Retained { .. } = &set.producer {
        explained["retained"] = json!(true);
        match (&origin, &set.scores) {
            (Origin::Venn { .. }, Some(scores)) if !outside && state != State::Unknown => {
                if let Some(detail) = super::recommend::venn_detail(scores.get(row)?) {
                    explained["venn"] = detail;
                }
            }
            (Origin::Neighbours, Some(scores)) if state == State::In => {
                explained["distance"] = json!(scores.get(row)?);
            }
            (Origin::Pattern { pattern }, _) if state != State::Unknown => {
                explained["pattern"] = json!(pattern);
                explained["matched"] = json!(state == State::In);
            }
            (Origin::Score { .. }, Some(scores)) if !outside => {
                let score = scores.get(row)?;
                if score.is_finite() {
                    explained["score"] = json!(score);
                    // Among the originals the receipt retained, not the whole snapshot.
                    explained["rank"] = json!(rank(set, scores, row)?);
                    *read += set.states.rows();
                    explained["ranked_among"] =
                        json!(set.tally.parent().0 - set.tally.of(State::Unknown).0);
                    explained["ranked_within"] = json!("the originals the receipt retained");
                }
            }
            (Origin::Filter, _) if state != State::Unknown => {
                explained["matched"] = json!(state == State::In);
            }
            (Origin::Issued, _) if !outside => {
                // The judgments as they were when the receipt was saved, and the run's
                // caller-reported model and prompt revision.
                if let Producer::Retained {
                    judged: Some(judged),
                    ..
                } = &set.producer
                {
                    let position = set.snapshot.scope.row(row)?.id.0;
                    let mut detail = match judged.detail(position) {
                        Some(text) => serde_json::from_str(text).unwrap_or(Value::Null),
                        None => json!({
                            "outcome": "not_issued",
                            "judgments": [],
                            "kind": "model suggestion; not a reviewed outcome",
                        }),
                    };
                    detail["run"] = json!({
                        "name": judged.run["name"],
                        "model": judged.run["model"],
                        "prompt_revision": judged.run["prompt_revision"],
                        "model_identity": judged.run["model_identity"],
                        "submissions": judged.run["submissions"],
                    });
                    explained["recommendation"] = detail;
                }
            }
            _ => {}
        }
        return Ok(explained);
    }
    match (&set.producer, &set.scores) {
        (Producer::Venn { .. }, Some(scores)) if !outside && state != State::Unknown => {
            // The route, the reason and the allocation the saved policy gave.
            if let Some(detail) = super::recommend::venn_detail(scores.get(row)?) {
                explained["venn"] = detail;
            }
        }
        (
            Producer::Issued {
                run, upto, batches, ..
            },
            Some(scores),
        ) if !outside => {
            let position = set.snapshot.scope.row(row)?.id.0;
            explained["recommendation"] =
                super::recommend::issued_detail(run, (*upto, *batches), position, scores.get(row)?);
        }
        (Producer::Neighbours { .. }, Some(scores)) if state == State::In => {
            explained["distance"] = json!(scores.get(row)?);
        }
        (Producer::Pattern { pattern, .. }, _) if state != State::Unknown => {
            explained["pattern"] = json!(pattern);
            explained["matched"] = json!(state == State::In);
        }
        (Producer::Score { .. }, Some(scores)) if !outside => {
            let score = scores.get(row)?;
            if score.is_finite() {
                explained["score"] = json!(score);
                explained["rank"] = json!(rank(set, scores, row)?);
                *read += set.states.rows();
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
        let mut membership = Map::new();
        for (name, set) in &task.sets {
            membership.insert(
                name.clone(),
                json!({"state": set.states.get(task.row)?.name()}),
            );
        }
        result.insert("membership".into(), Value::Object(membership));
    }
    // The row of the record itself, and what its explanation reads.
    let mut index_rows_read = 1;
    if wants("explanation") {
        let mut explained = Vec::with_capacity(task.sets.len() + 1);
        if let Population::Of(set) = population {
            explained.push(explain("handle", set, task.row, &mut index_rows_read)?);
        }
        for (name, set) in &task.sets {
            explained.push(explain(name, set, task.row, &mut index_rows_read)?);
        }
        result.insert("explanation".into(), Value::Array(explained));
        result.insert(
            "dependencies".into(),
            Value::Array(dependencies(&task.sets)),
        );
    }
    let mut disclosed = 0;
    let mut cost = warm_cost(index_rows_read);
    match excerpt {
        Some(excerpt) => {
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
            cost["index_rows_read"] = json!(index_rows_read);
        }
        None if snapshot.retained.is_some() => {
            result.insert(
                "excerpt_estimate".into(),
                json!({
                    "requires": "the source, which a reopened receipt does not retain: open \
                                 the source and replay the receipt's recipe",
                    "archive_scans": Value::Null,
                }),
            );
        }
        None => {
            // The text of a record is not kept: an excerpt is another read of the
            // source up to the record. What that takes at most is stated before a
            // client asks for it (performance.md, "Read and index only what the
            // investigation needs").
            let selected = snapshot.scope.selected_bytes().unwrap_or_default();
            result.insert(
                "excerpt_estimate".into(),
                json!({
                    "requires": "one read of the source up to the record",
                    "archive_scans": 1,
                    "records_at_most": snapshot.scope.completeness().records_scanned,
                    "bytes_unpacked_at_most": selected.unpacked,
                }),
            );
        }
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
    let payload = wire::with_response_bytes(Value::Object(result));
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
            "rows_scored": 0,
            "distinct_originals_read": scanned,
            "read_ended_at_record": true,
            "elapsed_ms": started.elapsed().as_millis() as u64,
            "queue_ms": worker.queue_ms,
            "memory_limit_bytes": report.memory_limit_bytes,
            "server_counted_peak_bytes": report.counted_peak_bytes,
            "server_process_peak_bytes": super::process::peak_bytes(),
            "response_bytes": 0,
        }),
    };
    evidence(shared, task, Some(excerpt))
}
