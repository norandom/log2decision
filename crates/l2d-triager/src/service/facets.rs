//! Field and time views over an occurrence projection (triager task 3.1; requirements
//! 2.4, 4.3, 5.2, 6.1 and 6.3): the fields of a population with typed counts, missing,
//! invalid and overflowing values; the values of one field; the time histogram with
//! its timestamp field, zone, range, bucket width and the occurrences without a time,
//! or `time_unavailable` with the occurrences by archive entry as the labelled
//! alternative; the drill into a field value or a time bucket, which makes a child
//! view on the same snapshot; and the occurrences of one original for its evidence.
//!
//! Every count is of occurrences and of distinct originals, both exact. An occurrence
//! counts when its original is a member of the population and, for a field or time
//! view, when it passes the view's occurrence filter: filtering to one host or bucket
//! selects the matching occurrences and their distinct originals (design.md, "What the
//! foundation already supplies"). An original is counted once in each value or bucket
//! one of its counted occurrences has, so a repeated original read from two entries
//! counts in both.
//!
//! These read the committed table of a projection and the membership of the
//! population, and nothing else: no source, no score, no model (requirement 6.3).
//! They run as jobs on the worker, within shares they ask the budget for: the block a
//! table is read through, a block of the population and one counter per value.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use l2d_core::L2dError;
use serde_json::{Map, Value, json};

use super::membership::{State, States, code_in, put_code};
use super::ops::{
    ARTIFACT_FIXED_BYTES, COMPOSE_FIXED_BYTES, Commit, Cost, Done, TALLY_BYTES, Worker, coverage,
    registry, require_fit, set_dir, share_of, tables_on_disk, tally,
};
use super::project::{
    Clause, Filter, MAX_CLAUSES, Occurrence, Projection, READER_BYTES, READER_RECORDS, Reader,
    Resolved, SOURCE_FIELD, VALUE_INVALID, VALUE_MISSING, VALUE_OVERFLOW, iso, resolve,
};
use super::state::{Population, Producer, Set, SetKind, Shared};
use super::tables::BLOCK_ROWS;
use super::wire::{self, Fault, MAX_ROWS, PAYLOAD_TRANSPORT_BYTES, applied_limits, cancelled};

/// What a walk over a table holds: the buffer the table is read through and a block of
/// the population's membership.
pub(crate) const WALK_BYTES: u64 = READER_BYTES + States::BLOCK_BYTES as u64;
/// What a view holds beside its walk and its counters: the result on its way out and
/// the rows of a page.
const VIEW_FIXED_BYTES: u64 = 256 * 1024;
/// What one counter takes: originals, occurrences and the last row it counted.
const CLASS_BYTES: u64 = 24;
/// The most buckets of a histogram.
pub(crate) const MAX_BUCKETS: u64 = 100;
/// The bucket widths a histogram is drawn with, in seconds.
pub(crate) const BUCKET_WIDTHS: [i64; 10] = [
    1, 60, 300, 900, 3_600, 21_600, 86_400, 604_800, 2_592_000, 31_536_000,
];
/// The most a page of values or entries may take of a result, in both its forms.
const PAGE_TRANSPORT_BYTES: usize = 20 * 1024;
/// The most bytes of a value a result shows.
const SHOWN_VALUE_BYTES: usize = 512;

/// Occurrences and distinct originals of one class: a value, a bucket, missing.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Class {
    pub originals: u64,
    pub occurrences: u64,
    /// One more than the last row counted; the occurrences of an original are read in
    /// a row, so it is counted once.
    last: u64,
}

impl Class {
    fn add(&mut self, row: u64) {
        self.occurrences += 1;
        if self.last != row + 1 {
            self.originals += 1;
            self.last = row + 1;
        }
    }

    fn json(&self) -> Value {
        json!({"distinct_originals": self.originals, "occurrences": self.occurrences})
    }
}

/// Reads the occurrences of `population` that pass `resolved`, in the order of the
/// table, and gives each to `each`. Returns the records read.
fn walk(
    cancel: &AtomicBool,
    projection: &Projection,
    population: &Population,
    resolved: &Resolved,
    each: &mut dyn FnMut(&Occurrence),
) -> Result<u64, L2dError> {
    let mut reader = Reader::new(projection, 0);
    let mut block = vec![0u8; States::BLOCK_BYTES];
    let mut current = u64::MAX;
    while let Some(occurrence) = reader.next()? {
        if reader.read.is_multiple_of(16 * READER_RECORDS as u64) && cancel.load(Ordering::Relaxed)
        {
            return Err(cancelled());
        }
        let first = occurrence.row - occurrence.row % BLOCK_ROWS;
        if first != current {
            population.read_block(first, &mut block)?;
            current = first;
        }
        if code_in(&block, (occurrence.row - first) as usize) != State::In as u8 {
            continue;
        }
        if resolved.passes(&occurrence) {
            each(&occurrence);
        }
    }
    Ok(reader.read)
}

/// Asks the budget for `bytes` beside `held` and holds them.
fn charge(worker: &mut Worker<'_>, held: u64, bytes: u64, what: &str) -> Result<u64, Fault> {
    worker.budget.hold(held);
    worker.budget.require_whole(bytes, what)?;
    worker.budget.hold(held.saturating_add(bytes));
    Ok(held.saturating_add(bytes))
}

/// The population block of a view's result: whose occurrences were counted.
fn population_block(population: &Population, filter: &Filter, total: &Class) -> Value {
    let (originals, occurrences) = population.size();
    json!({
        "handle": population.handle(),
        "members": {"distinct_originals": originals, "occurrences": occurrences},
        "occurrence_filter": filter.describe(),
        "counted": total.json(),
        "unit": "occurrences of the members that pass the occurrence filter, and their \
                 distinct originals",
    })
}

fn definition(projection: &Projection, field: Option<usize>) -> Value {
    match field {
        None => json!({"name": SOURCE_FIELD, "from": "built_in",
                       "meaning": "the archive entry the occurrence was read from"}),
        Some(field) => projection.spec.fields[field].describe(),
    }
}

// --- The fields of a population ---------------------------------------------------------

/// The fields summary of `population` within `projection`: for `source` and every
/// declared field, the occurrences and originals with a value, missing, invalid and
/// beyond the retained limit, numbers apart from strings, and how many distinct
/// retained values occur. `held` is what the caller holds.
pub(crate) fn summary_for(
    worker: &mut Worker<'_>,
    projection: &Projection,
    population: &Population,
    held: u64,
) -> Result<Value, Fault> {
    let filter = Filter::of(population);
    let resolved = resolve(&filter, projection)?;
    let fields = projection.spec.fields.len();
    let seen_bytes =
        fields as u64 * (projection.spec.max_values / 8 + 8) + projection.entries() as u64 / 8 + 8;
    let held = charge(
        worker,
        held,
        WALK_BYTES + seen_bytes + 8 * fields as u64 * CLASS_BYTES,
        "counting the fields of a population: a block of the table and of the population, \
         and a mark for every retained value,",
    )?;
    let mut seen: Vec<Vec<u64>> = (0..fields)
        .map(|_| vec![0u64; (projection.spec.max_values as usize).div_ceil(64)])
        .collect();
    let mut entries = vec![0u64; projection.entries().div_ceil(64)];
    // Per field: value, missing, invalid, overflow, number, string.
    let mut classes = vec![[Class::default(); 6]; fields];
    let mut total = Class::default();
    let read = walk(
        worker.cancel,
        projection,
        population,
        &resolved,
        &mut |occurrence| {
            total.add(occurrence.row);
            let entry = occurrence.entry() as usize;
            entries[entry / 64] |= 1 << (entry % 64);
            for (field, counted) in classes.iter_mut().enumerate() {
                let id = occurrence.ids[field];
                match id {
                    VALUE_MISSING => counted[1].add(occurrence.row),
                    VALUE_INVALID => counted[2].add(occurrence.row),
                    VALUE_OVERFLOW => {
                        counted[0].add(occurrence.row);
                        counted[3].add(occurrence.row);
                    }
                    id => {
                        counted[0].add(occurrence.row);
                        seen[field][id as usize / 64] |= 1 << (id % 64);
                        if projection.dictionaries[field].numeric[id as usize] {
                            counted[4].add(occurrence.row);
                        } else {
                            counted[5].add(occurrence.row);
                        }
                    }
                }
            }
        },
    )?;
    let mut listed = Vec::with_capacity(fields + 1);
    let distinct_entries: u64 = entries
        .iter()
        .map(|word| u64::from(word.count_ones()))
        .sum();
    listed.push(json!({
        "field": SOURCE_FIELD,
        "definition": definition(projection, None),
        "per_occurrence": true,
        "with_value": total.json(),
        "missing": Class::default().json(),
        "invalid": Class::default().json(),
        "overflow": Class::default().json(),
        "distinct_values": {"count": distinct_entries, "exact": true},
        "value_counts_exact": true,
    }));
    for (field, counted) in classes.iter().enumerate() {
        let distinct: u64 = seen[field]
            .iter()
            .map(|word| u64::from(word.count_ones()))
            .sum();
        let overflowing = counted[3].occurrences > 0;
        listed.push(json!({
            "field": projection.spec.fields[field].name,
            "definition": definition(projection, Some(field)),
            "per_occurrence": !projection.spec.fields[field].per_original(),
            "with_value": counted[0].json(),
            "missing": counted[1].json(),
            "invalid": counted[2].json(),
            "overflow": counted[3].json(),
            "types": {"number": counted[4].json(), "string": counted[5].json()},
            "distinct_values": {
                "count": distinct,
                "exact": !overflowing,
                "retained_limit": projection.spec.max_values,
                "meaning": if overflowing {
                    "retained values that occur; values beyond the retained limit are \
                     counted as overflow and not told apart"
                } else {
                    "every distinct value that occurs"
                },
            },
            "value_counts_exact": !overflowing,
        }));
    }
    drop(seen);
    drop(entries);
    worker.budget.hold(held);
    Ok(json!({
        "unit": "occurrences, with their distinct originals; an original is counted in each \
                 class one of its counted occurrences is in",
        "counted": total.json(),
        "records_read": read,
        "fields": listed,
    }))
}

/// What the occurrences of `population` hold of the declared timestamp: with a valid
/// time, missing, invalid, and the earliest and latest time. `null` without one.
pub(crate) fn time_for(
    worker: &mut Worker<'_>,
    projection: &Projection,
    population: &Population,
    held: u64,
) -> Result<Value, Fault> {
    let Some(time) = &projection.spec.time else {
        return Ok(Value::Null);
    };
    charge(
        worker,
        held,
        WALK_BYTES,
        "counting the times of a population: a block of the table and of the population,",
    )?;
    let filter = Filter::of(population);
    let resolved = resolve(&filter, projection)?;
    let counted = times(worker.cancel, projection, population, &resolved)?;
    worker.budget.hold(held);
    Ok(json!({
        "field": "_time",
        "declared": time.describe(),
        "timezone": "UTC",
        "with_time": counted.with_time.json(),
        "missing_time": counted.missing.json(),
        "invalid_time": counted.invalid.json(),
        "observed": counted.range.map(|(earliest, latest)| json!({
            "earliest": iso(earliest), "latest": iso(latest)})),
    }))
}

struct Times {
    with_time: Class,
    missing: Class,
    invalid: Class,
    total: Class,
    range: Option<(i64, i64)>,
    read: u64,
}

fn times(
    cancel: &AtomicBool,
    projection: &Projection,
    population: &Population,
    resolved: &Resolved,
) -> Result<Times, L2dError> {
    let mut counted = Times {
        with_time: Class::default(),
        missing: Class::default(),
        invalid: Class::default(),
        total: Class::default(),
        range: None,
        read: 0,
    };
    counted.read = walk(
        cancel,
        projection,
        population,
        resolved,
        &mut |occurrence| {
            counted.total.add(occurrence.row);
            if occurrence.has_time() {
                counted.with_time.add(occurrence.row);
                counted.range = Some(match counted.range {
                    None => (occurrence.time, occurrence.time),
                    Some((low, high)) => (low.min(occurrence.time), high.max(occurrence.time)),
                });
            } else if occurrence.time == super::project::TIME_MISSING {
                counted.missing.add(occurrence.row);
            } else {
                counted.invalid.add(occurrence.row);
            }
        },
    )?;
    Ok(counted)
}

// --- The view jobs ----------------------------------------------------------------------

/// Which field a values page counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Named {
    Source,
    Field(usize),
}

pub(crate) enum FacetKind {
    /// Every field, without values.
    Summary,
    /// The values of one field, most occurrences first, a page at a time.
    Values {
        field: Named,
        limit: usize,
        cursor: Option<String>,
    },
    /// The time histogram, or the occurrences by entry when there is no time.
    Histogram {
        bucket_seconds: Option<i64>,
        limit: usize,
        cursor: Option<String>,
    },
}

pub(crate) struct FacetTask {
    pub population: Population,
    pub projection: Arc<Projection>,
    pub kind: FacetKind,
}

fn head(view: &str, task: &FacetTask) -> Map<String, Value> {
    let mut head = super::inspect::head(view, &task.population);
    head.insert(
        "projection".into(),
        json!({"handle": task.projection.handle, "revision": task.projection.revision}),
    );
    head
}

fn warm(read: u64) -> Value {
    json!({"archive_scans": 0, "rows_scored": 0, "rows_encoded": 0, "genai_rows_submitted": 0,
           "index_rows_read": read, "occurrence_records_read": read, "response_bytes": 0})
}

fn finish(mut result: Map<String, Value>, rows: usize) -> Result<String, Fault> {
    result.insert("applied_limits".into(), applied_limits(rows));
    let mut payload = wire::with_response_bytes(Value::Object(result));
    if wire::transport_bytes(&payload) > PAYLOAD_TRANSPORT_BYTES {
        return Err(Fault::runtime(
            "an inspect result does not fit the response limit",
        ));
    }
    // The job table keeps the result: it takes what its text does, no more.
    payload.shrink_to_fit();
    Ok(payload)
}

/// The cursor binding of a page of `view` over `task`.
fn binding(shared: &Shared, task: &FacetTask, view: &str, extra: &str) -> String {
    wire::cursor_binding(
        &shared.config.project,
        &[
            task.population.handle(),
            task.population.revision(),
            &task.projection.handle,
            &task.projection.revision,
            view,
            extra,
        ],
    )
}

fn after(cursor: &Option<String>, binding: &str) -> Result<u64, Fault> {
    match cursor {
        None => Ok(0),
        Some(cursor) => wire::cursor_position(cursor, binding).ok_or_else(|| {
            Fault::new(
                "stale_cursor",
                "`cursor` was issued for another handle, revision, projection, field or \
                 view; start again without it",
            )
        }),
    }
}

/// Runs a fields or histogram view.
pub(crate) fn run(worker: &mut Worker<'_>, task: &FacetTask) -> Result<Done, Fault> {
    let held = worker.base;
    worker.budget.hold(held);
    let result = match &task.kind {
        FacetKind::Summary => {
            let started = Instant::now();
            let counted = summary_for(worker, &task.projection, &task.population, held)?;
            let mut result = head("fields", task);
            let filter = Filter::of(&task.population);
            let total = Class {
                originals: counted["counted"]["distinct_originals"]
                    .as_u64()
                    .unwrap_or(0),
                occurrences: counted["counted"]["occurrences"].as_u64().unwrap_or(0),
                last: 0,
            };
            result.insert(
                "population".into(),
                population_block(&task.population, &filter, &total),
            );
            let read = counted["records_read"].as_u64().unwrap_or(0);
            result.insert("unit".into(), counted["unit"].clone());
            result.insert("fields".into(), counted["fields"].clone());
            let mut cost = warm(read);
            cost["elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
            result.insert("cost".into(), cost);
            result.insert(
                "disclosure".into(),
                worker.shared.disclosure(task.population.snapshot()),
            );
            result.insert("page".into(), Value::Null);
            finish(result, 0)?
        }
        FacetKind::Values {
            field,
            limit,
            cursor,
        } => values(worker, task, *field, *limit, cursor)?,
        FacetKind::Histogram {
            bucket_seconds,
            limit,
            cursor,
        } => histogram(worker, task, *bucket_seconds, *limit, cursor)?,
    };
    worker.budget.hold(worker.base);
    Ok(Done {
        result,
        commit: None,
    })
}

/// A page of rows built while it fits [`PAGE_TRANSPORT_BYTES`]: `rows` gives the row
/// at each position of `order` from `from` on.
fn page(
    order_len: usize,
    from: usize,
    limit: usize,
    row: &mut dyn FnMut(usize) -> Value,
) -> (Vec<Value>, usize, bool) {
    let mut rows = Vec::with_capacity(limit.min(MAX_ROWS));
    let mut bytes = 0;
    let mut at = from;
    let mut cut = false;
    while at < order_len && rows.len() < limit {
        let value = row(at);
        let cost = wire::transport_bytes(&value.to_string());
        if bytes + cost > PAGE_TRANSPORT_BYTES && !rows.is_empty() {
            cut = true;
            break;
        }
        bytes += cost;
        rows.push(value);
        at += 1;
    }
    (rows, at, cut)
}

fn values(
    worker: &mut Worker<'_>,
    task: &FacetTask,
    field: Named,
    limit: usize,
    cursor: &Option<String>,
) -> Result<String, Fault> {
    let started = Instant::now();
    let projection = &task.projection;
    let name = match field {
        Named::Source => SOURCE_FIELD.to_string(),
        Named::Field(field) => projection.spec.fields[field].name.clone(),
    };
    let binding = binding(worker.shared, task, "fields", &name);
    let from = after(cursor, &binding)?;
    let values = match field {
        Named::Source => projection.entries(),
        Named::Field(field) => projection.dictionaries[field].values.len(),
    };
    let held = worker.base;
    let held = charge(
        worker,
        held,
        WALK_BYTES + VIEW_FIXED_BYTES + values as u64 * (CLASS_BYTES + 4) + 3 * CLASS_BYTES,
        "counting the values of a field: a block of the table and of the population, and a \
         counter for every retained value,",
    )?;
    let filter = Filter::of(&task.population);
    let resolved = resolve(&filter, projection)?;
    let mut counts = vec![Class::default(); values];
    let (mut missing, mut invalid, mut overflow, mut total) = (
        Class::default(),
        Class::default(),
        Class::default(),
        Class::default(),
    );
    let read = walk(
        worker.cancel,
        projection,
        &task.population,
        &resolved,
        &mut |occurrence| {
            total.add(occurrence.row);
            let id = match field {
                Named::Source => occurrence.entry(),
                Named::Field(field) => occurrence.ids[field],
            };
            match id {
                VALUE_MISSING => missing.add(occurrence.row),
                VALUE_INVALID => invalid.add(occurrence.row),
                VALUE_OVERFLOW => overflow.add(occurrence.row),
                id => counts[id as usize].add(occurrence.row),
            }
        },
    )?;
    let text = |id: u32| -> &str {
        match field {
            Named::Source => projection.entry_path(id),
            Named::Field(field) => projection.value(field, id).unwrap_or(""),
        }
    };
    let mut order: Vec<u32> = (0..values as u32)
        .filter(|id| counts[*id as usize].occurrences > 0)
        .collect();
    order.sort_by(|a, b| {
        counts[*b as usize]
            .occurrences
            .cmp(&counts[*a as usize].occurrences)
            .then_with(|| text(*a).cmp(text(*b)))
    });
    let from = (from as usize).min(order.len());
    let mut disclosed = 0u64;
    let kv = matches!(field, Named::Field(field) if projection.spec.fields[field].per_original());
    let (rows, next, cut) = page(order.len(), from, limit, &mut |at| {
        let id = order[at];
        let value = text(id);
        let (shown, clipped) = wire::display(value, SHOWN_VALUE_BYTES);
        if kv {
            disclosed += value.len() as u64;
        }
        let kind = match field {
            Named::Source => "string",
            Named::Field(field) if projection.dictionaries[field].numeric[id as usize] => "number",
            Named::Field(_) => "string",
        };
        let counted = counts[id as usize];
        json!({"value": shown, "value_clipped": clipped, "type": kind,
               "distinct_originals": counted.originals, "occurrences": counted.occurrences,
               "share": share_of(counted.occurrences, total.occurrences)})
    });
    // A value read from a record's text is evidence of the source: it is shown only as
    // the disclosure profile allows and counted against the session's allowance.
    if disclosed > 0 {
        worker
            .shared
            .disclose(task.population.snapshot(), disclosed)?;
    }
    let returned = rows.len();
    let overflowing = overflow.occurrences > 0;
    let mut result = head("fields", task);
    result.insert(
        "population".into(),
        population_block(&task.population, &filter, &total),
    );
    result.insert(
        "field".into(),
        json!({
            "name": name,
            "definition": definition(projection, match field {
                Named::Source => None,
                Named::Field(field) => Some(field),
            }),
            "values": rows,
            "missing": missing.json(),
            "invalid": invalid.json(),
            "overflow": overflow.json(),
            "distinct_values": {"count": order.len(), "exact": !overflowing,
                                "retained_limit": projection.spec.max_values},
            "value_counts_exact": true,
            "order": "occurrences descending, then value ascending",
            "ranking_exact": !overflowing,
            "ranking": if overflowing {
                "among retained values only: a value beyond the retained limit is counted as \
                 overflow and may occur more often than a value listed"
            } else {
                "over every value that occurs"
            },
            "unit": "occurrences of the members that pass the occurrence filter, and their \
                     distinct originals; shares are of the counted occurrences",
        }),
    );
    result.insert(
        "page".into(),
        json!({
            "returned": returned,
            "values_total": order.len(),
            "next_cursor": (next < order.len()).then(|| wire::cursor(next as u64, &binding)),
            "cut_by_bytes": cut,
        }),
    );
    result.insert(
        "disclosure".into(),
        worker.shared.disclosure(task.population.snapshot()),
    );
    let mut cost = warm(read);
    cost["elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
    result.insert("cost".into(), cost);
    drop(counts);
    drop(order);
    worker.budget.hold(held);
    finish(result, returned)
}

/// The smallest of [`BUCKET_WIDTHS`] that draws `[low, high]` in at most
/// [`MAX_BUCKETS`] buckets.
fn auto_width(low: i64, high: i64) -> Option<i64> {
    BUCKET_WIDTHS
        .iter()
        .copied()
        .find(|width| buckets_of(low, high, *width) <= MAX_BUCKETS)
}

fn buckets_of(low: i64, high: i64, width: i64) -> u64 {
    (high.div_euclid(width) - low.div_euclid(width) + 1) as u64
}

fn histogram(
    worker: &mut Worker<'_>,
    task: &FacetTask,
    asked: Option<i64>,
    limit: usize,
    cursor: &Option<String>,
) -> Result<String, Fault> {
    let started = Instant::now();
    let projection = &task.projection;
    let held = worker.base;
    let filter = Filter::of(&task.population);
    let resolved = resolve(&filter, projection)?;
    let held = charge(
        worker,
        held,
        WALK_BYTES + VIEW_FIXED_BYTES + MAX_BUCKETS * CLASS_BYTES,
        "drawing a time histogram: a block of the table and of the population, and a \
         counter for every bucket,",
    )?;
    // An undeclared timestamp is answered with what is missing before a job is
    // scheduled (`projection_missing`); `time_unavailable` is for a declared one.
    let Some(time) = projection.spec.time.clone() else {
        return Err(Fault::runtime(
            "a histogram was scheduled over a projection that declares no timestamp",
        ));
    };
    let counted = times(worker.cancel, projection, &task.population, &resolved)?;
    let Some((low, high)) = counted.range else {
        return alternative(
            worker,
            task,
            held,
            "no counted occurrence has a valid timestamp",
            Some(&counted),
            limit,
            cursor,
            started,
        );
    };
    let width = match asked {
        Some(width) => {
            if buckets_of(low, high, width) > MAX_BUCKETS {
                return Err(Fault::new(
                    "too_many_buckets",
                    format!(
                        "the counted occurrences span {} to {}: {} buckets of {width} s, more \
                         than {MAX_BUCKETS}; the smallest width that draws them is {} s, or \
                         drill into a bucket first",
                        iso(low),
                        iso(high),
                        buckets_of(low, high, width),
                        auto_width(low, high).map_or("none".into(), |width| width.to_string())
                    ),
                ));
            }
            width
        }
        None => auto_width(low, high).ok_or_else(|| {
            Fault::new(
                "too_many_buckets",
                format!(
                    "the counted occurrences span {} to {}, more than {MAX_BUCKETS} buckets of \
                     the widest width; drill into a part of the range first",
                    iso(low),
                    iso(high)
                ),
            )
        })?,
    };
    let first = low.div_euclid(width) * width;
    let count = buckets_of(low, high, width) as usize;
    let mut buckets = vec![Class::default(); count];
    let read = walk(
        worker.cancel,
        projection,
        &task.population,
        &resolved,
        &mut |occurrence| {
            if occurrence.has_time() {
                let at = ((occurrence.time - first) / width) as usize;
                buckets[at].add(occurrence.row);
            }
        },
    )?;
    let rows: Vec<Value> = buckets
        .iter()
        .enumerate()
        .map(|(at, bucket)| {
            json!({"start": iso(first + at as i64 * width),
                   "distinct_originals": bucket.originals, "occurrences": bucket.occurrences})
        })
        .collect();
    let mut result = head("histogram", task);
    result.insert(
        "population".into(),
        population_block(&task.population, &filter, &counted.total),
    );
    result.insert(
        "time".into(),
        json!({
            "field": "_time",
            "declared": time.describe(),
            "timezone": "UTC",
            "bucket_seconds": width,
            "bucket_boundaries": "[start, start + bucket_seconds), aligned to multiples of the \
                                  width since 1970-01-01T00:00:00Z",
            "range": {"start": iso(first), "end": iso(first + count as i64 * width)},
            "observed": {"earliest": iso(low), "latest": iso(high)},
            "buckets": rows,
            "with_time": counted.with_time.json(),
            "missing_time": counted.missing.json(),
            "invalid_time": counted.invalid.json(),
            "unit": "occurrences with their distinct originals; an original is counted once in \
                     each bucket one of its counted occurrences falls in",
            "exact": true,
        }),
    );
    result.insert("page".into(), Value::Null);
    let mut cost = warm(counted.read + read);
    cost["elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
    result.insert("cost".into(), cost);
    drop(buckets);
    worker.budget.hold(held);
    finish(result, count)
}

/// `time_unavailable`: the occurrences by archive entry in directory order, labelled as
/// what they are, with the occurrences without a time when a timestamp was declared.
#[allow(clippy::too_many_arguments)]
fn alternative(
    worker: &mut Worker<'_>,
    task: &FacetTask,
    held: u64,
    reason: &str,
    counted: Option<&Times>,
    limit: usize,
    cursor: &Option<String>,
    started: Instant,
) -> Result<String, Fault> {
    let projection = &task.projection;
    let entries = projection.entries();
    let held = charge(
        worker,
        held,
        entries as u64 * CLASS_BYTES,
        "counting the occurrences of a population by archive entry,",
    )?;
    let binding = binding(worker.shared, task, "histogram", "entries");
    let from = after(cursor, &binding)? as usize;
    let filter = Filter::of(&task.population);
    let resolved = resolve(&filter, projection)?;
    let mut by_entry = vec![Class::default(); entries];
    let mut total = Class::default();
    let read = walk(
        worker.cancel,
        projection,
        &task.population,
        &resolved,
        &mut |occurrence| {
            total.add(occurrence.row);
            by_entry[occurrence.entry() as usize].add(occurrence.row);
        },
    )?;
    let order: Vec<usize> = (0..entries)
        .filter(|entry| by_entry[*entry].occurrences > 0)
        .collect();
    let from = from.min(order.len());
    let (rows, next, cut) = page(order.len(), from, limit, &mut |at| {
        let entry = order[at];
        let (shown, clipped) =
            wire::display(projection.entry_path(entry as u32), wire::MAX_PATH_BYTES);
        json!({"entry": shown, "entry_clipped": clipped, "entry_index": entry,
               "distinct_originals": by_entry[entry].originals,
               "occurrences": by_entry[entry].occurrences})
    });
    let returned = rows.len();
    let mut result = head("histogram", task);
    result.insert("status".into(), json!("time_unavailable"));
    result.insert(
        "population".into(),
        population_block(&task.population, &filter, &total),
    );
    result.insert(
        "time".into(),
        json!({
            "available": false,
            "reason": reason,
            "declared": projection.spec.time.as_ref().map(|time| time.describe()),
            "with_time": counted.map(|counted| counted.with_time.json()),
            "missing_time": counted.map(|counted| counted.missing.json()),
            "invalid_time": counted.map(|counted| counted.invalid.json()),
        }),
    );
    result.insert(
        "alternative".into(),
        json!({
            "kind": "entry_order",
            "label": "occurrences by archive entry in directory order; this is not time",
            "field": SOURCE_FIELD,
            "drill": "refine with {\"kind\": \"field\", \"field\": \"source\", \"value\": ENTRY}",
            "rows": rows,
            "unit": "occurrences with their distinct originals; an original is counted once in \
                     each entry one of its counted occurrences was read from",
        }),
    );
    result.insert(
        "page".into(),
        json!({
            "returned": returned,
            "entries_total": order.len(),
            "next_cursor": (next < order.len()).then(|| wire::cursor(next as u64, &binding)),
            "cut_by_bytes": cut,
        }),
    );
    let mut cost = warm(read + counted.map_or(0, |counted| counted.read));
    cost["elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
    result.insert("cost".into(), cost);
    drop(by_entry);
    worker.budget.hold(held);
    finish(result, returned)
}

// --- Drilling into a field value or a time bucket ----------------------------------------

pub(crate) struct DrillTask {
    pub population: Population,
    pub projection: Arc<Projection>,
    pub clause: Clause,
}

/// The child view of `task`: the members of the population with at least one counted
/// occurrence that passes the clause, on the same snapshot. Its counts are of whole
/// originals in both units, as every view's; `matching_occurrences` counts the
/// occurrences that pass, with the population's counted occurrences as the parent.
pub(crate) fn drill(worker: &mut Worker<'_>, task: &DrillTask) -> Result<Done, Fault> {
    let started = Instant::now();
    let population = &task.population;
    let snapshot = population.snapshot().clone();
    let projection = &task.projection;
    let rows = snapshot.scope.originals();
    let action = task.clause.describe();
    let handle = wire::handle(
        "view",
        &worker.shared.config.project,
        &[
            &snapshot.handle,
            population.handle(),
            &json!({"refine": action}).to_string(),
        ],
    );
    if let Some(known) = registry(worker.shared).set(&handle) {
        return Ok(Done {
            result: known.result.clone(),
            commit: None,
        });
    }
    let parent_filter = Filter::of(population);
    let mut filter = parent_filter.clone();
    filter.clauses.push(task.clause.clone());
    if filter.clauses.len() > MAX_CLAUSES {
        return Err(Fault::invalid(format!(
            "a field or time view is at most {MAX_CLAUSES} drills deep; compose a view of this \
             one and drill into that"
        )));
    }
    let parent_resolved = resolve(&parent_filter, projection)?;
    let resolved = resolve(&filter, projection)?;
    let producer = Producer::Occurrence {
        described: json!({
            "kind": "occurrence",
            "action": action,
            "occurrence_filter": filter.describe(),
            "projection": projection.handle,
            "projection_revision": projection.revision,
        }),
        filter: Arc::new(filter),
    };
    let fixed = (WALK_BYTES + 2 * States::BLOCK_BYTES as u64 + BLOCK_ROWS / 8)
        .saturating_add(TALLY_BYTES + COMPOSE_FIXED_BYTES + ARTIFACT_FIXED_BYTES)
        .saturating_add(producer.held_bytes());
    let base = worker.base;
    let held = charge(
        worker,
        base,
        fixed,
        "drilling into a field value or a time bucket: a block of the table, of the \
         population and of the view at a time,",
    )?;
    let on_disk = tables_on_disk(worker, States::bytes_for(rows));
    if !on_disk {
        worker
            .budget
            .hold(held.saturating_add(States::bytes_for(rows)));
    }
    let mut states = if on_disk {
        let dir = set_dir(
            worker,
            States::bytes_for(rows),
            "placing the membership of a field or time view on disk",
        )?;
        States::on_disk(&dir, rows)?
    } else {
        States::in_memory(rows, State::Out)
    };
    let mut reader = Reader::new(projection, 0);
    let mut pending: Option<Occurrence> = None;
    let mut within = vec![0u8; States::BLOCK_BYTES];
    let mut codes = vec![0u8; States::BLOCK_BYTES];
    let mut matched = vec![0u64; (BLOCK_ROWS / 64) as usize];
    let (mut parent_occurrences, mut kept_occurrences) = (0u64, 0u64);
    let mut first = 0;
    while first < rows {
        worker.check_cancel()?;
        let in_block = population.read_block(first, &mut within)?;
        matched.fill(0);
        loop {
            let occurrence = match pending.take() {
                Some(occurrence) => occurrence,
                None => match reader.next()? {
                    Some(occurrence) => occurrence,
                    None => break,
                },
            };
            if occurrence.row >= first + BLOCK_ROWS {
                pending = Some(occurrence);
                break;
            }
            let at = (occurrence.row - first) as usize;
            if code_in(&within, at) != State::In as u8 || !parent_resolved.passes(&occurrence) {
                continue;
            }
            parent_occurrences += 1;
            if resolved.passes(&occurrence) {
                kept_occurrences += 1;
                matched[at / 64] |= 1 << (at % 64);
            }
        }
        for at in 0..in_block {
            let member = code_in(&within, at) == State::In as u8;
            let state = if member && matched[at / 64] & (1 << (at % 64)) != 0 {
                State::In
            } else {
                State::Out
            };
            put_code(&mut codes, at, state as u8);
        }
        states.write_block(first, &codes, in_block)?;
        first += BLOCK_ROWS;
    }
    let records_read = reader.read;
    drop(reader);
    let (tally, revision, _) = tally(&snapshot, population, &states)?;
    let (parent_originals, parent_all) = tally.parent();
    if (parent_originals, parent_all) != population.size() {
        return Err(Fault::runtime(
            "the members of a field or time view's parent do not add up to the parent",
        ));
    }
    let (kept_originals, kept_all) = tally.of(State::In);
    let mut cost = Cost::new(started);
    cost.set("index_rows_read", rows + records_read)
        .set("occurrence_records_read", records_read)
        .set("candidates", parent_originals)
        .set("output", kept_originals)
        .set("tables_on_disk", on_disk)
        .set("disk_bytes", states.disk_bytes())
        .set("disk_peak_bytes", states.disk_bytes());
    let result = wire::with_response_bytes(json!({
        "v": wire::VERSION,
        "status": "ready",
        "view": handle,
        "snapshot": snapshot.handle,
        "parent": population.handle(),
        "operation": {"mode": "refine", "action": action,
                      "projection": projection.handle,
                      "projection_revision": projection.revision},
        "revision": revision,
        "counts": tally.counts(),
        "occurrence_counts": tally.occurrence_counts(),
        "occurrence_counts_meaning": "every occurrence of the originals in each state, as set \
                                      algebra, regions and allocation count them",
        "matching_occurrences": {
            "unit": "occurrences",
            "parent": parent_occurrences,
            "in": kept_occurrences,
            "out": parent_occurrences - kept_occurrences,
            "retained_share": share_of(kept_occurrences, parent_occurrences),
            "meaning": "the parent's counted occurrences that pass the drill; the in-originals \
                        are their distinct originals",
        },
        "occurrence_filter": match &producer {
            Producer::Occurrence { filter, .. } => filter.describe(),
            _ => Value::Null,
        },
        "coverage": coverage(&tally, "not_evaluated"),
        "complete": snapshot.scope.completeness().complete,
        "root": {
            "snapshot": snapshot.handle,
            "distinct_originals": snapshot.scope.originals(),
            "occurrences": snapshot.scope.occurrences(),
            "retained_share": share_of(kept_originals, snapshot.scope.originals()),
            "occurrence_retained_share": share_of(kept_all, snapshot.scope.occurrences()),
        },
        "cost": cost.finish(worker),
    }));
    require_fit(&result)?;
    let mut result = result;
    result.shrink_to_fit();
    let kept = states
        .held_bytes()
        .saturating_add(producer.held_bytes())
        .saturating_add(2 * result.len() as u64 + handle.capacity() as u64)
        .saturating_add(ARTIFACT_FIXED_BYTES);
    let set = Set {
        handle,
        kind: SetKind::View,
        parent: population.as_set().cloned(),
        snapshot: snapshot.clone(),
        states,
        scores: None,
        owns_scores: false,
        producer,
        tally,
        revision,
        result: result.clone(),
        held: kept,
        inputs: Vec::new(),
        allocation: None,
    };
    worker.budget.hold(worker.base.saturating_add(kept));
    Ok(Done {
        result,
        commit: Some(Commit::Set(Arc::new(set))),
    })
}

// --- The occurrences of one original ------------------------------------------------------

/// The records an evidence listing reads at a time, and the most it lists.
const LISTING_RECORDS: usize = 64;
pub(crate) const MAX_LISTED: usize = 25;
/// The most a listing may take of an evidence result, in both its forms.
const LISTING_TRANSPORT_BYTES: usize = 6 * 1024;
/// What a listing holds on the calling thread: its block of records. Part of the
/// adapter's share ([`super::ADAPTER_FIXED_BYTES`]).
pub(crate) const LISTING_BYTES: u64 =
    (LISTING_RECORDS * super::project::MAX_RECORD_BYTES) as u64 + 16 * 1024;

/// Every occurrence of the original in `row`, from the projection: entry and line, the
/// time, the entry-derived field values and whether it passes `filter` (the drill-back
/// from a field or time view to the original in the wider scope). The first
/// [`MAX_LISTED`] by position are listed; all are counted.
pub(crate) fn listing(projection: &Projection, row: u64, filter: &Filter) -> Result<Value, Fault> {
    let resolved = resolve(filter, projection)?;
    let width = projection.width;
    let fields = projection.spec.fields.len();
    let mut at = projection.first_of_row(row)?;
    let mut buffer = vec![0u8; LISTING_RECORDS * width];
    let (mut total, mut matching) = (0u64, 0u64);
    let mut rows = Vec::new();
    let mut bytes = 0usize;
    let mut cut = false;
    'read: while at < projection.records {
        let now = (projection.records - at).min(LISTING_RECORDS as u64) as usize;
        projection
            .table
            .read(at * width as u64, &mut buffer[..now * width])?;
        for record in buffer[..now * width].chunks(width) {
            let occurrence = Occurrence::decode(record, fields);
            if occurrence.row != row {
                break 'read;
            }
            total += 1;
            let passes = resolved.passes(&occurrence);
            matching += u64::from(passes);
            if rows.len() < MAX_LISTED && !cut {
                let mut values = Map::new();
                for (field, spec) in projection.spec.fields.iter().enumerate() {
                    if spec.per_original() {
                        continue;
                    }
                    let id = occurrence.ids[field];
                    values.insert(
                        spec.name.clone(),
                        match projection.value(field, id) {
                            Some(value) => json!(wire::display(value, 128).0),
                            None if id == VALUE_OVERFLOW => json!({"overflow": true}),
                            None if id == VALUE_INVALID => json!({"invalid": true}),
                            None => Value::Null,
                        },
                    );
                }
                let time = if occurrence.has_time() {
                    json!(iso(occurrence.time))
                } else {
                    Value::Null
                };
                let time_state = if occurrence.has_time() {
                    "valid"
                } else if occurrence.time == super::project::TIME_MISSING {
                    "missing"
                } else {
                    "invalid"
                };
                let row = json!({
                    "entry": wire::display(projection.entry_path(occurrence.entry()), 160).0,
                    "line": occurrence.line(),
                    "time": time,
                    "time_state": if projection.spec.time.is_some() { time_state } else { "not_declared" },
                    "fields": values,
                    "in_view": passes,
                });
                let cost = wire::transport_bytes(&row.to_string());
                if bytes + cost > LISTING_TRANSPORT_BYTES {
                    cut = true;
                } else {
                    bytes += cost;
                    rows.push(row);
                }
            }
        }
        at += now as u64;
    }
    let listed = rows.len() as u64;
    Ok(json!({
        "projection": projection.handle,
        "order": "position ascending",
        "total": total,
        "listed": listed,
        "clipped": listed < total,
        "in_view": if filter.clauses.is_empty() { Value::Null } else { json!(matching) },
        "occurrence_filter": filter.describe(),
        "rows": rows,
        "meaning": "every occurrence of the original in the snapshot; `in_view` says whether \
                    it passes the occurrence filter of the handle the evidence was asked under",
    }))
}

/// The occurrences of a field or time view: its counted occurrences, for its overview.
pub(crate) fn matching(set: &Set) -> Option<Value> {
    match &set.producer {
        Producer::Occurrence { filter, .. } => {
            let result: Value = serde_json::from_str(&set.result).ok()?;
            Some(json!({
                "matching_occurrences": result["matching_occurrences"].clone(),
                "occurrence_filter": filter.describe(),
            }))
        }
        _ => None,
    }
}
