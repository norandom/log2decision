//! Pattern and neighbour drill-down (triager task 2.1, requirements 2.4 and 6.2): the
//! project's pattern catalogue, its refinement and its direct exemplar retrieval,
//! bound to the immutable sets of the service.
//!
//! Nothing here groups, splits, matches or measures a distance itself. A catalogue is
//! `PatternCatalogue::build_archive` over the `SelectionSet` the project's archive
//! search gives for the filter of a selection; a refinement revision is
//! `PatternRefinement` (`from_catalogue`, `split`, `merge`, `proposed_regex`,
//! `preview_query`); neighbours are `l2d_decide::exploration::explore`. What this
//! module adds is the binding: the saved search must name exactly the members of the
//! selection it is listed for, every revision is its own handle with its parent, and a
//! pattern or a neighbourhood becomes a selection like every other, with the states
//! of every original of the snapshot.
//!
//! # Memory
//!
//! A refinement is kept as the file its own `save` writes, in a directory of the work
//! directory, and is loaded for every action: between jobs the service holds only its
//! description. An action holds, for as long as it runs, what the project's own
//! loader asks for the file (64 times its bytes and [`NATIVE_FIXED_BYTES`]) and then
//! what the discovery commands hold for a loaded refinement and its output (32 times
//! what it keeps and [`NATIVE_FIXED_BYTES`]); the project's operations ask for what
//! they need beside that themselves, and an action that does not fit is refused with
//! the figures. The encoder of a vector search is held as its source states.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use l2d_core::patterns::refinement::PatternRefinement;
use l2d_core::patterns::{Pattern, PatternCatalogue, PatternSettings};
use l2d_core::selections::{Member, SearchQuery, SearchSettings, SelectionSet};
use l2d_core::{L2dError, MemoryBudget};
use l2d_decide::exploration::{self, CharacterTrigram, ExplorationEncoder, ExploreSettings};
use serde_json::{Map, Value, json};

use super::membership::{ABSTAINED, State};
use super::ops::{
    ARTIFACT_FIXED_BYTES, Commit, Cost, Done, Listed, Scratch, Worker, known_selection, registry,
    require_fit, select_listed,
};
use super::state::{Population, Producer, Refinement, Set, Snapshot};
use super::tables::require_disk;
use super::wire::{self, DEFAULT_ROWS, Fault, VERSION};
use crate::scope::archive_summary_bytes;

/// The fixed part of what the project's pattern operations ask the budget for.
const NATIVE_FIXED_BYTES: u64 = 4 * 1024 * 1024;
/// The most members of a selection a catalogue is built for: the cap of the
/// project's saved searches.
pub(crate) const MAX_PATTERN_MEMBERS: u64 = 100_000;
/// The most bytes of a template or a proposed query a result shows.
const MAX_TEMPLATE_BYTES: usize = 256;
/// The most the rows of one page of patterns take of a tool result, in both its
/// forms: what the response limit leaves beside the head of the result.
const PAGE_ROWS_TRANSPORT_BYTES: usize = wire::PAYLOAD_TRANSPORT_BYTES - 8 * 1024;
/// The most record handles a preview lists of each of its four groups.
const PREVIEW_IDS: usize = 10;
/// The most neighbours a search returns, and the most of them its result lists.
pub(crate) const MAX_NEIGHBOURS: u64 = 1_000;

/// Which members of the refinement a neighbour search considers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Within {
    Pattern,
    OtherPatterns,
    All,
}

impl Within {
    pub fn name(self) -> &'static str {
        match self {
            Within::Pattern => "pattern",
            Within::OtherPatterns => "other_patterns",
            Within::All => "all",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Method {
    CharacterTrigram,
    Vector,
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Method::CharacterTrigram => "character_trigram",
            Method::Vector => "vector",
        }
    }
}

/// One page of the patterns of a refinement.
pub(crate) struct Page {
    /// The pattern after which the page starts.
    pub after: Option<u64>,
    pub limit: usize,
    /// Whether the templates are shown: text of the source, counted as evidence.
    pub templates: bool,
}

pub(crate) enum RefineTask {
    /// Builds the catalogue of a filter selection and its first refinement.
    Patterns { set: Arc<Set> },
    List {
        refinement: Arc<Refinement>,
        page: Page,
    },
    Split {
        refinement: Arc<Refinement>,
        pattern: u64,
        regex: String,
    },
    Merge {
        refinement: Arc<Refinement>,
        patterns: Vec<usize>,
    },
    Preview {
        refinement: Arc<Refinement>,
        pattern: u64,
        regex: Option<String>,
    },
    Neighbours {
        refinement: Arc<Refinement>,
        /// The position of the exemplar.
        exemplar: u64,
        method: Method,
        within: Within,
        top_k: u64,
        radius: f64,
        /// The refinement of a wider selection to search as well.
        target: Option<Arc<Refinement>>,
    },
    /// The selection of the members of one pattern (`filter` with syntax `pattern`).
    Select {
        scope: Population,
        refinement: Arc<Refinement>,
        pattern: u64,
    },
}

fn runtime(doing: &str, path: &Path, error: &dyn std::fmt::Display) -> L2dError {
    L2dError::Runtime {
        detail: format!("{doing} {} failed: {error}", path.display()),
    }
}

/// Loads a refinement revision under the budget and returns it with what the action
/// holds for it from then on, beside `worker.base`.
fn load(
    worker: &mut Worker<'_>,
    refinement: &Refinement,
    beside: u64,
) -> Result<(PatternRefinement, u64), L2dError> {
    let base = worker.base.saturating_add(beside);
    worker.budget.hold(base);
    let loading = refinement
        .disk_bytes
        .saturating_mul(64)
        .saturating_add(NATIVE_FIXED_BYTES);
    worker
        .budget
        .require_whole(loading, "loading a pattern refinement of this server")?;
    worker.budget.hold(base.saturating_add(loading));
    let loaded = PatternRefinement::load(&refinement.path, worker.budget)?;
    if loaded.revision != refinement.revision {
        return Err(L2dError::Runtime {
            detail: "a refinement file holds another revision than it was saved with".into(),
        });
    }
    let held = work_share(&loaded);
    worker.budget.hold(base);
    worker.budget.require_whole(
        held,
        "working on a loaded pattern refinement and its output",
    )?;
    worker.budget.hold(base.saturating_add(held));
    Ok((loaded, held))
}

/// What an action holds for a loaded refinement and what it builds from it: the
/// figure the project's discovery commands hold.
fn work_share(refinement: &PatternRefinement) -> u64 {
    refinement
        .held_bytes()
        .saturating_mul(32)
        .saturating_add(NATIVE_FIXED_BYTES)
}

fn pattern_row(pattern: &Pattern) -> Map<String, Value> {
    let mut row = Map::new();
    row.insert("pattern".into(), json!(pattern.id));
    row.insert("members".into(), json!(pattern.counts.members));
    row.insert("occurrences".into(), json!(pattern.counts.occurrences));
    row.insert(
        "distinct_texts".into(),
        json!(pattern.counts.distinct_texts),
    );
    row.insert(
        "possible_outcome_collision".into(),
        json!(pattern.possible_outcome_collision),
    );
    row
}

/// The page of patterns after `page.after`, and the bytes of template text in it.
fn patterns_page(loaded: &PatternRefinement, page: &Page) -> (Value, u64) {
    let mut rows = Vec::with_capacity(page.limit);
    let mut shown_bytes = 0u64;
    let mut clipped = false;
    let mut last = None;
    let mut more = false;
    let (mut used, mut by_bytes) = (0usize, false);
    for pattern in &loaded.groups {
        if page.after.is_some_and(|after| pattern.id as u64 <= after) {
            continue;
        }
        if rows.len() == page.limit {
            more = true;
            break;
        }
        let mut row = pattern_row(pattern);
        let mut template_bytes = 0;
        let mut cut = false;
        if page.templates {
            let (shown, was_cut) = wire::display(&pattern.template, MAX_TEMPLATE_BYTES);
            template_bytes = shown.len() as u64;
            cut = was_cut;
            row.insert("template".into(), json!(shown));
        }
        let row = Value::Object(row);
        let written = row.to_string();
        let transport = 2 * (written.len() + 1) + wire::escape_bytes(&written);
        if used + transport > PAGE_ROWS_TRANSPORT_BYTES {
            // The page ends where the response would be full; it goes on from here.
            more = true;
            by_bytes = true;
            break;
        }
        used += transport;
        shown_bytes += template_bytes;
        clipped |= cut;
        last = Some(pattern.id);
        rows.push(row);
    }
    let returned = rows.len();
    (
        json!({
            "patterns": rows,
            "page": {
                "limit": page.limit,
                "returned": returned,
                "total": loaded.groups.len(),
                "next_after": if more { json!(last) } else { Value::Null },
            },
            "clipping": {"clipped": clipped || by_bytes, "rows_by_bytes": by_bytes,
                         "fields": if clipped { json!(["patterns.template"]) } else { json!([]) }},
        }),
        shown_bytes,
    )
}

/// What every result about a refinement states.
fn head(refinement: &str, set: &Set, loaded: &PatternRefinement, parent: Option<&str>) -> Value {
    let settings = &loaded.catalogue.settings;
    let (members, occurrences) = set.tally.of(State::In);
    json!({
        "v": VERSION,
        "status": "ready",
        "refinement": refinement,
        "set": set.handle,
        "snapshot": set.snapshot.handle,
        "revision": loaded.revision,
        "parent": parent,
        "action": wire::display(&loaded.action, 200).0,
        "groups": loaded.groups.len(),
        "members": {"distinct_originals": members, "occurrences": occurrences},
        "ungrouped": {"members": loaded.ungrouped.members,
                      "occurrences": loaded.ungrouped.occurrences},
        "method": {
            "catalogue": loaded.catalogue.algorithm_revision,
            "catalogue_revision": loaded.catalogue_revision,
            "selection_revision": loaded.selection_revision,
            "engine": settings.engine,
            "max_patterns": settings.max_patterns,
            "similarity_per_mille": settings.similarity_per_mille,
            "masks": settings.masks.revision,
            "membership": "exact saved rows; a template is a summary, not a matcher",
        },
    })
}

fn merged(mut head: Value, more: Value) -> Value {
    if let (Some(head), Value::Object(more)) = (head.as_object_mut(), more) {
        head.extend(more);
    }
    head
}

/// Saves `loaded` as a new refinement of `set` in a directory of its own and returns
/// the artifact. `changed` and the first page of patterns are part of its result.
fn publish(
    worker: &mut Worker<'_>,
    set: &Arc<Set>,
    loaded: &PatternRefinement,
    parent: Option<&Refinement>,
    dir: Option<PathBuf>,
    changed: Value,
    mut cost: Cost,
) -> Result<Done, Fault> {
    let project = &worker.shared.config.project;
    let handle = wire::handle("refn", project, &[&set.handle, &loaded.revision]);
    if let Some(known) = registry(worker.shared).refinement(&handle) {
        // The same action on the same revision is the same refinement.
        if let Some(dir) = dir {
            let _ = std::fs::remove_dir_all(dir);
        }
        return Ok(Done {
            result: known.result.clone(),
            commit: None,
        });
    }
    let dir = match dir {
        Some(dir) => dir,
        None => {
            let dir = worker.fresh_dir("refine");
            std::fs::create_dir_all(&dir).map_err(|error| runtime("creating", &dir, &error))?;
            dir
        }
    };
    // Removed again unless the refinement is committed with its file.
    let scratch = Scratch(dir.clone());
    let config = &worker.shared.config;
    require_disk(
        &config.work_root,
        loaded.held_bytes(),
        config.free_disk_bytes,
        "saving a pattern refinement",
    )?;
    let path = dir.join("refinement.json");
    loaded.save(&path, worker.budget)?;
    let disk_bytes = std::fs::metadata(&path)
        .map_err(|error| runtime("reading the size of", &path, &error))?
        .len();
    let first = Page {
        after: None,
        limit: DEFAULT_ROWS,
        templates: false,
    };
    let (page, _) = patterns_page(loaded, &first);
    cost.set("output", loaded.groups.len() as u64)
        .set("candidates", set.tally.of(State::In).0)
        .set("disk_bytes", disk_bytes)
        .set("disk_peak_bytes", disk_bytes);
    let mut result = merged(
        head(
            &handle,
            set,
            loaded,
            parent.map(|parent| parent.handle.as_str()),
        ),
        page,
    );
    result["changed"] = changed;
    result["disclosure"] = worker.shared.disclosure(&set.snapshot);
    result["cost"] = cost.finish(worker);
    let result = wire::with_response_bytes(result);
    require_fit(&result)?;
    let held = 2 * result.len() as u64
        + (handle.capacity() + loaded.revision.len() + path.as_os_str().len()) as u64
        + parent.map_or(0, |parent| parent.handle.len() as u64)
        + ARTIFACT_FIXED_BYTES;
    worker.budget.hold(worker.base);
    worker
        .budget
        .require_whole(held, "keeping the description of a pattern refinement")?;
    worker.budget.hold(worker.base.saturating_add(held));
    let refinement = Refinement {
        handle,
        set: set.clone(),
        parent: parent.map(|parent| parent.handle.clone()),
        revision: loaded.revision.clone(),
        path,
        disk_bytes,
        result: result.clone(),
        held,
    };
    std::mem::forget(scratch);
    Ok(Done {
        result,
        commit: Some(Commit::Refinement(Arc::new(refinement))),
    })
}

/// The search of the project that gives the members of a filter selection.
fn query_of(set: &Set) -> Result<SearchQuery, Fault> {
    let refused = |what: &str| {
        Fault::refused(format!(
            "patterns are listed for a `filter` selection (literal, glob or regex) over a \
             whole snapshot, the form the project's pattern catalogue takes its members in; \
             `{}` is {what}. Select the filter in the snapshot and refine that",
            set.handle
        ))
    };
    if set.parent.is_some() {
        return Err(refused("a set inside another set"));
    }
    match &set.producer {
        Producer::Filter {
            syntax,
            query,
            case_insensitive,
        } => {
            let (text, insensitive) = (query.clone(), *case_insensitive);
            Ok(match *syntax {
                "literal" => SearchQuery::Literal { text, insensitive },
                "glob" => SearchQuery::Glob {
                    pattern: text,
                    insensitive,
                },
                _ => SearchQuery::Regex {
                    pattern: text,
                    insensitive,
                },
            })
        }
        _ => Err(refused("made by another producer")),
    }
}

fn count_scan(worker: &Worker<'_>, snapshot: &Snapshot) -> u64 {
    let unpacked = snapshot.scope.selected_bytes().unwrap_or_default().unpacked;
    let activity = &worker.shared.activity;
    activity.archive_scans.fetch_add(1, Ordering::Relaxed);
    activity
        .bytes_unpacked
        .fetch_add(unpacked, Ordering::Relaxed);
    unpacked
}

fn patterns(worker: &mut Worker<'_>, set: &Arc<Set>, started: Instant) -> Result<Done, Fault> {
    let known = registry(worker.shared)
        .refinements()
        .find(|known| known.parent.is_none() && known.set.handle == set.handle)
        .cloned();
    if let Some(known) = known {
        // The catalogue of a selection is built once: nothing is read again.
        return Ok(Done {
            result: known.result.clone(),
            commit: None,
        });
    }
    let query = query_of(set)?;
    let (members, _) = set.tally.of(State::In);
    if members == 0 || members > MAX_PATTERN_MEMBERS {
        return Err(Fault::refused(format!(
            "the selection has {members} members, and a pattern catalogue is built for 1 to \
             {MAX_PATTERN_MEMBERS}; narrow the filter. Nothing was read"
        )));
    }
    let snapshot = set.snapshot.clone();
    let limits = worker.shared.config.patterns;
    let settings = PatternSettings {
        max_patterns: limits.max_patterns,
        max_text_bytes: limits.max_text_bytes,
        ..PatternSettings::default()
    };
    let config = &worker.shared.config;
    // The spooled originals, the membership stream and the refinement, each at most a
    // selected text of the longest kind written as JSON.
    let disk = members.saturating_mul(3 * (2 * settings.max_text_bytes as u64 + 512));
    require_disk(
        &config.work_root,
        disk,
        config.free_disk_bytes,
        "building a pattern catalogue",
    )?;
    let dir = worker.fresh_dir("refine");
    std::fs::create_dir_all(&dir).map_err(|error| runtime("creating", &dir, &error))?;
    let scratch = Scratch(dir.clone());
    worker.budget.note_spill();
    let base = worker.base;
    worker.budget.hold(base);
    let mut cost = Cost::new(started);

    // The members, as the project's archive search finds them.
    worker.check_cancel()?;
    let mut unpacked = count_scan(worker, &snapshot);
    let (selection, summary) = SelectionSet::search_archive(
        "triager",
        query,
        SearchSettings {
            max_members: MAX_PATTERN_MEMBERS,
        },
        &snapshot.path,
        &snapshot.zip,
        worker.budget,
    )?;
    let records = summary.input.records;
    let mut held = selection
        .held_bytes()
        .saturating_add(archive_summary_bytes(&summary));
    worker.budget.hold(base.saturating_add(held));
    drop(summary);
    held = selection.held_bytes();
    worker.budget.hold(base.saturating_add(held));
    // They must be the members of the selection, one for one: a catalogue of another
    // population would be stated under this selection's handle.
    let mut joined = 0u64;
    snapshot
        .scope
        .join_selection(&selection, worker.budget, |_, original| {
            let row = snapshot.scope.row_of(original.id.0)?;
            match row.map(|row| set.states.get(row)).transpose()? {
                Some(State::In) => {
                    joined += 1;
                    Ok(())
                }
                _ => Err(L2dError::Refused {
                    reason: format!(
                        "the project's search finds the record {}, which is no member of the \
                         selection; nothing was committed",
                        wire::record_id(original.id.0)
                    ),
                }),
            }
        })
        .map_err(L2dError::from)?;
    if joined != members {
        return Err(Fault::refused(format!(
            "the project's search finds {joined} of the {members} members of the selection; \
             nothing was committed"
        )));
    }
    worker.check_cancel()?;

    // The catalogue and its first refinement, as `l2d discover search` builds them.
    unpacked += count_scan(worker, &snapshot);
    let catalogue_dir = dir.join("catalogue");
    let catalogue = PatternCatalogue::build_archive(
        &selection,
        &snapshot.path,
        &snapshot.zip,
        &settings,
        &catalogue_dir,
        worker.budget,
    )?;
    held = held.saturating_add(catalogue.held_bytes());
    worker.budget.hold(base.saturating_add(held));
    worker.check_cancel()?;
    let loaded = PatternRefinement::from_catalogue(&catalogue, &catalogue_dir, worker.budget)?;
    let share = work_share(&loaded);
    worker
        .budget
        .require_whole(share, "working on a new pattern refinement and its output")?;
    worker
        .budget
        .hold(base.saturating_add(held).saturating_add(share));
    drop((selection, catalogue));
    let _ = std::fs::remove_dir_all(&catalogue_dir);
    worker.budget.hold(base.saturating_add(share));
    cost.set("archive_scans", 2)
        .set("bytes_unpacked", unpacked)
        .set("records_scanned", 2 * records)
        .set("distinct_originals", snapshot.scope.originals());
    std::mem::forget(scratch);
    publish(worker, set, &loaded, None, Some(dir), Value::Null, cost)
}

fn counts_of(loaded: &PatternRefinement, pattern: usize) -> Value {
    loaded
        .groups
        .iter()
        .find(|group| group.id == pattern)
        .map_or(Value::Null, |group| Value::Object(pattern_row(group)))
}

fn require_pattern(loaded: &PatternRefinement, pattern: u64) -> Result<usize, Fault> {
    loaded
        .groups
        .iter()
        .map(|group| group.id)
        .find(|id| *id as u64 == pattern)
        .ok_or_else(|| {
            Fault::new(
                "unknown_pattern",
                format!("this refinement revision has no pattern {pattern}"),
            )
        })
}

fn record_ids(members: &[Member]) -> Vec<String> {
    members
        .iter()
        .filter_map(|member| match member {
            Member::Position(position) => Some(wire::record_id(*position)),
            Member::StoreId(_) => None,
        })
        .take(PREVIEW_IDS)
        .collect()
}

/// The encoder of a neighbour search, open for as long as the search runs.
struct Encoder {
    budget_held: u64,
    encoder: Box<dyn ExplorationEncoder>,
}

fn open_encoder(worker: &mut Worker<'_>, method: Method, held: u64) -> Result<Encoder, Fault> {
    let shared = worker.shared;
    match method {
        Method::CharacterTrigram => Ok(Encoder {
            budget_held: 0,
            encoder: Box::new(CharacterTrigram),
        }),
        Method::Vector => {
            let source = shared.config.encoder.as_ref().ok_or_else(|| {
                Fault::unsupported(
                    "the neighbour method `vector` needs an encoder profile, and the host \
                     configured none; use `character_trigram`",
                )
            })?;
            let session = source.0.session_bytes();
            let budget: &mut MemoryBudget = worker.budget;
            budget.require_whole(session, "the encoder session of a neighbour search")?;
            let encoder = source.0.open(budget).map_err(|error| {
                // Why is the host's to read: the error names files of its profile
                // and runtime.
                eprintln!("triager: opening the encoder of a neighbour search failed: {error}");
                Fault::refused(format!(
                    "the encoder of the neighbour method `vector` could not be opened on this \
                     host ({}); the host's diagnostics state why. Nothing was encoded",
                    match error {
                        L2dError::Missing { .. } => "a file of its profile or runtime is missing",
                        L2dError::Refused { .. } => "it was refused",
                        L2dError::InvalidInput { .. } => "its configuration is not usable",
                        L2dError::Runtime { .. } => "it failed",
                    }
                ))
            })?;
            budget.hold(worker.base.saturating_add(held).saturating_add(session));
            Ok(Encoder {
                budget_held: session,
                encoder,
            })
        }
    }
}

struct NeighbourSearch<'a> {
    refinement: &'a Arc<Refinement>,
    exemplar: u64,
    method: Method,
    within: Within,
    top_k: u64,
    radius: f64,
    target: Option<&'a Arc<Refinement>>,
}

fn neighbours(
    worker: &mut Worker<'_>,
    search: &NeighbourSearch<'_>,
    started: Instant,
) -> Result<Done, Fault> {
    let refinement = search.refinement;
    let snapshot = refinement.set.snapshot.clone();
    // The search is named before anything is loaded or encoded: asked again, it is
    // the selection it made, and no encoder is opened for it.
    let (method, profile) = match search.method {
        Method::CharacterTrigram => (
            CharacterTrigram.method().to_string(),
            CharacterTrigram.profile().to_string(),
        ),
        Method::Vector => {
            let source = worker.shared.config.encoder.as_ref().ok_or_else(|| {
                Fault::unsupported(
                    "the neighbour method `vector` needs an encoder profile, and the host \
                     configured none; use `character_trigram`",
                )
            })?;
            (
                source.0.method().to_string(),
                source.0.profile().to_string(),
            )
        }
    };
    let mut described = json!({
        "kind": "neighbours",
        "version": 1,
        "refinement": refinement.handle,
        "refinement_revision": refinement.revision,
        "exemplar": wire::record_id(search.exemplar),
        "method": search.method.name(),
        "native_method": method,
        "profile": profile,
        "within": search.within.name(),
        "top_k": search.top_k,
        "radius": search.radius,
    });
    if let Some(target) = search.target {
        described["target"] = json!(target.handle);
        described["target_revision"] = json!(target.revision);
    }
    let producer = Producer::Neighbours { described };
    let population = Population::All(snapshot);
    if let (_, Some(known)) = known_selection(worker, &population, &producer) {
        return Ok(Done {
            result: known.result.clone(),
            commit: None,
        });
    }
    let (seed, mut held) = load(worker, refinement, 0)?;
    let target = match search.target {
        Some(target) => {
            let (loaded, more) = load(worker, target, held)?;
            held = held.saturating_add(more);
            Some(loaded)
        }
        None => None,
    };
    let exemplar_row = seed
        .rows
        .iter()
        .position(|row| row.member == Member::Position(search.exemplar))
        .ok_or_else(|| {
            Fault::new(
                "record_not_in_scope",
                format!(
                    "`{}` is no member of the selection this refinement groups",
                    wire::record_id(search.exemplar)
                ),
            )
        })?;
    let settings = ExploreSettings {
        exemplar_row,
        scope: match search.within {
            Within::Pattern => exploration::Scope::WithinPattern,
            Within::OtherPatterns => exploration::Scope::AcrossPatterns,
            Within::All => exploration::Scope::AllPatterns,
        },
        top_k: search.top_k as usize,
        radius: search.radius,
        expected_profile: None,
    };
    let mut encoder = open_encoder(worker, search.method, held)?;
    if encoder.encoder.profile() != profile || encoder.encoder.method() != method {
        return Err(Fault::runtime(
            "the encoder that was opened states another profile than its source",
        ));
    }
    let report = exploration::explore(
        &seed,
        target.as_ref(),
        &settings,
        encoder.encoder.as_mut(),
        worker.budget,
    );
    let session = encoder.budget_held;
    drop(encoder);
    // What the search admitted for its report stays held; the session is let go.
    let report = report?;
    let held_now = worker.budget.held_bytes().saturating_sub(session);
    worker.budget.hold(held_now);
    let held = held_now.saturating_sub(worker.base);
    let encoded = match search.method {
        Method::Vector => report.eligible + 1,
        Method::CharacterTrigram => 0,
    };
    worker
        .shared
        .activity
        .rows_encoded
        .fetch_add(encoded, Ordering::Relaxed);

    // Every member the search could have returned is evaluated: a neighbour is in,
    // another member out, and one the encoder could not represent not evaluated.
    let mut listing: BTreeMap<u64, (u8, f32)> = BTreeMap::new();
    for row in seed
        .rows
        .iter()
        .chain(target.iter().flat_map(|target| &target.rows))
    {
        if let Member::Position(position) = row.member {
            listing.insert(position, (State::Out as u8, f32::NAN));
        }
    }
    listing.insert(search.exemplar, (State::In as u8, 0.0));
    let position_of = |member: &Member| match member {
        Member::Position(position) => Ok(*position),
        Member::StoreId(_) => Err(Fault::runtime("a neighbour of an archive names a store")),
    };
    for unsupported in &report.unsupported {
        listing.insert(
            position_of(&unsupported.member.row.member)?,
            (ABSTAINED, f32::NAN),
        );
    }
    let mut shown = Vec::new();
    for neighbour in &report.neighbours {
        let position = position_of(&neighbour.row.member)?;
        let distance = neighbour.distance.unwrap_or(f64::NAN);
        listing.insert(position, (State::In as u8, distance as f32));
        if shown.len() < DEFAULT_ROWS {
            shown.push(json!({
                "id": wire::record_id(position),
                "distance": distance,
                "outside_seed": neighbour.outside_saved_search,
            }));
        }
    }
    let mut extra = Map::new();
    extra.insert(
        "neighbours".into(),
        json!({
            "returned": report.neighbours.len(),
            "includes_exemplar": true,
            "listed": shown,
            "distance_range": report.distance_range,
            "within_radius_before_top_k": {
                "members": report.support.members,
                "occurrences": report.support.occurrences,
                "distinct_texts": report.support.distinct_texts,
            },
            "scanned": report.scanned,
            "eligible": report.eligible,
            "not_represented": report.unsupported.len(),
            "connectivity": "direct reach to the exemplar only",
        }),
    );
    let mut cost = Cost::new(started);
    cost.set("rows_encoded", encoded)
        .set("candidates", report.eligible);
    drop(report);
    let listed = Listed {
        scope: &population,
        producer,
        with_values: true,
        reason: "outside_producer_coverage",
        extra,
        held,
        cost,
    };
    let (result, set) = select_listed(worker, listed, &mut |put| {
        listing
            .iter()
            .try_for_each(|(position, (code, value))| put(*position, *code, *value))
    })?;
    Ok(Done {
        result,
        commit: set.map(Commit::Set),
    })
}

pub(crate) fn run(worker: &mut Worker<'_>, task: &RefineTask) -> Result<Done, Fault> {
    let started = Instant::now();
    match task {
        RefineTask::Patterns { set } => patterns(worker, set, started),
        RefineTask::List { refinement, page } => {
            let (loaded, _) = load(worker, refinement, 0)?;
            let (rows, shown_bytes) = patterns_page(&loaded, page);
            let mut result = merged(
                head(
                    &refinement.handle,
                    &refinement.set,
                    &loaded,
                    refinement.parent.as_deref(),
                ),
                rows,
            );
            let snapshot = &refinement.set.snapshot;
            let mut disclosure = worker.shared.disclosure_after(snapshot, shown_bytes);
            disclosure["template_bytes"] = json!(shown_bytes);
            result["disclosure"] = disclosure;
            result["cost"] = Cost::new(started).finish(worker);
            let result = wire::with_response_bytes(result);
            require_fit(&result)?;
            // The result is whole and fits: only now is the allowance spent on it.
            if shown_bytes > 0 {
                worker.shared.disclose(snapshot, shown_bytes)?;
            }
            Ok(Done {
                result,
                commit: None,
            })
        }
        RefineTask::Split {
            refinement,
            pattern,
            regex,
        } => {
            let (loaded, _) = load(worker, refinement, 0)?;
            let id = require_pattern(&loaded, *pattern)?;
            let next = loaded.split(id, regex, worker.budget)?;
            let made = next.groups.iter().map(|group| group.id).max().unwrap_or(id);
            let changed = json!({
                "operation": "split",
                "pattern": id,
                "new_pattern": made,
                "before": counts_of(&loaded, id),
                "kept": counts_of(&next, id),
                "split_off": counts_of(&next, made),
            });
            let set = refinement.set.clone();
            publish(
                worker,
                &set,
                &next,
                Some(refinement),
                None,
                changed,
                Cost::new(started),
            )
        }
        RefineTask::Merge {
            refinement,
            patterns,
        } => {
            let (loaded, _) = load(worker, refinement, 0)?;
            for pattern in patterns {
                require_pattern(&loaded, *pattern as u64)?;
            }
            let next = loaded.merge(patterns, worker.budget)?;
            let kept = patterns.iter().copied().min().unwrap_or(0);
            let before: Vec<Value> = patterns
                .iter()
                .map(|pattern| counts_of(&loaded, *pattern))
                .collect();
            let changed = json!({
                "operation": "merge",
                "patterns": patterns,
                "pattern": kept,
                "before": before,
                "merged": counts_of(&next, kept),
            });
            let set = refinement.set.clone();
            publish(
                worker,
                &set,
                &next,
                Some(refinement),
                None,
                changed,
                Cost::new(started),
            )
        }
        RefineTask::Preview {
            refinement,
            pattern,
            regex,
        } => {
            let (loaded, _) = load(worker, refinement, 0)?;
            let id = require_pattern(&loaded, *pattern)?;
            let (query, proposed) = match regex {
                Some(regex) => (regex.clone(), false),
                None => (loaded.proposed_regex(id, worker.budget)?, true),
            };
            let preview = loaded.preview_query(id, &query, worker.budget)?;
            let (shown, clipped) = wire::display(&query, 4 * MAX_TEMPLATE_BYTES);
            let snapshot = &refinement.set.snapshot;
            let disclosed = if proposed { shown.len() as u64 } else { 0 };

            let group =
                |members: &[Member]| json!({"count": members.len(), "first": record_ids(members)});
            let mut disclosure = worker.shared.disclosure_after(snapshot, disclosed);
            disclosure["query_bytes"] = json!(disclosed);
            let result = wire::with_response_bytes(json!({
                "v": VERSION,
                "status": "ready",
                "preview": {
                    "refinement": refinement.handle,
                    "refinement_revision": preview.refinement_revision,
                    "pattern": id,
                    "regex": shown,
                    "regex_proposed": proposed,
                    "regex_sha256": l2d_core::input::content_sha256(query.as_bytes()),
                    "member_matches": group(&preview.member_matches),
                    "member_misses": group(&preview.member_misses),
                    "background_matches": group(&preview.background_matches),
                    "background_misses": group(&preview.background_misses),
                    "background_scope": preview.background_scope,
                },
                "set": refinement.set.handle,
                "snapshot": snapshot.handle,
                "clipping": {"clipped": clipped,
                             "fields": if clipped { json!(["preview.regex"]) } else { json!([]) }},
                "disclosure": disclosure,
                "cost": Cost::new(started).finish(worker),
            }));
            require_fit(&result)?;
            if proposed {
                // A proposed query is made of the literal tokens of the members: it
                // is counted as evidence, once the result is whole and fits.
                worker.shared.disclose(snapshot, disclosed)?;
            }
            Ok(Done {
                result,
                commit: None,
            })
        }
        RefineTask::Neighbours {
            refinement,
            exemplar,
            method,
            within,
            top_k,
            radius,
            target,
        } => neighbours(
            worker,
            &NeighbourSearch {
                refinement,
                exemplar: *exemplar,
                method: *method,
                within: *within,
                top_k: *top_k,
                radius: *radius,
                target: target.as_ref(),
            },
            started,
        ),
        RefineTask::Select {
            scope,
            refinement,
            pattern,
        } => {
            let (loaded, held) = load(worker, refinement, 0)?;
            let id = require_pattern(&loaded, *pattern)?;
            let listed = Listed {
                scope,
                producer: Producer::Pattern {
                    refinement: refinement.handle.clone(),
                    revision: refinement.revision.clone(),
                    pattern: *pattern,
                },
                with_values: false,
                reason: "outside_producer_coverage",
                extra: Map::new(),
                held,
                cost: Cost::new(started),
            };
            let (result, set) = select_listed(worker, listed, &mut |put| {
                loaded.rows.iter().try_for_each(|row| match row.member {
                    Member::Position(position) => {
                        put(position, u8::from(row.pattern == Some(id)), 0.0)
                    }
                    Member::StoreId(_) => Ok(()),
                })
            })?;
            Ok(Done {
                result,
                commit: set.map(Commit::Set),
            })
        }
    }
}
