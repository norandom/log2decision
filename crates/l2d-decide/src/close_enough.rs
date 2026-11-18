//! Bounded MarkSet grouping and support, reusing exploration's direct-exemplar
//! engine. Proximity is not deployment qualification. No neighbour vote changes a
//! score. Returned objects remain reserved until the caller drops them and releases
//! the reservation (restore the incoming `MemoryBudget::held_bytes`).
use crate::exploration::{DirectExemplar, DirectResult, ExplorationEncoder};
use crate::{Calibrated, ConformalCalibrator, ScorerKind};
use l2d_baseline::BaselineScorer;
use l2d_core::patterns::PatternStore;
use l2d_core::selections::Member;
use l2d_core::{InputSummary, L2dError, Label, LogsFrame, MarkSet, MemoryBudget};
use l2d_store::{Filter, Store};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
const FIXED: u64 = 4 * 1024 * 1024;
fn refused(reason: impl Into<String>) -> L2dError {
    L2dError::Refused {
        reason: reason.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provenance {
    Marked,
    ReviewedOpposite,
    AssumedBackground,
    Excluded,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Evidence {
    pub member: Member,
    pub text: String,
    pub label: Option<Label>,
    pub provenance: Provenance,
    pub group: Option<String>,
    pub distance: f64,
    pub occurrences: u64,
    /// Exact whole-original equality is the only positive payload evidence here.
    /// Nonidentical texts have unknown payload support, not a negative anomaly label.
    pub exact_original_match: bool,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct UnsupportedRepresentation {
    pub member: Member,
    pub reason: String,
}
#[derive(Clone, Debug)]
struct Row {
    member: Member,
    text: String,
    group: Option<String>,
    occurrences: u64,
    label: Option<Label>,
    provenance: Provenance,
}
/// A source-validated bounded history. Dataset labels are not promoted into review:
/// only explicit marks and opposite marks supply labels. This whole-history operation
/// has no disk form and refuses before loading when its conservative bound cannot fit.
pub struct MarkHistory {
    rows: Vec<Row>,
    marks: MarkSet,
    other: Option<MarkSet>,
    snapshot: Option<String>,
    retained: u64,
    metadata_revision: String,
}
fn mark_bytes(m: &MarkSet) -> u64 {
    FIXED
        .saturating_add(m.count().saturating_mul(1024))
        .saturating_add(
            m.redactions()
                .iter()
                .map(|r| (r.fragment.len() + r.placeholder.len() + 1024) as u64 * 32)
                .sum::<u64>(),
        )
}
fn selected(m: &MarkSet, member: &Member) -> Result<bool, L2dError> {
    match member {
        Member::Position(p) => Ok(m.is_selected(*p)),
        Member::StoreId(id) => m.is_selected_id(id),
    }
}
fn deselected(m: &MarkSet, member: &Member) -> Result<bool, L2dError> {
    match member {
        Member::Position(p) => Ok(m.deselected().contains(p)),
        Member::StoreId(id) => m.is_deselected_id(id),
    }
}
fn classify(m: &MarkSet, other: Option<&MarkSet>, row: &mut Row) -> Result<(), L2dError> {
    let a = selected(m, &row.member)?;
    let b = other
        .map(|o| selected(o, &row.member))
        .transpose()?
        .unwrap_or(false);
    if deselected(m, &row.member)?
        || other
            .map(|o| deselected(o, &row.member))
            .transpose()?
            .unwrap_or(false)
        || (a && b)
    {
        row.provenance = Provenance::Excluded;
    } else if a {
        row.label = Some(m.label());
        row.provenance = Provenance::Marked;
    } else if b {
        row.label = other.map(MarkSet::label);
        row.provenance = Provenance::ReviewedOpposite;
    }
    Ok(())
}
impl MarkHistory {
    fn finish(
        mut rows: Vec<Row>,
        marks: &MarkSet,
        other: Option<&MarkSet>,
        snapshot: Option<String>,
        retained: u64,
    ) -> Result<Self, L2dError> {
        if other
            .is_some_and(|o| o.label() == marks.label() || !o.source().same_history(marks.source()))
        {
            return Err(refused(
                "opposite marks must have the other label and the same source",
            ));
        }
        for row in &mut rows {
            classify(marks, other, row)?;
        }
        // Resolve ambiguity globally before radius/top-k: a conflicting/de-selected
        // original cannot re-enter through a different source-qualified ID.
        let mut states: BTreeMap<&str, (Option<Label>, bool)> = BTreeMap::new();
        for row in &rows {
            let state = states.entry(&row.text).or_default();
            state.1 |= row.provenance == Provenance::Excluded
                || (state.0.is_some() && row.label.is_some() && state.0 != row.label);
            if row.label.is_some() {
                state.0 = row.label;
            }
        }
        let excluded: BTreeSet<String> = states
            .into_iter()
            .filter(|(_, (_, conflict))| *conflict)
            .map(|(text, _)| text.to_owned())
            .collect();
        for row in &mut rows {
            if excluded.contains(&row.text) {
                row.provenance = Provenance::Excluded;
                row.label = None;
            }
        }
        rows.sort_by(|a, b| a.member.cmp(&b.member));
        let mut hash = Sha256::new();
        for row in &rows {
            // Length framing and explicit None prevent ambiguous concatenation.
            let bytes = serde_json::to_vec(&(&row.member, &row.text, &row.group, row.occurrences))
                .map_err(|e| refused(e.to_string()))?;
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
        let metadata_revision = format!("{:x}", hash.finalize());
        Ok(Self {
            rows,
            marks: marks.clone(),
            other: other.cloned(),
            snapshot,
            retained,
            metadata_revision,
        })
    }
    /// Read real stored groups. Re-encode originals through the chosen exploration
    /// route, so legacy stored vectors cannot silently supply a different profile.
    pub fn from_store(
        store: &Store,
        marks: &MarkSet,
        other: Option<&MarkSet>,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        let prior = budget.held_bytes();
        let result = (|| {
            let generation = store.read_generation()?;
            marks.require_store(store, budget)?;
            if let Some(o) = other {
                o.require_store(store, budget)?;
            }
            let before = store.pattern_snapshot(budget)?;
            let measured = store.matching_with(&Filter::default(), usize::MAX, budget)?;
            let retained = measured
                .read_bytes
                .saturating_mul(32)
                .saturating_add(mark_bytes(marks))
                .saturating_add(other.map_or(0, mark_bytes))
                .saturating_add(FIXED);
            budget.require_whole(
                retained,
                "bounded MarkSet history, metadata, duplicate index and output copies",
            )?;
            budget.hold(prior.saturating_add(retained));
            let frame = store.text_search_with(&Filter::default(), usize::MAX, budget)?;
            if (frame.estimated_size() as u64).saturating_mul(32) > retained {
                return Err(refused("store grew beyond admitted history bound; retry"));
            }
            let columns = || -> Result<_, l2d_core::polars::prelude::PolarsError> {
                Ok((
                    frame.column("id")?.str()?,
                    frame.column("text")?.str()?,
                    frame.column("group")?.str()?,
                    frame.column("occurrences")?.u32()?,
                ))
            };
            let (ids, texts, groups, counts) = columns().map_err(|e| refused(e.to_string()))?;
            let mut rows = Vec::with_capacity(frame.height());
            for i in 0..frame.height() {
                rows.push(Row {
                    member: Member::StoreId(ids.get(i).unwrap().into()),
                    text: texts.get(i).unwrap().into(),
                    group: groups.get(i).map(str::to_owned),
                    occurrences: u64::from(counts.get(i).unwrap()),
                    label: None,
                    provenance: Provenance::AssumedBackground,
                });
            }
            drop(frame);
            if before != store.pattern_snapshot(budget)? || generation != store.read_generation()? {
                return Err(refused("store changed while loading MarkSet history"));
            }
            marks.require_store(store, budget)?;
            Self::finish(rows, marks, other, Some(before), retained)
        })();
        if result.is_err() {
            budget.hold(prior);
        }
        result
    }
    /// Input adapters supply original text and genuine group keys; unknown stays null.
    pub fn from_input(
        logs: &LogsFrame,
        summary: &InputSummary,
        marks: &MarkSet,
        other: Option<&MarkSet>,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        Self::from_input_selection(logs, summary, None, marks, other, budget)
    }

    /// Capture original rows from a freshly read file or an exact archive entry selection.
    pub fn from_input_selection(
        logs: &LogsFrame,
        summary: &InputSummary,
        archive: Option<&l2d_core::ZipInputSettings>,
        marks: &MarkSet,
        other: Option<&MarkSet>,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        match archive {
            Some(settings) => marks.require_archive(summary, settings)?,
            None => marks.require_file(summary, logs, budget)?,
        }
        if other.is_some_and(|o| !o.source().same_history(marks.source())) {
            return Err(refused("opposite marks must bind the same input history"));
        }
        let retained = (logs.frame().estimated_size() as u64)
            .saturating_mul(32)
            .saturating_add(mark_bytes(marks))
            .saturating_add(other.map_or(0, mark_bytes))
            .saturating_add(FIXED);
        budget.require_whole(retained, "bounded input MarkSet history and output copies")?;
        let prior = budget.held_bytes();
        budget.hold(prior.saturating_add(retained));
        let rows = logs
            .records()
            .map(|r| Row {
                member: Member::Position(r.first_position),
                text: r.text,
                group: r.group,
                occurrences: u64::from(r.occurrences),
                label: None,
                provenance: Provenance::AssumedBackground,
            })
            .collect();
        let result = Self::finish(rows, marks, other, None, retained);
        if result.is_err() {
            budget.hold(prior);
        }
        result
    }
    /// Supply real stored endpoint keys to a grouped score report. Duplicate originals
    /// with conflicting or unknown keys have no unambiguous holdout group.
    pub fn stored_groups(
        &self,
        logs: &LogsFrame,
        budget: &mut MemoryBudget,
    ) -> Result<LogsFrame, L2dError> {
        use l2d_core::polars::prelude::{Column, IntoColumn, StringChunkedBuilder};
        if self.snapshot.is_none() {
            return Err(refused("stored group lookup requires a store history"));
        }
        let group_bytes = self.rows.iter().fold(0u64, |sum, r| {
            sum.saturating_add(r.group.as_ref().map_or(0, |g| g.len() as u64))
        });
        let needed = (self.rows.len() as u64)
            .saturating_mul(192)
            .saturating_add(group_bytes.saturating_mul(4))
            .saturating_add((logs.len() as u64).saturating_mul(128))
            .saturating_add(65536);
        budget.require_whole(needed, "stored group index and fitted group column")?;
        let mut groups = BTreeMap::new();
        for row in &self.rows {
            let group = row.group.as_deref();
            groups
                .entry(row.text.as_str())
                .and_modify(|known| {
                    if *known != group {
                        *known = None;
                    }
                })
                .or_insert(group);
        }
        let mut frame = logs.frame().clone();
        let texts = frame
            .column(LogsFrame::TEXT)
            .and_then(Column::str)
            .map_err(|e| refused(e.to_string()))?;
        let mut keys = StringChunkedBuilder::new(LogsFrame::GROUP.into(), logs.len());
        for text in texts.into_no_null_iter() {
            let key = groups
                .get(text)
                .ok_or_else(|| refused("fitted original missing from store history"))?;
            keys.append_option(*key);
        }
        frame
            .with_column(keys.finish().into_column())
            .map_err(|e| refused(e.to_string()))?;
        let enriched = LogsFrame::new(frame)?;
        budget.hold(budget.held_bytes().saturating_add(needed));
        Ok(enriched)
    }
    pub fn retained_bytes(&self) -> u64 {
        self.retained
    }
    /// Captured original identities, text, endpoint metadata and counts.
    pub fn metadata_revision(&self) -> &str {
        &self.metadata_revision
    }
    /// Input history is an immutable frame verified against InputSummary at capture.
    pub fn require_input_snapshot(&self) -> Result<(), L2dError> {
        if self.snapshot.is_some() {
            return Err(refused("store history requires a live store source"));
        }
        Ok(())
    }
    pub fn excluded_members(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| r.provenance == Provenance::Excluded)
            .count()
    }
    fn workspace(&self, text: &str, encoder: &dyn ExplorationEncoder) -> u64 {
        let longest = self
            .rows
            .iter()
            .map(|r| r.text.len())
            .max()
            .unwrap_or(0)
            .max(text.len()) as u64;
        self.retained
            .saturating_mul(2)
            .saturating_add(longest.saturating_mul(256))
            .saturating_add(encoder.scratch_bytes())
            .saturating_add(FIXED)
    }
    pub(crate) fn require_scorer(&self, scorer: &BaselineScorer) -> Result<(), L2dError> {
        let origin = scorer
            .mark_set_origin()
            .ok_or_else(|| refused("close-enough requires an actual MarkSet scorer"))?;
        if !origin.marks().try_eq(&self.marks)? {
            return Err(refused(
                "scorer marks differ from support history; rebuild history/refit after deselection",
            ));
        }
        match (origin.other(), self.other.as_ref()) {
            (None, None) => Ok(()),
            (Some(a), Some(b)) if a.try_eq(b)? => Ok(()),
            _ => Err(refused("scorer opposite marks differ from support history")),
        }
    }
    /// Check the source immediately before consuming a batch for a mutable store.
    pub fn require_store(&self, store: &Store, budget: &mut MemoryBudget) -> Result<(), L2dError> {
        let generation = store.read_generation()?;
        self.marks.require_store(store, budget)?;
        if self.snapshot.as_deref() != Some(store.pattern_snapshot(budget)?.as_str()) {
            return Err(refused(
                "store occurrence snapshot changed; rebuild support history",
            ));
        }
        let prior = budget.held_bytes();
        let result = (|| {
            budget.require_whole(self.retained, "verify immutable history endpoint metadata")?;
            let frame = store.text_search_with(&Filter::default(), usize::MAX, budget)?;
            let check = || -> Result<bool, l2d_core::polars::prelude::PolarsError> {
                let ids = frame.column("id")?.str()?;
                let texts = frame.column("text")?.str()?;
                let groups = frame.column("group")?.str()?;
                let counts = frame.column("occurrences")?.u32()?;
                Ok(frame.height() == self.rows.len()
                    && (0..frame.height()).all(|i| {
                        let Some(id) = ids.get(i) else {
                            return false;
                        };
                        self.rows
                            .binary_search_by(|r| match &r.member {
                                Member::StoreId(saved) => saved.as_str().cmp(id),
                                _ => std::cmp::Ordering::Less,
                            })
                            .ok()
                            .is_some_and(|j| {
                                let row = &self.rows[j];
                                texts.get(i) == Some(row.text.as_str())
                                    && groups.get(i) == row.group.as_deref()
                                    && counts.get(i).map(u64::from) == Some(row.occurrences)
                            })
                    }))
            };
            if !check().map_err(|e| refused(e.to_string()))?
                || generation != store.read_generation()?
            {
                return Err(refused("store metadata changed; rebuild support history"));
            }
            Ok(())
        })();
        budget.hold(prior);
        result
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReachSettings {
    pub radius: f64,
    /// Retain at most this many globally distinct labelled originals, and separately
    /// this many assumed-background originals. Background cannot evict review evidence.
    pub top_k: usize,
    pub minimum_labelled_support: usize,
    pub expected_profile: String,
}
impl ReachSettings {
    fn check(&self, encoder: &dyn ExplorationEncoder) -> Result<(), L2dError> {
        if !self.radius.is_finite()
            || !(0. ..=2.).contains(&self.radius)
            || self.top_k == 0
            || self.top_k > 10000
            || self.minimum_labelled_support == 0
            || self.minimum_labelled_support > self.top_k
        {
            return Err(refused(
                "radius must be 0..2; top-k 1..10000; labelled minimum 1..top-k",
            ));
        }
        if self.expected_profile != encoder.profile() {
            return Err(refused("incompatible exploration profile"));
        }
        Ok(())
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct MarkGroup {
    pub representative: Evidence,
    #[serde(default)]
    pub boundary: Option<Evidence>,
    pub size: usize,
    pub distinct_texts: usize,
    /// Distinct originals by actual endpoint/host relation to the representative.
    /// A text present in multiple groups may contribute in each applicable category.
    #[serde(default)]
    pub same_group_distinct_texts: usize,
    #[serde(default)]
    pub cross_group_distinct_texts: usize,
    #[serde(default)]
    pub unknown_group_distinct_texts: usize,
    pub occurrences: u64,
    pub distance_range: [f64; 2],
    members: Vec<Member>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Grouping {
    pub plan: l2d_core::BatchPlan,
    pub method: String,
    pub profile: String,
    pub groups: Vec<MarkGroup>,
    pub unsupported: Vec<Member>,
    pub unsupported_details: Vec<UnsupportedRepresentation>,
    pub excluded: usize,
    source: l2d_core::MarkedSource,
    marks_file: Vec<u8>,
    snapshot: Option<String>,
}
fn evidence(row: &Row, distance: f64, text: &str) -> Evidence {
    Evidence {
        member: row.member.clone(),
        text: row.text.clone(),
        label: row.label,
        provenance: row.provenance,
        group: row.group.clone(),
        distance,
        occurrences: row.occurrences,
        exact_original_match: row.text == text,
    }
}
/// Stable partition by direct reach to the first unassigned marked original. No
/// transitive closure, centroid, all-pairs matrix or second clustering backend.
pub fn group_marks(
    history: &MarkHistory,
    settings: &ReachSettings,
    encoder: &mut dyn ExplorationEncoder,
    budget: &mut MemoryBudget,
) -> Result<Grouping, L2dError> {
    settings.check(encoder)?;
    let guard = l2d_core::resource::ResourceGuard::tracked(
        "MarkSet grouping",
        l2d_core::resource::busy_choice(),
        budget.usage(),
    )?;
    let prior = budget.held_bytes();
    let needed = history.workspace("", encoder);
    budget.require_whole(needed, "MarkSet direct-exemplar grouping and result copies")?;
    budget.hold(prior.saturating_add(needed));
    let result = (|| {
        let mut plan = budget.plan(1, encoder.scratch_bytes().max(1))?;
        plan.batch_rows = 1;
        let mut assigned = BTreeSet::new();
        let mut result = Grouping {
            plan,
            method: encoder.method().into(),
            profile: encoder.profile().into(),
            groups: vec![],
            unsupported: vec![],
            unsupported_details: vec![],
            excluded: history.excluded_members(),
            source: history.marks.source().clone(),
            marks_file: history.marks.to_file_bytes()?,
            snapshot: history.snapshot.clone(),
        };
        for (i, row) in history.rows.iter().enumerate() {
            if row.provenance != Provenance::Marked || assigned.contains(&i) {
                continue;
            }
            budget.next_batch_rows(&mut result.plan, i as u64)?;
            let query = match DirectExemplar::new(&row.text, encoder)? {
                DirectResult::Supported(query) => query,
                DirectResult::Unsupported(reason) => {
                    result.unsupported.push(row.member.clone());
                    result.unsupported_details.push(UnsupportedRepresentation {
                        member: row.member.clone(),
                        reason,
                    });
                    assigned.insert(i);
                    continue;
                }
            };
            let mut group = MarkGroup {
                representative: evidence(row, 0., &row.text),
                boundary: Some(evidence(row, 0., &row.text)),
                size: 0,
                distinct_texts: 0,
                same_group_distinct_texts: 0,
                cross_group_distinct_texts: 0,
                unknown_group_distinct_texts: 0,
                occurrences: 0,
                distance_range: [0., 0.],
                members: vec![],
            };
            let mut texts = BTreeSet::new();
            let (mut same, mut cross, mut unknown) =
                (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
            for (j, candidate) in history.rows.iter().enumerate() {
                if candidate.provenance != Provenance::Marked || assigned.contains(&j) {
                    continue;
                }
                budget.next_batch_rows(&mut result.plan, j as u64)?;
                let outcome = if j == i {
                    DirectResult::Supported(0.)
                } else {
                    query.distance(&candidate.text, encoder)?
                };
                let d = match outcome {
                    DirectResult::Supported(d) => d,
                    DirectResult::Unsupported(reason) => {
                        result.unsupported.push(candidate.member.clone());
                        result.unsupported_details.push(UnsupportedRepresentation {
                            member: candidate.member.clone(),
                            reason,
                        });
                        assigned.insert(j);
                        continue;
                    }
                };
                if d <= settings.radius {
                    assigned.insert(j);
                    group.members.push(candidate.member.clone());
                    group.size += 1;
                    group.occurrences = group.occurrences.saturating_add(candidate.occurrences);
                    if d > group.distance_range[1] {
                        group.distance_range[1] = d;
                        group.boundary = Some(evidence(candidate, d, &row.text));
                    }
                    texts.insert(&candidate.text);
                    match (&row.group, &candidate.group) {
                        (Some(a), Some(b)) if a == b => same.insert(&candidate.text),
                        (Some(_), Some(_)) => cross.insert(&candidate.text),
                        _ => unknown.insert(&candidate.text),
                    };
                }
            }
            group.distinct_texts = texts.len();
            group.same_group_distinct_texts = same.len();
            group.cross_group_distinct_texts = cross.len();
            group.unknown_group_distinct_texts = unknown.len();
            result.groups.push(group);
        }
        Ok(result)
    })();
    let result = result.and_then(|value| {
        guard.finish()?;
        Ok(value)
    });
    if result.is_err() {
        budget.hold(prior);
    }
    result
}
impl Grouping {
    /// Stable preview identity excluding the host's adaptive memory telemetry.
    pub fn revision(
        &self,
        history: &MarkHistory,
        settings: &ReachSettings,
    ) -> Result<String, L2dError> {
        l2d_core::patterns::json_hash(&(
            "l2d-mark-group-preview-v1",
            &self.source,
            history.metadata_revision(),
            &self.marks_file,
            &self.snapshot,
            &self.method,
            &self.profile,
            settings,
            &self.groups,
            &self.unsupported_details,
        ))
    }
    /// Apply an untampered in-memory grouping against unchanged source/marks.
    pub fn deselect(
        &self,
        index: usize,
        marks: &mut MarkSet,
        store: Option<&Store>,
        budget: &mut MemoryBudget,
    ) -> Result<(), L2dError> {
        budget.require_whole(mark_bytes(marks), "group deselection mark copies")?;
        if marks.source() != &self.source || marks.to_file_bytes()? != self.marks_file {
            return Err(refused("grouping is stale for this MarkSet"));
        }
        if let Some(snapshot) = &self.snapshot {
            let store = store.ok_or_else(|| refused("store required for group deselection"))?;
            marks.require_store(store, budget)?;
            if &store.pattern_snapshot(budget)? != snapshot {
                return Err(refused("grouping store snapshot changed"));
            }
        }
        let group = self
            .groups
            .get(index)
            .ok_or_else(|| refused("group index out of range"))?;
        let positions: Vec<_> = group
            .members
            .iter()
            .filter_map(|m| {
                if let Member::Position(p) = m {
                    Some(*p)
                } else {
                    None
                }
            })
            .collect();
        let ids: Vec<_> = group
            .members
            .iter()
            .filter_map(|m| {
                if let Member::StoreId(id) = m {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        if !positions.is_empty() {
            marks.deselect(&positions)?;
        }
        if !ids.is_empty() {
            marks.deselect_ids(&ids, budget)?;
        }
        Ok(())
    }
}
pub use crate::neighbour_evidence::LabelEvidence;
#[derive(Debug, Serialize, Deserialize)]
pub struct CloseEnough {
    pub plan: l2d_core::BatchPlan,
    pub support_basis: String,
    pub source: l2d_core::MarkedSource,
    pub source_snapshot: Option<String>,
    pub evidence_revision: String,
    pub settings: ReachSettings,
    pub close_enough: bool,
    pub ratio: Option<f32>,
    pub scorer: String,
    pub scorer_revision: String,
    pub method: String,
    pub profile: String,
    pub reason: Option<String>,
    pub neighbours: Vec<Evidence>,
    pub bad: LabelEvidence,
    pub good: LabelEvidence,
    pub assumed_background: usize,
    pub same_group: usize,
    pub cross_group: usize,
    pub unknown_group: usize,
    pub distinct_cross_groups: usize,
    pub unsupported: usize,
    pub unsupported_details: Vec<UnsupportedRepresentation>,
    pub excluded: usize,
    pub calibrated: Option<Calibrated>,
    pub qualification_note: String,
}
pub struct Calibration<'a> {
    pub calibrator: &'a ConformalCalibrator,
    pub error_rate: f32,
}

impl CloseEnough {
    /// Owned evidence stays live after retrieval; the history index, encoder and
    /// scorer scratch do not. Include spare vector capacity and allocator overhead.
    fn retained_bytes(&self) -> u64 {
        let member_bytes = |member: &Member| match member {
            Member::StoreId(id) => id.capacity(),
            Member::Position(_) => 0,
        };
        let source = match &self.source {
            l2d_core::MarkedSource::Store(s) => s.name.capacity() + s.content_sha256.capacity(),
            l2d_core::MarkedSource::Input(s) => {
                s.path.capacity()
                    + s.content_sha256.capacity()
                    + s.archive_entries.as_ref().map_or(0, |entries| {
                        entries.capacity() * size_of::<String>()
                            + entries.iter().map(String::capacity).sum::<usize>()
                    })
            }
        };
        let calibration = self.calibrated.as_ref().map_or(0, |c| {
            [&c.good_stratum, &c.bad_stratum]
                .into_iter()
                .map(|s| match s {
                    crate::Stratum::Pooled => 0,
                    crate::Stratum::Group(g) => g.capacity(),
                })
                .sum::<usize>()
        });
        let headers = std::mem::size_of_val(self)
            + self.plan.changes.capacity() * size_of::<l2d_core::BatchChange>()
            + self.neighbours.capacity() * size_of::<Evidence>()
            + self.unsupported_details.capacity() * size_of::<UnsupportedRepresentation>();
        let descriptions = [
            &self.support_basis,
            &self.evidence_revision,
            &self.settings.expected_profile,
            &self.scorer,
            &self.scorer_revision,
            &self.method,
            &self.profile,
            &self.qualification_note,
        ]
        .into_iter()
        .map(String::capacity)
        .sum::<usize>()
            + self.reason.as_ref().map_or(0, String::capacity)
            + self.source_snapshot.as_ref().map_or(0, String::capacity);
        let neighbours = self
            .neighbours
            .iter()
            .map(|h| {
                member_bytes(&h.member)
                    + h.text.capacity()
                    + h.group.as_ref().map_or(0, String::capacity)
            })
            .sum::<usize>();
        let unsupported = self
            .unsupported_details
            .iter()
            .map(|row| member_bytes(&row.member) + row.reason.capacity())
            .sum::<usize>();
        ((headers + descriptions + source + calibration + neighbours + unsupported) as u64)
            .saturating_mul(2)
    }
}
/// `group` is a real endpoint/host key from the history's parser, never a pattern ID.
/// Store callers call `history.require_store` before consuming a batch of decisions.
#[allow(clippy::too_many_arguments)]
pub fn close_enough(
    history: &MarkHistory,
    text: &str,
    group: Option<&str>,
    scorer: &BaselineScorer,
    calibration: Option<Calibration<'_>>,
    settings: &ReachSettings,
    encoder: &mut dyn ExplorationEncoder,
    budget: &mut MemoryBudget,
) -> Result<CloseEnough, L2dError> {
    settings.check(encoder)?;
    history.require_scorer(scorer)?;
    let guard = l2d_core::resource::ResourceGuard::tracked(
        "MarkSet close-enough",
        l2d_core::resource::busy_choice(),
        budget.usage(),
    )?;
    if let Some(c) = &calibration {
        c.calibrator
            .require_scorer(ScorerKind::MarkSet, scorer.revision())
            .map_err(|e| refused(e.to_string()))?;
        c.calibrator
            .require_error_rate(c.error_rate)
            .map_err(|e| refused(e.to_string()))?;
    }
    let needed = history
        .workspace(text, encoder)
        .saturating_add(scorer.held_bytes())
        .saturating_add(scorer.scoring_bytes(text.len() as u64))
        .saturating_add(
            calibration
                .as_ref()
                .map_or(0, |c| c.calibrator.held_bytes()),
        );
    budget.require_whole(
        needed,
        "MarkSet proximity, scorer scratch, calibration and evidence copies",
    )?;
    let prior = budget.held_bytes();
    budget.hold(prior.saturating_add(needed));
    let result = (|| {
        let mut plan = budget.plan(1, encoder.scratch_bytes().max(1))?;
        plan.batch_rows = 1;
        let mut out=CloseEnough {
            plan,
            support_basis:"Counts/means describe returned nearest distinct originals within radius: up to top_k labelled and separately top_k assumed background; only explicit labels count toward minimum support.".into(),
            source:history.marks.source().clone(),source_snapshot:history.snapshot.clone(),evidence_revision:history.metadata_revision.clone(),settings:settings.clone(),
            close_enough:false,ratio:None,scorer:"mark-set".into(),scorer_revision:scorer.revision().into(),
            method:encoder.method().into(),profile:encoder.profile().into(),reason:None,neighbours:vec![],
            bad:LabelEvidence::default(),good:LabelEvidence::default(),assumed_background:0,
            same_group:0,cross_group:0,unknown_group:0,distinct_cross_groups:0,
            unsupported:0,unsupported_details:vec![],excluded:history.excluded_members(),calibrated:None,
            qualification_note:"Local support is not deployment qualification. Unmarked background is assumed, not verified good. Calibration error rates apply only to logs resembling the calibration logs.".into(),
        };
        budget.next_batch_rows(&mut out.plan, 0)?;
        let query = match DirectExemplar::new(text, encoder)? {
            DirectResult::Supported(query) => query,
            DirectResult::Unsupported(reason) => {
                out.reason = Some(format!("unsupported query: {reason}; review required"));
                return Ok(out);
            }
        };
        // Global original-text deduplication precedes top-k. Prefer explicit labels
        // over background copies; ambiguous endpoint metadata remains unknown.
        let mut unique: BTreeMap<&str, (usize, bool)> = BTreeMap::new();
        for (i, row) in history.rows.iter().enumerate() {
            if row.provenance == Provenance::Excluded {
                continue;
            }
            unique
                .entry(&row.text)
                .and_modify(|(j, ambiguous_group)| {
                    *ambiguous_group |= history.rows[*j].group != row.group;
                    if history.rows[*j].label.is_none() && row.label.is_some() {
                        *j = i;
                    }
                })
                .or_insert((i, false));
        }
        for (n, (_, (i, ambiguous_group))) in unique.iter().enumerate() {
            let row = &history.rows[*i];
            budget.next_batch_rows(&mut out.plan, n as u64)?;
            let d = match query.distance(&row.text, encoder)? {
                DirectResult::Supported(d) => d,
                DirectResult::Unsupported(reason) => {
                    out.unsupported += 1;
                    out.unsupported_details.push(UnsupportedRepresentation {
                        member: row.member.clone(),
                        reason,
                    });
                    continue;
                }
            };
            if d > settings.radius {
                continue;
            }
            let mut hit = evidence(row, d, text);
            if *ambiguous_group {
                hit.group = None;
            }
            out.neighbours.push(hit);
            out.neighbours.sort_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then(a.member.cmp(&b.member))
            });
            let (mut labelled, mut background) = (0, 0);
            out.neighbours.retain(|hit| {
                let count = if hit.label.is_some() {
                    &mut labelled
                } else {
                    &mut background
                };
                *count += 1;
                *count <= settings.top_k
            });
        }
        let mut bad_sum = 0.;
        let mut good_sum = 0.;
        let mut cross = BTreeSet::new();
        for h in &out.neighbours {
            match h.label {
                Some(Label::Bad) => {
                    out.bad.count += 1;
                    bad_sum += h.distance;
                }
                Some(Label::Good) => {
                    out.good.count += 1;
                    good_sum += h.distance;
                }
                _ => out.assumed_background += 1,
            }
            match (group, h.group.as_deref()) {
                (Some(a), Some(b)) if a == b => out.same_group += 1,
                (Some(_), Some(b)) => {
                    out.cross_group += 1;
                    cross.insert(b);
                }
                _ => out.unknown_group += 1,
            }
        }
        out.distinct_cross_groups = cross.len();
        out.bad.mean_distance = (out.bad.count > 0).then(|| bad_sum / out.bad.count as f64);
        out.good.mean_distance = (out.good.count > 0).then(|| good_sum / out.good.count as f64);
        let labelled = out.bad.count + out.good.count;
        out.reason = if labelled < settings.minimum_labelled_support {
            Some(format!(
                "not close enough: {labelled} returned nearest distinct labelled originals within reach; need {}; review required",
                settings.minimum_labelled_support
            ))
        } else if out.bad.count == 0
            && !out.neighbours.iter().any(|h| {
                h.label == Some(Label::Good)
                    && (h.exact_original_match
                        || matches!((group,h.group.as_deref()),(Some(a),Some(b)) if a!=b))
            })
        {
            Some("unsupported benign endpoint/template-only support: no measured payload match or known cross-endpoint support; review required".into())
        } else {
            None
        };
        if out.reason.is_none() {
            out.close_enough = true;
            let ratio = scorer.ratio(text);
            out.ratio = Some(ratio);
            if let Some(c) = calibration {
                out.calibrated = Some(
                    c.calibrator
                        .predict(ratio, group, c.error_rate)
                        .map_err(|e| refused(e.to_string()))?,
                );
            }
        }
        Ok(out)
    })();
    let result = result.and_then(|value| {
        // The retrieval closure dropped its scratch. Reserve only this result
        // beside the caller's incoming hold, preserving earlier batch outputs.
        budget.hold(prior);
        let retained = value.retained_bytes();
        budget.require_whole(retained, "retained MarkSet proximity evidence")?;
        budget.hold(prior.saturating_add(retained));
        guard.finish()?;
        Ok(value)
    });
    if result.is_err() {
        budget.hold(prior);
    }
    result
}
