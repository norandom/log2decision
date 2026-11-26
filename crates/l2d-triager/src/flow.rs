//! Immutable analyst views over a bounded, retained evidence projection.

pub(crate) mod query;

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::io::{self, Write};
use std::mem::size_of;
use std::sync::Arc;

use l2d_core::selections::{SearchQuery, compile};
use l2d_core::{MemoryBudget, MemorySettings};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAX_VIEWS: usize = 128;
const MAX_HISTORY_BYTES: u64 = 64 * 1024 * 1024;
const RECEIPT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize)]
pub struct Dataset {
    pub title: String,
    pub source_revision: String,
    pub provenance: String,
    pub occurrences_known: bool,
    pub c_producer: CProducer,
    pub records: Vec<Record>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Record {
    pub id: String,
    pub text: String,
    pub score: f64,
    pub recommendation: Recommendation,
    pub reason: String,
    pub occurrences: Vec<Occurrence>,
}

/// One group of equivalent locations. A multiplicity never expands into raw rows.
#[derive(Clone, Debug, Serialize)]
pub struct Occurrence {
    pub host: Option<String>,
    pub time: Option<String>,
    pub count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recommendation {
    In,
    Out,
    Unknown,
}

/// Selects the retained evidence projection used for C during replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CProducer {
    Fixture,
    Unavailable,
    PolicyInvestigateOrReview,
}

impl Recommendation {
    pub fn label(self) -> &'static str {
        match self {
            Self::In => "recommended",
            Self::Out => "passed",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchKind {
    Literal,
    Glob,
    Regex,
    Spl,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Search {
    pub kind: SearchKind,
    pub query: String,
}

impl Search {
    fn matcher(&self, budget: &MemoryBudget) -> Result<SearchMatcher, String> {
        let query = match self.kind {
            SearchKind::Literal => SearchQuery::literal(&self.query),
            SearchKind::Glob => SearchQuery::glob(&self.query),
            SearchKind::Regex => SearchQuery::regex(&self.query),
            SearchKind::Spl => {
                return query::Query::parse(&self.query, budget).map(SearchMatcher::Spl);
            }
        };
        compile(&query, budget)
            .map(SearchMatcher::Text)
            .map_err(|error| error.to_string())
    }
}

enum SearchMatcher {
    Text(regex::Regex),
    Spl(query::Query),
}

impl SearchMatcher {
    fn uses_occurrence(&self) -> bool {
        matches!(self, Self::Spl(query) if query.uses_occurrence())
    }

    fn is_predicate(&self) -> bool {
        match self {
            Self::Text(_) => true,
            Self::Spl(query) => query.is_predicate(),
        }
    }

    /// Members that pass, in pipeline order. Occurrence stages clear the
    /// scoped groups they reject in `scope`.
    fn select(
        &self,
        records: &[Record],
        members: Vec<Member>,
        scope: Option<query::Occurrences<'_>>,
        beside: u64,
        budget: &MemoryBudget,
    ) -> Result<Vec<Member>, String> {
        match self {
            // The caller's scratch covers the input copy and the filtered result.
            Self::Text(regex) => Ok(members
                .into_iter()
                .filter(|m| regex.is_match(&records[m.record].text))
                .collect()),
            Self::Spl(query) => query.select(records, members, scope, beside, budget),
        }
    }

    fn reserved_bytes(&self) -> u64 {
        match self {
            Self::Text(_) => l2d_core::marks::REGEX_HELD_BYTES,
            Self::Spl(query) => query.reserved_bytes(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Region {
    AOnly,
    Both,
    BOnly,
    Neither,
    Recommended,
    Pass,
    Unknown,
}

impl Region {
    pub const ALL: [Self; 7] = [
        Self::AOnly,
        Self::Both,
        Self::BOnly,
        Self::Neither,
        Self::Recommended,
        Self::Pass,
        Self::Unknown,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::AOnly => "A without B",
            Self::Both => "A and B",
            Self::BOnly => "B without A",
            Self::Neither => "Neither A nor B",
            Self::Recommended => "C recommended",
            Self::Pass => "C passed",
            Self::Unknown => "C unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pool {
    General,
    Unfamiliar,
    Audit,
}

impl Pool {
    pub fn label(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Unfamiliar => "unfamiliar",
            Self::Audit => "audit",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub general: usize,
    pub unfamiliar: usize,
    pub audit: usize,
    pub seed: u64,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            general: 2,
            unfamiliar: 1,
            audit: 1,
            seed: 5,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    Region(Region),
    Host(Option<String>),
    Hour(Option<String>),
    Search(Search),
    Allocate(Budget),
}

impl Step {
    fn title(&self) -> String {
        match self {
            Self::Region(region) => region.label().into(),
            Self::Host(host) => format!("Host · {}", host.as_deref().unwrap_or("missing")),
            Self::Hour(hour) => format!("Hour · {}", hour.as_deref().unwrap_or("missing")),
            Self::Search(search) => format!("{:?} · {}", search.kind, search.query),
            Self::Allocate(_) => "Analyst work set".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub originals: usize,
    pub occurrences: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptCounts {
    pub originals: usize,
    pub occurrences: Option<u64>,
}

impl Counts {
    pub fn for_receipt(self, occurrences_known: bool) -> ReceiptCounts {
        ReceiptCounts {
            originals: self.originals,
            occurrences: occurrences_known.then_some(self.occurrences),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Member {
    pub record: usize,
    pub occurrences: u64,
    pub allocation: Option<Pool>,
}

/// Facet drills keep their host/hour predicates. A search combining original
/// and location fields retains one bit per source group, shared by later drills.
#[derive(Clone, Debug, Default)]
struct LocationScope {
    host: Option<Option<String>>,
    hour: Option<Option<String>>,
    groups: Option<Arc<Vec<u64>>>,
}

fn hour_prefix(time: Option<&str>) -> Option<&str> {
    let time = time?;
    if !time.ends_with('Z') {
        return None;
    }
    let prefix = time.get(..13)?;
    (prefix.as_bytes().get(10) == Some(&b'T')).then_some(prefix)
}

impl LocationScope {
    fn locations<'a>(
        &'a self,
        record: &'a Record,
        offset: usize,
    ) -> impl Iterator<Item = (usize, &'a Occurrence)> {
        record
            .occurrences
            .iter()
            .enumerate()
            .filter(move |(index, group)| {
                let bit = offset + index;
                self.groups
                    .as_ref()
                    .is_none_or(|bits| bits[bit / 64] & (1_u64 << (bit % 64)) != 0)
                    && self.contains(group)
            })
    }

    fn contains(&self, location: &Occurrence) -> bool {
        if let Some(host) = &self.host
            && *host != location.host
        {
            return false;
        }
        if let Some(hour) = &self.hour {
            return hour.as_deref().and_then(|value| value.get(..13))
                == hour_prefix(location.time.as_deref());
        }
        true
    }
}

#[derive(Debug)]
pub struct View {
    pub id: usize,
    pub parent: Option<usize>,
    pub step: Option<Step>,
    pub title: String,
    pub counts: Counts,
    pub members: Vec<Member>,
    pub regions: BTreeMap<Region, Counts>,
    pub hosts: BTreeMap<Option<String>, Counts>,
    pub hours: BTreeMap<Option<String>, Counts>,
    scope: LocationScope,
}

impl View {
    pub fn is_final(&self) -> bool {
        matches!(self.step, Some(Step::Allocate(_)))
    }

    fn bytes(&self) -> u64 {
        let keys = self
            .hosts
            .keys()
            .chain(self.hours.keys())
            .map(|key| 192 + key.as_ref().map_or(0, String::capacity) as u64)
            .sum::<u64>();
        let scope_bytes = [&self.scope.host, &self.scope.hour]
            .into_iter()
            .filter_map(|value| value.as_ref()?.as_ref())
            .map(String::capacity)
            .sum::<usize>();
        let step_bytes = match &self.step {
            Some(Step::Search(search)) => search.query.capacity(),
            Some(Step::Host(Some(value)) | Step::Hour(Some(value))) => value.capacity(),
            _ => 0,
        };
        2048 + keys
            + (self.title.capacity() + scope_bytes + step_bytes) as u64
            + (self.members.capacity() * size_of::<Member>()) as u64
    }
}

#[derive(Clone, Copy)]
struct Membership {
    scored: bool,
    matched: bool,
    recommendation: Recommendation,
}

impl Membership {
    fn regions(self) -> [Region; 2] {
        let overlap = match (self.scored, self.matched) {
            (true, false) => Region::AOnly,
            (true, true) => Region::Both,
            (false, true) => Region::BOnly,
            (false, false) => Region::Neither,
        };
        let recommended = match self.recommendation {
            Recommendation::In => Region::Recommended,
            Recommendation::Out => Region::Pass,
            Recommendation::Unknown => Region::Unknown,
        };
        [overlap, recommended]
    }
}

type Facets = BTreeMap<Option<String>, (usize, Counts)>;

fn tally(facets: &mut Facets, key: Option<String>, original: usize, count: u64) {
    // Occurrences for an original are contiguous. Credit the original once per bucket.
    let (last, counts) = facets.entry(key).or_insert((usize::MAX, Counts::default()));
    if *last != original {
        counts.originals += 1;
        *last = original;
    }
    counts.occurrences += count;
}

fn summarize(data: &Dataset, membership: &[Membership], group_offsets: &[usize], view: &mut View) {
    let mut hosts = Facets::new();
    let mut hours = Facets::new();
    view.regions = Region::ALL
        .into_iter()
        .map(|region| (region, Counts::default()))
        .collect();
    for member in &view.members {
        view.counts.originals += 1;
        view.counts.occurrences += member.occurrences;
        for region in membership[member.record].regions() {
            let counts = view
                .regions
                .get_mut(&region)
                .expect("all regions initialized");
            counts.originals += 1;
            counts.occurrences += member.occurrences;
        }
        for (_, group) in view
            .scope
            .locations(&data.records[member.record], group_offsets[member.record])
        {
            tally(&mut hosts, group.host.clone(), member.record, group.count);
            let hour = hour_prefix(group.time.as_deref()).map(|prefix| format!("{prefix}:00Z"));
            tally(&mut hours, hour, member.record, group.count);
        }
    }
    view.hosts = hosts
        .into_iter()
        .map(|(key, (_, counts))| (key, counts))
        .collect();
    view.hours = hours
        .into_iter()
        .map(|(key, (_, counts))| (key, counts))
        .collect();
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Branch {
    pub parent: usize,
    pub step: Step,
}

/// Event text stays local. Both evidence and ordered output are verified during replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub version: u32,
    pub dataset_sha256: String,
    pub c_producer: CProducer,
    pub top_k: usize,
    pub filter: Search,
    pub branches: Vec<Branch>,
    pub active: usize,
    pub counts: ReceiptCounts,
    pub allocated: bool,
    pub result_sha256: String,
}

pub struct Workflow {
    data: Dataset,
    dataset_sha256: String,
    top_k: usize,
    filter: Search,
    membership: Vec<Membership>,
    /// Stable bit positions for occurrence groups; the last offset is the group count.
    group_offsets: Vec<usize>,
    views: Vec<View>,
    active: usize,
    history_bytes: u64,
    memory: MemoryBudget,
}

struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn dataset_bytes(data: &Dataset) -> u64 {
    let mut bytes = data.title.capacity()
        + data.provenance.capacity()
        + data.source_revision.capacity()
        + data.records.capacity() * size_of::<Record>();
    for record in &data.records {
        bytes += record.id.capacity()
            + record.text.capacity()
            + record.reason.capacity()
            + record.occurrences.capacity() * size_of::<Occurrence>();
        for group in &record.occurrences {
            bytes += group.host.as_ref().map_or(0, String::capacity)
                + group.time.as_ref().map_or(0, String::capacity);
        }
    }
    bytes as u64
}

fn score_order(a: &Record, b: &Record) -> Ordering {
    b.score
        .partial_cmp(&a.score)
        .expect("validated finite score")
        .then_with(|| a.id.cmp(&b.id))
}

fn audit_key(seed: u64, id: &str) -> u32 {
    // FNV-1a, same seeded key as the fixture's ordered audit pool.
    format!("{seed}/{id}")
        .encode_utf16()
        .fold(2166136261, |hash, unit| {
            (hash ^ u32::from(unit)).wrapping_mul(16777619)
        })
}

fn allocate(data: &Dataset, parent: &View, budget: &Budget) -> Result<Vec<Member>, String> {
    let total = budget
        .general
        .checked_add(budget.unfamiliar)
        .and_then(|n| n.checked_add(budget.audit))
        .ok_or("allocation capacity overflows")?;
    let mut selected = Vec::with_capacity(total.min(parent.members.len()));
    for (pool, state, limit) in [
        (Pool::General, Recommendation::In, budget.general),
        (Pool::Unfamiliar, Recommendation::Unknown, budget.unfamiliar),
        (Pool::Audit, Recommendation::Out, budget.audit),
    ] {
        if limit == 0 {
            continue;
        }
        let mut eligible: Vec<_> = parent
            .members
            .iter()
            .copied()
            .filter(|member| data.records[member.record].recommendation == state)
            .collect();
        // Each state has its own pool. Empty reserves intentionally remain unfilled.
        match pool {
            Pool::General => eligible.sort_unstable_by(|a, b| {
                score_order(&data.records[a.record], &data.records[b.record])
            }),
            Pool::Unfamiliar => eligible
                .sort_unstable_by(|a, b| data.records[a.record].id.cmp(&data.records[b.record].id)),
            Pool::Audit => eligible.sort_by_cached_key(|m| {
                (
                    audit_key(budget.seed, &data.records[m.record].id),
                    data.records[m.record].id.as_str(),
                )
            }),
        }
        selected.extend(eligible.into_iter().take(limit).map(|member| Member {
            allocation: Some(pool),
            ..member
        }));
    }
    Ok(selected)
}

impl Workflow {
    pub fn new(data: Dataset, top_k: usize, filter: Search) -> Result<Self, String> {
        let memory =
            MemoryBudget::observe(&MemorySettings::default()).map_err(|e| e.to_string())?;
        Self::with_budget(data, top_k, filter, memory)
    }

    pub fn with_budget(
        data: Dataset,
        top_k: usize,
        filter: Search,
        mut memory: MemoryBudget,
    ) -> Result<Self, String> {
        let held = dataset_bytes(&data) + filter.query.capacity() as u64 + 4096;
        memory
            .require_whole(held, "retaining native triager evidence")
            .map_err(|e| e.to_string())?;
        memory.hold(memory.held_bytes() + held);
        // Allow indexing, ID validation, two facet maps and their construction scratch.
        let mut scratch = 8192 + data.records.len() as u64 * 192 + data.title.capacity() as u64;
        for record in &data.records {
            for group in &record.occurrences {
                scratch += 1024 + group.host.as_ref().map_or(0, String::len) as u64 * 3;
            }
        }
        let matcher = filter.matcher(&memory)?;
        if matcher.uses_occurrence() {
            return Err("Set B selects whole originals. Apply a host or _time query inside a view to narrow its occurrences.".into());
        }
        if !matcher.is_predicate() {
            return Err(
                "Set B is a membership filter. Apply sort, head, tail or dedup inside a view."
                    .into(),
            );
        }
        memory
            .require_whole(
                scratch + matcher.reserved_bytes(),
                "preparing retained triager views",
            )
            .map_err(|e| e.to_string())?;
        let mut ids = HashSet::with_capacity(data.records.len());
        let mut membership = Vec::with_capacity(data.records.len());
        let mut members = Vec::with_capacity(data.records.len());
        let mut group_count = 0;
        let mut group_offsets = Vec::with_capacity(data.records.len() + 1);
        let mut total_occurrences = 0u64;
        for (index, record) in data.records.iter().enumerate() {
            if record.id.is_empty() || !ids.insert(&record.id) || !record.score.is_finite() {
                return Err("retained records need distinct nonempty IDs and finite scores".into());
            }
            if data.occurrences_known == record.occurrences.is_empty() {
                return Err(
                    "record locations disagree with snapshot occurrence availability".into(),
                );
            }
            if data.c_producer == CProducer::Unavailable
                && record.recommendation != Recommendation::Unknown
            {
                return Err("C membership requires retained producer evidence".into());
            }
            let count = record.occurrences.iter().try_fold(0u64, |sum, group| {
                if group.count == 0 {
                    return Err("occurrence groups need a positive count");
                }
                sum.checked_add(group.count)
                    .ok_or("occurrence total overflows")
            })?;
            total_occurrences = total_occurrences
                .checked_add(count)
                .ok_or("snapshot occurrence total overflows")?;
            group_offsets.push(group_count);
            group_count += record.occurrences.len();
            members.push(Member {
                record: index,
                occurrences: count,
                allocation: None,
            });
            membership.push(Membership {
                scored: false,
                matched: false,
                recommendation: record.recommendation,
            });
        }
        group_offsets.push(group_count);
        drop(ids);
        // Live beside the pipeline: the indexing scratch, the compiled query and
        // the copy of the members handed to it.
        let beside =
            scratch + matcher.reserved_bytes() + (members.len() * size_of::<Member>()) as u64;
        for m in matcher.select(&data.records, members.clone(), None, beside, &memory)? {
            membership[m.record].matched = true;
        }
        drop(matcher);
        let mut ranking: Vec<_> = (0..data.records.len()).collect();
        ranking.sort_unstable_by(|&a, &b| score_order(&data.records[a], &data.records[b]));
        for index in ranking.into_iter().take(top_k) {
            membership[index].scored = true;
        }
        members.sort_unstable_by(|a, b| data.records[a.record].id.cmp(&data.records[b.record].id));
        let mut hash = HashWriter(Sha256::new());
        serde_json::to_writer(&mut hash, &data).map_err(|e| e.to_string())?;
        let mut root = View {
            id: 0,
            parent: None,
            step: None,
            title: data.title.clone(),
            counts: Counts::default(),
            members,
            scope: LocationScope::default(),
            regions: BTreeMap::new(),
            hosts: BTreeMap::new(),
            hours: BTreeMap::new(),
        };
        summarize(&data, &membership, &group_offsets, &mut root);
        let history_bytes = root.bytes();
        if history_bytes > MAX_HISTORY_BYTES {
            return Err("root projection exceeds the native view history byte limit".into());
        }
        let root_bytes = history_bytes
            + (membership.capacity() * size_of::<Membership>()) as u64
            + (group_offsets.capacity() * size_of::<usize>()) as u64;
        memory
            .require_whole(root_bytes, "retaining the root view")
            .map_err(|e| e.to_string())?;
        memory.hold(memory.held_bytes() + root_bytes);
        Ok(Self {
            data,
            dataset_sha256: format!("{:x}", hash.0.finalize()),
            top_k,
            filter,
            membership,
            group_offsets,
            views: vec![root],
            active: 0,
            history_bytes,
            memory,
        })
    }

    pub fn dataset(&self) -> &Dataset {
        &self.data
    }
    pub fn memory(&self) -> &MemoryBudget {
        &self.memory
    }
    pub fn view(&self) -> &View {
        &self.views[self.active]
    }
    pub fn views(&self) -> &[View] {
        &self.views
    }
    pub fn top_k(&self) -> usize {
        self.top_k
    }
    pub fn root_search(&self) -> &Search {
        &self.filter
    }
    pub fn membership(&self, record_index: usize) -> (bool, bool, Recommendation) {
        let m = self.membership[record_index];
        (m.scored, m.matched, m.recommendation)
    }

    pub fn locations(&self, record_index: usize) -> impl Iterator<Item = &Occurrence> {
        let view = self.view();
        view.members
            .iter()
            .find(|m| m.record == record_index)
            .into_iter()
            .flat_map(move |member| {
                view.scope.locations(
                    &self.data.records[member.record],
                    self.group_offsets[member.record],
                )
            })
            .map(|(_, group)| group)
    }

    fn search_members(
        &self,
        parent: &View,
        matcher: &SearchMatcher,
        scope: &mut LocationScope,
        scratch: u64,
    ) -> Result<(Vec<Member>, u64), String> {
        let work_bytes = scratch + matcher.reserved_bytes();
        // The copy of the parent's members handed to the pipeline.
        let copy_bytes = (parent.members.len() * size_of::<Member>()) as u64;
        if !matcher.uses_occurrence() {
            self.memory
                .require_whole(work_bytes, "searching triager originals")
                .map_err(|e| e.to_string())?;
            let members = matcher.select(
                &self.data.records,
                parent.members.clone(),
                None,
                work_bytes + copy_bytes,
                &self.memory,
            )?;
            return Ok((members, 0));
        }
        if !self.data.occurrences_known {
            return Err(
                "Host and _time searches require occurrence locations, which this source does not retain."
                    .into(),
            );
        }
        let words = self.group_offsets.last().copied().unwrap_or(0).div_ceil(64);
        let mask_bytes = (words * size_of::<u64>()) as u64 + 64;
        self.memory
            .require_whole(
                work_bytes + mask_bytes,
                "projecting matching occurrence groups",
            )
            .map_err(|e| e.to_string())?;
        let mut mask = vec![0_u64; words];
        for m in &parent.members {
            let row = &self.data.records[m.record];
            let offset = self.group_offsets[m.record];
            for (index, _) in parent.scope.locations(row, offset) {
                let bit = offset + index;
                mask[bit / 64] |= 1_u64 << (bit % 64);
            }
        }
        let selected = matcher.select(
            &self.data.records,
            parent.members.clone(),
            Some(query::Occurrences {
                mask: &mut mask,
                offsets: &self.group_offsets,
            }),
            work_bytes + mask_bytes + copy_bytes,
            &self.memory,
        )?;
        let members = selected
            .into_iter()
            .map(|m| {
                let offset = self.group_offsets[m.record];
                let occurrences = self.data.records[m.record]
                    .occurrences
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| {
                        let bit = offset + index;
                        mask[bit / 64] & (1_u64 << (bit % 64)) != 0
                    })
                    .map(|(_, group)| group.count)
                    .sum();
                Member { occurrences, ..m }
            })
            .collect();
        scope.groups = Some(Arc::new(mask));
        // Descendants share the Arc. Only the creating view adds this allocation to history.
        Ok((members, mask_bytes))
    }

    pub fn goto(&mut self, id: usize) -> Result<(), String> {
        if id >= self.views.len() {
            return Err("unknown view handle".into());
        }
        self.active = id;
        Ok(())
    }

    pub fn back(&mut self) -> bool {
        if let Some(parent) = self.view().parent {
            self.active = parent;
            true
        } else {
            false
        }
    }

    pub fn drill(&mut self, step: Step) -> Result<usize, String> {
        let parent = self.view();
        if parent.is_final() {
            return Err(
                "This work set is allocated. Go back to its candidate to refine or allocate again."
                    .into(),
            );
        }
        if let Some(existing) = self
            .views
            .iter()
            .find(|v| v.parent == Some(parent.id) && v.step.as_ref() == Some(&step))
        {
            self.active = existing.id;
            return Ok(self.active);
        }
        if self.views.len() >= MAX_VIEWS {
            return Err("native session reached its 128-view limit".into());
        }
        // Compile before cloning the parent predicates or allocating branch scratch.
        let compiled = match &step {
            Step::Search(search) => Some(search.matcher(&self.memory)?),
            _ => None,
        };
        // The new view can never have more rows or facet keys than its navigation parent.
        let text_bytes = match &step {
            Step::Search(search) => search.query.len() as u64,
            _ => 0,
        };
        let scratch = 8192 + parent.bytes() * 6 + text_bytes * 24;
        let matcher_bytes = compiled.as_ref().map_or(0, SearchMatcher::reserved_bytes);
        self.memory
            .require_whole(scratch + matcher_bytes, "preparing triager branch")
            .map_err(|e| e.to_string())?;
        let mut scope = parent.scope.clone();
        match &step {
            Step::Host(value) if parent.hosts.contains_key(value) => {
                scope.host = Some(value.clone())
            }
            Step::Hour(value) if parent.hours.contains_key(value) => {
                scope.hour = Some(value.clone())
            }
            Step::Host(_) | Step::Hour(_) => {
                return Err("occurrence facet is unavailable in this view".into());
            }
            _ => (),
        }
        let mut projection_bytes = 0;
        let members = match &step {
            Step::Allocate(budget) => allocate(&self.data, parent, budget)?,
            Step::Search(_) => {
                let matcher = compiled
                    .as_ref()
                    .expect("search compiled before branch allocation");
                let (members, bytes) = self.search_members(parent, matcher, &mut scope, scratch)?;
                projection_bytes = bytes;
                members
            }
            Step::Region(region) => parent
                .members
                .iter()
                .copied()
                .filter(|m| self.membership[m.record].regions().contains(region))
                .map(|m| Member {
                    allocation: None,
                    ..m
                })
                .collect(),
            Step::Host(_) | Step::Hour(_) => parent
                .members
                .iter()
                .filter_map(|m| {
                    let occurrences = scope
                        .locations(&self.data.records[m.record], self.group_offsets[m.record])
                        .map(|(_, g)| g.count)
                        .sum();
                    (occurrences > 0).then_some(Member {
                        record: m.record,
                        occurrences,
                        allocation: None,
                    })
                })
                .collect(),
        };
        drop(compiled);
        let mut view = View {
            id: self.views.len(),
            parent: Some(parent.id),
            title: step.title(),
            step: Some(step),
            counts: Counts::default(),
            members,
            scope,
            regions: BTreeMap::new(),
            hosts: BTreeMap::new(),
            hours: BTreeMap::new(),
        };
        summarize(&self.data, &self.membership, &self.group_offsets, &mut view);
        let bytes = view.bytes() + projection_bytes;
        if self.history_bytes + bytes > MAX_HISTORY_BYTES {
            return Err("native session reached its view history byte limit".into());
        }
        self.memory
            .require_whole(bytes, "retaining a triager branch")
            .map_err(|e| e.to_string())?;
        self.history_bytes += bytes;
        self.memory.hold(self.memory.held_bytes() + bytes);
        self.active = view.id;
        self.views.push(view);
        Ok(self.active)
    }

    fn result_sha256(&self) -> String {
        let view = self.view();
        let mut hash = Sha256::new();
        for member in &view.members {
            let record = &self.data.records[member.record];
            hash.update((record.id.len() as u64).to_le_bytes());
            hash.update(record.id.as_bytes());
            hash.update([match member.allocation {
                None => 0,
                Some(Pool::General) => 1,
                Some(Pool::Unfamiliar) => 2,
                Some(Pool::Audit) => 3,
            }]);
            hash.update(member.occurrences.to_le_bytes());
            // The dataset pin fixes occurrence group order and content. Hashing group
            // positions distinguishes two host/time projections with the same count.
            for (index, _) in view
                .scope
                .locations(record, self.group_offsets[member.record])
            {
                hash.update((index as u64).to_le_bytes());
            }
            hash.update(u64::MAX.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    pub fn receipt(&self) -> Receipt {
        Receipt {
            version: RECEIPT_VERSION,
            dataset_sha256: self.dataset_sha256.clone(),
            c_producer: self.data.c_producer,
            top_k: self.top_k,
            filter: self.filter.clone(),
            branches: self
                .views
                .iter()
                .skip(1)
                .map(|view| Branch {
                    parent: view.parent.expect("child view"),
                    step: view.step.clone().expect("child step"),
                })
                .collect(),
            active: self.active,
            counts: self.view().counts.for_receipt(self.data.occurrences_known),
            allocated: self.view().is_final(),
            result_sha256: self.result_sha256(),
        }
    }

    pub fn reopen(data: Dataset, receipt: &Receipt) -> Result<Self, String> {
        let memory =
            MemoryBudget::observe(&MemorySettings::default()).map_err(|e| e.to_string())?;
        Self::reopen_with_budget(data, receipt, memory)
    }

    pub fn reopen_with_budget(
        data: Dataset,
        receipt: &Receipt,
        memory: MemoryBudget,
    ) -> Result<Self, String> {
        if receipt.version != RECEIPT_VERSION {
            return Err("unsupported native receipt version".into());
        }
        if receipt.branches.len() >= MAX_VIEWS {
            return Err("receipt exceeds view limit".into());
        }
        if receipt.c_producer != data.c_producer {
            return Err("receipt requires a different C producer projection".into());
        }
        let mut workflow = Self::with_budget(data, receipt.top_k, receipt.filter.clone(), memory)?;
        if receipt.dataset_sha256 != workflow.dataset_sha256 {
            return Err(
                "receipt refers to a different retained dataset or evidence projection".into(),
            );
        }
        for (i, branch) in receipt.branches.iter().enumerate() {
            workflow.goto(branch.parent)?;
            if workflow.drill(branch.step.clone())? != i + 1 {
                return Err("duplicate branch in saved receipt".into());
            }
        }
        workflow.goto(receipt.active)?;
        if workflow
            .view()
            .counts
            .for_receipt(workflow.data.occurrences_known)
            != receipt.counts
            || workflow.view().is_final() != receipt.allocated
            || workflow.result_sha256() != receipt.result_sha256
        {
            return Err(
                "receipt replay differs in membership, occurrence scope, order or allocation"
                    .into(),
            );
        }
        Ok(workflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory() -> MemoryBudget {
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(4_u64 << 30)).unwrap()
    }

    fn location(host: &str, hour: &str, count: u64) -> Occurrence {
        Occurrence {
            host: Some(host.into()),
            time: Some(format!("2026-11-18T{hour}:10:00Z")),
            count,
        }
    }

    fn data() -> Dataset {
        let records = [
            (
                "c",
                "login",
                0.5,
                Recommendation::Out,
                vec![location("red", "10", 6)],
            ),
            (
                "a",
                "token *",
                0.5,
                Recommendation::In,
                vec![
                    location("red", "10", 1_000_000_000),
                    location("red", "11", 1),
                    location("blue", "10", 3),
                ],
            ),
            (
                "b",
                "refresh",
                0.5,
                Recommendation::Unknown,
                vec![location("blue", "10", 2)],
            ),
        ]
        .into_iter()
        .map(|(id, text, score, recommendation, occurrences)| Record {
            id: id.into(),
            text: text.into(),
            score,
            recommendation,
            reason: "retained evidence".into(),
            occurrences,
        })
        .collect();
        Dataset {
            title: "exercise".into(),
            source_revision: "rev1".into(),
            provenance: "test".into(),
            occurrences_known: true,
            c_producer: CProducer::Fixture,
            records,
        }
    }

    fn open(data: Dataset) -> Workflow {
        Workflow::with_budget(
            data,
            2,
            Search {
                kind: SearchKind::Glob,
                query: "*login*".into(),
            },
            memory(),
        )
        .unwrap()
    }

    fn spl(query: &str) -> Search {
        Search {
            kind: SearchKind::Spl,
            query: query.into(),
        }
    }

    #[test]
    fn host_query_preserves_only_matching_groups_through_drills_allocation_and_replay() {
        let mut flow = open(data());
        let search = flow.drill(Step::Search(spl("host=blue OR id=c"))).unwrap();
        assert_eq!(
            flow.view().counts,
            Counts {
                originals: 3,
                occurrences: 11
            }
        );
        assert_eq!(flow.view().hosts[&Some("red".into())].occurrences, 6);
        assert_eq!(flow.view().hosts[&Some("blue".into())].occurrences, 5);
        assert_eq!(flow.view().hours.len(), 1);
        assert_eq!(flow.view().regions[&Region::AOnly].occurrences, 5);
        flow.drill(Step::Region(Region::AOnly)).unwrap();
        assert!(!flow.view().hosts.contains_key(&Some("red".into())));
        flow.drill(Step::Search(spl("id=a"))).unwrap();
        flow.drill(Step::Hour(Some("2026-11-18T10:00Z".into())))
            .unwrap();
        assert_eq!(
            flow.view().counts,
            Counts {
                originals: 1,
                occurrences: 3
            }
        );
        flow.drill(Step::Allocate(Budget::default())).unwrap();
        let receipt: Receipt =
            serde_json::from_slice(&serde_json::to_vec(&flow.receipt()).unwrap()).unwrap();
        let fresh = Workflow::reopen_with_budget(data(), &receipt, memory()).unwrap();
        assert!(fresh.view().is_final());
        assert_eq!(
            fresh
                .locations(1)
                .map(|g| g.host.as_deref())
                .collect::<Vec<_>>(),
            [Some("blue")]
        );
        assert_eq!(fresh.receipt().result_sha256, receipt.result_sha256);
        flow.goto(search).unwrap();
        flow.drill(Step::Search(spl("host=red"))).unwrap();
        assert_eq!(
            flow.view().counts,
            Counts {
                originals: 1,
                occurrences: 6
            }
        );
        assert_eq!(flow.view().members[0].record, 0);
    }

    #[test]
    fn group_digest_and_memory_track_the_projection_across_mask_words() {
        let mut data = data();
        let groups = (0..130)
            .map(|i| location(if i % 2 == 0 { "even" } else { "odd" }, "10", 1))
            .collect();
        data.records[1].occurrences = groups;
        let mut flow = open(data);
        let before = flow.memory().held_bytes();
        flow.drill(Step::Search(spl("host=even"))).unwrap();
        let expected_mask_bytes = 64 + 3 * size_of::<u64>() as u64;
        assert_eq!(
            flow.memory().held_bytes() - before,
            flow.view().bytes() + expected_mask_bytes
        );
        let parent_bytes = flow.memory().held_bytes();
        flow.drill(Step::Search(spl("id=a"))).unwrap();
        assert_eq!(
            flow.memory().held_bytes() - parent_bytes,
            flow.view().bytes()
        );
        let even = flow.receipt();
        assert_eq!(even.counts.occurrences, Some(65));
        flow.goto(0).unwrap();
        flow.drill(Step::Search(spl("host=odd"))).unwrap();
        let odd = flow.receipt();
        assert_eq!(even.counts, odd.counts);
        assert_ne!(even.result_sha256, odd.result_sha256);
    }

    #[test]
    fn spl_pipelines_order_views_and_replay_from_the_receipt() {
        let mut flow = open(data());
        let ids = |flow: &Workflow| {
            flow.view()
                .members
                .iter()
                .map(|m| flow.dataset().records[m.record].id.clone())
                .collect::<Vec<_>>()
        };
        flow.drill(Step::Search(spl("* | sort - id | head 2")))
            .unwrap();
        assert_eq!(ids(&flow), ["c", "b"]);
        assert_eq!(
            flow.view().counts,
            Counts {
                originals: 2,
                occurrences: 8
            }
        );
        flow.drill(Step::Search(spl("host=blue | tail 1"))).unwrap();
        assert_eq!(ids(&flow), ["b"]);
        assert_eq!(flow.view().counts.occurrences, 2);
        let receipt = flow.receipt();
        let reopened = Workflow::reopen_with_budget(data(), &receipt, memory()).unwrap();
        assert_eq!(reopened.receipt().result_sha256, receipt.result_sha256);
        // Facet and region drills keep the order a pipeline gave their parent.
        flow.goto(0).unwrap();
        flow.drill(Step::Search(spl("* | sort - id"))).unwrap();
        assert_eq!(ids(&flow), ["c", "b", "a"]);
        let sorted = flow.view().id;
        flow.drill(Step::Host(Some("red".into()))).unwrap();
        assert_eq!(ids(&flow), ["c", "a"]);
        flow.goto(sorted).unwrap();
        flow.drill(Step::Region(Region::AOnly)).unwrap();
        assert_eq!(ids(&flow), ["b", "a"]);
        for (filter, message) in [
            ("login | head 1", "Set B is a membership filter"),
            ("earliest=0", "Set B selects whole originals"),
        ] {
            let error = Workflow::with_budget(data(), 2, spl(filter), memory())
                .err()
                .unwrap();
            assert!(error.contains(message), "{filter}: {error}");
        }
    }

    #[test]
    fn record_level_search_works_without_locations_and_host_failure_preserves_the_view() {
        let mut input = data();
        input.occurrences_known = false;
        for row in &mut input.records {
            row.occurrences.clear();
        }
        assert!(
            Workflow::with_budget(input.clone(), 2, spl("host=blue"), memory())
                .err()
                .unwrap()
                .contains("Set B")
        );
        let mut flow = Workflow::with_budget(input.clone(), 2, spl("id=a"), memory()).unwrap();
        assert_eq!(flow.view().regions[&Region::Both].originals, 1);
        flow.drill(Step::Search(spl("score>=0.5"))).unwrap();
        let before = flow.receipt();
        assert_eq!(
            before.counts,
            ReceiptCounts {
                originals: 3,
                occurrences: None
            }
        );
        assert!(
            flow.drill(Step::Search(spl("NOT host=red")))
                .unwrap_err()
                .contains("occurrence locations")
        );
        assert_eq!(flow.view().id, before.active);
        assert_eq!(flow.views().len(), before.branches.len() + 1);
        assert_eq!(
            Workflow::reopen_with_budget(input, &before, memory())
                .unwrap()
                .receipt()
                .result_sha256,
            before.result_sha256
        );
    }

    #[test]
    fn occurrence_drills_keep_the_intersection_and_reuse_the_parent() {
        let mut flow = open(data());
        let red = flow.drill(Step::Host(Some("red".into()))).unwrap();
        assert_eq!(
            flow.view().counts,
            Counts {
                originals: 2,
                occurrences: 1_000_000_007
            }
        );
        let hour = Step::Hour(Some("2026-11-18T10:00Z".into()));
        let view = flow.drill(hour.clone()).unwrap();
        assert_eq!(flow.view().parent, Some(red));
        assert_eq!(
            flow.view().counts,
            Counts {
                originals: 2,
                occurrences: 1_000_000_006
            }
        );
        assert_eq!(
            flow.locations(1)
                .map(|group| group.count)
                .collect::<Vec<_>>(),
            [1_000_000_000]
        );
        assert_eq!(flow.locations(2).count(), 0);
        assert!(!flow.view().hosts.contains_key(&Some("blue".into())));
        assert!(flow.back());
        assert_eq!(flow.drill(hour).unwrap(), view);
        assert_eq!(flow.views().len(), 3);
        let receipt = flow.receipt();
        let reopened = Workflow::reopen_with_budget(data(), &receipt, memory()).unwrap();
        assert_eq!(reopened.view().id, view);
        assert_eq!(reopened.receipt().result_sha256, receipt.result_sha256);
    }

    #[test]
    fn shared_query_syntax_and_failed_drills_leave_the_session_recoverable() {
        let mut flow = open(data());
        // Tied originals sort by ID: A={a,b} even though the input order is c,a,b.
        assert_eq!(flow.membership(0), (false, true, Recommendation::Out));
        assert_eq!(flow.view().regions[&Region::AOnly].originals, 2);
        let step = Step::Search(Search {
            kind: SearchKind::Glob,
            query: r"*\*".into(),
        });
        flow.drill(step).unwrap();
        assert_eq!(flow.view().members[0].record, 1);
        let before = flow.receipt();
        assert!(
            flow.drill(Step::Search(Search {
                kind: SearchKind::Regex,
                query: "[".into()
            }))
            .is_err()
        );
        assert_eq!(flow.view().id, before.active);
        assert_eq!(flow.receipt().result_sha256, before.result_sha256);
        flow.goto(0).unwrap();
        flow.drill(Step::Search(Search {
            kind: SearchKind::Regex,
            query: "^ref.*sh$".into(),
        }))
        .unwrap();
        assert_eq!(flow.view().members[0].record, 2);
    }

    #[test]
    fn allocation_leaves_empty_reserves_and_pins_the_replay() {
        let mut flow = open(data());
        flow.drill(Step::Region(Region::AOnly)).unwrap();
        flow.drill(Step::Allocate(Budget::default())).unwrap();
        let allocations: Vec<_> = flow
            .view()
            .members
            .iter()
            .map(|m| (&flow.dataset().records[m.record].id[..], m.allocation))
            .collect();
        assert_eq!(
            allocations,
            [("a", Some(Pool::General)), ("b", Some(Pool::Unfamiliar))]
        );
        let receipt: Receipt =
            serde_json::from_str(&serde_json::to_string(&flow.receipt()).unwrap()).unwrap();
        assert!(receipt.allocated);
        assert_eq!(receipt.counts.occurrences, Some(1_000_000_006));
        let fresh = Workflow::reopen_with_budget(data(), &receipt, memory()).unwrap();
        assert!(fresh.view().is_final());
        let mut wrong_producer = receipt.clone();
        wrong_producer.c_producer = CProducer::PolicyInvestigateOrReview;
        assert!(Workflow::reopen_with_budget(data(), &wrong_producer, memory()).is_err());
        let mut changed = data();
        changed.records[0].reason.push('!');
        assert!(Workflow::reopen_with_budget(changed, &receipt, memory()).is_err());
        let mut tampered = receipt;
        tampered.result_sha256.clear();
        assert!(Workflow::reopen_with_budget(data(), &tampered, memory()).is_err());
    }

    #[test]
    fn final_work_sets_keep_their_allocation_and_reject_child_branches() {
        let mut flow = open(data());
        let candidate = flow.drill(Step::Region(Region::AOnly)).unwrap();
        let allocation = Step::Allocate(Budget::default());
        let final_id = flow.drill(allocation.clone()).unwrap();
        let receipt = flow.receipt();
        let saved = serde_json::to_value(&receipt).unwrap();
        for step in [
            allocation.clone(),
            Step::Region(Region::Recommended),
            Step::Host(Some("red".into())),
            Step::Hour(Some("2026-11-18T10:00Z".into())),
            Step::Search(Search {
                kind: SearchKind::Literal,
                query: "token".into(),
            }),
        ] {
            assert!(
                flow.drill(step)
                    .unwrap_err()
                    .contains("Go back to its candidate")
            );
        }
        assert_eq!(serde_json::to_value(flow.receipt()).unwrap(), saved);
        assert_eq!(flow.locations(1).count(), 3);
        let mut malformed = receipt;
        // Even a hidden child whose parent is the final violates the receipt graph.
        // Its active view and result digest still describe the legitimate final.
        malformed.branches.push(Branch {
            parent: final_id,
            step: Step::Region(Region::Recommended),
        });
        let error = Workflow::reopen_with_budget(data(), &malformed, memory())
            .err()
            .unwrap();
        assert!(error.contains("Go back to its candidate"));
        assert!(flow.back());
        assert_eq!(flow.view().id, candidate);
        assert_eq!(flow.drill(allocation).unwrap(), final_id);
        assert_eq!(serde_json::to_value(flow.receipt()).unwrap(), saved);
    }

    #[test]
    fn absent_locations_do_not_create_occurrences_or_suppress_originals() {
        let mut input = data();
        input.occurrences_known = false;
        for record in &mut input.records {
            record.occurrences.clear();
        }
        let mut flow = open(input);
        assert!(flow.view().hosts.is_empty());
        assert!(flow.drill(Step::Host(None)).is_err());
        flow.drill(Step::Region(Region::AOnly)).unwrap();
        assert_eq!(
            flow.view().counts,
            Counts {
                originals: 2,
                occurrences: 0
            }
        );
        assert_eq!(flow.view().regions[&Region::Unknown].originals, 1);
        assert_eq!(flow.receipt().counts.occurrences, None);
    }

    #[test]
    fn full_history_keeps_existing_branches_navigable() {
        let mut flow = open(data());
        for index in 1..MAX_VIEWS {
            flow.goto(0).unwrap();
            let search = Search {
                kind: SearchKind::Literal,
                query: format!("branch {index}"),
            };
            assert_eq!(flow.drill(Step::Search(search)).unwrap(), index);
        }
        assert!(flow.back());
        assert!(flow.drill(Step::Region(Region::Both)).is_err());
        assert_eq!(flow.view().id, 0);
        assert_eq!(
            flow.drill(Step::Search(Search {
                kind: SearchKind::Literal,
                query: "branch 1".into()
            }))
            .unwrap(),
            1
        );
    }
}
