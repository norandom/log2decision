//! Explicit development-selected Vector Decision Model routing. No qualified defaults.
use crate::ScorerKind;
use crate::close_enough::{Calibration, CloseEnough, MarkHistory, ReachSettings, close_enough};
use crate::exploration::ExplorationEncoder;
use l2d_baseline::BaselineScorer;
use l2d_core::resource::{ResourceGuard, busy_choice};
use l2d_core::{L2dError, Label, MemoryBudget};
use l2d_store::Store;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::Path;

/// Global conflict/deselection exclusion and original deduplication BEFORE top-k;
/// labelled and assumed-background pools have independent caps. No later filter.
pub const ELIGIBILITY: &str = "close-enough-originals-labelled-and-separate-background-v1";
pub const ALLOCATION: &str = "reserved-no-reassignment-general-priority-fnv1a-audit-v1";
/// Applies when the policy states `unsupported_slots`: unsupported records have their
/// own reservation and `unknown_slots` is reserved for disagreements alone.
pub const ALLOCATION_SEPARATE_UNSUPPORTED: &str =
    "reserved-unsupported-and-conflict-no-reassignment-general-priority-fnv1a-audit-v2";
const OVERHEAD: u64 = 1024 * 1024;
const QUALIFICATION: &str = "Exploratory; qualification incomplete. Policy mechanics and local support do not establish deployment quality.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Queue {
    Investigate,
    Review,
    LowPriority,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reason {
    Unsupported,
    Disagreement,
    HighCorroborated,
    LowCorroborated,
    Uncertain,
}

impl Reason {
    fn explanation(self, vector: Option<&CloseEnough>) -> &str {
        match self {
            Self::Unsupported => vector
                .and_then(|v| v.reason.as_deref())
                .unwrap_or("incompatible exploration profile; actionable ratio withheld"),
            Self::Disagreement => {
                "supported labels conflict with each other or with the configured score band"
            }
            Self::HighCorroborated => {
                "high saved character score corroborated by explicit bad-labelled originals"
            }
            Self::LowCorroborated => {
                "low saved character score corroborated by explicit good-labelled originals"
            }
            Self::Uncertain => {
                "neither corroborated score band applies, or required compatible calibration is absent"
            }
        }
    }
}

/// Every value is supplied by the caller; these are not operational defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriagePolicy {
    pub version: String,
    pub selection_provenance: String,
    pub low: f32,
    pub high: f32,
    pub corroboration: usize,
    pub reach: ReachSettings,
    pub require_calibration: bool,
    pub review_budget: usize,
    /// Shared by unsupported and disagreeing records, or reserved for disagreements
    /// alone when `unsupported_slots` is stated.
    pub unknown_slots: usize,
    /// A separate reservation for unsupported records. Absent in policies written
    /// before it existed; those keep the shared reservation and their saved form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsupported_slots: Option<usize>,
    pub audit_slots: usize,
    pub audit_seed: u64,
}

/// What routing reads from one record's evidence.
#[derive(Debug, Clone, Copy)]
struct Facts {
    ratio: Option<f32>,
    /// Within reach with enough labelled support, under the expected profile.
    supported: bool,
    bad: bool,
    good: bool,
    calibrated: bool,
}

fn refused(reason: impl Into<String>) -> L2dError {
    L2dError::Refused {
        reason: reason.into(),
    }
}

impl TriagePolicy {
    pub fn validate(&self) -> Result<(), L2dError> {
        if self.version.trim().is_empty()
            || self.selection_provenance.trim().is_empty()
            || !self.low.is_finite()
            || !self.high.is_finite()
            || self.low < 0.0
            || self.high > 1.0
            || self.low >= self.high
            || self.corroboration == 0
            || self.corroboration > self.reach.top_k
            || !self.reach.radius.is_finite()
            || !(0.0..=2.0).contains(&self.reach.radius)
            || self.reach.top_k == 0
            || self.reach.top_k > 10000
            || self.reach.minimum_labelled_support == 0
            || self.reach.minimum_labelled_support > self.reach.top_k
            || self.reach.expected_profile.trim().is_empty()
            || self.unknown_slots == 0
            || self.audit_slots == 0
            || self.unsupported_slots == Some(0)
            || self
                .unknown_slots
                .checked_add(self.audit_slots)
                .and_then(|n| n.checked_add(self.unsupported_slots.unwrap_or(0)))
                .is_none_or(|n| n > self.review_budget)
        {
            return Err(refused(
                "invalid explicit triage policy: versions, thresholds, support, reach and positive reserved allocations within the review budget are required, including any stated unsupported capacity",
            ));
        }
        Ok(())
    }

    /// The allocation revision this policy selects.
    pub fn allocation_rule(&self) -> &'static str {
        match self.unsupported_slots {
            None => ALLOCATION,
            Some(_) => ALLOCATION_SEPARATE_UNSUPPORTED,
        }
    }

    /// Only the saved character scorer ratio is actionable; distances are not scores.
    fn route(&self, evidence: &CloseEnough) -> (Queue, Reason) {
        self.route_facts(Facts {
            ratio: evidence.ratio,
            supported: evidence.close_enough && evidence.profile == self.reach.expected_profile,
            bad: evidence.bad.count >= self.corroboration,
            good: evidence.good.count >= self.corroboration,
            calibrated: evidence.calibrated.is_some(),
        })
    }

    fn route_facts(&self, facts: Facts) -> (Queue, Reason) {
        let Some(ratio) = facts
            .ratio
            .filter(|r| r.is_finite() && (0.0..=1.0).contains(r))
        else {
            return (Queue::Review, Reason::Unsupported);
        };
        if !facts.supported {
            return (Queue::Review, Reason::Unsupported);
        }
        let Facts { bad, good, .. } = facts;
        if (bad && good) || (bad && ratio <= self.low) || (good && ratio >= self.high) {
            return (Queue::Review, Reason::Disagreement);
        }
        if self.require_calibration && !facts.calibrated {
            return (Queue::Review, Reason::Uncertain);
        }
        if ratio >= self.high && bad {
            (Queue::Investigate, Reason::HighCorroborated)
        } else if ratio <= self.low && good {
            (Queue::LowPriority, Reason::LowCorroborated)
        } else {
            (Queue::Review, Reason::Uncertain)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyInput {
    /// Stable original-record identity; duplicates are refused, never silently merged.
    pub id: String,
    pub text: String,
    /// Real endpoint key, not a pattern ID. Unknown is None.
    pub group: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScoreEvidence {
    pub kind: ScorerKind,
    pub revision: String,
    pub comparison_class_assumed: bool,
    /// Always preserved, including unsupported cases; never used for queue ordering.
    pub diagnostic_ratio: f32,
    /// None when local evidence is unsupported.
    pub actionable_ratio: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum CalibrationEvidence {
    Missing,
    Invalid { revision: String, reason: String },
    Compatible { revision: String, error_rate: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Allocation {
    UnknownConflict,
    General,
    LowPriorityAudit,
    /// The separate reservation of `ALLOCATION_SEPARATE_UNSUPPORTED`.
    Unsupported,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub input: PolicyInput,
    pub score: ScoreEvidence,
    /// Complete secondary evidence, including identities/distances and assumed background.
    pub vector: Option<CloseEnough>,
    pub queue: Queue,
    pub reason: Reason,
    pub explanation: String,
    pub allocation: Option<Allocation>,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationCounts {
    pub configured: usize,
    pub selected: usize,
    pub unused: usize,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct PolicyReport {
    pub format: u32,
    /// Query snapshot supplied by a file/archive adapter. Library callers may supply
    /// independent original IDs without a file. Earlier reports omit this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_source: Option<l2d_core::MarkedInput>,
    pub policy: TriagePolicy,
    pub eligibility: String,
    pub allocation_rule: String,
    pub scorer_revision: String,
    pub evidence_revision: String,
    pub actual_profile: String,
    pub calibration: CalibrationEvidence,
    pub qualification: String,
    pub decisions: Vec<PolicyDecision>,
    /// Present exactly when the policy states `unsupported_slots`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsupported: Option<AllocationCounts>,
    pub unknown: AllocationCounts,
    pub general: AllocationCounts,
    pub audit: AllocationCounts,
    pub deferred: usize,
}

fn count(configured: usize, selected: usize) -> AllocationCounts {
    AllocationCounts {
        configured,
        selected,
        unused: configured - selected,
    }
}

fn priority(reason: Reason) -> u8 {
    match reason {
        Reason::HighCorroborated => 0,
        Reason::Disagreement => 1,
        Reason::Unsupported => 2,
        Reason::Uncertain => 3,
        Reason::LowCorroborated => 4,
    }
}

fn audit_key(id: &str, seed: u64) -> u64 {
    id.as_bytes()
        .iter()
        .fold(0xcbf29ce484222325 ^ seed, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        })
}

struct Allocated {
    unsupported: Option<AllocationCounts>,
    unknown: AllocationCounts,
    general: AllocationCounts,
    audit: AllocationCounts,
    deferred: usize,
}

fn allocate(rows: &mut [PolicyDecision], policy: &TriagePolicy) -> Allocated {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|a, b| {
        priority(rows[*a].reason)
            .cmp(&priority(rows[*b].reason))
            .then_with(|| {
                rows[*b]
                    .score
                    .actionable_ratio
                    .unwrap_or(-1.0)
                    .total_cmp(&rows[*a].score.actionable_ratio.unwrap_or(-1.0))
            })
            .then_with(|| rows[*a].input.id.cmp(&rows[*b].input.id))
    });
    let mut unsupported = 0;
    if let Some(slots) = policy.unsupported_slots {
        for &i in &order {
            if unsupported < slots && rows[i].reason == Reason::Unsupported {
                rows[i].allocation = Some(Allocation::Unsupported);
                unsupported += 1;
            }
        }
    }
    let mut unknown = 0;
    for &i in &order {
        // With a separate unsupported reservation these slots are for disagreements.
        let eligible = match policy.unsupported_slots {
            None => matches!(rows[i].reason, Reason::Unsupported | Reason::Disagreement),
            Some(_) => rows[i].reason == Reason::Disagreement,
        };
        if unknown < policy.unknown_slots && eligible {
            rows[i].allocation = Some(Allocation::UnknownConflict);
            unknown += 1;
        }
    }
    let general_capacity = policy.review_budget
        - policy.unknown_slots
        - policy.audit_slots
        - policy.unsupported_slots.unwrap_or(0);
    let mut general = 0;
    for &i in &order {
        if general < general_capacity
            && rows[i].allocation.is_none()
            && rows[i].queue != Queue::LowPriority
        {
            rows[i].allocation = Some(Allocation::General);
            general += 1;
        }
    }
    order.sort_by_key(|&i| {
        (
            audit_key(&rows[i].input.id, policy.audit_seed),
            &rows[i].input.id,
        )
    });
    let mut audit = 0;
    for i in order {
        if audit < policy.audit_slots && rows[i].queue == Queue::LowPriority {
            rows[i].allocation = Some(Allocation::LowPriorityAudit);
            audit += 1;
        }
    }
    Allocated {
        unsupported: policy.unsupported_slots.map(|n| count(n, unsupported)),
        unknown: count(policy.unknown_slots, unknown),
        general: count(general_capacity, general),
        audit: count(policy.audit_slots, audit),
        deferred: rows.len() - unsupported - unknown - general - audit,
    }
}

/// Source validation distinguishes mutable stores from captured file/archive frames.
#[derive(Clone, Copy)]
pub enum PolicySource<'a> {
    Store(&'a Store),
    InputSnapshot,
}
impl<'a> From<&'a Store> for PolicySource<'a> {
    fn from(store: &'a Store) -> Self {
        Self::Store(store)
    }
}

/// Consume a validated history and the exact scorer used to construct it.
/// Caller supplies original texts and endpoint metadata. Encoder/runtime errors
/// propagate; incompatible query representation/profile becomes explained unsupported.
/// Returned allocations remain held until dropped and the incoming budget is restored.
#[allow(clippy::too_many_arguments)]
pub fn triage_policy<'a>(
    inputs: &[PolicyInput],
    source: impl Into<PolicySource<'a>>,
    history: &MarkHistory,
    scorer: &BaselineScorer,
    calibration: Option<Calibration<'_>>,
    policy: &TriagePolicy,
    encoder: &mut dyn ExplorationEncoder,
    budget: &mut MemoryBudget,
) -> Result<PolicyReport, L2dError> {
    policy.validate()?;
    let guard = ResourceGuard::tracked("Venn triage policy", busy_choice(), budget.usage())?;
    let source = source.into();
    let generation = match source {
        PolicySource::Store(store) => {
            let generation = store.read_generation()?;
            history.require_store(store, budget)?;
            Some(generation)
        }
        PolicySource::InputSnapshot => {
            history.require_input_snapshot()?;
            None
        }
    };
    history.require_scorer(scorer)?;
    let input_bytes = inputs.iter().fold(0u64, |n, r| {
        n.saturating_add(
            (r.id.len() + r.text.len() + r.group.as_ref().map_or(0, String::len)) as u64,
        )
    });
    let mut policy_size = Counter(0);
    serde_json::to_writer(&mut policy_size, policy).map_err(|e| refused(e.to_string()))?;
    let needed = input_bytes
        .saturating_mul(4)
        .saturating_add(policy_size.0.saturating_mul(4))
        .saturating_add((inputs.len() as u64).saturating_mul(2048))
        .saturating_add(OVERHEAD);
    budget.require_whole(
        needed,
        "policy retained input, result, sorted queues and persistence metadata",
    )?;
    let prior = budget.held_bytes();
    budget.hold(prior.saturating_add(needed));
    let result = (|| {
        let mut ids = BTreeSet::new();
        for row in inputs {
            if row.id.is_empty() || !ids.insert(row.id.as_str()) {
                return Err(refused("policy input IDs must be nonempty and unique"));
            }
        }
        let calibration_evidence = match &calibration {
            None => CalibrationEvidence::Missing,
            Some(c) => match c
                .calibrator
                .require_scorer(ScorerKind::MarkSet, scorer.revision())
                .and_then(|()| c.calibrator.require_error_rate(c.error_rate))
            {
                Ok(()) => CalibrationEvidence::Compatible {
                    revision: c.calibrator.revision().into(),
                    error_rate: c.error_rate,
                },
                Err(e) => CalibrationEvidence::Invalid {
                    revision: c.calibrator.revision().into(),
                    reason: e.to_string(),
                },
            },
        };
        let compatible = matches!(calibration_evidence, CalibrationEvidence::Compatible { .. });
        let mut decisions = Vec::with_capacity(inputs.len());
        for input in inputs {
            budget.require_whole(
                scorer
                    .held_bytes()
                    .saturating_add(scorer.scoring_bytes(input.text.len() as u64)),
                "diagnostic saved scorer and scratch",
            )?;
            let diagnostic_ratio = scorer.ratio(&input.text);
            if !diagnostic_ratio.is_finite() || !(0.0..=1.0).contains(&diagnostic_ratio) {
                return Err(refused("scorer returned invalid ratio"));
            }
            let vector = if policy.reach.expected_profile == encoder.profile() {
                Some(close_enough(
                    history,
                    &input.text,
                    input.group.as_deref(),
                    scorer,
                    calibration
                        .as_ref()
                        .filter(|_| compatible)
                        .map(|c| Calibration {
                            calibrator: c.calibrator,
                            error_rate: c.error_rate,
                        }),
                    &policy.reach,
                    encoder,
                    budget,
                )?)
            } else {
                None
            };
            let (queue, reason) = vector
                .as_ref()
                .map_or((Queue::Review, Reason::Unsupported), |e| policy.route(e));
            let actionable_ratio = vector
                .as_ref()
                .and_then(|e| e.ratio)
                .filter(|_| reason != Reason::Unsupported);
            let explanation = reason.explanation(vector.as_ref()).to_owned();
            decisions.push(PolicyDecision {
                input: input.clone(),
                score: ScoreEvidence {
                    kind: ScorerKind::MarkSet,
                    revision: scorer.revision().into(),
                    comparison_class_assumed: scorer
                        .mark_set_origin()
                        .is_some_and(|o| o.other().is_none()),
                    diagnostic_ratio,
                    actionable_ratio,
                },
                vector,
                queue,
                reason,
                explanation,
                allocation: None,
            });
        }
        // Stable member order independent of input iteration order.
        decisions.sort_by(|a, b| a.input.id.cmp(&b.input.id));
        let allocated = allocate(&mut decisions, policy);
        if let PolicySource::Store(store) = source
            && generation != Some(store.read_generation()?)
        {
            return Err(refused(
                "store changed during policy batch; no mixed-generation result",
            ));
        }
        Ok(PolicyReport {
            format: 1,
            query_source: None,
            policy: policy.clone(),
            eligibility: ELIGIBILITY.into(),
            allocation_rule: policy.allocation_rule().into(),
            scorer_revision: scorer.revision().into(),
            evidence_revision: history.metadata_revision().into(),
            actual_profile: encoder.profile().into(),
            calibration: calibration_evidence,
            qualification: QUALIFICATION.into(),
            decisions,
            unsupported: allocated.unsupported,
            unknown: allocated.unknown,
            general: allocated.general,
            audit: allocated.audit,
            deferred: allocated.deferred,
        })
    })();
    let result = result.and_then(|report| {
        guard.finish()?;
        Ok(report)
    });
    if result.is_err() {
        budget.hold(prior);
    }
    result
}

/// Streaming byte counter prevents an unbounded serialization buffer.
struct Counter(u64);
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len() as u64);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
/// Check the saved aggregates against the original neighbours used for routing.
fn evidence_consistent(v: &CloseEnough, input: &PolicyInput, rate: Option<f32>) -> bool {
    use crate::close_enough::Provenance;
    let mut originals = BTreeSet::new();
    let mut groups = BTreeSet::new();
    let (mut same_group, mut cross_group) = (0, 0);
    let mut bad = Vec::new();
    let mut good = Vec::new();
    let mut benign_support = false;
    for h in &v.neighbours {
        if !originals.insert(&h.text)
            || !(0.0..=v.settings.radius).contains(&h.distance)
            || h.exact_original_match != (input.text == h.text)
        {
            return false;
        }
        let cross = match (&input.group, &h.group) {
            (Some(a), Some(b)) if a == b => {
                same_group += 1;
                false
            }
            (Some(_), Some(b)) => {
                cross_group += 1;
                groups.insert(b);
                true
            }
            _ => false,
        };
        match (h.label, h.provenance) {
            (Some(Label::Bad), Provenance::Marked | Provenance::ReviewedOpposite) => {
                bad.push(h.distance)
            }
            (Some(Label::Good), Provenance::Marked | Provenance::ReviewedOpposite) => {
                good.push(h.distance);
                benign_support |= h.exact_original_match || cross;
            }
            (None, Provenance::AssumedBackground) => {}
            _ => return false,
        }
    }
    let mean =
        |items: &[f64]| (!items.is_empty()).then(|| items.iter().sum::<f64>() / items.len() as f64);
    let labelled = bad.len() + good.len();
    let background = v.neighbours.len() - labelled;
    let supported =
        labelled >= v.settings.minimum_labelled_support && (!bad.is_empty() || benign_support);
    let calibrated = match (&v.calibrated, rate.filter(|_| supported)) {
        (None, None) => true,
        (Some(c), Some(alpha)) => {
            c.error_rate == alpha
                && (0.0..=1.0).contains(&c.p_bad)
                && (0.0..=1.0).contains(&c.p_good)
                && c.set.contains(Label::Bad) == (c.p_bad > alpha)
                && c.set.contains(Label::Good) == (c.p_good > alpha)
        }
        _ => false,
    };
    calibrated
        && v.close_enough == supported
        && v.ratio.is_some() == supported
        && v.reason.is_none() == supported
        && v.unsupported == v.unsupported_details.len()
        && v.bad.count == bad.len()
        && v.good.count == good.len()
        && v.bad.mean_distance == mean(&bad)
        && v.good.mean_distance == mean(&good)
        && labelled <= v.settings.top_k
        && background <= v.settings.top_k
        && v.assumed_background == background
        && v.same_group == same_group
        && v.cross_group == cross_group
        && v.unknown_group == v.neighbours.len() - same_group - cross_group
        && v.distinct_cross_groups == groups.len()
}

impl PolicyReport {
    /// Check the report against its embedded policy, including the retained routes,
    /// support and allocations. This reads no current scorer, history or source.
    pub fn validate_saved(mut self) -> Result<Self, L2dError> {
        self.policy.validate()?;
        if self.format != 1
            || self.qualification != QUALIFICATION
            || self.eligibility != ELIGIBILITY
            || self.allocation_rule != self.policy.allocation_rule()
        {
            return Err(refused(
                "incompatible saved policy report format, qualification, eligibility or allocation",
            ));
        }
        self.validate_members()?;
        Ok(self)
    }

    fn validate_members(&mut self) -> Result<(), L2dError> {
        let rate = match &self.calibration {
            CalibrationEvidence::Missing => None,
            CalibrationEvidence::Invalid { revision, reason }
                if !revision.is_empty() && !reason.is_empty() =>
            {
                None
            }
            CalibrationEvidence::Compatible {
                revision,
                error_rate,
            } if !revision.is_empty() && *error_rate > 0.0 && *error_rate < 1.0 => {
                Some(*error_rate)
            }
            _ => return Err(refused("inconsistent calibration identity or error rate")),
        };
        let mut previous = None;
        for row in &self.decisions {
            let vector = row.vector.as_ref();
            let expected = vector.map_or((Queue::Review, Reason::Unsupported), |v| {
                self.policy.route(v)
            });
            let ratio = vector.and_then(|v| v.ratio);
            if row.input.id.is_empty()
                || previous.is_some_and(|id| id >= row.input.id.as_str())
                || row.score.kind != ScorerKind::MarkSet
                || row.score.revision != self.scorer_revision
                || !(0.0..=1.0).contains(&row.score.diagnostic_ratio)
                || row.score.actionable_ratio != ratio
                || ratio.is_some_and(|r| r != row.score.diagnostic_ratio)
                || (row.queue, row.reason) != expected
                || row.explanation != expected.1.explanation(vector)
                || vector.is_some() != (self.actual_profile == self.policy.reach.expected_profile)
            {
                return Err(refused(
                    "inconsistent policy member identity, score or queue",
                ));
            }
            if vector.is_some_and(|v| {
                v.scorer != "mark-set"
                    || v.scorer_revision != self.scorer_revision
                    || v.evidence_revision != self.evidence_revision
                    || v.profile != self.actual_profile
                    || v.settings != self.policy.reach
                    || !evidence_consistent(v, &row.input, rate)
            }) {
                return Err(refused(
                    "inconsistent saved support or calibration evidence",
                ));
            }
            previous = Some(row.input.id.as_str());
        }
        let allocations: Vec<_> = self
            .decisions
            .iter_mut()
            .map(|r| r.allocation.take())
            .collect();
        let allocated = allocate(&mut self.decisions, &self.policy);
        if allocated.unsupported != self.unsupported
            || allocated.unknown != self.unknown
            || allocated.general != self.general
            || allocated.audit != self.audit
            || allocated.deferred != self.deferred
            || self.decisions.iter().map(|r| r.allocation).ne(allocations)
        {
            return Err(refused("inconsistent saved allocations or deferred count"));
        }
        Ok(())
    }

    /// Compare identities, evidence and queues, excluding adaptive memory telemetry.
    pub fn require_same_decisions(
        &self,
        other: &Self,
        budget: &MemoryBudget,
    ) -> Result<(), L2dError> {
        let guard = ResourceGuard::tracked(
            "Venn policy replay comparison",
            busy_choice(),
            budget.usage(),
        )?;
        let mut size = Counter(0);
        serde_json::to_writer(&mut size, self).map_err(|e| refused(e.to_string()))?;
        serde_json::to_writer(&mut size, other).map_err(|e| refused(e.to_string()))?;
        budget.require_whole(
            size.0.saturating_mul(32).saturating_add(OVERHEAD),
            "bounded policy replay comparison",
        )?;
        let normalized = |report: &Self| -> Result<serde_json::Value, L2dError> {
            let mut value = serde_json::to_value(report).map_err(|e| refused(e.to_string()))?;
            if let Some(rows) = value["decisions"].as_array_mut() {
                for row in rows {
                    if let Some(vector) = row["vector"].as_object_mut() {
                        vector.remove("plan");
                    }
                }
            }
            Ok(value)
        };
        if normalized(self)? != normalized(other)? {
            return Err(refused(
                "policy replay differs in artifacts, evidence, memberships or queues",
            ));
        }
        guard.finish()
    }
    /// Create a new artifact, refusing overwrite. Persistence is streamed and preflighted.
    pub fn save(&self, path: &Path, budget: &MemoryBudget) -> Result<(), L2dError> {
        save_with(path, budget, |file| {
            serde_json::to_writer(file, self).map_err(std::io::Error::other)
        })
    }

    /// Bounded reload. Caller-selected max_bytes limits both file input and expanded
    /// JSON object allocation; all retained parse memory remains charged on success.
    pub fn load(
        path: &Path,
        max_bytes: u64,
        policy: &TriagePolicy,
        scorer_revision: &str,
        profile: &str,
        evidence_revision: &str,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        policy.validate()?;
        let guard =
            ResourceGuard::tracked("Venn policy report reload", busy_choice(), budget.usage())?;
        let file = std::fs::File::open(path).map_err(|e| refused(e.to_string()))?;
        let len = file.metadata().map_err(|e| refused(e.to_string()))?.len();
        if len > max_bytes {
            return Err(refused("policy report exceeds caller persistence limit"));
        }
        let needed = len
            .saturating_add(1)
            .saturating_mul(32)
            .saturating_add(OVERHEAD);
        budget.require_whole(
            needed,
            "bounded policy report reload and expanded evidence/queue allocations",
        )?;
        let mut bytes = Vec::new();
        file.take(len.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|e| refused(e.to_string()))?;
        if bytes.len() as u64 != len {
            return Err(refused("policy report changed during reload"));
        }
        let out: Self = serde_json::from_slice(&bytes).map_err(|e| refused(e.to_string()))?;
        if out.scorer_revision != scorer_revision
            || out.actual_profile != profile
            || out.evidence_revision != evidence_revision
            || out.policy != *policy
        {
            return Err(refused(
                "incompatible report format, policy, scorer, eligibility, allocation, history or profile revision",
            ));
        }
        let out = out.validate_saved()?;
        guard.finish()?;
        budget.hold(budget.held_bytes().saturating_add(needed));
        Ok(out)
    }
}

fn save_with(
    path: &Path,
    budget: &MemoryBudget,
    write: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
) -> Result<(), L2dError> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let guard = ResourceGuard::tracked("Venn policy report save", busy_choice(), budget.usage())?;
    budget.require_whole(OVERHEAD, "streaming policy report persistence")?;
    let name = path
        .file_name()
        .ok_or_else(|| refused("policy output needs a filename"))?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let (temporary, mut file) = loop {
        let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
        let temp_name = format!(".venn-policy-{}-{sequence}.saving", std::process::id());
        if name
            .to_string_lossy()
            .trim_end_matches([' ', '.'])
            .eq_ignore_ascii_case(&temp_name)
        {
            continue;
        }
        let temporary = path.with_file_name(temp_name);
        match options.open(&temporary) {
            Ok(file) => break (temporary, file),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(refused(e.to_string())),
        }
    };
    let prepared = write(&mut file).and_then(|()| file.sync_all());
    drop(file);
    let result = prepared
        .map_err(|e| refused(e.to_string()))
        .and_then(|()| guard.finish())
        .and_then(|()| std::fs::hard_link(&temporary, path).map_err(|e| refused(e.to_string())));
    // The link is the commit point. Failure to remove an auxiliary name after
    // publication cannot turn a complete destination into a failed save.
    let _ = std::fs::remove_file(temporary);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_stream_is_invisible_and_can_be_retried() {
        let directory = std::env::temp_dir().join(format!(
            "venn-publish-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("report.json");
        let budget = MemoryBudget::observe_with(&Default::default(), || Ok(1 << 30)).unwrap();
        let failed = save_with(&path, &budget, |file| {
            file.write_all(b"partial")?;
            assert!(!path.exists(), "reader saw partially written report");
            Err(std::io::Error::other("injected stream failure"))
        });
        assert!(failed.is_err());
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        save_with(&path, &budget, |file| file.write_all(b"complete")).unwrap();
        assert!(save_with(&path, &budget, |file| file.write_all(b"replacement")).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"complete");
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    fn policy(unsupported_slots: Option<usize>) -> TriagePolicy {
        TriagePolicy {
            version: "unit".into(),
            selection_provenance: "unit fixture; exploratory only".into(),
            low: 0.4,
            high: 0.6,
            corroboration: 1,
            reach: ReachSettings {
                radius: 0.5,
                top_k: 3,
                minimum_labelled_support: 1,
                expected_profile: "unit-profile".into(),
            },
            require_calibration: false,
            review_budget: 6,
            unknown_slots: 2,
            unsupported_slots,
            audit_slots: 1,
            audit_seed: 5,
        }
    }

    fn queue_of(reason: Reason) -> Queue {
        match reason {
            Reason::HighCorroborated => Queue::Investigate,
            Reason::LowCorroborated => Queue::LowPriority,
            _ => Queue::Review,
        }
    }

    fn row(id: &str, reason: Reason) -> PolicyDecision {
        let ratio = (reason != Reason::Unsupported).then_some(0.5);
        PolicyDecision {
            input: PolicyInput {
                id: id.into(),
                text: id.into(),
                group: None,
            },
            score: ScoreEvidence {
                kind: ScorerKind::MarkSet,
                revision: "unit".into(),
                comparison_class_assumed: false,
                diagnostic_ratio: 0.5,
                actionable_ratio: ratio,
            },
            vector: None,
            queue: queue_of(reason),
            reason,
            explanation: String::new(),
            allocation: None,
        }
    }

    /// Three disagreements, three unsupported, one investigate and one low priority,
    /// in ID order as `triage_policy` supplies them.
    fn panel() -> Vec<PolicyDecision> {
        vec![
            row("d1", Reason::Disagreement),
            row("d2", Reason::Disagreement),
            row("d3", Reason::Disagreement),
            row("h1", Reason::HighCorroborated),
            row("l1", Reason::LowCorroborated),
            row("u1", Reason::Unsupported),
            row("u2", Reason::Unsupported),
            row("u3", Reason::Unsupported),
        ]
    }

    fn pools(rows: &[PolicyDecision]) -> Vec<(&str, Option<Allocation>)> {
        rows.iter()
            .map(|r| (r.input.id.as_str(), r.allocation))
            .collect()
    }

    #[test]
    fn every_routing_case_reaches_exactly_one_documented_route() {
        let mut seen = std::collections::BTreeMap::new();
        for required in [false, true] {
            let mut p = policy(None);
            p.require_calibration = required;
            for supported in [false, true] {
                for good in [false, true] {
                    for bad in [false, true] {
                        for calibrated in [false, true] {
                            // The low and high bands include their thresholds.
                            for (band, ratio) in [(-1, 0.4), (0, 0.5), (1, 0.6)] {
                                let got = p.route_facts(Facts {
                                    ratio: Some(ratio),
                                    supported,
                                    bad,
                                    good,
                                    calibrated,
                                });
                                // Precedence as written in the policy contract.
                                let expected = if !supported {
                                    Reason::Unsupported
                                } else if (good && bad) || (bad && band < 0) || (good && band > 0) {
                                    Reason::Disagreement
                                } else if required && !calibrated {
                                    Reason::Uncertain
                                } else if band > 0 && bad {
                                    Reason::HighCorroborated
                                } else if band < 0 && good {
                                    Reason::LowCorroborated
                                } else {
                                    Reason::Uncertain
                                };
                                assert_eq!(
                                    got,
                                    (queue_of(expected), expected),
                                    "required={required} supported={supported} good={good} bad={bad} calibrated={calibrated} band={band}"
                                );
                                *seen.entry(format!("{:?}", got.1)).or_insert(0usize) += 1;
                            }
                        }
                    }
                }
            }
        }
        // Counted by hand from the precedence: 96 cases, every route reachable.
        let expected: std::collections::BTreeMap<String, usize> = [
            ("Disagreement", 20),
            ("HighCorroborated", 3),
            ("LowCorroborated", 3),
            ("Uncertain", 22),
            ("Unsupported", 48),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        assert_eq!(seen, expected);
        assert_eq!(seen.values().sum::<usize>(), 96);
    }

    #[test]
    fn an_unusable_ratio_is_unsupported_whatever_the_evidence() {
        let p = policy(None);
        for ratio in [None, Some(f32::NAN), Some(-0.1), Some(1.1)] {
            let got = p.route_facts(Facts {
                ratio,
                supported: true,
                bad: true,
                good: false,
                calibrated: true,
            });
            assert_eq!(got, (Queue::Review, Reason::Unsupported));
        }
    }

    #[test]
    fn shared_reservation_is_unchanged_without_an_unsupported_capacity() {
        let p = policy(None);
        assert_eq!(p.allocation_rule(), ALLOCATION);
        let mut rows = panel();
        let out = allocate(&mut rows, &p);
        // Disagreements precede unsupported records, so they take both shared slots.
        // An unsupported record is reviewed only if the general pool reaches it.
        assert_eq!(
            pools(&rows),
            [
                ("d1", Some(Allocation::UnknownConflict)),
                ("d2", Some(Allocation::UnknownConflict)),
                ("d3", Some(Allocation::General)),
                ("h1", Some(Allocation::General)),
                ("l1", Some(Allocation::LowPriorityAudit)),
                ("u1", Some(Allocation::General)),
                ("u2", None),
                ("u3", None),
            ]
        );
        assert!(out.unsupported.is_none());
        assert_eq!(out.unknown, count(2, 2));
        assert_eq!(out.general, count(3, 3));
        assert_eq!(out.audit, count(1, 1));
        assert_eq!(out.deferred, 2);
    }

    #[test]
    fn unsupported_records_receive_their_own_slots_while_disagreements_fill_theirs() {
        let p = policy(Some(2));
        p.validate().unwrap();
        assert_eq!(p.allocation_rule(), ALLOCATION_SEPARATE_UNSUPPORTED);
        let mut rows = panel();
        let out = allocate(&mut rows, &p);
        assert_eq!(
            pools(&rows),
            [
                ("d1", Some(Allocation::UnknownConflict)),
                ("d2", Some(Allocation::UnknownConflict)),
                ("d3", None),
                ("h1", Some(Allocation::General)),
                ("l1", Some(Allocation::LowPriorityAudit)),
                ("u1", Some(Allocation::Unsupported)),
                ("u2", Some(Allocation::Unsupported)),
                ("u3", None),
            ]
        );
        assert_eq!(out.unsupported, Some(count(2, 2)));
        assert_eq!(out.unknown, count(2, 2));
        assert_eq!(out.general, count(1, 1));
        assert_eq!(out.audit, count(1, 1));
        assert_eq!(out.deferred, 2);
    }

    #[test]
    fn overflow_from_both_reservations_competes_in_the_general_pool() {
        let mut p = policy(Some(1));
        p.unknown_slots = 1;
        p.review_budget = 7;
        let mut rows = panel();
        let out = allocate(&mut rows, &p);
        // General holds four: investigate, then disagreements, then unsupported.
        assert_eq!(
            pools(&rows),
            [
                ("d1", Some(Allocation::UnknownConflict)),
                ("d2", Some(Allocation::General)),
                ("d3", Some(Allocation::General)),
                ("h1", Some(Allocation::General)),
                ("l1", Some(Allocation::LowPriorityAudit)),
                ("u1", Some(Allocation::Unsupported)),
                ("u2", Some(Allocation::General)),
                ("u3", None),
            ]
        );
        assert_eq!(out.general, count(4, 4));
        assert_eq!(out.deferred, 1);
    }

    #[test]
    fn an_empty_unsupported_pool_leaves_its_capacity_unused() {
        let p = policy(Some(2));
        let mut rows: Vec<_> = panel()
            .into_iter()
            .filter(|r| r.reason != Reason::Unsupported)
            .collect();
        let out = allocate(&mut rows, &p);
        assert_eq!(out.unsupported, Some(count(2, 0)));
        // No reassignment: the third disagreement stays deferred.
        assert_eq!(out.unknown, count(2, 2));
        assert_eq!(out.general, count(1, 1));
        assert_eq!(rows[2].allocation, None);
        assert_eq!(out.deferred, 1);
    }

    #[test]
    fn unsupported_capacity_must_be_positive_and_fit_the_budget() {
        policy(Some(3)).validate().unwrap();
        assert!(policy(Some(0)).validate().is_err());
        assert!(policy(Some(4)).validate().is_err());
        assert!(policy(Some(usize::MAX)).validate().is_err());
    }

    #[test]
    fn a_policy_without_the_capacity_keeps_its_saved_form() {
        let old = serde_json::to_value(policy(None)).unwrap();
        assert!(old.get("unsupported_slots").is_none());
        let back: TriagePolicy = serde_json::from_value(old).unwrap();
        assert_eq!(back, policy(None));
        let new = serde_json::to_value(policy(Some(2))).unwrap();
        assert_eq!(new["unsupported_slots"], 2);
        let back: TriagePolicy = serde_json::from_value(new).unwrap();
        assert_eq!(back, policy(Some(2)));
    }
}
