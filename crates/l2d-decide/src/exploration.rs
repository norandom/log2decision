//! Direct exemplar retrieval, separate from classification and transitive clustering.
use l2d_core::patterns::{PatternMember, refinement::PatternRefinement};
use l2d_core::{BatchPlan, L2dError, MarkedSource, MemoryBudget};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

fn refused(reason: impl Into<String>) -> L2dError {
    L2dError::Refused {
        reason: reason.into(),
    }
}

/// Cosine distance rejects unsupported vectors instead of assigning proximity.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f64, L2dError> {
    if a.is_empty() || a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err(refused("invalid vector dimension or non-finite component"));
    }
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let (an, bn) = (norm(a), norm(b));
    if an == 0. || bn == 0. {
        return Err(refused("zero vector is unsupported"));
    }
    Ok((1.
        - a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) / an * (f64::from(*y) / bn))
            .sum::<f64>())
    .clamp(0., 2.))
}

/// A named encoding route. `None` is an unsupported original, never a partial vector.
pub trait ExplorationEncoder {
    fn profile(&self) -> &str;
    fn encode(&mut self, text: &str) -> Result<Option<Representation>, L2dError>;
    fn method(&self) -> &str {
        "direct-exemplar-cosine"
    }
    fn unsupported_reason(&self) -> &str;
    fn scratch_bytes(&self) -> u64;
}
pub enum Representation {
    Vector(Vec<f32>),
    Trigrams {
        text: String,
        grams: BTreeSet<[char; 3]>,
    },
}
fn distance(a: &Representation, b: &Representation) -> Result<f64, L2dError> {
    match (a, b) {
        (Representation::Vector(a), Representation::Vector(b)) => cosine_distance(a, b),
        (
            Representation::Trigrams { text: at, grams: a },
            Representation::Trigrams { text: bt, grams: b },
        ) => {
            if at == bt {
                return Ok(0.);
            }
            let union = a.union(b).count();
            Ok(if union == 0 {
                1.
            } else {
                1. - a.intersection(b).count() as f64 / union as f64
            })
        }
        _ => Err(refused("incompatible vector and lexical representations")),
    }
}
impl ExplorationEncoder for l2d_encoder::hierarchy::HierarchicalSession {
    fn profile(&self) -> &str {
        l2d_encoder::hierarchy::PROFILE
    }
    fn encode(&mut self, text: &str) -> Result<Option<Representation>, L2dError> {
        Ok(self.encode(text)?.map(Representation::Vector))
    }
    fn unsupported_reason(&self) -> &str {
        "empty original or original exceeds pinned 1024 UTF-8 byte profile; no truncation performed"
    }
    fn scratch_bytes(&self) -> u64 {
        32 * 1024 * 1024
    }
}

/// Existing exact character-trigram/Jaccard arithmetic, on original text without masks.
pub struct CharacterTrigram;
impl ExplorationEncoder for CharacterTrigram {
    fn profile(&self) -> &str {
        "character-trigram-jaccard-original-unicode-v1"
    }
    fn method(&self) -> &str {
        "direct-exemplar-character-trigram-jaccard"
    }
    fn scratch_bytes(&self) -> u64 {
        65536
    }
    fn unsupported_reason(&self) -> &str {
        "empty original text"
    }
    fn encode(&mut self, text: &str) -> Result<Option<Representation>, L2dError> {
        if text.is_empty() {
            return Ok(None);
        }
        Ok(Some(Representation::Trigrams {
            text: text.into(),
            grams: l2d_core::patterns::character_trigrams(text),
        }))
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    WithinPattern,
    AcrossPatterns,
    AllPatterns,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExploreSettings {
    pub exemplar_row: usize,
    pub scope: Scope,
    pub top_k: usize,
    pub radius: f64,
    /// Replaying a report must explicitly require its profile.
    pub expected_profile: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExplorationMember {
    pub source: MarkedSource,
    pub refinement_revision: String,
    pub selection_revision: String,
    pub row: PatternMember,
    pub distance: Option<f64>,
    pub outside_saved_search: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnsupportedMember {
    pub member: ExplorationMember,
    pub reason: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Support {
    pub members: u64,
    pub occurrences: u64,
    pub distinct_texts: usize,
    pub same_group_distinct_texts: usize,
    pub cross_group_distinct_texts: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExplorationReport {
    pub format: u32,
    pub method: String,
    pub profile: String,
    pub settings: ExploreSettings,
    pub expansion_revision: Option<String>,
    pub representative: ExplorationMember,
    pub neighbours: Vec<ExplorationMember>,
    pub boundary: Option<ExplorationMember>,
    pub distance_range: Option<[f64; 2]>,
    pub unsupported: Vec<UnsupportedMember>,
    pub support: Support,
    pub support_basis: String,
    pub group_basis: String,
    pub connectivity: String,
    pub scanned: u64,
    pub eligible: u64,
    pub plan: BatchPlan,
}
fn member(
    r: &PatternRefinement,
    row: &PatternMember,
    distance: Option<f64>,
    outside: bool,
) -> ExplorationMember {
    let mut row = row.clone();
    // A suggestion is always unlabelled. It may have appeared in a reviewed source;
    // that source remains unchanged and its exact revision is kept above.
    if outside {
        row.review = None;
    }
    ExplorationMember {
        source: r.catalogue.source.clone(),
        refinement_revision: r.revision.clone(),
        selection_revision: r.selection_revision.clone(),
        row,
        distance,
        outside_saved_search: outside,
    }
}
fn origin_bytes(r: &PatternRefinement) -> u64 {
    let source = match &r.catalogue.source {
        MarkedSource::Input(s) => {
            s.path.len()
                + s.content_sha256.len()
                + s.archive_entries
                    .as_ref()
                    .map_or(0, |entries| entries.iter().map(|e| e.len() + 64).sum())
        }
        MarkedSource::Store(s) => s.name.len() + s.content_sha256.len(),
    };
    (source + r.revision.len() + r.selection_revision.len() + 1024) as u64
}

/// Bounded direct reach. Pattern/source filters precede top-k. No all-pairs matrix,
/// chaining, score labels, or mutation of either saved artifact is performed.
pub fn explore(
    seed: &PatternRefinement,
    expansion: Option<&PatternRefinement>,
    settings: &ExploreSettings,
    encoder: &mut dyn ExplorationEncoder,
    budget: &mut MemoryBudget,
) -> Result<ExplorationReport, L2dError> {
    let guard = l2d_core::resource::ResourceGuard::tracked(
        "vector exploration",
        l2d_core::resource::busy_choice(),
        budget.usage(),
    )?;
    if settings.top_k == 0
        || settings.top_k > 10000
        || !settings.radius.is_finite()
        || !(0. ..=2.).contains(&settings.radius)
    {
        return Err(refused("top-k must be 1..10000 and cosine radius 0..2"));
    }
    if let Some(expected) = &settings.expected_profile
        && expected != encoder.profile()
    {
        return Err(refused(format!(
            "incompatible exploration profiles: requested {expected}; actual {}",
            encoder.profile()
        )));
    }
    let exemplar = seed
        .rows
        .get(settings.exemplar_row)
        .ok_or_else(|| refused("exemplar row is outside saved membership"))?;
    if settings.scope == Scope::WithinPattern && expansion.is_some() {
        return Err(refused(
            "outside-query expansion requires all-patterns or across-patterns scope",
        ));
    }
    // Bounded saved artifacts are admitted before loading by the CLI; also charge all
    // retained results, distinct-text trees, serde copies and worst-case unsupported
    // records here. This intentionally refuses large whole-membership exploration.
    let total = seed.rows.len() + expansion.map_or(0, |r| r.rows.len());
    let retained = seed
        .held_bytes()
        .saturating_add(expansion.map_or(0, |r| r.held_bytes()))
        .saturating_add(origin_bytes(seed).saturating_mul(seed.rows.len() as u64 + 2))
        .saturating_add(expansion.map_or(0, |r| {
            origin_bytes(r).saturating_mul(r.rows.len() as u64 + 2)
        }))
        .saturating_mul(32)
        .saturating_add(encoder.scratch_bytes())
        .saturating_add(4 * 1024 * 1024);
    budget.require_whole(
        retained,
        "bounded vector exploration results, support trees and encoder scratch",
    )?;
    let prior = budget.held_bytes();
    budget.hold(prior.saturating_add(retained));
    let result = (|| {
        let mut plan = budget.plan(1, encoder.scratch_bytes().max(1))?;
        // One original at a time is a bounded batch. Recheck host memory on every
        // candidate; model images have a fixed shape and are reserved above.
        plan.batch_rows = 1;
        budget.next_batch_rows(&mut plan, 0)?;
        let query = encoder.encode(&exemplar.text)?.ok_or_else(|| {
            refused(format!(
                "unsupported exemplar: {}",
                encoder.unsupported_reason()
            ))
        })?;
        distance(&query, &query)?;
        let mut report=ExplorationReport { format:1,method:encoder.method().into(),profile:encoder.profile().into(),settings:settings.clone(),expansion_revision:expansion.map(|r|r.revision.clone()),representative:member(seed,exemplar,Some(0.),false),neighbours:Vec::new(),boundary:None,distance_range:None,unsupported:Vec::new(),support:Support::default(),support_basis:"all eligible members within radius before top-k; distinct-text counts exclude repetitions of the exemplar; representative is analyst-selected, and boundary/distance range describe returned top-k only".into(),group_basis:"source snapshot + refinement revision + pattern ID; ungrouped is not independent group evidence; endpoint/host independence unavailable; same/cross counts may overlap for texts appearing in both".into(),connectivity:"direct reach to the recorded exemplar only; transitive connectivity/chaining not computed".into(),scanned:0,eligible:0,plan };
        let mut texts = BTreeSet::new();
        let mut same = BTreeSet::new();
        let mut cross = BTreeSet::new();
        let seed_members: BTreeSet<_> = seed.rows.iter().map(|r| &r.member).collect();
        for (r, outside) in std::iter::once((seed, false)).chain(expansion.map(|r| (r, true))) {
            for row in &r.rows {
                report.scanned += 1;
                let same_source = r.catalogue.source == seed.catalogue.source;
                if outside && same_source && seed_members.contains(&row.member) {
                    continue;
                }
                if !outside && row.member == exemplar.member {
                    continue;
                }
                let same_group = same_source
                    && r.revision == seed.revision
                    && row.pattern.is_some()
                    && row.pattern == exemplar.pattern;
                if (settings.scope == Scope::WithinPattern && !same_group)
                    || (settings.scope == Scope::AcrossPatterns && same_group)
                {
                    continue;
                }
                report.eligible += 1;
                budget.next_batch_rows(&mut report.plan, report.eligible - 1)?;
                report.plan.min_rows_used = 1;
                report.plan.max_rows_used = 1;
                let vector = encoder.encode(&row.text)?;
                let distance = match vector {
                    Some(v) => distance(&query, &v),
                    None => Err(refused(encoder.unsupported_reason())),
                };
                let d = match distance {
                    Ok(d) => d,
                    Err(e) => {
                        report.unsupported.push(UnsupportedMember {
                            member: member(r, row, None, outside),
                            reason: e.to_string(),
                        });
                        continue;
                    }
                };
                if d > settings.radius {
                    continue;
                }
                report.support.members += 1;
                report.support.occurrences =
                    report.support.occurrences.saturating_add(row.occurrences);
                // Exemplar repetitions are occurrences, not independent corroboration.
                if row.text != exemplar.text {
                    texts.insert(row.text.as_str());
                    if same_group {
                        same.insert(row.text.as_str());
                    } else if row.pattern.is_some() {
                        cross.insert(row.text.as_str());
                    }
                }
                let hit = member(r, row, Some(d), outside);
                report.neighbours.push(hit);
                report.neighbours.sort_by(|a, b| {
                    a.distance
                        .unwrap()
                        .total_cmp(&b.distance.unwrap())
                        .then(a.refinement_revision.cmp(&b.refinement_revision))
                        .then(a.row.member.cmp(&b.row.member))
                });
                report.neighbours.truncate(settings.top_k);
            }
        }
        report.support.distinct_texts = texts.len();
        report.support.same_group_distinct_texts = same.len();
        report.support.cross_group_distinct_texts = cross.len();
        report.boundary = report.neighbours.last().cloned();
        report.distance_range = report
            .neighbours
            .first()
            .zip(report.neighbours.last())
            .map(|(a, b)| [a.distance.unwrap(), b.distance.unwrap()]);
        debug_assert!(report.scanned <= total as u64);
        Ok(report)
    })();
    // The report, its provenance and output copies remain admitted until the caller
    // releases this reservation; failures restore the caller's prior holdings.
    if result.is_err() {
        budget.hold(prior);
    }
    guard.finish()?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_cosine_uses_direction_and_refuses_invalid_vectors() {
        assert_eq!(cosine_distance(&[2., 0.], &[8., 0.]).unwrap(), 0.);
        assert_eq!(cosine_distance(&[2., 0.], &[0., 8.]).unwrap(), 1.);
        assert_eq!(cosine_distance(&[2., 0.], &[-8., 0.]).unwrap(), 2.);
        for bad in [
            vec![],
            vec![0., 0.],
            vec![f32::NAN, 1.],
            vec![f32::INFINITY, 1.],
            vec![1.],
        ] {
            assert!(cosine_distance(&[2., 0.], &bad).is_err());
        }
    }
}
