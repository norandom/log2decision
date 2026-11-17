//! Squared-L2 neighbourhood distances, never classification probabilities.
//!
//! Original source/text identifies a query: a stored match must have identical
//! vector bits and its derived ID is excluded. Equal vectors of different records
//! remain neighbours. Counts describe rows, not independent analyst support.
//! All stored records need known original-record flags and finite nonzero vectors.
//!
//! Borrowed inputs are conservatively charged even if already held by the caller.
//! Queries run one at a time; Store scans candidates in adaptive batches. Sorting
//! requires compact results in memory, preflighted before search. Returned results
//! and a JSON allowance stay reserved: drop them before restoring the incoming
//! budget held_bytes. Errors restore that reservation automatically.
use crate::neighbour_evidence::NeighbourEvidence;
use l2d_core::{BatchPlan, L2dError, MemoryBudget};
use l2d_store::{Filter, Store, VectorOrigin};
use serde::Serialize;

pub struct OutlierQuery<'a> {
    pub source: &'a str,
    pub text: &'a str,
    pub vector: &'a [f32],
}
#[derive(Debug, Serialize)]
pub struct Outlier {
    /// Input position, also the tie breaker.
    pub query: usize,
    pub distance: f32,
    pub self_excluded: bool,
    pub evidence: NeighbourEvidence,
}
#[derive(Debug, Serialize)]
pub struct OutlierRanking {
    pub encoder: l2d_core::EncoderIdentity,
    pub runtime_release: Option<String>,
    pub store_path: String,
    /// Connection-local consistency token, not a portable dataset revision.
    pub store_generation: u64,
    pub resource_record: Option<std::path::PathBuf>,
    pub k: usize,
    pub distance_metric: &'static str,
    pub rows: Vec<Outlier>,
    /// Candidate-scan plans in INPUT order, including memory changes.
    pub searches: Vec<BatchPlan>,
    pub validation: Option<BatchPlan>,
    pub retained_bytes: u64,
}
fn refused(reason: impl Into<String>) -> L2dError {
    L2dError::Refused {
        reason: reason.into(),
    }
}
pub fn rank_outliers(
    store: &Store,
    origin: VectorOrigin<'_>,
    queries: &[OutlierQuery<'_>],
    k: Option<usize>,
    budget: &mut MemoryBudget,
) -> Result<OutlierRanking, L2dError> {
    store.require_origin(origin)?;
    let guard = l2d_core::resource::ResourceGuard::tracked(
        "outlier ranking",
        l2d_core::resource::busy_choice(),
        budget.usage(),
    )?;
    let k = k.unwrap_or(5);
    if k == 0 || k == usize::MAX {
        return Err(refused(
            "outlier k must be positive and leave room for self exclusion",
        ));
    }
    let incoming = budget.held_bytes();
    let mut retained = (queries.len() as u64)
        .saturating_mul(2048)
        .saturating_add(4096)
        .saturating_add(
            8 * (origin.identity.name.len() as u64
                + origin.identity.revision.len() as u64
                + origin.runtime_release.map_or(0, |r| r.len() as u64)
                + store.path().as_os_str().len() as u64),
        );
    let input = queries.iter().fold(0u64, |sum, q| {
        sum.saturating_add(128)
            .saturating_add(q.source.len() as u64)
            .saturating_add(q.text.len() as u64)
            .saturating_add((q.vector.len() as u64).saturating_mul(4))
    });
    let result = (|| {
        let generation = store.read_generation()?;
        let stored = store.len()?;
        // At most one adjustment per stored row; cover transient scan diagnostics.
        let scratch = stored.saturating_mul(256).saturating_add(65536);
        let working = input.saturating_add(scratch);
        budget.require_whole(
            retained.saturating_add(working),
            "outlier input, retained ranking and scan diagnostics memory",
        )?;
        budget.hold(incoming.saturating_add(retained).saturating_add(working));
        for (i, q) in queries.iter().enumerate() {
            if q.vector.len() != usize::from(origin.identity.dimension)
                || q.vector.iter().any(|x| !x.is_finite())
                || q.vector.iter().all(|&x| x == 0.)
            {
                return Err(refused(format!(
                    "unsupported query {i}: wrong dimension, zero or nonfinite vector"
                )));
            }
        }
        let validation = if queries.is_empty() {
            None
        } else {
            Some(store.require_original_vectors_with(origin, budget)?)
        };
        let extra = validation
            .as_ref()
            .map_or(0, |p| p.changes.capacity() as u64 * 256);
        budget.require_whole(extra, "retained vector validation memory changes")?;
        retained = retained.saturating_add(extra);
        budget.hold(budget.held_bytes().saturating_add(extra));
        let mut out = OutlierRanking {
            encoder: origin.identity.clone(),
            runtime_release: origin.runtime_release.map(str::to_owned),
            store_path: store.path().to_string_lossy().into_owned(),
            store_generation: generation,
            resource_record: guard.record_path().map(std::path::Path::to_path_buf),
            k,
            distance_metric: "squared_l2_distance",
            rows: Vec::with_capacity(queries.len()),
            searches: Vec::with_capacity(queries.len()),
            validation,
            retained_bytes: 0,
        };
        for (query, q) in queries.iter().enumerate() {
            if store.read_generation()? != generation {
                return Err(refused(
                    "store changed during outlier ranking; no ranking published",
                ));
            }
            let filter_bytes = (q.source.len() as u64 + q.text.len() as u64)
                .saturating_mul(4)
                .saturating_add(1024);
            budget.require_whole(filter_bytes, "stored query identity filter")?;
            let base = budget.held_bytes();
            budget.hold(base.saturating_add(filter_bytes));
            let filter = Filter {
                source: Some(q.source.into()),
                equals: Some(q.text.into()),
                ..Default::default()
            };
            let matched = store.matching_with(&filter, 1, budget)?;
            let self_excluded = matched.rows != 0;
            let needed = (k as u64).saturating_add(u64::from(self_excluded));
            if stored < needed {
                return Err(refused(format!(
                    "outlier ranking: {stored} stored, {needed} needed for k={k} besides the query"
                )));
            }
            if self_excluded {
                budget.require_whole(matched.read_bytes, "stored query vector verification")?;
                budget.hold(budget.held_bytes().saturating_add(matched.read_bytes));
                let own = store.text_search_with(&filter, 1, budget)?;
                let vector = own
                    .column(l2d_store::VECTOR)
                    .and_then(|c| c.list())
                    .map_err(|e| refused(e.to_string()))?
                    .get_as_series(0)
                    .ok_or_else(|| refused("stored query disappeared"))?;
                if vector
                    .f32()
                    .map_err(|e| refused(e.to_string()))?
                    .into_no_null_iter()
                    .map(f32::to_bits)
                    .ne(q.vector.iter().map(|x| x.to_bits()))
                {
                    return Err(refused(format!(
                        "query {query} differs from its stored vector"
                    )));
                }
            }
            drop(filter);
            budget.hold(base);
            let self_id = l2d_store::log_id(q.source, q.text);
            let nearest = store.nearest_with(
                q.vector,
                origin,
                k + usize::from(self_excluded),
                &Filter::default(),
                budget,
            )?;
            let hits = &nearest.hits;
            budget.hold(base.saturating_add(nearest.hits_bytes));
            budget.require_held(query as u64)?;
            let ids = hits
                .column(l2d_store::ID)
                .and_then(|c| c.str())
                .map_err(|e| refused(e.to_string()))?;
            let distances = hits
                .column(l2d_store::DISTANCE)
                .and_then(|c| c.f32())
                .map_err(|e| refused(e.to_string()))?;
            let labels = hits
                .column(l2d_store::LABEL)
                .and_then(|c| c.str())
                .map_err(|e| refused(e.to_string()))?;
            let mut evidence = NeighbourEvidence::default();
            let mut used = 0;
            let mut distance = 0.;
            for row in 0..hits.height() {
                if ids.get(row) == Some(self_id.as_str()) {
                    continue;
                }
                distance = distances
                    .get(row)
                    .ok_or_else(|| refused("missing neighbour distance"))?;
                if !distance.is_finite() {
                    return Err(refused("unsupported squared-L2 distance overflow"));
                }
                evidence.add(labels.get(row).unwrap_or("unknown"), f64::from(distance));
                used += 1;
                if used == k {
                    break;
                }
            }
            if used != k {
                return Err(refused(format!(
                    "outlier ranking: {} stored, {needed} needed; store changed",
                    nearest.candidates
                )));
            }
            let plan = nearest.plan;
            drop(nearest.hits);
            budget.hold(base);
            if let Some(plan) = plan {
                let extra = (plan.changes.capacity() as u64).saturating_mul(256);
                budget.require_whole(extra, "retained outlier scan memory changes")?;
                retained = retained.saturating_add(extra);
                budget.hold(budget.held_bytes().saturating_add(extra));
                out.searches.push(plan);
            }
            out.rows.push(Outlier {
                query,
                distance,
                self_excluded,
                evidence,
            });
        }
        if store.read_generation()? != generation {
            return Err(refused(
                "store changed during outlier ranking; no ranking published",
            ));
        }
        out.rows.sort_unstable_by(|a, b| {
            b.distance
                .total_cmp(&a.distance)
                .then(a.query.cmp(&b.query))
        });
        out.retained_bytes = retained;
        Ok(out)
    })();
    budget.hold(incoming.saturating_add(if result.is_ok() { retained } else { 0 }));
    if let Err(error) = guard.finish() {
        budget.hold(incoming);
        return Err(error);
    }
    result
}
