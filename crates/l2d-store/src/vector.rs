//! The exact search by vector (requirements 5.3, 5.4, 5.6; task 1.9).
//!
//! [`Store::nearest`] compares query vectors with the stored vectors of the logs that a
//! [`Filter`] names and returns, for each query, the nearest of them with their
//! distances. Every stored vector that passes the filter is compared: there is no
//! approximate index.
//!
//! # The distance
//!
//! The distance is the squared L2 distance, in 32-bit floats:
//! `(v - q) · (v - q)` for a stored vector `v` and a query `q`. The difference is
//! formed first and its product with itself, a `1 × d` by `d × 1` matrix product, is
//! computed by `faer` (`faer::linalg::matmul::dot::inner_prod`, its SIMD kernel for the
//! CPU at hand). It is not computed as `|v|² + |q|² - 2 v·q` with one matrix product
//! over a whole batch, although that is faster: that form loses the small distances to
//! cancellation (two equal unit vectors come out at about 1e-7, not 0, and the nearest
//! are the ones that suffer), and the store does not require unit vectors. As it is, a
//! stored vector equal to the query is at the distance 0 exactly, and a distance
//! depends on its two vectors only, so it is the same whatever batch the stored vector
//! was read in (requirement 12.6). It can differ in its last bits between CPUs with
//! different SIMD widths.
//!
//! # Order
//!
//! The hits of a query come nearest first. Among equal distances the log that was
//! stored first comes first (the order of `seq`, the insertion order of the store), and
//! where the cut at `k` falls inside such a tie, the logs stored first are kept. Two
//! distances are equal when their 32-bit values are.
//!
//! # Filter before distance (requirement 5.4)
//!
//! The filter is applied by the statement that yields the stored vectors (the text
//! index, the label and the source in SQL, the rest per row, see [`crate::query`]), so
//! only the logs it names are compared and ranked.
//!
//! # Memory (requirement 12)
//!
//! The search holds, for as long as it runs, the share of any search with its filter
//! ([`crate::Explain::held_bytes`]: SQLite's fixed share, and with a phrase what the
//! text index and the reading of texts take, see [`crate::query`]) and the best hits
//! so far of every query (16 bytes per hit kept). The stored vectors are read in
//! batches sized by the caller's budget, [`NEAREST_ROW_BYTES`] and the vector per row,
//! with the free memory read again before each batch; a batch is compared with every
//! query and then dropped. Many queries are answered in one pass over the store. The
//! hits are then read back as rows, which is counted first and asked of the budget
//! like any reading of rows.
//!
//! faer is used for that one inner product and for nothing else.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use faer::linalg::matmul::dot::inner_prod;
use faer::{Conj, col::ColRef, row::RowRef};
use l2d_core::polars::prelude::{Column, DataFrame};
use l2d_core::{BatchPlan, EncoderIdentity, MemoryBudget, MemorySettings};

use crate::query::{
    Compiled, DISTANCE, QUERY, READ_FIXED_BYTES, ROW_COLUMNS, RowsBuilder, TEXT_READ_FACTOR,
    decode, row_bytes,
};
use crate::{Filter, Store, StoreError, failed};

/// What the budget is asked for per stored vector of a batch, beside the vector
/// itself: its place in the insertion order (8 bytes), doubled.
pub const NEAREST_ROW_BYTES: u64 = 16;

/// What a kept hit takes while the store is searched: a distance and a row number.
const HIT_BYTES: u64 = 16;

/// What [`Store::nearest`] found.
#[derive(Debug, Clone)]
pub struct Nearest {
    /// The hits frame ([`crate::hits_schema`]): for each query, in the order of the
    /// queries, its nearest logs, nearest first. `query` is the number of the query
    /// from 0, `distance` the squared L2 distance; the other columns are those of the
    /// stored row without the vector (`group` for requirement 6.7).
    pub hits: DataFrame,
    /// The query vectors given.
    pub queries: u64,
    /// The stored logs the filter named, each of which was compared with every query.
    pub candidates: u64,
    /// What the search held of the budget for as long as it ran: the share of a search
    /// with its filter ([`crate::Explain::held_bytes`]) and the best hits of every
    /// query. 0 when nothing was read.
    pub held_bytes: u64,
    /// What the budget was asked for the hits frame.
    pub hits_bytes: u64,
    /// The free memory observed, the limit, the bytes per stored vector and the batch
    /// sizes used (requirement 12.3, 12.8); `None` when nothing was read (no query,
    /// `k` = 0, an empty store, a phrase no text can hold).
    pub plan: Option<BatchPlan>,
}
/// A stored log as a hit: ordered by distance, then by insertion order.
#[derive(Debug, Clone, Copy)]
struct Near {
    distance: f32,
    seq: i64,
}

impl Ord for Near {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.seq.cmp(&other.seq))
    }
}

impl PartialOrd for Near {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Near {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Near {}

/// The squared L2 distance of `a` and `b`; `difference` is scratch of their length.
fn squared_l2(a: &[f32], b: &[f32], difference: &mut [f32]) -> f32 {
    for ((d, x), y) in difference.iter_mut().zip(a).zip(b) {
        *d = x - y;
    }
    inner_prod(
        RowRef::from_slice(difference),
        Conj::No,
        ColRef::from_slice(difference),
        Conj::No,
    )
}

/// Compares the stored vectors of a batch (`vectors`, one after the other, with their
/// places `seqs` in the insertion order) with every query and keeps the `keep` best of
/// each.
fn rank(
    vectors: &[f32],
    seqs: &[i64],
    queries: &[f32],
    dimension: usize,
    keep: usize,
    best: &mut [BinaryHeap<Near>],
    difference: &mut [f32],
) {
    for (query, best) in queries.chunks_exact(dimension).zip(best) {
        for (vector, &seq) in vectors.chunks_exact(dimension).zip(seqs) {
            let near = Near {
                distance: squared_l2(vector, query, difference),
                seq,
            };
            if best.len() < keep {
                best.push(near);
            } else if let Some(mut worst) = best.peek_mut()
                && near < *worst
            {
                *worst = near;
            }
        }
    }
}

impl Store {
    /// [`Store::nearest_with`] under a budget observed now with the default settings,
    /// in which nothing is held: for callers that hold nothing that matters.
    pub fn nearest(
        &self,
        queries: &[f32],
        identity: &EncoderIdentity,
        k: usize,
        filter: &Filter,
    ) -> Result<Nearest, StoreError> {
        let mut budget =
            MemoryBudget::observe(&MemorySettings::default()).map_err(StoreError::Memory)?;
        self.nearest_with(queries, identity, k, filter, &mut budget)
    }

    /// For each query vector, the `k` stored logs nearest to it by squared L2 distance
    /// among the logs that `filter` names (see the module description for the
    /// distance, the order among equal distances, and the memory).
    ///
    /// `queries` holds the query vectors one after the other, each of the store's
    /// dimension, produced by the encoder `identity`.
    ///
    /// - Another identity than the store's is [`StoreError::IdentityMismatch`], naming
    ///   both (requirement 5.6); it is checked before anything else.
    /// - `queries` that is not whole vectors, or holds a value that is not finite, and
    ///   a filter that cannot be searched for, are [`StoreError::InvalidQuery`].
    /// - Fewer logs than `k` pass the filter: all of them are returned.
    /// - No query, `k` = 0, or a phrase no stored text can hold: no hits. Nothing is
    ///   read and nothing is asked of the budget, so this is never refused;
    ///   `candidates` is 0 and there is no plan.
    /// - When the budget does not have the search's own share, the smallest batch or
    ///   the hits, the search is refused ([`StoreError::Memory`]) with the memory
    ///   needed and the limit.
    /// - When the file no longer describes this open store, [`StoreError::Failed`],
    ///   naming what it was opened as and what the file says now.
    ///
    /// The hits do not depend on the batch size. Nothing in the store is changed, and
    /// the budget holds afterwards what it held before.
    pub fn nearest_with(
        &self,
        queries: &[f32],
        identity: &EncoderIdentity,
        k: usize,
        filter: &Filter,
        budget: &mut MemoryBudget,
    ) -> Result<Nearest, StoreError> {
        self.require_identity(identity)?;
        let dimension = usize::from(self.identity().dimension);
        if !queries.len().is_multiple_of(dimension) {
            return Err(StoreError::InvalidQuery {
                detail: format!(
                    "{} values were given as query vectors; the store's encoder has the \
                     dimension {dimension}, so that is {} whole vectors and {} values more",
                    queries.len(),
                    queries.len() / dimension,
                    queries.len() % dimension
                ),
            });
        }
        if let Some(at) = queries.iter().position(|value| !value.is_finite()) {
            return Err(StoreError::InvalidQuery {
                detail: format!(
                    "query {}: value {} of the vector is {}, not a finite number",
                    at / dimension + 1,
                    at % dimension + 1,
                    queries[at]
                ),
            });
        }
        let compiled = Compiled::new(filter)?;
        let count = queries.len() / dimension;
        let nothing = |plan: Option<BatchPlan>, held_bytes: u64| -> Result<Nearest, StoreError> {
            Ok(Nearest {
                hits: hits_frame(
                    Vec::new(),
                    Vec::new(),
                    RowsBuilder::new(0, dimension, false),
                )
                .map_err(|error| failed("searching the store", self.path(), error))?,
                queries: count as u64,
                candidates: 0,
                held_bytes,
                hits_bytes: 0,
                plan,
            })
        };
        if count == 0 || k == 0 || compiled.nothing {
            return nothing(None, 0);
        }

        let vector_bytes = 4 * dimension as u64;
        let row = vector_bytes + NEAREST_ROW_BYTES;
        // At most `k` hits are kept per query, and no more than the store holds.
        let stored = self.len()?;
        let keep = k.min(usize::try_from(stored).unwrap_or(usize::MAX));
        if keep == 0 {
            return nothing(None, 0);
        }
        let best_bytes = (count as u64)
            .saturating_mul((keep as u64 + 1).saturating_mul(HIT_BYTES) + 32)
            .saturating_add(vector_bytes);
        let more_parts = format!(
            ", and {best_bytes} bytes for the {keep} best hits of each of the {count} queries"
        );
        self.search(&compiled, best_bytes, &more_parts, budget, |budget, own| {
            let mut plan = budget.plan(stored, row).map_err(StoreError::Memory)?;
            let scan = self.scan(&compiled, queries, keep, stored, &mut plan, budget)?;
            let (hits, hits_bytes) = self.hits(&scan.best, budget)?;
            Ok(Nearest {
                hits,
                queries: count as u64,
                candidates: scan.candidates,
                held_bytes: own,
                hits_bytes,
                plan: Some(plan),
            })
        })
    }

    /// Reads the stored vectors that `compiled` names in batches and ranks them against
    /// every query.
    fn scan(
        &self,
        compiled: &Compiled,
        queries: &[f32],
        keep: usize,
        stored: u64,
        plan: &mut BatchPlan,
        budget: &mut MemoryBudget,
    ) -> Result<Scan, StoreError> {
        let dimension = usize::from(self.identity().dimension);
        // The largest batch of the plan, or the whole store when that is less.
        let capacity = usize::try_from(plan.batch_rows.min(stored).max(1)).unwrap_or(usize::MAX);
        let mut vectors: Vec<f32> = Vec::with_capacity(capacity * dimension);
        let mut seqs: Vec<i64> = Vec::with_capacity(capacity);
        let mut difference = vec![0.0f32; dimension];
        let mut best: Vec<BinaryHeap<Near>> = (0..queries.len() / dimension)
            .map(|_| BinaryHeap::with_capacity(keep + 1))
            .collect();

        let fail = |error| self.failure(error);
        let columns = if compiled.reads_text() {
            "logs.seq, logs.vector, octet_length(logs.text), logs.text"
        } else {
            "logs.seq, logs.vector"
        };
        let mut statement = self
            .connection
            .prepare(&compiled.sql(columns))
            .map_err(fail)?;
        let mut rows = statement
            .query(compiled.parameters().as_slice())
            .map_err(fail)?;
        let mut candidates = 0u64;
        let mut exhausted = false;
        while !exhausted {
            let batch = budget
                .next_batch_rows(plan, candidates)
                .map_err(StoreError::Memory)?
                .min(capacity as u64) as usize;
            vectors.clear();
            seqs.clear();
            while seqs.len() < batch {
                let Some(row) = rows.next().map_err(fail)? else {
                    exhausted = true;
                    break;
                };
                if !compiled.accepts(row, 2, 3).map_err(fail)? {
                    continue;
                }
                let blob = row
                    .get_ref(1)
                    .and_then(|value| value.as_blob().map_err(rusqlite::Error::from))
                    .map_err(fail)?;
                decode(blob, dimension, &mut vectors).map_err(fail)?;
                seqs.push(row.get(0).map_err(fail)?);
            }
            rank(
                &vectors,
                &seqs,
                queries,
                dimension,
                keep,
                &mut best,
                &mut difference,
            );
            candidates += seqs.len() as u64;
        }
        Ok(Scan {
            best: best.into_iter().map(BinaryHeap::into_sorted_vec).collect(),
            candidates,
        })
    }

    /// Reads the hits back as rows, after asking the budget for them.
    fn hits(
        &self,
        best: &[Vec<Near>],
        budget: &MemoryBudget,
    ) -> Result<(DataFrame, u64), StoreError> {
        let fail = |error| self.failure(error);
        let total: usize = best.iter().map(Vec::len).sum();
        let mut sizes = self
            .connection
            .prepare(
                "SELECT octet_length(text), octet_length(source), \
                 ifnull(octet_length(group_key), 0) FROM logs WHERE seq = ?1",
            )
            .map_err(fail)?;
        let mut bytes = 0u64;
        let mut longest = 0u64;
        for near in best.iter().flatten() {
            let (text, source, group): (u64, u64, u64) = sizes
                .query_row([near.seq], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })
                .map_err(fail)?;
            // The number of the query and the distance, beside the row.
            bytes += row_bytes(text, source, group, 16);
            longest = longest.max(text);
        }
        bytes += READ_FIXED_BYTES + TEXT_READ_FACTOR * longest;
        budget
            .require_whole(
                bytes,
                &format!(
                    "reading the {total} nearest logs found in the store {} back",
                    self.name()
                ),
            )
            .map_err(StoreError::Memory)?;

        let mut builder = RowsBuilder::new(total, usize::from(self.identity().dimension), false);
        let mut query_numbers: Vec<u32> = Vec::with_capacity(total);
        let mut distances: Vec<f32> = Vec::with_capacity(total);
        let mut fetch = self
            .connection
            .prepare(&format!("SELECT {ROW_COLUMNS} FROM logs WHERE seq = ?1"))
            .map_err(fail)?;
        for (query, hits) in best.iter().enumerate() {
            for near in hits {
                fetch
                    .query_row([near.seq], |row| builder.push(row, 0))
                    .map_err(fail)?;
                query_numbers.push(query as u32);
                distances.push(near.distance);
            }
        }
        let frame = hits_frame(query_numbers, distances, builder)
            .map_err(|error| failed("searching the store", self.path(), error))?;
        Ok((frame, bytes))
    }
}

/// What a pass over the stored vectors found.
struct Scan {
    /// The best hits of each query, nearest first.
    best: Vec<Vec<Near>>,
    candidates: u64,
}

fn hits_frame(
    queries: Vec<u32>,
    distances: Vec<f32>,
    rows: RowsBuilder,
) -> l2d_core::polars::prelude::PolarsResult<DataFrame> {
    let mut columns = vec![
        Column::new(QUERY.into(), queries),
        Column::new(DISTANCE.into(), distances),
    ];
    columns.extend(rows.finish());
    DataFrame::new(columns)
}
