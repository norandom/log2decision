//! One review budget allocated once (`triager_compose` with `mode=allocate`,
//! requirements 3.1 and 3.2; interface.md, "The final call identifies the three
//! populations explicitly").
//!
//! A request names a candidate expression and a budget of named pools. Each pool has
//! a population (`from`, an expression over the same named sets, which may reach
//! beyond the candidate expression), a limit and an order. The pools are filled in
//! the order they are stated; an original that an earlier pool took is not taken
//! again (`first_pool_wins`), and a pool whose population runs short leaves its slots
//! unused (`leave_unfilled`). These are the semantics of the parent policy's rule
//! `reserved-no-reassignment-general-priority-fnv1a-audit` over named pools instead
//! of fixed routes: reserved pools first, no reassignment, no silent spill-over, and
//! the audit sample ordered by the policy's own seeded FNV-1a key
//! ([`l2d_decide::policy::audit_key`]) over the record handle with the handle as the
//! tie-breaker. The rule this allocation states is [`ALLOCATION_RULE`].
//!
//! The result is a final work set: its members are the selected originals in the
//! order of the work set (pool by pool, each in its order), and it keeps why each
//! slot was filled. A deferred original, one in the candidate expression or in a
//! pool's population that no slot took, is explained when it is inspected: the pools
//! whose population held it were full.
//!
//! # Memory
//!
//! The allocation asks the budget before it reads: a block of every operand, of the
//! parent, of the final set and of every score table it orders by; a run of rows of
//! the relation; the candidates of the widest pool; the slots of the whole budget
//! twice (in the order of the work set and by row); the result. The final set keeps
//! its states (in memory or on disk, as every set) and the slots.

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

use l2d_core::L2dError;
use serde_json::{Map, Value, json};

use super::membership::{Expr, State, States, Tally, code_in, put_code};
use super::ops::{
    self, ARTIFACT_FIXED_BYTES, COMPOSE_FIXED_BYTES, Commit, Cost, Done, Worker, registry,
};
use super::state::{Origin, Population, Producer, Set, SetKind, Snapshot};
use super::tables::{BLOCK_ROWS, Scores, score_in};
use super::wire::{self, Fault, VERSION};
use crate::scope::RowBlock;

/// The rule an allocation states: named pools filled in their order, an original
/// taken once, unused slots left unused, the seeded audit key of the parent policy.
pub(crate) const ALLOCATION_RULE: &str =
    "named-pools-first-pool-wins-leave-unfilled-fnv1a-audit-v1";
/// The most review slots of one budget, and the most pools.
pub(crate) const MAX_SLOTS: u64 = 10_000;
pub(crate) const MAX_POOLS: usize = 8;
/// The most order keys of one pool.
const MAX_KEYS: usize = 3;

/// One key of a pool's order. Every order ends with the record ID, so it is total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OrderKey {
    /// The score of a score set, highest first, an original without one last; the
    /// index is into the allocation's score sets.
    ScoreDesc(usize),
    /// The policy's seeded FNV-1a key over the record handle, lowest first.
    SeededHash,
    RecordIdAscending,
}

/// A pool as a request states it, resolved.
#[derive(Debug, Clone)]
pub(crate) struct Pool {
    pub name: String,
    pub limit: u64,
    pub from: Expr,
    pub order: Vec<OrderKey>,
    pub seed: Option<u64>,
}

/// An allocation request with every handle resolved before it was scheduled.
pub(crate) struct AllocateTask {
    pub snapshot: Arc<Snapshot>,
    pub scope: Population,
    /// The named sets, in the order the expressions index them.
    pub sets: Vec<(String, Arc<Set>)>,
    pub expr: Expr,
    pub total: u64,
    pub pools: Vec<Pool>,
    /// The sets whose scores a `score_desc` key reads.
    pub scores: Vec<Arc<Set>>,
}

/// One filled slot of a final work set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Slot {
    pub row: u64,
    pub pool: u32,
    /// 1-based within its pool.
    pub rank: u32,
    /// Drawn from a pool's population outside the candidate expression.
    pub outside_candidate: bool,
}

/// What one pool did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PoolOutcome {
    pub name: String,
    pub limit: u64,
    pub selected: u64,
    /// Its population within the parent: distinct originals and occurrences.
    pub eligible: (u64, u64),
    /// Of its population, what another pool took.
    pub taken_elsewhere: u64,
    /// Of its population, what no slot took.
    pub deferred: (u64, u64),
    pub outside_candidate: u64,
}

/// A final work set's allocation: the request, the slots and the counts.
pub(crate) struct Allocation {
    /// The canonical request, every name replaced by its handle.
    pub spec: Value,
    pub total: u64,
    pub names: Vec<String>,
    pub operands: Vec<Arc<Set>>,
    pub expr: Expr,
    pub pools: Vec<Pool>,
    pub outcomes: Vec<PoolOutcome>,
    /// In the order of the work set.
    pub slots: Vec<Slot>,
    /// Row and index into `slots`, by row.
    pub by_row: Vec<(u64, u32)>,
    /// The candidate expression within the parent: in, and not evaluated.
    pub candidate: (u64, u64),
    pub candidate_unknown: (u64, u64),
    /// In the candidate expression or a pool's population, and in no slot.
    pub deferred: (u64, u64),
}

impl Allocation {
    /// What the allocation keeps in memory beside the final set's states.
    pub fn held_bytes(&self) -> u64 {
        (self.slots.capacity() * std::mem::size_of::<Slot>()
            + self.by_row.capacity() * std::mem::size_of::<(u64, u32)>()
            + self.names.iter().map(String::capacity).sum::<usize>()
            + self
                .pools
                .iter()
                .map(|pool| pool.name.capacity() + 256)
                .sum::<usize>()
            + self.outcomes.capacity() * std::mem::size_of::<PoolOutcome>()) as u64
            + 16 * self.spec.to_string().len() as u64
            + 64 * 64
    }

    /// The slot that holds `row`, if any.
    pub fn slot_of(&self, row: u64) -> Option<(usize, &Slot)> {
        let at = self.by_row.binary_search_by_key(&row, |(of, _)| *of).ok()?;
        let index = self.by_row[at].1 as usize;
        Some((index, &self.slots[index]))
    }

    /// Why `row` has a slot or none: the pool, the slot and the order, or the pools
    /// whose population held it and were full.
    pub fn explain(&self, row: u64) -> Result<Value, L2dError> {
        if let Some((index, slot)) = self.slot_of(row) {
            let pool = &self.pools[slot.pool as usize];
            let order: Vec<Value> = pool.order.iter().map(|key| key_name(*key)).collect();
            let mut reason = format!(
                "fills slot {} of the work set: rank {} of the pool `{}` by {}",
                index + 1,
                slot.rank,
                pool.name,
                order
                    .iter()
                    .map(|key| key.as_str().unwrap_or("score_desc").to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            if slot.outside_candidate {
                reason.push_str(
                    "; drawn from the pool's population outside the candidate expression",
                );
            }
            return Ok(json!({
                "allocation": pool.name,
                "slot": index + 1,
                "pool_rank": slot.rank,
                "outside_candidate": slot.outside_candidate,
                "seed": pool.seed,
                "reason": reason,
            }));
        }
        let code = |index: usize| -> State {
            self.operands[index]
                .states
                .code(row)
                .map(State::of)
                .unwrap_or(State::Unknown)
        };
        let candidate = self.expr.eval(&code);
        let full: Vec<&str> = self
            .pools
            .iter()
            .filter(|pool| pool.from.eval(&code) == State::In)
            .map(|pool| pool.name.as_str())
            .collect();
        let (deferred, reason) = match (full.is_empty(), candidate) {
            (false, _) => (
                true,
                format!(
                    "deferred: every pool whose population holds it was full ({}); its slots \
                     went to originals before it in the pool's order",
                    full.iter()
                        .map(|name| format!("`{name}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ),
            (true, State::In) => (
                true,
                "deferred: in the candidate expression and in no pool's population".to_string(),
            ),
            (true, State::Unknown) => (
                false,
                "not eligible: the candidate expression is not evaluated for it and no pool's \
                 population holds it"
                    .to_string(),
            ),
            (true, State::Out) => (
                false,
                "not eligible: outside the candidate expression and every pool's population"
                    .to_string(),
            ),
        };
        Ok(json!({
            "allocation": Value::Null,
            "deferred": deferred,
            "eligible_for": full,
            "candidate": candidate.name(),
            "reason": reason,
        }))
    }

    /// The allocation block of a result or a receipt: the budget, the pools and the
    /// counts in both units.
    pub fn describe(&self) -> Value {
        let selected = self.slots.len() as u64;
        let outside: u64 = self
            .outcomes
            .iter()
            .map(|pool| pool.outside_candidate)
            .sum();
        json!({
            "rule": ALLOCATION_RULE,
            "audit_key": "l2d_decide::policy::audit_key: FNV-1a 64-bit over the record handle, \
                          offset basis XOR seed, the handle breaks a tie",
            "duplicate_rule": "first_pool_wins",
            "shortfall": "leave_unfilled",
            "capacity": self.total,
            "selected": selected,
            "unused": self.total - selected,
            "pools": self.outcomes.iter().zip(&self.pools).map(|(outcome, pool)| json!({
                "name": outcome.name,
                "limit": outcome.limit,
                "selected": outcome.selected,
                "unused": outcome.limit - outcome.selected,
                "order": pool.order.iter().map(|key| key_name(*key)).collect::<Vec<_>>(),
                "seed": pool.seed,
                "eligible": {"distinct_originals": outcome.eligible.0,
                             "occurrences": outcome.eligible.1},
                "taken_by_other_pools": outcome.taken_elsewhere,
                "deferred": {"distinct_originals": outcome.deferred.0,
                             "occurrences": outcome.deferred.1},
                "outside_candidate": outcome.outside_candidate,
            })).collect::<Vec<_>>(),
            "candidate": {
                "distinct_originals": self.candidate.0,
                "occurrences": self.candidate.1,
                "not_evaluated": {"distinct_originals": self.candidate_unknown.0,
                                  "occurrences": self.candidate_unknown.1},
            },
            "deferred": {"distinct_originals": self.deferred.0, "occurrences": self.deferred.1},
            "outside_candidate": {
                "distinct_originals": outside,
                "reason": "drawn by a pool whose population reaches beyond the candidate \
                           expression",
            },
        })
    }
}

/// An order key as a request and a result state it; a score key names its set by
/// position in the allocation's score sets, which the canonical form replaces.
fn key_name(key: OrderKey) -> Value {
    match key {
        OrderKey::ScoreDesc(_) => json!("score_desc"),
        OrderKey::SeededHash => json!("seeded_hash"),
        OrderKey::RecordIdAscending => json!("record_id_ascending"),
    }
}

/// Whether `set` keeps the scores of a frozen scorer, as a `score_desc` key reads them.
fn keeps_scores(set: &Set) -> bool {
    set.scores.is_some()
        && matches!(
            &set.producer,
            Producer::Score { .. }
                | Producer::Retained {
                    origin: Origin::Score { .. },
                    ..
                }
        )
}

/// The score sets a set was made from: itself, the sets of a view's expression or an
/// allocation, and what they were made from; each table once, in the order found.
pub(crate) fn score_sources(set: &Arc<Set>, found: &mut Vec<Arc<Set>>, depth: usize) {
    if depth > 16 {
        return;
    }
    if keeps_scores(set) {
        let table = set.scores.as_ref().map(Arc::as_ptr);
        if !found
            .iter()
            .any(|known| known.scores.as_ref().map(Arc::as_ptr) == table)
        {
            found.push(set.clone());
        }
        return;
    }
    for input in &set.inputs {
        score_sources(input, found, depth + 1);
    }
}

/// A budget as a request states it: the total, the pools and the score sets their
/// orders read.
pub(crate) type Budget = (u64, Vec<Pool>, Vec<Arc<Set>>);

/// Reads the `budget` of an allocation request over the named `sets`: the pools with
/// their populations, limits and orders, and the score sets their orders read.
pub(crate) fn parse_budget(
    budget: &Value,
    names: &[&str],
    sets: &[(String, Arc<Set>)],
) -> Result<Budget, Fault> {
    let fields = budget.as_object().ok_or_else(|| {
        Fault::invalid(
            "`budget` is an object with `total`, `duplicate_rule`, `shortfall` and `pools`",
        )
    })?;
    if let Some(field) = fields
        .keys()
        .find(|field| !["total", "duplicate_rule", "shortfall", "pools"].contains(&field.as_str()))
    {
        return Err(Fault::invalid(format!(
            "`budget.{}` is no argument here; the arguments are total, duplicate_rule, \
             shortfall, pools (every review slot belongs to a named pool)",
            wire::clip(field, 64).0
        )));
    }
    let total = fields
        .get("total")
        .and_then(Value::as_u64)
        .filter(|total| (1..=MAX_SLOTS).contains(total))
        .ok_or_else(|| {
            Fault::invalid(format!(
                "`budget.total` is a whole number from 1 to {MAX_SLOTS}"
            ))
        })?;
    match fields.get("duplicate_rule").and_then(Value::as_str) {
        Some("first_pool_wins") => {}
        _ => {
            return Err(Fault::invalid(
                "`budget.duplicate_rule` is `first_pool_wins`: an original takes one slot, \
                 in the first pool that reaches it",
            ));
        }
    }
    match fields.get("shortfall").and_then(Value::as_str) {
        Some("leave_unfilled") => {}
        _ => {
            return Err(Fault::invalid(
                "`budget.shortfall` is `leave_unfilled`: a pool whose population runs short \
                 leaves its slots unused, nothing is reassigned",
            ));
        }
    }
    let listed = fields
        .get("pools")
        .and_then(Value::as_array)
        .filter(|pools| (1..=MAX_POOLS).contains(&pools.len()))
        .ok_or_else(|| {
            Fault::invalid(format!(
                "`budget.pools` is a list of 1 to {MAX_POOLS} pools"
            ))
        })?;
    let mut pools: Vec<Pool> = Vec::with_capacity(listed.len());
    let mut scores: Vec<Arc<Set>> = Vec::new();
    let mut sum = 0u64;
    for (at, pool) in listed.iter().enumerate() {
        let fields = pool.as_object().ok_or_else(|| {
            Fault::invalid(format!(
                "`budget.pools[{at}]` is an object with `name`, `limit`, `from`, `order` and, \
                 for `seeded_hash`, `seed`"
            ))
        })?;
        if let Some(field) = fields
            .keys()
            .find(|field| !["name", "limit", "from", "order", "seed"].contains(&field.as_str()))
        {
            return Err(Fault::invalid(format!(
                "`budget.pools[{at}].{}` is no argument here; the arguments are name, limit, \
                 from, order, seed",
                wire::clip(field, 64).0
            )));
        }
        let name = fields
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| {
                (1..=32).contains(&name.len())
                    && name.as_bytes()[0].is_ascii_lowercase()
                    && name.bytes().all(|byte| {
                        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'
                    })
            })
            .ok_or_else(|| {
                Fault::invalid(format!(
                    "`budget.pools[{at}].name` is 1 to 32 lower-case letters, digits or '_' and \
                     starts with a letter"
                ))
            })?;
        if pools.iter().any(|known| known.name == name) {
            return Err(Fault::invalid(format!(
                "the pool `{name}` is named twice in `budget.pools`"
            )));
        }
        let limit = fields
            .get("limit")
            .and_then(Value::as_u64)
            .filter(|limit| (1..=total).contains(limit))
            .ok_or_else(|| {
                Fault::invalid(format!(
                    "`budget.pools[{at}].limit` is a whole number from 1 to `budget.total`"
                ))
            })?;
        sum += limit;
        let from = Expr::parse(
            fields
                .get("from")
                .ok_or_else(|| Fault::invalid(format!("`budget.pools[{at}].from` is required")))?,
            names,
        )?;
        let keys = fields
            .get("order")
            .and_then(Value::as_array)
            .filter(|keys| (1..=MAX_KEYS).contains(&keys.len()))
            .ok_or_else(|| {
                Fault::invalid(format!(
                    "`budget.pools[{at}].order` is a list of 1 to {MAX_KEYS} keys ending with \
                     `record_id_ascending`"
                ))
            })?;
        let mut order = Vec::with_capacity(keys.len());
        for key in keys {
            let parsed = match key {
                Value::String(key) if key == "seeded_hash" => OrderKey::SeededHash,
                Value::String(key) if key == "record_id_ascending" => OrderKey::RecordIdAscending,
                Value::String(key) if key == "score_desc" => {
                    let mut found = Vec::new();
                    for (_, set) in sets {
                        score_sources(set, &mut found, 0);
                    }
                    match found.len() {
                        1 => OrderKey::ScoreDesc(score_index(&mut scores, &found[0])),
                        0 => {
                            return Err(Fault::invalid(
                                "`score_desc` orders by the scores of a score set, and none of \
                                 `sets` is one or was composed from one; name a score set in \
                                 `sets` and order by {\"score_desc\": NAME}",
                            ));
                        }
                        _ => {
                            return Err(Fault::invalid(
                                "`score_desc` is ambiguous: `sets` reach the scores of more \
                                 than one scorer; order by {\"score_desc\": NAME} with a score \
                                 set of `sets`",
                            ));
                        }
                    }
                }
                Value::Object(named) if named.len() == 1 && named.contains_key("score_desc") => {
                    let name = named["score_desc"].as_str().unwrap_or_default();
                    let set = sets
                        .iter()
                        .find(|(known, _)| known == name)
                        .map(|(_, set)| set)
                        .filter(|set| keeps_scores(set))
                        .ok_or_else(|| {
                            Fault::invalid(format!(
                                "`{{\"score_desc\": \"{}\"}}` names no score set of `sets`",
                                wire::clip(name, 64).0
                            ))
                        })?;
                    OrderKey::ScoreDesc(score_index(&mut scores, set))
                }
                _ => {
                    return Err(Fault::invalid(format!(
                        "an order key of `budget.pools[{at}]` is `score_desc`, \
                         {{\"score_desc\": NAME}}, `seeded_hash` or `record_id_ascending`"
                    )));
                }
            };
            let same_kind =
                |known: &OrderKey| std::mem::discriminant(known) == std::mem::discriminant(&parsed);
            if order.iter().any(same_kind) {
                return Err(Fault::invalid(format!(
                    "`budget.pools[{at}].order` names a key twice"
                )));
            }
            order.push(parsed);
        }
        if order.last() != Some(&OrderKey::RecordIdAscending) {
            return Err(Fault::invalid(format!(
                "`budget.pools[{at}].order` ends with `record_id_ascending`, the tie rule that \
                 makes every order total"
            )));
        }
        let seeded = order.contains(&OrderKey::SeededHash);
        let seed = match (fields.get("seed"), seeded) {
            (Some(seed), true) => Some(seed.as_u64().ok_or_else(|| {
                Fault::invalid(format!(
                    "`budget.pools[{at}].seed` is a whole number of 0 to 2^64 - 1"
                ))
            })?),
            (None, true) => {
                return Err(Fault::invalid(format!(
                    "`budget.pools[{at}]` orders by `seeded_hash` and needs its `seed`"
                )));
            }
            (Some(_), false) => {
                return Err(Fault::invalid(format!(
                    "`budget.pools[{at}].seed` is taken only with the order key `seeded_hash`"
                )));
            }
            (None, false) => None,
        };
        pools.push(Pool {
            name: name.to_string(),
            limit,
            from,
            order,
            seed,
        });
    }
    if sum != total {
        return Err(Fault::invalid(format!(
            "the limits of `budget.pools` add up to {sum} and `budget.total` is {total}; every \
             review slot belongs to exactly one pool"
        )));
    }
    Ok((total, pools, scores))
}

fn score_index(scores: &mut Vec<Arc<Set>>, set: &Arc<Set>) -> usize {
    match scores.iter().position(|known| Arc::ptr_eq(known, set)) {
        Some(at) => at,
        None => {
            scores.push(set.clone());
            scores.len() - 1
        }
    }
}

/// The canonical request: every name replaced by its handle, as the handle of the
/// final set is derived from it and a receipt records it.
pub(crate) fn canonical(task: &AllocateTask) -> Value {
    let handle = |index: usize| json!(task.sets[index].1.handle);
    let pools: Vec<Value> = task
        .pools
        .iter()
        .map(|pool| {
            let order: Vec<Value> = pool
                .order
                .iter()
                .map(|key| match key {
                    OrderKey::ScoreDesc(at) => json!({"score_desc": task.scores[*at].handle}),
                    other => key_name(*other),
                })
                .collect();
            let mut described = json!({
                "name": pool.name,
                "limit": pool.limit,
                "from": pool.from.describe(&handle),
                "order": order,
            });
            if let Some(seed) = pool.seed {
                described["seed"] = json!(seed);
            }
            described
        })
        .collect();
    json!({
        "mode": "allocate",
        "scope": task.scope.handle(),
        "expr": task.expr.describe(&handle),
        "budget": {
            "total": task.total,
            "duplicate_rule": "first_pool_wins",
            "shortfall": "leave_unfilled",
            "pools": pools,
        },
        "rule": ALLOCATION_RULE,
    })
}

/// A candidate of a pool: its sort keys, the smaller the better, and its row. The
/// greatest is the worst, so that a heap of the best keeps the worst on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    keys: [u64; MAX_KEYS],
    row: u64,
}

const CANDIDATE_BYTES: u64 = std::mem::size_of::<Candidate>() as u64;

/// A score as a key that sorts the highest score first and no score last.
fn score_key(score: f32) -> u64 {
    if !score.is_finite() {
        return u64::MAX;
    }
    let bits = score.to_bits();
    // The order of `f32::total_cmp`, as an unsigned number.
    let ordered = if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    };
    u64::from(u32::MAX - ordered)
}

/// The record handle of `position`, written into `out` without an allocation.
pub(crate) fn record_id_into(position: u64, out: &mut [u8; 18]) -> &str {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out[0] = b'r';
    out[1] = b'-';
    for at in 0..16 {
        out[2 + at] = HEX[((position >> (60 - 4 * at)) & 0xf) as usize];
    }
    std::str::from_utf8(out).expect("ASCII")
}

/// What fills the pools: the candidates of the pool being filled, the slots of the
/// whole budget and the rows taken so far. [`Filler::share`] is what it holds at
/// most, asked for before it is made.
struct Filler {
    heap: BinaryHeap<Candidate>,
    slots: Vec<Slot>,
    taken: Vec<u64>,
    limit: u64,
}

impl Filler {
    /// What a filler for a budget of `total` slots whose widest pool has `widest`
    /// takes at most: the candidates of one pool, the slots, the rows taken, and the
    /// slots by row that [`Filler::finish`] makes.
    fn share(total: u64, widest: u64) -> u64 {
        widest
            .saturating_mul(CANDIDATE_BYTES)
            .saturating_add(total.saturating_mul(
                (std::mem::size_of::<Slot>()
                    + std::mem::size_of::<u64>()
                    + std::mem::size_of::<(u64, u32)>()) as u64,
            ))
    }

    fn new(total: u64, widest: u64) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(widest as usize),
            slots: Vec::with_capacity(total as usize),
            taken: Vec::with_capacity(total as usize),
            limit: 0,
        }
    }

    fn begin(&mut self, limit: u64) {
        self.heap.clear();
        self.limit = limit;
    }

    fn is_taken(&self, row: u64) -> bool {
        self.taken.binary_search(&row).is_ok()
    }

    fn offer(&mut self, candidate: Candidate) {
        if (self.heap.len() as u64) < self.limit {
            self.heap.push(candidate);
        } else if self.heap.peek().is_some_and(|worst| candidate < *worst) {
            self.heap.pop();
            self.heap.push(candidate);
        }
    }

    /// The best of the pool take its slots, best first; the heap's room is kept for
    /// the next pool.
    fn end(&mut self, pool: u32) {
        let chosen = std::mem::take(&mut self.heap).into_sorted_vec();
        for (rank, candidate) in chosen.iter().enumerate() {
            self.slots.push(Slot {
                row: candidate.row,
                pool,
                rank: rank as u32 + 1,
                outside_candidate: false,
            });
            self.taken.push(candidate.row);
        }
        self.taken.sort_unstable();
        self.heap = BinaryHeap::from(chosen);
        self.heap.clear();
    }

    /// The slots in the order of the work set, and by row.
    fn finish(self) -> (Vec<Slot>, Vec<(u64, u32)>) {
        let Filler {
            heap, slots, taken, ..
        } = self;
        drop(heap);
        drop(taken);
        let mut by_row: Vec<(u64, u32)> = Vec::with_capacity(slots.len());
        by_row.extend(
            slots
                .iter()
                .enumerate()
                .map(|(index, slot)| (slot.row, index as u32)),
        );
        by_row.sort_unstable();
        (slots, by_row)
    }
}

/// The keys of `row` under `order`.
fn keys_of(
    order: &[OrderKey],
    seed: Option<u64>,
    position: u64,
    scores: &[f32],
) -> [u64; MAX_KEYS] {
    let mut keys = [0u64; MAX_KEYS];
    for (at, key) in order.iter().enumerate() {
        keys[at] = match key {
            OrderKey::ScoreDesc(index) => score_key(scores[*index]),
            OrderKey::SeededHash => {
                let mut id = [0u8; 18];
                l2d_decide::policy::audit_key(record_id_into(position, &mut id), seed.unwrap_or(0))
            }
            OrderKey::RecordIdAscending => position,
        };
    }
    keys
}

/// The blocks a pass over the relation reads at a time.
struct Blocks {
    within: Vec<u8>,
    codes: Vec<Vec<u8>>,
    values: Vec<Vec<u8>>,
    run: RowBlock,
    run_start: u64,
    run_end: u64,
}

impl Blocks {
    fn bytes(sets: usize, scores: usize) -> u64 {
        (sets as u64 + 1) * States::BLOCK_BYTES as u64
            + scores as u64 * Scores::BLOCK_BYTES as u64
            + RowBlock::BYTES
            + (sets + scores) as u64 * std::mem::size_of::<Vec<u8>>() as u64
    }

    fn new(sets: usize, scores: usize) -> Self {
        Self {
            within: vec![0u8; States::BLOCK_BYTES],
            codes: (0..sets).map(|_| vec![0u8; States::BLOCK_BYTES]).collect(),
            values: (0..scores)
                .map(|_| vec![0u8; Scores::BLOCK_BYTES])
                .collect(),
            run: RowBlock::new(),
            run_start: 0,
            run_end: 0,
        }
    }

    fn load(&mut self, task: &AllocateTask, first: u64) -> Result<usize, L2dError> {
        let rows = task.scope.read_block(first, &mut self.within)?;
        for ((_, set), codes) in task.sets.iter().zip(&mut self.codes) {
            set.states.read_block(first, codes)?;
        }
        for (set, values) in task.scores.iter().zip(&mut self.values) {
            if let Some(scores) = &set.scores {
                scores.read_block(first, values)?;
            }
        }
        Ok(rows)
    }

    /// The record ID and the occurrences of `row`.
    fn row(&mut self, snapshot: &Snapshot, row: u64) -> Result<(u64, u64), L2dError> {
        if !(self.run_start..self.run_end).contains(&row) {
            let rows = snapshot.scope.rows(row, true, &mut self.run)?;
            self.run_start = row;
            self.run_end = row + rows;
        }
        Ok((self.run.position(row), self.run.occurrences(row)))
    }

    fn state(&self, at: usize) -> impl Fn(usize) -> State + '_ {
        move |index| State::of(code_in(&self.codes[index], at))
    }
}

/// Allocates the budget of `task` once and returns the final work set.
pub(crate) fn allocate(worker: &mut Worker<'_>, task: &AllocateTask) -> Result<Done, Fault> {
    let started = Instant::now();
    let snapshot = &task.snapshot;
    let rows = snapshot.scope.originals();
    let spec = canonical(task);
    let handle = wire::handle(
        "final",
        &worker.shared.config.project,
        &[&snapshot.handle, task.scope.handle(), &spec.to_string()],
    );
    if let Some(known) = registry(worker.shared).set(&handle) {
        return Ok(Done {
            result: known.result.clone(),
            commit: None,
        });
    }
    let producer = Producer::Allocation { spec: spec.clone() };
    // Everything the allocation holds, asked for before any of it is allocated.
    let widest = task.pools.iter().map(|pool| pool.limit).max().unwrap_or(0);
    let fixed = Blocks::bytes(task.sets.len() + 1, task.scores.len())
        .saturating_add(Filler::share(task.total, widest))
        .saturating_add(task.scores.len() as u64 * 4)
        .saturating_add(COMPOSE_FIXED_BYTES + ARTIFACT_FIXED_BYTES)
        // The canonical request three times: here, in the producer and in the
        // allocation the final set keeps.
        .saturating_add(3 * producer.held_bytes())
        .saturating_add(64 * 1024);
    worker.budget.hold(worker.base);
    worker.budget.require_whole(
        fixed,
        "allocating a review budget: a block of every operand, the candidates of the widest \
         pool and the slots of the budget,",
    )?;
    worker.budget.hold(worker.base.saturating_add(fixed));
    let on_disk = ops::tables_on_disk(worker, States::bytes_for(rows));
    if !on_disk {
        worker
            .budget
            .hold(worker.base.saturating_add(fixed + States::bytes_for(rows)));
    }
    let mut states = if on_disk {
        let dir = ops::set_dir(
            worker,
            States::bytes_for(rows),
            "placing the membership of a final work set on disk",
        )?;
        States::on_disk(&dir, rows)?
    } else {
        States::in_memory(rows, State::Out)
    };

    let mut blocks = Blocks::new(task.sets.len(), task.scores.len());
    let mut scores = vec![f32::NAN; task.scores.len()];
    let mut filler = Filler::new(task.total, widest);
    for (index, pool) in task.pools.iter().enumerate() {
        filler.begin(pool.limit);
        let mut first = 0;
        while first < rows {
            worker.check_cancel()?;
            let in_block = blocks.load(task, first)?;
            for at in 0..in_block {
                if code_in(&blocks.within, at) != State::In as u8 {
                    continue;
                }
                let row = first + at as u64;
                if pool.from.eval(&blocks.state(at)) != State::In || filler.is_taken(row) {
                    continue;
                }
                for (score, values) in scores.iter_mut().zip(&blocks.values) {
                    *score = score_in(values, at);
                }
                let (position, _) = blocks.row(snapshot, row)?;
                filler.offer(Candidate {
                    keys: keys_of(&pool.order, pool.seed, position, &scores),
                    row,
                });
            }
            first += BLOCK_ROWS;
        }
        filler.end(index as u32);
    }
    let (mut slots, by_row) = filler.finish();

    // One more pass: the states of the final set and the counts of every pool.
    let mut outcomes: Vec<PoolOutcome> = task
        .pools
        .iter()
        .map(|pool| PoolOutcome {
            name: pool.name.clone(),
            limit: pool.limit,
            selected: 0,
            eligible: (0, 0),
            taken_elsewhere: 0,
            deferred: (0, 0),
            outside_candidate: 0,
        })
        .collect();
    let (mut candidate, mut candidate_unknown, mut deferred) = ((0, 0), (0, 0), (0, 0));
    let mut codes = vec![0u8; States::BLOCK_BYTES];
    let mut first = 0;
    while first < rows {
        worker.check_cancel()?;
        let in_block = blocks.load(task, first)?;
        codes[..in_block.div_ceil(4)].fill(0);
        for at in 0..in_block {
            if code_in(&blocks.within, at) != State::In as u8 {
                continue;
            }
            let row = first + at as u64;
            let (_, occurrences) = blocks.row(snapshot, row)?;
            let state = blocks.state(at);
            let wanted = task.expr.eval(&state);
            match wanted {
                State::In => {
                    candidate.0 += 1;
                    candidate.1 += occurrences;
                }
                State::Unknown => {
                    candidate_unknown.0 += 1;
                    candidate_unknown.1 += occurrences;
                }
                State::Out => {}
            }
            let slot = by_row
                .binary_search_by_key(&row, |(of, _)| *of)
                .ok()
                .map(|at| by_row[at].1 as usize);
            let mut eligible = wanted == State::In;
            for (index, pool) in task.pools.iter().enumerate() {
                if pool.from.eval(&state) != State::In {
                    continue;
                }
                eligible = true;
                let outcome = &mut outcomes[index];
                outcome.eligible.0 += 1;
                outcome.eligible.1 += occurrences;
                match slot {
                    Some(slot) if slots[slot].pool as usize == index => {}
                    Some(_) => outcome.taken_elsewhere += 1,
                    None => {
                        outcome.deferred.0 += 1;
                        outcome.deferred.1 += occurrences;
                    }
                }
            }
            match slot {
                Some(slot) => {
                    put_code(&mut codes, at, State::In as u8);
                    let filled = &mut slots[slot];
                    let outcome = &mut outcomes[filled.pool as usize];
                    outcome.selected += 1;
                    if wanted != State::In {
                        filled.outside_candidate = true;
                        outcome.outside_candidate += 1;
                    }
                }
                None if eligible => {
                    deferred.0 += 1;
                    deferred.1 += occurrences;
                }
                None => {}
            }
        }
        states.write_block(first, &codes, in_block)?;
        first += BLOCK_ROWS;
    }
    drop(blocks);
    let (tally, revision, _) = ops::tally(snapshot, &task.scope, &states)?;
    if tally.parent() != task.scope.size() || tally.of(State::In).0 != slots.len() as u64 {
        return Err(Fault::runtime(
            "the members of a final work set do not add up to its parent and its slots",
        ));
    }
    let names: Vec<String> = task.sets.iter().map(|(name, _)| name.clone()).collect();
    let mut inputs: Vec<Arc<Set>> = task.sets.iter().map(|(_, set)| set.clone()).collect();
    for set in &task.scores {
        if !inputs.iter().any(|known| Arc::ptr_eq(known, set)) {
            inputs.push(set.clone());
        }
    }
    let allocation = Allocation {
        spec,
        total: task.total,
        names,
        operands: task.sets.iter().map(|(_, set)| set.clone()).collect(),
        expr: task.expr.clone(),
        pools: task.pools.clone(),
        outcomes,
        slots,
        by_row,
        candidate,
        candidate_unknown,
        deferred,
    };
    let mut cost = Cost::new(started);
    cost.set("index_rows_read", rows * (task.pools.len() as u64 + 1))
        .set("candidates", tally.parent().0)
        .set("output", allocation.slots.len() as u64)
        .set("tables_on_disk", on_disk)
        .set("disk_bytes", states.disk_bytes())
        .set("disk_peak_bytes", states.disk_bytes());
    let result = final_result(
        &handle,
        snapshot,
        &task.scope,
        &task.sets,
        &allocation,
        &tally,
        &revision,
        cost.finish(worker),
        None,
    );
    ops::require_fit(&result)?;
    let kept = states
        .held_bytes()
        .saturating_add(producer.held_bytes())
        .saturating_add(allocation.held_bytes())
        .saturating_add(2 * result.len() as u64 + handle.capacity() as u64)
        .saturating_add(ARTIFACT_FIXED_BYTES);
    let set = Set {
        handle,
        kind: SetKind::Final,
        parent: task.scope.as_set().cloned(),
        snapshot: snapshot.clone(),
        states,
        scores: None,
        owns_scores: false,
        producer,
        tally,
        revision,
        result: result.clone(),
        held: kept,
        inputs,
        allocation: Some(Arc::new(allocation)),
    };
    worker.budget.hold(worker.base.saturating_add(kept));
    Ok(Done {
        result,
        commit: Some(Commit::Set(Arc::new(set))),
    })
}

/// The result of a final work set, as `triager_compose` returns it and its overview
/// splices it in.
#[allow(clippy::too_many_arguments)]
pub(crate) fn final_result(
    handle: &str,
    snapshot: &Snapshot,
    scope: &Population,
    sets: &[(String, Arc<Set>)],
    allocation: &Allocation,
    tally: &Tally,
    revision: &str,
    cost: Value,
    retained: Option<Value>,
) -> String {
    let named: Map<String, Value> = sets
        .iter()
        .map(|(name, set)| (name.clone(), json!(set.handle)))
        .collect();
    let mut operation = allocation.spec.clone();
    operation["sets"] = Value::Object(named);
    let selected = allocation.slots.len() as u64;
    let mut result = json!({
        "v": VERSION,
        "status": "ready",
        "set": handle,
        "kind": "final",
        "snapshot": snapshot.handle,
        "parent": scope.handle(),
        "operation": operation,
        "revision": revision,
        "capacity": allocation.total,
        "selected": selected,
        "allocation": allocation.describe(),
        "counts": tally.counts(),
        "occurrence_counts": tally.occurrence_counts(),
        "suggestion_only": "an allocation orders review; it creates no analyst mark",
        "evidence": {
            "tool": "triager_inspect",
            "handle": handle,
            "views": ["records", "evidence"],
            "order": "allocation",
        },
        "cost": cost,
    });
    if let Some(retained) = retained {
        result["retained_from"] = retained;
    }
    wire::with_response_bytes(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Filling the pools of the largest budget allocates no more than the allocation
    /// asks the budget for: every pool offered three times its limit, the widest as
    /// wide as the budget.
    #[test]
    fn filling_the_pools_allocates_no_more_than_its_share() {
        for (total, limits) in [
            (MAX_SLOTS, vec![MAX_SLOTS]),
            (MAX_SLOTS, vec![6_000, 3_000, 1_000]),
            (24, vec![18, 4, 2]),
        ] {
            let widest = *limits.iter().max().unwrap();
            let (slots, peak) = crate::test_heap::measure(|| {
                let mut filler = Filler::new(total, widest);
                let mut row = 0u64;
                for (pool, limit) in limits.iter().enumerate() {
                    filler.begin(*limit);
                    for _ in 0..3 * limit {
                        row += 1;
                        if filler.is_taken(row) {
                            continue;
                        }
                        filler.offer(Candidate {
                            keys: [row.wrapping_mul(0x9e37_79b9) % 1_000, row, 0],
                            row,
                        });
                    }
                    filler.end(pool as u32);
                }
                filler.finish()
            });
            let share = Filler::share(total, widest);
            assert!(
                peak <= share,
                "filling {limits:?} allocated {peak} bytes and asks for {share}"
            );
            assert_eq!(slots.0.len() as u64, total);
            assert!(slots.1.windows(2).all(|pair| pair[0].0 < pair[1].0));
        }
    }

    /// The blocks of a pass are what their share says.
    #[test]
    fn the_blocks_of_a_pass_allocate_what_they_ask_for() {
        for (sets, scores) in [(1, 0), (8, 3), (wire::MAX_SETS + 1, 8)] {
            let (blocks, peak) = crate::test_heap::measure(|| Blocks::new(sets, scores));
            assert!(
                peak <= Blocks::bytes(sets, scores),
                "{sets} sets, {scores} score tables: {peak} bytes, {} asked",
                Blocks::bytes(sets, scores)
            );
            drop(blocks);
        }
    }

    #[test]
    fn a_score_key_puts_the_highest_score_first_and_none_last() {
        let ordered = [f32::INFINITY, 2.5, 1.0, 0.0, -0.0, -1.0, f32::NEG_INFINITY];
        let mut keys: Vec<u64> = [1.0f32, -1.0, 2.5, 0.0, -0.0]
            .iter()
            .map(|score| score_key(*score))
            .collect();
        keys.sort_unstable();
        let expected: Vec<u64> = [2.5f32, 1.0, 0.0, -0.0, -1.0]
            .iter()
            .map(|score| score_key(*score))
            .collect();
        assert_eq!(keys, expected);
        assert!(score_key(2.5) < score_key(1.0));
        assert_eq!(score_key(f32::NAN), u64::MAX);
        // Infinities are no scores either: they sort last, as a missing one.
        assert_eq!(score_key(ordered[0]), u64::MAX);
        for pair in [1.0f32, 0.5, 0.25].windows(2) {
            assert!(score_key(pair[0]) < score_key(pair[1]));
        }
    }

    #[test]
    fn the_seeded_key_is_the_policys_over_the_record_handle() {
        let mut id = [0u8; 18];
        assert_eq!(record_id_into(0x1_0000_00ff, &mut id), "r-00000001000000ff");
        assert_eq!(
            keys_of(
                &[OrderKey::SeededHash, OrderKey::RecordIdAscending],
                Some(5),
                0x1_0000_00ff,
                &[]
            ),
            [
                l2d_decide::policy::audit_key("r-00000001000000ff", 5),
                0x1_0000_00ff,
                0
            ]
        );
        // Another seed is another order.
        assert_ne!(
            l2d_decide::policy::audit_key("r-00000001000000ff", 5),
            l2d_decide::policy::audit_key("r-00000001000000ff", 6)
        );
        let ((), peak) = crate::test_heap::measure(|| {
            let mut id = [0u8; 18];
            std::hint::black_box(record_id_into(7, &mut id));
        });
        assert_eq!(peak, 0);
    }
}
