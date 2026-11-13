//! Folds of group keys: all logs of a group key are on one side of every split
//! (requirement 9.2).
//!
//! A split is made in two steps, and neither looks at a label, a text or a ratio:
//!
//! 1. [`units`] counts the logs of every group key and merges the group keys that must
//!    stay together into one [`Unit`] (the two CSIC endpoints with the same form, for
//!    instance). Units come out in byte order of their key, so the order of the rows
//!    plays no part.
//! 2. A rule deals the units into folds: [`Folds::balanced`], [`Folds::dealt`] or
//!    [`Folds::leave_one_out`]. A fold is the set of group keys it holds out.
//!
//! The rules are deterministic: the same group keys with the same numbers of logs give
//! the same folds, on any host and in any row order. There is no random number.
//!
//! The caller decides what a group key is. For HTTP CSIC 2010 it is the URL path alone
//! (decision of 2026-11-11); a log without a group key is its own group
//! ([`crate::evaluate::EvalLogs`] gives it a key of its own).

use std::collections::{BTreeMap, BTreeSet};

use l2d_core::L2dError;
use sha2::{Digest, Sha256};

/// What a fold is dealt: one group key, or several that must stay on the same side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    /// The first of `groups` in byte order; names the unit.
    pub key: String,
    /// The group keys of the unit, in byte order.
    pub groups: Vec<String>,
    /// The number of logs (rows) of all its groups.
    pub logs: u64,
}

fn invalid(detail: String) -> L2dError {
    L2dError::InvalidInput { detail }
}

/// The units of `groups`, the group key of every log, in byte order of the unit keys.
///
/// `together` lists sets of group keys that are kept on one side of every split: each
/// set becomes one unit. A key of a set that no log has is ignored, so the same sets
/// can be used on a part of the logs. A group key named in two sets is
/// [`L2dError::InvalidInput`].
pub fn units<S: AsRef<str>>(groups: &[S], together: &[Vec<String>]) -> Result<Vec<Unit>, L2dError> {
    let mut logs: BTreeMap<&str, u64> = BTreeMap::new();
    for group in groups {
        *logs.entry(group.as_ref()).or_default() += 1;
    }
    // The set each named group key belongs to.
    let mut set_of: BTreeMap<&str, usize> = BTreeMap::new();
    for (index, set) in together.iter().enumerate() {
        for key in set {
            match set_of.insert(key.as_str(), index) {
                Some(other) if other != index => {
                    return Err(invalid(format!(
                        "the group key '{key}' is named in two sets of groups that stay together"
                    )));
                }
                _ => {}
            }
        }
    }
    let mut merged: BTreeMap<usize, Unit> = BTreeMap::new();
    let mut found = Vec::new();
    // `logs` is walked in byte order, so the first key met of a set is its smallest.
    for (&group, &count) in &logs {
        match set_of.get(group) {
            Some(&set) => {
                let unit = merged.entry(set).or_insert_with(|| Unit {
                    key: group.to_string(),
                    groups: Vec::new(),
                    logs: 0,
                });
                unit.groups.push(group.to_string());
                unit.logs += count;
            }
            None => found.push(Unit {
                key: group.to_string(),
                groups: vec![group.to_string()],
                logs: count,
            }),
        }
    }
    found.extend(merged.into_values());
    found.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(found)
}

/// Folds: for each fold the group keys it holds out. Everything else is its fitting
/// side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Folds {
    held_out: Vec<BTreeSet<String>>,
}

/// Units from the largest to the smallest, ties in byte order of the key.
fn largest_first<'a>(units: impl Iterator<Item = &'a Unit>) -> Vec<&'a Unit> {
    let mut sorted: Vec<&Unit> = units.collect();
    sorted.sort_by(|a, b| b.logs.cmp(&a.logs).then_with(|| a.key.cmp(&b.key)));
    sorted
}

fn check_count(units: &[Unit], folds: usize) -> Result<(), L2dError> {
    if folds < 2 {
        return Err(invalid(format!(
            "a grouped split needs at least 2 folds, not {folds}"
        )));
    }
    if units.len() < folds {
        return Err(invalid(format!(
            "{folds} folds need at least {folds} groups (or sets of groups that stay \
             together); there are {}",
            units.len()
        )));
    }
    Ok(())
}

impl Folds {
    fn from_parts(held_out: Vec<BTreeSet<String>>) -> Result<Self, L2dError> {
        if let Some(empty) = held_out.iter().position(BTreeSet::is_empty) {
            return Err(invalid(format!(
                "fold {empty} would hold no group; use fewer folds"
            )));
        }
        Ok(Self { held_out })
    }

    /// Folds of nearly equal numbers of logs.
    ///
    /// The units are taken from the largest to the smallest (ties in byte order of the
    /// key) and each goes to the fold that holds the fewest logs so far, the lowest
    /// fold on a tie. This is the rule of scikit-learn's `GroupKFold`.
    ///
    /// Fewer than 2 folds, or fewer units than folds, is [`L2dError::InvalidInput`].
    pub fn balanced(units: &[Unit], folds: usize) -> Result<Self, L2dError> {
        check_count(units, folds)?;
        let mut held_out = vec![BTreeSet::new(); folds];
        let mut logs = vec![0_u64; folds];
        for unit in largest_first(units.iter()) {
            let fold = (0..folds)
                .min_by_key(|&fold| (logs[fold], fold))
                .expect("there are at least 2 folds");
            logs[fold] += unit.logs;
            held_out[fold].extend(unit.groups.iter().cloned());
        }
        Self::from_parts(held_out)
    }

    /// Folds dealt in turn, the rule of the science gate (task 2.1,
    /// `tools/science/l2dsci.py`, `cv5_fold_of_group`).
    ///
    /// The units that hold a group key of `first` are taken from the largest to the
    /// smallest (ties in byte order of the key) and dealt in turn to fold 0, 1, 2, ...;
    /// then the other units are taken in byte order of the key and dealt in turn, from
    /// fold 0 again. With the group keys that hold both labels as `first`, each fold
    /// gets one of the large endpoints.
    ///
    /// Fewer than 2 folds, fewer units than folds, or a fold left without a group is
    /// [`L2dError::InvalidInput`].
    pub fn dealt(units: &[Unit], folds: usize, first: &BTreeSet<String>) -> Result<Self, L2dError> {
        check_count(units, folds)?;
        let is_first = |unit: &Unit| unit.groups.iter().any(|group| first.contains(group));
        let mut held_out = vec![BTreeSet::new(); folds];
        let leading = largest_first(units.iter().filter(|unit| is_first(unit)));
        for (index, unit) in leading.into_iter().enumerate() {
            held_out[index % folds].extend(unit.groups.iter().cloned());
        }
        // `units` may come in any order; the rest is dealt in byte order of the key.
        let mut rest: Vec<&Unit> = units.iter().filter(|unit| !is_first(unit)).collect();
        rest.sort_by(|a, b| a.key.cmp(&b.key));
        for (index, unit) in rest.into_iter().enumerate() {
            held_out[index % folds].extend(unit.groups.iter().cloned());
        }
        Self::from_parts(held_out)
    }

    /// One fold per key of `keys`, in their order: the fold holds out the unit of that
    /// key and nothing else. Units that are not named are on the fitting side of every
    /// fold.
    ///
    /// No key, a key that is not the key of a unit, or a key named twice is
    /// [`L2dError::InvalidInput`].
    pub fn leave_one_out(units: &[Unit], keys: &[String]) -> Result<Self, L2dError> {
        if keys.is_empty() {
            return Err(invalid("no group was named to be left out".into()));
        }
        let mut seen = BTreeSet::new();
        let mut held_out = Vec::with_capacity(keys.len());
        for key in keys {
            let unit = units.iter().find(|unit| &unit.key == key).ok_or_else(|| {
                invalid(format!(
                    "'{key}' is not a group key of the logs (or not the first key of a set \
                     of groups that stay together)"
                ))
            })?;
            if !seen.insert(key) {
                return Err(invalid(format!("the group key '{key}' is named twice")));
            }
            held_out.push(unit.groups.iter().cloned().collect());
        }
        Self::from_parts(held_out)
    }

    /// The number of folds.
    pub fn len(&self) -> usize {
        self.held_out.len()
    }

    pub fn is_empty(&self) -> bool {
        self.held_out.is_empty()
    }

    /// The group keys fold `fold` holds out.
    pub fn held_out(&self, fold: usize) -> &BTreeSet<String> {
        &self.held_out[fold]
    }

    /// The fold that holds `group` out; `None` when no fold does.
    pub fn fold_of(&self, group: &str) -> Option<usize> {
        self.held_out.iter().position(|held| held.contains(group))
    }

    /// For every log, by its group key: whether fold `fold` holds it out.
    pub fn mask<S: AsRef<str>>(&self, fold: usize, groups: &[S]) -> Vec<bool> {
        let held = &self.held_out[fold];
        groups
            .iter()
            .map(|group| held.contains(group.as_ref()))
            .collect()
    }

    /// The lower-case hexadecimal SHA-256 of the held-out group keys of fold `fold`, in
    /// byte order, joined by line feeds: the figure `held_out_paths_sha256` of the
    /// science gate's run records, so two splits can be compared without their lists.
    pub fn membership_sha256(&self, fold: usize) -> String {
        let mut hasher = Sha256::new();
        for (index, group) in self.held_out[fold].iter().enumerate() {
            if index > 0 {
                hasher.update(b"\n");
            }
            hasher.update(group.as_bytes());
        }
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}
