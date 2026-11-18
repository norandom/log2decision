//! Explicit, immutable analyst refinements. Membership, never templates, is authority.
use super::*;
use crate::selections::{ReviewAction, ReviewDecision, ReviewPreview};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Normal,
    Promoted,
    Hidden,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatternRefinement {
    pub format: u32,
    pub catalogue: PatternCatalogue,
    pub catalogue_revision: String,
    pub selection_revision: String,
    pub parent_revision: Option<String>,
    pub action: String,
    pub groups: Vec<Pattern>,
    pub priorities: BTreeMap<usize, Priority>,
    pub ungrouped: PatternCounts,
    pub rows: Vec<PatternMember>,
    pub revision: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueryPreview {
    pub refinement_revision: String,
    pub selection_revision: String,
    pub pattern: usize,
    pub regex: String,
    pub background_scope: String,
    pub member_matches: Vec<Member>,
    pub member_misses: Vec<Member>,
    pub background_matches: Vec<Member>,
    pub background_misses: Vec<Member>,
}

/// Atomic immutable publication; a collision never overwrites an existing artifact.
fn publish(value: &impl Serialize, path: &Path) -> Result<(), L2dError> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let parent = path
        .parent()
        .ok_or_else(|| fail("output needs a parent directory"))?;
    fs::create_dir_all(parent).map_err(fail)?;
    let temp = parent.join(format!(
        ".refinement-{}-{}.saving",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(fail)?;
        serde_json::to_writer(&mut file, value).map_err(fail)?;
        file.sync_all().map_err(fail)?;
        drop(file);
        fs::hard_link(&temp, path).map_err(fail)?;
        Ok(())
    })();
    let _ = fs::remove_file(temp);
    result
}
fn read<T: serde::de::DeserializeOwned>(path: &Path, budget: &MemoryBudget) -> Result<T, L2dError> {
    let size = fs::metadata(path).map_err(fail)?.len();
    budget.require_whole(
        size.saturating_mul(64).saturating_add(FIXED),
        "loading complete bounded refinement artifact",
    )?;
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(fail)?
        .take(size.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(fail)?;
    if bytes.len() as u64 != size {
        return Err(fail("artifact changed during read"));
    }
    serde_json::from_slice(&bytes).map_err(fail)
}
impl QueryPreview {
    pub fn held_bytes(&self) -> u64 {
        let lists = [
            &self.member_matches,
            &self.member_misses,
            &self.background_matches,
            &self.background_misses,
        ];
        self.regex.capacity() as u64
            + self.background_scope.capacity() as u64
            + 512
            + lists
                .iter()
                .map(|v| {
                    v.capacity() as u64 * 64
                        + v.iter()
                            .map(|m| match m {
                                Member::StoreId(s) => s.capacity() as u64,
                                _ => 0,
                            })
                            .sum::<u64>()
                })
                .sum::<u64>()
    }
    pub fn load(path: &Path, budget: &MemoryBudget) -> Result<Self, L2dError> {
        read(path, budget)
    }
}
impl PatternRefinement {
    /// Revalidate the original bytes, including the complete archive and its saved scope.
    pub fn require_current_input(&self) -> Result<(), L2dError> {
        match &self.catalogue.source {
            MarkedSource::Input(s) if hash_file(s.saved_location()?)? == s.content_sha256 => Ok(()),
            _ => Err(fail(
                "discovery input changed or is not a file/archive; search again",
            )),
        }
    }
    /// Unlike the selection ID hash, this snapshot also binds occurrence counts.
    pub fn require_current_store(
        &self,
        store: &dyn PatternStore,
        budget: &mut MemoryBudget,
    ) -> Result<(), L2dError> {
        self.reserve(budget)?;
        match &self.catalogue.source {
            MarkedSource::Store(s)
                if s.name == store.store_name()
                    && Some(store.pattern_snapshot(budget)?)
                        == self.catalogue.source_data_revision =>
            {
                Ok(())
            }
            _ => Err(fail(
                "discovery store identity/count snapshot changed; search again",
            )),
        }
    }
    pub fn review_saved_input(
        &self,
        selection: &mut SelectionSet,
        preview: ReviewPreview,
        action: ReviewAction,
        budget: &MemoryBudget,
    ) -> Result<(), L2dError> {
        self.require_selection(selection, budget)?;
        self.require_current_input()?;
        selection.apply_review(preview, action, budget)?;
        self.require_current_input()
    }
    pub fn from_catalogue(
        c: &PatternCatalogue,
        directory: &Path,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        let guard =
            ResourceGuard::tracked("create pattern refinement", busy_choice(), budget.usage())?;
        let size = fs::metadata(directory.join("members.jsonl"))
            .map_err(fail)?
            .len();
        let reserve = size
            .saturating_mul(64)
            .saturating_add(c.held_bytes().saturating_mul(4))
            .saturating_add(FIXED);
        budget.require_whole(
            reserve,
            "whole bounded refinement membership (no disk refinement form)",
        )?;
        let before = budget.held_bytes();
        budget.hold(before.saturating_add(reserve));
        let result = (|| {
            let mut rows = Vec::new();
            c.visit_members(directory, budget, |row| {
                rows.push(row);
                Ok(())
            })?;
            let mut r = Self {
                format: 1,
                catalogue: c.clone(),
                catalogue_revision: c.revision.clone(),
                selection_revision: c.selection_revision.clone(),
                parent_revision: None,
                action: "initial catalogue membership".into(),
                groups: c.patterns.clone(),
                priorities: c
                    .patterns
                    .iter()
                    .map(|p| (p.id, Priority::Normal))
                    .collect(),
                ungrouped: c.ungrouped.clone(),
                rows,
                revision: String::new(),
            };
            r.check_counts()?;
            r.revision = json_hash(&r)?;
            Ok(r)
        })();
        budget.hold(before);
        finish(guard, result)
    }
    pub fn held_bytes(&self) -> u64 {
        self.catalogue
            .held_bytes()
            .saturating_mul(3)
            .saturating_add(self.rows.capacity() as u64 * 256)
            .saturating_add(
                self.rows
                    .iter()
                    .map(|r| {
                        r.text.capacity() as u64 * 4
                            + match &r.member {
                                Member::StoreId(s) => s.capacity() as u64,
                                _ => 0,
                            }
                    })
                    .sum::<u64>(),
            )
            .saturating_add(
                self.groups
                    .iter()
                    .map(|g| {
                        g.template.capacity() as u64
                            + g.representative.capacity() as u64
                            + g.boundary_example.capacity() as u64
                            + g.variable_examples
                                .iter()
                                .chain(&g.differing_literals)
                                .map(|s| s.capacity() as u64 + 32)
                                .sum::<u64>()
                            + 256
                    })
                    .sum::<u64>(),
            )
            .saturating_add(self.action.capacity() as u64 + 4096)
    }
    fn reserve(&self, budget: &MemoryBudget) -> Result<(), L2dError> {
        budget.require_whole(
            self.held_bytes()
                .saturating_mul(12)
                .saturating_add(FIXED)
                .saturating_add(crate::marks::REGEX_HELD_BYTES),
            "refinement copies, summaries, query buffers and exact membership",
        )
    }
    fn check_counts(&self) -> Result<(), L2dError> {
        if self.rows.len() as u64 != self.catalogue.selected_members
            || self.rows.windows(2).any(|p| p[0].member >= p[1].member)
            || self.groups.windows(2).any(|p| p[0].id >= p[1].id)
            || self
                .groups
                .iter()
                .any(|g| !self.priorities.contains_key(&g.id))
            || self.priorities.len() != self.groups.len()
        {
            return Err(fail("invalid refinement membership/order/priority"));
        }
        let mut counts: BTreeMap<usize, PatternCounts> = self
            .groups
            .iter()
            .map(|p| (p.id, PatternCounts::default()))
            .collect();
        let mut ungrouped = PatternCounts::default();
        let mut support = Support::default();
        for row in &self.rows {
            if row.occurrences == 0 || row.text.len() > self.catalogue.settings.max_text_bytes {
                return Err(fail("invalid refinement text/count"));
            }
            let target = match row.pattern {
                Some(id) => counts
                    .get_mut(&id)
                    .ok_or_else(|| fail("unknown refinement group"))?,
                None => &mut ungrouped,
            };
            support.add(&row.text, row.pattern, row.review, target, row.occurrences)?;
        }
        if ungrouped != self.ungrouped
            || self
                .groups
                .iter()
                .any(|g| counts.get(&g.id) != Some(&g.counts))
        {
            return Err(fail("refinement counts differ from exact membership"));
        }
        Ok(())
    }
    fn validate(&self, budget: &MemoryBudget) -> Result<(), L2dError> {
        self.reserve(budget)?;
        self.catalogue.settings.validate()?;
        if self.format != 1 || self.catalogue_revision != self.catalogue.revision {
            return Err(fail("unsupported refinement or catalogue identity"));
        }
        if self.catalogue.format != 1
            || self.catalogue.algorithm_revision != ALGORITHM_REVISION
            || self.catalogue.patterns.len() > self.catalogue.settings.max_patterns
            || self
                .catalogue
                .patterns
                .iter()
                .enumerate()
                .any(|(i, p)| p.id != i)
        {
            return Err(fail("invalid source catalogue contract"));
        }
        let mut c = self.catalogue.clone();
        let revision = std::mem::take(&mut c.revision);
        if json_hash(&c)? != revision {
            return Err(fail("catalogue revision mismatch"));
        }
        let mut r = self.clone();
        let revision = std::mem::take(&mut r.revision);
        if json_hash(&r)? != revision {
            return Err(fail("refinement revision mismatch"));
        }
        self.check_counts()
    }
    pub fn save(&self, path: &Path, budget: &MemoryBudget) -> Result<(), L2dError> {
        let guard =
            ResourceGuard::tracked("save pattern refinement", busy_choice(), budget.usage())?;
        let result = self.validate(budget).and_then(|_| publish(self, path));
        finish(guard, result)
    }
    pub fn load(path: &Path, budget: &MemoryBudget) -> Result<Self, L2dError> {
        let guard =
            ResourceGuard::tracked("load pattern refinement", busy_choice(), budget.usage())?;
        let result = (|| {
            let r: Self = read(path, budget)?;
            r.validate(budget)?;
            Ok(r)
        })();
        finish(guard, result)
    }
    fn group(&self, id: usize) -> Result<&Pattern, L2dError> {
        self.groups
            .iter()
            .find(|p| p.id == id)
            .ok_or_else(|| fail("unknown pattern id"))
    }
    fn revised(&self, action: String, budget: &MemoryBudget) -> Result<Self, L2dError> {
        self.validate(budget)?;
        budget.require_whole(
            self.held_bytes()
                .saturating_mul(12)
                .saturating_add(action.len() as u64 * 4)
                .saturating_add(FIXED),
            "refinement action and lineage",
        )?;
        let mut next = self.clone();
        next.parent_revision = Some(self.revision.clone());
        next.action = action;
        next.revision.clear();
        Ok(next)
    }
    fn summarize(&mut self) -> Result<(), L2dError> {
        let ids: BTreeSet<usize> = self.rows.iter().filter_map(|r| r.pattern).collect();
        self.groups.clear();
        self.priorities.retain(|id, _| ids.contains(id));
        for id in ids {
            let mut members = self.rows.iter().filter(|r| r.pattern == Some(id));
            let first = members.next().ok_or_else(|| fail("empty pattern"))?;
            let mut words = tokens(&first.text);
            let mut p = Pattern {
                id,
                template: first.text.clone(),
                representative: first.text.clone(),
                variable_examples: Vec::new(),
                boundary_example: first.text.clone(),
                minimum_similarity_per_mille: 1000,
                differing_literals: Vec::new(),
                possible_outcome_collision: false,
                counts: PatternCounts::default(),
            };
            let mut variable_structure = false;
            for row in members {
                if row.text != p.representative
                    && p.variable_examples.len() < 2
                    && !p.variable_examples.contains(&row.text)
                {
                    p.variable_examples.push(row.text.clone());
                }
                let other = tokens(&row.text);
                let score = similarity(
                    PatternEngine::DrainStyle,
                    &tokens(&p.representative),
                    &other,
                );
                if score < p.minimum_similarity_per_mille {
                    p.minimum_similarity_per_mille = score;
                    p.boundary_example = row.text.clone();
                }
                if words.len() != other.len() {
                    variable_structure = true;
                    p.possible_outcome_collision = true;
                }
                for (a, b) in words.iter_mut().zip(&other) {
                    if a != b {
                        p.possible_outcome_collision = true;
                        if p.differing_literals.len() < 8 {
                            p.differing_literals.push(format!("{a} -> {b}"));
                        }
                        *a = "<*>".into();
                    }
                }
            }
            p.template = if variable_structure {
                "[variable structure; inspect exact members]".into()
            } else {
                words.concat()
            };
            self.priorities.entry(id).or_insert(Priority::Normal);
            self.groups.push(p);
        }
        self.ungrouped = PatternCounts::default();
        let mut support = Support::default();
        for row in &self.rows {
            let counts = match row.pattern {
                Some(id) => {
                    &mut self
                        .groups
                        .iter_mut()
                        .find(|p| p.id == id)
                        .ok_or_else(|| fail("missing group"))?
                        .counts
                }
                None => &mut self.ungrouped,
            };
            support.add(&row.text, row.pattern, row.review, counts, row.occurrences)?;
        }
        self.revision = json_hash(self)?;
        Ok(())
    }
    pub fn split(&self, id: usize, pattern: &str, budget: &MemoryBudget) -> Result<Self, L2dError> {
        let guard = ResourceGuard::tracked("split pattern", busy_choice(), budget.usage())?;
        self.group(id)?;
        self.reserve(budget)?;
        let matcher = matcher(pattern, budget)?;
        let mut r = self.revised(format!("split pattern {id} by regex {pattern}"), budget)?;
        let next = self
            .groups
            .iter()
            .map(|p| p.id)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| fail("pattern id overflow"))?;
        let mut yes = 0;
        let mut no = 0;
        for row in &mut r.rows {
            if row.pattern == Some(id) {
                if matcher.is_match(&row.text) {
                    row.pattern = Some(next);
                    yes += 1;
                } else {
                    no += 1;
                }
            }
        }
        if yes == 0 || no == 0 {
            return Err(fail("split must leave two nonempty groups"));
        }
        r.priorities.insert(next, self.priorities[&id]);
        r.summarize()?;
        finish(guard, Ok(r))
    }
    pub fn merge(&self, ids: &[usize], budget: &MemoryBudget) -> Result<Self, L2dError> {
        let guard = ResourceGuard::tracked("merge patterns", busy_choice(), budget.usage())?;
        if ids.len() < 2 || ids.len() > self.groups.len() {
            return Err(fail("merge needs at least two distinct groups"));
        }
        for (i, id) in ids.iter().enumerate() {
            self.group(*id)?;
            if ids[..i].contains(id) {
                return Err(fail("duplicate merge group"));
            }
        }
        let target = *ids.iter().min().ok_or_else(|| fail("empty merge"))?;
        let mut r = self.revised(
            format!("merge patterns {ids:?}; priority from pattern {target}"),
            budget,
        )?;
        for row in &mut r.rows {
            if row.pattern.is_some_and(|id| ids.contains(&id)) {
                row.pattern = Some(target);
            }
        }
        r.summarize()?;
        finish(guard, Ok(r))
    }
    pub fn prioritize(
        &self,
        id: usize,
        priority: Priority,
        budget: &MemoryBudget,
    ) -> Result<Self, L2dError> {
        let guard = ResourceGuard::tracked("pattern priority", busy_choice(), budget.usage())?;
        self.group(id)?;
        let mut r = self.revised(
            format!("priority pattern {id}: {priority:?}; labels unchanged"),
            budget,
        )?;
        r.priorities.insert(id, priority);
        r.revision = json_hash(&r)?;
        finish(guard, Ok(r))
    }
    pub fn proposed_regex(&self, id: usize, budget: &MemoryBudget) -> Result<String, L2dError> {
        let guard = ResourceGuard::tracked("propose pattern query", busy_choice(), budget.usage())?;
        self.group(id)?;
        self.reserve(budget)?;
        let mut rows = self.rows.iter().filter(|r| r.pattern == Some(id));
        let first = rows.next().ok_or_else(|| fail("empty pattern"))?;
        let literal = tokens(&first.text);
        let masked = masked(&first.text, &self.catalogue.settings.masks);
        let mut variable: Vec<bool> = literal.iter().zip(&masked).map(|(a, b)| a != b).collect();
        for row in rows {
            let other = tokens(&row.text);
            if other.len() != literal.len() {
                return Err(fail(
                    "variable token structure: provide an explicit regex to preview",
                ));
            }
            for ((slot, a), b) in variable.iter_mut().zip(&literal).zip(&other) {
                *slot |= a != b;
            }
        }
        let mut query = String::from(r"\A(?:");
        for (token, variable) in literal.iter().zip(variable) {
            if variable {
                query.push_str("(?s:.*?)");
            } else {
                query.push_str(&regex::escape(token));
            }
        }
        query.push_str(r")\z");
        finish(guard, Ok(query))
    }
    pub fn preview_query(
        &self,
        id: usize,
        query: &str,
        budget: &MemoryBudget,
    ) -> Result<QueryPreview, L2dError> {
        let guard = ResourceGuard::tracked("preview pattern query", busy_choice(), budget.usage())?;
        self.group(id)?;
        self.validate(budget)?;
        let matcher = matcher(query, budget)?;
        let mut p = QueryPreview { refinement_revision:self.revision.clone(), selection_revision:self.selection_revision.clone(), pattern:id, regex:query.into(), background_scope:"all other members of this saved selection, including ungrouped; no outside-source sample or corpus precision claim".into(), member_matches:Vec::new(), member_misses:Vec::new(), background_matches:Vec::new(), background_misses:Vec::new() };
        for row in &self.rows {
            match (row.pattern == Some(id), matcher.is_match(&row.text)) {
                (true, true) => &mut p.member_matches,
                (true, false) => &mut p.member_misses,
                (false, true) => &mut p.background_matches,
                (false, false) => &mut p.background_misses,
            }
            .push(row.member.clone());
        }
        finish(guard, Ok(p))
    }
    pub fn save_query(
        &self,
        preview: &QueryPreview,
        path: &Path,
        budget: &MemoryBudget,
    ) -> Result<(), L2dError> {
        let guard = ResourceGuard::tracked("save pattern query", busy_choice(), budget.usage())?;
        if self.preview_query(preview.pattern, &preview.regex, budget)? != *preview {
            return Err(fail("query preview is stale or altered; preview again"));
        }
        finish(guard, publish(preview, path))
    }
    pub fn require_selection(
        &self,
        selection: &SelectionSet,
        budget: &MemoryBudget,
    ) -> Result<(), L2dError> {
        self.validate(budget)?;
        if selection.revision(budget)? != self.selection_revision
            || selection.source() != &self.catalogue.source
            || selection.members().len() != self.rows.len()
            || selection
                .members()
                .iter()
                .zip(&self.rows)
                .any(|(m, r)| m != &r.member)
            || self.rows.iter().any(|row| {
                row.review
                    != selection.review().and_then(|review| {
                        review
                            .decisions
                            .binary_search_by(|d| d.member.cmp(&row.member))
                            .ok()
                            .map(|i| review.decisions[i].state)
                    })
            })
        {
            return Err(fail(
                "refinement selection/source/revision/membership mismatch",
            ));
        }
        Ok(())
    }
    /// The external preview names only target rows; the applied preview carries unchanged others.
    pub fn review_revision(
        &self,
        selection: &SelectionSet,
        id: usize,
        state: ReviewState,
        action: &ReviewAction,
        budget: &MemoryBudget,
    ) -> Result<String, L2dError> {
        self.group(id)?;
        self.require_selection(selection, budget)?;
        if action.analyst.trim().is_empty() || action.action.trim().is_empty() {
            return Err(fail("review needs a named analyst and action"));
        }
        budget.require_whole(
            (action.analyst.len() as u64)
                .saturating_add(action.action.len() as u64)
                .saturating_mul(4)
                .saturating_add(FIXED),
            "binding explicit review preview",
        )?;
        json_hash(&(
            "pattern-review-v1",
            &self.revision,
            &self.selection_revision,
            id,
            state,
            action,
        ))
    }
    pub fn preview_review(
        &self,
        selection: &SelectionSet,
        id: usize,
        state: ReviewState,
        budget: &MemoryBudget,
    ) -> Result<ReviewPreview, L2dError> {
        self.group(id)?;
        self.require_selection(selection, budget)?;
        budget.require_whole(
            self.held_bytes()
                .saturating_mul(4)
                .saturating_add(selection.held_bytes().saturating_mul(4))
                .saturating_add(FIXED),
            "partial review preserving all prior decisions",
        )?;
        let decisions: Vec<ReviewDecision> = self
            .rows
            .iter()
            .map(|row| ReviewDecision {
                member: row.member.clone(),
                state: if row.pattern == Some(id) {
                    state
                } else {
                    selection
                        .review()
                        .and_then(|r| {
                            r.decisions
                                .binary_search_by(|d| d.member.cmp(&row.member))
                                .ok()
                                .map(|i| r.decisions[i].state)
                        })
                        .unwrap_or(ReviewState::Unresolved)
                },
            })
            .collect();
        selection.preview_review(&decisions, budget)
    }
    pub fn with_review(
        &self,
        reviewed: &SelectionSet,
        budget: &MemoryBudget,
    ) -> Result<Self, L2dError> {
        let receipt = reviewed
            .review()
            .ok_or_else(|| fail("explicit review receipt required"))?;
        if receipt.selection_revision != self.selection_revision
            || reviewed.source() != &self.catalogue.source
            || reviewed.members().len() != self.rows.len()
            || reviewed
                .members()
                .iter()
                .zip(&self.rows)
                .any(|(m, r)| m != &r.member)
        {
            return Err(fail("review lineage/source/membership mismatch"));
        }
        let mut r = self.revised(
            format!(
                "explicit review by {}: {}",
                receipt.action.analyst, receipt.action.action
            ),
            budget,
        )?;
        for row in &mut r.rows {
            row.review = receipt
                .decisions
                .binary_search_by(|d| d.member.cmp(&row.member))
                .ok()
                .map(|i| receipt.decisions[i].state);
        }
        r.selection_revision = reviewed.revision(budget)?;
        r.summarize()?;
        Ok(r)
    }
}
fn matcher(query: &str, budget: &MemoryBudget) -> Result<regex::Regex, L2dError> {
    budget.require_whole(
        crate::marks::REGEX_HELD_BYTES.saturating_add(query.len() as u64 * 32),
        "bounded refinement regex",
    )?;
    regex::RegexBuilder::new(query)
        .size_limit(crate::marks::REGEX_SIZE_LIMIT)
        .dfa_size_limit(crate::marks::REGEX_DFA_SIZE_LIMIT)
        .build()
        .map_err(|e| L2dError::InvalidInput {
            detail: format!("invalid discovery regex: {e}"),
        })
}
