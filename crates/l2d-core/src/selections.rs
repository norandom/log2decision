//! Unlabelled bounded discovery selections, with explicit review receipts.
//! Existing objects/frames belong in the caller's budget; all new buffers are
//! reserved before allocation. A member cap refuses overflow, never truncates.
use crate::polars::prelude::Column;
use crate::resource::{ResourceGuard, busy_choice};
use crate::{
    ArchiveSummary, InputSummary, L2dError, Label, LogsFrame, MarkQuery, MarkSet, MarkStore,
    MarkedInput, MarkedSource, MarkedStore, MemoryBudget, ZipInput, ZipInputSettings,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};
const FIXED: u64 = 64 * 1024;
const MEMBER_BYTES: u64 = 128;
const SAVE_BUFFER: u64 = 64 * 1024;

/// Literal/regex search anywhere; glob matches the whole record with Unicode
/// characters and backslash escapes. FTS5Trigram is the store's intrinsically
/// insensitive SQLite expression language, not word tokens or substring search.
/// Unquoted prefix expansion (`*`) is explicitly refused because its native
/// readers cannot be bounded by expression length; quoted literal stars work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum SearchQuery {
    Literal { text: String, insensitive: bool },
    Glob { pattern: String, insensitive: bool },
    Regex { pattern: String, insensitive: bool },
    Fts5Trigram { expression: String },
}
impl SearchQuery {
    pub fn literal(text: impl Into<String>) -> Self {
        Self::Literal {
            text: text.into(),
            insensitive: false,
        }
    }
    pub fn glob(pattern: impl Into<String>) -> Self {
        Self::Glob {
            pattern: pattern.into(),
            insensitive: false,
        }
    }
    pub fn regex(pattern: impl Into<String>) -> Self {
        Self::Regex {
            pattern: pattern.into(),
            insensitive: false,
        }
    }
    fn text(&self) -> &str {
        match self {
            Self::Literal { text, .. } => text,
            Self::Glob { pattern, .. } | Self::Regex { pattern, .. } => pattern,
            Self::Fts5Trigram { expression } => expression,
        }
    }
    fn check(&self) -> Result<(), L2dError> {
        let maximum = if matches!(self, Self::Fts5Trigram { .. }) {
            4096
        } else {
            crate::marks::MAX_PATTERN_BYTES
        };
        if self.text().is_empty() || self.text().len() > maximum {
            return Err(invalid(
                "search text must have 1 to 65536 bytes (FTS5: at most 4096)",
            ));
        }
        Ok(())
    }
}
fn invalid(s: &str) -> L2dError {
    L2dError::InvalidInput { detail: s.into() }
}
fn refused(s: &str) -> L2dError {
    L2dError::Refused { reason: s.into() }
}
fn failed(e: impl std::fmt::Display) -> L2dError {
    L2dError::Runtime {
        detail: format!("selection: {e}"),
    }
}
/// Compile a selection with the same matching rules and resource limits as discovery.
pub fn compile(query: &SearchQuery, budget: &MemoryBudget) -> Result<regex::Regex, L2dError> {
    query.check()?;
    budget.require_whole(
        crate::marks::REGEX_HELD_BYTES + 16 * query.text().len() as u64,
        "compiling selection query",
    )?;
    let (pattern, insensitive) = match query {
        SearchQuery::Literal { text, insensitive } => (regex::escape(text), *insensitive),
        SearchQuery::Regex {
            pattern,
            insensitive,
        } => (pattern.clone(), *insensitive),
        SearchQuery::Glob {
            pattern,
            insensitive,
        } => {
            let mut out = String::with_capacity(pattern.len() * 4 + 16);
            out.push_str(r"\A(?s:");
            let mut chars = pattern.chars();
            while let Some(ch) = chars.next() {
                match ch {
                    '*' => out.push_str(".*"),
                    '?' => out.push('.'),
                    '\\' => {
                        let escaped = chars
                            .next()
                            .ok_or_else(|| invalid("content glob ends with an escape"))?;
                        out.push_str(&regex::escape(escaped.encode_utf8(&mut [0; 4])));
                    }
                    other => out.push_str(&regex::escape(other.encode_utf8(&mut [0; 4]))),
                }
            }
            out.push_str(r")\z");
            (out, *insensitive)
        }
        SearchQuery::Fts5Trigram { .. } => {
            return Err(invalid("FTS5 trigram search requires a store"));
        }
    };
    regex::RegexBuilder::new(&pattern)
        .case_insensitive(insensitive)
        .size_limit(crate::marks::REGEX_SIZE_LIMIT)
        .dfa_size_limit(crate::marks::REGEX_DFA_SIZE_LIMIT)
        .build()
        .map_err(|e| invalid(&format!("invalid selection query: {e}")))
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchSettings {
    pub max_members: u64,
}
impl Default for SearchSettings {
    fn default() -> Self {
        Self {
            max_members: 100_000,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Member {
    Position(u64),
    StoreId(String),
}
/// Counts refer to the selected source scope. Skipped records and unsupported
/// archive entries make completeness false; entry-filter exclusions are outside scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Completeness {
    pub complete: bool,
    pub records: Option<u64>,
    /// Distinct source members scanned: distinct texts for input files/archives,
    /// distinct log IDs for stores. Store IDs include source, so repeated text from
    /// two sources counts twice here. This is not independent support or a store
    /// distinct-text count; catalogues must deduplicate text separately.
    pub distinct_scanned: u64,
    pub matched: u64,
    pub skipped_records: u64,
    pub omitted_entries: u64,
}
/// Returns total stored rows, streaming matching IDs under the budget.
pub trait SelectionStore: MarkStore {
    fn fts5_trigram_ids(
        &self,
        expression: &str,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&str) -> Result<(), L2dError>,
    ) -> Result<u64, L2dError>;
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewState {
    Good,
    Bad,
    Unresolved,
    Conflict,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewDecision {
    pub member: Member,
    pub state: ReviewState,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewAction {
    pub analyst: String,
    pub action: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewRecord {
    pub selection_revision: String,
    pub action: ReviewAction,
    pub decisions: Vec<ReviewDecision>,
}
/// Only preview_review constructs this revision-bound list. Unspecified members
/// are explicitly unresolved; applying it requires a fresh source check.
#[derive(Debug)]
pub struct ReviewPreview {
    revision: String,
    decisions: Vec<ReviewDecision>,
}
impl ReviewPreview {
    pub fn affected(&self) -> usize {
        self.decisions.len()
    }
    pub fn decisions(&self) -> &[ReviewDecision] {
        &self.decisions
    }
    pub fn selection_revision(&self) -> &str {
        &self.revision
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionSet {
    format: u32,
    kind: String,
    name: String,
    query: SearchQuery,
    settings: SearchSettings,
    source: MarkedSource,
    completeness: Completeness,
    members: Vec<Member>,
    review: Option<ReviewRecord>,
}
fn source_bytes(source: &MarkedSource) -> u64 {
    match source {
        MarkedSource::Input(s) => {
            s.path.len() as u64
                + s.content_sha256.len() as u64
                + s.archive_entries
                    .as_ref()
                    .map_or(0, |e| e.iter().map(|s| s.len() as u64 + 32).sum())
        }
        MarkedSource::Store(s) => s.name.len() as u64 + s.content_sha256.len() as u64,
    }
}
fn input_bytes(summary: &InputSummary, entries: Option<&[String]>) -> u64 {
    summary.source_path.as_os_str().len() as u64 * 4
        + summary.content_sha256.len() as u64
        + entries.map_or(0, |e| e.iter().map(|s| s.len() as u64 + 32).sum())
}
fn input_source(
    summary: &InputSummary,
    logs: &LogsFrame,
    budget: &MemoryBudget,
) -> Result<MarkedSource, L2dError> {
    Ok(MarkedSource::Input(MarkedInput {
        path: MarkedInput::source_location(&summary.source_path)?,
        content_sha256: summary.content_sha256.clone(),
        archive_entries: None,
        interpretation: Some(crate::marks::InputInterpretation::file(logs, budget)?),
    }))
}
fn check_name(name: &str) -> Result<(), L2dError> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(invalid(
            "selection name must be 1 to 64 ASCII letters, digits, '-' or '_'",
        ));
    }
    Ok(())
}
fn hash_ok(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl SelectionSet {
    fn reservation(
        name: &str,
        query: &SearchQuery,
        settings: SearchSettings,
        metadata: u64,
        budget: &MemoryBudget,
    ) -> Result<u64, L2dError> {
        check_name(name)?;
        query.check()?;
        if settings.max_members == 0 {
            return Err(invalid("max_members must be positive"));
        }
        let bytes = settings
            .max_members
            .checked_mul(MEMBER_BYTES)
            .and_then(|v| v.checked_add(metadata))
            .and_then(|v| {
                v.checked_add(
                    FIXED + crate::marks::REGEX_HELD_BYTES + 16 * query.text().len() as u64,
                )
            })
            .ok_or_else(|| refused("selection reservation overflow; lower max_members"))?;
        usize::try_from(settings.max_members)
            .map_err(|_| refused("max_members exceeds host capacity"))?;
        budget.require_whole(
            bytes,
            "reserving selection metadata, members and matcher before allocation",
        )?;
        Ok(bytes)
    }
    fn empty(
        name: &str,
        query: SearchQuery,
        settings: SearchSettings,
        source: MarkedSource,
    ) -> Self {
        Self {
            format: 1,
            kind: "selection-set".into(),
            name: name.into(),
            query,
            settings,
            source,
            completeness: Completeness {
                complete: true,
                records: None,
                distinct_scanned: 0,
                matched: 0,
                skipped_records: 0,
                omitted_entries: 0,
            },
            members: Vec::with_capacity(settings.max_members as usize),
            review: None,
        }
    }
    fn room(&self) -> Result<(), L2dError> {
        if self.members.len() as u64 == self.settings.max_members {
            return Err(refused(
                "selection exceeds max_members; no partial selection was returned or saved",
            ));
        }
        Ok(())
    }
    fn take(&mut self, logs: &LogsFrame, matcher: &regex::Regex) -> Result<(), L2dError> {
        let texts = logs
            .frame()
            .column(LogsFrame::TEXT)
            .and_then(Column::str)
            .map_err(failed)?;
        let positions = logs
            .frame()
            .column(LogsFrame::FIRST_POSITION)
            .and_then(Column::u64)
            .map_err(failed)?;
        for (text, position) in texts.into_no_null_iter().zip(positions.into_no_null_iter()) {
            self.completeness.distinct_scanned += 1;
            if matcher.is_match(text) {
                self.room()?;
                self.members.push(Member::Position(position));
            }
        }
        Ok(())
    }
    fn finish(&mut self) {
        self.members.sort_unstable();
        self.members.dedup();
        self.completeness.matched = self.members.len() as u64;
    }
    fn counts(&mut self, summary: &InputSummary) {
        self.completeness.records = Some(summary.records);
        self.completeness.skipped_records = summary.skipped.len() as u64;
        self.completeness.complete =
            self.completeness.skipped_records == 0 && self.completeness.omitted_entries == 0;
    }
    pub fn search_input(
        name: &str,
        query: SearchQuery,
        settings: SearchSettings,
        logs: &LogsFrame,
        summary: &InputSummary,
        budget: &MemoryBudget,
    ) -> Result<Self, L2dError> {
        let guard =
            ResourceGuard::tracked("selection input search", busy_choice(), budget.usage())?;
        Self::reservation(
            name,
            &query,
            settings,
            input_bytes(summary, None) * 16,
            budget,
        )?;
        if !hash_ok(&summary.content_sha256) {
            return Err(invalid("input summary has an invalid SHA-256"));
        }
        if logs.len() as u64 != summary.distinct_texts {
            return Err(invalid(
                "input frame does not cover the summary's distinct texts",
            ));
        }
        let source = input_source(summary, logs, budget)?;
        let matcher = compile(&query, budget)?;
        let mut set = Self::empty(name, query, settings, source);
        set.take(logs, &matcher)?;
        set.counts(summary);
        set.finish();
        guard.finish()?;
        Ok(set)
    }
    pub fn search_archive(
        name: &str,
        query: SearchQuery,
        settings: SearchSettings,
        path: &Path,
        options: &ZipInputSettings,
        budget: &mut MemoryBudget,
    ) -> Result<(Self, ArchiveSummary), L2dError> {
        Self::search_archive_reported(
            name,
            query,
            settings,
            path,
            options,
            budget,
            &mut |_| Ok(()),
        )
    }
    /// The unchanged archive search with a bounded caller-owned batch-plan observer.
    pub fn search_archive_reported(
        name: &str,
        query: SearchQuery,
        settings: SearchSettings,
        path: &Path,
        options: &ZipInputSettings,
        budget: &mut MemoryBudget,
        report: &mut dyn FnMut(&crate::BatchPlan) -> Result<(), L2dError>,
    ) -> Result<(Self, ArchiveSummary), L2dError> {
        let guard =
            ResourceGuard::tracked("selection archive search", busy_choice(), budget.usage())?;
        if matches!(query, SearchQuery::Fts5Trigram { .. }) {
            return Err(invalid("FTS5 trigram search requires a store"));
        }
        let metadata = path.as_os_str().len() as u64 * 4
            + options
                .entries
                .iter()
                .map(|e| e.len() as u64 + 32)
                .sum::<u64>()
            + 128;
        let reserve = Self::reservation(name, &query, settings, metadata * 16, budget)?;
        let matcher = compile(&query, budget)?;
        let before = budget.held_bytes();
        budget.hold(before + reserve);
        let result = (|| {
            let mut input = ZipInput::open(path, options, budget)?;
            input.hold_beside(before + reserve);
            let source = MarkedSource::Input(MarkedInput {
                path: MarkedInput::source_location(path)?,
                content_sha256: input.content_sha256().into(),
                archive_entries: Some(options.entries.clone()),
                interpretation: Some(crate::marks::InputInterpretation::archive(options)),
            });
            let mut set = Self::empty(name, query, settings, source);
            // Constant reservation avoids ZipBatches' hold_beside one-batch lag.
            for batch in input.batches(budget) {
                set.take(&batch?, &matcher)?;
            }
            if let Some(plan) = input.batch_plan() {
                report(plan)?;
            }
            let summary = input.finish();
            let mut file = fs::File::open(path).map_err(failed)?;
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 64 * 1024];
            loop {
                let read = file.read(&mut buffer).map_err(failed)?;
                if read == 0 {
                    break;
                }
                hash.update(&buffer[..read]);
            }
            if format!("{:x}", hash.finalize()) != summary.input.content_sha256 {
                return Err(refused("archive changed during selection; search again"));
            }
            set.completeness.omitted_entries = summary
                .entries
                .iter()
                .filter(|e| {
                    e.skipped.as_ref().is_some_and(|s| {
                        !matches!(
                            s,
                            crate::EntrySkip::NoPatternMatch | crate::EntrySkip::Directory
                        )
                    })
                })
                .count() as u64;
            set.counts(&summary.input);
            set.finish();
            Ok((set, summary))
        })();
        budget.hold(before);
        finish_result(guard, result)
    }
    pub fn search_store(
        name: &str,
        query: SearchQuery,
        settings: SearchSettings,
        store: &dyn SelectionStore,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        let guard =
            ResourceGuard::tracked("selection store search", busy_choice(), budget.usage())?;
        let reserve = Self::reservation(
            name,
            &query,
            settings,
            store.store_name().len() as u64 + 128,
            budget,
        )?;
        let matcher = if matches!(query, SearchQuery::Fts5Trigram { .. }) {
            None
        } else {
            Some(compile(&query, budget)?)
        };
        let before = budget.held_bytes();
        budget.hold(before + reserve);
        let result = (|| {
            let hash = store.content_sha256(budget)?;
            let mut set = Self::empty(
                name,
                query,
                settings,
                MarkedSource::Store(MarkedStore {
                    name: store.store_name().into(),
                    content_sha256: hash.clone(),
                }),
            );
            if let SearchQuery::Fts5Trigram { expression } = &set.query {
                let expression = expression.clone();
                set.completeness.distinct_scanned =
                    store.fts5_trigram_ids(&expression, budget, &mut |id| {
                        set.room()?;
                        if !hash_ok(id) {
                            return Err(invalid("invalid store ID"));
                        }
                        set.members.push(Member::StoreId(id.into()));
                        Ok(())
                    })?;
            } else {
                let matcher = matcher.as_ref().expect("non-FTS matcher");
                store.text_batches(0, 0, budget, &mut |batch| {
                    let ids = batch.column("id").and_then(Column::str).map_err(failed)?;
                    let texts = batch.column("text").and_then(Column::str).map_err(failed)?;
                    for (id, text) in ids.into_no_null_iter().zip(texts.into_no_null_iter()) {
                        set.completeness.distinct_scanned += 1;
                        if matcher.is_match(text) {
                            set.room()?;
                            if !hash_ok(id) {
                                return Err(invalid("invalid store ID"));
                            }
                            set.members.push(Member::StoreId(id.into()));
                        }
                    }
                    Ok(0)
                })?;
            }
            if store.content_sha256(budget)? != hash {
                return Err(refused("store changed during selection; search again"));
            }
            set.finish();
            Ok(set)
        })();
        budget.hold(before);
        finish_result(guard, result)
    }
    pub fn members(&self) -> &[Member] {
        &self.members
    }
    pub fn query(&self) -> &SearchQuery {
        &self.query
    }
    pub fn source(&self) -> &MarkedSource {
        &self.source
    }
    pub fn settings(&self) -> SearchSettings {
        self.settings
    }
    pub fn completeness(&self) -> &Completeness {
        &self.completeness
    }
    pub fn review(&self) -> Option<&ReviewRecord> {
        self.review.as_ref()
    }
    pub fn held_bytes(&self) -> u64 {
        FIXED
            + self.members.capacity() as u64 * MEMBER_BYTES
            + source_bytes(&self.source)
            + self.query.text().len() as u64
            + self.review.as_ref().map_or(0, |r| {
                r.decisions.capacity() as u64 * (MEMBER_BYTES + 16)
                    + r.action.analyst.len() as u64
                    + r.action.action.len() as u64
            })
    }
    pub fn require_input(
        &self,
        summary: &InputSummary,
        entries: Option<&[String]>,
    ) -> Result<(), L2dError> {
        match &self.source {
            MarkedSource::Input(s)
                if s.content_sha256 == summary.content_sha256
                    && s.archive_entries.as_deref() == entries =>
            {
                Ok(())
            }
            _ => Err(refused(
                "selection source hash or archive scope differs; search again",
            )),
        }
    }
    pub fn require_store(
        &self,
        store: &dyn MarkStore,
        budget: &mut MemoryBudget,
    ) -> Result<(), L2dError> {
        budget.require_whole(self.held_bytes(), "checking selection source")?;
        let before = budget.held_bytes();
        budget.hold(before + self.held_bytes());
        let result = store.content_sha256(budget);
        budget.hold(before);
        match (&self.source, result?) {
            (MarkedSource::Store(s), hash)
                if s.name == store.store_name() && s.content_sha256 == hash =>
            {
                Ok(())
            }
            _ => Err(refused("selection store snapshot differs; search again")),
        }
    }
    pub fn revision(&self, budget: &MemoryBudget) -> Result<String, L2dError> {
        budget.require_whole(self.held_bytes() + FIXED, "hashing selection revision")?;
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
        let mut writer = HashWriter(Sha256::new());
        serde_json::to_writer(&mut writer, self).map_err(failed)?;
        Ok(format!("{:x}", writer.0.finalize()))
    }
    pub fn preview_review(
        &self,
        decisions: &[ReviewDecision],
        budget: &MemoryBudget,
    ) -> Result<ReviewPreview, L2dError> {
        budget.require_whole(
            self.held_bytes() + FIXED + self.members.len() as u64 * (MEMBER_BYTES + 32),
            "previewing exact review members",
        )?;
        if decisions.len() > self.members.len() {
            return Err(invalid("too many review decisions"));
        }
        let mut ordered: Vec<&ReviewDecision> = decisions.iter().collect();
        ordered.sort_unstable_by(|a, b| a.member.cmp(&b.member));
        if ordered.windows(2).any(|p| p[0].member == p[1].member)
            || ordered
                .iter()
                .any(|d| self.members.binary_search(&d.member).is_err())
        {
            return Err(invalid(
                "review decisions must name unique selection members",
            ));
        }
        let decisions = self
            .members
            .iter()
            .map(|m| ReviewDecision {
                member: m.clone(),
                state: ordered
                    .binary_search_by(|d| d.member.cmp(m))
                    .map_or(ReviewState::Unresolved, |i| ordered[i].state),
            })
            .collect();
        Ok(ReviewPreview {
            revision: self.revision(budget)?,
            decisions,
        })
    }
    pub fn review_input(
        &mut self,
        preview: ReviewPreview,
        action: ReviewAction,
        summary: &InputSummary,
        entries: Option<&[String]>,
        budget: &MemoryBudget,
    ) -> Result<(), L2dError> {
        self.require_input(summary, entries)?;
        self.apply_review(preview, action, budget)
    }
    pub fn review_store(
        &mut self,
        preview: ReviewPreview,
        action: ReviewAction,
        store: &dyn MarkStore,
        budget: &mut MemoryBudget,
    ) -> Result<(), L2dError> {
        let preview_bytes = preview.decisions.capacity() as u64 * (MEMBER_BYTES + 16)
            + action.action.len() as u64
            + action.analyst.len() as u64
            + FIXED;
        budget.require_whole(
            preview_bytes + self.held_bytes(),
            "checking reviewed store snapshot with preview held",
        )?;
        let before = budget.held_bytes();
        budget.hold(before + preview_bytes);
        let checked = self.require_store(store, budget);
        budget.hold(before);
        checked?;
        self.apply_review(preview, action, budget)
    }
    pub(crate) fn apply_review(
        &mut self,
        preview: ReviewPreview,
        action: ReviewAction,
        budget: &MemoryBudget,
    ) -> Result<(), L2dError> {
        check_action(&action)?;
        budget.require_whole(
            self.held_bytes()
                + FIXED
                + preview.decisions.len() as u64 * (MEMBER_BYTES + 16)
                + action.action.len() as u64
                + action.analyst.len() as u64,
            "recording explicit review",
        )?;
        if preview.revision != self.revision(budget)? {
            return Err(refused(
                "review preview is stale; preview this selection revision again",
            ));
        }
        self.review = Some(ReviewRecord {
            selection_revision: preview.revision,
            action,
            decisions: preview.decisions,
        });
        Ok(())
    }
    /// Format 3 export carries exact reviewed members and revision references.
    /// `count()` includes unresolved/conflicting members as de-selected exclusions;
    /// `selected_count()` is only the requested resolved label. This prevents the
    /// existing one-MarkSet fitter from assuming excluded members are background.
    /// Persist this reviewed SelectionSet as the detailed action/decision receipt.
    pub fn export_mark_set(
        &self,
        name: &str,
        label: Label,
        budget: &MemoryBudget,
    ) -> Result<MarkSet, L2dError> {
        check_name(name)?;
        let state = match label {
            Label::Good => ReviewState::Good,
            Label::Bad => ReviewState::Bad,
            _ => return Err(invalid("export label must be good or bad")),
        };
        let review = self.review.as_ref().ok_or_else(|| {
            refused("preview and explicitly review selection before label export")
        })?;
        budget.require_whole(
            self.held_bytes()
                + FIXED
                + self.members.len() as u64 * MEMBER_BYTES * 6
                + crate::mark_ids::WORK_BYTES * 2
                + source_bytes(&self.source),
            "exporting reviewed exact members",
        )?;
        // Downstream MarkSet consumers account the actual member lengths, so do
        // not retain spare capacity or an unused vector for the other source kind.
        let excluded_count = review
            .decisions
            .iter()
            .filter(|d| matches!(d.state, ReviewState::Unresolved | ReviewState::Conflict))
            .count();
        let included_count =
            excluded_count + review.decisions.iter().filter(|d| d.state == state).count();
        let input = matches!(self.source, MarkedSource::Input(_));
        let mut positions = Vec::with_capacity(if input { included_count } else { 0 });
        let mut ids = Vec::with_capacity(if input { 0 } else { included_count });
        let mut excluded_positions = Vec::with_capacity(if input { excluded_count } else { 0 });
        let mut excluded_ids = Vec::with_capacity(if input { 0 } else { excluded_count });
        for d in &review.decisions {
            let excluded = matches!(d.state, ReviewState::Unresolved | ReviewState::Conflict);
            if d.state == state || excluded {
                match &d.member {
                    Member::Position(p) => {
                        positions.push(*p);
                        if excluded {
                            excluded_positions.push(*p);
                        }
                    }
                    Member::StoreId(id) => {
                        ids.push(id.clone());
                        if excluded {
                            excluded_ids.push(id.clone());
                        }
                    }
                }
            }
        }
        let mut marks = MarkSet::from_reviewed_selection(
            name,
            label,
            MarkQuery::ReviewedSelection {
                selection_revision: review.selection_revision.clone(),
                review_revision: self.revision(budget)?,
            },
            self.source.clone(),
            positions,
            ids,
        )?;
        if matches!(self.source, MarkedSource::Input(_)) {
            marks.deselect(&excluded_positions)?;
        } else {
            marks.deselect_ids(&excluded_ids, budget)?;
        }
        Ok(marks)
    }
    /// Immutable artifact publication: an existing destination is never replaced.
    pub fn save(&self, path: &Path, budget: &MemoryBudget) -> Result<(), L2dError> {
        let guard = ResourceGuard::tracked("saving selection", busy_choice(), budget.usage())?;
        budget.require_whole(
            self.held_bytes() + 2 * SAVE_BUFFER,
            "saving selection manifest",
        )?;
        self.validate()?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let parent = path
            .parent()
            .ok_or_else(|| invalid("selection output needs a parent directory"))?;
        fs::create_dir_all(parent).map_err(failed)?;
        let temporary = parent.join(format!(
            ".selection-{}-{}.saving",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let result = (|| {
            let file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(failed)?;
            let mut out = io::BufWriter::with_capacity(SAVE_BUFFER as usize, file);
            serde_json::to_writer_pretty(&mut out, self).map_err(failed)?;
            out.flush().map_err(failed)?;
            out.get_ref().sync_all().map_err(failed)?;
            drop(out);
            fs::hard_link(&temporary, path).map_err(failed)?;
            fs::remove_file(&temporary).map_err(failed)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        finish_result(guard, result)
    }
    pub fn load(path: &Path, budget: &MemoryBudget) -> Result<Self, L2dError> {
        let guard = ResourceGuard::tracked("loading selection", busy_choice(), budget.usage())?;
        let file = fs::File::open(path).map_err(failed)?;
        let bytes = file.metadata().map_err(failed)?.len();
        let reserve = bytes
            .checked_mul(64)
            .and_then(|v| v.checked_add(FIXED))
            .ok_or_else(|| refused("selection file too large"))?;
        budget.require_whole(
            reserve,
            "loading selection metadata and members before allocation",
        )?;
        let mut input = Vec::with_capacity(
            usize::try_from(bytes).map_err(|_| refused("selection file exceeds host capacity"))?,
        );
        file.take(bytes).read_to_end(&mut input).map_err(failed)?;
        let set: Self = serde_json::from_slice(&input).map_err(failed)?;
        set.validate()?;
        guard.finish()?;
        Ok(set)
    }
    fn validate(&self) -> Result<(), L2dError> {
        check_name(&self.name)?;
        self.query.check()?;
        if self.format != 1
            || self.kind != "selection-set"
            || self.settings.max_members == 0
            || self.members.len() as u64 > self.settings.max_members
            || self.completeness.matched != self.members.len() as u64
            || self.completeness.distinct_scanned < self.completeness.matched
            || self.completeness.complete
                != (self.completeness.skipped_records == 0
                    && self.completeness.omitted_entries == 0)
            || !self.members.windows(2).all(|p| p[0] < p[1])
        {
            return Err(invalid(
                "invalid selection version, membership or completeness",
            ));
        }
        let store = matches!(self.source, MarkedSource::Store(_));
        let hash = match &self.source {
            MarkedSource::Input(s) => &s.content_sha256,
            MarkedSource::Store(s) => &s.content_sha256,
        };
        if !hash_ok(hash) || (!store && matches!(self.query, SearchQuery::Fts5Trigram { .. })) {
            return Err(invalid("invalid selection source/query"));
        }
        if self.members.iter().any(|m| match m {
            Member::Position(_) => store,
            Member::StoreId(id) => !store || !hash_ok(id),
        }) {
            return Err(invalid("invalid selection member identity"));
        }
        if let Some(r) = &self.review {
            check_action(&r.action)?;
            if !hash_ok(&r.selection_revision)
                || r.decisions.len() != self.members.len()
                || r.decisions
                    .iter()
                    .zip(&self.members)
                    .any(|(d, m)| &d.member != m)
            {
                return Err(invalid("invalid selection review provenance"));
            }
        }
        Ok(())
    }
}
fn check_action(a: &ReviewAction) -> Result<(), L2dError> {
    if a.analyst.trim().is_empty()
        || a.analyst.len() > 256
        || a.action.trim().is_empty()
        || a.action.len() > 4096
    {
        return Err(invalid(
            "review requires analyst (1..256 bytes) and explicit action (1..4096 bytes)",
        ));
    }
    Ok(())
}

fn finish_result<T>(guard: ResourceGuard, result: Result<T, L2dError>) -> Result<T, L2dError> {
    match result {
        Ok(value) => {
            guard.finish()?;
            Ok(value)
        }
        Err(error) => {
            drop(guard);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemorySettings;
    #[test]
    fn search_modes_escaping_case_unicode_and_newlines() {
        let budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();
        let literal = SearchQuery::Literal {
            text: "a.b".into(),
            insensitive: false,
        };
        assert!(
            compile(&literal, &budget)
                .unwrap()
                .is_match("prefix a.b suffix")
        );
        assert!(!compile(&literal, &budget).unwrap().is_match("axb"));
        let glob = SearchQuery::Glob {
            pattern: "a?b".into(),
            insensitive: false,
        };
        let matcher = compile(&glob, &budget).unwrap();
        assert!(matcher.is_match("aéb"));
        assert!(matcher.is_match("a\nb"));
        assert!(!matcher.is_match("prefix aéb"));
        assert!(!matcher.is_match("ab"));
        let escaped = SearchQuery::Glob {
            pattern: r"\*\?\\".into(),
            insensitive: false,
        };
        assert!(compile(&escaped, &budget).unwrap().is_match("*?\\"));
        let any = SearchQuery::Glob {
            pattern: "a*b".into(),
            insensitive: true,
        };
        assert!(compile(&any, &budget).unwrap().is_match("AB"));
        assert!(compile(&any, &budget).unwrap().is_match("a\n\nb"));
        let regex = SearchQuery::Regex {
            pattern: "a.b".into(),
            insensitive: false,
        };
        assert!(compile(&regex, &budget).unwrap().is_match("prefix axb"));
        assert!(
            compile(
                &SearchQuery::Glob {
                    pattern: "x\\".into(),
                    insensitive: false
                },
                &budget
            )
            .is_err()
        );
    }
}
