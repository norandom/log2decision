//! Bounded, source-bound discovery summaries. Templates are not exact matchers and
//! discovery masks never authorize fitting redactions. Membership is the authority.
use crate::polars::prelude::Column;
use crate::resource::{ResourceGuard, busy_choice};
use crate::selections::{Completeness, Member, ReviewState, SelectionSet};
use crate::{InputSummary, L2dError, LogsFrame, MarkStore, MarkedSource, MemoryBudget};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
pub mod refinement;

/// Store adapter with original multiplicities. Its snapshot includes counts and
/// text identity; all methods restore the incoming budget hold on errors too.
pub trait PatternStore: MarkStore {
    fn pattern_snapshot(&self, budget: &mut MemoryBudget) -> Result<String, L2dError>;
    fn pattern_batches(
        &self,
        beside: u64,
        per_row: u64,
        budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&crate::polars::prelude::DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError>;
}

const FIXED: u64 = 4 * 1024 * 1024;
const ALGORITHM_REVISION: &str = "bounded-patterns-v1";
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PatternEngine {
    DrainStyle,
    CharacterTrigram,
}
/// Key/value discovery masks only; never a RedactionRule. Exact case-sensitive key,
/// one alphanumeric value. Punctuation and all other tokens remain visible.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveryMasks {
    pub revision: String,
    pub keys: Vec<String>,
}
impl Default for DiscoveryMasks {
    fn default() -> Self {
        Self {
            revision: "key-values-v1".into(),
            keys: vec!["id".into(), "user".into(), "host".into()],
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatternSettings {
    pub engine: PatternEngine,
    pub max_patterns: usize,
    pub max_text_bytes: usize,
    pub similarity_per_mille: u16,
    pub masks: DiscoveryMasks,
}
impl Default for PatternSettings {
    fn default() -> Self {
        Self {
            engine: PatternEngine::DrainStyle,
            max_patterns: 256,
            max_text_bytes: 4096,
            similarity_per_mille: 800,
            masks: DiscoveryMasks::default(),
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatternCounts {
    pub members: u64,
    /// Original input or stored occurrence count (the source count can saturate at u32::MAX).
    pub occurrences: u64,
    pub distinct_texts: u64,
    /// Distinct texts with a good/bad/conflict review; unresolved is separate.
    pub reviewed_texts: u64,
    pub reviewed_good_texts: u64,
    pub reviewed_bad_texts: u64,
    pub unresolved_texts: u64,
    pub conflicting_texts: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Pattern {
    pub id: usize,
    pub template: String,
    pub representative: String,
    pub variable_examples: Vec<String>,
    pub boundary_example: String,
    pub minimum_similarity_per_mille: u16,
    /// Differing original literal tokens are retained for inspection, not silently
    /// treated as harmless. Known outcome signatures additionally prevent merging.
    pub differing_literals: Vec<String>,
    pub possible_outcome_collision: bool,
    pub counts: PatternCounts,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatternCatalogue {
    pub format: u32,
    pub algorithm_revision: String,
    pub selection_revision: String,
    pub source: MarkedSource,
    pub source_data_revision: Option<String>,
    pub completeness: Completeness,
    pub settings: PatternSettings,
    pub occurrence_basis: String,
    pub patterns: Vec<Pattern>,
    pub ungrouped: PatternCounts,
    pub selected_members: u64,
    pub memberships_sha256: String,
    pub revision: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatternMember {
    pub member: Member,
    pub pattern: Option<usize>,
    pub text: String,
    pub occurrences: u64,
    pub review: Option<ReviewState>,
}
fn fail(e: impl std::fmt::Display) -> L2dError {
    L2dError::Refused {
        reason: format!("pattern catalogue: {e}"),
    }
}
fn checked(a: u64, b: u64) -> Result<u64, L2dError> {
    a.checked_add(b).ok_or_else(|| fail("size/count overflow"))
}
fn hash_file(path: &Path) -> Result<String, L2dError> {
    let mut f = File::open(path).map_err(fail)?;
    let mut h = Sha256::new();
    let mut bytes = [0; 65536];
    loop {
        let n = f.read(&mut bytes).map_err(fail)?;
        if n == 0 {
            break;
        }
        h.update(&bytes[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
/// SHA-256 of the streamed JSON representation, in serialization order.
pub fn json_hash<T: Serialize>(value: &T) -> Result<String, L2dError> {
    struct Hash(Sha256);
    impl Write for Hash {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.update(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut h = Hash(Sha256::new());
    serde_json::to_writer(&mut h, value).map_err(fail)?;
    Ok(format!("{:x}", h.0.finalize()))
}
fn finish<T>(guard: ResourceGuard, result: Result<T, L2dError>) -> Result<T, L2dError> {
    match result {
        Ok(x) => {
            guard.finish()?;
            Ok(x)
        }
        Err(e) => Err(e),
    }
}
impl PatternSettings {
    fn validate(&self) -> Result<(), L2dError> {
        if self.max_patterns == 0
            || self.max_patterns > 65536
            || self.max_text_bytes == 0
            || self.max_text_bytes > 1024 * 1024
            || self.similarity_per_mille == 0
            || self.similarity_per_mille > 1000
            || self.masks.keys.len() > 64
            || self.masks.revision.len() > 128
            || self
                .masks
                .keys
                .iter()
                .any(|k| k.is_empty() || k.len() > 64 || !k.chars().all(char::is_alphanumeric))
        {
            return Err(fail("invalid bounded settings"));
        }
        Ok(())
    }
    fn reservation(&self, selection: &SelectionSet) -> Result<u64, L2dError> {
        self.validate()?;
        // Includes template tokens, three full examples, signatures, trigram work sets,
        // hash indexes (including tree overhead), spool offsets, serde and write buffers.
        let groups = (self.max_patterns as u64)
            .checked_mul(self.max_text_bytes as u64 * 160 + 4096)
            .ok_or_else(|| fail("catalogue reservation overflow"))?;
        checked(
            checked(FIXED + selection.held_bytes() * 2, groups)?,
            selection.members().len() as u64 * 512 + self.max_text_bytes as u64 * 256,
        )
    }
}
struct Spool<'a> {
    selection: &'a SelectionSet,
    file: File,
    order: File,
    directory: std::path::PathBuf,
    offsets: Vec<Option<u64>>,
    max: usize,
}
impl<'a> Spool<'a> {
    fn new(selection: &'a SelectionSet, path: &Path, max: usize) -> Result<Self, L2dError> {
        Ok(Self {
            selection,
            file: File::options()
                .read(true)
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(fail)?,
            order: File::options()
                .read(true)
                .write(true)
                .create_new(true)
                .open(path.with_extension("order"))
                .map_err(fail)?,
            directory: path.parent().unwrap().to_path_buf(),
            offsets: vec![None; selection.members().len()],
            max,
        })
    }
    fn add(&mut self, member: &Member, text: &str, occurrences: u64) -> Result<(), L2dError> {
        let Ok(i) = self.selection.members().binary_search(member) else {
            return Ok(());
        };
        if text.len() > self.max {
            return Err(fail(
                "selected text exceeds max_text_bytes; increase the explicit bound",
            ));
        }
        if occurrences == 0 || self.offsets[i].is_some() {
            return Err(fail("duplicate or invalid selected source identity"));
        }
        self.offsets[i] = Some(self.file.stream_position().map_err(fail)?);
        self.file
            .write_all(&occurrences.to_le_bytes())
            .map_err(fail)?;
        self.file
            .write_all(&(text.len() as u64).to_le_bytes())
            .map_err(fail)?;
        self.file.write_all(text.as_bytes()).map_err(fail)
    }
    fn frame(&mut self, logs: &LogsFrame) -> Result<(), L2dError> {
        let frame = logs.frame();
        let text = frame
            .column(LogsFrame::TEXT)
            .and_then(Column::str)
            .map_err(fail)?;
        let positions = frame
            .column(LogsFrame::FIRST_POSITION)
            .and_then(Column::u64)
            .map_err(fail)?;
        let counts = frame
            .column(LogsFrame::OCCURRENCES)
            .and_then(Column::u32)
            .map_err(fail)?;
        for ((t, p), n) in text
            .into_no_null_iter()
            .zip(positions.into_no_null_iter())
            .zip(counts.into_no_null_iter())
        {
            let member = Member::Position(p);
            self.add(&member, t, u64::from(n))?;
            let offset = self
                .selection
                .members()
                .binary_search(&member)
                .ok()
                .and_then(|i| self.offsets[i])
                .unwrap_or(u64::MAX);
            self.order.write_all(&offset.to_le_bytes()).map_err(fail)?;
        }
        Ok(())
    }
    fn read(&mut self, i: usize) -> Result<(String, u64), L2dError> {
        let pos = self.offsets[i]
            .ok_or_else(|| fail("selected member missing from source; search again"))?;
        self.file.seek(SeekFrom::Start(pos)).map_err(fail)?;
        let mut bytes = [0; 8];
        self.file.read_exact(&mut bytes).map_err(fail)?;
        let count = u64::from_le_bytes(bytes);
        self.file.read_exact(&mut bytes).map_err(fail)?;
        let len = u64::from_le_bytes(bytes);
        if len > self.max as u64 {
            return Err(fail("invalid spool length"));
        }
        let mut text = vec![0; len as usize];
        self.file.read_exact(&mut text).map_err(fail)?;
        Ok((String::from_utf8(text).map_err(fail)?, count))
    }
}
// Every punctuation and whitespace character is a token, preserving byte-level
// distinctions. Alphanumeric runs form words; no Unicode/case normalization.
fn tokens(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            word.push(c);
        } else {
            if !word.is_empty() {
                result.push(std::mem::take(&mut word));
            }
            result.push(c.to_string());
        }
    }
    if !word.is_empty() {
        result.push(word);
    }
    result
}
fn masked(text: &str, masks: &DiscoveryMasks) -> Vec<String> {
    let mut t = tokens(text);
    for i in 2..t.len() {
        if t[i - 1] == "="
            && masks.keys.contains(&t[i - 2])
            && t[i].chars().all(char::is_alphanumeric)
        {
            t[i] = "<*>".into();
        }
    }
    t
}
fn protected(text: &str) -> Vec<String> {
    let t = tokens(text);
    let mut result = Vec::new();
    for (i, token) in t.iter().enumerate() {
        let lower = token.to_lowercase();
        if [
            "not",
            "no",
            "never",
            "without",
            "failed",
            "failure",
            "succeeded",
            "success",
            "error",
            "denied",
            "permitted",
            "accepted",
            "rejected",
            "revoked",
            "true",
            "false",
        ]
        .contains(&lower.as_str())
            || (i >= 2
                && t[i - 1] == "="
                && ["status", "code", "rc", "result", "outcome"].contains(&t[i - 2].as_str()))
        {
            result.push(token.clone());
        }
    }
    result
}
/// Exact Unicode character trigrams shared by lexical discovery and exploration.
pub fn character_trigrams(text: &str) -> BTreeSet<[char; 3]> {
    let chars: Vec<_> = text.chars().collect();
    chars.windows(3).map(|w| [w[0], w[1], w[2]]).collect()
}
struct Cluster {
    pattern: Pattern,
    tokens: Vec<String>,
    exemplar: Vec<String>,
    protected: Vec<String>,
}
fn similarity(engine: PatternEngine, a: &[String], b: &[String]) -> u16 {
    match engine {
        PatternEngine::DrainStyle => {
            // Fixed-depth route: token count then first token. Candidate leaves are capped
            // by max_patterns; no corpus-sized pair matrix or eviction.
            if a.len() != b.len() || a.first() != b.first() {
                return 0;
            }
            // Whitespace/punctuation must match; only word tokens may generalize.
            if a.iter().zip(b).any(|(x, y)| {
                x != y
                    && ((x != "<*>" && !x.chars().all(char::is_alphanumeric))
                        || (y != "<*>" && !y.chars().all(char::is_alphanumeric)))
            }) {
                return 0;
            }
            let pairs: Vec<_> = a
                .iter()
                .zip(b)
                .filter(|(x, _)| !x.chars().all(char::is_whitespace))
                .collect();
            if pairs.is_empty() {
                return if a == b { 1000 } else { 0 };
            }
            (pairs
                .iter()
                .filter(|(x, y)| x == y || x.as_str() == "<*>" || y.as_str() == "<*>")
                .count()
                * 1000
                / pairs.len()) as u16
        }
        PatternEngine::CharacterTrigram => {
            let aa = a.concat();
            let bb = b.concat();
            if aa == bb {
                return 1000;
            }
            let aa = character_trigrams(&aa);
            let bb = character_trigrams(&bb);
            let union = aa.union(&bb).count();
            (aa.intersection(&bb).count() * 1000)
                .checked_div(union)
                .unwrap_or(0) as u16
        }
    }
}
#[derive(Default)]
struct Support {
    seen: BTreeMap<[u8; 32], (Option<usize>, u8)>,
}
impl Support {
    fn add(
        &mut self,
        text: &str,
        group: Option<usize>,
        review: Option<ReviewState>,
        counts: &mut PatternCounts,
        occ: u64,
    ) -> Result<(), L2dError> {
        counts.members = checked(counts.members, 1)?;
        counts.occurrences = checked(counts.occurrences, occ)?;
        let key: [u8; 32] = Sha256::digest(text.as_bytes()).into();
        let bit = match review {
            None => 0,
            Some(ReviewState::Good) => 1,
            Some(ReviewState::Bad) => 2,
            Some(ReviewState::Unresolved) => 4,
            Some(ReviewState::Conflict) => 8,
        };
        let old = self.seen.get(&key).copied();
        if old.is_some_and(|(g, _)| g != group) {
            return Err(fail("identical text assigned to different patterns"));
        }
        if old.is_none() {
            counts.distinct_texts += 1;
        }
        let previous = old.map_or(0, |(_, bits)| bits);
        let bits = previous | bit;
        if previous & 11 == 0 && bits & 11 != 0 {
            counts.reviewed_texts += 1;
        }
        if previous & 1 == 0 && bits & 1 != 0 {
            counts.reviewed_good_texts += 1;
        }
        if previous & 2 == 0 && bits & 2 != 0 {
            counts.reviewed_bad_texts += 1;
        }
        if previous & 4 == 0 && bits & 4 != 0 {
            counts.unresolved_texts += 1;
        }
        let conflict = |v: u8| v & 8 != 0 || v & 3 == 3;
        if !conflict(previous) && conflict(bits) {
            counts.conflicting_texts += 1;
        }
        self.seen.insert(key, (group, bits));
        Ok(())
    }
}
impl PatternCatalogue {
    pub fn build_input(
        selection: &SelectionSet,
        logs: &LogsFrame,
        summary: &InputSummary,
        settings: &PatternSettings,
        destination: &Path,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        let guard =
            ResourceGuard::tracked("pattern input catalogue", busy_choice(), budget.usage())?;
        let before = budget.held_bytes();
        let result = (|| {
            let summary_bytes = summary.skipped.capacity() as u64 * 32
                + summary
                    .skipped
                    .iter()
                    .map(|s| s.reason.capacity() as u64)
                    .sum::<u64>()
                + summary.source_path.as_os_str().len() as u64 * 4
                + 128;
            let reserve = checked(
                checked(
                    settings.reservation(selection)?,
                    logs.frame().estimated_size() as u64 * 2,
                )?,
                summary_bytes,
            )?;
            budget.require_whole(reserve, "pattern catalogue and retained input")?;
            selection.require_input(summary, None)?;
            if let MarkedSource::Input(source) = selection.source() {
                source.require_interpretation(&crate::marks::InputInterpretation::file(
                    logs, budget,
                )?)?;
            }
            if hash_file(&summary.source_path)? != summary.content_sha256 {
                return Err(fail("input changed since selection"));
            }
            budget.hold(checked(before, reserve)?);
            Self::build(selection, settings, destination, budget, |spool, _| {
                spool.frame(logs)?;
                if hash_file(&summary.source_path)? != summary.content_sha256 {
                    return Err(fail("input changed during discovery"));
                }
                Ok(None)
            })
        })();
        budget.hold(before);
        finish(guard, result)
    }
    pub fn build_archive(
        selection: &SelectionSet,
        path: &Path,
        options: &crate::ZipInputSettings,
        settings: &PatternSettings,
        destination: &Path,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        Self::build_archive_reported(
            selection,
            path,
            options,
            settings,
            destination,
            budget,
            &mut |_| Ok(()),
        )
    }
    /// The unchanged catalogue builder with an observer for the actual archive plan.
    pub fn build_archive_reported(
        selection: &SelectionSet,
        path: &Path,
        options: &crate::ZipInputSettings,
        settings: &PatternSettings,
        destination: &Path,
        budget: &mut MemoryBudget,
        report: &mut dyn FnMut(&crate::BatchPlan) -> Result<(), L2dError>,
    ) -> Result<Self, L2dError> {
        let guard =
            ResourceGuard::tracked("pattern archive catalogue", busy_choice(), budget.usage())?;
        let before = budget.held_bytes();
        let result = (|| {
            let reserve = settings.reservation(selection)?;
            budget.require_whole(reserve, "pattern archive catalogue")?;
            budget.hold(checked(before, reserve)?);
            Self::build(selection, settings, destination, budget, |spool, budget| {
                let mut input = crate::ZipInput::open(path, options, budget)?;
                input.hold_beside(checked(before, reserve)?);
                input.spill_into(&spool.directory.join("archive-counts"), None)?;
                match selection.source() {
                    MarkedSource::Input(s)
                        if s.content_sha256 == input.content_sha256()
                            && s.archive_entries.as_ref() == Some(&options.entries) =>
                    {
                        s.require_interpretation(&crate::marks::InputInterpretation::archive(
                            options,
                        ))?;
                    }
                    _ => return Err(fail("selection archive source or scope differs")),
                }
                for batch in input.batches(budget) {
                    spool.frame(&batch?)?;
                }
                if let Some(plan) = input.batch_plan() {
                    report(plan)?;
                }
                // Final counts are in emitted-row order, including late duplicate occurrences.
                spool.order.seek(SeekFrom::Start(0)).map_err(fail)?;
                for chunk in input.occurrence_counts() {
                    for count in chunk? {
                        let mut bytes = [0; 8];
                        spool.order.read_exact(&mut bytes).map_err(fail)?;
                        let offset = u64::from_le_bytes(bytes);
                        if offset != u64::MAX {
                            if count == 0 {
                                return Err(fail("missing archive occurrence count"));
                            }
                            spool.file.seek(SeekFrom::Start(offset)).map_err(fail)?;
                            spool
                                .file
                                .write_all(&u64::from(count).to_le_bytes())
                                .map_err(fail)?;
                        }
                    }
                }
                let mut extra = [0];
                if spool.order.read(&mut extra).map_err(fail)? != 0 {
                    return Err(fail("archive count stream incomplete"));
                }
                let summary = input.finish();
                selection.require_input(&summary.input, Some(&options.entries))?;
                if hash_file(path)? != summary.input.content_sha256 {
                    return Err(fail("archive changed during discovery"));
                }
                Ok(None)
            })
        })();
        budget.hold(before);
        finish(guard, result)
    }
    pub fn build_store(
        selection: &SelectionSet,
        store: &dyn PatternStore,
        settings: &PatternSettings,
        destination: &Path,
        budget: &mut MemoryBudget,
    ) -> Result<Self, L2dError> {
        let guard =
            ResourceGuard::tracked("pattern store catalogue", busy_choice(), budget.usage())?;
        let before = budget.held_bytes();
        let result = (|| {
            let reserve = settings.reservation(selection)?;
            budget.require_whole(reserve, "pattern catalogue")?;
            selection.require_store(store, budget)?;
            budget.hold(checked(before, reserve)?);
            Self::build(selection, settings, destination, budget, |spool, budget| {
                let snapshot = store.pattern_snapshot(budget)?;
                store.pattern_batches(0, 0, budget, &mut |frame| {
                    let ids = frame.column("id").and_then(Column::str).map_err(fail)?;
                    let texts = frame.column("text").and_then(Column::str).map_err(fail)?;
                    let counts = frame
                        .column("occurrences")
                        .and_then(Column::u32)
                        .map_err(fail)?;
                    for ((id, text), count) in ids
                        .into_no_null_iter()
                        .zip(texts.into_no_null_iter())
                        .zip(counts.into_no_null_iter())
                    {
                        spool.add(&Member::StoreId(id.into()), text, u64::from(count))?;
                    }
                    Ok(0)
                })?;
                selection.require_store(store, budget)?;
                if store.pattern_snapshot(budget)? != snapshot {
                    return Err(fail("store occurrence snapshot changed during discovery"));
                }
                Ok(Some(snapshot))
            })
        })();
        budget.hold(before);
        finish(guard, result)
    }
    fn build(
        selection: &SelectionSet,
        settings: &PatternSettings,
        destination: &Path,
        budget: &mut MemoryBudget,
        feed: impl FnOnce(&mut Spool<'_>, &mut MemoryBudget) -> Result<Option<String>, L2dError>,
    ) -> Result<Self, L2dError> {
        if destination.exists() {
            return Err(fail(
                "destination exists; use a new immutable catalogue directory",
            ));
        }
        // A uniquely created sibling is never externally visible as a finished artifact.
        let parent = destination
            .parent()
            .ok_or_else(|| fail("catalogue requires parent directory"))?;
        fs::create_dir_all(parent).map_err(fail)?;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let staging = parent.join(format!(
            ".patterns-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir(&staging).map_err(fail)?;
        let result = (|| {
            budget.note_spill();
            let mut spool = Spool::new(selection, &staging.join("spool"), settings.max_text_bytes)?;
            let source_data_revision = feed(&mut spool, budget)?;
            let mut clusters: Vec<Cluster> = Vec::new();
            let mut support = Support::default();
            let mut ungrouped = PatternCounts::default();
            let mut members = File::create(staging.join("members.jsonl")).map_err(fail)?;
            for (i, member) in selection.members().iter().enumerate() {
                let (text, occurrences) = spool.read(i)?;
                let words = masked(&text, &settings.masks);
                let signature = protected(&text);
                let review = selection.review().and_then(|r| {
                    r.decisions
                        .binary_search_by(|d| d.member.cmp(member))
                        .ok()
                        .map(|i| r.decisions[i].state)
                });
                // Reuse exact-text assignments, so template evolution cannot split duplicates.
                let digest: [u8; 32] = Sha256::digest(text.as_bytes()).into();
                let existing = support.seen.get(&digest).map(|(g, _)| *g);
                let mut best = None;
                let mut score = 0;
                if let Some(g) = existing {
                    best = g;
                    score = 1000;
                } else {
                    for (id, c) in clusters.iter().enumerate() {
                        if c.protected != signature {
                            continue;
                        }
                        let s = similarity(
                            settings.engine,
                            if settings.engine == PatternEngine::DrainStyle {
                                &c.tokens
                            } else {
                                &c.exemplar
                            },
                            &words,
                        );
                        // Equal scores retain the earliest canonical cluster.
                        if s >= settings.similarity_per_mille && (best.is_none() || s > score) {
                            best = Some(id);
                            score = s;
                        }
                    }
                    if best.is_none() && clusters.len() < settings.max_patterns {
                        let id = clusters.len();
                        clusters.push(Cluster {
                            pattern: Pattern {
                                id,
                                template: words.concat(),
                                representative: text.clone(),
                                variable_examples: Vec::new(),
                                boundary_example: text.clone(),
                                minimum_similarity_per_mille: 1000,
                                differing_literals: Vec::new(),
                                possible_outcome_collision: false,
                                counts: PatternCounts::default(),
                            },
                            tokens: words.clone(),
                            exemplar: words.clone(),
                            protected: signature,
                        });
                        best = Some(id);
                        score = 1000;
                    }
                }
                if let Some(id) = best {
                    let c = &mut clusters[id];
                    let p = &mut c.pattern;
                    if text != p.representative
                        && p.variable_examples.len() < 2
                        && !p.variable_examples.contains(&text)
                    {
                        p.variable_examples.push(text.clone());
                    }
                    if score < p.minimum_similarity_per_mille {
                        p.minimum_similarity_per_mille = score;
                        p.boundary_example = text.clone();
                    }
                    if c.tokens.len() == words.len() {
                        for (a, b) in c.tokens.iter_mut().zip(&words) {
                            if a != b {
                                if a != "<*>" && b != "<*>" {
                                    p.possible_outcome_collision = true;
                                    if p.differing_literals.len() < 8 {
                                        p.differing_literals.push(format!("{a} -> {b}"));
                                    }
                                }
                                *a = "<*>".into();
                            }
                        }
                        p.template = c.tokens.concat();
                    } else {
                        p.possible_outcome_collision = true;
                        p.template = "[variable structure; inspect members]".into();
                    }
                    support.add(&text, best, review, &mut p.counts, occurrences)?;
                } else {
                    support.add(&text, None, review, &mut ungrouped, occurrences)?;
                }
                let row = PatternMember {
                    member: member.clone(),
                    pattern: best,
                    text,
                    occurrences,
                    review,
                };
                serde_json::to_writer(&mut members, &row).map_err(fail)?;
                members.write_all(b"\n").map_err(fail)?;
            }
            members.sync_all().map_err(fail)?;
            drop(members);
            drop(spool);
            fs::remove_file(staging.join("spool")).map_err(fail)?;
            fs::remove_file(staging.join("spool.order")).map_err(fail)?;
            let mut catalogue = Self {
                format: 1,
                algorithm_revision: ALGORITHM_REVISION.into(),
                selection_revision: selection.revision(budget)?,
                source: selection.source().clone(),
                source_data_revision,
                completeness: selection.completeness().clone(),
                settings: settings.clone(),
                occurrence_basis: if matches!(selection.source(), MarkedSource::Store(_)) {
                    "stored_occurrences"
                } else {
                    "input_occurrences"
                }
                .into(),
                patterns: clusters.into_iter().map(|c| c.pattern).collect(),
                ungrouped,
                selected_members: selection.members().len() as u64,
                memberships_sha256: hash_file(&staging.join("members.jsonl"))?,
                revision: String::new(),
            };
            catalogue.revision = json_hash(&catalogue)?;
            let mut manifest = File::create(staging.join("manifest.json")).map_err(fail)?;
            serde_json::to_writer(&mut manifest, &catalogue).map_err(fail)?;
            manifest.sync_all().map_err(fail)?;
            drop(manifest);
            fs::rename(&staging, destination).map_err(fail)?;
            Ok(catalogue)
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }
    pub fn load(directory: &Path, budget: &mut MemoryBudget) -> Result<Self, L2dError> {
        let guard =
            ResourceGuard::tracked("pattern catalogue load", busy_choice(), budget.usage())?;
        let result = (|| {
            let path = directory.join("manifest.json");
            let size = fs::metadata(&path).map_err(fail)?.len();
            budget.require_whole(
                size.saturating_mul(32).saturating_add(FIXED),
                "loading bounded pattern metadata",
            )?;
            let mut f = File::open(&path).map_err(fail)?.take(size + 1);
            let mut bytes = Vec::new();
            f.read_to_end(&mut bytes).map_err(fail)?;
            if bytes.len() as u64 != size {
                return Err(fail("manifest changed while reading"));
            }
            let mut c: Self = serde_json::from_slice(&bytes).map_err(fail)?;
            c.settings.validate()?;
            if c.patterns.len() > c.settings.max_patterns
                || c.patterns.iter().enumerate().any(|(i, p)| {
                    p.id != i
                        || p.representative.len() > c.settings.max_text_bytes
                        || p.boundary_example.len() > c.settings.max_text_bytes
                        || p.variable_examples.len() > 2
                        || p.differing_literals.len() > 8
                })
                || c.patterns
                    .iter()
                    .map(|p| p.counts.members)
                    .try_fold(c.ungrouped.members, |a, b| a.checked_add(b))
                    != Some(c.selected_members)
            {
                return Err(fail("invalid catalogue shape/counts"));
            }
            if c.format != 1 || c.algorithm_revision != ALGORITHM_REVISION {
                return Err(fail("unsupported catalogue version"));
            }
            let expected = std::mem::take(&mut c.revision);
            if json_hash(&c)? != expected {
                return Err(fail("catalogue revision mismatch"));
            }
            c.revision = expected;
            if hash_file(&directory.join("members.jsonl"))? != c.memberships_sha256 {
                return Err(fail("membership digest mismatch"));
            }
            Ok(c)
        })();
        finish(guard, result)
    }
    /// Resident capacity estimate, independent of any mutable manifest on disk.
    pub fn held_bytes(&self) -> u64 {
        let string = |s: &String| s.capacity() as u64;
        let strings = |v: &Vec<String>| {
            v.capacity() as u64 * size_of::<String>() as u64 + v.iter().map(string).sum::<u64>()
        };
        let source = match &self.source {
            MarkedSource::Input(s) => {
                string(&s.path)
                    + string(&s.content_sha256)
                    + s.archive_entries.as_ref().map_or(0, strings)
            }
            MarkedSource::Store(s) => string(&s.name) + string(&s.content_sha256),
        };
        size_of::<Self>() as u64
            + string(&self.algorithm_revision)
            + string(&self.selection_revision)
            + string(&self.occurrence_basis)
            + string(&self.memberships_sha256)
            + string(&self.revision)
            + self.source_data_revision.as_ref().map_or(0, string)
            + source
            + string(&self.settings.masks.revision)
            + strings(&self.settings.masks.keys)
            + self.patterns.capacity() as u64 * size_of::<Pattern>() as u64
            + self
                .patterns
                .iter()
                .map(|p| {
                    string(&p.template)
                        + string(&p.representative)
                        + string(&p.boundary_example)
                        + strings(&p.variable_examples)
                        + strings(&p.differing_literals)
                })
                .sum::<u64>()
    }
    pub fn visit_members(
        &self,
        directory: &Path,
        budget: &mut MemoryBudget,
        mut each: impl FnMut(PatternMember) -> Result<(), L2dError>,
    ) -> Result<(), L2dError> {
        let guard =
            ResourceGuard::tracked("pattern membership read", busy_choice(), budget.usage())?;
        let result = (|| {
            self.settings.validate()?;
            budget.require_whole(
                self.held_bytes()
                    .saturating_add(self.settings.max_text_bytes as u64 * 32)
                    .saturating_add(FIXED),
                "streaming pattern membership and resident catalogue",
            )?;
            if hash_file(&directory.join("manifest.json"))? != json_hash(self)? {
                return Err(fail("directory manifest differs from resident catalogue"));
            }
            let path = directory.join("members.jsonl");
            if hash_file(&path)? != self.memberships_sha256 {
                return Err(fail("membership digest mismatch"));
            }
            let mut reader = BufReader::new(File::open(&path).map_err(fail)?);
            let mut count = 0;
            let mut previous = None;
            loop {
                let mut line = String::new();
                let read = reader
                    .by_ref()
                    .take(self.settings.max_text_bytes as u64 * 6 + 1024)
                    .read_line(&mut line)
                    .map_err(fail)?;
                if read == 0 {
                    break;
                }
                if !line.ends_with('\n') {
                    return Err(fail("oversized/truncated membership row"));
                }
                let row: PatternMember = serde_json::from_str(&line).map_err(fail)?;
                if row.text.len() > self.settings.max_text_bytes
                    || row.pattern.is_some_and(|i| i >= self.patterns.len())
                    || previous.as_ref().is_some_and(|p| p >= &row.member)
                {
                    return Err(fail("invalid membership identity/order"));
                }
                previous = Some(row.member.clone());
                each(row)?;
                count += 1;
            }
            if count != self.selected_members || hash_file(&path)? != self.memberships_sha256 {
                return Err(fail("membership count or digest mismatch"));
            }
            Ok(())
        })();
        finish(guard, result)
    }
}
