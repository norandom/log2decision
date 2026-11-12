//! Contract tests of triage (task 1.10; requirements 6.4 to 6.6, 6.9, 11.6, 11.7, 11.9,
//! 11.10, 12.6).
//!
//! The engine is tested against the `Scorer` contract with a scripted scorer (which can
//! abstain and can break the contract) and with the baseline scorer on the mini
//! fixture. What a result must be is computed here a second time, directly, from the
//! ratios and from `ConformalCalibrator::predict`. Everything writes only below the
//! system temporary directory.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use l2d_baseline::{BaselineScorer, BaselineSettings};
use l2d_core::polars::prelude::{Column, DataFrame, DataType};
use l2d_core::{
    InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat, LogRecord,
    LogsFrame, MemoryBudget, MemorySettings,
};
use l2d_decide::conformal::{
    CalibrationError, CalibrationPoint, CalibrationSettings, ConformalCalibrator,
    EXCHANGEABILITY_NOTE, LabelSet, Stratum, points_frame,
};
use l2d_decide::scorer::{Scorer, ScoringCost};
use l2d_decide::triage::{
    AbstainReason, ListUse, ScoredDownBy, TriageCalibration, TriageError, TriageList, TriageResult,
    TriageSettings, triage_with,
};
use l2d_decide::{ScorerKind, triage};

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const OTHER_REVISION: &str = "fedcba9876543210fedcba9876543210fedcba98";

/// A budget with a fixed free-memory reading, so the tests do not depend on the host.
fn budget_of(limit: Option<u64>, batch_rows: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: limit,
        max_batch_rows: batch_rows,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).expect("the budget is usable")
}

fn budget() -> MemoryBudget {
    budget_of(None, None)
}

type Log<'a> = (&'a str, Label, Option<&'a str>);

fn logs_of(rows: &[Log<'_>]) -> LogsFrame {
    let records: Vec<LogRecord> = rows
        .iter()
        .enumerate()
        .map(|(index, (text, label, group))| LogRecord {
            text: (*text).to_owned(),
            label: *label,
            group: group.map(str::to_owned),
            first_position: index as u64 + 1,
            occurrences: index as u32 % 3 + 1,
        })
        .collect();
    LogsFrame::from_records(&records).expect("the records make a logs frame")
}

/// How a scripted scorer breaks the contract of the scores frame.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Break {
    OneRowShort,
    OtherText,
    OtherName,
    RatioAboveOne,
    RatioNotANumber,
    NullRatioNotAbstained,
    NoAbstainedColumn,
    RatioAsDouble,
    EvidenceInBytes,
    EvidenceColumnNotStated,
    Fails,
}

/// A scorer that gives each text the ratio it was told to, or abstains on it.
struct Scripted {
    kind: ScorerKind,
    revision: String,
    ratios: BTreeMap<String, Option<f32>>,
    /// The number of rows of every batch it was asked to score.
    batches: Vec<usize>,
    broken: Option<Break>,
    cost: ScoringCost,
}

impl Scripted {
    fn new(kind: ScorerKind, ratios: &[(&str, Option<f32>)]) -> Self {
        Self {
            kind,
            revision: REVISION.into(),
            ratios: ratios
                .iter()
                .map(|(text, ratio)| ((*text).to_owned(), *ratio))
                .collect(),
            batches: Vec::new(),
            broken: None,
            cost: ScoringCost {
                fixed_bytes: 1_000,
                row_bytes: 100,
                evidence_row_bytes: 4,
                evidence_columns: 1,
            },
        }
    }

    fn baseline(ratios: &[(&str, Option<f32>)]) -> Self {
        Self::new(ScorerKind::Baseline, ratios)
    }

    fn broken(mut self, how: Break) -> Self {
        self.broken = Some(how);
        self
    }
}

impl Scorer for Scripted {
    fn kind(&self) -> ScorerKind {
        self.kind
    }

    fn revision(&self) -> &str {
        &self.revision
    }

    fn cost(&self, _logs: &LogsFrame) -> ScoringCost {
        self.cost
    }

    fn score(&mut self, logs: &LogsFrame) -> Result<DataFrame, L2dError> {
        self.batches.push(logs.len());
        if self.broken == Some(Break::Fails) {
            return Err(L2dError::Runtime {
                detail: "the scripted scorer fails".into(),
            });
        }
        let mut texts: Vec<String> = logs.records().map(|record| record.text).collect();
        let mut ratios: Vec<Option<f32>> = texts
            .iter()
            .map(|text| {
                *self
                    .ratios
                    .get(text)
                    .unwrap_or_else(|| panic!("no ratio scripted for '{text}'"))
            })
            .collect();
        let mut abstained: Vec<bool> = ratios.iter().map(Option::is_none).collect();
        let bytes: Vec<u32> = texts.iter().map(|text| text.len() as u32).collect();
        let mut name = self.kind.as_str();
        let mut bytes = bytes;
        match self.broken {
            Some(Break::OneRowShort) => {
                texts.pop();
                ratios.pop();
                abstained.pop();
                bytes.pop();
            }
            Some(Break::OtherText) => texts[0] = "another text".into(),
            Some(Break::OtherName) => name = "someone",
            Some(Break::RatioAboveOne) => ratios[0] = Some(1.5),
            Some(Break::RatioNotANumber) => ratios[0] = Some(f32::NAN),
            Some(Break::NullRatioNotAbstained) => {
                ratios[0] = None;
                abstained[0] = false;
            }
            _ => {}
        }
        let rows = texts.len();
        let mut columns = vec![
            Column::new("text".into(), texts),
            Column::new("scorer".into(), vec![name; rows]),
            Column::new("ratio".into(), ratios),
            Column::new("abstained".into(), abstained),
            Column::new("text_bytes".into(), bytes),
        ];
        match self.broken {
            Some(Break::NoAbstainedColumn) => {
                columns.remove(3);
            }
            Some(Break::RatioAsDouble) => {
                columns[2] = columns[2].cast(&DataType::Float64).unwrap();
            }
            Some(Break::EvidenceInBytes) => {
                columns[4] = Column::new("text_bytes".into(), vec![b"raw".as_slice(); rows]);
            }
            Some(Break::EvidenceColumnNotStated) => {
                columns.push(Column::new("more".into(), vec![1_u32; rows]));
            }
            _ => {}
        }
        Ok(DataFrame::new(columns).expect("the columns have one length"))
    }
}

/// One row of a triage frame.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    text: String,
    scorer: String,
    ratio: Option<f32>,
    abstained: bool,
    list: String,
    rank: Option<u32>,
    scored_down_by: Option<String>,
    abstain_reason: Option<String>,
    label_set: Option<String>,
    p_good: Option<f32>,
    p_bad: Option<f32>,
    good_stratum: Option<String>,
    bad_stratum: Option<String>,
}

fn rows_of(frame: &DataFrame) -> Vec<Row> {
    let text = |name: &str| -> Vec<Option<String>> {
        frame
            .column(name)
            .unwrap_or_else(|_| panic!("the triage frame has a column '{name}'"))
            .str()
            .unwrap_or_else(|_| panic!("column '{name}' is a String column"))
            .iter()
            .map(|value| value.map(str::to_owned))
            .collect()
    };
    let float = |name: &str| -> Vec<Option<f32>> {
        frame
            .column(name)
            .unwrap_or_else(|_| panic!("the triage frame has a column '{name}'"))
            .f32()
            .unwrap_or_else(|_| panic!("column '{name}' is a Float32 column"))
            .iter()
            .collect()
    };
    let texts = text("text");
    let scorers = text("scorer");
    let lists = text("list");
    let ratios = float("ratio");
    let abstained: Vec<Option<bool>> = frame
        .column("abstained")
        .unwrap()
        .bool()
        .unwrap()
        .iter()
        .collect();
    let ranks: Vec<Option<u32>> = frame
        .column("rank")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .collect();
    let by = text("scored_down_by");
    let reasons = text("abstain_reason");
    let sets = text("label_set");
    let (p_good, p_bad) = (float("p_good"), float("p_bad"));
    let (good_stratum, bad_stratum) = (text("good_stratum"), text("bad_stratum"));
    (0..frame.height())
        .map(|row| Row {
            text: texts[row].clone().expect("no text is null"),
            scorer: scorers[row].clone().expect("no scorer is null"),
            ratio: ratios[row],
            abstained: abstained[row].expect("no flag is null"),
            list: lists[row].clone().expect("no list is null"),
            rank: ranks[row],
            scored_down_by: by[row].clone(),
            abstain_reason: reasons[row].clone(),
            label_set: sets[row].clone(),
            p_good: p_good[row],
            p_bad: p_bad[row],
            good_stratum: good_stratum[row].clone(),
            bad_stratum: bad_stratum[row].clone(),
        })
        .collect()
}

fn texts_of(frame: &DataFrame) -> Vec<String> {
    frame
        .column("text")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|text| text.expect("no text is null").to_owned())
        .collect()
}

fn plain(primary: ScorerKind, review_budget: Option<u64>) -> TriageSettings<'static> {
    TriageSettings {
        primary,
        review_budget,
        calibration: None,
    }
}

fn calibrated(
    calibrator: &ConformalCalibrator,
    error_rate: f32,
    review_budget: Option<u64>,
) -> TriageSettings<'_> {
    TriageSettings {
        primary: ScorerKind::Baseline,
        review_budget,
        calibration: Some(TriageCalibration {
            calibrator,
            error_rate,
        }),
    }
}

fn run(
    logs: &LogsFrame,
    scorer: &mut dyn Scorer,
    settings: &TriageSettings<'_>,
) -> Result<TriageResult, TriageError> {
    triage_with(logs, &mut [scorer], settings, &mut budget())
}

/// What triage must give for one log, computed directly.
#[derive(Debug, Clone, PartialEq)]
struct Expected {
    list: &'static str,
    rank: Option<u32>,
    scored_down_by: Option<&'static str>,
    abstain_reason: Option<&'static str>,
    label_set: Option<&'static str>,
}

/// The placement of logs with these ratios (`None`: the scorer abstained) and groups,
/// worked out the plain way: place by label set where there is a calibration, order the
/// candidates for review by ratio with a stable sort, cut at the budget.
fn expected(
    ratios: &[Option<f32>],
    groups: &[Option<&str>],
    calibration: Option<(&ConformalCalibrator, f32)>,
    review_budget: Option<u64>,
) -> Vec<Expected> {
    let mut placed: Vec<Expected> = Vec::new();
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for (row, ratio) in ratios.iter().enumerate() {
        let mut one = Expected {
            list: "abstained",
            rank: None,
            scored_down_by: None,
            abstain_reason: None,
            label_set: None,
        };
        match (ratio, calibration) {
            (None, _) => one.abstain_reason = Some("scorer_abstained"),
            (Some(ratio), None) => candidates.push((row, *ratio)),
            (Some(ratio), Some((calibrator, error_rate))) => {
                let set = calibrator
                    .predict(*ratio, groups[row], error_rate)
                    .expect("the error rate is supported")
                    .set;
                one.label_set = Some(set.as_str());
                match set {
                    LabelSet::Bad => candidates.push((row, *ratio)),
                    LabelSet::Good => {
                        one.list = "scored_down";
                        one.scored_down_by = Some("calibration");
                    }
                    LabelSet::Both => one.abstain_reason = Some("uncertain"),
                    LabelSet::Neither => one.abstain_reason = Some("unlike_calibration"),
                }
            }
        }
        placed.push(one);
    }
    // Stable: equal ratios stay in the order of the logs frame.
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("ratios are numbers"));
    for (place, (row, _)) in candidates.iter().enumerate() {
        let rank = place as u64 + 1;
        let one = &mut placed[*row];
        one.rank = Some(rank as u32);
        if review_budget.is_none_or(|budget| rank <= budget) {
            one.list = "review";
        } else {
            one.list = "scored_down";
            one.scored_down_by = Some("budget_only");
        }
    }
    placed
}

fn assert_placed(found: &[Row], wanted: &[Expected]) {
    assert_eq!(found.len(), wanted.len(), "one row per log");
    for (row, (found, wanted)) in found.iter().zip(wanted).enumerate() {
        let got = Expected {
            list: match found.list.as_str() {
                "review" => "review",
                "scored_down" => "scored_down",
                "abstained" => "abstained",
                other => panic!("row {row}: the list '{other}' is none of the three"),
            },
            rank: found.rank,
            scored_down_by: found.scored_down_by.as_deref().map(|by| match by {
                "calibration" => "calibration",
                "budget_only" => "budget_only",
                other => panic!("row {row}: scored down by '{other}'"),
            }),
            abstain_reason: found.abstain_reason.as_deref().map(|reason| match reason {
                "scorer_abstained" => "scorer_abstained",
                "uncertain" => "uncertain",
                "unlike_calibration" => "unlike_calibration",
                other => panic!("row {row}: abstained because '{other}'"),
            }),
            label_set: found.label_set.as_deref().map(|set| match set {
                "good" => "good",
                "bad" => "bad",
                "both" => "both",
                "neither" => "neither",
                other => panic!("row {row}: the label set '{other}'"),
            }),
        };
        assert_eq!(&got, wanted, "row {row} ({found:?})");
    }
}

/// SplitMix64: a fixed sequence, so every run of a test sees the same draws.
struct Draws(u64);

impl Draws {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, limit: u64) -> u64 {
        self.next() % limit
    }

    /// A ratio around `centre`, on a grid of 1/64 so that ties are frequent.
    fn ratio(&mut self, centre: f64, spread: f64) -> f32 {
        let sum: f64 = (0..12).map(|_| self.unit()).sum::<f64>() - 6.0;
        let value = (centre + spread * sum).clamp(0.0, 1.0);
        ((value * 64.0).round() / 64.0) as f32
    }
}

const GROUPS: [&str; 3] = ["/a", "/b", "/c"];

/// A calibration on synthetic ratios. The groups `/a` and `/b` hold enough logs of each
/// label for lists of their own, `/c` only for good, and logs without a group or of
/// another group use the pooled lists. In `/a` the labels lie far apart (ratios between
/// them are unlike either), in `/b` they overlap (ratios between them are uncertain).
fn synthetic_calibrator(revision: &str) -> ConformalCalibrator {
    let mut draws = Draws(1_010);
    let mut points = Vec::new();
    for (group, good, bad, apart) in [
        (Some("/a"), 150, 150, 0.25),
        (Some("/b"), 150, 150, 0.08),
        (Some("/c"), 150, 20, 0.2),
        (None, 60, 60, 0.2),
    ] {
        for (label, count, centre) in [
            (Label::Good, good, 0.5 - apart),
            (Label::Bad, bad, 0.5 + apart),
        ] {
            for _ in 0..count {
                points.push(CalibrationPoint {
                    ratio: draws.ratio(centre, 0.12),
                    label,
                    group: group.map(str::to_owned),
                });
            }
        }
    }
    ConformalCalibrator::fit_with(
        &points_frame(&points).unwrap(),
        ScorerKind::Baseline,
        revision,
        &CalibrationSettings { min_per_group: 100 },
        &budget(),
    )
    .expect("the synthetic points calibrate")
}

/// `count` logs with drawn ratios, groups (the three calibrated ones, one the
/// calibration has never seen, and none) and, with `abstain_every`, abstentions.
struct Population {
    texts: Vec<String>,
    groups: Vec<Option<&'static str>>,
    ratios: Vec<Option<f32>>,
}

impl Population {
    fn draw(seed: u64, count: usize, abstain_every: Option<u64>) -> Self {
        let mut draws = Draws(seed);
        let mut population = Self {
            texts: Vec::new(),
            groups: Vec::new(),
            ratios: Vec::new(),
        };
        for index in 0..count {
            population.texts.push(format!("GET /log/{seed}/{index}"));
            population.groups.push(match draws.below(5) {
                0 => Some(GROUPS[0]),
                1 => Some(GROUPS[1]),
                2 => Some(GROUPS[2]),
                3 => Some("/never-calibrated"),
                _ => None,
            });
            let centre = if draws.below(2) == 0 { 0.3 } else { 0.7 };
            let ratio = draws.ratio(centre, 0.2);
            let abstains = abstain_every.is_some_and(|every| draws.below(every) == 0);
            population.ratios.push((!abstains).then_some(ratio));
        }
        population
    }

    fn logs(&self) -> LogsFrame {
        let rows: Vec<Log<'_>> = self
            .texts
            .iter()
            .zip(&self.groups)
            .map(|(text, group)| (text.as_str(), Label::Unknown, *group))
            .collect();
        logs_of(&rows)
    }

    fn scorer(&self) -> Scripted {
        let ratios: Vec<(&str, Option<f32>)> = self
            .texts
            .iter()
            .zip(&self.ratios)
            .map(|(text, ratio)| (text.as_str(), *ratio))
            .collect();
        Scripted::baseline(&ratios)
    }
}

/// A project root in the system temporary directory, removed when dropped.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-decide-triage-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("the scratch root can be created");
        Self { root }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

// ---------------------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------------------

#[test]
fn lists_and_reasons_have_fixed_names() {
    assert_eq!(TriageList::Review.as_str(), "review");
    assert_eq!(TriageList::ScoredDown.as_str(), "scored_down");
    assert_eq!(TriageList::Abstained.as_str(), "abstained");
    assert_eq!(AbstainReason::ScorerAbstained.as_str(), "scorer_abstained");
    assert_eq!(AbstainReason::Uncertain.as_str(), "uncertain");
    assert_eq!(
        AbstainReason::UnlikeCalibration.as_str(),
        "unlike_calibration"
    );
    assert_eq!(ScoredDownBy::Calibration.as_str(), "calibration");
    assert_eq!(ScoredDownBy::BudgetOnly.as_str(), "budget_only");
    assert_eq!(
        serde_json::to_string(&AbstainReason::UnlikeCalibration).unwrap(),
        "\"unlike_calibration\""
    );
    assert_eq!(
        serde_json::to_string(&ScoredDownBy::BudgetOnly).unwrap(),
        "\"budget_only\""
    );
}

// ---------------------------------------------------------------------------------------
// Ranking, ties, budget, abstention (requirements 6.4, 6.5, 6.6)
// ---------------------------------------------------------------------------------------

#[test]
fn without_a_budget_every_scored_log_is_ranked_for_review_highest_ratio_first() {
    let logs = logs_of(&[
        ("a", Label::Unknown, None),
        ("b", Label::Good, Some("/x")),
        ("c", Label::Bad, None),
        ("d", Label::Unknown, Some("/y")),
    ]);
    let mut scorer = Scripted::baseline(&[
        ("a", Some(0.25)),
        ("b", Some(0.75)),
        ("c", Some(1.0)),
        ("d", Some(0.0)),
    ]);
    let result = run(&logs, &mut scorer, &plain(ScorerKind::Baseline, None)).unwrap();

    // The triage frame has one row per log, in the order of the logs frame, and starts
    // with the columns of the logs frame.
    let frame = result.frame();
    assert_eq!(texts_of(frame), ["a", "b", "c", "d"]);
    let names: Vec<&str> = frame
        .get_column_names()
        .into_iter()
        .map(|name| name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "text",
            "label",
            "group",
            "first_position",
            "occurrences",
            "scorer",
            "ratio",
            "abstained",
            "list",
            "rank",
            "scored_down_by",
            "abstain_reason",
            "label_set",
            "p_good",
            "p_bad",
            "good_stratum",
            "bad_stratum",
            "text_bytes",
        ]
    );
    for name in ["text", "label", "group", "first_position", "occurrences"] {
        assert!(
            frame
                .column(name)
                .unwrap()
                .as_materialized_series()
                .equals_missing(logs.frame().column(name).unwrap().as_materialized_series()),
            "column '{name}' is the logs frame's"
        );
    }

    let rows = rows_of(frame);
    let ranks: Vec<Option<u32>> = rows.iter().map(|row| row.rank).collect();
    assert_eq!(ranks, [Some(3), Some(2), Some(1), Some(4)]);
    for row in &rows {
        assert_eq!(row.list, "review");
        assert_eq!(row.scorer, "baseline", "requirement 6.6");
        assert!(!row.abstained);
        assert_eq!(row.scored_down_by, None);
        assert_eq!(row.abstain_reason, None);
        // Without a calibration there is no label set.
        assert_eq!((&row.label_set, row.p_good, row.p_bad), (&None, None, None));
        assert_eq!((&row.good_stratum, &row.bad_stratum), (&None, &None));
    }
    let ratios: Vec<Option<f32>> = rows.iter().map(|row| row.ratio).collect();
    assert_eq!(ratios, [Some(0.25), Some(0.75), Some(1.0), Some(0.0)]);

    // The review list is the ranking; the other lists are empty.
    assert_eq!(texts_of(&result.review().unwrap()), ["c", "b", "a", "d"]);
    assert_eq!(result.scored_down().unwrap().height(), 0);
    assert_eq!(result.abstained().unwrap().height(), 0);
    assert_eq!(result.primary(), ScorerKind::Baseline);
    assert_eq!(result.scorers(), [ScorerKind::Baseline]);
    assert_eq!(result.review_budget(), None);
    assert!(result.calibration().is_none());
    let counts = result.counts();
    assert_eq!((counts.logs, counts.review), (4, 4));
    assert_eq!((counts.scored_down(), counts.abstained()), (0, 0));
}

#[test]
fn equal_ratios_keep_the_order_of_the_logs_frame() {
    let logs = logs_of(&[
        ("first", Label::Unknown, None),
        ("second", Label::Unknown, None),
        ("third", Label::Unknown, None),
        ("fourth", Label::Unknown, None),
        ("fifth", Label::Unknown, None),
        ("sixth", Label::Unknown, None),
    ]);
    let scripted = [
        ("first", Some(0.5)),
        ("second", Some(0.0)),
        ("third", Some(0.5)),
        ("fourth", Some(0.75)),
        // The two zeros are one ratio.
        ("fifth", Some(-0.0)),
        ("sixth", Some(0.5)),
    ];
    let mut scorer = Scripted::baseline(&scripted);
    let result = run(&logs, &mut scorer, &plain(ScorerKind::Baseline, None)).unwrap();
    assert_eq!(
        texts_of(&result.review().unwrap()),
        ["fourth", "first", "third", "sixth", "second", "fifth"]
    );

    // The same logs the other way round: the tied ones change places with the input.
    let reversed: Vec<Log<'_>> = ["sixth", "fifth", "fourth", "third", "second", "first"]
        .into_iter()
        .map(|text| (text, Label::Unknown, None))
        .collect();
    let mut scorer = Scripted::baseline(&scripted);
    let result = run(
        &logs_of(&reversed),
        &mut scorer,
        &plain(ScorerKind::Baseline, None),
    )
    .unwrap();
    assert_eq!(
        texts_of(&result.review().unwrap()),
        ["fourth", "sixth", "third", "first", "fifth", "second"]
    );

    // A tie at the budget is cut by the same rule: the earlier log goes to review.
    let mut scorer = Scripted::baseline(&scripted);
    let result = run(&logs, &mut scorer, &plain(ScorerKind::Baseline, Some(2))).unwrap();
    assert_eq!(texts_of(&result.review().unwrap()), ["fourth", "first"]);
    assert_eq!(
        texts_of(&result.scored_down().unwrap()),
        ["third", "sixth", "second", "fifth"]
    );
}

#[test]
fn the_budget_sends_the_top_logs_to_review_and_marks_the_rest_scored_down() {
    let population = Population::draw(11, 200, None);
    let logs = population.logs();
    for review_budget in [0, 1, 37, 199, 200, 201, u64::MAX] {
        let mut scorer = population.scorer();
        let result = run(
            &logs,
            &mut scorer,
            &plain(ScorerKind::Baseline, Some(review_budget)),
        )
        .unwrap();
        let rows = rows_of(result.frame());
        assert_placed(
            &rows,
            &expected(
                &population.ratios,
                &population.groups,
                None,
                Some(review_budget),
            ),
        );
        let in_review = review_budget.min(200);
        let counts = result.counts();
        assert_eq!(counts.review, in_review, "budget {review_budget}");
        assert_eq!(counts.scored_down_by_budget, 200 - in_review);
        assert_eq!(counts.scored_down_by_calibration, 0);
        assert_eq!(counts.abstained(), 0);
        assert_eq!(result.review_budget(), Some(review_budget));

        // Requirement 6.5: the review list holds the highest ratios, in order; every
        // other log is marked scored down, and only the budget did it.
        let review = rows_of(&result.review().unwrap());
        assert_eq!(review.len() as u64, in_review);
        assert!(
            review
                .windows(2)
                .all(|pair| pair[0].ratio >= pair[1].ratio && pair[0].rank < pair[1].rank)
        );
        let down = rows_of(&result.scored_down().unwrap());
        assert_eq!(down.len() as u64, 200 - in_review);
        assert!(
            down.iter()
                .all(|row| row.scored_down_by.as_deref() == Some("budget_only"))
        );
        assert!(down.windows(2).all(|pair| pair[0].rank < pair[1].rank));
        if let (Some(last), Some(first)) = (review.last(), down.first()) {
            assert!(last.ratio >= first.ratio);
        }
    }
}

#[test]
fn logs_the_scorer_abstained_on_are_listed_separately_and_take_none_of_the_budget() {
    let logs = logs_of(&[
        ("a", Label::Unknown, None),
        ("b", Label::Unknown, None),
        ("c", Label::Unknown, None),
        ("d", Label::Unknown, None),
        ("e", Label::Unknown, None),
    ]);
    let mut scorer = Scripted::new(
        ScorerKind::Neighbour,
        &[
            ("a", None),
            ("b", Some(0.25)),
            ("c", Some(0.75)),
            ("d", None),
            ("e", Some(0.5)),
        ],
    );
    let result = run(&logs, &mut scorer, &plain(ScorerKind::Neighbour, Some(2))).unwrap();
    assert_eq!(texts_of(&result.review().unwrap()), ["c", "e"]);
    assert_eq!(texts_of(&result.scored_down().unwrap()), ["b"]);
    // In the order of the logs frame.
    assert_eq!(texts_of(&result.abstained().unwrap()), ["a", "d"]);
    for row in rows_of(&result.abstained().unwrap()) {
        assert!(row.abstained);
        assert_eq!(row.ratio, None, "an abstention has no ratio");
        assert_eq!(row.rank, None);
        assert_eq!(row.abstain_reason.as_deref(), Some("scorer_abstained"));
        assert_eq!(row.scored_down_by, None);
        assert_eq!(row.scorer, "neighbour");
    }
    let counts = result.counts();
    assert_eq!(counts.scorer_abstained, 2);
    assert_eq!((counts.review, counts.scored_down_by_budget), (2, 1));
}

#[test]
fn an_empty_set_of_logs_gives_three_empty_lists() {
    let logs = LogsFrame::empty();
    let mut scorer = Scripted::baseline(&[]);
    let result = run(&logs, &mut scorer, &plain(ScorerKind::Baseline, Some(3))).unwrap();
    assert_eq!(result.frame().height(), 0);
    assert_eq!(result.counts().logs, 0);
    assert_eq!(result.review().unwrap().height(), 0);
    assert_eq!(result.scored_down().unwrap().height(), 0);
    assert_eq!(result.abstained().unwrap().height(), 0);
    assert!(result.frame().column("list").is_ok());
}

// ---------------------------------------------------------------------------------------
// Several scorers (requirement 6.6)
// ---------------------------------------------------------------------------------------

#[test]
fn every_scorer_is_named_and_the_primary_one_ranks() {
    let logs = logs_of(&[
        ("a", Label::Unknown, None),
        ("b", Label::Unknown, None),
        ("c", Label::Unknown, None),
    ]);
    let baseline = [("a", Some(0.25)), ("b", Some(0.5)), ("c", Some(0.75))];
    let neighbour = [("a", Some(1.0)), ("b", None), ("c", Some(0.0))];
    for primary in [ScorerKind::Baseline, ScorerKind::Neighbour] {
        let mut first = Scripted::baseline(&baseline);
        let mut second = Scripted::new(ScorerKind::Neighbour, &neighbour);
        let result = triage_with(
            &logs,
            &mut [&mut first, &mut second],
            &plain(primary, None),
            &mut budget(),
        )
        .unwrap();
        assert_eq!(result.primary(), primary);
        assert_eq!(
            result.scorers(),
            [ScorerKind::Baseline, ScorerKind::Neighbour]
        );
        let rows = rows_of(result.frame());
        assert!(rows.iter().all(|row| row.scorer == primary.as_str()));
        let other = |name: &str| -> Vec<Option<f32>> {
            result
                .frame()
                .column(name)
                .unwrap_or_else(|_| panic!("the frame has '{name}'"))
                .f32()
                .unwrap()
                .iter()
                .collect()
        };
        if primary == ScorerKind::Baseline {
            assert_eq!(texts_of(&result.review().unwrap()), ["c", "b", "a"]);
            assert_eq!(other("ratio_neighbour"), [Some(1.0), None, Some(0.0)]);
            assert!(result.frame().column("ratio_baseline").is_err());
            assert_eq!(result.counts().abstained(), 0);
        } else {
            assert_eq!(texts_of(&result.review().unwrap()), ["a", "c"]);
            assert_eq!(texts_of(&result.abstained().unwrap()), ["b"]);
            assert_eq!(other("ratio_baseline"), [Some(0.25), Some(0.5), Some(0.75)]);
            assert!(result.frame().column("ratio_neighbour").is_err());
        }
    }
}

#[test]
fn scorers_that_cannot_be_used_as_asked_are_refused_before_anything_is_scored() {
    let logs = logs_of(&[("a", Label::Unknown, None)]);
    let ratios = [("a", Some(0.5))];

    let error = triage_with(
        &logs,
        &mut [],
        &plain(ScorerKind::Baseline, None),
        &mut budget(),
    )
    .unwrap_err();
    assert!(
        matches!(error, TriageError::InvalidScorers { .. }),
        "{error:?}"
    );
    assert_eq!(L2dError::from(error).exit_code(), 2);

    let mut only = Scripted::baseline(&ratios);
    let error = run(&logs, &mut only, &plain(ScorerKind::Neighbour, None)).unwrap_err();
    assert!(
        matches!(&error, TriageError::InvalidScorers { detail } if detail.contains("neighbour")),
        "{error:?}"
    );
    assert!(only.batches.is_empty(), "nothing was scored");

    let (mut one, mut two) = (Scripted::baseline(&ratios), Scripted::baseline(&ratios));
    let error = triage_with(
        &logs,
        &mut [&mut one, &mut two],
        &plain(ScorerKind::Baseline, None),
        &mut budget(),
    )
    .unwrap_err();
    assert!(
        matches!(&error, TriageError::InvalidScorers { detail } if detail.contains("baseline")),
        "{error:?}"
    );
    assert!(one.batches.is_empty() && two.batches.is_empty());
}

#[test]
fn a_scores_frame_that_breaks_the_contract_is_refused_naming_the_scorer() {
    let logs = logs_of(&[
        ("a", Label::Unknown, None),
        ("b", Label::Unknown, None),
        ("c", Label::Unknown, None),
    ]);
    let ratios = [("a", Some(0.25)), ("b", Some(0.5)), ("c", None)];
    for (how, word) in [
        (Break::OneRowShort, "rows"),
        (Break::OtherText, "text"),
        (Break::OtherName, "someone"),
        (Break::RatioAboveOne, "1.5"),
        (Break::RatioNotANumber, "NaN"),
        (Break::NullRatioNotAbstained, "ratio"),
        (Break::NoAbstainedColumn, "abstained"),
        (Break::RatioAsDouble, "ratio"),
        // Evidence is words, flags or numbers, and as many columns as the cost states.
        (Break::EvidenceInBytes, "String, Boolean or numbers"),
        (
            Break::EvidenceColumnNotStated,
            "2 evidence columns where its cost states 1",
        ),
    ] {
        let mut scorer = Scripted::baseline(&ratios).broken(how);
        let error = run(&logs, &mut scorer, &plain(ScorerKind::Baseline, None)).unwrap_err();
        match &error {
            TriageError::Scores { scorer, detail } => {
                assert_eq!(*scorer, "baseline", "{how:?}");
                assert!(detail.contains(word), "{how:?}: {detail}");
            }
            other => panic!("{how:?}: expected a scores refusal, got {other:?}"),
        }
        assert_eq!(L2dError::from(error).exit_code(), 5, "{how:?}");
    }

    // A scorer's own failure is passed on as it is.
    let mut scorer = Scripted::baseline(&ratios).broken(Break::Fails);
    let error = run(&logs, &mut scorer, &plain(ScorerKind::Baseline, None)).unwrap_err();
    assert_eq!(
        error,
        TriageError::Scorer(L2dError::Runtime {
            detail: "the scripted scorer fails".into()
        })
    );
}

// ---------------------------------------------------------------------------------------
// Every log in exactly one list; nothing is altered (task postcondition, requirement 6.9)
// ---------------------------------------------------------------------------------------

/// The postcondition on one result: the frame has the logs of the input in their order,
/// each with exactly one of the three lists, and the three lists together hold every
/// log once.
fn assert_partition(logs: &LogsFrame, result: &TriageResult) {
    let input = texts_of(logs.frame());
    let rows = rows_of(result.frame());
    assert_eq!(
        rows.iter().map(|row| row.text.clone()).collect::<Vec<_>>(),
        input,
        "one row per log, in the order of the logs frame"
    );
    let (review, down, abstained) = (
        result.review().unwrap(),
        result.scored_down().unwrap(),
        result.abstained().unwrap(),
    );
    let mut seen: BTreeMap<String, &str> = BTreeMap::new();
    for (list, frame) in [
        ("review", &review),
        ("scored_down", &down),
        ("abstained", &abstained),
    ] {
        assert_eq!(
            frame.get_column_names(),
            result.frame().get_column_names(),
            "the lists have the columns of the triage frame"
        );
        for row in rows_of(frame) {
            assert_eq!(
                row.list, list,
                "'{}' is listed where its row says",
                row.text
            );
            let before = seen.insert(row.text.clone(), list);
            assert_eq!(before, None, "'{}' is in two lists", row.text);
        }
    }
    assert_eq!(
        seen.keys().cloned().collect::<BTreeSet<_>>(),
        input.iter().cloned().collect::<BTreeSet<_>>(),
        "every log is in a list"
    );
    assert_eq!(
        seen.len(),
        input.len(),
        "the texts of the input are distinct"
    );
    for row in &rows {
        // What goes with each list, and nothing else.
        match row.list.as_str() {
            "review" => {
                assert!(row.rank.is_some() && row.ratio.is_some(), "{row:?}");
                assert!(
                    row.scored_down_by.is_none() && row.abstain_reason.is_none(),
                    "{row:?}"
                );
            }
            "scored_down" => {
                assert!(
                    row.ratio.is_some() && row.abstain_reason.is_none(),
                    "{row:?}"
                );
                match row.scored_down_by.as_deref() {
                    Some("budget_only") => assert!(row.rank.is_some(), "{row:?}"),
                    Some("calibration") => {
                        assert!(row.rank.is_none(), "{row:?}");
                        assert_eq!(row.label_set.as_deref(), Some("good"), "{row:?}");
                    }
                    other => panic!("scored down by {other:?}: {row:?}"),
                }
            }
            "abstained" => {
                assert!(
                    row.rank.is_none() && row.scored_down_by.is_none(),
                    "{row:?}"
                );
                assert!(row.abstain_reason.is_some(), "{row:?}");
            }
            other => panic!("the list '{other}' is none of the three"),
        }
    }
    let counts = result.counts();
    assert_eq!(counts.logs as usize, input.len());
    assert_eq!(
        (counts.review, counts.scored_down(), counts.abstained()),
        (
            review.height() as u64,
            down.height() as u64,
            abstained.height() as u64
        )
    );
    assert_eq!(
        counts.review + counts.scored_down() + counts.abstained(),
        counts.logs
    );
}

#[test]
fn every_log_lands_in_exactly_one_list_and_the_logs_are_not_altered() {
    let calibrator = synthetic_calibrator(REVISION);
    let mut cases = 0;
    for seed in 1..=6_u64 {
        for count in [1_usize, 2, 17, 300] {
            for abstain_every in [None, Some(4), Some(1)] {
                let population = Population::draw(seed * 100 + count as u64, count, abstain_every);
                let logs = population.logs();
                let before = logs.frame().clone();
                for review_budget in [None, Some(0), Some(5), Some(count as u64 + 1)] {
                    for with_calibration in [false, true] {
                        let settings = if with_calibration {
                            calibrated(&calibrator, 0.1, review_budget)
                        } else {
                            plain(ScorerKind::Baseline, review_budget)
                        };
                        let mut scorer = population.scorer();
                        let result = run(&logs, &mut scorer, &settings).unwrap();
                        assert_partition(&logs, &result);
                        assert_placed(
                            &rows_of(result.frame()),
                            &expected(
                                &population.ratios,
                                &population.groups,
                                with_calibration.then_some((&calibrator, 0.1)),
                                review_budget,
                            ),
                        );
                        cases += 1;
                    }
                }
                // Requirement 6.9: the logs are what they were.
                assert!(
                    logs.frame().equals_missing(&before),
                    "triage altered the logs frame"
                );
            }
        }
    }
    assert_eq!(cases, 6 * 4 * 3 * 4 * 2);
}

// ---------------------------------------------------------------------------------------
// With a calibration (requirements 11.6, 11.7, 11.9)
// ---------------------------------------------------------------------------------------

#[test]
fn with_a_calibration_logs_are_placed_by_their_label_set() {
    const E: f32 = 0.1;
    let calibrator = synthetic_calibrator(REVISION);
    let population = Population::draw(77, 1_500, Some(40));
    let logs = population.logs();
    let mut scorer = population.scorer();
    let result = run(&logs, &mut scorer, &calibrated(&calibrator, E, None)).unwrap();
    let rows = rows_of(result.frame());
    assert_placed(
        &rows,
        &expected(
            &population.ratios,
            &population.groups,
            Some((&calibrator, E)),
            None,
        ),
    );

    // Each row carries the decision of the calibration: set, p-values and which list
    // each p-value came from.
    let mut sets = [0_u64; 4];
    let (mut good_use, mut bad_use) = (ListUse::default(), ListUse::default());
    for ((row, ratio), group) in rows.iter().zip(&population.ratios).zip(&population.groups) {
        let Some(ratio) = ratio else {
            assert_eq!(
                (&row.label_set, row.p_good, row.p_bad),
                (&None, None, None),
                "an abstention is not given a label set"
            );
            assert_eq!((&row.good_stratum, &row.bad_stratum), (&None, &None));
            continue;
        };
        let decision = calibrator.predict(*ratio, *group, E).unwrap();
        assert_eq!(row.label_set.as_deref(), Some(decision.set.as_str()));
        assert_eq!(
            (row.p_good, row.p_bad),
            (Some(decision.p_good), Some(decision.p_bad))
        );
        let name = |stratum: &Stratum, used: &mut ListUse| match stratum {
            Stratum::Pooled => {
                used.pooled += 1;
                "pooled"
            }
            Stratum::Group(own) => {
                assert_eq!(Some(own.as_str()), *group);
                used.group += 1;
                "group"
            }
        };
        assert_eq!(
            row.good_stratum.as_deref(),
            Some(name(&decision.good_stratum, &mut good_use))
        );
        assert_eq!(
            row.bad_stratum.as_deref(),
            Some(name(&decision.bad_stratum, &mut bad_use))
        );
        sets[match decision.set {
            LabelSet::Good => 0,
            LabelSet::Bad => 1,
            LabelSet::Both => 2,
            LabelSet::Neither => 3,
        }] += 1;
    }
    // The draws put logs in every list, so each rule is exercised.
    assert!(sets.iter().all(|count| *count >= 10), "{sets:?}");

    // Requirement 11.6 in the lists: review is the bad sets, ranked; scored down is the
    // good sets, by calibration; abstained says why.
    let review = rows_of(&result.review().unwrap());
    assert_eq!(review.len() as u64, sets[1]);
    assert!(
        review
            .iter()
            .all(|row| row.label_set.as_deref() == Some("bad"))
    );
    assert!(review.windows(2).all(|pair| pair[0].ratio >= pair[1].ratio));
    let down = rows_of(&result.scored_down().unwrap());
    assert_eq!(down.len() as u64, sets[0]);
    assert!(down.iter().all(|row| {
        row.label_set.as_deref() == Some("good")
            && row.scored_down_by.as_deref() == Some("calibration")
    }));
    assert!(down.windows(2).all(|pair| pair[0].ratio >= pair[1].ratio));
    let abstained = rows_of(&result.abstained().unwrap());
    let reason = |wanted: &str| {
        abstained
            .iter()
            .filter(|row| row.abstain_reason.as_deref() == Some(wanted))
            .count() as u64
    };
    assert_eq!(reason("uncertain"), sets[2]);
    assert_eq!(reason("unlike_calibration"), sets[3]);
    assert!(
        abstained
            .iter()
            .filter(|row| row.abstain_reason.as_deref() == Some("uncertain"))
            .all(|row| row.label_set.as_deref() == Some("both"))
    );
    assert!(
        abstained
            .iter()
            .filter(|row| row.abstain_reason.as_deref() == Some("unlike_calibration"))
            .all(|row| row.label_set.as_deref() == Some("neither"))
    );
    let scorer_abstained = population.ratios.iter().filter(|r| r.is_none()).count() as u64;
    assert!(scorer_abstained >= 10);
    assert_eq!(reason("scorer_abstained"), scorer_abstained);

    // What the result says about the calibration (requirement 11.9, owner decision of
    // 2026-11-12): the note, the error rate, and how many logs used a group's own list
    // and how many the pooled list, per label.
    let used = result
        .calibration()
        .expect("the result names the calibration");
    assert_eq!(used.note, EXCHANGEABILITY_NOTE);
    assert!(used.note.contains("resemble the calibration logs"));
    assert_eq!(used.error_rate, E);
    assert_eq!(used.revision, calibrator.revision());
    assert_eq!(used.held_bytes, calibrator.held_bytes());
    assert_eq!((used.good, used.bad), (good_use, bad_use));
    assert_eq!(
        [used.set_good, used.set_bad, used.set_both, used.set_neither],
        sets
    );
    let judged = 1_500 - scorer_abstained;
    assert_eq!(used.good.group + used.good.pooled, judged);
    assert_eq!(used.bad.group + used.bad.pooled, judged);
    // `/c` has a list of its own for good only, so more logs use a group list for good.
    assert!(used.good.group > used.bad.group && used.bad.group > 0);
    assert!(used.good.pooled > 0);
    let counts = result.counts();
    assert_eq!(
        (
            counts.review,
            counts.scored_down_by_calibration,
            counts.scored_down_by_budget,
            counts.uncertain,
            counts.unlike_calibration,
            counts.scorer_abstained
        ),
        (sets[1], sets[0], 0, sets[2], sets[3], scorer_abstained)
    );
}

#[test]
fn beyond_the_budget_logs_are_scored_down_by_the_budget_only_and_each_says_which_applied() {
    const E: f32 = 0.1;
    let calibrator = synthetic_calibrator(REVISION);
    let population = Population::draw(78, 800, None);
    let logs = population.logs();
    let mut scorer = population.scorer();
    let unlimited = run(&logs, &mut scorer, &calibrated(&calibrator, E, None)).unwrap();
    let bad_sets = unlimited.counts().review;
    assert!(bad_sets > 50, "{bad_sets} logs have the set bad");

    for review_budget in [0, 10, bad_sets - 1, bad_sets, bad_sets + 100] {
        let mut scorer = population.scorer();
        let result = run(
            &logs,
            &mut scorer,
            &calibrated(&calibrator, E, Some(review_budget)),
        )
        .unwrap();
        assert_placed(
            &rows_of(result.frame()),
            &expected(
                &population.ratios,
                &population.groups,
                Some((&calibrator, E)),
                Some(review_budget),
            ),
        );
        let in_review = review_budget.min(bad_sets);
        let counts = result.counts();
        assert_eq!(counts.review, in_review);
        assert_eq!(counts.scored_down_by_budget, bad_sets - in_review);
        // The budget counts the review ranking only: what calibration scored down and
        // what is abstained on stays where it was.
        assert_eq!(
            counts.scored_down_by_calibration,
            unlimited.counts().scored_down_by_calibration
        );
        assert_eq!(counts.abstained(), unlimited.counts().abstained());

        // Requirement 11.7: every scored-down log says which applied. The budget's are
        // listed first, in rank order, and their label set is bad; the calibration's
        // follow and their label set is good.
        let down = rows_of(&result.scored_down().unwrap());
        let by_budget = (bad_sets - in_review) as usize;
        for (place, row) in down.iter().enumerate() {
            if place < by_budget {
                assert_eq!(row.scored_down_by.as_deref(), Some("budget_only"));
                assert_eq!(row.label_set.as_deref(), Some("bad"));
                assert_eq!(row.rank, Some((in_review + place as u64 + 1) as u32));
            } else {
                assert_eq!(row.scored_down_by.as_deref(), Some("calibration"));
                assert_eq!(row.label_set.as_deref(), Some("good"));
                assert_eq!(row.rank, None);
            }
        }
    }
}

#[test]
fn a_log_of_an_endpoint_the_calibration_does_not_cover_is_placed_like_any_other() {
    const E: f32 = 0.1;
    let calibrator = synthetic_calibrator(REVISION);
    // The same ratio on a calibrated endpoint, an endpoint never calibrated, and
    // without a group.
    let logs = logs_of(&[
        ("covered low", Label::Unknown, Some("/a")),
        ("uncovered low", Label::Unknown, Some("/never-calibrated")),
        ("no group low", Label::Unknown, None),
        ("covered high", Label::Unknown, Some("/a")),
        ("uncovered high", Label::Unknown, Some("/never-calibrated")),
        ("no group high", Label::Unknown, None),
    ]);
    let ratios = [
        Some(0.125),
        Some(0.125),
        Some(0.125),
        Some(0.875),
        Some(0.875),
        Some(0.875),
    ];
    let scripted: Vec<(&str, Option<f32>)> = [
        "covered low",
        "uncovered low",
        "no group low",
        "covered high",
        "uncovered high",
        "no group high",
    ]
    .into_iter()
    .zip(ratios)
    .collect();
    let mut scorer = Scripted::baseline(&scripted);
    let result = run(&logs, &mut scorer, &calibrated(&calibrator, E, None)).unwrap();
    let rows = rows_of(result.frame());
    let groups = [
        Some("/a"),
        Some("/never-calibrated"),
        None,
        Some("/a"),
        Some("/never-calibrated"),
        None,
    ];
    assert_placed(
        &rows,
        &expected(&ratios, &groups, Some((&calibrator, E)), None),
    );
    // The uncovered endpoint is judged on the pooled lists, exactly as a log without a
    // group is, and lands where that log lands.
    for (uncovered, no_group) in [(1, 2), (4, 5)] {
        assert_eq!(rows[uncovered].good_stratum.as_deref(), Some("pooled"));
        assert_eq!(rows[uncovered].bad_stratum.as_deref(), Some("pooled"));
        assert_eq!(rows[uncovered].list, rows[no_group].list);
        assert_eq!(rows[uncovered].label_set, rows[no_group].label_set);
        assert_eq!(
            (rows[uncovered].p_good, rows[uncovered].p_bad),
            (rows[no_group].p_good, rows[no_group].p_bad)
        );
    }
    assert_eq!(rows[0].good_stratum.as_deref(), Some("group"));
    // The low ratio is scored down by calibration and the high one ranked for review,
    // on the uncovered endpoint as elsewhere.
    assert_eq!(rows[1].list, "scored_down");
    assert_eq!(rows[4].list, "review");
    let used = result.calibration().unwrap();
    assert_eq!(
        used.good,
        ListUse {
            group: 2,
            pooled: 4
        }
    );
    assert_eq!(
        used.bad,
        ListUse {
            group: 2,
            pooled: 4
        }
    );
}

#[test]
fn a_calibration_that_does_not_fit_is_refused_before_anything_is_scored() {
    let population = Population::draw(5, 20, None);
    let logs = population.logs();

    // Made for another revision of the scorer.
    let other = synthetic_calibrator(OTHER_REVISION);
    let mut scorer = population.scorer();
    let error = run(&logs, &mut scorer, &calibrated(&other, 0.1, None)).unwrap_err();
    assert!(
        matches!(
            &error,
            TriageError::Calibration(CalibrationError::OtherScorer { .. })
        ),
        "{error:?}"
    );
    assert_eq!(L2dError::from(error).exit_code(), 4);
    assert!(scorer.batches.is_empty(), "nothing was scored");

    // The calibration judges the primary scorer, whichever else scores.
    let calibrator = synthetic_calibrator(REVISION);
    let mut neighbour = Scripted::new(ScorerKind::Neighbour, &[]);
    let mut baseline = population.scorer();
    let settings = TriageSettings {
        primary: ScorerKind::Neighbour,
        ..calibrated(&calibrator, 0.1, None)
    };
    let error = triage_with(
        &logs,
        &mut [&mut baseline, &mut neighbour],
        &settings,
        &mut budget(),
    )
    .unwrap_err();
    assert!(
        matches!(
            &error,
            TriageError::Calibration(CalibrationError::OtherScorer { kind, .. }) if *kind == "neighbour"
        ),
        "{error:?}"
    );

    // An error rate the calibration does not support (the shortest list holds 150 logs,
    // so nothing below 1/151), and one that is not an error rate.
    for (error_rate, exit) in [(0.001_f32, 4), (0.0, 2), (1.0, 2), (f32::NAN, 2)] {
        let mut scorer = population.scorer();
        let error = run(
            &logs,
            &mut scorer,
            &calibrated(&calibrator, error_rate, None),
        )
        .unwrap_err();
        match &error {
            TriageError::Calibration(CalibrationError::TooFewLogs { .. }) => assert_eq!(exit, 4),
            TriageError::Calibration(CalibrationError::InvalidErrorRate { .. }) => {
                assert_eq!(exit, 2)
            }
            other => panic!("error rate {error_rate}: {other:?}"),
        }
        assert_eq!(L2dError::from(error).exit_code(), exit);
        assert!(scorer.batches.is_empty(), "nothing was scored");
    }
}

// ---------------------------------------------------------------------------------------
// Repeats and batches (requirements 11.10, 12.6), memory (12.4, 12.9)
// ---------------------------------------------------------------------------------------

#[test]
fn the_result_is_the_same_whatever_the_batch_size_and_on_repeat() {
    const E: f32 = 0.1;
    let calibrator = synthetic_calibrator(REVISION);
    let population = Population::draw(99, 2_500, Some(30));
    let logs = population.logs();
    for settings in [
        plain(ScorerKind::Baseline, Some(100)),
        calibrated(&calibrator, E, Some(100)),
    ] {
        let mut scorer = population.scorer();
        let whole = triage_with(&logs, &mut [&mut scorer], &settings, &mut budget()).unwrap();
        assert_eq!(scorer.batches, [2_500], "one batch by default");

        for batch_rows in [1_u64, 7, 64, 1_024, 2_499, 2_500, 100_000] {
            let mut scorer = population.scorer();
            let mut budget = budget_of(None, Some(batch_rows));
            let result = triage_with(&logs, &mut [&mut scorer], &settings, &mut budget).unwrap();
            assert!(
                result.frame().equals_missing(whole.frame()),
                "batches of {batch_rows} rows give another triage frame"
            );
            assert_eq!(result.counts(), whole.counts());
            assert_eq!(result.calibration(), whole.calibration());
            for (list, other) in [
                (result.review(), whole.review()),
                (result.scored_down(), whole.scored_down()),
                (result.abstained(), whole.abstained()),
            ] {
                assert!(list.unwrap().equals_missing(&other.unwrap()));
            }
            // The logs really went through in batches of that size.
            let size = batch_rows.min(2_500) as usize;
            assert_eq!(scorer.batches.iter().sum::<usize>(), 2_500);
            assert!(
                scorer.batches[..scorer.batches.len() - 1]
                    .iter()
                    .all(|rows| *rows == size),
                "batches of {batch_rows}: {:?}",
                &scorer.batches[..scorer.batches.len().min(5)]
            );
            assert_eq!(scorer.batches.len(), 2_500_usize.div_ceil(size));
            // Requirement 12.3: the plan the batches followed is in the result.
            assert_eq!(result.plan().batch_rows, batch_rows);
            assert_eq!(result.plan().bytes_per_row, 100);
        }

        // Requirement 11.10: the same again.
        let mut scorer = population.scorer();
        let again = triage_with(&logs, &mut [&mut scorer], &settings, &mut budget()).unwrap();
        assert!(again.frame().equals_missing(whole.frame()));
    }
}

#[test]
fn triage_counts_its_result_its_scorers_and_the_calibration_against_the_memory_limit() {
    const E: f32 = 0.1;
    let calibrator = synthetic_calibrator(REVISION);
    assert!(
        calibrator.held_bytes() >= 8 * 1_640,
        "a calibration whose lists hold 1,640 values keeps {} bytes",
        calibrator.held_bytes()
    );
    let population = Population::draw(123, 3_000, None);
    let logs = population.logs();

    // What triage holds is stated in its result: the calibration is part of it.
    let mut budget = budget();
    let left_before = budget.left_bytes();
    let mut scorer = population.scorer();
    let without = triage_with(
        &logs,
        &mut [&mut scorer],
        &plain(ScorerKind::Baseline, None),
        &mut budget,
    )
    .unwrap();
    let mut scorer = population.scorer();
    let with = triage_with(
        &logs,
        &mut [&mut scorer],
        &calibrated(&calibrator, E, None),
        &mut budget,
    )
    .unwrap();
    assert_eq!(
        with.held_bytes(),
        without.held_bytes() + calibrator.held_bytes()
    );
    // The scripted scorer asks for 1,000 bytes whatever the batch, 100 per row of a
    // batch and 4 per log for its evidence.
    assert!(without.held_bytes() >= 1_000 + 3_000 * 4);
    assert_eq!(without.plan().bytes_per_row, 100);
    // Triage gives back what it held: the caller's budget is as before.
    assert_eq!(budget.left_bytes(), left_before);

    // What the caller holds already stays counted, and is still held afterwards.
    let mut budget = budget_of(Some(64 * MIB), None);
    budget.hold(10 * MIB);
    let left_before = budget.left_bytes();
    let mut scorer = population.scorer();
    triage_with(
        &logs,
        &mut [&mut scorer],
        &calibrated(&calibrator, E, None),
        &mut budget,
    )
    .unwrap();
    assert_eq!(budget.left_bytes(), left_before);

    // A limit that leaves less than triage holds: refused before anything is scored,
    // with the bytes needed and the bytes left, and the budget as before.
    let needs = with.held_bytes();
    let mut budget = budget_of(Some(64 * MIB), None);
    budget.hold(32 * MIB - needs + 1);
    let left = budget.left_bytes();
    assert_eq!(left, needs - 1);
    let mut scorer = population.scorer();
    let error = triage_with(
        &logs,
        &mut [&mut scorer],
        &calibrated(&calibrator, E, None),
        &mut budget,
    )
    .unwrap_err();
    match &error {
        TriageError::Memory(L2dError::Refused { reason }) => {
            assert!(reason.contains(&format!("{needs} bytes")), "{reason}");
            assert!(reason.contains(&format!("{left} bytes")), "{reason}");
            assert!(
                reason.contains(&format!("{} bytes", 64 * MIB)),
                "the limit is named: {reason}"
            );
            assert!(
                reason.contains(&format!("{} bytes", calibrator.held_bytes())),
                "the calibration's share is named: {reason}"
            );
        }
        other => panic!("expected a memory refusal, got {other:?}"),
    }
    assert_eq!(L2dError::from(error).exit_code(), 4);
    assert!(scorer.batches.is_empty(), "nothing was scored");
    assert_eq!(budget.left_bytes(), left);
    // One byte more is enough for what is held, though not yet for a batch.
    budget.hold(32 * MIB - needs);
    let mut scorer = population.scorer();
    let error = triage_with(
        &logs,
        &mut [&mut scorer],
        &calibrated(&calibrator, E, None),
        &mut budget,
    )
    .unwrap_err();
    assert!(
        matches!(&error, TriageError::Memory(L2dError::Refused { reason }) if reason.contains("smallest batch")),
        "{error:?}"
    );
    assert!(scorer.batches.is_empty());
    assert_eq!(budget.left_bytes(), needs, "the budget is as before");
}

/// Review of task 1.10: triage rebuilt what the caller held from what was left, so a
/// budget that held more than its limit leaves came back holding exactly the limit.
#[test]
fn a_budget_that_already_holds_more_than_the_limit_leaves_comes_back_as_it_was() {
    let mut budget = budget_of(Some(64 * MIB), None);
    // 32 MiB are usable; the caller holds 40.
    budget.hold(40 * MIB);
    assert!(budget.require_held(0).is_err());
    assert_eq!(budget.left_bytes(), 0);

    // A scorer that costs nothing on no logs: nothing stands in the way of the run.
    let mut scorer = Scripted::baseline(&[]);
    scorer.cost = ScoringCost::default();
    let result = triage_with(
        &LogsFrame::empty(),
        &mut [&mut scorer],
        &plain(ScorerKind::Baseline, None),
        &mut budget,
    );
    assert!(result.is_ok(), "{result:?}");
    assert!(
        budget.require_held(0).is_err(),
        "the budget no longer holds what the caller held"
    );
    assert_eq!(budget.held_bytes(), 40 * MIB);

    // And after a refusal.
    let population = Population::draw(8, 50, None);
    let mut scorer = population.scorer();
    let result = triage_with(
        &population.logs(),
        &mut [&mut scorer],
        &plain(ScorerKind::Baseline, None),
        &mut budget,
    );
    assert!(matches!(result, Err(TriageError::Memory(_))), "{result:?}");
    assert!(budget.require_held(0).is_err());
    assert_eq!(budget.held_bytes(), 40 * MIB);
}

#[test]
fn triage_without_a_budget_of_its_own_observes_the_host() {
    let logs = logs_of(&[("a", Label::Unknown, None), ("b", Label::Unknown, None)]);
    let mut scorer = Scripted::baseline(&[("a", Some(0.25)), ("b", Some(0.5))]);
    let result = triage(
        &logs,
        &mut [&mut scorer],
        &plain(ScorerKind::Baseline, Some(1)),
    )
    .unwrap();
    assert_eq!(texts_of(&result.review().unwrap()), ["b"]);
    assert!(result.plan().free_bytes > 0);
}

// ---------------------------------------------------------------------------------------
// The mini fixture with the baseline scorer (the task's done-criterion)
// ---------------------------------------------------------------------------------------

/// The four of the fixture's eight groups that are held out for calibration.
const HELD_OUT: [&str; 4] = [
    "GET /tienda1/index.jsp",
    "GET /tienda1/publico/buscar.jsp",
    "POST /tienda1/miembros/editar.jsp",
    "POST /tienda1/publico/autenticar.jsp",
];

struct Fixture {
    _scratch: Scratch,
    whole: LogsFrame,
    held_out: (LogsFrame, InputSummary),
    scorer: BaselineScorer,
}

/// The mini fixture: the scorer is fitted on four of its groups, the other four are
/// held out for calibration, and the whole fixture is what is triaged.
fn fixture(name: &str) -> Fixture {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv");
    let text = std::fs::read_to_string(&source).expect("the fixture is readable");
    let mut lines = text.lines();
    let header = lines.next().expect("the fixture has a header");
    let (mut fitting, mut held_out) = (vec![header], vec![header]);
    for line in lines {
        let held = HELD_OUT
            .iter()
            .any(|group| line.ends_with(&format!(",{group}")));
        if held { &mut held_out } else { &mut fitting }.push(line);
    }
    let scratch = Scratch::new(name);
    let adapter = || LabelledAdapter::new(LabelledFormat::Csv).with_group_column("group");
    let read = |file: &str, lines: &[&str]| {
        let path = scratch.root.join(file);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        adapter().read(&path).expect("the part is readable")
    };
    let fitting = read("fitting.csv", &fitting);
    let held_out = read("held-out.csv", &held_out);
    let (whole, _) = adapter().read(&source).expect("the fixture is readable");
    let small = BaselineSettings {
        buckets: 4_096,
        ..BaselineSettings::default()
    };
    let scorer = BaselineScorer::fit_with(
        &fitting.0,
        std::slice::from_ref(&fitting.1),
        &small,
        &budget(),
    )
    .expect("the fitting part fits");
    Fixture {
        _scratch: scratch,
        whole,
        held_out,
        scorer,
    }
}

impl Fixture {
    fn ratios(&self) -> Vec<Option<f32>> {
        self.whole
            .records()
            .map(|record| Some(self.scorer.ratio(&record.text)))
            .collect()
    }

    fn groups(&self) -> Vec<Option<String>> {
        self.whole.records().map(|record| record.group).collect()
    }
}

#[test]
fn the_fixture_is_partitioned_by_the_budget_with_the_baseline_scorer_named() {
    let mut fixture = fixture("plain");
    let logs = fixture.whole.clone();
    let before = logs.frame().clone();
    let ratios = fixture.ratios();
    let groups = fixture.groups();
    let groups: Vec<Option<&str>> = groups.iter().map(|group| group.as_deref()).collect();
    assert!(
        logs.len() > 150,
        "the fixture has {} distinct logs",
        logs.len()
    );

    for review_budget in [None, Some(0), Some(25), Some(10_000)] {
        let settings = plain(ScorerKind::Baseline, review_budget);
        let result = run(&logs, &mut fixture.scorer, &settings).unwrap();
        assert_partition(&logs, &result);
        let rows = rows_of(result.frame());
        assert_placed(&rows, &expected(&ratios, &groups, None, review_budget));
        for (row, ratio) in rows.iter().zip(&ratios) {
            assert_eq!(row.scorer, "baseline", "requirement 6.6");
            assert_eq!(row.ratio, *ratio, "the ratio is the baseline scorer's");
            assert!(!row.abstained, "the baseline scorer never abstains");
        }
        let in_review = review_budget.map_or(logs.len(), |n| (n as usize).min(logs.len()));
        assert_eq!(result.review().unwrap().height(), in_review);
        assert_eq!(
            result.scored_down().unwrap().height(),
            logs.len() - in_review
        );
        assert_eq!(result.abstained().unwrap().height(), 0);
        // The baseline's evidence column comes through.
        assert_eq!(
            result.frame().column("no_features").unwrap().dtype(),
            &DataType::Boolean
        );
        assert_eq!(result.scorers(), [ScorerKind::Baseline]);
    }
    assert!(logs.frame().equals_missing(&before));
}

#[test]
fn the_fixture_is_partitioned_by_label_set_and_budget_with_a_calibration() {
    const E: f32 = 0.1;
    let mut fixture = fixture("calibrated");
    let calibrator = ConformalCalibrator::fit_baseline(
        &fixture.scorer,
        &fixture.held_out.0,
        std::slice::from_ref(&fixture.held_out.1),
        &CalibrationSettings { min_per_group: 10 },
        &budget(),
    )
    .expect("the held-out part calibrates");
    let logs = fixture.whole.clone();
    let before = logs.frame().clone();
    let ratios = fixture.ratios();
    let groups = fixture.groups();
    let groups: Vec<Option<&str>> = groups.iter().map(|group| group.as_deref()).collect();

    let mut bad_sets = 0;
    for review_budget in [None, Some(0), Some(5), Some(10_000)] {
        let settings = calibrated(&calibrator, E, review_budget);
        let result = run(&logs, &mut fixture.scorer, &settings).unwrap();
        assert_partition(&logs, &result);
        let rows = rows_of(result.frame());
        assert_placed(
            &rows,
            &expected(&ratios, &groups, Some((&calibrator, E)), review_budget),
        );
        assert!(rows.iter().all(|row| row.scorer == "baseline"));
        let counts = *result.counts();
        if review_budget.is_none() {
            bad_sets = counts.review;
            // The fixture puts logs in all three lists (58 for review, 96 scored down by
            // calibration, 15 unlike the calibration logs, none uncertain, when this was
            // written); the uncertain ones are covered by the synthetic tests above.
            println!("L2D_TRIAGE_FIXTURE {counts:?} {:?}", result.calibration());
            assert!(counts.review > 5, "{counts:?}");
            assert!(counts.scored_down_by_calibration > 5, "{counts:?}");
            assert!(counts.unlike_calibration > 5, "{counts:?}");
        }
        let in_review = review_budget.map_or(bad_sets, |n| n.min(bad_sets));
        assert_eq!(counts.review, in_review);
        assert_eq!(counts.scored_down_by_budget, bad_sets - in_review);

        let used = result
            .calibration()
            .expect("the result names the calibration");
        assert_eq!(used.note, EXCHANGEABILITY_NOTE);
        assert_eq!(used.error_rate, E);
        // The four held-out groups have a list of their own for good (12 logs each, the
        // minimum is 10) and none for bad (9 each); the four fitting groups have none.
        let in_held_out = groups
            .iter()
            .filter(|group| group.is_some_and(|group| HELD_OUT.contains(&group)))
            .count() as u64;
        assert_eq!(
            used.good,
            ListUse {
                group: in_held_out,
                pooled: logs.len() as u64 - in_held_out
            }
        );
        assert_eq!(
            used.bad,
            ListUse {
                group: 0,
                pooled: logs.len() as u64
            }
        );
    }
    assert!(logs.frame().equals_missing(&before));
}
