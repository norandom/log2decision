//! A score fitted from a mark set (requirements 16.6, 16.7, 16.8, 16.3, 6.1, 6.6; task
//! 1.23).
//!
//! The fixture is `tests/fixtures/mini-labelled.csv` (169 logs read by the CSV adapter).
//! Its labels play no part: the analyst's marks are the labels. The marked logs are found
//! by a regular expression or named by position, as an analyst would mark them.

use std::io::Write;
use std::path::{Path, PathBuf};

use l2d_baseline::markscore::{GroupKeyRule, MarkFitSettings, OtherClass, fit_from_marks};
use l2d_baseline::redact::{RedactionInput, RedactionRules, RedactionSettings, propose_redactions};
use l2d_baseline::scorer::MARK_SET_SCORER_KIND;
use l2d_baseline::{BaselineScorer, BaselineSettings};
use l2d_core::marks::{MarkQuery, MarkSet, MarkStore};
use l2d_core::memory::PROCESS_BASELINE_BYTES;
use l2d_core::polars::prelude::{Column, DataFrame};
use l2d_core::{
    InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat, LinesAdapter,
    LogsFrame, MemoryBudget, MemorySettings, ProjectPaths, RedactionKind, RedactionRule,
    ZipInputSettings,
};
use sha2::{Digest, Sha256};

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const SQL_INJECTION: &str = r"(?i)union(\s|\+)+select";
const DROP_TABLE: &str = r"(?i)drop(\s|\+)+table";
const HOST: &str = "localhost:8080";
const ANADIR: &str = "/tienda1/publico/anadir.jsp";
const BUSCAR: &str = "/tienda1/publico/buscar.jsp";

/// A directory of its own in the system temp folder, removed when dropped.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "l2d-baseline-markscore-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    fn project(&self, name: &str) -> ProjectPaths {
        ProjectPaths::new(self.dir.join(name))
    }

    fn archive(&self, name: &str, entries: &[(&str, &[String])]) -> PathBuf {
        let path = self.dir.join(name);
        let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let mut writer = zip::ZipWriter::new(file);
        for (entry, lines) in entries {
            writer
                .start_file(*entry, zip::write::SimpleFileOptions::default())
                .unwrap();
            for line in *lines {
                writer.write_all(line.as_bytes()).unwrap();
                writer.write_all(b"\n").unwrap();
            }
        }
        writer.finish().unwrap().flush().unwrap();
        path
    }

    fn lines(&self, name: &str, lines: &[String]) -> PathBuf {
        let path = self.dir.join(name);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("mini-labelled.csv")
}

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * GIB)).unwrap()
}

fn read_fixture() -> (LogsFrame, InputSummary, MemoryBudget) {
    let mut budget = budget();
    let (logs, summary, _) = LabelledAdapter::new(LabelledFormat::Csv)
        .read_with(&fixture_path(), &mut budget)
        .unwrap();
    (logs, summary, budget)
}

fn settings() -> MarkFitSettings {
    MarkFitSettings {
        baseline: BaselineSettings {
            buckets: 4096,
            ..BaselineSettings::default()
        },
        group_key: None,
    }
}

fn regex(pattern: &str) -> MarkQuery {
    MarkQuery::Regex {
        pattern: pattern.into(),
    }
}

fn mark(
    name: &str,
    label: Label,
    query: MarkQuery,
    logs: &LogsFrame,
    summary: &InputSummary,
) -> MarkSet {
    MarkSet::mark(name, label, query, logs, summary, &budget()).unwrap()
}

fn texts(logs: &LogsFrame) -> Vec<String> {
    logs.records().map(|record| record.text).collect()
}

fn positions(logs: &LogsFrame) -> Vec<u64> {
    logs.records().map(|record| record.first_position).collect()
}

fn literal(fragment: &str) -> RedactionRule {
    RedactionRule {
        kind: RedactionKind::Literal,
        fragment: fragment.into(),
        placeholder: "\u{1a}".into(),
    }
}

fn refusal(error: L2dError) -> String {
    match error {
        L2dError::Refused { reason } => reason,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------
// Refusal below the counts of requirement 2.3 (16.8)
// ---------------------------------------------------------------------------------------

#[test]
fn too_few_marked_logs_after_de_selection_are_refused_with_the_counts() {
    let (logs, summary, mut budget) = read_fixture();
    let mut set = mark("drop", Label::Bad, regex(DROP_TABLE), &logs, &summary);
    assert_eq!(set.count(), 8);
    let marked: Vec<u64> = set.positions().to_vec();
    set.deselect(&marked[..4]).unwrap();
    let reason = refusal(
        fit_from_marks(
            &set,
            None,
            RedactionInput::File {
                logs: &logs,
                summary: &summary,
            },
            &settings(),
            &mut budget,
        )
        .unwrap_err(),
    );
    assert!(reason.contains("4 marked logs"), "{reason}");
    assert!(reason.contains("8 marked, 4 de-selected"), "{reason}");
    assert!(reason.contains("161"), "{reason}");
    assert!(reason.contains("at least 10"), "{reason}");
    assert!(reason.contains("at least 5"), "{reason}");
    // Five left is enough against the unmarked logs.
    let mut set = mark("drop", Label::Bad, regex(DROP_TABLE), &logs, &summary);
    set.deselect(&marked[..3]).unwrap();
    let fit = fit_from_marks(
        &set,
        None,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &settings(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(fit.counts.marked, 5);
    assert_eq!(fit.scorer.fit_report().bad, 5);
}

#[test]
fn too_few_logs_of_the_other_mark_set_are_refused_with_the_counts() {
    let (logs, summary, mut budget) = read_fixture();
    let all = positions(&logs);
    let bad = mark(
        "five-bad",
        Label::Bad,
        MarkQuery::Positions {
            positions: all[..5].to_vec(),
        },
        &logs,
        &summary,
    );
    let four_good = mark(
        "four-good",
        Label::Good,
        MarkQuery::Positions {
            positions: all[5..9].to_vec(),
        },
        &logs,
        &summary,
    );
    let input = || RedactionInput::File {
        logs: &logs,
        summary: &summary,
    };
    let reason = refusal(
        fit_from_marks(&bad, Some(&four_good), input(), &settings(), &mut budget).unwrap_err(),
    );
    assert!(reason.contains("5 marked logs"), "{reason}");
    assert!(reason.contains("4 logs of the other class"), "{reason}");
    assert!(reason.contains("the mark set `four-good`"), "{reason}");
    let five_good = mark(
        "five-good",
        Label::Good,
        MarkQuery::Positions {
            positions: all[5..10].to_vec(),
        },
        &logs,
        &summary,
    );
    let fit = fit_from_marks(&bad, Some(&five_good), input(), &settings(), &mut budget).unwrap();
    let report = fit.scorer.fit_report();
    assert_eq!((report.rows, report.good, report.bad), (10, 5, 5));
    assert_eq!(fit.counts.not_marked, 169 - 10);
}

// ---------------------------------------------------------------------------------------
// The other class, stated (16.6)
// ---------------------------------------------------------------------------------------

#[test]
fn the_score_states_what_it_was_fitted_against() {
    let (logs, summary, mut budget) = read_fixture();
    let attacks = mark("attacks", Label::Bad, regex(SQL_INJECTION), &logs, &summary);
    let input = || RedactionInput::File {
        logs: &logs,
        summary: &summary,
    };
    let fit = fit_from_marks(&attacks, None, input(), &settings(), &mut budget).unwrap();
    assert_eq!(fit.other_class, OtherClass::Unmarked);
    assert!(
        fit.other_class
            .to_string()
            .contains("unmarked logs of the same input")
    );
    assert_eq!(fit.counts.marked + fit.counts.other, 169);
    let origin = fit.scorer.mark_set_origin().unwrap();
    assert!(origin.marks().try_eq(&attacks).unwrap());
    assert!(origin.other().is_none());
    assert_eq!(origin.other_class(), OtherClass::Unmarked);

    // Good marks of the same input: fitted against them, and stated.
    let good = mark(
        "plain",
        Label::Good,
        regex(r"^GET [^ ]*pagina=\d+$"),
        &logs,
        &summary,
    );
    assert!(good.count() >= 5);
    let fit = fit_from_marks(&attacks, Some(&good), input(), &settings(), &mut budget).unwrap();
    assert_eq!(
        fit.other_class,
        OtherClass::MarkSet {
            name: "plain".into()
        }
    );
    assert!(fit.other_class.to_string().contains("the mark set `plain`"));
    assert_eq!(fit.counts.other, good.count());
    assert_eq!(fit.scorer.fit_report().good, good.count());

    // An other mark set of the same label, or of another input, is not usable.
    let same_label = mark("more", Label::Bad, regex(DROP_TABLE), &logs, &summary);
    assert!(matches!(
        fit_from_marks(
            &attacks,
            Some(&same_label),
            input(),
            &settings(),
            &mut budget
        ),
        Err(L2dError::InvalidInput { .. })
    ));
    let mut other_summary = summary.clone();
    other_summary.content_sha256 = "0".repeat(64);
    let elsewhere = mark(
        "elsewhere",
        Label::Good,
        regex("GET"),
        &logs,
        &other_summary,
    );
    assert!(matches!(
        fit_from_marks(
            &attacks,
            Some(&elsewhere),
            input(),
            &settings(),
            &mut budget
        ),
        Err(L2dError::Refused { .. })
    ));
}

#[test]
fn a_log_in_both_mark_sets_is_left_out_and_counted() {
    let (logs, summary, mut budget) = read_fixture();
    let attacks = mark(
        "attacks",
        Label::Bad,
        regex(r"(?i)union(\s|\+)+select|drop(\s|\+)+table|ping|passwd|' OR "),
        &logs,
        &summary,
    );
    // Every GET is "good" here, so the GET attacks are in both.
    let gets = mark("gets", Label::Good, regex("^GET "), &logs, &summary);
    let both: Vec<u64> = attacks
        .selected()
        .filter(|position| gets.is_selected(*position))
        .collect();
    assert!(!both.is_empty());
    let fit = fit_from_marks(
        &attacks,
        Some(&gets),
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &settings(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(fit.counts.conflicting, both.len() as u64);
    assert_eq!(fit.counts.marked, attacks.count() - both.len() as u64);
    assert_eq!(fit.counts.other, gets.count() - both.len() as u64);
    let texts = texts(&logs);
    let all = positions(&logs);
    for position in both {
        let text = &texts[all.iter().position(|p| *p == position).unwrap()];
        assert_eq!(fit.scorer.was_fitted_on_text(text), Some(false));
    }
}

// ---------------------------------------------------------------------------------------
// De-selection (16.3): out of fitting and out of the proposals
// ---------------------------------------------------------------------------------------

#[test]
fn a_de_selected_log_is_kept_out_of_fitting_and_out_of_the_proposals() {
    let (logs, summary, mut budget) = read_fixture();
    let mut set = mark("attacks", Label::Bad, regex(SQL_INJECTION), &logs, &summary);
    let marked = set.count();
    let left_out = set.positions()[0];
    set.deselect(&[left_out]).unwrap();
    let all_texts = texts(&logs);
    let all = positions(&logs);
    let text = &all_texts[all.iter().position(|p| *p == left_out).unwrap()];

    let fit = fit_from_marks(
        &set,
        None,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &settings(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(fit.counts.deselected, 1);
    assert_eq!(fit.counts.marked, marked - 1);
    // Not the other class either: 169 logs, less the marked ones (de-selected included).
    assert_eq!(fit.counts.other, 169 - marked);
    assert_eq!(fit.scorer.fit_report().rows, 168);
    assert_eq!(fit.scorer.was_fitted_on_text(text), Some(false));
    assert!(texts(&fit.logs).iter().all(|fitted| fitted != text));

    // The proposals count it on neither side.
    let proposals = propose_redactions(
        &set,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &RedactionSettings::default(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(proposals.marked_logs, marked - 1);
    assert_eq!(proposals.unmarked_logs, 169 - marked);
    assert_eq!(proposals.deselected_logs, 1);
}

// ---------------------------------------------------------------------------------------
// Rules: applied at fitting and first at scoring (16.6, 16.7)
// ---------------------------------------------------------------------------------------

#[test]
fn the_score_applies_its_rules_first_at_scoring() {
    let (logs, summary, mut budget) = read_fixture();
    let mut set = mark("attacks", Label::Bad, regex(SQL_INJECTION), &logs, &summary);
    let plain = fit_from_marks(
        &set,
        None,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &settings(),
        &mut budget,
    )
    .unwrap()
    .scorer;
    set.set_redactions(vec![literal(ANADIR), literal(BUSCAR), literal(HOST)])
        .unwrap();
    let fit = fit_from_marks(
        &set,
        None,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &settings(),
        &mut budget,
    )
    .unwrap();
    let scorer = &fit.scorer;
    let rules = RedactionRules::of_mark_set(&set, &budget).unwrap();
    assert_eq!(scorer.rules().content_sha256(), rules.content_sha256());
    assert!(RedactionRules::none().content_sha256() != rules.content_sha256());

    // Two logs that differ only in a redacted endpoint: one ratio, bit for bit.
    let anadir = format!("GET http://{HOST}{ANADIR}?id=2+UNION+SELECT+password+FROM+users");
    let buscar = format!("GET http://{HOST}{BUSCAR}?id=2+UNION+SELECT+password+FROM+users");
    assert_eq!(
        scorer.ratio(&anadir).to_bits(),
        scorer.ratio(&buscar).to_bits()
    );
    // The same marks without rules tell them apart: the rules are what makes them equal.
    assert_ne!(
        plain.ratio(&anadir).to_bits(),
        plain.ratio(&buscar).to_bits()
    );
    // A text and its redacted form are one text to the score.
    for text in texts(&logs) {
        let redacted = rules.apply(&text);
        assert_eq!(
            scorer.ratio(&text).to_bits(),
            scorer.ratio(&redacted).to_bits()
        );
    }
    // The scores frame applies them too.
    let frame = scorer
        .score_texts(&[anadir.as_str(), buscar.as_str()])
        .unwrap();
    let ratios: Vec<f32> = frame
        .column("ratio")
        .unwrap()
        .f32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(ratios[0].to_bits(), scorer.ratio(&anadir).to_bits());
    assert_eq!(ratios[0].to_bits(), ratios[1].to_bits());
    // The frame names the scorer as a score fitted from a mark set (6.6).
    let named: Vec<&str> = frame
        .column("scorer")
        .unwrap()
        .str()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(named, vec![MARK_SET_SCORER_KIND; 2]);
    assert_eq!(scorer.scorer_kind(), MARK_SET_SCORER_KIND);
    assert_eq!(plain.scorer_kind(), "mark-set");
    let (logs_plain, summary_plain, budget_plain) = read_fixture();
    let labelled = BaselineScorer::fit_with(
        &logs_plain,
        std::slice::from_ref(&summary_plain),
        &settings().baseline,
        &budget_plain,
    )
    .unwrap();
    assert_eq!(labelled.scorer_kind(), "baseline");
    // The fitted texts are the logs as read, not their redacted forms.
    for record in fit.logs.records() {
        assert_eq!(scorer.was_fitted_on_text(&record.text), Some(true));
    }
    // Every marked or unmarked log is fitted.
    assert_eq!(scorer.fit_report().rows, 169);
}

#[test]
fn texts_that_are_equal_after_redaction_stay_rows_of_their_own() {
    let scratch = Scratch::new("equal-after");
    let mut lines = Vec::new();
    for n in 0..12 {
        for path in ["/shop/a.php", "/shop/b.php"] {
            lines.push(format!("GET http://h{path}?q={n}"));
            lines.push(format!("GET http://h{path}?q={n}'+OR+1=1--"));
        }
    }
    let path = scratch.lines("logs.txt", &lines);
    let mut budget = budget();
    let (logs, summary, _) = LinesAdapter.read_with(&path, &mut budget).unwrap();
    assert_eq!(logs.len(), 48);
    let mut set = mark("or", Label::Bad, regex("OR"), &logs, &summary);
    set.set_redactions(vec![literal("/shop/a.php"), literal("/shop/b.php")])
        .unwrap();
    let fit = fit_from_marks(
        &set,
        None,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &settings(),
        &mut budget,
    )
    .unwrap();
    // 48 distinct texts read, 24 distinct after redaction: every one is fitted as a row
    // of its own, with its own text and hash; nothing is merged.
    let rules = RedactionRules::of_mark_set(&set, &budget).unwrap();
    let mut redacted: Vec<String> = lines.iter().map(|l| rules.apply(l).into_owned()).collect();
    redacted.sort();
    redacted.dedup();
    assert_eq!(redacted.len(), 24);
    assert_eq!(fit.scorer.fit_report().rows, 48);
    assert_eq!(fit.scorer.fitted_text_hashes(), Some(48));
    assert_eq!(fit.logs.len(), 48);
    let mut fitted = texts(&fit.logs);
    fitted.sort();
    fitted.dedup();
    assert_eq!(fitted.len(), 48);
}

// ---------------------------------------------------------------------------------------
// Saved with the mark set, the de-selections, the rules and the fitting-text hashes
// ---------------------------------------------------------------------------------------

#[test]
fn the_score_survives_save_and_load_with_its_mark_set_and_rules() {
    let scratch = Scratch::new("save-load");
    let paths = scratch.project("project");
    let (logs, summary, mut budget) = read_fixture();
    let mut set = mark("attacks", Label::Bad, regex(SQL_INJECTION), &logs, &summary);
    set.deselect(&[set.positions()[1]]).unwrap();
    set.set_redactions(vec![literal(HOST), literal(ANADIR)])
        .unwrap();
    let fit = fit_from_marks(
        &set,
        None,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &settings(),
        &mut budget,
    )
    .unwrap();
    let dir = fit.scorer.save(&paths, "attacks-score").unwrap();
    assert!(dir.join("mark-set.json").is_file());
    assert!(!dir.join("other-mark-set.json").exists());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("baseline.json")).unwrap()).unwrap();
    assert_eq!(manifest["format"], 3);
    let entry = &manifest["mark_set"];
    assert_eq!(entry["name"], "attacks");
    assert_eq!(entry["label"], "bad");
    assert_eq!(entry["other"]["kind"], "unmarked");
    assert_eq!(entry["deselected"], 1);
    assert_eq!(
        entry["sha256"].as_str().unwrap(),
        format!(
            "{:x}",
            Sha256::digest(std::fs::read(dir.join("mark-set.json")).unwrap())
        )
    );
    let rules = RedactionRules::of_mark_set(&set, &budget).unwrap();
    assert_eq!(manifest["redactions"]["sha256"], rules.content_sha256());
    assert_eq!(manifest["redactions"]["rules"].as_array().unwrap().len(), 2);
    // The copy of the mark set holds the de-selection and the rules.
    let copy: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("mark-set.json")).unwrap()).unwrap();
    assert_eq!(copy["deselected"].as_array().unwrap().len(), 1);
    assert_eq!(copy["redactions"].as_array().unwrap().len(), 2);

    let loaded = BaselineScorer::load_under(&paths, &dir).unwrap();
    assert_eq!(loaded.revision(), fit.scorer.revision());
    assert_eq!(loaded.rules().content_sha256(), rules.content_sha256());
    let origin = loaded.mark_set_origin().unwrap();
    assert!(origin.marks().try_eq(&set).unwrap());
    assert_eq!(
        origin.counts(),
        fit.scorer.mark_set_origin().unwrap().counts()
    );
    assert_eq!(loaded.fitted_text_hashes(), Some(168));
    for text in texts(&logs) {
        assert_eq!(
            loaded.ratio(&text).to_bits(),
            fit.scorer.ratio(&text).to_bits()
        );
        assert_eq!(
            loaded.was_fitted_on_text(&text),
            fit.scorer.was_fitted_on_text(&text)
        );
    }
    assert!(loaded.held_bytes() >= rules.held_bytes());
    assert!(BaselineScorer::load_bytes(&dir) > 0);
    // Saving again is the same directory and changes nothing.
    let before = std::fs::read(dir.join("baseline.json")).unwrap();
    assert_eq!(loaded.save(&paths, "attacks-score").unwrap(), dir);
    assert_eq!(std::fs::read(dir.join("baseline.json")).unwrap(), before);

    // A copy of the mark set that was changed is refused.
    let path = dir.join("mark-set.json");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(b'\n');
    std::fs::write(&path, bytes).unwrap();
    assert!(matches!(
        BaselineScorer::load_under(&paths, &dir),
        Err(L2dError::Refused { .. })
    ));
}

#[test]
fn the_revision_covers_the_rules_and_the_marks() {
    let (logs, summary, mut budget) = read_fixture();
    let input = || RedactionInput::File {
        logs: &logs,
        summary: &summary,
    };
    let mut set = mark("attacks", Label::Bad, regex(SQL_INJECTION), &logs, &summary);
    let none = fit_from_marks(&set, None, input(), &settings(), &mut budget).unwrap();
    set.set_redactions(vec![literal(HOST)]).unwrap();
    let host = fit_from_marks(&set, None, input(), &settings(), &mut budget).unwrap();
    set.set_redactions(vec![literal(ANADIR)]).unwrap();
    let anadir = fit_from_marks(&set, None, input(), &settings(), &mut budget).unwrap();
    let revisions = [
        none.scorer.revision(),
        host.scorer.revision(),
        anadir.scorer.revision(),
    ];
    assert_ne!(revisions[0], revisions[1]);
    assert_ne!(revisions[1], revisions[2]);
    assert_ne!(revisions[0], revisions[2]);
    // A plain fit of the same rows is another model description: no marks, no rules.
    let plain = BaselineScorer::fit_with(
        &none.logs,
        std::slice::from_ref(&summary),
        &settings().baseline,
        &budget,
    )
    .unwrap();
    assert_ne!(plain.revision(), none.scorer.revision());
    assert!(plain.mark_set_origin().is_none());
    assert!(plain.rules().is_empty());
    // The same weights: marks without rules fit what a plain fit of their rows fits.
    let bits = |scorer: &BaselineScorer| -> Vec<u64> {
        scorer.weights().iter().map(|w| w.to_bits()).collect()
    };
    assert_eq!(bits(&plain), bits(&none.scorer));
    // A de-selection is part of the marks and so of the revision.
    let mut fewer = mark("attacks", Label::Bad, regex(SQL_INJECTION), &logs, &summary);
    fewer.deselect(&[fewer.positions()[0]]).unwrap();
    let fewer = fit_from_marks(&fewer, None, input(), &settings(), &mut budget).unwrap();
    assert_ne!(fewer.scorer.revision(), none.scorer.revision());
}

#[test]
fn a_plain_scorer_is_saved_as_before() {
    let scratch = Scratch::new("plain");
    let paths = scratch.project("project");
    let (logs, summary, budget) = read_fixture();
    let scorer = BaselineScorer::fit_with(
        &logs,
        std::slice::from_ref(&summary),
        &settings().baseline,
        &budget,
    )
    .unwrap();
    let redacted_none = BaselineScorer::fit_redacted_with(
        &logs,
        std::slice::from_ref(&summary),
        &settings().baseline,
        &RedactionRules::none(),
        &budget,
    )
    .unwrap();
    assert_eq!(scorer.revision(), redacted_none.revision());
    let dir = scorer.save(&paths, "plain").unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("baseline.json")).unwrap()).unwrap();
    assert_eq!(manifest["format"], 2);
    assert!(manifest.get("mark_set").is_none());
    assert!(manifest.get("redactions").is_none());
    assert!(!dir.join("mark-set.json").exists());
}

// ---------------------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------------------

#[test]
fn fitting_from_marks_is_deterministic() {
    let scratch = Scratch::new("deterministic");
    let (logs, summary, mut budget) = read_fixture();
    let mut set = mark("attacks", Label::Bad, regex(SQL_INJECTION), &logs, &summary);
    set.set_redactions(vec![literal(HOST), literal(ANADIR)])
        .unwrap();
    let mut fits = Vec::new();
    for _ in 0..2 {
        fits.push(
            fit_from_marks(
                &set,
                None,
                RedactionInput::File {
                    logs: &logs,
                    summary: &summary,
                },
                &settings(),
                &mut budget,
            )
            .unwrap(),
        );
    }
    assert_eq!(fits[0].scorer, fits[1].scorer);
    assert_eq!(fits[0].counts, fits[1].counts);
    let mut saved = Vec::new();
    for (index, fit) in fits.iter().enumerate() {
        let paths = scratch.project(&format!("p{index}"));
        let dir = fit.scorer.save(&paths, "score").unwrap();
        let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    std::fs::read(entry.path()).unwrap(),
                )
            })
            .collect();
        files.sort();
        saved.push((dir.file_name().unwrap().to_owned(), files));
    }
    assert_eq!(saved[0], saved[1]);
}

// ---------------------------------------------------------------------------------------
// An archive and a store
// ---------------------------------------------------------------------------------------

#[test]
fn an_archive_mark_set_is_fitted_with_the_host_of_each_entry_as_group_key() {
    let scratch = Scratch::new("archive");
    let mut web = Vec::new();
    let mut mail = Vec::new();
    for n in 0..20 {
        web.push(format!("GET /index.php?id={n}"));
        mail.push(format!("postfix/smtpd[{n}]: connect from client{n}"));
        if n % 3 == 0 {
            web.push(format!("GET /index.php?id={n}'+OR+1=1--"));
            mail.push(format!(
                "postfix/smtpd[{n}]: warning: ' OR 1=1-- from client{n}"
            ));
        }
    }
    let path = scratch.archive(
        "testbed.zip",
        &[
            ("gather/webserver/logs/access.log", &web),
            ("gather/mail/logs/mail.log", &mail),
            ("README.txt", &["notes".to_string()]),
        ],
    );
    let zip_settings = ZipInputSettings::default();
    let mut budget = budget();
    let set = MarkSet::mark_archive(
        "or",
        Label::Bad,
        regex("OR 1=1|OR\\+1=1"),
        &path,
        &zip_settings,
        &mut budget,
    )
    .unwrap()
    .0;
    assert_eq!(set.count(), 14);
    let held_before = budget.held_bytes();
    let fit = fit_from_marks(
        &set,
        None,
        RedactionInput::Archive {
            path: &path,
            settings: &zip_settings,
        },
        &settings(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(fit.group_key, GroupKeyRule::ArchiveHost);
    assert!(fit.group_key.name().contains("host"));
    assert_eq!((fit.counts.marked, fit.counts.other), (14, 41));
    let mut groups: Vec<Option<String>> = fit.logs.records().map(|r| r.group).collect();
    groups.sort();
    groups.dedup();
    assert_eq!(
        groups,
        vec![
            None,
            Some("mail".to_string()),
            Some("webserver".to_string())
        ]
    );
    assert!(fit.archive.is_some());
    assert_eq!(budget.held_bytes(), held_before);

    // The group key is settable: a pattern over the text.
    let by_pattern = MarkFitSettings {
        group_key: Some(GroupKeyRule::Pattern {
            pattern: r"^(\S+)".into(),
        }),
        ..settings()
    };
    let fit = fit_from_marks(
        &set,
        None,
        RedactionInput::Archive {
            path: &path,
            settings: &zip_settings,
        },
        &by_pattern,
        &mut budget,
    )
    .unwrap();
    assert!(fit.group_key.name().contains(r"^(\S+)"));
    assert!(
        fit.logs
            .records()
            .any(|r| r.group.as_deref() == Some("GET"))
    );
}

/// The fixture's logs as a store would hand them out: ids are the SHA-256 of the text.
struct FixtureStore {
    rows: Vec<(String, String)>,
    hash: String,
}

impl FixtureStore {
    fn new(texts: &[String]) -> Self {
        let rows: Vec<(String, String)> = texts
            .iter()
            .map(|text| {
                (
                    format!("{:x}", Sha256::digest(text.as_bytes())),
                    text.clone(),
                )
            })
            .collect();
        let mut ids: Vec<&str> = rows.iter().map(|(id, _)| id.as_str()).collect();
        ids.sort_unstable();
        let hash = format!("{:x}", Sha256::digest(ids.concat().as_bytes()));
        Self { rows, hash }
    }
}

impl MarkStore for FixtureStore {
    fn store_name(&self) -> &str {
        "fixture"
    }

    fn content_sha256(&self, _budget: &mut MemoryBudget) -> Result<String, L2dError> {
        Ok(self.hash.clone())
    }

    fn phrase_ids(
        &self,
        phrase: &str,
        _id_bytes: u64,
        _budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&str),
    ) -> Result<(), L2dError> {
        for (id, text) in &self.rows {
            if text.contains(phrase) {
                each(id);
            }
        }
        Ok(())
    }

    fn text_batches(
        &self,
        _beside: u64,
        _per_row: u64,
        _budget: &mut MemoryBudget,
        each: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError> {
        for (number, chunk) in self.rows.chunks(10).enumerate() {
            let seq: Vec<i64> = (0..chunk.len())
                .map(|row| (number * 10 + row) as i64)
                .collect();
            let ids: Vec<&str> = chunk.iter().map(|(id, _)| id.as_str()).collect();
            let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
            let frame = DataFrame::new(vec![
                Column::new("seq".into(), seq),
                Column::new("id".into(), ids),
                Column::new("text".into(), texts),
            ])
            .unwrap();
            each(&frame)?;
        }
        Ok(())
    }
}

#[test]
fn a_store_mark_set_fits_the_same_model_as_the_same_logs_of_a_file() {
    let (logs, summary, mut budget) = read_fixture();
    let store = FixtureStore::new(&texts(&logs));
    let on_file = mark("attacks", Label::Bad, regex(SQL_INJECTION), &logs, &summary);
    let on_store = MarkSet::mark_store(
        "attacks",
        Label::Bad,
        regex(SQL_INJECTION),
        &store,
        &mut budget,
    )
    .unwrap();
    let file_fit = fit_from_marks(
        &on_file,
        None,
        RedactionInput::File {
            logs: &logs,
            summary: &summary,
        },
        &settings(),
        &mut budget,
    )
    .unwrap();
    let store_fit = fit_from_marks(
        &on_store,
        None,
        RedactionInput::Store(&store),
        &settings(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(store_fit.counts.marked, file_fit.counts.marked);
    assert_eq!(store_fit.counts.other, file_fit.counts.other);
    let bits = |scorer: &BaselineScorer| -> Vec<u64> {
        scorer.weights().iter().map(|w| w.to_bits()).collect()
    };
    assert_eq!(bits(&store_fit.scorer), bits(&file_fit.scorer));
    // The marks differ in kind, so the revisions do.
    assert_ne!(store_fit.scorer.revision(), file_fit.scorer.revision());
    // A store has no group keys by default.
    assert_eq!(store_fit.group_key, GroupKeyRule::None);
    assert!(store_fit.logs.records().all(|r| r.group.is_none()));
}

// ---------------------------------------------------------------------------------------
// Memory (16.12)
// ---------------------------------------------------------------------------------------

#[test]
fn a_fit_that_does_not_fit_the_memory_limit_is_refused_with_the_figures() {
    let (logs, summary, _) = read_fixture();
    let set = mark("attacks", Label::Bad, regex(SQL_INJECTION), &logs, &summary);
    let mut tight = MemoryBudget::observe_with(
        &MemorySettings {
            memory_limit_bytes: Some(PROCESS_BASELINE_BYTES + MIB),
            ..MemorySettings::default()
        },
        || Ok(8 * GIB),
    )
    .unwrap();
    let wide = MarkFitSettings {
        baseline: BaselineSettings::default(),
        group_key: None,
    };
    let reason = refusal(
        fit_from_marks(
            &set,
            None,
            RedactionInput::File {
                logs: &logs,
                summary: &summary,
            },
            &wide,
            &mut tight,
        )
        .unwrap_err(),
    );
    assert!(reason.contains("memory limit"), "{reason}");
    assert_eq!(tight.held_bytes(), 0);
}

// ---------------------------------------------------------------------------------------
// Scoring memory with rules that lengthen a text (review round 1)
// ---------------------------------------------------------------------------------------

#[test]
fn scoring_memory_covers_rules_that_lengthen_a_text() {
    let budget = budget();
    let long = "x".repeat(64);
    let rules = RedactionRules::new(
        vec![
            RedactionRule {
                kind: RedactionKind::Literal,
                fragment: "a".into(),
                placeholder: long.clone(),
            },
            RedactionRule {
                kind: RedactionKind::Pattern,
                fragment: "z*".into(),
                placeholder: "yy".into(),
            },
        ],
        &budget,
    )
    .unwrap();
    for text in ["aaaa", "abc", "", "zzz", "a z a z"] {
        let redacted = rules.apply(text).len() as u64;
        let bound = rules.redacted_bound(text.len() as u64);
        assert!(redacted <= bound, "{text:?}: {redacted} > {bound}");
    }
    // A text the rules lengthen far: three bounded "a" of 64 bytes each.
    assert!(rules.apply("a a a").len() > 3 * 64);
    // The rules of confirmed proposals (one-byte placeholders) never lengthen a text.
    let confirmed = RedactionRules::new(vec![literal(HOST), literal(ANADIR)], &budget).unwrap();
    assert_eq!(confirmed.redacted_bound(1000), 1000);
    assert_eq!(RedactionRules::none().redacted_bound(1000), 1000);

    let (logs, summary, budget) = read_fixture();
    let scorer = BaselineScorer::fit_redacted_with(
        &logs,
        std::slice::from_ref(&summary),
        &settings().baseline,
        &rules,
        &budget,
    )
    .unwrap();
    let plain = BaselineScorer::fit_with(
        &logs,
        std::slice::from_ref(&summary),
        &settings().baseline,
        &budget,
    )
    .unwrap();
    // The scoring figure of a 5-byte text with these rules is that of its longest
    // redacted form and its copy, not that of 4 bytes.
    let redacted = rules.apply("a a a").len() as u64;
    assert!(scorer.scoring_bytes(5) >= plain.scoring_bytes(redacted) + redacted);
    assert!(scorer.scoring_bytes(5) > plain.scoring_bytes(5));
}
