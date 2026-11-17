//! A score fitted from a CSIC mark set, and its held-out-group report (task 1.23, "done
//! when"): the score gives ratios, survives save and load, and reports held-out against
//! fitted groups, without and with the redaction rules the analyst confirms.
//!
//! The mark set stands in for an analyst's marks: the SQL injections that drop a table,
//! found by a regular expression (as in task 1.22). The CSIC labels play no part. This is
//! not gate 2.18: one mark set, one report each, no study over folds and seeds.
//!
//! On request, in release mode:
//!
//! ```text
//! cargo test --release -p l2d-eval --test markscore_csic -- --ignored --nocapture
//! ```
//!
//! The dataset is `data/csic-2010/csic_database.csv` below the repository, or the file
//! or directory `L2D_CSIC_CSV` names; without it the test is skipped with a note.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use l2d_baseline::BaselineScorer;
use l2d_baseline::markscore::{MarkFit, MarkFitSettings, OtherClass, fit_from_marks};
use l2d_baseline::redact::{RedactionInput, RedactionRules, RedactionSettings, propose_redactions};
use l2d_core::adapters::csic2010::group_key;
use l2d_core::marks::{MarkQuery, MarkSet};
use l2d_core::{
    Csic2010Adapter, InputAdapter, InputSummary, L2dError, Label, LogsFrame, MemoryBudget,
    MemorySettings, ProjectPaths,
};
use l2d_eval::markscore::{HeldOutSettings, held_out_report};

const DROP_TABLE: &str = r"(?i)drop(\+|%20|\s)+table";
const HOST: &str = "localhost:8080";

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// A directory of its own in the system temp folder, removed when dropped.
struct Scratch {
    dir: PathBuf,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn fit_and_report(
    label: &str,
    marks: &MarkSet,
    logs: &LogsFrame,
    summary: &InputSummary,
    paths: &ProjectPaths,
    budget: &mut MemoryBudget,
) -> MarkFit {
    let started = Instant::now();
    let fit = fit_from_marks(
        marks,
        None,
        RedactionInput::File { logs, summary },
        &MarkFitSettings::default(),
        budget,
    )
    .unwrap();
    let fitted = started.elapsed();
    assert_eq!(fit.other_class, OtherClass::Unmarked);
    let report = fit.scorer.fit_report();
    println!(
        "CSIC {label}: fitted in {fitted:.2?}: {} rows ({} bad = marked, {} good = unmarked), \
         {} iterations, {:?}, rules {} (sha256 {}), revision {}",
        report.rows,
        report.bad,
        report.good,
        report.iterations,
        report.termination,
        fit.scorer.rules().rules().len(),
        fit.scorer.rules().content_sha256(),
        fit.scorer.revision()
    );

    // Ratios for every log, and the same after save and load.
    let texts: Vec<String> = logs.records().map(|record| record.text).collect();
    let started = Instant::now();
    let ratios: Vec<f32> = texts.iter().map(|text| fit.scorer.ratio(text)).collect();
    let scoring = started.elapsed();
    assert!(ratios.iter().all(|ratio| (0.0..=1.0).contains(ratio)));
    let dir = fit.scorer.save(paths, label).unwrap();
    let loaded = BaselineScorer::load_under(paths, &dir).unwrap();
    assert_eq!(loaded.revision(), fit.scorer.revision());
    assert_eq!(loaded.mark_set_origin().unwrap().marks(), marks);
    for (text, ratio) in texts.iter().zip(&ratios) {
        assert_eq!(loaded.ratio(text).to_bits(), ratio.to_bits());
    }
    println!(
        "CSIC {label}: {} logs scored in {scoring:.2?}; saved to {} and loaded with equal \
         ratios ({} bytes to load)",
        texts.len(),
        dir.display(),
        BaselineScorer::load_bytes(&dir)
    );

    let started = Instant::now();
    let report = held_out_report(&fit, &HeldOutSettings::default(), budget)
        .unwrap()
        .expect("CSIC logs have group keys");
    println!(
        "CSIC {label}: held-out-group report in {:.2?}\n{report}",
        started.elapsed()
    );
    assert_eq!(report.logs, texts.len() as u64);
    assert!(report.held_out.roc_auc.is_some());
    assert!(report.fitted.roc_auc.is_some());
    fit
}

#[test]
#[ignore = "CSIC measurement; run in release mode on request"]
fn a_score_fitted_from_a_csic_mark_set_reports_held_out_against_fitted_groups() {
    let supplied = std::env::var_os("L2D_CSIC_CSV").map(PathBuf::from);
    let path = match Csic2010Adapter::locate(&ProjectPaths::new(repository()), supplied.as_deref())
    {
        Ok(path) => path,
        Err(error @ L2dError::Missing { .. }) => {
            eprintln!("SKIPPED: {error} (set L2D_CSIC_CSV to csic_database.csv)");
            return;
        }
        Err(other) => panic!("locating the dataset failed: {other:?}"),
    };
    let scratch = Scratch {
        dir: std::env::temp_dir().join(format!("l2d-eval-markscore-csic-{}", std::process::id())),
    };
    let _ = std::fs::remove_dir_all(&scratch.dir);
    let paths = ProjectPaths::new(scratch.dir.join("project"));
    let mut budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();
    let (logs, summary, _) = Csic2010Adapter::new()
        .read_with(&path, &mut budget)
        .unwrap();
    let mut set = MarkSet::mark(
        "drop-table",
        Label::Bad,
        MarkQuery::Regex {
            pattern: DROP_TABLE.into(),
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap();
    println!(
        "CSIC: {} logs; mark set `{}`: {} marked",
        logs.len(),
        set.name(),
        set.count()
    );

    let plain = fit_and_report(
        "drop-table-plain",
        &set,
        &logs,
        &summary,
        &paths,
        &mut budget,
    );

    // The rules of task 1.22: the host and the endpoint paths of the marked logs that are
    // proposed, confirmed by the analyst and saved with the mark set.
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
    let endpoints: BTreeSet<String> = plain
        .logs
        .records()
        .filter(|record| record.label == Label::Bad)
        .map(|record| group_key(&record.text))
        .collect();
    let confirmed: Vec<_> = proposals
        .proposals
        .iter()
        .filter(|proposal| proposal.fragment == HOST || endpoints.contains(&proposal.fragment))
        .collect();
    let rules = RedactionRules::confirmed(confirmed.iter().copied(), &budget).unwrap();
    println!(
        "CSIC rules: {} confirmed of {} proposals (sha256 {}): {:?}",
        rules.rules().len(),
        proposals.proposals.len(),
        rules.content_sha256(),
        rules
            .rules()
            .iter()
            .map(|rule| rule.fragment.as_str())
            .collect::<Vec<_>>()
    );
    rules.save_with(&mut set, &paths, &budget).unwrap();
    let redacted = fit_and_report(
        "drop-table-redacted",
        &set,
        &logs,
        &summary,
        &paths,
        &mut budget,
    );
    assert_ne!(redacted.scorer.revision(), plain.scorer.revision());
    assert_eq!(redacted.scorer.rules(), &rules);
}
