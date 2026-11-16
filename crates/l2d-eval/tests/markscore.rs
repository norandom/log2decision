//! The held-out-group report of a score fitted from a mark set (requirement 16.9; task
//! 1.23).
//!
//! The logs are made here from a seeded generator: requests to 40 endpoints whose names
//! are random words, with random parameter names and values. Two mark sets are made on
//! them the way an analyst would make them, and neither looks at a fold:
//!
//! - **agnostic**: the logs that carry an injection, found by a regular expression; the
//!   injections are spread over every endpoint;
//! - **endpoint-specific** (planted): every log of 20 endpoints drawn by the generator,
//!   named by position. Nothing but the endpoint tells these logs apart.
//!
//! The report decides by itself which groups it holds out (a hash of the group key) and
//! whether to warn. The test only reads its verdict.

use std::collections::BTreeSet;

use l2d_baseline::BaselineSettings;
use l2d_baseline::markscore::{MarkFit, MarkFitSettings, fit_from_marks};
use l2d_baseline::redact::RedactionInput;
use l2d_core::marks::{MarkQuery, MarkSet};
use l2d_core::{InputSummary, Label, LogRecord, LogsFrame, MemoryBudget, MemorySettings};
use l2d_eval::markscore::{DEFAULT_MARGIN, HeldOutSettings, held_out_report};
use l2d_eval::split::{Folds, hash_fold, units};
use sha2::{Digest, Sha256};

const GIB: u64 = 1024 * 1024 * 1024;
const ENDPOINTS: usize = 40;
const PER_ENDPOINT: usize = 24;

fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * GIB)).unwrap()
}

/// xorshift64*: a fixed stream of numbers for a seed.
struct Generator(u64);

impl Generator {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn word(&mut self, shortest: u64, longest: u64) -> String {
        let length = shortest + self.below(longest - shortest + 1);
        (0..length)
            .map(|_| char::from(b'a' + self.below(26) as u8))
            .collect()
    }

    fn alphanumeric(&mut self, length: u64) -> String {
        const SET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
        (0..length)
            .map(|_| char::from(SET[self.below(SET.len() as u64) as usize]))
            .collect()
    }
}

struct Synthetic {
    logs: LogsFrame,
    summary: InputSummary,
    /// The endpoints the planted mark set covers.
    planted: BTreeSet<String>,
}

fn synthetic(seed: u64) -> Synthetic {
    let mut generator = Generator(seed);
    let endpoints: Vec<String> = (0..ENDPOINTS)
        .map(|_| format!("/{}/{}.php", generator.word(4, 7), generator.word(5, 9)))
        .collect();
    let mut planted = BTreeSet::new();
    while planted.len() < ENDPOINTS / 2 {
        planted.insert(endpoints[generator.below(ENDPOINTS as u64) as usize].clone());
    }
    let mut records = Vec::new();
    let mut seen = BTreeSet::new();
    for endpoint in &endpoints {
        let mut made = 0;
        while made < PER_ENDPOINT {
            let parameters: Vec<String> = (0..1 + generator.below(3))
                .map(|_| {
                    let name = generator.word(2, 6);
                    let length = 3 + generator.below(8);
                    format!("{name}={}", generator.alphanumeric(length))
                })
                .collect();
            let mut query = parameters.join("&");
            if generator.below(3) == 0 {
                let a = generator.below(100);
                query.push_str(&format!("'+OR+{a}={a}--"));
            }
            let text = format!("GET http://shop.example{endpoint}?{query}");
            if !seen.insert(text.clone()) {
                continue;
            }
            made += 1;
            records.push(LogRecord {
                text,
                label: Label::Unknown,
                group: Some(endpoint.clone()),
                first_position: records.len() as u64 + 1,
                occurrences: 1,
            });
        }
    }
    let logs = LogsFrame::from_records(&records).unwrap();
    let mut hash = Sha256::new();
    for record in &records {
        hash.update(record.text.as_bytes());
        hash.update(b"\n");
    }
    let summary = InputSummary {
        source_path: "synthetic.log".into(),
        records: records.len() as u64,
        distinct_texts: records.len() as u64,
        content_sha256: format!("{:x}", hash.finalize()),
        skipped: Vec::new(),
        label_conflicts: 0,
    };
    Synthetic {
        logs,
        summary,
        planted,
    }
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

fn fit(data: &Synthetic, marks: &MarkSet) -> MarkFit {
    let mut budget = budget();
    fit_from_marks(
        marks,
        None,
        RedactionInput::File {
            logs: &data.logs,
            summary: &data.summary,
        },
        &settings(),
        &mut budget,
    )
    .unwrap()
}

fn agnostic_marks(data: &Synthetic) -> MarkSet {
    MarkSet::mark(
        "injections",
        Label::Bad,
        MarkQuery::Regex {
            pattern: r"'\+OR\+(\d+)=\d+--".into(),
        },
        &data.logs,
        &data.summary,
        &budget(),
    )
    .unwrap()
}

fn planted_marks(data: &Synthetic) -> MarkSet {
    let positions: Vec<u64> = data
        .logs
        .records()
        .filter(|record| data.planted.contains(record.group.as_deref().unwrap()))
        .map(|record| record.first_position)
        .collect();
    MarkSet::mark(
        "endpoints",
        Label::Bad,
        MarkQuery::Positions { positions },
        &data.logs,
        &data.summary,
        &budget(),
    )
    .unwrap()
}

#[test]
fn folds_are_dealt_by_a_hash_of_the_group_key() {
    let groups: Vec<String> = (0..50).map(|n| format!("/path/{n}")).collect();
    let mut repeated: Vec<&str> = Vec::new();
    for (index, group) in groups.iter().enumerate() {
        for _ in 0..=index % 4 {
            repeated.push(group);
        }
    }
    let found = units(&repeated, &[]).unwrap();
    let folds = Folds::hashed(&found, 5).unwrap();
    assert_eq!(folds.len(), 5);
    for group in &groups {
        let digest = Sha256::digest(group.as_bytes());
        let expected = (u64::from_le_bytes(digest[..8].try_into().unwrap()) % 5) as usize;
        assert_eq!(hash_fold(group, 5), expected);
        assert_eq!(folds.fold_of(group), Some(expected));
    }
    // The order of the logs and the sizes of the groups play no part.
    let mut reversed = repeated.clone();
    reversed.reverse();
    let other = Folds::hashed(&units(&reversed[..60], &[]).unwrap(), 5).unwrap();
    for group in &groups[..10] {
        if other.fold_of(group).is_some() {
            assert_eq!(other.fold_of(group), folds.fold_of(group));
        }
    }
    // A fold no group hashes to stays empty rather than taking another group's logs.
    let two = units(&["a", "b"], &[]).unwrap();
    let wide = Folds::hashed(&two, 2).unwrap();
    for group in ["a", "b"] {
        assert_eq!(wide.fold_of(group), Some(hash_fold(group, 2)));
    }
}

#[test]
fn the_warning_fires_on_an_endpoint_specific_score_and_not_on_an_agnostic_one() {
    let data = synthetic(0x5eed_0001);
    let report_of = |marks: &MarkSet| {
        let fit = fit(&data, marks);
        held_out_report(&fit, &HeldOutSettings::default(), &mut budget())
            .unwrap()
            .expect("the logs have group keys")
    };
    let agnostic = report_of(&agnostic_marks(&data));
    let planted = report_of(&planted_marks(&data));
    eprintln!("agnostic:\n{agnostic}\nendpoint-specific:\n{planted}");

    assert_eq!(agnostic.margin, DEFAULT_MARGIN);
    assert!(agnostic.warning.is_none(), "{agnostic}");
    assert!(planted.warning.is_some(), "{planted}");
    let warning = planted.warning.as_deref().unwrap();
    assert!(
        warning.contains("depends on the groups it has seen"),
        "{warning}"
    );
    // Both name the group key and hold every group out exactly once.
    for report in [&agnostic, &planted] {
        assert!(report.group_key.contains("group"), "{}", report.group_key);
        assert_eq!(report.groups, ENDPOINTS as u64);
        assert_eq!(report.held_out.logs, (ENDPOINTS * PER_ENDPOINT) as u64);
        let held: u64 = report.folds.iter().map(|fold| fold.held_out_groups).sum();
        assert_eq!(held, ENDPOINTS as u64);
    }
    // The figures behind the verdicts.
    let held = planted.held_out.roc_auc.unwrap();
    let fitted = planted.fitted.roc_auc.unwrap();
    assert!(fitted - held > DEFAULT_MARGIN, "{planted}");
    assert_eq!(planted.difference, Some(fitted - held));
}

#[test]
fn the_report_is_deterministic() {
    let data = synthetic(0x5eed_0002);
    let marks = agnostic_marks(&data);
    let first = held_out_report(
        &fit(&data, &marks),
        &HeldOutSettings::default(),
        &mut budget(),
    )
    .unwrap()
    .unwrap();
    let second = held_out_report(
        &fit(&data, &marks),
        &HeldOutSettings::default(),
        &mut budget(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.to_string(), second.to_string());
}

#[test]
fn the_margin_and_the_folds_are_settable_and_checked() {
    let data = synthetic(0x5eed_0003);
    let fit = fit(&data, &planted_marks(&data));
    // A margin no gap reaches: no warning.
    let lenient = HeldOutSettings {
        margin: 1.0,
        ..HeldOutSettings::default()
    };
    let report = held_out_report(&fit, &lenient, &mut budget())
        .unwrap()
        .unwrap();
    assert!(report.warning.is_none());
    assert_eq!(report.margin, 1.0);
    let three = HeldOutSettings {
        folds: 3,
        ..HeldOutSettings::default()
    };
    let report = held_out_report(&fit, &three, &mut budget())
        .unwrap()
        .unwrap();
    assert_eq!(report.folds.len(), 3);
    for bad in [
        HeldOutSettings {
            folds: 1,
            ..HeldOutSettings::default()
        },
        HeldOutSettings {
            margin: -0.1,
            ..HeldOutSettings::default()
        },
        HeldOutSettings {
            margin: f64::NAN,
            ..HeldOutSettings::default()
        },
    ] {
        assert!(matches!(
            held_out_report(&fit, &bad, &mut budget()),
            Err(l2d_core::L2dError::InvalidInput { .. })
        ));
    }
}

#[test]
fn without_group_keys_there_is_no_report() {
    let data = synthetic(0x5eed_0004);
    let records: Vec<LogRecord> = data
        .logs
        .records()
        .map(|record| LogRecord {
            group: None,
            ..record
        })
        .collect();
    let logs = LogsFrame::from_records(&records).unwrap();
    let marks = MarkSet::mark(
        "injections",
        Label::Bad,
        MarkQuery::Regex {
            pattern: r"'\+OR\+".into(),
        },
        &logs,
        &data.summary,
        &budget(),
    )
    .unwrap();
    let mut budget = budget();
    let fit = fit_from_marks(
        &marks,
        None,
        RedactionInput::File {
            logs: &logs,
            summary: &data.summary,
        },
        &settings(),
        &mut budget,
    )
    .unwrap();
    assert!(
        held_out_report(&fit, &HeldOutSettings::default(), &mut budget)
            .unwrap()
            .is_none()
    );
    assert_eq!(budget.held_bytes(), 0);
}
