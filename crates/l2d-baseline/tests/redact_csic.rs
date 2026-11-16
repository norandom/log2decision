//! Redaction on HTTP CSIC 2010 (task 1.22): the proposals for a mark set of SQL
//! injections name the host and the endpoint paths with their counts, no payload token
//! of the attack is proposed, confirmed rules change the features as stated, and the
//! routing option of requirement 2.6 gives the features of its rule on every log.
//!
//! On request, in release mode:
//!
//! ```text
//! cargo test --release -p l2d-baseline --test redact_csic -- --ignored --nocapture
//! ```
//!
//! The dataset is `data/csic-2010/csic_database.csv` below the repository, or the file
//! or directory `L2D_CSIC_CSV` names; without it the test is skipped with a note. The
//! counts asserted are computed here from the texts by a restatement of the cutting
//! rule, independently of the code under test.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use l2d_baseline::redact::{
    DEFAULT_PLACEHOLDER, RedactionInput, RedactionRules, RedactionSettings, propose_redactions,
};
use l2d_baseline::{CharHasher, HashingSettings, RoutingMode};
use l2d_core::adapters::csic2010::group_key;
use l2d_core::marks::{MarkQuery, MarkSet};
use l2d_core::{
    Csic2010Adapter, InputAdapter, L2dError, Label, LogsFrame, MemoryBudget, MemorySettings,
    ProjectPaths,
};

/// The SQL injections that drop a table: what an analyst finds with a regular expression.
const DROP_TABLE: &str = r"(?i)drop(\+|%20|\s)+table";
const HOST: &str = "localhost:8080";

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn texts(logs: &LogsFrame) -> Vec<String> {
    logs.records().map(|record| record.text).collect()
}

/// The words of a text: the runs between the separators of the cutting rule.
fn words(text: &str) -> Vec<&str> {
    text.split(|c: char| {
        c.is_ascii_whitespace() || c.is_ascii_control() || "\"'(),;<=>?[\\]`{|}&".contains(c)
    })
    .filter(|word| !word.is_empty())
    .collect()
}

/// Whether `path` (starting with `/`) is a path suffix of a word of `text`.
fn has_path(text: &str, path: &str) -> bool {
    words(text).iter().any(|word| {
        word.strip_suffix(path)
            .is_some_and(|before| !before.ends_with('/'))
            && !path[1..].starts_with('/')
    })
}

/// Whether `c` continues a token: not a separator of the cutting rule and not `/`.
fn continues_token(c: char) -> bool {
    !(c == '/'
        || c.is_ascii_whitespace()
        || c.is_ascii_control()
        || "\"'(),;<=>?[\\]`{|}&".contains(c))
}

/// `text` with `fragment` replaced by the placeholder wherever the character before a
/// fragment that starts with a token character, and the one after a fragment that ends
/// with one, do not continue the token.
fn replace_where_bounded(text: &str, fragment: &str) -> String {
    let opens = fragment.chars().next().is_some_and(continues_token);
    let closes = fragment.chars().last().is_some_and(continues_token);
    let mut out = String::new();
    let mut rest = text;
    let mut before: Option<char> = None;
    while let Some(at) = rest.find(fragment) {
        let prior = rest[..at].chars().last().or(before);
        let after = rest[at + fragment.len()..].chars().next();
        let bounded = (!opens || !prior.is_some_and(continues_token))
            && (!closes || !after.is_some_and(continues_token));
        if bounded {
            out.push_str(&rest[..at]);
            out.push_str(DEFAULT_PLACEHOLDER);
            before = fragment.chars().last();
            rest = &rest[at + fragment.len()..];
        } else {
            let step = at + rest[at..].chars().next().map_or(1, char::len_utf8);
            out.push_str(&rest[..step]);
            before = rest[..step].chars().last();
            rest = &rest[step..];
        }
    }
    out.push_str(rest);
    out
}

/// Whether `token` is a segment of a word of `text`.
fn has_token(text: &str, token: &str) -> bool {
    words(text)
        .iter()
        .any(|word| word.split('/').any(|segment| segment == token))
}

#[test]
#[ignore = "CSIC measurement; run in release mode on request"]
fn csic_proposals_name_the_host_and_the_endpoint_paths_with_their_counts() {
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
    let mut budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();
    let started = Instant::now();
    let (logs, summary, _) = Csic2010Adapter::new()
        .read_with(&path, &mut budget)
        .unwrap();
    let read = started.elapsed();
    let set = MarkSet::mark(
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
    let all = texts(&logs);
    let positions: Vec<u64> = {
        let frame = logs.frame();
        frame
            .column(LogsFrame::FIRST_POSITION)
            .unwrap()
            .u64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    };
    let mut marked: Vec<&String> = Vec::new();
    let mut unmarked: Vec<&String> = Vec::new();
    for (text, position) in all.iter().zip(&positions) {
        if set.is_selected(*position) {
            marked.push(text);
        } else {
            unmarked.push(text);
        }
    }

    let started = Instant::now();
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
    let proposing = started.elapsed();
    println!(
        "CSIC: {} logs read in {read:.2?}; mark set `{}`: {} marked, {} unmarked; \
         proposals in {proposing:.2?}: {} (at least {} marked and {} unmarked logs; \
         counters {:.1} MiB, {} dropped, {} candidates recounted)",
        all.len(),
        set.name(),
        proposals.marked_logs,
        proposals.unmarked_logs,
        proposals.proposals.len(),
        proposals.min_marked,
        proposals.min_unmarked,
        proposals.counters_bytes as f64 / (1024.0 * 1024.0),
        proposals.dropped,
        proposals.candidates
    );
    for proposal in &proposals.proposals {
        println!(
            "CSIC proposal {:<6?} marked {:>5} unmarked {:>6}  {}",
            proposal.kind, proposal.marked, proposal.unmarked, proposal.fragment
        );
    }
    assert_eq!(proposals.marked_logs, marked.len() as u64);
    assert_eq!(proposals.unmarked_logs, unmarked.len() as u64);

    // The host, with its counts.
    let host = proposals.get(HOST).expect("the host is proposed");
    let count = |side: &[&String], test: &dyn Fn(&str) -> bool| {
        side.iter().filter(|text| test(text)).count() as u64
    };
    assert_eq!(
        (host.marked, host.unmarked),
        (
            count(&marked, &|text| has_token(text, HOST)),
            count(&unmarked, &|text| has_token(text, HOST))
        )
    );
    // Every endpoint path of the marked logs that is frequent on both sides is
    // proposed, with its counts; the others are not.
    let mut endpoints: BTreeMap<String, u64> = BTreeMap::new();
    for text in &marked {
        *endpoints.entry(group_key(text)).or_default() += 1;
    }
    let mut named = 0;
    for path in endpoints.keys() {
        let counts = (
            count(&marked, &|text| has_path(text, path)),
            count(&unmarked, &|text| has_path(text, path)),
        );
        let qualifies = counts.0 >= proposals.min_marked && counts.1 >= proposals.min_unmarked;
        match proposals.get(path) {
            Some(proposal) => {
                assert!(qualifies, "{path} {counts:?}");
                assert_eq!((proposal.marked, proposal.unmarked), counts, "{path}");
                named += 1;
                println!(
                    "CSIC endpoint {path}: marked {} unmarked {}",
                    counts.0, counts.1
                );
            }
            None => assert!(!qualifies, "{path} {counts:?} is not proposed"),
        }
    }
    assert!(named >= 2, "{endpoints:?}");
    // No payload token of the attack.
    for proposal in &proposals.proposals {
        let upper = proposal.fragment.to_uppercase();
        for payload in ["DROP", "TABLE", "SELECT", "%27", "%3B"] {
            assert!(!upper.contains(payload), "{proposal:?}");
        }
    }

    // The analyst confirms the host and the endpoint paths: those fragments become the
    // placeholder, and the features are those of the text so written.
    let confirmed: Vec<_> = proposals
        .proposals
        .iter()
        .filter(|proposal| proposal.fragment == HOST || endpoints.contains_key(&proposal.fragment))
        .collect();
    let rules = RedactionRules::confirmed(confirmed.iter().copied(), &budget).unwrap();
    let hasher = CharHasher::default();
    let mut changed = 0;
    for text in &all {
        // What the rules are stated to do: each fragment replaced where it does not
        // continue a token (`localhost:8080.old` keeps its host).
        let mut stated = text.clone();
        for rule in rules.rules() {
            stated = replace_where_bounded(&stated, &rule.fragment);
        }
        let redacted = rules.apply(text);
        assert_eq!(redacted, stated);
        if redacted != *text {
            changed += 1;
            assert_eq!(hasher.row_redacted(text, &rules), hasher.row(&stated));
        }
    }
    println!(
        "CSIC rules: {} confirmed (sha256 {}), {changed} of {} logs changed",
        rules.rules().len(),
        rules.content_sha256(),
        all.len()
    );
    assert!(changed > 0);

    // The routing option of 2.6 equals its rule on every log.
    let keep = CharHasher::default();
    let strip = CharHasher::new(HashingSettings {
        routing: RoutingMode::Strip,
        ..HashingSettings::default()
    })
    .unwrap();
    let routing = RedactionRules::routing(&budget).unwrap();
    let started = Instant::now();
    for text in &all {
        assert_eq!(keep.row_redacted(text, &routing), strip.row(text), "{text}");
    }
    println!(
        "CSIC routing rule = strip on {} logs in {:.2?}",
        all.len(),
        started.elapsed()
    );
}
