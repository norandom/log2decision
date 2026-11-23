//! Opens an immutable, local projection of evidence already retained by the analyst.
use crate::flow::{CProducer, Dataset, Occurrence, Recommendation, Record};
use l2d_core::input::content_sha256;
use l2d_core::marks::InputInterpretation;
use l2d_core::{L2dError, MemoryBudget};
use l2d_decide::policy::{PolicyReport, Queue};
use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const DEMO: &[u8] = include_bytes!("../fixtures/auth-demo.json");
const MAX_REPORT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_RECORDS: usize = 25_000;

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum DemoOpinion {
    Recommend,
    Pass,
    Abstain,
}

#[derive(Deserialize)]
struct DemoRow {
    id: String,
    text: String,
    score: f64,
    opinion: Option<DemoOpinion>,
    reason: Option<String>,
    events: Vec<(String, String)>,
}

/// The public synthetic sign-in fixture from the analyst walkthrough.
pub fn demo() -> Dataset {
    let rows: Vec<DemoRow> = serde_json::from_slice(DEMO).expect("compiled auth demo fixture");
    let records = rows
        .into_iter()
        .map(|row| Record {
            id: row.id,
            text: row.text,
            score: row.score,
            recommendation: match row.opinion {
                Some(DemoOpinion::Recommend) => Recommendation::In,
                Some(DemoOpinion::Pass) => Recommendation::Out,
                Some(DemoOpinion::Abstain) | None => Recommendation::Unknown,
            },
            reason: row
                .reason
                .unwrap_or_else(|| "Outside the fixture recommendation batch.".into()),
            occurrences: row
                .events
                .into_iter()
                .map(|(host, minute)| Occurrence {
                    host: Some(host),
                    time: Some(format!("2026-11-18T{minute}:00Z")),
                    count: 1,
                })
                .collect(),
        })
        .collect();
    Dataset {
        title: "Synthetic sign-in walkthrough".into(),
        provenance: "Public design fixture · fixture-score · fixture-review issued batch on A minus B · synthetic UTC occurrences".into(),
        source_revision: content_sha256(DEMO),
        c_producer: CProducer::Fixture,
        occurrences_known: true,
        records,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyMembership {
    NoRecommendations,
    /// Saved Investigate and Review routes are in C, including unsupported cases sent to Review.
    InvestigateOrReview,
}

pub fn load_report(path: &Path, budget: &MemoryBudget) -> Result<Dataset, L2dError> {
    load_report_with_membership(path, budget, PolicyMembership::NoRecommendations)
}

/// The JSON length is known before decoding. The expansion estimate follows the
/// saved policy loader's 32x charge for input, parsed support, and working memory.
pub fn load_report_with_membership(
    path: &Path,
    budget: &MemoryBudget,
    membership: PolicyMembership,
) -> Result<Dataset, L2dError> {
    let io_error = |e| L2dError::Runtime {
        detail: format!("reading policy report {}: {e}", path.display()),
    };
    let file = File::open(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => L2dError::Missing {
            item: "policy report".into(),
            expected_at: path.to_owned(),
        },
        _ => io_error(e),
    })?;
    let metadata = file.metadata().map_err(io_error)?;
    if !metadata.is_file() {
        return Err(L2dError::InvalidInput {
            detail: "Expected a saved policy report in a regular file.".into(),
        });
    }
    let len = metadata.len();
    if len > MAX_REPORT_BYTES {
        return Err(L2dError::Refused {
            reason: format!(
                "Report is {len} bytes. Open a smaller saved policy report (maximum {MAX_REPORT_BYTES} bytes)."
            ),
        });
    }
    let needed = (len + 1) * 32 + 1024 * 1024;
    budget.require_whole(needed, "opening a saved triager policy report")?;
    let mut bytes = Vec::with_capacity(len as usize + 1);
    file.take(len + 1)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if bytes.len() as u64 != len {
        return Err(L2dError::Refused {
            reason: "Policy report changed size while opening.".into(),
        });
    }
    project(&bytes, membership).map_err(|detail| L2dError::InvalidInput { detail })
}

fn project(bytes: &[u8], membership: PolicyMembership) -> Result<Dataset, String> {
    let report: PolicyReport = serde_json::from_slice(bytes)
        .map_err(|e| format!("Expected a saved l2d triage policy report: {e}"))?;
    if report.decisions.len() > MAX_RECORDS {
        return Err(format!(
            "Report has {} distinct originals. Open a smaller saved report (maximum {MAX_RECORDS} originals).",
            report.decisions.len()
        ));
    }
    let report = report.validate_saved().map_err(|e| e.to_string())?;
    if [
        &report.actual_profile,
        &report.evidence_revision,
        &report.scorer_revision,
    ]
    .iter()
    .any(|revision| revision.trim().is_empty())
    {
        return Err("Missing saved profile, scorer or evidence revision.".into());
    }
    let source = match &report.query_source {
        Some(source) => {
            if source.path.trim().is_empty() || !is_digest(&source.content_sha256) {
                return Err(
                    "Missing source path or invalid source SHA-256 in the policy report.".into(),
                );
            }
            let interpretation = match (&source.archive_entries, &source.interpretation) {
                (_, None) => "interpretation unrecorded".into(),
                (None, Some(InputInterpretation::FileRowsV1 { sha256 })) if is_digest(sha256) => {
                    format!("file rows {sha256}")
                }
                (Some(entries), Some(InputInterpretation::ArchiveLinesV1 { max_record_bytes }))
                    if *max_record_bytes > 0 =>
                {
                    format!(
                        "archive entry patterns {entries:?}, max record {max_record_bytes} bytes"
                    )
                }
                _ => return Err("Inconsistent source interpretation in the policy report.".into()),
            };
            for row in &report.decisions {
                if l2d_store::log_id(&source.path, &row.input.text) != row.input.id {
                    return Err(
                        "A policy original ID differs from its saved source path and text.".into(),
                    );
                }
            }
            format!(
                "{} · SHA-256 {} · {interpretation}",
                source.path, source.content_sha256
            )
        }
        None => "Report-local IDs; query source was not recorded".into(),
    };
    let (c_producer, producer) = match membership {
        PolicyMembership::NoRecommendations => (CProducer::Unavailable, "C unavailable"),
        PolicyMembership::InvestigateOrReview => (
            CProducer::PolicyInvestigateOrReview,
            "C = Venn Investigate or Review",
        ),
    };
    let provenance = format!(
        "{source}\nSaved policy {} · {}\nA uses the saved mark-set diagnostic ratio · scorer {}\nHistory {} · vector profile {}\nSaved qualification: {}\n{producer}. The TUI creates a new analyst work set. The original policy allocation is recorded in the report.",
        report.policy.version,
        report.policy.selection_provenance,
        report.scorer_revision,
        report.evidence_revision,
        report.actual_profile,
        report.qualification,
    );
    let records = report.decisions.into_iter().map(|row| {
        let recommendation = match membership {
            PolicyMembership::NoRecommendations => Recommendation::Unknown,
            PolicyMembership::InvestigateOrReview => match row.queue {
                Queue::Investigate | Queue::Review => Recommendation::In,
                Queue::LowPriority => Recommendation::Out,
            },
        };
        let actionable = row.score.actionable_ratio.map_or_else(|| "withheld".to_owned(), |n| n.to_string());
        let group = row.input.group.as_deref().unwrap_or("unrecorded");
        let comparison = if row.score.comparison_class_assumed { "assumed" } else { "explicit" };
        let reason = format!(
            "{producer}. Saved policy {:?} ({:?}). Actionable ratio {actionable}. Comparison class {comparison}. Endpoint group: {group}. {}",
            row.queue, row.reason, row.explanation,
        );
        Record {
            id: row.input.id,
            text: row.input.text,
            score: f64::from(row.score.diagnostic_ratio),
            recommendation,
            reason,
            occurrences: Vec::new(),
        }
    }).collect();
    Ok(Dataset {
        title: format!(
            "{} · A diagnostic mark-set · {producer}",
            report.policy.version
        ),
        provenance,
        source_revision: content_sha256(bytes),
        c_producer,
        occurrences_known: false,
        records,
    })
}

fn is_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::io::Write;

    fn budget() -> MemoryBudget {
        MemoryBudget::observe_with(&l2d_core::MemorySettings::default(), || Ok(4_u64 << 30))
            .unwrap()
    }

    fn report() -> Value {
        let source_path = "/cases/auth.csv";
        let text = "authentication challenge rejected";
        json!({
            "format": 1,
            "query_source": {
                "path": source_path, "content_sha256": "0".repeat(64),
                "archive_entries": null, "interpretation": {"kind":"file_rows_v1","sha256":"1".repeat(64)}
            },
            "policy": {
                "version":"test-v1", "selection_provenance":"synthetic test", "low":0.2,"high":0.8,
                "corroboration":1,"reach":{"radius":0.2,"top_k":2,"minimum_labelled_support":1,"expected_profile":"expected"},
                "require_calibration":false,"review_budget":4,"unknown_slots":1,"audit_slots":1,"audit_seed":5
            },
            "eligibility": l2d_decide::policy::ELIGIBILITY, "allocation_rule": l2d_decide::policy::ALLOCATION,
            "scorer_revision":"scorer-v1", "evidence_revision":"history-v1", "actual_profile":"incompatible",
            "calibration":"Missing",
            "qualification": "Exploratory; qualification incomplete. Policy mechanics and local support do not establish deployment quality.",
            "decisions": [{
                "input":{"id":l2d_store::log_id(source_path, text),"text":text,"group":"/auth/login"},
                "score":{"kind":"mark-set","revision":"scorer-v1","comparison_class_assumed":true,"diagnostic_ratio":0.75,"actionable_ratio":null},
                "vector": null,"queue":"Review","reason":"Unsupported",
                "explanation":"incompatible exploration profile; actionable ratio withheld", "allocation":"UnknownConflict"
            }],
            "unknown":{"configured":1,"selected":1,"unused":0}, "general":{"configured":2,"selected":0,"unused":2},
            "audit":{"configured":1,"selected":0,"unused":1},"deferred":0
        })
    }

    #[test]
    fn default_and_explicit_policy_keep_score_coverage_and_units_distinct() {
        let bytes = serde_json::to_vec(&report()).unwrap();
        let path = tempfile::NamedTempFile::new().unwrap();
        path.as_file().write_all(&bytes).unwrap();
        let budget = budget();
        let before = budget.usage().report().counted_peak_bytes;
        let plain = load_report(path.path(), &budget).unwrap();
        assert!(budget.usage().report().counted_peak_bytes >= before + 32 * bytes.len() as u64);
        let row = &plain.records[0];
        assert_eq!(plain.source_revision, content_sha256(&bytes));
        assert!(!plain.occurrences_known);
        assert!(row.occurrences.is_empty());
        assert_eq!(row.score, 0.75);
        assert_eq!(row.recommendation, Recommendation::Unknown);
        assert!(row.reason.contains("Actionable ratio withheld"));
        assert!(row.reason.contains("Endpoint group: /auth/login"));
        assert!(plain.provenance.contains("history-v1"));
        let venn = load_report_with_membership(
            path.path(),
            &budget,
            PolicyMembership::InvestigateOrReview,
        )
        .unwrap();
        assert_eq!(venn.records[0].recommendation, Recommendation::In);
        assert!(venn.title.contains("Venn Investigate or Review"));
        assert_eq!(venn.source_revision, plain.source_revision);
    }

    /// Parent task 2.23: a report allocated with a separate unsupported reservation.
    #[test]
    fn a_report_with_a_separate_unsupported_reservation_projects_the_same_records() {
        let mut value = report();
        value["policy"]["unsupported_slots"] = json!(1);
        value["allocation_rule"] = json!(l2d_decide::policy::ALLOCATION_SEPARATE_UNSUPPORTED);
        value["decisions"][0]["allocation"] = json!("Unsupported");
        value["unsupported"] = json!({"configured":1,"selected":1,"unused":0});
        value["unknown"] = json!({"configured":1,"selected":0,"unused":1});
        value["general"] = json!({"configured":1,"selected":0,"unused":1});
        let shared = project(
            &serde_json::to_vec(&report()).unwrap(),
            PolicyMembership::InvestigateOrReview,
        )
        .unwrap();
        let separate = project(
            &serde_json::to_vec(&value).unwrap(),
            PolicyMembership::InvestigateOrReview,
        )
        .unwrap();
        assert_eq!(separate.records.len(), 1);
        assert_eq!(separate.records[0].id, shared.records[0].id);
        assert_eq!(separate.records[0].score, shared.records[0].score);
        assert_eq!(separate.records[0].recommendation, Recommendation::In);
        // The rule and the stated capacity must agree.
        value["allocation_rule"] = json!(l2d_decide::policy::ALLOCATION);
        assert!(
            project(
                &serde_json::to_vec(&value).unwrap(),
                PolicyMembership::InvestigateOrReview
            )
            .is_err()
        );
    }

    #[test]
    fn validates_retained_decisions_and_source_join_before_projection() {
        let bad = [
            ("/decisions/0/queue", json!("LowPriority")),
            ("/decisions/0/score/revision", json!("changed")),
            ("/decisions/0/input/text", json!("different original")),
            ("/query_source/content_sha256", json!("unbound")),
            ("/audit/selected", json!(2)),
        ];
        for (pointer, change) in bad {
            let mut value = report();
            *value.pointer_mut(pointer).unwrap() = change;
            assert!(
                project(
                    &serde_json::to_vec(&value).unwrap(),
                    PolicyMembership::InvestigateOrReview
                )
                .is_err(),
                "{pointer}"
            );
        }
        let mut legacy = report();
        legacy.as_object_mut().unwrap().remove("query_source");
        let dataset = project(
            &serde_json::to_vec(&legacy).unwrap(),
            PolicyMembership::NoRecommendations,
        )
        .unwrap();
        assert!(dataset.provenance.contains("Report-local IDs"));
    }

    #[test]
    fn refuses_input_size_or_working_set_before_json_decoding() {
        let path = tempfile::NamedTempFile::new().unwrap();
        path.as_file().set_len(MAX_REPORT_BYTES + 1).unwrap();
        assert!(
            load_report(path.path(), &budget())
                .err()
                .unwrap()
                .to_string()
                .contains("maximum")
        );
        path.as_file().set_len(16).unwrap();
        let mut limited = budget();
        limited.hold(limited.usable_bytes() - 1024);
        assert!(matches!(
            load_report(path.path(), &limited),
            Err(L2dError::Refused { .. })
        ));
    }

    #[test]
    fn walkthrough_retains_real_event_times_and_an_explained_abstention() {
        let set = demo();
        assert!(set.occurrences_known);
        assert_eq!(set.records.len(), 12);
        assert_eq!(
            set.records
                .iter()
                .map(|r| r.occurrences.len())
                .sum::<usize>(),
            20
        );
        assert_eq!(set.records[2].id, "R03");
        assert_eq!(
            set.records[2].occurrences[1].time.as_deref(),
            Some("2026-11-18T08:46:00Z")
        );
        assert_eq!(set.records[7].recommendation, Recommendation::Unknown);
        assert!(set.records[7].reason.contains("abstained"));
        assert!(set.records[8].reason.contains("Outside"));
    }
}
