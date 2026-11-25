//! What the service tests share: an archive written into a temporary directory, an
//! independent oracle over its lines, a deterministic scorer, and a host around the
//! service under test.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use l2d_core::polars::prelude::Column;
use l2d_core::resource::BusyChoice;
use l2d_core::{ArchivePosition, L2dError, LogsFrame, MemoryBudget, MemorySettings};
use l2d_triager::scope::IndexPlacement;
use l2d_triager::service::{
    DisclosureProfile, FrozenScorer, HostConfig, MembershipRef, Reply, ScorerIdentity,
    ScorerSource, Service, SourceRef, TablePlacement,
};
use serde_json::{Value, json};

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;
pub const BASELINE: u64 = l2d_core::memory::PROCESS_BASELINE_BYTES;

/// The score the test scorer gives a text: a pure function with many ties.
pub fn score_of(text: &str) -> f32 {
    (text.bytes().map(u32::from).sum::<u32>() % 97) as f32 / 97.0
}

pub struct SumScorer {
    delay: Duration,
    /// Panics on its first batch: a job that ends abnormally.
    broken: bool,
}

impl FrozenScorer for SumScorer {
    fn identity(&self) -> ScorerIdentity {
        ScorerIdentity {
            kind: "test".into(),
            name: "sum".into(),
            revision: "rev-1".into(),
        }
    }

    fn held_bytes(&self) -> u64 {
        1024
    }

    fn scoring_bytes(&self, _longest_text_bytes: u64) -> u64 {
        0
    }

    fn score(&self, logs: &LogsFrame, out: &mut Vec<f32>) -> Result<(), L2dError> {
        assert!(
            !self.broken,
            "the scorer `boom` breaks down while it scores"
        );
        std::thread::sleep(self.delay);
        let texts = logs
            .frame()
            .column(LogsFrame::TEXT)
            .and_then(Column::str)
            .map_err(|error| L2dError::Runtime {
                detail: error.to_string(),
            })?;
        out.extend(texts.into_no_null_iter().map(score_of));
        Ok(())
    }
}

/// Knows the model `sum`, and `boom`, which panics while it scores; every batch
/// scored takes `delay`.
pub struct TestScorers {
    pub delay: Duration,
}

impl ScorerSource for TestScorers {
    fn load(
        &self,
        model: &str,
        budget: &mut MemoryBudget,
    ) -> Result<Box<dyn FrozenScorer>, L2dError> {
        if model != "sum" && model != "boom" {
            return Err(L2dError::Missing {
                item: format!("scorer '{model}'"),
                expected_at: PathBuf::from("models/baselines"),
            });
        }
        budget.require_whole(1024, "loading the test scorer")?;
        Ok(Box::new(SumScorer {
            delay: self.delay,
            broken: model == "boom",
        }))
    }
}

/// Writes an archive with the entries `(name, lines)`.
pub fn write_archive(path: &Path, entries: &[(String, Vec<String>)]) {
    let mut writer = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    for (name, lines) in entries {
        writer
            .start_file(name.as_str(), zip::write::SimpleFileOptions::default())
            .unwrap();
        for line in lines {
            writer.write_all(line.as_bytes()).unwrap();
            writer.write_all(b"\n").unwrap();
        }
    }
    writer.finish().unwrap();
}

/// The sign-in fixture: three entries, texts repeated within and across entries, and
/// one entry the `*.log` filter leaves out.
pub fn auth_entries() -> Vec<(String, Vec<String>)> {
    let mut auth = Vec::new();
    for index in 0..240 {
        auth.push(format!("login ok user=u{:02}", index % 30));
        if index % 3 == 0 {
            auth.push(format!(
                "login failed user=u{:02} attempt={}",
                index % 30,
                index % 7
            ));
        }
        if index % 5 == 0 {
            auth.push(format!("session opened id={index}"));
        }
    }
    let mut web = Vec::new();
    for index in 0..200 {
        web.push(format!("GET /page/{} 200", index % 60));
        if index % 4 == 0 {
            web.push(format!("POST /login user=u{:02} 302", index % 30));
        }
        if index % 10 == 0 {
            // The same text as in the first entry: its first position stays there.
            web.push(format!("login ok user=u{:02}", index % 30));
        }
        if index % 9 == 0 {
            web.push(format!("challenge rejected ip=10.0.0.{}", index % 20));
        }
    }
    let notes = vec!["login notes that the entry filter leaves out".to_string()];
    vec![
        ("app/auth.log".into(), auth),
        ("app/web.log".into(), web),
        ("doc/notes.txt".into(), notes),
    ]
}

#[derive(Debug, Clone, PartialEq)]
pub struct Orig {
    pub position: u64,
    pub entry: String,
    pub line: u64,
    pub text: String,
    pub occurrences: u64,
}

impl Orig {
    pub fn id(&self) -> String {
        record_id(self.position)
    }
}

pub fn record_id(position: u64) -> String {
    format!("r-{position:016x}")
}

/// The distinct originals of the entries whose name ends in one of `suffixes`, by
/// first position, computed without any code of the product beyond the packing of a
/// position.
pub fn oracle(entries: &[(String, Vec<String>)], suffix: &str) -> Vec<Orig> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    let mut originals: Vec<Orig> = Vec::new();
    for (index, (name, lines)) in entries.iter().enumerate() {
        if !name.ends_with(suffix) {
            continue;
        }
        for (line, text) in lines.iter().enumerate() {
            match seen.get(text.as_str()) {
                Some(&at) => originals[at].occurrences += 1,
                None => {
                    seen.insert(text, originals.len());
                    originals.push(Orig {
                        position: ArchivePosition {
                            entry: index as u32,
                            line: line as u64 + 1,
                        }
                        .pack(),
                        entry: name.clone(),
                        line: line as u64 + 1,
                        text: text.clone(),
                        occurrences: 1,
                    });
                }
            }
        }
    }
    originals.sort_by_key(|original| original.position);
    originals
}

/// The top `k` of `among` by score, the lower record ID first among equal scores.
pub fn top_k(among: &[Orig], k: usize) -> Vec<Orig> {
    let mut ranked: Vec<&Orig> = among.iter().collect();
    ranked.sort_by(|a, b| {
        score_of(&b.text)
            .total_cmp(&score_of(&a.text))
            .then(a.position.cmp(&b.position))
    });
    let mut top: Vec<Orig> = ranked.into_iter().take(k).cloned().collect();
    top.sort_by_key(|original| original.position);
    top
}

pub fn occurrences(originals: &[Orig]) -> u64 {
    originals.iter().map(|original| original.occurrences).sum()
}

pub fn ids(originals: &[Orig]) -> Vec<String> {
    originals.iter().map(Orig::id).collect()
}

pub struct HostOptions {
    pub usable: u64,
    pub batch_rows: Option<u64>,
    pub dedup_set_texts: Option<u64>,
    pub index: IndexPlacement,
    /// Where selections and views keep their membership and scores.
    pub tables: TablePlacement,
    /// Membership files the host registers: a name and the lines of the file.
    pub memberships: Vec<(String, String)>,
    pub profile: DisclosureProfile,
    pub inline_wait: Option<Duration>,
    pub busy: BusyChoice,
    pub delay: Duration,
    pub evidence_allowance_bytes: u64,
}

impl Default for HostOptions {
    fn default() -> Self {
        Self {
            usable: 256 * MIB,
            batch_rows: None,
            dedup_set_texts: None,
            index: IndexPlacement::MemoryWhenAdmitted,
            tables: TablePlacement::MemoryWhenAdmitted,
            memberships: Vec::new(),
            profile: DisclosureProfile::PrivateLocal,
            inline_wait: None,
            busy: BusyChoice::Wait,
            delay: Duration::ZERO,
            evidence_allowance_bytes: 64 * 1024,
        }
    }
}

/// A service over a temporary directory with the archive `private.zip` registered as
/// `private` and `public.zip` as the public fixture `public-fixture`.
pub struct Host {
    pub dir: tempfile::TempDir,
    pub service: Service,
}

impl Host {
    pub fn new(entries: &[(String, Vec<String>)], options: HostOptions) -> Self {
        let dir = tempfile::tempdir().unwrap();
        write_archive(&dir.path().join("private.zip"), entries);
        write_archive(&dir.path().join("public.zip"), entries);
        std::fs::create_dir_all(dir.path().join("allowed")).unwrap();
        write_archive(&dir.path().join("allowed").join("mine.zip"), entries);
        let service = start(dir.path(), "project-one", &options).unwrap();
        Self { dir, service }
    }

    pub fn work(&self) -> PathBuf {
        self.dir.path().join("work")
    }

    pub fn reply(&self, tool: &str, args: Value) -> Reply {
        let reply = self.service.call(tool, &args);
        assert!(
            l2d_triager::service::transport_bytes(&reply.json)
                <= l2d_triager::service::MAX_RESULT_BYTES,
            "{tool}: a result of {} bytes",
            reply.json.len()
        );
        reply
    }

    /// Calls a tool that must succeed; a job that is still running is waited for.
    pub fn ok(&self, tool: &str, args: Value) -> Value {
        let reply = self.reply(tool, args.clone());
        let value: Value = serde_json::from_str(&reply.json).unwrap();
        assert!(!reply.is_error, "{tool} {args}: {value}");
        assert_eq!(value["v"], 1, "{value}");
        if value["status"] == "running" {
            return self.wait(value["job"].as_str().unwrap());
        }
        value
    }

    /// Calls a tool that must fail, and returns the code and the message.
    pub fn err(&self, tool: &str, args: Value) -> (String, String) {
        let reply = self.reply(tool, args.clone());
        let value: Value = serde_json::from_str(&reply.json).unwrap();
        assert!(reply.is_error, "{tool} {args} was accepted: {value}");
        assert_eq!(value["status"], "error", "{value}");
        (
            value["error"]["code"].as_str().unwrap().to_string(),
            value["error"]["message"].as_str().unwrap().to_string(),
        )
    }

    pub fn job(&self, job: &str, action: &str) -> Value {
        let reply = self.reply("triager_job", json!({"v": 1, "job": job, "action": action}));
        let value: Value = serde_json::from_str(&reply.json).unwrap();
        assert!(!reply.is_error, "{value}");
        value
    }

    /// Polls a job until it has ended and returns its committed result.
    pub fn wait(&self, job: &str) -> Value {
        let status = self.wait_status(job);
        assert_eq!(status["status"], "completed", "{status}");
        status["result"].clone()
    }

    pub fn wait_status(&self, job: &str) -> Value {
        let started = Instant::now();
        loop {
            let status = self.job(job, "status");
            let state = status["status"].as_str().unwrap().to_string();
            if !matches!(state.as_str(), "waiting" | "running" | "cancel_requested") {
                return status;
            }
            assert!(started.elapsed() < Duration::from_secs(120), "{status}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    pub fn open(&self, reference: &str) -> Value {
        self.ok(
            "triager_open",
            json!({"v": 1, "source": {"ref": reference, "entries": ["*.log"]}}),
        )
    }

    pub fn select_top(&self, scope: &str, k: u64) -> Value {
        self.ok(
            "triager_select",
            json!({"v": 1, "scope": scope, "producer": {
                "kind": "score", "model": "sum", "top_k": k, "ties": "record_id_ascending"}}),
        )
    }

    pub fn select_filter(&self, scope: &str, syntax: &str, query: &str) -> Value {
        self.ok(
            "triager_select",
            json!({"v": 1, "scope": scope, "producer": {
                "kind": "filter", "syntax": syntax, "query": query}}),
        )
    }

    /// Every record of `handle`, read page by page with `limit` rows asked for.
    pub fn all_records(&self, handle: &str, sets: Value, limit: u64) -> Vec<Value> {
        let mut rows = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let mut args =
                json!({"v": 1, "handle": handle, "view": "records", "limit": limit, "sets": sets});
            if let Some(cursor) = &cursor {
                args["cursor"] = json!(cursor);
            }
            let page = self.ok("triager_inspect", args);
            let returned = page["rows"].as_array().unwrap();
            assert!(returned.len() as u64 <= limit, "{page}");
            assert_eq!(page["page"]["returned"], returned.len() as u64);
            rows.extend(returned.iter().cloned());
            match page["page"]["next_cursor"].as_str() {
                Some(next) => cursor = Some(next.to_string()),
                None => return rows,
            }
        }
    }

    pub fn activity(&self, snapshot: &str) -> Value {
        self.ok(
            "triager_inspect",
            json!({"v": 1, "handle": snapshot, "view": "overview"}),
        )["activity"]
            .clone()
    }
}

pub fn budget(options: &HostOptions) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: Some(BASELINE + options.usable),
        max_batch_rows: options.batch_rows,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(64 * GIB)).unwrap()
}

pub fn config(dir: &Path, project: &str, options: &HostOptions) -> HostConfig {
    let memberships = options
        .memberships
        .iter()
        .map(|(name, lines)| {
            let path = dir.join(format!("membership-{name}.tsv"));
            std::fs::write(&path, lines).unwrap();
            MembershipRef {
                name: name.clone(),
                path,
            }
        })
        .collect();
    HostConfig {
        project: project.into(),
        work_root: dir.join("work"),
        sources: vec![
            SourceRef {
                name: "private".into(),
                path: dir.join("private.zip"),
                public_fixture: false,
            },
            SourceRef {
                name: "public-fixture".into(),
                path: dir.join("public.zip"),
                public_fixture: true,
            },
        ],
        allowed_roots: vec![dir.join("allowed")],
        profile: options.profile,
        evidence_allowance_bytes: options.evidence_allowance_bytes,
        inline_wait: options.inline_wait,
        busy: options.busy,
        dedup_set_texts: options.dedup_set_texts,
        index: options.index,
        tables: options.tables,
        memberships,
        free_disk_bytes: None,
    }
}

pub fn start(dir: &Path, project: &str, options: &HostOptions) -> Result<Service, L2dError> {
    Service::start(
        config(dir, project, options),
        budget(options),
        Box::new(TestScorers {
            delay: options.delay,
        }),
    )
}

pub fn count(value: &Value, unit: &str, field: &str) -> u64 {
    value[unit][field]
        .as_u64()
        .unwrap_or_else(|| panic!("{unit}.{field} of {value}"))
}
