//! End-to-end analyst workflow, including immutable receipts and refusal paths.
use serde_json::Value;
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let p = std::env::temp_dir().join(format!(
            "l2d-discover-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn path(&self, p: &str) -> String {
        self.0.join(p).to_string_lossy().into_owned()
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_l2d"))
            .arg("--project")
            .arg(&self.0)
            .args(args)
            .output()
            .unwrap()
    }
    fn json(&self, args: &[&str]) -> Value {
        let mut all = vec!["--json"];
        all.extend_from_slice(args);
        let o = self.run(&all);
        assert!(
            o.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
        serde_json::from_slice(&o.stdout).unwrap()
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn exploration_is_explicit_source_bound_unlabelled_and_replayable() {
    let s = Scratch::new();
    let file = s.path("explore.log");
    fs::write(&file,"login failed user=alice\nlogin failed user=bob\nlogin denied user=carol\nlogin succeeded user=dave\n").unwrap();
    s.json(&[
        "discover",
        "search",
        "--input",
        &file,
        "--phrase",
        "failed",
        "--out",
        &s.path("seed"),
    ]);
    s.json(&[
        "discover",
        "search",
        "--input",
        &file,
        "--glob",
        "*",
        "--out",
        &s.path("scope"),
    ]);
    let seed = s.path("seed/refinement.json");
    let scope = s.path("scope/refinement.json");
    let before = fs::read(&seed).unwrap();
    let args = [
        "discover",
        "explore",
        "--artifact",
        &seed,
        "--exemplar-row",
        "0",
        "--method",
        "character-trigram",
        "--radius",
        "1",
        "--top-k",
        "5",
        "--expand-artifact",
        &scope,
        "--out",
        &s.path("report"),
    ];
    let output = s.json(&args);
    let r = &output["exploration"];
    assert_eq!(r["method"], "direct-exemplar-character-trigram-jaccard");
    assert_eq!(r["neighbours"].as_array().unwrap().len(), 3);
    assert_eq!(r["support"]["distinct_texts"], 3);
    assert_eq!(
        r["neighbours"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|h| h["outside_saved_search"] == true)
            .count(),
        2
    );
    assert!(
        r["neighbours"]
            .as_array()
            .unwrap()
            .iter()
            .all(|h| h["row"]["review"].is_null())
    );
    assert_eq!(fs::read(&seed).unwrap(), before);
    let replay = s.json(&[
        "discover",
        "explore",
        "--artifact",
        &seed,
        "--exemplar-row",
        "0",
        "--method",
        "character-trigram",
        "--radius",
        "1",
        "--top-k",
        "5",
        "--expand-artifact",
        &scope,
        "--expect-profile",
        r["profile"].as_str().unwrap(),
        "--out",
        &s.path("replay"),
    ]);
    assert_eq!(r["neighbours"], replay["exploration"]["neighbours"]);
    assert!(!s.run(&args).status.success());
    let mismatch = s.run(&[
        "discover",
        "explore",
        "--artifact",
        &seed,
        "--exemplar-row",
        "0",
        "--method",
        "character-trigram",
        "--expect-profile",
        "wrong",
        "--out",
        &s.path("mismatch"),
    ]);
    assert!(!mismatch.status.success());
    assert!(String::from_utf8_lossy(&mismatch.stderr).contains("incompatible"));
    assert!(!s.0.join("mismatch").exists());
    let npu = s.run(&[
        "discover",
        "explore",
        "--artifact",
        &seed,
        "--exemplar-row",
        "0",
        "--method",
        "npu",
        "--out",
        &s.path("no-fallback"),
    ]);
    assert!(!npu.status.success());
    assert!(!s.0.join("no-fallback").exists());
    fs::write(&file, "changed source\n").unwrap();
    assert!(
        !s.run(&[
            "discover",
            "explore",
            "--artifact",
            &seed,
            "--exemplar-row",
            "0",
            "--method",
            "character-trigram",
            "--out",
            &s.path("stale")
        ])
        .status
        .success()
    );
}
#[test]
fn broad_search_refinement_preview_review_and_export_replay() {
    let s = Scratch::new();
    let file = s.path("logs.txt");
    fs::write(&file,"login failed user=alice\nlogin failed user=bob\nlogin failed user=alice\nlogin succeeded user=carol\nrare [x].* exception\n").unwrap();
    let found = s.json(&[
        "--batch-rows",
        "2",
        "discover",
        "search",
        "--input",
        &file,
        "--glob",
        "*",
        "--out",
        &s.path("found"),
    ]);
    assert_eq!(found["completeness"]["matched"], 4);
    assert_eq!(found["patterns"][0]["counts"]["occurrences"], 3);
    assert!(found["memory"]["free_bytes"].as_u64().unwrap() > 0);
    assert_eq!(found["memory"]["operations"][0]["batches"]["batch_rows"], 2);
    assert_eq!(
        found["memory"]["operations"][0]["batches"]["max_rows_used"],
        2
    );
    let artifact = s.path("found/refinement.json");
    let selection = s.path("found/selection.json");
    let human = s.run(&["discover", "patterns", "--artifact", &artifact]);
    assert!(
        human.status.success(),
        "{}",
        String::from_utf8_lossy(&human.stderr)
    );
    assert!(String::from_utf8_lossy(&human.stdout).contains("login failed"));
    assert!(!String::from_utf8_lossy(&human.stdout).starts_with('{'));
    let replay = s.json(&["discover", "patterns", "--artifact", &artifact]);
    assert_eq!(replay["revision"], found["refinement_revision"]);
    let details = s.json(&[
        "discover",
        "details",
        "--artifact",
        &artifact,
        "--pattern",
        "0",
    ]);
    assert_eq!(details["members"].as_array().unwrap().len(), 2);
    let split = s.json(&[
        "discover",
        "split",
        "--artifact",
        &artifact,
        "--pattern",
        "0",
        "--regex",
        "alice",
        "--out",
        &s.path("split.json"),
    ]);
    let next = split["refinement"]["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["id"].as_u64().unwrap())
        .max()
        .unwrap();
    let merged = s.json(&[
        "discover",
        "merge",
        "--artifact",
        &s.path("split.json"),
        "--patterns",
        &format!("0,{next}"),
        "--out",
        &s.path("merged.json"),
    ]);
    assert_eq!(merged["refinement"]["rows"].as_array().unwrap().len(), 4);
    let p = s.json(&[
        "discover",
        "preview",
        "--artifact",
        &artifact,
        "--pattern",
        "0",
        "--out",
        &s.path("preview.json"),
    ]);
    assert_eq!(p["member_misses"].as_array().unwrap().len(), 0);
    assert_eq!(p["background_matches"].as_array().unwrap().len(), 0);
    let broad = s.json(&[
        "discover",
        "preview",
        "--artifact",
        &artifact,
        "--pattern",
        "0",
        "--regex",
        "login",
    ]);
    assert_eq!(broad["background_matches"].as_array().unwrap().len(), 1);
    s.json(&[
        "discover",
        "save-query",
        "--artifact",
        &artifact,
        "--preview",
        &s.path("preview.json"),
        "--out",
        &s.path("query.json"),
    ]);
    let promoted = s.json(&[
        "discover",
        "priority",
        "--artifact",
        &artifact,
        "--pattern",
        "0",
        "--priority",
        "promoted",
        "--out",
        &s.path("promoted.json"),
    ]);
    assert_eq!(promoted["labels"], "unchanged");
    let stale = s.run(&[
        "discover",
        "save-query",
        "--artifact",
        &s.path("promoted.json"),
        "--preview",
        &s.path("preview.json"),
        "--out",
        &s.path("stale.json"),
    ]);
    assert!(!stale.status.success());
    assert!(!s.0.join("stale.json").exists());
    let plan = s.json(&[
        "discover",
        "review",
        "--artifact",
        &artifact,
        "--selection",
        &selection,
        "--pattern",
        "0",
        "--state",
        "bad",
        "--analyst",
        "fixture",
        "--action",
        "inspect failed logins",
    ]);
    assert_eq!(plan["affected_members"].as_array().unwrap().len(), 2);
    let altered = s.run(&[
        "discover",
        "review",
        "--artifact",
        &artifact,
        "--selection",
        &selection,
        "--pattern",
        "0",
        "--state",
        "good",
        "--analyst",
        "fixture",
        "--action",
        "inspect failed logins",
        "--apply",
        "--expect-revision",
        plan["review_revision"].as_str().unwrap(),
        "--out",
        &s.path("altered-review"),
    ]);
    assert!(!altered.status.success());
    assert!(!s.0.join("altered-review").exists());
    s.json(&[
        "discover",
        "review",
        "--artifact",
        &artifact,
        "--selection",
        &selection,
        "--pattern",
        "0",
        "--state",
        "bad",
        "--analyst",
        "fixture",
        "--action",
        "inspect failed logins",
        "--apply",
        "--expect-revision",
        plan["review_revision"].as_str().unwrap(),
        "--out",
        &s.path("review1"),
    ]);
    let r1 = s.path("review1/refinement.json");
    let sel1 = s.path("review1/selection.json");
    let plan = s.json(&[
        "discover",
        "review",
        "--artifact",
        &r1,
        "--selection",
        &sel1,
        "--pattern",
        "1",
        "--state",
        "good",
        "--analyst",
        "fixture",
        "--action",
        "inspect successes",
    ]);
    s.json(&[
        "discover",
        "review",
        "--artifact",
        &r1,
        "--selection",
        &sel1,
        "--pattern",
        "1",
        "--state",
        "good",
        "--analyst",
        "fixture",
        "--action",
        "inspect successes",
        "--apply",
        "--expect-revision",
        plan["review_revision"].as_str().unwrap(),
        "--out",
        &s.path("review2"),
    ]);
    let r2 = s.path("review2/refinement.json");
    let sel2 = s.path("review2/selection.json");
    let exported = s.json(&[
        "discover",
        "export",
        "--artifact",
        &r2,
        "--selection",
        &sel2,
        "--label",
        "bad",
        "--name",
        "failed",
        "--out",
        &s.path("export"),
    ]);
    assert_eq!(exported["selected"], 2);
    assert_eq!(exported["excluded_unresolved_or_conflict"], 1);
    assert!(s.0.join("export/selection.json").exists());
    assert!(s.0.join("export/marks/failed.json").exists());
    let saved: Value =
        serde_json::from_slice(&fs::read(s.0.join("export/marks/failed.json")).unwrap()).unwrap();
    assert_eq!(saved["format"], 3);
    let refused = s.run(&[
        "discover",
        "review",
        "--artifact",
        &r2,
        "--selection",
        &sel2,
        "--pattern",
        "1",
        "--state",
        "conflict",
        "--analyst",
        "fixture",
        "--action",
        "stale",
        "--apply",
        "--expect-revision",
        "wrong",
        "--out",
        &s.path("refused"),
    ]);
    assert!(!refused.status.success());
    assert!(!s.0.join("refused").exists());
    let bad = s.run(&[
        "discover",
        "preview",
        "--artifact",
        &artifact,
        "--pattern",
        "0",
        "--regex",
        "[",
    ]);
    assert_eq!(bad.status.code(), Some(2));
    let tiny = s.run(&[
        "--memory-limit",
        "33MB",
        "discover",
        "search",
        "--input",
        &file,
        "--glob",
        "*",
        "--out",
        &s.path("tiny"),
    ]);
    assert!(!tiny.status.success());
    assert!(!s.0.join("tiny").exists());
    fs::write(&file, "changed source\n").unwrap();
    let changed = s.run(&[
        "discover",
        "export",
        "--artifact",
        &r2,
        "--selection",
        &sel2,
        "--label",
        "bad",
        "--name",
        "changed",
        "--out",
        &s.path("changed"),
    ]);
    assert!(!changed.status.success());
    assert!(!s.0.join("changed").exists());
}

#[test]
fn archive_scope_and_store_count_changes_are_refused() {
    use l2d_core::{
        EncoderIdentity, InputAdapter, LinesAdapter, MemoryBudget, MemorySettings, ProjectPaths,
    };
    use l2d_store::{LogFlags, Store, VectorOrigin, rows_frame};
    use std::io::Write;
    let s = Scratch::new();
    let archive = s.path("logs.zip");
    let mut zip = zip::ZipWriter::new(fs::File::create(&archive).unwrap());
    for (name, text) in [
        ("a.log", "login failed user=a\nlogin failed user=b\n"),
        ("b.log", "outside ignored\n"),
    ] {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(text.as_bytes()).unwrap();
    }
    zip.finish().unwrap();
    let found = s.json(&[
        "--batch-rows",
        "1",
        "discover",
        "search",
        "--input",
        &archive,
        "--entry",
        "a.log",
        "--glob",
        "*",
        "--out",
        &s.path("archive"),
    ]);
    assert_eq!(found["completeness"]["matched"], 2);
    assert_eq!(found["memory"]["operations"].as_array().unwrap().len(), 2);
    for operation in found["memory"]["operations"].as_array().unwrap() {
        assert_eq!(operation["batches"]["batch_rows"], 1);
        assert_eq!(operation["batches"]["min_rows_used"], 1);
        assert_eq!(operation["batches"]["max_rows_used"], 1);
    }
    let repeated = s.json(&[
        "discover",
        "search",
        "--input",
        &archive,
        "--entry",
        "a.log",
        "--glob",
        "*",
        "--out",
        &s.path("archive-repeat"),
    ]);
    assert_eq!(
        repeated["refinement_revision"],
        found["refinement_revision"]
    );
    assert_eq!(
        found["source"]["archive_entries"],
        serde_json::json!(["a.log"])
    );
    let archive_plan = s.json(&[
        "discover",
        "review",
        "--artifact",
        &s.path("archive/refinement.json"),
        "--selection",
        &s.path("archive/selection.json"),
        "--pattern",
        "0",
        "--state",
        "conflict",
        "--analyst",
        "fixture",
        "--action",
        "archive review",
    ]);
    let reviewed = s.json(&[
        "discover",
        "review",
        "--artifact",
        &s.path("archive/refinement.json"),
        "--selection",
        &s.path("archive/selection.json"),
        "--pattern",
        "0",
        "--state",
        "conflict",
        "--analyst",
        "fixture",
        "--action",
        "archive review",
        "--apply",
        "--expect-revision",
        archive_plan["review_revision"].as_str().unwrap(),
        "--out",
        &s.path("archive-reviewed"),
    ]);
    assert!(reviewed["revision"].is_string());
    let input = s.path("store.log");
    fs::write(&input, "login failed user=a\nlogin failed user=b\n").unwrap();
    let mut b =
        MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * 1024 * 1024 * 1024))
            .unwrap();
    let (logs, _, _) = LinesAdapter
        .read_with(std::path::Path::new(&input), &mut b)
        .unwrap();
    let identity = EncoderIdentity {
        name: "fixture".into(),
        revision: "a".repeat(64),
        dimension: 2,
    };
    let origin = VectorOrigin::without_release(&identity);
    let mut store = Store::create(&ProjectPaths::new(&s.0), "fixture", origin).unwrap();
    let rows = rows_frame(
        &logs,
        "fixture",
        &[1.0, 0.0, 1.0, 0.0],
        2,
        &[LogFlags::default(); 2],
    )
    .unwrap();
    store.add_with(&rows, origin, &mut b).unwrap();
    let found = s.json(&[
        "--batch-rows",
        "1",
        "discover",
        "search",
        "--store",
        "fixture",
        "--phrase",
        "login",
        "--out",
        &s.path("store-found"),
    ]);
    assert_eq!(found["completeness"]["matched"], 2);
    for scan in found["store_batches"].as_array().unwrap() {
        assert_eq!(scan["batches"], 2);
        assert_eq!(scan["min_delivered_rows"], 1);
        assert_eq!(scan["max_delivered_rows"], 1);
    }
    let fts = s.json(&[
        "discover",
        "search",
        "--store",
        "fixture",
        "--fts",
        "login",
        "--out",
        &s.path("fts"),
    ]);
    assert_eq!(fts["store_batches"][0]["batches"], 0);
    assert!(
        fts["store_batches"][0]["sizing"]
            .as_str()
            .unwrap()
            .contains("no selection frame batches")
    );
    store.add_with(&rows, origin, &mut b).unwrap();
    let refused = s.run(&[
        "discover",
        "review",
        "--artifact",
        &s.path("store-found/refinement.json"),
        "--selection",
        &s.path("store-found/selection.json"),
        "--pattern",
        "0",
        "--state",
        "bad",
        "--analyst",
        "fixture",
        "--action",
        "stale counts",
        "--apply",
        "--expect-revision",
        found["refinement_revision"].as_str().unwrap(),
        "--out",
        &s.path("store-refused"),
    ]);
    assert!(!refused.status.success());
    assert!(!s.0.join("store-refused").exists());
}
