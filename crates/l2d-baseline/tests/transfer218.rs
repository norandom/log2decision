//! Opt-in scientific measurement of the existing product MarkSet scorer.
use l2d_baseline::{
    markscore::{MarkFitSettings, fit_from_marks},
    redact::{RedactionInput, RedactionRules, RedactionSettings, propose_redactions},
};
use l2d_core::{
    Csic2010Adapter, InputAdapter, Label, MarkQuery, MarkSet, MemoryBudget, MemorySettings,
    ProjectPaths,
};
use std::{
    io::{BufRead, Write},
    path::PathBuf,
};

#[test]
#[ignore = "task 2.18 exploratory measurement; explicit fixture required"]
fn export_markset_comparison() {
    let started = std::time::Instant::now();
    let guard = l2d_core::resource::ResourceGuard::enter(
        "2.18 mark-set comparison",
        l2d_core::resource::BusyChoice::Refuse,
    )
    .unwrap();
    let root = PathBuf::from(std::env::var_os("L2D_TRANSFER218").expect("L2D_TRANSFER218"));
    let fold = std::env::var("L2D_TRANSFER_FOLD").unwrap();
    let out = root.join(format!("f{fold}"));
    let plan: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("plan.json")).unwrap()).unwrap();
    let mut budget = MemoryBudget::observe(&MemorySettings::default()).unwrap();
    let (logs, summary, _) = Csic2010Adapter::new()
        .read_with(&out.join("train.csv"), &mut budget)
        .unwrap();
    // Input frame plus bounded proposals/paths/serialization and one report record.
    let held = 64 * 1024 * 1024 + 2 * logs.frame().estimated_size() as u64;
    budget
        .require_whole(held, "2.18 bounded input and report workspace")
        .unwrap();
    budget.hold(held);
    let mut marks = MarkSet::mark(
        "seed",
        Label::Bad,
        MarkQuery::Regex {
            pattern: plan["seed_query"].as_str().unwrap().into(),
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap();
    let input = || RedactionInput::File {
        logs: &logs,
        summary: &summary,
    };
    let proposals =
        propose_redactions(&marks, input(), &RedactionSettings::default(), &mut budget).unwrap();
    let paths: std::collections::HashSet<_> = logs.records().filter_map(|r| r.group).collect();
    let rules = RedactionRules::confirmed(
        proposals
            .proposals
            .iter()
            .filter(|p| p.fragment == "localhost:8080" || paths.contains(&p.fragment)),
        &budget,
    )
    .unwrap();
    let settings = MarkFitSettings::default();
    let raw = fit_from_marks(&marks, None, input(), &settings, &mut budget).unwrap();
    let project = ProjectPaths::new(out.join("scorers"));
    let raw_path = raw.scorer.save(&project, "raw").unwrap();
    let raw_held = raw.held_bytes() + raw.scorer.held_bytes();
    budget
        .require_whole(raw_held, "2.18 retained raw fit")
        .unwrap();
    budget.hold(held + raw_held);
    marks.set_redactions(rules.rules().to_vec()).unwrap();
    let red = fit_from_marks(&marks, None, input(), &settings, &mut budget).unwrap();
    let red_path = red.scorer.save(&project, "redacted").unwrap();
    let red_held = red.held_bytes() + red.scorer.held_bytes();
    budget
        .require_whole(red_held, "2.18 retained redacted fit")
        .unwrap();
    budget.hold(held + raw_held + red_held);
    // Freeze every fitted artifact and rule before accessing report texts or labels.
    std::fs::write(out.join("frozen.json"),serde_json::to_vec_pretty(&serde_json::json!({"raw":raw_path,"redacted":red_path,"rules":rules.rules(),"rule_hash":rules.content_sha256(),"counts":raw.counts,"proposals":proposals.proposals.iter().map(|p| serde_json::json!({"fragment":p.fragment,"marked":p.marked,"unmarked":p.unmarked})).collect::<Vec<_>>(),"background":"assumed unmarked; labels ignored at fitting"})).unwrap()).unwrap();
    let records = std::io::BufReader::new(std::fs::File::open(root.join("records.jsonl")).unwrap());
    let mut scores =
        std::io::BufWriter::new(std::fs::File::create(out.join("baseline.jsonl")).unwrap());
    let mut texts =
        std::io::BufWriter::new(std::fs::File::create(out.join("paired.jsonl")).unwrap());
    for row in records.lines() {
        let row: serde_json::Value = serde_json::from_str(&row.unwrap()).unwrap();
        let text = row["text"].as_str().unwrap();
        budget
            .require_whole(
                raw.scorer
                    .scoring_bytes(text.len() as u64)
                    .max(red.scorer.scoring_bytes(text.len() as u64)),
                "2.18 scoring scratch",
            )
            .unwrap();
        let redacted = rules.apply(text);
        writeln!(scores,"{}",serde_json::json!({"i":row["i"],"raw":raw.scorer.ratio(text),"redacted":red.scorer.ratio(text)})).unwrap();
        writeln!(texts,"{}",serde_json::json!({"i":row["i"],"raw":text,"redacted":redacted,"original_bytes":text.len()})).unwrap();
    }
    scores.flush().unwrap();
    texts.flush().unwrap();
    println!(
        "fold {fold}: raw={} redacted={} rules={} marked={}",
        raw.scorer.revision(),
        red.scorer.revision(),
        rules.rules().len(),
        marks.count()
    );
    println!(
        "2.18 resources: {}",
        serde_json::json!({"seconds":started.elapsed().as_secs_f64(),"memory":budget.usage().report(),"waited":guard.waited,"spill_files":0,"native_peak":"not measured by this harness"})
    );
    guard
        .track("2.18 baseline resources", budget.usage())
        .unwrap()
        .finish()
        .unwrap();
    guard.finish().unwrap();
}
