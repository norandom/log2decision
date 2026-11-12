//! The conformal calibration against what task 1.13 produced (task 1.18).
//!
//! Task 1.13 computed its p-values with the crate `conformal-prediction` 2.0.1; task
//! 1.18 replaced the crate with own code. The files below
//! `tests/fixtures/conformal-1.13/` were written by the code of task 1.13, before the
//! crate was removed, and are kept as they were written:
//!
//! - `saved/crate-2.0.1-<rev12>/`: a calibration saved by that code (`calibration.json`
//!   and `scores.bin`) from the calibration points of [`points`].
//! - `revision.txt`: its revision.
//! - `decisions.tsv`: what that code decided for a fixed grid of ratios, groups and
//!   error rates against that calibration, and what it refused. Tab-separated; 32-bit
//!   floats are written as their bit patterns in hexadecimal:
//!   - `smallest <bits>`: the smallest supported error rate.
//!   - `rates <bits>,<bits>,..`: the error rates of the decision lines.
//!   - `d <ratio> <group or -> <p_good> <p_bad> <good list> | <bad list> | <sets>`:
//!     one ratio and group, with the label set at each of the rates, in their order.
//!   - `r <ratio> <group or -> <rate> <error, Debug> | <error, Display>`: a refusal.
//!
//! The tests take every input from these files, ask the present code, and compare
//! bit for bit and byte for byte.
//!
//! # Regenerating
//!
//! The fixtures say what the crate-based code did, so they are regenerated from that
//! code only: check out the commit of task 1.13 (the last one whose `Cargo.lock` names
//! `conformal-prediction`), copy this file into it, and run
//!
//! ```text
//! $env:L2D_CONFORMAL_FIXTURE_OUT = "<an empty directory>"
//! cargo test -p l2d-decide --test conformal_fixture -- --ignored regenerate
//! ```
//!
//! then replace `tests/fixtures/conformal-1.13/` with what was written. Run on the
//! present code, the same command writes what the present code gives, which must be
//! identical to the committed files; that is a check, not a new reference. The files
//! are stored byte for byte (`tests/fixtures/.gitattributes`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use l2d_core::{Label, MemoryBudget, MemorySettings, ProjectPaths};
use l2d_decide::ScorerKind;
use l2d_decide::conformal::{
    CalibrationPoint, CalibrationSettings, ConformalCalibrator, LabelSet, MANIFEST_FILE,
    SCORES_FILE, points_frame,
};

const GIB: u64 = 1024 * 1024 * 1024;
const SCORER_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const NAME: &str = "crate-2.0.1";
const MIN_PER_GROUP: u32 = 20;
/// A group whose name needs escaping in JSON and is not ASCII.
const QUOTED: &str = "/\u{fc} \"q\"";
const REVISION_FILE: &str = "revision.txt";
const DECISIONS_FILE: &str = "decisions.tsv";
const OUT_VARIABLE: &str = "L2D_CONFORMAL_FIXTURE_OUT";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("conformal-1.13")
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
}

fn read_text(path: &Path) -> String {
    String::from_utf8(read(path)).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn fixture_revision() -> String {
    read_text(&fixture_dir().join(REVISION_FILE))
        .trim()
        .to_owned()
}

fn saved_dir_in(dir: &Path, revision: &str) -> PathBuf {
    dir.join("saved")
        .join(format!("{NAME}-{}", &revision[..12]))
}

/// A budget with a fixed free-memory reading, so the tests do not depend on the host.
fn budget() -> MemoryBudget {
    MemoryBudget::observe_with(&MemorySettings::default(), || Ok(8 * GIB))
        .expect("the budget is usable")
}

/// A project root in the system temporary directory, removed when dropped.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-decide-conformal-fixture-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("the scratch root can be created");
        Self { root }
    }

    fn paths(&self) -> ProjectPaths {
        ProjectPaths::new(&self.root)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// The calibration points of the fixture. Only integers and single IEEE operations go
/// into a ratio, so every host builds the same values.
///
/// - Logs without a group, with many equal ratios, for the pooled lists.
/// - `/a`: enough logs of both labels for a list of its own for each.
/// - `/b`: exactly enough good logs for a list of its own, one bad log too few.
/// - [`QUOTED`]: bad logs only, with a list of its own.
/// - `/c`: too few logs of either label.
/// - The ends of the range for both labels: 0, negative zero, the smallest positive
///   ratio, the largest ratio below 1, and 1.
/// - Logs labelled unknown, which are not calibration logs.
fn points() -> Vec<CalibrationPoint> {
    let mut points = Vec::new();
    let mut add = |ratio: f32, label: Label, group: Option<&str>| {
        points.push(CalibrationPoint {
            ratio,
            label,
            group: group.map(str::to_owned),
        });
    };
    for index in 0..30_u32 {
        add((index * 7 % 13) as f32 / 20.0, Label::Good, None);
        add(1.0 - (index * 5 % 11) as f32 / 20.0, Label::Bad, None);
    }
    for index in 0..24_u32 {
        add((index * 3 % 8) as f32 / 40.0, Label::Good, Some("/a"));
    }
    for index in 0..22_u32 {
        add(0.6 + (index % 6) as f32 / 16.0, Label::Bad, Some("/a"));
    }
    for index in 0..20_u32 {
        add(
            0.3 + (index * 9 % 20) as f32 / 50.0,
            Label::Good,
            Some("/b"),
        );
    }
    for index in 0..19_u32 {
        add(0.45 + (index % 10) as f32 / 20.0, Label::Bad, Some("/b"));
    }
    for index in 0..20_u32 {
        add(
            0.5 + (index * 3 % 7) as f32 / 14.0,
            Label::Bad,
            Some(QUOTED),
        );
    }
    for ratio in [0.2, 0.55, 0.55] {
        add(ratio, Label::Good, Some("/c"));
    }
    for ratio in [0.5, 0.95] {
        add(ratio, Label::Bad, Some("/c"));
    }
    for ratio in [0.0, -0.0, f32::MIN_POSITIVE, 1.0 - f32::EPSILON, 1.0] {
        add(ratio, Label::Good, None);
        add(ratio, Label::Bad, None);
    }
    for ratio in [0.1, 0.9] {
        add(ratio, Label::Unknown, None);
        add(ratio, Label::Unknown, Some("/a"));
    }
    points
}

fn fit(points: &[CalibrationPoint]) -> ConformalCalibrator {
    ConformalCalibrator::fit_with(
        &points_frame(points).expect("the points make a frame"),
        ScorerKind::Baseline,
        SCORER_REVISION,
        &CalibrationSettings {
            min_per_group: MIN_PER_GROUP,
        },
        &budget(),
    )
    .expect("the calibration fits")
}

fn bits(value: f32) -> String {
    format!("{:08x}", value.to_bits())
}

fn from_bits(text: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(text, 16).unwrap_or_else(|_| panic!("bits '{text}'")))
}

fn group_name(group: Option<&str>) -> &str {
    group.unwrap_or("-")
}

fn group_of(name: &str) -> Option<&str> {
    (name != "-").then_some(name)
}

/// The `rates` line.
fn rates_line(rates: &[f32]) -> String {
    let rates: Vec<String> = rates.iter().copied().map(bits).collect();
    format!("rates\t{}", rates.join(","))
}

/// The `d` line of a ratio and group: the p-values, the lists and the set at each rate.
fn decision_line(
    calibrator: &ConformalCalibrator,
    ratio: f32,
    group: Option<&str>,
    rates: &[f32],
) -> String {
    let decisions: Vec<_> = rates
        .iter()
        .map(|rate| {
            let decision = calibrator
                .predict(ratio, group, *rate)
                .unwrap_or_else(|error| panic!("ratio {ratio} at {rate}: {error}"));
            assert_eq!(decision.error_rate.to_bits(), rate.to_bits());
            decision
        })
        .collect();
    let first = &decisions[0];
    for decision in &decisions {
        assert_eq!(
            (decision.p_good.to_bits(), decision.p_bad.to_bits()),
            (first.p_good.to_bits(), first.p_bad.to_bits()),
            "the p-values of ratio {ratio} do not depend on the error rate"
        );
        assert_eq!(
            (&decision.good_stratum, &decision.bad_stratum),
            (&first.good_stratum, &first.bad_stratum)
        );
    }
    let sets: Vec<&str> = decisions
        .iter()
        .map(|decision| decision.set.as_str())
        .collect();
    format!(
        "d\t{}\t{}\t{}\t{}\t{} | {} | {}",
        bits(ratio),
        group_name(group),
        bits(first.p_good),
        bits(first.p_bad),
        first.good_stratum,
        first.bad_stratum,
        sets.join(",")
    )
}

/// The `r` line of a refused decision.
fn refusal_line(
    calibrator: &ConformalCalibrator,
    ratio: f32,
    group: Option<&str>,
    rate: f32,
) -> String {
    let outcome = match calibrator.predict(ratio, group, rate) {
        Ok(decision) => format!("decided {decision:?}"),
        Err(error) => format!("{error:?} | {error}"),
    };
    format!(
        "r\t{}\t{}\t{}\t{outcome}",
        bits(ratio),
        group_name(group),
        bits(rate)
    )
}

fn below(value: f32) -> f32 {
    f32::from_bits(value.to_bits() - 1)
}

fn above(value: f32) -> f32 {
    f32::from_bits(value.to_bits() + 1)
}

/// The whole of `decisions.tsv` as the given calibrator decides. Used by the generator;
/// the tests take their inputs from the committed file instead.
fn decisions_table(calibrator: &ConformalCalibrator) -> String {
    let smallest = calibrator.smallest_supported();
    let rates = [smallest, 0.05, 0.1, 0.25, 0.5, 0.9];
    let groups = [None, Some("/a"), Some("/b"), Some("/c"), Some(QUOTED)];
    let mut ratios: Vec<f32> = (0..=40_u32).map(|step| step as f32 / 40.0).collect();
    ratios.extend([
        -0.0,
        f32::MIN_POSITIVE,
        1e-30,
        1e-12,
        below(0.5),
        above(0.5),
        0.3 + 9.0 / 50.0,
        0.6 + 1.0 / 16.0,
        1.0 - f32::EPSILON,
    ]);

    let mut lines = vec![format!("smallest\t{}", bits(smallest)), rates_line(&rates)];
    for group in groups {
        for ratio in &ratios {
            lines.push(decision_line(calibrator, *ratio, group, &rates));
        }
    }
    // Error rates that are refused: too small for the lists, and not error rates.
    for rate in [
        below(smallest),
        0.04,
        0.01,
        f32::MIN_POSITIVE,
        0.0,
        -0.0,
        1.0,
        -0.5,
        f32::INFINITY,
        f32::NAN,
    ] {
        for group in [None, Some("/b")] {
            lines.push(refusal_line(calibrator, 0.5, group, rate));
        }
    }
    // Ratios that are refused.
    let unusable = [
        -0.1,
        -f32::MIN_POSITIVE,
        above(1.0),
        2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for ratio in unusable {
        for group in [None, Some("/a")] {
            lines.push(refusal_line(calibrator, ratio, group, 0.05));
        }
    }
    // Both at once: the error rate is checked first.
    lines.push(refusal_line(calibrator, 2.0, None, 0.01));
    lines.push(refusal_line(calibrator, f32::NAN, None, f32::NAN));
    lines.join("\n") + "\n"
}

/// `calibration.json` of the fixture as the present build writes it: the same bytes,
/// with the tool version of the present build where the fixture names its own.
fn expected_manifest(saved: &Path) -> Vec<u8> {
    let text = read_text(&saved.join(MANIFEST_FILE));
    let manifest: serde_json::Value = serde_json::from_str(&text).expect("the manifest is JSON");
    let written_by = manifest["tool_version"]
        .as_str()
        .expect("the manifest names a tool version");
    let named = format!("\"tool_version\": \"{written_by}\"");
    assert_eq!(
        text.matches(&named).count(),
        1,
        "the manifest names its tool version once"
    );
    text.replace(
        &named,
        &format!("\"tool_version\": \"{}\"", env!("CARGO_PKG_VERSION")),
    )
    .into_bytes()
}

/// Checks that `dir` holds the files of the fixture's saved calibration.
fn assert_same_files(dir: &Path, saved: &Path) {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, [MANIFEST_FILE, SCORES_FILE]);
    assert_eq!(dir.file_name(), saved.file_name(), "the directory name");
    assert!(
        read(&dir.join(SCORES_FILE)) == read(&saved.join(SCORES_FILE)),
        "{SCORES_FILE} differs from the one task 1.13 wrote"
    );
    assert_eq!(
        String::from_utf8(read(&dir.join(MANIFEST_FILE))).unwrap(),
        String::from_utf8(expected_manifest(saved)).unwrap(),
        "{MANIFEST_FILE} differs from the one task 1.13 wrote"
    );
}

/// Asks `calibrator` for every line of the committed `decisions.tsv` and compares.
fn assert_decides_as_recorded(calibrator: &ConformalCalibrator) {
    let table = read_text(&fixture_dir().join(DECISIONS_FILE));
    let mut rates: Vec<f32> = Vec::new();
    let (mut decided, mut refused, mut pooled, mut own, mut mixed) = (0, 0, 0, 0, 0);
    let mut sets = std::collections::BTreeSet::new();
    let mut refusals = std::collections::BTreeSet::new();
    let mut differing = Vec::new();
    for (number, line) in table.lines().enumerate() {
        let fields: Vec<&str> = line.split('\t').collect();
        let found = match fields.as_slice() {
            ["smallest", _] => format!("smallest\t{}", bits(calibrator.smallest_supported())),
            ["rates", listed] => {
                rates = listed.split(',').map(from_bits).collect();
                rates_line(&rates)
            }
            ["d", ratio, group, _, _, rest] => {
                decided += 1;
                let parts: Vec<&str> = rest.split(" | ").collect();
                match (parts[0] == "pooled", parts[1] == "pooled") {
                    (true, true) => pooled += 1,
                    (false, false) => own += 1,
                    _ => mixed += 1,
                }
                sets.extend(parts[2].split(',').map(str::to_owned));
                decision_line(calibrator, from_bits(ratio), group_of(group), &rates)
            }
            ["r", ratio, group, rate, outcome] => {
                refused += 1;
                refusals.insert(outcome.split([' ', '{']).next().unwrap().to_owned());
                refusal_line(
                    calibrator,
                    from_bits(ratio),
                    group_of(group),
                    from_bits(rate),
                )
            }
            _ => panic!("line {} of {DECISIONS_FILE} is not understood", number + 1),
        };
        if found != line {
            differing.push(format!(
                "line {}:\n  task 1.13: {line}\n  now:       {found}",
                number + 1
            ));
        }
    }
    assert!(
        differing.is_empty(),
        "{} of {} lines differ from what task 1.13 decided; the first:\n{}",
        differing.len(),
        table.lines().count(),
        differing[..differing.len().min(5)].join("\n")
    );

    // The table is worth comparing with: it is not empty and exercises every case.
    assert_eq!(rates.len(), 6);
    assert!(decided >= 200 && refused >= 30, "{decided} and {refused}");
    assert!(pooled > 0 && own > 0 && mixed > 0, "{pooled} {own} {mixed}");
    let all: Vec<String> = [
        LabelSet::Bad,
        LabelSet::Both,
        LabelSet::Good,
        LabelSet::Neither,
    ]
    .iter()
    .map(|set| set.as_str().to_owned())
    .collect();
    assert_eq!(sets.into_iter().collect::<Vec<_>>(), all);
    assert_eq!(
        refusals.into_iter().collect::<Vec<_>>(),
        ["InvalidErrorRate", "InvalidRatio", "TooFewLogs"]
    );
}

#[test]
fn the_calibration_saved_by_task_1_13_loads_with_its_revision_and_decides_the_same() {
    let revision = fixture_revision();
    assert_eq!(revision.len(), 40);
    let saved = saved_dir_in(&fixture_dir(), &revision);
    let loaded = ConformalCalibrator::load(&saved).expect("the saved calibration loads");
    assert_eq!(loaded.revision(), revision);
    assert_eq!(loaded.scorer_kind(), ScorerKind::Baseline);
    assert_eq!(loaded.scorer_revision(), SCORER_REVISION);
    assert_eq!(loaded.settings().min_per_group, MIN_PER_GROUP);
    assert_eq!(loaded.strata().len(), 6);
    assert_decides_as_recorded(&loaded);
}

#[test]
fn saving_the_loaded_calibration_again_writes_the_same_bytes() {
    let revision = fixture_revision();
    let saved = saved_dir_in(&fixture_dir(), &revision);
    let loaded = ConformalCalibrator::load(&saved).expect("the saved calibration loads");
    let scratch = Scratch::new("save-again");
    let dir = loaded.save(&scratch.paths(), NAME).expect("it saves");
    assert!(dir.starts_with(std::env::temp_dir()));
    assert_same_files(&dir, &saved);
}

#[test]
fn fitting_the_same_points_gives_the_same_revision_files_and_decisions() {
    let revision = fixture_revision();
    let saved = saved_dir_in(&fixture_dir(), &revision);
    let loaded = ConformalCalibrator::load(&saved).expect("the saved calibration loads");

    let mut points = points();
    let forwards = fit(&points);
    points.reverse();
    let backwards = fit(&points);
    for (order, fitted) in [("forwards", forwards), ("backwards", backwards)] {
        assert_eq!(fitted.revision(), revision, "{order}");
        assert_eq!(fitted, loaded, "{order}");
        assert_eq!(fitted.strata(), loaded.strata(), "{order}");
        assert_eq!(
            fitted.smallest_supported().to_bits(),
            loaded.smallest_supported().to_bits()
        );
        assert_decides_as_recorded(&fitted);
        let scratch = Scratch::new(order);
        let dir = fitted.save(&scratch.paths(), NAME).expect("it saves");
        assert_same_files(&dir, &saved);
    }
}

/// Writes the fixture files as the code this is compiled with gives them, into the
/// directory named by `L2D_CONFORMAL_FIXTURE_OUT`. See "Regenerating" above.
#[test]
#[ignore = "writes fixture files; run on request with L2D_CONFORMAL_FIXTURE_OUT set"]
fn regenerate_the_fixture_files() {
    let out = std::env::var_os(OUT_VARIABLE)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {OUT_VARIABLE} to the directory to write into"));
    let calibrator = fit(&points());
    let revision = calibrator.revision().to_owned();

    let scratch = Scratch::new("regenerate");
    let dir = calibrator.save(&scratch.paths(), NAME).expect("it saves");
    let target = saved_dir_in(&out, &revision);
    std::fs::create_dir_all(&target).expect("the output directory can be created");
    for file in [MANIFEST_FILE, SCORES_FILE] {
        std::fs::copy(dir.join(file), target.join(file)).expect("the file can be copied");
    }
    std::fs::write(out.join(REVISION_FILE), format!("{revision}\n")).unwrap();
    std::fs::write(out.join(DECISIONS_FILE), decisions_table(&calibrator)).unwrap();
    println!(
        "L2D_CONFORMAL_FIXTURE revision={revision} written to {}",
        out.display()
    );
}

/// The time of a decision against a large calibration, for the notes of task 1.18.
#[test]
#[ignore = "a measurement; run in release mode on request"]
fn decision_speed_measured() {
    for logs in [5_000_u32, 200_000] {
        let points: Vec<CalibrationPoint> = (0..logs)
            .map(|index| CalibrationPoint {
                ratio: (u64::from(index) * 7_919 % 100_003) as f32 / 100_003.0,
                label: if index % 2 == 0 {
                    Label::Good
                } else {
                    Label::Bad
                },
                group: None,
            })
            .collect();
        let calibrator = fit(&points);
        let decisions = 20_000_u32;
        let started = Instant::now();
        let mut good = 0_u32;
        for index in 0..decisions {
            let ratio = index as f32 / decisions as f32;
            let decision = calibrator.predict(ratio, None, 0.05).expect("supported");
            good += u32::from(decision.set == LabelSet::Good);
        }
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "L2D_CONFORMAL_SPEED calibration_logs={logs} decisions={decisions} good={good} \
             seconds={seconds:.4} microseconds_per_decision={:.3}",
            1e6 * seconds / f64::from(decisions)
        );
    }
}
