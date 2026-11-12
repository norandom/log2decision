//! Contract tests of the conformal calibration layer (task 1.13, requirement 11).
//!
//! Everything here runs on synthetic ratios or on the mini fixture, writes only below
//! the system temporary directory and needs no dataset. The evaluation on HTTP CSIC
//! 2010 is in `conformal_csic.rs` and runs on request.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use l2d_baseline::{BaselineScorer, BaselineSettings};
use l2d_core::input::content_sha256;
use l2d_core::polars::prelude::{Column, DataFrame, DataType};
use l2d_core::{
    InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat, LogsFrame,
    MemoryBudget, MemorySettings, ProjectPaths,
};
use l2d_decide::ScorerKind;
use l2d_decide::conformal::{
    Calibrated, CalibrationError, CalibrationPoint, CalibrationSettings, ConformalCalibrator,
    DEFAULT_ERROR_RATE, DEFAULT_MIN_PER_GROUP, EXCHANGEABILITY_NOTE, LabelSet, MANIFEST_FILE,
    SCORES_FILE, Stratum, estimated_bytes, points_frame,
};

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

/// A budget with a fixed free-memory reading, so the tests do not depend on the host.
fn budget_with_limit(limit: Option<u64>) -> MemoryBudget {
    let settings = MemorySettings {
        memory_limit_bytes: limit,
        ..MemorySettings::default()
    };
    MemoryBudget::observe_with(&settings, || Ok(8 * GIB)).expect("the budget is usable")
}

fn budget() -> MemoryBudget {
    budget_with_limit(None)
}

type Row<'a> = (f32, Label, Option<&'a str>);

fn frame(rows: &[Row<'_>]) -> DataFrame {
    let points: Vec<CalibrationPoint> = rows
        .iter()
        .map(|(ratio, label, group)| CalibrationPoint {
            ratio: *ratio,
            label: *label,
            group: group.map(str::to_owned),
        })
        .collect();
    points_frame(&points).expect("the points make a frame")
}

fn settings(min_per_group: u32) -> CalibrationSettings {
    CalibrationSettings { min_per_group }
}

fn fit_rows(rows: &[Row<'_>], min_per_group: u32) -> Result<ConformalCalibrator, CalibrationError> {
    ConformalCalibrator::fit_with(
        &frame(rows),
        ScorerKind::Baseline,
        REVISION,
        &settings(min_per_group),
        &budget(),
    )
}

fn fitted(rows: &[Row<'_>], min_per_group: u32) -> ConformalCalibrator {
    fit_rows(rows, min_per_group).expect("the calibration fits")
}

/// `good` and `bad` ratios without groups.
fn ungrouped<'a>(good: &[f32], bad: &[f32]) -> Vec<Row<'a>> {
    good.iter()
        .map(|ratio| (*ratio, Label::Good, None))
        .chain(bad.iter().map(|ratio| (*ratio, Label::Bad, None)))
        .collect()
}

/// The nonconformity of the design: `ratio` for good, `1 - ratio` for bad, in double
/// precision.
fn nonconformity(ratio: f32, label: Label) -> f64 {
    match label {
        Label::Good => f64::from(ratio),
        Label::Bad => 1.0 - f64::from(ratio),
        Label::Unknown => unreachable!("unknown logs have no nonconformity"),
    }
}

/// The p-value of the design computed directly: `(1 + calibration logs of the label
/// whose nonconformity is at least the new log's) / (n + 1)`.
fn direct_p(calibration: &[f32], label: Label, ratio: f32) -> f32 {
    let new = nonconformity(ratio, label);
    let at_least = calibration
        .iter()
        .filter(|other| nonconformity(**other, label) >= new)
        .count();
    ((at_least as f64 + 1.0) / (calibration.len() as f64 + 1.0)) as f32
}

fn direct_set(p_good: f32, p_bad: f32, error_rate: f32) -> LabelSet {
    match (p_good > error_rate, p_bad > error_rate) {
        (true, false) => LabelSet::Good,
        (false, true) => LabelSet::Bad,
        (true, true) => LabelSet::Both,
        (false, false) => LabelSet::Neither,
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

    /// Uniform in `0.0..1.0`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Close to a standard normal draw: the sum of twelve uniform draws, less six.
    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
    }

    /// A ratio around `centre`, kept within `0.0..=1.0`.
    fn ratio(&mut self, centre: f64, spread: f64) -> f32 {
        (centre + spread * self.normal()).clamp(0.0, 1.0) as f32
    }

    fn ratios(&mut self, count: usize, centre: f64, spread: f64) -> Vec<f32> {
        (0..count).map(|_| self.ratio(centre, spread)).collect()
    }
}

/// The share of `ratios`, all of the true label `label`, whose label set leaves that
/// label out: the achieved error rate.
fn achieved_error(
    calibrator: &ConformalCalibrator,
    ratios: &[f32],
    label: Label,
    group: Option<&str>,
    error_rate: f32,
) -> f64 {
    let missed = ratios
        .iter()
        .filter(|ratio| {
            let calibrated = calibrator
                .predict(**ratio, group, error_rate)
                .expect("the error rate is supported");
            !calibrated.set.contains(label)
        })
        .count();
    missed as f64 / ratios.len() as f64
}

/// A project root in the system temporary directory, removed when dropped.
struct Scratch {
    root: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "l2d-decide-conformal-{}-{}-{name}",
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

/// The little-endian doubles in `bytes`.
fn doubles(bytes: &[u8]) -> Vec<f64> {
    let (chunks, rest) = bytes.as_chunks::<8>();
    assert!(
        rest.is_empty(),
        "{} bytes are not whole doubles",
        bytes.len()
    );
    chunks
        .iter()
        .map(|bits| f64::from_bits(u64::from_le_bytes(*bits)))
        .collect()
}

fn files_of(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Overlapping ratios: 300 good around 0.35 and 300 bad around 0.65.
fn overlapping() -> (Vec<f32>, Vec<f32>) {
    let mut draws = Draws(7);
    (draws.ratios(300, 0.35, 0.15), draws.ratios(300, 0.65, 0.15))
}

/// Ratios from 0 to 1 in steps of 1/200, plus the values around the ends.
fn grid() -> Vec<f32> {
    let mut grid: Vec<f32> = (0..=200).map(|step| step as f32 / 200.0).collect();
    grid.extend([f32::MIN_POSITIVE, 1e-30, 1e-12, 1.0 - f32::EPSILON]);
    grid
}

fn decisions(calibrator: &ConformalCalibrator, group: Option<&str>, e: f32) -> Vec<Calibrated> {
    grid()
        .into_iter()
        .map(|ratio| calibrator.predict(ratio, group, e).expect("supported"))
        .collect()
}

// ---------------------------------------------------------------------------------------
// Defaults and names
// ---------------------------------------------------------------------------------------

#[test]
fn the_defaults_are_five_percent_and_one_hundred_logs_per_group() {
    assert_eq!(DEFAULT_ERROR_RATE, 0.05);
    assert_eq!(DEFAULT_MIN_PER_GROUP, 100);
    assert_eq!(CalibrationSettings::default().min_per_group, 100);
}

#[test]
fn label_sets_strata_and_scorer_kinds_have_fixed_names() {
    assert_eq!(LabelSet::Good.as_str(), "good");
    assert_eq!(LabelSet::Bad.as_str(), "bad");
    assert_eq!(LabelSet::Both.as_str(), "both");
    assert_eq!(LabelSet::Neither.as_str(), "neither");
    assert!(LabelSet::Good.contains(Label::Good) && !LabelSet::Good.contains(Label::Bad));
    assert!(LabelSet::Bad.contains(Label::Bad) && !LabelSet::Bad.contains(Label::Good));
    assert!(LabelSet::Both.contains(Label::Good) && LabelSet::Both.contains(Label::Bad));
    assert!(!LabelSet::Neither.contains(Label::Good) && !LabelSet::Neither.contains(Label::Bad));
    assert!(!LabelSet::Both.contains(Label::Unknown));

    assert_eq!(ScorerKind::Baseline.as_str(), "baseline");
    assert_eq!(ScorerKind::Neighbour.as_str(), "neighbour");
    assert_eq!(
        ScorerKind::from_name("baseline"),
        Some(ScorerKind::Baseline)
    );
    assert_eq!(
        ScorerKind::from_name("neighbour"),
        Some(ScorerKind::Neighbour)
    );
    assert_eq!(ScorerKind::from_name("Baseline"), None);
    // The baseline scorer writes the same name into its scores frame.
    assert_eq!(
        ScorerKind::Baseline.as_str(),
        l2d_baseline::scorer::SCORER_KIND
    );

    assert_eq!(Stratum::Pooled.to_string(), "pooled");
    assert_eq!(Stratum::Group("/a".into()).to_string(), "group /a");
}

#[test]
fn the_output_note_says_what_the_error_rate_depends_on() {
    // Requirement 11.9: the commands print this next to calibrated decisions.
    assert!(EXCHANGEABILITY_NOTE.contains("resemble the calibration logs"));
}

#[test]
fn a_calibrator_can_be_shared_between_threads() {
    fn shareable<T: Send + Sync>() {}
    shareable::<ConformalCalibrator>();
}

// ---------------------------------------------------------------------------------------
// P-values and label sets
// ---------------------------------------------------------------------------------------

#[test]
fn p_values_follow_the_documented_formula_on_a_set_small_enough_to_count_by_hand() {
    let rows = ungrouped(&[0.10, 0.20, 0.20, 0.30], &[0.70, 0.80, 0.80, 0.90]);
    let calibrator = fitted(&rows, 100);
    assert_eq!((calibrator.good(), calibrator.bad()), (4, 4));

    // Good: three of four calibration ratios are at least 0.20 (the tie counts).
    // Bad: no bad calibration log is as unlike bad as a log with ratio 0.20.
    let low = calibrator.predict(0.20, None, 0.25).unwrap();
    assert_eq!((low.p_good, low.p_bad), (0.8, 0.2));
    assert_eq!(low.set, LabelSet::Good);
    assert_eq!(low.error_rate, 0.25);
    assert_eq!(
        (&low.good_stratum, &low.bad_stratum),
        (&Stratum::Pooled, &Stratum::Pooled)
    );

    let high = calibrator.predict(0.80, None, 0.25).unwrap();
    assert_eq!((high.p_good, high.p_bad), (0.2, 0.8));
    assert_eq!(high.set, LabelSet::Bad);

    // Between the two clouds the log is unlike both.
    let between = calibrator.predict(0.50, None, 0.25).unwrap();
    assert_eq!((between.p_good, between.p_bad), (0.2, 0.2));
    assert_eq!(between.set, LabelSet::Neither);

    // A label stays in the set only when its p-value is greater than the error rate:
    // a p-value equal to it is ruled out.
    let equal = calibrator.predict(0.20, None, 0.2).unwrap();
    assert_eq!((equal.p_good, equal.p_bad), (0.8, 0.2));
    assert_eq!(equal.set, LabelSet::Good);

    // The lowest and the highest ratio: every calibration log of the fitting label is
    // at least as unlike it, so its p-value is 1.
    let lowest = calibrator.predict(0.0, None, 0.25).unwrap();
    assert_eq!((lowest.p_good, lowest.p_bad), (1.0, 0.2));
    let highest = calibrator.predict(1.0, None, 0.25).unwrap();
    assert_eq!((highest.p_good, highest.p_bad), (0.2, 1.0));
}

#[test]
fn overlapping_labels_give_the_set_both() {
    let rows = ungrouped(&[0.40, 0.50, 0.60, 0.70], &[0.30, 0.40, 0.50, 0.60]);
    let calibrator = fitted(&rows, 100);
    let middle = calibrator.predict(0.50, None, 0.25).unwrap();
    assert_eq!((middle.p_good, middle.p_bad), (0.8, 0.8));
    assert_eq!(middle.set, LabelSet::Both);
}

/// The p-values the calibrator returns, and the sets made from them, against a
/// direct count over the calibration ratios. The calibration set holds many equal
/// ratios, and every calibration ratio is itself asked for, so ties are exercised.
#[test]
fn p_values_and_sets_equal_a_direct_computation_ties_included() {
    let mut draws = Draws(11);
    // Ratios rounded to two decimals: about 500 logs on at most 101 values.
    let rounded = |ratio: f32| (ratio * 100.0).round() / 100.0;
    let good: Vec<f32> = draws
        .ratios(500, 0.35, 0.2)
        .into_iter()
        .map(rounded)
        .collect();
    let bad: Vec<f32> = draws
        .ratios(400, 0.65, 0.2)
        .into_iter()
        .map(rounded)
        .collect();
    let calibrator = fitted(&ungrouped(&good, &bad), 100);

    let mut asked: Vec<f32> = grid();
    asked.extend(good.iter().chain(bad.iter()).copied());
    let mut ties = 0;
    for error_rate in [0.01, 0.05, 0.1, 0.2, 0.5] {
        for ratio in &asked {
            let found = calibrator.predict(*ratio, None, error_rate).unwrap();
            let p_good = direct_p(&good, Label::Good, *ratio);
            let p_bad = direct_p(&bad, Label::Bad, *ratio);
            assert_eq!(
                (found.p_good.to_bits(), found.p_bad.to_bits()),
                (p_good.to_bits(), p_bad.to_bits()),
                "p-values of ratio {ratio}"
            );
            assert_eq!(
                found.set,
                direct_set(p_good, p_bad, error_rate),
                "set of ratio {ratio}"
            );
            ties += usize::from(good.contains(ratio) || bad.contains(ratio));
        }
    }
    assert!(ties > 4_000, "the test asked for {ties} tied ratios");
}

#[test]
fn every_p_value_is_a_multiple_of_one_over_n_plus_one_and_never_zero() {
    let (good, bad) = overlapping();
    let calibrator = fitted(&ungrouped(&good, &bad), 100);
    for decision in decisions(&calibrator, None, 0.05) {
        for p in [decision.p_good, decision.p_bad] {
            assert!((1.0 / 301.0..=1.0).contains(&p), "p-value {p}");
            let steps = f64::from(p) * 301.0;
            assert!(
                (steps - steps.round()).abs() < 1e-4,
                "p-value {p} is not k/301"
            );
        }
    }
}

#[test]
fn a_higher_ratio_never_makes_good_more_plausible_or_bad_less() {
    let (good, bad) = overlapping();
    let calibrator = fitted(&ungrouped(&good, &bad), 100);
    let found = decisions(&calibrator, None, 0.05);
    let found = &found[..201]; // the ascending part of the grid
    for pair in found.windows(2) {
        assert!(pair[1].p_good <= pair[0].p_good);
        assert!(pair[1].p_bad >= pair[0].p_bad);
    }
    assert_eq!(found[0].set, LabelSet::Good);
    assert_eq!(found[200].set, LabelSet::Bad);
}

// ---------------------------------------------------------------------------------------
// Achieved error rate on known distributions
// ---------------------------------------------------------------------------------------

/// Calibration and test ratios are drawn from the same two distributions. Split
/// conformal prediction then errs, per label, at a rate whose expectation is at most
/// the chosen rate and which varies with the calibration draw by about
/// `sqrt(e (1 - e) / n)`: 0.005 for `e = 0.05` and `n = 2,000`, 0.009 for `e = 0.2`.
/// 20,000 test logs per label add 0.002 to 0.003. The tolerance of 0.03 is three to
/// six of those standard deviations; the draws are fixed, so the outcome is too.
#[test]
fn the_chosen_error_rate_is_achieved_per_label_on_known_distributions() {
    let mut draws = Draws(2010);
    let good = draws.ratios(2_000, 0.35, 0.15);
    let bad = draws.ratios(2_000, 0.65, 0.15);
    let calibrator = fitted(&ungrouped(&good, &bad), 100);
    let test_good = draws.ratios(20_000, 0.35, 0.15);
    let test_bad = draws.ratios(20_000, 0.65, 0.15);

    for error_rate in [0.05_f32, 0.1, 0.2] {
        let chosen = f64::from(error_rate);
        for (label, ratios) in [(Label::Good, &test_good), (Label::Bad, &test_bad)] {
            let achieved = achieved_error(&calibrator, ratios, label, None, error_rate);
            println!("L2D_CONFORMAL_SYNTHETIC e={error_rate} {label:?} achieved={achieved:.4}");
            assert!(
                (achieved - chosen).abs() <= 0.03,
                "{label:?} at {error_rate}: achieved {achieved:.4}"
            );
        }
        // The sets are not trivially right: most logs get exactly one label.
        let single = test_good
            .iter()
            .chain(test_bad.iter())
            .filter(|ratio| {
                let set = calibrator.predict(**ratio, None, error_rate).unwrap().set;
                matches!(set, LabelSet::Good | LabelSet::Bad)
            })
            .count();
        assert!(
            single > 20_000,
            "only {single} of 40,000 sets name one label"
        );
    }
}

/// Two groups whose ratios sit in different places. Calibrated per group, each group
/// reaches the chosen rate for each label. Calibrated on the pooled lists alone, the
/// rate holds only on average: the good logs of the group with the higher ratios are
/// ruled out far more often than chosen.
#[test]
fn group_strata_reach_the_error_rate_in_each_group_where_the_pooled_list_does_not() {
    const E: f32 = 0.05;
    let mut draws = Draws(1632);
    // (group, centre of good, centre of bad)
    let groups = [("/low", 0.20, 0.60), ("/high", 0.50, 0.90)];
    let mut calibration: Vec<(f32, Label, Option<&str>)> = Vec::new();
    for (group, good, bad) in groups {
        for ratio in draws.ratios(1_500, good, 0.1) {
            calibration.push((ratio, Label::Good, Some(group)));
        }
        for ratio in draws.ratios(1_500, bad, 0.1) {
            calibration.push((ratio, Label::Bad, Some(group)));
        }
    }
    let per_group = fitted(&calibration, 100);
    let pooled_only = fitted(&calibration, u32::MAX);

    for (group, good, bad) in groups {
        let test_good = draws.ratios(10_000, good, 0.1);
        let test_bad = draws.ratios(10_000, bad, 0.1);
        let decision = per_group.predict(0.5, Some(group), E).unwrap();
        assert_eq!(decision.good_stratum, Stratum::Group(group.into()));
        assert_eq!(decision.bad_stratum, Stratum::Group(group.into()));
        let decision = pooled_only.predict(0.5, Some(group), E).unwrap();
        assert_eq!(
            (decision.good_stratum, decision.bad_stratum),
            (Stratum::Pooled, Stratum::Pooled)
        );

        for (label, ratios) in [(Label::Good, &test_good), (Label::Bad, &test_bad)] {
            let achieved = achieved_error(&per_group, ratios, label, Some(group), E);
            let pooled = achieved_error(&pooled_only, ratios, label, Some(group), E);
            println!(
                "L2D_CONFORMAL_GROUPS {group} {label:?} per_group={achieved:.4} pooled={pooled:.4}"
            );
            // n = 1,500 per list: one standard deviation is about 0.006.
            assert!(
                (achieved - 0.05).abs() <= 0.025,
                "{group} {label:?}: {achieved:.4}"
            );
        }
    }

    let high_good = draws.ratios(10_000, 0.50, 0.1);
    let low_good = draws.ratios(10_000, 0.20, 0.1);
    let missed_high = achieved_error(&pooled_only, &high_good, Label::Good, Some("/high"), E);
    let missed_low = achieved_error(&pooled_only, &low_good, Label::Good, Some("/low"), E);
    assert!(
        missed_high > 0.08,
        "pooled, good logs of /high: {missed_high:.4}"
    );
    assert!(
        missed_low < 0.02,
        "pooled, good logs of /low: {missed_low:.4}"
    );
}

// ---------------------------------------------------------------------------------------
// Repeatability
// ---------------------------------------------------------------------------------------

#[test]
fn repeated_calibration_gives_identical_label_sets_and_the_same_revision() {
    let (good, bad) = overlapping();
    let rows = ungrouped(&good, &bad);
    let first = fitted(&rows, 100);
    let second = fitted(&rows, 100);
    assert_eq!(first.revision(), second.revision());
    assert_eq!(first, second);
    for error_rate in [0.05, 0.1, 0.3] {
        assert_eq!(
            decisions(&first, None, error_rate),
            decisions(&second, None, error_rate)
        );
        // Asking the same calibrator again changes nothing either.
        assert_eq!(
            decisions(&first, None, error_rate),
            decisions(&first, None, error_rate)
        );
    }
}

#[test]
fn the_order_of_the_calibration_logs_does_not_matter() {
    let (good, bad) = overlapping();
    let rows = ungrouped(&good, &bad);
    let mut reversed = rows.clone();
    reversed.reverse();
    let (forward, backward) = (fitted(&rows, 100), fitted(&reversed, 100));
    assert_eq!(forward.revision(), backward.revision());
    assert_eq!(
        decisions(&forward, None, 0.05),
        decisions(&backward, None, 0.05)
    );
}

#[test]
fn the_revision_names_the_scorer_the_settings_and_the_calibration_logs() {
    let (good, bad) = overlapping();
    let rows = ungrouped(&good, &bad);
    let fit = |rows: &[Row<'_>], kind, revision: &str, min| {
        ConformalCalibrator::fit_with(&frame(rows), kind, revision, &settings(min), &budget())
            .unwrap()
            .revision()
            .to_owned()
    };
    let base = fit(&rows, ScorerKind::Baseline, REVISION, 100);
    assert_eq!(base.len(), 40);
    assert!(
        base.bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    );

    let mut changed = rows.clone();
    changed[0].0 += 0.001;
    let others = [
        fit(&rows, ScorerKind::Neighbour, REVISION, 100),
        fit(
            &rows,
            ScorerKind::Baseline,
            &REVISION.replace('0', "1"),
            100,
        ),
        fit(&rows, ScorerKind::Baseline, REVISION, 99),
        fit(&changed, ScorerKind::Baseline, REVISION, 100),
        fit(&rows[1..], ScorerKind::Baseline, REVISION, 100),
    ];
    for (index, other) in others.iter().enumerate() {
        assert_ne!(&base, other, "change {index} left the revision as it was");
    }
}

// ---------------------------------------------------------------------------------------
// Strata
// ---------------------------------------------------------------------------------------

/// 100 good logs in `/big`, 99 in `/small`, 5 without a group; 100 bad logs in `/small`
/// and 30 in `/big`.
fn strata_rows() -> Vec<Row<'static>> {
    let mut draws = Draws(5);
    let mut rows: Vec<Row<'static>> = Vec::new();
    let mut add = |count: usize, centre: f64, label: Label, group: Option<&'static str>| {
        for ratio in draws.ratios(count, centre, 0.05) {
            rows.push((ratio, label, group));
        }
    };
    add(100, 0.10, Label::Good, Some("/big"));
    add(99, 0.40, Label::Good, Some("/small"));
    add(5, 0.40, Label::Good, None);
    add(100, 0.90, Label::Bad, Some("/small"));
    add(30, 0.60, Label::Bad, Some("/big"));
    rows
}

#[test]
fn a_group_gets_its_own_list_for_a_label_from_the_configured_number_of_logs() {
    let calibrator = fitted(&strata_rows(), 100);
    assert_eq!((calibrator.good(), calibrator.bad()), (204, 130));

    // The lists, in their fixed order: per label the pooled list, then the groups.
    let strata: Vec<(Label, Stratum, u64)> = calibrator
        .strata()
        .into_iter()
        .map(|count| (count.label, count.stratum, count.logs))
        .collect();
    assert_eq!(
        strata,
        [
            (Label::Good, Stratum::Pooled, 204),
            (Label::Good, Stratum::Group("/big".into()), 100),
            (Label::Bad, Stratum::Pooled, 130),
            (Label::Bad, Stratum::Group("/small".into()), 100),
        ]
    );

    // Each decision names the list used for each label.
    let big = calibrator.predict(0.3, Some("/big"), 0.05).unwrap();
    assert_eq!(big.good_stratum, Stratum::Group("/big".into()));
    assert_eq!(big.bad_stratum, Stratum::Pooled);
    let small = calibrator.predict(0.3, Some("/small"), 0.05).unwrap();
    assert_eq!(small.good_stratum, Stratum::Pooled);
    assert_eq!(small.bad_stratum, Stratum::Group("/small".into()));
    for group in [None, Some("/never-seen"), Some("")] {
        let other = calibrator.predict(0.3, group, 0.05).unwrap();
        assert_eq!(
            (other.good_stratum, other.bad_stratum),
            (Stratum::Pooled, Stratum::Pooled)
        );
    }

    // The group's list and the pooled list give different answers: 0.3 is far above
    // every good log of /big, but ordinary among the good logs of all groups.
    assert_eq!(big.p_good, 1.0 / 101.0);
    assert!(small.p_good > 0.5, "pooled p-value {}", small.p_good);
    assert_eq!(big.set, LabelSet::Neither);
    assert_eq!(small.set, LabelSet::Good);
}

#[test]
fn the_pooled_list_of_a_label_holds_the_logs_of_every_group() {
    let rows = strata_rows();
    let calibrator = fitted(&rows, 100);
    let all_good: Vec<f32> = rows
        .iter()
        .filter(|row| row.1 == Label::Good)
        .map(|row| row.0)
        .collect();
    let big_good: Vec<f32> = rows
        .iter()
        .filter(|row| row.1 == Label::Good && row.2 == Some("/big"))
        .map(|row| row.0)
        .collect();
    for ratio in grid() {
        let pooled = calibrator.predict(ratio, None, 0.05).unwrap();
        assert_eq!(pooled.p_good, direct_p(&all_good, Label::Good, ratio));
        let own = calibrator.predict(ratio, Some("/big"), 0.05).unwrap();
        assert_eq!(own.p_good, direct_p(&big_good, Label::Good, ratio));
    }
}

#[test]
fn group_names_are_compared_exactly() {
    let calibrator = fitted(&strata_rows(), 100);
    for near in ["/BIG", "/big ", "/big/", "big"] {
        let decision = calibrator.predict(0.3, Some(near), 0.05).unwrap();
        assert_eq!(decision.good_stratum, Stratum::Pooled, "group '{near}'");
    }
}

#[test]
fn logs_labelled_unknown_are_not_calibration_logs() {
    let (good, bad) = overlapping();
    let mut rows = ungrouped(&good, &bad);
    let without = fitted(&rows, 100);
    rows.push((0.99, Label::Unknown, None));
    rows.insert(0, (0.01, Label::Unknown, Some("/g")));
    let with = fitted(&rows, 100);
    assert_eq!((with.good(), with.bad()), (300, 300));
    assert_eq!(with.revision(), without.revision());
}

// ---------------------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------------------

fn evenly(count: usize, from: f32, to: f32) -> Vec<f32> {
    (0..count)
        .map(|step| from + (to - from) * step as f32 / count as f32)
        .collect()
}

#[test]
fn an_error_rate_the_counts_do_not_support_is_refused_with_counts_and_smallest_rate() {
    // 19 logs support 1/20 = 0.05; 18 logs support 1/19 only.
    let enough = fitted(
        &ungrouped(&evenly(19, 0.0, 0.4), &evenly(40, 0.6, 1.0)),
        100,
    );
    assert_eq!(enough.smallest_supported(), 0.05);
    assert!(enough.predict(0.5, None, 0.05).is_ok());
    assert!(enough.require_error_rate(0.05).is_ok());

    let short = fitted(
        &ungrouped(&evenly(40, 0.0, 0.4), &evenly(18, 0.6, 1.0)),
        100,
    );
    let smallest = (1.0_f64 / 19.0) as f32;
    assert_eq!(short.smallest_supported(), smallest);
    let refused = short.predict(0.5, None, 0.05).unwrap_err();
    assert_eq!(
        refused,
        CalibrationError::TooFewLogs {
            good: 40,
            bad: 18,
            smallest_supported: smallest
        }
    );
    assert_eq!(short.require_error_rate(0.05).unwrap_err(), refused);
    let message = refused.to_string();
    assert!(
        message.contains("40 good") && message.contains("18 bad"),
        "{message}"
    );
    assert!(message.contains("0.0526"), "{message}");
    let as_tool_error = L2dError::from(refused);
    assert_eq!(as_tool_error.exit_code(), 4);

    // The smallest supported rate itself, and anything above it, is accepted.
    assert!(short.predict(0.5, None, smallest).is_ok());
    assert!(short.predict(0.5, None, 0.06).is_ok());
}

#[test]
fn a_decimal_error_rate_exactly_at_the_limit_is_accepted() {
    // 1/100 is supported by 99 logs. 0.01 as a 32-bit float lies just below 1/100; the
    // comparison is made on the 32-bit values, so the rate the analyst typed counts.
    let exact = fitted(
        &ungrouped(&evenly(99, 0.0, 0.4), &evenly(99, 0.6, 1.0)),
        100,
    );
    assert!(exact.predict(0.5, None, 0.01).is_ok());
    let short = fitted(
        &ungrouped(&evenly(98, 0.0, 0.4), &evenly(99, 0.6, 1.0)),
        100,
    );
    assert!(matches!(
        short.predict(0.5, None, 0.01),
        Err(CalibrationError::TooFewLogs {
            good: 98,
            bad: 99,
            ..
        })
    ));
}

#[test]
fn the_smallest_list_in_use_decides_which_error_rates_are_supported() {
    // With ten logs per group as the minimum, the group lists of ten logs support 1/11.
    let mut rows = ungrouped(&evenly(200, 0.0, 0.4), &evenly(200, 0.6, 1.0));
    for ratio in evenly(10, 0.0, 0.4) {
        rows.push((ratio, Label::Good, Some("/ten")));
    }
    let calibrator = fitted(&rows, 10);
    let smallest = (1.0_f64 / 11.0) as f32;
    assert_eq!(calibrator.smallest_supported(), smallest);
    // The refusal is the same for every log, whichever list the log would use.
    for group in [Some("/ten"), None] {
        assert_eq!(
            calibrator.predict(0.5, group, 0.05).unwrap_err(),
            CalibrationError::TooFewLogs {
                good: 10,
                bad: 200,
                smallest_supported: smallest
            }
        );
        assert!(calibrator.predict(0.5, group, 0.1).is_ok());
    }
    // With the default minimum the group is pooled and 0.05 is supported.
    assert!(fitted(&rows, 100).predict(0.5, Some("/ten"), 0.05).is_ok());
}

#[test]
fn calibration_without_logs_of_a_label_is_refused() {
    for (good, bad) in [(0, 50), (50, 0), (0, 0)] {
        let rows = ungrouped(&evenly(good, 0.0, 0.4), &evenly(bad, 0.6, 1.0));
        assert_eq!(
            fit_rows(&rows, 100).unwrap_err(),
            CalibrationError::TooFewLogs {
                good: good as u64,
                bad: bad as u64,
                smallest_supported: 1.0
            }
        );
    }
}

#[test]
fn error_rates_outside_zero_to_one_are_invalid() {
    let (good, bad) = overlapping();
    let calibrator = fitted(&ungrouped(&good, &bad), 100);
    for error_rate in [0.0, 1.0, -0.05, 1.5, f32::NAN, f32::INFINITY] {
        match calibrator.predict(0.5, None, error_rate) {
            Err(error @ CalibrationError::InvalidErrorRate { .. }) => {
                assert_eq!(L2dError::from(error).exit_code(), 2);
            }
            other => panic!("error rate {error_rate}: {other:?}"),
        }
    }
}

#[test]
fn ratios_outside_zero_to_one_are_invalid() {
    let (good, bad) = overlapping();
    let calibrator = fitted(&ungrouped(&good, &bad), 100);
    for ratio in [-0.01, 1.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            matches!(
                calibrator.predict(ratio, None, 0.05),
                Err(CalibrationError::InvalidRatio { .. })
            ),
            "ratio {ratio}"
        );
        let mut rows = ungrouped(&good, &bad);
        rows.push((ratio, Label::Good, None));
        match fit_rows(&rows, 100) {
            Err(CalibrationError::InvalidPoints { detail }) => {
                assert!(detail.contains("row 600"), "{detail}");
            }
            other => panic!("ratio {ratio}: {other:?}"),
        }
    }
}

#[test]
fn a_frame_that_is_not_a_calibration_points_frame_is_invalid() {
    let good = frame(&ungrouped(&[0.1; 30], &[0.9; 30]));
    assert_eq!(
        good.get_column_names_str(),
        ["ratio", "label", "group"],
        "the documented columns"
    );
    assert_eq!(
        good.dtypes(),
        [DataType::Float32, DataType::String, DataType::String]
    );
    let fit = |frame: &DataFrame| {
        ConformalCalibrator::fit_with(
            frame,
            ScorerKind::Baseline,
            REVISION,
            &settings(100),
            &budget(),
        )
    };
    assert!(fit(&good).is_ok());

    let invalid = |frame: DataFrame, what: &str| match fit(&frame) {
        Err(CalibrationError::InvalidPoints { .. }) => {}
        other => panic!("{what}: {other:?}"),
    };
    // A column missing, one too many, another order, another type.
    invalid(good.drop("group").unwrap(), "no group column");
    let mut extra = good.clone();
    extra
        .with_column(Column::new("text".into(), vec!["t"; 60]))
        .unwrap();
    invalid(extra, "an extra column");
    invalid(
        good.select(["label", "ratio", "group"]).unwrap(),
        "another order",
    );
    let mut wide = good.clone();
    wide.with_column(Column::new("ratio".into(), vec![0.5_f64; 60]))
        .unwrap();
    invalid(wide, "ratio as Float64");
    // A null ratio, a null label, a label that is not a label.
    let nullable = |ratio: Option<f32>, label: Option<&str>| {
        let mut ratios: Vec<Option<f32>> = vec![Some(0.1); 59];
        ratios.push(ratio);
        let mut labels: Vec<Option<&str>> = vec![Some("good"); 30];
        labels.extend(vec![Some("bad"); 29]);
        labels.push(label);
        DataFrame::new(vec![
            Column::new("ratio".into(), ratios),
            Column::new("label".into(), labels),
            Column::new("group".into(), vec![None::<&str>; 60]),
        ])
        .unwrap()
    };
    assert!(fit(&nullable(Some(0.5), Some("bad"))).is_ok());
    invalid(nullable(None, Some("bad")), "a null ratio");
    invalid(nullable(Some(0.5), None), "a null label");
    invalid(
        nullable(Some(0.5), Some("Bad")),
        "a label that is not a label",
    );
}

#[test]
fn a_minimum_of_zero_logs_per_group_is_invalid() {
    let (good, bad) = overlapping();
    let error = fit_rows(&ungrouped(&good, &bad), 0).unwrap_err();
    assert!(
        matches!(error, CalibrationError::InvalidSettings { .. }),
        "{error:?}"
    );
    assert_eq!(L2dError::from(error).exit_code(), 2);
}

#[test]
fn a_calibration_above_the_memory_limit_is_refused_with_estimate_and_limit() {
    let mut draws = Draws(3);
    let rows = ungrouped(
        &draws.ratios(60_000, 0.3, 0.1),
        &draws.ratios(60_000, 0.7, 0.1),
    );
    let points = frame(&rows);
    let estimate = estimated_bytes(points.height() as u64, 0);
    assert!(
        estimate > MIB,
        "the estimate of 120,000 points is {estimate} bytes"
    );

    // 33 MiB as the limit leaves 1 MiB after what the tool keeps for itself.
    let tight = budget_with_limit(Some(33 * MIB));
    let fit = |budget: &MemoryBudget| {
        ConformalCalibrator::fit_with(
            &points,
            ScorerKind::Baseline,
            REVISION,
            &settings(100),
            budget,
        )
    };
    match fit(&tight) {
        Err(CalibrationError::Memory(error @ L2dError::Refused { .. })) => {
            let message = error.to_string();
            assert!(message.contains("calibrating"), "{message}");
            assert!(
                message.contains("needs the whole dataset in memory"),
                "{message}"
            );
            assert_eq!(
                L2dError::from(CalibrationError::Memory(error)).exit_code(),
                4
            );
        }
        other => panic!("expected a memory refusal, got {other:?}"),
    }
    assert!(fit(&budget()).is_ok());
    // The estimate grows with the number of points.
    assert!(estimated_bytes(0, 0) < estimated_bytes(1_000, 0));
    assert!(estimated_bytes(1_000, 0) < estimated_bytes(1_000_000, 0));
}

#[test]
fn a_calibration_is_tied_to_the_scorer_it_was_made_for() {
    let (good, bad) = overlapping();
    let calibrator = fitted(&ungrouped(&good, &bad), 100);
    assert_eq!(calibrator.scorer_kind(), ScorerKind::Baseline);
    assert_eq!(calibrator.scorer_revision(), REVISION);
    assert!(
        calibrator
            .require_scorer(ScorerKind::Baseline, REVISION)
            .is_ok()
    );
    let other = REVISION.replace('0', "f");
    for (kind, revision) in [
        (ScorerKind::Neighbour, REVISION),
        (ScorerKind::Baseline, &other),
    ] {
        let error = calibrator.require_scorer(kind, revision).unwrap_err();
        assert!(
            matches!(error, CalibrationError::OtherScorer { .. }),
            "{error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains(REVISION) && message.contains(revision),
            "{message}"
        );
        assert_eq!(L2dError::from(error).exit_code(), 4);
    }
}

// ---------------------------------------------------------------------------------------
// The fixture: provenance, saving and loading
// ---------------------------------------------------------------------------------------

/// The four of the fixture's eight groups that are held out for calibration.
const HELD_OUT: [&str; 4] = [
    "GET /tienda1/index.jsp",
    "GET /tienda1/publico/buscar.jsp",
    "POST /tienda1/miembros/editar.jsp",
    "POST /tienda1/publico/autenticar.jsp",
];

struct Fixture {
    scratch: Scratch,
    fitting: (LogsFrame, InputSummary),
    held_out: (LogsFrame, InputSummary),
    scorer: BaselineScorer,
}

/// The mini fixture split by group into two files below a scratch root: the scorer is
/// fitted on one, the other is held out for calibration.
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
    let read = |file: &str, lines: &[&str]| {
        let path = scratch.root.join(file);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        LabelledAdapter::new(LabelledFormat::Csv)
            .with_group_column("group")
            .read(&path)
            .expect("the part is readable")
    };
    let fitting = read("fitting.csv", &fitting);
    let held_out = read("held-out.csv", &held_out);
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
        scratch,
        fitting,
        held_out,
        scorer,
    }
}

impl Fixture {
    /// Calibrates on the held-out part with ten logs per group as the minimum.
    fn calibrate(&self) -> ConformalCalibrator {
        ConformalCalibrator::fit_baseline(
            &self.scorer,
            &self.held_out.0,
            std::slice::from_ref(&self.held_out.1),
            &settings(10),
            &budget(),
        )
        .expect("the held-out part calibrates")
    }
}

#[test]
fn the_held_out_part_of_the_fixture_calibrates_the_scorer() {
    let fixture = fixture("calibrates");
    let calibrator = fixture.calibrate();
    assert_eq!(calibrator.scorer_kind(), ScorerKind::Baseline);
    assert_eq!(calibrator.scorer_revision(), fixture.scorer.revision());
    assert_eq!(calibrator.settings(), &settings(10));

    // One calibration log per labelled distinct text of the held-out part.
    let held = fixture.held_out.0.to_records();
    let count = |label: Label| held.iter().filter(|record| record.label == label).count() as u64;
    assert_eq!(
        (calibrator.good(), calibrator.bad()),
        (count(Label::Good), count(Label::Bad))
    );
    assert!(calibrator.good() >= 40 && calibrator.bad() >= 30);

    // The ratios are the scorer's: the pooled p-values equal the direct computation.
    let ratios = |label: Label| -> Vec<f32> {
        held.iter()
            .filter(|record| record.label == label)
            .map(|record| fixture.scorer.ratio(&record.text))
            .collect()
    };
    let (good, bad) = (ratios(Label::Good), ratios(Label::Bad));
    for ratio in grid() {
        let found = calibrator.predict(ratio, None, 0.1).unwrap();
        assert_eq!(found.p_good, direct_p(&good, Label::Good, ratio));
        assert_eq!(found.p_bad, direct_p(&bad, Label::Bad, ratio));
    }

    // Each held-out group has 12 good and 9 bad texts: with ten as the minimum the good
    // logs are calibrated per group and the bad logs on the pooled list.
    let strata = calibrator.strata();
    assert_eq!(strata.len(), 6, "{strata:?}");
    for group in HELD_OUT {
        let decision = calibrator.predict(0.5, Some(group), 0.1).unwrap();
        assert_eq!(decision.good_stratum, Stratum::Group(group.into()));
        assert_eq!(decision.bad_stratum, Stratum::Pooled);
    }
    // Lists of 12 logs support 1/13, so the default rate is refused with that figure.
    assert_eq!(
        calibrator
            .predict(0.5, None, DEFAULT_ERROR_RATE)
            .unwrap_err(),
        CalibrationError::TooFewLogs {
            good: 12,
            bad: calibrator.bad(),
            smallest_supported: (1.0_f64 / 13.0) as f32
        }
    );
}

#[test]
fn logs_the_scorer_was_fitted_on_are_refused_for_calibration() {
    let fixture = fixture("provenance");
    let calibrate = |scorer: &BaselineScorer, logs: &LogsFrame, inputs: &[InputSummary]| {
        ConformalCalibrator::fit_baseline(scorer, logs, inputs, &settings(10), &budget())
    };
    let (fit_logs, fit_summary) = &fixture.fitting;
    let (held_logs, held_summary) = &fixture.held_out;

    // The fitting input itself.
    let error =
        calibrate(&fixture.scorer, fit_logs, std::slice::from_ref(fit_summary)).unwrap_err();
    assert_eq!(
        error,
        CalibrationError::SameInputAsFitting {
            content_sha256: fit_summary.content_sha256.clone()
        }
    );
    assert!(error.to_string().contains(&fit_summary.content_sha256));
    assert_eq!(L2dError::from(error).exit_code(), 4);

    // The fitting input among several, and its hash written in upper case.
    let mut shouted = fit_summary.clone();
    shouted.content_sha256 = shouted.content_sha256.to_uppercase();
    let both = [held_summary.clone(), shouted];
    assert!(matches!(
        calibrate(&fixture.scorer, held_logs, &both),
        Err(CalibrationError::SameInputAsFitting { .. })
    ));

    // A scorer that does not say what it was fitted on.
    let unknown = BaselineScorer::fit_with(
        fit_logs,
        &[],
        &BaselineSettings {
            buckets: 4_096,
            ..BaselineSettings::default()
        },
        &budget(),
    )
    .unwrap();
    let error = calibrate(&unknown, held_logs, std::slice::from_ref(held_summary)).unwrap_err();
    assert_eq!(error, CalibrationError::UnknownFittingInputs);
    assert_eq!(L2dError::from(error).exit_code(), 4);

    // Calibration logs that do not say where they come from.
    let error = calibrate(&fixture.scorer, held_logs, &[]).unwrap_err();
    assert_eq!(error, CalibrationError::UnknownCalibrationInputs);
    assert_eq!(L2dError::from(error).exit_code(), 4);

    // The held-out input is accepted.
    assert!(
        calibrate(
            &fixture.scorer,
            held_logs,
            std::slice::from_ref(held_summary)
        )
        .is_ok()
    );
}

#[test]
fn a_calibration_fitted_on_the_held_out_part_loads_back_identically() {
    let fixture = fixture("round-trip");
    let paths = fixture.scratch.paths();
    let calibrator = fixture.calibrate();

    let dir = calibrator.save(&paths, "mini").unwrap();
    assert_eq!(
        dir,
        paths
            .calibrations_dir()
            .join(format!("mini-{}", &calibrator.revision()[..12]))
    );
    assert_eq!(dir, paths.calibration_dir("mini", calibrator.revision()));
    assert!(
        dir.starts_with(std::env::temp_dir()),
        "{} is a scratch place",
        dir.display()
    );

    for loaded in [
        ConformalCalibrator::load(&dir).unwrap(),
        ConformalCalibrator::load_under(&paths, &dir).unwrap(),
    ] {
        assert_eq!(loaded, calibrator);
        assert_eq!(loaded.revision(), calibrator.revision());
        assert_eq!(loaded.scorer_kind(), ScorerKind::Baseline);
        assert_eq!(loaded.scorer_revision(), fixture.scorer.revision());
        assert_eq!(loaded.settings(), calibrator.settings());
        assert_eq!(loaded.strata(), calibrator.strata());
        assert_eq!(loaded.smallest_supported(), calibrator.smallest_supported());
        for group in [
            None,
            Some(HELD_OUT[0]),
            Some(HELD_OUT[3]),
            Some("/elsewhere"),
        ] {
            for error_rate in [0.1, 0.25, 0.5] {
                assert_eq!(
                    decisions(&loaded, group, error_rate),
                    decisions(&calibrator, group, error_rate)
                );
            }
        }
        // The held-out logs themselves get the same sets from both.
        for record in fixture.held_out.0.records() {
            let ratio = fixture.scorer.ratio(&record.text);
            let group = record.group.as_deref();
            assert_eq!(
                loaded.predict(ratio, group, 0.1).unwrap(),
                calibrator.predict(ratio, group, 0.1).unwrap()
            );
        }
    }
}

#[test]
fn the_saved_files_have_the_documented_layout_and_revision() {
    let fixture = fixture("layout");
    let paths = fixture.scratch.paths();
    let calibrator = fixture.calibrate();
    let dir = calibrator.save(&paths, "mini").unwrap();
    assert_eq!(files_of(&dir), [MANIFEST_FILE, SCORES_FILE]);
    assert_eq!(
        (MANIFEST_FILE, SCORES_FILE),
        ("calibration.json", "scores.bin")
    );
    assert_eq!(
        files_of(&paths.calibrations_dir()).len(),
        1,
        "no temporary directory is left"
    );
    assert_eq!(files_of(&paths.models_dir()), ["calibrations"]);

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
    assert_eq!(manifest["format"], 1);
    assert_eq!(manifest["kind"], "conformal-calibration");
    assert_eq!(manifest["revision"], calibrator.revision());
    assert_eq!(manifest["tool_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(manifest["scorer"]["kind"], "baseline");
    assert_eq!(manifest["scorer"]["revision"], fixture.scorer.revision());
    assert_eq!(manifest["settings"]["min_per_group"], 10);
    assert_eq!(manifest["good"], calibrator.good());
    assert_eq!(manifest["bad"], calibrator.bad());

    // The lists: per label the pooled list, then the groups in byte order of the name.
    let lists = manifest["lists"].as_array().unwrap();
    let strata = calibrator.strata();
    assert_eq!(lists.len(), strata.len());
    let scores = std::fs::read(dir.join(SCORES_FILE)).unwrap();
    let total: u64 = strata.iter().map(|count| count.logs).sum();
    assert_eq!(scores.len() as u64, total * 8);

    // The revision is the documented hash of the scorer, the settings and the lists.
    let mut bytes = b"l2d-conformal-calibration/1\n".to_vec();
    bytes.push(0); // baseline
    let scorer_revision = fixture.scorer.revision().as_bytes();
    bytes.extend_from_slice(&(scorer_revision.len() as u64).to_le_bytes());
    bytes.extend_from_slice(scorer_revision);
    bytes.extend_from_slice(&10_u32.to_le_bytes());
    bytes.extend_from_slice(&(lists.len() as u64).to_le_bytes());
    let mut offset = 0;
    for (list, count) in lists.iter().zip(&strata) {
        assert_eq!(list["label"], count.label.as_str());
        assert_eq!(list["logs"], count.logs);
        let length = count.logs as usize * 8;
        let values = &scores[offset..offset + length];
        offset += length;
        // Each list is sorted ascending and holds nonconformity values in 0..=1.
        let numbers = doubles(values);
        assert!(numbers.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(numbers.iter().all(|value| (0.0..=1.0).contains(value)));

        bytes.push(u8::from(count.label == Label::Bad));
        match &count.stratum {
            Stratum::Pooled => {
                assert!(list["group"].is_null());
                bytes.push(0);
                bytes.extend_from_slice(&0_u64.to_le_bytes());
            }
            Stratum::Group(group) => {
                assert_eq!(list["group"], group.as_str());
                bytes.push(1);
                bytes.extend_from_slice(&(group.len() as u64).to_le_bytes());
                bytes.extend_from_slice(group.as_bytes());
            }
        }
        bytes.extend_from_slice(&count.logs.to_le_bytes());
        bytes.extend_from_slice(values);
    }
    assert_eq!(calibrator.revision(), &content_sha256(&bytes)[..40]);

    // The bad pooled list holds 1 - ratio of every bad held-out log.
    let mut expected: Vec<f64> = fixture
        .held_out
        .0
        .records()
        .filter(|record| record.label == Label::Bad)
        .map(|record| nonconformity(fixture.scorer.ratio(&record.text), Label::Bad))
        .collect();
    expected.sort_by(f64::total_cmp);
    let pooled_bad = strata
        .iter()
        .position(|count| count.label == Label::Bad && count.stratum == Stratum::Pooled)
        .unwrap();
    let before: u64 = strata[..pooled_bad].iter().map(|count| count.logs).sum();
    let stored = doubles(&scores[before as usize * 8..][..expected.len() * 8]);
    assert_eq!(stored, expected);
}

#[test]
fn saving_again_keeps_the_directory_and_a_foreign_directory_is_not_overwritten() {
    let fixture = fixture("again");
    let paths = fixture.scratch.paths();
    let calibrator = fixture.calibrate();
    let dir = calibrator.save(&paths, "mini").unwrap();
    let written = std::fs::metadata(dir.join(SCORES_FILE))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(calibrator.save(&paths, "mini").unwrap(), dir);
    assert_eq!(
        std::fs::metadata(dir.join(SCORES_FILE))
            .unwrap()
            .modified()
            .unwrap(),
        written,
        "the same calibration is not written twice"
    );
    assert_eq!(files_of(&paths.calibrations_dir()).len(), 1);

    // Something else in the place of the calibration.
    std::fs::write(dir.join(MANIFEST_FILE), b"something else").unwrap();
    match calibrator.save(&paths, "mini") {
        Err(L2dError::Refused { reason }) => {
            assert!(reason.contains("left unchanged"), "{reason}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(
        std::fs::read(dir.join(MANIFEST_FILE)).unwrap(),
        b"something else"
    );
    assert_eq!(
        files_of(&paths.calibrations_dir()).len(),
        1,
        "no temporary directory is left"
    );
}

#[test]
fn changed_files_make_load_refuse() {
    let fixture = fixture("tampered");
    let paths = fixture.scratch.paths();
    let calibrator = fixture.calibrate();
    let dir = calibrator.save(&paths, "mini").unwrap();
    let manifest = std::fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap();
    let scores = std::fs::read(dir.join(SCORES_FILE)).unwrap();
    let restore = || {
        std::fs::write(dir.join(MANIFEST_FILE), &manifest).unwrap();
        std::fs::write(dir.join(SCORES_FILE), &scores).unwrap();
        assert_eq!(ConformalCalibrator::load(&dir).unwrap(), calibrator);
    };
    let refused = |what: &str| match ConformalCalibrator::load(&dir) {
        Err(error @ L2dError::Refused { .. }) => assert_eq!(error.exit_code(), 4, "{what}"),
        other => panic!("{what}: expected a refusal, got {other:?}"),
    };

    // One changed value: lower, so the list stays sorted and only the revision tells.
    let mut flipped = scores.clone();
    flipped[..8].copy_from_slice(&0.0_f64.to_bits().to_le_bytes());
    assert_ne!(
        flipped, scores,
        "the first value of the fixture is not zero"
    );
    std::fs::write(dir.join(SCORES_FILE), &flipped).unwrap();
    refused("a changed value");
    restore();

    // A list that is no longer sorted, and a value outside 0..=1.
    let mut unsorted = scores.clone();
    unsorted[..8].copy_from_slice(&1.0_f64.to_bits().to_le_bytes());
    std::fs::write(dir.join(SCORES_FILE), &unsorted).unwrap();
    refused("an unsorted list");
    let mut outside = scores.clone();
    outside[..8].copy_from_slice(&(-0.5_f64).to_bits().to_le_bytes());
    std::fs::write(dir.join(SCORES_FILE), &outside).unwrap();
    refused("a value below zero");
    restore();

    // A truncated and a lengthened scores file.
    std::fs::write(dir.join(SCORES_FILE), &scores[..scores.len() - 8]).unwrap();
    refused("a truncated scores file");
    let mut longer = scores.clone();
    longer.extend_from_slice(&[0; 8]);
    std::fs::write(dir.join(SCORES_FILE), &longer).unwrap();
    refused("a lengthened scores file");
    restore();

    // Another scorer revision, another minimum, another kind, an unreadable manifest.
    let scorer_revision = fixture.scorer.revision();
    let other_revision = format!("{}0", &scorer_revision[..scorer_revision.len() - 1]);
    let other_revision = if other_revision == scorer_revision {
        format!("{}1", &scorer_revision[..scorer_revision.len() - 1])
    } else {
        other_revision
    };
    for (what, from, to) in [
        (
            "another scorer revision",
            scorer_revision,
            other_revision.as_str(),
        ),
        (
            "another minimum",
            "\"min_per_group\": 10",
            "\"min_per_group\": 11",
        ),
        (
            "another scorer kind",
            "\"kind\": \"baseline\"",
            "\"kind\": \"neighbour\"",
        ),
        (
            "another kind of directory",
            "\"conformal-calibration\"",
            "\"baseline\"",
        ),
        ("another format", "\"format\": 1", "\"format\": 2"),
    ] {
        assert!(manifest.contains(from), "the manifest holds {from}");
        std::fs::write(dir.join(MANIFEST_FILE), manifest.replacen(from, to, 1)).unwrap();
        refused(what);
    }
    std::fs::write(dir.join(MANIFEST_FILE), "{").unwrap();
    refused("an unreadable manifest");
    restore();

    // The directory under another name.
    let renamed = paths.calibrations_dir().join("mini-000000000000");
    std::fs::rename(&dir, &renamed).unwrap();
    match ConformalCalibrator::load(&renamed) {
        Err(L2dError::Refused { reason }) => {
            assert!(reason.contains("mini-000000000000"), "{reason}")
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_missing_calibration_names_the_item_and_where_it_was_expected() {
    let fixture = fixture("missing");
    let paths = fixture.scratch.paths();
    let absent = paths.calibration_dir("absent", "0123456789abcdef");
    match ConformalCalibrator::load(&absent) {
        Err(error @ L2dError::Missing { .. }) => {
            assert_eq!(error.exit_code(), 3);
            assert!(error.to_string().contains("absent-0123456789ab"));
        }
        other => panic!("expected Missing, got {other:?}"),
    }

    let dir = fixture.calibrate().save(&paths, "mini").unwrap();
    for file in [SCORES_FILE, MANIFEST_FILE] {
        let kept = std::fs::read(dir.join(file)).unwrap();
        std::fs::remove_file(dir.join(file)).unwrap();
        match ConformalCalibrator::load(&dir) {
            Err(L2dError::Missing { item, expected_at }) => {
                assert!(item.contains(file), "{item}");
                assert_eq!(expected_at, dir.join(file));
            }
            other => panic!("without {file}: {other:?}"),
        }
        std::fs::write(dir.join(file), kept).unwrap();
    }
    assert!(ConformalCalibrator::load(&dir).is_ok());
}

#[test]
fn unusable_names_and_places_outside_the_model_directory_are_refused() {
    let fixture = fixture("places");
    let paths = fixture.scratch.paths();
    let calibrator = fixture.calibrate();
    let too_long = "n".repeat(65);
    for name in [
        "",
        ".hidden",
        "a/b",
        "a\\b",
        "..",
        "a b",
        "c:d",
        too_long.as_str(),
    ] {
        match calibrator.save(&paths, name) {
            Err(error @ L2dError::InvalidInput { .. }) => assert_eq!(error.exit_code(), 2),
            other => panic!("name '{name}': {other:?}"),
        }
    }
    assert!(!paths.models_dir().exists(), "nothing was written");

    // A project root written with a parent-directory component.
    let dotted = ProjectPaths::new(fixture.scratch.root.join("sub").join(".."));
    assert!(matches!(
        calibrator.save(&dotted, "mini"),
        Err(L2dError::Refused { .. })
    ));
    assert!(!paths.models_dir().exists(), "nothing was written");

    // Loading through the project only reads below its model directory.
    let dir = calibrator.save(&paths, "mini").unwrap();
    let elsewhere = fixture
        .scratch
        .root
        .join("elsewhere")
        .join(dir.file_name().unwrap());
    std::fs::create_dir_all(elsewhere.parent().unwrap()).unwrap();
    std::fs::rename(&dir, &elsewhere).unwrap();
    assert!(ConformalCalibrator::load(&elsewhere).is_ok());
    assert!(matches!(
        ConformalCalibrator::load_under(&paths, &elsewhere),
        Err(L2dError::Refused { .. })
    ));
}

#[test]
fn group_names_of_any_content_survive_saving() {
    let scratch = Scratch::new("names");
    let odd = ["", "/a b", "/ü/\"quoted\"\\n", "/tab\there", "line\nbreak"];
    let mut rows = Vec::new();
    let mut draws = Draws(9);
    for group in odd {
        for ratio in draws.ratios(20, 0.3, 0.1) {
            rows.push((ratio, Label::Good, Some(group)));
        }
        for ratio in draws.ratios(20, 0.7, 0.1) {
            rows.push((ratio, Label::Bad, Some(group)));
        }
    }
    let calibrator = fitted(&rows, 20);
    assert_eq!(calibrator.strata().len(), 2 + 2 * odd.len());
    let dir = calibrator.save(&scratch.paths(), "odd.names_1-x").unwrap();
    let loaded = ConformalCalibrator::load(&dir).unwrap();
    assert_eq!(loaded, calibrator);
    for group in odd {
        let decision = loaded.predict(0.5, Some(group), 0.1).unwrap();
        assert_eq!(decision.good_stratum, Stratum::Group(group.into()));
        assert_eq!(decision, calibrator.predict(0.5, Some(group), 0.1).unwrap());
    }
}
