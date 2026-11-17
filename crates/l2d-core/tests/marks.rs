//! Mark sets: the analyst's marked logs (requirements 16.1, 16.3, 16.12; task 1.21).
//!
//! The fixture is `tests/fixtures/mini-labelled.csv`: 200 data rows. A log is a
//! distinct text, named by the position of its first occurrence (the data-row index
//! for CSV, the packed entry and line for an archive). Row 121 has two
//! fields and is skipped by the CSV adapter, so the adapter gives 169 logs. The expected
//! positions below were read off the fixture by hand: the four rows with
//! `UNION SELECT` are the first occurrences at 7, 57, 83 and 188 (rows 64, 146 and 192
//! repeat three of them), and the eight rows that drop a table are at 8, 27, 46, 52,
//! 65, 84, 170 and 189.

use std::io::Write;
use std::path::{Path, PathBuf};

use std::cell::Cell;

use l2d_core::marks::{MarkQuery, MarkSet, MarkStore};
use l2d_core::polars::prelude::DataFrame;
use l2d_core::{
    ArchivePosition, InputAdapter, InputSummary, L2dError, Label, LabelledAdapter, LabelledFormat,
    LogsFrame, MemoryBudget, MemorySettings, ProjectPaths, ZipInputSettings,
};

const UNION_SELECT: [u64; 4] = [7, 57, 83, 188];
const DROP_TABLE: [u64; 8] = [8, 27, 46, 52, 65, 84, 170, 189];
const DROP_TABLE_PATTERN: &str = r"(?i)drop(\s|\+)+table";

/// A directory of its own in the system temp folder, removed when dropped.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("l2d-core-marks-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir }
    }

    fn project(&self) -> ProjectPaths {
        ProjectPaths::new(self.dir.join("project"))
    }

    /// An archive called `name` whose entries are `(entry path, lines)`, deflated.
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
    MemoryBudget::observe(&MemorySettings::default()).unwrap()
}

/// The fixture read as the tool reads it: the logs frame and the summary.
fn read(path: &Path) -> (LogsFrame, InputSummary, MemoryBudget) {
    let mut budget = budget();
    let (logs, summary, _) = LabelledAdapter::new(LabelledFormat::Csv)
        .read_with(path, &mut budget)
        .unwrap();
    (logs, summary, budget)
}

/// Every text of the fixture's 200 data rows, in order (repeats included, and the text
/// of row 121, which has a field too few for the adapter).
fn fixture_lines() -> Vec<String> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(fixture_path())
        .unwrap();
    reader
        .records()
        .map(|record| record.unwrap()[0].to_string())
        .collect()
}

fn texts(logs: &LogsFrame) -> Vec<String> {
    logs.records().map(|record| record.text).collect()
}

fn mark(query: MarkQuery) -> MarkSet {
    let (logs, summary, budget) = read(&fixture_path());
    MarkSet::mark("attacks", Label::Bad, query, &logs, &summary, &budget).unwrap()
}

#[test]
fn phrase_regex_and_positions_mark_the_expected_logs_on_the_fixture() {
    let (logs, summary, budget) = read(&fixture_path());
    // Row 121 has two fields and is a skipped record: 169 logs.
    assert_eq!(logs.len(), 169);
    assert_eq!(summary.skipped.len(), 1);
    assert_eq!(summary.skipped[0].position, 121);

    let phrase = MarkSet::mark(
        "union",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "UNION SELECT".into(),
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap();
    assert_eq!(phrase.positions(), UNION_SELECT);
    assert_eq!(phrase.count(), 4);
    assert_eq!(phrase.name(), "union");
    assert_eq!(phrase.label(), Label::Bad);
    assert_eq!(
        phrase.query(),
        &MarkQuery::Phrase {
            phrase: "UNION SELECT".into()
        }
    );
    let input = phrase.input().expect("a mark set of an input");
    assert_eq!(input.content_sha256, summary.content_sha256);
    assert_eq!(input.path, fixture_path().to_string_lossy());
    assert_eq!(input.archive_entries, None);
    assert!(phrase.store().is_none());
    assert!(phrase.ids(&budget).unwrap().is_empty());
    // The same logs as a plain scan of the rows: every text that contains the phrase.
    let scanned: Vec<u64> = logs
        .records()
        .filter(|record| record.text.contains("UNION SELECT"))
        .map(|record| record.first_position)
        .collect();
    assert_eq!(phrase.positions(), scanned);

    // A phrase is compared byte for byte: no other case, no syntax.
    let lower = MarkSet::mark(
        "lower",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "union select".into(),
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap();
    assert_eq!(lower.count(), 0);
    let dots = MarkSet::mark(
        "dots",
        Label::Good,
        MarkQuery::Phrase {
            phrase: "index.jsp?lang".into(),
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap();
    assert!(dots.count() > 0);
    assert!(
        logs.records()
            .filter(|record| dots.positions().contains(&record.first_position))
            .all(|record| record.text.contains("index.jsp?lang"))
    );

    let regex = MarkSet::mark(
        "drop",
        Label::Bad,
        MarkQuery::Regex {
            pattern: DROP_TABLE_PATTERN.into(),
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap();
    assert_eq!(regex.positions(), DROP_TABLE);
    assert_eq!(regex.count(), 8);

    let positions = MarkSet::mark(
        "picked",
        Label::Good,
        MarkQuery::Positions {
            positions: vec![8, 1, 7, 8],
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap();
    assert_eq!(positions.positions(), [1, 7, 8]);
    assert_eq!(positions.count(), 3);
    assert_eq!(positions.label(), Label::Good);
}

#[test]
fn a_mark_set_lists_the_marked_logs_of_a_frame() {
    let (logs, _, _) = read(&fixture_path());
    let set = mark(MarkQuery::Phrase {
        phrase: "UNION SELECT".into(),
    });
    let marked = set.marked_in(&logs).unwrap();
    assert_eq!(marked.len(), 4);
    let positions: Vec<u64> = marked
        .records()
        .map(|record| record.first_position)
        .collect();
    assert_eq!(positions, UNION_SELECT);
    assert!(
        texts(&marked)
            .iter()
            .all(|text| text.contains("UNION SELECT"))
    );
    // The rows are the frame's own: labels and groups as read.
    assert!(marked.records().all(|record| record.label == Label::Bad));
}

#[test]
fn positions_that_name_no_log_are_refused() {
    let (logs, summary, budget) = read(&fixture_path());
    // 64 repeats the text of 57: the log is named by 57. 121 is a skipped record, 999
    // is beyond the input.
    let error = MarkSet::mark(
        "picked",
        Label::Bad,
        MarkQuery::Positions {
            positions: vec![7, 64, 121, 999],
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap_err();
    let L2dError::InvalidInput { detail } = error else {
        panic!("expected InvalidInput, got {error:?}");
    };
    assert!(detail.contains("64, 121, 999"), "{detail}");
    assert!(detail.contains("3 positions"), "{detail}");
}

#[test]
fn unusable_names_labels_and_queries_are_refused() {
    let (logs, summary, budget) = read(&fixture_path());
    let attempt = |name: &str, label: Label, query: MarkQuery| {
        MarkSet::mark(name, label, query, &logs, &summary, &budget).unwrap_err()
    };
    let phrase = || MarkQuery::Phrase {
        phrase: "UNION".into(),
    };
    for name in ["", "../escape", ".hidden", "a/b", &"n".repeat(65)] {
        assert!(
            matches!(
                attempt(name, Label::Bad, phrase()),
                L2dError::InvalidInput { .. }
            ),
            "{name:?}"
        );
    }
    let unknown = attempt("x", Label::Unknown, phrase());
    assert!(
        matches!(&unknown, L2dError::InvalidInput { detail } if detail.contains("bad or good")),
        "{unknown:?}"
    );
    let empty = attempt(
        "x",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: String::new(),
        },
    );
    assert!(matches!(empty, L2dError::InvalidInput { .. }));
    let broken = attempt(
        "x",
        Label::Bad,
        MarkQuery::Regex {
            pattern: "(unclosed".into(),
        },
    );
    assert!(
        matches!(&broken, L2dError::InvalidInput { detail } if detail.contains("(unclosed")),
        "{broken:?}"
    );
    let none = attempt(
        "x",
        Label::Bad,
        MarkQuery::Positions {
            positions: Vec::new(),
        },
    );
    assert!(matches!(none, L2dError::InvalidInput { .. }));
}

#[test]
fn a_mark_set_survives_save_and_load() {
    let scratch = Scratch::new("save-load");
    let paths = scratch.project();
    let set = mark(MarkQuery::Regex {
        pattern: DROP_TABLE_PATTERN.into(),
    });

    let saved = set.save(&paths, &budget()).unwrap();
    assert_eq!(saved, paths.root.join("marks").join("attacks.json"));
    assert_eq!(saved, paths.mark_set_path("attacks"));

    let loaded = MarkSet::load(&paths, "attacks", &budget()).unwrap();
    assert!(loaded.try_eq(&set).unwrap());
    assert_eq!(loaded.positions(), DROP_TABLE);
    assert_eq!(loaded.count(), 8);

    // The file names what the requirement asks for, readable as JSON.
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&saved).unwrap()).unwrap();
    assert_eq!(json["name"], "attacks");
    assert_eq!(json["label"], "bad");
    assert_eq!(json["query"]["kind"], "regex");
    assert_eq!(json["query"]["pattern"], DROP_TABLE_PATTERN);
    assert_eq!(json["source"]["kind"], "input");
    assert_eq!(
        json["source"]["content_sha256"],
        set.input().unwrap().content_sha256.as_str()
    );
    assert_eq!(json["count"], 8);
    assert_eq!(json["ids"].as_array().unwrap().len(), 0);
    assert_eq!(json["positions"].as_array().unwrap().len(), 8);
    assert_eq!(json["deselected"].as_array().unwrap().len(), 0);

    // Saving the same mark set again changes nothing; nothing else is left behind.
    assert_eq!(set.save(&paths, &budget()).unwrap(), saved);
    let (logs, summary, budget) = read(&fixture_path());
    let other = MarkSet::mark(
        "union",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "UNION SELECT".into(),
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap();
    other.save(&paths, &budget).unwrap();
    assert_eq!(MarkSet::list(&paths).unwrap(), ["attacks", "union"]);
    let files: Vec<String> = std::fs::read_dir(paths.marks_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(files.len(), 2, "{files:?}");
}

#[test]
fn a_name_in_use_by_another_mark_set_is_refused() {
    let scratch = Scratch::new("name-in-use");
    let paths = scratch.project();
    mark(MarkQuery::Regex {
        pattern: DROP_TABLE_PATTERN.into(),
    })
    .save(&paths, &budget())
    .unwrap();
    let error = mark(MarkQuery::Phrase {
        phrase: "UNION SELECT".into(),
    })
    .save(&paths, &budget())
    .unwrap_err();
    assert!(
        matches!(&error, L2dError::Refused { reason } if reason.contains("attacks")),
        "{error:?}"
    );
    // The first one is still there as it was.
    let loaded = MarkSet::load(&paths, "attacks", &budget()).unwrap();
    assert_eq!(loaded.positions(), DROP_TABLE);
}

#[test]
fn deselection_of_single_logs_is_recorded_and_kept() {
    let scratch = Scratch::new("deselect");
    let paths = scratch.project();
    let (logs, _, _) = read(&fixture_path());
    let mut set = mark(MarkQuery::Phrase {
        phrase: "UNION SELECT".into(),
    });
    set.save(&paths, &budget()).unwrap();

    set.deselect(&[57, 188]).unwrap();
    set.deselect(&[57]).unwrap();
    assert_eq!(set.deselected(), [57, 188]);
    assert_eq!(set.count(), 4, "the count is of the logs marked");
    assert_eq!(set.selected().collect::<Vec<_>>(), [7, 83]);
    assert_eq!(set.selected_count(), 2);
    let marked: Vec<u64> = set
        .marked_in(&logs)
        .unwrap()
        .records()
        .map(|record| record.first_position)
        .collect();
    assert_eq!(marked, [7, 83], "de-selected logs are left out");

    // A position that is not marked cannot be de-selected; nothing changes.
    let error = set.deselect(&[1, 7]).unwrap_err();
    assert!(
        matches!(&error, L2dError::InvalidInput { detail } if detail.contains('1')),
        "{error:?}"
    );
    assert_eq!(set.deselected(), [57, 188]);

    // The de-selection is saved over the same mark set, and loaded back.
    set.save(&paths, &budget()).unwrap();
    let loaded = MarkSet::load(&paths, "attacks", &budget()).unwrap();
    assert_eq!(loaded.deselected(), [57, 188]);
    assert!(loaded.try_eq(&set).unwrap());
}

#[test]
fn an_input_whose_hash_differs_is_refused() {
    let scratch = Scratch::new("changed-input");
    let set = mark(MarkQuery::Phrase {
        phrase: "UNION SELECT".into(),
    });
    let (_, same, _) = read(&fixture_path());
    set.require_input(&same, None).unwrap();

    // The fixture with one row changed: same rows, other content.
    let original = std::fs::read_to_string(fixture_path()).unwrap();
    let changed = original.replacen("pagina=12,good", "pagina=13,good", 1);
    assert_ne!(changed, original);
    let path = scratch.dir.join("changed.csv");
    std::fs::write(&path, changed).unwrap();
    let (_, other, _) = read(&path);

    let error = set.require_input(&other, None).unwrap_err();
    let L2dError::Refused { reason } = error else {
        panic!("expected Refused, got {error:?}");
    };
    assert!(reason.contains(&same.content_sha256), "{reason}");
    assert!(reason.contains(&other.content_sha256), "{reason}");
    assert!(reason.contains("attacks"), "{reason}");
}

#[test]
fn loading_what_is_not_a_mark_set_is_refused() {
    let scratch = Scratch::new("not-a-mark-set");
    let paths = scratch.project();
    let missing = MarkSet::load(&paths, "nothing", &budget()).unwrap_err();
    assert!(matches!(missing, L2dError::Missing { .. }), "{missing:?}");
    assert_eq!(MarkSet::list(&paths).unwrap(), Vec::<String>::new());

    std::fs::create_dir_all(paths.marks_dir()).unwrap();
    std::fs::write(paths.mark_set_path("junk"), b"{ not json").unwrap();
    let junk = MarkSet::load(&paths, "junk", &budget()).unwrap_err();
    assert!(matches!(junk, L2dError::Refused { .. }), "{junk:?}");

    // A file whose count does not match its positions is not taken either.
    let set = mark(MarkQuery::Phrase {
        phrase: "UNION SELECT".into(),
    });
    set.save(&paths, &budget()).unwrap();
    let path = paths.mark_set_path("attacks");
    let edited = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\"count\": 4", "\"count\": 5");
    std::fs::write(&path, edited).unwrap();
    let wrong = MarkSet::load(&paths, "attacks", &budget()).unwrap_err();
    assert!(
        matches!(&wrong, L2dError::Refused { reason } if reason.contains("count")),
        "{wrong:?}"
    );
}

#[test]
fn marking_an_archive_gives_the_logs_of_the_same_lines() {
    let scratch = Scratch::new("archive");
    let lines = fixture_lines();
    assert_eq!(lines.len(), 200);
    let first = vec!["first entry, one line".to_string()];
    let path = scratch.archive(
        "fixture.zip",
        &[("logs/first.log", &first), ("logs/fixture.log", &lines)],
    );
    let settings = ZipInputSettings {
        entries: vec!["logs/*.log".into()],
        ..ZipInputSettings::default()
    };

    let mut budget = budget();
    let (set, summary) = MarkSet::mark_archive(
        "union",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "UNION SELECT".into(),
        },
        &path,
        &settings,
        &mut budget,
    )
    .unwrap();
    // The lines of the second entry (index 1), by the archive's own position encoding.
    let expected: Vec<u64> = UNION_SELECT
        .iter()
        .map(|&line| ArchivePosition { entry: 1, line }.pack())
        .collect();
    assert_eq!(set.positions(), expected);
    assert_eq!(set.count(), 4);
    assert_eq!(
        set.input().unwrap().content_sha256,
        summary.input.content_sha256
    );
    assert_eq!(
        set.input().unwrap().archive_entries.as_deref(),
        Some(&["logs/*.log".to_string()][..])
    );
    assert_eq!(summary.input.records, 201);
    set.require_input(&summary.input, Some(&settings.entries))
        .unwrap();
    // The same archive read with other entry patterns: its positions may name other
    // logs (another entry can be left out before the marked one), so it is refused,
    // and so is the archive taken for a plain file.
    let other = ["logs/fixture.log".to_string()];
    let error = set.require_input(&summary.input, Some(&other)).unwrap_err();
    assert!(
        matches!(&error, L2dError::Refused { reason }
            if reason.contains("logs/*.log") && reason.contains("logs/fixture.log")),
        "{error:?}"
    );
    let error = set.require_input(&summary.input, None).unwrap_err();
    assert!(
        matches!(&error, L2dError::Refused { reason } if reason.contains("archive")),
        "{error:?}"
    );
    // A mark set of a file is refused for an archive read of the same bytes.
    let (logs, file_summary, budget_file) = read(&fixture_path());
    let of_file = MarkSet::mark(
        "file",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "UNION SELECT".into(),
        },
        &logs,
        &file_summary,
        &budget_file,
    )
    .unwrap();
    let error = of_file
        .require_input(&file_summary, Some(&settings.entries))
        .unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error:?}");

    let (regex, _) = MarkSet::mark_archive(
        "drop",
        Label::Bad,
        MarkQuery::Regex {
            pattern: DROP_TABLE_PATTERN.into(),
        },
        &path,
        &settings,
        &mut budget,
    )
    .unwrap();
    let expected: Vec<u64> = DROP_TABLE
        .iter()
        .map(|&line| ArchivePosition { entry: 1, line }.pack())
        .collect();
    assert_eq!(regex.positions(), expected);

    let wanted = vec![
        ArchivePosition { entry: 0, line: 1 }.pack(),
        ArchivePosition { entry: 1, line: 7 }.pack(),
    ];
    let (picked, _) = MarkSet::mark_archive(
        "picked",
        Label::Good,
        MarkQuery::Positions {
            positions: wanted.clone(),
        },
        &path,
        &settings,
        &mut budget,
    )
    .unwrap();
    assert_eq!(picked.positions(), wanted);

    // A repeat (line 64 repeats line 57) names no log of the archive.
    let error = MarkSet::mark_archive(
        "picked",
        Label::Good,
        MarkQuery::Positions {
            positions: vec![ArchivePosition { entry: 1, line: 64 }.pack()],
        },
        &path,
        &settings,
        &mut budget,
    )
    .unwrap_err();
    assert!(matches!(error, L2dError::InvalidInput { .. }), "{error:?}");
}

#[test]
fn marked_positions_that_do_not_fit_are_refused_with_the_figures() {
    let (logs, summary, _) = read(&fixture_path());
    // A limit of the tool's own share and 64 KiB, of which nearly all is held: the
    // positions of every log do not fit.
    let settings = MemorySettings {
        memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 64 * 1024),
        ..MemorySettings::default()
    };
    let mut budget = MemoryBudget::observe_with(&settings, || Ok(8 << 30)).unwrap();
    budget.hold(64 * 1024 - 100);
    let error = MarkSet::mark(
        "everything",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "tienda1".into(),
        },
        &logs,
        &summary,
        &budget,
    )
    .unwrap_err();
    let L2dError::Refused { reason } = error else {
        panic!("expected Refused, got {error:?}");
    };
    assert!(reason.contains("marked positions"), "{reason}");
    assert!(reason.contains("memory limit"), "{reason}");
}

/// Saving is held to the budget (review of task 1.21, round 1): the comparison read of a
/// file of the same name is counted as loading is, and refused with the figures when it
/// does not fit, leaving that file as it was; a new file is written as a stream.
#[test]
fn a_save_over_a_file_that_does_not_fit_the_budget_is_refused_and_leaves_it() {
    let scratch = Scratch::new("save-budget");
    let paths = scratch.project();
    let mut set = mark(MarkQuery::Regex {
        pattern: DROP_TABLE_PATTERN.into(),
    });
    let saved = set.save(&paths, &budget()).unwrap();
    let before = std::fs::read(&saved).unwrap();
    let file_bytes = before.len() as u64;

    // A budget whose room is below 16 times the file: the comparison read does not fit.
    let settings = MemorySettings {
        memory_limit_bytes: Some(l2d_core::memory::PROCESS_BASELINE_BYTES + 1024 * 1024),
        ..MemorySettings::default()
    };
    let mut tight = MemoryBudget::observe_with(&settings, || Ok(8 << 30)).unwrap();
    tight.hold(1024 * 1024 - 16 * file_bytes + 1);
    set.deselect(&[8]).unwrap();
    let error = set.save(&paths, &tight).unwrap_err();
    let L2dError::Refused { reason } = error else {
        panic!("expected Refused, got {error:?}");
    };
    assert!(reason.contains("saving the mark set `attacks`"), "{reason}");
    assert!(reason.contains(&(16 * file_bytes).to_string()), "{reason}");
    assert_eq!(
        std::fs::read(&saved).unwrap(),
        before,
        "the old file is unchanged"
    );
    assert_eq!(
        std::fs::read_dir(paths.marks_dir()).unwrap().count(),
        1,
        "nothing else is left behind"
    );

    // With room for the comparison read and the write buffer (one after the other) it
    // is saved; a new name needs no comparison read.
    tight.hold(1024 * 1024 - (16 * file_bytes).max(l2d_core::marks::SAVE_BUFFER_BYTES));
    set.save(&paths, &tight).unwrap();
    assert_eq!(
        MarkSet::load(&paths, "attacks", &budget())
            .unwrap()
            .deselected(),
        [8]
    );
    // A new file is written through a buffer of `SAVE_BUFFER_BYTES`, whatever its size.
    let mut nothing_left = MemoryBudget::observe_with(&settings, || Ok(8 << 30)).unwrap();
    nothing_left.hold(1024 * 1024 - l2d_core::marks::SAVE_BUFFER_BYTES);
    let union = MarkSet::mark(
        "union",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "UNION SELECT".into(),
        },
        &read(&fixture_path()).0,
        &read(&fixture_path()).1,
        &budget(),
    )
    .unwrap();
    union.save(&paths, &nothing_left).unwrap();
    nothing_left.hold(1024 * 1024 - l2d_core::marks::SAVE_BUFFER_BYTES + 1);
    let lower = MarkSet::mark(
        "lower",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "union select".into(),
        },
        &read(&fixture_path()).0,
        &read(&fixture_path()).1,
        &budget(),
    )
    .unwrap();
    let error = lower.save(&paths, &nothing_left).unwrap_err();
    assert!(matches!(error, L2dError::Refused { .. }), "{error:?}");
    assert!(!paths.mark_set_path("lower").exists());
}

/// A store whose logs change after its content hash is first taken: every call of
/// `content_sha256` gives the next hash. Its one log contains the phrase.
struct ChangingStore {
    hashes_taken: Cell<u32>,
}

impl MarkStore for ChangingStore {
    fn store_name(&self) -> &str {
        "changing"
    }

    fn content_sha256(&self, _: &mut MemoryBudget) -> Result<String, L2dError> {
        let taken = self.hashes_taken.get();
        self.hashes_taken.set(taken + 1);
        Ok(format!("{taken:064x}"))
    }

    fn phrase_ids(
        &self,
        _: &str,
        _: u64,
        _: &mut MemoryBudget,
        each: &mut dyn FnMut(&str),
    ) -> Result<(), L2dError> {
        each(&"a".repeat(64));
        Ok(())
    }

    fn text_batches(
        &self,
        _: u64,
        _: u64,
        _: &mut MemoryBudget,
        _: &mut dyn FnMut(&DataFrame) -> Result<u64, L2dError>,
    ) -> Result<(), L2dError> {
        Ok(())
    }
}

#[test]
fn a_store_that_changes_while_it_is_marked_is_refused() {
    let store = ChangingStore {
        hashes_taken: Cell::new(0),
    };
    let error = MarkSet::mark_store(
        "changing",
        Label::Bad,
        MarkQuery::Phrase {
            phrase: "UNION".into(),
        },
        &store,
        &mut budget(),
    )
    .unwrap_err();
    let L2dError::Refused { reason } = error else {
        panic!("expected Refused, got {error:?}");
    };
    assert!(reason.contains(&format!("{:064x}", 0)), "{reason}");
    assert!(reason.contains(&format!("{:064x}", 1)), "{reason}");
    assert!(reason.contains("changed while"), "{reason}");
}
