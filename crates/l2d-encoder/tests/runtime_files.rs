//! `RuntimeFiles::check` against temporary directories (task 2.3, requirements 4.3 and
//! 10.5), and on request against the real runtime directory with timings.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, UNIX_EPOCH};

use l2d_core::{L2dError, ProjectPaths};
use l2d_encoder::runtime::{
    CheckDepth, ManifestFile, ManifestSource, RECEIPT_FILE, RecordRefresh, RecordState,
    RuntimeFiles, RuntimeManifest,
};
use sha2::{Digest, Sha256};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "l2d-encoder-test-{}-runtime-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The three files of the test release, with their content.
const FILES: &[(&str, &[u8], &str)] = &[
    ("onnxruntime.dll", b"runtime of release two", "1.30.0"),
    ("provider.dll", b"the provider, a longer file", "2.0.0"),
    ("dispatch.dll", b"dispatch core", "2.0.0"),
];
const ALL: [&str; 3] = ["onnxruntime.dll", "provider.dll", "dispatch.dll"];
/// `provider.dll` with other content of the same length.
const ALTERED: &[u8] = b"the provider, a LONGER file";

fn manifest() -> RuntimeManifest {
    RuntimeManifest {
        format: 1,
        release: "2.0.0".into(),
        onnxruntime: "1.30.0".into(),
        source: ManifestSource {
            kind: "wheel".into(),
            package: "runtime-package".into(),
            version: "1.30.0".into(),
            directory: "onnxruntime/capi".into(),
            detail: [
                ("index", "https://example.invalid/2.0.0/simple/"),
                ("wheel", "runtime_package-1.30.0-py3-none-win_amd64.whl"),
            ]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
        },
        files: FILES
            .iter()
            .map(|(name, content, version)| ManifestFile {
                name: (*name).into(),
                bytes: content.len() as u64,
                sha256: sha256_hex(content),
                version: (*version).into(),
            })
            .collect(),
    }
}

/// A directory that holds exactly the files of the test release, and no record.
fn complete(name: &str) -> PathBuf {
    let dir = scratch(name);
    for (file, content, _) in FILES {
        std::fs::write(dir.join(file), content).unwrap();
    }
    dir
}

fn modified_ns(path: &Path) -> u64 {
    let modified = std::fs::metadata(path).unwrap().modified().unwrap();
    modified.duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

fn set_modified(path: &Path, to: std::time::SystemTime) {
    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_modified(to).unwrap();
}

/// The record the provisioning step leaves in the directory, for the files as they are
/// on disk now.
fn write_provisioning_record(dir: &Path, release: &str, onnxruntime: &str) {
    let files: Vec<serde_json::Value> = FILES
        .iter()
        .map(|(name, content, _)| {
            serde_json::json!({
                "name": name,
                "bytes": content.len(),
                "sha256": sha256_hex(content),
                "modified_unix_ns": modified_ns(&dir.join(name)),
            })
        })
        .collect();
    let record = serde_json::json!({
        "format": 1,
        "release": release,
        "onnxruntime": onnxruntime,
        "written_by": "provisioning",
        "provisioned_utc": "2026-11-12T12:00:00Z",
        "source": { "package": "runtime-package", "version": "1.30.0",
                    "installed_from": "https://example.invalid/wheel.whl" },
        "files": files,
    });
    std::fs::write(
        dir.join(RECEIPT_FILE),
        serde_json::to_string_pretty(&record).unwrap(),
    )
    .unwrap();
}

fn read_record(dir: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join(RECEIPT_FILE)).unwrap()).unwrap()
}

fn fast(dir: &Path) -> Result<RuntimeFiles, L2dError> {
    RuntimeFiles::check_against(&manifest(), dir, CheckDepth::Fast)
}

fn refusal(result: Result<RuntimeFiles, L2dError>) -> String {
    match result {
        Err(L2dError::Refused { reason }) => reason,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// --- No record: everything is hashed, and the record is written -------------------------

#[test]
fn without_a_record_every_file_is_hashed_and_the_record_is_written() {
    let dir = complete("no-record");

    let first = fast(&dir).unwrap();
    assert_eq!(first.dir, dir);
    assert_eq!(first.release, "2.0.0");
    assert_eq!(first.onnxruntime, "1.30.0");
    assert_eq!(first.hashed, ALL, "nothing is trusted without a record");
    assert_eq!(first.record, RecordState::Written);
    assert_eq!(first.refresh, RecordRefresh::NotNeeded);

    // What the check wrote: marked as its own, with no source, and with what it verified.
    let record = read_record(&dir);
    assert_eq!(record["format"], 1);
    assert_eq!(record["release"], "2.0.0");
    assert_eq!(record["onnxruntime"], "1.30.0");
    assert_eq!(record["written_by"], "check");
    assert!(record.get("source").is_none(), "{record}");
    assert!(record.get("installed_from").is_none(), "{record}");
    assert!(record["verified_unix_s"].as_u64().unwrap() > 1_700_000_000);
    let files = record["files"].as_array().unwrap();
    assert_eq!(files.len(), 3);
    for (entry, (name, content, _)) in files.iter().zip(FILES) {
        assert_eq!(entry["name"], *name);
        assert_eq!(entry["bytes"], content.len());
        assert_eq!(entry["sha256"], sha256_hex(content));
        assert_eq!(entry["modified_unix_ns"], modified_ns(&dir.join(name)));
    }
    // Nothing but the files and the record is left in the directory.
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "dispatch.dll",
            "onnxruntime.dll",
            "provider.dll",
            RECEIPT_FILE
        ]
    );

    // The second start is fast: nothing is hashed, the record is the check's.
    let second = fast(&dir).unwrap();
    assert!(second.hashed.is_empty(), "{:?}", second.hashed);
    assert_eq!(second.record, RecordState::Check);

    // The full check hashes everything, record or not.
    let full = RuntimeFiles::check_against(&manifest(), &dir, CheckDepth::Full).unwrap();
    assert_eq!(full.hashed, ALL);
    assert_eq!(full.record, RecordState::Check);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The case of a directory filled by hand from another package: same names, same sizes,
/// other content.
#[test]
fn a_same_size_replacement_without_a_record_is_refused_by_the_fast_check() {
    let dir = complete("same-size-no-record");
    assert_eq!(ALTERED.len(), FILES[1].1.len());
    std::fs::write(dir.join("provider.dll"), ALTERED).unwrap();
    let manifest = manifest();

    for depth in [CheckDepth::Fast, CheckDepth::Full] {
        let reason = refusal(RuntimeFiles::check_against(&manifest, &dir, depth));
        assert!(
            reason.contains(&dir.join("provider.dll").display().to_string()),
            "{reason}"
        );
        assert!(reason.contains(&sha256_hex(ALTERED)), "found: {reason}");
        assert!(
            reason.contains(&manifest.files[1].sha256),
            "expected: {reason}"
        );
        assert!(
            !dir.join(RECEIPT_FILE).exists(),
            "no record of a directory that failed"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A directory the record cannot be written into. The write goes through a file of this
/// process that is renamed into place; a folder of that name stands in for a directory
/// without write permission.
#[test]
fn when_the_record_cannot_be_written_every_start_hashes_and_says_so() {
    let dir = complete("not-writable");
    let blocker = dir.join(format!(".{RECEIPT_FILE}.{}.partial", std::process::id()));
    std::fs::create_dir(&blocker).unwrap();

    for _ in 0..2 {
        let checked = fast(&dir).expect("the files are right; the record is a convenience");
        assert_eq!(checked.hashed, ALL);
        let RecordState::NotWritten { reason } = &checked.record else {
            panic!("{:?}", checked.record);
        };
        assert!(
            reason.contains(&dir.join(RECEIPT_FILE).display().to_string()),
            "{reason}"
        );
        assert!(
            checked.record.describe().contains("every start"),
            "{}",
            checked.record.describe()
        );
        assert!(!dir.join(RECEIPT_FILE).exists());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

// --- A record written by the check is used and invalidated like the provisioning one ----

#[test]
fn a_record_is_invalidated_by_a_changed_modification_time_whoever_wrote_it() {
    for written_by_check in [false, true] {
        let dir = complete(if written_by_check {
            "record-check"
        } else {
            "record-provisioning"
        });
        let expected_state = if written_by_check {
            assert_eq!(fast(&dir).unwrap().record, RecordState::Written);
            RecordState::Check
        } else {
            write_provisioning_record(&dir, "2.0.0", "1.30.0");
            RecordState::Provisioning
        };

        // Untouched since: nothing is hashed.
        let checked = fast(&dir).unwrap();
        assert_eq!(checked.record, expected_state);
        assert!(checked.hashed.is_empty(), "{:?}", checked.hashed);

        // Touched, content unchanged: one start hashes that file, finds it right and
        // brings the record up to date.
        let target = dir.join("provider.dll");
        let later =
            std::fs::metadata(&target).unwrap().modified().unwrap() + Duration::from_secs(90);
        set_modified(&target, later);
        let before = read_record(&dir);
        let checked = fast(&dir).unwrap();
        assert_eq!(checked.hashed, vec!["provider.dll"]);
        assert_eq!(checked.record, expected_state);
        assert_eq!(
            checked.refresh,
            RecordRefresh::Refreshed {
                files: vec!["provider.dll".into()]
            }
        );
        assert!(checked.refresh.describe().contains("provider.dll"));

        // The record: the new time for that file, a note of the refresh, and everything
        // else as it was. Who wrote it and where the files came from stay true.
        let after = read_record(&dir);
        assert_eq!(after["files"][1]["name"], "provider.dll");
        assert_eq!(after["files"][1]["modified_unix_ns"], modified_ns(&target));
        assert_eq!(after["files"][1]["sha256"], sha256_hex(FILES[1].1));
        assert_eq!(after["files"][0], before["files"][0]);
        assert_eq!(after["files"][2], before["files"][2]);
        assert_eq!(after["reverified"]["by"], "check");
        assert_eq!(
            after["reverified"]["files"],
            serde_json::json!(["provider.dll"])
        );
        assert!(after["reverified"]["unix_s"].as_u64().unwrap() > 1_700_000_000);
        for key in ["format", "release", "onnxruntime", "written_by"] {
            assert_eq!(after[key], before[key], "{key}");
        }
        if written_by_check {
            assert_eq!(after["written_by"], "check");
            assert_eq!(after["verified_unix_s"], before["verified_unix_s"]);
            assert!(after.get("source").is_none());
        } else {
            assert_eq!(after["written_by"], "provisioning");
            assert_eq!(after["source"], before["source"]);
            assert_eq!(after["provisioned_utc"], before["provisioned_utc"]);
        }

        // The start after it is fast again.
        let checked = fast(&dir).unwrap();
        assert!(checked.hashed.is_empty(), "{:?}", checked.hashed);
        assert_eq!(checked.refresh, RecordRefresh::NotNeeded);
        assert_eq!(checked.record, expected_state);

        // Replaced by a file of the same size: found through the hash, refused, and the
        // record is not rewritten.
        std::fs::write(&target, ALTERED).unwrap();
        set_modified(&target, later + Duration::from_secs(90));
        let before = std::fs::read(dir.join(RECEIPT_FILE)).unwrap();
        for _ in 0..2 {
            let reason = refusal(fast(&dir));
            assert!(reason.contains(&target.display().to_string()), "{reason}");
            assert!(reason.contains(&sha256_hex(ALTERED)), "{reason}");
            assert_eq!(std::fs::read(dir.join(RECEIPT_FILE)).unwrap(), before);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// One touched file that is right and one that was replaced: refused, and the record
/// keeps the old time of the right one too (nothing is rewritten for a directory that
/// failed).
#[test]
fn a_refused_directory_does_not_get_its_record_refreshed() {
    let dir = complete("refresh-refused");
    write_provisioning_record(&dir, "2.0.0", "1.30.0");
    let before = std::fs::read(dir.join(RECEIPT_FILE)).unwrap();
    let later = std::time::SystemTime::now() + Duration::from_secs(300);
    set_modified(&dir.join("onnxruntime.dll"), later);
    std::fs::write(dir.join("provider.dll"), ALTERED).unwrap();
    set_modified(&dir.join("provider.dll"), later);
    refusal(fast(&dir));
    assert_eq!(std::fs::read(dir.join(RECEIPT_FILE)).unwrap(), before);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_file_the_record_does_not_name_is_hashed_and_added_to_the_record() {
    let dir = complete("record-incomplete");
    write_provisioning_record(&dir, "2.0.0", "1.30.0");
    let mut record = read_record(&dir);
    record["files"].as_array_mut().unwrap().remove(1);
    std::fs::write(dir.join(RECEIPT_FILE), record.to_string()).unwrap();
    let checked = fast(&dir).unwrap();
    assert_eq!(checked.hashed, vec!["provider.dll"]);
    assert_eq!(
        checked.refresh,
        RecordRefresh::Refreshed {
            files: vec!["provider.dll".into()]
        }
    );
    assert_eq!(read_record(&dir)["files"].as_array().unwrap().len(), 3);
    assert!(fast(&dir).unwrap().hashed.is_empty());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A record that cannot be rewritten: the check passes, says so, and hashes the touched
/// file at every start. (A folder at the name of the temporary file stands in for a
/// directory without write permission.)
#[test]
fn when_the_record_cannot_be_refreshed_the_check_passes_and_says_so() {
    let dir = complete("refresh-not-writable");
    write_provisioning_record(&dir, "2.0.0", "1.30.0");
    let before = std::fs::read(dir.join(RECEIPT_FILE)).unwrap();
    let target = dir.join("provider.dll");
    set_modified(
        &target,
        std::time::SystemTime::now() + Duration::from_secs(300),
    );
    std::fs::create_dir(dir.join(format!(".{RECEIPT_FILE}.{}.partial", std::process::id())))
        .unwrap();
    for _ in 0..2 {
        let checked = fast(&dir).unwrap();
        assert_eq!(checked.hashed, vec!["provider.dll"]);
        let RecordRefresh::Failed { files, reason } = &checked.refresh else {
            panic!("{:?}", checked.refresh);
        };
        assert_eq!(files, &vec!["provider.dll".to_string()]);
        assert!(reason.contains(RECEIPT_FILE), "{reason}");
        assert!(checked.refresh.describe().contains("every start"));
        assert_eq!(std::fs::read(dir.join(RECEIPT_FILE)).unwrap(), before);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_directory_recorded_for_another_release_is_refused_with_both_versions() {
    // A record of the provisioning step for another release.
    let dir = complete("other-release");
    write_provisioning_record(&dir, "1.8.0", "1.27.0");
    for depth in [CheckDepth::Fast, CheckDepth::Full] {
        let reason = refusal(RuntimeFiles::check_against(&manifest(), &dir, depth));
        assert!(reason.contains("1.8.0"), "found release: {reason}");
        assert!(reason.contains("1.27.0"), "found runtime: {reason}");
        assert!(reason.contains("2.0.0"), "expected release: {reason}");
        assert!(reason.contains("1.30.0"), "expected runtime: {reason}");
        assert!(reason.contains(&dir.display().to_string()), "{reason}");
    }
    std::fs::remove_dir_all(&dir).unwrap();

    // A record the check wrote, read by a build that expects another release.
    let dir = complete("other-release-check");
    assert_eq!(fast(&dir).unwrap().record, RecordState::Written);
    let mut newer = manifest();
    newer.release = "2.1.0".into();
    newer.onnxruntime = "1.31.0".into();
    let reason = refusal(RuntimeFiles::check_against(&newer, &dir, CheckDepth::Fast));
    for part in ["2.0.0", "1.30.0", "2.1.0", "1.31.0"] {
        assert!(reason.contains(part), "{part}: {reason}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_record_of_another_format_is_refused_naming_it() {
    let dir = complete("format-record");
    let path = dir.join(RECEIPT_FILE);
    write_provisioning_record(&dir, "2.0.0", "1.30.0");
    let mut record = read_record(&dir);
    record["format"] = 7.into();
    std::fs::write(&path, record.to_string()).unwrap();
    let before = std::fs::read(&path).unwrap();
    let reason = refusal(fast(&dir));
    assert!(reason.contains(&path.display().to_string()), "{reason}");
    assert!(reason.contains("format 7"), "the format found: {reason}");
    assert!(reason.contains("format 1"), "the format read: {reason}");
    assert!(reason.contains("remove it"), "how to recover: {reason}");

    // Also when the rest of it does not have the shape this build knows.
    std::fs::write(&path, r#"{"format": 9, "anything": true}"#).unwrap();
    let reason = refusal(fast(&dir));
    assert!(reason.contains("format 9"), "{reason}");
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        r#"{"format": 9, "anything": true}"#
    );
    drop(before);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A record that cannot be read is treated as no record: the files decide. Every file is
/// hashed, and when all are right a new record replaces the unreadable one, so the user
/// has nothing to repair by hand.
#[test]
fn a_record_that_cannot_be_read_is_treated_as_no_record_and_replaced() {
    for (name, content) in [
        ("not-json", "{ not json"),
        ("empty", ""),
        ("not-a-record", r#"{"format": 1, "release": "2.0.0"}"#),
        ("a-list", "[1, 2, 3]"),
    ] {
        let dir = complete(&format!("unreadable-{name}"));
        let path = dir.join(RECEIPT_FILE);
        std::fs::write(&path, content).unwrap();

        let checked = fast(&dir).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(checked.hashed, ALL, "{name}: nothing is trusted");
        let RecordState::Replaced { reason } = &checked.record else {
            panic!("{name}: {:?}", checked.record);
        };
        assert!(reason.contains(&path.display().to_string()), "{reason}");
        assert!(
            checked.record.describe().contains("new record"),
            "{}",
            checked.record.describe()
        );
        // The new record is the check's, and the next start is fast.
        assert_eq!(read_record(&dir)["written_by"], "check");
        let again = fast(&dir).unwrap();
        assert!(again.hashed.is_empty(), "{name}: {:?}", again.hashed);
        assert_eq!(again.record, RecordState::Check);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // With a wrong file the directory is refused, and the unreadable record stays.
    let dir = complete("unreadable-and-wrong");
    let path = dir.join(RECEIPT_FILE);
    std::fs::write(&path, "{ not json").unwrap();
    std::fs::write(dir.join("provider.dll"), ALTERED).unwrap();
    let reason = refusal(fast(&dir));
    assert!(reason.contains(&sha256_hex(ALTERED)), "{reason}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ not json");
    std::fs::remove_dir_all(&dir).unwrap();

    // A folder in the record's place cannot be read or replaced: the check passes, hashes
    // at every start and says both.
    let dir = complete("unreadable-folder");
    std::fs::create_dir(dir.join(RECEIPT_FILE)).unwrap();
    let checked = fast(&dir).unwrap();
    assert_eq!(checked.hashed, ALL);
    assert!(
        matches!(checked.record, RecordState::NotWritten { .. }),
        "{:?}",
        checked.record
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

// --- The record is bound to the manifest -------------------------------------------------

/// The same release from two sources (AMD's wheel and AMD's NuGet package hold different
/// builds of the same files, of the same sizes). A directory provisioned and recorded
/// from the other source is not trusted by a build that carries this manifest: the
/// record's hashes are not the manifest's, so the files are hashed, and refused.
#[test]
fn a_record_whose_hashes_belong_to_another_source_does_not_make_files_trusted() {
    let dir = scratch("other-source");
    // The other source's files: same names, same sizes, other content.
    let other: Vec<(&str, Vec<u8>)> = FILES
        .iter()
        .map(|(name, content, _)| (*name, content.to_ascii_uppercase()))
        .collect();
    for (name, content) in &other {
        std::fs::write(dir.join(name), content).unwrap();
    }
    // Its record is right about them in every respect: release, sizes, times, hashes.
    let files: Vec<serde_json::Value> = other
        .iter()
        .map(|(name, content)| {
            serde_json::json!({
                "name": name,
                "bytes": content.len(),
                "sha256": sha256_hex(content),
                "modified_unix_ns": modified_ns(&dir.join(name)),
            })
        })
        .collect();
    let record = serde_json::json!({
        "format": 1, "release": "2.0.0", "onnxruntime": "1.30.0",
        "written_by": "provisioning",
        "source": { "kind": "nuget", "package": "OtherPackage" },
        "files": files,
    });
    std::fs::write(dir.join(RECEIPT_FILE), record.to_string()).unwrap();
    let before = std::fs::read(dir.join(RECEIPT_FILE)).unwrap();

    let reason = refusal(fast(&dir));
    for ((name, content), file) in other.iter().zip(&manifest().files) {
        assert!(
            reason.contains(&dir.join(name).display().to_string()),
            "{reason}"
        );
        assert!(reason.contains(&sha256_hex(content)), "found: {reason}");
        assert!(reason.contains(&file.sha256), "expected: {reason}");
    }
    assert_eq!(std::fs::read(dir.join(RECEIPT_FILE)).unwrap(), before);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A record without sizes and hashes, or with other ones, over files that are right: the
/// files are hashed once, found right, and the record gets the manifest's values.
#[test]
fn a_record_without_the_manifests_hashes_is_not_trusted_and_is_brought_up_to_date() {
    for strip in [true, false] {
        let dir = complete(if strip {
            "record-no-hashes"
        } else {
            "record-other-hashes"
        });
        write_provisioning_record(&dir, "2.0.0", "1.30.0");
        let mut record = read_record(&dir);
        for entry in record["files"].as_array_mut().unwrap() {
            let entry = entry.as_object_mut().unwrap();
            if strip {
                entry.remove("sha256");
                entry.remove("bytes");
            } else {
                entry.insert("sha256".into(), "0".repeat(64).into());
            }
        }
        std::fs::write(dir.join(RECEIPT_FILE), record.to_string()).unwrap();

        let checked = fast(&dir).unwrap();
        assert_eq!(checked.hashed, ALL, "strip {strip}");
        assert_eq!(
            checked.refresh,
            RecordRefresh::Refreshed {
                files: ALL.iter().map(|name| name.to_string()).collect()
            }
        );
        let after = read_record(&dir);
        for (entry, file) in after["files"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&manifest().files)
        {
            assert_eq!(entry["sha256"], file.sha256);
            assert_eq!(entry["bytes"], file.bytes);
        }
        assert_eq!(after["written_by"], "provisioning");
        assert!(fast(&dir).unwrap().hashed.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// Temporary record files left by an interrupted write are removed by the next write
/// that succeeds; a young one (a write that may be under way) and other files are left.
#[test]
fn leftover_partial_record_files_are_removed_by_the_next_successful_write() {
    let dir = complete("leftovers");
    let hour_ago = std::time::SystemTime::now() - Duration::from_secs(3600);
    let old_names = [
        format!(".{RECEIPT_FILE}.4242.partial"),
        format!(".{RECEIPT_FILE}.partial"),
    ];
    for name in &old_names {
        std::fs::write(dir.join(name), "{").unwrap();
        set_modified(&dir.join(name), hour_ago);
    }
    let young = dir.join(format!(".{RECEIPT_FILE}.4343.partial"));
    std::fs::write(&young, "{").unwrap();
    let unrelated = dir.join(".onnxruntime.dll.partial");
    std::fs::write(&unrelated, "half a file of the provisioning step").unwrap();
    set_modified(&unrelated, hour_ago);

    assert_eq!(fast(&dir).unwrap().record, RecordState::Written);
    for name in &old_names {
        assert!(!dir.join(name).exists(), "{name} is still there");
    }
    assert!(
        young.exists(),
        "a write that may be under way is left alone"
    );
    assert!(unrelated.exists(), "only record files are removed");
    assert!(dir.join(RECEIPT_FILE).is_file());
    std::fs::remove_dir_all(&dir).unwrap();
}

// --- Missing and wrong files ------------------------------------------------------------

#[test]
fn a_missing_file_is_named_with_its_expected_path() {
    let dir = complete("one-missing");
    std::fs::remove_file(dir.join("dispatch.dll")).unwrap();

    for depth in [CheckDepth::Fast, CheckDepth::Full] {
        let error = RuntimeFiles::check_against(&manifest(), &dir, depth).unwrap_err();
        let L2dError::Missing { item, expected_at } = &error else {
            panic!("expected Missing, got {error:?}");
        };
        assert_eq!(expected_at, &dir.join("dispatch.dll"));
        assert!(item.contains("dispatch.dll"), "{item}");
        assert!(item.contains("2.0.0"), "the release is named: {item}");
        assert_eq!(error.exit_code(), l2d_core::EXIT_MISSING);
        assert!(!dir.join(RECEIPT_FILE).exists());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn every_missing_file_is_named_with_its_expected_path() {
    let dir = complete("two-missing");
    std::fs::remove_file(dir.join("provider.dll")).unwrap();
    std::fs::remove_file(dir.join("dispatch.dll")).unwrap();

    let error = fast(&dir).unwrap_err();
    let L2dError::Missing { item, expected_at } = &error else {
        panic!("expected Missing, got {error:?}");
    };
    // The first in manifest order is the error's location; the message names both paths.
    assert_eq!(expected_at, &dir.join("provider.dll"));
    let message = error.to_string();
    assert!(
        message.contains(&dir.join("provider.dll").display().to_string()),
        "{message}"
    );
    assert!(
        message.contains(&dir.join("dispatch.dll").display().to_string()),
        "{message}"
    );
    assert!(item.contains("2 of 3"), "{item}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_missing_directory_is_named() {
    let dir = scratch("no-directory").join("ryzenai");
    let error = fast(&dir).unwrap_err();
    let L2dError::Missing { item, expected_at } = &error else {
        panic!("expected Missing, got {error:?}");
    };
    assert_eq!(expected_at, &dir);
    assert!(item.contains("runtime directory"), "{item}");
    std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
}

#[test]
fn a_folder_named_like_a_required_file_is_refused_as_not_a_file() {
    let dir = complete("folder");
    std::fs::remove_file(dir.join("dispatch.dll")).unwrap();
    std::fs::create_dir(dir.join("dispatch.dll")).unwrap();
    let reason = refusal(fast(&dir));
    assert!(
        reason.contains(&format!(
            "{} is not a file",
            dir.join("dispatch.dll").display()
        )),
        "{reason}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_file_of_the_wrong_size_is_refused_with_both_sizes() {
    let dir = complete("wrong-size");
    std::fs::write(dir.join("provider.dll"), b"another build").unwrap();

    let reason = refusal(fast(&dir));
    assert!(
        reason.contains(&dir.join("provider.dll").display().to_string()),
        "{reason}"
    );
    assert!(reason.contains("13 bytes"), "found size: {reason}");
    assert!(reason.contains("27 bytes"), "expected size: {reason}");
    assert!(reason.contains("2.0.0"), "the expected release: {reason}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_unexpected_dll_is_refused_and_other_files_are_ignored() {
    let dir = complete("extra");
    // Not DLLs: the record, notes, anything else.
    std::fs::write(dir.join("notes.txt"), "kept").unwrap();
    std::fs::create_dir(dir.join("subfolder")).unwrap();
    fast(&dir).expect("files that are not DLLs do not matter");

    std::fs::write(dir.join("Stray_Provider.DLL"), b"from another release").unwrap();
    let reason = refusal(fast(&dir));
    assert!(
        reason.contains(&dir.join("Stray_Provider.DLL").display().to_string()),
        "{reason}"
    );
    assert!(reason.contains("not part of"), "{reason}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_and_wrong_files_together_are_one_refusal_that_names_all() {
    let dir = complete("mixed");
    std::fs::remove_file(dir.join("dispatch.dll")).unwrap();
    std::fs::write(dir.join("provider.dll"), b"another build").unwrap();

    let reason = refusal(fast(&dir));
    assert!(
        reason.contains(&dir.join("dispatch.dll").display().to_string()),
        "{reason}"
    );
    assert!(
        reason.contains(&dir.join("provider.dll").display().to_string()),
        "{reason}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

// --- The manifest -----------------------------------------------------------------------

#[test]
fn the_library_is_the_first_file_of_the_manifest_checked_against() {
    let dir = complete("library");
    assert_eq!(fast(&dir).unwrap().library, "onnxruntime.dll");
    assert_eq!(
        fast(&dir).unwrap().onnxruntime_dll(),
        dir.join("onnxruntime.dll")
    );

    let mut reordered = manifest();
    reordered.files.swap(0, 1);
    let checked = RuntimeFiles::check_against(&reordered, &dir, CheckDepth::Fast).unwrap();
    assert_eq!(checked.library, "provider.dll");
    assert_eq!(checked.onnxruntime_dll(), dir.join("provider.dll"));

    let mut empty = manifest();
    empty.files.clear();
    assert!(matches!(
        RuntimeFiles::check_against(&empty, &dir, CheckDepth::Fast),
        Err(L2dError::InvalidInput { .. })
    ));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_manifest_is_validated_when_parsed() {
    let good = serde_json::to_string(&manifest()).unwrap();
    assert_eq!(RuntimeManifest::parse(&good).unwrap(), manifest());

    let mut other_format = manifest();
    other_format.format = 2;
    let mut no_files = manifest();
    no_files.files.clear();
    let mut path_name = manifest();
    path_name.files[0].name = "..\\onnxruntime.dll".into();
    let mut bad_hash = manifest();
    bad_hash.files[0].sha256 = "ABC".into();
    let mut twice = manifest();
    twice.files[1].name = "ONNXRUNTIME.DLL".into();
    for broken in [other_format, no_files, path_name, bad_hash, twice] {
        let text = serde_json::to_string(&broken).unwrap();
        let error = RuntimeManifest::parse(&text).unwrap_err();
        assert!(
            matches!(error, L2dError::InvalidInput { .. }),
            "{text}: {error:?}"
        );
    }
    assert!(matches!(
        RuntimeManifest::parse("{"),
        Err(L2dError::InvalidInput { .. })
    ));
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn tracked_manifest(release: &str) -> RuntimeManifest {
    let path = workspace_root()
        .join("runtime")
        .join(format!("ryzenai-{release}.manifest.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    RuntimeManifest::parse(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// The release the tool is built for is one choice, and everything that names it agrees:
/// the feature of this crate, the manifest compiled in, the API level of `ort`, and
/// `runtime/README.md`.
#[test]
fn the_built_release_is_one_choice_that_everything_agrees_on() {
    let expected = RuntimeManifest::expected();
    let by_feature = if cfg!(feature = "ryzenai-1-9") {
        "1.9.0"
    } else {
        "1.8.0"
    };
    assert_eq!(expected.release, by_feature);
    assert_eq!(&tracked_manifest(by_feature), expected);

    // `ort` asks the runtime for the API level of the release's ONNX Runtime (1.NN.x).
    let minor: u32 = expected
        .onnxruntime
        .split('.')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(
        ort::MINOR_VERSION,
        minor,
        "the `ort` API level and ONNX Runtime {} of AMD Ryzen AI {}",
        expected.onnxruntime,
        expected.release
    );

    let readme =
        std::fs::read_to_string(workspace_root().join("runtime").join("README.md")).unwrap();
    let stated: Vec<&str> = readme
        .lines()
        .filter(|line| line.starts_with("The tool is built for **AMD Ryzen AI "))
        .collect();
    assert_eq!(
        stated,
        vec![
            format!(
                "The tool is built for **AMD Ryzen AI {}** (ONNX Runtime {}).",
                expected.release, expected.onnxruntime
            )
            .as_str()
        ]
    );
}

/// Both releases have a tracked manifest, and the tables of `runtime/README.md` say the
/// same as the manifests.
#[test]
fn the_tracked_manifests_agree_with_the_tables_of_the_readme() {
    let readme =
        std::fs::read_to_string(workspace_root().join("runtime").join("README.md")).unwrap();
    for (release, onnxruntime, kind, count, total) in [
        ("1.9.0", "1.30.0", "wheel", 5, 713_594_768u64),
        ("1.8.0", "1.27.0", "nuget", 6, 451_655_648u64),
    ] {
        let manifest = tracked_manifest(release);
        assert_eq!(manifest.release, release);
        assert_eq!(manifest.onnxruntime, onnxruntime);
        assert_eq!(manifest.source.kind, kind);
        assert_eq!(manifest.files.len(), count);
        assert_eq!(manifest.files[0].name, "onnxruntime.dll");
        assert_eq!(manifest.total_bytes(), total);

        // The section of the release: from its heading to the next heading.
        let heading = format!("### AMD Ryzen AI {release}");
        let section: Vec<&str> = readme
            .lines()
            .skip_while(|line| !line.starts_with(&heading))
            .skip(1)
            .take_while(|line| !line.starts_with('#'))
            .collect();
        assert!(!section.is_empty(), "no section {heading:?} in the README");
        // Rows of the table: | `name` | bytes | version | `sha256` |
        let rows: Vec<(String, u64, String, String)> = section
            .iter()
            .filter(|line| line.starts_with("| `") && line.contains(".dll`"))
            .map(|line| {
                let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
                assert_eq!(cells.len(), 4, "{line}");
                (
                    cells[0].trim_matches('`').to_string(),
                    cells[1].replace(',', "").parse().unwrap(),
                    cells[2].to_string(),
                    cells[3].trim_matches('`').to_string(),
                )
            })
            .collect();
        let manifest_rows: Vec<(String, u64, String, String)> = manifest
            .files
            .iter()
            .map(|file| {
                (
                    file.name.clone(),
                    file.bytes,
                    file.version.clone(),
                    file.sha256.clone(),
                )
            })
            .collect();
        assert_eq!(rows, manifest_rows, "{release}");
    }
    let wheel = tracked_manifest("1.9.0");
    assert_eq!(wheel.source.package, "onnxruntime-vitisai");
    assert!(wheel.source.detail["index"].contains("/1.9.0/"));
    let nuget = tracked_manifest("1.8.0");
    assert_eq!(nuget.source.package, "RyzenAI_Deployment");
    assert_eq!(
        nuget.source.detail["nupkg"],
        "RyzenAI_Deployment.1.8.0.nupkg"
    );
    assert_eq!(nuget.source.detail["nupkg_sha256"].len(), 64);
}

/// A runtime directory of the other release is refused, and both releases are named: by
/// its record, and without a record by its files.
#[test]
fn a_directory_of_the_other_release_is_refused_naming_both() {
    let built = RuntimeManifest::expected();
    let other = tracked_manifest(if built.release == "1.9.0" {
        "1.8.0"
    } else {
        "1.9.0"
    });
    // The files of the other release, by name and size, without a record. Their content
    // does not matter: nothing of that size is read, except where the two releases have
    // a file of the same size, and that one is then refused by its hash.
    let dir = scratch("other-release-files");
    for file in &other.files {
        let handle = std::fs::File::create(dir.join(&file.name)).unwrap();
        handle.set_len(file.bytes).unwrap();
    }
    let reason = refusal(RuntimeFiles::check(&dir, CheckDepth::Fast));
    // Both releases are named although no record says which one the directory holds.
    for part in [
        &built.release,
        &built.onnxruntime,
        &other.release,
        &other.onnxruntime,
    ] {
        assert!(reason.contains(part.as_str()), "{part}: {reason}");
    }
    assert!(
        reason.contains(&format!(
            "{} of the {} files of AMD Ryzen AI {}",
            other.files.len(),
            other.files.len(),
            other.release
        )),
        "{reason}"
    );
    assert!(reason.contains("onnxruntime.dll"), "{reason}");
    assert!(!dir.join(RECEIPT_FILE).exists());

    // With the record a check built for the other release wrote, or provisioning did.
    std::fs::write(
        dir.join(RECEIPT_FILE),
        serde_json::json!({
            "format": 1, "release": other.release, "onnxruntime": other.onnxruntime,
            "written_by": "check", "files": [],
        })
        .to_string(),
    )
    .unwrap();
    let reason = refusal(RuntimeFiles::check(&dir, CheckDepth::Fast));
    for part in [
        &built.release,
        &built.onnxruntime,
        &other.release,
        &other.onnxruntime,
    ] {
        assert!(reason.contains(part.as_str()), "{part}: {reason}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
// --- File versions (Windows) ------------------------------------------------------------

/// The version of a real DLL is read, and a refusal names it: on a copy of a system
/// library, which is only read.
#[cfg(windows)]
#[test]
fn the_file_version_of_a_wrong_dll_is_named_in_the_refusal() {
    use l2d_encoder::runtime::file_version;

    let system = PathBuf::from(std::env::var_os("SystemRoot").expect("SystemRoot"))
        .join("System32")
        .join("version.dll");
    let dir = complete("file-version");
    let copy = dir.join("provider.dll");
    std::fs::copy(&system, &copy).unwrap();

    let version = file_version(&copy).expect("a system DLL has a version resource");
    let parts: Vec<&str> = version.split('.').collect();
    assert!(
        (3..=4).contains(&parts.len()) && parts.iter().all(|part| part.parse::<u32>().is_ok()),
        "{version}"
    );
    // The copy is read, not the file in `System32`: for files there Windows reports
    // another major and minor version to a program that does not declare the Windows
    // version it was built for (seen: 6.2.26100.1150 for the original, 10.0.26100.1150
    // for its copy).
    assert!(parts[0] == "10" || parts[0] == "6", "{version}");
    // Text is not a DLL, and a missing file has no version.
    assert_eq!(file_version(&dir.join("onnxruntime.dll")), None);
    assert_eq!(file_version(&dir.join("absent.dll")), None);

    // A file of another release: its version against the manifest's.
    let reason = refusal(fast(&dir));
    assert!(
        reason.contains(&format!(
            "it reports file version {version}, the release has 2.0.0: a file of another release"
        )),
        "{reason}"
    );
    // The same version as the release's: a damaged file, not another release.
    let mut same_version = manifest();
    same_version.files[1].version = version.clone();
    let reason = refusal(RuntimeFiles::check_against(
        &same_version,
        &dir,
        CheckDepth::Fast,
    ));
    assert!(
        reason.contains(&format!(
            "it reports file version {version}, as the release does"
        )),
        "{reason}"
    );
    // A stray DLL is named with its version too.
    std::fs::write(&copy, FILES[1].1).unwrap();
    std::fs::copy(&system, dir.join("stray.dll")).unwrap();
    let reason = refusal(fast(&dir));
    assert!(
        reason.contains(&format!("it reports file version {version}")),
        "{reason}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

// --- On request: the real directory -----------------------------------------------------

/// The real runtime directory (`L2D_RUNTIME_DIR`, default `runtime/ryzenai/` of the
/// workspace), with the time the checks take. In a directory without a record the first
/// check hashes everything and writes the record: the directory is written to.
///
/// ```text
/// cargo test -p l2d-encoder --test runtime_files -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs the AMD runtime files and may write the record; run on request"]
fn the_real_runtime_directory_passes_and_the_checks_are_timed() {
    let dir = ProjectPaths::new(workspace_root())
        .runtime_files_dir(l2d_core::paths::runtime_dir_from_environment().as_deref());
    println!(
        "runtime_files: directory {} (record present: {})",
        dir.display(),
        dir.join(RECEIPT_FILE).exists()
    );

    let mut with_record = Vec::new();
    for _ in 0..6 {
        let started = Instant::now();
        let checked = RuntimeFiles::check(&dir, CheckDepth::Fast)
            .unwrap_or_else(|error| panic!("fast check: {error}"));
        let elapsed = started.elapsed();
        assert_eq!(checked.release, RuntimeManifest::expected().release);
        println!(
            "runtime_files: fast check {elapsed:?}, hashed {}, record: {}",
            checked.hashed.len(),
            checked.record.describe()
        );
        if matches!(
            checked.record,
            RecordState::Check | RecordState::Provisioning
        ) {
            with_record.push(elapsed);
        }
    }
    let started = Instant::now();
    let checked = RuntimeFiles::check(&dir, CheckDepth::Full)
        .unwrap_or_else(|error| panic!("full check: {error}"));
    let full = started.elapsed();
    assert_eq!(
        checked.hashed.len(),
        RuntimeManifest::expected().files.len()
    );
    let megabytes = RuntimeManifest::expected().total_bytes() as f64 / 1e6;
    println!(
        "runtime_files: full check {full:?} over {megabytes:.1} MB ({:.0} MB/s), release {} \
         (ONNX Runtime {})",
        megabytes / full.as_secs_f64(),
        checked.release,
        checked.onnxruntime
    );
    println!(
        "runtime_files: fast check with a record, lowest and highest of {}: {:?} and {:?}",
        with_record.len(),
        with_record.iter().min(),
        with_record.iter().max()
    );
    #[cfg(windows)]
    for file in &RuntimeManifest::expected().files {
        let reported = l2d_encoder::runtime::file_version(&dir.join(&file.name));
        println!(
            "runtime_files: {} reports file version {reported:?}, the manifest says {}",
            file.name, file.version
        );
        assert_eq!(reported.as_deref(), Some(file.version.as_str()));
    }
}
