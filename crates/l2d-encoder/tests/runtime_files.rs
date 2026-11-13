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

    // Bytes that are not text are content that is not a record, too.
    let dir = complete("unreadable-binary");
    std::fs::write(dir.join(RECEIPT_FILE), [0xff, 0xfe, 0x00, 0x80]).unwrap();
    assert!(matches!(
        fast(&dir).unwrap().record,
        RecordState::Replaced { .. }
    ));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A record that is there but cannot be opened for the moment is not a record that
/// cannot be read: it may be a good one that another process has open. Nothing is
/// trusted for this start, and the record is neither replaced nor rewritten.
#[test]
fn a_record_that_cannot_be_opened_is_left_as_it_is() {
    // A folder in the record's place cannot be opened as a file.
    let dir = complete("unavailable-folder");
    std::fs::create_dir(dir.join(RECEIPT_FILE)).unwrap();
    for _ in 0..2 {
        let checked = fast(&dir).unwrap();
        assert_eq!(checked.hashed, ALL);
        let RecordState::Unavailable { reason } = &checked.record else {
            panic!("{:?}", checked.record);
        };
        assert!(reason.contains(RECEIPT_FILE), "{reason}");
        assert!(checked.record.describe().contains("left as it is"));
        assert!(dir.join(RECEIPT_FILE).is_dir());
    }
    std::fs::remove_dir_all(&dir).unwrap();

    // A record another process holds open without sharing it (Windows): the check
    // passes, hashes everything, and the record is byte for byte what it was; once it is
    // free again it is used as before.
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;

        let dir = complete("unavailable-locked");
        write_provisioning_record(&dir, "2.0.0", "1.30.0");
        let path = dir.join(RECEIPT_FILE);
        let before = std::fs::read(&path).unwrap();
        // Touched, so that with a readable record this file would be re-verified and
        // the record rewritten.
        set_modified(
            &dir.join("provider.dll"),
            std::time::SystemTime::now() + Duration::from_secs(300),
        );
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(&path)
            .unwrap();
        let checked = fast(&dir).unwrap();
        assert_eq!(checked.hashed, ALL);
        assert!(
            matches!(checked.record, RecordState::Unavailable { .. }),
            "{:?}",
            checked.record
        );
        assert_eq!(checked.refresh, RecordRefresh::NotNeeded);
        drop(lock);
        assert_eq!(std::fs::read(&path).unwrap(), before);

        let checked = fast(&dir).unwrap();
        assert_eq!(checked.record, RecordState::Provisioning);
        assert_eq!(checked.hashed, vec!["provider.dll"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
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
/// files are hashed once and found right. A record that lacked the values gets them and
/// stays what it was; a record that held other hashes described other content, so its
/// source is not kept: a record of the check replaces it.
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
        let files: Vec<String> = ALL.iter().map(|name| name.to_string()).collect();
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
        if strip {
            assert_eq!(checked.refresh, RecordRefresh::Refreshed { files });
            assert_eq!(after["written_by"], "provisioning");
            assert!(after.get("source").is_some());
        } else {
            assert_eq!(checked.refresh, RecordRefresh::Rewritten { files });
            assert!(checked.refresh.describe().contains("another source"));
            assert_eq!(after["written_by"], "check");
            assert!(after.get("source").is_none(), "{after}");
        }
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
    let mut broken_names = Vec::new();
    // Names that are not plainly a file of that name in the directory: a stream of
    // another file, names Windows changes (trailing dot or space), device names with or
    // without an extension, and characters that have a meaning in a path.
    for name in [
        "second.dll:stream",
        "onnxruntime.dll::$DATA",
        "C:onnxruntime.dll",
        "trailing.dll.",
        "trailing.dll ",
        "CON",
        "con.dll",
        "NUL.dll",
        "COM1.dll",
        "lpt9.txt.dll",
        "sub/onnxruntime.dll",
        "what?.dll",
        "star*.dll",
        "pipe|.dll",
        "tab\t.dll",
        ".",
        "..",
        "",
    ] {
        assert!(!l2d_encoder::runtime::is_plain_file_name(name), "{name:?}");
        let mut named = manifest();
        named.files[0].name = name.into();
        broken_names.push(named);
    }
    for name in [
        "onnxruntime.dll",
        "onnxruntime-genai.dll",
        "a b.dll",
        "CONSOLE.dll",
        "x.y.z",
    ] {
        assert!(l2d_encoder::runtime::is_plain_file_name(name), "{name:?}");
    }
    for broken in [other_format, no_files, path_name, bad_hash, twice]
        .into_iter()
        .chain(broken_names)
    {
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

/// The tracked manifest of a release from a source: `runtime/ryzenai-<release>.<kind>.manifest.json`.
fn tracked_manifest(release: &str, kind: &str) -> RuntimeManifest {
    let path = workspace_root()
        .join("runtime")
        .join(format!("ryzenai-{release}.{kind}.manifest.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    RuntimeManifest::parse(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// The release the tool is built for is AMD Ryzen AI 1.9.0, and everything that names it
/// agrees: the feature of this crate, the manifests compiled in, the tracked manifests,
/// the API level of `ort`, and `runtime/README.md`.
#[test]
fn the_built_release_is_one_choice_that_everything_agrees_on() {
    let expected = RuntimeManifest::expected();
    // The feature `ryzenai-1-9` is on, or the crate would not have compiled.
    assert_eq!(expected.release, "1.9.0");
    // The manifests a directory may match: one per source of the built release.
    let built = vec![
        tracked_manifest("1.9.0", "wheel"),
        tracked_manifest("1.9.0", "nuget"),
    ];
    assert_eq!(RuntimeManifest::built(), built.as_slice());
    // No manifest is tracked that the tool does not compile in.
    let mut tracked: Vec<String> = std::fs::read_dir(workspace_root().join("runtime"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".manifest.json"))
        .collect();
    tracked.sort();
    assert_eq!(
        tracked,
        [
            "ryzenai-1.9.0.nuget.manifest.json",
            "ryzenai-1.9.0.wheel.manifest.json"
        ]
    );
    assert_eq!(expected, &built[0]);
    // The sources of a release differ in the hashes only: same release, same names, same
    // sizes, same file versions (the builds are the same, signed separately).
    for manifest in RuntimeManifest::built() {
        assert_eq!(manifest.release, expected.release);
        assert_eq!(manifest.onnxruntime, expected.onnxruntime);
        let shape = |manifest: &RuntimeManifest| -> Vec<(String, u64, String)> {
            manifest
                .files
                .iter()
                .map(|file| (file.name.clone(), file.bytes, file.version.clone()))
                .collect()
        };
        assert_eq!(shape(manifest), shape(expected));
    }
    let kinds: Vec<&str> = RuntimeManifest::built()
        .iter()
        .map(|manifest| manifest.source.kind.as_str())
        .collect();
    let mut distinct = kinds.clone();
    distinct.dedup();
    assert_eq!(kinds, distinct, "one manifest per source");

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

/// Every tracked manifest has its table in `runtime/README.md`, and the tables say the
/// same as the manifests.
#[test]
fn the_tracked_manifests_agree_with_the_tables_of_the_readme() {
    let readme =
        std::fs::read_to_string(workspace_root().join("runtime").join("README.md")).unwrap();
    for (release, onnxruntime, kind, count, total) in [
        ("1.9.0", "1.30.0", "wheel", 5, 713_594_768u64),
        ("1.9.0", "1.30.0", "nuget", 5, 713_594_768u64),
    ] {
        let manifest = tracked_manifest(release, kind);
        assert_eq!(manifest.release, release);
        assert_eq!(manifest.onnxruntime, onnxruntime);
        assert_eq!(manifest.source.kind, kind);
        assert_eq!(manifest.files.len(), count);
        assert_eq!(manifest.files[0].name, "onnxruntime.dll");
        assert_eq!(manifest.total_bytes(), total);

        // The section of the release: from its heading to the next heading.
        let heading = format!("### AMD Ryzen AI {release}, {kind}");
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
        assert_eq!(rows, manifest_rows, "{release} {kind}");
    }
    let wheel = tracked_manifest("1.9.0", "wheel");
    assert_eq!(wheel.source.package, "onnxruntime-vitisai");
    assert!(wheel.source.detail["index"].contains("/1.9.0/"));
    let nuget = tracked_manifest("1.9.0", "nuget");
    assert_eq!(nuget.source.package, "RyzenAI_Deployment");
    assert_eq!(nuget.source.version, "1.9.0");
    assert_eq!(
        nuget.source.detail["nupkg"],
        "RyzenAI_Deployment.1.9.0.nupkg"
    );
    assert_eq!(nuget.source.detail["nupkg_sha256"].len(), 64);
    assert_eq!(nuget.source.detail["nupkg_bytes"], "289269284");
    // The two sources of 1.9.0 have no hash in common.
    for (a, b) in wheel.files.iter().zip(&nuget.files) {
        assert_ne!(a.sha256, b.sha256, "{}", a.name);
    }
}

/// A runtime directory of another release is refused by the real check. The tool has the
/// manifests of its own release only, so what it can name depends on the directory:
/// without a record, the built release and every file that is not that release's (its
/// size against the expected one, or its hash, and each DLL the release does not have);
/// with a record, both releases.
#[test]
fn a_directory_of_another_release_is_refused() {
    let built = RuntimeManifest::expected();
    // The names and sizes of the six files of AMD Ryzen AI 1.8.0 (ONNX Runtime 1.27.0),
    // as a directory provisioned for it holds them. Their content does not matter:
    // nothing of a wrong size is read; the one file that has the size of the built
    // release's is refused by its hash.
    let other_files: [(&str, u64); 6] = [
        ("onnxruntime.dll", 24_030_800),
        ("onnxruntime_providers_shared.dll", 117_840),
        ("onnxruntime_providers_vitisai.dll", 465_488),
        ("onnxruntime_vitisai_ep.dll", 119_166_032),
        ("dyn_dispatch_core.dll", 158_389_328),
        ("aiecompiler_client.dll", 149_486_160),
    ];
    let dir = scratch("other-release-files");
    for (name, bytes) in other_files {
        let handle = std::fs::File::create(dir.join(name)).unwrap();
        handle.set_len(bytes).unwrap();
    }
    let reason = refusal(RuntimeFiles::check(&dir, CheckDepth::Fast));
    assert!(
        reason.contains(&format!(
            "does not hold the files of AMD Ryzen AI {} (ONNX Runtime {})",
            built.release, built.onnxruntime
        )),
        "{reason}"
    );
    for file in &built.files {
        let (_, found) = other_files
            .iter()
            .find(|(name, _)| *name == file.name)
            .unwrap();
        let stated = if *found == file.bytes {
            format!("{} has SHA-256 ", dir.join(&file.name).display())
        } else {
            format!(
                "{} has {found} bytes, the release has {} bytes",
                dir.join(&file.name).display(),
                file.bytes
            )
        };
        assert!(reason.contains(&stated), "{stated}: {reason}");
    }
    assert!(
        reason.contains(&format!(
            "{} is not part of the release",
            dir.join("aiecompiler_client.dll").display()
        )),
        "{reason}"
    );
    assert!(!dir.join(RECEIPT_FILE).exists());

    // With the record a check built for that release wrote, or provisioning did: both
    // releases are named.
    std::fs::write(
        dir.join(RECEIPT_FILE),
        serde_json::json!({
            "format": 1, "release": "1.8.0", "onnxruntime": "1.27.0",
            "written_by": "check", "files": [],
        })
        .to_string(),
    )
    .unwrap();
    let reason = refusal(RuntimeFiles::check(&dir, CheckDepth::Fast));
    for part in [
        built.release.as_str(),
        built.onnxruntime.as_str(),
        "1.8.0",
        "1.27.0",
    ] {
        assert!(reason.contains(part), "{part}: {reason}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
// --- Two sources of one release ----------------------------------------------------------

/// The other source of the test release: the same names and sizes, other content.
fn other_source() -> (RuntimeManifest, Vec<(&'static str, Vec<u8>)>) {
    let files: Vec<(&'static str, Vec<u8>)> = FILES
        .iter()
        .map(|(name, content, _)| (*name, content.to_ascii_uppercase()))
        .collect();
    let mut other = manifest();
    other.source.kind = "nuget".into();
    other.source.package = "RuntimePackage".into();
    other.source.detail = [("nupkg", "RuntimePackage.2.0.0.nupkg")]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    for (file, (_, content)) in other.files.iter_mut().zip(&files) {
        file.sha256 = sha256_hex(content);
    }
    (other, files)
}

fn among(dir: &Path) -> Result<RuntimeFiles, L2dError> {
    RuntimeFiles::check_among(&[manifest(), other_source().0], dir, CheckDepth::Fast)
}

/// A directory that matches either manifest of the release passes, and the check says
/// which source it matched; the record holds that source's hashes, and the next start
/// hashes nothing.
#[test]
fn a_directory_of_either_source_passes_and_the_source_is_reported() {
    let wheel = complete("source-wheel");
    let first = among(&wheel).unwrap();
    assert_eq!(first.source, "wheel");
    assert_eq!(first.hashed, ALL);
    assert_eq!(first.record, RecordState::Written);
    assert_eq!(read_record(&wheel)["matched_source"], "wheel");
    let again = among(&wheel).unwrap();
    assert_eq!(again.source, "wheel");
    assert!(again.hashed.is_empty(), "{:?}", again.hashed);

    let (other, files) = other_source();
    let nuget = scratch("source-nuget");
    for (name, content) in &files {
        std::fs::write(nuget.join(name), content).unwrap();
    }
    let first = among(&nuget).unwrap();
    assert_eq!(first.source, "nuget");
    assert_eq!(
        first.hashed, ALL,
        "each file is hashed once, not once per manifest"
    );
    assert_eq!(first.record, RecordState::Written);
    let record = read_record(&nuget);
    assert_eq!(record["matched_source"], "nuget");
    for (entry, file) in record["files"].as_array().unwrap().iter().zip(&other.files) {
        assert_eq!(entry["sha256"], file.sha256);
    }
    let again = among(&nuget).unwrap();
    assert_eq!(again.source, "nuget");
    assert!(again.hashed.is_empty(), "{:?}", again.hashed);
    // The full check passes for it too; against the wheel manifest alone it is refused.
    let full = RuntimeFiles::check_among(&[manifest(), other], &nuget, CheckDepth::Full).unwrap();
    assert_eq!((full.source.as_str(), full.hashed.len()), ("nuget", 3));
    refusal(RuntimeFiles::check_against(
        &manifest(),
        &nuget,
        CheckDepth::Fast,
    ));
    std::fs::remove_dir_all(&wheel).unwrap();
    std::fs::remove_dir_all(&nuget).unwrap();
}

/// A directory with some files of each source matches neither manifest: it is refused,
/// and the refusal says that it is a mix and which file is from which source. With and
/// without a record.
#[test]
fn a_directory_that_mixes_the_two_sources_is_refused_naming_the_mix() {
    let (_, files) = other_source();
    for with_record in [false, true] {
        let dir = complete(if with_record { "mix-record" } else { "mix" });
        if with_record {
            assert_eq!(among(&dir).unwrap().source, "wheel");
        }
        let before = std::fs::read(dir.join(RECEIPT_FILE)).ok();
        std::fs::write(dir.join("provider.dll"), &files[1].1).unwrap();

        let reason = refusal(among(&dir));
        assert!(reason.contains("mixes files of 2 sources"), "{reason}");
        assert!(
            reason.contains("onnxruntime.dll, dispatch.dll from the source `wheel`"),
            "{reason}"
        );
        assert!(
            reason.contains("provider.dll from the source `nuget`"),
            "{reason}"
        );
        assert!(reason.contains("one source"), "{reason}");
        assert!(
            reason.contains(&dir.join("provider.dll").display().to_string()),
            "{reason}"
        );
        // Nothing was written for a directory that is refused.
        assert_eq!(std::fs::read(dir.join(RECEIPT_FILE)).ok(), before);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// A record stays bound to the manifest it was made against. A directory verified as the
/// wheel's whose files are then replaced by the NuGet package's (same sizes) is not
/// trusted through that record: every file is hashed; they are all the other source's,
/// so the directory passes as that source, and the record is replaced by one that says
/// so (its old source would no longer be true). And the other way round.
#[test]
fn a_record_of_one_source_does_not_make_the_other_sources_files_trusted() {
    let (other, other_files) = other_source();
    let own_files: Vec<(&str, Vec<u8>)> = FILES
        .iter()
        .map(|(name, content, _)| (*name, content.to_vec()))
        .collect();
    for (from, to, to_manifest, to_files) in [
        ("wheel", "nuget", &other, &other_files),
        ("nuget", "wheel", &manifest(), &own_files),
    ] {
        let dir = scratch(&format!("bound-{from}-to-{to}"));
        let from_files = if from == "wheel" {
            &own_files
        } else {
            &other_files
        };
        for (name, content) in from_files {
            std::fs::write(dir.join(name), content).unwrap();
        }
        // A provisioning record of the first source, as the script writes it.
        let entries: Vec<serde_json::Value> = from_files
            .iter()
            .map(|(name, content)| {
                serde_json::json!({
                    "name": name, "bytes": content.len(), "sha256": sha256_hex(content),
                    "modified_unix_ns": modified_ns(&dir.join(name)),
                })
            })
            .collect();
        let record = serde_json::json!({
            "format": 1, "release": "2.0.0", "onnxruntime": "1.30.0",
            "written_by": "provisioning", "source": { "kind": from }, "files": entries,
        });
        std::fs::write(dir.join(RECEIPT_FILE), record.to_string()).unwrap();
        let checked = among(&dir).unwrap();
        assert_eq!((checked.source.as_str(), checked.hashed.len()), (from, 0));
        assert_eq!(checked.record, RecordState::Provisioning);

        // The binding: a build that carried only the other source's manifest finds this
        // record, whose times are right, and trusts nothing through it, because the
        // record's hashes are not that manifest's. Every file is hashed and refused.
        let before = std::fs::read(dir.join(RECEIPT_FILE)).unwrap();
        let reason = refusal(RuntimeFiles::check_against(
            to_manifest,
            &dir,
            CheckDepth::Fast,
        ));
        for ((_, content), file) in from_files.iter().zip(&to_manifest.files) {
            assert!(reason.contains(&sha256_hex(content)), "found: {reason}");
            assert!(reason.contains(&file.sha256), "expected: {reason}");
        }
        assert_eq!(std::fs::read(dir.join(RECEIPT_FILE)).unwrap(), before);

        // The files are replaced by the other source's (same sizes).
        for (name, content) in to_files {
            std::fs::write(dir.join(name), content).unwrap();
            set_modified(
                &dir.join(name),
                std::time::SystemTime::now() + Duration::from_secs(300),
            );
        } // Against the first source's manifest alone: refused (the record makes nothing
        // trusted, the hashes are the other source's).
        let first_manifest = if from == "wheel" {
            manifest()
        } else {
            other.clone()
        };
        let reason = refusal(RuntimeFiles::check_against(
            &first_manifest,
            &dir,
            CheckDepth::Full,
        ));
        assert!(reason.contains(&to_manifest.files[0].sha256), "{reason}");

        // Among both: the directory is the other source's, whole.
        let checked = among(&dir).unwrap();
        assert_eq!(checked.source, to);
        assert_eq!(checked.hashed, ALL, "{from} to {to}: nothing was trusted");
        assert_eq!(
            checked.refresh,
            RecordRefresh::Rewritten {
                files: ALL.iter().map(|name| name.to_string()).collect()
            }
        );
        let after = read_record(&dir);
        assert_eq!(after["written_by"], "check");
        assert_eq!(after["matched_source"], to);
        assert!(
            after.get("source").is_none(),
            "the old source is gone: {after}"
        );
        for (entry, file) in after["files"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&to_manifest.files)
        {
            assert_eq!(entry["sha256"], file.sha256);
        }
        let again = among(&dir).unwrap();
        assert_eq!((again.source.as_str(), again.hashed.len()), (to, 0));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn a_check_among_no_manifests_is_an_error() {
    let dir = complete("no-manifests");
    assert!(matches!(
        RuntimeFiles::check_among(&[], &dir, CheckDepth::Fast),
        Err(L2dError::InvalidInput { .. })
    ));
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
            "runtime_files: fast check {elapsed:?}, source `{}`, hashed {}, record: {}",
            checked.source,
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
