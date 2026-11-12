//! `RuntimeFiles::check` against temporary directories (task 2.3, requirements 4.3 and
//! 10.5), and on request against the real runtime directory with timings.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, UNIX_EPOCH};

use l2d_core::{L2dError, ProjectPaths};
use l2d_encoder::runtime::{
    CheckDepth, ManifestFile, ManifestSource, RECEIPT_FILE, RuntimeFiles, RuntimeManifest,
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

fn manifest() -> RuntimeManifest {
    RuntimeManifest {
        format: 1,
        release: "2.0.0".into(),
        onnxruntime: "1.30.0".into(),
        source: ManifestSource {
            index: "https://example.invalid/2.0.0/simple/".into(),
            package: "runtime-package".into(),
            version: "1.30.0".into(),
            wheel: "runtime_package-1.30.0-py3-none-win_amd64.whl".into(),
            directory: "onnxruntime/capi".into(),
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

/// A directory that holds exactly the files of the test release.
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

/// The record the provisioning step leaves in the directory, for the files as they are
/// on disk now.
fn write_receipt(dir: &Path, release: &str, onnxruntime: &str) {
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
    let receipt = serde_json::json!({
        "format": 1,
        "release": release,
        "onnxruntime": onnxruntime,
        "provisioned_utc": "2026-11-12T12:00:00Z",
        "source": { "package": "runtime-package", "version": "1.30.0" },
        "files": files,
    });
    std::fs::write(
        dir.join(RECEIPT_FILE),
        serde_json::to_string_pretty(&receipt).unwrap(),
    )
    .unwrap();
}

fn refusal(result: Result<RuntimeFiles, L2dError>) -> String {
    match result {
        Err(L2dError::Refused { reason }) => reason,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_complete_directory_passes_the_fast_and_the_full_check() {
    let dir = complete("complete");
    let manifest = manifest();

    let fast = RuntimeFiles::check_against(&manifest, &dir, CheckDepth::Fast).unwrap();
    assert_eq!(fast.dir, dir);
    assert_eq!(fast.release, "2.0.0");
    assert_eq!(fast.onnxruntime, "1.30.0");
    assert_eq!(fast.onnxruntime_dll(), dir.join("onnxruntime.dll"));
    assert!(!fast.provisioned, "no provisioning record in the directory");
    assert!(fast.hashed.is_empty(), "the fast check hashes nothing here");

    let full = RuntimeFiles::check_against(&manifest, &dir, CheckDepth::Full).unwrap();
    assert_eq!(
        full.hashed,
        vec!["onnxruntime.dll", "provider.dll", "dispatch.dll"]
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

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
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn every_missing_file_is_named_with_its_expected_path() {
    let dir = complete("two-missing");
    std::fs::remove_file(dir.join("provider.dll")).unwrap();
    std::fs::remove_file(dir.join("dispatch.dll")).unwrap();

    let error = RuntimeFiles::check_against(&manifest(), &dir, CheckDepth::Fast).unwrap_err();
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
    let error = RuntimeFiles::check_against(&manifest(), &dir, CheckDepth::Fast).unwrap_err();
    let L2dError::Missing { item, expected_at } = &error else {
        panic!("expected Missing, got {error:?}");
    };
    assert_eq!(expected_at, &dir);
    assert!(item.contains("runtime directory"), "{item}");
    std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
}

#[test]
fn a_file_of_the_wrong_size_is_refused_with_both_sizes() {
    let dir = complete("wrong-size");
    std::fs::write(dir.join("provider.dll"), b"another build").unwrap();

    let reason = refusal(RuntimeFiles::check_against(
        &manifest(),
        &dir,
        CheckDepth::Fast,
    ));
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
fn a_wrong_hash_is_found_by_the_full_check_only() {
    let dir = complete("wrong-hash");
    // Same length, other content.
    let altered = b"the provider, a LONGER file";
    assert_eq!(altered.len(), FILES[1].1.len());
    std::fs::write(dir.join("provider.dll"), altered).unwrap();
    let manifest = manifest();

    // Without a provisioning record the fast check compares presence and size only.
    RuntimeFiles::check_against(&manifest, &dir, CheckDepth::Fast)
        .expect("the fast check does not hash a directory without a provisioning record");

    let reason = refusal(RuntimeFiles::check_against(
        &manifest,
        &dir,
        CheckDepth::Full,
    ));
    assert!(
        reason.contains(&dir.join("provider.dll").display().to_string()),
        "{reason}"
    );
    assert!(reason.contains(&sha256_hex(altered)), "found: {reason}");
    assert!(
        reason.contains(&manifest.files[1].sha256),
        "expected: {reason}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_directory_provisioned_for_another_release_is_refused_with_both_versions() {
    let dir = complete("other-release");
    write_receipt(&dir, "1.8.0", "1.27.0");

    for depth in [CheckDepth::Fast, CheckDepth::Full] {
        let reason = refusal(RuntimeFiles::check_against(&manifest(), &dir, depth));
        assert!(reason.contains("1.8.0"), "found release: {reason}");
        assert!(reason.contains("1.27.0"), "found runtime: {reason}");
        assert!(reason.contains("2.0.0"), "expected release: {reason}");
        assert!(reason.contains("1.30.0"), "expected runtime: {reason}");
        assert!(reason.contains(&dir.display().to_string()), "{reason}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_unreadable_provisioning_record_is_refused() {
    let dir = complete("bad-receipt");
    std::fs::write(dir.join(RECEIPT_FILE), "{ not json").unwrap();
    let reason = refusal(RuntimeFiles::check_against(
        &manifest(),
        &dir,
        CheckDepth::Fast,
    ));
    assert!(
        reason.contains(&dir.join(RECEIPT_FILE).display().to_string()),
        "{reason}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn with_a_provisioning_record_the_fast_check_hashes_only_files_changed_since() {
    let dir = complete("receipt");
    write_receipt(&dir, "2.0.0", "1.30.0");
    let manifest = manifest();

    // Untouched since provisioning: nothing is hashed.
    let checked = RuntimeFiles::check_against(&manifest, &dir, CheckDepth::Fast).unwrap();
    assert!(checked.provisioned);
    assert!(checked.hashed.is_empty(), "{:?}", checked.hashed);

    // Touched, content unchanged: that file is hashed and passes.
    let target = dir.join("provider.dll");
    let later = std::fs::metadata(&target).unwrap().modified().unwrap() + Duration::from_secs(90);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&target)
        .unwrap();
    file.set_modified(later).unwrap();
    drop(file);
    let checked = RuntimeFiles::check_against(&manifest, &dir, CheckDepth::Fast).unwrap();
    assert_eq!(checked.hashed, vec!["provider.dll"]);

    // Replaced by a file of the same size: the fast check finds it through the hash.
    let altered = b"the provider, a LONGER file";
    std::fs::write(&target, altered).unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&target)
        .unwrap();
    file.set_modified(later + Duration::from_secs(90)).unwrap();
    drop(file);
    let reason = refusal(RuntimeFiles::check_against(
        &manifest,
        &dir,
        CheckDepth::Fast,
    ));
    assert!(reason.contains(&target.display().to_string()), "{reason}");
    assert!(reason.contains(&sha256_hex(altered)), "{reason}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_unexpected_dll_is_refused_and_other_files_are_ignored() {
    let dir = complete("extra");
    // Not DLLs: the provisioning record's neighbours, notes, anything else.
    std::fs::write(dir.join("notes.txt"), "kept").unwrap();
    std::fs::create_dir(dir.join("subfolder")).unwrap();
    RuntimeFiles::check_against(&manifest(), &dir, CheckDepth::Fast)
        .expect("files that are not DLLs do not matter");

    std::fs::write(dir.join("Stray_Provider.DLL"), b"from another release").unwrap();
    let reason = refusal(RuntimeFiles::check_against(
        &manifest(),
        &dir,
        CheckDepth::Fast,
    ));
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

    let reason = refusal(RuntimeFiles::check_against(
        &manifest(),
        &dir,
        CheckDepth::Fast,
    ));
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

/// The manifest compiled into the tool is the tracked file `runtime/ryzenai.manifest.json`,
/// and the table in `runtime/README.md` says the same.
#[test]
fn the_compiled_in_manifest_is_the_tracked_one_and_agrees_with_the_readme() {
    let expected = RuntimeManifest::expected();
    let tracked = std::fs::read_to_string(
        workspace_root()
            .join("runtime")
            .join("ryzenai.manifest.json"),
    )
    .unwrap();
    assert_eq!(&RuntimeManifest::parse(&tracked).unwrap(), expected);

    assert_eq!(expected.release, "1.9.0");
    assert_eq!(expected.onnxruntime, "1.30.0");
    assert_eq!(expected.source.package, "onnxruntime-vitisai");
    assert_eq!(expected.source.version, "1.30.0");
    assert!(
        expected.source.index.contains("/1.9.0/"),
        "{}",
        expected.source.index
    );
    assert_eq!(expected.files.len(), 5);
    assert_eq!(expected.total_bytes(), 713_594_768);

    // Rows of the README table: | `name` | bytes | version | `sha256` |
    let readme =
        std::fs::read_to_string(workspace_root().join("runtime").join("README.md")).unwrap();
    let rows: Vec<(String, u64, String, String)> = readme
        .lines()
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
    let manifest_rows: Vec<(String, u64, String, String)> = expected
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
    assert_eq!(rows, manifest_rows);
}

/// On request: the real runtime directory (`L2D_RUNTIME_DIR`, default `runtime/ryzenai/`
/// of the workspace), with the time the fast and the full check take.
///
/// ```text
/// cargo test -p l2d-encoder --test runtime_files -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs the AMD runtime files (714 MB); run on request"]
fn the_real_runtime_directory_passes_and_the_checks_are_timed() {
    let dir = ProjectPaths::new(workspace_root())
        .runtime_files_dir(l2d_core::paths::runtime_dir_from_environment().as_deref());
    println!("runtime_files: directory {}", dir.display());

    let mut fast_times = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        let checked = RuntimeFiles::check(&dir, CheckDepth::Fast)
            .unwrap_or_else(|error| panic!("fast check: {error}"));
        fast_times.push(started.elapsed());
        assert_eq!(checked.release, "1.9.0");
        println!(
            "runtime_files: fast check {:?} (provisioning record: {}, hashed: {:?})",
            fast_times.last().unwrap(),
            checked.provisioned,
            checked.hashed
        );
    }
    let started = Instant::now();
    let checked = RuntimeFiles::check(&dir, CheckDepth::Full)
        .unwrap_or_else(|error| panic!("full check: {error}"));
    let full = started.elapsed();
    assert_eq!(checked.hashed.len(), 5);
    let megabytes = RuntimeManifest::expected().total_bytes() as f64 / 1e6;
    println!(
        "runtime_files: full check {full:?} over {megabytes:.1} MB ({:.0} MB/s), release {} \
         (ONNX Runtime {})",
        megabytes / full.as_secs_f64(),
        checked.release,
        checked.onnxruntime
    );
    println!(
        "runtime_files: fast check, lowest of five: {:?}",
        fast_times.iter().min().unwrap()
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
