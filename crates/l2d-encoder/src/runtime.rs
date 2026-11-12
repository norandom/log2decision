//! The AMD runtime files: what must be in the runtime directory, and the check that it is.
//!
//! The check runs before anything of the runtime is loaded. That order is not a courtesy:
//! with `dyn_dispatch_core.dll` missing the process ends with an access violation while
//! the provider is registered, and no error comes back (task 2.2).
//!
//! # What is expected
//!
//! [`RuntimeManifest::expected`] is a tracked file, `runtime/ryzenai-<release>.manifest.json`,
//! compiled into the tool (which release: the feature `ryzenai-1-9` or `ryzenai-1-8` of
//! this crate, one choice that also sets the API level of `ort`): the release, where its files come from, and for each file its
//! name, size, SHA-256 and file version. It is compiled in because the tool is built for
//! one release (the `api-30` feature of `ort` is the API level of that release's ONNX
//! Runtime), and because the runtime directory may lie outside the project
//! ([`l2d_core::paths::RUNTIME_DIR_VARIABLE`]), where no tracked file is at hand.
//!
//! # The record in the directory
//!
//! [`RECEIPT_FILE`] in the runtime directory records what was verified there: the release
//! and for each file its size, SHA-256 and modification time. It is written by the
//! provisioning step (`tools/runtime/provision.py`), which also records where the files
//! came from, or by this check after it hashed every file of a directory that had no
//! record (marked `"written_by": "check"`, with no source). The check treats the two
//! alike: it reads the release and the modification times and trusts none of the hashes.
//!
//! # What is checked when
//!
//! The rule: a file is trusted without hashing only when a record says it was hashed
//! with this modification time.
//!
//! Always ([`CheckDepth::Fast`]; measured 0.3 to 2.2 ms on the reference host when a record
//! is there, 0.84 s for the one start that hashes and writes it):
//! - the directory exists;
//! - the record, if there is one, has a known format and names the expected release;
//! - every file of the manifest is present, is a file and has the expected size;
//! - no other DLL is in the directory;
//! - a file whose modification time differs from the record, or that the record does not
//!   name, is hashed; when it is right the record is brought up to date (the file's new
//!   time, and a note `reverified` saying that this check did it and when), so that the
//!   start after it is fast again. Who wrote the record and where the files came from
//!   stay as they were;
//! - without a record every file is hashed, and when all are right the record is written,
//!   so that the next start is fast. If it cannot be written (a directory that is not
//!   writable), every start hashes, and [`RuntimeFiles::record`] says so.
//!
//! On request ([`CheckDepth::Full`]): the SHA-256 of every file, record or not. Hashing
//! the 714 MB of release 1.9.0 takes 0.7 to 0.9 s (about 1 GB/s with the files in the file
//! cache; 24 s when `sha2` is compiled without optimization, which is why the workspace
//! optimizes it in debug builds too).
//!
//! So a directory filled by hand, for example from a package that has files of the same
//! names and sizes but other content, is refused at its first use. Not found by the fast
//! check: a replacement of the same size whose modification time was set back to the
//! recorded one, and a record written by hand; the full check finds both.
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::UNIX_EPOCH;

use l2d_core::L2dError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The record of what was verified in the runtime directory, written by the provisioning
/// step or by the check.
pub const RECEIPT_FILE: &str = "provisioned.json";

/// The manifest format this build reads.
pub const MANIFEST_FORMAT: u32 = 1;

/// The format of the record this build reads and writes.
pub const RECEIPT_FORMAT: u32 = 1;

/// How to fill the runtime directory, for messages.
const PROVISION_HINT: &str = "see runtime/README.md (uv run --project tools/runtime python \
                              tools/runtime/provision.py)";

/// One required file of a release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFile {
    /// File name inside the runtime directory.
    pub name: String,
    pub bytes: u64,
    /// Lower-case hexadecimal SHA-256 of the file.
    pub sha256: String,
    /// The file version the DLL reports in its version resource.
    pub version: String,
}

/// Where the files of a release come from. Which package that is depends on the kind:
/// AMD's wheel on its release-specific package index (`wheel`), or AMD's NuGet package
/// inside the ZIP AMD publishes (`nuget`). The two hold different builds of the same
/// files, so a manifest is for one of them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestSource {
    /// `wheel` or `nuget`.
    pub kind: String,
    /// The package that carries the files, and its version.
    pub package: String,
    pub version: String,
    /// Directory inside the package that holds the files.
    pub directory: String,
    /// What else names the source. For `wheel`: `index` (AMD's index) and `wheel` (the
    /// file name there). For `nuget`: `archive` (the ZIP as AMD names it), `nupkg` (the
    /// package inside it) and `nupkg_sha256`.
    #[serde(flatten)]
    pub detail: std::collections::BTreeMap<String, String>,
}
/// The required runtime files of one AMD Ryzen AI release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeManifest {
    pub format: u32,
    /// The Ryzen AI release, for example `1.9.0`.
    pub release: String,
    /// The ONNX Runtime version of that release, for example `1.30.0`.
    pub onnxruntime: String,
    pub source: ManifestSource,
    /// The files, the library `ort` loads first.
    pub files: Vec<ManifestFile>,
}

// The release the tool is built for is one choice: the feature `ryzenai-1-9` or
// `ryzenai-1-8` of this crate (its default, `crates/l2d-encoder/Cargo.toml`). The feature
// selects the manifest compiled in here and the API level `ort` asks the runtime for.
#[cfg(all(feature = "ryzenai-1-9", feature = "ryzenai-1-8"))]
compile_error!(
    "the features `ryzenai-1-9` and `ryzenai-1-8` are both enabled: the tool is built for \
     one release of the AMD runtime"
);
#[cfg(not(any(feature = "ryzenai-1-9", feature = "ryzenai-1-8")))]
compile_error!(
    "neither `ryzenai-1-9` nor `ryzenai-1-8` is enabled: choose the release of the AMD \
     runtime the tool is built for"
);

#[cfg(feature = "ryzenai-1-9")]
const EXPECTED_MANIFEST: &str = include_str!("../../../runtime/ryzenai-1.9.0.manifest.json");
#[cfg(all(feature = "ryzenai-1-8", not(feature = "ryzenai-1-9")))]
const EXPECTED_MANIFEST: &str = include_str!("../../../runtime/ryzenai-1.8.0.manifest.json");
#[cfg(not(any(feature = "ryzenai-1-9", feature = "ryzenai-1-8")))]
const EXPECTED_MANIFEST: &str = "";

static EXPECTED: LazyLock<RuntimeManifest> = LazyLock::new(|| {
    RuntimeManifest::parse(EXPECTED_MANIFEST)
        .unwrap_or_else(|error| panic!("the compiled-in runtime manifest is unusable: {error}"))
});

impl RuntimeManifest {
    /// The manifest this build of the tool expects: `runtime/ryzenai-<release>.manifest.json` as it
    /// was when the tool was compiled.
    pub fn expected() -> &'static RuntimeManifest {
        &EXPECTED
    }

    /// Reads a manifest and refuses one this build cannot rely on.
    pub fn parse(text: &str) -> Result<Self, L2dError> {
        let invalid = |detail: String| L2dError::InvalidInput {
            detail: format!("runtime manifest: {detail}"),
        };
        let manifest: RuntimeManifest =
            serde_json::from_str(text).map_err(|error| invalid(error.to_string()))?;
        if manifest.format != MANIFEST_FORMAT {
            return Err(invalid(format!(
                "format {} is not the format {MANIFEST_FORMAT} this build reads",
                manifest.format
            )));
        }
        if manifest.files.is_empty() {
            return Err(invalid("it names no file".into()));
        }
        for (index, file) in manifest.files.iter().enumerate() {
            let plain_name = !file.name.is_empty()
                && Path::new(&file.name)
                    .file_name()
                    .and_then(|name| name.to_str())
                    == Some(file.name.as_str());
            if !plain_name {
                return Err(invalid(format!("{:?} is not a plain file name", file.name)));
            }
            let is_digest = file.sha256.len() == 64
                && file
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
            if !is_digest {
                return Err(invalid(format!(
                    "the SHA-256 of {} is not 64 lower-case hexadecimal digits",
                    file.name
                )));
            }
            if manifest.files[..index]
                .iter()
                .any(|earlier| earlier.name.eq_ignore_ascii_case(&file.name))
            {
                return Err(invalid(format!("{} is named twice", file.name)));
            }
        }
        Ok(manifest)
    }

    /// Sum of the file sizes.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|file| file.bytes).sum()
    }

    fn release_text(&self) -> String {
        format!(
            "AMD Ryzen AI {} (ONNX Runtime {})",
            self.release, self.onnxruntime
        )
    }
}

/// How much of the runtime directory is verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckDepth {
    /// Presence, sizes, no stray DLL, the release of the record; a hash only for files
    /// that changed since the record was written, and for all files when there is no
    /// record (which is then written). For every start of the tool.
    Fast,
    /// As `Fast`, and the SHA-256 of every file. For `l2d runtime check` and after the
    /// files were placed.
    Full,
}

/// A runtime directory that passed the check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeFiles {
    /// The directory, as it was given.
    pub dir: PathBuf,
    /// The Ryzen AI release of the files.
    pub release: String,
    /// The ONNX Runtime version of the files.
    pub onnxruntime: String,
    /// File name of the library `ort` is initialised from: the manifest's first file.
    pub library: String,
    /// The record of the directory ([`RECEIPT_FILE`]): found, written now, or not writable.
    pub record: RecordState,
    /// Whether this check brought a record it found up to date.
    pub refresh: RecordRefresh,
    /// The files whose SHA-256 was computed and found right, in manifest order.
    pub hashed: Vec<String>,
}

/// What this check did to a record it found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordRefresh {
    /// Nothing: every file has the modification time of the record, or there was no
    /// record to refresh.
    NotNeeded,
    /// These files had another modification time than the record (or were not in it);
    /// they were hashed, found right, and the record now holds their times. The next
    /// start hashes nothing.
    Refreshed { files: Vec<String> },
    /// As `Refreshed`, but the record could not be rewritten: the files are right, and
    /// they are hashed again at every start until the directory is writable.
    Failed { files: Vec<String>, reason: String },
}

impl RecordRefresh {
    /// The refresh in words, for the tool's output; empty when nothing was done.
    pub fn describe(&self) -> String {
        match self {
            RecordRefresh::NotNeeded => String::new(),
            RecordRefresh::Refreshed { files } => format!(
                "{} changed since the record, hashed and found right; the record was \
                 brought up to date",
                files.join(", ")
            ),
            RecordRefresh::Failed { files, reason } => format!(
                "{} changed since the record, hashed and found right; the record could not \
                 be brought up to date ({reason}), so they are hashed at every start",
                files.join(", ")
            ),
        }
    }
}

/// The record of a checked directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordState {
    /// Found; written by the provisioning step.
    Provisioning,
    /// Found; written by an earlier check that hashed every file.
    Check,
    /// There was none: every file was hashed and the record written by this check.
    Written,
    /// There was none and it could not be written: every file was hashed, and will be
    /// again at every start until the directory is writable.
    NotWritten { reason: String },
}

impl RecordState {
    /// The state in words, for the tool's output.
    pub fn describe(&self) -> String {
        match self {
            RecordState::Provisioning => "written by the provisioning step".into(),
            RecordState::Check => "written by an earlier check".into(),
            RecordState::Written => {
                "there was none; every file was hashed and the record written".into()
            }
            RecordState::NotWritten { reason } => format!(
                "there is none and it could not be written ({reason}); every file is hashed \
                 at every start"
            ),
        }
    }
}

/// The record in the directory; only what the check uses is read.
#[derive(Debug, Deserialize)]
struct Receipt {
    format: u32,
    release: String,
    onnxruntime: String,
    /// `provisioning` or `check`; a record without it is the provisioning step's.
    #[serde(default)]
    written_by: Option<String>,
    files: Vec<ReceiptFile>,
    /// The whole record as it was read, so that a refresh keeps what it does not change:
    /// who wrote it and where the files came from.
    #[serde(skip)]
    whole: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ReceiptFile {
    name: String,
    modified_unix_ns: u64,
}

impl RuntimeFiles {
    /// Checks `dir` against the manifest compiled into the tool. Call it before the
    /// runtime is loaded.
    ///
    /// A missing directory or file is [`L2dError::Missing`] with its expected path; files
    /// of the wrong size or content, a stray DLL and a directory recorded for another
    /// release are [`L2dError::Refused`], and the reason names every finding. In a
    /// directory without a record every file is hashed and the record is written.
    pub fn check(dir: &Path, depth: CheckDepth) -> Result<Self, L2dError> {
        Self::check_against(RuntimeManifest::expected(), dir, depth)
    }

    /// [`RuntimeFiles::check`] against a given manifest.
    pub fn check_against(
        manifest: &RuntimeManifest,
        dir: &Path,
        depth: CheckDepth,
    ) -> Result<Self, L2dError> {
        if !dir.is_dir() {
            return Err(L2dError::Missing {
                item: format!(
                    "runtime directory with the files of {} ({PROVISION_HINT})",
                    manifest.release_text()
                ),
                expected_at: dir.to_path_buf(),
            });
        }
        if manifest.files.is_empty() {
            return Err(L2dError::InvalidInput {
                detail: "runtime manifest: it names no file".into(),
            });
        }
        let receipt = read_receipt(manifest, dir)?;

        let mut missing: Vec<&ManifestFile> = Vec::new();
        let mut wrong: Vec<String> = Vec::new();
        let mut hashed = Vec::new();
        // What a record written by this check holds: the modification times the hashes
        // were computed under, read before the files are.
        let mut verified: Vec<serde_json::Value> = Vec::new();
        // Files that are right but not as the record has them: the record is refreshed.
        let mut stale: Vec<serde_json::Value> = Vec::new();
        for file in &manifest.files {
            let path = dir.join(&file.name);
            let metadata = match std::fs::metadata(&path) {
                Ok(metadata) if metadata.is_file() => metadata,
                Ok(_) => {
                    wrong.push(format!("{} is not a file", path.display()));
                    continue;
                }
                Err(_) => {
                    missing.push(file);
                    continue;
                }
            };
            if metadata.len() != file.bytes {
                wrong.push(format!(
                    "{} has {} bytes, the release has {} bytes{}",
                    path.display(),
                    metadata.len(),
                    file.bytes,
                    version_note(&path, file)
                ));
                continue;
            }
            let modified = modified_unix_ns(&metadata);
            // Trusted without a hash only when a record names the file with this
            // modification time. Without a record nothing is.
            let recorded_as_it_is = receipt.as_ref().is_some_and(|receipt| {
                let recorded = receipt
                    .files
                    .iter()
                    .find(|recorded| recorded.name.eq_ignore_ascii_case(&file.name))
                    .map(|recorded| recorded.modified_unix_ns);
                recorded.is_some() && recorded == modified
            });
            if depth == CheckDepth::Full || !recorded_as_it_is {
                let found = sha256_of(&path)?;
                if found == file.sha256 {
                    hashed.push(file.name.clone());
                    let entry = serde_json::json!({
                        "name": file.name,
                        "bytes": file.bytes,
                        "sha256": file.sha256,
                        "modified_unix_ns": modified,
                    });
                    if receipt.is_some() && !recorded_as_it_is {
                        stale.push(entry.clone());
                    }
                    verified.push(entry);
                } else {
                    wrong.push(format!(
                        "{} has SHA-256 {found}, the release has {}{}",
                        path.display(),
                        file.sha256,
                        version_note(&path, file)
                    ));
                }
            }
        }
        for stray in stray_dlls(manifest, dir)? {
            wrong.push(format!(
                "{} is not part of the release{} (the directory was proven with exactly the \
                 {} files of the manifest; a DLL beside them can be loaded in their place)",
                stray.display(),
                file_version(&stray)
                    .map(|version| format!("; it reports file version {version}"))
                    .unwrap_or_default(),
                manifest.files.len()
            ));
        }

        let expected_at = |file: &ManifestFile| dir.join(&file.name).display().to_string();
        if !wrong.is_empty() {
            wrong.extend(
                missing
                    .iter()
                    .map(|file| format!("{} is missing", expected_at(file))),
            );
            return Err(L2dError::Refused {
                reason: format!(
                    "the runtime directory {} does not hold the files of {}: {} ({PROVISION_HINT})",
                    dir.display(),
                    manifest.release_text(),
                    wrong.join("; ")
                ),
            });
        }
        if let Some(first) = missing.first() {
            let item = if missing.len() == 1 {
                format!("runtime file {} of {}", first.name, manifest.release_text())
            } else {
                format!(
                    "{} of {} runtime files of {}: {}",
                    missing.len(),
                    manifest.files.len(),
                    manifest.release_text(),
                    missing
                        .iter()
                        .map(|file| format!("{} (expected at {})", file.name, expected_at(file)))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            return Err(L2dError::Missing {
                item: format!("{item}; {PROVISION_HINT}"),
                expected_at: dir.join(&first.name),
            });
        }
        let record = match &receipt {
            Some(receipt) if receipt.written_by.as_deref() == Some(WRITTEN_BY_CHECK) => {
                RecordState::Check
            }
            Some(_) => RecordState::Provisioning,
            // No record: every file was hashed above and is right.
            None => match write_receipt(manifest, dir, &verified) {
                Ok(()) => RecordState::Written,
                Err(reason) => RecordState::NotWritten { reason },
            },
        };
        // Every file is right here. A record that no longer has their modification times
        // is brought up to date, so that only this start was slow.
        let refresh = match &receipt {
            Some(receipt) if !stale.is_empty() => {
                let files = stale
                    .iter()
                    .filter_map(|entry| entry["name"].as_str().map(str::to_string))
                    .collect();
                match refresh_receipt(dir, receipt, &stale) {
                    Ok(()) => RecordRefresh::Refreshed { files },
                    Err(reason) => RecordRefresh::Failed { files, reason },
                }
            }
            _ => RecordRefresh::NotNeeded,
        };
        Ok(RuntimeFiles {
            dir: dir.to_path_buf(),
            release: manifest.release.clone(),
            onnxruntime: manifest.onnxruntime.clone(),
            library: manifest.files[0].name.clone(),
            record,
            refresh,
            hashed,
        })
    }

    /// The library `ort` is initialised from: the first file of the manifest the directory
    /// was checked against (`onnxruntime.dll` in the manifest of the tool). The path is
    /// below [`RuntimeFiles::dir`] as that was given; make it absolute, without the `\\?\`
    /// prefix, before it is passed to `ort::init_from`.
    pub fn onnxruntime_dll(&self) -> PathBuf {
        self.dir.join(&self.library)
    }
}

/// The record of `dir`, if there is one, whoever wrote it. A record of another release
/// is a refusal that names both; so is a record of another format or one that cannot be
/// read.
fn read_receipt(manifest: &RuntimeManifest, dir: &Path) -> Result<Option<Receipt>, L2dError> {
    let path = dir.join(RECEIPT_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(L2dError::Refused {
                reason: format!("the record {} cannot be read: {error}", path.display()),
            });
        }
    };
    let unreadable = |error: serde_json::Error| L2dError::Refused {
        reason: format!(
            "the record {} cannot be read: {error} ({PROVISION_HINT})",
            path.display()
        ),
    };
    let whole: serde_json::Value = serde_json::from_str(&text).map_err(unreadable)?;
    let mut receipt: Receipt = serde_json::from_value(whole.clone()).map_err(unreadable)?;
    receipt.whole = whole;
    if receipt.format != RECEIPT_FORMAT {
        return Err(L2dError::Refused {
            reason: format!(
                "the record {} has format {}, this build of l2d reads format {RECEIPT_FORMAT}; \
                 remove it to have the files verified again, or provision again \
                 ({PROVISION_HINT})",
                path.display(),
                receipt.format
            ),
        });
    }
    if receipt.release != manifest.release || receipt.onnxruntime != manifest.onnxruntime {
        return Err(L2dError::Refused {
            reason: format!(
                "the runtime directory {} was verified as AMD Ryzen AI {} (ONNX Runtime {}) \
                 according to its record, this build of l2d expects {}; the two are not \
                 mixed ({PROVISION_HINT})",
                dir.display(),
                receipt.release,
                receipt.onnxruntime,
                manifest.release_text()
            ),
        });
    }
    Ok(Some(receipt))
}

/// `written_by` of a record this check wrote.
const WRITTEN_BY_CHECK: &str = "check";

/// Writes the record of a directory whose files were all hashed and found right. The
/// error is the reason it could not be written; the check then stands without it.
fn write_receipt(
    manifest: &RuntimeManifest,
    dir: &Path,
    verified: &[serde_json::Value],
) -> Result<(), String> {
    let complete = verified.len() == manifest.files.len()
        && verified
            .iter()
            .all(|file| file["modified_unix_ns"].is_u64());
    if !complete {
        return Err("a modification time could not be read".into());
    }
    let record = serde_json::json!({
        "format": RECEIPT_FORMAT,
        "release": manifest.release,
        "onnxruntime": manifest.onnxruntime,
        // Not the provisioning step: this check hashed the files it found. Where they
        // came from is not known here, so no source is recorded.
        "written_by": WRITTEN_BY_CHECK,
        "verified_unix_s": unix_seconds_now(),

        "files": verified,
    });
    write_record(dir, &record)
}

/// Brings a record up to date after files that it did not have with their present
/// modification time were hashed and found right. Everything else of the record stays:
/// `written_by` still names who verified the directory first, and a provisioning record
/// keeps its source. The refresh itself is noted under `reverified`.
fn refresh_receipt(
    dir: &Path,
    receipt: &Receipt,
    stale: &[serde_json::Value],
) -> Result<(), String> {
    if !stale.iter().all(|file| file["modified_unix_ns"].is_u64()) {
        return Err("a modification time could not be read".into());
    }
    let mut record = receipt.whole.clone();
    let files = record
        .get_mut("files")
        .and_then(|files| files.as_array_mut())
        .ok_or("the record has no list of files")?;
    for entry in stale {
        let name = entry["name"].as_str().unwrap_or_default();
        let recorded = files.iter_mut().find(|recorded| {
            recorded["name"]
                .as_str()
                .is_some_and(|recorded| recorded.eq_ignore_ascii_case(name))
        });
        match recorded {
            Some(recorded) => *recorded = entry.clone(),
            None => files.push(entry.clone()),
        }
    }
    record["reverified"] = serde_json::json!({
        "by": WRITTEN_BY_CHECK,
        "unix_s": unix_seconds_now(),
        "files": stale.iter().map(|entry| entry["name"].clone()).collect::<Vec<_>>(),
    });
    write_record(dir, &record)
}

fn unix_seconds_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since_epoch| since_epoch.as_secs())
        .unwrap_or(0)
}

/// Writes a record into `dir`, whole or not at all.
fn write_record(dir: &Path, record: &serde_json::Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(record).map_err(|error| error.to_string())?;
    let path = dir.join(RECEIPT_FILE);
    // Through a file of this process, renamed into place: two processes that check the
    // same directory at once each write a whole record.
    let partial = dir.join(format!(".{RECEIPT_FILE}.{}.partial", std::process::id()));
    let written = std::fs::write(&partial, text + "\n")
        .and_then(|()| std::fs::rename(&partial, &path))
        .map_err(|error| format!("writing {}: {error}", path.display()));
    if written.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    written
}

/// DLLs in `dir` that the manifest does not name, sorted. Other files and folders do not
/// matter: nothing loads them.
fn stray_dlls(manifest: &RuntimeManifest, dir: &Path) -> Result<Vec<PathBuf>, L2dError> {
    let failed = |error: std::io::Error| L2dError::Runtime {
        detail: format!("listing the runtime directory {}: {error}", dir.display()),
    };
    let mut stray = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(failed)? {
        let path = entry.map_err(failed)?.path();
        let is_dll = path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"));
        let named = path.file_name().is_some_and(|name| {
            manifest
                .files
                .iter()
                .any(|file| name.eq_ignore_ascii_case(&file.name))
        });
        if is_dll && !named {
            stray.push(path);
        }
    }
    stray.sort();
    Ok(stray)
}

fn modified_unix_ns(metadata: &std::fs::Metadata) -> Option<u64> {
    let since_epoch = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
    u64::try_from(since_epoch.as_nanos()).ok()
}

/// Lower-case hexadecimal SHA-256 of a file, read in pieces of 1 MiB.
pub fn sha256_of(path: &Path) -> Result<String, L2dError> {
    let failed = |error: std::io::Error| L2dError::Runtime {
        detail: format!("reading {}: {error}", path.display()),
    };
    let mut file = std::fs::File::open(path).map_err(failed)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(failed)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// For a file that is not the release's: the version it reports against the one expected,
/// which is what tells a file of another release from a damaged one.
fn version_note(path: &Path, expected: &ManifestFile) -> String {
    match file_version(path) {
        Some(found) if found != expected.version => format!(
            " (it reports file version {found}, the release has {}: a file of another \
             release)",
            expected.version
        ),
        Some(found) => format!(" (it reports file version {found}, as the release does)"),
        None => String::new(),
    }
}

/// The file version in the version resource of a DLL or program, as `a.b.c` (or `a.b.c.d`
/// when the fourth part is not zero). The file is read as data; it is not loaded and none
/// of its code runs. None when it has no such resource.
#[cfg(windows)]
pub fn file_version(path: &Path) -> Option<String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_VER_GET_NEUTRAL, GetFileVersionInfoExW, GetFileVersionInfoSizeExW, VS_FIXEDFILEINFO,
        VerQueryValueW,
    };

    let name: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // The file's own resource is asked for (`FILE_VER_GET_NEUTRAL`), not that of a
    // language file beside it.
    let mut handle = 0u32;
    // SAFETY: `name` is a NUL-terminated UTF-16 path and `handle` a valid output.
    let size =
        unsafe { GetFileVersionInfoSizeExW(FILE_VER_GET_NEUTRAL, name.as_ptr(), &mut handle) };
    if size == 0 {
        return None;
    }
    let mut block = vec![0u8; size as usize];
    // SAFETY: `block` is at least `size` bytes long.
    let read = unsafe {
        GetFileVersionInfoExW(
            FILE_VER_GET_NEUTRAL,
            name.as_ptr(),
            0,
            size,
            block.as_mut_ptr().cast(),
        )
    };
    if read == 0 {
        return None;
    }
    let root: [u16; 2] = [u16::from(b'\\'), 0];
    let mut info: *mut core::ffi::c_void = std::ptr::null_mut();
    let mut length = 0u32;
    // SAFETY: `block` holds the version information read above, `root` is NUL-terminated,
    // and the two outputs are valid for the call.
    let found =
        unsafe { VerQueryValueW(block.as_ptr().cast(), root.as_ptr(), &mut info, &mut length) };
    if found == 0 || info.is_null() || (length as usize) < size_of::<VS_FIXEDFILEINFO>() {
        return None;
    }
    // SAFETY: the call returned a pointer into `block`, which is alive, to at least
    // `length` bytes, enough for one such structure; the read does not rely on alignment.
    let fixed = unsafe { info.cast::<VS_FIXEDFILEINFO>().read_unaligned() };
    // The fixed part of a version resource starts with this signature; anything else is
    // not one.
    if fixed.dwSignature != 0xFEEF_04BD {
        return None;
    }
    let parts = [
        fixed.dwFileVersionMS >> 16,
        fixed.dwFileVersionMS & 0xffff,
        fixed.dwFileVersionLS >> 16,
        fixed.dwFileVersionLS & 0xffff,
    ];
    Some(if parts[3] == 0 {
        format!("{}.{}.{}", parts[0], parts[1], parts[2])
    } else {
        format!("{}.{}.{}.{}", parts[0], parts[1], parts[2], parts[3])
    })
}

/// File versions are a Windows resource; elsewhere none is read.
#[cfg(not(windows))]
pub fn file_version(_path: &Path) -> Option<String> {
    None
}
