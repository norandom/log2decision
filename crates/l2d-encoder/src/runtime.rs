//! The AMD runtime files: what must be in the runtime directory, and the check that it is.
//!
//! The check runs before anything of the runtime is loaded. That order is not a courtesy:
//! with `dyn_dispatch_core.dll` missing the process ends with an access violation while
//! the provider is registered, and no error comes back (task 2.2).
//!
//! # What is expected
//!
//! [`RuntimeManifest::expected`] is the tracked file `runtime/ryzenai.manifest.json`,
//! compiled into the tool: the release, where its files come from, and for each file its
//! name, size, SHA-256 and file version. It is compiled in because the tool is built for
//! one release (the `api-30` feature of `ort` is the API level of that release's ONNX
//! Runtime), and because the runtime directory may lie outside the project
//! ([`l2d_core::paths::RUNTIME_DIR_VARIABLE`]), where no tracked file is at hand.
//!
//! The provisioning step (`tools/runtime/provision.py`) writes a record of what it
//! copied, [`RECEIPT_FILE`], into the runtime directory: the release and for each file
//! its size, SHA-256 and modification time. A directory filled by hand has none.
//!
//! # What is checked when
//!
//! Always ([`CheckDepth::Fast`]; measured 0.16 to 1.0 ms on the reference host):
//! - the directory exists;
//! - a provisioning record, if there is one, names the expected release;
//! - every file of the manifest is present and has the expected size;
//! - no other DLL is in the directory;
//! - a file whose modification time differs from the provisioning record is hashed.
//!
//! On request ([`CheckDepth::Full`]; measured 0.7 s for the 714 MB of release 1.9.0, about
//! 1 GB/s with the files in the file cache, and 24 s when `sha2` is compiled without
//! optimization, which is why the workspace optimizes it in debug builds too): the
//! SHA-256 of every file. Too slow for every start of a command-line tool, so it is not
//! the default.
//!
//! So a file that was replaced by another of the same size is found at once in a
//! provisioned directory (its modification time changed), and by the full check in a
//! directory without a provisioning record. Not found by the fast check: a replacement of
//! the same size whose modification time was set back to the recorded one.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::UNIX_EPOCH;

use l2d_core::L2dError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The record the provisioning step leaves in the runtime directory.
pub const RECEIPT_FILE: &str = "provisioned.json";

/// The manifest format this build reads.
pub const MANIFEST_FORMAT: u32 = 1;

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

/// Where the files of a release come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestSource {
    /// AMD's release-specific package index.
    pub index: String,
    /// The package that carries the files, and its version.
    pub package: String,
    pub version: String,
    /// File name of the wheel on the index.
    pub wheel: String,
    /// Directory inside the wheel that holds the files.
    pub directory: String,
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

static EXPECTED: LazyLock<RuntimeManifest> = LazyLock::new(|| {
    RuntimeManifest::parse(include_str!("../../../runtime/ryzenai.manifest.json"))
        .unwrap_or_else(|error| panic!("the compiled-in runtime manifest is unusable: {error}"))
});

impl RuntimeManifest {
    /// The manifest this build of the tool expects: `runtime/ryzenai.manifest.json` as it
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
    /// Presence, sizes, no stray DLL, the release of the provisioning record; a hash only
    /// for files that changed since they were provisioned. For every start of the tool.
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
    /// Whether the directory holds a provisioning record ([`RECEIPT_FILE`]).
    pub provisioned: bool,
    /// The files whose SHA-256 was computed and found right, in manifest order.
    pub hashed: Vec<String>,
}

/// What the provisioning step recorded; only what the check uses is read.
#[derive(Debug, Deserialize)]
struct Receipt {
    release: String,
    onnxruntime: String,
    files: Vec<ReceiptFile>,
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
    /// of the wrong size or content, a stray DLL and a directory provisioned for another
    /// release are [`L2dError::Refused`], and the reason names every finding.
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
        let receipt = read_receipt(manifest, dir)?;

        let mut missing: Vec<&ManifestFile> = Vec::new();
        let mut wrong: Vec<String> = Vec::new();
        let mut hashed = Vec::new();
        for file in &manifest.files {
            let path = dir.join(&file.name);
            let metadata = match std::fs::metadata(&path) {
                Ok(metadata) if metadata.is_file() => metadata,
                _ => {
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
            let changed_since_provisioning = receipt.as_ref().is_some_and(|receipt| {
                let recorded = receipt
                    .files
                    .iter()
                    .find(|recorded| recorded.name.eq_ignore_ascii_case(&file.name))
                    .map(|recorded| recorded.modified_unix_ns);
                recorded.is_none() || recorded != modified_unix_ns(&metadata)
            });
            if depth == CheckDepth::Full || changed_since_provisioning {
                let found = sha256_of(&path)?;
                if found == file.sha256 {
                    hashed.push(file.name.clone());
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
        Ok(RuntimeFiles {
            dir: dir.to_path_buf(),
            release: manifest.release.clone(),
            onnxruntime: manifest.onnxruntime.clone(),
            provisioned: receipt.is_some(),
            hashed,
        })
    }

    /// The library `ort` is initialised from: the first file of the manifest.
    pub fn onnxruntime_dll(&self) -> PathBuf {
        self.dir.join("onnxruntime.dll")
    }
}

/// The provisioning record of `dir`, if there is one. A record of another release is a
/// refusal that names both; so is a record that cannot be read.
fn read_receipt(manifest: &RuntimeManifest, dir: &Path) -> Result<Option<Receipt>, L2dError> {
    let path = dir.join(RECEIPT_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(L2dError::Refused {
                reason: format!(
                    "the provisioning record {} cannot be read: {error}",
                    path.display()
                ),
            });
        }
    };
    let receipt: Receipt = serde_json::from_str(&text).map_err(|error| L2dError::Refused {
        reason: format!(
            "the provisioning record {} cannot be read: {error} ({PROVISION_HINT})",
            path.display()
        ),
    })?;
    if receipt.release != manifest.release || receipt.onnxruntime != manifest.onnxruntime {
        return Err(L2dError::Refused {
            reason: format!(
                "the runtime directory {} was provisioned for AMD Ryzen AI {} (ONNX Runtime \
                 {}), this build of l2d expects {}; the two are not mixed ({PROVISION_HINT})",
                dir.display(),
                receipt.release,
                receipt.onnxruntime,
                manifest.release_text()
            ),
        });
    }
    Ok(Some(receipt))
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
        GetFileVersionInfoSizeW, GetFileVersionInfoW, VS_FIXEDFILEINFO, VerQueryValueW,
    };

    let name: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: `name` is a NUL-terminated UTF-16 path; the handle output may be null.
    let size = unsafe { GetFileVersionInfoSizeW(name.as_ptr(), std::ptr::null_mut()) };
    if size == 0 {
        return None;
    }
    let mut block = vec![0u8; size as usize];
    // SAFETY: `block` is at least `size` bytes long.
    let read = unsafe { GetFileVersionInfoW(name.as_ptr(), 0, size, block.as_mut_ptr().cast()) };
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
