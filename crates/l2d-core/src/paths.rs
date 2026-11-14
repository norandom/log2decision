//! Fixed project locations: models, data, stores, runtime files and reports.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use crate::error::L2dError;
use crate::types::short_revision;

/// Environment variable that names the directory of the AMD runtime files, in place of
/// `runtime/ryzenai/` of the project. The files are 714 MB and are the same for every
/// checkout on a host, so one copy can serve several working copies. It is the one
/// location of the tool that an environment variable can move; an empty value counts as
/// not set.
pub const RUNTIME_DIR_VARIABLE: &str = "L2D_RUNTIME_DIR";

/// The runtime directory a value of [`RUNTIME_DIR_VARIABLE`] names: none when the
/// variable is not set or empty.
pub fn runtime_dir_from(value: Option<OsString>) -> Option<PathBuf> {
    value.filter(|value| !value.is_empty()).map(PathBuf::from)
}

/// The runtime directory named by [`RUNTIME_DIR_VARIABLE`] in the process environment,
/// to be passed to [`ProjectPaths::runtime_files_dir`].
pub fn runtime_dir_from_environment() -> Option<PathBuf> {
    runtime_dir_from(std::env::var_os(RUNTIME_DIR_VARIABLE))
}

/// The project root and the fixed directories below it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectPaths {
    pub root: PathBuf,
}

impl ProjectPaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `models/` — the only place encoders and baselines are read from or written to.
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    /// `models/encoders/`
    pub fn encoders_dir(&self) -> PathBuf {
        self.models_dir().join("encoders")
    }

    /// `models/baselines/`
    pub fn baselines_dir(&self) -> PathBuf {
        self.models_dir().join("baselines")
    }

    /// `models/encoders/<name>-<rev12>/`
    pub fn encoder_dir(&self, name: &str, revision: &str) -> PathBuf {
        self.encoders_dir()
            .join(format!("{name}-{}", short_revision(revision)))
    }

    /// `models/baselines/<name>-<rev12>/`
    pub fn baseline_dir(&self, name: &str, revision: &str) -> PathBuf {
        self.baselines_dir()
            .join(format!("{name}-{}", short_revision(revision)))
    }

    /// `models/calibrations/`
    pub fn calibrations_dir(&self) -> PathBuf {
        self.models_dir().join("calibrations")
    }

    /// `models/calibrations/<name>-<rev12>/`
    pub fn calibration_dir(&self, name: &str, revision: &str) -> PathBuf {
        self.calibrations_dir()
            .join(format!("{name}-{}", short_revision(revision)))
    }

    /// `data/` — where datasets are read from unless the analyst supplies a path.
    pub fn data_dir(&self) -> PathBuf {
        self.root.join("data")
    }

    /// `stores/`
    pub fn stores_dir(&self) -> PathBuf {
        self.root.join("stores")
    }

    /// `stores/<name>.db` — one store is one database file.
    pub fn store_path(&self, name: &str) -> PathBuf {
        self.stores_dir().join(format!("{name}.db"))
    }

    /// `runtime/`
    pub fn runtime_dir(&self) -> PathBuf {
        self.root.join("runtime")
    }

    /// `runtime/ryzenai/` — the AMD runtime files.
    pub fn ryzenai_dir(&self) -> PathBuf {
        self.runtime_dir().join("ryzenai")
    }

    /// Where the AMD runtime files are read from: the directory the user supplied (the
    /// value of [`RUNTIME_DIR_VARIABLE`], see [`runtime_dir_from_environment`]), or
    /// `runtime/ryzenai/` otherwise. A supplied directory replaces the default, as a
    /// supplied dataset path does in [`ProjectPaths::dataset`]; it is used as given. The
    /// directory is not required to exist here: the file check of the encoder crate names
    /// what is missing.
    ///
    /// Precedence, for the callers: a directory given by an explicit option of a command
    /// beats the environment variable, which beats the default. So a command passes
    /// `option.or(runtime_dir_from_environment())`. A relative directory, from either,
    /// is relative to the current directory of the process, not to the project root; the
    /// result is returned as given. Before a path below it is handed to the runtime it
    /// must be made absolute and plain (`canonicalize`, then without the `\\?\` prefix):
    /// only that form was proven with the provider (task 2.2).
    pub fn runtime_files_dir(&self, supplied: Option<&Path>) -> PathBuf {
        match supplied {
            Some(path) => path.to_path_buf(),
            None => self.ryzenai_dir(),
        }
    }

    /// `runtime/cache/` — the provider's compilation cache. It stays in the project
    /// whatever directory the runtime files are read from.
    pub fn runtime_cache_dir(&self) -> PathBuf {
        self.runtime_dir().join("cache")
    }

    /// `work/` — below it every run that writes result parts gets a directory of its
    /// own (`work/<run>/`), unless the analyst names one. Untracked.
    pub fn work_dir(&self) -> PathBuf {
        self.root.join("work")
    }

    /// `reports/`
    pub fn reports_dir(&self) -> PathBuf {
        self.root.join("reports")
    }

    /// Whether `path` lies below the project's model directory. The check is lexical and
    /// rejects any path that contains a parent-directory component.
    pub fn is_under_models(&self, path: &Path) -> bool {
        path.starts_with(self.models_dir())
            && !path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
    }

    /// Returns `path` when it exists; otherwise names the missing `item` and its expected
    /// location.
    pub fn require(&self, item: &str, path: impl Into<PathBuf>) -> Result<PathBuf, L2dError> {
        let path = path.into();
        if path.exists() {
            Ok(path)
        } else {
            Err(L2dError::Missing {
                item: item.to_string(),
                expected_at: path,
            })
        }
    }

    /// Locates a dataset: the path the analyst supplied, or `data/<name>` otherwise.
    pub fn dataset(&self, name: &str, supplied: Option<&Path>) -> Result<PathBuf, L2dError> {
        let location = match supplied {
            Some(path) => path.to_path_buf(),
            None => self.data_dir().join(name),
        };
        self.require(&format!("dataset {name}"), location)
    }
}
