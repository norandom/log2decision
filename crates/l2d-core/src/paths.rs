//! Fixed project locations: models, data, stores, runtime files and reports.

use std::path::{Component, Path, PathBuf};

use crate::error::L2dError;
use crate::types::short_revision;

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

    /// `runtime/cache/` — the provider's compilation cache.
    pub fn runtime_cache_dir(&self) -> PathBuf {
        self.runtime_dir().join("cache")
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
