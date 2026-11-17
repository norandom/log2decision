//! Admission and counted Rust reservations for opt-in speed measurements.
//! Native provider allocations remain unaccounted; OS process peaks are separate.
#![allow(dead_code)]
use l2d_core::memory::{MemoryBudget, MemorySettings, parse_memory_size};
use l2d_core::resource::{BusyChoice, ResourceGuard};
use l2d_core::{Csic2010Adapter, InputAdapter, LogsFrame};
use std::{cell::RefCell, path::Path, rc::Rc};

thread_local! { static ACTIVE: RefCell<Option<Rc<RefCell<MemoryBudget>>>> = const { RefCell::new(None) }; }

type ProcessMemoryObservation = fn() -> (u64, u64, u64);

pub struct Measurement {
    operation: String,
    budget: Rc<RefCell<MemoryBudget>>,
    guard: ResourceGuard,
    native: bool,
    process_memory: Option<ProcessMemoryObservation>,
}
impl Measurement {
    pub fn start(operation: &str, native: bool) -> Self {
        let choice = match std::env::var("L2D_BUSY").as_deref() {
            Ok("refuse") => BusyChoice::Refuse,
            Ok("wait") | Err(_) => BusyChoice::Wait,
            Ok(other) => panic!("L2D_BUSY must be wait or refuse, got {other}"),
        };
        let guard = if native {
            ResourceGuard::npu(operation, choice)
        } else {
            ResourceGuard::enter(operation, choice)
        }
        .unwrap();
        let settings = MemorySettings {
            memory_limit_bytes: std::env::var("L2D_MEMORY_LIMIT")
                .ok()
                .map(|value| parse_memory_size(&value).unwrap()),
            ..MemorySettings::default()
        };
        let budget = Rc::new(RefCell::new(MemoryBudget::observe(&settings).unwrap()));
        ACTIVE.with(|active| {
            assert!(
                active.borrow().is_none(),
                "a measurement is already active on this thread"
            );
            *active.borrow_mut() = Some(budget.clone());
        });
        Self {
            operation: operation.into(),
            budget,
            guard,
            native,
            process_memory: None,
        }
    }
    pub fn with_process_memory(mut self, observe: ProcessMemoryObservation) -> Self {
        self.process_memory = Some(observe);
        self
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "operation": self.operation, "gate_held": true,
            "observation": self.guard.observation, "waited_for_host": self.guard.waited,
            "memory": self.budget.borrow().usage().report(),
            "accounting_scope": "counted conservative Rust, fixture, input and output reservations; this process",
            "native_memory_unaccounted": self.native,
            "measured_process_memory": self.process_memory.map(|observe| {
                let (working, peak, commit) = observe();
                serde_json::json!({"working_set_bytes": working, "peak_working_set_bytes": peak,
                    "private_commit_bytes": commit, "scope": "this process; OS observation, not the counted reservation peak"})
            }),
            "ended_while_panicking": std::thread::panicking(),
        })
    }
}
impl Drop for Measurement {
    fn drop(&mut self) {
        let report = self.report();
        eprintln!("measurement resources: {report}");
        let directory = std::env::temp_dir().join("log2decision-resource-records");
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = directory.join(format!("measurement-{}-{stamp}.json", std::process::id()));
        if let Err(error) = std::fs::create_dir_all(&directory)
            .and_then(|()| std::fs::write(&path, report.to_string()))
        {
            eprintln!(
                "could not write measurement record {}: {error}",
                path.display()
            );
        } else {
            eprintln!("measurement record: {}", path.display());
        }
        ACTIVE.with(|active| *active.borrow_mut() = None);
    }
}

pub fn reserve(bytes: u64, what: &str) {
    ACTIVE.with(|active| {
        if let Some(budget) = active.borrow().as_ref() {
            let mut budget = budget.borrow_mut();
            budget
                .require_whole(bytes, what)
                .unwrap_or_else(|error| panic!("{error}"));
            let held = budget.held_bytes();
            budget.hold(held.saturating_add(bytes));
        }
    });
}

pub fn file(path: &Path) {
    // Retain this conservative reservation through the measurement, including parsed
    // metadata/model edits and copies made while packaging/opening the fixture.
    let size = std::fs::metadata(path).unwrap().len();
    reserve(
        size.saturating_mul(16).saturating_add(64 * 1024),
        "loading speed measurement fixture",
    );
}

pub fn csic(path: &Path) -> LogsFrame {
    ACTIVE.with(|active| {
        let active = active.borrow();
        if let Some(budget) = active.as_ref() {
            Csic2010Adapter::new()
                .read_with(path, &mut budget.borrow_mut())
                .unwrap()
                .0
        } else {
            Csic2010Adapter::new().read(path).unwrap().0
        }
    })
}

pub fn copied_logs(logs: &LogsFrame) {
    reserve(
        (logs.frame().estimated_size() as u64)
            .saturating_mul(2)
            .saturating_add(logs.len() as u64 * 192),
        "copying speed measurement texts and references",
    );
}
