//! Host-wide admission for expensive work, independent of the project directory.
//!
//! OS file locks release on normal exit and on crashes. The metadata is only a
//! description: an old metadata file never proves that a process is still running.
//! A session takes the host gate before the NPU gate and holds both for its lifetime.
//! Nested calls on the owning thread reuse the gate; unrelated threads wait/refuse.
//! NPU ownership describes cooperating tool sessions. Other software's hardware
//! utilization is unknown, including when the tool's NPU gate is unlocked.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{
    L2dError,
    memory::{MemoryUsage, MemoryUsageReport, free_physical_memory},
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BusyChoice {
    Refuse,
    #[default]
    Wait,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Holder {
    pub process_id: u32,
    pub operation: String,
    pub lease_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub free_memory_bytes: u64,
    pub host_holder: Option<Holder>,
    pub npu_holder: Option<Holder>,
    pub external_npu_utilization: String,
}

#[derive(Debug)]
struct Lease {
    // The handle MUST outlive every nested owner. Never unlink the lock file:
    // replacing its inode would allow two processes to lock different files.
    _file: File,
    witness: Option<File>,
    witness_path: PathBuf,
    metadata: PathBuf,
}

impl Drop for Lease {
    fn drop(&mut self) {
        // Remove the publication before releasing the host. Crash leftovers are
        // rejected by their unlocked witness, independently of PID reuse.
        let _ = fs::remove_file(&self.metadata);
        self.witness.take();
        let _ = fs::remove_file(&self.witness_path);
    }
}

thread_local! {
    static OWNED: RefCell<HashMap<PathBuf, Weak<Lease>>> = RefCell::new(HashMap::new());
    static CHOICE: std::cell::Cell<BusyChoice> = const { std::cell::Cell::new(BusyChoice::Wait) };
}

/// Policy for subsequent operations on this thread (CLI `--busy`). Library callers
/// can pass a choice directly to `enter`, `npu`, or `tracked`.
pub fn set_busy_choice(choice: BusyChoice) {
    CHOICE.with(|setting| setting.set(choice));
}
pub fn busy_choice() -> BusyChoice {
    CHOICE.with(std::cell::Cell::get)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReport {
    pub operation: String,
    pub process_id: u32,
    pub gate_held: bool,
    pub observation: Observation,
    pub waited_for_host: bool,
    #[serde(flatten)]
    pub memory: MemoryUsageReport,
    pub ended_while_panicking: bool,
}

#[derive(Debug)]
/// A live operation is thread-affine. Move work to its worker thread before admission;
/// nested calls are synchronous on that thread. This prevents a moved guard leaving a
/// reusable lease behind in the original thread's nesting context.
///
/// ```compile_fail
/// use l2d_core::resource::{BusyChoice, ResourceGuard};
/// let guard = ResourceGuard::enter("training", BusyChoice::Refuse).unwrap();
/// std::thread::spawn(move || drop(guard));
/// ```
pub struct ResourceGuard {
    _host: Arc<Lease>,
    _npu: Option<Arc<Lease>>,
    pub observation: Observation,
    pub waited: bool,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
    record: Option<(String, MemoryUsage, PathBuf)>,
}

fn failed(doing: &str, error: impl std::fmt::Display) -> L2dError {
    L2dError::Refused {
        reason: format!("resource admission could not {doing}: {error}"),
    }
}

fn root() -> PathBuf {
    #[cfg(windows)]
    let base = std::env::var_os("ProgramData")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"));
    #[cfg(not(windows))]
    let base = PathBuf::from("/tmp");
    base.join("log2decision-resource-gates")
}

fn open(path: &Path) -> Result<File, L2dError> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| failed("open the host lock", e))
}

fn owner(path: &Path, waited: &mut bool) -> Result<Option<Holder>, L2dError> {
    let metadata = path.with_extension("holder.json");
    let started = std::time::Instant::now();
    loop {
        let published = (|| {
            use std::io::Read;
            let file = File::open(&metadata).map_err(|e| failed("read the lock holder", e))?;
            let holder: Holder = serde_json::from_reader(file.take(4096))
                .map_err(|e| failed("read the lock holder", e))?;
            let name = Path::new(&holder.lease_file);
            if name.file_name() != Some(name.as_os_str()) || holder.lease_file.len() > 160 {
                return Err(failed("read the lock holder", "invalid lease witness name"));
            }
            let witness = open(&path.with_file_name(&holder.lease_file))?;
            match witness.try_lock() {
                Err(TryLockError::WouldBlock) => Ok(holder),
                _ => Err(failed(
                    "read the lock holder",
                    "holder publication is stale or incomplete",
                )),
            }
        })();
        if let Ok(holder) = published {
            return Ok(Some(holder));
        }
        // The holder may have exited while publication was being read.
        let probe = open(path)?;
        if probe.try_lock().is_ok() {
            return Ok(None);
        }
        if started.elapsed() >= Duration::from_secs(5) {
            return published.map(Some);
        }
        *waited = true;
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn local(path: &Path) -> Option<Arc<Lease>> {
    OWNED.with(|owned| owned.borrow().get(path).and_then(Weak::upgrade))
}

fn observe(path: &Path, waited: &mut bool) -> Result<Option<Holder>, L2dError> {
    if local(path).is_some() {
        return owner(path, waited);
    }
    let file = open(path)?;
    match file.try_lock() {
        Ok(()) => Ok(None),
        Err(TryLockError::WouldBlock) => owner(path, waited),
        Err(TryLockError::Error(error)) => Err(failed("observe the host lock", error)),
    }
}

fn acquire(
    path: &Path,
    operation: &str,
    choice: BusyChoice,
    waited: &mut bool,
) -> Result<Arc<Lease>, L2dError> {
    if let Some(lease) = local(path) {
        return Ok(lease);
    }
    let mut file = open(path)?;
    let mut announced = false;
    loop {
        match file.try_lock() {
            Ok(()) => break,
            Err(TryLockError::WouldBlock) => {
                let Some(holder) = owner(path, waited)? else {
                    continue;
                };
                let reason = format!(
                    "{} (process {}) holds {}; requested {operation}",
                    holder.operation,
                    holder.process_id,
                    path.file_stem().unwrap().to_string_lossy()
                );
                if choice == BusyChoice::Refuse {
                    return Err(L2dError::Refused { reason });
                }
                if !announced {
                    eprintln!("waiting: {reason}");
                    announced = true;
                }
                *waited = true;
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(TryLockError::Error(error)) => return Err(failed("take the host lock", error)),
        }
    }
    static LEASES: AtomicU64 = AtomicU64::new(0);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| failed("timestamp the lease", e))?
        .as_nanos();
    let lease_file = format!(
        "lease-{}-{stamp}-{}.lock",
        std::process::id(),
        LEASES.fetch_add(1, Ordering::Relaxed)
    );
    let witness_path = path.with_file_name(&lease_file);
    let witness = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&witness_path)
        .map_err(|e| failed("create the lease witness", e))?;
    witness
        .try_lock()
        .map_err(|e| failed("lock the lease witness", e))?;
    let holder = Holder {
        process_id: std::process::id(),
        operation: operation.to_string(),
        lease_file,
    };
    let metadata = path.with_extension("holder.json");
    let bytes = serde_json::to_vec(&holder).map_err(|e| failed("describe the lock holder", e))?;
    let publication = witness_path.with_extension("publishing");
    fs::write(&publication, bytes).map_err(|e| failed("write the lock holder", e))?;
    fs::rename(&publication, &metadata).map_err(|e| failed("publish the lock holder", e))?;
    // Force any opening/locking failures to occur while admission is still fallible.
    file.flush().map_err(|e| failed("flush the host lock", e))?;
    let lease = Arc::new(Lease {
        _file: file,
        witness: Some(witness),
        witness_path,
        metadata,
    });
    OWNED.with(|owned| {
        owned
            .borrow_mut()
            .insert(path.to_path_buf(), Arc::downgrade(&lease));
    });
    Ok(lease)
}

impl ResourceGuard {
    /// Gate plus final accounting. The final record is written even when an operation
    /// returns an error; it describes resources, not a successful result. The caller
    /// calls `finish` before reporting success so a failed record write is returned.
    pub fn tracked(
        operation: &str,
        choice: BusyChoice,
        usage: MemoryUsage,
    ) -> Result<Self, L2dError> {
        Self::enter(operation, choice)?.with_accounting(operation, usage)
    }

    /// Account a phase of this admitted operation, retaining its actual wait and
    /// initial holder observations. The returned guard shares ownership of the gates;
    /// it does not perform a new admission or describe the caller as its own blocker.
    pub fn track(&self, operation: &str, usage: MemoryUsage) -> Result<Self, L2dError> {
        Self {
            _host: self._host.clone(),
            _npu: self._npu.clone(),
            observation: self.observation.clone(),
            waited: self.waited,
            _thread: std::marker::PhantomData,
            record: None,
        }
        .with_accounting(operation, usage)
    }

    fn with_accounting(mut self, operation: &str, usage: MemoryUsage) -> Result<Self, L2dError> {
        if operation.is_empty() || operation.len() > 512 {
            return Err(failed("name the operation", "expected 1 to 512 bytes"));
        }
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join("log2decision-resource-records");
        fs::create_dir_all(&directory)
            .map_err(|e| failed("create the resource record directory", e))?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| failed("timestamp the resource record", e))?
            .as_nanos();
        let path = directory.join(format!(
            "{}-{stamp}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        self.record = Some((operation.into(), usage, path));
        Ok(self)
    }

    pub fn record_path(&self) -> Option<&Path> {
        self.record.as_ref().map(|(_, _, path)| path.as_path())
    }

    fn write_record(&self) -> Result<(), L2dError> {
        let Some((operation, usage, path)) = &self.record else {
            return Ok(());
        };
        let report = ResourceReport {
            operation: operation.clone(),
            process_id: std::process::id(),
            gate_held: true,
            observation: self.observation.clone(),
            waited_for_host: self.waited,
            memory: usage.report(),
            ended_while_panicking: std::thread::panicking(),
        };
        let file = File::create(path).map_err(|e| failed("create the final resource record", e))?;
        serde_json::to_writer_pretty(file, &report)
            .map_err(|e| failed("write the final resource record", e))
    }

    pub fn finish(mut self) -> Result<(), L2dError> {
        self.write_record()?;
        self.record = None;
        Ok(())
    }
    pub fn enter(operation: &str, choice: BusyChoice) -> Result<Self, L2dError> {
        Self::at(&root(), operation, choice, false)
    }

    pub fn npu(operation: &str, choice: BusyChoice) -> Result<Self, L2dError> {
        Self::at(&root(), operation, choice, true)
    }

    /// A separate namespace for isolated process tests; production always uses `enter`
    /// or `npu`, whose namespace does not depend on the selected project or worktree.
    #[doc(hidden)]
    pub fn at(
        directory: &Path,
        operation: &str,
        choice: BusyChoice,
        npu: bool,
    ) -> Result<Self, L2dError> {
        if operation.is_empty() || operation.len() > 512 {
            return Err(failed("name the operation", "expected 1 to 512 bytes"));
        }
        fs::create_dir_all(directory).map_err(|e| failed("create the lock directory", e))?;
        let directory =
            fs::canonicalize(directory).map_err(|e| failed("resolve the lock directory", e))?;
        let host_path = directory.join("host.lock");
        let npu_path = directory.join("npu.lock");
        let mut waited = false;
        let observation = Observation {
            free_memory_bytes: free_physical_memory()?,
            host_holder: observe(&host_path, &mut waited)?,
            npu_holder: observe(&npu_path, &mut waited)?,
            external_npu_utilization: "unknown".into(),
        };
        let host = acquire(&host_path, operation, choice, &mut waited)?;
        let npu = if npu {
            Some(acquire(&npu_path, operation, choice, &mut waited)?)
        } else {
            None
        };
        Ok(Self {
            _host: host,
            _npu: npu,
            observation,
            waited,
            _thread: std::marker::PhantomData,
            record: None,
        })
    }
}

impl Drop for ResourceGuard {
    fn drop(&mut self) {
        if let Err(error) = self.write_record() {
            eprintln!("{error}");
        }
    }
}
