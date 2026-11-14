//! Batches sized to the free memory of the host (requirement 12).
//!
//! # How an operation uses it
//!
//! 1. [`MemoryBudget::observe`] reads the free physical memory once, before anything is
//!    loaded, and derives the memory limit (the analyst's, or a share of the free
//!    memory) and the reserve (a share of the free memory that is left to other
//!    programs).
//! 2. The operation estimates its bytes per row ([`estimate_rows`]: source bytes per row
//!    from a sample of the first rows, times an expansion factor the operation
//!    declares, plus a fixed cost per row: [`BatchCost`]) and asks for a [`BatchPlan`]
//!    with [`MemoryBudget::plan`]. The first batch is the maximum batch size, or what
//!    the limit leaves when that is less.
//! 3. Before every batch the operation calls [`MemoryBudget::next_batch_rows`], which
//!    reads the free memory again, halves the batch below the reserve, doubles it when
//!    the memory has recovered, and records every change in the plan with its cause.
//! 4. An operation that needs all rows at once calls [`MemoryBudget::require_whole`]
//!    with its estimate before it loads anything.
//!
//! [`Batcher`] packages steps 2 and 3 for an operation that walks a source record by
//! record.
//!
//! # What the limit covers
//!
//! The memory limit is for the whole process. [`PROCESS_BASELINE_BYTES`] of it are kept
//! for the tool itself, so a limit that is not above that figure is rejected. What an
//! operation needs between batches ([`MemoryBudget::hold`]: for a read, the peak of
//! keeping and merging the rows and skipped records collected so far, and the fixed
//! cost of a batch) is taken off the rest. A batch may take what is then left.
//!
//! # Refusals
//!
//! A batch is never started when its estimated memory exceeds what is available: the
//! free memory just read, or what is left of the limit when that is less. The batch
//! shrinks until it fits; when the smallest allowed batch does not fit, or what the
//! operation holds alone exceeds the limit, the operation ends with
//! [`L2dError::Refused`], which states the memory needed, the memory available and,
//! mid-run, the rows completed. A refusal is always preferred to exceeding the limit:
//! the estimates are set above what was measured, so an operation may be refused
//! although it would just have fitted.
//!
//! # Free memory
//!
//! The figure is the physical memory the operating system reports as available
//! ([`free_physical_memory`]). The reading is injectable
//! ([`MemoryBudget::observe_with`]), so that tests can let it fall and recover.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::L2dError;

/// The largest batch when the analyst sets none.
pub const DEFAULT_MAX_BATCH_ROWS: u64 = 65_536;
/// The smallest batch the tool shrinks to. An analyst's batch size below it is its own
/// minimum.
pub const MIN_BATCH_ROWS: u64 = 1_024;
/// The share of the free memory that is the memory limit when the analyst sets none.
pub const DEFAULT_FREE_SHARE: f32 = 0.5;
/// The share of the free memory observed at the start that is kept in reserve.
pub const DEFAULT_RESERVE_SHARE: f32 = 0.25;

/// The share of every memory limit that is kept for the tool itself: program code,
/// stacks, the thread pool of the data layer and its buffers, none of which grows with
/// the data. Measured on the reference host (release build, peak working set of reads
/// whose data is negligible): 9 MiB for one line, 20 to 22 MiB for 1,000,000 lines
/// with 1,000 distinct texts. A limit that is not above this figure cannot be kept and
/// is rejected ([`MemoryBudget::observe`]).
pub const PROCESS_BASELINE_BYTES: u64 = 32 * 1024 * 1024;

/// How many of the first rows are sampled for the bytes per row, at most.
pub const SAMPLE_ROWS: u64 = 256;
/// How many source bytes the sample of the first rows may take, one row excepted.
pub const SAMPLE_BYTES: u64 = 1024 * 1024;

/// What the analyst may set; `None` leaves the choice to the tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemorySettings {
    /// `--memory-limit`: the memory limit in bytes, in place of a share of the free
    /// memory.
    pub memory_limit_bytes: Option<u64>,
    /// `--batch-rows`: the largest batch, in place of [`DEFAULT_MAX_BATCH_ROWS`].
    pub max_batch_rows: Option<u64>,
    /// The share of the free memory used as the limit: above 0, at most 1.
    pub free_share: f32,
    /// The share of the free memory kept in reserve: at least 0, below 1.
    pub reserve_share: f32,
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            memory_limit_bytes: None,
            max_batch_rows: None,
            free_share: DEFAULT_FREE_SHARE,
            reserve_share: DEFAULT_RESERVE_SHARE,
        }
    }
}

impl MemorySettings {
    fn validate(&self) -> Result<(), L2dError> {
        let invalid = |detail: String| Err(L2dError::InvalidInput { detail });
        if self.memory_limit_bytes == Some(0) {
            return invalid("the memory limit must be above zero".into());
        }
        if self.max_batch_rows == Some(0) {
            return invalid("the batch size must be at least one row".into());
        }
        if !(self.free_share > 0.0 && self.free_share <= 1.0) {
            return invalid(format!(
                "the share of free memory used as the limit must be above 0 and at most 1, \
                 not {}",
                self.free_share
            ));
        }
        if !(self.reserve_share >= 0.0 && self.reserve_share < 1.0) {
            return invalid(format!(
                "the share of free memory kept in reserve must be at least 0 and below 1, \
                 not {}",
                self.reserve_share
            ));
        }
        Ok(())
    }
}

/// Why the batch size changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchCause {
    /// The free memory had fallen below the reserve: the batch was halved.
    BelowReserve,
    /// The free memory was above twice the reserve again: the batch was doubled.
    Recovered,
    /// The batch would have taken more than the free memory: it was cut until it fits.
    FreeTooSmall,
    /// The batch would have taken more than the memory limit leaves beside what the
    /// tool itself needs and what the operation already holds: it was cut until it fits.
    LimitLessHeld,
}

/// One change of the batch size during an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchChange {
    /// How many rows were completed when the size changed.
    pub at_row: u64,
    pub from_rows: u64,
    pub to_rows: u64,
    /// The free memory read before the batch.
    pub free_bytes: u64,
    /// What the batch could take: the free memory, or what was left of the memory
    /// limit when that was less.
    pub available_bytes: u64,
    /// Which of the two figures caused the change, and how.
    pub cause: BatchCause,
}

impl fmt::Display for BatchChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "at row {}: {} to {} rows, ",
            self.at_row, self.from_rows, self.to_rows
        )?;
        let free = human(self.free_bytes);
        match self.cause {
            BatchCause::BelowReserve => write!(f, "{free} free is below the reserve"),
            BatchCause::Recovered => write!(f, "{free} free is above twice the reserve"),
            BatchCause::FreeTooSmall => write!(f, "{free} free is less than the batch needs"),
            BatchCause::LimitLessHeld => write!(
                f,
                "{} left of the memory limit beside the rows held ({free} free)",
                human(self.available_bytes)
            ),
        }
    }
}

/// What was observed and chosen for one operation, and how the batch size moved.
///
/// Serializable for the run record ([`crate::RunRecord::record_batch_plan`]); its
/// `Display` form is one line for the analyst.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchPlan {
    /// The free physical memory observed before loading.
    pub free_bytes: u64,
    /// The memory limit applied.
    pub limit_bytes: u64,
    /// Free memory below this halves the next batch.
    pub reserve_bytes: u64,
    /// The estimated memory one row takes while it is processed.
    pub bytes_per_row: u64,
    /// The first batch size, which is also the largest the operation grows back to.
    pub batch_rows: u64,
    /// Whether the analyst set the memory limit or the batch size.
    pub overridden: bool,
    /// Every change of the batch size, in order.
    pub changes: Vec<BatchChange>,
    /// The smallest batch size used; 0 as long as no batch was started.
    pub min_rows_used: u64,
    /// The largest batch size used; 0 as long as no batch was started.
    pub max_rows_used: u64,
}

impl BatchPlan {
    /// The batch size in force: the last change, or the first batch.
    pub fn current_rows(&self) -> u64 {
        self.changes
            .last()
            .map_or(self.batch_rows, |change| change.to_rows)
    }
}

impl fmt::Display for BatchPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "free memory {}, memory limit {}, reserve {}, {} bytes per row, batches of {} rows \
             ({})",
            human(self.free_bytes),
            human(self.limit_bytes),
            human(self.reserve_bytes),
            self.bytes_per_row,
            self.batch_rows,
            if self.overridden {
                "limit or batch size set by the analyst"
            } else {
                "chosen from the free memory"
            },
        )?;
        match self.changes.len() {
            0 => write!(f, ", no change of the batch size")?,
            count => {
                write!(f, ", {count} change{} (", if count == 1 { "" } else { "s" })?;
                for (index, change) in self.changes.iter().enumerate() {
                    if index > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{change}")?;
                }
                write!(f, ")")?;
            }
        }
        if self.max_rows_used == 0 {
            write!(f, ", no batch read")
        } else {
            write!(
                f,
                ", smallest {} rows, largest {} rows",
                self.min_rows_used, self.max_rows_used
            )
        }
    }
}

/// `bytes` in the largest binary unit that keeps it at or above one, with one decimal.
fn human(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

/// `bytes` for a message that must state the figure: readable and exact.
pub(crate) fn exact(bytes: u64) -> String {
    format!("{} ({bytes} bytes)", human(bytes))
}

/// What a sample of the first rows says about a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowEstimate {
    /// The estimated number of rows in the whole source.
    pub total_rows: u64,
    /// The estimated memory one row takes while the operation processes it.
    pub bytes_per_row: u64,
}

/// Estimates rows and bytes per row of a source of `source_bytes` bytes from a sample:
/// the first `sample_rows` rows, which take `sample_bytes` bytes in the source.
///
/// `expansion` is declared by the operation: how many bytes of memory it needs per byte
/// of source while it processes a batch (parsing: [`crate::input::PARSE_EXPANSION`]).
/// Both figures are rounded up.
pub fn estimate_rows(
    source_bytes: u64,
    sample_bytes: u64,
    sample_rows: u64,
    expansion: f64,
) -> RowEstimate {
    if sample_rows == 0 {
        return RowEstimate {
            total_rows: 0,
            bytes_per_row: 0,
        };
    }
    let source_bytes_per_row = sample_bytes.div_ceil(sample_rows).max(1);
    let total_rows = if sample_bytes >= source_bytes {
        sample_rows
    } else {
        source_bytes.div_ceil(source_bytes_per_row).max(sample_rows)
    };
    RowEstimate {
        total_rows,
        bytes_per_row: (source_bytes_per_row as f64 * expansion).ceil() as u64,
    }
}

/// The physical memory the operating system reports as available, in bytes.
#[cfg(windows)]
pub fn free_physical_memory() -> Result<u64, L2dError> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    // SAFETY: `status` is a valid, writable MEMORYSTATUSEX whose `dwLength` is set to
    // its size, as GlobalMemoryStatusEx requires; the call writes nothing else.
    let succeeded = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
    if succeeded {
        Ok(status.ullAvailPhys)
    } else {
        Err(L2dError::Runtime {
            detail: format!(
                "reading the free physical memory failed: {}",
                std::io::Error::last_os_error()
            ),
        })
    }
}

/// The physical memory the operating system reports as available, in bytes
/// (`MemAvailable` of `/proc/meminfo`).
#[cfg(target_os = "linux")]
pub fn free_physical_memory() -> Result<u64, L2dError> {
    let failed = |what: String| L2dError::Runtime {
        detail: format!("reading the free physical memory failed: {what}"),
    };
    let meminfo =
        std::fs::read_to_string("/proc/meminfo").map_err(|error| failed(error.to_string()))?;
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.trim().strip_suffix("kB"))
        .and_then(|kib| kib.trim().parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024))
        .ok_or_else(|| failed("/proc/meminfo has no MemAvailable line in kB".into()))
}

/// Not available on this operating system: the caller must supply a memory reading
/// ([`MemoryBudget::observe_with`]).
#[cfg(not(any(windows, target_os = "linux")))]
pub fn free_physical_memory() -> Result<u64, L2dError> {
    Err(L2dError::Runtime {
        detail: "reading the free physical memory is not supported on this operating system".into(),
    })
}

/// Reads a memory size as the analyst types it: a whole number of bytes above zero,
/// optionally followed by `B`, `K`/`KB`/`KiB`, `M`/`MB`/`MiB`, `G`/`GB`/`GiB` or
/// `T`/`TB`/`TiB` in any case. All units are multiples of 1,024, as memory is counted.
pub fn parse_memory_size(text: &str) -> Result<u64, L2dError> {
    let trimmed = text.trim();
    let digits_end = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (digits, suffix) = trimmed.split_at(digits_end);
    let unit: Option<u64> = match suffix.trim_start().to_ascii_lowercase().as_str() {
        "" | "b" => Some(1),
        "k" | "kb" | "kib" => Some(1 << 10),
        "m" | "mb" | "mib" => Some(1 << 20),
        "g" | "gb" | "gib" => Some(1 << 30),
        "t" | "tb" | "tib" => Some(1 << 40),
        _ => None,
    };
    digits
        .parse::<u64>()
        .ok()
        .zip(unit)
        .and_then(|(count, unit)| count.checked_mul(unit))
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| L2dError::InvalidInput {
            detail: format!(
                "'{trimmed}' is not a memory size: expected a whole number of bytes above \
                 zero, optionally followed by KB, MB, GB or TB (multiples of 1,024), for \
                 example 512MB or 2GB"
            ),
        })
}

/// The memory an operation may use, observed before it loads anything.
///
/// Of the memory limit, [`PROCESS_BASELINE_BYTES`] are kept for the tool itself; what
/// an operation holds between batches ([`MemoryBudget::hold`]) is taken off the rest;
/// a batch may take what is then left ([`MemoryBudget::left_bytes`]).
pub struct MemoryBudget {
    settings: MemorySettings,
    free_bytes: u64,
    limit_bytes: u64,
    reserve_bytes: u64,
    held_bytes: u64,
    read_free: Box<dyn FnMut() -> Result<u64, L2dError> + Send>,
}

impl fmt::Debug for MemoryBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryBudget")
            .field("settings", &self.settings)
            .field("free_bytes", &self.free_bytes)
            .field("limit_bytes", &self.limit_bytes)
            .field("reserve_bytes", &self.reserve_bytes)
            .field("held_bytes", &self.held_bytes)
            .finish_non_exhaustive()
    }
}

fn share_of(bytes: u64, share: f32) -> u64 {
    (bytes as f64 * f64::from(share)) as u64
}

impl MemoryBudget {
    /// Reads the free physical memory of the host once and derives limit and reserve.
    ///
    /// Settings out of range are [`L2dError::InvalidInput`], and so is an analyst's
    /// memory limit that is not above [`PROCESS_BASELINE_BYTES`]: the option value
    /// cannot be honoured. When the tool's own limit (a share of the free memory) is
    /// not above it, the host has too little free memory and the operation is
    /// [`L2dError::Refused`].
    pub fn observe(settings: &MemorySettings) -> Result<Self, L2dError> {
        Self::observe_with(settings, free_physical_memory)
    }

    /// Like [`MemoryBudget::observe`], with `read_free` in place of the operating
    /// system's figure. It is called once now and once before every batch.
    pub fn observe_with(
        settings: &MemorySettings,
        mut read_free: impl FnMut() -> Result<u64, L2dError> + Send + 'static,
    ) -> Result<Self, L2dError> {
        settings.validate()?;
        let free_bytes = read_free()?;
        let limit_bytes = match settings.memory_limit_bytes {
            Some(limit) if limit <= PROCESS_BASELINE_BYTES => {
                return Err(L2dError::InvalidInput {
                    detail: format!(
                        "the memory limit of {} is not above the {} the tool itself needs \
                         before it reads anything",
                        exact(limit),
                        exact(PROCESS_BASELINE_BYTES),
                    ),
                });
            }
            Some(limit) => limit,
            None => {
                let limit = share_of(free_bytes, settings.free_share);
                if limit <= PROCESS_BASELINE_BYTES {
                    return Err(L2dError::Refused {
                        reason: format!(
                            "{} of physical memory is free, of which the tool may use {}; \
                             that is not above the {} it needs itself before it reads \
                             anything",
                            exact(free_bytes),
                            exact(limit),
                            exact(PROCESS_BASELINE_BYTES),
                        ),
                    });
                }
                limit
            }
        };
        Ok(Self {
            settings: settings.clone(),
            free_bytes,
            limit_bytes,
            reserve_bytes: share_of(free_bytes, settings.reserve_share),
            held_bytes: 0,
            read_free: Box::new(read_free),
        })
    }

    /// The free physical memory observed at the start.
    pub fn free_bytes(&self) -> u64 {
        self.free_bytes
    }

    /// The memory limit: the analyst's, or the configured share of the free memory.
    pub fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }

    /// The reserve: free memory below it halves the next batch.
    pub fn reserve_bytes(&self) -> u64 {
        self.reserve_bytes
    }

    /// What the memory limit leaves for data: the limit less
    /// [`PROCESS_BASELINE_BYTES`].
    pub fn usable_bytes(&self) -> u64 {
        self.limit_bytes.saturating_sub(PROCESS_BASELINE_BYTES)
    }

    /// What is left for the next batch or for a whole dataset: the usable memory less
    /// what the operation holds.
    pub fn left_bytes(&self) -> u64 {
        self.usable_bytes().saturating_sub(self.held_bytes)
    }

    /// What the operation has said it holds ([`MemoryBudget::hold`]). An operation that
    /// holds a share of its own on top of the caller's reads this first and puts it
    /// back when it is done; the figure can be more than the limit leaves.
    pub fn held_bytes(&self) -> u64 {
        self.held_bytes
    }

    /// The largest batch: the analyst's, or [`DEFAULT_MAX_BATCH_ROWS`].
    pub fn max_batch_rows(&self) -> u64 {
        self.settings
            .max_batch_rows
            .unwrap_or(DEFAULT_MAX_BATCH_ROWS)
    }

    /// The smallest batch: [`MIN_BATCH_ROWS`], or the largest batch when that is less.
    pub fn min_batch_rows(&self) -> u64 {
        MIN_BATCH_ROWS.min(self.max_batch_rows())
    }

    /// Tells the budget how much memory the operation needs between batches (for a
    /// read: what it takes to keep and to merge the deduplicated rows and the skipped
    /// records so far). It counts against the limit, so the batches shrink as it grows
    /// and the operation is refused when nothing is left for the smallest batch.
    pub fn hold(&mut self, bytes: u64) {
        self.held_bytes = bytes;
    }

    /// [`L2dError::Refused`] when what the operation holds is more than the usable
    /// memory; `rows_done` rows are completed.
    pub fn require_held(&self, rows_done: u64) -> Result<(), L2dError> {
        if self.held_bytes <= self.usable_bytes() {
            return Ok(());
        }
        Err(L2dError::Refused {
            reason: format!(
                "what is held of the rows read so far needs {} to be kept and merged, but \
                 the memory limit of {} leaves {} beside the {} kept for the tool itself; \
                 stopped after {rows_done} rows were completed",
                exact(self.held_bytes),
                exact(self.limit_bytes),
                exact(self.usable_bytes()),
                exact(PROCESS_BASELINE_BYTES),
            ),
        })
    }

    /// The plan for an operation whose rows take `bytes_per_row` bytes each.
    ///
    /// The first batch is the largest batch, or as many rows as fit what is left of the
    /// limit when that is less. [`L2dError::Refused`] when the smallest batch does not
    /// fit.
    ///
    /// A source of fewer rows than the smallest batch is judged by its own rows
    /// (`total_rows_estimate`): three very long records are read when the three fit,
    /// although 1,024 of them would not. The first batch is then below
    /// [`MIN_BATCH_ROWS`] and is not shrunk further.
    pub fn plan(
        &self,
        total_rows_estimate: u64,
        bytes_per_row: u64,
    ) -> Result<BatchPlan, L2dError> {
        let smallest = self.min_batch_rows().min(total_rows_estimate.max(1));
        let left = self.left_bytes();
        let fitting = left.checked_div(bytes_per_row).unwrap_or(u64::MAX);
        if fitting < smallest {
            return Err(L2dError::Refused {
                reason: format!(
                    "the smallest batch of {smallest} rows needs {} at {bytes_per_row} bytes \
                     per row, but {} is left of the memory limit of {} ({} is kept for the \
                     tool itself, {} is held; {} of physical memory was free); none of the \
                     about {total_rows_estimate} rows was loaded",
                    exact(smallest.saturating_mul(bytes_per_row)),
                    exact(left),
                    exact(self.limit_bytes),
                    exact(PROCESS_BASELINE_BYTES),
                    exact(self.held_bytes),
                    exact(self.free_bytes),
                ),
            });
        }
        Ok(BatchPlan {
            free_bytes: self.free_bytes,
            limit_bytes: self.limit_bytes,
            reserve_bytes: self.reserve_bytes,
            bytes_per_row,
            batch_rows: self.max_batch_rows().min(fitting),
            overridden: self.settings.memory_limit_bytes.is_some()
                || self.settings.max_batch_rows.is_some(),
            changes: Vec::new(),
            min_rows_used: 0,
            max_rows_used: 0,
        })
    }

    /// The size of the batch about to start, after `rows_done` rows are completed.
    ///
    /// Reads the free memory again. Below the reserve the batch is halved, down to the
    /// smallest batch; above twice the reserve it is doubled, up to the first batch of
    /// the plan. A batch whose estimated memory exceeds what is available (the free
    /// memory just read, or what is left of the limit when that is less) is halved
    /// until it fits. A change is recorded in the plan with its cause and the figures
    /// that caused it.
    ///
    /// [`L2dError::Refused`] when the smallest batch does not fit; the message states
    /// the memory needed and available and the rows completed.
    pub fn next_batch_rows(
        &mut self,
        plan: &mut BatchPlan,
        rows_done: u64,
    ) -> Result<u64, L2dError> {
        let free = (self.read_free)()?;
        let current = plan.current_rows();
        let largest = plan.batch_rows;
        let smallest = self.min_batch_rows().min(largest);

        let mut rows = current;
        let mut cause = None;
        if free < self.reserve_bytes {
            rows = (rows / 2).max(smallest);
            cause = Some(BatchCause::BelowReserve);
        } else if free > self.reserve_bytes.saturating_mul(2) {
            rows = rows.saturating_mul(2).min(largest);
            cause = Some(BatchCause::Recovered);
        }
        let left = self.left_bytes();
        let available = free.min(left);
        let needed = |rows: u64| rows.saturating_mul(plan.bytes_per_row);
        if needed(rows) > available {
            cause = Some(if left < free {
                BatchCause::LimitLessHeld
            } else {
                BatchCause::FreeTooSmall
            });
            while rows > smallest && needed(rows) > available {
                rows = (rows / 2).max(smallest);
            }
        }
        if needed(rows) > available {
            return Err(L2dError::Refused {
                reason: format!(
                    "the smallest batch of {rows} rows needs {} at {} bytes per row, but only \
                     {} is available (free physical memory {}; {} left of the memory limit \
                     of {}, of which {} is kept for the tool itself and {} is held by the \
                     rows read so far); stopped after {rows_done} rows were completed",
                    exact(needed(rows)),
                    plan.bytes_per_row,
                    exact(available),
                    exact(free),
                    exact(left),
                    exact(self.limit_bytes),
                    exact(PROCESS_BASELINE_BYTES),
                    exact(self.held_bytes),
                ),
            });
        }

        if rows != current
            && let Some(cause) = cause
        {
            plan.changes.push(BatchChange {
                at_row: rows_done,
                from_rows: current,
                to_rows: rows,
                free_bytes: free,
                available_bytes: available,
                cause,
            });
        }
        plan.min_rows_used = match plan.min_rows_used {
            0 => rows,
            used => used.min(rows),
        };
        plan.max_rows_used = plan.max_rows_used.max(rows);
        Ok(rows)
    }

    /// The check of an operation that needs the whole dataset in memory at once.
    ///
    /// [`L2dError::Refused`], stating the estimate and the limit, when
    /// `estimated_bytes` exceeds what is left of the memory limit. Call it before
    /// loading.
    pub fn require_whole(&self, estimated_bytes: u64, operation: &str) -> Result<(), L2dError> {
        let left = self.left_bytes();
        if estimated_bytes <= left {
            return Ok(());
        }
        Err(L2dError::Refused {
            reason: format!(
                "{operation} needs the whole dataset in memory at once: estimated {}, but \
                 {} is left of the memory limit of {} ({} is kept for the tool itself, {} is \
                 held; {} of physical memory was free); nothing was loaded",
                exact(estimated_bytes),
                exact(left),
                exact(self.limit_bytes),
                exact(PROCESS_BASELINE_BYTES),
                exact(self.held_bytes),
                exact(self.free_bytes),
            ),
        })
    }
}

/// What an operation declares about the memory it needs for one batch.
///
/// The memory of a batch of `n` rows that take `s` source bytes is estimated as
/// `batch_bytes + n * row_bytes + s * expansion`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchCost {
    /// Bytes of memory per byte of source.
    pub expansion: f64,
    /// Bytes of memory per row, whatever its length.
    pub row_bytes: u64,
    /// Bytes of memory per batch, whatever its size.
    pub batch_bytes: u64,
}

/// The most one record of a source may take, and the limit it follows from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordLimit {
    /// The most source bytes one record may have.
    pub max_bytes: u64,
    /// The memory limit.
    pub limit_bytes: u64,
}

impl RecordLimit {
    /// The refusal for the record at `position` (0: the header row), which is larger
    /// than allowed. Nothing of it is kept.
    pub fn refusal(&self, position: u64) -> L2dError {
        let record = match position {
            0 => "the header row".to_string(),
            position => format!("the record at position {position}"),
        };
        L2dError::Refused {
            reason: format!(
                "{record} is larger than {}, the most one record may take within the memory \
                 limit of {}; stopped after {} records were read",
                exact(self.max_bytes),
                exact(self.limit_bytes),
                position.saturating_sub(1),
            ),
        }
    }
}

/// Cuts a source that is walked record by record into batches (steps 2 and 3 of the
/// module description).
///
/// The first records, at most [`SAMPLE_ROWS`] and about [`SAMPLE_BYTES`], are the
/// sample for the bytes per row; they are part of the first batch. A batch ends when it
/// holds the rows [`MemoryBudget::next_batch_rows`] chose for it, or earlier when its
/// records turn out so much larger than the sample that their estimated memory reaches
/// what the limit leaves. The free memory is read when the first record of a batch
/// arrives, so it is read once for every batch that has rows.
///
/// The fixed memory of a batch ([`BatchCost::batch_bytes`]) is held from the start.
pub struct Batcher<'b> {
    budget: &'b mut MemoryBudget,
    source_bytes: u64,
    cost: BatchCost,
    sample_rows: u64,
    plan: Option<BatchPlan>,
    /// The size chosen for the current batch; 0 before its first record.
    target_rows: u64,
    batch_rows: u64,
    batch_bytes: u64,
    rows_done: u64,
}

impl<'b> Batcher<'b> {
    /// A batcher for a source of `source_bytes` bytes, processed by an operation with
    /// the memory needs `cost`.
    pub fn new(budget: &'b mut MemoryBudget, source_bytes: u64, cost: BatchCost) -> Self {
        let sample_rows = SAMPLE_ROWS.min(budget.max_batch_rows());
        budget.hold(cost.batch_bytes);
        Self {
            budget,
            source_bytes,
            cost,
            sample_rows,
            plan: None,
            target_rows: 0,
            batch_rows: 0,
            batch_bytes: 0,
            rows_done: 0,
        }
    }

    /// The most one record may take: its estimated memory must fit what the limit
    /// leaves.
    pub fn record_limit(&self) -> RecordLimit {
        let left = self.budget.left_bytes().saturating_sub(self.cost.row_bytes);
        RecordLimit {
            max_bytes: ((left as f64 / self.cost.expansion) as u64).max(1),
            limit_bytes: self.budget.limit_bytes(),
        }
    }

    /// The budget the batches are cut within, as it stands now.
    pub fn budget(&self) -> &MemoryBudget {
        self.budget
    }

    /// The rows of the batches processed so far.
    pub fn rows_done(&self) -> u64 {
        self.rows_done
    }

    /// Notes one record of `source_bytes` bytes, read or skipped, as part of the
    /// current batch. `true` means the batch is complete: process it, then call
    /// [`Batcher::processed`].
    pub fn record(&mut self, source_bytes: u64) -> Result<bool, L2dError> {
        self.batch_rows += 1;
        self.batch_bytes += source_bytes;
        if self.plan.is_none() {
            if self.batch_rows < self.sample_rows && self.batch_bytes < SAMPLE_BYTES {
                return Ok(false);
            }
            self.make_plan()?;
        }
        self.choose_size()?;
        let estimated = self.batch_bytes as f64 * self.cost.expansion
            + (self.batch_rows * self.cost.row_bytes) as f64;
        Ok(self.batch_rows >= self.target_rows || estimated >= self.budget.left_bytes() as f64)
    }

    /// Notes that the current batch was processed and that the operation now needs
    /// `held_bytes` bytes of memory between batches.
    ///
    /// [`L2dError::Refused`] when that is more than the limit leaves; the message
    /// states the rows completed.
    pub fn processed(&mut self, held_bytes: u64) -> Result<(), L2dError> {
        self.rows_done += self.batch_rows;
        self.batch_rows = 0;
        self.batch_bytes = 0;
        self.target_rows = 0;
        self.budget
            .hold(held_bytes.saturating_add(self.cost.batch_bytes));
        self.budget.require_held(self.rows_done)
    }

    /// Notes the end of the source. `true` means records are left in a last, partial
    /// batch: process it, then call [`Batcher::processed`].
    pub fn end(&mut self) -> Result<bool, L2dError> {
        if self.plan.is_none() {
            self.make_plan()?;
        }
        if self.batch_rows > 0 {
            self.choose_size()?;
        }
        Ok(self.batch_rows > 0)
    }

    /// The plan with the changes recorded during the run.
    pub fn into_plan(mut self) -> Result<BatchPlan, L2dError> {
        if self.plan.is_none() {
            self.make_plan()?;
        }
        self.plan.ok_or_else(|| L2dError::Runtime {
            detail: "the batch plan was not made".into(),
        })
    }

    fn make_plan(&mut self) -> Result<(), L2dError> {
        let estimate = estimate_rows(
            self.source_bytes,
            self.batch_bytes,
            self.batch_rows,
            self.cost.expansion,
        );
        let bytes_per_row = match estimate.total_rows {
            0 => 0,
            _ => estimate.bytes_per_row.saturating_add(self.cost.row_bytes),
        };
        self.plan = Some(self.budget.plan(estimate.total_rows, bytes_per_row)?);
        Ok(())
    }

    fn choose_size(&mut self) -> Result<(), L2dError> {
        if self.target_rows == 0
            && let Some(plan) = &mut self.plan
        {
            self.target_rows = self.budget.next_batch_rows(plan, self.rows_done)?;
        }
        Ok(())
    }
}
