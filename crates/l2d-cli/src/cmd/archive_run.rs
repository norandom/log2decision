//! What `l2d ingest` and `l2d baseline score` share when they read a zip archive with
//! deduplication on disk (requirement 13.12; tasks 1.17 and 1.20).
//!
//! # The work directory
//!
//! `--work-dir DIR`, or `<project>/work/<command>-<UTC time>/` when it is not given
//! (`ingest`, which keeps nothing there, adds its process id to the name, so that two
//! runs in one second do not meet). The directory must be empty or absent; the run
//! makes it and writes nothing outside it. The files of deduplication on disk are in
//! `dedup/` below it, have names the tool makes up, and are removed when the read
//! ends, however it ends.
//!
//! What the run made of the directory is removed again as far as it is empty: the
//! directory itself when it was absent, and the directories above it that were. So a
//! command that writes no result (`ingest`) leaves nothing behind, and a run that is
//! refused before it wrote a result leaves nothing either. A directory that was there
//! before the run stays.
//!
//! # When the disk is used (task 1.20, remediation)
//!
//! `ingest` uses the disk only when it needs it ([`spill_when_needed`]): while the set
//! of distinct texts fits the memory limit nothing is made, no disk is looked at and
//! nothing more is held, as before task 1.20. A `--work-dir` the analyst gives is
//! checked before the read (below a file, not empty) and made only then. When the set
//! is frozen, the directory is made, the free disk is checked, and the read goes on on
//! disk; a refusal there comes during the read, with what was read until then.
//!
//! `baseline score` writes parts from its first batch, so its directory is made and the
//! disk is checked for the parts before the read ([`ArchiveRun::prepare`]); the key of
//! every row is written to `dedup/` from the start, because the occurrence counts of
//! the parts are read out by them, so the check before the read also counts the most
//! that key file can take ([`l2d_core::most_key_file_bytes`]; review round 2 of task
//! 1.20). The disk for deduplication on disk is checked at the
//! freeze, as for `ingest`, together with the whole estimate of the parts again: a run
//! that stays in memory is not refused for the disk deduplication on disk would need.
//!
//! # The free-disk estimate
//!
//! For deduplication on disk the estimate is the unpacked bytes of the selected entries
//! (a text that does not fit the set in memory is written once per batch it occurs
//! in) and the 32 bytes a row on disk takes beside its text, for as many rows as
//! records of 6 bytes would make of those bytes. It is an estimate, not a bound: the
//! rows are not known before the read, and a gzip entry declares its compressed size
//! only. At the freeze it is the same figure, for all the selected entries.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use l2d_core::parts::WorkDir;
use l2d_core::{ArchiveSummary, DiskDedup, FreezeDiskCheck, L2dError, ZipInput};
use serde::Serialize;

use super::common::Context;
use crate::output::human_bytes;

/// `yyyymmddThhmmssZ` of `seconds` since 1970-01-01 UTC.
fn utc_stamp(seconds: u64) -> String {
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Days to a civil date (Howard Hinnant's algorithm), for the proleptic Gregorian
    // calendar.
    let shifted = days as i64 + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_shifted = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_shifted + 2) / 5 + 1;
    let month = if month_shifted < 10 {
        month_shifted + 3
    } else {
        month_shifted - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

/// The directory of this run: the analyst's, or `name` below the project's work
/// directory, with the time of the run in place of `<UTC time>`.
fn run_directory(context: &Context, name: &str, given: Option<&Path>) -> Result<PathBuf, L2dError> {
    if let Some(given) = given {
        return Ok(given.to_path_buf());
    }
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| L2dError::Runtime {
            detail: format!("the clock of the host is before 1970: {error}"),
        })?
        .as_secs();
    Ok(context
        .paths()?
        .work_dir()
        .join(name.replace("<UTC time>", &utc_stamp(seconds))))
}

/// The work directory of a run. When it is let go, what the run made of it is removed
/// as far as it is empty (module documentation).
pub(crate) struct RunDir(WorkDir);

impl RunDir {
    pub(crate) fn work(&self) -> &WorkDir {
        &self.0
    }
}

impl Drop for RunDir {
    fn drop(&mut self) {
        self.0.remove_created();
    }
}

/// Has `input` use the disk only when its set of distinct texts outgrows the memory
/// limit (module documentation), for a command that keeps nothing in the directory and
/// wants no occurrence counts (`ingest`). Nothing is made here.
///
/// `name` is the name of the directory below the project's work directory when the
/// analyst gave none; `<UTC time>` in it becomes the time of the run.
///
/// [`L2dError::InvalidInput`] when the directory the analyst gave would be below a
/// file, [`L2dError::Refused`] when it holds something.
pub(crate) fn spill_when_needed(
    context: &Context,
    input: &mut ZipInput,
    name: &str,
    given: Option<&Path>,
) -> Result<(), L2dError> {
    if let Some(given) = given {
        WorkDir::check(given)?;
    }
    input.spill_when_needed(&run_directory(context, name, given)?, None);
    input.check_disk_at_freeze(FreezeDiskCheck::default());
    Ok(())
}

/// An archive that is read with deduplication on disk in the work directory of its
/// run, for a command that writes result parts there (`baseline score`).
pub(crate) struct ArchiveRun {
    /// The archive. It is let go before the directory, so that its files are gone
    /// when the directory is removed.
    pub(crate) input: ZipInput,
    dir: RunDir,
    /// The free space of the directory's volume before the run.
    disk_free: u64,
    /// What the parts were estimated to need there.
    disk_estimated: u64,
}

impl ArchiveRun {
    /// Makes the work directory of the run, checks the free space of its volume
    /// against what the result parts are estimated to need (`parts`) and the most the
    /// key file of the occurrence counts can take (`keys`), and names the
    /// directory for deduplication on disk to the archive. The disk for deduplication
    /// on disk is checked when the set is frozen, with `parts` again beside it. Nothing
    /// of the archive's entries is read.
    ///
    /// `name` is the name of the directory below the project's work directory when
    /// the analyst gave none; `<UTC time>` in it becomes the time of the run.
    ///
    /// [`L2dError::InvalidInput`] when the directory would be below a file,
    /// [`L2dError::Refused`] when it holds something or the free space is less than
    /// the estimate. What was made of the directory until then is removed again.
    pub(crate) fn prepare(
        context: &Context,
        mut input: ZipInput,
        name: &str,
        given: Option<&Path>,
        parts: u64,
        keys: u64,
    ) -> Result<Self, L2dError> {
        let dir = RunDir(WorkDir::create(&run_directory(context, name, given)?)?);
        let disk_free = dir.work().require_free(&[
            ("the result parts", parts),
            ("the keys of the occurrence counts", keys),
        ])?;
        input.spill_into(&dir.work().path().join("dedup"), None)?;
        input.check_disk_at_freeze(FreezeDiskCheck {
            beside: vec![("the result parts".to_string(), parts)],
            free_bytes: None,
        });
        Ok(Self {
            input,
            dir,
            disk_free,
            disk_estimated: parts.saturating_add(keys),
        })
    }

    /// The work directory.
    pub(crate) fn work(&self) -> &WorkDir {
        self.dir.work()
    }

    /// The free disk space and the estimate of the last check: at the freeze when the
    /// read went on on disk, before the read otherwise.
    pub(crate) fn disk_check(&self) -> (u64, u64) {
        match self.input.freeze_disk() {
            Some(disk) => (disk.free_bytes, disk.estimated_bytes),
            None => (self.disk_free, self.disk_estimated),
        }
    }

    /// Ends the read: the summary of the archive, and the work directory, from which
    /// the files of deduplication on disk are gone.
    pub(crate) fn finish(self) -> (ArchiveSummary, RunDir) {
        let Self { input, dir, .. } = self;
        (input.finish(), dir)
    }
}

/// Whether the deduplication went on on disk: the `deduplication` object of the
/// output.
#[derive(Debug, Serialize)]
pub(crate) struct DedupReport {
    on_disk: bool,
    /// Distinct texts the set in memory held.
    set_texts: u64,
    /// Rows written to disk for deduplication, and their bytes.
    spilled_records: u64,
    spilled_bytes: u64,
    /// Distinct texts among them.
    spilled_distinct: u64,
}

impl DedupReport {
    /// `distinct` are the distinct texts of the archive.
    pub(crate) fn of(dedup: Option<DiskDedup>, distinct: u64) -> Self {
        match dedup {
            Some(dedup) => Self {
                on_disk: true,
                set_texts: dedup.set_texts,
                spilled_records: dedup.spilled_records,
                spilled_bytes: dedup.spilled_bytes,
                spilled_distinct: dedup.spilled_distinct,
            },
            None => Self {
                on_disk: false,
                set_texts: distinct,
                spilled_records: 0,
                spilled_bytes: 0,
                spilled_distinct: 0,
            },
        }
    }

    /// The lines of the text output: where the texts were deduplicated, and the disk
    /// space that was free and estimated.
    pub(crate) fn write_text(
        &self,
        out: &mut dyn Write,
        disk_free: u64,
        disk_estimated: u64,
    ) -> io::Result<()> {
        if self.on_disk {
            writeln!(
                out,
                "  deduplication: on disk after {} distinct texts in memory ({} rows, {}, written \
                 and removed again; {} further distinct texts)",
                self.set_texts,
                self.spilled_records,
                human_bytes(self.spilled_bytes),
                self.spilled_distinct
            )?;
        } else {
            writeln!(out, "  deduplication: in memory")?;
        }
        writeln!(
            out,
            "  disk: {} was free, the run was estimated at {}",
            human_bytes(disk_free),
            human_bytes(disk_estimated)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::utc_stamp;

    #[test]
    fn a_time_is_stamped_as_a_utc_date_and_time() {
        assert_eq!(utc_stamp(0), "19700101T000000Z");
        assert_eq!(utc_stamp(951_782_399), "20000228T235959Z");
        assert_eq!(utc_stamp(951_782_400), "20000229T000000Z");
        assert_eq!(utc_stamp(951_868_800), "20000301T000000Z");
        assert_eq!(utc_stamp(1_794_652_496), "20261114T103456Z");
        assert_eq!(utc_stamp(4_102_444_799), "20991231T235959Z");
    }
}
