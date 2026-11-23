//! A local handoff saves operations and evidence digests. Raw log text stays in the source.

use crate::flow::Receipt;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use tempfile::NamedTempFile;

const MAX_BYTES: u64 = 1024 * 1024;
const LOCK_NAME: &str = ".l2d-triager-receipts.lock";

pub fn load(path: &Path) -> Result<Receipt, String> {
    load_versioned(path).map(|(receipt, _)| receipt)
}

pub fn load_versioned(path: &Path) -> Result<(Receipt, String), String> {
    let file = File::open(path).map_err(|e| format!("opening receipt {}: {e}", path.display()))?;
    let meta = file.metadata().map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > MAX_BYTES {
        return Err("expected a regular native receipt of at most 1 MiB".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err("receipt exceeds 1 MiB".into());
    }
    let receipt = serde_json::from_slice(&bytes)
        .map_err(|e| format!("invalid native receipt {}: {e}", path.display()))?;
    Ok((receipt, format!("{:x}", Sha256::digest(&bytes))))
}

struct ReceiptWriter<'a> {
    file: &'a mut File,
    sha: Sha256,
    written: u64,
}

impl Write for ReceiptWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() as u64 > MAX_BYTES - self.written {
            return Err(io::Error::other("receipt exceeds 1 MiB"));
        }
        let n = self.file.write(bytes)?;
        self.sha.update(&bytes[..n]);
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

/// Create exclusively, or replace the exact revision that the caller last read.
pub fn save(path: &Path, receipt: &Receipt, expected: Option<&str>) -> Result<String, String> {
    if path
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case(LOCK_NAME))
    {
        return Err("choose a receipt path distinct from the directory's receipt lock".into());
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut tmp = NamedTempFile::new_in(parent).map_err(|e| format!("creating receipt: {e}"))?;
    let mut writer = BufWriter::new(ReceiptWriter {
        file: tmp.as_file_mut(),
        written: 0,
        sha: Sha256::new(),
    });
    serde_json::to_writer(&mut writer, receipt).map_err(|e| e.to_string())?;
    writer.write_all(b"\n").map_err(|e| e.to_string())?;
    let writer = writer.into_inner().map_err(|e| e.to_string())?;
    let revision = format!("{:x}", writer.sha.finalize());
    tmp.as_file().sync_all().map_err(|e| e.to_string())?;
    if let Some(expected) = expected {
        // All cooperating writers lock the same stable file. Renaming the receipt
        // replaces its inode, so locking the receipt itself would permit lost updates.
        // Leave the lock file in place; closing the handle releases the advisory lock.
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(parent.join(LOCK_NAME))
            .map_err(|e| format!("opening receipt lock: {e}"))?;
        lock.lock()
            .map_err(|e| format!("locking receipt directory: {e}"))?;
        let (previous, actual) = load_versioned(path)?;
        if actual != expected {
            return Err("receipt changed since opening; reopen it or save to a new path".into());
        }
        if previous.version != receipt.version
            || previous.dataset_sha256 != receipt.dataset_sha256
            || previous.c_producer != receipt.c_producer
        {
            return Err(
                "destination belongs to another investigation; choose a new receipt path".into(),
            );
        }
        tmp.persist(path)
            .map_err(|e| format!("updating receipt {}: {e}", path.display()))?;
    } else {
        tmp.persist_noclobber(path)
            .map_err(|e| format!("creating new receipt {}: {e}", path.display()))?;
    }
    Ok(revision)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        flow::{Region, Search, SearchKind, Step, Workflow},
        input,
    };

    #[test]
    fn save_reopen_replace_and_protect_existing_evidence() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("hunt.json");
        let mut flow = Workflow::new(
            input::demo(),
            8,
            Search {
                kind: SearchKind::Glob,
                query: "*login*".into(),
            },
        )
        .unwrap();
        let initial = save(&path, &flow.receipt(), None).unwrap();
        assert!(!tmp.path().join(LOCK_NAME).exists());
        assert_eq!(load_versioned(&path).unwrap().1, initial);
        let lock = tmp.path().join(LOCK_NAME);
        std::fs::write(&lock, b"existing content").unwrap();
        flow.drill(Step::Region(Region::AOnly)).unwrap();
        let revision = save(&path, &flow.receipt(), Some(&initial)).unwrap();
        assert_ne!(initial, revision);
        assert_eq!(std::fs::read(lock).unwrap(), b"existing content");
        assert_eq!(
            Workflow::reopen(input::demo(), &load(&path).unwrap())
                .unwrap()
                .view()
                .counts
                .originals,
            6
        );
        let saved = std::fs::read(&path).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&saved)), revision);
        // Save As exclusively creates; even an identical receipt must not replace the target.
        assert!(save(&path, &flow.receipt(), None).is_err());
        assert!(save(&path, &flow.receipt(), Some(&initial)).is_err());
        let mut foreign = flow.receipt();
        foreign.dataset_sha256 = "0".repeat(64);
        assert!(save(&path, &foreign, Some(&revision)).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), saved);
        std::fs::write(&path, b"original log evidence").unwrap();
        assert!(save(&path, &flow.receipt(), None).is_err());
        assert!(save(&path, &flow.receipt(), Some(&revision)).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"original log evidence");
    }

    #[test]
    fn concurrent_sessions_with_the_same_revision_have_one_winner() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("concurrent.json");
        let mut flow = Workflow::new(
            input::demo(),
            8,
            Search {
                kind: SearchKind::Glob,
                query: "*login*".into(),
            },
        )
        .unwrap();
        let base = save(&path, &flow.receipt(), None).unwrap();
        flow.drill(Step::Region(Region::AOnly)).unwrap();
        let left = flow.receipt();
        flow.back();
        flow.drill(Step::Region(Region::Both)).unwrap();
        let right = flow.receipt();
        let gate = std::sync::Barrier::new(2);
        let update = |receipt: &Receipt| {
            gate.wait();
            save(&path, receipt, Some(&base))
        };
        let (a, b) = std::thread::scope(|threads| {
            let a = threads.spawn(|| update(&left));
            let b = update(&right);
            (a.join().unwrap(), b)
        });
        let (winner, loser) = match (a, b) {
            (Ok(winner), Err(loser)) | (Err(loser), Ok(winner)) => (winner, loser),
            _ => panic!("one writer must reject the stale revision"),
        };
        assert!(loser.contains("changed since opening"));
        assert_eq!(load_versioned(&path).unwrap().1, winner);
    }
}
