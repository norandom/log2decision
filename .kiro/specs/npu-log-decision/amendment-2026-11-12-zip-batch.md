# Amendment draft: zip archives as input, processed in batches under the memory limit

Status: **approved by the owner on 2026-11-12 and merged** into
`requirements.md` (Requirement 13), `design.md` ("ZipInput and result parts")
and `tasks.md` (tasks 1.16, 1.17, 1.19). Where this file and those documents
differ, those documents win. This file is kept as the record of the checks and
of the options considered.

Owner's answers to the open points in section 6: both tasks, 1.16 first; gzip
is the only nested format read; the default work directory is
`<project>/work/<run>/`; parallel reading of entries is a follow-up task
(1.19); the AIT label adapter is a separate amendment later. Later the same
day the owner set the default maximum record length to 64 KiB (this file's
section 4 still says 1 MiB).

Requested by the owner on 2026-11-12: "we should be able to process zipped
files in batch mode, mem capped", with the questions whether Rust has a memory
compression buffer usable with Polars, and whether dataframes can spill over to
disk.

## 1. What was checked

Verified on 2026-11-12:

- `data/ait-lds-v2/russellmitchell.zip` (the smallest AIT testbed, 6.64 GB):
  11,992 entries with data, 13.95 GB uncompressed. The largest entries are
  packet captures of about 1 GB each. Most entries are not logs: 1,842 `.py`,
  1,444 `.yml`, 911 `.ini`, 848 `.conf`. 108 entries are themselves archives
  (`.gz`, `.zip`, `.bz2`, `.xz` or `.tar`).
- Dependency check in a scratch copy of the workspace
  (`D:\BuildArtifacts\log2decision\zip-amendment-check\ws`): with `polars-io`
  features `parquet`, `ipc`, `zstd` added and `zip` 2.6.1
  (`default-features = false`, feature `deflate`), `cargo tree -p l2d-core
  -e normal` and the resolved `Cargo.lock` name none of the crates the network
  guard test lists. 29 packages are added, among them `polars-parquet`, `zip`,
  `flate2`, `zstd`, `zstd-sys`, `lz4`, `lz4-sys`, `brotli`, `snap`.

Not verified:

- Nothing was built. `zstd-sys` and `lz4-sys` compile C code; whether they
  build on the reference host is open.
- The compression methods used inside the AIT archives were not read. The
  design assumes deflate; an entry with another method is skipped and reported.
- Which Polars version first supports what is described under "Options
  considered" was not checked against the 0.52.0 source; the statement about
  the streaming feature comes from the comment in the workspace `Cargo.toml`.

## 2. Options considered

| Option | Verdict |
| --- | --- |
| Compressed buffer in memory (hold batches zstd- or lz4-compressed, decompress on use) | Rejected. Arrow and Polars have no transparent compressed buffer; it would be our own code, costs CPU on every access, and still has a ceiling. |
| Polars streaming engine with spill to disk | Not available here. The facade feature `streaming` links `polars-stream` and through it `reqwest`, `hyper`, `object_store` and `rustls` (workspace `Cargo.toml`), which requirement 10.2 and the guard test forbid. |
| Memory-mapped Arrow IPC files | Possible later for reading parts back. Not needed for the first cut; Parquet parts are smaller on disk. |
| Our own batches (task 1.15) with results written to Parquet parts on disk | **Chosen.** Disk is the spill by design. Memory holds one batch, the decompressor state and the deduplication state. |

## 3. Requirements (proposed Requirement 13)

### Requirement 13: Zip archives as input, in batches

**Objective:** As a log analyst, I want to point the tool at a zip archive of
logs that is larger than my laptop's memory, so that I can work on large
datasets without extracting them and without the tool exhausting memory.

#### Acceptance Criteria

1. When the analyst supplies a zip archive, the Log Triage Tool shall read log
   records from the entries of the archive without extracting the archive to
   disk.
2. Where the analyst gives entry patterns, the Log Triage Tool shall read only
   the entries whose path matches a pattern.
3. When an archive has been read, the Log Triage Tool shall report the number
   of entries read and, for every entry it did not read, the entry's path and
   the reason.
4. If an entry is not text, is encrypted, or uses a compression method the
   tool does not support, then the Log Triage Tool shall skip the entry,
   continue with the next entry, and report it.
5. When an entry is a gzip-compressed text file, the Log Triage Tool shall
   read the log records inside it.
6. If an entry is damaged or fails its checksum, then the Log Triage Tool
   shall stop reading that entry, keep the records read from other entries,
   report the entry, and exit with a non-zero status.
7. The Log Triage Tool shall give each record read from an archive a position
   that names the entry path and the line within the entry.
8. While an archive is read, the Log Triage Tool shall process it in batches
   under Requirement 12, so that its memory use stays within the memory limit
   whatever the size of the archive or of a single entry.
9. If a single log record is longer than the configured maximum record
   length, then the Log Triage Tool shall skip the record and report its
   position and length.
10. The Log Triage Tool shall produce the same records, occurrence counts and
    input summary for an archive as for the same files read unpacked, whatever
    the batch size.
11. When an operation writes per-log results for an input that is processed in
    batches, the Log Triage Tool shall write the results to files in parts as
    the batches complete, and shall state where the parts are.
12. If the deduplication state of an input would exceed the memory limit,
    then the Log Triage Tool shall either continue with deduplication on disk
    or refuse and state the memory needed and the memory available; it shall
    not exceed the limit.
13. The Log Triage Tool shall record for an archive input the archive path,
    the archive's content hash, and for each entry read its path and record
    count.

Changes to existing text if this is approved:

- Boundary Context, in scope: add "Reading log records from zip archives in
  batches".
- Boundary Context, out of scope: unchanged. Evaluation on AIT V2 stays out;
  this amendment only makes such an archive readable. The AIT label files
  (labels kept apart from the logs, by line) need their own input adapter and
  are not part of this amendment.

## 4. Design (proposed additions)

**Crates**

- `zip` 2.x, `default-features = false`, feature `deflate` only. No `aes`,
  `bzip2`, `time` or `zstd` features. Zip64 is needed (archives up to
  24.7 GB) and is part of the crate's core.
- `flate2` (arrives with `zip`) for gzip entries.
- `polars-io` gains the features `parquet` and `zstd` for writing parts. `ipc`
  is not added until something needs it.
- The guard test over `Cargo.lock` stays as it is; the comment at the Polars
  entry in the workspace `Cargo.toml` gains the new features and the date of
  the check.

**Reading (`l2d-core`, next to `InputAdapter`)**

- `ZipInput` opens the archive file, reads the central directory, and walks
  the entries in central-directory order. That order is the record order, so
  results do not depend on timing.
- Each selected entry is read as a stream: decompressor, then the existing
  line splitting and byte-order-mark handling of task 1.3. Nothing is written
  to disk for reading.
- Entry selection: glob patterns over the entry path (`--entry`, repeatable).
  Without a pattern every entry is tried.
- Text check per entry on its first 8 KiB: a NUL byte or invalid UTF-8 marks
  the entry as not text. This is also what keeps packet captures out.
- A gzip entry (by magic bytes, not by name) is read through a second
  decompressor. Other nested archives are skipped and reported.
- Maximum record length: default 1 MiB, option `--max-record-bytes`. The
  store's own limit on text length (task 1.8 remediation) must not be larger.
- The archive's content hash is SHA-256 over the archive file, computed in one
  streaming pass before reading. Entry checksums (CRC-32) are verified by the
  crate when an entry is read to its end.

**Batches and memory**

- The reader yields batches through the `Batcher` of task 1.15. The cost of a
  batch is estimated from the uncompressed bytes of its records. Fixed costs
  held for the whole run: the decompressor window and buffers, and the
  deduplication state.
- Deduplication state, first stage: a set of 128-bit text hashes with their
  occurrence counts, held against the budget and grown batch by batch. When it
  no longer fits, the operation refuses (criterion 12, refusal branch).
- Deduplication state, second stage: when the set would exceed the budget,
  records are written to parts partitioned by hash prefix, and each partition
  is deduplicated on its own. Results are identical to the in-memory path.
- Entries are read one at a time. Reading several entries in parallel is left
  out of this amendment.

**Results in parts**

- An operation that produces per-log results for a batched input writes one
  Parquet part (zstd) per batch into a run directory and a small manifest that
  lists the parts, their row counts and their hashes.
- Location: `--work-dir`, default `<project>/work/<run>/`, which is ignored by
  git. Open point 3 below.
- If the free disk space is less than the estimated size of the parts, the
  operation refuses before it starts and states both figures.

**Contracts (sketch)**

```rust
pub struct ZipInputSettings { pub entries: Vec<String>, pub max_record_bytes: u64 }

pub enum EntrySkip { NotText, Encrypted, UnsupportedMethod(u16), NestedArchive, NoPatternMatch, Damaged(String) }

pub struct EntryReport { pub path: String, pub records: u64, pub skipped: Option<EntrySkip> }

pub struct ArchiveSummary { pub input: InputSummary, pub entries: Vec<EntryReport> }

impl ZipInput {
    pub fn open(path: &Path, settings: &ZipInputSettings) -> Result<Self, L2dError>;
    // Yields frames of at most the planned batch rows; the position column names entry and line.
    pub fn batches<'a>(&'a mut self, budget: &'a mut MemoryBudget) -> impl Iterator<Item = Result<LogsFrame, L2dError>> + 'a;
    pub fn finish(self) -> ArchiveSummary;
}
```

## 5. Tasks (proposed, after 1.12)

- [ ] 1.16 Read zip archives in batches under the memory limit
  - `ZipInput` in `l2d-core`: entry walk in central-directory order, entry
    patterns, text check, gzip entries, maximum record length, entry reports,
    archive content hash, positions naming entry and line
  - Batches through `Batcher`; deduplication set held against the budget, with
    refusal when it does not fit
  - `l2d ingest` accepts an archive and prints the entry report
  - Tests first, on archives built in a temporary directory: equality with the
    unpacked files at several batch sizes; a binary entry, an encrypted entry,
    a damaged entry and a gzip entry; an over-long record; a zip64 archive
  - On-request test in release on `data/ait-lds-v2/russellmitchell.zip` with a
    log pattern: records read, time, and the heap peak against the limit
  - Done when the fixture archive gives the same summary as its unpacked
    files, and the AIT archive is read within the memory limit
  - _Depends: 1.11, 1.15_
  - _Requirements: 13.1 to 13.10, 13.13, 12.4, 12.9_

- [ ] 1.17 Write per-log results in parts and deduplicate on disk
  - Parquet parts per batch with a manifest, `--work-dir`, free-disk check
  - Hash-partitioned deduplication on disk when the in-memory set would exceed
    the budget; equality with the in-memory path asserted on the fixture
  - `l2d baseline score` on an archive writes parts
  - Done when scoring an archive larger than the memory limit completes within
    the limit and its parts hold one row per distinct log
  - _Depends: 1.16_
  - _Requirements: 13.11, 13.12, 12.6, 12.9_

## 6. Open points for the owner

1. One task or two. 1.16 alone makes archives readable and refuses when the
   deduplication state outgrows memory. 1.17 adds the parts and the on-disk
   deduplication. 1.17 is the larger piece of work.
2. Nested archives other than gzip (`.bz2`, `.xz`, `.tar`, `.zip` inside the
   zip) are skipped and reported in this draft. Each one that should be read
   adds a crate.
3. Default work directory. The draft says `<project>/work/<run>/`. On this
   host the storage rules point generated artifacts at
   `D:\BuildArtifacts\log2decision\`.
4. Whether to read entries in parallel later. It would speed up large
   archives and make the record order a thing to define.
5. The AIT label adapter is a separate amendment. Without it an AIT archive
   can be read and scored, not evaluated.
