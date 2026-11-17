# Handover: npu-log-decision

## Current roadmap update — 2026-11-17

The owner accepted the recommended pattern-discovery/Venn workflow and requested
integration into Kiro. Requirements 17–20, design and the active roadmap in
tasks.md now contain it; amendment-2026-11-17-pattern-discovery-and-triage.md
records the rationale. Scope: lexical patterns first, NPU vector expansion next,
independently qualified reusable score, then investigate/review/low-priority
queues with reserved review for unknown/conflicting cases and explicit feedback.
Assumed background remains available for exploration; qualification requires
reviewed good/bad evaluation examples. Venn means Vector Decision Models.

Task 1.23 wrap-up completed on 2026-11-17 as merge `883ce5e`. Fresh independent
review found and then approved fixes for redaction expansion before a budget check
and held-out reporting copies before refusal. Verification: 788 passed, 0 failed,
49 ignored; fmt, Clippy and CLI help passed. Logs are in
`D:\BuildArtifacts\log2decision\fix-1.23-memory\`. Both old task worktrees were
verified clean/merged and removed. Task 1.25 is complete (reviewed commit `cd103f6`):
file/archive phrase and regex marking, matching text/positions, whole JSON and
optional saved MarkSets. Independent review approved; 9 task tests and 797
workspace tests passed (0 failed, 49 ignored); fmt, Clippy and actual CLI smoke
passed. Logs: `D:\BuildArtifacts\log2decision\task-1.25\`.

Task 1.24 is complete (reviewed commit `84bb15a`): host/NPU admission with
wait/refuse policy, final resource accounting, bounded store-ID spilling, reader
fault/cancellation coverage and a real concurrent store-writer test. Independent
review caught two reporting defects; re-review approved their fixes. Verification:
820 passed, 0 failed, 50 ignored; fmt, Clippy, three Python regressions and actual
CLI/AIT smoke passed. Evidence: `D:\BuildArtifacts\log2decision\task-1.24\`.
The two-round AIT comparison retains 64 pieces: 128 is faster with two readers
forced but slower with defaults, which admit only one reader at 512 MiB. All
eight runs preserved 22,880,272 records and 5,642,082 distinct texts. A batch
boundary accounting fix also preserves the results and keeps the counted peak
within the limit. Native NPU memory remains explicitly unaccounted in the Rust
throughput reservations; outside-tool NPU utilization is unknown. These are not
claims that all of Requirements 15.2/15.12 are closed.

Keep work scoped to this project: do not monitor, inspect or wait for unrelated
processes. Serialize this project's own heavy jobs.

Task 1.26 is complete (reviewed commit `65a00fb`): unlabelled SelectionSets,
literal/whole-record glob/regex and explicit store FTS5 trigram modes, source-bound
review previews and format-3 reviewed MarkSet exports. Unresolved/conflicting
members remain explicit exclusions from both supervised classes. Existing MarkSet
formats 1/2 retain their behavior. Persist the reviewed SelectionSet receipt
alongside exports when building the discovery CLI (1.28). Store counts are
source-qualified log IDs, not independent text support.
Independent re-review approved the native FTS memory fix. The full workspace had
831 passed, 0 failed, 50 ignored before scoped capacity/FTS corrections; all 54
affected tests, native SQLite heap regression, fmt, Clippy and CLI help passed on
the corrected source. Evidence: `D:\BuildArtifacts\log2decision\task-1.26\`.
FTS is store-only and intrinsically insensitive; unquoted prefix expansion
refuses because its native allocation cannot be bounded by expression length.
Selection membership overflow refuses explicitly. Next: task 2.19.

Task 1.27 is complete (reviewed commit `b62236d`): bounded pattern catalogue,
disk membership, deterministic source-bound revisions, representative/variable/
boundary evidence, separate occurrence and distinct/reviewed support, explicit
ungrouped overflow and an occurrence-aware store adapter. Independent review:
APPROVED; 26 independently replayed tests passed. Workspace evidence: 843 passed,
0 failed, 51 ignored before the final transaction admission fix; after that fix,
19 store, 39 baseline/redaction, 8 core and one release qualification test passed.
Fmt, Clippy and CLI help passed. Parent rechecked artifact hashes, fmt and the
release qualification before merging. Evidence: `D:\BuildArtifacts\log2decision\task-1.27`.
The frozen 20-text engineering comparison selected Drain-style (TP7/FP0/FN1)
over trigram (TP6/FP1/FN2); this is not human-labelled deployment qualification.

Task 1.28 uses the owner-selected `l2d discover ...` group. Partial batch review
must preserve existing decisions outside its target; save the reviewed SelectionSet
receipt alongside exported MarkSets. Query preview must state its background scope.
Build storage now follows `AGENTS.md`: one reused project target, no debug symbols
or incremental compilation, 30 GiB target/50 GiB free preflight limits, serialized
heavy jobs. The old project debug cache was cleaned (27.7 GiB); release evidence
was retained. The owner resolved the shared cargo-target cleanup separately.
Task 1.28 is complete (reviewed commit `6f03c0b`): the owner-selected
`l2d discover` group implements search, pattern inspection, split/merge, priority,
query preview/save, explicit partial review and receipt-bundled format-3 export.
Independent review APPROVED; source and binary hashes matched. Full workspace
before final metadata reporting: 851 passed, 0 failed, 51 ignored. Final reporting
changes passed 17 affected tests, fmt, Clippy and actual human/JSON CLI smoke;
reviewer replayed 18 tests. No full-suite-after-reporting claim. Native heap growth
536,225 B; refused load 1,045 B; counted peak 79,560,704 B under 128 MiB.
Evidence: `D:\BuildArtifacts\log2decision\task-1.28\validation.md`; command guide:
`docs/qualification/pattern-refinement-contract.md`. Refinement explicitly refuses
oversized whole-membership snapshots. Query background is other selected members.
Next task 2.19 adds vector exploration, keeping retrieval distinct from qualification.
Remaining execution order: 2.19 → 2.18 → 2.8 →
2.11 → 2.20 → 2.9 → 2.22 → 2.21. This supersedes sequencing below.
First usable discovery milestone 1.28 is complete. Complete qualified triage: 2.22.
Review-budget values, reserved/audit allocations and missed-case targets must be
recorded before final evaluation; no universal thresholds were selected.

## Previous handover — retained historical record

Written 2026-11-16, end of session. Branch `main`, pushed (this file's commit
follows the merge of task 1.23). Remote: https://github.com/norandom/log2decision

This file describes the state as it is now. It replaces the handover of
2026-11-15.

## Where things stand

The spec `npu-log-decision` is in implementation
(`.kiro/specs/npu-log-decision/`). Each finished task was implemented by one
agent, checked by an independent reviewer, then committed and pushed.

Tests for the merged 1.23 source (fresh validation 2026-11-17): 788 passed,
0 failed, 49 ignored. `cargo fmt` and
`clippy -D warnings` are clean.

What works from the command line: a file or a zip archive of logs is read,
scored, calibrated and triaged with the baseline scorer; `baseline score` on an
archive writes Parquet parts with a manifest; `ingest` reads two archive
entries at once by default. What works as library code, not yet as commands:
the store and its search, mark sets (1.21), redaction (1.22), a score fitted
from a mark set with the held-out-group report (1.23), and encoding logs into
the store on the NPU (2.7).

**No agent is running. No worktree is open. Next is task 1.25, the `l2d mark`
command (owner decision of 2026-11-16), then 1.24, then gate 2.18.**

### Finished tasks

| Task | What it is | Commit |
| --- | --- | --- |
| 1.1 | Rust workspace, shared core types | `92f2714` |
| 1.2 | Command-line skeleton (stubs), test fixture | `fd64eb9` |
| 1.3 | Readers for lines, CSV, JSONL; deduplication | `5895ba7` |
| 1.4 | HTTP CSIC 2010 reader | `90da638` |
| 1.5 | Character n-gram hashing with routing modes | `ff89b83` |
| 1.6 | Deterministic logistic regression, sparse and dense | `3675d61` |
| 1.14 | Polars as the data layer; CSIC group key = URL path | `b1e9198` |
| 1.15 | Batches sized to free memory; memory limit that holds | `786ff1b` |
| 1.7 | Baseline scorer: fit, score, save, load | `d5d0be7` |
| 1.13 | Conformal calibration over any scorer | `fc1f253` |
| 1.8 | SQLite store with a text index and the vectors | `5f4a9b5` |
| 1.18 | Own conformal code; the `conformal-prediction` crate is gone | `a60f981` |
| 1.9 | Search by phrase, label, source and nearest vectors | `b0def83` |
| 1.10 | Triage: ranking, review budget, placement by label set | `52918fc` |
| 2.2 | Hardware gate: the 8-bit spike model runs on the NPU from Rust | `bdabad2` |
| 2.3 | Runtime file check, NPU device query, `L2D_RUNTIME_DIR` | `3b12989` |
| 2.13 | Provisioning from AMD's NuGet ZIP (release 1.8.0) | `e3413ef` |
| 2.14 | Provisioning release 1.9.0 from AMD's NuGet package | `bb2f4f1` |
| 2.5 | Log text to the encoder's input (`ByteCodec`) | `9ced82b` |
| 1.11 | Commands `ingest`, `baseline fit`, `baseline score`, `calibrate`, `triage` | `7e70f43` |
| 2.4 | Encoder package with a verified revision; the 1.8.0 build switch removed | `150cef6` |
| 1.12 | Baseline gate: grouped evaluation on CSIC in `l2d-eval` | `775b08a` |
| 2.12 | Rows right and fast on the NPU: the texts as rows of one image | `51aa056` |
| 2.1 | Science gate: decided by the owner on 2026-11-13 | `51aa056` |
| 1.16 | Zip archives read in batches under the memory limit | `8ba2e95` |
| 2.6 | Encoder session on the NPU, NPU only by evidence | `0491f8d` |
| 2.15 | Gate: embedding qualification with diagrams (did not pass for the published encoder) | `a713fd8` |
| 1.17 | Results in Parquet parts on disk, deduplication on disk | `e7cbbce` |
| 2.16 | Gate: hierarchical byte encoder, entropy patches against two controls (record) | `acb9767` |
| 2.7 | Encode logs and store the vectors, one transaction, release recorded | `7e1bc77` |
| 1.20 | `l2d ingest` deduplicates on disk only when needed; partition room by distinct keys | `15ee402` |
| 2.17 | Gate: stable entropy-patch encoder; vectors do not classify on unseen endpoints; owner decision | `28c69b3`, `8fded88` |
| 1.19 | Archive entries read at once: `ingest` 2 by default (faster), `baseline score` 1 (not faster) | `170e013` (merge of `6088015`) |
| 1.21 | Mark sets on a file, archive or store | `4f7deeb` (merge of `a53708e`) |
| 1.22 | Redaction: proposals with counts, confirmed rules before hashing | `ac3580f` (merge of `cfd2ade`) |
| 1.23 | A score fitted from a mark set; held-out-group report | merge of `747c3a5` |

Review rounds on 2026-11-16: 1.19 approved in round 1; 1.21 in round 2; 1.22
in round 1; 1.23 in round 3 (rounds 1 and 2 rejected memory held outside the
budget). The reviews' figures are in the `- <task>, review` notes of tasks.md.

Spec changes on 2026-11-16 (owner decisions, in tasks.md):
- **Task 1.24** added: guards for a busy machine (15.6 to 15.10: one
  resource-hungry operation per host, one encoder session on the NPU across
  processes, wait or refuse, record at the end), reader robustness tests from
  1.19, 128 reader pieces measured against 64, store mark sets on disk, a real
  concurrent-writer test for store marking. Runs before gate 2.18 (confirmed
  at the end of the session).
- **Task 1.25** added: `l2d mark` on a file or an archive (phrase or regex,
  prints the matches with positions, saves the mark set), pulled forward from
  2.9 so a new dataset can be searched from the command line. Runs next,
  before 1.24.

## Next, in order (one heavy job at a time)

1. **1.25** `l2d mark` on a file or an archive. Needs neither the NPU nor the
   store. Read the `- 1.21:` notes (positions, entry patterns,
   `require_input(summary, archive_entries)`) and the `- 1.19:` notes
   (entries-at-once default of `ingest`).
2. **1.24** hardening (see above). The measurement of 128 reader pieces runs
   alone, in release, two rounds, on `russellmitchell.zip`; D: needs 88 to 99
   GiB free for all-entries runs.
3. **Gate 2.18:** does redaction make scores agnostic? On the 2.17 folds (five
   by endpoint), CSIC attacks found by regular expressions standing in for an
   analyst's marks, the mark-set score with and without the proposed rules;
   held-out against fitted ROC AUC with spreads. Run alone. The 1.23 CSIC run
   (a `drop table` mark set) gave 1.0000 on both sides with and without rules:
   that regex marks its own payload, so it says nothing about agnosticism; the
   gate needs mark sets whose texts carry endpoint fragments.
4. **2.8** grouping of a mark set and "close enough" (Venn's
   `nearest_outcome`), **2.11** outlier ranking, **2.9** the remaining commands.
5. Milestones 3 and 4.

Distance to the owner's goals, as stated on 2026-11-16: a score fitted from
marks that reports its own agnosticism exists as library code (1.23); evidence
that redaction makes it agnostic is gate 2.18; Venn's "close enough" is 2.8;
the decision to investigate (where score and decision model meet) is not a
task yet.

## Decisions waiting for the owner

None.

## Decisions in force

From 2026-11-11:

- Output is a bad/good ratio per log for triage, not a verdict.
- Rust only. No Python in the tool; Python only during development, in a `uv`
  environment that belongs to this project.
- General text embeddings were dropped (EmbeddingGemma: 2.3 logs/s on the NPU,
  worse than character n-grams).
- The baseline scorer is character n-grams plus logistic regression.
- Polars is the one data layer, pinned to 0.52.0 (later releases link
  networking code).
- No crate that can open a network connection. A test guards the lock file.
- CSIC is grouped by URL path alone. Single-label groups are reported
  separately from the cross-endpoint score.
- Memory (requirement 12): read free memory before loading, size and rescale
  batches, refuse instead of running out.
- The store is SQLite with its text-search extension.

From 2026-11-12:

- Conformal calibration is own code.
- Triage places logs by label set and states that the error rate holds only for
  logs that resemble the calibration logs.
- A single log text in the store may be at most 1 MiB. `faer` stays for the
  vector search.
- Requirement 13: zip archives read in batches without extracting; gzip the
  only nested format; results to `<project>/work/<run>/`; maximum record length
  64 KiB.
- The runtime files are provisioned from AMD's packages by the method of the
  sibling project `npu_rag_from_folder`.
- The project code is under the MIT licence.
- Design text is brought in line after each approved task.

From 2026-11-13:

- **Decide on measured metrics, not estimates.** AMD release 1.9.0, "always
  forward".
- **NPU only.** No CPU fallback, no NPU-against-CPU comparison. 4.5 withdrawn;
  4.6: the same vector for the same text whatever its place in a batch. A host
  without an NPU uses the baseline scorer.
- **Baseline gate figure: pooled,** the mean over folds beside it. L2 penalty
  stays at 1 until repeated on AIT.
- **NPU input stays float.** Requirement 1.5 has two truncation flags.
- **Zip input (1.16):** own zip-directory reader; lines before damage kept;
  exit code 2 for a damaged entry.

From 2026-11-14:

- **Encoder direction:** convolutional architecture; bytes, then patches, then
  one vector per line; entropy patches after BLT; no cutting; n dimensions.
  Vector sets are qualified for classification by distinct feature ablation
  (information view).
- **No `blaze`.**
- **Zip and parts (1.17):** `flate2` on `zlib-rs` (built at `opt-level = 3` in
  dev); five Parquet codecs; disk estimate 6 bytes per row until a second
  archive is measured; `statvfs` later.
- **1.20:**
  - `ingest` touches the disk only when the set outgrows memory;
  - partition room is judged by distinct keys;
  - `baseline score` checks its parts and a bound for `keys.bin` before the
    read, and deduplication at the freeze;
  - buffers are sized at the freeze.
- **Store (2.7):** both truncation flags plus "empty"; a fill is one
  transaction; the store stays usable after a refused fill.
- **Field machines (requirement 15, design's operating criteria):**
  - batching;
  - spilling datatypes and one global memory limit;
  - streaming where possible;
  - one hungry operation at a time.

  The reference host kills processes at 92 % of memory. Development pipelines
  follow the same rules.

From 2026-11-15:

- **Gate 2.17, option 1:** the baseline scorer classifies logs. The encoder, in
  the stable layout of 2.17 (cut from the 8-bit byte graph, line graph of
  stride-1 convolutions), serves search, grouping and near-duplicate views.
- **The score is agnostic, not dataset-specific.** A score that works because
  it learned certain hosts or endpoints is against the idea. The flow
  (Requirement 16):
  1. the analyst marks n samples (phrase, regex, positions);
  2. they are grouped by string similarity and the analyst de-selects;
  3. hardcoded fragments are redacted, proposed by the tool and confirmed by
     the analyst;
  4. the baseline scorer is fitted on the redacted texts;
  5. Venn decides "close enough", and a log that is not close enough gets no
     ratio.

  CSIC and AIT labels measure agnosticism only (they stand in for an analyst's
  marks); they are not the source of a product score.
- **Requirement 16 details:**
  - the other class is the unmarked logs of the same input by default (good
    marks allowed);
  - the group key is per kind of log and settable (CSIC: URL path; AIT: host);
  - the margin for "depends on seen groups" is 0.05 ROC AUC;
  - without an NPU, grouping is by character n-gram similarity.
- **Requirement 15 wording approved as written.**

From 2026-11-16:

- **Reading archive entries at once (1.19):** kept only where faster within the
  same limit. `ingest` reads 2 at once by default when the readers fit an
  eighth of what the limit leaves; `baseline score` reads 1 by default;
  `--entries-at-once` 1 to 8 on both. Each reader holds a fixed 19 MiB (64
  pieces of 256 KiB).
- **Marking a store (1.21):** a stored log is named by its `log_id`; the
  store's content hash is a SHA-256 over its `log_id`s in sorted order,
  streamed; a mark set records its kind of source (input or store) and refuses
  the other; phrase through the store's text index, regex through a batched
  read; a trait in `l2d-core` implemented by `l2d-store`. A mark set of an
  archive is checked with its entry patterns; a store is hashed again after
  marking. `regex` is a direct dependency of `l2d-core` (it was already
  linked).
- **Redaction (1.22):** proposals in two passes (a bounded counter, then an
  exact count), refused when what the counter dropped could hide a qualifying
  fragment; literal rules bounded by tokens (`/tienda1/` but not `/tienda10/`);
  placeholder SUB (0x1A); the routing option `strip` is itself a rule; rules
  saved in the mark-set file (format 2).
- **Mark-set score (1.23):** de-selected logs are kept out of fitting and out
  of the proposals (neither side); texts that become equal after redaction stay
  rows of their own (redaction inside the hashing); `baseline.json` format 3
  with copies of the mark sets, revision over the rules' digest and the copies;
  the scorer column says `mark-set`; the held-out report folds by a hash of the
  group key (`Folds::hashed`, the science gate's rule). A fit from an archive
  keeps room for two batches' copies, because the reader sizes one batch ahead;
  each batch is checked against the figure it was sized with.
- **Order:** 1.25 (`l2d mark`), then 1.24, then gate 2.18.

### Follow-ups, not yet tasks

- Calibration (`ScorerKind` in `l2d-decide`) has no mark-set kind; a mark-set
  score is calibrated as a plain baseline scorer. Needed before 2.8 gives
  label sets for a mark-set score (16.11).
- `statvfs` free-disk query on systems other than Windows.
- A measured disk check at the freeze (replaces the 6-byte estimate; would also
  tighten the `keys.bin` bound, about three quarters of the limit under the
  default limit). AIT all-entries asks 88 to 99 GiB and writes about 1.2 GB.
- `"incomplete": true` in the JSON of `triage` and `baseline score` on plain
  inputs.
- `ingest` reads its frames back from disk although only their number is
  needed.
- Closed on 2026-11-17: the held-out report now preflights its copies and retained
  workspace, with dedicated allocator regression tests (merge `883ce5e`).
- A refused archive read in a mark-set fit pushes one stray `BatchRoom` while
  building its message (harmless).
- The session's placement rule against the provider's duplicated dequantize
  node: stated, not changed.
- The 8-bit cut of 2.17 holds only while the byte graph stays bit-identical on
  the NPU: re-check on every new driver or Ryzen AI release.
- 2.17 record, minor: `hg17_run.py:741` treats the published encoder's
  `seen_linear` as seed-free (0.973, n = 5; over 15 runs 0.9726 ± 0.0089).
- From the 2.7 review: the refusal message wording; `VectorOrigin`'s public
  fields; the message of `Fill::store`; a free-disk check in 2.9; the
  `npu_gate` cache directory under `L2D_NPU_WORK_DIR`.
- Remove the CPU provider comparison from `tests/npu_gate.rs`; repeat the
  penalty comparison on AIT; process CPU seconds as the energy proxy
  (requirement 9.4); AIT labels (a separate amendment later).

The task file's `## Implementation Notes` section holds what each finished task
left for the next ones. Read the notes of the tasks a new task depends on
before starting it.

## How the work is run

Autonomous mode, review required (`.claude/skills/kiro-impl/`):

1. One heavy task at a time on this machine. Dispatch an implementer agent in
   its own worktree with the task text, the spec sections, the implementation
   notes and the host rules. Tests first. Its report must end with the exact
   `## Status Report` block; ask for it again if missing.
2. Dispatch an independent reviewer (`.claude/skills/kiro-review/SKILL.md`).
   Give it the code and the implementer's claims to verify, not the conclusion.
   On a rejection, send the findings back to the same implementer, then
   re-review with the same reviewer. At most two remediation rounds; after a
   third rejection, a debug agent (`kiro-debug`).
3. Commit the reviewed work on the task branch, merge into `main` with
   `git merge --no-ff --no-commit <branch>` (`tasks.md` may conflict in the
   notes; keep both sides), re-run the checks there, mark the task `[x]`, add a
   `- <task>, review` note, update the README status table and this file, then
   commit and push (the owner asked for a push after each approved task).
   Commit messages end with the `Co-Authored-By` line.
4. Remove the worktree and delete its branch. Agent worktrees are locked by the
   harness: check that the branch is in `main` (`git merge-base
   --is-ancestor`) and the worktree is clean, then `git worktree unlock` before
   `git worktree remove`.

Put into every brief: never stop processes you did not start (stop your own
run by its process tree, never all `cargo`/`rustc` on the host); for anything
that reads in batches and keeps something per row, charge everything held,
including the first batch and the one-batch lag of `ZipBatches::hold_beside`,
with a counting-allocator memory test over archive and store and a red run.

Checks, from the repository root, with a target folder under
`D:\BuildArtifacts\log2decision\`:

```
cargo fmt --all -- --check
cargo test --workspace -j 4
cargo clippy --workspace --all-targets -j 4 -- -D warnings
```

The workspace tests take about 10 minutes in debug on this laptop.

Hardware tests are ignored by default and run on request, in release, one job
on the NPU at a time, with a reduced `PATH` to show that no Python is involved:

```
$env:PATH = 'C:\Windows\System32;C:\Windows;C:\Users\mariu\.cargo\bin'
$env:L2D_RUNTIME_DIR = 'D:\Source\log2decision\runtime\ryzenai'
cargo test --release -j 4 -p l2d-encoder --test npu_rows -- --ignored --nocapture
```

Others: `npu_gate`, `npu_throughput`, `npu_quality`, `npu_session`,
`npu_embedding_export` (2.15), `npu_hierarchical_gate` (2.16, 2.17). The CSIC
tests of 1.22 and 1.23 (`redact_csic`, `markscore_csic`) are ignored and run in
release with `L2D_CSIC_CSV=data\csic-2010\csic_database.csv`.

With the owner: answer the question asked first and briefly, say plainly
whether anything needs attention, and bring open decisions as a menu (choices
with a recommendation first).

## Results worth remembering

- **CSIC 2010:** 61,065 records, 25,608 distinct texts, 1,632 path groups, 8 of
  them with both labels.
- **Baseline gate (1.12):** cross-endpoint ROC AUC 0.914 pooled, 0.919 mean
  over five folds by URL path, routing stripped; bar 0.90.
- **Calibration on CSIC:** the 5 % error rate holds on endpoints the
  calibration logs cover, not on others.
- **NPU, rows (2.12):** texts as the rows of one image; every row
  bit-identical to the text encoded alone.
- **NPU, speed:** 16,000 to 42,000 logs/s end to end for the published encoder
  over recorded runs; 2,037 to 2,166 logs/s for the hierarchical encoder of
  2.17 (two graphs); requirement 1,000. Ranges; the laptop's variation is not
  pursued.
- **NPU session (2.6):** NPU only by evidence (placement report plus a profile
  of the first run). A damaged provider cache can run on the CPU while its
  report claims the NPU.
- **Embedding qualification (2.15):** the published encoder's vectors are nearly
  one-dimensional (first component 87.9 %, participation ratio 1.3 of 128).
  The cause is a label-only objective.
- **Hierarchical encoder (2.16):** 12 to 32 components for 95 %; classification
  on unseen endpoints unsettled.
- **Stability gate (2.17)**, `docs/2026-11-15-entropy-encoder-stability-gate.md`:
  randomness ruled out; cuts stable from 8 bits to the NPU; on five folds by
  endpoint × three seeds the masked-byte encoder's linear head reaches 0.786
  seen and 0.722 ± 0.070 unseen, the baseline 0.961 and 0.923 ± 0.097 and wins
  every fold; the vectors mostly hold the endpoint's template.
- **AIT `russellmitchell.zip` (7.1 GB), 512 MB, ranges over 2–3 rounds (1.19):**
  - `ingest`, `*.log` entries (22.9 million lines, 5.6 million distinct, in
    memory): 33–34 s one entry at a time, 28 s two at once, peaks under 420 MiB;
  - `ingest`, all entries (30.9 million lines, 8.3 million distinct, on disk):
    74–81 s one at a time, 50–71 s two at once;
  - `baseline score`, all entries: 434–482 s one at a time, 376–458 s two at
    once, 451–469 s four at once (not consistently faster);
  - counts and `rows_sha256` identical in every run.
- **Redaction on CSIC (1.22):** a `drop table` mark set (168 marked, 25,440
  unmarked) gives 56 proposals; the host `localhost:8080` and all five endpoint
  paths are proposed with exact counts, no payload token; six rules change
  25,596 of 25,608 logs; the routing rule equals `strip` on all logs.
- **Mark-set score (1.23):** the held-out report warns on a planted
  endpoint-specific score (0.53 held out against 0.996 fitted) and not on an
  agnostic one (1.0 against 1.0); on CSIC the score gives ratios and survives
  save and load (revisions `43977c27...` plain, `aa217bf8...` with rules).
- **Store:** CSIC's 25,608 logs added in about 3.5 s into a 46 MiB file;
  nearest-vector search about 36 ms; store format 2 records vector origin and
  runtime release.

## Things that will bite

- **The host kills processes at 92 % of physical memory** (29.6 GiB). Exit code
  -536870911 (0xE0000001) means killed for memory. A died process is never a
  mystery: say which jobs ran together and how much each held. Cures:
  batching, one heavy job at a time.
- **One heavy job at a time:** one training, one measurement, one large build,
  one job on the NPU. Builds use `-j 4`.
- **Never stop other processes.** On 2026-11-16 an implementer stopped every
  cargo/rustc process on the host to end its own run. Stop only your own
  process tree.
- **Spike scripts must stream.** `tools/science/hg17.py` (2.17) holds bounded,
  stated amounts only. `tools/science/memguard.py` runs a step under an
  OS-enforced memory limit and records its peak.
- **Shared build folder.** `CARGO_TARGET_DIR` is `D:\BuildArtifacts\cargo-target`
  for every Rust project on this machine. Worktrees, reviewers and scratch
  copies build with their own target under `D:\BuildArtifacts\log2decision\`.
  Each task target is 4 to 20 GiB; clean them with `cargo clean --target-dir`
  once the task is merged.
- **Disk.** The disk check of an all-entries AIT run asks 88 to 99 GiB free on
  D:; on 2026-11-15 a run was refused at 84.3 GiB free. Recheck free space before
  each measurement; build targets and other host jobs change it.
- **Debug builds are slow where it matters.** Run evaluations and large archive
  cases in release.
- **Timings on this host vary** (laptop power management; the owner sometimes
  runs it in low-power, silent mode overnight). State ranges; do not
  investigate.
- **Polars.** Import only as `l2d_core::polars`. Do not enable its `csv`,
  `json`, `strings` or `streaming` facade features or upgrade it without
  checking `cargo tree` for network crates. It drops a byte-order mark at the
  start of any buffer it parses (handled since 1.16).
- **Runtime directory.** `runtime\ryzenai\` holds the five files of release
  1.9.0, provisioned from
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` with
  `uv run --no-project --python 3.12 python tools/runtime/provision.py --nupkg <path>`.
  Do not provision or run hardware tests while another process has the DLLs
  loaded. A missing DLL ends the process without an error.
- **NPU provider traps:** `error_on_failure` and the placement report do not
  make a session NPU only; the compiled-model cache trusts its key alone.
- **What reviews reject.** Figures and causes stated but not measured;
  comparisons across different heads or splits (2.17 round 1); code that
  accepts input it cannot handle; and above all memory held outside the
  budget. Every task that touched memory was rejected at least once on it: on
  2026-11-16, 1.21 (a regex compiled before the archive's directory was
  charged; a save that built the whole file) and 1.23 twice (no room for the
  first batch's copies; then the reader's one-batch lag). Tests that cannot show
  a fault in the heap show it through a refusal point or a per-batch check.
- **Host rules.** PowerShell only, no Git Bash, no `wsl`; containers through
  `docker` and `docker compose`. Models in `./models` (never C:), datasets in
  `./data/<dataset>/`, documents in `./docs/` with a date prefix and a README
  row. Caches and temp under `D:\BuildArtifacts\`, never a drive root.
- **Large files.** A hook blocks reading a file of 150 lines or more in one go;
  read in ranges or through `smart_read`.
- **Untracked on purpose.** `.openidea/` is IDE state; `.claude/worktrees/`
  holds agent worktrees. `log2ml/`, `models/`, `reports/`, `runtime/ryzenai/`,
  `runtime/cache/` and the dataset files are ignored.

## Data and artifacts outside git

- **AIT Log Data Set V2:** all eight testbeds, 121.7 GB, in `data/ait-lds-v2/`,
  verified against their MD5 sums, read in place.
- **CSIC 2010:** `data/csic-2010/csic_database.csv`.
- **Science gate:** runs and models in
  `D:\BuildArtifacts\log2decision\science-2.1\runs\`; the published model in
  `models/science-2.1/` (the weakest of three seeds).
- **Encoder gates:** `D:\BuildArtifacts\log2decision\task-2.15\`, `task-2.16\`,
  `task-2.17\` (with the review's figures and the option 2(a) result in
  `task-2.17\review\`); 2.17 model weights in `models/task-2.17/`.
- **Task 1.19 measurements:** `D:\BuildArtifacts\log2decision\task-1.19\`
  (`ait-measurement.log`, `runs\`, `scripts\measure-ait.ps1`, `run-all.ps1`).
- **Baseline gate reports:** `reports/2026-11-13-csic-2010-baseline.{json,md,run.json}`.
- **Evidence of the tasks:** the `task-*` and `review-*` log folders under
  `D:\BuildArtifacts\log2decision\` (red and green runs, check outputs, CSIC
  runs). Build target folders of finished tasks were removed.
- **AMD packages on the owner's machine:**
  `C:\Users\mariu\Downloads\RyzenAI_Deployment.1.9.0.nupkg` and
  `C:\Users\mariu\Downloads\ryzen_ai_nuget_1.8.0 (1).zip`.

## Other records

- Session memory for agents: `mmhmem` project `log2decision` holds a checkpoint
  of the same state.
- Research documents: `docs/`, indexed in `README.md`.
- Amendment records in `.kiro/specs/npu-log-decision/`:
  `amendment-2026-11-12-zip-batch.md`,
  `amendment-2026-11-12-vector-decision-heads.md`,
  `amendment-2026-11-15-analyst-flow.md`. All merged; where they differ from the
  spec, the spec wins.
- Spec input and early decisions: `brief.md` and `research.md`. Where they
  disagree with `design.md`, the design wins.
