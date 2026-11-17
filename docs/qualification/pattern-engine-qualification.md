# Task 1.27: bounded pattern-engine qualification

Date: 2026-11-17. Selected engine: **Drain-style**, agreement threshold 800/1000.
This is an engineering-fixture selection, not deployment or real-log generalization
qualification. No Python runtime or package is added to the product.

## Frozen evaluation and provenance

`crates/l2d-core/tests/fixtures/pattern_pairs.json` contains 20 agent-authored text
cases with explicit same-pattern group annotations and individual rationale. The
parent controller inspected the annotations before measurements and found them
coherent. That review is not claimed to be human ground truth. Fixture SHA-256:
`ff4d57d2c0c5daebda28e5c6af504ca8ff22e3e39f7f074723d87c754a744c88`.

The groups induce eight positive and 182 negative unordered pairs. All-pairs
measurement is restricted to this 20-case fixture; product discovery never allocates
an all-pairs corpus matrix. Rare singleton events, explicit negation, success/failure,
numeric statuses versus variable IDs, free word variables and payload punctuation
are represented. Identical-text replication is not used to inflate independent
support or recall.

Before implementation or results, the fixture froze these criteria: zero protected
outcome collisions; pair precision at least .95; then maximize recall, minimize the
number of representatives, and finally use median timing as a tie-break. No-predicted
positive precision is undefined, so an all-singleton result cannot win on accuracy.
The two thresholds were fixed at .800 token agreement and .720 character-trigram
Jaccard. No retuning followed the missed free-word pair or held-out stress tests.

Both candidates use 64 maximum patterns, 4,096 maximum UTF-8 text bytes, a 512 MiB
process budget (32 MiB process baseline), default discovery-mask revision
`key-values-v1`, keys `id,user,host`, and algorithm revision `bounded-patterns-v1`.
The protected-token gate and canonical source-member ordering are shared. The full
contract and implementation differences are in [pattern-catalogue-contract.md](pattern-catalogue-contract.md).

## Measured quality and review usefulness

| Metric | Drain-style | Character trigrams |
| --- | ---: | ---: |
| True positive pairs | 7 | 6 |
| False positive pairs | 0 | 1 |
| False negative pairs | 1 | 2 |
| True negative pairs | 182 | 181 |
| Pair precision | 1.000000 | 0.857143 |
| Pair recall | 0.875000 | 0.750000 |
| Selected-member coverage | 20/20 | 20/20 |
| Pattern representatives | 13 | 13 |
| Protected outcome collisions | 0 | 0 |
| Conservative literal-variation flags | 1 | 1 |
| Rare singleton cases retained | 2/2 | 2/2 |

Drain-style meets the frozen precision gate and has higher recall. Character trigrams
fail the precision gate by merging `payload=[ok]` with `payload={ok}`; their differing
punctuation remains exposed in the literal-variation flag and boundary example.
Both engines miss `worker alpha completed batch` versus `worker beta completed batch`:
three of four word tokens agree, below the fixed .800 Drain-style threshold. Trigrams
also split the annotated cache-shard pair. Drain-style groups that pair and flags
`blue -> green` as a literal variation to inspect. A flag is deliberately not a claim
that the variation truly changes outcome.

Thirteen representatives instead of twenty records reduce the initial summary list
by seven. Inspecting all retained varied examples still reaches all twenty distinct
texts on this small fixture; this is not evidence of reduced total analyst effort.
Representatives and variation/boundary examples are actual member texts. Templates
make the variable `id`/`user` slots and distinct statuses visible, while rare events
remain separate. No user study, human review-time saving, background recall, or
unseen-format quality is inferred.

## Measurements and reproducibility

The Rust release test is:

```powershell
$env:CARGO_TARGET_DIR='D:\BuildArtifacts\log2decision\task-1.26-target'
cargo test -p l2d-core --test pattern_qualification --release -j4 -- --ignored --nocapture
```

Run from the task repository, it rebuilds each catalogue seven times with the same
settings and budget, asserts repeat revisions, and prints pair counts, templates,
heap measurements and total discovery/persistence elapsed time. The allocator
counter measures incremental **Rust heap**, not native allocations or OS peak RSS.
The MemoryBudget report includes its process baseline and accumulated input/search
accounting, and is a conservative admission estimate, not measured physical memory.
Timing includes small-file source checks, spooling, membership writes, sync and
resource accounting; it is noisy and does not establish large-corpus throughput or
exclusive-host performance. Timing did not decide the chosen engine.

The first measured run (`qualification-release.log`) recorded median 22.350 ms
(Drain-style) and 21.099 ms (trigrams), Rust heap peaks 17,214 and 18,811–18,812 bytes,
and the same 85,516,660-byte conservative process estimate. Counts and revisions
were stable across all seven runs. The flag was subsequently removed, spool usage
was marked in resource telemetry, and the division spelling was made Clippy-clean;
no grouping settings or algorithm behavior changed. Final replay evidence is kept
with the task validation logs under `D:\BuildArtifacts\log2decision\task-1.27`.

## Held-out correctness and resource checks

Separate tests, not used to tune thresholds, cover three-member template evolution,
an unfamiliar `aborted` variation exposed as a potential outcome collision,
`rc=0`/`rc=1`, rare retention, reversed input frame order, mask revision changes,
long-text refusal, catalogue overflow, duplicate texts reappearing across store
sources after overflow, and distinct/reviewed support with contradictory explicit
reviews. Store tests use real SQLite occurrence increments and a controlled
count-snapshot mutation with unchanged ID identity. Archive tests include duplicate
occurrences in later entries and reject changed entry scope.

Disk tests cover corruption, deletion, immutable-destination refusal, no final
artifact after source/record/snapshot refusal, and manifest replacement. A resident
catalogue's actual capacities are charged even when its disk manifest is replaced
with a tiny file; a different manifest cannot be used to understate memory or return
members. A 2,000-member allocator test measured 206,062 bytes incremental Rust heap
for catalogue construction, retained exactly 2,000 distinct texts, and exercised
bounded loading/streaming plus low-budget and oversized-metadata refusals. These
resource tests establish their bounded cases; they are not a claim about all native
memory paths or hardware inference.

## Primary algorithm reference and limits

The [official Drain3 documentation](https://github.com/logpai/Drain3) describes its
fixed-depth parse tree, token similarity threshold, configurable masking and
`max_clusters` eviction behavior. It links the original 2017 Drain paper. Consulted
2026-11-17; no reference implementation source was copied. Our implementation uses
a token-count/first-token routing predicate with a bounded linear candidate scan,
counts wildcard slots as agreement, retains punctuation, gates on protected original
outcome signatures, and retains
explicit overflow instead of LRU eviction. It is called **Drain-style**, not Drain3
compatibility. The competing engine is the explicitly documented bounded Unicode
character-trigram/Jaccard exemplar route, not a CPU fallback for an NPU encoder.

This fixture is small, synthetic, and search-selected. Mask coverage is deliberately
narrow; arbitrary timestamps, paths, free-form identifiers and multilingual outcomes
are not comprehensively qualified. Outcome protection is a finite vocabulary;
unknown changes require the conservative variation flag and original membership
inspection. Templates are summaries, not exact regexes or class labels. Production
transfer qualification, refinement and vector evaluation remain their later tasks.


The store admission follow-up was reproduced with a real second-process writer,
using a valid 1 MiB log text committed at the reader's `BEGIN`. The old pre-transaction
maximum admitted two callback rows and grew native SQLite memory by 1,063,752 bytes
with only 1 MiB of read headroom. After moving the longest-text lookup and native
reservation into the same read transaction, the operation refuses before handing
out any rows: 3 MiB needed, 1 MiB available, 13,112 bytes native peak growth. The
unit test's trace registration is test-only, and it is unregistered before its
userdata is dropped. The normal three-column marking interface remains unchanged.
Evidence: `snapshot-red.log`, `snapshot-green.log`, and unflagged `store-final.log`.

## Final replay and validation evidence

The final release replay is authoritative: all seven repetitions retain the table's
same pair counts, coverage, representatives and revisions. Median elapsed times were
23.330 ms (Drain-style) and 18.830 ms (trigrams); Rust heap peaks remained 17,214 bytes
and 18,811–18,812 bytes. Timing varied from the initial run, as expected for this tiny
I/O-heavy fixture; quality, not latency, selected the engine. Spool telemetry now
correctly reports `spilled: true`. The final 2,000-member allocator replay measured
206,068 bytes incremental Rust heap, within the same tested bound.

[Final per-repeat results](pattern-engine-results.csv) preserve the measured rows.
[Artifact identities and parameters](pattern-engine-artifacts.json) pin the fixture,
Rust source, benchmark source and release executable hashes and compiler version.
Initial raw measurements remain in `qualification-release.log`; final measurements
are in `qualification-final.log` in the task artifact directory.

The full workspace run before the narrow transaction-admission follow-up passed
843 tests, failed none and ignored 51. After that follow-up, the focused store suite
passed 19 tests, baseline fitting/redaction and memory suites passed 39, catalogue
and allocator suites passed eight, and the explicit release comparison passed one.
This distinction avoids claiming a second full run after the store change. Hardware,
large dataset and request-only measurements remain ignored and are not qualified by
this task. Final fmt, workspace Clippy with warnings denied and CLI help are recorded
alongside these logs.
