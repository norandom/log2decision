# Pattern catalogue contract (task 1.27)

`l2d_core::patterns::PatternCatalogue` is a discovery artifact, not a classifier,
regex matcher, or authorization to redact fitting inputs. It consumes an immutable
`SelectionSet` and reads original file/archive/store text under its source identity.
The selection's revision includes its review receipt. Discovery never modifies it.

## Persistence and identity

`build_input`, `build_archive`, and `build_store` write a new directory. Selected
original records are spooled first; their offsets are bounded by the already capped
selection. Processing follows ascending `Member` identity, independent of frame or
store insertion order. The membership stream contains every selected member exactly
once, its original text, original occurrence count, explicit review state, and an
optional pattern ID. A missing pattern ID means explicit catalogue-cap overflow.
There is no eviction. Missing selected identities and disk errors refuse publication.

`members.jsonl` is flushed and synced before `manifest.json` is serialized and
synced. The private sibling staging directory is renamed only after both succeed.
Failed operations remove their own staging directory; an existing destination is
refused. Membership and manifest hashes bind the canonical stream, source,
selection review revision, settings, algorithm revision and discovery-mask revision.
Readers check version, digest, bounded metadata size and member ordering; damaged,
missing and truncated membership files are errors, never empty results.

File and archive hashes are checked against the selection and checked after reading.
Archive scope includes the entry filters. Final archive occurrence counts come from
the existing spill count stream so duplicates after a record's first batch count.
The store adapter preserves the existing marking batch schema, exposes a separate
occurrence-aware batch route, and compares a count-sensitive snapshot before and
after reading as well as the selection's ID snapshot. A same-ID occurrence increase
therefore changes the catalogue revision; an increase during discovery refuses.

## Counts and review

* `members` counts source-qualified selected identities.
* `occurrences` sums the source's original multiplicities. Existing source counts
  can saturate at `u32::MAX`; discovery cannot recover occurrences already lost there.
* `distinct_texts` counts SHA-256 identities of original UTF-8 texts, including across
  store sources. Masked text is never used as independent support.
* `reviewed_texts` counts distinct texts with a good, bad or conflict receipt.
  Unresolved receipts (including unspecified decisions in a reviewed selection) are
  reported separately and do not inflate this count.
* Good and bad support can overlap when the same text received different explicit
  reviews at different source identities. `conflicting_texts` exposes that overlap.

An exact-text assignment is reused even after template evolution or catalogue
capacity is reached. This includes earlier ungrouped assignments. Duplicate sources
and occurrence volume cannot inflate distinct or reviewed support. Selection
completeness and skipped/omitted counts are retained unchanged; all-member accounting
is a claim about the selected scope, not about omitted source records.

## Bounded engines and inspection

Both Rust engines use the same canonical input order, limits, key/value masks and
protected-outcome gate. Their implementation is original; no Python dependency or
third-party algorithm code is included. Every punctuation and whitespace character
is retained as a token; alphanumeric runs form word tokens. There is no case folding.
The masks replace a single alphanumeric value following an explicit case-sensitive
`key=` token sequence. Defaults are `id`, `user`, and `host`. They are independent
of `RedactionRule`; changing discovery settings creates a new catalogue revision.

The Drain-style candidate uses token count and first token as a fixed-depth routing
predicate, then scans at most `max_patterns` candidates. It measures non-whitespace
token agreement; existing wildcard slots count as agreement. Punctuation must agree
and differing word slots can generalize. This wildcard rule is part of our variant,
not a claim of equivalence to upstream Drain3.
Equal scores choose the first canonical cluster. This is a bounded adaptation with
a linear capped leaf search, not a full Drain3 parse tree or implementation of all
its options. It keeps overflow explicitly rather than evicting old clusters.

The character-trigram candidate measures set Jaccard over Unicode character trigrams
of masked text against the cluster's original exemplar. It scans the same bounded
catalogue and uses the same tie-break. It has no chaining through member vectors and
allocates no corpus-wide all-pairs matrix. Both candidates produce inspectable token
summaries; variable-length n-gram groups explicitly say to inspect member structure.

Known negation/outcome words and values of `status`, `code`, `rc`, `result`, and
`outcome` form an original-text signature that must agree before grouping. This is
not an exhaustive semantic vocabulary. Any generalized unmasked word difference
sets `possible_outcome_collision` and retains literal differences for inspection;
that conservative flag also catches harmless name variation. Real representatives,
two varied member examples and the lowest routing-similarity boundary example are
retained. The full membership stream remains available when samples are insufficient.

## Resource boundary

ResourceGuard enters before operation work and records final accounting on success
and failure. A conservative reservation precedes allocation of catalogue/token/sample
storage, exact-text support indexes, offsets, retained file frame/summary, serialization
and bounded read/write buffers. Archive reservation is fixed before the first batch
so the existing one-batch `hold_beside` lag cannot undercount catalogue growth.
Memberships and original texts stream to disk; distinct-text support and offsets
remain proportional to the capped selection and are explicitly precharged. Oversized
texts, metadata, settings or whole-workspace reservations refuse before publication.
This design can conservatively refuse a catalogue that would physically fit; raising
explicit limits requires enough MemoryBudget headroom. Store SQLite read budgets
remain owned by the existing store query infrastructure. The longest-text bound
and native text-reader reservation are taken inside the same read transaction as
the cursor, before preparing or stepping the text query.

Requirements: 17.3 and 17.7; resource contracts in 12 and 15. Refinement, generated
query previews and the discovery CLI belong to 1.28; vector discovery belongs to
2.19. No NPU execution or deployment qualification is claimed here.


