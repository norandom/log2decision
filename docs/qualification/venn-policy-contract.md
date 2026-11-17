# Explicit Venn triage policy (task 2.20)

`l2d_decide::policy::triage_policy` composes the existing fitted character MarkSet
scorer and `close_enough` evidence engine. It adds no classifier, distance-to-badness
conversion or neighbour-vote score. Venn means Vector Decision Models.

This is an exploratory library milestone. It does not qualify deployment, choose
operational thresholds, consume final holdout data or create analyst labels. CLI
integration remains task 2.9, qualification task 2.22 and feedback task 2.21.

## Configuration and routing

Callers must supply `TriagePolicy` with a version and selection provenance, ordered
low/high thresholds, corroboration count, complete reach/profile/top-k/support
settings, calibration requirement, total review capacity, positive unknown/conflict
and audit capacities, and audit seed. There is no `Default` policy. Invalid values,
including a budget smaller than the two reserved capacities, refuse the operation.
Fixture thresholds are synthetic test inputs, not deployment recommendations.

The fixed eligibility revision is
`close-enough-originals-labelled-and-separate-background-v1`: global conflicting or
deselected originals are excluded and originals deduplicated before reach/top-k.
Labelled originals and assumed-background originals have separate top-k limits.
Background never counts as labelled corroboration. No policy-specific filter is
applied after retrieval. The existing benign support guard remains in force.

Precedence is:

1. Unsupported representation/profile, insufficient labelled support or out of
   reach: review, unsupported; no actionable ratio.
2. Both labels meet the configured corroboration count, or the corroborating label
   contradicts the low/high score band: review, disagreement.
3. If compatible calibration is explicitly required but unavailable: review,
   uncertain. Otherwise a high score with bad corroboration investigates, and a low
   score with good corroboration becomes low priority.
4. All remaining supported cases: review, uncertain.

Routing is internal to the validated operation, not a public API for reinterpreting
arbitrary saved evidence under changed settings. Good/bad are evidence labels;
investigate/review/low priority are workflow outcomes. Allocation never changes an
outcome into an observed label.

## Preserved signals and calibration

Every member retains the character scorer kind (`MarkSet`), revision, diagnostic
ratio, whether its comparison class was assumed, and a separately optional
actionable ratio. Unsupported diagnostics are never used for allocation ranking.
The complete vector signal retains original identities/text, explicit or assumed
provenance, distances, support counts, source/snapshot, metadata revision,
representation failures and explanation. Profile mismatch has no vector result
and is explicitly explained at the member level; the report still binds the
history metadata revision and actual/expected profiles.

Calibration has a separate Missing, Invalid or Compatible state. Compatibility
requires MarkSet scorer kind, exact scorer revision and supported error rate.
Legacy Baseline-tag calibration is invalid even if its revision string matches.
Invalid calibration never reaches prediction. Compatible prediction retains label
sets, p-values and strata separately from the score and distance. Missing calibration
does not itself prevent an uncalibrated supported decision unless configured to do
so; no such decision claims calibration or deployment qualification.

## Allocation and deferred members

`reserved-no-reassignment-general-priority-fnv1a-audit-v1` allocates capacities:
unknown/conflict, audit, and general = total minus the two reservations. Empty or
undersized pools leave unused reserved capacity; there is no silent reassignment.
The same rule handles budgets larger than the input population.

Unknown/conflict candidates are selected first. General selects remaining non-low
priority members, including overflow unknown/conflict. Order is investigate,
disagreement, unsupported, uncertain; then descending actionable ratio (missing
last), then stable original-record ID. Diagnostics cannot promote unsupported cases.
Low-priority audit uses seeded FNV-1a over the original ID, with ID as the collision
tie-breaker. This is a reproducible audit sample, not a cryptographic or statistical
randomness claim. Changing the seed changes the full policy identity.

All members are persisted in ID order, with an optional allocation. None means
deferred; no member disappears. Configured, selected and unused capacities are
separate from deferred counts. Duplicate member IDs refuse; distinct IDs with equal
text remain distinct input records, while support still deduplicates originals.

## Source consistency, replay and resource limits

`PolicySource::Store` validates live history metadata and checks the SQLite read
generation before and after the run. History capture also checks generation before
and after reading. An earlier endpoint-only change is caught by comparing captured
IDs/text/groups/counts against the live frame, even when occurrence snapshots match.
The metadata revision is SHA-256 over length-framed JSON tuples in member order.
Support labels come from exact saved MarkSets, bound by the scorer revision; store
label columns are not substituted. Stored vectors are not consumed: original text
is re-encoded through the selected exploration encoder.

`PolicySource::InputSnapshot` accepts only immutable input histories constructed
with `MarkHistory::from_input`, which checks the supplied source summary against the
MarkSets. It describes captured file input, not an assertion that a later file is
unchanged. Adapters must recreate and validate the snapshot when loading changed
inputs. It cannot be used to bypass live validation of a store history. Archive
adapters must provide a validated supported input history; this API does not add an
archive reader or weaken the existing source-kind checks.

Genuine encoder/runtime errors propagate, and reservations return to their incoming
value on error. Representation unsupported is a successful explained review result.
Queues, input copies and retrieval scratch are admitted through `MemoryBudget`.
After each query, the caller retains a conservative charge for the returned
evidence, including its owned strings and spare vector capacity. The next query
can reuse the released history, scorer, calibration and encoder scratch allowance.
The host resource gate covers each complete triage, save, reload and replay
comparison, and writes a final resource record for each operation.

Save streams JSON into a separate file in the destination directory and syncs it.
A hard link atomically publishes the complete report and refuses overwrite. A
write or sync failure leaves the destination available for a retry. Reload checks a caller
byte limit and admits expanded parse memory before reading. It requires exact
policy configuration, scorer, profile, history metadata and format/eligibility/
allocation revisions. Reload checks sorted member IDs, per-member revisions,
score and vector consistency, labelled support and calibration summaries, and
recomputes routing and budget allocation against the saved evidence. Replay
against the original inputs and artifacts checks whether that evidence reproduces.
Distances and policy settings round-trip through binary64-precise JSON parsing.
Replay compares the full reports except adaptive memory
telemetry, with comparison memory preflighted. Policy, calibration, evidence or
input changes produce a replay mismatch. Returned data remains budget-held until
the caller drops it and restores the incoming hold. An interruption can leave an
auxiliary staging file. Published reports remain complete.
