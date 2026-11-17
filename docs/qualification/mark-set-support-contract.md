# MarkSet grouping and local support (task 2.8)

`l2d_decide::close_enough` is a library boundary. CLI integration belongs to 2.9,
and operational policy, queue allocation and broader compatibility belong to 2.20.
Local support is not transfer or deployment qualification. The exploratory 2.18
comparison did not demonstrate an agnostic MarkSet score; it supplies no default
operational reach, ratio threshold or quality guarantee.

## Source and grouping

`MarkHistory::from_store` validates both MarkSets against the store, checks the
occurrence snapshot before/after reading, and retains genuine stored group keys.
`from_input` accepts an adapter's logs and matching input summary. Histories are
immutable snapshots. Call `history.require_store` before consuming decisions
against a live store; if it fails, rebuild the history. A batch consumer should
check again before publishing its batch if concurrent writers are possible.
Unknown group keys stay unknown. Pattern IDs are never endpoint identities.

Grouping and proximity reuse exploration's `DirectExemplar` engine. The caller
explicitly supplies either the pinned `HierarchicalSession` or the named
`CharacterTrigram` route and its expected profile. There is no automatic CPU
encoder fallback. Original texts are re-encoded, so legacy vectors in a store
cannot silently substitute another profile. NPU originals beyond the pinned
1024 UTF-8 bytes are unsupported and never truncated.

Successful encoding can return invalid data: zero/non-finite vectors or invalid
comparison dimensions are unsupported representation outcomes, distinct from an
encoder/runtime error. Grouping records those members once and continues with
valid members; proximity skips/counts them without counting labelled support.
Both reports retain member-specific reasons in `unsupported_details`. An invalid
query returns an unsupported reason with no ratio or calibration. Actual encoder
errors still abort the operation and restore its incoming memory hold. No error
class is inferred from an error message or treated as an automatic fallback.

Groups form a deterministic partition: the first unassigned marked member in
stable member order is the real exemplar; every member is directly within its
radius. This is not transitive connectivity or an all-pairs matrix. Group size
counts source members, distinct texts are reported separately, and the distance
range includes the exemplar at zero. Occurrences remain descriptive counts.
`Grouping::deselect` verifies exact current marks and the store snapshot, then
records every group member through the existing MarkSet deselection methods.
Persist that MarkSet with its existing save operation and refit before scoring
against the changed history. Applying a stale grouping is refused.

## Support and ratio

Both the primary MarkSet and its explicitly reviewed opposite supply labelled
evidence symmetrically. Unmarked originals have `AssumedBackground` provenance
and no label; they never become verified good or count toward minimum support.
De-selected or conflicting copies exclude that entire original text before
radius and top-k. Source-qualified IDs and repeated occurrences cannot inflate
independent support. A duplicate original with differing group metadata becomes
unknown, not cross-endpoint evidence.

`ReachSettings` has no default operational thresholds. The caller supplies the
radius, minimum labelled support and top-k. The returned evidence keeps up to
top-k distinct labelled originals and, separately, top-k distinct background
originals within the radius. Background cannot evict explicit reviewed support.
Counts and per-label mean distances describe these returned neighbours. The
same/cross/unknown group counts include returned evidence of both provenances;
the explicit label and background counts distinguish them. Distinct cross-group
keys are reported separately. The report records source, occurrence snapshot,
profile, settings, batch plan and scorer revision.

Insufficient labelled support gives a review reason and no ratio. Benign-only
support with no known cross-endpoint evidence and no exact original match also
gives a template/payload-support review reason and no ratio. Exact whole-original
equality is the only positive payload-match measurement this API makes; it does
not infer anomaly absence from a template, invent a payload classifier, or claim
that nonidentical payloads cannot match. This conservative choice can abstain
on a genuine same-endpoint payload match. Unknown group metadata cannot supply
cross-endpoint reassurance. An eligible result is exactly `scorer.ratio(text)`,
including its saved redactions; neighbours never average, lower or replace it.

The history must match the fitted scorer's saved MarkSets exactly. The upstream
`Scorer` implementation now exposes `ScorerKind::MarkSet` (`mark-set`) for such
scorers, matching their existing score frames. Calibration must match that kind
and the full fitted revision. Legacy calibration tagged `baseline` is refused
even when its revision string matches; recalibrate under the correct identity.
Existing baseline/neighbour serialization tags remain 0/1, and MarkSet uses 2.
Existing baseline calibration fixture bytes are unchanged. A compatible
calibrator supplies the existing label set, error rate and group/pooled strata;
an abstained result has no calibration prediction.

## Resources and verification

History loading is explicitly whole-history and bounded: it preflights metadata,
duplicate indexes, mark copies and output copies, refusing before loading if
they do not fit. Grouping has no all-pairs allocation. Queries and groups encode
one original at a time and recheck memory between candidates. Encoder scratch,
two live representations, scorer/redaction scratch, calibration and result
copies are reserved before allocation. Grouping/scoring use the shared heavy-job
guard. Failed operations restore the incoming budget hold. Successful history,
group and decision results remain held until the caller drops them and releases
their reservation. Callers also hold already-loaded inputs and NPU session memory
(`SESSION_BYTES`, currently a conservative 1 GiB estimate).

`tests/close_enough.rs` exercises a real SQLite store, saved/reloaded MarkSet
scorer, grouping/deselection, near/far evidence, duplicate/conflict exclusion,
background isolation, primary/opposite symmetry, template refusal, compatible
and incompatible calibration, stale snapshots, profile/memory refusals and
unsupported originals. Its opt-in pinned-NPU test compares every evidence
distance with independently re-encoded vectors and checks the exact scorer ratio.
`tests/close_enough_memory.rs` measures Rust heap growth including JSON copies,
input grouping/deselection, refusal before encoding and restoration on encoder
failure. This is not measurement of all native provider allocations and does
not close Requirement 15.2 for the provider.
