# Amendment: pattern discovery, reusable scores and Venn triage

Date: 2026-11-17. Status: recommended choices accepted by the owner and integrated
into requirements 17–20, the design extension and the active roadmap in tasks.md.
Those documents are authoritative; the proposal below is retained as rationale.
Tasks 1.26–1.28, 2.19–2.22 were added and 2.8/2.9/2.18 extended. Numeric review
budgets, allocation shares and qualification targets remain measurement inputs.
This is a CLI and saved-artifact workflow;
a graphical interface and live ingestion remain outside the current scope.

## 1. Intent and boundaries

The owner wants a score derived from log material that works on new, future log
cases. The investigation starts with scoring and/or broad string, glob or regex
search; discovers patterns without requiring a finished regex; refines candidates
using NPU vectors and cosine distance; and uses Venn-style decisions for triage.
Generalization is the acceptance criterion, not merely fit on the original logs.

Here Venn means the Vector Decision Model described in the
[existing gap analysis](../../../docs/2026-11-12-venn-vdm-gap-list.md), not set
diagrams or Venn probability prediction. Re-implement applicable operations in
Rust; do not add Venn's Python package. Boolean set operations remain useful search
features, but they do not define the decision model.

Keep the decision from gate 2.17: the character n-gram baseline learns the score;
NPU vectors support retrieval, grouping and evidence. Vector similarity is not
proof of the same outcome or of generalization. Do not reopen a vector classifier
or introduce a generic natural-language decision head through this amendment.

## 2. Recommended investigation loop

| Step | Analyst action | Saved result and purpose |
| --- | --- | --- |
| 1. Establish scope | Choose files/archive entries/store snapshot and, when available, time/source filters | Dataset identity, completeness, filters and record identities; all later counts refer to this scope |
| 2. Survey | Run an existing score, or proceed without one; inspect high scores, rare patterns and a background sample | Scoring revision and scored/unscored/unsupported counts; no automatic training labels |
| 3. Search broadly | Use literal text, a content glob or regex; save and combine searches | An unlabelled candidate set with query semantics and provenance |
| 4. Extract patterns | Group candidates by recurring message structure; inspect templates, counts and variable examples | Versioned pattern catalogue and exact member assignments |
| 5. Refine patterns | Drill into groups, split on meaningful tokens/fields, merge compatible groups, retain exceptions | Revised groups, explicit exclusions, representative and boundary examples |
| 6. Explore vectors | Find nearby examples inside and across patterns; explicitly expand to the wider dataset | NPU profile, cosine distances, seed evidence and newly suggested candidates |
| 7. Review and mark | Inspect examples and exceptions, then mark good/bad or leave unresolved | Explicit labels; selected groups do not become labelled merely by being selected |
| 8. Prepare and fit | Confirm nuisance redactions; fit the score using reviewed marks and a stated comparison class | Frozen scorer, redaction rules, label provenance and validation report |
| 9. Triage future cases | Apply the frozen score and retrieve applicable historical evidence | Investigate, review, or low-priority queue with reasons; unsupported cases remain visible |
| 10. Record outcomes | Record what review established, including corrections | A new outcome-store revision and an explicit later retraining cycle |

Steps 3–7 are iterative. A precise regex is an optional product of exploration,
not an entry requirement. Existing scores can guide discovery; new investigations
can begin with search and pattern extraction when no suitable score exists.
Reserve review capacity for low-scoring, ungrouped and out-of-search cases so that
the initial query or an old score does not define everything the analyst sees.

## 3. Pattern extraction and refinement

### Analyst-facing behavior

Start with a broad query such as `login` or the content glob `*login*fail*`.
An illustrative result might be:

```text
P17  login failed for user <USER> from <IP>   620 occurrences / 180 distinct texts
P18  login failed: account locked <USER>      45 occurrences /  20 distinct texts
P19  login failed: backend unavailable        8 occurrences /   2 distinct texts
```

Placeholders are illustrative inferred slots unless an explicit masking rule
assigns a type. Each group shows a representative, varied slot values, source/time
coverage when available, and score distribution/unsupported count when scored.
Counts distinguish original occurrences, distinct texts, and reviewed members.
One representative cannot justify labelling every member.

The analyst can inspect all members, split P17 by an informative token or field,
promote a group for review, hide it from the current view, or generate a proposed
query from its template. Hiding or promoting changes review priority only.
An explicit batch-label operation previews its membership and records that the
analyst chose to apply one label to the entire group.

Keep an ungrouped/exception set. A signature is a summary, not necessarily an
executable regex or a guarantee that every member matches its displayed text.
When proposing a regex, escape literals, make variable bounds explicit, preview
its actual matches and misses against the saved membership and a background
sample, and require an explicit save before using it as a reusable search.
Do not claim measured recall from a search-selected sample alone.

### Algorithm recommendation

Evaluate a bounded Drain-style template miner first, implemented in Rust or
using a vetted Rust implementation consistent with the dependency rules. It is
a candidate, not a selected dependency or a promise to reproduce proprietary
Splunk/Sumo algorithms. Compare it with the existing character n-gram grouping
on representative formats and analyst-labelled same-pattern/different-pattern pairs.

Tokenize with punctuation and meaningful path/key/value boundaries retained.
Use configurable discovery masks for volatile timestamps, IDs and addresses.
Preserve negation, outcomes, error codes and payload tokens by default; show
collisions where different outcomes have been collapsed into one template.
Discovery masks do not authorize redaction for score fitting: those rules still
require the existing analyst confirmation and are separately versioned.

Bound the active template catalogue, parser/token buffers and membership state.
Spill memberships to disk; on reaching a template limit, report overflow as
ungrouped or refuse explicitly, rather than silently deleting patterns. Use a
canonical input order and deterministic tie-breaking; save parameters and seed.
Changing masks, splits or thresholds creates a new catalogue revision. Membership
IDs are stable within a frozen revision, not promised stable across refits.

### Search semantics to specify

Literal search means substring matching, distinct from store full-text token
search. Content glob `*` matches zero or more characters and `?` one character
within a record; match the whole record, so substring globs need surrounding `*`.
Make escaping and case sensitivity explicit; recommend case-sensitive default
with an explicit insensitive option. Archive-entry globs are a separate filter.
Regex matching uses the supported Rust regex syntax and returns match positions.
Do not implement literal substring search through FTS if their results differ.

## 4. NPU vectors as a second view of the same records

Encode original member texts with the qualified encoder profile, in bounded
batches using the existing NPU session. Keep record identity shared between
patterns, vectors, scores and marks. Pin encoder revision, dimension, codec,
normalization and any preprocessing digest; refuse incompatible vector profiles.
Templates are navigation summaries; encoding only their text loses member detail.

Use cosine distance on valid normalized vectors. Report zero/invalid vectors as
unsupported. Nearest-neighbour search and clustering may use CPU/store arithmetic;
the NPU supplies embeddings, and this proposal does not claim NPU clustering.
Without an NPU, retain the existing explicit character n-gram similarity route;
never silently substitute CPU execution of the NPU encoder.

Offer two operations: regroup within a selected pattern and find neighbours
across patterns. The latter is necessary because lexical pre-filtering can exclude
other expressions of the same event. Expansion outside the saved search scope
is explicit and preserves why each new candidate was suggested.

Start with bounded top-k/radius neighbourhoods around reviewed exemplars. Avoid
requiring an all-pairs distance matrix or knowing the number of clusters upfront.
For materialized groups, show a real representative, distance spread and boundary
members; document any chaining rule. A chain of similar pairs must not be
presented as evidence that every member is close to the reviewed exemplar.
Choose distance thresholds from measured retrieval quality, not a universal
cosine cutoff. Duplicated occurrences must not inflate independent support.

## 5. Learning a score that transfers

An unlabelled candidate set is a new object, separate from the existing labelled
MarkSet. Searching, cluster membership, promotion and high scores never assign
ground-truth labels. Review can emit compatible good/bad MarkSets; unresolved and
excluded records stay identifiable and outside supervised labels.

The existing default comparison class is unmarked input (Requirement 16.6).
Keep that compatibility but describe it as an assumed background class, not
verified good logs. Recommend explicit reviewed good examples for qualification,
including hard negatives from similar templates. Record conflicting labels and
exclude them until resolved. Review-priority outcomes are separate from good/bad
labels: an urgent operational failure need not be a malicious log.

Freeze the discovery/training material, approved redactions, model, neighbour
evidence store and policy before final evaluation. Learn data-dependent masks,
redactions, template rules and thresholds from development data only. Do not use
final holdout labels or content to improve those artifacts and then call that
same holdout unseen. Do not silently adapt them during a future-data scoring run.

Evaluate both unseen groups (hosts/endpoints as applicable) and later time windows
when timestamps exist. Prevent exact/near-duplicate leakage and record the split
policy; report exclusions and folds with insufficient support. When no suitable
time data exists, report temporal generalization as untested. Evaluate on material
beyond the seed query, using independent reviewed labels rather than the query
itself as truth. Do not claim universality from one dataset or one regex.

Report pooled and per-fold ROC AUC, precision/recall at the review budget,
precision-recall AUC where applicable, abstention/coverage, and quality among
supported cases. Include unsupported positive cases in end-to-end missed-case
accounting. Compare score alone, pattern-assisted review, and score plus vector
evidence on the same split and review budget. The current 0.05 AUC gap is a warning
criterion, not sufficient evidence of good absolute performance. Deployment
thresholds and acceptable missed-case rates require a stated use case.

## 6. Venn-style decision policy: where evidence and score meet

Use the nearest-reference/outcome-store pattern: retrieve relevant reviewed
historical cases, apply eligibility filters before top-k, check reach and minimum
independent support, and return evidence or abstain. Keep outcome labels, source,
time, profile and provenance on each case. Report per-label counts, distances,
same-group/cross-group support and conflicting outcomes. A neighbour vote is not
a calibrated probability and is not a replacement for the fitted baseline score.

The first policy should be explicit and inspectable rather than a new learned
fusion model. Recommended routing, with support/validity checks taking precedence:

| Evidence and score | Proposed queue | Explanation |
| --- | --- | --- |
| Invalid input/profile, insufficient support or out of reach | Review: unsupported | No actionable mark-set ratio, following 16.10; retain novelty and any separately identified diagnostic score |
| Supported, high score, corroborating reviewed bad cases | Investigate | Show ratio, evidence and applicable calibration label set |
| Supported, score conflicts with neighbours or labels are ambiguous | Review: disagreement | Preserve both signals; do not average away the conflict |
| Supported, low score, corroborating reviewed good cases | Low priority | Keep searchable and include an audit sample; do not delete or auto-close |
| Supported, all other cases | Review: uncertain | Explain which threshold or evidence condition was unmet |

Here high/low, reach, minimum support and corroboration are versioned policy
parameters to choose on development data. Existing minimum-support requirements
remain the floor. Conformal label sets may inform routing only where calibration
is valid for the scorer and population. Missing calibration is shown explicitly.
Novelty is a review signal, not evidence of badness. Review budgets rank within
queues and record deferred counts; they must not discard unsupported cases.

Venn-style similarity gates may reject valid new examples whose surface template
changed. Measure that loss directly: report coverage and missed positives for the
gate, rather than claiming a high supported-case AUC proves future usefulness.
If gate 2.17's endpoint bias makes the gate too restrictive, bring a measured
revision of 16.10; do not silently bypass it or lower the threshold to pass.

Record analyst outcomes separately from suggested routes. New outcomes can enter
a subsequent evidence-store revision. Retraining, recalibration and policy changes
produce new revisions and require revalidation; no automatic label feedback loop.

## 7. Acceptance cases and spec integration

The implementation proposal must cover these observable cases:

1. A broad search yields useful templates without a hand-written full regex;
   drill-down and split preserve membership and occurrence counts.
2. Success/failure or negation variants are inspectable and not silently labelled
   together; every candidate remains accounted for, including ungrouped records.
3. A proposed regex reports differences from its source cluster; content glob,
   literal substring, FTS and archive-entry filters have distinct tested semantics.
4. Search and vector expansion preserve provenance; an outside-query neighbour
   remains unlabelled until review. Near duplicates do not manufacture support.
5. A frozen run is reproducible; incompatible profiles, stale input identities,
   invalid vectors and resource exhaustion give explicit results/refusals.
6. Unseen-group and temporal results expose leakage and unsupported positives;
   end-to-end triage is compared at equal analyst budgets, not only by AUC.
7. Policy conflicts and insufficient support route to review with evidence;
   replaying the same artifact revisions produces the same queues.
8. File, archive and store paths stay within Requirements 12 and 15, with memory
   charged for catalogues, memberships, vectors, queues and persistence buffers.

Proposed work packages, deliberately not assigned active task numbers yet:

| Package | Integration |
| --- | --- |
| Unlabelled selections and explicit query semantics | Extend 1.21/1.25 and 16.1 without breaking existing MarkSet files; glob is new scope |
| Pattern extraction and iterative refinement | Extend 16.2/16.3 with templates, exceptions, split/merge and query previews |
| NPU exploration and evidence | Separate grouping/navigation from close-enough logic currently combined in 2.8; reuse 1.9 and 2.7 |
| Generalization qualification | Extend gate 2.18 beyond seed-query labels; add independent review and later-time evaluation where available |
| Decision policy and feedback | Add an explicit decision-to-investigate task; connect 2.8, 2.11 and calibration provenance for mark-set scorers |
| CLI and saved artifacts | Extend 2.9 after the contracts are defined; no GUI implied |

Recommended sequence: finish/verify the pending 1.23 wrap-up; retain 1.25 and 1.24
as the command/resource foundation; add selections and lexical discovery; qualify
the score and vector retrieval; implement the explicit triage policy. Reconcile
this ordering and the scope of gate 2.18 in tasks.md when integrating the amendment.
Do not mark existing work completed or reorder active tasks through this draft.

Decisions for refinement, with recommendations:

- First decision target: keep good/bad labels for the learned score and separate
  investigate/review/low-priority queues. Later domain-specific outcome families
  should be explicit additions, not inferred from textual similarity.
- First pattern engine: evaluate Drain-style mining against n-gram grouping;
  choose based on review usefulness, rare-pattern retention, memory and speed.
- Comparison class: preserve the current background option, while requiring
  independent reviewed labels for claims of generalization.
- Qualification: choose an analyst review budget and acceptable missed-case rate
  before setting deployment thresholds; no universal numeric defaults proposed.

## 8. Sources and interpretation

Consulted 2026-11-17. Product behavior informs the workflow; this proposal does
not assert algorithmic equivalence or measured performance in log2decision.

- [Splunk Patterns](https://help.splunk.com/en/splunk-cloud-platform/search/search-manual/9.3.2408/retrieve-events/identify-event-patterns-with-the-patterns-tab): pattern exploration is based on search results and may use a subset. Our proposal makes coverage explicit.
- [Sumo Logic LogReduce](https://www.sumologic.com/help/docs/search/behavior-insights/logreduce/detect-patterns-with-logreduce/): signatures summarize similar structures, with variable placeholders, drilling into messages, editing/splitting and an Others group. A signature is not necessarily an exact matcher.
- [Drain3](https://github.com/logpai/Drain3): an open log-template mining reference; candidate algorithm family only, not a Python runtime dependency.
- [Venn](https://markrussinovich.github.io/venn-vdm/): typed decisions over embeddings and recorded outcomes. The local Rust policy and the NPU byte encoder require their own validation.
- Local authority: [requirements](requirements.md), [design](design.md), [tasks](tasks.md), [prior analyst-flow amendment](amendment-2026-11-15-analyst-flow.md), and [Venn gap analysis](../../../docs/2026-11-12-venn-vdm-gap-list.md).
