# Amendment draft: a fitted linear head, an outlier ranking and richer evidence

Status: **draft, awaiting owner approval** (2026-11-12). Nothing here is part of
the approved spec yet. On approval the parts below are merged into
`requirements.md`, `design.md` and `tasks.md`.

Background: `docs/2026-11-12-venn-vdm-gap-list.md` compares Venn's operations
with the spec. The owner chose the spec plus three small additions, to be
built after the NPU encoder, and confirmed that the baseline ratio and the
vector scorers stay separate scorers for now. The decision to investigate,
which brings them together, is later work and not part of this amendment.

## 1. One correction to the gap list

The gap list names "spread and mean distance" as the missing evidence
figures. Venn's spread is the standard deviation of numeric outcomes among the
neighbours. With the labels good and bad it follows from the ratio and adds
nothing. This draft therefore adds, per label, the count and the mean distance
of the neighbours used, and leaves spread out.

## 2. Requirements (proposed Requirement 14)

### Requirement 14: Fitted linear head and outlier ranking on vectors

**Objective:** As a log analyst, I want a second decision model on the encoder's
vectors that is fitted to my labelled logs, and a signal for logs unlike
anything stored, so that I can compare it with the nearest-neighbour ratio and
the baseline, and see what is new.

#### Acceptance Criteria

1. When labelled stored logs are supplied for fitting, the Log Triage Tool
   shall fit a linear head on their vectors that returns a bad/good ratio
   between 0 and 1 for a vector, where higher means more likely bad.
2. If fewer than ten labelled logs, or fewer than five of either label, are
   supplied for fitting, then the Log Triage Tool shall refuse to fit and
   state the counts it found.
3. Where group keys exist, the Log Triage Tool shall choose the head's penalty
   by cross-validation in which all logs of a group fall on one side of every
   split, and shall report the penalty chosen and the score of each candidate.
4. The Log Triage Tool shall produce the same head and the same ratios when
   fitting is repeated on the same logs with the same settings.
5. If a vector from one encoder identity is given to a head fitted on another,
   then the Log Triage Tool shall refuse and name both identities.
6. When a head is saved, the Log Triage Tool shall give it a revision derived
   from its content and record the encoder identity and the logs it was fitted
   on, so that calibration can exclude them.
7. When a log is triaged with the linear head, the Log Triage Tool shall name
   the linear head as the scorer that produced the ratio.
8. The Log Triage Tool shall calibrate the linear head under Requirement 11 in
   the same way as any other scorer.
9. When an evaluation is run, the Log Triage Tool shall report the linear head
   next to the baseline scorer and the neighbour scorer on the same splits.
10. When the analyst requests an outlier ranking for a set of logs against a
    store, the Log Triage Tool shall return for each log the distance to its
    k-th nearest stored log and rank the logs from the largest distance down.
11. The Log Triage Tool shall describe the outlier figure in its output as a
    distance and not as a probability.
12. If the store holds fewer logs than the outlier ranking needs, then the Log
    Triage Tool shall refuse and state the number stored and the number
    needed.
13. When evidence is reported for a neighbour-based decision, the Log Triage
    Tool shall report, for each label, the number of neighbours used and their
    mean distance.
14. The Log Triage Tool shall keep fitting a head and ranking outliers within
    the memory limit of Requirement 12, refusing before loading when the
    vectors needed do not fit.

Change to existing text: requirement 6.6 lists "the baseline scorer, the
encoder-based scorer, or both"; it gains the linear head as a named scorer.

## 3. Design (proposed additions)

**LinearHead (`l2d-decide`, `head.rs`)**

- Model: logistic regression on the stored, L2-normalised vectors, fitted with
  the deterministic regression of task 1.6 through its dense path
  (`l2d-baseline` is already a dependency of `l2d-decide`). No new crate.
- Penalty: chosen from 0.01, 0.1, 1, 10, 100 (the grid Venn's linear family
  documents) by 5-fold grouped cross-validation on log loss. Folds are
  assigned from a hash of the group key, so they do not depend on row order.
  Without group keys the folds are assigned from the log id, and the report
  says that the splits were not grouped.
- No abstention of its own. Uncertainty comes from the calibration layer.
- Identity: the head stores the encoder identity of its vectors and refuses
  others.
- Saved under `models/heads/<name>-<rev12>/`: weights, bias, penalty,
  cross-validation scores, encoder identity, and the hashes of the fitting
  texts (the per-log held-out check of task 1.11 applies to it as to the
  baseline scorer). The revision is the SHA-256 of that content.
- `ScorerKind` gains `Linear`. Triage and calibration take it like any scorer.
- Memory: fitting holds all labelled vectors (`require_whole`); scoring runs
  in batches.

**Outlier ranking (`l2d-decide`, on the search of task 1.9)**

- Figure: squared L2 distance from a log's vector to its k-th nearest stored
  log, k = 5 by default (Venn's `knn5`), option `--k`.
- A log that is itself stored does not count as its own neighbour.
- Computed in batches of query vectors against the stored vectors, under the
  memory budget. Refused when the store holds fewer than k logs besides the
  query.
- Output: the ranking with the distance, described as a distance. It is a
  signal for the later decision to investigate, not a score.

**Evidence**

- `Evidence` gains, per label, the count and the mean distance of the
  neighbours used.

## 4. Tasks (proposed, in milestone 2 after 2.8)

- [ ] 2.10 Fitted linear head on vectors
  - `LinearHead` in `l2d-decide`: fit with grouped cross-validation for the
    penalty, score, save, load, identity check, hashes of fitting texts
  - `ScorerKind::Linear`; triage and calibration accept it
  - Tests first: refusal counts; determinism; identity mismatch; grouped folds
    keep a group on one side; calibration of the head on the fixture
  - Done when a head fitted on the fixture store gives ratios, survives save
    and load with the same revision, and is calibrated and triaged like the
    baseline
  - _Depends: 1.6, 1.13, 2.7, 2.8_
  - _Requirements: 14.1 to 14.8, 14.14_

- [ ] 2.11 Outlier ranking and evidence figures
  - Distance to the k-th nearest stored log in batches under the memory
    budget, with the refusal for small stores
  - Per-label count and mean distance in `Evidence`
  - Done when a test store with a planted far log ranks it first, and evidence
    for a known neighbourhood shows the expected counts and mean distances
  - _Depends: 1.9, 2.8_
  - _Requirements: 14.10 to 14.14_

- Task 2.9 (command wiring) gains `l2d head fit`, `l2d head score` and
  `l2d outliers`. Task 3.7 (evaluation harness) gains the linear head as a
  third scorer (requirement 14.9).

## 5. Open points for the owner

1. The penalty grid and five folds are Venn's documented values. The fixture
   and CSIC may want others; the task records what it finds.
2. k = 5 for the outlier figure is Venn's. Logs repeat heavily, so a larger k
   may separate "rare" from "new" better. To be measured in 2.11.
3. Not in this draft, by the owner's choice: the trained residual head, heads
   compiled from an outcome store, numeric outcomes, and the decision to
   investigate.
