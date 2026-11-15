# Amendment draft: from the analyst's marks to an agnostic score

Status: **approved by the owner on 2026-11-15 and merged** into
`requirements.md` (Requirement 16; 6.1, 6.6; 14.1 to 14.9 deferred; boundary
context), `design.md` ("MarkSet, Redaction and the mark-set score",
traceability, implementation order) and `tasks.md` (1.21 to 1.23, 2.18; 2.8
re-scoped; 2.10 deferred; 2.9 extended). Where this file and those documents
differ, those documents win. This file is kept as the record of the proposal.

## 1. Why

The owner, 2026-11-15: the score must be derived in an agnostic sense. A score
that works because it has learned certain hosts or endpoints is a dataset-specific
model, and that is against the idea of the tool. The intended flow:

1. the analyst marks n samples (grep, regex, and the like);
2. the samples are grouped by string similarity, and the analyst de-selects;
3. hardcoded fragments are redacted from the training data;
4. the score is built;
5. Venn decides whether a new log is "close enough".

Owner decisions that shape this draft:
- **The baseline scorer classifies logs; the encoder serves search, grouping and
  near-duplicate views** (gate 2.17, 2026-11-15). Its vectors do not classify on
  endpoints they have not seen: linear 0.722 ± 0.070 against the baseline's
  0.923 ± 0.097 on the same folds.
- **Redaction is proposed by the tool and confirmed by the analyst**
  (2026-11-15).

What the spec has today is a scorer fitted on a labelled dataset. CSIC and AIT
labels stay in the spec, but their role changes. They measure whether a score is
agnostic, for example by standing in for an analyst's marks on held-out
endpoints. They are not the source a product score is trained on.

## 2. Requirements

### Proposed Requirement 16: From the analyst's marks to an agnostic score

**Objective:** As a log analyst, I want to mark a few logs I care about, have the
tool group them, strip what is specific to my hosts and endpoints, and build a
score from them, so that the score finds the same kind of log elsewhere and not
only on the hosts I marked.

#### Acceptance Criteria

1. When the analyst marks logs by a phrase, a regular expression or a list of
   log positions, against an input or a store, the Log Triage Tool shall save
   the marked logs as a named mark set. The mark set records the query, the
   label the analyst gives it (bad or good), the input's content hash and the
   number of logs marked.
2. When a mark set is grouped, the Log Triage Tool shall group its logs by
   string similarity and show each group with its size, a representative log,
   and the spread of distances within it.
3. When the analyst de-selects groups or single logs from a mark set, the Log
   Triage Tool shall keep them out of fitting and record what was de-selected.
4. When a mark set is prepared for fitting, the Log Triage Tool shall propose as
   redactions the fragments that recur across the marked logs and are also
   frequent in unmarked logs of the same input (host names, addresses, URL
   paths, fixed parameter names and values). Each proposal shows how many
   marked and unmarked logs it occurs in.
5. When the analyst confirms, edits or rejects a proposed redaction, or adds
   one, the Log Triage Tool shall apply only the confirmed redactions and record
   them as rules.
6. When a score is fitted from a mark set, the Log Triage Tool shall fit the
   baseline scorer (Requirement 2) on the redacted texts. It shall save with the
   score the mark set, the de-selections, the redaction rules and the hashes of
   the fitting texts.
7. When a log is scored with a score fitted from a mark set, the Log Triage Tool
   shall apply the score's redaction rules to the log first, the same rules as
   at fitting.
8. If a mark set holds fewer logs after de-selection than Requirement 2.3
   needs, then the Log Triage Tool shall refuse to fit and state the counts.
9. When a score is fitted and group keys exist, the Log Triage Tool shall report
   its ratio quality on groups held out of fitting next to its quality on the
   groups it was fitted on. It shall state when the held-out figure is the lower
   one by more than the configured margin: the score then depends on the groups
   it has seen.
10. When a new log is triaged against a mark set, the Log Triage Tool shall
    decide whether it is close enough to the marked logs: the nearest marked
    and unmarked logs within reach, with the minimum support of Requirement
    6.3. For a log that is not close enough, it shall state that and not give
    the mark set's ratio.
11. When a log is close enough, the Log Triage Tool shall give the score's ratio
    with the evidence of Requirements 6.2, 6.7 and 14.13, and the label set of
    Requirement 11 where the score is calibrated.
12. The Log Triage Tool shall keep marking, grouping, redaction and fitting
    within the memory limit of Requirement 12, and under Requirement 15.

### Changes to existing requirements

- **6.1** "derived from stored logs with known labels" becomes "derived from the
  analyst's mark sets (Requirement 16) or from a labelled dataset".
- **6.6** gains "a score fitted from a mark set" as a named scorer.
- **14.1 to 14.9 (linear head on vectors):** deferred. The number is kept so
  that later numbers do not move. Reason: gate 2.17.
- **14.10 to 14.14 (outlier ranking, evidence figures):** kept. "Unlike
  anything stored" is a use of the vectors that 2.17 supports.
- **Boundary Context, in scope:** add "a score built from the analyst's marks,
  with redaction of host- and endpoint-specific fragments". The existing
  "agnostic threat scoring barrier" bullet is the principle Requirement 16
  carries out.

## 3. Tasks (proposed)

| Task | What | Replaces |
| --- | --- | --- |
| 1.21 | Mark sets: phrase, regular expression and positions against an input or a store; saved and listed (`l2d mark`) | new |
| 1.22 | Redaction: proposals (frequent in marked and unmarked logs), confirmed rules, applied at fitting and at scoring | new |
| 1.23 | A score fitted from a mark set on the redacted texts; the held-out-group report (16.9) | new |
| 2.8 | Re-scoped: grouping of a mark set by vector similarity (16.2, 16.3), and the "close enough" decision with reach, minimum support and evidence (16.10, 16.11) | the neighbour scorer as a classifier |
| 2.10 | Deferred (linear head on vectors) | |
| 2.11 | Kept: outlier ranking and evidence figures | |
| 2.18 | Gate, a measurement: does redaction make scores agnostic? On the 2.17 folds, with CSIC attacks found by regular expression standing in for an analyst's marks: baseline and encoder each with and without redaction, figures on held-out endpoints, means and spread | new |
| 2.9 | Wiring extended by `l2d mark`, `l2d group`, `l2d redact`, `l2d score fit` | |

Order: 1.21, 1.22, 1.23 first. They need neither the NPU nor the encoder and
make steps a, c and d usable with the baseline. Then 2.18, which measures
whether redaction holds up. Then 2.8 (steps b and e) and 2.9.

## 4. Points decided by the owner (2026-11-15)

1. **The other class:** by default the unmarked logs of the same input stand in
   as the other class, and the output says so. The analyst may also mark good
   logs. (16.6 fits against the background unless good marks are given.)
2. **The group key:** one default key per kind of log (CSIC: the URL path;
   AIT: the host), named in the output and settable by the analyst.
3. **The margin of 16.9:** 0.05 ROC AUC, to be checked by gate 2.18.
4. **Grouping without an NPU:** step b groups by character n-gram similarity
   (the baseline's features), with the method named in the output.

The owner approved the draft as a whole on 2026-11-15.
