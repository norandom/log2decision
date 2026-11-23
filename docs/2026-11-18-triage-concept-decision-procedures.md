# Triage concept: the model proposes, set algebra decides, the analyst judges

Date: 2026-11-18. Status: concept for owner review, cut down the same day to what is cheap or already paid for. It changes no requirement and creates no task.

**The claim.** Every stage of log2decision already produces exact, versioned, replayable facts about a population of logs. A model or a heuristic may choose where to look next. It never decides what was found, and it never decides what a finding means. Automation comes after the triage itself has been shown to beat the plain score.

## The pipeline has five stages, and each one hands over a contract

| Stage | Hands over | Exists today | Open |
|---|---|---|---|
| 1. Source | A pinned snapshot: bytes or store revision, entry filter, parser settings | File, ZIP, local store | Remote APIs |
| 2. Dataset U | Distinct originals, their occurrences, both counts, skipped and damaged input | Streaming readers, disk dedup, adaptive batches, spill | One scoped identity for score, filter and policy keys (triager 1.1) |
| 3. Evidence | Named sets with coverage: A from the fitted score, B from search or a pattern, neighbours from NPU vectors | Score, SPL search, pattern catalogue, NPU exploration | Disk-backed membership (1.3), field and time projections (3.1) |
| 4. Decision | C from the Venn policy, regions of A/B/C with unknowns kept apart, one budget allocation, a receipt | Policy library and CLI, native workbench on retained reports | MCP adapter (1.2), replay across clients (2.2), qualification (parent 2.22) |
| 5. Flows | A hunt that runs without the analyst steering every step | The recipe and the walkthrough | The `triager` skill. Agents are parked |

A reviewed mark enters a new MarkSet revision. Nothing feeds back automatically.

## "Venn" names three things

| Name to use | What it is | Role |
|---|---|---|
| Vector Decision Models | The method family the name comes from | Background |
| Venn policy | `triage_policy`: score band plus reviewed neighbours gives investigate, review or low priority | One producer of set C |
| Region algebra | Union, intersection and difference over A, B and C in three-valued logic | The drill-down itself |

## What Venn changes, as measured

All figures are from the [96-record development panel](2026-11-18-triage-workflow-comparison.md): 69 benchmark positives, 12 reviews per workflow, one endpoint supplying 84 records.

**The policy does not rank better than the score.** Score alone found 12 of 12, the policy 11 of 12. The policy recovered three positives the score list missed and lost four. It cost 6,240 encoding calls and 91.8 seconds.

**This panel could not have shown a gain.** With 72 % positives and 12 reviews, score alone reached the maximum. A comparison needs a lower base rate or a budget the score does not saturate.

**What the policy adds is a partition with reasons.**

| Policy queue | Records | Positive | Share |
|---|---:|---:|---:|
| Unsupported | 17 | 17 | 100 % |
| Investigate, disagreement, uncertain | 18 | 18 | 100 % |
| Low priority | 61 | 34 | 56 % |
| Panel | 96 | 69 | 72 % |

- "Unlike anything reviewed" was the strongest signal on this panel, and the allocation deferred all 17 of those records.
- "Low priority" is only mildly cleaner than the panel. It cannot be used to close cases.

**At the top of the ranking the partition restates the score.** The 35 highest-scoring records are also 35 of 35 positive. The two sets of 35 differ in seven records each way, all positive. The score's first negative is at rank 42.

**Below rank 41 the unsupported route finds what the score cannot.** Ranks 42 to 96 hold 55 records, 28 of them positive. The score has no order left there that separates them.

| Ranks 42 to 96, by policy reason | Records | Positive | Share |
|---|---:|---:|---:|
| Unsupported | 7 | 7 | 100 % |
| Low priority | 48 | 21 | 44 % |
| All | 55 | 28 | 51 % |

Unsupported records sit at score ranks 2 to 96, four of them in the last twenty. So the measured difference Venn makes is narrow and specific: the vector evidence marks records unlike anything reviewed, and on this panel those were positive wherever the score put them. Seven records on one development panel with one dominant endpoint is a lead, not a result.

Computed on 2026-11-18 from the frozen score order, the persisted policy decisions and the benchmark labels of the 2.18 export, whose hash matched the panel manifest. No new run.

**The region algebra is the part that needs no model.** "Which high-scoring records did my search miss" is answered from A and B alone, with counts before text and unknowns kept apart from negatives. Its value is analyst time per confirmed case, which is not yet measured.

## Two cheap things that follow

**An exhaustive test of the policy.** Seven yes/no facts describe a record under one policy revision: supported, good corroboration, bad corroboration, high band, low band, calibration required, calibration compatible. High and low exclude each other, which leaves 96 cases. One test enumerates them and checks that each reaches exactly one queue, that every queue is reachable, and which cases change route between two revisions.

**Own review slots for unsupported records.** Disagreements filled the two slots that unknown and conflicting records share, so every unsupported record was deferred. This follows from the allocation order in the [policy contract](qualification/venn-policy-contract.md). Owner decision of 2026-11-18: unsupported records get their own reservation, as a new exploratory policy revision. It is not a final allocation.

## RAM: one process is bounded, the machine is not

Checked against `crates/l2d-core/src/memory.rs` and `resource.rs`. A process has a budget from live free memory, re-reads it before every batch, spills or refuses, and one heavy job and one NPU job run at a time across processes.

| Missing | Status |
|---|---|
| No machine-wide sum of what processes hold. Light reads each run under their own budget | Read from the design, not provoked in a test |
| Model runtime memory is observed, not charged | The NPU session reserves 1 GiB |
| Free memory is sampled, not reserved | Read from the code structure, not provoked in a test |
| The workbench holds its report in memory | Planned in triager 1.1 and 1.3 |

Nothing is built for this until one of the rows is provoked in a measurement.

## Rule for any later automation

An automated step answers established, refuted or unknown with a reason, and its claim counts only if the receipt replays. It never creates an analyst mark and never turns unknown into negative. Receipts and three-valued membership already exist, so this is a rule and not new machinery.

## Order

1. Triager 1.1 and 1.2: shared identity and the MCP adapter.
2. The exhaustive policy test, and the policy revision with own slots for unsupported records.
3. A new exploratory panel (owner decision of 2026-11-18): held CSIC records subsampled to a low positive share on a larger panel, several review budgets reported instead of one, and the score's purity at each budget beside the policy's. The question it must answer is the one above: are unsupported records positive at a higher rate than their score rank predicts?
4. Parent 2.22 on a protected cohort, set up the same way.
5. Only if triage wins there: the `triager` skill with one automated flow, "what did the search miss".

Steps 2 and 3 need tasks in the parent spec before any work starts.

## Parked

Direction, not work. Each returns only when its precondition exists.

| Idea | Returns when |
|---|---|
| Several small hunting agents | The skill and one flow have shown a gain at equal review budget |
| A host memory ledger | A measured kill or refusal traces to memory held across processes |
| Remote sources. Splunk first, since the search language is already SPL | Local triage is qualified |
| An agent that translates SPL for other tools, tuned offline with MIPRO or GEPA, checked by counts before and after each fetch and by audited sample windows, with such snapshots marked statistically audited and never exact | A second remote source is wanted |
| Kroening and Strichman beyond propositional logic: equality for entities, difference logic for time-ordered hypotheses, combination of theories for several sources | Entity resolution, occurrence time projections or a second source exist |

## We are wrong if

- On a low-base-rate panel, unsupported records are positive at no higher a rate than records of the same score rank. Then the vector evidence adds nothing to the score and the NPU cost of the policy is not earned.
- The exhaustive policy test finds nothing the existing tests of `policy.rs` do not already cover.
- Receipts and replay cost more analyst time than the reduction saves.

## Decided on 2026-11-18

- Unsupported records get their own review slots, in a new exploratory policy revision.
- The next comparison uses a low base rate and a sweep of review budgets.
- The top-35 comparison was computed from the frozen files. Result above.
- The memory gaps stay parked until concurrent processes or a local model exist.

Final review capacity, missed-case targets and final allocations remain unset, as decided on 2026-11-17.

Sources: [triager requirements](../.kiro/specs/triager/requirements.md) · [design](../.kiro/specs/triager/design.md) · [recipe](../.kiro/specs/triager/recipe.md) · [performance](../.kiro/specs/triager/performance.md) · [Venn gap list](2026-11-12-venn-vdm-gap-list.md) · [vectorization assessment](2026-11-18-vectorization-assessment.md)
