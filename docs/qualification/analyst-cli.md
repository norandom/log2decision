# Using the analyst CLI

All commands accept `--project DIR`, `--json` and the global memory and busy-host settings. Diagnostics go to standard error. A policy decision is a suggested queue. Only explicit analyst marks supply reviewed labels.

## Encode originals and inspect distances

```text
l2d runtime check
l2d encoder info --profile-dir PACKAGE
l2d encoder encode --profile-dir PACKAGE --input originals.txt
l2d store add --store history --profile-dir PACKAGE --input originals.csv --group-column endpoint
l2d outliers --store history --profile-dir PACKAGE --input future.txt --k 5
```

The default package location is `models/encoders/f0-s0-mask` under the project. The adapter verifies the two graphs, lookup tables and layout before opening the pinned Ryzen AI runtime and checking the NPU driver. The run record includes the source hashes, actual node placement, first-execution evidence for each graph, driver, input plan and measured encoding rate.

The fixed profile accepts original UTF-8 records up to 1,024 bytes. Encoding output includes each position, occurrence count, support status and the separate cut, length and empty-input flags. The store and outlier commands require a supported, finite, nonzero vector for every submitted original. An existing store must match the encoder identity and runtime release. Outlier results report squared L2 distance to the k-th neighbour, with identical stored query IDs excluded. The encoder emits raw pooled f32 vectors. Cosine retrieval normalizes during distance calculation.

## Prepare marks and fit a score

```text
l2d mark --store history --phrase denied --label bad --name denied
l2d mark --store history --regex accepted --label good --name accepted
l2d group --mark-set denied --method character-trigram --radius RADIUS
l2d group --mark-set denied --method character-trigram --radius RADIUS --deselect-group INDEX --expect-grouping REVISION
l2d redact --mark-set denied
l2d redact --mark-set denied --rules confirmed-rules.json
l2d score fit --mark-set denied --other-mark-set accepted --name analyst-score --out fit-evidence
```

Choose the grouping radius for the named method. The NPU method accepts `--profile-dir`. Grouping and both mark-based triage modes report the completed encoding calls, throughput and provider placement separately from replayable decisions. The character-trigram method compares Unicode trigrams locally. The group preview identifies the real representative, farthest direct member, spread, occurrences, distinct originals and endpoint support. Individual exclusions accept `--deselect-id` for stores or `--deselect-position` for inputs. A preview revision binds a group index to its members, settings and source.

A rules file is an ordered JSON array. For example, `[{"kind":"literal","fragment":"host-A","placeholder":""}]` removes one confirmed literal. Previewing proposals shows marked and unmarked counts. Applying a file replaces the confirmed list, so omitted proposals stay out of fitting. The fitted scorer saves its rules, mark copies and fitted-text hashes. Without an opposite MarkSet, the score uses the unmarked logs as assumed background.

File and archive marks and discovery selections record a resolved absolute source location, so reopening them from another directory uses the same input. An older relative reference has no saved working directory and requires a fresh search or mark.

New file marks bind the complete ordered position, original text, label, endpoint and occurrence mapping. Archive marks bind the archive bytes, entry patterns and original maximum record length. Older input marks without this evidence must be reviewed again under a new name. A repeated named mark reuses an identical, unrefined set and refuses to overwrite analyst refinements.

For a file, use the original column choices via `--history-adapter`, `--history-text-column`, `--history-label-column`, `--history-no-labels` and `--history-group-column` as applicable. For archives the saved entry patterns identify the historical selection. Repeat an original record-length override with `--history-max-record-bytes`. File and archive marks also accept repeated `--position` values. Store IDs remain the stable identities for stored originals.

The grouped report records the fold membership hashes, held-out and fitted-group metrics, gap, margin and warning. The group key comes from the input adapter, archive host or stored endpoint metadata. Conflicting stored keys for the same original yield an unknown key. Use `--group-pattern` to select a group from text, `--folds` for the fold count and `--margin` for the gap warning. The optional new evidence directory contains `score-redaction.json` with the scorer, marks and rules revisions, model-file hashes and the report. Its `model/` directory freezes the exact weights, fitted-text hashes, mark copies and model manifest. It records training evidence. Deployment qualification requires an independent holdout and operational targets.

## Triage and replay

```text
l2d triage --input future.txt --baseline analyst-score --mark-set denied --method character-trigram --radius RADIUS --top-k K --min-support N
l2d triage --input future.txt --baseline analyst-score --mark-set denied --method npu --profile-dir PACKAGE --policy policy.json --out decisions.json
l2d triage --input future.txt --baseline analyst-score --mark-set denied --method npu --profile-dir PACKAGE --policy policy.json --replay decisions.json
```

Supply the complete versioned policy JSON described in the [Venn policy contract](venn-policy-contract.md). It records threshold-selection provenance, evidence profile, support and corroboration, review capacity, positive unknown/conflict and audit reservations, and audit seed. Add `"unsupported_slots": N` to give unsupported records their own reservation; the unknown/conflict slots then serve disagreements alone. The CLI opens the live marks used by the score and rebuilds history from their original source. Supported decisions carry the fitted character scorer's ratio. Unsupported decisions withhold an actionable ratio and explain why. Policy output retains all decisions and allocation or deferral for each.

For calibrated evidence, create a calibration with `l2d calibrate --baseline analyst-score --input held-out.csv --name SCORE-CAL`. Then supply `--calibration SCORE-CAL` and the applicable `--error-rate` on triage. Replay checks score, policy, evidence, profile and decisions against the saved report.
