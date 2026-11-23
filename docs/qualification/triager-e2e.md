# Triager end-to-end acceptance

The analyst's question is “Which high-scoring events did my filter miss, what evidence supports them, and which ones will I review?” Both Kiro specs use the same investigation to answer it. The parent spec owns the evidence producers. The triager spec owns how an analyst explores that evidence and hands off a reproducible work set. [Parent roadmap](../../.kiro/specs/npu-log-decision/tasks.md) · [Triager roadmap](../../.kiro/specs/triager/tasks.md)

## Drive the workbench

These examples use an installed `l2d` on your `PATH`. For the PowerShell command that runs the build directly from a source checkout, see the [quick start](../../README.md#try-it).

```sh
l2d tui --demo --session hunt.json
```

Press `1` for A − B, then `c` to compare recommendation coverage. Choose `i` for recommended, `o` for assessed passes or `u` for unknown. Press Enter on an original to read its evidence. Use `h` for host, `t` for UTC hour and `/` to search inside the current view. The editor opens in SPL-style search mode. Tab cycles through literal, whole-record glob and regex, then back to SPL. `b` returns to the parent. `v` revisits a retained branch. `[` and `]` resize the fields pane, and `-` and `+` resize the search pane.

From the demo root, enter `host=edge-01 (login OR challenge) score>=0.9`. R01, R02 and R03 match. Their four matching occurrences feed the histogram and sidebar. Press `1` on this result to narrow to R03 and its two occurrences at edge-01. R03 also occurred on idp-01 in the original dataset. That occurrence stays outside this branch, including after saving and reopening it.

| Search expression | What it selects |
|---|---|
| `"challenge rejected"` | Case-insensitive phrase in the retained raw text. Unquoted terms are implicit AND. |
| `id=R0* score>=0.9` | Records whose ID matches the wildcard and whose retained numeric score meets the threshold. |
| `(login OR session) NOT "cache cleared"` | Either raw term, excluding the phrase. Parentheses make grouping explicit. The search operators bind in the order NOT, OR, AND. |
| `c=recommended` | Records the active C producer recommended. Other C values are `passed` and `unknown`. |
| `c=recommended \| search score>=0.5` | A second `search` stage keeps recommendations whose retained score is at least 0.5. On a policy report this is the saved diagnostic ratio. |
| `host=edge-* _raw="*challenge*"` | Occurrences on matching hosts whose retained record contains “challenge”. Reports without host data return an explanation when a query needs that field. |

Use `--query 'EXPRESSION'` to apply the same search as a child of the starting view, including a reopened candidate. Use `--match-mode spl --filter 'EXPRESSION'` to define set B from original-level fields when starting an investigation. A query refines the current branch and keeps the original A/B definitions. The pipeline supports `search`, `where`, `regex`, `rex`, `fields`, `head`, `tail`, `sort` and `dedup`, with Splunk term segmentation. Commands that build tables, such as `stats`, are refused. The separate regex editor uses Rust regex syntax. [Search language reference](../2026-11-18-triager-search-language.md) · [Splunk search command primer](https://help.splunk.com/en/splunk-enterprise/search/search-manual/10.4/search-primer/search-command-primer)

Press `r` on a result or in its evidence detail to read the full retained log text. ↑/↓ move through distinct originals in the active view's order. In a final work set, that is allocation order. Use PgUp/PgDn to read a long record a screen at a time, or ←/→ to jump between chunks. Esc returns to the row or detail that opened the browser. For a policy report, the text comes from each decision’s saved input. Line breaks remain visible. The detail pane escapes control characters and doubles literal backslashes so you can distinguish a tab from the two characters `\t`. Byte ranges refer to the retained text before escaping.

Press `s` to save a candidate before spending any review slots. At the desired parent, press `f` to edit the general, unassessed and audit capacities. Enter confirms the allocation and fixes the ordered IDs and pool assignments. To try another budget, press `b` and allocate from the candidate. Use uppercase `S` to save to a fresh path and keep the draft. An update checks the revision the analyst opened, so a stale session can save its work under a different name. Reopen using the same source and receipt. Add `--json` to inspect the active counts and up to 16 ordered IDs without entering the terminal.

```sh
l2d tui --report policy-report.json --policy-recommendations --top-k 8 --match-mode glob --filter "*login*" --session review.json
l2d --json tui --report policy-report.json --session review.json
```

Use the file written by `l2d triage --policy … --out policy-report.json`. A ranks saved diagnostic MarkSet scores with record ID as the tie breaker. The upstream policy may withhold an actionable score. By default C is unknown. With `--policy-recommendations`, C includes the saved Investigate and Review routes, including Unsupported routed to Review. LowPriority is an assessed pass under that predicate. These routes depend on the parent scorer. Allocating a work set creates an analyst selection and preserves the original policy allocation and labels. The demo supplies synthetic, scoped C decisions. Neither input invokes a GenAI model.

The native importer admits reports up to 16 MiB and 25,000 distinct originals, subject to the shared memory budget. A policy report supplies stable original IDs and evidence revisions but lacks occurrence multiplicity, timestamps and host locators. Its terminal counts therefore cover originals. Its JSON occurrence counts are `null`. Native receipts bind the retained evidence, producer, filters, branch operations, active counts and ordered result digest. Reopening reads the same retained input. Live-source verification is available through the parent's `triage --replay`.

## Shared cases

| Case | Action and expected outcome | Acceptance witness |
|---|---|---|
| A/B reduction | From 12 originals and 20 occurrences, top-eight A and glob B yield six originals and 11 occurrences in A − B. The parent denominator remains 12. | [Independent walkthrough expectations](../../.kiro/specs/triager/prototype/walkthrough.json) and [native PTY driver](../../tools/triager-e2e/run.py) |
| Search to Venn | Search host, raw terms and score from the root. Three originals represent four matching occurrences. Select A − B, inspect R03’s two matching occurrences and reopen the saved result with identical membership. | [Native search walkthrough](../../tools/triager-e2e/run.py) |
| Scoped evidence | Inside the six, C has three recommended, two assessed passes and one unknown. R03 has two occurrences at edge-01 and three in its parent. Returning to D and choosing unknown gives R08 with the abstention reason. An empty search under the 08:00 UTC bucket can be undone. | [Native PTY driver](../../tools/triager-e2e/run.py) and [flow implementation and tests](../../crates/l2d-triager/src/flow.rs) |
| Raw logs | Open the second original in A − B, read adjacent originals and return to the same row. Reopen the final set and browse its four originals in allocation order. Verify long records can be read beyond the evidence excerpt, and terminal controls are displayed safely. | [Native PTY driver](../../tools/triager-e2e/run.py) and [raw browser tests](../../crates/l2d-triager/src/ui/raw.rs) |
| Candidate and final handoff | Reopen the candidate in a fresh process with all seats available. Allocate 2 + 1 + 1 once. A second reader recovers R03 and R04 in general, R08 in the unassessed reserve and R07 in audit. Together they account for eight occurrences. | [Walkthrough allocation oracle](../../.kiro/specs/triager/prototype/walkthrough.json) and [PTY replay](../../tools/triager-e2e/run.py) |
| Parent producer to terminal | Fit the fixture scorer, run `triage --out` and open its three originals via `tui --json`. Compare IDs and digest against the producer, retain missing occurrence coverage, and compare opt-in C to the original policy routes. | [Parent CLI integration test](../../crates/l2d-cli/tests/mark.rs), `analyst_workflow_pins_rules_score_history_and_replays_policy` |
| Retained policy replay | Open a real report in a PTY, choose the policy region, save, and reopen from a fresh process. C and ordered membership stay bound to that report. Altering its bytes must refuse reopen. | [Report PTY driver](../../tools/triager-e2e/report.py) |
| Terminal behavior | Resize to 80×24 and expand again, change the pane dimensions and verify that the saved investigation is byte-identical. Leaving the workbench restores the parent terminal's input mode. | [PTY driver](../../tools/triager-e2e/run.py), [input-mode probe](../../tools/triager-e2e/probe.py) and [Rust UI tests](../../crates/l2d-triager/src/ui.rs) |

For a changed score, route or rendering rule, run the relevant parent and terminal case together. If results differ, identify the first divergent step and its source, denominator or producer. A deliberate contract change updates both specs and the independent expectation before accepting new results. A renderer fix must continue to satisfy the existing expectation.

## Replay

From the repository root with the project's configured Cargo target, run the following. The Python packages belong to the test environment. The shipped Rust binary runs independently.

```sh
cargo test --locked -p l2d-triager
cargo test --locked -p l2d-cli --test mark analyst_workflow_pins_rules_score_history_and_replays_policy
node .kiro/specs/triager/prototype/workflow.js
python -m pip install -r tools/triager-e2e/requirements.txt
python tools/triager-e2e/run.py --binary <built-l2d> --evidence <project-test-output>
python tools/triager-e2e/report.py --binary <built-l2d> --report <saved-policy-report> --evidence <project-test-output>
```

Each PTY run creates its own evidence directory. The synthetic run saves candidate and final receipts, terminal views and the ordered result. The report run saves its digest and counts. The 18 November CSIC development report contains 96 originals. Selecting the saved policy routes gives 35 in C and 61 assessed passes. The original 17 unsupported cases are included in the review route. [Development cohort and upstream policy evidence](../2026-11-18-triage-workflow-comparison.md)

On the Windows reference host on 18 November, the native ConPTY walkthrough passed the 80×24 resize, exclusive Save As, repeat-allocation refusal, fresh-process reopen and restoration of the caller's mixed console settings. The second PTY case reproduced the ordered membership digest of all 35 recommended originals directly from the 96 parent decisions. It refused a changed report. The run directories under `D:\BuildArtifacts\log2decision\triager-e2e` are `native-pty-twv80o4i` and `policy-pty-i90g7t2p`.

The raw-browser runs in `native-pty-sq1ym9uk` and `policy-pty-edbibj5s` also verified the return to the selected result and evidence detail, final allocation order, and displayed text from the saved policy report. The synthetic run includes a terminal capture in `raw.txt`.

After the Splunk search language (task 3.2b) and its review fixes, both drivers passed again on 18 November with the installed binary SHA256 `E72B2310DAA43A248B072F1F2B1373E98178793E0BA77A49D8F5F220C8175D85`. The synthetic run `native-pty-fin2p7vi` gave a candidate of 6 originals and 11 occurrences, coverage 3/2/1, search then A − B of 1 original and 2 occurrences, and final R03, R04, R08, R07. The policy run `policy-pty-_np94b92` gave 96 originals, C 35 in and 61 out, 16 originals for `c=recommended | search score>=0.5`, refusal of the changed report, and a reopen counted peak of 49,395,456 bytes.

To replay the synthetic walkthrough over SSH, use a Linux analysis host with the matching binary, a known host key, batch login and an existing private test directory. The driver creates a fresh run directory there and copies the evidence back locally. It checks input-mode restoration at both ends.

```sh
python tools/triager-e2e/run.py --ssh-host <ssh-alias> --remote-binary /path/to/l2d --remote-dir /path/to/test-output --evidence <local-test-output>
```

## Remaining design gates

| Owner | Next end-to-end case | Evidence required to close it |
|---|---|---|
| Triager 1.1–1.3 and 3.1 | Open a multi-million-occurrence archive over real MCP stdio, derive the same A/B regions, drill into a repeated original and resume after spill. | Exact distinct and occurrence accounting, source identity, memory and disk costs, cold/warm times, and zero archive or inference repeats on prepared views. [Performance gate](../../.kiro/specs/triager/performance.md) |
| Triager 2.1–2.2 | Start from a broad search, select a mined template or vector neighbourhood and collect recommendations on that branch. Reopen the candidate and final receipt from a different MCP/TUI client. | Explainable pattern exclusions, vector revision, explicit C coverage including abstentions and unissued records, identical parent counts and ordered IDs. |
| Triager 3.2 | Replay the native walkthrough over an actual SSH connection, resize and cancel, then reopen an MCP-produced receipt. | Terminal restoration, stable counts/IDs, layout and redraw measurements, and unchanged producer counters. Local PTY evidence covers the renderer and native receipts. |
| Parent 2.22 | Freeze the rule and review budget, run future or held-out groups and obtain independent relevance outcomes. | Compare missed relevant cases and recovery at equal review capacity, including reserved and audit slots, against predeclared targets. [Parent qualification evidence](../2026-11-18-triage-workflow-comparison.md) |
