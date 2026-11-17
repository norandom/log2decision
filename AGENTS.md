# Build storage discipline

- Run commands from this repository or its task worktree, never a drive root.
- Keep disposable outputs under `D:\BuildArtifacts\log2decision`. Reuse one Cargo target across sequential tasks and reviewers; do not create a target per agent or task. The current reusable target is `D:\BuildArtifacts\log2decision\task-1.26-target` (the historical name does not imply task ownership).
- Set `CARGO_TARGET_DIR` explicitly to that target for this project's jobs. Preserve configured `CARGO_HOME`, dependency caches and TEMP/TMP. Do not change machine-wide settings or clean the shared `D:\BuildArtifacts\cargo-target` as routine project maintenance.
- For ordinary validation set `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, and `CARGO_INCREMENTAL=0`. Use at most four Cargo jobs and one heavy build/test job at a time. Reuse the same features/profile across checks where possible.
- Before each heavy job, measure target size and free space. Default admission limits: target below 30 GiB and at least 50 GiB free on D:. These are preflight limits, not a hard filesystem quota. If either fails, stop admitting builds and reclaim only this project's inactive generated artifacts before retrying. Do not silently allocate another cache.
- After each task, inspect cache growth. Prefer `cargo clean --profile dev --target-dir <verified-project-target>` when cleanup is needed and no project job uses it. Resolve and verify the absolute path first. Preserve release qualification binaries until their evidence is recorded and no pending review needs them.
- Never delete source, logs/evidence, session history, datasets, models or active build output as cache cleanup. Do not monitor or interfere with unrelated projects/processes.
- Run affected tests while iterating; run the required broader checks at the review gate. Reuse validated binaries for review/smoke when their provenance matches the current source. Do not rebuild just to repeat unchanged evidence.
