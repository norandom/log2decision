//! Pinned two-graph NPU encoder integration.
use super::common::{self, Context, InputArgs};
use clap::{Args, Subcommand};
use l2d_core::{L2dError, LogRecord, LogsFrame, MemoryBudget, ProjectPaths};
use l2d_decide::outlier::{OutlierQuery, rank_outliers};
use l2d_encoder::hierarchy::{
    HierarchicalSession, MAX_BYTES, OriginalFlags, PROFILE, SESSION_BYTES, identity, verify_profile,
};
use l2d_store::{LogFlags, Store, VectorOrigin};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

#[derive(Debug, Clone, Args)]
pub struct ProfileArgs {
    /// Verified f0-s0-mask package directory
    #[arg(long)]
    pub profile_dir: Option<PathBuf>,
}
impl ProfileArgs {
    fn directory(&self, paths: &ProjectPaths) -> PathBuf {
        self.profile_dir
            .clone()
            .unwrap_or_else(|| paths.encoders_dir().join("f0-s0-mask"))
    }
    pub fn open(
        &self,
        paths: &ProjectPaths,
        budget: &mut MemoryBudget,
    ) -> Result<HierarchicalSession, L2dError> {
        let runtime =
            paths.runtime_files_dir(l2d_core::paths::runtime_dir_from_environment().as_deref());
        let cache = std::env::var_os("L2D_NPU_WORK_DIR")
            .map(PathBuf::from)
            .map(|d| d.join("cache"))
            .unwrap_or_else(|| paths.runtime_cache_dir());
        let session = HierarchicalSession::open(&self.directory(paths), &runtime, &cache, budget)?;
        budget.hold(budget.held_bytes().saturating_add(SESSION_BYTES));
        Ok(session)
    }
}
#[derive(Debug, Clone, Args)]
pub struct EncodeArgs {
    #[command(flatten)]
    pub input: InputArgs,
    #[command(flatten)]
    pub archive: common::ArchiveArgs,
    #[command(flatten)]
    pub profile: ProfileArgs,
}
#[derive(Debug, Clone, Subcommand)]
pub enum EncoderCommand {
    /// Train an encoder and write its package
    Train,
    /// Encode original records on the pinned NPU
    Encode(EncodeArgs),
    /// Verify the package and show its identity
    Info(ProfileArgs),
}
pub enum Operation<'a> {
    Encode,
    StoreAdd(&'a str),
    Outliers(&'a str, Option<usize>),
}
pub fn run(context: &Context, command: &EncoderCommand) -> Result<(), L2dError> {
    match command {
        EncoderCommand::Train => Err(super::not_implemented("encoder train")),
        EncoderCommand::Encode(args) => encoded(context, args, Operation::Encode),
        EncoderCommand::Info(args) => {
            let paths = context.paths()?;
            let budget = context.budget()?;
            let dir = args.directory(&paths);
            verify_profile(&dir, &budget)?;
            crate::output::report(
                context,
                &budget,
                &json!({
                    "encoder": identity(), "profile": PROFILE, "package": dir,
                    "input_bytes": MAX_BYTES, "dimension": 128, "input_mode": "original UTF-8, no cut"
                }),
            )
        }
    }
}
pub fn encoded(
    context: &Context,
    args: &EncodeArgs,
    operation: Operation<'_>,
) -> Result<(), L2dError> {
    let paths = context.paths()?;
    let mut budget = context.budget()?;
    let _guard = l2d_core::resource::ResourceGuard::enter(
        "CLI NPU encoding",
        l2d_core::resource::busy_choice(),
    )?;
    let input = super::analyst_source::query_input(
        context,
        &paths,
        &args.input,
        &args.archive,
        &mut budget,
    )?;
    let identity = identity();
    let origin = VectorOrigin::made_under(&identity, "1.9.0");
    let mut store = match operation {
        Operation::Outliers(name, _) => Some(Store::open(&paths, name)?),
        Operation::StoreAdd(name) if paths.store_path(name).exists() => {
            Some(Store::open(&paths, name)?)
        }
        _ => None,
    };
    if let Some(store) = &store {
        store.require_origin(origin)?;
    }
    let working = (input.logs.frame().estimated_size() as u64)
        .saturating_mul(12)
        .saturating_add((input.logs.len() as u64).saturating_mul(128 * 128 + 4096))
        .saturating_add(256 * 1024);
    budget.require_whole(
        working,
        "original records, NPU vectors, store frame and output",
    )?;
    budget.hold(budget.held_bytes().saturating_add(working));
    let run_start = Instant::now();
    let mut session = args.profile.open(&paths, &mut budget)?;
    let start = Instant::now();
    let mut rows = Vec::new();
    let mut records: Vec<LogRecord> = Vec::new();
    let mut stored_flags = Vec::new();
    let mut values = Vec::new();
    let mut encoded_count = 0;
    for row in input.logs.records() {
        let vector = session.encode(&row.text)?;
        let supported = vector.as_ref().is_some_and(|v| {
            v.len() == 128 && v.iter().all(|x| x.is_finite()) && v.iter().any(|x| *x != 0.)
        });
        let flags = OriginalFlags::of(&row.text);
        if matches!(operation, Operation::Encode) {
            rows.push(json!({"first_position": row.first_position, "occurrences": row.occurrences,
                "supported": supported, "flags": flags, "vector":vector.as_ref().filter(|_|supported)}));
        } else if !supported {
            return Err(L2dError::Refused {
                reason: format!(
                    "unsupported original at position {} under {PROFILE}; flags {}; store/ranking requires complete finite nonzero original vectors",
                    row.first_position,
                    json!(flags)
                ),
            });
        }
        if supported {
            encoded_count += 1;
            if !matches!(operation, Operation::Encode) {
                stored_flags.push(LogFlags {
                    truncated: flags.truncated,
                    text_longer_than_input: flags.text_longer_than_input,
                    empty: flags.empty,
                });
                values.extend(vector.unwrap());
                records.push(row);
            }
        }
    }
    let mut record = session.run_record(
        input.summary.clone(),
        encoded_count,
        start.elapsed(),
        run_start.elapsed(),
    );
    record.record_batch_plan(&input.plan)?;
    if input.adapter == "zip" {
        record.settings["archive_entries"] = json!(args.archive.entries);
        record.settings["max_record_bytes"] = json!(
            args.archive
                .settings(common::AtOnceDefault::One)
                .max_record_bytes
        );
    }
    let mut result = json!({"run_record":record,"encoded":encoded_count});
    drop(session);
    budget.hold(budget.held_bytes().saturating_sub(SESSION_BYTES));
    let source = input.summary.source_path.to_string_lossy();
    match operation {
        Operation::Encode => result["rows"] = rows.into(),
        Operation::StoreAdd(name) => {
            let records = LogsFrame::from_records(&records)?;
            let frame = l2d_store::rows_frame(
                &records,
                &source,
                &values,
                identity.dimension,
                &stored_flags,
            )?;
            let report = match store.take() {
                Some(mut store) => store.add_with(&frame, origin, &mut budget)?,
                None => Store::create_with_rows(&paths, name, origin, &frame, &mut budget)?,
            };
            result["store"] = json!({"name":name, "rows":report.rows, "added":report.added, "repeated":report.repeated,
                "truncated":0,"text_longer_than_input":0,"empty":0});
        }
        Operation::Outliers(_, k) => {
            let queries: Vec<_> = records
                .iter()
                .zip(values.as_chunks::<128>().0)
                .map(|(r, v)| OutlierQuery {
                    source: &source,
                    text: &r.text,
                    vector: v,
                })
                .collect();
            result["ranking"] = json!(rank_outliers(
                store.as_ref().unwrap(),
                origin,
                &queries,
                k,
                &mut budget
            )?);
            result["query_positions"] =
                json!(records.iter().map(|r| r.first_position).collect::<Vec<_>>());
        }
    }
    crate::output::report(context, &budget, &result)
}
