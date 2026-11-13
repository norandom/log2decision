//! `l2d ingest`: read and summarize an input (requirements 1.1 to 1.4, 1.6).
//!
//! Nothing is written: the command shows what a later command would read.

use std::io::{self, Write};

use clap::Args;
use l2d_core::L2dError;

use super::common::{
    Context, DEFAULT_TEXT_LIMIT, InputArgs, InputReport, MemoryReport, Operation, read_input,
};
use crate::output::{JsonOut, output_failed};

/// The options of `l2d ingest`.
#[derive(Debug, Clone, Args)]
pub struct IngestArgs {
    #[command(flatten)]
    pub input: InputArgs,

    /// Most skipped records listed in the text output [default: 20]; JSON lists all
    #[arg(long, value_name = "N")]
    pub limit: Option<u64>,
}

/// Runs `l2d ingest`.
pub fn run(context: &Context, args: &IngestArgs) -> Result<(), L2dError> {
    let paths = context.paths()?;
    let mut budget = context.budget()?;
    let input = read_input(&paths, &args.input, &mut budget)?;
    let report = InputReport::of(&input.logs, &input.summary, input.adapter, &budget)?;
    let memory = context.memory_report(
        &budget,
        vec![Operation::batches(
            "read",
            input.summary.records,
            input.plan,
        )],
    );

    let out = io::stdout().lock();
    let written = if context.json {
        json(out, &report, &memory)
    } else {
        text(
            out,
            &report,
            &memory,
            args.limit.unwrap_or(DEFAULT_TEXT_LIMIT),
        )
    };
    written.map_err(output_failed)
}

fn json(out: impl Write, input: &InputReport<'_>, memory: &MemoryReport) -> io::Result<()> {
    let mut json = JsonOut::begin(io::BufWriter::new(out))?;
    json.field("command", "ingest")?;
    json.field("input", input)?;
    json.field("memory", memory)?;
    json.end()
}

fn text(
    out: impl Write,
    input: &InputReport<'_>,
    memory: &MemoryReport,
    limit: u64,
) -> io::Result<()> {
    let mut out = io::BufWriter::new(out);
    input.write_text(&mut out, limit)?;
    memory.write_text(&mut out)?;
    out.flush()
}
