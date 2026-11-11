//! Reads one input in batches and prints the batch plan, the totals and the time.
//!
//! The measuring tool behind the figures in `PARSE_EXPANSION` and the task notes: run
//! it under a process monitor to see the peak working set for a batch size.
//!
//! ```text
//! cargo run --release -p l2d-core --example read_batched -- csic data/csic-2010/csic_database.csv
//! cargo run --release -p l2d-core --example read_batched -- csic <path> --batch-rows 1024
//! cargo run --release -p l2d-core --example read_batched -- lines <path> --memory-limit 512MB
//! ```
//!
//! The input is data only; nothing in it is executed.

use std::path::Path;
use std::process::ExitCode;

use l2d_core::memory::parse_memory_size;
use l2d_core::{
    Csic2010Adapter, InputAdapter, L2dError, LabelledAdapter, LabelledFormat, LinesAdapter,
    MemoryBudget, MemorySettings,
};

fn run(args: &[String]) -> Result<(), L2dError> {
    let invalid = |detail: &str| L2dError::InvalidInput {
        detail: format!(
            "{detail}; usage: read_batched <csic|lines|csv|jsonl> <path> [--batch-rows N] \
             [--memory-limit SIZE]"
        ),
    };
    let [kind, path, options @ ..] = args else {
        return Err(invalid("adapter and path are required"));
    };
    let mut settings = MemorySettings::default();
    for pair in options.chunks(2) {
        match pair {
            [name, value] if name == "--batch-rows" => {
                let rows = value
                    .parse()
                    .map_err(|_| invalid("unreadable --batch-rows"))?;
                settings.max_batch_rows = Some(rows);
            }
            [name, value] if name == "--memory-limit" => {
                settings.memory_limit_bytes = Some(parse_memory_size(value)?);
            }
            _ => return Err(invalid("unknown option")),
        }
    }
    let adapter: Box<dyn InputAdapter> = match kind.as_str() {
        "csic" => Box::new(Csic2010Adapter::new()),
        "lines" => Box::new(LinesAdapter),
        "csv" => Box::new(LabelledAdapter::new(LabelledFormat::Csv)),
        "jsonl" => Box::new(LabelledAdapter::new(LabelledFormat::Jsonl)),
        _ => return Err(invalid("unknown adapter")),
    };

    let mut budget = MemoryBudget::observe(&settings)?;
    let started = std::time::Instant::now();
    let (logs, summary, plan) = adapter.read_with(Path::new(path), &mut budget)?;
    let elapsed = started.elapsed();
    println!("plan: {plan}");
    println!(
        "read: {} records, {} distinct texts, {} skipped, {} label conflicts, sha256 {}, in {:.0?}",
        summary.records,
        logs.len(),
        summary.skipped.len(),
        summary.label_conflicts,
        summary.content_sha256,
        elapsed
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(error.exit_code())
        }
    }
}
