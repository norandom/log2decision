//! The `l2d` command-line binary: argument parsing, dispatch, exit code.

mod cmd;
#[cfg(feature = "heap-report")]
mod heap;
mod output;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::error::ErrorKind;
use clap::{Parser, Subcommand};
use l2d_core::{L2dError, MemorySettings};

/// Local log triage: read logs, score them, and rank what to review first.
#[derive(Debug, Parser)]
#[command(name = "l2d", version, propagate_version = true)]
struct Cli {
    /// Print the result as JSON on standard output; diagnostics stay on standard error
    #[arg(long, global = true)]
    json: bool,

    /// Largest number of rows read in one batch [default: 65536, less when memory is short]
    #[arg(
        long,
        global = true,
        value_name = "N",
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    batch_rows: Option<u64>,

    /// Memory the tool may use, in bytes or as 512MB, 2GB (multiples of 1,024)
    /// [default: half of the free memory]
    #[arg(long, global = true, value_name = "SIZE", value_parser = memory_size)]
    memory_limit: Option<u64>,

    /// Share of the free memory used as the memory limit when --memory-limit is not
    /// given: above 0, below 1 [default: 0.5]
    #[arg(long, global = true, value_name = "SHARE", value_parser = share)]
    memory_share: Option<f32>,

    /// Share of the free memory kept as the reserve: batches are halved while less than
    /// this is free; above 0, below 1 [default: 0.25]
    #[arg(long, global = true, value_name = "SHARE", value_parser = share)]
    reserve_share: Option<f32>,

    /// The project directory: models are read and written below its models/, datasets
    /// are looked for below its data/
    #[arg(long, global = true, value_name = "DIR", default_value = ".")]
    project: PathBuf,

    #[command(subcommand)]
    command: Command,
}

/// Reads the value of `--memory-share` and `--reserve-share`: a number above 0 and
/// below 1 (owner decision of 2026-11-12, task 1.11).
fn share(text: &str) -> Result<f32, String> {
    match text.trim().parse::<f32>() {
        Ok(share) if share > 0.0 && share < 1.0 => Ok(share),
        _ => Err("a share is a number above 0 and below 1, for example 0.5".into()),
    }
}

/// Reads the value of `--memory-limit`.
fn memory_size(text: &str) -> Result<u64, String> {
    l2d_core::memory::parse_memory_size(text).map_err(|error| match error {
        L2dError::InvalidInput { detail } => detail,
        other => other.to_string(),
    })
}

impl Cli {
    /// The analyst's batch size, memory limit and memory shares (requirement 12.2);
    /// what is not given is left to the tool.
    fn memory_settings(&self) -> MemorySettings {
        let defaults = MemorySettings::default();
        MemorySettings {
            memory_limit_bytes: self.memory_limit,
            max_batch_rows: self.batch_rows,
            free_share: self.memory_share.unwrap_or(defaults.free_share),
            reserve_share: self.reserve_share.unwrap_or(defaults.reserve_share),
        }
    }

    /// What every command gets beside its own options.
    fn context(&self) -> cmd::common::Context {
        cmd::common::Context {
            json: self.json,
            project: self.project.clone(),
            memory: self.memory_settings(),
            memory_share_given: self.memory_share.is_some(),
            reserve_share_given: self.reserve_share.is_some(),
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Read and summarize an input
    Ingest(cmd::ingest::IngestArgs),
    /// Fit or apply the baseline scorer
    #[command(subcommand)]
    Baseline(cmd::baseline::BaselineCommand),
    /// Train an encoder, encode logs, show an encoder identity
    #[command(subcommand)]
    Encoder(cmd::encoder::EncoderCommand),
    /// Fill and query a store
    #[command(subcommand)]
    Store(cmd::store::StoreCommand),
    /// Rank logs for review, score down the rest, list what cannot be decided
    Triage(cmd::triage::TriageArgs),
    /// Calibrate a fitted scorer on held-out labelled logs
    Calibrate(cmd::calibrate::CalibrateArgs),
    /// Compare two groups of logs for a population shift
    Shift,
    /// Measure a vector set
    Assess,
    /// Write an evaluation report
    Eval,
    /// Check the runtime files, the NPU and its driver
    #[command(subcommand)]
    Runtime(cmd::runtime::RuntimeCommand),
}

/// Runs one operation. `context` carries what the analyst set for the project, the
/// output form and the memory; every operation that loads a dataset observes its memory
/// budget from it before it reads anything.
fn dispatch(command: &Command, context: &cmd::common::Context) -> Result<(), L2dError> {
    match command {
        Command::Ingest(args) => cmd::ingest::run(context, args),
        Command::Baseline(sub) => cmd::baseline::run(context, sub),
        Command::Encoder(sub) => cmd::encoder::run(sub),
        Command::Store(sub) => cmd::store::run(sub),
        Command::Triage(args) => cmd::triage::run(context, args),
        Command::Calibrate(args) => cmd::calibrate::run(context, args),
        Command::Shift => cmd::shift::run(),
        Command::Assess => cmd::assess::run(),
        Command::Eval => cmd::eval::run(),
        Command::Runtime(sub) => cmd::runtime::run(sub),
    }
}

/// Turns a command-line usage error into the analyst-error envelope (exit code 2).
fn usage_error(error: &clap::Error) -> L2dError {
    let rendered = error.render().to_string();
    let first_line = rendered.lines().next().unwrap_or_default();
    let detail = first_line.strip_prefix("error: ").unwrap_or(first_line);
    L2dError::InvalidInput {
        detail: format!("{detail} (see `l2d --help`)"),
    }
}

/// Whether `--json` was typed, for failures that happen before the arguments are parsed.
fn json_requested(args: &[String]) -> bool {
    args.iter()
        .skip(1)
        .take_while(|arg| *arg != "--")
        .any(|arg| arg == "--json")
}

fn fail(error: &L2dError, json: bool) -> ExitCode {
    output::report_error(error, json);
    ExitCode::from(error.exit_code())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let cli = match Cli::try_parse_from(&args) {
        Ok(cli) => cli,
        Err(error) => match error.kind() {
            // Help and version are answers, not failures; clap prints them and exits
            // (0 when asked for, 2 with the usage on standard error when nothing was typed).
            ErrorKind::DisplayHelp
            | ErrorKind::DisplayVersion
            | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand => error.exit(),
            _ => return fail(&usage_error(&error), json_requested(&args)),
        },
    };
    let code = match dispatch(&cli.command, &cli.context()) {
        Ok(()) => ExitCode::from(l2d_core::EXIT_SUCCESS),
        Err(error) => fail(&error, cli.json),
    };
    #[cfg(feature = "heap-report")]
    heap::report();
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(args: &[&str]) -> MemorySettings {
        Cli::try_parse_from(args)
            .unwrap_or_else(|error| panic!("{args:?}: {error}"))
            .memory_settings()
    }

    #[test]
    fn without_the_options_the_memory_settings_are_the_defaults() {
        assert_eq!(settings(&["l2d", "ingest"]), MemorySettings::default());
    }

    #[test]
    fn batch_rows_and_memory_limit_become_memory_settings() {
        assert_eq!(
            settings(&["l2d", "--batch-rows", "10000", "ingest"]),
            MemorySettings {
                max_batch_rows: Some(10_000),
                ..MemorySettings::default()
            }
        );
        assert_eq!(
            settings(&["l2d", "ingest", "--memory-limit", "2GB"]),
            MemorySettings {
                memory_limit_bytes: Some(2 * 1024 * 1024 * 1024),
                ..MemorySettings::default()
            }
        );
        assert_eq!(
            settings(&[
                "l2d",
                "store",
                "--memory-limit=512MB",
                "search",
                "--batch-rows=1024"
            ]),
            MemorySettings {
                memory_limit_bytes: Some(512 * 1024 * 1024),
                max_batch_rows: Some(1_024),
                free_share: 0.5,
                reserve_share: 0.25,
            }
        );
        assert_eq!(
            settings(&["l2d", "--memory-limit", "123456789", "eval"]).memory_limit_bytes,
            Some(123_456_789)
        );
    }

    #[test]
    fn zero_and_nonsense_are_usage_errors() {
        for args in [
            &["l2d", "--memory-limit", "0", "ingest"][..],
            &["l2d", "--memory-limit", "lots", "ingest"][..],
            &["l2d", "--batch-rows", "0", "ingest"][..],
            &["l2d", "--batch-rows", "2GB", "ingest"][..],
        ] {
            let error = Cli::try_parse_from(args).expect_err("must be refused");
            assert_eq!(error.kind(), ErrorKind::ValueValidation, "{args:?}");
            assert_eq!(usage_error(&error).exit_code(), 2);
        }
    }

    #[test]
    fn the_memory_shares_become_memory_settings() {
        let cli = Cli::try_parse_from([
            "l2d",
            "--memory-share",
            "0.3",
            "triage",
            "--reserve-share=0.1",
        ])
        .unwrap();
        assert_eq!(
            cli.memory_settings(),
            MemorySettings {
                free_share: 0.3,
                reserve_share: 0.1,
                ..MemorySettings::default()
            }
        );
        let context = cli.context();
        assert!(context.memory_share_given && context.reserve_share_given);
        let context = Cli::try_parse_from(["l2d", "ingest"]).unwrap().context();
        assert!(!context.memory_share_given && !context.reserve_share_given);
        assert_eq!(context.project, PathBuf::from("."));
    }

    #[test]
    fn a_share_lies_strictly_between_zero_and_one() {
        for text in ["0.5", "0.000001", "0.999", " 0.25 "] {
            assert!(share(text).is_ok(), "{text}");
        }
        for text in [
            "0", "1", "1.0", "-0.5", "2", "NaN", "inf", "-inf", "half", "", "50%",
        ] {
            assert!(share(text).is_err(), "{text}");
            for option in ["--memory-share", "--reserve-share"] {
                let error = Cli::try_parse_from(["l2d", option, text, "ingest"])
                    .expect_err("must be refused");
                assert_eq!(usage_error(&error).exit_code(), 2, "{option} {text}");
            }
        }
    }

    #[test]
    fn a_share_together_with_a_limit_is_refused_before_anything_is_read() {
        let context = Cli::try_parse_from([
            "l2d",
            "--memory-share",
            "0.3",
            "--memory-limit",
            "1GB",
            "ingest",
        ])
        .unwrap()
        .context();
        let error = context.budget().unwrap_err();
        assert_eq!(error.exit_code(), 2);
        assert!(error.to_string().contains("--memory-share"), "{error}");
    }
}
