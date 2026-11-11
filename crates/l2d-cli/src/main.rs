//! The `l2d` command-line binary: argument parsing, dispatch, exit code.

mod cmd;
mod output;

use std::process::ExitCode;

use clap::error::ErrorKind;
use clap::{Parser, Subcommand};
use l2d_core::L2dError;

/// Local log triage: read logs, score them, and rank what to review first.
#[derive(Debug, Parser)]
#[command(name = "l2d", version, propagate_version = true)]
struct Cli {
    /// Print the result as JSON on standard output; diagnostics stay on standard error
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Read and summarize an input
    Ingest,
    /// Fit or apply the baseline scorer
    #[command(subcommand)]
    Baseline(cmd::baseline::BaselineCommand),
    /// Train an encoder, encode logs, show an encoder identity
    #[command(subcommand)]
    Encoder(cmd::encoder::EncoderCommand),
    /// Fill and query a store
    #[command(subcommand)]
    Store(cmd::store::StoreCommand),
    /// Rank logs for review
    Triage,
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

fn dispatch(command: &Command) -> Result<(), L2dError> {
    match command {
        Command::Ingest => cmd::ingest::run(),
        Command::Baseline(sub) => cmd::baseline::run(sub),
        Command::Encoder(sub) => cmd::encoder::run(sub),
        Command::Store(sub) => cmd::store::run(sub),
        Command::Triage => cmd::triage::run(),
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
    match dispatch(&cli.command) {
        Ok(()) => ExitCode::from(l2d_core::EXIT_SUCCESS),
        Err(error) => fail(&error, cli.json),
    }
}
