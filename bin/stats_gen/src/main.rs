use clap::{Parser, ValueEnum};
use log::info;
use std::{error::Error, path::PathBuf};

use crate::config::load_config;

mod config;
mod fraction;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Input config file path.
    #[arg(short, long)]
    pub config: PathBuf,

    /// Output directory path.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Number of times to repeat each decomposition operation.
    #[arg(short, long, default_value_t = 1)]
    pub repetitions: u32,

    /// Output filter.
    #[arg(short, long, value_enum, default_value_t = OutputFilter::All)]
    pub filter: OutputFilter,
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum OutputFilter {
    #[default]
    All,
    Clamped,
    Exact,
}

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();

    let args = Args::parse();
    info!("Config path: {:?}", args.config);
    info!("Output path: {:?}", args.output);
    info!("Repetitions: {:?}", args.repetitions);

    let config = load_config(&args.config)?;
    let stats = config.create_stats(&args.output, args.repetitions)?;
    stats.write_stats(&args.output.join("stats.txt"), args.filter)?;

    Ok(())
}
