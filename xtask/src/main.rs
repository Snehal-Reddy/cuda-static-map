mod build;
mod build_ptx;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Build automation for cuda-static-map", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Non-device build (host)
    Build {
        #[arg(long)]
        release: bool,
        #[arg(short, long)]
        package: Option<String>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// PTX build (device)
    BuildPtx {
        #[arg(long)]
        release: bool,
        #[arg(long, default_value = "compute_75")]
        arch: String,
        #[arg(short, long)]
        package: Option<String>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Build both host and device code
    BuildAll {
        #[arg(long)]
        release: bool,
        #[arg(long, default_value = "compute_75")]
        arch: String,
        #[arg(short, long)]
        package: Option<String>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build {
            release,
            package,
            args,
        } => {
            build::run(release, package, args)?;
        }
        Commands::BuildPtx {
            release,
            arch,
            package,
            args,
        } => {
            build_ptx::run(release, arch, package, args)?;
        }
        Commands::BuildAll {
            release,
            arch,
            package,
            args,
        } => {
            // Build host
            println!("--- Building Host ---");
            build::run(release, package.clone(), args.clone())?;

            // Build device
            println!("\n--- Building PTX ---");
            build_ptx::run(release, arch, package, args)?;
        }
    }

    Ok(())
}
