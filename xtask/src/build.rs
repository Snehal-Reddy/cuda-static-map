use anyhow::Result;
use std::process::Command;

pub fn run(release: bool, package: Option<String>, extra_args: Vec<String>) -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.arg("build");

    if release {
        cmd.arg("--release");
    }

    if let Some(pkg) = package {
        cmd.arg("-p").arg(pkg);
    }

    cmd.args(extra_args);

    let status = cmd.status()?;

    if !status.success() {
        anyhow::bail!("Build failed");
    }

    Ok(())
}
