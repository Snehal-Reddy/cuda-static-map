use anyhow::{Context, Result};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn run(release: bool, arch: String, package: Option<String>, extra_args: Vec<String>) -> Result<()> {
    let workspace_root = env::current_dir()?;
    let codegen_backend = find_codegen_backend(&workspace_root)?;
    let codegen_backend_dir = codegen_backend.parent().unwrap();

    // Construct RUSTFLAGS
    // Note: We use CARGO_ENCODED_RUSTFLAGS with \x1f separator to avoid escaping issues
    let rustflags = vec![
        format!("-Zcodegen-backend={}", codegen_backend.display()),
        "-Zcrate-attr=feature(register_tool)".to_string(),
        "-Zcrate-attr=register_tool(nvvm_internal)".to_string(),
        "-Zcrate-attr=no_std".to_string(),
        "-Zsaturating_float_casts=false".to_string(),
        "-Cembed-bitcode=no".to_string(),
        "-Cdebuginfo=0".to_string(),
        "-Coverflow-checks=off".to_string(),
        "-Copt-level=3".to_string(),
        "-Cpanic=abort".to_string(),
        "-Cno-redzone=yes".to_string(),
        format!("-Cllvm-args=-arch={} --override-libm", arch),
        format!("-Ctarget-feature=+{}", arch),
    ];
    
    let encoded_rustflags = rustflags.join("\x1f");

    // LD_LIBRARY_PATH
    let old_ld_path = env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let new_ld_path = if old_ld_path.is_empty() {
        codegen_backend_dir.display().to_string()
    } else {
        format!("{}:{}", codegen_backend_dir.display(), old_ld_path)
    };

    // Target dir
    let target_dir = workspace_root.join("target/cuda-builder-direct");

    // CUDA_ARCH
    // extract "compute_75" -> "75"
    let cuda_arch_ver = arch.strip_prefix("compute_").unwrap_or("75");

    // Default package to "cuda-static-map-kernels" if not specified
    let package = package.or_else(|| Some("cuda-static-map-kernels".to_string()));

    let mut cmd = Command::new("cargo");
    cmd.arg("build");
    cmd.arg("--target=nvptx64-nvidia-cuda");
    cmd.arg("--target-dir").arg(&target_dir);
    cmd.arg("-Zbuild-std=core,alloc");
    cmd.arg("-Zbuild-std-features=panic_immediate_abort");

    if release {
        cmd.arg("--release");
    }

    if let Some(pkg) = package {
        cmd.arg("-p").arg(pkg);
    }

    cmd.args(extra_args);

    cmd.env("CARGO_ENCODED_RUSTFLAGS", encoded_rustflags);
    cmd.env("LD_LIBRARY_PATH", new_ld_path);
    cmd.env("CUDA_ARCH", cuda_arch_ver);

    // Print what we are doing
    println!("Building PTX for arch: {}", arch);
    println!("Using codegen backend: {}", codegen_backend.display());

    let status = cmd.status().context("Failed to execute cargo build-ptx")?;

    if !status.success() {
        anyhow::bail!("PTX build failed");
    }

    Ok(())
}

fn find_codegen_backend(workspace_root: &Path) -> Result<PathBuf> {
    let search_paths = [
        "target/debug/deps/librustc_codegen_nvvm.so",
        "target/release/deps/librustc_codegen_nvvm.so",
        "target/cuda-builder-codegen/debug/deps/librustc_codegen_nvvm.so",
        "target/cuda-builder-codegen/release/deps/librustc_codegen_nvvm.so",
        "rust-cuda/target/debug/deps/librustc_codegen_nvvm.so",
        "rust-cuda/target/release/deps/librustc_codegen_nvvm.so",
    ];

    for path_str in &search_paths {
        let path = workspace_root.join(path_str);
        if path.exists() {
            return Ok(path);
        }
    }

    println!("Building rustc_codegen_nvvm...");
    // We try to build it using the package from the current workspace or sub-workspace
    // Since rust-cuda is in the tree, we might need to point to it or just rely on cargo finding it if it's in a workspace.
    // If rustc_codegen_nvvm is part of rust-cuda which is not in the root workspace, we might need to go into rust-cuda dir or use -p if it's available.
    // Given the shell script does: (cd "$workspace_root" && cargo build -p rustc_codegen_nvvm ...)
    // It assumes rustc_codegen_nvvm is available from root.
    
    let status = Command::new("cargo")
        .args(&["build", "-p", "rustc_codegen_nvvm"])
        .current_dir(workspace_root)
        .status()?;

    if !status.success() {
        anyhow::bail!("Failed to build rustc_codegen_nvvm");
    }

    // Try again
    for path_str in &search_paths {
        let path = workspace_root.join(path_str);
        if path.exists() {
            return Ok(path);
        }
    }
    
    // As a fallback, try to find it via `find` equivalent if necessary, but shell script didn't.
    // However, cargo build -p rustc_codegen_nvvm might put it in target/debug/deps of the root workspace.
    
    anyhow::bail!("Could not find librustc_codegen_nvvm.so after building")
}

