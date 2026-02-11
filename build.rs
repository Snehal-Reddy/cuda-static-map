use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");

    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let mut builder =
        CudaBuilder::new(manifest_dir.join("kernels")).copy_to(out_path.join("kernels.ptx"));

    if env::var("CARGO_FEATURE_TEST_KERNELS").is_ok() {
        builder = builder.build_args(&["--features", "test-kernels"]);
    }

    builder.build().unwrap();
}
