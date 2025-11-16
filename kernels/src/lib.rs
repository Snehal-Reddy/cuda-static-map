//! Shared kernels and device code
//! This crate can be compiled for both CPU and GPU targets
//! - For CPU: used as a regular Rust library dependency
//! - For GPU: compiled to PTX via cuda_builder with --target=nvptx64-nvidia-cuda

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

#[cfg(target_arch = "nvptx64")]
use cuda_std;

pub mod pair;
pub mod static_map_ref;

pub use pair::Pair;
pub use static_map_ref::StaticMapRef;

