//! GPU-side kernels and device code
//! This crate is compiled for nvptx64-nvidia-cuda target

use cuda_std::prelude::*;

pub mod pair;
pub mod static_map_ref;

pub use pair::Pair;
pub use static_map_ref::StaticMapRef;

