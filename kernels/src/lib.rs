//! Shared kernels and device code
//! This crate can be compiled for both CPU and GPU targets
//! - For CPU: used as a regular Rust library dependency
//! - For GPU: compiled to PTX via cuda_builder with --target=nvptx64-nvidia-cuda

#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![allow(incomplete_features)]

#[cfg(target_arch = "nvptx64")]
use cuda_std;

pub mod hash;
pub mod open_addressing;
pub mod pair;
pub mod probing;
pub mod static_map;
pub mod static_map_ref;
pub mod storage;

pub use hash::{Hash, IdentityHash, XXHash32, XXHash64};
pub use pair::{IsTupleLike, Pair, PairLike};
pub use static_map::StaticMap;
pub use static_map_ref::StaticMapRef;
pub use storage::BucketStorageRef;
#[cfg(not(target_arch = "nvptx64"))]
pub use storage::{BucketStorage, Extent, make_valid_extent, make_valid_extent_for_scheme};

#[cfg(all(target_arch = "nvptx64", feature = "test-kernels"))]
pub mod test_kernels;
