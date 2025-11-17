//! CUDA Static Map - A GPU-accelerated hash map implementation

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

pub mod static_map;

// Re-export main types
// Pair is defined in cuda-static-map-kernels and works on both CPU and GPU
pub use cuda_static_map_kernels::Pair;
pub use static_map::StaticMap;
