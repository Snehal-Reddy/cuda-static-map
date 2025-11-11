//! CUDA Static Map - A GPU-accelerated hash map implementation


static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

pub mod pair;
pub mod static_map;

// Re-export main types
pub use pair::Pair;
pub use static_map::StaticMap;
