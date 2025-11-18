//! Host-side StaticMap wrapper
//! 
//! This is a thin wrapper around the shared StaticMap implementation
//! that adds host-specific functionality like kernel launching and
//! device memory management.
//!
//! The core StaticMap type is defined in the kernels crate and is
//! shared between host and device code.

// Re-export the shared StaticMap type from kernels
pub use cuda_static_map_kernels::StaticMap;

// The StaticMap implementation in kernels/src/static_map.rs already has
// host-specific methods (marked with #[cfg(not(target_arch = "nvptx64"))]),
// so this module can add additional host-only functionality if needed.

// For now, we just re-export everything. In the future, we might add:
// - Additional host-side convenience methods
// - Integration with cust/cuda_std for kernel launching
// - Device memory management helpers
