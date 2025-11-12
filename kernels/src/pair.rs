//! Shared pair implementation for both CPU and GPU
//!
//! This code compiles for both host (CPU) and device (GPU) targets.
//! When compiled for GPU (nvptx64-nvidia-cuda), it generates PTX code.
//! When compiled for CPU, it's used as a regular Rust library.

use cust_core::DeviceCopy;

/// Pair type for both device and host code
#[repr(C)]
#[derive(Clone, Copy, Debug, DeviceCopy)]
pub struct Pair<First, Second> {
    pub first: First,
    pub second: Second,
}

impl<First, Second> Pair<First, Second> {
    /// Create a new pair
    ///
    /// This method works on both CPU and GPU.
    pub fn new(first: First, second: Second) -> Self {
        Self { first, second }
    }
}

// GPU-specific implementations (only compiled when targeting nvptx64)
#[cfg(target_arch = "nvptx64")]
impl<First, Second> Pair<First, Second> {
    // Add GPU-specific methods here if needed in the future
}

// CPU-specific implementations (only compiled when NOT targeting nvptx64)
#[cfg(not(target_arch = "nvptx64"))]
impl<First, Second> Pair<First, Second> {
    // Add CPU-specific methods here if needed in the future
}
