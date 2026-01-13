use cust_core::DeviceCopy;

/// Marker trait for valid hash return types.
/// 
/// Only `u32` and `u64` implement this trait, ensuring type safety
/// for hash function return values.
pub trait HashOutput: Copy + DeviceCopy {
    /// Convert the hash value to `usize` for use in modulo operations.
    fn to_usize(self) -> usize;
}

impl HashOutput for u32 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl HashOutput for u64 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

/// Trait for hash functions that can hash keys on both host and device.
/// 
/// Hash functions must be `Copy` and device-compatible to work in CUDA kernels.
pub trait Hash<Key>: Copy + DeviceCopy {
    /// The hash return type. Must be either `u32` or `u64`.
    type HashType: HashOutput;
    
    /// Hash a key to a `u32` or `u64` value.
    /// 
    /// This method must be callable from both host and device code.
    fn hash(&self, key: &Key) -> Self::HashType;
}

// Submodules for hash function implementations
pub mod identity;
pub mod xxhash;

// Re-export hash functions
pub use identity::IdentityHash;
pub use xxhash::{XXHash32, XXHash64};
