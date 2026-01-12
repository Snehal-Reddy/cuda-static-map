use core::marker::PhantomData;
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
    type HashType: HashOutput;
    /// Hash a key to a `u32/u64` value.
    /// 
    /// This method must be callable from both host and device code.
    fn hash(&self, key: &Key) -> Self::HashType;
}

/// Identity hash function that returns the key as-is (with appropriate type conversion).
/// 
/// This is a perfect hash function when `hash_table_capacity >= |input set|`.
/// 
/// # Notes
/// - Identity hash is only intended to be used perfectly
/// - Perfect hashes are deterministic, and thus do not need seeds
/// - Returns `u32` cast to `u64` for keys ≤ 4 bytes, `u64` for keys > 4 bytes
/// 
/// # Type Parameters
/// * `Key` - The type of the values to hash. Must be convertible to `u32` or `u64`.
/// 
/// # Safety
/// The key type must be safely convertible to the result type. For keys ≤ 4 bytes,
/// the key must be convertible to `u32`. For keys > 4 bytes, the key must be
/// convertible to `u64`.
#[repr(C)]
#[derive(Debug)]
pub struct IdentityHash<Key> {
    _phantom: PhantomData<Key>,
}

// Manual Clone and Copy implementations - IdentityHash is a ZST, so it's Clone/Copy regardless of Key
impl<Key> Clone for IdentityHash<Key> {
    fn clone(&self) -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<Key> Copy for IdentityHash<Key> {}

impl<Key> IdentityHash<Key> {
    /// Create a new identity hash function.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

// Implementation for keys ≤ 4 bytes (u8, u16, u32, i8, i16, i32)
macro_rules! impl_identity_hash_small {
    ($($t:ty),*) => {
        $(
            impl Hash<$t> for IdentityHash<$t> {
                type HashType = u32;
                fn hash(&self, key: &$t) -> Self::HashType {
                    // Cast to u32 first, then to u64 (matches C++ behavior)
                    *key as Self::HashType
                }
            }
        )*
    };
}

impl_identity_hash_small!(u8, u16, u32, i8, i16, i32);

// Implementation for keys > 4 bytes (u64, i64, usize, isize)
macro_rules! impl_identity_hash_large {
    ($($t:ty),*) => {
        $(
            impl Hash<$t> for IdentityHash<$t> {
                type HashType = u64;
                fn hash(&self, key: &$t) -> Self::HashType {
                    // Direct cast to u64 (matches C++ behavior)
                    *key as Self::HashType
                }
            }
        )*
    };
}

impl_identity_hash_large!(u64, i64);

// For usize/isize, we need to handle platform-dependent sizes
// On 64-bit platforms, they're 8 bytes; on 32-bit platforms, they're 4 bytes
#[cfg(target_pointer_width = "64")]
impl_identity_hash_large!(usize, isize);

#[cfg(target_pointer_width = "32")]
impl_identity_hash_small!(usize, isize);

// Ensure IdentityHash is device-compatible
unsafe impl<Key> DeviceCopy for IdentityHash<Key> {}
