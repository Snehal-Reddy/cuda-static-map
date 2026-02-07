use core::marker::PhantomData;
use cust_core::DeviceCopy;

use crate::hash::Hash;

/// Identity hash function that returns the key as-is (with appropriate type conversion).
///
/// This is a perfect hash function when `hash_table_capacity >= |input set|`.
///
/// # Notes
/// - Identity hash is only intended to be used perfectly
/// - Perfect hashes are deterministic, and thus do not need seeds
/// - Returns `u32` for keys ≤ 4 bytes, `u64` for keys > 4 bytes
///
/// # Type Parameters
/// * `Key` - The type of the values to hash. Must be convertible to `u32` or `u64`.
///
/// The key type must be convertible to the result type. For keys ≤ 4 bytes,
/// the key must be convertible to `u32`. For keys > 4 bytes, the key must be
/// convertible to `u64`. This is enforced by the trait implementations.
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
                    // Cast to u32 (matches C++ uint32_t result_type for small keys)
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
                    // Direct cast to u64 (matches C++ uint64_t result_type for large keys)
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

// Safety: This is a zero-sized type containing only PhantomData<Key>.
unsafe impl<Key> DeviceCopy for IdentityHash<Key> {}
