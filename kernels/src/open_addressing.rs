//! Open addressing primitives and abstractions.
//!
//! This module provides the `KeyEqual` trait and `EqualWrapper` for
//! handling key comparisons and sentinel values in open addressing schemes.

/// Enum of equality comparison results.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(i8)]
pub enum EqualResult {
    /// Keys are unequal.
    Unequal = 0,
    /// Keys are equal.
    Equal = 1,
    /// Slot is empty (used for find/contains to stop probing).
    Empty = 2,
    /// Slot is available (empty or erased, used for insert).
    Available = 3,
}

/// Trait for customizable key comparison.
///
/// User-provided equality predicates cannot be used to compare against sentinel values
/// directly. The `EqualWrapper` handles sentinel checking automatically.
///
/// # Order Sensitivity
///
/// The comparison is order-sensitive: `probe_key` (LHS) is the key being searched for,
/// and `slot_key` (RHS) is the key stored in the slot. Container slots are always on
/// the right-hand side.
pub trait KeyEqual<Key: ?Sized> {
    /// Compares a probe key with a slot key.
    ///
    /// # Arguments
    /// * `probe_key` - The key being searched for (LHS)
    /// * `slot_key` - The key stored in the slot (RHS)
    ///
    /// # Returns
    /// * `EqualResult::Equal` if the keys are equivalent
    /// * `EqualResult::Unequal` otherwise
    fn equal<ProbeKey: ?Sized>(&self, probe_key: &ProbeKey, slot_key: &Key) -> EqualResult
    where
        ProbeKey: PartialEq<Key>;
}

/// Default implementation of `KeyEqual` using standard `PartialEq`.
#[derive(Copy, Clone, Default)]
pub struct DefaultKeyEqual;

impl<Key: ?Sized> KeyEqual<Key> for DefaultKeyEqual {
    fn equal<ProbeKey: ?Sized>(&self, probe_key: &ProbeKey, slot_key: &Key) -> EqualResult
    where
        ProbeKey: PartialEq<Key>,
    {
        if probe_key == slot_key {
            EqualResult::Equal
        } else {
            EqualResult::Unequal
        }
    }
}

/// Performs a strict bitwise comparison of two values.
///
/// Optimizations are added for 4-byte and 8-byte types for better performance.
#[inline(always)]
fn bitwise_compare<T: Copy>(a: &T, b: &T) -> bool {
    // Safety: T is Copy, and we're reading from valid references.
    // We use read_unaligned to safely read bytes regardless of alignment.
    unsafe {
        let size = core::mem::size_of::<T>();
        
        // Handle zero-sized types
        if size == 0 {
            return true;
        }
        
        // Get byte pointers directly from references (no copying)
        let a_ptr = a as *const T as *const u8;
        let b_ptr = b as *const T as *const u8;
        
        // Optimize for common sizes (matches C++ specializations)
        // These reduce to single register operations on GPU
        match size {
            4 => {
                // For 4-byte types: single u32 comparison
                let a_val = core::ptr::read_unaligned(a_ptr as *const u32);
                let b_val = core::ptr::read_unaligned(b_ptr as *const u32);
                a_val == b_val
            }
            8 => {
                // For 8-byte types: single u64 comparison
                let a_val = core::ptr::read_unaligned(a_ptr as *const u64);
                let b_val = core::ptr::read_unaligned(b_ptr as *const u64);
                a_val == b_val
            }
            _ => {
                // Byte-by-byte comparison for other sizes
                // Compare raw bytes using a simple loop (similar to C++ cuda_memcmp)
                for i in 0..size {
                    let a_byte = *a_ptr.add(i);
                    let b_byte = *b_ptr.add(i);
                    if a_byte != b_byte {
                        return false;
                    }
                }
                true
            }
        }
    }
}

/// Wrapper around `KeyEqual` that handles sentinel values.

#[derive(Copy, Clone)]
pub struct EqualWrapper<Key, KeyEqual> {
    key_equal: KeyEqual,
    empty_sentinel: Key,
    erased_sentinel: Key,
}

impl<Key, KQ> EqualWrapper<Key, KQ>
where
    Key: PartialEq + Copy,
    KQ: KeyEqual<Key>,
{
    /// Creates a new `EqualWrapper`.
    ///
    /// # Arguments
    /// * `empty_sentinel` - The key value that represents an empty slot
    /// * `erased_sentinel` - The key value that represents an erased slot
    /// * `key_equal` - The user-provided key equality predicate
    pub const fn new(empty_sentinel: Key, erased_sentinel: Key, key_equal: KQ) -> Self {
        Self {
            key_equal,
            empty_sentinel,
            erased_sentinel,
        }
    }

    /// Equality check with the given equality predicate.
    ///
    /// This method directly calls the underlying `KeyEqual::equal` without
    /// checking for sentinels. Use `equal_for_insert` or `equal_for_find` for
    /// sentinel-aware comparisons.
    ///
    /// # Arguments
    /// * `probe_key` - The key being searched for (LHS)
    /// * `slot_key` - The key stored in the slot (RHS)
    ///
    /// # Returns
    /// * `EqualResult::Equal` if the keys are equivalent
    /// * `EqualResult::Unequal` otherwise
    #[inline]
    pub fn equal_to<ProbeKey: ?Sized>(
        &self,
        probe_key: &ProbeKey,
        slot_key: &Key,
    ) -> EqualResult
    where
        ProbeKey: PartialEq<Key>,
    {
        self.key_equal.equal(probe_key, slot_key)
    }

    /// Equality check for insertion.
    ///
    /// This function always compares the right-hand side element (slot key) against
    /// sentinel values first, then performs an equality check with the given equality
    /// predicate, i.e., `key_equal.equal(probe_key, slot_key)`.
    ///
    /// Insert probing stops when it finds an empty or erased slot (available for insertion).
    ///
    /// # Arguments
    /// * `probe_key` - The key to insert (LHS)
    /// * `slot_key` - The key currently in the slot (RHS)
    /// * `allows_duplicates` - Whether duplicate keys are allowed (skips expensive equality check)
    ///
    /// # Returns
    /// * `EqualResult::Available` if the slot is empty or erased (available for insertion)
    /// * `EqualResult::Equal` if the keys match (duplicate found, insertion may fail)
    /// * `EqualResult::Unequal` if the keys don't match (continue probing)
    ///
    /// # Note
    /// For containers that allow duplicates, the expensive key equality check is skipped
    /// during insertion since we always insert regardless of whether the key already exists.
    #[inline]
    pub fn equal_for_insert<ProbeKey: ?Sized>(
        &self,
        probe_key: &ProbeKey,
        slot_key: &Key,
        allows_duplicates: bool,
    ) -> EqualResult
    where
        ProbeKey: PartialEq<Key>,
    {
        // Check for sentinels (Available = Empty or Erased)
        // Use bitwise comparison to handle NaNs and ensure we only match
        // strict sentinel values (including padding).
        if bitwise_compare(slot_key, &self.empty_sentinel) 
            || bitwise_compare(slot_key, &self.erased_sentinel) 
        {
            return EqualResult::Available;
        }

        // Optimization: if duplicates allowed, we don't care about equality,
        // we only look for available slots. Existing keys are treated as "Unequal"
        // (i.e., continue probing).
        if allows_duplicates {
            return EqualResult::Unequal;
        }

        self.equal_to(probe_key, slot_key)
    }

    /// Equality check for find/contains.
    ///
    /// This function always compares the right-hand side element (slot key) against
    /// the empty sentinel first, then performs an equality check with the given equality
    /// predicate.
    ///
    /// Query probing stops only when it finds an empty slot. Erased slots are skipped
    /// (continue probing).
    ///
    /// # Arguments
    /// * `probe_key` - The key being searched for (LHS)
    /// * `slot_key` - The key currently in the slot (RHS)
    ///
    /// # Returns
    /// * `EqualResult::Empty` if the slot is empty (stop probing, key not found)
    /// * `EqualResult::Equal` if the keys match (key found)
    /// * `EqualResult::Unequal` if the keys don't match (continue probing)
    #[inline]
    pub fn equal_for_find<ProbeKey: ?Sized>(
        &self,
        probe_key: &ProbeKey,
        slot_key: &Key,
    ) -> EqualResult
    where
        ProbeKey: PartialEq<Key>,
    {
        // Check for empty sentinel (Empty = stop probing)
        // Use bitwise comparison to ensure we only stop on true empty slots.
        if bitwise_compare(slot_key, &self.empty_sentinel) {
            return EqualResult::Empty;
        }

        self.equal_to(probe_key, slot_key)
    }
}
