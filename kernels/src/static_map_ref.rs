//! Device-side reference type for static map.
//!
//! This  allow kernels to access the static map on the GPU. The ref holds the same
//! fields as `OpenAddressingRefImpl` and delegates find/insert/contains to it on device.

use crate::open_addressing::{EqualWrapper, ThreadScope};
use crate::pair::{AlignedTo, Pair, alignment};
use crate::storage::{BucketStorageRef, Extent};
use cust_core::DeviceCopy;

#[cfg(target_arch = "nvptx64")]
use crate::open_addressing::OpenAddressingRefImpl;
#[cfg(target_arch = "nvptx64")]
use crate::probing::ProbingScheme;

/// Non-owning device-side reference to a static map.
///
/// This type is trivially copyable and is intended to be passed by value to GPU kernels.
/// It holds the same components as `OpenAddressingRefImpl` (storage ref, sentinels,
/// predicate, probing scheme). On device, `find()`, `insert()`, and `contains()` delegate
/// to the ref implementation for bucket-based access and atomic operations.
///
/// # Type Parameters
/// * `Key` - Key type
/// * `Value` - Mapped (value) type
/// * `Scheme` - Probing scheme type
/// * `BUCKET_SIZE` - Slots per bucket
/// * `KeyEqual` - Key equality predicate type
/// * `SCOPE` - Thread scope for atomic operations
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StaticMapRef<
    Key,
    Value,
    Scheme,
    const BUCKET_SIZE: usize,
    KeyEqual,
    const SCOPE: ThreadScope,
> where
    Key: Copy,
    Value: Copy,
    Scheme: Copy,
    KeyEqual: Copy,
    (): AlignedTo<{ alignment::<Key, Value>() }>,
    [(); alignment::<Key, Value>()]:,
{
    storage_ref: BucketStorageRef<Pair<Key, Value>, BUCKET_SIZE>,
    empty_slot_sentinel: Pair<Key, Value>,
    erased_key_sentinel: Key,
    predicate: EqualWrapper<Key, KeyEqual>,
    probing_scheme: Scheme,
}

// Safety: StaticMapRef has the same layout as the ref impl components. All fields are
// Copy and represent device-safe values (BucketStorageRef, Pair, Key, EqualWrapper, Scheme).
// The pointer in BucketStorageRef is only dereferenced on device.
unsafe impl<Key, Value, Scheme, const BUCKET_SIZE: usize, KeyEqual, const SCOPE: ThreadScope>
    DeviceCopy for StaticMapRef<Key, Value, Scheme, BUCKET_SIZE, KeyEqual, SCOPE>
where
    Key: DeviceCopy + Copy,
    Value: DeviceCopy + Copy,
    Scheme: DeviceCopy + Copy,
    KeyEqual: DeviceCopy + Copy,
    Pair<Key, Value>: DeviceCopy,
    EqualWrapper<Key, KeyEqual>: DeviceCopy,
    (): AlignedTo<{ alignment::<Key, Value>() }>,
    [(); alignment::<Key, Value>()]:,
{
}

impl<Key, Value, Scheme, const BUCKET_SIZE: usize, KeyEqual, const SCOPE: ThreadScope>
    StaticMapRef<Key, Value, Scheme, BUCKET_SIZE, KeyEqual, SCOPE>
where
    Key: Copy,
    Value: Copy,
    Scheme: Copy,
    KeyEqual: Copy,
    (): AlignedTo<{ alignment::<Key, Value>() }>,
    [(); alignment::<Key, Value>()]:,
{
    /// Constructs a static map ref from storage ref and configuration.
    ///
    /// # Arguments
    /// * `empty_slot_sentinel` - Pair of (empty key, empty value) for empty slots
    /// * `erased_key_sentinel` - Key value denoting erased slots
    /// * `predicate` - Key equality wrapper (handles sentinel checks)
    /// * `probing_scheme` - Probing scheme for the table
    /// * `storage_ref` - Non-owning ref to bucket storage
    pub const fn new(
        empty_slot_sentinel: Pair<Key, Value>,
        erased_key_sentinel: Key,
        predicate: EqualWrapper<Key, KeyEqual>,
        probing_scheme: Scheme,
        storage_ref: BucketStorageRef<Pair<Key, Value>, BUCKET_SIZE>,
    ) -> Self {
        Self {
            storage_ref,
            empty_slot_sentinel,
            erased_key_sentinel,
            predicate,
            probing_scheme,
        }
    }

    /// Returns the maximum number of elements the container can hold.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.storage_ref.capacity()
    }

    /// Returns the extent of the storage.
    #[inline]
    pub const fn extent(&self) -> Extent {
        self.storage_ref.extent()
    }

    /// Returns the sentinel value used for empty keys.
    #[inline]
    pub const fn empty_key_sentinel(&self) -> Key {
        self.empty_slot_sentinel.first
    }

    /// Returns the sentinel value used for empty payloads.
    #[inline]
    pub const fn empty_value_sentinel(&self) -> Value {
        self.empty_slot_sentinel.second
    }

    /// Returns the sentinel value used for erased keys.
    #[inline]
    pub const fn erased_key_sentinel(&self) -> Key {
        self.erased_key_sentinel
    }

    /// Returns the key equality predicate (wrapper).
    #[inline]
    pub const fn key_eq(&self) -> &EqualWrapper<Key, KeyEqual> {
        &self.predicate
    }

    /// Returns the non-owning storage ref.
    #[inline]
    pub const fn storage_ref(&self) -> BucketStorageRef<Pair<Key, Value>, BUCKET_SIZE> {
        self.storage_ref
    }

    /// Returns the probing scheme.
    #[inline]
    pub const fn probing_scheme(&self) -> &Scheme {
        &self.probing_scheme
    }
}

// Device-only operations: delegate to OpenAddressingRefImpl.
#[cfg(target_arch = "nvptx64")]
impl<Key, Value, Scheme, const BUCKET_SIZE: usize, KeyEqual, const SCOPE: ThreadScope>
    StaticMapRef<Key, Value, Scheme, BUCKET_SIZE, KeyEqual, SCOPE>
where
    Key: DeviceCopy + Copy + PartialEq,
    Value: DeviceCopy + Copy + PartialEq,
    Scheme: ProbingScheme<Key> + Copy,
    KeyEqual: crate::open_addressing::KeyEqual<Key> + Copy,
    Pair<Key, Value>: DeviceCopy,
    (): AlignedTo<{ alignment::<Key, Value>() }>,
    [(); alignment::<Key, Value>()]:,
{
    /// Builds the open-addressing ref impl from this ref's fields.
    #[inline]
    fn as_ref_impl(
        &self,
    ) -> OpenAddressingRefImpl<
        Key,
        Value,
        Scheme,
        BucketStorageRef<Pair<Key, Value>, BUCKET_SIZE>,
        KeyEqual,
        SCOPE,
        false,
    > {
        OpenAddressingRefImpl::new(
            self.storage_ref,
            self.empty_slot_sentinel,
            self.erased_key_sentinel,
            self.predicate,
            self.probing_scheme,
        )
    }

    /// Finds the value associated with the given key.
    ///
    /// # Returns
    /// * `Some(value)` if the key is found
    /// * `None` if the key is not found
    #[inline]
    pub fn find(&self, key: &Key) -> Option<Value> {
        self.as_ref_impl().find(key)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// # Returns
    /// * `true` if the insertion was successful or the key already existed
    /// * `false` if the map is full (should not happen with proper sizing)
    #[inline]
    pub fn insert(&self, value: Pair<Key, Value>) -> bool {
        self.as_ref_impl().insert(value)
    }

    /// Returns whether the given key is present in the map.
    #[inline]
    pub fn contains(&self, key: &Key) -> bool {
        self.find(key).is_some()
    }
}
