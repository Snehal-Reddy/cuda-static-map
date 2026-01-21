//! Open addressing primitives and abstractions.

#[cfg(not(target_arch = "nvptx64"))]
use cust::stream::Stream;
#[cfg(not(target_arch = "nvptx64"))]
use cust::error::CudaResult;
#[cfg(not(target_arch = "nvptx64"))]
use crate::storage::{BucketStorage, make_valid_extent_for_scheme};

use crate::storage::{BucketStorageRef};
use crate::pair::{Pair, AlignedTo, alignment};
use crate::probing::ProbingScheme;
use cust_core::DeviceCopy;

/// Thread scope for atomic operations.
///
/// This enum is used as a const generic parameter to control the scope of
/// atomic operations (e.g., system-wide, device-wide, block-wide).
#[derive(Copy, Clone, PartialEq, Eq, Debug, core::marker::ConstParamTy)]
pub enum ThreadScope {
    /// System-wide scope (GPU + CPU via zero-copy memory).
    ///
    /// Use this scope when:
    /// - Using zero-copy memory accessible from both CPU and GPU
    /// - Maximum compatibility is required across all memory spaces
    /// - Performance is less critical than correctness
    ///
    /// **Note**: This is the widest scope and may have the highest overhead.
    System,

    /// Device-wide scope (all threads on the GPU).
    ///
    /// **Recommended default** for most use cases.
    ///
    /// Use this scope when:
    /// - All operations are within a single GPU device
    /// - Best balance of performance and correctness
    /// - Most common case for `static_map` operations
    ///
    /// This is the default scope for `StaticMap` and provides good
    /// performance while maintaining correctness guarantees.
    Device,

    /// Block-wide scope (all threads in the same block).
    ///
    /// Use this scope when:
    /// - Operations are guaranteed to be within a single thread block
    /// - Maximum performance is needed within a block
    /// - You can guarantee block-level synchronization (e.g., via barriers)
    ///
    /// **Warning**: Incorrect use can lead to data races if threads from
    /// different blocks access the same memory locations.
    Block,

    /// Thread scope (no atomicity relative to other threads).
    ///
    /// Use this scope when:
    /// - Each thread operates on distinct memory locations
    /// - No cross-thread synchronization is needed
    /// - Volatile operations are sufficient for compiler optimization prevention
    ///
    /// **Warning**: This scope provides no atomicity guarantees. Only use
    /// when you can guarantee that no two threads will access the same
    /// memory location concurrently.
    Thread,
}

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

/// Open addressing implementation layer (Host-side).
///
/// This struct acts as a host-side abstraction layer that encapsulates:
/// - Storage management (`BucketStorage`)
/// - Probing scheme
/// - Key equality predicate
/// - Sentinel values
///
/// It provides access to the underlying storage and configuration for high-level APIs.
///
/// # Type Parameters
/// * `Key` - Key type
/// * `Value` - Value type
/// * `Scheme` - Probing scheme type
/// * `BUCKET_SIZE` - Size of each bucket (default 1)
/// * `KeyEqual` - Key equality predicate
/// * `SCOPE` - Thread scope for atomic operations
#[cfg(not(target_arch = "nvptx64"))]
pub struct OpenAddressingImpl<
    Key,
    Value,
    Scheme,
    const BUCKET_SIZE: usize,
    KeyEqual,
    const SCOPE: ThreadScope,
>
where
    Key: DeviceCopy + Copy + PartialEq,
    Value: DeviceCopy + Copy,
    Scheme: ProbingScheme<Key>,
    KeyEqual: self::KeyEqual<Key>,
    Pair<Key, Value>: DeviceCopy,
    (): AlignedTo<{ alignment::<Key, Value>() }>,
{
    storage: BucketStorage<Pair<Key, Value>, BUCKET_SIZE>,
    empty_slot_sentinel: Pair<Key, Value>,
    erased_key_sentinel: Key,
    predicate: KeyEqual,
    probing_scheme: Scheme,
}

#[cfg(not(target_arch = "nvptx64"))]
impl<Key, Value, Scheme, const BUCKET_SIZE: usize, KeyEqual, const SCOPE: ThreadScope>
    OpenAddressingImpl<Key, Value, Scheme, BUCKET_SIZE, KeyEqual, SCOPE>
where
    Key: DeviceCopy + Copy + PartialEq,
    Value: DeviceCopy + Copy,
    Scheme: ProbingScheme<Key>,
    KeyEqual: self::KeyEqual<Key> + Copy,
    Pair<Key, Value>: DeviceCopy,
    (): AlignedTo<{ alignment::<Key, Value>() }>,
{
    /// Constructs a new `OpenAddressingImpl` with auto-extracted erased key sentinel.
    ///
    /// The erased key sentinel is automatically extracted from the empty slot sentinel
    /// using `extract_key(empty_slot_sentinel)`. This constructor allows the erased and
    /// empty key sentinels to be the same value.
    ///
    /// # Arguments
    /// * `capacity` - Requested capacity (lower bound)
    /// * `empty_slot_sentinel` - Sentinel value for empty slots
    /// * `pred` - Key equality predicate
    /// * `probing_scheme` - Probing scheme
    /// * `stream` - CUDA stream for initialization
    pub fn new(
        capacity: usize,
        empty_slot_sentinel: Pair<Key, Value>,
        pred: KeyEqual,
        probing_scheme: Scheme,
        stream: &Stream,
    ) -> CudaResult<Self> {
        // Auto-extract erased key sentinel from empty slot sentinel
        let erased_key_sentinel = empty_slot_sentinel.first;
        Self::new_internal(
            capacity,
            empty_slot_sentinel,
            erased_key_sentinel,
            pred,
            probing_scheme,
            stream,
        )
    }

    /// Constructs a new `OpenAddressingImpl` with explicit erased key sentinel.
    ///
    /// This constructor requires that the empty and erased key sentinels be different.
    ///
    /// # Arguments
    /// * `capacity` - Requested capacity (lower bound)
    /// * `empty_slot_sentinel` - Sentinel value for empty slots
    /// * `erased_key_sentinel` - Sentinel key for erased slots
    /// * `pred` - Key equality predicate
    /// * `probing_scheme` - Probing scheme
    /// * `stream` - CUDA stream for initialization
    ///
    /// # Panics
    /// Panics if `empty_key_sentinel() == erased_key_sentinel()` (they must be different).
    pub fn new_with_erased(
        capacity: usize,
        empty_slot_sentinel: Pair<Key, Value>,
        erased_key_sentinel: Key,
        pred: KeyEqual,
        probing_scheme: Scheme,
        stream: &Stream,
    ) -> CudaResult<Self> {
        // Validate that empty and erased sentinels are different
        // Use equality check without Debug formatting (matches C++ CUCO_EXPECTS behavior)
        let empty_key_sentinel = empty_slot_sentinel.first;
        if empty_key_sentinel == erased_key_sentinel {
            panic!("The empty key sentinel and erased key sentinel cannot be the same value.");
        }

        Self::new_internal(
            capacity,
            empty_slot_sentinel,
            erased_key_sentinel,
            pred,
            probing_scheme,
            stream,
        )
    }

    /// Constructs a new `OpenAddressingImpl` with a desired load factor.
    ///
    /// The capacity is computed as `ceil(n / desired_load_factor)`, where `n` is the
    /// number of elements to insert. The erased key sentinel is automatically extracted
    /// from the empty slot sentinel. This constructor allows the erased and empty key
    /// sentinels to be the same value.
    ///
    /// # Arguments
    /// * `n` - The number of elements to insert
    /// * `desired_load_factor` - The desired load factor (e.g., 0.5 implies 50% load)
    /// * `empty_slot_sentinel` - Sentinel value for empty slots
    /// * `pred` - Key equality predicate
    /// * `probing_scheme` - Probing scheme
    /// * `stream` - CUDA stream for initialization
    ///
    /// # Panics
    /// Panics if `desired_load_factor <= 0.0` or `desired_load_factor >= 1.0`.
    pub fn with_load_factor(
        n: usize,
        desired_load_factor: f64,
        empty_slot_sentinel: Pair<Key, Value>,
        pred: KeyEqual,
        probing_scheme: Scheme,
        stream: &Stream,
    ) -> CudaResult<Self> {
        // Validate load factor
        assert!(
            desired_load_factor > 0.0 && desired_load_factor < 1.0,
            "The desired load factor must be in the range (0.0, 1.0)"
        );

        // Compute requested capacity from load factor
        let requested_capacity = (n as f64 / desired_load_factor).ceil() as usize;

        // Auto-extract erased key sentinel from empty slot sentinel
        let erased_key_sentinel = empty_slot_sentinel.first;

        Self::new_internal(
            requested_capacity,
            empty_slot_sentinel,
            erased_key_sentinel,
            pred,
            probing_scheme,
            stream,
        )
    }

    /// Internal constructor that performs the actual construction without validation.
    ///
    /// This is used by all public constructors to avoid code duplication.
    fn new_internal(
        capacity: usize,
        empty_slot_sentinel: Pair<Key, Value>,
        erased_key_sentinel: Key,
        pred: KeyEqual,
        probing_scheme: Scheme,
        stream: &Stream,
    ) -> CudaResult<Self> {
        // Compute valid extent
        let extent = make_valid_extent_for_scheme(capacity, &probing_scheme, BUCKET_SIZE);

        // Allocate storage
        let mut storage = BucketStorage::new(extent, stream)?;

        // Initialize storage with empty sentinel
        // Safety: We return the object, so it's the caller's responsibility to synchronize
        // if using a non-default stream, as documented.
        unsafe {
            storage.initialize_async(empty_slot_sentinel, stream, None)?;
        }

        Ok(Self {
            storage,
            empty_slot_sentinel,
            erased_key_sentinel,
            predicate: pred,
            probing_scheme,
        })
    }

    /// Clears the container (synchronous).
    ///
    /// This method synchronizes the stream after clearing, ensuring the operation
    /// is complete before returning.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream used for the operation
    pub fn clear(&mut self, stream: &Stream) -> CudaResult<()> {
        self.storage.initialize(self.empty_slot_sentinel, stream, None)
    }

    /// Clears the container asynchronously.
    ///
    /// This method does not synchronize the stream.
    ///
    /// # Safety
    /// The caller is responsible for synchronization if needed.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream used for the operation
    pub unsafe fn clear_async(&mut self, stream: &Stream) -> CudaResult<()> {
        unsafe {
            self.storage.initialize_async(self.empty_slot_sentinel, stream, None)
        }
    }

    /// Gets the capacity of the container.
    pub fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    /// Gets the empty key sentinel.
    pub fn empty_key_sentinel(&self) -> Key {
        self.empty_slot_sentinel.first
    }

    /// Gets the erased key sentinel.
    pub fn erased_key_sentinel(&self) -> Key {
        self.erased_key_sentinel
    }

    /// Gets the key equality predicate.
    pub fn key_eq(&self) -> KeyEqual {
        self.predicate
    }

    /// Gets the probing scheme.
    pub fn probing_scheme(&self) -> &Scheme {
        &self.probing_scheme
    }

    /// Gets the storage reference for device access.
    pub fn storage_ref(&self) -> BucketStorageRef<Pair<Key, Value>, BUCKET_SIZE> {
        self.storage.storage_ref()
    }

    /// Extracts the key from a value type (pair).
    ///
    /// For maps, this returns the first element of the pair (the key).
    /// This method is used internally for consistency with the C++ reference implementation.
    ///
    /// # Arguments
    /// * `slot` - The slot value (pair) to extract the key from
    ///
    /// # Returns
    /// A reference to the key
    pub fn extract_key<'a>(&self, slot: &'a Pair<Key, Value>) -> &'a Key {
        &slot.first
    }
}

/// Open addressing reference implementation (Device-side).
///
/// This struct encapsulates the device-side logic for open addressing, including:
/// - Bucket-based probing
/// - Atomic operations for insert/find
/// - Sentinel checking
/// - Key equality testing
///
/// # Type Parameters
/// * `Key` - Key type
/// * `Value` - Value type (payload)
/// * `Scheme` - Probing scheme type
/// * `StorageRef` - Bucket storage reference type
/// * `KeyEqual` - Key equality predicate type
/// * `SCOPE` - Thread scope for atomic operations
/// * `ALLOWS_DUPLICATES` - Whether to allow duplicate keys
#[cfg(target_arch = "nvptx64")]
struct KeySizeCheck<T>(core::marker::PhantomData<T>);

#[cfg(target_arch = "nvptx64")]
impl<T> KeySizeCheck<T> {
    const CHECK: () = assert!(core::mem::size_of::<T>() <= 8, "Container does not support key types larger than 8 bytes.");
}

#[cfg(target_arch = "nvptx64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OpenAddressingRefImpl<
    Key,
    Value,
    Scheme,
    StorageRef,
    KeyEqual,
    const SCOPE: ThreadScope,
    const ALLOWS_DUPLICATES: bool,
> where
    (): AlignedTo<{ alignment::<Key, Value>() }>,
    [(); alignment::<Key, Value>()]:,
{
    storage_ref: StorageRef,
    empty_slot_sentinel: Pair<Key, Value>,
    erased_key_sentinel: Key,
    predicate: EqualWrapper<Key, KeyEqual>,
    probing_scheme: Scheme,
}

#[cfg(target_arch = "nvptx64")]
impl<
        Key,
        Value,
        Scheme,
        const BUCKET_SIZE: usize,
        KeyEqual,
        const SCOPE: ThreadScope,
        const ALLOWS_DUPLICATES: bool,
    >
    OpenAddressingRefImpl<
        Key,
        Value,
        Scheme,
        BucketStorageRef<Pair<Key, Value>, BUCKET_SIZE>,
        KeyEqual,
        SCOPE,
        ALLOWS_DUPLICATES,
    >
where
    Key: DeviceCopy + Copy + PartialEq,
    Value: DeviceCopy + Copy + PartialEq,
    Scheme: ProbingScheme<Key>,
    KeyEqual: self::KeyEqual<Key> + Copy,
    Pair<Key, Value>: DeviceCopy,
    (): AlignedTo<{ alignment::<Key, Value>() }>,
    [(); alignment::<Key, Value>()]:,
{
    /// Creates a new `OpenAddressingRefImpl`.
    pub const fn new(
        storage_ref: BucketStorageRef<Pair<Key, Value>, BUCKET_SIZE>,
        empty_slot_sentinel: Pair<Key, Value>,
        erased_key_sentinel: Key,
        predicate: EqualWrapper<Key, KeyEqual>,
        probing_scheme: Scheme,
    ) -> Self {
        Self {
            storage_ref,
            empty_slot_sentinel,
            erased_key_sentinel,
            predicate,
            probing_scheme,
        }
    }

    /// Finds a key in the map.
    ///
    /// # Arguments
    /// * `key` - The key to search for
    ///
    /// # Returns
    /// * `Some(value)` if the key is found
    /// * `None` if the key is not found
    #[inline]
    pub fn find(&self, key: &Key) -> Option<Value> {
        // Use the probing scheme to get an iterator over the probe sequence
        let capacity = self.storage_ref.capacity();
        let mut iter = self.probing_scheme.make_iterator(key, BUCKET_SIZE, capacity);
        let init_idx = iter.current(); // Store initial index for wrap-around detection

        loop {
            let bucket_idx = iter.current() / BUCKET_SIZE;
            let bucket_ptr = self.storage_ref.get_bucket(bucket_idx);

            // Iterate over slots in the bucket
            for i in 0..BUCKET_SIZE {
                let slot_ptr = unsafe { bucket_ptr.add(i) };

                // Safety: slot_ptr is valid.
                let slot_val = unsafe { *slot_ptr };
                let slot_key = slot_val.first;

                match self.predicate.equal_for_find(key, &slot_key) {
                    EqualResult::Equal => {
                        let pair_size = core::mem::size_of::<Pair<Key, Value>>();

                        // Wait for payload to be written (for concurrent operations)
                        // Only needed when pair_size > 8 (back-to-back CAS strategy)
                        if pair_size > 8 {
                            use crate::storage::atomic_ops;
                            // Safety:
                            // - `slot_ptr` points into storage created via `BucketStorageRef::new`, so it is in-bounds and
                            //   properly aligned for `Pair<Key, Value>`.
                            // - Taking `&mut (*(slot_ptr as *mut Pair<Key, Value>)).second` produces a unique, in-bounds
                            //   reference to the `Value` field of that pair (we never alias this slot mutably elsewhere in
                            //   this function), so casting it to `*mut u8` yields a valid, aligned pointer to the value.
                            let val_ptr = unsafe {
                                &mut (*(slot_ptr as *mut Pair<Key, Value>)).second as *mut Value as *mut u8
                            };

                            let val_size = core::mem::size_of::<Value>();
                            // Safety:
                            // - `self.empty_slot_sentinel.second` is a fully-initialized `Value` living for `'self`, so
                            //   taking a raw pointer to it is valid for reads.
                            // - We copy at most `val_size.min(8)` bytes, where `val_size` is `size_of::<Value>()`, so the
                            //   read range stays within the source object and the write range stays within the local `u64`
                            //   `temp`; these regions are disjoint, satisfying `copy_nonoverlapping`’s requirements.
                            let empty_val_u64 = unsafe {
                                let mut temp = 0u64;
                                core::ptr::copy_nonoverlapping(
                                    &self.empty_slot_sentinel.second as *const Value as *const u8,
                                    &mut temp as *mut u64 as *mut u8,
                                    val_size.min(8), // Only copy up to 8 bytes
                                );
                                temp
                            };

                            // Safety:
                            // - `val_ptr` is derived from `slot_ptr` (aligned, in-bounds `Pair<Key, Value>` obtained via
                            //   `get_bucket`), so it is valid and aligned for `Value`.
                            // - `empty_val_u64` copies up to `val_size.min(8)` bytes from the valid
                            //   `self.empty_slot_sentinel.second` into a local `u64`; the copy is bounded and
                            //   non-overlapping.
                            // - `atomic_ops::wait_for_payload` only dereferences `value_ptr` when `value_size` is 4 or 8;
                            //   for other sizes it exits immediately, so passing larger `val_size` is still safe. When
                            //   `val_size` is 4 or 8, `Pair` alignment guarantees `val_ptr` meets the alignment the helper
                            //   requires.
                            // - If `SCOPE == ThreadScope::Thread`, callers configuring `OpenAddressingImpl<..., SCOPE>` must
                            //   enforce one-thread-per-slot to satisfy the volatile-only semantics of that mode; other scopes
                            //   rely on CUDA atomics for synchronization.
                            unsafe {
                                atomic_ops::wait_for_payload::<SCOPE>(
                                    val_ptr,
                                    empty_val_u64,
                                    val_size
                                );
                            }
                        }

                        // Safety:
                        // - `slot_ptr` points to an in-bounds, properly aligned `Pair<Key, Value>` obtained via
                        //   `get_bucket`, so dereferencing it is valid.
                        // - We only read `second`; no mutable alias exists at this point, so the shared reference rules
                        //   are respected. Copying out the `Value` (which is `Copy`) is safe.
                        let final_val = unsafe { (*slot_ptr).second };
                        return Some(final_val);
                    }
                    EqualResult::Empty => {
                        return None;
                    }
                    EqualResult::Unequal => {}
                    EqualResult::Available => {}
                }
            }

            iter.next();
            // Check if we've wrapped around to the initial index
            if iter.current() == init_idx {
                return None;  // Key not found
            }
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// # Arguments
    /// * `value` - The key-value pair to insert
    ///
    /// # Returns
    /// * `true` if the insertion was successful or key already existed
    /// * `false` if the map is full (should not happen with proper sizing)
    #[inline]
    pub fn insert(&self, value: Pair<Key, Value>) -> bool {
        use crate::storage::atomic_ops;

        let key = value.first;
        let capacity = self.storage_ref.capacity();

        let mut iter = self.probing_scheme.make_iterator(&key, BUCKET_SIZE, capacity);
        let init_idx = iter.current(); // Store initial index for wrap-around detection

        let pair_size = core::mem::size_of::<Pair<Key, Value>>();
        let key_size = core::mem::size_of::<Key>();
        let val_size = core::mem::size_of::<Value>();

        loop {
            let bucket_idx = iter.current() / BUCKET_SIZE;
            let bucket_ptr = self.storage_ref.get_bucket(bucket_idx);

            for i in 0..BUCKET_SIZE {
                // Safety (pointer math/cast):
                // - `bucket_ptr` was produced by `storage_ref.get_bucket`, which guarantees
                //   a properly aligned bucket base within the allocated backing array.
                // - `i` is bounded by `0..BUCKET_SIZE`, so `bucket_ptr.add(i)` stays within
                //   that bucket.
                // - We cast to `*mut` because we may write the slot later, but the map
                //   coordinates mutation through atomic operations so aliasing is controlled.
                let slot_ptr = unsafe { bucket_ptr.add(i) as *mut Pair<Key, Value> };
                let slot_u8 = slot_ptr as *mut u8;

                // Safety (raw deref):
                // - `slot_ptr` is derived from the in-bounds, aligned bucket pointer above.
                // - The storage holds initialized `Pair<Key, Value>` slots, and concurrent
                //   updates use atomics (or CUDA atomics) so reading here observes a valid
                //   representation.
                let slot_val = unsafe { *slot_ptr };
                let slot_key = slot_val.first;

                match self.predicate.equal_for_insert(&key, &slot_key, ALLOWS_DUPLICATES) {
                    EqualResult::Equal => {
                        return true;
                    }
                    EqualResult::Available => {
                        let success = if pair_size <= 8 {
                            // Packed CAS
                            // Safety:
                            // - `slot_u8` points into `slot_ptr` derived from `bucket_ptr` obtained via
                            //   `get_bucket`, so it is in-bounds, properly aligned for `Pair<Key, Value>`, and
                            //   thus aligned for `pair_size` bytes.
                            // - The copies into `expected_u64`/`desired_u64` read exactly `pair_size` bytes from
                            //   `self.empty_slot_sentinel` and `value`, both valid `Pair<Key, Value>` instances;
                            //   destinations are local temps and non-overlapping with sources.
                            // - `atomic_ops::packed_cas` demands a valid, aligned address and correct size (4 or 8);
                            //   we pass the aligned `slot_u8` and `pair_size` (<= 8) with bit-patterns produced
                            //   directly from the corresponding `Pair` values.
                            // - If `SCOPE == ThreadScope::Thread`, the caller configuring `OpenAddressingImpl<..., SCOPE>`
                            //   must ensure one-thread-per-slot so the volatile-thread fallback in `packed_cas`
                            //   sees no concurrent access; other scopes rely on CUDA atomics for synchronization.
                            unsafe {
                                let mut expected_u64 = 0u64;
                                core::ptr::copy_nonoverlapping(
                                    &self.empty_slot_sentinel as *const Pair<Key, Value> as *const u8,
                                    &mut expected_u64 as *mut u64 as *mut u8,
                                    pair_size
                                );

                                let mut desired_u64 = 0u64;
                                core::ptr::copy_nonoverlapping(
                                    &value as *const Pair<Key, Value> as *const u8,
                                    &mut desired_u64 as *mut u64 as *mut u8,
                                    pair_size
                                );

                                atomic_ops::packed_cas::<SCOPE>(
                                    slot_u8,
                                    expected_u64,
                                    desired_u64,
                                    pair_size
                                )
                            }
                        } else {
                            // Back-to-back CAS or CAS + dependent write
                            // Key size must be <= 8 bytes for this strategy
                            let _ = KeySizeCheck::<Key>::CHECK;

                            // Safety:
                            // - `bucket_ptr` was obtained from `self.storage_ref.get_bucket(bucket_idx)`, which is
                            //   constructed from `BucketStorageRef::new` and only ever created with a valid, in-bounds,
                            //   properly aligned device allocation for `Pair<Key, Value>` slots.
                            // - `Pair`’s `#[repr(C)]` + alignment machinery means `(*slot_ptr).first/second` are laid out
                            //   and aligned exactly as `Key`/`Value`, so casting their references to `*mut u8` and using
                            //   `key_size`/`val_size` matches the guarantees required by the atomic helpers.
                            // - `get_u64` always copies at most `key_size`/`val_size` bytes (each derived from
                            //   `core::mem::size_of`) from `&self.empty_slot_sentinel.*` / `&value.*` into a local `u64`,
                            //   so the source ranges are valid and non-overlapping with the destination, and the copy
                            //   length never exceeds the 8 bytes available in `temp`.
                            // - Both `atomic_ops::back_to_back_cas` and `atomic_ops::cas_dependent_write` are given the
                            //   same `key_ptr`/`value_ptr` produced from the well-aligned `slot_ptr` above, and the
                            //   `key_size`/`val_size` we pass are exactly the element sizes, satisfying their pointer and
                            //   size preconditions.
                            // - For scopes other than `ThreadScope::Thread`, cross-thread safety is provided by the CUDA
                            //   atomics used inside `atomic_ops`; if `ThreadScope::Thread` is chosen for `SCOPE`, the
                            //   caller constructing `OpenAddressingImpl<_, _, _, _, _, SCOPE>` must ensure one-thread-per-
                            //   slot access, which is the only remaining requirement of those intrinsics.
                            unsafe {
                                let get_u64 = |ptr: *const u8, size: usize| -> u64 {
                                    let mut temp = 0u64;
                                    core::ptr::copy_nonoverlapping(ptr, &mut temp as *mut u64 as *mut u8, size);
                                    temp
                                };

                                let expected_key_u64 = get_u64(&self.empty_slot_sentinel.first as *const Key as *const u8, key_size);
                                let desired_key_u64 = get_u64(&value.first as *const Key as *const u8, key_size);

                                if val_size <= 8 {
                                    let expected_val_u64 = get_u64(&self.empty_slot_sentinel.second as *const Value as *const u8, val_size);
                                    let desired_val_u64 = get_u64(&value.second as *const Value as *const u8, val_size);

                                     atomic_ops::back_to_back_cas::<SCOPE>(
                                        &mut (*slot_ptr).first as *mut Key as *mut u8,
                                        &mut (*slot_ptr).second as *mut Value as *mut u8,
                                        expected_key_u64,
                                        desired_key_u64,
                                        expected_val_u64,
                                        desired_val_u64,
                                        key_size,
                                        val_size
                                    )
                                } else {
                                    let desired_val_u64 = get_u64(&value.second as *const Value as *const u8, 8.min(val_size));

                                     atomic_ops::cas_dependent_write::<SCOPE>(
                                        &mut (*slot_ptr).first as *mut Key as *mut u8,
                                        &mut (*slot_ptr).second as *mut Value as *mut u8,
                                        expected_key_u64,
                                        desired_key_u64,
                                        desired_val_u64,
                                        key_size,
                                        val_size
                                    )
                                }
                            }
                        };

                        if success {
                            return true;
                        }
                    }
                    EqualResult::Unequal => {}
                    EqualResult::Empty => {}
                }
            }
            iter.next();
            // Check if we've wrapped around to the initial index
            if iter.current() == init_idx {
                return false;  // Map is full (should not happen with proper sizing)
            }
        }
    }
}
