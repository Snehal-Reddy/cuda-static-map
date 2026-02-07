//! Shared static map implementation for both CPU and GPU
//!
//! This code compiles for both host (CPU) and device (GPU) targets.
//! When compiled for GPU (nvptx64-nvidia-cuda), it generates PTX code.
//! When compiled for CPU, it's used as a regular Rust library.

use crate::pair::{alignment, AlignedTo, Pair};

#[cfg(not(target_arch = "nvptx64"))]
use crate::open_addressing::OpenAddressingImpl;
#[cfg(not(target_arch = "nvptx64"))]
use crate::probing::ProbingScheme;
#[cfg(not(target_arch = "nvptx64"))]
use cust::stream::Stream;
#[cfg(not(target_arch = "nvptx64"))]
use cust::error::CudaResult;
use cust_core::DeviceCopy;

/// A GPU-accelerated, unordered, associative container of key-value pairs with unique keys.
///
/// This is a thin wrapper around `OpenAddressingImpl`. Capacity is computed via
/// `make_valid_extent` (inside `OpenAddressingImpl::new`) based on the probing scheme,
/// cooperative group size, and bucket size.
///
/// # Type Parameters
/// * `Key` - Key type
/// * `Value` - Value (mapped) type
/// * `Scheme` - Probing scheme (e.g. linear probing)
/// * `BUCKET_SIZE` - Slots per bucket (default: 1)
/// * `KeyEqual` - Key equality predicate (default: [`DefaultKeyEqual`](crate::open_addressing::DefaultKeyEqual))
/// * `SCOPE` - Thread scope for atomics (default: [`ThreadScope::Device`](crate::open_addressing::ThreadScope::Device))
///
/// For the common case you can omit the trailing parameters and use
/// `StaticMap<Key, Value, Scheme>`.
pub struct StaticMap<
    Key,
    Value,
    Scheme,
    const BUCKET_SIZE: usize = 1,
    KeyEqual = crate::open_addressing::DefaultKeyEqual,
    const SCOPE: crate::open_addressing::ThreadScope = { crate::open_addressing::ThreadScope::Device },
>
where
    Key: DeviceCopy + Copy + PartialEq,
    Value: DeviceCopy + Copy,
    Scheme: crate::probing::ProbingScheme<Key>,
    KeyEqual: crate::open_addressing::KeyEqual<Key> + Copy,
    Pair<Key, Value>: DeviceCopy,
    (): AlignedTo<{ alignment::<Key, Value>() }>,
{
    #[cfg(not(target_arch = "nvptx64"))]
    impl_: OpenAddressingImpl<Key, Value, Scheme, BUCKET_SIZE, KeyEqual, SCOPE>,
    #[cfg(not(target_arch = "nvptx64"))]
    empty_value_sentinel: Value,
    #[cfg(target_arch = "nvptx64")]
    _phantom: core::marker::PhantomData<(Key, Value, Scheme, KeyEqual)>,
}

// CPU-specific implementations
#[cfg(not(target_arch = "nvptx64"))]
impl<
    Key,
    Value,
    Scheme,
    const BUCKET_SIZE: usize,
    KeyEqual,
    const SCOPE: crate::open_addressing::ThreadScope,
> StaticMap<Key, Value, Scheme, BUCKET_SIZE, KeyEqual, SCOPE>
where
    Key: DeviceCopy + Copy + PartialEq,
    Value: DeviceCopy + Copy,
    Scheme: ProbingScheme<Key>,
    KeyEqual: crate::open_addressing::KeyEqual<Key> + Copy,
    Pair<Key, Value>: DeviceCopy,
    (): AlignedTo<{ alignment::<Key, Value>() }>,
{
    /// Constructs a statically-sized map with the specified initial capacity.
    ///
    /// The actual capacity is computed via `make_valid_extent` inside `OpenAddressingImpl::new`
    /// from the requested capacity, probing scheme, CG size, and bucket size.
    ///
    /// # Arguments
    /// * `capacity` - Requested lower-bound map size
    /// * `empty_key_sentinel` - Reserved key value for empty slots
    /// * `empty_value_sentinel` - Reserved mapped value for empty slots
    /// * `pred` - Key equality predicate
    /// * `probing_scheme` - Probing scheme
    /// * `stream` - CUDA stream for initialization
    pub fn new(
        capacity: usize,
        empty_key_sentinel: Key,
        empty_value_sentinel: Value,
        pred: KeyEqual,
        probing_scheme: Scheme,
        stream: &Stream,
    ) -> CudaResult<Self> {
        let empty_slot_sentinel = Pair::new(empty_key_sentinel, empty_value_sentinel);
        let impl_ =
            OpenAddressingImpl::new(capacity, empty_slot_sentinel, pred, probing_scheme, stream)?;
        Ok(Self {
            impl_,
            empty_value_sentinel: empty_value_sentinel,
        })
    }

    /// Constructs a map with explicit erased key sentinel.
    ///
    /// The empty and erased key sentinels must be different.
    pub fn new_with_erased(
        capacity: usize,
        empty_key_sentinel: Key,
        empty_value_sentinel: Value,
        erased_key_sentinel: Key,
        pred: KeyEqual,
        probing_scheme: Scheme,
        stream: &Stream,
    ) -> CudaResult<Self> {
        let empty_slot_sentinel = Pair::new(empty_key_sentinel, empty_value_sentinel);
        let impl_ = OpenAddressingImpl::new_with_erased(
            capacity,
            empty_slot_sentinel,
            erased_key_sentinel,
            pred,
            probing_scheme,
            stream,
        )?;
        Ok(Self {
            impl_,
            empty_value_sentinel: empty_value_sentinel,
        })
    }

    /// Constructs a map with a desired load factor.
    ///
    /// Capacity is computed as `ceil(n / desired_load_factor)` then passed through
    /// `make_valid_extent`.
    ///
    /// # Panics
    /// Panics if `desired_load_factor` is not in (0.0, 1.0).
    pub fn with_load_factor(
        n: usize,
        desired_load_factor: f64,
        empty_key_sentinel: Key,
        empty_value_sentinel: Value,
        pred: KeyEqual,
        probing_scheme: Scheme,
        stream: &Stream,
    ) -> CudaResult<Self> {
        let empty_slot_sentinel = Pair::new(empty_key_sentinel, empty_value_sentinel);
        let impl_ = OpenAddressingImpl::with_load_factor(
            n,
            desired_load_factor,
            empty_slot_sentinel,
            pred,
            probing_scheme,
            stream,
        )?;
        Ok(Self {
            impl_,
            empty_value_sentinel: empty_value_sentinel,
        })
    }

    /// Clears the container (synchronous).
    pub fn clear(&mut self, stream: &Stream) -> CudaResult<()> {
        self.impl_.clear(stream)
    }

    /// Clears the container asynchronously.
    ///
    /// # Safety
    /// Caller is responsible for stream synchronization if needed.
    pub unsafe fn clear_async(&mut self, stream: &Stream) -> CudaResult<()> {
        unsafe { self.impl_.clear_async(stream) }
    }

    /// Gets the capacity of the map.
    pub fn capacity(&self) -> usize {
        self.impl_.capacity()
    }

    /// Gets the empty key sentinel.
    pub fn empty_key_sentinel(&self) -> Key {
        self.impl_.empty_key_sentinel()
    }

    /// Gets the empty value (mapped) sentinel stored for ref construction.
    pub fn empty_value_sentinel(&self) -> Value {
        self.empty_value_sentinel
    }

    /// Gets the erased key sentinel.
    pub fn erased_key_sentinel(&self) -> Key {
        self.impl_.erased_key_sentinel()
    }

    /// Gets the storage reference for device access.
    pub fn storage_ref(&self) -> crate::storage::BucketStorageRef<Pair<Key, Value>, BUCKET_SIZE> {
        self.impl_.storage_ref()
    }

    /// Inserts all key-value pairs in the range (bulk operation).
    ///
    /// TODO: launch bulk insert kernel using container_ref.
    pub fn insert(
        &mut self,
        _pairs: &[Pair<Key, Value>],
    ) -> Result<usize, Box<dyn std::error::Error>> {
        todo!("Implement bulk insert kernel launch")
    }

    /// Finds values for all keys in the range (bulk operation).
    ///
    /// TODO: launch bulk find kernel using container_ref.
    pub fn find(
        &self,
        _keys: &[Key],
        _output: &mut [Value],
    ) -> Result<(), Box<dyn std::error::Error>> {
        todo!("Implement bulk find kernel launch")
    }

    /// Checks if keys are contained in the map (bulk operation).
    ///
    /// TODO: launch bulk contains kernel using container_ref.
    pub fn contains(
        &self,
        _keys: &[Key],
        _output: &mut [bool],
    ) -> Result<(), Box<dyn std::error::Error>> {
        todo!("Implement bulk contains kernel launch")
    }

    /// Gets the number of elements in the map.
    ///
    /// TODO: May require kernel launch to count (CounterStorage).
    pub fn size(&self) -> usize {
        todo!("Implement size")
    }

    /// Gets a device-side reference for use in custom kernels.
    ///
    /// The returned ref can be passed by value to GPU kernels, which may then call
    /// `.find()`, `.insert()`, and `.contains()` on it.
    pub fn device_ref(&self) -> crate::static_map_ref::StaticMapRef<
        Key,
        Value,
        Scheme,
        BUCKET_SIZE,
        KeyEqual,
        SCOPE,
    > {
        use crate::open_addressing::EqualWrapper;
        let empty_key = self.impl_.empty_key_sentinel();
        let empty_value = self.empty_value_sentinel();
        let erased_key = self.impl_.erased_key_sentinel();
        let predicate = EqualWrapper::new(empty_key, erased_key, self.impl_.key_eq());
        let empty_slot_sentinel = Pair::new(empty_key, empty_value);
        crate::static_map_ref::StaticMapRef::new(
            empty_slot_sentinel,
            erased_key,
            predicate,
            *self.impl_.probing_scheme(),
            self.impl_.storage_ref(),
        )
    }
}

// Shared traits and types that work on both CPU and GPU

/// Trait for types that can be used as keys in the static map
pub trait MapKey: Copy + core::fmt::Debug {
    /// Check if this key is the empty sentinel value
    fn is_empty_sentinel(&self) -> bool;

    /// Check if this key is the erased sentinel value
    fn is_erased_sentinel(&self) -> bool;
}

/// Trait for types that can be used as values in the static map
pub trait MapValue: Copy + core::fmt::Debug {
    /// Check if this value is the empty sentinel value
    fn is_empty_sentinel(&self) -> bool;
}
