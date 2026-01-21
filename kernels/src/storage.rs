//! Device memory storage abstraction for static map
//!
//! This module provides a storage abstraction, which manages a device 
//! memory buffer storing `Pair<Key, Value>` slots.
//!
//! The modern architecture supports bucket_size >= 1, with bucket-based storage
//! that aligns to cooperative group boundaries for efficient GPU access.
//!
//! Architecture:
//! - `Extent`: Represents capacity (dynamic for MVP)
//! - `BucketStorageRef<T, BUCKET_SIZE>`: Device-side reference for bucket access
//! - `BucketStorage<T, BUCKET_SIZE>`: Host-side storage with device memory allocation
//! - Atomic operations: Helpers for device-side concurrent operations

use crate::open_addressing::ThreadScope;
use crate::probing::ProbingScheme;

#[cfg(not(target_arch = "nvptx64"))]
use cust::error::{CudaError, CudaResult};
#[cfg(not(target_arch = "nvptx64"))]
use cust::launch;
#[cfg(not(target_arch = "nvptx64"))]
use cust::memory::{AsyncCopyDestination, DeviceBuffer, DevicePointer};
#[cfg(not(target_arch = "nvptx64"))]
use cust::module::Module;
#[cfg(not(target_arch = "nvptx64"))]
use cust::stream::Stream;
#[cfg(not(target_arch = "nvptx64"))]
use cust_raw::driver_sys;
#[cfg(not(target_arch = "nvptx64"))]
use std::mem::size_of;

#[cfg(not(target_arch = "nvptx64"))]
use core::sync::atomic::{AtomicU32, AtomicU64};
use core::marker::PhantomData;
use core::sync::atomic::Ordering;
#[cfg(target_arch = "nvptx64")]
use cuda_std::atomic::mid;

use cust_core::DeviceCopy;

// Include the compile-time generated primes array
include!(concat!(env!("OUT_DIR"), "/primes.rs"));

/// Extent type that represents container capacity.
///
/// For MVP, only dynamic extent is supported (wraps `usize`).
/// Static (compile-time) extent can be added later if needed.
///
/// # Example
/// ```no_run
/// use cuda_static_map_kernels::storage::Extent;
/// let extent = Extent::new(1000);
/// assert_eq!(extent.value(), 1000);
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Extent {
    value: usize,
}

impl Extent {
    /// Creates a new extent with the given value.
    pub const fn new(value: usize) -> Self {
        Self { value }
    }

    /// Gets the extent value.
    pub const fn value(&self) -> usize {
        self.value
    }
}

// Safety: Extent is a simple wrapper around usize, which is a primitive type that is
// trivially copyable and contains no references or pointers. It is safe to copy
// to/from CUDA device memory.
unsafe impl DeviceCopy for Extent {}

/// Finds the smallest prime >= `n` using binary search on the precomputed primes array.
///
/// This is used for double hashing to ensure the step size is coprime with the table size.
/// Returns `n` if no prime is found (shouldn't happen for reasonable values within PRIMES range).
///
/// # Arguments
/// * `n` - The value to find the next prime for
///
/// # Returns
/// The smallest prime >= `n`, or `n` if not found
fn next_prime_at_least(n: usize) -> usize {
    // Binary search for the smallest prime >= n
    match PRIMES.binary_search(&n) {
        Ok(i) => PRIMES[i], // Exact match
        Err(i) => {
            // Not found, but i is the insertion point
            if i < PRIMES.len() {
                PRIMES[i] // Next prime >= n
            } else {
                // Beyond the array, return the last prime or n
                PRIMES.last().copied().unwrap_or(n)
            }
        }
    }
}

/// Computes a valid extent based on requested capacity, probing scheme, and bucket size.
///
/// The valid capacity is aligned to bucket boundaries and stride (CG_size * bucket_size).
/// - For non-double-hashing schemes: `valid_capacity = ceil(requested / stride) * stride`
/// - For double-hashing schemes: Uses prime number lookup to ensure coprime step sizes
///
/// # Arguments
/// * `requested` - The requested capacity
/// * `cg_size` - Cooperative group size from the probing scheme
/// * `bucket_size` - Size of each bucket
/// * `is_double_hashing` - Whether this is for double hashing (requires prime lookup)
///
/// # Returns
/// A valid extent with capacity aligned to stride boundaries (and prime for double hashing)
///
/// # Example
/// ```no_run
/// use cuda_static_map_kernels::storage::{Extent, make_valid_extent};
/// // Linear probing: cg_size=1, bucket_size=4, requested=10
/// // stride = 1 * 4 = 4
/// // num_buckets = ceil(10/4) = 3
/// // valid_capacity = 3 * 4 = 12
/// let extent = make_valid_extent(10, 1, 4, false);
/// assert_eq!(extent.value(), 12);
///
/// // Double hashing: same parameters but uses prime lookup
/// // num_groups = ceil(10/4) = 3
/// // next_prime >= 3 is 3
/// // valid_capacity = 3 * 4 = 12
/// let extent = make_valid_extent(10, 1, 4, true);
/// assert_eq!(extent.value(), 12);
/// ```
pub fn make_valid_extent(
    requested: usize,
    cg_size: usize,
    bucket_size: usize,
    is_double_hashing: bool,
) -> Extent {
    let stride = cg_size * bucket_size;
    if stride == 0 {
        // Edge case: if stride is 0, return requested (shouldn't happen in practice)
        return Extent::new(requested.max(1));
    }
    
    let num_groups = (requested + stride - 1) / stride; // Ceil division
    
    if is_double_hashing {
        // For double hashing, find smallest prime >= num_groups
        // This ensures step size will be coprime with the table size
        let prime = next_prime_at_least(num_groups);
        Extent::new(prime * stride)
    } else {
        // For linear probing, just align to stride
        Extent::new(num_groups * stride)
    }
}

/// Computes a valid extent for a probing scheme and bucket size.
///
/// This is a convenience function that extracts CG size from the probing scheme
/// and automatically detects if the scheme is double hashing.
///
/// # Type Parameters
/// * `Scheme` - Probing scheme type that implements `ProbingScheme<Key>`
/// * `Key` - Key type (used for trait bounds)
///
/// # Arguments
/// * `requested` - The requested capacity
/// * `scheme` - The probing scheme (used to get CG size and detect double hashing)
/// * `bucket_size` - Size of each bucket
///
/// # Returns
/// A valid extent with capacity aligned to stride boundaries (and prime for double hashing)
pub fn make_valid_extent_for_scheme<Key>(
    requested: usize,
    scheme: &impl ProbingScheme<Key>,
    bucket_size: usize,
) -> Extent {
    let cg_size = scheme.cg_size();
    let is_double_hashing = scheme.is_double_hashing();
    make_valid_extent(requested, cg_size, bucket_size, is_double_hashing)
}

/// Device-side reference to bucket storage.
///
/// This is a lightweight, trivially-copyable type that can be passed to CUDA kernels.
/// It provides bucket-based access to storage slots.
///
/// # Type Parameters
/// * `T` - The slot type (typically `Pair<Key, Value>`)
/// * `BUCKET_SIZE` - Number of slots per bucket
///
/// # Safety
/// The pointer must remain valid for the lifetime of this reference.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BucketStorageRef<T, const BUCKET_SIZE: usize> {
    extent: Extent,
    slots: *const T,
    _phantom: PhantomData<T>,
}

// Safety: BucketStorageRef contains:
// - `extent: Extent` which implements DeviceCopy
// - `slots: *const T` which is a raw pointer (raw pointers are Copy and safe to transfer)
// - `_phantom: PhantomData<T>` which is a zero-sized type
// All fields are trivially copyable and contain no references to CPU memory.
// The pointer itself is just a value that can be safely copied; the caller is responsible
// for ensuring the pointed-to memory is valid device memory.
unsafe impl<T: Copy, const BUCKET_SIZE: usize> DeviceCopy for BucketStorageRef<T, BUCKET_SIZE> {}

/// Type alias for bucket type (array of slots).
pub type BucketType<T, const BUCKET_SIZE: usize> = [T; BUCKET_SIZE];

impl<T, const BUCKET_SIZE: usize> BucketStorageRef<T, BUCKET_SIZE> {
    /// Creates a new bucket storage reference.
    ///
    /// # Safety
    /// - `slots` must point to valid device memory
    /// - The memory must be properly aligned (see `alignment()`)
    /// - The memory must contain at least `extent.value()` elements
    ///
    /// # Arguments
    /// * `extent` - The extent (capacity) of the storage
    /// * `slots` - Pointer to the first slot
    pub const unsafe fn new(extent: Extent, slots: *const T) -> Self
    where
        T: Copy,
    {
        Self {
            extent,
            slots,
            _phantom: PhantomData,
        }
    }

    /// Gets a pointer to the bucket at the given bucket index.
    ///
    /// # Arguments
    /// * `bucket_idx` - The bucket index
    ///
    /// # Returns
    /// Pointer to the first slot of the bucket
    #[cfg(target_arch = "nvptx64")]
    pub fn get_bucket(&self, bucket_idx: usize) -> *const T {
        // Safety: `self.slots` is a valid pointer to device memory (guaranteed by BucketStorageRef::new).
        // The calculation `bucket_idx * BUCKET_SIZE` is bounded by `num_buckets()` which ensures
        // the resulting pointer is within the allocated memory region.
        unsafe { self.slots.add(bucket_idx * BUCKET_SIZE) }
    }

    /// Gets a pointer to the bucket at the given bucket index (host-side).
    ///
    /// # Arguments
    /// * `bucket_idx` - The bucket index
    ///
    /// # Returns
    /// Pointer to the first slot of the bucket
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn get_bucket(&self, bucket_idx: usize) -> *const T {
        // Safety: `self.slots` is a valid pointer to device memory (guaranteed by BucketStorageRef::new).
        // The calculation `bucket_idx * BUCKET_SIZE` is bounded by `num_buckets()` which ensures
        // the resulting pointer is within the allocated memory region.
        unsafe { self.slots.add(bucket_idx * BUCKET_SIZE) }
    }

    /// Gets a pointer to the data (first slot).
    ///
    /// # Returns
    /// Pointer to the first slot
    pub const fn data(&self) -> *const T {
        self.slots
    }

    /// Gets the number of buckets.
    ///
    /// # Returns
    /// Number of buckets (capacity / bucket_size)
    pub const fn num_buckets(&self) -> usize {
        if BUCKET_SIZE == 0 {
            0
        } else {
            self.extent.value() / BUCKET_SIZE
        }
    }

    /// Gets the total capacity (number of slots).
    ///
    /// # Returns
    /// Total number of slots
    pub const fn capacity(&self) -> usize {
        self.extent.value()
    }

    /// Gets the extent.
    ///
    /// # Returns
    /// The extent
    pub const fn extent(&self) -> Extent {
        self.extent
    }

    /// Computes the alignment required for bucket storage.
    ///
    /// Alignment is `min(sizeof(T) * bucket_size, 16)`.
    ///
    /// # Returns
    /// Required alignment in bytes
    pub const fn alignment() -> usize {
        let bucket_size_bytes = core::mem::size_of::<T>() * BUCKET_SIZE;
        if bucket_size_bytes > 16 {
            16
        } else {
            bucket_size_bytes.next_power_of_two()
        }
    }
}

/// Host-side bucket storage that manages device memory allocation.
///
/// This type allocates and manages device memory for bucket-based storage.
/// The storage uses `value_type = T` (for maps, this is `Pair<Key, Value>`, not just `Value`).
///
/// # Type Parameters
/// * `T` - The slot type (typically `Pair<Key, Value>`)
/// * `BUCKET_SIZE` - Number of slots per bucket
///
/// # Example
/// ```no_run
/// use cuda_static_map_kernels::storage::{BucketStorage, Extent};
/// use cuda_static_map_kernels::pair::Pair;
/// use cust::stream::Stream;
/// use cust::stream::StreamFlags;
///
/// let _ctx = cust::quick_init()?;
/// let extent = Extent::new(1000);
/// let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
/// let storage = BucketStorage::<Pair<u32, u32>, 1>::new(extent, &stream)?;
/// ```
#[cfg(not(target_arch = "nvptx64"))]
#[derive(Debug)]
pub struct BucketStorage<T, const BUCKET_SIZE: usize>
where
    T: DeviceCopy,
{
    extent: Extent,
    buffer: DeviceBuffer<T>,
}

#[cfg(not(target_arch = "nvptx64"))]
impl<T, const BUCKET_SIZE: usize> BucketStorage<T, BUCKET_SIZE>
where
    T: DeviceCopy,
{
    /// Creates a new bucket storage with the specified extent.
    ///
    /// Allocates device memory for `extent.value()` slots but does not initialize them.
    /// Use `initialize()` or `initialize_async()` to fill slots with sentinel values.
    ///
    /// # Arguments
    /// * `extent` - The extent (capacity) for the storage
    /// * `stream` - CUDA stream for asynchronous allocation
    ///
    /// # Errors
    /// Returns `CudaResult` if device memory allocation fails.
    pub fn new(extent: Extent, stream: &Stream) -> CudaResult<Self> {
        let capacity = extent.value();
        // Safety: The buffer is only used after `initialize()` or `initialize_async()` fills all slots,
        // ensuring memory is initialized before reads. Stream ordering is maintained: `initialize()`
        // synchronizes, and callers must synchronize after `initialize_async()` before use.
        // `capacity` is a valid usize, so overflow is handled by the allocation function.
        let buffer = unsafe { DeviceBuffer::uninitialized_async(capacity, stream)? };
        Ok(Self { extent, buffer })
    }

    /// Gets a pointer to the device memory buffer.
    ///
    /// # Returns
    /// Device pointer to the first slot
    pub fn data(&self) -> DevicePointer<T> {
        self.buffer.as_slice().as_device_ptr()
    }

    /// Gets a bucket storage reference for device-side access.
    ///
    /// # Returns
    /// A `BucketStorageRef` that can be passed to CUDA kernels
    pub fn storage_ref(&self) -> BucketStorageRef<T, BUCKET_SIZE> {
        // Safety: `self.data()` returns a DevicePointer<T> from `DeviceBuffer`, which points to
        // valid device memory allocated by CUDA (valid for the lifetime of `self`). CUDA's
        // `cudaMalloc` returns memory aligned to at least 256 bytes, which satisfies the
        // `alignment()` requirement (max 16 bytes). The buffer contains `extent.value()` elements
        // matching `self.extent`, ensuring sufficient capacity.
        unsafe { BucketStorageRef::new(self.extent, self.data().as_raw() as *const T) }
    }

    /// Initializes all slots with the given sentinel value.
    ///
    /// This is a synchronous operation. For asynchronous initialization, use `initialize_async()`.
    ///
    /// # Arguments
    /// * `value` - The sentinel value to initialize all slots with
    /// * `stream` - CUDA stream for the operation
    /// * `module` - Optional CUDA module containing the initialization kernel.
    ///              If `None`, falls back to host-to-device copy.
    ///
    /// # Errors
    /// Returns `CudaResult` if the initialization fails.
    pub fn initialize(
        &mut self,
        value: T,
        stream: &Stream,
        module: Option<&Module>,
    ) -> CudaResult<()> {
        // Safety: We immediately synchronize after the call, ensuring the storage
        // is fully initialized before returning.
        unsafe {
            self.initialize_async(value, stream, module)?;
        }
        stream.synchronize()?;
        Ok(())
    }

    /// Asynchronously initializes all slots with the given sentinel value.
    ///
    /// The caller must synchronize the stream before using the storage.
    ///
    /// # Arguments
    /// * `value` - The sentinel value to initialize all slots with
    /// * `stream` - CUDA stream for asynchronous initialization
    /// * `module` - Optional CUDA module containing the initialization kernel.
    ///              If `None`, falls back to host-to-device copy.
    ///
    /// # Errors
    /// Returns `CudaResult` if the initialization fails.
    ///
    /// # Safety
    /// The storage cannot be used until the stream operation completes.
    pub unsafe fn initialize_async(
        &mut self,
        value: T,
        stream: &Stream,
        module: Option<&Module>,
    ) -> CudaResult<()> {
        let capacity = self.capacity();
        if capacity == 0 {
            return Ok(());
        }

        // Try kernel-based initialization if module is available
        if let Some(module) = module {
            // Safety: `self.buffer` is not accessed elsewhere during kernel execution (exclusive
            // mutable access). The function's safety requirements ensure the caller synchronizes
            // the stream before using the storage, guaranteeing the kernel completes before any reads/writes.
            return unsafe { Self::initialize_async_kernel(self, value, stream, module) };
        }

        // Fallback to host-to-device copy
        let host_data: Vec<T> = vec![value; capacity];
        // Safety: `host_data` is a local vector that lives for the function duration and is not
        // modified after creation. `self.buffer` is not accessed elsewhere during the copy.
        // The function's safety requirements ensure the caller synchronizes the stream before
        // using the storage, guaranteeing the copy completes before any reads/writes.
        unsafe {
            self.buffer.async_copy_from(&host_data, stream)?;
        }

        Ok(())
    }

    /// Kernel-based initialization using the compiled PTX module.
    ///
    /// # Safety
    /// The caller must synchronize the stream before using the storage. The kernel
    /// will write to `self.buffer` asynchronously, and the storage is not valid until
    /// the kernel completes.
    unsafe fn initialize_async_kernel(
        &mut self,
        value: T,
        stream: &Stream,
        module: &Module,
    ) -> CudaResult<()> {
        // Allocate device memory for sentinel value
        let sentinel_device = DeviceBuffer::from_slice(&[value])?;
        let sentinel_ptr = sentinel_device.as_device_ptr();

        // Calculate launch configuration
        const BLOCK_SIZE: u32 = 128;
        let capacity = self.capacity();
        let grid_size = ((capacity as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);

        // Get the kernel function
        let init_kernel = module.get_function("initialize_storage_slots")?;

        // Launch the kernel
        // Safety: The kernel launch is safe because:
        // - `self.buffer.as_device_ptr()` returns a valid device pointer to allocated memory
        // - `sentinel_ptr` points to valid device memory containing the sentinel value
        // - `size_of::<T>()` and `capacity` are correct values matching the buffer size
        // - The kernel function signature matches the expected parameters
        // - The launch configuration (grid_size, BLOCK_SIZE) is valid
        unsafe {
            launch!(init_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                self.buffer.as_device_ptr().as_raw() as *mut u8,
                sentinel_ptr.as_raw() as *const u8,
                size_of::<T>(),
                capacity
            ))?;
        }

        Ok(())
    }

    /// Gets the total capacity (number of slots).
    ///
    /// # Returns
    /// Total number of slots
    pub const fn capacity(&self) -> usize {
        self.extent.value()
    }

    /// Gets the number of buckets.
    ///
    /// # Returns
    /// Number of buckets (capacity / bucket_size)
    pub const fn num_buckets(&self) -> usize {
        if BUCKET_SIZE == 0 {
            0
        } else {
            self.capacity() / BUCKET_SIZE
        }
    }

    /// Gets the extent.
    ///
    /// # Returns
    /// The extent
    pub const fn extent(&self) -> Extent {
        self.extent
    }

    /// Gets a reference to the underlying device buffer.
    ///
    /// Useful for advanced use cases where direct access to the buffer is needed.
    ///
    /// # Returns
    /// Reference to the device buffer
    pub fn as_buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }
}

/// atomic reference wrapper.
///
/// This struct wraps a raw pointer and provides atomic operations.
/// It uses `cuda_std` intrinsics on device and standard atomics on host.
pub struct AtomicRef<'a, T, const SCOPE: ThreadScope> {
    ptr: *mut T,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, const SCOPE: ThreadScope> AtomicRef<'a, T, SCOPE> {
    /// Creates a new `AtomicRef` from a raw pointer.
    ///
    /// # Safety
    /// The pointer must be valid and properly aligned for atomic operations.
    pub const unsafe fn new(ptr: *mut T) -> Self {
        Self { ptr, _phantom: PhantomData }
    }
}

#[cfg(target_arch = "nvptx64")]
impl<'a, const SCOPE: ThreadScope> AtomicRef<'a, u32, SCOPE> {
    /// Atomically adds a value to the current value.
    #[inline(always)]
    pub fn fetch_add(&self, val: u32, order: Ordering) -> u32 {
        unsafe {
            // Safety: from `cuda_std::atomic::mid` — pointer must be valid device
            // memory for `u32`, 4-byte aligned, and initialized; ordering must match
            // intended sync; `ThreadScope::Thread` requires no cross-thread aliasing
            // (volatile). `AtomicRef::new` is unsafe and expects the caller to supply
            // such a pointer; we forward ordering/scope unchanged.
            match SCOPE {
                 ThreadScope::System => mid::atomic_fetch_add_u32_system(self.ptr, order, val),
                 ThreadScope::Device => mid::atomic_fetch_add_u32_device(self.ptr, order, val),
                 ThreadScope::Block => mid::atomic_fetch_add_u32_block(self.ptr, order, val),
                 ThreadScope::Thread => {
                     let old = core::ptr::read_volatile(self.ptr);
                     core::ptr::write_volatile(self.ptr, old.wrapping_add(val));
                     old
                 }
            }
        }
    }

    /// Atomically loads the value.
    #[inline(always)]
    pub fn load(&self, order: Ordering) -> u32 {
        unsafe {
            // Safety: same as `fetch_add` — valid, 4-byte aligned, initialized `u32`
            // pointer; caller-chosen ordering/scope; no aliasing under
            // `ThreadScope::Thread` (volatile load).
            match SCOPE {
                ThreadScope::System => mid::atomic_load_32_system(self.ptr, order),
                ThreadScope::Device => mid::atomic_load_32_device(self.ptr, order),
                ThreadScope::Block => mid::atomic_load_32_block(self.ptr, order),
                ThreadScope::Thread => core::ptr::read_volatile(self.ptr),
            }
        }
    }

    /// Atomically stores a value.
    #[inline(always)]
    pub fn store(&self, val: u32, order: Ordering) {
        unsafe {
            // Safety: same as above — valid, 4-byte aligned, initialized pointer;
            // caller-selected ordering/scope; no aliasing for `ThreadScope::Thread`
            // (volatile store).
            match SCOPE {
                ThreadScope::System => mid::atomic_store_32_system(self.ptr, order, val),
                ThreadScope::Device => mid::atomic_store_32_device(self.ptr, order, val),
                ThreadScope::Block => mid::atomic_store_32_block(self.ptr, order, val),
                ThreadScope::Thread => core::ptr::write_volatile(self.ptr, val),
            }
        }
    }
}

#[cfg(not(target_arch = "nvptx64"))]
impl<'a, const SCOPE: ThreadScope> AtomicRef<'a, u32, SCOPE> {
    /// Atomically adds a value to the current value (Host fallback).
    #[inline(always)]
    pub fn fetch_add(&self, val: u32, order: Ordering) -> u32 {
        // Safety: Creating a reference from `self.ptr` is safe because:
        // 1. `AtomicRef::new()` requires the pointer to be valid (not null, not dangling)
        //    and properly aligned for atomic operations, which guarantees 4-byte alignment
        //    required for `AtomicU32`.
        // 2. `u32` and `AtomicU32` have the same size and alignment (guaranteed by Rust
        //    standard library), so the cast from `*mut u32` to `*const AtomicU32` is valid.
        // 3. The memory must be initialized (required for atomic operations to work correctly),
        //    which is the caller's responsibility when creating the `AtomicRef`.
        // 4. The reference lifetime is scoped to this function call, so it cannot outlive
        //    the memory it points to.
        unsafe {
            let atomic = &*(self.ptr as *const AtomicU32);
            atomic.fetch_add(val, order)
        }
    }

    /// Atomically loads the value (Host fallback).
    #[inline(always)]
    pub fn load(&self, order: Ordering) -> u32 {
        unsafe {
            // Safety: Host fallback uses `AtomicU32`; caller must supply a valid,
            // 4-byte aligned, initialized pointer. Ordering is forwarded unchanged.
            // `AtomicRef::new` is unsafe and relies on the caller for those guarantees.
            let atomic = &*(self.ptr as *const AtomicU32);
            atomic.load(order)
        }
    }

    /// Atomically stores a value (Host fallback).
    #[inline(always)]
    pub fn store(&self, val: u32, order: Ordering) {
        unsafe {
            // Safety: Same as `load`: valid, 4-byte aligned, initialized `u32`
            // pointer supplied by the caller; ordering forwarded unchanged.
            let atomic = &*(self.ptr as *const AtomicU32);
            atomic.store(val, order)
        }
    }
}

#[cfg(target_arch = "nvptx64")]
impl<'a, const SCOPE: ThreadScope> AtomicRef<'a, u64, SCOPE> {
    /// Atomically adds a value to the current value.
    ///
    /// # Safety
    /// From `cuda_std::atomic::mid` requirements: pointer must be valid device
    /// memory for `u64`, 8-byte aligned, and initialized; ordering must match
    /// intended synchronization; for `ThreadScope::Thread`, no cross-thread
    /// aliasing (volatile-only). `AtomicRef::new` is unsafe and expects the
    /// caller to provide a valid, aligned pointer; storage initialization
    /// satisfies initialization; we forward ordering/scope unchanged.
    #[inline(always)]
    pub fn fetch_add(&self, val: u64, order: Ordering) -> u64 {
        unsafe {
            match SCOPE {
                 ThreadScope::System => mid::atomic_fetch_add_u64_system(self.ptr, order, val),
                 ThreadScope::Device => mid::atomic_fetch_add_u64_device(self.ptr, order, val),
                 ThreadScope::Block => mid::atomic_fetch_add_u64_block(self.ptr, order, val),
                 ThreadScope::Thread => {
                     let old = core::ptr::read_volatile(self.ptr);
                     core::ptr::write_volatile(self.ptr, old.wrapping_add(val));
                     old
                 }
            }
        }
    }

    /// Atomically loads the value.
    ///
    /// # Safety
    /// Same preconditions as `fetch_add`: valid device pointer, 8-byte
    /// alignment, initialized `u64`, appropriate ordering/scope, and no
    /// aliasing under `ThreadScope::Thread` (volatile load).
    #[inline(always)]
    pub fn load(&self, order: Ordering) -> u64 {
        unsafe {
            match SCOPE {
                ThreadScope::System => mid::atomic_load_64_system(self.ptr, order),
                ThreadScope::Device => mid::atomic_load_64_device(self.ptr, order),
                ThreadScope::Block => mid::atomic_load_64_block(self.ptr, order),
                ThreadScope::Thread => core::ptr::read_volatile(self.ptr),
            }
        }
    }

    /// Atomically stores a value.
    ///
    /// # Safety
    /// Same as above: valid device pointer, 8-byte alignment, initialized
    /// storage, appropriate ordering/scope, and no aliasing under
    /// `ThreadScope::Thread` (volatile store).
    #[inline(always)]
    pub fn store(&self, val: u64, order: Ordering) {
        unsafe {
            match SCOPE {
                ThreadScope::System => mid::atomic_store_64_system(self.ptr, order, val),
                ThreadScope::Device => mid::atomic_store_64_device(self.ptr, order, val),
                ThreadScope::Block => mid::atomic_store_64_block(self.ptr, order, val),
                ThreadScope::Thread => core::ptr::write_volatile(self.ptr, val),
            }
        }
    }
}

#[cfg(not(target_arch = "nvptx64"))]
impl<'a, const SCOPE: ThreadScope> AtomicRef<'a, u64, SCOPE> {
    /// Atomically adds a value to the current value (Host fallback).
    #[inline(always)]
    pub fn fetch_add(&self, val: u64, order: Ordering) -> u64 {
        // Safety: Creating a reference from `self.ptr` is safe because:
        // 1. `AtomicRef::new()` requires the pointer to be valid (not null, not dangling)
        //    and properly aligned for atomic operations, which guarantees 8-byte alignment
        //    required for `AtomicU64`.
        // 2. `u64` and `AtomicU64` have the same size and alignment (guaranteed by Rust
        //    standard library), so the cast from `*mut u64` to `*const AtomicU64` is valid.
        // 3. The memory must be initialized (required for atomic operations to work correctly),
        //    which is the caller's responsibility when creating the `AtomicRef`.
        // 4. The reference lifetime is scoped to this function call, so it cannot outlive
        //    the memory it points to.
        unsafe {
            let atomic = &*(self.ptr as *const AtomicU64);
            atomic.fetch_add(val, order)
        }
    }

    /// Atomically loads the value (Host fallback).
    #[inline(always)]
    pub fn load(&self, order: Ordering) -> u64 {
        unsafe {
            // Safety: Host fallback uses `AtomicU64`; pointer must be valid, 8-byte
            // aligned, and initialized. `AtomicRef::new` is unsafe and expects the
            // caller to provide such a pointer. Ordering is forwarded unchanged.
            let atomic = &*(self.ptr as *const AtomicU64);
            atomic.load(order)
        }
    }

    /// Atomically stores a value (Host fallback).
    #[inline(always)]
    pub fn store(&self, val: u64, order: Ordering) {
        unsafe {
            // Safety: Same as `load`: valid, 8-byte aligned, initialized `u64` pointer
            // supplied by the caller; ordering forwarded unchanged.
            let atomic = &*(self.ptr as *const AtomicU64);
            atomic.store(val, order)
        }
    }
}

/// Device-side reference to counter storage.
///
/// This struct acts as a lightweight reference to the counter storage on the device.
/// It provides access to the atomic counter via `data()`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CounterStorageRef<T, const SCOPE: ThreadScope> {
    ptr: *mut T,
    _phantom: PhantomData<T>,
}

unsafe impl<T: Copy, const SCOPE: ThreadScope> DeviceCopy for CounterStorageRef<T, SCOPE> {}

impl<T, const SCOPE: ThreadScope> CounterStorageRef<T, SCOPE> {
    /// Creates a new `CounterStorageRef`.
    ///
    /// # Safety
    /// `ptr` must point to valid device memory.
    pub const unsafe fn new(ptr: *mut T) -> Self {
        Self { ptr, _phantom: PhantomData }
    }

    /// Gets an atomic reference to the counter.
    ///
    /// # Returns
    /// An `AtomicRef` wrapper for atomic operations.
    pub fn data<'a>(&self) -> AtomicRef<'a, T, SCOPE> {
        unsafe {
            // Safety: `self.ptr` is valid and properly aligned because `CounterStorageRef`
            // is only created via `CounterStorage::storage_ref()`, which uses a valid device
            // pointer from `DeviceBuffer::as_device_ptr()`. Device buffers guarantee valid,
            // properly aligned pointers for their element type.
            AtomicRef::new(self.ptr)
        }
    }
}

/// Counter storage for atomic counting operations.
///
/// This struct manages a device buffer for storing a single atomic counter.
/// It provides methods to reset the counter and load the value to the host.
///
/// # Type Parameters
/// * `T` - The counter type (e.g., `usize`, `u64`, `u32`)
/// * `SCOPE` - Thread scope for atomic operations (default: `Device`)
#[cfg(not(target_arch = "nvptx64"))]
pub struct CounterStorage<T: DeviceCopy, const SCOPE: ThreadScope = { ThreadScope::Device }> {
    buffer: DeviceBuffer<T>,
}

/// Helper function to convert CUDA driver error to CudaError
#[cfg(not(target_arch = "nvptx64"))]
fn cuda_error_from_enum(err: cust_raw::driver_sys::cudaError_enum) -> CudaError {
    use cust_raw::driver_sys::cudaError_enum;
    match err {
        cudaError_enum::CUDA_SUCCESS => unreachable!(),
        cudaError_enum::CUDA_ERROR_INVALID_VALUE => CudaError::InvalidValue,
        cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY => CudaError::OutOfMemory,
        cudaError_enum::CUDA_ERROR_NOT_INITIALIZED => CudaError::NotInitialized,
        cudaError_enum::CUDA_ERROR_DEINITIALIZED => CudaError::Deinitialized,
        cudaError_enum::CUDA_ERROR_INVALID_CONTEXT => CudaError::InvalidContext,
        cudaError_enum::CUDA_ERROR_INVALID_DEVICE => CudaError::InvalidDevice,
        _ => CudaError::UnknownError,
    }
}

#[cfg(not(target_arch = "nvptx64"))]
impl<T, const SCOPE: ThreadScope> CounterStorage<T, SCOPE>
where
    T: DeviceCopy + Default + Clone,
{
    /// Creates a new counter storage.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream for allocation
    pub fn new(stream: &Stream) -> CudaResult<Self> {
        // Allocate a single element buffer
        let buffer = unsafe { DeviceBuffer::uninitialized_async(1, stream)? };
        Ok(Self { buffer })
    }

    /// Resets the counter to zero.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream for the operation
    ///
    /// # Safety
    /// - `stream` must be a valid CUDA stream.
    /// - The memset is enqueued asynchronously; callers must ensure the stream completes
    ///   before reading the counter on host or device (e.g., synchronize or use stream ordering).
    /// - `T` must have the same size as its atomic representation (typically `u32` or `u64`).
    ///   This ensures that `cuMemsetD8Async` correctly zeros the atomic counter value.
    pub unsafe fn reset(&mut self, stream: &Stream) -> CudaResult<()> {
        // SAFETY: `cuMemsetD8Async` requirements are met:
        // - Pointer: `self.buffer` owns a single-element `DeviceBuffer<T>`, so
        //   `as_device_ptr().as_raw()` points to `size_of::<T>()` bytes of valid device memory.
        // - Size: we pass `size_of::<T>()` (exact allocation for one element); guarded against ZSTs.
        // - Stream: `stream.as_inner()` yields a valid `CUstream`.
        // - Host aliasing: `&mut self` ensures exclusive host access during enqueue.
        unsafe {
            if size_of::<T>() != 0 {
                let result = driver_sys::cuMemsetD8Async(
                    self.buffer.as_device_ptr().as_raw(),
                    0,
                    size_of::<T>(),
                    stream.as_inner(),
                );
                match result {
                    driver_sys::cudaError_enum::CUDA_SUCCESS => Ok(()),
                    e => Err(cuda_error_from_enum(e)),
                }?;
            }
            Ok(())
        }
    }

    /// Loads the counter value to the host.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream for the operation
    ///
    /// # Returns
    /// The counter value on the host
    pub fn load_to_host(&self, stream: &Stream) -> CudaResult<T> {
        let mut host_val = vec![T::default()];
        // Safety: buffer has size 1, host_val has size 1. Pointers are valid.
        unsafe {
            self.buffer.async_copy_to(&mut host_val, stream)?;
        }
        stream.synchronize()?;
        Ok(host_val[0])
    }

    /// Gets a device pointer to the counter.
    ///
    /// # Returns
    /// Device pointer to the counter
    pub fn data(&self) -> DevicePointer<T> {
        self.buffer.as_device_ptr()
    }

    /// Gets a storage reference for device access.
    ///
    /// # Returns
    /// A `CounterStorageRef` that can be passed to CUDA kernels.
    pub fn storage_ref(&self) -> CounterStorageRef<T, SCOPE> {
        // Safety: data() returns a valid device pointer.
        // The pointer is valid for the lifetime of self.
        unsafe { CounterStorageRef::new(self.data().as_raw() as *mut T) }
    }
}

/// Atomic operations for device-side concurrent access.
///
/// This module provides helpers for atomic compare-and-swap operations
/// on storage slots. Different strategies are used based on slot size:
/// - Packed CAS: For `value_type <= 8 bytes` (single atomic operation)
/// - Back-to-back CAS: For `value_type > 8 bytes` (CAS key, then write value)
/// - CAS + dependent write: Alternative strategy for larger types
#[cfg(target_arch = "nvptx64")]
pub mod atomic_ops {
    use crate::open_addressing::ThreadScope;
    use core::sync::atomic::Ordering;
    use cuda_std::atomic::mid;

    /// Performs atomic compare-and-swap for types <= 8 bytes (packed).
    ///
    /// This uses a single atomic operation on the entire value by reinterpreting
    /// it as u32 (for 4 bytes) or u64 (for 8 bytes)
    ///
    /// # Type Parameters
    /// * `SCOPE` - The thread scope for the atomic operation (System, Device, Block, Thread).
    ///
    /// # Arguments
    /// * `address` - Pointer to the slot value
    /// * `expected` - Expected value (as bytes)
    /// * `desired` - Desired value (as bytes)
    /// * `size` - Size of the value type in bytes (must be 4 or 8)
    ///
    /// # Returns
    /// `true` if the swap succeeded, `false` otherwise. Returns `false` if `size` is not 4 or 8.
    ///
    /// # Safety
    /// * `address` must point to valid device memory.
    /// * If `size` is 4, `address` must be 4-byte aligned.
    /// * If `size` is 8, `address` must be 8-byte aligned.
    /// * `expected` and `desired` must be valid bit patterns for the value type
    /// * When `SCOPE` is `ThreadScope::Thread`, each thread must operate on distinct
    ///   memory locations (no concurrent access to the same location by multiple threads).
    ///   Volatile operations are used which prevent compiler reordering but provide no
    ///   cross-thread synchronization guarantees.
    #[inline(always)]
    pub unsafe fn packed_cas<const SCOPE: ThreadScope>(
        address: *mut u8,
        expected: u64,
        desired: u64,
        size: usize,
    ) -> bool {
        match size {
            4 => {
                let slot_ptr = address as *mut u32;
                let expected_u32 = expected as u32;
                let desired_u32 = desired as u32;

                // Safety:
                // 1. `slot_ptr` is derived from `address`.
                // 2. Caller guarantees `address` is valid and 4-byte aligned when size is 4.
                // 3. `mid` intrinsics require valid pointers and appropriate memory ordering.
                let old = match SCOPE {
                    ThreadScope::System => unsafe {
                        mid::atomic_compare_and_swap_u32_system(
                            slot_ptr, expected_u32, desired_u32, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Device => unsafe {
                        mid::atomic_compare_and_swap_u32_device(
                            slot_ptr, expected_u32, desired_u32, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Block => unsafe {
                        mid::atomic_compare_and_swap_u32_block(
                            slot_ptr, expected_u32, desired_u32, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Thread => unsafe {
                        // See safety requirements: each thread must operate on distinct memory locations.
                        let current = core::ptr::read_volatile(slot_ptr);
                        if current == expected_u32 {
                            core::ptr::write_volatile(slot_ptr, desired_u32);
                            expected_u32
                        } else {
                            current
                        }
                    },
                };
                old == expected_u32
            }
            8 => {
                let slot_ptr = address as *mut u64;

                // Safety:
                // 1. `slot_ptr` is derived from `address`.
                // 2. Caller guarantees `address` is valid and 8-byte aligned when size is 8.
                let old = match SCOPE {
                    ThreadScope::System => unsafe {
                        mid::atomic_compare_and_swap_u64_system(
                            slot_ptr, expected, desired, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Device => unsafe {
                        mid::atomic_compare_and_swap_u64_device(
                            slot_ptr, expected, desired, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Block => unsafe {
                        mid::atomic_compare_and_swap_u64_block(
                            slot_ptr, expected, desired, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Thread => unsafe {
                        let current = core::ptr::read_volatile(slot_ptr);
                        if current == expected {
                            core::ptr::write_volatile(slot_ptr, desired);
                            expected
                        } else {
                            current
                        }
                    },
                };
                old == expected
            }
            _ => false,
        }
    }

    /// Performs back-to-back CAS for pairs > 8 bytes.
    ///
    /// First CAS the key, then CAS the value if successful.
    ///
    /// # Type Parameters
    /// * `SCOPE` - The thread scope for the atomic operation.
    ///
    /// # Arguments
    /// * `key_ptr` - Pointer to the key field
    /// * `value_ptr` - Pointer to the value field
    /// * `expected_key` - Expected key value
    /// * `desired_key` - Desired key value
    /// * `expected_value` - Expected value (empty sentinel)
    /// * `desired_value` - Desired value
    /// * `key_size` - Size of key in bytes (must be 4 or 8)
    /// * `value_size` - Size of value in bytes (must be 4 or 8)
    ///
    /// # Returns
    /// `true` if both CAS operations succeeded, `false` otherwise.
    /// Returns `false` if `key_size` or `value_size` is not 4 or 8.
    ///
    /// # Safety
    /// * Both pointers must point to valid device memory
    /// * Both pointers must be properly aligned for their respective sizes (4 or 8 bytes)
    /// * When `SCOPE` is `ThreadScope::Thread`, each thread must operate on distinct
    ///   memory locations (no concurrent access to the same location by multiple threads).
    ///   Volatile operations are used which prevent compiler reordering but provide no
    ///   cross-thread synchronization guarantees.
    #[inline(always)]
    pub unsafe fn back_to_back_cas<const SCOPE: ThreadScope>(
        key_ptr: *mut u8,
        value_ptr: *mut u8,
        expected_key: u64,
        desired_key: u64,
        expected_value: u64,
        desired_value: u64,
        key_size: usize,
        value_size: usize,
    ) -> bool {
        // First CAS the key
        let key_success = match key_size {
            4 => {
                let key_ptr_u32 = key_ptr as *mut u32;
                let expected_key_u32 = expected_key as u32;
                let desired_key_u32 = desired_key as u32;

                // Safety: Caller guarantees key_ptr is valid and 4-byte aligned.
                let old = match SCOPE {
                    ThreadScope::System => unsafe {
                        mid::atomic_compare_and_swap_u32_system(
                            key_ptr_u32, expected_key_u32, desired_key_u32, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Device => unsafe {
                        mid::atomic_compare_and_swap_u32_device(
                            key_ptr_u32, expected_key_u32, desired_key_u32, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Block => unsafe {
                        mid::atomic_compare_and_swap_u32_block(
                            key_ptr_u32, expected_key_u32, desired_key_u32, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Thread => unsafe {
                        let current = core::ptr::read_volatile(key_ptr_u32);
                        if current == expected_key_u32 {
                            core::ptr::write_volatile(key_ptr_u32, desired_key_u32);
                            expected_key_u32
                        } else {
                            current
                        }
                    },
                };
                old == expected_key_u32
            }
            8 => {
                let key_ptr_u64 = key_ptr as *mut u64;

                // Safety: Caller guarantees key_ptr is valid and 8-byte aligned.
                let old = match SCOPE {
                    ThreadScope::System => unsafe {
                        mid::atomic_compare_and_swap_u64_system(
                            key_ptr_u64, expected_key, desired_key, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Device => unsafe {
                        mid::atomic_compare_and_swap_u64_device(
                            key_ptr_u64, expected_key, desired_key, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Block => unsafe {
                        mid::atomic_compare_and_swap_u64_block(
                            key_ptr_u64, expected_key, desired_key, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Thread => unsafe {
                        let current = core::ptr::read_volatile(key_ptr_u64);
                        if current == expected_key {
                            core::ptr::write_volatile(key_ptr_u64, desired_key);
                            expected_key
                        } else {
                            current
                        }
                    },
                };
                old == expected_key
            }
            _ => false,
        };

        if !key_success {
            return false;
        }

        // Then CAS the value
        let mut value_success = false;
        let mut current_expected = expected_value;

        while !value_success {
            value_success = match value_size {
                4 => {
                    let value_ptr_u32 = value_ptr as *mut u32;
                    let expected_value_u32 = current_expected as u32;
                    let desired_value_u32 = desired_value as u32;

                    // Safety: Caller guarantees value_ptr is valid and 4-byte aligned.
                    let old = match SCOPE {
                        ThreadScope::System => unsafe {
                            mid::atomic_compare_and_swap_u32_system(
                                value_ptr_u32, expected_value_u32, desired_value_u32, Ordering::Relaxed
                            )
                        },
                        ThreadScope::Device => unsafe {
                            mid::atomic_compare_and_swap_u32_device(
                                value_ptr_u32, expected_value_u32, desired_value_u32, Ordering::Relaxed
                            )
                        },
                        ThreadScope::Block => unsafe {
                            mid::atomic_compare_and_swap_u32_block(
                                value_ptr_u32, expected_value_u32, desired_value_u32, Ordering::Relaxed
                            )
                        },
                        ThreadScope::Thread => unsafe {
                            let current = core::ptr::read_volatile(value_ptr_u32);
                            if current == expected_value_u32 {
                                core::ptr::write_volatile(value_ptr_u32, desired_value_u32);
                                expected_value_u32
                            } else {
                                current
                            }
                        },
                    };

                    if old == expected_value_u32 {
                        true
                    } else {
                        current_expected = old as u64;
                        false
                    }
                }
                8 => {
                    let value_ptr_u64 = value_ptr as *mut u64;

                    // Safety: Caller guarantees value_ptr is valid and 8-byte aligned.
                    let old = match SCOPE {
                        ThreadScope::System => unsafe {
                            mid::atomic_compare_and_swap_u64_system(
                                value_ptr_u64, current_expected, desired_value, Ordering::Relaxed
                            )
                        },
                        ThreadScope::Device => unsafe {
                            mid::atomic_compare_and_swap_u64_device(
                                value_ptr_u64, current_expected, desired_value, Ordering::Relaxed
                            )
                        },
                        ThreadScope::Block => unsafe {
                            mid::atomic_compare_and_swap_u64_block(
                                value_ptr_u64, current_expected, desired_value, Ordering::Relaxed
                            )
                        },
                        ThreadScope::Thread => unsafe {
                            let current = core::ptr::read_volatile(value_ptr_u64);
                            if current == current_expected {
                                core::ptr::write_volatile(value_ptr_u64, desired_value);
                                current_expected
                            } else {
                                current
                            }
                        },
                    };

                    if old == current_expected {
                        true
                    } else {
                        current_expected = old;
                        false
                    }
                }
                _ => false,
            };
        }

        true
    }

    /// Performs CAS + dependent write for larger types.
    ///
    /// First CAS the key, then write the value if successful (non-atomic write).
    ///
    /// # Type Parameters
    /// * `SCOPE` - The thread scope for the atomic operation.
    ///
    /// # Arguments
    /// * `key_ptr` - Pointer to the key field
    /// * `value_ptr` - Pointer to the value field
    /// * `expected_key` - Expected key value
    /// * `desired_key` - Desired key value
    /// * `desired_value` - Desired value to write
    /// * `key_size` - Size of key in bytes (must be 4 or 8)
    /// * `value_size` - Size of value in bytes
    ///
    /// # Returns
    /// `true` if the key CAS succeeded and value was written, `false` otherwise.
    /// Returns `false` if `key_size` is not 4 or 8.
    ///
    /// # Safety
    /// * Both pointers must point to valid device memory
    /// * `key_ptr` must be aligned to 4 or 8 bytes depending on `key_size`.
    /// * `value_ptr` must be properly aligned for `value_size`.
    /// * When `SCOPE` is `ThreadScope::Thread`, each thread must operate on distinct
    ///   memory locations (no concurrent access to the same location by multiple threads).
    ///   Volatile operations are used which prevent compiler reordering but provide no
    ///   cross-thread synchronization guarantees.
    #[inline(always)]
    pub unsafe fn cas_dependent_write<const SCOPE: ThreadScope>(
        key_ptr: *mut u8,
        value_ptr: *mut u8,
        expected_key: u64,
        desired_key: u64,
        desired_value: u64,
        key_size: usize,
        value_size: usize,
    ) -> bool {
        // First CAS the key
        let key_success = match key_size {
            4 => {
                let key_ptr_u32 = key_ptr as *mut u32;
                let expected_key_u32 = expected_key as u32;
                let desired_key_u32 = desired_key as u32;

                // Safety: Caller guarantees key_ptr is valid and 4-byte aligned.
                let old = match SCOPE {
                    ThreadScope::System => unsafe {
                        mid::atomic_compare_and_swap_u32_system(
                            key_ptr_u32, expected_key_u32, desired_key_u32, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Device => unsafe {
                        mid::atomic_compare_and_swap_u32_device(
                            key_ptr_u32, expected_key_u32, desired_key_u32, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Block => unsafe {
                        mid::atomic_compare_and_swap_u32_block(
                            key_ptr_u32, expected_key_u32, desired_key_u32, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Thread => unsafe {
                        let current = core::ptr::read_volatile(key_ptr_u32);
                        if current == expected_key_u32 {
                            core::ptr::write_volatile(key_ptr_u32, desired_key_u32);
                            expected_key_u32
                        } else {
                            current
                        }
                    },
                };
                old == expected_key_u32
            }
            8 => {
                let key_ptr_u64 = key_ptr as *mut u64;

                // Safety: Caller guarantees key_ptr is valid and 8-byte aligned.
                let old = match SCOPE {
                    ThreadScope::System => unsafe {
                        mid::atomic_compare_and_swap_u64_system(
                            key_ptr_u64, expected_key, desired_key, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Device => unsafe {
                        mid::atomic_compare_and_swap_u64_device(
                            key_ptr_u64, expected_key, desired_key, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Block => unsafe {
                        mid::atomic_compare_and_swap_u64_block(
                            key_ptr_u64, expected_key, desired_key, Ordering::Relaxed
                        )
                    },
                    ThreadScope::Thread => unsafe {
                        let current = core::ptr::read_volatile(key_ptr_u64);
                        if current == expected_key {
                            core::ptr::write_volatile(key_ptr_u64, desired_key);
                            expected_key
                        } else {
                            current
                        }
                    },
                };
                old == expected_key
            }
            _ => false,
        };

        if key_success {
            // Write value non-atomically (exclusive access)
            // Safety:
            // 1. `key_success` being true implies we won the CAS, granting exclusive access.
            // 2. Caller guarantees `value_ptr` is valid and properly aligned.
            unsafe {
                match value_size {
                    4 => {
                        let value_ptr_u32 = value_ptr as *mut u32;
                        *value_ptr_u32 = desired_value as u32;
                    }
                    8 => {
                        let value_ptr_u64 = value_ptr as *mut u64;
                        *value_ptr_u64 = desired_value;
                    }
                    _ => {
                        core::ptr::copy_nonoverlapping(
                            &desired_value as *const u64 as *const u8,
                            value_ptr,
                            value_size,
                        );
                    }
                }
            }
        }

        key_success
    }

    /// Waits for payload to be written (for concurrent operations).
    ///
    /// This is used when a key CAS succeeds but the value might not be written yet.
    /// Spins until the value is no longer the empty sentinel.
    ///
    /// # Type Parameters
    /// * `SCOPE` - The thread scope for the atomic operation.
    ///
    /// # Arguments
    /// * `value_ptr` - Pointer to atomic value
    /// * `empty_value` - The empty sentinel value to wait for
    /// * `value_size` - Size of value in bytes (must be 4 or 8)
    ///
    /// # Safety
    /// * `value_ptr` must point to valid device memory.
    /// * `value_ptr` must be properly aligned for `value_size` (4 or 8 bytes).
    /// * When `SCOPE` is `ThreadScope::Thread`, each thread must operate on distinct
    ///   memory locations (no concurrent access to the same location by multiple threads).
    ///   Volatile operations are used which prevent compiler reordering but provide no
    ///   cross-thread synchronization guarantees.
    #[inline]
    pub unsafe fn wait_for_payload<const SCOPE: ThreadScope>(
        value_ptr: *mut u8,
        empty_value: u64,
        value_size: usize,
    ) {
        loop {
            let current = match value_size {
                4 => {
                    let value_ptr_u32 = value_ptr as *mut u32;
                    // Safety: Caller guarantees value_ptr is valid and 4-byte aligned.
                    match SCOPE {
                        ThreadScope::System => unsafe {
                            mid::atomic_load_32_system(value_ptr_u32, Ordering::Acquire) as u64
                        },
                        ThreadScope::Device => unsafe {
                            mid::atomic_load_32_device(value_ptr_u32, Ordering::Acquire) as u64
                        },
                        ThreadScope::Block => unsafe {
                            mid::atomic_load_32_block(value_ptr_u32, Ordering::Acquire) as u64
                        },
                        ThreadScope::Thread => unsafe {
                            // See safety requirements: each thread must operate on distinct memory locations.
                            core::ptr::read_volatile(value_ptr_u32) as u64
                        },
                    }
                }
                8 => {
                    let value_ptr_u64 = value_ptr as *mut u64;
                    // Safety: Caller guarantees value_ptr is valid and 8-byte aligned.
                    match SCOPE {
                        ThreadScope::System => unsafe {
                            mid::atomic_load_64_system(value_ptr_u64, Ordering::Acquire)
                        },
                        ThreadScope::Device => unsafe {
                            mid::atomic_load_64_device(value_ptr_u64, Ordering::Acquire)
                        },
                        ThreadScope::Block => unsafe {
                            mid::atomic_load_64_block(value_ptr_u64, Ordering::Acquire)
                        },
                        ThreadScope::Thread => unsafe {
                            // See safety requirements: each thread must operate on distinct memory locations.
                            core::ptr::read_volatile(value_ptr_u64)
                        },
                    }
                }
                _ => break,
            };

            if current != empty_value {
                break;
            }

            // Simple busy wait
            core::hint::spin_loop();
        }
    }
}

// Host-side placeholder implementations (for testing/development)
#[cfg(not(target_arch = "nvptx64"))]
pub mod atomic_ops {
    use crate::open_addressing::ThreadScope;
    use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    /// Host-side placeholder for packed CAS.
    ///
    /// Ignores `SCOPE` as standard Rust atomics don't support it directly.
    ///
    /// # Safety
    /// * `address` must be valid and aligned.
    #[inline]
    pub unsafe fn packed_cas<const SCOPE: ThreadScope>(
        address: *mut u8,
        expected: u64,
        desired: u64,
        size: usize,
    ) -> bool {
        match size {
            4 => {
                let slot_ptr = address as *mut AtomicU32;
                // Safety: address is valid and aligned per function safety requirements
                let atomic = unsafe { &*slot_ptr };
                atomic
                    .compare_exchange(
                        expected as u32,
                        desired as u32,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            }
            8 => {
                let slot_ptr = address as *mut AtomicU64;
                // Safety: address is valid and aligned per function safety requirements
                let atomic = unsafe { &*slot_ptr };
                atomic
                    .compare_exchange(expected, desired, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            }
            _ => false,
        }
    }

    /// Host-side placeholder for back-to-back CAS.
    ///
    /// Ignores `SCOPE` as standard Rust atomics don't support it directly.
    ///
    /// # Safety
    /// * Pointers must be valid and aligned.
    #[inline]
    pub unsafe fn back_to_back_cas<const SCOPE: ThreadScope>(
        key_ptr: *mut u8,
        value_ptr: *mut u8,
        expected_key: u64,
        desired_key: u64,
        expected_value: u64,
        desired_value: u64,
        key_size: usize,
        value_size: usize,
    ) -> bool {
        let key_success = match key_size {
            4 => {
                // Safety: key_ptr is valid and aligned per function safety requirements
                let key_atomic = unsafe { &*(key_ptr as *mut AtomicU32) };
                key_atomic
                    .compare_exchange(
                        expected_key as u32,
                        desired_key as u32,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            }
            8 => {
                // Safety: key_ptr is valid and aligned per function safety requirements
                let key_atomic = unsafe { &*(key_ptr as *mut AtomicU64) };
                key_atomic
                    .compare_exchange(expected_key, desired_key, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            }
            _ => false,
        };

        if !key_success {
            return false;
        }

        let mut value_success = false;
        let mut current_expected = expected_value;
        while !value_success {
            value_success = match value_size {
                4 => {
                    // Safety: value_ptr is valid and aligned per function safety requirements
                    let value_atomic = unsafe { &*(value_ptr as *mut AtomicU32) };
                    match value_atomic.compare_exchange(
                        current_expected as u32,
                        desired_value as u32,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => true,
                        Err(old) => {
                            current_expected = old as u64;
                            false
                        }
                    }
                }
                8 => {
                    // Safety: value_ptr is valid and aligned per function safety requirements
                    let value_atomic = unsafe { &*(value_ptr as *mut AtomicU64) };
                    match value_atomic.compare_exchange(
                        current_expected,
                        desired_value,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => true,
                        Err(old) => {
                            current_expected = old;
                            false
                        }
                    }
                }
                _ => false,
            };
        }

        true
    }

    /// Host-side placeholder for CAS + dependent write.
    ///
    /// Ignores `SCOPE` as standard Rust atomics don't support it directly.
    ///
    /// # Safety
    /// * Pointers must be valid and aligned.
    #[inline]
    pub unsafe fn cas_dependent_write<const SCOPE: ThreadScope>(
        key_ptr: *mut u8,
        value_ptr: *mut u8,
        expected_key: u64,
        desired_key: u64,
        desired_value: u64,
        key_size: usize,
        value_size: usize,
    ) -> bool {
        let key_success = match key_size {
            4 => {
                // Safety: key_ptr is valid and aligned per function safety requirements
                let key_atomic = unsafe { &*(key_ptr as *mut AtomicU32) };
                key_atomic
                    .compare_exchange(
                        expected_key as u32,
                        desired_key as u32,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            }
            8 => {
                // Safety: key_ptr is valid and aligned per function safety requirements
                let key_atomic = unsafe { &*(key_ptr as *mut AtomicU64) };
                key_atomic
                    .compare_exchange(expected_key, desired_key, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            }
            _ => false,
        };

        if key_success {
            // Safety: key_success being true implies we won the CAS, granting exclusive access.
            // Caller guarantees value_ptr is valid and properly aligned.
            unsafe {
                match value_size {
                    4 => {
                        *(value_ptr as *mut u32) = desired_value as u32;
                    }
                    8 => {
                        *(value_ptr as *mut u64) = desired_value;
                    }
                    _ => {
                        core::ptr::copy_nonoverlapping(
                            &desired_value as *const u64 as *const u8,
                            value_ptr,
                            value_size,
                        );
                    }
                }
            }
        }

        key_success
    }

    /// Host-side placeholder for wait-for-payload.
    ///
    /// Ignores `SCOPE` as standard Rust atomics don't support it directly.
    ///
    /// # Safety
    /// * Pointers must be valid and aligned.
    #[inline]
    pub unsafe fn wait_for_payload<const SCOPE: ThreadScope>(
        value_ptr: *mut u8,
        empty_value: u64,
        value_size: usize
    ) {
        loop {
            let current = match value_size {
                4 => {
                    // Safety: value_ptr is valid and aligned per function safety requirements
                    let value_atomic = unsafe { &*(value_ptr as *mut AtomicU32) };
                    value_atomic.load(Ordering::Acquire) as u64
                }
                8 => {
                    // Safety: value_ptr is valid and aligned per function safety requirements
                    let value_atomic = unsafe { &*(value_ptr as *mut AtomicU64) };
                    value_atomic.load(Ordering::Acquire)
                }
                _ => break,
            };

            if current != empty_value {
                break;
            }

            core::hint::spin_loop();
        }
    }
}

// Device-side kernel for initializing storage
// This kernel is compiled to PTX and loaded as a module.
#[cfg(target_arch = "nvptx64")]
mod device_kernel {
    use cuda_std::prelude::*;

    /// Device kernel to initialize storage slots by copying sentinel bytes.
    /// This works for any type by copying the raw bytes of the sentinel value.
    /// The sentinel is passed as a byte array and copied to each slot.
    ///
    /// # Safety
    /// - `slots` must point to valid device memory of at least `capacity * slot_size` bytes
    /// - `sentinel_bytes` must point to valid device memory of at least `slot_size` bytes
    /// - `capacity` must be the actual capacity of the buffer
    /// - `slot_size` must be the size in bytes of one slot (sizeof(Pair<Key, Value>))
    #[kernel]
    #[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
    pub unsafe fn initialize_storage_slots(
        slots: *mut u8,
        sentinel_bytes: *const u8,
        slot_size: usize,
        capacity: usize,
    ) {
        let idx = thread::index_1d() as usize;
        if idx < capacity {
            // Safety: idx is within capacity, so the offset is within the slots allocation.
            let dest = unsafe { slots.add(idx * slot_size) };

            // Safety: slots and sentinel_bytes are distinct buffers of at least slot_size,
            // and u8 pointers are always aligned.
            unsafe {
                core::ptr::copy_nonoverlapping(sentinel_bytes, dest, slot_size);
            }
        }
    }
}
