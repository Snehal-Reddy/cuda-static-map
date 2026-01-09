//! Device memory storage abstraction for static map
//!
//! This module provides a storage abstraction, which manages a device 
//! memory buffer storing `Pair<Key, Value>` slots.
//!
//! For MVP, bucket_size = 1 (flat storage) is used, which simplifies the implementation.
//! `DeviceBuffer` is used directly for simplicity. An allocator abstraction can be added later 
//! if needed.

use crate::pair::Pair;

#[cfg(not(target_arch = "nvptx64"))]
use cust::error::CudaResult;
#[cfg(not(target_arch = "nvptx64"))]
use cust::launch;
#[cfg(not(target_arch = "nvptx64"))]
use cust::memory::{
    AsyncCopyDestination, CopyDestination, DeviceBuffer, DeviceCopy, DevicePointer,
};
#[cfg(not(target_arch = "nvptx64"))]
use cust::module::Module;
#[cfg(not(target_arch = "nvptx64"))]
use cust::stream::Stream;
#[cfg(not(target_arch = "nvptx64"))]
use std::mem::size_of;

/// Storage abstraction for managing device memory buffer of key-value pairs.
/// # Type Parameters
/// - `Key`: The key type (must be ≤ 8 bytes, bitwise comparable)
/// - `Value`: The value type (must be ≤ 8 bytes, bitwise comparable)
///
/// # Example
/// ```no_run
/// use cuda_static_map_kernels::storage::Storage;
/// use cuda_static_map_kernels::pair::Pair;
/// use cust::module::Module;
/// use cust::stream::Stream;
/// use cust::stream::StreamFlags;
///
/// // Initialize CUDA context (required for all CUDA operations)
/// let _ctx = cust::quick_init()?;
///
/// // Create storage with capacity 1000
/// let mut storage = Storage::<u32, u32>::new(1000)?;
///
/// // Initialize all slots with sentinel pair
/// let sentinel = Pair::new(0xFFFFFFFF, 0xFFFFFFFF);
///
/// // Option 1: Use host-to-device copy (simpler, works without module)
/// storage.initialize(sentinel, None)?;
///
/// // Option 2: Use kernel-based initialization (faster for large capacities)
/// // Load the PTX module compiled by build.rs
/// static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
/// let module = Module::from_ptx(PTX, &[])?;
/// storage.initialize(sentinel, Some(&module))?;
///
/// // For asynchronous initialization:
/// let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
/// unsafe {
///     storage.initialize_async(sentinel, &stream, Some(&module))?;
///     stream.synchronize()?;
/// }
/// ```
#[cfg(not(target_arch = "nvptx64"))]
pub struct Storage<Key, Value>
where
    (): crate::pair::AlignedTo<{ crate::pair::alignment::<Key, Value>() }>,
    Pair<Key, Value>: DeviceCopy,
{
    /// Device buffer storing the slot pairs
    buffer: DeviceBuffer<Pair<Key, Value>>,
    /// Capacity (number of slots)
    capacity: usize,
}

#[cfg(not(target_arch = "nvptx64"))]
impl<Key, Value> Storage<Key, Value>
where
    (): crate::pair::AlignedTo<{ crate::pair::alignment::<Key, Value>() }>,
    Pair<Key, Value>: DeviceCopy,
{
    /// Creates a new storage with the specified capacity.
    ///
    /// Allocates device memory for `capacity` slots but does not initialize them.
    /// Use `initialize()` or `initialize_async()` to fill slots with sentinel values.
    ///
    /// # Arguments
    /// * `capacity` - Number of slots to allocate
    ///
    /// # Errors
    /// Returns `CudaResult` if device memory allocation fails.
    ///
    /// # Example
    /// ```no_run
    /// let storage = Storage::<u32, u32>::new(1000)?;
    /// ```
    pub fn new(capacity: usize) -> CudaResult<Self> {
        // Allocate uninitialized device buffer
        let buffer = unsafe { DeviceBuffer::uninitialized(capacity)? };
        Ok(Self { buffer, capacity })
    }

    /// Creates a new storage with the specified capacity, allocating asynchronously on a stream.
    ///
    /// # Safety
    /// The allocated memory cannot be used until the stream operation completes.
    /// The stream must be synchronized before using the storage.
    ///
    /// # Arguments
    /// * `capacity` - Number of slots to allocate
    /// * `stream` - CUDA stream for asynchronous allocation
    ///
    /// # Errors
    /// Returns `CudaResult` if device memory allocation fails.
    pub unsafe fn new_async(capacity: usize, stream: &Stream) -> CudaResult<Self> {
        // Safety: Stream must be synchronized before using the buffer
        let buffer = unsafe { DeviceBuffer::uninitialized_async(capacity, stream)? };
        Ok(Self { buffer, capacity })
    }

    /// Gets a pointer to the device memory buffer.
    ///
    /// This returns a device pointer that can be passed to kernels.
    /// The pointer is valid for the lifetime of the `Storage` object.
    ///
    /// # Returns
    /// Device pointer to the first slot
    pub fn data(&self) -> DevicePointer<Pair<Key, Value>> {
        self.buffer.as_slice().as_device_ptr()
    }

    /// Gets the capacity (number of slots) of the storage.
    ///
    /// # Returns
    /// Number of slots in the storage
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Gets the length of the buffer (same as capacity for storage).
    ///
    /// # Returns
    /// Number of slots in the storage
    pub fn len(&self) -> usize {
        self.capacity
    }

    /// Checks if the storage is empty (capacity == 0).
    ///
    /// # Returns
    /// `true` if capacity is 0, `false` otherwise
    pub fn is_empty(&self) -> bool {
        self.capacity == 0
    }

    /// Initializes all slots in the storage with the given sentinel pair.
    ///
    /// This is a synchronous operation that fills all slots with the sentinel value.
    /// For asynchronous initialization, `initialize_async()` can be used.
    ///
    /// # Arguments
    /// * `sentinel` - The pair value to initialize all slots with
    /// * `module` - Optional CUDA module containing the initialization kernel.
    ///              If `None`, falls back to host-to-device copy.
    ///
    /// # Errors
    /// Returns `CudaResult` if the initialization fails.
    ///
    /// # Note
    /// If a module is provided, uses a GPU kernel for parallel initialization.
    /// Otherwise, data is copied from host to device.
    pub fn initialize(
        &mut self,
        sentinel: Pair<Key, Value>,
        module: Option<&Module>,
    ) -> CudaResult<()> {
        if self.capacity == 0 {
            return Ok(());
        }

        // Try kernel-based initialization if module is available
        if let Some(module) = module {
            let stream = Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None)?;
            unsafe {
                Self::initialize_async_kernel(self, sentinel, &stream, module)?;
            }
            stream.synchronize()?;
            return Ok(());
        }

        // Fallback to host-to-device copy
        let host_data: Vec<Pair<Key, Value>> = vec![sentinel; self.capacity];
        self.buffer.copy_from(&host_data)?;
        
        Ok(())
    }

    /// Asynchronously initializes all slots in the storage with the given sentinel pair.
    ///
    /// This is an asynchronous operation that fills all slots with the sentinel value on the given stream.
    /// The caller must synchronize the stream before using the storage.
    ///
    /// # Arguments
    /// * `sentinel` - The pair value to initialize all slots with
    /// * `stream` - CUDA stream for asynchronous initialization
    /// * `module` - Optional CUDA module containing the initialization kernel.
    ///              If `None`, falls back to host-to-device copy.
    ///
    /// # Errors
    /// Returns `CudaResult` if the initialization fails.
    ///
    /// # Safety
    /// The storage cannot be used until the stream operation completes.
    /// The caller must synchronize the stream before using the storage.
    pub unsafe fn initialize_async(
        &mut self,
        sentinel: Pair<Key, Value>,
        stream: &Stream,
        module: Option<&Module>,
    ) -> CudaResult<()> {
        if self.capacity == 0 {
            return Ok(());
        }

        // Try to use kernel-based initialization if module is available
        if let Some(module) = module {
            // Safety: initialize_async_kernel is unsafe but we're in an unsafe function
            return unsafe { Self::initialize_async_kernel(self, sentinel, stream, module) };
        }

        // Fallback to host-to-device copy
        let host_data: Vec<Pair<Key, Value>> = vec![sentinel; self.capacity];
        unsafe {
            self.buffer.async_copy_from(&host_data, stream)?;
        }
        
        Ok(())
    }

    /// Kernel-based initialization using the compiled PTX module.
    unsafe fn initialize_async_kernel(
        &mut self,
        sentinel: Pair<Key, Value>,
        stream: &Stream,
        module: &Module,
    ) -> CudaResult<()> {
        // Allocate device memory for sentinel value
        let sentinel_device = DeviceBuffer::from_slice(&[sentinel])?;
        let sentinel_ptr = sentinel_device.as_device_ptr();

        // Calculate launch configuration
        // For just initializing the storage, a block size of 128 is sufficient.
        // TODO: Tune this later if needed.
        const BLOCK_SIZE: u32 = 128;
        let grid_size = ((self.capacity as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);

        // Get the kernel function
        let init_kernel = module.get_function("initialize_storage_slots")?;

        // Launch the kernel
        // Parameters: slots, sentinel_bytes, slot_size, capacity
        // Safety: Parameters are valid.
        unsafe {
            launch!(init_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                self.buffer.as_device_ptr().as_raw() as *mut u8,
                sentinel_ptr.as_raw() as *const u8,
                size_of::<Pair<Key, Value>>(),
                self.capacity
            ))?;
        }

        Ok(())
    }

    /// Gets a reference to the underlying device buffer.
    ///
    /// Useful for advanced use cases where direct access to the buffer is needed.
    ///
    /// # Returns
    /// Reference to the device buffer
    pub fn as_buffer(&self) -> &DeviceBuffer<Pair<Key, Value>> {
        &self.buffer
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
            let dest = unsafe { slots.add(idx * slot_size) };
            unsafe {
                core::ptr::copy_nonoverlapping(sentinel_bytes, dest, slot_size);
            }
        }
    }
}

// Note: For MVP, no StorageRef type is implemented yet.
// Design will be evaluated again later.

