//! Device memory storage abstraction for static map
//!
//! This module provides a storage abstraction, which manages a device 
//! memory buffer storing `Pair<Key, Value>` slots.
//!
//! For MVP, bucket_size = 1 (flat storage) is used, which simplifies the implementation.
//! `DeviceBuffer` is used directly for simplicity. An allocator abstraction can be added later 
//! if needed.

use crate::pair::Pair;
use cust::error::CudaResult;
use cust::memory::{
    AsyncCopyDestination, CopyDestination, DeviceBuffer, DeviceCopy, DevicePointer,
};
use cust::stream::Stream;

/// Storage abstraction for managing device memory buffer of key-value pairs.
/// # Type Parameters
/// - `Key`: The key type (must be ≤ 8 bytes, bitwise comparable)
/// - `Value`: The value type (must be ≤ 8 bytes, bitwise comparable)
///
/// # Example
/// ```
/// use cuda_static_map_kernels::storage::Storage;
/// use cuda_static_map_kernels::pair::Pair;
///
/// // Create storage with capacity 1000
/// let storage = Storage::<u32, u32>::new(1000)?;
///
/// // Initialize all slots with sentinel pair
/// let sentinel = Pair::new(0xFFFFFFFF, 0xFFFFFFFF);
/// storage.initialize(sentinel)?;
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
    ///
    /// # Errors
    /// Returns `CudaResult` if the initialization fails.
    ///
    /// # Note
    /// Data is copied from host to device. For large capacities,
    /// `initialize_async()` may provide better performance.
    pub fn initialize(&mut self, sentinel: Pair<Key, Value>) -> CudaResult<()> {
        if self.capacity == 0 {
            return Ok(());
        }

        // Create a host vector filled with the sentinel value
        let host_data: Vec<Pair<Key, Value>> = vec![sentinel; self.capacity];
        
        // Copy from host to device (synchronous)
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
    ///
    /// # Errors
    /// Returns `CudaResult` if the initialization fails.
    ///
    /// # Safety
    /// The storage cannot be used until the stream operation completes.
    /// The caller must synchronize the stream before using the storage.
    /// The host data must remain valid until the copy completes.
    pub unsafe fn initialize_async(
        &mut self,
        sentinel: Pair<Key, Value>,
        stream: &Stream,
    ) -> CudaResult<()> {
        if self.capacity == 0 {
            return Ok(());
        }

        // Create a host vector filled with the sentinel value
        let host_data: Vec<Pair<Key, Value>> = vec![sentinel; self.capacity];
        
        // Copy from host to device (asynchronous)
        // Safety: The host_data is valid for the duration of the copy, and the stream
        // must be synchronized before using the storage.
        unsafe {
            self.buffer.async_copy_from(&host_data, stream)?;
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

// Note: For MVP, no StorageRef type is implemented yet.
// Design will be evaluated again later.

