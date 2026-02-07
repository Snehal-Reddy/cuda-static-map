//! Shared static map implementation for both CPU and GPU
//!
//! This code compiles for both host (CPU) and device (GPU) targets.
//! When compiled for GPU (nvptx64-nvidia-cuda), it generates PTX code.
//! When compiled for CPU, it's used as a regular Rust library.

use crate::pair::{AlignedTo, Pair, alignment};

#[cfg(not(target_arch = "nvptx64"))]
use crate::open_addressing::OpenAddressingImpl;
#[cfg(not(target_arch = "nvptx64"))]
use crate::probing::ProbingScheme;
#[cfg(not(target_arch = "nvptx64"))]
use cust::error::CudaResult;
#[cfg(not(target_arch = "nvptx64"))]
use cust::launch;
#[cfg(not(target_arch = "nvptx64"))]
use cust::memory::{AsyncCopyDestination, DeviceBuffer};
#[cfg(not(target_arch = "nvptx64"))]
use cust::module::Module;
#[cfg(not(target_arch = "nvptx64"))]
use cust::stream::Stream;
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
///
/// **Bulk operations** (`insert`, `find`, `contains` that launch device kernels) are implemented
/// only for the canonical type (u64/u64/LinearProbing/DefaultKeyEqual/Device) with
/// `BUCKET_SIZE` in [`BULK_SUPPORTED_BUCKET_SIZES`] (1, 2, 4, 8).
pub struct StaticMap<
    Key,
    Value,
    Scheme,
    const BUCKET_SIZE: usize = 1,
    KeyEqual = crate::open_addressing::DefaultKeyEqual,
    const SCOPE: crate::open_addressing::ThreadScope = {
        crate::open_addressing::ThreadScope::Device
    },
> where
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
    ///
    /// The caller must synchronize `stream` before observing the cleared state or reusing the container.
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
    pub fn device_ref(
        &self,
    ) -> crate::static_map_ref::StaticMapRef<Key, Value, Scheme, BUCKET_SIZE, KeyEqual, SCOPE> {
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

/// Bulk device kernels and host impls are generated only for these sizes.
pub const BULK_SUPPORTED_BUCKET_SIZES: &[usize] = &[1, 2, 4, 8];

/// Generates one impl block per supported BUCKET_SIZE with bulk insert/find/contains
/// that dispatch to kernels named bulk_insert_n_bs{N}, bulk_find_n_bs{N}, bulk_contains_n_bs{N}.
#[cfg(not(target_arch = "nvptx64"))]
macro_rules! impl_bulk_ops_for_canonical_type {
    ($($bs:literal),+ $(,)?) => {
        $(
            impl StaticMap<
                u64,
                u64,
                crate::probing::LinearProbing<u64, crate::hash::IdentityHash<u64>>,
                $bs,
                crate::open_addressing::DefaultKeyEqual,
                { crate::open_addressing::ThreadScope::Device },
            >
            where
                (): AlignedTo<{ alignment::<u64, u64>() }>,
            {
                /// Bulk insert; launches device kernel and returns success count.
                pub fn insert(
                    &mut self,
                    pairs: &[Pair<u64, u64>],
                    stream: &Stream,
                    module: &Module,
                ) -> Result<usize, Box<dyn std::error::Error>> {
                    use crate::open_addressing::ThreadScope;
                    use crate::storage::CounterStorage;

                    if pairs.is_empty() {
                        return Ok(0);
                    }
                    let n = pairs.len();
                    let pairs_buf = DeviceBuffer::from_slice(pairs).map_err(|e| e.to_string())?;
                    // SAFETY: Counter is written before any read: `reset()` is called immediately.
                    // Same `stream` for reset, launch, and `load_to_host()`; `load_to_host()` synchronizes before return. `T` is `u64` (atomic-sized).
                    let mut counter =
                        unsafe { CounterStorage::<u64, { ThreadScope::Device }>::new(stream)? };
                    unsafe { counter.reset(stream)? };
                    let counter_ref = counter.storage_ref();
                    let container_ref = self.device_ref();
                    let cg_size = self.impl_.probing_scheme().cg_size();
                    const BLOCK_SIZE: u32 = 128;
                    let grid_size = grid_size_for_bulk(n, cg_size, BLOCK_SIZE);
                    let kernel_name = concat!("bulk_insert_n_bs", stringify!($bs));
                    let kernel = module.get_function(kernel_name).map_err(|e| e.to_string())?;
                    // SAFETY: Kernel pointer arguments valid for kernel duration: `pairs_buf` and `counter` owned, not dropped until after `load_to_host()`. `container_ref` from `&self`. Return only after `load_to_host()`.
                    unsafe {
                        launch!(kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                            pairs_buf.as_device_ptr().as_raw(),
                            n,
                            counter_ref,
                            container_ref
                        ))?;
                    }
                    let count = counter.load_to_host(stream).map_err(|e| e.to_string())?;
                    Ok(count as usize)
                }

                /// Bulk find; launches device kernel.
                ///
                /// # Safety
                ///
                /// `output` must refer to page-locked (pinned) host memory, as it is the destination of an asynchronous device-to-host copy.
                pub unsafe fn find(
                    &self,
                    keys: &[u64],
                    output: &mut [u64],
                    stream: &Stream,
                    module: &Module,
                ) -> Result<(), Box<dyn std::error::Error>> {
                    if keys.len() != output.len() {
                        return Err("keys and output length mismatch".into());
                    }
                    if keys.is_empty() {
                        return Ok(());
                    }
                    let n = keys.len();
                    let keys_buf = DeviceBuffer::from_slice(keys).map_err(|e| e.to_string())?;
                    let out_buf = unsafe { DeviceBuffer::uninitialized(n).map_err(|e| e.to_string())? };
                    let empty_value = self.empty_value_sentinel();
                    let container_ref = self.device_ref();
                    let cg_size = self.impl_.probing_scheme().cg_size();
                    const BLOCK_SIZE: u32 = 128;
                    let grid_size = grid_size_for_bulk(n, cg_size, BLOCK_SIZE);
                    let kernel_name = concat!("bulk_find_n_bs", stringify!($bs));
                    let kernel = module.get_function(kernel_name).map_err(|e| e.to_string())?;
                    // SAFETY: Kernel pointer arguments valid for kernel duration: `keys_buf` and `out_buf` owned, not dropped until `stream.synchronize()` below; `container_ref` from `&self`.
                    unsafe {
                        launch!(kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                            keys_buf.as_device_ptr().as_raw(),
                            n,
                            out_buf.as_device_ptr().as_raw(),
                            empty_value,
                            container_ref
                        ))?;
                    }
                    // SAFETY: Caller guarantees `output` is page-locked (see `# Safety`). Source and destination not deallocated, modified, or read until `stream.synchronize()` below.
                    unsafe { out_buf.async_copy_to(&mut output[..n], stream)? };
                    stream.synchronize().map_err(|e| e.to_string())?;
                    Ok(())
                }

                /// Bulk contains; launches device kernel.
                ///
                /// # Safety
                ///
                /// `output` must refer to page-locked (pinned) host memory, as it is the destination of an asynchronous device-to-host copy.
                pub unsafe fn contains(
                    &self,
                    keys: &[u64],
                    output: &mut [bool],
                    stream: &Stream,
                    module: &Module,
                ) -> Result<(), Box<dyn std::error::Error>> {
                    if keys.len() != output.len() {
                        return Err("keys and output length mismatch".into());
                    }
                    if keys.is_empty() {
                        return Ok(());
                    }
                    let n = keys.len();
                    let keys_buf = DeviceBuffer::from_slice(keys).map_err(|e| e.to_string())?;
                    let out_buf = unsafe { DeviceBuffer::uninitialized(n).map_err(|e| e.to_string())? };
                    let container_ref = self.device_ref();
                    let cg_size = self.impl_.probing_scheme().cg_size();
                    const BLOCK_SIZE: u32 = 128;
                    let grid_size = grid_size_for_bulk(n, cg_size, BLOCK_SIZE);
                    let kernel_name = concat!("bulk_contains_n_bs", stringify!($bs));
                    let kernel = module.get_function(kernel_name).map_err(|e| e.to_string())?;
                    // SAFETY: Kernel pointer arguments valid for kernel duration: `keys_buf` and `out_buf` owned, not dropped until `stream.synchronize()` below; `container_ref` from `&self`.
                    unsafe {
                        launch!(kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                            keys_buf.as_device_ptr().as_raw(),
                            n,
                            out_buf.as_device_ptr().as_raw(),
                            container_ref
                        ))?;
                    }
                    // SAFETY: Caller guarantees `output` is page-locked (see `# Safety`). Source and destination not deallocated, modified, or read until `stream.synchronize()` below.
                    unsafe { out_buf.async_copy_to(&mut output[..n], stream)? };
                    stream.synchronize().map_err(|e| e.to_string())?;
                    Ok(())
                }
            }
        )+
    };
}
#[cfg(not(target_arch = "nvptx64"))]
impl_bulk_ops_for_canonical_type!(1, 2, 4, 8);

/// Grid size for bulk kernels: ceil((num_keys * cg_size) / block_size).
#[cfg(not(target_arch = "nvptx64"))]
fn grid_size_for_bulk(num_keys: usize, cg_size: usize, block_size: u32) -> u32 {
    let block = block_size as usize;
    let g = (num_keys
        .saturating_mul(cg_size)
        .saturating_add(block.saturating_sub(1)))
        / block;
    (g.min(core::u32::MAX as usize) as u32).max(1)
}

// Device-side bulk kernels: one variant per supported BUCKET_SIZE (1, 2, 4, 8).
// Each module compiles to PTX symbols bulk_insert_n_bs{N}, bulk_find_n_bs{N}, bulk_contains_n_bs{N}.
#[cfg(target_arch = "nvptx64")]
macro_rules! bulk_device_kernels {
    (@inner $n:tt) => {
        paste::paste! {
            mod [<bulk_kernels_bs $n>] {
                use core::sync::atomic::Ordering;
                use cuda_std::prelude::*;
                use cuda_std::warp;

                use crate::hash::IdentityHash;
                use crate::open_addressing::{DefaultKeyEqual, ThreadScope};
                use crate::pair::Pair;
                use crate::probing::{LinearProbing, ProbingScheme};
                use crate::storage::CounterStorageRef;
                use crate::static_map_ref::StaticMapRef;

                type K = u64;
                type V = u64;
                type S = LinearProbing<K, IdentityHash<K>>;
                type Ref = StaticMapRef<K, V, S, $n, DefaultKeyEqual, { ThreadScope::Device }>;

                // Ref/CounterStorageRef are repr(C) and passed per NVVM kernel ABI.

                /// Bulk insert kernel.
                ///
                /// # Safety
                ///
                /// * `pairs` must be a valid device pointer to at least `num_pairs` elements.
                /// * `pairs` must be valid for reads for the duration of the kernel.
                /// * `counter_ref` and `container_ref` must be valid references (guaranteed by `impl_bulk_ops_for_canonical_type` setup).
                #[kernel]
                #[allow(improper_ctypes_definitions)]
                pub unsafe fn [<bulk_insert_n_bs $n>](
                    pairs: *const Pair<K, V>,
                    num_pairs: usize,
                    counter_ref: CounterStorageRef<u64, { ThreadScope::Device }>,
                    container_ref: Ref,
                ) {
                    let cg_size = container_ref.probing_scheme().cg_size();
                    let grid_stride = (thread::grid_dim_x() * thread::block_dim_x()) as usize;
                    let loop_stride = grid_stride / cg_size;
                    let mut idx = (thread::index_1d() as usize) / cg_size;

                    while idx < num_pairs {
                        // SAFETY:
                        // 1. We checked `idx < num_pairs` in the while loop condition.
                        // 2. The caller guarantees `pairs` is valid for `num_pairs` elements.
                        // 3. Therefore, `pairs.add(idx)` is within bounds and valid for read.
                        let pair = unsafe { *pairs.add(idx) };

                        if cg_size == 1 {
                            if container_ref.insert(pair) {
                                // Relaxed ordering: host reads count after stream sync; no cross-thread ordering required.
                                counter_ref.data().fetch_add(1, Ordering::Relaxed);
                            }
                        } else {
                            let lane_id = warp::lane_id();
                            let base_lane = (lane_id / cg_size as u32) * cg_size as u32;
                            let tile_mask = ((1u32 << cg_size) - 1) << base_lane;

                            // SAFETY:
                            // 1. `tile_mask` is constructed to partition the warp into `cg_size` groups.
                            // 2. We are in a converged control flow (all threads in the warp execute this path if `cg_size > 1`).
                            // 3. All threads in the mask are active.
                            if unsafe { container_ref.insert_cooperative(tile_mask, pair) } {
                                // Only the first thread in the group updates the counter
                                if (lane_id % cg_size as u32) == 0 {
                                    counter_ref.data().fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }

                        idx += loop_stride;
                    }
                }

                // Ref/CounterStorageRef are repr(C) and passed per NVVM kernel ABI.

                /// Bulk find kernel.
                ///
                /// # Safety
                ///
                /// * `keys` must be a valid device pointer to at least `num_pairs` elements.
                /// * `output` must be a valid device pointer to at least `num_pairs` elements.
                /// * Both pointers must be valid for the duration of the kernel.
                /// * `container_ref` must be a valid reference.
                #[kernel]
                #[allow(improper_ctypes_definitions)]
                pub unsafe fn [<bulk_find_n_bs $n>](
                    keys: *const K,
                    num_pairs: usize,
                    output: *mut V,
                    empty_value: V,
                    container_ref: Ref,
                ) {
                    let cg_size = container_ref.probing_scheme().cg_size();
                    let grid_stride = (thread::grid_dim_x() * thread::block_dim_x()) as usize;
                    let loop_stride = grid_stride / cg_size;
                    let mut idx = (thread::index_1d() as usize) / cg_size;

                    while idx < num_pairs {
                        // SAFETY:
                        // 1. We checked `idx < num_pairs`.
                        // 2. The caller guarantees `keys` is valid for `num_pairs` elements.
                        let key = unsafe { *keys.add(idx) };
                        let val;

                        if cg_size == 1 {
                            val = container_ref.find(&key);
                        } else {
                            let lane_id = warp::lane_id();
                            let base_lane = (lane_id / cg_size as u32) * cg_size as u32;
                            let tile_mask = ((1u32 << cg_size) - 1) << base_lane;

                            // SAFETY:
                            // 1. `tile_mask` partitions the warp correctly for `cg_size`.
                            // 2. All threads in the mask are active and converged.
                            val = unsafe { container_ref.find_cooperative(tile_mask, &key) };
                        }

                        if cg_size == 1 || (warp::lane_id() % cg_size as u32) == 0 {
                            // SAFETY:
                            // 1. We checked `idx < num_pairs`.
                            // 2. The caller guarantees `output` is valid for `num_pairs` elements.
                            // 3. We are writing to a unique index `idx` assigned to this thread (or group).
                            unsafe { *output.add(idx) = val.unwrap_or(empty_value) };
                        }

                        idx += loop_stride;
                    }
                }

                // Ref/CounterStorageRef are repr(C) and passed per NVVM kernel ABI.

                /// Bulk contains kernel.
                ///
                /// # Safety
                ///
                /// * `keys` must be a valid device pointer to at least `num_pairs` elements.
                /// * `output` must be a valid device pointer to at least `num_pairs` elements.
                /// * Both pointers must be valid for the duration of the kernel.
                /// * `container_ref` must be a valid reference.
                #[kernel]
                #[allow(improper_ctypes_definitions)]
                pub unsafe fn [<bulk_contains_n_bs $n>](
                    keys: *const K,
                    num_pairs: usize,
                    output: *mut bool,
                    container_ref: Ref,
                ) {
                    let cg_size = container_ref.probing_scheme().cg_size();
                    let grid_stride = (thread::grid_dim_x() * thread::block_dim_x()) as usize;
                    let loop_stride = grid_stride / cg_size;
                    let mut idx = (thread::index_1d() as usize) / cg_size;

                    while idx < num_pairs {
                        // SAFETY:
                        // 1. We checked `idx < num_pairs`.
                        // 2. The caller guarantees `keys` is valid for `num_pairs` elements.
                        let key = unsafe { *keys.add(idx) };
                        let res;

                        if cg_size == 1 {
                            res = container_ref.contains(&key);
                        } else {
                            let lane_id = warp::lane_id();
                            let base_lane = (lane_id / cg_size as u32) * cg_size as u32;
                            let tile_mask = ((1u32 << cg_size) - 1) << base_lane;

                            // SAFETY:
                            // 1. `tile_mask` partitions the warp correctly for `cg_size`.
                            // 2. All threads in the mask are active and converged.
                            res = unsafe { container_ref.contains_cooperative(tile_mask, &key) };
                        }

                        if cg_size == 1 || (warp::lane_id() % cg_size as u32) == 0 {
                            // SAFETY:
                            // 1. We checked `idx < num_pairs`.
                            // 2. The caller guarantees `output` is valid for `num_pairs` elements.
                            // 3. We are writing to a unique index `idx` assigned to this thread (or group).
                            unsafe { *output.add(idx) = res };
                        }

                        idx += loop_stride;
                    }
                }
            }
        }
    };
    ($($n:tt),+ $(,)?) => {
        $( bulk_device_kernels!(@inner $n); )+
    };
}
#[cfg(target_arch = "nvptx64")]
bulk_device_kernels!(1, 2, 4, 8);

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
