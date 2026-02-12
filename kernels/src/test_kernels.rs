//! Test kernels for StaticMapRef
//! These are compiled only when `test-kernels` feature is enabled.

use crate::hash::IdentityHash;
use crate::open_addressing::{DefaultKeyEqual, ThreadScope};
use crate::pair::Pair;
use crate::probing::LinearProbing;
use crate::probing::ProbingScheme;
use crate::static_map_ref::StaticMapRef;
use cuda_std::prelude::*;
use cuda_std::warp;

type K = u64;
type V = u64;
type S = LinearProbing<K, IdentityHash<K>>;
type RefBs1 = StaticMapRef<K, V, S, 1, DefaultKeyEqual, { ThreadScope::Device }>;

/// Simple insert kernel for testing.
/// Each thread inserts one pair.
///
/// # Safety
/// Pointers must be valid.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_insert_bs1(
    pairs: *const Pair<K, V>,
    num_pairs: usize,
    out_results: *mut bool,
    container_ref: RefBs1,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_pairs {
        let pair = unsafe { *pairs.add(idx) };
        let result = container_ref.insert(pair);
        unsafe { *out_results.add(idx) = result };
    }
}

/// Simple find kernel for testing.
/// Each thread finds one key.
///
/// # Safety
/// Pointers must be valid.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_find_bs1(
    keys: *const K,
    num_keys: usize,
    out_values: *mut V,
    empty_val: V,
    container_ref: RefBs1,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_keys {
        let key = unsafe { *keys.add(idx) };
        let val = container_ref.find(&key).unwrap_or(empty_val);
        unsafe { *out_values.add(idx) = val };
    }
}

/// Simple contains kernel for testing.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_contains_bs1(
    keys: *const K,
    num_keys: usize,
    out_results: *mut bool,
    container_ref: RefBs1,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_keys {
        let key = unsafe { *keys.add(idx) };
        let result = container_ref.contains(&key);
        unsafe { *out_results.add(idx) = result };
    }
}

// Cooperative Group Test Kernels
type SCg2 = LinearProbing<K, IdentityHash<K>, 2>;
type RefBs1Cg2 = StaticMapRef<K, V, SCg2, 1, DefaultKeyEqual, { ThreadScope::Device }>;

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_cg_insert_bs1_cg2(
    pairs: *const Pair<K, V>,
    num_pairs: usize, // Number of logical insertions (groups)
    out_results: *mut bool,
    container_ref: RefBs1Cg2,
) {
    let cg_size = 2;
    let tid = thread::index_1d() as usize;
    let idx = tid / cg_size;
    let lane_id = warp::lane_id();
    let base_lane = (lane_id / cg_size as u32) * cg_size as u32;
    let tile_mask = ((1u32 << cg_size) - 1) << base_lane;

    if idx < num_pairs {
        let pair = unsafe { *pairs.add(idx) };

        let result = unsafe { container_ref.insert_cooperative(tile_mask, pair) };
        if (lane_id % cg_size as u32) == 0 {
            unsafe { *out_results.add(idx) = result };
        }
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_cg_find_bs1_cg2(
    keys: *const K,
    num_keys: usize,
    out_values: *mut V,
    empty_val: V,
    container_ref: RefBs1Cg2,
) {
    let cg_size = 2;
    let tid = thread::index_1d() as usize;
    let idx = tid / cg_size;
    let lane_id = warp::lane_id();
    let base_lane = (lane_id / cg_size as u32) * cg_size as u32;
    let tile_mask = ((1u32 << cg_size) - 1) << base_lane;

    if idx < num_keys {
        let key = unsafe { *keys.add(idx) };
        let val = unsafe { container_ref.find_cooperative(tile_mask, &key) };
        if (lane_id % cg_size as u32) == 0 {
            unsafe { *out_values.add(idx) = val.unwrap_or(empty_val) };
        }
    }
}

// Test kernels for different bucket sizes
type RefBs2 = StaticMapRef<K, V, S, 2, DefaultKeyEqual, { ThreadScope::Device }>;
type RefBs4 = StaticMapRef<K, V, S, 4, DefaultKeyEqual, { ThreadScope::Device }>;
type RefBs8 = StaticMapRef<K, V, S, 8, DefaultKeyEqual, { ThreadScope::Device }>;

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_insert_bs2(
    pairs: *const Pair<K, V>,
    num_pairs: usize,
    out_results: *mut bool,
    container_ref: RefBs2,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_pairs {
        let pair = unsafe { *pairs.add(idx) };
        let result = container_ref.insert(pair);
        unsafe { *out_results.add(idx) = result };
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_find_bs2(
    keys: *const K,
    num_keys: usize,
    out_values: *mut V,
    empty_val: V,
    container_ref: RefBs2,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_keys {
        let key = unsafe { *keys.add(idx) };
        let val = container_ref.find(&key).unwrap_or(empty_val);
        unsafe { *out_values.add(idx) = val };
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_insert_bs4(
    pairs: *const Pair<K, V>,
    num_pairs: usize,
    out_results: *mut bool,
    container_ref: RefBs4,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_pairs {
        let pair = unsafe { *pairs.add(idx) };
        let result = container_ref.insert(pair);
        unsafe { *out_results.add(idx) = result };
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_find_bs4(
    keys: *const K,
    num_keys: usize,
    out_values: *mut V,
    empty_val: V,
    container_ref: RefBs4,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_keys {
        let key = unsafe { *keys.add(idx) };
        let val = container_ref.find(&key).unwrap_or(empty_val);
        unsafe { *out_values.add(idx) = val };
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_insert_bs8(
    pairs: *const Pair<K, V>,
    num_pairs: usize,
    out_results: *mut bool,
    container_ref: RefBs8,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_pairs {
        let pair = unsafe { *pairs.add(idx) };
        let result = container_ref.insert(pair);
        unsafe { *out_results.add(idx) = result };
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_find_bs8(
    keys: *const K,
    num_keys: usize,
    out_values: *mut V,
    empty_val: V,
    container_ref: RefBs8,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_keys {
        let key = unsafe { *keys.add(idx) };
        let val = container_ref.find(&key).unwrap_or(empty_val);
        unsafe { *out_values.add(idx) = val };
    }
}

// Cooperative Group kernels for different CG sizes
type SCg4 = LinearProbing<K, IdentityHash<K>, 4>;
type SCg8 = LinearProbing<K, IdentityHash<K>, 8>;
type SCg16 = LinearProbing<K, IdentityHash<K>, 16>;
type SCg32 = LinearProbing<K, IdentityHash<K>, 32>;
type RefBs1Cg4 = StaticMapRef<K, V, SCg4, 1, DefaultKeyEqual, { ThreadScope::Device }>;
type RefBs1Cg8 = StaticMapRef<K, V, SCg8, 1, DefaultKeyEqual, { ThreadScope::Device }>;
type RefBs1Cg16 = StaticMapRef<K, V, SCg16, 1, DefaultKeyEqual, { ThreadScope::Device }>;
type RefBs1Cg32 = StaticMapRef<K, V, SCg32, 1, DefaultKeyEqual, { ThreadScope::Device }>;

macro_rules! cg_kernels {
    ($cg_size:literal, $cg_type:ty, $ref_type:ty, $insert_name:ident, $find_name:ident) => {
        #[kernel]
        #[allow(improper_ctypes_definitions)]
        pub unsafe fn $insert_name(
            pairs: *const Pair<K, V>,
            num_pairs: usize,
            out_results: *mut bool,
            container_ref: $ref_type,
        ) {
            let cg_size = $cg_size;
            let tid = thread::index_1d() as usize;
            let idx = tid / cg_size;
            let lane_id = warp::lane_id();
            let base_lane = (lane_id / cg_size as u32) * cg_size as u32;
            // For CG size 32, use 0xFFFFFFFF (all bits set), otherwise use bit shift
            let tile_mask = if cg_size == 32 {
                0xFFFFFFFFu32
            } else {
                ((1u32 << cg_size) - 1) << base_lane
            };

            if idx < num_pairs {
                let pair = unsafe { *pairs.add(idx) };
                let result = unsafe { container_ref.insert_cooperative(tile_mask, pair) };
                if (lane_id % cg_size as u32) == 0 {
                    unsafe { *out_results.add(idx) = result };
                }
            }
        }

        #[kernel]
        #[allow(improper_ctypes_definitions)]
        pub unsafe fn $find_name(
            keys: *const K,
            num_keys: usize,
            out_values: *mut V,
            empty_val: V,
            container_ref: $ref_type,
        ) {
            let cg_size = $cg_size;
            let tid = thread::index_1d() as usize;
            let idx = tid / cg_size;
            let lane_id = warp::lane_id();
            let base_lane = (lane_id / cg_size as u32) * cg_size as u32;
            // For CG size 32, use 0xFFFFFFFF (all bits set), otherwise use bit shift
            let tile_mask = if cg_size == 32 {
                0xFFFFFFFFu32
            } else {
                ((1u32 << cg_size) - 1) << base_lane
            };

            if idx < num_keys {
                let key = unsafe { *keys.add(idx) };
                let val = unsafe { container_ref.find_cooperative(tile_mask, &key) };
                if (lane_id % cg_size as u32) == 0 {
                    unsafe { *out_values.add(idx) = val.unwrap_or(empty_val) };
                }
            }
        }
    };
}

cg_kernels!(
    4,
    SCg4,
    RefBs1Cg4,
    test_cg_insert_bs1_cg4,
    test_cg_find_bs1_cg4
);
cg_kernels!(
    8,
    SCg8,
    RefBs1Cg8,
    test_cg_insert_bs1_cg8,
    test_cg_find_bs1_cg8
);
cg_kernels!(
    16,
    SCg16,
    RefBs1Cg16,
    test_cg_insert_bs1_cg16,
    test_cg_find_bs1_cg16
);
cg_kernels!(
    32,
    SCg32,
    RefBs1Cg32,
    test_cg_insert_bs1_cg32,
    test_cg_find_bs1_cg32
);

// Block-level test kernel - all threads in a block cooperate
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_block_insert_bs1(
    pairs: *const Pair<K, V>,
    num_pairs: usize,
    out_results: *mut bool,
    container_ref: RefBs1,
) {
    let block_idx = thread::block_idx_x() as usize;
    let thread_idx = thread::thread_idx_x() as usize;
    let block_dim = thread::block_dim_x() as usize;
    let idx = block_idx * block_dim + thread_idx;

    if idx < num_pairs {
        let pair = unsafe { *pairs.add(idx) };
        let result = container_ref.insert(pair);
        unsafe { *out_results.add(idx) = result };
    }
}

// Test kernel that verifies device ref fields
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_device_ref_fields(
    container_ref: RefBs1,
    out_capacity: *mut usize,
    out_empty_key: *mut K,
    out_empty_val: *mut V,
    out_erased_key: *mut K,
) {
    let idx = thread::index_1d() as usize;
    if idx == 0 {
        unsafe {
            *out_capacity = container_ref.capacity();
            *out_empty_key = container_ref.empty_key_sentinel();
            *out_empty_val = container_ref.empty_value_sentinel();
            *out_erased_key = container_ref.erased_key_sentinel();
        }
    }
}

// Test kernel that copies ref between functions (simulating passing ref between kernels)
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_ref_copying_kernel1(
    pairs: *const Pair<K, V>,
    num_pairs: usize,
    container_ref: RefBs1,
    out_ref_copy: *mut RefBs1,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_pairs {
        let pair = unsafe { *pairs.add(idx) };
        container_ref.insert(pair);
    }
    // Copy ref to output (simulating passing to another kernel)
    if idx == 0 {
        unsafe {
            *out_ref_copy = container_ref;
        }
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn test_ref_copying_kernel2(
    keys: *const K,
    num_keys: usize,
    out_values: *mut V,
    empty_val: V,
    container_ref: RefBs1,
) {
    let idx = thread::index_1d() as usize;
    if idx < num_keys {
        let key = unsafe { *keys.add(idx) };
        let val = container_ref.find(&key).unwrap_or(empty_val);
        unsafe { *out_values.add(idx) = val };
    }
}
