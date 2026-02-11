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
    let idx = (thread::index_1d() as usize);
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
    let idx = (thread::index_1d() as usize);
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
    let idx = (thread::index_1d() as usize);
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
