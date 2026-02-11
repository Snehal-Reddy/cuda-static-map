#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![allow(incomplete_features)]

use cuda_static_map::{StaticMap, get_ptx};
use cuda_static_map_kernels::hash::IdentityHash;
use cuda_static_map_kernels::open_addressing::{DefaultKeyEqual, ThreadScope};
use cuda_static_map_kernels::pair::{Pair};
use cuda_static_map_kernels::probing::LinearProbing;
use cust::prelude::*;
use std::error::Error;

// Helper to init CUDA
fn setup_cuda() -> Result<(Context, Stream), Box<dyn Error>> {
    cust::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let ctx = Context::new(device)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    Ok((ctx, stream))
}

#[test]
fn test_device_insert_find_bs1() -> Result<(), Box<dyn Error>> {
    let (_ctx, stream) = setup_cuda()?;
    let ptx = get_ptx();
    let module = Module::from_ptx(ptx, &[])?;

    // Create host map
    let capacity = 1024;
    let empty_key = u64::MAX;
    let empty_val = u64::MAX;
    let probing = LinearProbing::<u64, IdentityHash<u64>>::new(IdentityHash::new());
    let pred = DefaultKeyEqual;

    let mut map = StaticMap::<
        u64,
        u64,
        LinearProbing<u64, IdentityHash<u64>>,
        1,
        DefaultKeyEqual,
        { ThreadScope::Device },
    >::new(capacity, empty_key, empty_val, pred, probing, &stream)?;

    // Prepare input data
    let num_items = 100;
    let mut pairs = Vec::with_capacity(num_items);
    for i in 0..num_items {
        pairs.push(Pair::new(i as u64, (i * 10) as u64));
    }
    let pairs_buf = DeviceBuffer::from_slice(&pairs)?;

    // Output buffer for results (bool)
    let out_results = unsafe { DeviceBuffer::uninitialized(num_items)? };

    // Get kernel
    let kernel = module.get_function("test_insert_bs1")?;

    // Launch kernel
    let block_size = 128;
    let grid_size = (num_items + block_size - 1) / block_size;

    let map_ref = map.device_ref();

    unsafe {
        launch!(kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
            pairs_buf.as_device_ptr(),
            num_items,
            out_results.as_device_ptr(),
            map_ref
        ))?;
    }
    stream.synchronize()?;

    // Verify results on host
    let mut host_results = vec![false; num_items];
    out_results.copy_to(&mut host_results)?;

    for (i, &res) in host_results.iter().enumerate() {
        assert!(res, "Insert failed at index {}", i);
    }

    // Now test find
    let mut keys = Vec::with_capacity(num_items);
    for i in 0..num_items {
        keys.push(i as u64);
    }
    let keys_buf = DeviceBuffer::from_slice(&keys)?;
    let out_values = unsafe { DeviceBuffer::uninitialized(num_items)? };

    let kernel_find = module.get_function("test_find_bs1")?;

    unsafe {
        launch!(kernel_find<<<grid_size as u32, block_size as u32, 0, stream>>>(
            keys_buf.as_device_ptr(),
            num_items,
            out_values.as_device_ptr(),
            empty_val,
            map_ref
        ))?;
    }
    stream.synchronize()?;

    let mut host_values = vec![0u64; num_items];
    out_values.copy_to(&mut host_values)?;

    for (i, &val) in host_values.iter().enumerate() {
        assert_eq!(val, (i * 10) as u64, "Find mismatch at index {}", i);
    }

    // Test contains
    let out_contains = unsafe { DeviceBuffer::uninitialized(num_items)? };
    let kernel_contains = module.get_function("test_contains_bs1")?;

    unsafe {
        launch!(kernel_contains<<<grid_size as u32, block_size as u32, 0, stream>>>(
            keys_buf.as_device_ptr(),
            num_items,
            out_contains.as_device_ptr(),
            map_ref
        ))?;
    }
    stream.synchronize()?;

    let mut host_contains = vec![false; num_items];
    out_contains.copy_to(&mut host_contains)?;

    for (i, &res) in host_contains.iter().enumerate() {
        assert!(res, "Contains failed at index {}", i);
    }

    Ok(())
}

#[test]
fn test_cg_insert_find_bs1_cg2() -> Result<(), Box<dyn Error>> {
    let (_ctx, stream) = setup_cuda()?;
    let ptx = get_ptx();
    let module = Module::from_ptx(ptx, &[])?;

    // Create host map with CG=2
    let capacity = 1024;
    let empty_key = u64::MAX;
    let empty_val = u64::MAX;
    let probing = LinearProbing::<u64, IdentityHash<u64>, 2>::new(IdentityHash::new());
    let pred = DefaultKeyEqual;

    let mut map = StaticMap::<
        u64,
        u64,
        LinearProbing<u64, IdentityHash<u64>, 2>,
        1,
        DefaultKeyEqual,
        { ThreadScope::Device },
    >::new(capacity, empty_key, empty_val, pred, probing, &stream)?;

    // Prepare input data
    let num_items = 100;
    let mut pairs = Vec::with_capacity(num_items);
    for i in 0..num_items {
        pairs.push(Pair::new(i as u64, (i * 10) as u64));
    }
    let pairs_buf = DeviceBuffer::from_slice(&pairs)?;
    let out_results = unsafe { DeviceBuffer::uninitialized(num_items)? };

    let kernel = module.get_function("test_cg_insert_bs1_cg2")?;

    let cg_size = 2;
    let block_size = 128;
    // Total threads needed = num_items * cg_size
    let total_threads = num_items * cg_size;
    let grid_size = (total_threads + block_size - 1) / block_size;

    let map_ref = map.device_ref();

    unsafe {
        launch!(kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
            pairs_buf.as_device_ptr(),
            num_items,
            out_results.as_device_ptr(),
            map_ref
        ))?;
    }
    stream.synchronize()?;

    let mut host_results = vec![false; num_items];
    out_results.copy_to(&mut host_results)?;

    for (i, &res) in host_results.iter().enumerate() {
        assert!(res, "CG Insert failed at index {}", i);
    }

    // CG Find

    let mut keys = Vec::with_capacity(num_items);

    for i in 0..num_items {
        keys.push(i as u64);
    }

    let keys_buf = DeviceBuffer::from_slice(&keys)?;

    let out_values = unsafe { DeviceBuffer::uninitialized(num_items)? };

    let kernel_find = module.get_function("test_cg_find_bs1_cg2")?;

    unsafe {
        launch!(kernel_find<<<grid_size as u32, block_size as u32, 0, stream>>>(
            keys_buf.as_device_ptr(),
            num_items,
            out_values.as_device_ptr(),
            empty_val,
            map_ref
        ))?;
    }

    stream.synchronize()?;

    let mut host_values = vec![0u64; num_items];

    out_values.copy_to(&mut host_values)?;

    for (i, &val) in host_values.iter().enumerate() {
        assert_eq!(val, (i * 10) as u64, "CG Find mismatch at index {}", i);
    }

    Ok(())
}
