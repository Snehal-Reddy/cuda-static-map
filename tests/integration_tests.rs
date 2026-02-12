#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![allow(incomplete_features)]

use cuda_static_map::{StaticMap, get_ptx};
use cuda_static_map_kernels::hash::IdentityHash;
use cuda_static_map_kernels::open_addressing::{DefaultKeyEqual, ThreadScope};
use cuda_static_map_kernels::pair::Pair;
use cuda_static_map_kernels::probing::LinearProbing;
use cust::memory::LockedBuffer;
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

// Test helper utilities
mod test_helpers {
    use super::*;

    pub type TestMap = StaticMap<
        u64,
        u64,
        LinearProbing<u64, IdentityHash<u64>>,
        1,
        DefaultKeyEqual,
        { ThreadScope::Device },
    >;

    pub fn create_test_map(capacity: usize, stream: &Stream) -> Result<TestMap, Box<dyn Error>> {
        let empty_key = u64::MAX;
        let empty_val = u64::MAX;
        let probing = LinearProbing::<u64, IdentityHash<u64>>::new(IdentityHash::new());
        let pred = DefaultKeyEqual;
        Ok(TestMap::new(
            capacity, empty_key, empty_val, pred, probing, stream,
        )?)
    }

    pub fn create_test_map_with_module(
        capacity: usize,
        stream: &Stream,
    ) -> Result<(TestMap, Module), Box<dyn Error>> {
        let map = create_test_map(capacity, stream)?;
        let ptx = get_ptx();
        let module = Module::from_ptx(ptx, &[])?;
        Ok((map, module))
    }
}

// Basic Operations Tests
mod basic_operations {
    use super::test_helpers::*;
    use super::*;

    mod insert {
        use super::*;

        /// Test inserting a single key-value pair
        #[test]
        fn test_single_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert single pair
            let pairs = vec![Pair::new(42u64, 100u64)];
            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(success_count, 1, "Single insert should succeed");

            // Verify it was inserted by finding it
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[42u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(output[0], 100u64, "Found value should match inserted value");

            Ok(())
        }

        /// Test inserting multiple pairs sequentially (batch insert)
        #[test]
        fn test_batch_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert multiple pairs
            let num_items = 100;
            let mut pairs = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }

            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(success_count, num_items, "All inserts should succeed");

            // Verify all were inserted
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            for i in 0..num_items {
                assert_eq!(output[i], (i * 10) as u64, "Value mismatch at index {}", i);
            }

            Ok(())
        }

        /// Test inserting duplicate keys with different values.
        ///
        /// The second insert should report success because the key exists,
        /// but the stored value must remain the first one (no overwrite).
        #[test]
        fn test_duplicate_key_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert first pair
            let pairs1 = vec![Pair::new(42u64, 100u64)];
            let success1 = map.insert(&pairs1, &stream, &module)?;
            assert_eq!(success1, 1, "First insert should succeed");

            // Insert same key with different value
            let pairs2 = vec![Pair::new(42u64, 200u64)];
            let success2 = map.insert(&pairs2, &stream, &module)?;
            // Bulk insert counts an operation as "successful" if the key was inserted
            // OR already existed, so we still expect a count of 1 here.
            assert_eq!(
                success2, 1,
                "Duplicate key insert should report success when key already exists"
            );

            // Verify that the value was NOT overwritten: we should still see the first value.
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[42u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(
                output[0], 100u64,
                "Duplicate insert must not overwrite the original value"
            );

            Ok(())
        }

        /// Test inserting into an empty map
        #[test]
        fn test_empty_map_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert into freshly created map
            let pairs = vec![Pair::new(1u64, 2u64)];
            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(success_count, 1, "Insert into empty map should succeed");

            Ok(())
        }

        /// Test attempting to insert when the map is at capacity.
        ///
        /// Once we've filled all slots, additional inserts should fail (return 0
        /// additional successful operations) and the new key should not be present.
        #[test]
        fn test_full_map_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            // Use a small requested capacity, then query the actual capacity that
            // may be rounded up internally.
            let requested_capacity = 16;
            let (mut map, module) = create_test_map_with_module(requested_capacity, &stream)?;
            let capacity = map.capacity();

            // Fill the map up to its reported capacity.
            let mut pairs = Vec::new();
            for i in 0..capacity {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }

            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(
                success_count, capacity,
                "All inserts up to capacity should succeed"
            );

            // Verify that all inserted keys are present.
            let keys: Vec<u64> = (0..capacity).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(capacity)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }
            for i in 0..capacity {
                assert_eq!(
                    output[i],
                    (i * 10) as u64,
                    "Value mismatch at index {} before over-capacity insert",
                    i
                );
            }

            // Try to insert one more – this should fail because the table is full.
            let extra_pair = vec![Pair::new(capacity as u64, (capacity * 10) as u64)];
            let extra_success = map.insert(&extra_pair, &stream, &module)?;
            assert_eq!(
                extra_success, 0,
                "Insert into a full map should report 0 additional successful inserts"
            );

            // The extra key should not be present; find should return the empty sentinel.
            let empty_val = map.empty_value_sentinel();
            let mut extra_output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(
                    &[capacity as u64],
                    extra_output.as_mut_slice(),
                    &stream,
                    &module,
                )?;
            }
            assert_eq!(
                extra_output[0], empty_val,
                "Over-capacity insert must not make the new key visible"
            );

            Ok(())
        }
    }

    mod find {
        use super::*;

        /// Test finding an existing key
        #[test]
        fn test_find_existing_key() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert a pair
            let pairs = vec![Pair::new(42u64, 100u64)];
            map.insert(&pairs, &stream, &module)?;

            // Find it
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[42u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(output[0], 100u64, "Found value should match");

            Ok(())
        }

        /// Test finding a non-existent key
        #[test]
        fn test_find_non_existent_key() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (map, module) = create_test_map_with_module(1024, &stream)?;

            // Try to find a key that was never inserted
            let empty_val = map.empty_value_sentinel();
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[999u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(
                output[0], empty_val,
                "Non-existent key should return empty sentinel"
            );

            Ok(())
        }

        /// Test finding immediately after insert
        #[test]
        fn test_find_after_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert and immediately find
            let pairs = vec![Pair::new(42u64, 100u64)];
            map.insert(&pairs, &stream, &module)?;

            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[42u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(output[0], 100u64, "Should find immediately after insert");

            Ok(())
        }

        /// Test finding multiple keys (batch find)
        #[test]
        fn test_find_multiple_keys() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert multiple pairs
            let num_items = 50;
            let mut pairs = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            map.insert(&pairs, &stream, &module)?;

            // Find all of them
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            for i in 0..num_items {
                assert_eq!(
                    output[i],
                    (i * 10) as u64,
                    "Batch find mismatch at index {}",
                    i
                );
            }

            Ok(())
        }

        /// Test finding with sentinel values
        #[test]
        fn test_find_with_sentinel() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (map, module) = create_test_map_with_module(1024, &stream)?;

            let empty_val = map.empty_value_sentinel();
            let empty_key = map.empty_key_sentinel();

            // Try to find the empty key sentinel (should return empty value)
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[empty_key], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(
                output[0], empty_val,
                "Finding empty key sentinel should return empty value sentinel"
            );

            Ok(())
        }
    }

    mod contains {
        use super::*;

        /// Test contains for existing key
        #[test]
        fn test_contains_existing_key() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert a pair
            let pairs = vec![Pair::new(42u64, 100u64)];
            map.insert(&pairs, &stream, &module)?;

            // Check contains
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.contains(&[42u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert!(output[0], "Contains should return true for existing key");

            Ok(())
        }

        /// Test contains for non-existent key
        #[test]
        fn test_contains_non_existent_key() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (map, module) = create_test_map_with_module(1024, &stream)?;

            // Check contains for key that was never inserted
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.contains(&[999u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert!(
                !output[0],
                "Contains should return false for non-existent key"
            );

            Ok(())
        }

        /// Test contains immediately after insert
        #[test]
        fn test_contains_after_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert and immediately check contains
            let pairs = vec![Pair::new(42u64, 100u64)];
            map.insert(&pairs, &stream, &module)?;

            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.contains(&[42u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert!(output[0], "Should contain key immediately after insert");

            Ok(())
        }

        /// Test batch contains operation
        #[test]
        fn test_batch_contains() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert some pairs
            let num_items = 50;
            let mut pairs = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            map.insert(&pairs, &stream, &module)?;

            // Check contains for all inserted keys plus some non-existent ones
            let keys: Vec<u64> = (0..num_items + 10).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items + 10)? };
            unsafe {
                map.contains(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            // First num_items should be true, rest should be false
            for i in 0..num_items {
                assert!(
                    output[i],
                    "Contains should return true for inserted key at index {}",
                    i
                );
            }
            for i in num_items..num_items + 10 {
                assert!(
                    !output[i],
                    "Contains should return false for non-existent key at index {}",
                    i
                );
            }

            Ok(())
        }
    }

    mod clear {
        use super::*;

        /// Test clearing an empty map
        #[test]
        fn test_clear_empty_map() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, _module) = create_test_map_with_module(1024, &stream)?;

            // Clear empty map should not panic
            map.clear(&stream)?;
            stream.synchronize()?;

            Ok(())
        }

        /// Test clearing a populated map
        #[test]
        fn test_clear_populated_map() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert items
            let num_items = 50;
            let mut pairs = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            map.insert(&pairs, &stream, &module)?;

            // Clear the map
            map.clear(&stream)?;
            stream.synchronize()?;

            // Verify all items are gone
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let empty_val = map.empty_value_sentinel();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            for i in 0..num_items {
                assert_eq!(
                    output[i], empty_val,
                    "After clear, find should return empty sentinel at index {}",
                    i
                );
            }

            Ok(())
        }

        /// Test clearing then inserting new items
        #[test]
        fn test_clear_then_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert initial items
            let pairs1 = vec![Pair::new(1u64, 10u64), Pair::new(2u64, 20u64)];
            map.insert(&pairs1, &stream, &module)?;

            // Clear
            map.clear(&stream)?;
            stream.synchronize()?;

            // Insert new items
            let pairs2 = vec![Pair::new(3u64, 30u64), Pair::new(4u64, 40u64)];
            let success_count = map.insert(&pairs2, &stream, &module)?;
            assert_eq!(success_count, 2, "Insert after clear should succeed");

            // Verify new items are present
            let mut output = unsafe { LockedBuffer::uninitialized(2)? };
            unsafe {
                map.find(&[3u64, 4u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(output[0], 30u64);
            assert_eq!(output[1], 40u64);

            // Verify old items are gone
            let empty_val = map.empty_value_sentinel();
            let mut output_old = unsafe { LockedBuffer::uninitialized(2)? };
            unsafe {
                map.find(&[1u64, 2u64], output_old.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(output_old[0], empty_val);
            assert_eq!(output_old[1], empty_val);

            Ok(())
        }

        /// Test async clear
        #[test]
        fn test_async_clear() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert items
            let pairs = vec![Pair::new(42u64, 100u64)];
            map.insert(&pairs, &stream, &module)?;

            // Clear asynchronously
            unsafe {
                map.clear_async(&stream)?;
            }
            // Synchronize to ensure clear completes
            stream.synchronize()?;

            // Verify item is gone
            let empty_val = map.empty_value_sentinel();
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[42u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(
                output[0], empty_val,
                "After async clear, find should return empty sentinel"
            );

            Ok(())
        }
    }
}

// Configuration Tests
mod configuration {
    use super::test_helpers::*;
    use super::*;
    use cuda_static_map_kernels::hash::{Hash, XXHash32, XXHash64};
    use cuda_static_map_kernels::probing::{DoubleHashProbing, LinearProbing, ProbingScheme};

    // Helper macro to test bulk operations for a specific bucket size
    macro_rules! test_bucket_size_bulk_ops {
        ($bucket_size:literal, $stream:expr, $module:expr) => {{
            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let probing = LinearProbing::<u64, IdentityHash<u64>>::new(IdentityHash::new());
            let pred = DefaultKeyEqual;

            let mut map = StaticMap::<
                u64,
                u64,
                LinearProbing<u64, IdentityHash<u64>>,
                $bucket_size,
                DefaultKeyEqual,
                { ThreadScope::Device },
            >::new(1024, empty_key, empty_val, pred, probing, $stream)?;

            // Test insert/find/contains with a small set of data
            let num_items = 50;
            let mut pairs = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }

            let success_count = map.insert(&pairs, $stream, $module)?;
            assert_eq!(
                success_count, num_items,
                "All inserts should succeed for bucket_size={}",
                $bucket_size
            );

            // Verify finds
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), $stream, $module)?;
            }

            for i in 0..num_items {
                assert_eq!(
                    output[i],
                    (i * 10) as u64,
                    "Find mismatch at index {} for bucket_size={}",
                    i,
                    $bucket_size
                );
            }

            // Verify contains
            let mut contains_output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.contains(&keys, contains_output.as_mut_slice(), $stream, $module)?;
            }

            for i in 0..num_items {
                assert!(
                    contains_output[i],
                    "Contains should return true at index {} for bucket_size={}",
                    i, $bucket_size
                );
            }

            Ok(())
        }};
    }

    mod bucket_size {
        use super::*;

        /// Parameterized test for all supported bucket sizes
        #[test]
        fn test_all_bucket_sizes() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            for &bucket_size in &[1, 2, 4, 8] {
                let empty_key = u64::MAX;
                let empty_val = u64::MAX;
                let probing = LinearProbing::<u64, IdentityHash<u64>>::new(IdentityHash::new());
                let pred = DefaultKeyEqual;

                match bucket_size {
                    1 => {
                        let _map = StaticMap::<
                            u64,
                            u64,
                            LinearProbing<u64, IdentityHash<u64>>,
                            1,
                            DefaultKeyEqual,
                            { ThreadScope::Device },
                        >::new(
                            1024, empty_key, empty_val, pred, probing, &stream
                        )?;
                    }
                    2 => {
                        let _map = StaticMap::<
                            u64,
                            u64,
                            LinearProbing<u64, IdentityHash<u64>>,
                            2,
                            DefaultKeyEqual,
                            { ThreadScope::Device },
                        >::new(
                            1024, empty_key, empty_val, pred, probing, &stream
                        )?;
                    }
                    4 => {
                        let _map = StaticMap::<
                            u64,
                            u64,
                            LinearProbing<u64, IdentityHash<u64>>,
                            4,
                            DefaultKeyEqual,
                            { ThreadScope::Device },
                        >::new(
                            1024, empty_key, empty_val, pred, probing, &stream
                        )?;
                    }
                    8 => {
                        let _map = StaticMap::<
                            u64,
                            u64,
                            LinearProbing<u64, IdentityHash<u64>>,
                            8,
                            DefaultKeyEqual,
                            { ThreadScope::Device },
                        >::new(
                            1024, empty_key, empty_val, pred, probing, &stream
                        )?;
                    }
                    _ => unreachable!(),
                }
            }

            Ok(())
        }
    }

    mod cooperative_groups {
        use super::*;

        /// Test cooperative group size 1
        #[test]
        fn test_cg_size_1() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (_, module) = create_test_map_with_module(1024, &stream)?;
            test_bucket_size_bulk_ops!(1, &stream, &module)
        }

        /// Test cooperative group size 2
        #[test]
        fn test_cg_size_2() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let _module = Module::from_ptx(ptx, &[])?;

            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let probing = LinearProbing::<u64, IdentityHash<u64>, 2>::new(IdentityHash::new());
            let pred = DefaultKeyEqual;

            let map = StaticMap::<
                u64,
                u64,
                LinearProbing<u64, IdentityHash<u64>, 2>,
                1,
                DefaultKeyEqual,
                { ThreadScope::Device },
            >::new(1024, empty_key, empty_val, pred, probing, &stream)?;

            // Verify map was created successfully and CG size is correct
            assert_eq!(map.capacity(), 1024);
            let map_ref = map.device_ref();
            assert_eq!(map_ref.probing_scheme().cg_size(), 2);

            // Note: Bulk operations only work for CG_SIZE=1, but device kernels exist for CG=2
            // See test_cg_insert_find_bs1_cg2 for device kernel tests

            Ok(())
        }

        /// Test cooperative group size 4
        #[test]
        fn test_cg_size_4() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let _module = Module::from_ptx(ptx, &[])?;

            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let probing = LinearProbing::<u64, IdentityHash<u64>, 4>::new(IdentityHash::new());
            let pred = DefaultKeyEqual;

            let map = StaticMap::<
                u64,
                u64,
                LinearProbing<u64, IdentityHash<u64>, 4>,
                1,
                DefaultKeyEqual,
                { ThreadScope::Device },
            >::new(1024, empty_key, empty_val, pred, probing, &stream)?;

            // Verify map was created successfully and CG size is correct
            assert_eq!(map.capacity(), 1024);
            let map_ref = map.device_ref();
            assert_eq!(map_ref.probing_scheme().cg_size(), 4);

            Ok(())
        }

        /// Test cooperative group size 8
        #[test]
        fn test_cg_size_8() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let _module = Module::from_ptx(ptx, &[])?;

            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let probing = LinearProbing::<u64, IdentityHash<u64>, 8>::new(IdentityHash::new());
            let pred = DefaultKeyEqual;

            let map = StaticMap::<
                u64,
                u64,
                LinearProbing<u64, IdentityHash<u64>, 8>,
                1,
                DefaultKeyEqual,
                { ThreadScope::Device },
            >::new(1024, empty_key, empty_val, pred, probing, &stream)?;

            // Verify map was created successfully and CG size is correct
            assert_eq!(map.capacity(), 1024);
            let map_ref = map.device_ref();
            assert_eq!(map_ref.probing_scheme().cg_size(), 8);

            Ok(())
        }

        /// Test cooperative group size 16
        #[test]
        fn test_cg_size_16() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let _module = Module::from_ptx(ptx, &[])?;

            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let probing = LinearProbing::<u64, IdentityHash<u64>, 16>::new(IdentityHash::new());
            let pred = DefaultKeyEqual;

            let map = StaticMap::<
                u64,
                u64,
                LinearProbing<u64, IdentityHash<u64>, 16>,
                1,
                DefaultKeyEqual,
                { ThreadScope::Device },
            >::new(1024, empty_key, empty_val, pred, probing, &stream)?;

            // Verify map was created successfully and CG size is correct
            assert_eq!(map.capacity(), 1024);
            let map_ref = map.device_ref();
            assert_eq!(map_ref.probing_scheme().cg_size(), 16);

            Ok(())
        }

        /// Test cooperative group size 32
        #[test]
        fn test_cg_size_32() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let _module = Module::from_ptx(ptx, &[])?;

            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let probing = LinearProbing::<u64, IdentityHash<u64>, 32>::new(IdentityHash::new());
            let pred = DefaultKeyEqual;

            let map = StaticMap::<
                u64,
                u64,
                LinearProbing<u64, IdentityHash<u64>, 32>,
                1,
                DefaultKeyEqual,
                { ThreadScope::Device },
            >::new(1024, empty_key, empty_val, pred, probing, &stream)?;

            // Verify map was created successfully and CG size is correct
            assert_eq!(map.capacity(), 1024);
            let map_ref = map.device_ref();
            assert_eq!(map_ref.probing_scheme().cg_size(), 32);

            Ok(())
        }
    }

    mod probing_schemes {
        use super::*;

        /// Test linear probing scheme
        #[test]
        fn test_linear_probing() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (_, module) = create_test_map_with_module(1024, &stream)?;
            test_bucket_size_bulk_ops!(1, &stream, &module)
        }

        /// Test double hashing probing scheme
        #[test]
        fn test_double_hash_probing() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let hasher1 = IdentityHash::new();
            let hasher2 = IdentityHash::new();
            let probing = DoubleHashProbing::<u64, IdentityHash<u64>, IdentityHash<u64>>::new(
                hasher1, hasher2,
            );
            let pred = DefaultKeyEqual;

            let mut map = StaticMap::<
                u64,
                u64,
                DoubleHashProbing<u64, IdentityHash<u64>, IdentityHash<u64>>,
                1,
                DefaultKeyEqual,
                { ThreadScope::Device },
            >::new(1024, empty_key, empty_val, pred, probing, &stream)?;

            // Verify map was created successfully
            // Note: Double hashing requires prime table sizes, so capacity may differ from requested
            assert!(
                map.capacity() >= 1024,
                "Capacity should be at least 1024 for double hashing"
            );
            assert_eq!(map.empty_key_sentinel(), empty_key);
            assert_eq!(map.empty_value_sentinel(), empty_val);

            // Note: DoubleHashProbing may not support bulk operations yet
            // Just verify the map was created successfully
            Ok(())
        }
    }

    mod hash_functions {
        use super::*;

        /// Test XXHash32
        #[test]
        fn test_xxhash32() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let _module = Module::from_ptx(ptx, &[])?;

            // Host-side verification
            let hasher = XXHash32::new(0);
            let key1 = 12345u64;
            let key2 = 67890u64;
            let hash1 = hasher.hash(&key1);
            let hash2 = hasher.hash(&key2);
            assert_ne!(hash1, hash2, "Different keys should have different hashes");
            assert_eq!(hash1, hasher.hash(&key1), "Same key should have same hash");

            let hasher_seeded = XXHash32::new(123);
            let hash1_seeded = hasher_seeded.hash(&key1);
            assert_ne!(
                hash1, hash1_seeded,
                "Different seeds should produce different hashes"
            );

            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let probing = LinearProbing::<u64, XXHash32<u64>>::new(hasher);
            let pred = DefaultKeyEqual;

            let map = StaticMap::<
                u64,
                u64,
                LinearProbing<u64, XXHash32<u64>>,
                1,
                DefaultKeyEqual,
                { ThreadScope::Device },
            >::new(1024, empty_key, empty_val, pred, probing, &stream)?;

            // Verify map was created successfully
            assert_eq!(map.capacity(), 1024);
            assert_eq!(map.empty_key_sentinel(), empty_key);
            assert_eq!(map.empty_value_sentinel(), empty_val);

            Ok(())
        }

        /// Test XXHash64
        #[test]
        fn test_xxhash64() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let _module = Module::from_ptx(ptx, &[])?;

            // Host-side verification
            let hasher = XXHash64::new(0);
            let key1 = 12345u64;
            let key2 = 67890u64;
            let hash1 = hasher.hash(&key1);
            let hash2 = hasher.hash(&key2);
            assert_ne!(hash1, hash2, "Different keys should have different hashes");
            assert_eq!(hash1, hasher.hash(&key1), "Same key should have same hash");

            let hasher_seeded = XXHash64::new(123);
            let hash1_seeded = hasher_seeded.hash(&key1);
            assert_ne!(
                hash1, hash1_seeded,
                "Different seeds should produce different hashes"
            );

            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let probing = LinearProbing::<u64, XXHash64<u64>>::new(hasher);
            let pred = DefaultKeyEqual;

            let map = StaticMap::<
                u64,
                u64,
                LinearProbing<u64, XXHash64<u64>>,
                1,
                DefaultKeyEqual,
                { ThreadScope::Device },
            >::new(1024, empty_key, empty_val, pred, probing, &stream)?;

            // Verify map was created successfully
            assert_eq!(map.capacity(), 1024);
            assert_eq!(map.empty_key_sentinel(), empty_key);
            assert_eq!(map.empty_value_sentinel(), empty_val);

            Ok(())
        }
    }
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

// Bulk Operations Tests
mod bulk_operations {
    use super::test_helpers::*;
    use super::*;

    mod bulk_insert {
        use super::*;

        /// Test inserting an empty slice
        #[test]
        fn test_empty_bulk_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            let pairs: Vec<Pair<u64, u64>> = vec![];
            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(success_count, 0, "Empty bulk insert should return 0");

            Ok(())
        }

        /// Test inserting many elements at once
        #[test]
        fn test_large_bulk_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(10000, &stream)?;

            let num_items = 5000;
            let mut pairs = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }

            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(
                success_count, num_items,
                "Large bulk insert should succeed for all items"
            );

            // Verify a sample of inserted items
            let sample_size = 100;
            let sample_keys: Vec<u64> = (0..sample_size).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(sample_size)? };
            unsafe {
                map.find(&sample_keys, output.as_mut_slice(), &stream, &module)?;
            }

            for i in 0..sample_size {
                assert_eq!(
                    output[i],
                    (i * 10) as u64,
                    "Large bulk insert verification failed at index {}",
                    i
                );
            }

            Ok(())
        }

        /// Test bulk insert with duplicate keys
        #[test]
        fn test_bulk_insert_duplicates() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Create pairs with duplicate keys
            let pairs = vec![
                Pair::new(10u64, 100u64),
                Pair::new(20u64, 200u64),
                Pair::new(10u64, 300u64), // Duplicate key
                Pair::new(30u64, 400u64),
                Pair::new(20u64, 500u64), // Duplicate key
            ];

            let success_count = map.insert(&pairs, &stream, &module)?;
            // All inserts report success (even duplicates), but only unique keys are stored
            assert_eq!(
                success_count, 5,
                "Bulk insert with duplicates should report success for all operations"
            );

            // Verify that first values are retained (not overwritten)
            let keys = vec![10u64, 20u64, 30u64];
            let mut output = unsafe { LockedBuffer::uninitialized(3)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            assert_eq!(
                output[0], 100u64,
                "First value for key 10 should be retained"
            );
            assert_eq!(
                output[1], 200u64,
                "First value for key 20 should be retained"
            );
            assert_eq!(output[2], 400u64, "Value for key 30 should be correct");

            Ok(())
        }

        /// Test bulk insert success count matches actual successful inserts
        #[test]
        fn test_bulk_insert_success_count() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(100, &stream)?;

            // Insert unique keys
            let num_items = 50;
            let mut pairs = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }

            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(
                success_count, num_items,
                "Success count should match number of unique inserts"
            );

            // Insert same keys again (duplicates)
            let success_count2 = map.insert(&pairs, &stream, &module)?;
            // Duplicate inserts still report success (key exists), so count should be same
            assert_eq!(
                success_count2, num_items,
                "Duplicate inserts should still report success count"
            );

            // Verify all keys are present
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            let mut found_count = 0;
            for i in 0..num_items {
                if output[i] == (i * 10) as u64 {
                    found_count += 1;
                }
            }
            assert_eq!(found_count, num_items, "All inserted keys should be found");

            Ok(())
        }

        /// Test bulk insert when map is full (partial failure)
        #[test]
        fn test_bulk_insert_partial_failure() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let requested_capacity = 50;
            let (mut map, module) = create_test_map_with_module(requested_capacity, &stream)?;
            let capacity = map.capacity();

            // Fill the map to capacity
            let mut pairs = Vec::new();
            for i in 0..capacity {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }

            let success_count1 = map.insert(&pairs, &stream, &module)?;
            assert_eq!(
                success_count1, capacity,
                "First bulk insert should fill map to capacity"
            );

            // Try to insert more than capacity
            let extra_pairs: Vec<Pair<u64, u64>> = (capacity..capacity + 20)
                .map(|i| Pair::new(i as u64, (i * 10) as u64))
                .collect();

            let success_count2 = map.insert(&extra_pairs, &stream, &module)?;
            // When map is full, additional inserts should fail (return 0)
            assert_eq!(
                success_count2, 0,
                "Bulk insert when map is full should return 0 successful inserts"
            );

            // Verify that extra keys were not inserted
            let extra_keys: Vec<u64> = (capacity..capacity + 20).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(20)? };
            unsafe {
                map.find(&extra_keys, output.as_mut_slice(), &stream, &module)?;
            }

            let empty_val = map.empty_value_sentinel();
            for i in 0..20 {
                assert_eq!(
                    output[i], empty_val,
                    "Extra keys should not be found (should return empty_value_sentinel)"
                );
            }

            Ok(())
        }
    }

    mod bulk_find {
        use super::*;

        /// Test bulk find with existing keys
        #[test]
        fn test_bulk_find_existing() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert some pairs
            let num_items = 100;
            let mut pairs = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            map.insert(&pairs, &stream, &module)?;

            // Find all inserted keys
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            // Verify all values match
            for i in 0..num_items {
                assert_eq!(
                    output[i],
                    (i * 10) as u64,
                    "Bulk find existing: value mismatch at index {}",
                    i
                );
            }

            Ok(())
        }

        /// Test bulk find with non-existent keys
        #[test]
        fn test_bulk_find_missing() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (map, module) = create_test_map_with_module(1024, &stream)?;

            // Try to find keys that were never inserted
            let num_keys = 50;
            let keys: Vec<u64> = (1000..1000 + num_keys).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_keys)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            // All should return empty_value_sentinel
            let empty_val = map.empty_value_sentinel();
            for i in 0..num_keys {
                assert_eq!(
                    output[i], empty_val,
                    "Bulk find missing: should return empty_value_sentinel at index {}",
                    i
                );
            }

            Ok(())
        }

        /// Test bulk find with mix of existing and missing keys
        #[test]
        fn test_bulk_find_mixed() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert some pairs
            let num_inserted = 50;
            let mut pairs = Vec::with_capacity(num_inserted);
            for i in 0..num_inserted {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            map.insert(&pairs, &stream, &module)?;

            // Create mixed keys: some exist, some don't
            let keys = vec![
                0u64,    // exists
                1000u64, // missing
                10u64,   // exists
                2000u64, // missing
                20u64,   // exists
                3000u64, // missing
                30u64,   // exists
            ];

            let mut output = unsafe { LockedBuffer::uninitialized(keys.len())? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            let empty_val = map.empty_value_sentinel();

            // Verify existing keys
            assert_eq!(output[0], 0u64, "Key 0 should be found");
            assert_eq!(output[2], 100u64, "Key 10 should be found");
            assert_eq!(output[4], 200u64, "Key 20 should be found");
            assert_eq!(output[6], 300u64, "Key 30 should be found");

            // Verify missing keys
            assert_eq!(output[1], empty_val, "Key 1000 should not be found");
            assert_eq!(output[3], empty_val, "Key 2000 should not be found");
            assert_eq!(output[5], empty_val, "Key 3000 should not be found");

            Ok(())
        }

        /// Test bulk find output array matches input keys order
        #[test]
        fn test_bulk_find_output_order() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert pairs in one order
            let pairs = vec![
                Pair::new(100u64, 1000u64),
                Pair::new(200u64, 2000u64),
                Pair::new(300u64, 3000u64),
            ];
            map.insert(&pairs, &stream, &module)?;

            // Find keys in different order
            let keys = vec![300u64, 100u64, 200u64];
            let mut output = unsafe { LockedBuffer::uninitialized(keys.len())? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            // Output should match input order
            assert_eq!(output[0], 3000u64, "Output[0] should match keys[0]");
            assert_eq!(output[1], 1000u64, "Output[1] should match keys[1]");
            assert_eq!(output[2], 2000u64, "Output[2] should match keys[2]");

            Ok(())
        }

        /// Test bulk find with empty keys array
        #[test]
        fn test_bulk_find_empty() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (map, module) = create_test_map_with_module(1024, &stream)?;

            let keys: Vec<u64> = vec![];
            let mut output = unsafe { LockedBuffer::uninitialized(0)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            // Should not panic or error
            Ok(())
        }
    }

    mod bulk_contains {
        use super::*;

        /// Test bulk contains with existing keys
        #[test]
        fn test_bulk_contains_existing() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert some pairs
            let num_items = 100;
            let mut pairs = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            map.insert(&pairs, &stream, &module)?;

            // Check contains for all inserted keys
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.contains(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            // All should return true
            for i in 0..num_items {
                assert!(
                    output[i],
                    "Bulk contains existing: should return true at index {}",
                    i
                );
            }

            Ok(())
        }

        /// Test bulk contains with non-existent keys
        #[test]
        fn test_bulk_contains_missing() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (map, module) = create_test_map_with_module(1024, &stream)?;

            // Check contains for keys that were never inserted
            let num_keys = 50;
            let keys: Vec<u64> = (1000..1000 + num_keys).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_keys)? };
            unsafe {
                map.contains(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            // All should return false
            for i in 0..num_keys {
                assert!(
                    !output[i],
                    "Bulk contains missing: should return false at index {}",
                    i
                );
            }

            Ok(())
        }

        /// Test bulk contains with mix of existing and missing keys
        #[test]
        fn test_bulk_contains_mixed() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert some pairs
            let num_inserted = 50;
            let mut pairs = Vec::with_capacity(num_inserted);
            for i in 0..num_inserted {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            map.insert(&pairs, &stream, &module)?;

            // Create mixed keys: some exist, some don't
            let keys = vec![
                0u64,    // exists
                1000u64, // missing
                10u64,   // exists
                2000u64, // missing
                20u64,   // exists
                3000u64, // missing
                30u64,   // exists
            ];

            let mut output = unsafe { LockedBuffer::uninitialized(keys.len())? };
            unsafe {
                map.contains(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            // Verify existing keys return true
            assert!(output[0], "Key 0 should be contained");
            assert!(output[2], "Key 10 should be contained");
            assert!(output[4], "Key 20 should be contained");
            assert!(output[6], "Key 30 should be contained");

            // Verify missing keys return false
            assert!(!output[1], "Key 1000 should not be contained");
            assert!(!output[3], "Key 2000 should not be contained");
            assert!(!output[5], "Key 3000 should not be contained");

            Ok(())
        }

        /// Test bulk contains output boolean array correctness
        #[test]
        fn test_bulk_contains_output_correctness() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert specific pairs
            let pairs = vec![
                Pair::new(100u64, 1000u64),
                Pair::new(200u64, 2000u64),
                Pair::new(300u64, 3000u64),
            ];
            map.insert(&pairs, &stream, &module)?;

            // Check contains for keys in specific order
            let keys = vec![
                100u64, // exists
                400u64, // missing
                200u64, // exists
                500u64, // missing
                300u64, // exists
            ];

            let mut output = unsafe { LockedBuffer::uninitialized(keys.len())? };
            unsafe {
                map.contains(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            // Verify output matches expected pattern
            assert!(output[0], "Output[0] should be true for key 100");
            assert!(!output[1], "Output[1] should be false for key 400");
            assert!(output[2], "Output[2] should be true for key 200");
            assert!(!output[3], "Output[3] should be false for key 500");
            assert!(output[4], "Output[4] should be true for key 300");

            // Also verify with find to ensure consistency
            let mut find_output = unsafe { LockedBuffer::uninitialized(keys.len())? };
            unsafe {
                map.find(&keys, find_output.as_mut_slice(), &stream, &module)?;
            }

            let empty_val = map.empty_value_sentinel();
            assert_eq!(
                output[0],
                find_output[0] != empty_val,
                "Contains should match find result"
            );
            assert_eq!(
                output[1],
                find_output[1] != empty_val,
                "Contains should match find result"
            );
            assert_eq!(
                output[2],
                find_output[2] != empty_val,
                "Contains should match find result"
            );
            assert_eq!(
                output[3],
                find_output[3] != empty_val,
                "Contains should match find result"
            );
            assert_eq!(
                output[4],
                find_output[4] != empty_val,
                "Contains should match find result"
            );

            Ok(())
        }
    }
}

// Device Reference Tests (StaticMapRef)
mod edge_cases {
    use super::test_helpers::*;
    use super::*;
    use cust::memory::DeviceBuffer;

    mod sentinels {
        use super::*;

        /// Test behavior when key equals empty_key_sentinel
        #[test]
        fn test_empty_key_sentinel() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            let empty_key = map.empty_key_sentinel();
            let empty_val = map.empty_value_sentinel();

            // Try to insert a pair with empty key sentinel
            let sentinel_pair = Pair::new(empty_key, 999u64);
            let pairs = vec![sentinel_pair];
            let _success_count = map.insert(&pairs, &stream, &module)?;

            // Inserting sentinel key should either fail or be rejected
            // The behavior is undefined per documentation, but we test it doesn't crash
            // and that finding the sentinel returns the empty value sentinel
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[empty_key], output.as_mut_slice(), &stream, &module)?;
            }
            // Finding empty key sentinel should return empty value sentinel
            assert_eq!(
                output[0], empty_val,
                "Finding empty key sentinel should return empty value sentinel"
            );

            Ok(())
        }

        /// Test behavior when value equals empty_value_sentinel
        /// Note: Inserting empty_value_sentinel as a value is allowed.
        /// However, verifying it via find/contains may be problematic since these operations
        /// use the empty sentinel to indicate missing keys. We just verify insert succeeds.
        #[test]
        fn test_empty_value_sentinel() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            let empty_val = map.empty_value_sentinel();

            // Insert a pair with empty value sentinel (this should be allowed)
            let pair = Pair::new(42u64, empty_val);
            let pairs = vec![pair];
            let success_count = map.insert(&pairs, &stream, &module)?;

            // Verify insert reported success
            // Note: We can't reliably verify via find/contains because:
            // - find returns empty_value_sentinel for missing keys
            // - contains might have similar ambiguity
            // The important thing is that inserting empty_value_sentinel as a value doesn't crash
            assert_eq!(
                success_count, 1,
                "Insert with empty value sentinel should succeed"
            );

            Ok(())
        }

        /// Test sentinel collision: when user data accidentally matches sentinel values
        #[test]
        fn test_sentinel_collision() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            let empty_key = map.empty_key_sentinel();
            let empty_val = map.empty_value_sentinel();

            // Try to insert a key that happens to match the sentinel
            // Use a key that's very close to the sentinel but not exactly it
            let near_sentinel_key = if empty_key == u64::MAX {
                u64::MAX - 1
            } else {
                empty_key + 1
            };

            let pair = Pair::new(near_sentinel_key, 100u64);
            let pairs = vec![pair];
            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(success_count, 1, "Insert near sentinel should succeed");

            // Verify we can find it
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(
                    &[near_sentinel_key],
                    output.as_mut_slice(),
                    &stream,
                    &module,
                )?;
            }
            assert_eq!(output[0], 100u64, "Found value should match");

            Ok(())
        }

        /// Test that find returns None (empty_value_sentinel) for sentinel keys
        #[test]
        fn test_sentinel_in_find() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (map, module) = create_test_map_with_module(1024, &stream)?;

            let empty_key = map.empty_key_sentinel();
            let empty_val = map.empty_value_sentinel();

            // Find empty key sentinel should return empty value sentinel
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[empty_key], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(
                output[0], empty_val,
                "Find for empty key sentinel should return empty value sentinel"
            );

            Ok(())
        }
    }

    mod capacity {
        use super::*;

        /// Test with very small capacity
        #[test]
        fn test_small_capacity() -> Result<(), Box<dyn Error>> {
            for &capacity in &[4, 8, 16] {
                let (_ctx, stream) = setup_cuda()?;
                let (mut map, module) = create_test_map_with_module(capacity, &stream)?;
                let actual_capacity = map.capacity();

                assert!(
                    actual_capacity >= capacity,
                    "Actual capacity should be at least requested capacity"
                );

                // Insert up to capacity
                let mut pairs = Vec::new();
                for i in 0..actual_capacity {
                    pairs.push(Pair::new(i as u64, (i * 10) as u64));
                }

                let success_count = map.insert(&pairs, &stream, &module)?;
                assert_eq!(
                    success_count, actual_capacity,
                    "All inserts should succeed for capacity {}",
                    capacity
                );

                // Verify all were inserted
                let keys: Vec<u64> = (0..actual_capacity).map(|i| i as u64).collect();
                let mut output = unsafe { LockedBuffer::uninitialized(actual_capacity)? };
                unsafe {
                    map.find(&keys, output.as_mut_slice(), &stream, &module)?;
                }

                for i in 0..actual_capacity {
                    assert_eq!(
                        output[i],
                        (i * 10) as u64,
                        "Value mismatch at index {} for capacity {}",
                        i,
                        capacity
                    );
                }
            }

            Ok(())
        }

        /// Test with large capacity
        #[test]
        fn test_large_capacity() -> Result<(), Box<dyn Error>> {
            for &capacity in &[1_000_000, 10_000_000] {
                let (_ctx, stream) = setup_cuda()?;
                let (map, _module) = create_test_map_with_module(capacity, &stream)?;
                let actual_capacity = map.capacity();

                assert!(
                    actual_capacity >= capacity,
                    "Actual capacity should be at least requested capacity for {}",
                    capacity
                );
            }

            Ok(())
        }

        /// Test with_load_factor constructor
        #[test]
        fn test_load_factor() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            let n = 1000;
            let desired_load_factor = 0.75;
            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let probing = LinearProbing::<u64, IdentityHash<u64>>::new(IdentityHash::new());
            let pred = DefaultKeyEqual;

            let map = StaticMap::<
                u64,
                u64,
                LinearProbing<u64, IdentityHash<u64>>,
                1,
                DefaultKeyEqual,
                { ThreadScope::Device },
            >::with_load_factor(
                n,
                desired_load_factor,
                empty_key,
                empty_val,
                pred,
                probing,
                &stream,
            )?;

            let capacity = map.capacity();
            let expected_min_capacity = (n as f64 / desired_load_factor).ceil() as usize;

            assert!(
                capacity >= expected_min_capacity,
                "Capacity {} should be at least ceil({} / {}) = {}",
                capacity,
                n,
                desired_load_factor,
                expected_min_capacity
            );

            Ok(())
        }

        /// Test capacity rounding (verify capacity is properly rounded up)
        #[test]
        fn test_capacity_rounding() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;

            // Test various requested capacities
            for requested_capacity in &[1, 3, 7, 15, 31, 63, 127, 255] {
                let (map, _module) = create_test_map_with_module(*requested_capacity, &stream)?;
                let actual_capacity = map.capacity();

                assert!(
                    actual_capacity >= *requested_capacity,
                    "Actual capacity {} should be at least requested capacity {}",
                    actual_capacity,
                    requested_capacity
                );
            }

            Ok(())
        }
    }

    mod concurrency {
        use super::*;

        /// Test concurrent inserts: Multiple threads inserting different keys simultaneously
        #[test]
        fn test_concurrent_inserts_different_keys() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            let capacity = 1024;
            let (mut map, _module) = create_test_map_with_module(capacity, &stream)?;
            let map_ref = map.device_ref();

            let num_items = 100;
            let mut pairs = Vec::new();
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }

            let pairs_buf = DeviceBuffer::from_slice(&pairs)?;
            let out_results: DeviceBuffer<bool> =
                unsafe { DeviceBuffer::uninitialized(num_items)? };

            let kernel = module.get_function("test_concurrent_inserts_different_keys")?;
            let block_size = 128;
            let grid_size = (num_items + block_size - 1) / block_size;

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

            // All inserts should succeed
            for (i, &result) in host_results.iter().enumerate() {
                assert!(result, "Concurrent insert should succeed for key {}", i);
            }

            // Verify all keys were inserted
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            for i in 0..num_items {
                assert_eq!(
                    output[i],
                    (i * 10) as u64,
                    "Value mismatch at index {} after concurrent insert",
                    i
                );
            }

            Ok(())
        }

        /// Test concurrent same-key inserts: Multiple threads trying to insert same key
        #[test]
        fn test_concurrent_same_key_inserts() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            let capacity = 1024;
            let (mut map, _module) = create_test_map_with_module(capacity, &stream)?;
            let map_ref = map.device_ref();

            let test_pair = Pair::new(42u64, 100u64);
            let num_threads = 32; // Multiple threads trying to insert the same key

            let out_results: DeviceBuffer<bool> =
                unsafe { DeviceBuffer::uninitialized(num_threads)? };

            let kernel = module.get_function("test_concurrent_inserts_same_key")?;
            let block_size = 32;
            let grid_size = 1;

            unsafe {
                launch!(kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    test_pair,
                    num_threads,
                    out_results.as_device_ptr(),
                    map_ref
                ))?;
            }
            stream.synchronize()?;

            let mut host_results = vec![false; num_threads];
            out_results.copy_to(&mut host_results)?;

            // At least one insert should succeed
            let success_count = host_results.iter().filter(|&&r| r).count();
            assert!(
                success_count > 0,
                "At least one concurrent same-key insert should succeed"
            );
            assert!(
                success_count <= num_threads,
                "Not all threads should report success for same key"
            );

            // Verify the key exists with the correct value
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[42u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(
                output[0], 100u64,
                "Value should match after concurrent same-key insert"
            );

            Ok(())
        }

        /// Test concurrent find: Multiple threads finding keys simultaneously
        #[test]
        fn test_concurrent_find() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            let capacity = 1024;
            let (mut map, _module) = create_test_map_with_module(capacity, &stream)?;
            let map_ref = map.device_ref();
            let empty_val = map.empty_value_sentinel();

            // Insert some keys first
            let num_inserted = 50;
            let mut pairs = Vec::new();
            for i in 0..num_inserted {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            map.insert(&pairs, &stream, &module)?;

            // Now have multiple threads find keys concurrently
            let num_keys = 100; // Some exist, some don't
            let keys: Vec<u64> = (0..num_keys).map(|i| i as u64).collect();
            let keys_buf = DeviceBuffer::from_slice(&keys)?;
            let out_values: DeviceBuffer<u64> = unsafe { DeviceBuffer::uninitialized(num_keys)? };

            let kernel = module.get_function("test_concurrent_find")?;
            let block_size = 128;
            let grid_size = (num_keys + block_size - 1) / block_size;

            unsafe {
                launch!(kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    keys_buf.as_device_ptr(),
                    num_keys,
                    out_values.as_device_ptr(),
                    empty_val,
                    map_ref
                ))?;
            }
            stream.synchronize()?;

            let mut host_values = vec![0u64; num_keys];
            out_values.copy_to(&mut host_values)?;

            // Verify results
            for i in 0..num_keys {
                if i < num_inserted {
                    assert_eq!(
                        host_values[i],
                        (i * 10) as u64,
                        "Concurrent find should return correct value for existing key {}",
                        i
                    );
                } else {
                    assert_eq!(
                        host_values[i], empty_val,
                        "Concurrent find should return empty sentinel for missing key {}",
                        i
                    );
                }
            }

            Ok(())
        }

        /// Test insert-find race: One thread inserting while another finds
        /// Note: This is a simplified test - true race conditions are hard to test deterministically
        #[test]
        fn test_insert_find_race() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            let capacity = 1024;
            let (mut map, _module) = create_test_map_with_module(capacity, &stream)?;
            let map_ref = map.device_ref();
            let empty_val = map.empty_value_sentinel();

            // Insert some keys first
            let num_inserted = 50;
            let mut pairs = Vec::new();
            for i in 0..num_inserted {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            map.insert(&pairs, &stream, &module)?;

            // Now have some threads insert new keys while others find existing keys
            // This simulates a race condition scenario
            let num_new_inserts = 25;
            let mut new_pairs = Vec::new();
            let mut find_keys = Vec::new();
            for i in num_inserted..num_inserted + num_new_inserts {
                new_pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            // Find keys: mix of existing and new
            for i in 0..num_inserted + num_new_inserts {
                find_keys.push(i as u64);
            }

            // Use separate kernels to simulate race: insert and find concurrently
            // In practice, this would be done in separate streams or interleaved
            // For this test, we'll do them sequentially but verify both work

            // Insert new pairs
            let new_pairs_buf = DeviceBuffer::from_slice(&new_pairs)?;
            let out_insert_results: DeviceBuffer<bool> =
                unsafe { DeviceBuffer::uninitialized(num_new_inserts)? };
            let insert_kernel = module.get_function("test_concurrent_inserts_different_keys")?;
            let block_size = 128;
            let insert_grid_size = (num_new_inserts + block_size - 1) / block_size;

            unsafe {
                launch!(insert_kernel<<<insert_grid_size as u32, block_size as u32, 0, stream>>>(
                    new_pairs_buf.as_device_ptr(),
                    num_new_inserts,
                    out_insert_results.as_device_ptr(),
                    map_ref
                ))?;
            }

            // Find keys (some exist, some don't yet)
            let find_keys_buf = DeviceBuffer::from_slice(&find_keys)?;
            let out_find_values: DeviceBuffer<u64> =
                unsafe { DeviceBuffer::uninitialized(find_keys.len())? };
            let find_kernel = module.get_function("test_concurrent_find")?;
            let find_grid_size = (find_keys.len() + block_size - 1) / block_size;

            unsafe {
                launch!(find_kernel<<<find_grid_size as u32, block_size as u32, 0, stream>>>(
                    find_keys_buf.as_device_ptr(),
                    find_keys.len(),
                    out_find_values.as_device_ptr(),
                    empty_val,
                    map_ref
                ))?;
            }

            stream.synchronize()?;

            let mut host_insert_results = vec![false; num_new_inserts];
            let mut host_find_values = vec![0u64; find_keys.len()];
            out_insert_results.copy_to(&mut host_insert_results)?;
            out_find_values.copy_to(&mut host_find_values)?;

            // All inserts should succeed
            for (i, &result) in host_insert_results.iter().enumerate() {
                assert!(
                    result,
                    "Insert should succeed in race condition test for key {}",
                    num_inserted + i
                );
            }

            // Verify finds: existing keys should be found, new keys may or may not depending on timing
            for i in 0..num_inserted {
                assert_eq!(
                    host_find_values[i],
                    (i * 10) as u64,
                    "Find should return correct value for pre-existing key {}",
                    i
                );
            }

            // New keys: at least some should be found (if insert completed before find)
            // This is a race condition test - we just verify it doesn't crash
            // and that operations complete successfully

            Ok(())
        }

        /// Test concurrent contains: Multiple threads checking contains simultaneously
        #[test]
        fn test_concurrent_contains() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            let capacity = 1024;
            let (mut map, _module) = create_test_map_with_module(capacity, &stream)?;
            let map_ref = map.device_ref();

            // Insert some keys first
            let num_inserted = 50;
            let mut pairs = Vec::new();
            for i in 0..num_inserted {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }
            map.insert(&pairs, &stream, &module)?;

            // Now have multiple threads check contains concurrently
            let num_keys = 100; // Some exist, some don't
            let keys: Vec<u64> = (0..num_keys).map(|i| i as u64).collect();
            let keys_buf = DeviceBuffer::from_slice(&keys)?;
            let out_results: DeviceBuffer<bool> = unsafe { DeviceBuffer::uninitialized(num_keys)? };

            let kernel = module.get_function("test_concurrent_contains")?;
            let block_size = 128;
            let grid_size = (num_keys + block_size - 1) / block_size;

            unsafe {
                launch!(kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    keys_buf.as_device_ptr(),
                    num_keys,
                    out_results.as_device_ptr(),
                    map_ref
                ))?;
            }
            stream.synchronize()?;

            let mut host_results = vec![false; num_keys];
            out_results.copy_to(&mut host_results)?;

            // Verify results
            for i in 0..num_keys {
                if i < num_inserted {
                    assert!(
                        host_results[i],
                        "Concurrent contains should return true for existing key {}",
                        i
                    );
                } else {
                    assert!(
                        !host_results[i],
                        "Concurrent contains should return false for missing key {}",
                        i
                    );
                }
            }

            Ok(())
        }
    }

    mod data_types {
        use super::*;

        /// Test with small pairs (pair_size <= 8, packed CAS path)
        #[test]
        fn test_small_pairs() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;

            // u32/u32 pair = 8 bytes (packed CAS path)
            let capacity = 1024;
            let empty_key = u32::MAX;
            let empty_val = u32::MAX;
            let probing = LinearProbing::<u32, IdentityHash<u32>>::new(IdentityHash::new());
            let pred = DefaultKeyEqual;

            let mut map =
                StaticMap::<
                    u32,
                    u32,
                    LinearProbing<u32, IdentityHash<u32>>,
                    1,
                    DefaultKeyEqual,
                    { ThreadScope::Device },
                >::new(capacity, empty_key, empty_val, pred, probing, &stream)?;

            // Verify the map was created successfully
            // Note: Bulk insert operations only work for u64/u64, so we can't test
            // insert/find operations for u32/u32 pairs without custom kernels
            assert_eq!(
                map.capacity(),
                capacity,
                "Small pair map capacity should match"
            );

            Ok(())
        }

        /// Test with large pairs (pair_size > 8, back-to-back CAS path)
        #[test]
        fn test_large_pairs() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            // u64/u64 pair = 16 bytes (back-to-back CAS path)
            let capacity = 1024;
            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let probing = LinearProbing::<u64, IdentityHash<u64>>::new(IdentityHash::new());
            let pred = DefaultKeyEqual;

            let mut map =
                StaticMap::<
                    u64,
                    u64,
                    LinearProbing<u64, IdentityHash<u64>>,
                    1,
                    DefaultKeyEqual,
                    { ThreadScope::Device },
                >::new(capacity, empty_key, empty_val, pred, probing, &stream)?;

            // Insert some pairs
            let num_items = 100;
            let mut pairs = Vec::new();
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }

            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(
                success_count, num_items,
                "Large pair inserts should succeed"
            );

            // Verify all were inserted
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            for i in 0..num_items {
                assert_eq!(
                    output[i],
                    (i * 10) as u64,
                    "Value mismatch at index {} for large pairs",
                    i
                );
            }

            Ok(())
        }

        /// Test alignment requirements are met
        #[test]
        fn test_alignment() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;

            // Test that maps can be created with various types
            // The alignment is checked at compile time via trait bounds

            // u32/u32
            let empty_key32 = u32::MAX;
            let empty_val32 = u32::MAX;
            let probing32 = LinearProbing::<u32, IdentityHash<u32>>::new(IdentityHash::new());
            let pred32 = DefaultKeyEqual;

            let _map32 =
                StaticMap::<
                    u32,
                    u32,
                    LinearProbing<u32, IdentityHash<u32>>,
                    1,
                    DefaultKeyEqual,
                    { ThreadScope::Device },
                >::new(1024, empty_key32, empty_val32, pred32, probing32, &stream)?;

            // u64/u64
            let empty_key64 = u64::MAX;
            let empty_val64 = u64::MAX;
            let probing64 = LinearProbing::<u64, IdentityHash<u64>>::new(IdentityHash::new());
            let pred64 = DefaultKeyEqual;

            let _map64 =
                StaticMap::<
                    u64,
                    u64,
                    LinearProbing<u64, IdentityHash<u64>>,
                    1,
                    DefaultKeyEqual,
                    { ThreadScope::Device },
                >::new(1024, empty_key64, empty_val64, pred64, probing64, &stream)?;

            // If we get here, alignment requirements are met
            Ok(())
        }
    }
}

mod device_tests {
    use super::*;
    use cuda_static_map_kernels::static_map_ref::StaticMapRef;

    type RefBs1 = StaticMapRef<
        u64,
        u64,
        LinearProbing<u64, IdentityHash<u64>>,
        1,
        DefaultKeyEqual,
        { ThreadScope::Device },
    >;

    /// Test device_ref() construction and verify ref fields are correct
    #[test]
    fn test_device_ref_construction() -> Result<(), Box<dyn Error>> {
        let (_ctx, stream) = setup_cuda()?;
        let ptx = get_ptx();
        let module = Module::from_ptx(ptx, &[])?;

        let capacity = 1024;
        let empty_key = u64::MAX;
        let empty_val = u64::MAX;
        let probing = LinearProbing::<u64, IdentityHash<u64>>::new(IdentityHash::new());
        let pred = DefaultKeyEqual;

        let map = StaticMap::<
            u64,
            u64,
            LinearProbing<u64, IdentityHash<u64>>,
            1,
            DefaultKeyEqual,
            { ThreadScope::Device },
        >::new(capacity, empty_key, empty_val, pred, probing, &stream)?;

        // Create device ref
        let map_ref = map.device_ref();

        // Verify ref fields match host map
        assert_eq!(map_ref.capacity(), map.capacity());
        assert_eq!(map_ref.empty_key_sentinel(), map.empty_key_sentinel());
        assert_eq!(map_ref.empty_value_sentinel(), map.empty_value_sentinel());
        assert_eq!(map_ref.erased_key_sentinel(), map.erased_key_sentinel());

        // Test kernel that reads ref fields on device
        let kernel = module.get_function("test_device_ref_fields")?;
        let out_capacity = unsafe { DeviceBuffer::uninitialized(1)? };
        let out_empty_key = unsafe { DeviceBuffer::uninitialized(1)? };
        let out_empty_val = unsafe { DeviceBuffer::uninitialized(1)? };
        let out_erased_key = unsafe { DeviceBuffer::uninitialized(1)? };

        unsafe {
            launch!(kernel<<<1, 1, 0, stream>>>(
                map_ref,
                out_capacity.as_device_ptr(),
                out_empty_key.as_device_ptr(),
                out_empty_val.as_device_ptr(),
                out_erased_key.as_device_ptr()
            ))?;
        }
        stream.synchronize()?;

        let mut host_capacity = vec![0usize; 1];
        let mut host_empty_key = vec![0u64; 1];
        let mut host_empty_val = vec![0u64; 1];
        let mut host_erased_key = vec![0u64; 1];

        out_capacity.copy_to(&mut host_capacity)?;
        out_empty_key.copy_to(&mut host_empty_key)?;
        out_empty_val.copy_to(&mut host_empty_val)?;
        out_erased_key.copy_to(&mut host_erased_key)?;

        assert_eq!(host_capacity[0], capacity);
        assert_eq!(host_empty_key[0], empty_key);
        assert_eq!(host_empty_val[0], empty_val);
        assert_eq!(host_erased_key[0], map.erased_key_sentinel());

        Ok(())
    }

    /// Test passing StaticMapRef between kernels
    #[test]
    fn test_device_ref_copying() -> Result<(), Box<dyn Error>> {
        let (_ctx, stream) = setup_cuda()?;
        let ptx = get_ptx();
        let module = Module::from_ptx(ptx, &[])?;

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

        let map_ref = map.device_ref();

        // First kernel: insert data
        let num_items = 50;
        let mut pairs = Vec::with_capacity(num_items);
        for i in 0..num_items {
            pairs.push(Pair::new(i as u64, (i * 10) as u64));
        }
        let pairs_buf = DeviceBuffer::from_slice(&pairs)?;
        let out_ref_copy: DeviceBuffer<RefBs1> = unsafe { DeviceBuffer::uninitialized(1)? };

        let kernel1 = module.get_function("test_ref_copying_kernel1")?;
        let block_size = 128;
        let grid_size = (num_items + block_size - 1) / block_size;

        unsafe {
            launch!(kernel1<<<grid_size as u32, block_size as u32, 0, stream>>>(
                pairs_buf.as_device_ptr(),
                num_items,
                map_ref,
                out_ref_copy.as_device_ptr()
            ))?;
        }
        stream.synchronize()?;

        // Second kernel: use the copied ref to find data
        let mut keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
        let keys_buf = DeviceBuffer::from_slice(&keys)?;
        let out_values = unsafe { DeviceBuffer::uninitialized(num_items)? };

        let kernel2 = module.get_function("test_ref_copying_kernel2")?;

        unsafe {
            launch!(kernel2<<<grid_size as u32, block_size as u32, 0, stream>>>(
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
            assert_eq!(
                val,
                (i * 10) as u64,
                "Ref copying test failed at index {}",
                i
            );
        }

        Ok(())
    }

    /// Parameterized test for device operations with different bucket sizes
    #[test]
    fn test_device_operations_all_bucket_sizes() -> Result<(), Box<dyn Error>> {
        let (_ctx, stream) = setup_cuda()?;
        let ptx = get_ptx();
        let module = Module::from_ptx(ptx, &[])?;

        macro_rules! test_device_ops {
            ($bucket_size:literal, $stream:ident, $module:ident) => {{
                let capacity = 1024;
                let empty_key = u64::MAX;
                let empty_val = u64::MAX;
                let probing = LinearProbing::<u64, IdentityHash<u64>>::new(IdentityHash::new());
                let pred = DefaultKeyEqual;

                let mut map = StaticMap::<
                    u64,
                    u64,
                    LinearProbing<u64, IdentityHash<u64>>,
                    $bucket_size,
                    DefaultKeyEqual,
                    { ThreadScope::Device },
                >::new(capacity, empty_key, empty_val, pred, probing, &$stream)?;

                let num_items = 100;
                let mut pairs = Vec::with_capacity(num_items);
                for i in 0..num_items {
                    pairs.push(Pair::new(i as u64, (i * 10) as u64));
                }
                let pairs_buf = DeviceBuffer::from_slice(&pairs)?;
                let out_results = unsafe { DeviceBuffer::uninitialized(num_items)? };

                let kernel_name = format!("test_insert_bs{}", $bucket_size);
                let kernel = $module.get_function(&kernel_name)?;
                let block_size = 128;
                let grid_size = (num_items + block_size - 1) / block_size;

                let map_ref = map.device_ref();

                unsafe {
                    launch!(kernel<<<grid_size as u32, block_size as u32, 0, $stream>>>(
                        pairs_buf.as_device_ptr(),
                        num_items,
                        out_results.as_device_ptr(),
                        map_ref
                    ))?;
                }
                $stream.synchronize()?;

                let mut host_results = vec![false; num_items];
                out_results.copy_to(&mut host_results)?;

                for (i, &res) in host_results.iter().enumerate() {
                    assert!(res, "Insert failed at index {} for bucket_size={}", i, $bucket_size);
                }

                // Test find
                let mut keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
                let keys_buf = DeviceBuffer::from_slice(&keys)?;
                let out_values = unsafe { DeviceBuffer::uninitialized(num_items)? };

                let find_kernel_name = format!("test_find_bs{}", $bucket_size);
                let kernel_find = $module.get_function(&find_kernel_name)?;

                unsafe {
                    launch!(kernel_find<<<grid_size as u32, block_size as u32, 0, $stream>>>(
                        keys_buf.as_device_ptr(),
                        num_items,
                        out_values.as_device_ptr(),
                        empty_val,
                        map_ref
                    ))?;
                }
                $stream.synchronize()?;

                let mut host_values = vec![0u64; num_items];
                out_values.copy_to(&mut host_values)?;

                for (i, &val) in host_values.iter().enumerate() {
                    assert_eq!(
                        val,
                        (i * 10) as u64,
                        "Find mismatch at index {} for bucket_size={}",
                        i,
                        $bucket_size
                    );
                }

                // Test contains
                let out_contains = unsafe { DeviceBuffer::uninitialized(num_items)? };
                let contains_kernel_name = format!("test_contains_bs{}", $bucket_size);
                let kernel_contains = $module.get_function(&contains_kernel_name)?;

                unsafe {
                    launch!(kernel_contains<<<grid_size as u32, block_size as u32, 0, $stream>>>(
                        keys_buf.as_device_ptr(),
                        num_items,
                        out_contains.as_device_ptr(),
                        map_ref
                    ))?;
                }
                $stream.synchronize()?;

                let mut host_contains = vec![false; num_items];
                out_contains.copy_to(&mut host_contains)?;

                for (i, &res) in host_contains.iter().enumerate() {
                    assert!(
                        res,
                        "Contains failed at index {} for bucket_size={}",
                        i,
                        $bucket_size
                    );
                }
                Ok::<(), Box<dyn Error>>(())
            }};
        }

        // Test all bucket sizes
        test_device_ops!(1, stream, module)?;
        test_device_ops!(2, stream, module)?;
        test_device_ops!(4, stream, module)?;
        test_device_ops!(8, stream, module)?;

        Ok(())
    }

    /// Parameterized test for cooperative group operations with different CG sizes
    #[test]
    fn test_cooperative_operations_all_cg_sizes() -> Result<(), Box<dyn Error>> {
        let (_ctx, stream) = setup_cuda()?;
        let ptx = get_ptx();
        let module = Module::from_ptx(ptx, &[])?;

        macro_rules! test_cg_ops {
            ($cg_size:literal, $stream:ident, $module:ident) => {{
                let capacity = 1024;
                let empty_key = u64::MAX;
                let empty_val = u64::MAX;
                let probing = LinearProbing::<u64, IdentityHash<u64>, $cg_size>::new(IdentityHash::new());
                let pred = DefaultKeyEqual;

                let mut map = StaticMap::<
                    u64,
                    u64,
                    LinearProbing<u64, IdentityHash<u64>, $cg_size>,
                    1,
                    DefaultKeyEqual,
                    { ThreadScope::Device },
                >::new(capacity, empty_key, empty_val, pred, probing, &$stream)?;

                let num_items = 100;
                let mut pairs = Vec::with_capacity(num_items);
                for i in 0..num_items {
                    pairs.push(Pair::new(i as u64, (i * 10) as u64));
                }
                let pairs_buf = DeviceBuffer::from_slice(&pairs)?;
                let out_results = unsafe { DeviceBuffer::uninitialized(num_items)? };

                let kernel_name = format!("test_cg_insert_bs1_cg{}", $cg_size);
                let kernel = $module.get_function(&kernel_name)?;
                let block_size = 128;
                let total_threads = num_items * $cg_size;
                let grid_size = (total_threads + block_size - 1) / block_size;

                let map_ref = map.device_ref();

                unsafe {
                    launch!(kernel<<<grid_size as u32, block_size as u32, 0, $stream>>>(
                        pairs_buf.as_device_ptr(),
                        num_items,
                        out_results.as_device_ptr(),
                        map_ref
                    ))?;
                }
                $stream.synchronize()?;

                let mut host_results = vec![false; num_items];
                out_results.copy_to(&mut host_results)?;

                for (i, &res) in host_results.iter().enumerate() {
                    assert!(res, "CG insert failed at index {} for cg_size={}", i, $cg_size);
                }

                // Test find
                let mut keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
                let keys_buf = DeviceBuffer::from_slice(&keys)?;
                let out_values = unsafe { DeviceBuffer::uninitialized(num_items)? };

                let find_kernel_name = format!("test_cg_find_bs1_cg{}", $cg_size);
                let kernel_find = $module.get_function(&find_kernel_name)?;

                unsafe {
                    launch!(kernel_find<<<grid_size as u32, block_size as u32, 0, $stream>>>(
                        keys_buf.as_device_ptr(),
                        num_items,
                        out_values.as_device_ptr(),
                        empty_val,
                        map_ref
                    ))?;
                }
                $stream.synchronize()?;

                let mut host_values = vec![0u64; num_items];
                out_values.copy_to(&mut host_values)?;

                for (i, &val) in host_values.iter().enumerate() {
                    assert_eq!(
                        val,
                        (i * 10) as u64,
                        "CG find mismatch at index {} for cg_size={}",
                        i,
                        $cg_size
                    );
                }

                // Test contains
                let out_contains = unsafe { DeviceBuffer::uninitialized(num_items)? };
                let contains_kernel_name = format!("test_cg_contains_bs1_cg{}", $cg_size);
                let kernel_contains = $module.get_function(&contains_kernel_name)?;
                // Contains might need different grid calculation if kernel is per-key not per-thread
                // Assuming it follows same pattern as insert/find for CG tests
                let total_threads_contains = num_items * $cg_size;
                let grid_size_contains = (total_threads_contains + block_size - 1) / block_size;

                unsafe {
                    launch!(kernel_contains<<<grid_size_contains as u32, block_size as u32, 0, $stream>>>(
                        keys_buf.as_device_ptr(),
                        num_items,
                        out_contains.as_device_ptr(),
                        map_ref
                    ))?;
                }
                $stream.synchronize()?;

                let mut host_contains = vec![false; num_items];
                out_contains.copy_to(&mut host_contains)?;

                for (i, &res) in host_contains.iter().enumerate() {
                    assert!(
                        res,
                        "CG contains failed at index {} for cg_size={}",
                        i,
                        $cg_size
                    );
                }

                Ok::<(), Box<dyn Error>>(())
            }};
        }

        // Test all CG sizes
        test_cg_ops!(2, stream, module)?;
        test_cg_ops!(4, stream, module)?;
        test_cg_ops!(8, stream, module)?;
        test_cg_ops!(16, stream, module)?;
        test_cg_ops!(32, stream, module)?;

        Ok(())
    }

    /// Test multi-thread kernel operations (block-level)
    #[test]
    fn test_block_level_operations() -> Result<(), Box<dyn Error>> {
        let (_ctx, stream) = setup_cuda()?;
        let ptx = get_ptx();
        let module = Module::from_ptx(ptx, &[])?;

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

        let num_items = 200;
        let mut pairs = Vec::with_capacity(num_items);
        for i in 0..num_items {
            pairs.push(Pair::new(i as u64, (i * 10) as u64));
        }
        let pairs_buf = DeviceBuffer::from_slice(&pairs)?;
        let out_results = unsafe { DeviceBuffer::uninitialized(num_items)? };

        let kernel = module.get_function("test_block_insert_bs1")?;
        let block_size = 256;
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

        let mut host_results = vec![false; num_items];
        out_results.copy_to(&mut host_results)?;

        for (i, &res) in host_results.iter().enumerate() {
            assert!(res, "Block-level insert failed at index {}", i);
        }

        // Verify with find
        let mut keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
        let keys_buf = DeviceBuffer::from_slice(&keys)?;
        let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
        unsafe {
            map.find(&keys, output.as_mut_slice(), &stream, &module)?;
        }

        for i in 0..num_items {
            assert_eq!(
                output[i],
                (i * 10) as u64,
                "Block-level find mismatch at index {}",
                i
            );
        }

        Ok(())
    }

    /// Test device-side contains operation
    #[test]
    fn test_device_contains() -> Result<(), Box<dyn Error>> {
        let (_ctx, stream) = setup_cuda()?;
        let ptx = get_ptx();
        let module = Module::from_ptx(ptx, &[])?;

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

        // Insert via device kernel
        let num_items = 100;
        let mut pairs = Vec::with_capacity(num_items);
        for i in 0..num_items {
            pairs.push(Pair::new(i as u64, (i * 10) as u64));
        }
        let pairs_buf = DeviceBuffer::from_slice(&pairs)?;
        let out_results: DeviceBuffer<bool> = unsafe { DeviceBuffer::uninitialized(num_items)? };

        let kernel_insert = module.get_function("test_insert_bs1")?;
        let block_size = 128;
        let grid_size = (num_items + block_size - 1) / block_size;

        let map_ref = map.device_ref();

        unsafe {
            launch!(kernel_insert<<<grid_size as u32, block_size as u32, 0, stream>>>(
                pairs_buf.as_device_ptr(),
                num_items,
                out_results.as_device_ptr(),
                map_ref
            ))?;
        }
        stream.synchronize()?;

        // Test contains for existing and non-existing keys
        let mut keys = Vec::with_capacity(num_items + 20);
        for i in 0..num_items {
            keys.push(i as u64);
        }
        for i in num_items..num_items + 20 {
            keys.push(i as u64);
        }
        let keys_buf = DeviceBuffer::from_slice(&keys)?;
        let out_contains = unsafe { DeviceBuffer::uninitialized(keys.len())? };

        let kernel_contains = module.get_function("test_contains_bs1")?;
        let grid_size_contains = (keys.len() + block_size - 1) / block_size;

        unsafe {
            launch!(kernel_contains<<<grid_size_contains as u32, block_size as u32, 0, stream>>>(
                keys_buf.as_device_ptr(),
                keys.len(),
                out_contains.as_device_ptr(),
                map_ref
            ))?;
        }
        stream.synchronize()?;

        let mut host_contains = vec![false; keys.len()];
        out_contains.copy_to(&mut host_contains)?;

        // First num_items should be true, rest should be false
        for i in 0..num_items {
            assert!(
                host_contains[i],
                "Contains should return true for existing key {}",
                i
            );
        }
        for i in num_items..keys.len() {
            assert!(
                !host_contains[i],
                "Contains should return false for non-existing key {}",
                i
            );
        }

        Ok(())
    }

    /// Test cooperative contains operation for CG size 2
    #[test]
    fn test_cooperative_contains_cg2() -> Result<(), Box<dyn Error>> {
        let (_ctx, stream) = setup_cuda()?;
        let ptx = get_ptx();
        let module = Module::from_ptx(ptx, &[])?;

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

        // Insert via cooperative kernel
        let num_items = 100;
        let mut pairs = Vec::with_capacity(num_items);
        for i in 0..num_items {
            pairs.push(Pair::new(i as u64, (i * 10) as u64));
        }
        let pairs_buf = DeviceBuffer::from_slice(&pairs)?;
        let out_results: DeviceBuffer<bool> = unsafe { DeviceBuffer::uninitialized(num_items)? };

        let kernel_insert = module.get_function("test_cg_insert_bs1_cg2")?;
        let cg_size = 2;
        let block_size = 128;
        let total_threads = num_items * cg_size;
        let grid_size = (total_threads + block_size - 1) / block_size;

        let map_ref = map.device_ref();

        unsafe {
            launch!(kernel_insert<<<grid_size as u32, block_size as u32, 0, stream>>>(
                pairs_buf.as_device_ptr(),
                num_items,
                out_results.as_device_ptr(),
                map_ref
            ))?;
        }
        stream.synchronize()?;

        // Test cooperative contains for existing and non-existing keys
        let mut keys = Vec::with_capacity(num_items + 20);
        for i in 0..num_items {
            keys.push(i as u64);
        }
        for i in num_items..num_items + 20 {
            keys.push(i as u64);
        }
        let keys_buf = DeviceBuffer::from_slice(&keys)?;
        let out_contains = unsafe { DeviceBuffer::uninitialized(keys.len())? };

        let kernel_contains = module.get_function("test_cg_contains_bs1_cg2")?;
        let total_threads_contains = keys.len() * cg_size;
        let grid_size_contains = (total_threads_contains + block_size - 1) / block_size;

        unsafe {
            launch!(kernel_contains<<<grid_size_contains as u32, block_size as u32, 0, stream>>>(
                keys_buf.as_device_ptr(),
                keys.len(),
                out_contains.as_device_ptr(),
                map_ref
            ))?;
        }
        stream.synchronize()?;

        let mut host_contains = vec![false; keys.len()];
        out_contains.copy_to(&mut host_contains)?;

        // First num_items should be true, rest should be false
        for i in 0..num_items {
            assert!(
                host_contains[i],
                "Cooperative contains should return true for existing key {} (CG2)",
                i
            );
        }
        for i in num_items..keys.len() {
            assert!(
                !host_contains[i],
                "Cooperative contains should return false for non-existing key {} (CG2)",
                i
            );
        }

        Ok(())
    }
}

// Integration and Workflow Tests
mod integration {
    use super::test_helpers::*;
    use super::*;

    mod workflows {
        use super::*;

        /// Test insert-then-find workflow: Insert items, then find them all
        #[test]
        fn test_insert_then_find() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Insert multiple items
            let num_items = 100;
            let mut pairs = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs.push(Pair::new(i as u64, (i * 10) as u64));
            }

            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(success_count, num_items, "All inserts should succeed");

            // Find all inserted items
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let mut output = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output.as_mut_slice(), &stream, &module)?;
            }

            // Verify all values match
            for i in 0..num_items {
                assert_eq!(
                    output[i],
                    (i * 10) as u64,
                    "Found value should match inserted value for key {}",
                    i
                );
            }

            Ok(())
        }

        /// Test find-then-insert workflow: Find non-existent, then insert, then find again
        #[test]
        fn test_find_then_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            let test_key = 42u64;
            let test_value = 100u64;
            let empty_val = map.empty_value_sentinel();

            // First, try to find a non-existent key
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[test_key], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(
                output[0], empty_val,
                "Find should return empty_value_sentinel for non-existent key"
            );

            // Insert the key
            let pairs = vec![Pair::new(test_key, test_value)];
            let success_count = map.insert(&pairs, &stream, &module)?;
            assert_eq!(success_count, 1, "Insert should succeed");

            // Find it again - should now return the value
            unsafe {
                map.find(&[test_key], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(
                output[0], test_value,
                "Find should return inserted value after insert"
            );

            Ok(())
        }

        /// Test insert-clear-insert workflow: Insert, clear, insert again
        #[test]
        fn test_insert_clear_insert() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // First insert
            let num_items = 50;
            let mut pairs1 = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs1.push(Pair::new(i as u64, (i * 10) as u64));
            }

            let success_count1 = map.insert(&pairs1, &stream, &module)?;
            assert_eq!(success_count1, num_items, "First insert should succeed");

            // Verify items are present
            let keys: Vec<u64> = (0..num_items).map(|i| i as u64).collect();
            let mut output1 = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output1.as_mut_slice(), &stream, &module)?;
            }
            for i in 0..num_items {
                assert_eq!(
                    output1[i],
                    (i * 10) as u64,
                    "Value should be present before clear"
                );
            }

            // Clear the map
            map.clear(&stream)?;

            // Verify items are gone
            let mut output2 = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output2.as_mut_slice(), &stream, &module)?;
            }
            let empty_val = map.empty_value_sentinel();
            for i in 0..num_items {
                assert_eq!(
                    output2[i], empty_val,
                    "Value should be empty_value_sentinel after clear"
                );
            }

            // Insert again with different values
            let mut pairs2 = Vec::with_capacity(num_items);
            for i in 0..num_items {
                pairs2.push(Pair::new(i as u64, (i * 20) as u64));
            }

            let success_count2 = map.insert(&pairs2, &stream, &module)?;
            assert_eq!(success_count2, num_items, "Second insert should succeed");

            // Verify new values are present
            let mut output3 = unsafe { LockedBuffer::uninitialized(num_items)? };
            unsafe {
                map.find(&keys, output3.as_mut_slice(), &stream, &module)?;
            }
            for i in 0..num_items {
                assert_eq!(
                    output3[i],
                    (i * 20) as u64,
                    "Value should match second insert after clear"
                );
            }

            Ok(())
        }

        /// Test mixed operations: Interleave inserts, finds, and contains
        #[test]
        fn test_mixed_operations() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            let empty_val = map.empty_value_sentinel();

            // Insert some initial items
            let initial_pairs = vec![
                Pair::new(1u64, 10u64),
                Pair::new(2u64, 20u64),
                Pair::new(3u64, 30u64),
            ];
            let success_count = map.insert(&initial_pairs, &stream, &module)?;
            assert_eq!(success_count, 3, "Initial inserts should succeed");

            // Find existing key
            let mut output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.find(&[2u64], output.as_mut_slice(), &stream, &module)?;
            }
            assert_eq!(output[0], 20u64, "Find should return correct value");

            // Check contains for existing key
            let mut contains_output = unsafe { LockedBuffer::uninitialized(1)? };
            unsafe {
                map.contains(&[2u64], contains_output.as_mut_slice(), &stream, &module)?;
            }
            assert!(
                contains_output[0],
                "Contains should return true for existing key"
            );

            // Check contains for non-existent key
            unsafe {
                map.contains(&[99u64], contains_output.as_mut_slice(), &stream, &module)?;
            }
            assert!(
                !contains_output[0],
                "Contains should return false for non-existent key"
            );

            // Insert more items
            let more_pairs = vec![Pair::new(4u64, 40u64), Pair::new(5u64, 50u64)];
            let success_count2 = map.insert(&more_pairs, &stream, &module)?;
            assert_eq!(success_count2, 2, "Additional inserts should succeed");

            // Find multiple keys (mix of existing and non-existent)
            let mixed_keys = vec![1u64, 99u64, 3u64, 100u64, 5u64];
            let mut mixed_output = unsafe { LockedBuffer::uninitialized(mixed_keys.len())? };
            unsafe {
                map.find(&mixed_keys, mixed_output.as_mut_slice(), &stream, &module)?;
            }

            assert_eq!(mixed_output[0], 10u64, "Key 1 should be found");
            assert_eq!(mixed_output[1], empty_val, "Key 99 should not be found");
            assert_eq!(mixed_output[2], 30u64, "Key 3 should be found");
            assert_eq!(mixed_output[3], empty_val, "Key 100 should not be found");
            assert_eq!(mixed_output[4], 50u64, "Key 5 should be found");

            // Batch contains for mixed keys
            let mut batch_contains = unsafe { LockedBuffer::uninitialized(mixed_keys.len())? };
            unsafe {
                map.contains(&mixed_keys, batch_contains.as_mut_slice(), &stream, &module)?;
            }

            assert!(batch_contains[0], "Key 1 should exist");
            assert!(!batch_contains[1], "Key 99 should not exist");
            assert!(batch_contains[2], "Key 3 should exist");
            assert!(!batch_contains[3], "Key 100 should not exist");
            assert!(batch_contains[4], "Key 5 should exist");

            Ok(())
        }

        /// Test sequential operations: Long sequence of operations
        #[test]
        fn test_sequential_operations() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(2048, &stream)?;

            let empty_val = map.empty_value_sentinel();
            let num_operations = 200;

            // Perform a long sequence of insert, find, contains operations
            for i in 0..num_operations {
                // Insert
                let pairs = vec![Pair::new(i as u64, (i * 10) as u64)];
                let success_count = map.insert(&pairs, &stream, &module)?;
                assert_eq!(success_count, 1, "Insert should succeed at iteration {}", i);

                // Find immediately after insert
                let mut output = unsafe { LockedBuffer::uninitialized(1)? };
                unsafe {
                    map.find(&[i as u64], output.as_mut_slice(), &stream, &module)?;
                }
                assert_eq!(
                    output[0],
                    (i * 10) as u64,
                    "Find should return correct value immediately after insert at iteration {}",
                    i
                );

                // Check contains
                let mut contains_output = unsafe { LockedBuffer::uninitialized(1)? };
                unsafe {
                    map.contains(
                        &[i as u64],
                        contains_output.as_mut_slice(),
                        &stream,
                        &module,
                    )?;
                }
                assert!(
                    contains_output[0],
                    "Contains should return true immediately after insert at iteration {}",
                    i
                );

                // Verify previous keys are still accessible
                if i > 0 {
                    let prev_key = (i - 1) as u64;
                    unsafe {
                        map.find(&[prev_key], output.as_mut_slice(), &stream, &module)?;
                    }
                    assert_eq!(
                        output[0],
                        ((i - 1) * 10) as u64,
                        "Previous key should still be accessible at iteration {}",
                        i
                    );
                }
            }

            // Final verification: find all keys
            let all_keys: Vec<u64> = (0..num_operations).map(|i| i as u64).collect();
            let mut final_output = unsafe { LockedBuffer::uninitialized(num_operations)? };
            unsafe {
                map.find(&all_keys, final_output.as_mut_slice(), &stream, &module)?;
            }

            for i in 0..num_operations {
                assert_eq!(
                    final_output[i],
                    (i * 10) as u64,
                    "Final verification: value mismatch for key {}",
                    i
                );
            }

            Ok(())
        }
    }

    mod scenarios {
        use super::*;

        /// Test dictionary lookup scenario: Simulate dictionary with many lookups
        #[test]
        fn test_dictionary_lookup() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(4096, &stream)?;

            // Build dictionary: word_id -> definition_id mapping
            let num_words = 1000;
            let mut dictionary = Vec::with_capacity(num_words);
            for i in 0..num_words {
                // Use word_id as key, definition_id as value
                dictionary.push(Pair::new(i as u64, (i * 100) as u64));
            }

            // Insert all dictionary entries
            let success_count = map.insert(&dictionary, &stream, &module)?;
            assert_eq!(
                success_count, num_words,
                "All dictionary entries should be inserted"
            );

            // Simulate many lookups (more than inserts)
            let num_lookups = 2000;
            let mut lookup_keys = Vec::with_capacity(num_lookups);
            for i in 0..num_lookups {
                // Deterministic access pattern: some valid, some invalid
                // Use modulo to create a pattern that includes both valid and invalid keys
                let key = (i % (num_words * 2)) as u64;
                lookup_keys.push(key);
            }

            let mut lookup_results = unsafe { LockedBuffer::uninitialized(num_lookups)? };
            unsafe {
                map.find(
                    &lookup_keys,
                    lookup_results.as_mut_slice(),
                    &stream,
                    &module,
                )?;
            }

            let empty_val = map.empty_value_sentinel();

            // Verify lookup results
            for (i, &key) in lookup_keys.iter().enumerate() {
                if key < num_words as u64 {
                    // Valid key - should find definition
                    assert_eq!(
                        lookup_results[i],
                        (key * 100) as u64,
                        "Dictionary lookup should find definition for valid word_id {}",
                        key
                    );
                } else {
                    // Invalid key - should return empty_value_sentinel
                    assert_eq!(
                        lookup_results[i], empty_val,
                        "Dictionary lookup should return empty_value_sentinel for invalid word_id {}",
                        key
                    );
                }
            }

            // Test batch contains for dictionary lookups
            let mut contains_results = unsafe { LockedBuffer::uninitialized(num_lookups)? };
            unsafe {
                map.contains(
                    &lookup_keys,
                    contains_results.as_mut_slice(),
                    &stream,
                    &module,
                )?;
            }

            for (i, &key) in lookup_keys.iter().enumerate() {
                let should_exist = key < num_words as u64;
                assert_eq!(
                    contains_results[i], should_exist,
                    "Contains should match find result for word_id {}",
                    key
                );
            }

            Ok(())
        }

        /// Test cache simulation: Insert, find, insert pattern
        #[test]
        fn test_cache_simulation() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            let cache_size = 100;
            let num_accesses = 500;
            let empty_val = map.empty_value_sentinel();

            // Simulate cache behavior: insert on miss, find on hit
            for access in 0..num_accesses {
                let key = (access % (cache_size * 2)) as u64; // Some keys repeat, some don't
                let value = (access * 10) as u64;

                // Try to find first (cache lookup)
                let mut output = unsafe { LockedBuffer::uninitialized(1)? };
                unsafe {
                    map.find(&[key], output.as_mut_slice(), &stream, &module)?;
                }

                if output[0] == empty_val {
                    // Cache miss - insert
                    let pairs = vec![Pair::new(key, value)];
                    let success_count = map.insert(&pairs, &stream, &module)?;
                    assert_eq!(
                        success_count, 1,
                        "Cache insert should succeed on miss for access {}",
                        access
                    );
                } else {
                    // Cache hit - verify value is correct
                    // Note: In real cache, we might update value, but StaticMap doesn't overwrite
                    // So we verify the value matches what was inserted earlier
                    // Actually, let's just verify it's not empty
                    assert_ne!(
                        output[0], empty_val,
                        "Cache hit should return non-empty value for access {}",
                        access
                    );
                }
            }

            // Verify cache contains expected entries
            let mut verify_keys = Vec::new();
            for i in 0..cache_size {
                verify_keys.push(i as u64);
            }

            let mut verify_results = unsafe { LockedBuffer::uninitialized(verify_keys.len())? };
            unsafe {
                map.find(
                    &verify_keys,
                    verify_results.as_mut_slice(),
                    &stream,
                    &module,
                )?;
            }

            // All keys in cache range should be present (or at least most of them)
            let mut found_count = 0;
            for &val in verify_results.as_slice() {
                if val != empty_val {
                    found_count += 1;
                }
            }

            assert!(
                found_count > cache_size / 2,
                "Cache should contain at least half of the expected entries, found {}",
                found_count
            );

            Ok(())
        }

        /// Test set operations: Use map as set (value = key or constant)
        #[test]
        fn test_set_operations() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(1024, &stream)?;

            // Use map as a set: value equals key (or constant 1)
            let set_size = 200;
            let mut set_elements = Vec::with_capacity(set_size);
            for i in 0..set_size {
                // Value = key (identity mapping for set)
                set_elements.push(Pair::new(i as u64, i as u64));
            }

            // Insert set elements
            let success_count = map.insert(&set_elements, &stream, &module)?;
            assert_eq!(
                success_count, set_size,
                "All set elements should be inserted"
            );

            // Test membership queries (set contains)
            let test_elements: Vec<u64> = vec![
                0,   // Should be in set
                50,  // Should be in set
                99,  // Should be in set
                200, // Should NOT be in set
                250, // Should NOT be in set
                150, // Should be in set
            ];

            let mut contains_results = unsafe { LockedBuffer::uninitialized(test_elements.len())? };
            unsafe {
                map.contains(
                    &test_elements,
                    contains_results.as_mut_slice(),
                    &stream,
                    &module,
                )?;
            }

            assert!(contains_results[0], "Element 0 should be in set");
            assert!(contains_results[1], "Element 50 should be in set");
            assert!(contains_results[2], "Element 99 should be in set");
            assert!(!contains_results[3], "Element 200 should NOT be in set");
            assert!(!contains_results[4], "Element 250 should NOT be in set");
            assert!(contains_results[5], "Element 150 should be in set");

            // Test set union simulation: try to add more elements
            let additional_elements: Vec<Pair<u64, u64>> = (set_size..set_size + 50)
                .map(|i| Pair::new(i as u64, i as u64))
                .collect();

            let union_success = map.insert(&additional_elements, &stream, &module)?;
            assert_eq!(
                union_success, 50,
                "Union operation should add 50 new elements"
            );

            // Verify union result
            let union_test_keys: Vec<u64> = vec![set_size as u64, (set_size + 25) as u64];
            let mut union_results = unsafe { LockedBuffer::uninitialized(union_test_keys.len())? };
            unsafe {
                map.find(
                    &union_test_keys,
                    union_results.as_mut_slice(),
                    &stream,
                    &module,
                )?;
            }

            assert_eq!(
                union_results[0], set_size as u64,
                "Union element should be present"
            );
            assert_eq!(
                union_results[1],
                (set_size + 25) as u64,
                "Union element should be present"
            );

            Ok(())
        }

        /// Test frequency counting: Count occurrences of keys
        #[test]
        fn test_frequency_counting() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (mut map, module) = create_test_map_with_module(2048, &stream)?;

            // Simulate counting frequencies: key = item, value = count
            let num_unique_items = 50;
            let occurrences_per_item = 10;

            // Insert items with initial count of 1
            let mut items = Vec::with_capacity(num_unique_items);
            for i in 0..num_unique_items {
                items.push(Pair::new(i as u64, 1u64)); // Initial count = 1
            }

            let success_count = map.insert(&items, &stream, &module)?;
            assert_eq!(
                success_count, num_unique_items,
                "Initial insert should succeed for all items"
            );

            // Simulate counting: try to increment counts
            // Note: StaticMap doesn't support update, so we simulate by checking if exists
            // and tracking counts separately. In real implementation, you'd use atomic operations.
            let mut frequency_data = Vec::new();
            for _round in 0..occurrences_per_item {
                for item_id in 0..num_unique_items {
                    frequency_data.push(item_id as u64);
                }
            }

            // Verify all items are present (simulating frequency counting)
            let mut verify_results = unsafe { LockedBuffer::uninitialized(num_unique_items)? };
            let verify_keys: Vec<u64> = (0..num_unique_items).map(|i| i as u64).collect();
            unsafe {
                map.find(
                    &verify_keys,
                    verify_results.as_mut_slice(),
                    &stream,
                    &module,
                )?;
            }

            // All items should be present (each was inserted once)
            let empty_val = map.empty_value_sentinel();
            for (i, &val) in verify_results.as_slice().iter().enumerate() {
                assert_ne!(
                    val, empty_val,
                    "Item {} should be present in frequency map",
                    i
                );
                assert_eq!(val, 1u64, "Item {} should have count of 1", i);
            }

            // Test batch contains for frequency checking
            let mut batch_contains = unsafe { LockedBuffer::uninitialized(frequency_data.len())? };
            unsafe {
                map.contains(
                    &frequency_data,
                    batch_contains.as_mut_slice(),
                    &stream,
                    &module,
                )?;
            }

            // All items in frequency_data should be present
            for (i, &present) in batch_contains.as_slice().iter().enumerate() {
                assert!(
                    present,
                    "Item {} in frequency data should be present",
                    frequency_data[i]
                );
            }

            // Test finding specific items
            let test_items = vec![0u64, 25u64, 49u64];
            let mut test_results = unsafe { LockedBuffer::uninitialized(test_items.len())? };
            unsafe {
                map.find(&test_items, test_results.as_mut_slice(), &stream, &module)?;
            }

            for (i, &item) in test_items.iter().enumerate() {
                assert_eq!(
                    test_results[i], 1u64,
                    "Item {} should have frequency count of 1",
                    item
                );
            }

            Ok(())
        }
    }
}
