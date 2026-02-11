#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![allow(incomplete_features)]

use cuda_static_map::{StaticMap, get_ptx};
use cuda_static_map_kernels::hash::IdentityHash;
use cuda_static_map_kernels::open_addressing::{DefaultKeyEqual, ThreadScope};
use cuda_static_map_kernels::pair::{Pair};
use cuda_static_map_kernels::probing::LinearProbing;
use cust::prelude::*;
use cust::memory::LockedBuffer;
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

    pub fn create_test_map(
        capacity: usize,
        stream: &Stream,
    ) -> Result<TestMap, Box<dyn Error>> {
        let empty_key = u64::MAX;
        let empty_val = u64::MAX;
        let probing = LinearProbing::<u64, IdentityHash<u64>>::new(IdentityHash::new());
        let pred = DefaultKeyEqual;
        Ok(TestMap::new(capacity, empty_key, empty_val, pred, probing, stream)?)
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
    use super::*;
    use super::test_helpers::*;

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
            assert_eq!(
                success_count, num_items,
                "All inserts should succeed"
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
                    "Value mismatch at index {}",
                    i
                );
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
            assert!(!output[0], "Contains should return false for non-existent key");

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
    use super::*;
    use super::test_helpers::*;
    use cuda_static_map_kernels::hash::{XXHash32, XXHash64};
    use cuda_static_map_kernels::probing::{DoubleHashProbing, LinearProbing, ProbingScheme};

    // Helper macro to test bulk operations for a specific bucket size
    macro_rules! test_bucket_size_bulk_ops {
        ($bucket_size:literal, $stream:expr, $module:expr) => {
            {
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
                        i, $bucket_size
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
            }
        };
    }

    mod bucket_size {
        use super::*;

        /// Test bucket size 1
        #[test]
        fn test_bucket_size_1() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (_, module) = create_test_map_with_module(1024, &stream)?;
            test_bucket_size_bulk_ops!(1, &stream, &module)
        }

        /// Test bucket size 2
        #[test]
        fn test_bucket_size_2() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;
            test_bucket_size_bulk_ops!(2, &stream, &module)
        }

        /// Test bucket size 4
        #[test]
        fn test_bucket_size_4() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;
            test_bucket_size_bulk_ops!(4, &stream, &module)
        }

        /// Test bucket size 8
        #[test]
        fn test_bucket_size_8() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;
            test_bucket_size_bulk_ops!(8, &stream, &module)
        }

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
                        >::new(1024, empty_key, empty_val, pred, probing, &stream)?;
                    }
                    2 => {
                        let _map = StaticMap::<
                            u64,
                            u64,
                            LinearProbing<u64, IdentityHash<u64>>,
                            2,
                            DefaultKeyEqual,
                            { ThreadScope::Device },
                        >::new(1024, empty_key, empty_val, pred, probing, &stream)?;
                    }
                    4 => {
                        let _map = StaticMap::<
                            u64,
                            u64,
                            LinearProbing<u64, IdentityHash<u64>>,
                            4,
                            DefaultKeyEqual,
                            { ThreadScope::Device },
                        >::new(1024, empty_key, empty_val, pred, probing, &stream)?;
                    }
                    8 => {
                        let _map = StaticMap::<
                            u64,
                            u64,
                            LinearProbing<u64, IdentityHash<u64>>,
                            8,
                            DefaultKeyEqual,
                            { ThreadScope::Device },
                        >::new(1024, empty_key, empty_val, pred, probing, &stream)?;
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
            assert!(map.capacity() >= 1024, "Capacity should be at least 1024 for double hashing");
            assert_eq!(map.empty_key_sentinel(), empty_key);
            assert_eq!(map.empty_value_sentinel(), empty_val);

            // Note: DoubleHashProbing may not support bulk operations yet
            // Just verify the map was created successfully
            Ok(())
        }
    }

    mod hash_functions {
        use super::*;

        /// Test IdentityHash
        #[test]
        fn test_identity_hash() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let (_, module) = create_test_map_with_module(1024, &stream)?;
            test_bucket_size_bulk_ops!(1, &stream, &module)
        }

        /// Test XXHash32
        #[test]
        fn test_xxhash32() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let hasher = XXHash32::new(0);
            let probing = LinearProbing::<u64, XXHash32<u64>>::new(hasher);
            let pred = DefaultKeyEqual;

            let mut map = StaticMap::<
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

            // Note: XXHash32 may not support bulk operations yet (only canonical type supports bulk)
            // Just verify the map was created successfully
            Ok(())
        }

        /// Test XXHash64
        #[test]
        fn test_xxhash64() -> Result<(), Box<dyn Error>> {
            let (_ctx, stream) = setup_cuda()?;
            let ptx = get_ptx();
            let module = Module::from_ptx(ptx, &[])?;

            let empty_key = u64::MAX;
            let empty_val = u64::MAX;
            let hasher = XXHash64::new(0);
            let probing = LinearProbing::<u64, XXHash64<u64>>::new(hasher);
            let pred = DefaultKeyEqual;

            let mut map = StaticMap::<
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

            // Note: XXHash64 may not support bulk operations yet (only canonical type supports bulk)
            // Just verify the map was created successfully
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
