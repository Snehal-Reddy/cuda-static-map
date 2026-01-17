use core::marker::PhantomData;
use cust_core::DeviceCopy;

use crate::hash::Hash;

/// XXHash-32 hash function implementation.
#[repr(C)]
#[derive(Debug)]
pub struct XXHash32<Key> {
    seed: u32,
    _phantom: PhantomData<Key>,
}

impl<Key> Clone for XXHash32<Key> {
    fn clone(&self) -> Self {
        Self {
            seed: self.seed,
            _phantom: PhantomData,
        }
    }
}

impl<Key> Copy for XXHash32<Key> {}

impl<Key> Default for XXHash32<Key> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<Key> XXHash32<Key> {
    const PRIME1: u32 = 0x9e3779b1;
    const PRIME2: u32 = 0x85ebca77;
    const PRIME3: u32 = 0xc2b2ae3d;
    const PRIME4: u32 = 0x27d4eb2f;
    const PRIME5: u32 = 0x165667b1;

    /// Create a new XXHash-32 hash function instance with the given seed.
    ///
    /// # Arguments
    ///
    /// * `seed` - A custom number to randomize the resulting hash value (defaults to 0)
    pub const fn new(seed: u32) -> Self {
        Self {
            seed,
            _phantom: PhantomData,
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that `bytes` contains enough data to read a `T` at the given `index`.
    /// Specifically, `bytes.len()` must be >= `index * size_of::<T>() + size_of::<T>()`.
    unsafe fn load_chunk<T: Copy>(bytes: &[u8], index: usize) -> T {
        // Safety: `read_unaligned` requires:
        // 1. Valid pointer: `bytes.as_ptr()` is valid (from a valid slice), and `add()` keeps it within bounds
        // 2. Initialized memory: `bytes` comes from a valid `Key` reference, so all bytes are initialized
        // 3. Readable memory: `bytes` is a valid slice, so the memory is readable
        // 4. Sufficient size: The function's safety requirements guarantee `bytes.len() >= index * size_of::<T>() + size_of::<T>()`,
        //    so there are at least `size_of::<T>()` bytes available starting at the computed pointer
        unsafe {
            let ptr = bytes.as_ptr().add(index * core::mem::size_of::<T>());
            core::ptr::read_unaligned(ptr as *const T)
        }
    }

    fn finalize(mut h: u32) -> u32 {
        h ^= h >> 15;
        h = h.wrapping_mul(Self::PRIME2);
        h ^= h >> 13;
        h = h.wrapping_mul(Self::PRIME3);
        h ^= h >> 16;
        h
    }

    pub fn compute_hash(&self, key: &Key) -> u32 {
        let size = core::mem::size_of::<Key>();
        // Safety: `key` is a valid reference to a `Key` value. We reinterpret it as a byte slice
        // of length `size_of::<Key>()`. This is safe because:
        // - The pointer `key as *const Key as *const u8` is valid and points to the start of the Key
        // - The size is exactly `size_of::<Key>()`, so the slice covers exactly one Key value
        // - The slice is only used for reading bytes and doesn't outlive the key reference
        let bytes = unsafe { core::slice::from_raw_parts(key as *const Key as *const u8, size) };

        let mut h32: u32;
        let mut offset = 0;

        if size >= 16 {
            let limit = size - 16;
            let mut v1 = self.seed.wrapping_add(Self::PRIME1).wrapping_add(Self::PRIME2);
            let mut v2 = self.seed.wrapping_add(Self::PRIME2);
            let mut v3 = self.seed;
            let mut v4 = self.seed.wrapping_sub(Self::PRIME1);

            while offset <= limit {
                let pipeline_offset = offset / 4;
                // SAFETY: We checked `size >= 16` and `limit = size - 16`.
                // `offset <= limit` ensures we have at least 16 bytes remaining.
                // We read 4 u32s (16 bytes) at `offset`, `offset+4`, `offset+8`, `offset+12` (byte offsets).
                // `pipeline_offset` is `offset / 4`, so indices are `pipeline_offset + 0..3`.
                unsafe {
                    v1 = v1.wrapping_add(
                        Self::load_chunk::<u32>(bytes, pipeline_offset).wrapping_mul(Self::PRIME2),
                    );
                    v1 = v1.rotate_left(13);
                    v1 = v1.wrapping_mul(Self::PRIME1);

                    v2 = v2.wrapping_add(
                        Self::load_chunk::<u32>(bytes, pipeline_offset + 1).wrapping_mul(Self::PRIME2),
                    );
                    v2 = v2.rotate_left(13);
                    v2 = v2.wrapping_mul(Self::PRIME1);

                    v3 = v3.wrapping_add(
                        Self::load_chunk::<u32>(bytes, pipeline_offset + 2).wrapping_mul(Self::PRIME2),
                    );
                    v3 = v3.rotate_left(13);
                    v3 = v3.wrapping_mul(Self::PRIME1);

                    v4 = v4.wrapping_add(
                        Self::load_chunk::<u32>(bytes, pipeline_offset + 3).wrapping_mul(Self::PRIME2),
                    );
                    v4 = v4.rotate_left(13);
                    v4 = v4.wrapping_mul(Self::PRIME1);
                }

                offset += 16;
            }

            h32 = v1
                .rotate_left(1)
                .wrapping_add(v2.rotate_left(7))
                .wrapping_add(v3.rotate_left(12))
                .wrapping_add(v4.rotate_left(18));
        } else {
            h32 = self.seed.wrapping_add(Self::PRIME5);
        }

        h32 = h32.wrapping_add(size as u32);

        if (size % 16) >= 4 {
            while offset <= size - 4 {
                // SAFETY: `offset <= size - 4` ensures we have at least 4 bytes remaining.
                h32 = h32.wrapping_add(
                    unsafe { Self::load_chunk::<u32>(bytes, offset / 4) }.wrapping_mul(Self::PRIME3),
                );
                h32 = h32.rotate_left(17).wrapping_mul(Self::PRIME4);
                offset += 4;
            }
        }

        if size % 4 != 0 {
            while offset < size {
                h32 = h32.wrapping_add((bytes[offset] as u32 & 0xff).wrapping_mul(Self::PRIME5));
                h32 = h32.rotate_left(11).wrapping_mul(Self::PRIME1);
                offset += 1;
            }
        }

        Self::finalize(h32)
    }
}

// Safety: XXHash32 contains only a u32 seed and PhantomData<Key>.
// Both u32 and PhantomData implement DeviceCopy, and XXHash32 contains no references
// to CPU data, making it safe to copy to/from CUDA device memory.
unsafe impl<Key> DeviceCopy for XXHash32<Key> {}

impl<Key> Hash<Key> for XXHash32<Key> {
    type HashType = u32;
    fn hash(&self, key: &Key) -> Self::HashType {
        // For keys <= 16 bytes, copy the key first to ensure proper alignment.
        // SAFETY: We're copying the key to ensure alignment. This is safe because:
        // 1. We only do this for keys <= 16 bytes (small, stack-allocated types)
        // 2. We immediately use the copy and it doesn't escape
        // 3. The key must be trivially copyable for the hash function to work anyway
        if core::mem::size_of::<Key>() <= 16 {
            let key_copy = unsafe { core::ptr::read(key as *const Key) };
            self.compute_hash(&key_copy)
        } else {
            self.compute_hash(key)
        }
    }
}

/// XXHash-64 hash function implementation.
#[repr(C)]
#[derive(Debug)]
pub struct XXHash64<Key> {
    seed: u64,
    _phantom: PhantomData<Key>,
}

impl<Key> Clone for XXHash64<Key> {
    fn clone(&self) -> Self {
        Self {
            seed: self.seed,
            _phantom: PhantomData,
        }
    }
}

impl<Key> Copy for XXHash64<Key> {}

impl<Key> Default for XXHash64<Key> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<Key> XXHash64<Key> {
    const PRIME1: u64 = 11400714785074694791;
    const PRIME2: u64 = 14029467366897019727;
    const PRIME3: u64 = 1609587929392839161;
    const PRIME4: u64 = 9650029242287828579;
    const PRIME5: u64 = 2870177450012600261;

    /// Create a new XXHash-64 hash function instance with the given seed.
    ///
    /// # Arguments
    ///
    /// * `seed` - A custom number to randomize the resulting hash value (defaults to 0)
    pub const fn new(seed: u64) -> Self {
        Self {
            seed,
            _phantom: PhantomData,
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that `bytes` contains enough data to read a `T` at the given `index`.
    /// Specifically, `bytes.len()` must be >= `index * size_of::<T>() + size_of::<T>()`.
    unsafe fn load_chunk<T: Copy>(bytes: &[u8], index: usize) -> T {
        // Safety: `read_unaligned` requires:
        // 1. Valid pointer: `bytes.as_ptr()` is valid (from a valid slice), and `add()` keeps it within bounds
        // 2. Initialized memory: `bytes` comes from a valid `Key` reference, so all bytes are initialized
        // 3. Readable memory: `bytes` is a valid slice, so the memory is readable
        // 4. Sufficient size: The function's safety requirements guarantee `bytes.len() >= index * size_of::<T>() + size_of::<T>()`,
        //    so there are at least `size_of::<T>()` bytes available starting at the computed pointer
        unsafe {
            let ptr = bytes.as_ptr().add(index * core::mem::size_of::<T>());
            core::ptr::read_unaligned(ptr as *const T)
        }
    }

    fn finalize(mut h: u64) -> u64 {
        h ^= h >> 33;
        h = h.wrapping_mul(Self::PRIME2);
        h ^= h >> 29;
        h = h.wrapping_mul(Self::PRIME3);
        h ^= h >> 32;
        h
    }

    pub fn compute_hash(&self, key: &Key) -> u64 {
        let size = core::mem::size_of::<Key>();
        // Safety: `key` is a valid reference to a `Key` value. We reinterpret it as a byte slice
        // of length `size_of::<Key>()`. This is safe because:
        // - The pointer `key as *const Key as *const u8` is valid and points to the start of the Key
        // - The size is exactly `size_of::<Key>()`, so the slice covers exactly one Key value
        // - The slice is only used for reading bytes and doesn't outlive the key reference
        let bytes = unsafe { core::slice::from_raw_parts(key as *const Key as *const u8, size) };

        let mut h64: u64;
        let mut offset = 0;

        // Data can be processed in 32-byte chunks
        if size >= 32 {
            let limit = size - 32;
            let mut v1 = self.seed.wrapping_add(Self::PRIME1).wrapping_add(Self::PRIME2);
            let mut v2 = self.seed.wrapping_add(Self::PRIME2);
            let mut v3 = self.seed;
            let mut v4 = self.seed.wrapping_sub(Self::PRIME1);

            while offset <= limit {
                let pipeline_offset = offset / 8;
                // SAFETY: We checked `size >= 32` and `limit = size - 32`.
                // `offset <= limit` ensures we have at least 32 bytes remaining.
                // We read 4 u64s (32 bytes) at `offset`, `offset+8`, `offset+16`, `offset+24` (byte offsets).
                // `pipeline_offset` is `offset / 8`, so indices are `pipeline_offset + 0..3`.
                unsafe {
                    v1 = v1.wrapping_add(
                        Self::load_chunk::<u64>(bytes, pipeline_offset).wrapping_mul(Self::PRIME2),
                    );
                    v1 = v1.rotate_left(31);
                    v1 = v1.wrapping_mul(Self::PRIME1);

                    v2 = v2.wrapping_add(
                        Self::load_chunk::<u64>(bytes, pipeline_offset + 1).wrapping_mul(Self::PRIME2),
                    );
                    v2 = v2.rotate_left(31);
                    v2 = v2.wrapping_mul(Self::PRIME1);

                    v3 = v3.wrapping_add(
                        Self::load_chunk::<u64>(bytes, pipeline_offset + 2).wrapping_mul(Self::PRIME2),
                    );
                    v3 = v3.rotate_left(31);
                    v3 = v3.wrapping_mul(Self::PRIME1);

                    v4 = v4.wrapping_add(
                        Self::load_chunk::<u64>(bytes, pipeline_offset + 3).wrapping_mul(Self::PRIME2),
                    );
                    v4 = v4.rotate_left(31);
                    v4 = v4.wrapping_mul(Self::PRIME1);
                }

                offset += 32;
            }

            h64 = v1
                .rotate_left(1)
                .wrapping_add(v2.rotate_left(7))
                .wrapping_add(v3.rotate_left(12))
                .wrapping_add(v4.rotate_left(18));

            // Merge v1-v4 into h64
            v1 = v1.wrapping_mul(Self::PRIME2);
            v1 = v1.rotate_left(31);
            v1 = v1.wrapping_mul(Self::PRIME1);
            h64 ^= v1;
            h64 = h64.wrapping_mul(Self::PRIME1).wrapping_add(Self::PRIME4);

            v2 = v2.wrapping_mul(Self::PRIME2);
            v2 = v2.rotate_left(31);
            v2 = v2.wrapping_mul(Self::PRIME1);
            h64 ^= v2;
            h64 = h64.wrapping_mul(Self::PRIME1).wrapping_add(Self::PRIME4);

            v3 = v3.wrapping_mul(Self::PRIME2);
            v3 = v3.rotate_left(31);
            v3 = v3.wrapping_mul(Self::PRIME1);
            h64 ^= v3;
            h64 = h64.wrapping_mul(Self::PRIME1).wrapping_add(Self::PRIME4);

            v4 = v4.wrapping_mul(Self::PRIME2);
            v4 = v4.rotate_left(31);
            v4 = v4.wrapping_mul(Self::PRIME1);
            h64 ^= v4;
            h64 = h64.wrapping_mul(Self::PRIME1).wrapping_add(Self::PRIME4);
        } else {
            h64 = self.seed.wrapping_add(Self::PRIME5);
        }

        h64 = h64.wrapping_add(size as u64);

        // Remaining data can be processed in 8-byte chunks
        if (size % 32) >= 8 {
            while offset <= size - 8 {
                // SAFETY: `offset <= size - 8` ensures we have at least 8 bytes remaining.
                let mut k1 = unsafe { Self::load_chunk::<u64>(bytes, offset / 8) }.wrapping_mul(Self::PRIME2);
                k1 = k1.rotate_left(31).wrapping_mul(Self::PRIME1);
                h64 ^= k1;
                h64 = h64.rotate_left(27).wrapping_mul(Self::PRIME1).wrapping_add(Self::PRIME4);
                offset += 8;
            }
        }

        // Remaining data can be processed in 4-byte chunks
        if (size % 8) >= 4 {
            while offset <= size - 4 {
                // SAFETY: `offset <= size - 4` ensures we have at least 4 bytes remaining.
                h64 ^= (unsafe { Self::load_chunk::<u32>(bytes, offset / 4) } as u64 & 0xffffffffu64).wrapping_mul(Self::PRIME1);
                h64 = h64.rotate_left(23).wrapping_mul(Self::PRIME2).wrapping_add(Self::PRIME3);
                offset += 4;
            }
        }

        // Process remaining bytes
        if size % 4 != 0 {
            while offset < size {
                h64 ^= (bytes[offset] as u64 & 0xff).wrapping_mul(Self::PRIME5);
                h64 = h64.rotate_left(11).wrapping_mul(Self::PRIME1);
                offset += 1;
            }
        }

        Self::finalize(h64)
    }
}

// Safety: XXHash64 contains only a u64 seed and PhantomData<Key>.
// Both u64 and PhantomData implement DeviceCopy, and XXHash64 contains no references
// to CPU data, making it safe to copy to/from CUDA device memory.
unsafe impl<Key> DeviceCopy for XXHash64<Key> {}

impl<Key> Hash<Key> for XXHash64<Key> {
    type HashType = u64;
    fn hash(&self, key: &Key) -> Self::HashType {
        // For keys <= 16 bytes, copy the key first to ensure proper alignment.
        // SAFETY: We're copying the key to ensure alignment. This is safe because:
        // 1. We only do this for keys <= 16 bytes (small, stack-allocated types)
        // 2. We immediately use the copy and it doesn't escape
        // 3. The key must be trivially copyable for the hash function to work anyway
        if core::mem::size_of::<Key>() <= 16 {
            let key_copy = unsafe { core::ptr::read(key as *const Key) };
            self.compute_hash(&key_copy)
        } else {
            self.compute_hash(key)
        }
    }
}
