use core::marker::PhantomData;
use cust_core::DeviceCopy;

/// Trait for hash functions that can hash keys on both host and device.
/// 
/// Hash functions must be `Copy` and device-compatible to work in CUDA kernels.
pub trait Hash<Key>: Copy + DeviceCopy {
    /// Hash a key to a `u64` value.
    /// 
    /// This method must be callable from both host and device code.
    fn hash(&self, key: &Key) -> u64;
}

/// Probing iterator that generates slot indices in a probe sequence.
/// 
/// The iterator starts at an initial index and advances by a step size,
/// wrapping around when it reaches the upper bound. Wrap-around detection
/// should be handled by the calling code (compare current index with the
/// initial index stored separately).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProbingIterator {
    /// Current slot index in the probe sequence
    curr_index: usize,
    /// Step size for advancing to the next slot
    step_size: usize,
    /// Upper bound (capacity) - iterator wraps around at this value
    upper_bound: usize,
}

impl ProbingIterator {
    /// Create a new probing iterator.
    /// 
    /// # Arguments
    /// * `start` - Initial slot index to start probing from
    /// * `step_size` - Step size for advancing (bucket_size for linear probing, computed for double hashing)
    /// * `upper_bound` - Upper bound (capacity) - iterator wraps at this value
    pub const fn new(start: usize, step_size: usize, upper_bound: usize) -> Self {
        Self {
            curr_index: start,
            step_size,
            upper_bound,
        }
    }

    /// Get the current slot index.
    pub const fn current(&self) -> usize {
        self.curr_index
    }

    /// Advance to the next slot in the probe sequence.
    pub fn next(&mut self) {
        self.curr_index = (self.curr_index + self.step_size) % self.upper_bound;
    }
}

// Ensure ProbingIterator is device-compatible
unsafe impl DeviceCopy for ProbingIterator {}

/// Trait for probing schemes that define collision resolution strategies.
/// 
/// Probing schemes encapsulate both the hash function(s) and the strategy
/// for resolving collisions in open-addressing hash tables.
pub trait ProbingScheme<Key>: Copy + DeviceCopy {
    /// Type of the hash function(s) used by this probing scheme.
    /// For linear probing, this is a single hash function.
    /// For double hashing, this could be a tuple of two hash functions.
    type Hash: Hash<Key>;

    /// Create a probing iterator for the given key.
    /// 
    /// # Arguments
    /// * `key` - The key to create a probe sequence for
    /// * `bucket_size` - Size of each bucket (typically 1 for simple storage)
    /// * `capacity` - Total capacity of the hash table
    /// 
    /// # Returns
    /// A `ProbingIterator` that will generate slot indices in the probe sequence
    fn make_iterator(&self, key: &Key, bucket_size: usize, capacity: usize) -> ProbingIterator;

    /// Get the hash function(s) used by this probing scheme.
    fn hash_function(&self) -> Self::Hash;

    /// Get the cooperative group size used by this probing scheme.
    /// 
    /// For MVP, this is typically 1 (scalar operations).
    /// TODO: Support larger CG sizes for vectorized operations.
    fn cg_size(&self) -> usize;
}

/// Linear probing scheme.
/// 
/// Uses a simple linear probe sequence: `(hash(key) + i * bucket_size) % capacity`
/// where `i` is the iteration number. This is efficient for low occupancy scenarios.
/// 
/// # Type Parameters
/// * `Key` - The key type that this probing scheme operates on
/// * `H` - Hash function type that implements `Hash<Key>`
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LinearProbing<Key, H> {
    hash: H,
    _phantom: PhantomData<Key>,
}

impl<Key, H> LinearProbing<Key, H>
where
    H: Hash<Key> + Copy + DeviceCopy,
{
    /// Create a new linear probing scheme with the given hash function.
    pub const fn new(hash: H) -> Self {
        Self {
            hash,
            _phantom: PhantomData,
        }
    }
}

impl<Key, H> ProbingScheme<Key> for LinearProbing<Key, H>
where
    H: Hash<Key> + Copy + DeviceCopy,
{
    type Hash = H;

    fn make_iterator(&self, key: &Key, bucket_size: usize, capacity: usize) -> ProbingIterator {
        // Compute initial position: hash(key) % (capacity / bucket_size) * bucket_size
        // This aligns the start position to bucket boundaries
        let hash_value = self.hash.hash(key);
        let num_buckets = capacity / bucket_size;
        let init = ((hash_value as usize) % num_buckets) * bucket_size;
        
        // Step size is fixed at bucket_size for linear probing
        ProbingIterator::new(init, bucket_size, capacity)
    }

    fn hash_function(&self) -> Self::Hash {
        self.hash
    }

    fn cg_size(&self) -> usize {
        1 // Scalar operations for MVP
    }
}

unsafe impl<Key, H> DeviceCopy for LinearProbing<Key, H>
where
    H: Hash<Key> + Copy + DeviceCopy,
{
}

