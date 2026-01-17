use core::marker::PhantomData;
use cust_core::DeviceCopy;

use crate::hash::{Hash, HashOutput};

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

// Safety: ProbingIterator contains only primitive types (usize fields) and no references
// or pointers. All fields (curr_index, step_size, upper_bound) are trivially copyable
// and safe to copy to/from CUDA device memory.
unsafe impl DeviceCopy for ProbingIterator {}

/// Trait for probing schemes that define collision resolution strategies.
/// 
/// Probing schemes encapsulate both the hash function(s) and the strategy
/// for resolving collisions in open-addressing hash tables.
pub trait ProbingScheme<Key>: Copy + DeviceCopy {
    /// Type of the hash function(s) used by this probing scheme.
    /// For linear probing, this is a single hash function.
    /// For double hashing, this could be a tuple of two hash functions.
    type Hasher;

    /// Create a probing iterator for the given key.
    /// 
    /// Thread rank is automatically computed from the current thread's index in device code,
    /// or defaults to 0 in host code. This allows the same API to work for both scalar and
    /// cooperative group operations.
    /// 
    /// # Arguments
    /// * `key` - The key to create a probe sequence for
    /// * `bucket_size` - Size of each bucket (typically 1 for simple storage)
    /// * `capacity` - Total capacity of the hash table
    /// 
    /// # Returns
    /// A `ProbingIterator` that will generate slot indices in the probe sequence
    /// 
    /// # Notes
    /// - In device code, thread rank is computed as `thread::thread_idx_x() % cg_size()`
    /// - In host code, thread rank defaults to 0 (scalar operation)
    /// - This method can be called from both host and device code
    fn make_iterator(&self, key: &Key, bucket_size: usize, capacity: usize) -> ProbingIterator {
        // Compute thread_rank automatically:
        // - In device code: use actual thread index modulo cg_size
        // - In host code: default to 0 (scalar operation)
        #[cfg(target_arch = "nvptx64")]
        let thread_rank = {
            use cuda_std::thread;
            let cg_size = self.cg_size();
            (thread::thread_idx_x() as usize) % cg_size
        };
        #[cfg(not(target_arch = "nvptx64"))]
        let thread_rank = 0;
        
        self.make_iterator_with_rank(key, bucket_size, capacity, thread_rank)
    }
    
    /// Internal method that creates a probing iterator with an explicit thread rank.
    /// 
    /// This is used internally by `make_iterator` and can be used for testing or
    /// when you need explicit control over thread rank.
    fn make_iterator_with_rank(&self, key: &Key, bucket_size: usize, capacity: usize, thread_rank: usize) -> ProbingIterator;

    /// Get the hash function(s) used by this probing scheme.
    fn hash_function(&self) -> Self::Hasher;

    /// Get the cooperative group size used by this probing scheme.
    /// 
    /// For MVP, this is typically 1 (scalar operations).
    /// TODO: Support larger CG sizes for vectorized operations.
    fn cg_size(&self) -> usize;
    
    /// Returns whether this probing scheme uses double hashing.
    /// 
    /// This is used to determine if prime number lookup is needed for capacity calculation.
    /// Double hashing requires prime table sizes to ensure step sizes are coprime.
    fn is_double_hashing(&self) -> bool;
}

/// Linear probing scheme.
/// 
/// Uses a simple linear probe sequence: `(hash(key) + i * bucket_size) % capacity`
/// where `i` is the iteration number. This is efficient for low occupancy scenarios.
/// 
/// # Type Parameters
/// * `Key` - The key type that this probing scheme operates on
/// * `Hasher` - Hash function type
#[repr(C)]
#[derive(Debug)]
pub struct LinearProbing<Key, Hasher> {
    hasher: Hasher,
    _phantom: PhantomData<Key>,
}

impl<Key, Hasher> Clone for LinearProbing<Key, Hasher>
where
    Hasher: Clone,
{
    fn clone(&self) -> Self {
        Self {
            hasher: self.hasher.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<Key, Hasher> Copy for LinearProbing<Key, Hasher>
where
    Hasher: Copy,
{
}

impl<Key, Hasher> LinearProbing<Key, Hasher>
where
    Hasher: Hash<Key> + Copy + DeviceCopy,
{
    /// Create a new linear probing scheme with the given hash function.
    pub const fn new(hasher: Hasher) -> Self {
        Self {
            hasher,
            _phantom: PhantomData,
        }
    }
}

impl<Key, Hasher> ProbingScheme<Key> for LinearProbing<Key, Hasher>
where
    Hasher: Hash<Key> + Copy + DeviceCopy,
{
    type Hasher = Hasher;

    fn make_iterator(&self, key: &Key, bucket_size: usize, capacity: usize) -> ProbingIterator {
        #[cfg(target_arch = "nvptx64")]
        let thread_rank = {
            use cuda_std::thread;
            let cg_size = self.cg_size();
            (thread::thread_idx_x() as usize) % cg_size
        };
        #[cfg(not(target_arch = "nvptx64"))]
        let thread_rank = 0;

        self.make_iterator_with_rank(key, bucket_size, capacity, thread_rank)
    }

    fn make_iterator_with_rank(&self, key: &Key, bucket_size: usize, capacity: usize, thread_rank: usize) -> ProbingIterator {
        let cg_size = self.cg_size();
        let stride = bucket_size * cg_size;
        
        // Compute group-aligned starting position
        // hash % (capacity / stride) gives us the group index
        // * stride aligns it to a group boundary
        let hash_value = self.hasher.hash(key);
        let num_groups = capacity / stride;
        let init_base = ((hash_value.to_usize()) % num_groups) * stride;
        
        // Add thread-specific offset within the group
        // Each thread gets its own bucket within the aligned chunk
        let init = init_base + thread_rank * bucket_size;
        
        // Step size is stride for cooperative groups (all threads advance together)
        // For scalar case (cg_size == 1), stride == bucket_size, so it's equivalent
        ProbingIterator::new(init, stride, capacity)
    }

    fn hash_function(&self) -> Self::Hasher {
        self.hasher
    }

    fn cg_size(&self) -> usize {
        1 // Scalar operations for MVP
    }
    
    fn is_double_hashing(&self) -> bool {
        false
    }
}

// Safety: LinearProbing contains:
// - `hasher: Hasher` which implements DeviceCopy (by trait bound)
// - `_phantom: PhantomData<Key>` which is a zero-sized type
// Both fields are trivially copyable and contain no references to CPU memory.
// The Key type parameter is only used in PhantomData, which doesn't affect memory layout.
unsafe impl<Key, Hasher> DeviceCopy for LinearProbing<Key, Hasher>
where
    Hasher: Copy + DeviceCopy,
{
}

/// Double hashing probing scheme.
/// 
/// Uses two hash functions to compute probe sequence:
/// - Initial position: `hash1(key) % (capacity / stride) * stride + thread_rank * bucket_size`
/// - Step size: `(hash2(key) % (num_groups - 1) + 1) * stride`
/// 
/// This reduces clustering compared to linear probing and is superior for
/// high occupancy/high multiplicity scenarios.
/// 
/// # Type Parameters
/// * `Key` - The key type that this probing scheme operates on
/// * `Hasher1` - First hash function type that implements `Hash<Key>`
/// * `Hasher2` - Second hash function type that implements `Hash<Key>`
#[repr(C)]
#[derive(Debug)]
pub struct DoubleHashProbing<Key, Hasher1, Hasher2> {
    hasher1: Hasher1,
    hasher2: Hasher2,
    _phantom: PhantomData<Key>,
}

impl<Key, Hasher1, Hasher2> Clone for DoubleHashProbing<Key, Hasher1, Hasher2>
where
    Hasher1: Hash<Key> + Clone,
    Hasher2: Hash<Key> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            hasher1: self.hasher1.clone(),
            hasher2: self.hasher2.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<Key, Hasher1, Hasher2> Copy for DoubleHashProbing<Key, Hasher1, Hasher2>
where
    Hasher1: Hash<Key> + Copy,
    Hasher2: Hash<Key> + Copy,
{
}

impl<Key, Hasher1, Hasher2> DoubleHashProbing<Key, Hasher1, Hasher2>
where
    Hasher1: Hash<Key> + Copy + DeviceCopy,
    Hasher2: Hash<Key> + Copy + DeviceCopy,
{
    /// Create a new double hashing scheme with the given hash functions.
    pub const fn new(hasher1: Hasher1, hasher2: Hasher2) -> Self {
        Self {
            hasher1,
            hasher2,
            _phantom: PhantomData,
        }
    }
    
}

impl<Key, Hasher1, Hasher2> ProbingScheme<Key> for DoubleHashProbing<Key, Hasher1, Hasher2> 
where
    Hasher1: Hash<Key> + Copy + DeviceCopy,
    Hasher2: Hash<Key> + Copy + DeviceCopy, {
    
    type Hasher = (Hasher1, Hasher2);

    fn make_iterator(&self, key: &Key, bucket_size: usize, capacity: usize) -> ProbingIterator {
        #[cfg(target_arch = "nvptx64")]
        let thread_rank = {
            use cuda_std::thread;
            let cg_size = self.cg_size();
            (thread::thread_idx_x() as usize) % cg_size
        };
        #[cfg(not(target_arch = "nvptx64"))]
        let thread_rank = 0;

        self.make_iterator_with_rank(key, bucket_size, capacity, thread_rank)
    }

    fn make_iterator_with_rank(&self, key: &Key, bucket_size: usize, capacity: usize, thread_rank: usize) -> ProbingIterator {
        let cg_size = self.cg_size();
        let stride = bucket_size * cg_size;
        
        // Compute initial position using first hash function
        let hash1_value = self.hasher1.hash(key);
        let num_groups = capacity / stride;
        let init_base = ((hash1_value.to_usize()) % num_groups) * stride;
        let init = init_base + thread_rank * bucket_size;
        
        // Compute step size using second hash function
        // Step size must be in range [1, num_groups - 1] to ensure full coverage
        // For num_groups == 1, we use step_size = stride to avoid division by zero
        let hash2_value = self.hasher2.hash(key);
        let step_base = if num_groups > 1 {
            ((hash2_value.to_usize()) % (num_groups - 1)) + 1
        } else {
            1
        };
        let step_size = step_base * stride;
        
        ProbingIterator::new(init, step_size, capacity)
    }

    fn hash_function(&self) -> Self::Hasher {
        (self.hasher1, self.hasher2)
    }

    fn cg_size(&self) -> usize {
        1 // Scalar operations for MVP
    }
    
    fn is_double_hashing(&self) -> bool {
        true
    }
}

// Safety: DoubleHashProbing contains:
// - `hasher1: Hasher1` which implements DeviceCopy (by trait bound)
// - `hasher2: Hasher2` which implements DeviceCopy (by trait bound)
// - `_phantom: PhantomData<Key>` which is a zero-sized type
// All fields are trivially copyable and contain no references to CPU memory.
// The Key type parameter is only used in PhantomData, which doesn't affect memory layout.
unsafe impl<Key, Hasher1, Hasher2> DeviceCopy for DoubleHashProbing<Key, Hasher1, Hasher2>
where
    Hasher1: Hash<Key> + Copy + DeviceCopy,
    Hasher2: Hash<Key> + Copy + DeviceCopy,
{
}
