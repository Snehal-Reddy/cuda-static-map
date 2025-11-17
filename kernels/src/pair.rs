//! Shared pair implementation for both CPU and GPU
//!
//! This code compiles for both host (CPU) and device (GPU) targets.
//! When compiled for GPU (nvptx64-nvidia-cuda), it generates PTX code.
//! When compiled for CPU, it's used as a regular Rust library.

use core::mem::{size_of, zeroed};
use cust_core::DeviceCopy;

/// Computes the alignment for a pair type.
pub const fn alignment<First, Second>() -> usize {
    let x = (size_of::<First>() + size_of::<Second>()).next_power_of_two();
    // Const-compatible min: if x > 16, return 16, else return x
    if x > 16 { 16 } else { x }
}

/// Trait for mapping alignment values to aligned types
pub trait AlignedTo<const ALIGN: usize> {
    type Aligned: Clone + Copy + core::fmt::Debug + cust_core::DeviceCopy;
}

/// Type alias to get the aligned type for a given alignment value
pub type Aligned<const ALIGN: usize> = <() as AlignedTo<ALIGN>>::Aligned;

/// Macro to generate `AlignedTo` implementations for specific alignment values
macro_rules! aligned_to {
    ( $($align:literal),* $(,)? ) => {
        $(
            const _: () = {
                #[repr(align($align))]
                #[derive(Clone, Copy, Debug)]
                pub struct Aligned;
                
                impl AlignedTo<$align> for () {
                    type Aligned = Aligned;
                }
                
                unsafe impl cust_core::DeviceCopy for Aligned {}
            };
        )*
    }
}

// Generate implementations for all possible alignment values (1, 2, 4, 8, 16, 32)
// The alignment function returns min(16, next_power_of_two(size)), so these cover all cases
aligned_to!(1, 2, 4, 8, 16, 32);

/// Pair type for both device and host code
#[repr(C)]
#[derive(Clone, Copy, Debug, DeviceCopy)]
pub struct Pair<First, Second>
where
    (): AlignedTo<{ alignment::<First, Second>() }>,
{
    _marker: Aligned<{ alignment::<First, Second>() }>,
    pub first: First,
    pub second: Second,
}

impl<First, Second> Pair<First, Second>
where
    (): AlignedTo<{ alignment::<First, Second>() }>,
{
    /// Create a new pair
    ///
    /// This method works on both CPU and GPU.
    pub fn new(first: First, second: Second) -> Self {
        Self {
            _marker: unsafe { zeroed::<Aligned<{ alignment::<First, Second>() }>>() },
            first,
            second,
        }
    }
}

/// Trait for types that can be treated as pairs.
///
/// This trait enables generic code to work with different pair-like types.
pub trait PairLike {
    /// Type of the first element
    type First;
    /// Type of the second element
    type Second;

    /// Get a reference to the first element
    fn first(&self) -> &Self::First;
    /// Get a reference to the second element
    fn second(&self) -> &Self::Second;

    /// Convert to owned values (requires Clone)
    fn into_pair(self) -> (Self::First, Self::Second)
    where
        Self: Sized,
        Self::First: Clone,
        Self::Second: Clone,
    {
        (self.first().clone(), self.second().clone())
    }
}

// Implement PairLike for standard Rust tuples (A, B)
impl<T1, T2> PairLike for (T1, T2) {
    type First = T1;
    type Second = T2;

    fn first(&self) -> &Self::First {
        &self.0
    }

    fn second(&self) -> &Self::Second {
        &self.1
    }
}

// Implement PairLike for our custom Pair type
impl<First, Second> PairLike for Pair<First, Second>
where
    (): AlignedTo<{ alignment::<First, Second>() }>,
{
    type First = First;
    type Second = Second;

    fn first(&self) -> &Self::First {
        &self.first
    }

    fn second(&self) -> &Self::Second {
        &self.second
    }
}

// Enable conversion from tuples using From trait
impl<First, Second> From<(First, Second)> for Pair<First, Second>
where
    (): AlignedTo<{ alignment::<First, Second>() }>,
{
    fn from((first, second): (First, Second)) -> Self {
        Self::new(first, second)
    }
}

/// Type detection trait for tuple-like types.
pub trait IsTupleLike {}

impl<T1, T2> IsTupleLike for (T1, T2) {}
impl<First, Second> IsTupleLike for Pair<First, Second>
where
    (): AlignedTo<{ alignment::<First, Second>() }>,
{}

// GPU-specific implementations (only compiled when targeting nvptx64)
#[cfg(target_arch = "nvptx64")]
impl<First, Second> Pair<First, Second>
where
    (): AlignedTo<{ alignment::<First, Second>() }>,
{
    // Add GPU-specific methods here if needed in the future
}

// CPU-specific implementations (only compiled when NOT targeting nvptx64)
#[cfg(not(target_arch = "nvptx64"))]
impl<First, Second> Pair<First, Second>
where
    (): AlignedTo<{ alignment::<First, Second>() }>,
{
    // Add CPU-specific methods here if needed in the future
}
