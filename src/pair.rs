//! Host-side pair type

/// Host-side pair type
/// 
/// This is used on the CPU side for constructing pairs before
/// sending them to the GPU.
#[derive(Clone, Debug)]
pub struct Pair<First, Second> {
    pub first: First,
    pub second: Second,
}

impl<First, Second> Pair<First, Second> {
    /// Create a new pair
    pub fn new(first: First, second: Second) -> Self {
        Self { first, second }
    }
}

// TODO: Implement conversion to/from GPU pair type if needed
// TODO: Implement serialization/deserialization if needed

