//! GPU-side pair implementation
//! This will be the Rust equivalent of cuco::pair

/// GPU-side pair type
/// 
/// Just a placeholder for now
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Pair<First, Second> {
    pub first: First,
    pub second: Second,
}

// TODO: Implement

