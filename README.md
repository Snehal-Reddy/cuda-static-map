# CUDA Static Map

A GPU-accelerated, statically-sized hash map implementation for CUDA-enabled devices, written in Rust. This library provides a high-performance associative container that runs operations like insert, find, and contains directly on the GPU.

## Status

🚧 **Spinning on an atomic CAS** 🚧

The core implementation is functional and supports bulk operations and various configurations.

## Features

- **Hash map logic**: Open-addressing map with multiple collision resolution strategies.
- **Probing schemes**:
    - Linear Probing: Best for low occupancy.
    - Double Hashing: Reduces clustering and handles high-occupancy better.
- **Cooperative groups**: Uses warp-level parallelism for vectorized GPU operations, with group sizes from 1 to 32.
- **Bucket storage**: Configurable bucket sizes (1, 2, 4, 8) to help with memory alignment and access patterns.
- **Atomic operations**: Concurrent access using GPU atomics
- **Two APIs**:
    - Host Bulk API: Batch operations that launch optimized kernels.
    - Device Reference API: `StaticMapRef` for direct access inside your own CUDA kernels.
- **Hash functions**: Includes `IdentityHash`, `XXHash32`, and `XXHash64`.
- **Testing**: Integration tests covering basic operations and edge cases.

## Prerequisites

To run the Docker container with GPU support, ensure you have the following installed on your host system:

*   **NVIDIA GPU Drivers**: Your system must have compatible NVIDIA drivers installed. Verify with `nvidia-smi`.
*   **NVIDIA Container Toolkit**: This is required for Docker to interact with your NVIDIA GPU.

## Building

This project uses `xtask` for unified build management:

*   **Build Host Code**: `cargo xtask build`
*   **Build PTX (Device Code)**: `cargo xtask build-ptx`
*   **Build Everything**: `cargo xtask build-all`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
