Rusanta API Design

Overview

Rusanta is a Rust-native scientific computing ecosystem designed as a
modular collection of crates.

Architecture:

Application | Machine Learning | rusanta-ml / rusanta-nn |
rusanta-tensor | rusanta-array | rusanta-core

Design Principles

Rust First: Rusanta focuses on native Rust implementation instead of
wrapping existing frameworks.

Goals: - Memory safety - Predictable performance - Zero-cost
abstractions - Strong Rust ecosystem integration

Crate Responsibilities:

rusanta-core: Foundation utilities, traits, and shared infrastructure.

rusanta-array: Numerical arrays and scientific computing primitives.

rusanta-frame: Data processing and dataframe functionality.

rusanta-tensor: Tensor engine, automatic differentiation, and
optimization.

rusanta-ml: Machine learning algorithms.

rusanta-viz: Visualization tools.

rusanta-stat-viz: Statistical visualization.

rusanta-triton: GPU compiler and execution layer.

Tensor API Philosophy

Inspired by NumPy, PyTorch, and JAX while maintaining Rust safety.

Automatic Differentiation:

Tensor -> Operation -> Graph Node -> Backward Pass -> Gradient

Backend Design:

Tensor | Backend | CPU / GPU

Future backends: - Vulkan - Metal - ROCm

Versioning: Rusanta follows semantic versioning: MAJOR.MINOR.PATCH
