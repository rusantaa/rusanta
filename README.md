# Rusanta

## Native Rust Data Science, Machine Learning, Visualization & GPU Computing Ecosystem

Rusanta is a modular Rust-native scientific computing ecosystem designed
to provide a complete alternative stack for data science, analytics,
visualization, machine learning, and GPU acceleration without depending
on Python runtimes.

The project is inspired by the roles of NumPy, Pandas, Matplotlib,
Seaborn, Scikit-learn, XGBoost, LightGBM, PyTorch, and Triton, but
rebuilt around Rust's strengths:

-   Memory safety
-   Native performance
-   Zero-cost abstractions
-   Parallel execution
-   Systems-level deployment

------------------------------------------------------------------------

# Workspace Architecture

    Rusanta
    ├── rusanta-array
    ├── rusanta-core
    ├── rusanta-frame
    ├── rusanta-viz
    ├── rusanta-stat-viz
    ├── rusanta-ml
    ├── rusanta-triton

Dependency flow:

                    Applications
                         |
                         v
                  rusanta-ml
                         |
                  rusanta-frame
                         |
                  rusanta-array
                         |
                  rusanta-core
                         |
                  rusanta-triton
                         |
                     GPU Layer

Visualization branches:

    rusanta-array
          |
          +---- rusanta-viz
          |
          +---- rusanta-stat-viz

------------------------------------------------------------------------

# Crate Overview

## rusanta-core

The foundation crate.

Responsibilities:

-   Shared traits
-   Error handling
-   Common types
-   Memory abstractions
-   Utilities used by every crate

Everything depends on core.

------------------------------------------------------------------------

# rusanta-array

The numerical foundation layer.

Role:

Rust-native multidimensional array computing.

Equivalent roles:

-   NumPy ndarray
-   Tensor storage layer

Features:

-   N-dimensional arrays
-   Shape management
-   Indexing
-   Broadcasting
-   Mathematical operations
-   Memory-efficient storage

Used by:

-   DataFrame operations
-   Machine learning tensors
-   GPU buffers

------------------------------------------------------------------------

# rusanta-frame

DataFrame and data manipulation engine.

Equivalent roles:

-   Pandas DataFrame
-   Polars DataFrame

Provides:

-   DataFrame
-   Series
-   Index system
-   CSV IO
-   JSON IO
-   Parquet IO
-   XLSX IO
-   TSV IO
-   TOON format IO
-   Aggregation operations
-   Data transformations

Example:

``` rust
let df = DataFrame::from_csv("data.csv")?;

let result = df
    .groupby("category")
    .mean("value");
```

------------------------------------------------------------------------

# rusanta-viz

General purpose visualization engine.

Equivalent roles:

-   Matplotlib
-   Plotters

Provides:

    figure
     └── axes
          ├── line
          ├── scatter
          ├── bar
          └── histogram

Backends:

-   SVG
-   PNG
-   WGPU GPU rendering

Designed for:

-   Scientific plots
-   Reports
-   Dashboards

------------------------------------------------------------------------

# rusanta-stat-viz

Statistical visualization layer.

Built on top of rusanta-viz.

Provides:

-   Categorical plots
-   Distribution visualization
-   Regression plots
-   Statistical themes

Conceptually similar to:

-   Seaborn
-   ggplot statistical layers

------------------------------------------------------------------------

# rusanta-ml

Machine learning framework.

Equivalent roles:

-   Scikit-learn
-   XGBoost ecosystem

Implemented algorithms include:

## Supervised Learning

-   Linear Regression
-   Logistic Regression
-   SVM
-   KNN
-   Decision Trees
-   Random Forest
-   Naive Bayes
    -   Gaussian
    -   Bernoulli
    -   Multinomial

## Gradient Boosting

-   XGBoost Regression
-   XGBoost Classification
-   LightGBM style histogram boosting
-   CatBoost style boosting
-   AdaBoost

## Clustering

-   K-Means
-   DBSCAN
-   OPTICS
-   Hierarchical clustering

## Online Learning

-   FTRL-Proximal

Supports:

-   L1 regularization
-   L2 regularization
-   Incremental updates

## Dimensionality Reduction

-   PCA
-   ICA
-   t-SNE Barnes-Hut approximation

## Feature Methods

-   Kernel mapping
-   Polynomial expansion
-   RBF mapping

## Regression Algorithms

-   LARS
-   Partial Least Squares

------------------------------------------------------------------------

# rusanta-triton

GPU compiler and execution layer.

Equivalent roles:

-   CUDA runtime
-   Triton language concepts
-   GPU compute backend

Backends:

## WGPU

Portable GPU compute:

-   Vulkan
-   Metal
-   DX12
-   WebGPU

## CUDA

NVIDIA acceleration:

-   CUDA kernels
-   PTX
-   NVRTC compilation

Future targets:

-   ROCm
-   HIP
-   Metal native
-   Vulkan compute

------------------------------------------------------------------------

# Complete Data Science Pipeline

Example architecture:

    Raw Data
       |
       v
    rusanta-frame
       |
       v
    rusanta-array
       |
       v
    rusanta-ml
       |
       v
    rusanta-triton
       |
       v
    GPU acceleration

Visualization:

    DataFrame
        |
        v
    stat-viz
        |
        v
    viz
        |
        v
    SVG / PNG / GPU

------------------------------------------------------------------------

# Design Philosophy

## Rust First

No Python interpreter dependency.

## Modular

Each crate has one responsibility.

## Hardware Aware

CPU and GPU execution can coexist.

## Production Ready

Designed for:

-   Embedded analytics
-   Server applications
-   Scientific computing
-   AI systems
-   High performance pipelines

------------------------------------------------------------------------

# Future Roadmap

## Core

-   SIMD acceleration
-   Better memory allocator
-   Parallel execution runtime

## Array

-   GPU tensors
-   Automatic differentiation

## Frame

-   Lazy execution engine
-   Query optimizer

## ML

-   Neural network module
-   Automatic model selection
-   Distributed training

## Triton

-   Kernel compiler
-   Automatic optimization
-   GPU graph execution

------------------------------------------------------------------------

# License

Part of the Rusanta project.
