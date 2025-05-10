# GPU Acceleration for xatlas

This document explains how to use GPU acceleration for UV unwrapping in xatlas.

## Overview

The GPU acceleration feature uses CUDA to accelerate the computation-intensive parts of the UV unwrapping process, specifically the LSCM (Least Squares Conformal Maps) parameterization. This can significantly improve performance for large meshes.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 10.0 or newer
- CMake 3.10 or newer

## Building with GPU Acceleration

To build xatlas with GPU acceleration support:

```bash
mkdir build
cd build
cmake -DXA_GPU_ACCELERATION=ON ..
make
```

## Using GPU Acceleration

To use GPU acceleration in your code, set the `useGPU` option to `true` in the `ChartOptions` structure:

```cpp
xatlas::ChartOptions chartOptions;
chartOptions.useGPU = true;
```

When this option is enabled, xatlas will automatically try to use the GPU for the LSCM parameterization. If GPU acceleration is not available or fails, it will automatically fall back to the CPU implementation.

## Implementation Details

The GPU acceleration is implemented in the following files:

- `source/xatlas/xatlas_gpu.h`: Header file with GPU-related declarations
- `source/xatlas/xatlas_gpu.cu`: CUDA implementation of the accelerated algorithms

The main accelerated functions are:

1. `computeLSCM`: Accelerates the Least Squares Conformal Maps parameterization
2. `computeChartMetrics`: Accelerates the computation of chart quality metrics

## Performance

Performance improvements depend on the size and complexity of the mesh:

- For small meshes (<10k triangles), the overhead of transferring data to the GPU might outweigh the benefits
- For medium meshes (10k-100k triangles), expect 2-5x speedup
- For large meshes (>100k triangles), expect 5-10x speedup or more

## Limitations

Current limitations of the GPU acceleration implementation:

1. Only the LSCM parameterization algorithm is accelerated
2. Not all chart quality metrics are computed on the GPU
3. Requires a CUDA-capable NVIDIA GPU

## Troubleshooting

If you encounter issues with GPU acceleration:

1. Make sure your GPU drivers are up to date
2. Check that CUDA is properly installed and configured
3. Try disabling GPU acceleration to see if the issue is related to the GPU implementation
4. Check console output for any CUDA-related error messages
