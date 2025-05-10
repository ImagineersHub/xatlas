# Installation Instructions for GPU Acceleration

This document provides detailed instructions for installing and setting up GPU acceleration in xatlas.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 10.0 or newer
- CMake 3.10 or newer

## Installation Steps

1. **Apply the GPU acceleration patch:**

   ```bash
   cd /path/to/xatlas
   git apply gpu_acceleration.patch
   ```

2. **Create and customize the CMake build files:**

   ```bash
   mkdir build
   cd build
   cmake -DXA_GPU_ACCELERATION=ON ..
   ```

3. **Build the library:**

   ```bash
   make
   ```

   Or on Windows with Visual Studio:

   ```bash
   cmake --build . --config Release
   ```

4. **Installation (optional):**

   ```bash
   make install
   ```

## Integration into Existing Projects

If you're integrating xatlas with GPU acceleration into an existing project, you need to:

1. Include `xatlas.h` in your project
2. Define `XA_GPU_ACCELERATION` as 1 before including the header
3. Link against CUDA libraries (cublas, cusparse, cusolver)

Example CMake configuration for your project:

```cmake
# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Define GPU acceleration
add_definitions(-DXA_GPU_ACCELERATION=1)

# Add xatlas with GPU acceleration
add_subdirectory(path/to/xatlas)

# Link your target
target_link_libraries(your_target xatlas ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY})
```

## Testing GPU Acceleration

To verify that GPU acceleration is working correctly:

1. Create a test application that uses xatlas
2. Enable GPU acceleration in the chart options:
   ```cpp
   xatlas::ChartOptions chartOptions;
   chartOptions.useGPU = true;
   ```
3. Run the application and look for the message "Using GPU acceleration for LSCM" in the console output

## Troubleshooting

If you encounter issues with GPU acceleration:

1. **CUDA not found:** Make sure CUDA toolkit is installed and in your PATH
2. **Build errors:** Ensure you're using a compatible compiler for your CUDA version
3. **Runtime errors:** Check that your GPU is supported and has up-to-date drivers
4. **Performance issues:** Try different mesh sizes to find the optimal performance threshold between CPU and GPU implementations
