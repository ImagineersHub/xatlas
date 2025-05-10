cmake_minimum_required(VERSION 3.10)
project(xatlas LANGUAGES CXX)

option(XA_GPU_ACCELERATION "Enable GPU acceleration using CUDA" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(XATLAS_SOURCES
    source/xatlas/xatlas.cpp
)

# Include directories
include_directories(source)

# CUDA support
if(XA_GPU_ACCELERATION)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    set(XATLAS_CUDA_SOURCES
        source/xatlas/xatlas_gpu.cu
    )
    add_definitions(-DXA_GPU_ACCELERATION=1)
    include_directories(${CUDA_INCLUDE_DIRS})

    # Set CUDA architectures if not already set
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
    endif()
endif()

# Create library
if(XA_GPU_ACCELERATION)
    add_library(xatlas ${XATLAS_SOURCES} ${XATLAS_CUDA_SOURCES})
    target_link_libraries(xatlas ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY})
else()
    add_library(xatlas ${XATLAS_SOURCES})
endif()

# Example viewer application (optional)
option(XA_BUILD_VIEWER "Build the viewer application" OFF)

if(XA_BUILD_VIEWER)
    add_subdirectory(source/viewer)
endif()

# Installation
install(TARGETS xatlas
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES source/xatlas/xatlas.h DESTINATION include)