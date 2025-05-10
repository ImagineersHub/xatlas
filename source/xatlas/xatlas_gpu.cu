#include "xatlas.h"

#if XA_GPU_ACCELERATION

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolver_common.h>
#include <cusolverSp.h>

#include "xatlas_gpu.h"

namespace xatlas
{
    namespace gpu
    {

        // CUDA error checking helper function
        static bool checkCudaError(cudaError_t error, const char *functionName)
        {
            if (error != cudaError_t::cudaSuccess)
            {
                XA_PRINT("CUDA error in %s: %s\n", functionName, cudaGetErrorString(error));
                return false;
            }
            return true;
        }

        static int deviceCount = 0;
        static bool initialized = false;
        static cublasHandle_t cublasHandle = nullptr;
        static cusparseHandle_t cusparseHandle = nullptr;
        static cusolverSpHandle_t cusolverHandle = nullptr;

        bool isGpuAvailable()
        {
            if (!initialized)
            {
                cudaError_t error = cudaGetDeviceCount(&deviceCount);
                initialized = true;
                if (error != cudaSuccess)
                {
                    deviceCount = 0;
                    return false;
                }
            }
            return deviceCount > 0;
        }

        bool initialize()
        {
            if (!isGpuAvailable())
                return false;

            if (cublasHandle != nullptr)
                return true; // Already initialized

            // Initialize CUDA libraries
            if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
            {
                XA_PRINT("Failed to initialize cuBLAS\n");
                return false;
            }

            if (cusparseCreate(&cusparseHandle) != CUSPARSE_STATUS_SUCCESS)
            {
                XA_PRINT("Failed to initialize cuSPARSE\n");
                cublasDestroy(cublasHandle);
                cublasHandle = nullptr;
                return false;
            }

            if (cusolverSpCreate(&cusolverHandle) != CUSOLVER_STATUS_SUCCESS)
            {
                XA_PRINT("Failed to initialize cuSOLVER\n");
                cublasDestroy(cublasHandle);
                cusparseDestroy(cusparseHandle);
                cublasHandle = nullptr;
                cusparseHandle = nullptr;
                return false;
            }

            XA_PRINT("GPU acceleration initialized successfully\n");
            return true;
        }

        void shutdown()
        {
            if (cublasHandle != nullptr)
            {
                cublasDestroy(cublasHandle);
                cublasHandle = nullptr;
            }

            if (cusparseHandle != nullptr)
            {
                cusparseDestroy(cusparseHandle);
                cusparseHandle = nullptr;
            }

            if (cusolverHandle != nullptr)
            {
                cusolverSpDestroy(cusolverHandle);
                cusolverHandle = nullptr;
            }
        }

        // CUDA kernels and device functions
        __global__ void setupSystemKernel(
            int *d_rowPtr, int *d_colIdx, float *d_values, float *d_b,
            const float *d_positions, const int *d_indices,
            int vertexCount, int faceCount,
            int lockedVertex0, int lockedVertex1,
            float2 initUv0, float2 initUv1)
        {
            int faceIdx = blockIdx.x * blockDim.x + threadIdx.x;

            if (faceIdx >= faceCount)
                return;

            // Process triangle faces - build matrix and right-hand side
            int v0 = d_indices[faceIdx * 3];
            int v1 = d_indices[faceIdx * 3 + 1];
            int v2 = d_indices[faceIdx * 3 + 2];

            // Get vertex positions
            float3 p0 = make_float3(d_positions[v0 * 3], d_positions[v0 * 3 + 1], d_positions[v0 * 3 + 2]);
            float3 p1 = make_float3(d_positions[v1 * 3], d_positions[v1 * 3 + 1], d_positions[v1 * 3 + 2]);
            float3 p2 = make_float3(d_positions[v2 * 3], d_positions[v2 * 3 + 1], d_positions[v2 * 3 + 2]);

            // Project triangle to 2D
            float3 edge1 = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
            float3 edge2 = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);

            // Triangle normal
            float3 normal = make_float3(
                edge1.y * edge2.z - edge1.z * edge2.y,
                edge1.z * edge2.x - edge1.x * edge2.z,
                edge1.x * edge2.y - edge1.y * edge2.x);

            // Normalize normal
            float len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
            if (len > 1e-10f)
            {
                normal.x /= len;
                normal.y /= len;
                normal.z /= len;
            }

            // Project to 2D
            float2 z0 = make_float2(0.0f, 0.0f);
            float2 z1, z2;

            // Choose a suitable tangent vector (perpendicular to normal)
            float3 tangent;
            if (fabsf(normal.x) < fabsf(normal.y) && fabsf(normal.x) < fabsf(normal.z))
                tangent = make_float3(1, 0, 0);
            else if (fabsf(normal.y) < fabsf(normal.z))
                tangent = make_float3(0, 1, 0);
            else
                tangent = make_float3(0, 0, 1);

            // Make tangent orthogonal to normal
            float dot = tangent.x * normal.x + tangent.y * normal.y + tangent.z * normal.z;
            tangent.x -= dot * normal.x;
            tangent.y -= dot * normal.y;
            tangent.z -= dot * normal.z;

            // Normalize tangent
            len = sqrtf(tangent.x * tangent.x + tangent.y * tangent.y + tangent.z * tangent.z);
            if (len > 1e-10f)
            {
                tangent.x /= len;
                tangent.y /= len;
                tangent.z /= len;
            }

            // Compute bitangent
            float3 bitangent = make_float3(
                normal.y * tangent.z - normal.z * tangent.y,
                normal.z * tangent.x - normal.x * tangent.z,
                normal.x * tangent.y - normal.y * tangent.x);

            // Project vertices to 2D plane
            z1.x = (p1.x - p0.x) * tangent.x + (p1.y - p0.y) * tangent.y + (p1.z - p0.z) * tangent.z;
            z1.y = (p1.x - p0.x) * bitangent.x + (p1.y - p0.y) * bitangent.y + (p1.z - p0.z) * bitangent.z;

            z2.x = (p2.x - p0.x) * tangent.x + (p2.y - p0.y) * tangent.y + (p2.z - p0.z) * tangent.z;
            z2.y = (p2.x - p0.x) * bitangent.x + (p2.y - p0.y) * bitangent.y + (p2.z - p0.z) * bitangent.z;

            float a = z1.x;
            float b = z1.y;
            float c = z2.x;
            float d = z2.y;

            // Create matrix entries for this triangle
            // For vertex indices: 2*v+0 is u coordinate, 2*v+1 is v coordinate
            int u0_id = 2 * v0;
            int v0_id = 2 * v0 + 1;
            int u1_id = 2 * v1;
            int v1_id = 2 * v1 + 1;
            int u2_id = 2 * v2;
            int v2_id = 2 * v2 + 1;

            // Lock vertices if needed
            if (v0 == lockedVertex0)
            {
                // Handle locked vertex 0
                d_b[u0_id] = initUv0.x;
                d_b[v0_id] = initUv0.y;
            }

            if (v1 == lockedVertex0)
            {
                // Handle locked vertex 0
                d_b[u1_id] = initUv0.x;
                d_b[v1_id] = initUv0.y;
            }

            if (v2 == lockedVertex0)
            {
                // Handle locked vertex 0
                d_b[u2_id] = initUv0.x;
                d_b[v2_id] = initUv0.y;
            }

            if (v0 == lockedVertex1)
            {
                // Handle locked vertex 1
                d_b[u0_id] = initUv1.x;
                d_b[v0_id] = initUv1.y;
            }

            if (v1 == lockedVertex1)
            {
                // Handle locked vertex 1
                d_b[u1_id] = initUv1.x;
                d_b[v1_id] = initUv1.y;
            }

            if (v2 == lockedVertex1)
            {
                // Handle locked vertex 1
                d_b[u2_id] = initUv1.x;
                d_b[v2_id] = initUv1.y;
            }

            // Add matrix entries for this triangle
            // We're building A^T A x = A^T b linear system

            // Row 1 (real part)
            atomicAdd(&d_values[u0_id * vertexCount * 2 + u0_id], (-a + c) * (-a + c) + (b - d) * (b - d));
            atomicAdd(&d_values[u0_id * vertexCount * 2 + v0_id], (-a + c) * (b - d) + (b - d) * (-a + c));
            atomicAdd(&d_values[u0_id * vertexCount * 2 + u1_id], (-a + c) * (-c) + (b - d) * d);
            atomicAdd(&d_values[u0_id * vertexCount * 2 + v1_id], (-a + c) * d + (b - d) * (-c));
            atomicAdd(&d_values[u0_id * vertexCount * 2 + u2_id], (-a + c) * a);

            // Row 2 (imaginary part)
            atomicAdd(&d_values[v0_id * vertexCount * 2 + u0_id], (b - d) * (-a + c) + (-a + c) * (b - d));
            atomicAdd(&d_values[v0_id * vertexCount * 2 + v0_id], (b - d) * (b - d) + (-a + c) * (-a + c));
            atomicAdd(&d_values[v0_id * vertexCount * 2 + u1_id], (b - d) * (-c) + (-a + c) * d);
            atomicAdd(&d_values[v0_id * vertexCount * 2 + v1_id], (b - d) * d + (-a + c) * (-c));
            atomicAdd(&d_values[v0_id * vertexCount * 2 + v2_id], (b - d) * a);

            // Remaining rows follow the same pattern
            // (Implementation simplified for clarity - full matrix construction would be needed)
        }

        bool computeLSCM(const internal::Mesh *mesh, const internal::Array<uint32_t> &faces,
                         internal::Array<internal::Vector2> &texcoords)
        {
            if (!isGpuAvailable())
            {
                XA_PRINT("GPU acceleration not available\n");
                return false;
            }

            if (!initialize())
            {
                XA_PRINT("Failed to initialize GPU acceleration\n");
                return false;
            }

            // Find locked vertices (similar to CPU implementation)
            uint32_t lockedVertex0 = 0, lockedVertex1 = 0;

            // Find two vertices that are farthest apart to pin
            float maxDist = 0.0f;
            const uint32_t vertexCount = mesh->vertexCount();
            for (uint32_t i = 0; i < vertexCount; i++)
            {
                if (!mesh->isBoundaryVertex(i))
                    continue;

                for (uint32_t j = i + 1; j < vertexCount; j++)
                {
                    if (!mesh->isBoundaryVertex(j))
                        continue;

                    const internal::Vector3 &p1 = mesh->position(i);
                    const internal::Vector3 &p2 = mesh->position(j);

                    float dist = internal::length(p2 - p1);
                    if (dist > maxDist)
                    {
                        maxDist = dist;
                        lockedVertex0 = i;
                        lockedVertex1 = j;
                    }
                }
            }

            if (maxDist <= 0.0f)
            {
                // Couldn't find boundary vertices, mesh has no boundaries
                return false;
            }

            // Set fixed positions for the two locked vertices
            float2 initUv0 = make_float2(0.0f, 0.0f);
            float2 initUv1 = make_float2(1.0f, 0.0f);

            // Allocate device memory
            const uint32_t faceCount = faces.size() / 3;
            const uint32_t systemSize = vertexCount * 2; // 2 coordinates (u,v) per vertex

            // Position data
            float *d_positions = nullptr;
            size_t positionsSize = vertexCount * 3 * sizeof(float);
            if (cudaMalloc(&d_positions, positionsSize) != cudaSuccess)
            {
                XA_PRINT("Failed to allocate GPU memory for positions\n");
                return false;
            }

            // Copy positions to device
            float *h_positions = new float[vertexCount * 3];
            for (uint32_t i = 0; i < vertexCount; i++)
            {
                const internal::Vector3 &pos = mesh->position(i);
                h_positions[i * 3] = pos.x;
                h_positions[i * 3 + 1] = pos.y;
                h_positions[i * 3 + 2] = pos.z;
            }

            cudaMemcpy(d_positions, h_positions, positionsSize, cudaMemcpyHostToDevice);
            delete[] h_positions;

            // Index data
            int *d_indices = nullptr;
            size_t indicesSize = faceCount * 3 * sizeof(int);
            if (cudaMalloc(&d_indices, indicesSize) != cudaSuccess)
            {
                cudaFree(d_positions);
                XA_PRINT("Failed to allocate GPU memory for indices\n");
                return false;
            }

            // Copy indices to device
            int *h_indices = new int[faceCount * 3];
            for (uint32_t i = 0; i < faceCount; i++)
            {
                for (uint32_t j = 0; j < 3; j++)
                {
                    h_indices[i * 3 + j] = faces[i * 3 + j];
                }
            }

            cudaMemcpy(d_indices, h_indices, indicesSize, cudaMemcpyHostToDevice);
            delete[] h_indices;

            // Sparse matrix in CSR format
            int *d_rowPtr = nullptr;
            int *d_colIdx = nullptr;
            float *d_values = nullptr;
            float *d_b = nullptr;
            float *d_x = nullptr;

            // We're using a simple dense matrix for this example
            // In a production implementation, this should be a sparse matrix
            if (cudaMalloc(&d_values, systemSize * systemSize * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&d_b, systemSize * sizeof(float)) != cudaSuccess ||
                cudaMalloc(&d_x, systemSize * sizeof(float)) != cudaSuccess)
            {
                cudaFree(d_positions);
                cudaFree(d_indices);
                XA_PRINT("Failed to allocate GPU memory for system matrix\n");
                return false;
            }

            // Initialize to zero
            cudaMemset(d_values, 0, systemSize * systemSize * sizeof(float));
            cudaMemset(d_b, 0, systemSize * sizeof(float));

            // Set up initial values for u,v coordinates
            cudaMemset(d_x, 0, systemSize * sizeof(float));

            // Launch kernel to set up system
            int threadsPerBlock = 256;
            int numBlocks = (faceCount + threadsPerBlock - 1) / threadsPerBlock;

            setupSystemKernel<<<numBlocks, threadsPerBlock>>>(
                d_rowPtr, d_colIdx, d_values, d_b,
                d_positions, d_indices,
                vertexCount, faceCount,
                lockedVertex0, lockedVertex1,
                initUv0, initUv1);

            // Check for kernel errors
            if (!checkCudaError(cudaGetLastError(), "setupSystemKernel"))
            {
                cudaFree(d_positions);
                cudaFree(d_indices);
                cudaFree(d_values);
                cudaFree(d_b);
                cudaFree(d_x);
                return false;
            }

            // Wait for kernel to finish
            cudaDeviceSynchronize();

            // For a production implementation, we would solve the sparse system using cuSOLVER
            // This is a simplified version that just demonstrates the approach

            // Copy solution back from device
            float *h_x = new float[systemSize];
            cudaMemcpy(h_x, d_x, systemSize * sizeof(float), cudaMemcpyDeviceToHost);

            // Copy solution to texcoords
            texcoords.resize(vertexCount);
            for (uint32_t i = 0; i < vertexCount; i++)
            {
                texcoords[i].x = h_x[i * 2];
                texcoords[i].y = h_x[i * 2 + 1];
            }

            // Clean up
            delete[] h_x;
            cudaFree(d_positions);
            cudaFree(d_indices);
            cudaFree(d_values);
            cudaFree(d_b);
            cudaFree(d_x);

            // For now, return false to fall back to CPU implementation
            // In a complete implementation, return true when the system is solved properly
            return false;
        }

        // GPU implementation of chart quality metrics calculation
        bool computeChartMetrics(const internal::Mesh *mesh, internal::param::Quality &quality)
        {
            // This would compute flipped triangles, boundary intersections, etc. on the GPU
            // For now, return false to fall back to CPU implementation
            return false;
        }

    } // namespace gpu
} // namespace xatlas

#endif // XA_GPU_ACCELERATION