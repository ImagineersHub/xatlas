#ifndef XATLAS_GPU_H
#define XATLAS_GPU_H

#if XA_GPU_ACCELERATION

namespace xatlas
{
    namespace internal
    {

        // Forward declarations
        class Mesh;
        class Vector2;
        template <typename T>
        class Array;

        namespace param
        {
            class Quality;
        }

    } // namespace internal

    namespace gpu
    {

        // Compute LSCM parameterization using GPU acceleration
        bool computeLSCM(const internal::Mesh *mesh, const internal::Array<uint32_t> &faces,
                         internal::Array<internal::Vector2> &texcoords);

        // Compute chart quality metrics using GPU
        bool computeChartMetrics(const internal::Mesh *mesh, internal::param::Quality &quality);

        // Initialize GPU resources
        bool initialize();

        // Check if GPU is available
        bool isGpuAvailable();

        // Shutdown GPU resources
        void shutdown();

    } // namespace gpu
} // namespace xatlas

#endif // XA_GPU_ACCELERATION

#endif // XATLAS_GPU_H