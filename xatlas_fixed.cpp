// Fix for the end of xatlas.cpp
// Add this at the appropriate location

const char *StringForEnum(ProgressCategory category)
{
    switch (category)
    {
    case ProgressCategory::AddMesh:
        return "Adding mesh";
    case ProgressCategory::ComputeCharts:
        return "Computing charts";
    case ProgressCategory::PackCharts:
        return "Packing charts";
    case ProgressCategory::BuildOutputMeshes:
        return "Building output meshes";
    default:
        return "";
    }
}

#if XA_GPU_ACCELERATION
namespace gpu
{

    bool isGpuAvailable()
    {
        // Simple stub implementation
        return false;
    }

    void shutdown()
    {
        // Simple stub implementation
    }

} // namespace gpu
#endif

} // namespace xatlas