$file = 'source\xatlas\xatlas.cpp'
$content = Get-Content $file
$lineNumber = 16483  # The line with the closing brace of StringForEnum function

# Create the content to insert
$insertContent = @"
#if XA_GPU_ACCELERATION
namespace gpu {

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
"@

# Insert content after the specified line
$newContent = @()
for ($i = 0; $i -lt $content.Length; $i++) {
    $newContent += $content[$i]
    if ($i -eq $lineNumber) {
        $newContent += ""  # Add a blank line
        $newContent += $insertContent -split "`n"
    }
}

# Write the modified content back to the file
$newContent | Set-Content $file

Write-Host "Fix applied successfully!" 