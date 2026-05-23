$project = "D:\projects\finbench_agent"

$output = "D:\projects\finbench_agent_full_bundle.zip"

$excludePatterns = @(
    "\.git\\",
    "\\__pycache__\\",
    "\\venv\\",
    "\\node_modules\\",
    "\\chromadb\\",
    "\\bm25_index\\",
    "\\results\\",
    "\\logs\\",
    "\\checkpoints\\",
    "\\models\\",
    "\\cache\\",
    "\\tmp\\",
    "\\temp\\",
    "\.pytest_cache\\",
    "\\dist\\",
    "\\build\\"
)

Write-Host ""
Write-Host "Scanning project..."
Write-Host ""

$allFiles = Get-ChildItem `
    -Path $project `
    -Recurse `
    -File `
    -ErrorAction SilentlyContinue

$filtered = $allFiles | Where-Object {

    $path = $_.FullName

    $exclude = $false

    foreach ($pattern in $excludePatterns) {
        if ($path -match $pattern) {
            $exclude = $true
            break
        }
    }

    -not $exclude
}

$tempDir = Join-Path $env:TEMP "finbench_bundle"

if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}

New-Item `
    -ItemType Directory `
    -Path $tempDir | Out-Null

Write-Host "Copying filtered files..."
Write-Host ""

foreach ($file in $filtered) {

    $relative = $file.FullName.Substring($project.Length).TrimStart("\")

    $dest = Join-Path $tempDir $relative

    $destDir = Split-Path $dest -Parent

    if (-not (Test-Path $destDir)) {
        New-Item `
            -ItemType Directory `
            -Path $destDir `
            -Force | Out-Null
    }

    Copy-Item $file.FullName $dest -Force
}

Write-Host ""
Write-Host "Creating zip..."
Write-Host ""

if (Test-Path $output) {
    Remove-Item $output -Force
}

Compress-Archive `
    -Path "$tempDir\*" `
    -DestinationPath $output `
    -Force

$size = (Get-Item $output).Length / 1MB

Write-Host ""
Write-Host "DONE"
Write-Host ""
Write-Host "ZIP:"
Write-Host $output
Write-Host ""
Write-Host ("SIZE: {0:N2} MB" -f $size)
Write-Host ""