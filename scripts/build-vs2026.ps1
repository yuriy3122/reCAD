[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug', 'Both')]
    [string]$Configuration = 'Release',
    [switch]$SkipDependencies
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$cmake = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (-not (Test-Path -LiteralPath $cmake)) {
    throw "Visual Studio 2026 bundled CMake was not found at $cmake"
}

if (-not $SkipDependencies -and (
    -not (Test-Path -LiteralPath (Join-Path $projectRoot '.deps\installed\x64-windows\share\opencascade')) -or
    -not (Test-Path -LiteralPath (Join-Path $projectRoot '.deps\tinygltf\include\tiny_gltf.h')))) {
    & (Join-Path $PSScriptRoot 'bootstrap-dependencies.ps1')
}

Push-Location $projectRoot
try {
    & $cmake --preset vs2026
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
    if ($Configuration -eq 'Release' -or $Configuration -eq 'Both') {
        & $cmake --build --preset release
        if ($LASTEXITCODE -ne 0) { throw 'Release build failed' }
    }
    if ($Configuration -eq 'Debug' -or $Configuration -eq 'Both') {
        & $cmake --build --preset debug
        if ($LASTEXITCODE -ne 0) { throw 'Debug build failed' }
    }
} finally {
    Pop-Location
}
