[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dependencyRoot = Join-Path $projectRoot '.deps'
$vcpkgVersion = '2026.05.25'
$vcpkgRoot = Join-Path $dependencyRoot "vcpkg-$vcpkgVersion"
$archive = Join-Path $dependencyRoot "vcpkg-$vcpkgVersion.zip"
$installRoot = Join-Path $dependencyRoot 'installed'
$tinyGltfInclude = Join-Path $dependencyRoot 'tinygltf\include'

New-Item -ItemType Directory -Force -Path $dependencyRoot | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $vcpkgRoot 'bootstrap-vcpkg.bat'))) {
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest `
            -Uri "https://github.com/microsoft/vcpkg/archive/refs/tags/$vcpkgVersion.zip" `
            -OutFile $archive
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $dependencyRoot -Force
}

$vcpkg = Join-Path $vcpkgRoot 'vcpkg.exe'
if (-not (Test-Path -LiteralPath $vcpkg)) {
    & (Join-Path $vcpkgRoot 'bootstrap-vcpkg.bat') -disableMetrics
    if ($LASTEXITCODE -ne 0) { throw 'vcpkg bootstrap failed' }
}

# Dependencies built by any v14x MSVC toolset are binary compatible with the
# v145 application. Prefer the installed VS 2022 toolset because this pinned
# vcpkg release predates full VS 2026 instance discovery.
$vs2022 = 'C:\Program Files\Microsoft Visual Studio\2022\Community'
$vs2026 = 'C:\Program Files\Microsoft Visual Studio\18\Community'
if (Test-Path -LiteralPath $vs2022) {
    $env:VCPKG_VISUAL_STUDIO_PATH = $vs2022
} elseif (Test-Path -LiteralPath $vs2026) {
    $env:VCPKG_VISUAL_STUDIO_PATH = $vs2026
} else {
    throw 'No supported Visual Studio C++ installation was found.'
}

& $vcpkg install `
    --classic `
    draco:x64-windows `
    opencascade:x64-windows `
    stb:x64-windows `
    eigen3:x64-windows `
    nlohmann-json:x64-windows `
    --x-install-root=$installRoot `
    --clean-after-build
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }

New-Item -ItemType Directory -Force -Path $tinyGltfInclude | Out-Null
$tinyGltfHeader = Join-Path $tinyGltfInclude 'tiny_gltf.h'
if (-not (Test-Path -LiteralPath $tinyGltfHeader)) {
    Invoke-WebRequest `
        -Uri 'https://raw.githubusercontent.com/syoyo/tinygltf/v3.0.0/tiny_gltf.h' `
        -OutFile $tinyGltfHeader
}

Write-Host "Dependencies installed in $installRoot"
