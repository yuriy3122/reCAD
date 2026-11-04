[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug', 'Both')]
    [string]$Configuration = 'Release',
    [switch]$SkipDependencies,
    [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$msbuild = 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe'
if (-not (Test-Path -LiteralPath $msbuild)) {
    $vswhere = Join-Path ([Environment]::GetFolderPath('ProgramFilesX86')) 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $msbuild = & $vswhere -latest -products '*' -version '[18.0,19.0)' `
            -requires Microsoft.Component.MSBuild -find 'MSBuild\Current\Bin\amd64\MSBuild.exe' |
            Select-Object -First 1
    }
}
if (-not $msbuild -or -not (Test-Path -LiteralPath $msbuild)) {
    throw 'Visual Studio 2026 MSBuild was not found. Install the Desktop development with C++ workload.'
}

if (-not $SkipDependencies -and (
    -not (Test-Path -LiteralPath (Join-Path $projectRoot '.deps\installed\x64-windows\lib\draco.lib')) -or
    -not (Test-Path -LiteralPath (Join-Path $projectRoot '.deps\installed\x64-windows\lib\TKXSBase.lib')) -or
    -not (Test-Path -LiteralPath (Join-Path $projectRoot '.deps\tinygltf\include\tiny_gltf.h')))) {
    & (Join-Path $PSScriptRoot 'bootstrap-dependencies.ps1')
}

$configurations = if ($Configuration -eq 'Both') { @('Debug', 'Release') } else { @($Configuration) }
$buildTarget = if ($Rebuild) { 'Rebuild' } else { 'Build' }
$solution = Join-Path $projectRoot 'reCAD.sln'
foreach ($buildConfiguration in $configurations) {
    & $msbuild $solution /m "/t:$buildTarget" "/p:Configuration=$buildConfiguration" `
        /p:Platform=x64 /v:minimal /nologo /nodeReuse:false
    if ($LASTEXITCODE -ne 0) { throw "$buildConfiguration build failed" }
}
