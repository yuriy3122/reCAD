[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$ctest = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
if (-not (Test-Path -LiteralPath $ctest)) {
    throw "Visual Studio 2026 bundled CTest was not found at $ctest"
}

Push-Location $projectRoot
try {
    & $ctest --preset release
    if ($LASTEXITCODE -ne 0) { throw 'mesh2brep integration test failed' }
} finally {
    Pop-Location
}

