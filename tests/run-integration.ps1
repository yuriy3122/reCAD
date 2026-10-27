[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $SkipBuild) {
    & (Join-Path $projectRoot 'scripts\build-vs2026.ps1') -Configuration $Configuration
}
$binaryRoot = Join-Path $projectRoot "build\native\bin\$Configuration"
$testRunner = Join-Path $binaryRoot 'recad_tests.exe'
$application = Join-Path $binaryRoot 'mesh2brep.exe'
if (-not (Test-Path -LiteralPath $testRunner) -or -not (Test-Path -LiteralPath $application)) {
    throw "Build reCAD.sln ($Configuration|x64) before running the tests."
}

& $testRunner
if ($LASTEXITCODE -ne 0) { throw 'CUDA, primitive, NURBS, or reference validation test failed' }
& $application --input (Join-Path $projectRoot 'testData\box_sphere_test-01.glb') `
    --input-scale 1 --output (Join-Path $projectRoot "build\native\test-output\$Configuration") `
    --verify-analytic
if ($LASTEXITCODE -ne 0) { throw 'GLB reconstruction integration test failed' }
Write-Output "All native $Configuration tests passed."
