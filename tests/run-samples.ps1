[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$Executable,
    [string]$OutputRoot,
    [double]$MaximumReferenceRmsRelative = 0.001,
    [double]$MaximumReferenceDistanceRelative = 0.005,
    [double]$MaximumReferenceRmsMm = 0.005,
    [double]$MaximumReferenceDistanceMm = 0.1,
    [double]$MaximumRelativeVolumeError = 0.001
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Executable) { $Executable = Join-Path $projectRoot 'build\native\bin\Release\mesh2brep.exe' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $projectRoot 'build\sample-validation' }
$dataset = (Resolve-Path -LiteralPath $DatasetRoot).Path
$inputs = @(Get-ChildItem -LiteralPath (Join-Path $dataset 'GLB') -Filter '*.glb' | Sort-Object Name)
if ($inputs.Count -eq 0) { throw "No GLB samples found in $dataset" }
$results = @()
foreach ($sample in $inputs) {
    $reference = Join-Path (Join-Path $dataset 'BRep') ($sample.BaseName + '.step')
    if (-not (Test-Path -LiteralPath $reference)) { throw "Missing reference: $reference" }
    $output = Join-Path $OutputRoot $sample.BaseName
    & $Executable --input $sample.FullName --output $output --reference-step $reference --reference-z-up --verify-analytic
    if ($LASTEXITCODE -ne 0) { throw "Reconstruction failed: $($sample.Name)" }
    $report = Get-Content -LiteralPath (Join-Path $output 'reconstruction.json') -Raw | ConvertFrom-Json
    $measured = $report.reconstruction.properties
    $expected = $report.reference_validation
    $signatureMatches = $measured.faces -eq $expected.reference_faces -and
        $measured.planes -eq $expected.reference_planes -and
        $measured.cylinders -eq $expected.reference_cylinders -and
        $measured.spheres -eq $expected.reference_spheres -and
        $measured.tori -eq $expected.reference_tori
    if (-not $signatureMatches) { throw "Surface signature mismatch: $($sample.Name)" }
    if ($expected.normalized_rms -gt $MaximumReferenceRmsRelative -or
        $expected.normalized_max -gt $MaximumReferenceDistanceRelative -or
        $expected.symmetric_rms_mm -gt $MaximumReferenceRmsMm -or
        $expected.symmetric_max_mm -gt $MaximumReferenceDistanceMm -or
        $expected.volume_relative_error -gt $MaximumRelativeVolumeError) {
        throw "Reference distance exceeds tolerance: $($sample.Name)"
    }
    $results += [pscustomobject]@{
        Sample = $sample.BaseName
        Planes = $measured.planes
        Cylinders = $measured.cylinders
        Spheres = $measured.spheres
        Tori = $measured.tori
        ReferenceRmsMm = $expected.symmetric_rms_mm
        ReferenceMaxMm = $expected.symmetric_max_mm
        RelativeVolumeError = $expected.volume_relative_error
    }
}
$results | Format-Table -AutoSize | Out-String -Width 240 | Write-Output
Write-Output "All $($results.Count) GLB/STEP sample pairs passed. Detailed metrics are in $OutputRoot."
