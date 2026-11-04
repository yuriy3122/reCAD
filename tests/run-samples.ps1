[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$ReferenceRoot,
    [switch]$RequireSurfaceSignature,
    [switch]$ReuseCompleted,
    [ValidateRange(1, 1000000)][int]$ReferenceSamples = 512,
    [ValidateRange(1, 86400)][int]$TimeoutSeconds = 300,
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
$inputs = @(Get-ChildItem -LiteralPath $dataset -Filter '*.glb' -Recurse | Sort-Object FullName)
if ($inputs.Count -eq 0) { throw "No GLB samples found in $dataset" }
if (-not $ReferenceRoot) {
    $ReferenceRoot = if (Test-Path (Join-Path $dataset 'BRep')) { Join-Path $dataset 'BRep' } else { $dataset }
}
$references = @(Get-ChildItem -LiteralPath $ReferenceRoot -Recurse -File | Where-Object { $_.Extension -in '.step', '.stp' })
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$settings = [ordered]@{
    DatasetRoot = $dataset; ReferenceRoot = $ReferenceRoot; Executable = $Executable
    ReferenceSamples = $ReferenceSamples; TimeoutSeconds = $TimeoutSeconds
    MaximumReferenceRmsRelative = $MaximumReferenceRmsRelative
    MaximumReferenceDistanceRelative = $MaximumReferenceDistanceRelative
    MaximumReferenceRmsMm = $MaximumReferenceRmsMm
    MaximumReferenceDistanceMm = $MaximumReferenceDistanceMm
    MaximumRelativeVolumeError = $MaximumRelativeVolumeError
    RequireSurfaceSignature = [bool]$RequireSurfaceSignature
}
$settings | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputRoot 'validation_settings.json') -Encoding UTF8
$results = @()
foreach ($sample in $inputs) {
    $elapsed = [System.Diagnostics.Stopwatch]::StartNew()
    $row = [ordered]@{ Sample = $sample.BaseName; Passed = $false; Error = $null }
    try {
    $matches = @($references | Where-Object {
        $sample.BaseName -eq $_.BaseName -or $sample.BaseName.StartsWith($_.BaseName + '_')
    } | Sort-Object { $_.BaseName.Length } -Descending)
    if ($matches.Count -eq 0) { throw "Missing STEP reference: $($sample.Name)" }
    $best = @($matches | Where-Object { $_.BaseName.Length -eq $matches[0].BaseName.Length })
    if ($best.Count -ne 1) { throw "Ambiguous references; specify -ReferenceRoot" }
    $reference = $best[0].FullName
    $output = Join-Path $OutputRoot $sample.BaseName
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $statusPath = Join-Path $output 'run_status.json'
    $reportPath = Join-Path $output 'reconstruction.json'
    $reusable = $false
    if ($ReuseCompleted -and (Test-Path -LiteralPath $statusPath) -and (Test-Path -LiteralPath $reportPath)) {
        $statusFile = Get-Item -LiteralPath $statusPath
        $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        $previous = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $reusable = $status.status -eq 'completed' -and
            $previous.input.path -eq $sample.FullName -and
            $previous.input.coordinate_scale_to_millimetres -eq 1000 -and
            $previous.reference_validation.path -eq $reference -and
            $previous.reference_validation.reference_z_up_to_gltf -and
            $previous.reference_validation.sample_limit_per_direction -eq $ReferenceSamples -and
            $sample.LastWriteTimeUtc -le $statusFile.LastWriteTimeUtc -and
            (Get-Item -LiteralPath $reference).LastWriteTimeUtc -le $statusFile.LastWriteTimeUtc -and
            (Get-Item -LiteralPath $Executable).LastWriteTimeUtc -le $statusFile.LastWriteTimeUtc
    }
    if (-not $reusable) {
    $arguments = '--input "{0}" --output "{1}" --reference-step "{2}" --reference-z-up --reference-samples {3} --verify-analytic' -f $sample.FullName, $output, $reference, $ReferenceSamples
    $process = Start-Process -FilePath $Executable -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'stdout.log') -RedirectStandardError (Join-Path $output 'stderr.log')
    # Retain the native handle before exit; otherwise Windows PowerShell can
    # lose ExitCode when a redirected process is waited on asynchronously.
    $taskHandle = $process.Handle
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        $process.Kill()
        $process.WaitForExit()
        @{ status = 'timed_out'; error = "Reconstruction exceeded $TimeoutSeconds seconds" } |
            ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
        throw "Reconstruction exceeded $TimeoutSeconds seconds"
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "Reconstruction exit $($process.ExitCode): $(Get-Content (Join-Path $output 'stderr.log') -Raw)" }
    }
    $row.Reused = $reusable
    $report = Get-Content -LiteralPath (Join-Path $output 'reconstruction.json') -Raw | ConvertFrom-Json
    $measured = $report.reconstruction.properties
    if (-not $measured.valid -or -not $measured.closed_solid -or $measured.solids -lt 1 -or
        $measured.bspline_surfaces -ne 0 -or $measured.other_surfaces -ne 0) { throw 'Expected a valid closed analytic solid' }
    $expected = $report.reference_validation
    $row.Faces = $measured.faces
    $row.Planes = $measured.planes
    $row.Cylinders = $measured.cylinders
    $row.Cones = $measured.cones
    $row.Spheres = $measured.spheres
    $row.Tori = $measured.tori
    $row.ReferenceRmsMm = $expected.symmetric_rms_mm
    $row.ReferenceMaxMm = $expected.symmetric_max_mm
    $row.RelativeVolumeError = $expected.volume_relative_error
    $signatureMatches = $measured.faces -eq $expected.reference_faces -and
        $measured.planes -eq $expected.reference_planes -and
        $measured.cylinders -eq $expected.reference_cylinders -and
        $measured.cones -eq $expected.reference_cones -and
        $measured.spheres -eq $expected.reference_spheres -and
        $measured.tori -eq $expected.reference_tori
    if ($RequireSurfaceSignature -and -not $signatureMatches) { throw "Surface signature mismatch: $($sample.Name)" }
    if (-not $expected.distances_compared -or -not $expected.reference_valid -or
        -not $expected.reference_closed -or $expected.failed_samples -ne 0) { throw 'Incomplete or invalid reference comparison' }
    foreach ($metric in @('normalized_rms', 'normalized_max', 'symmetric_rms_mm', 'symmetric_max_mm', 'volume_relative_error')) {
        if ($null -eq $expected.$metric -or [double]::IsNaN($expected.$metric) -or [double]::IsInfinity($expected.$metric)) { throw "Invalid metric: $metric" }
    }
    if ($expected.normalized_rms -gt $MaximumReferenceRmsRelative -or
        $expected.normalized_max -gt $MaximumReferenceDistanceRelative -or
        $expected.symmetric_rms_mm -gt $MaximumReferenceRmsMm -or
        $expected.symmetric_max_mm -gt $MaximumReferenceDistanceMm -or
        $expected.volume_relative_error -gt $MaximumRelativeVolumeError) {
        throw "Reference distance exceeds tolerance: $($sample.Name)"
    }
    $row.Passed = $true
    Write-Output "$($sample.Name): PASS"
    } catch {
        $row.Error = $_.Exception.Message.Trim()
        Write-Output "$($sample.Name): FAIL - $($_.Exception.Message.Trim())"
    }
    $row.Seconds = [math]::Round($elapsed.Elapsed.TotalSeconds, 3)
    $results += [pscustomobject]$row
    $results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputRoot 'results.json') -Encoding UTF8
}
$results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputRoot 'results.json') -Encoding UTF8
$results | Select-Object Sample, Passed, Reused, Faces, ReferenceRmsMm, ReferenceMaxMm, RelativeVolumeError, Seconds | Format-Table -AutoSize | Out-String -Width 240 | Write-Output
if (@($results | Where-Object { -not $_.Passed }).Count) { throw "Sample validation failed. See $OutputRoot\results.json" }
Write-Output "All $($results.Count) GLB/STEP sample pairs passed. Detailed metrics are in $OutputRoot."
