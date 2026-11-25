$ErrorActionPreference = 'Continue'
$kernel = 'nguynbquangminh/fact-grounding-five-benchmark-evaluation'
$base = Split-Path -Parent $MyInvocation.MyCommand.Path
$outDir = Join-Path $base 'kaggle_run_v5_output'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$statusLog = Join-Path $outDir 'monitor_status.log'
$downloadLog = Join-Path $outDir 'artifact_download.log'
while ($true) {
    $status = (& kaggle kernels status $kernel 2>&1 | Out-String).Trim()
    $exitCode = $LASTEXITCODE
    $record = "$(Get-Date -Format o)`t(exit=$exitCode) $status"
    Add-Content -LiteralPath $statusLog -Value $record
    if ($status -match 'KernelWorkerStatus\.COMPLETE') {
        "$(Get-Date -Format o)`tDownloading final artifacts" | Add-Content -LiteralPath $statusLog
        & kaggle kernels output $kernel -p $outDir 2>&1 | Tee-Object -FilePath $downloadLog -Append | Out-Null
        "$(Get-Date -Format o)`tArtifact download finished (exit=$LASTEXITCODE)" | Add-Content -LiteralPath $statusLog
        break
    }
    if ($status -match 'KernelWorkerStatus\.(ERROR|CANCELLED|FAILED|CANCELED)') {
        "$(Get-Date -Format o)`tKernel ended without COMPLETE" | Add-Content -LiteralPath $statusLog
        break
    }
    Start-Sleep -Seconds 1800
}
