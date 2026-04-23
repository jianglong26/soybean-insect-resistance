Param(
    [switch]$SkipDataPreparation
)

$ErrorActionPreference = 'Stop'

Set-Location (Join-Path $PSScriptRoot '..')

Write-Host 'Step 1/3: image_and_label_preparation.py' -ForegroundColor Cyan
if (-not $SkipDataPreparation) {
    python .\experiments\image_and_label_preparation.py
} else {
    Write-Host 'Skipped data preparation.' -ForegroundColor Yellow
}

Write-Host 'Step 2/3: extract_vi_dinov3.py' -ForegroundColor Cyan
python .\experiments\extract_vi_dinov3.py

Write-Host 'Step 3/3: run_insect_resistance_analysis.py' -ForegroundColor Cyan
python .\experiments\run_insect_resistance_analysis.py

Write-Host 'Workflow completed.' -ForegroundColor Green
