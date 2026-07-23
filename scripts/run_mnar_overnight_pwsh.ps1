$ErrorActionPreference = "Continue"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonExe = "C:\Users\LingChengTseng\anaconda3\envs\fed-marl\python.exe"
$tensorDir = "..\2026_vitalDB\tensor-file-for-4feature-20260304T112438Z-3-001\tensor-file-for-4feature\vitaldb_14feats_tensor_T300"
$device = "cuda"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "[$timestamp] $Message"
}

function Run-Job {
    param(
        [string]$Scenario,
        [string]$Method,
        [string]$MissingRate
    )

    Write-Log "------------------------------------------------------------"
    Write-Log "scenario=$Scenario"
    Write-Log "seeds=0 1 2 3 4"
    Write-Log "method=$Method"
    Write-Log "missing-rate=$MissingRate"
    Write-Log "device=$device"
    Write-Log "tensor-dir=$tensorDir"

    & $pythonExe experiments/run_mnar_experiment.py `
        --scenario $Scenario `
        --seeds 0 1 2 3 4 `
        --mnar-method $Method `
        --missing-rate $MissingRate `
        --device $device `
        --tensor-dir $tensorDir

    Write-Log "exit_code=$LASTEXITCODE"
}

Write-Log "MNAR overnight batch started"
Write-Log "projectRoot=$projectRoot"
Write-Log "python=$pythonExe"
Write-Log "device=$device"
Write-Log "tensor-dir=$tensorDir"

Run-Job -Scenario "S1" -Method "quantile" -MissingRate "0.3"
Run-Job -Scenario "S4" -Method "quantile" -MissingRate "0.3"

Write-Log "MNAR overnight batch finished"
