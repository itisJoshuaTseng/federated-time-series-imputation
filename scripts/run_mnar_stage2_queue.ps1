param(
    [switch]$ForceRerun,
    [int[]]$Seeds = @(0, 1, 2, 3, 4)
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonExe = "C:\Users\LingChengTseng\anaconda3\envs\fed-marl\python.exe"
$tensorDir = "..\2026_vitalDB\tensor-file-for-4feature-20260304T112438Z-3-001\tensor-file-for-4feature\vitaldb_14feats_tensor_T300"
$device = "cuda"
$seedList = $Seeds
$resultDir = Join-Path $projectRoot "logs/saits_mnar"
$stateDir = Join-Path $projectRoot "logs/saits_mnar_queue_state"

$jobs = @(
    @{ Phase = "RUN_LOGIT"; Scenario = "S1"; Method = "logit";    MissingRate = "0.3" },
    @{ Phase = "RUN_LOGIT"; Scenario = "S4"; Method = "logit";    MissingRate = "0.3" },
    @{ Phase = "RUN_MISSING"; Scenario = "S1"; Method = "quantile"; MissingRate = "0.5" },
    @{ Phase = "RUN_MISSING"; Scenario = "S4"; Method = "quantile"; MissingRate = "0.5" },
    @{ Phase = "RUN_MISSING"; Scenario = "S1"; Method = "quantile"; MissingRate = "0.7" },
    @{ Phase = "RUN_MISSING"; Scenario = "S4"; Method = "quantile"; MissingRate = "0.7" }
)

New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
New-Item -ItemType Directory -Path $stateDir -Force | Out-Null

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Output "[$timestamp] $Message"
}

function Convert-MissingRateToTag {
    param([string]$MissingRate)
    return ($MissingRate -replace "\.", "p")
}

function Get-JobKey {
    param(
        [string]$Scenario,
        [string]$Method,
        [string]$MissingRate,
        [int[]]$Seeds
    )

    $seedKey = ($Seeds | Sort-Object | ForEach-Object { [string]$_ }) -join "-"
    $rateKey = Convert-MissingRateToTag -MissingRate $MissingRate
    return "${Scenario}_${Method}_rho${rateKey}_seeds_${seedKey}"
}

function Get-StatePath {
    param([string]$JobKey)
    return Join-Path $stateDir "${JobKey}.json"
}

function Write-State {
    param(
        [string]$JobKey,
        [hashtable]$Payload
    )

    $statePath = Get-StatePath -JobKey $JobKey
    try {
        $json = $Payload | ConvertTo-Json -Depth 8
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($statePath, $json, $utf8NoBom)
    }
    catch {
        Write-Log "job=$JobKey state_write_warning=$($_.Exception.Message)"
    }
}

function Remove-State {
    param([string]$JobKey)
    $statePath = Get-StatePath -JobKey $JobKey
    if (Test-Path $statePath) {
        Remove-Item -LiteralPath $statePath -Force
    }
}

function Test-ResultMatches {
    param(
        [string]$ResultPath,
        [string]$Scenario,
        [string]$Method,
        [double]$MissingRate,
        [int[]]$Seeds
    )

    try {
        $json = Get-Content -Path $ResultPath -Raw | ConvertFrom-Json
    }
    catch {
        return $false
    }

    if ($json.scenario -ne $Scenario) { return $false }
    if ($json.mnar_method -ne $Method) { return $false }

    $jsonRate = [double]$json.missing_rate
    if ([math]::Abs($jsonRate - $MissingRate) -gt 1e-9) { return $false }

    $expectedSeeds = @($Seeds | ForEach-Object { [int]$_ } | Sort-Object)
    $actualSeeds = @($json.seeds | ForEach-Object { [int]$_ } | Sort-Object)
    if (($expectedSeeds -join ",") -ne ($actualSeeds -join ",")) { return $false }

    foreach ($seed in $Seeds) {
        $seedResults = @($json.results | Where-Object { [int]$_.seed -eq [int]$seed })
        if ($seedResults.Count -lt 3) { return $false }

        $methods = @($seedResults | ForEach-Object { $_.method } | Sort-Object -Unique)
        if ((@("fedavg", "fedprox", "local") -join ",") -ne ($methods -join ",")) {
            return $false
        }
    }

    return $true
}

function Find-CompletedResult {
    param(
        [string]$Scenario,
        [string]$Method,
        [double]$MissingRate,
        [int[]]$Seeds
    )

    if (-not (Test-Path $resultDir)) { return $null }

    $pattern = "mnar_recon_${Scenario}_${Method}_*.json"
    $candidates = Get-ChildItem -Path $resultDir -Filter $pattern -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    foreach ($f in $candidates) {
        if (Test-ResultMatches -ResultPath $f.FullName -Scenario $Scenario -Method $Method -MissingRate $MissingRate -Seeds $Seeds) {
            return $f.FullName
        }
    }

    return $null
}

function New-ResultPath {
    param(
        [string]$Scenario,
        [string]$Method,
        [string]$MissingRate
    )

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $rateTag = Convert-MissingRateToTag -MissingRate $MissingRate
    $fileName = "mnar_recon_${Scenario}_${Method}_rho${rateTag}_${timestamp}.json"
    return Join-Path $resultDir $fileName
}

function Run-Job {
    param(
        [string]$Phase,
        [string]$Scenario,
        [string]$Method,
        [string]$MissingRate
    )

    $jobKey = Get-JobKey -Scenario $Scenario -Method $Method -MissingRate $MissingRate -Seeds $seedList

    Write-Log "------------------------------------------------------------"
    Write-Log "job=$jobKey"
    Write-Log "phase=$Phase"
    Write-Log "scenario=$Scenario"
    Write-Log "seeds=$($seedList -join ' ')"
    Write-Log "method=$Method"
    Write-Log "missing-rate=$MissingRate"
    Write-Log "device=$device"
    Write-Log "tensor-dir=$tensorDir"

    if (-not $ForceRerun) {
        $existing = Find-CompletedResult -Scenario $Scenario -Method $Method -MissingRate ([double]$MissingRate) -Seeds $seedList
        if ($null -ne $existing) {
            Write-Log "job=$jobKey status=SKIP reason=completed_result_found result=$existing"
            return
        }
    }

    $outputPath = New-ResultPath -Scenario $Scenario -Method $Method -MissingRate $MissingRate
    Write-State -JobKey $jobKey -Payload @{
        job = $jobKey
        status = "running"
        phase = $Phase
        scenario = $Scenario
        method = $Method
        missing_rate = [double]$MissingRate
        seeds = @($seedList)
        started_at = (Get-Date).ToString("s")
        output_path = $outputPath
    }

    try {
        & $pythonExe experiments/run_mnar_experiment.py `
            --scenario $Scenario `
            --seeds $seedList `
            --mnar-method $Method `
            --missing-rate $MissingRate `
            --device $device `
            --tensor-dir $tensorDir `
            --output-path $outputPath `
            --fail-if-output-exists

        if ($LASTEXITCODE -ne 0) {
            throw "Python exited with code $LASTEXITCODE"
        }

        $verified = Find-CompletedResult -Scenario $Scenario -Method $Method -MissingRate ([double]$MissingRate) -Seeds $seedList
        if ($null -eq $verified) {
            throw "Job finished but no verified completed result was found."
        }

        Write-State -JobKey $jobKey -Payload @{
            job = $jobKey
            status = "completed"
            phase = $Phase
            scenario = $Scenario
            method = $Method
            missing_rate = [double]$MissingRate
            seeds = @($seedList)
            completed_at = (Get-Date).ToString("s")
            output_path = $verified
        }

        Write-Log "job=$jobKey status=COMPLETED result=$verified"
    }
    catch {
        Write-State -JobKey $jobKey -Payload @{
            job = $jobKey
            status = "failed"
            phase = $Phase
            scenario = $Scenario
            method = $Method
            missing_rate = [double]$MissingRate
            seeds = @($seedList)
            failed_at = (Get-Date).ToString("s")
            output_path = $outputPath
            error = $_.Exception.Message
        }
        throw
    }
}

Write-Log "MNAR stage2 queue started"
Write-Log "projectRoot=$projectRoot"
Write-Log "python=$pythonExe"
Write-Log "device=$device"
Write-Log "tensor-dir=$tensorDir"
Write-Log "force-rerun=$ForceRerun"
Write-Log "planned-jobs=$($jobs.Count)"

foreach ($job in $jobs) {
    Run-Job -Phase $job.Phase -Scenario $job.Scenario -Method $job.Method -MissingRate $job.MissingRate
}

Write-Log "MNAR stage2 queue finished"
