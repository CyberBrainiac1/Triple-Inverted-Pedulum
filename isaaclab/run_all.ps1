param(
    [string]$IsaacLabRoot = "$HOME\Downloads\IsaacLab",
    [string]$IsaacSimRoot = "C:\isaacsim",
    [int]$NumEnvs = 64,
    [int]$MaxIterations = 0,
    [int]$RetriesPerStage = 3,
    [switch]$Headless = $true
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pkgPath = Join-Path $repoRoot "source\triple_pendulum_isaac"
$zipPath = Join-Path $repoRoot "FinalTripleInvertedPendulum.zip"
$extractPath = Join-Path $repoRoot "_extracted_urdf"
$sanitizedUrdf = Join-Path $repoRoot "_sanitized_urdf\finaltripleinvertedpendulum\urdf\finaltripleinvertedpendulum_sanitized.urdf"
$sanitizeScript = Join-Path $repoRoot "isaaclab\sanitize_urdf_assets.py"
$trainScript = Join-Path $repoRoot "isaaclab\train_sb3.py"
$playScript = Join-Path $repoRoot "isaaclab\play_sb3.py"
$isaacLabBat = Join-Path $IsaacLabRoot "isaaclab.bat"
$isaacSimPython = Join-Path $IsaacSimRoot "python.bat"
$isaacSimBat = Join-Path $IsaacSimRoot "isaac-sim.bat"
$isaacLabLink = Join-Path $IsaacLabRoot "_isaac_sim"

function Invoke-IsaacLab {
    param(
        [string[]]$IsaacLabArgs
    )
    & $isaacLabBat @IsaacLabArgs
    if ($LASTEXITCODE -ne 0) {
        throw "isaaclab.bat failed with exit code $LASTEXITCODE"
    }
}

function Invoke-TrainingStage {
    param(
        [string]$TaskName,
        [string[]]$StageArgs
    )

    $stageLogDir = Join-Path $IsaacLabRoot ("logs\sb3\" + $TaskName)
    $success = $false

    for ($attempt = 1; $attempt -le $RetriesPerStage; $attempt++) {
        $stageStart = Get-Date
        Write-Host "Starting $TaskName (attempt $attempt/$RetriesPerStage)..."

        try {
            Invoke-IsaacLab -IsaacLabArgs $StageArgs
        }
        catch {
            Write-Host "$TaskName failed on attempt $attempt."
            if ($attempt -lt $RetriesPerStage) {
                Write-Host "Waiting 30 seconds before retry..."
                Start-Sleep -Seconds 30
                continue
            }
            throw
        }

        $latestModel = $null
        if (Test-Path $stageLogDir) {
            $latestModel = Get-ChildItem -Path $stageLogDir -Filter model.zip -Recurse -File |
                Where-Object { $_.LastWriteTime -ge $stageStart.AddSeconds(-2) } |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
        }

        if ($null -ne $latestModel) {
            Write-Host "Checkpoint written: $($latestModel.FullName)"
            $success = $true
            break
        }

        Write-Host "$TaskName finished but no new model.zip was found."
        if ($attempt -lt $RetriesPerStage) {
            Write-Host "Waiting 30 seconds before retry..."
            Start-Sleep -Seconds 30
        }
    }

    if (-not $success) {
        throw "Stage $TaskName did not produce a checkpoint after $RetriesPerStage attempt(s)."
    }
}

if (-not (Test-Path $isaacLabBat)) {
    throw "Could not find isaaclab.bat at $isaacLabBat"
}
if (-not (Test-Path $isaacSimPython)) {
    throw "Could not find Isaac Sim python at $isaacSimPython"
}
if (-not (Test-Path $isaacSimBat)) {
    throw "Could not find isaac-sim.bat at $isaacSimBat"
}
if (-not (Test-Path $zipPath)) {
    throw "Could not find URDF zip at $zipPath"
}

if (-not (Test-Path $extractPath)) {
    Write-Host "Extracting URDF zip..."
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractPath -Force
}

if ((-not (Test-Path $isaacLabLink)) -or (-not (Test-Path (Join-Path $isaacLabLink "python.bat")))) {
    if (Test-Path $isaacLabLink) {
        Remove-Item $isaacLabLink -Recurse -Force
    }
    Write-Host "Linking Isaac Lab to Isaac Sim..."
    cmd /c "mklink /J `"$isaacLabLink`" `"$IsaacSimRoot`"" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create Isaac Sim junction at $isaacLabLink"
    }
}

Push-Location $IsaacLabRoot
try {
    Write-Host "Checking Isaac Sim Python..."
    & $isaacSimPython -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Sim python check failed"
    }

    Write-Host "Sanitizing URDF asset names..."
    Invoke-IsaacLab -IsaacLabArgs @("-p", $sanitizeScript)
    if (-not (Test-Path $sanitizedUrdf)) {
        throw "Sanitized URDF was not generated at $sanitizedUrdf"
    }

    Write-Host "Checking Stable-Baselines3 support..."
    & $isaacSimPython -c "import stable_baselines3, h5py; print(stable_baselines3.__version__)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Stable-Baselines3 is missing; installing Isaac Lab SB3 support once..."
        Invoke-IsaacLab -IsaacLabArgs @("-i", "sb3")
    }

    Write-Host "Installing the triple pendulum task package..."
    Invoke-IsaacLab -IsaacLabArgs @("-p", "-m", "pip", "install", "-e", $pkgPath)

    $commonArgs = @("--num_envs", "$NumEnvs")
    if ($Headless) {
        $commonArgs += "--headless"
    }
    if ($MaxIterations -gt 0) {
        $commonArgs += @("--max_iterations", "$MaxIterations")
    }

    Write-Host "Training balance task..."
    Invoke-TrainingStage -TaskName "TriplePendulum-Balance-Direct-v0" -StageArgs (@("-p", $trainScript, "--task", "TriplePendulum-Balance-Direct-v0") + $commonArgs)
    Start-Sleep -Seconds 10

    Write-Host "Training swing-up task..."
    Invoke-TrainingStage -TaskName "TriplePendulum-SwingUp-Direct-v0" -StageArgs (@("-p", $trainScript, "--task", "TriplePendulum-SwingUp-Direct-v0") + $commonArgs)
    Start-Sleep -Seconds 10

    Write-Host "Training combined swing-up + balance task..."
    Invoke-TrainingStage -TaskName "TriplePendulum-SwingUpBalance-Direct-v0" -StageArgs (@("-p", $trainScript, "--task", "TriplePendulum-SwingUpBalance-Direct-v0") + $commonArgs)

    Write-Host ""
    Write-Host "Training completed."
    Write-Host "Play the latest combined checkpoint with:"
    Write-Host "  `"$isaacLabBat`" -p `"$playScript`" --task TriplePendulum-SwingUpBalance-Direct-v0 --checkpoint `"C:\path\to\model.zip`""
}
finally {
    Pop-Location
}
