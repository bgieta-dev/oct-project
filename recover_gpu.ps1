<#
.SYNOPSIS
    Recovers lost NVIDIA GPU on Windows ("GPU is lost" / Xid crash) without full system reboot.
.DESCRIPTION
    1. Self-elevates to Administrator if required.
    2. Kills lingering processes holding CUDA handles (python, torch).
    3. Restarts NVIDIA Display Container services.
    4. Cycles the PCIe device state via pnputil / PnPDevice API (Disable -> Sleep -> Enable).
    5. Tests recovery with nvidia-smi.
#>

param(
    [switch]$ForceRebootOnFailure
)

# 1. Require / self-elevate Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[!] Administrator privileges required. Relaunching elevated..." -ForegroundColor Yellow
    $argsList = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if ($ForceRebootOnFailure) { $argsList += " -ForceRebootOnFailure" }
    Start-Process powershell.exe -Verb RunAs -ArgumentList $argsList
    exit
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting NVIDIA GPU Recovery Procedure" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 2. Terminate hanging processes attached to CUDA
Write-Host "[1/5] Terminating lingering python/torch compute processes..." -ForegroundColor Yellow
$procNames = @("python", "python3", "pythonw")
foreach ($p in $procNames) {
    Get-Process -Name $p -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  Stopping process: $($_.Name) (PID: $($_.Id))"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 1

# 3. Stop NVIDIA Display Services
Write-Host "[2/5] Stopping NVIDIA container services..." -ForegroundColor Yellow
$services = @("NVDisplay.ContainerLocalSystem", "NvContainerLocalSystem")
foreach ($s in $services) {
    if (Get-Service -Name $s -ErrorAction SilentlyContinue) {
        Write-Host "  Stopping service: $s"
        Stop-Service -Name $s -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 1

# 4. Locate and restart NVIDIA GPU PCIe devices
Write-Host "[3/5] Locating NVIDIA Display Adapter in PnP tree..." -ForegroundColor Yellow
$gpuDevices = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue | Where-Object {
    $_.FriendlyName -like "*NVIDIA*" -or $_.HardwareID -like "*VEN_10DE*"
}

if (-not $gpuDevices) {
    Write-Host "[!] No NVIDIA display devices found via PnP. Scanning PCI bus via pnputil..." -ForegroundColor Red
} else {
    foreach ($dev in $gpuDevices) {
        Write-Host "  Found GPU: $($dev.FriendlyName) [$($dev.InstanceId)] (Status: $($dev.Status))" -ForegroundColor Cyan

        # Try pnputil device restart first
        Write-Host "  Attempting pnputil device restart..."
        $restartOut = pnputil /restart-device "$($dev.InstanceId)" 2>&1
        Write-Host "  pnputil output: $restartOut"

        Start-Sleep -Seconds 2

        # Cycle disable/enable if still degraded or for thorough reset
        Write-Host "  Power-cycling device state (Disable -> Enable)..."
        try {
            Disable-PnpDevice -InstanceId $dev.InstanceId -Confirm:$false -ErrorAction Stop
            Start-Sleep -Seconds 3
            Enable-PnpDevice -InstanceId $dev.InstanceId -Confirm:$false -ErrorAction Stop
            Write-Host "  [OK] Device re-enabled successfully." -ForegroundColor Green
        } catch {
            Write-Host "  [WARN] Disable/Enable PnPDevice warning: $_" -ForegroundColor Yellow
        }
        Start-Sleep -Seconds 2
    }
}

# 5. Restart NVIDIA services
Write-Host "[4/5] Restarting NVIDIA container services..." -ForegroundColor Yellow
foreach ($s in $services) {
    if (Get-Service -Name $s -ErrorAction SilentlyContinue) {
        Write-Host "  Starting service: $s"
        Start-Service -Name $s -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 2

# 6. Verify via nvidia-smi
Write-Host "[5/5] Testing GPU recovery status via nvidia-smi..." -ForegroundColor Yellow
$smiCmd = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if (-not $smiCmd) {
    # Check default installation path
    $defaultSmi = "C:\Windows\System32\nvidia-smi.exe"
    if (Test-Path $defaultSmi) {
        $smiCmd = $defaultSmi
    }
}

$recovered = $false
if ($smiCmd) {
    $smiResult = & $smiCmd 2>&1
    if ($LASTEXITCODE -eq 0 -and ($smiResult -notmatch "GPU is lost|Unable to determine")) {
        $recovered = $true
        Write-Host "==========================================================" -ForegroundColor Green
        Write-Host " [SUCCESS] GPU successfully recovered without rebooting!" -ForegroundColor Green
        Write-Host "==========================================================" -ForegroundColor Green
        & $smiCmd --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    } else {
        Write-Host "==========================================================" -ForegroundColor Red
        Write-Host " [FAIL] nvidia-smi still reporting error:" -ForegroundColor Red
        Write-Host $smiResult -ForegroundColor Red
        Write-Host "==========================================================" -ForegroundColor Red
    }
} else {
    Write-Host "[WARN] nvidia-smi executable not found in PATH." -ForegroundColor Yellow
}

if (-not $recovered) {
    Write-Host "`nPCIe link dropped off the bus at hardware/firmware level." -ForegroundColor Yellow
    Write-Host "Software reset was insufficient. A system reboot is required." -ForegroundColor Yellow
    if ($ForceRebootOnFailure) {
        Write-Host "ForceRebootOnFailure switch provided. Rebooting in 5 seconds..." -ForegroundColor Red
        Start-Sleep -Seconds 5
        Restart-Computer -Force
    } else {
        $answer = Read-Host "Would you like to reboot the computer now? (y/N)"
        if ($answer -eq 'y' -or $answer -eq 'Y') {
            Restart-Computer -Force
        }
    }
}
