<#
.SYNOPSIS
    Registreert eci_json_collector.py als dagelijkse taak in Windows Task Scheduler.

.DESCRIPTION
    Maakt een taak die het script elke dag draait, met logging naar een bestand
    per dag. Als de PC uit stond op het geplande moment, draait de taak alsnog
    zodra hij weer aan is.

    Geen admin-rechten nodig: de taak draait onder je eigen account en alleen
    wanneer je bent ingelogd.

.EXAMPLE
    .\register_eci_task.ps1
    .\register_eci_task.ps1 -Time "07:15"
    .\register_eci_task.ps1 -Show
    .\register_eci_task.ps1 -RunNow
    .\register_eci_task.ps1 -Unregister
#>

[CmdletBinding()]
param(
    [string] $ScriptPath = "C:\Users\Gebruiker\Documents\Betmobile\eci_json_collector.py",
    [string] $TaskName   = "Betmobile - ECI JSON collector",
    [string] $Time       = "06:30",
    [switch] $Show,
    [switch] $RunNow,
    [switch] $Unregister
)

$ErrorActionPreference = "Stop"

# --- opruimen ---------------------------------------------------------------

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "[OK] taak '$TaskName' verwijderd" -ForegroundColor Green
    } else {
        Write-Host "[--] taak '$TaskName' bestaat niet" -ForegroundColor Yellow
    }
    return
}

# --- status tonen -----------------------------------------------------------

if ($Show) {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "[--] taak '$TaskName' bestaat niet" -ForegroundColor Yellow
        return
    }
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host "Taak    : $TaskName"
    Write-Host "Status  : $($task.State)"
    Write-Host "Laatste : $($info.LastRunTime)  (resultaat $($info.LastTaskResult))"
    Write-Host "Volgende: $($info.NextRunTime)"
    Write-Host ""
    Write-Host "Resultaat 0 betekent goed afgelopen. Alles daarboven is een fout;"
    Write-Host "kijk dan in het logbestand van die dag."
    return
}

# --- controles vooraf -------------------------------------------------------

if (-not (Test-Path $ScriptPath)) {
    throw "Script niet gevonden: $ScriptPath"
}

$WorkDir = Split-Path -Parent $ScriptPath
$LogDir  = Join-Path $WorkDir "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Volledig pad naar python.exe. Task Scheduler kent je PATH niet altijd,
# dus we zetten het hard in de taak.
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    throw "python niet gevonden in PATH. Geef het pad handmatig op in dit script."
}
Write-Host "Python  : $python"
Write-Host "Script  : $ScriptPath"
Write-Host "Werkmap : $WorkDir"
Write-Host "Logs    : $LogDir"

# --- de taak zelf -----------------------------------------------------------

# cmd.exe als omhulsel, puur om stdout en stderr naar een dagelijks log te
# kunnen sturen. Een ScheduledTaskAction kan zelf niet omleiden.
$logPattern = Join-Path $LogDir "eci_collect_%date:~-4%%date:~3,2%%date:~0,2%.log"
$command    = "`"$python`" `"$ScriptPath`" collect >> `"$logPattern`" 2>&1"

$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c $command" `
    -WorkingDirectory $WorkDir

$trigger = New-ScheduledTaskTrigger -Daily -At $Time

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "`n[--] bestaande taak wordt vervangen" -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName    $TaskName `
    -Action      $action `
    -Trigger     $trigger `
    -Settings    $settings `
    -Principal   $principal `
    -Description "Haalt dagelijks de ECI match-odds feed op (append-only, schrijft niet naar PostgreSQL)." `
    | Out-Null

Write-Host "`n[OK] taak geregistreerd: '$TaskName', dagelijks om $Time" -ForegroundColor Green

if ($RunNow) {
    Write-Host "[--] taak wordt nu gestart..."
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 3
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host "     status: $((Get-ScheduledTask -TaskName $TaskName).State)"
}

Write-Host ""
Write-Host "Handig:"
Write-Host "  .\register_eci_task.ps1 -Show          status en laatste resultaat"
Write-Host "  .\register_eci_task.ps1 -RunNow        nu draaien"
Write-Host "  .\register_eci_task.ps1 -Unregister    weghalen"