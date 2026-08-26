#Requires -Version 5.1

<#
.SYNOPSIS
Collects low-overhead DWM trends and a short, restart-aligned incident bundle.

.DESCRIPTION
StartMonitor creates a bounded circular PerfMon log while the machine is healthy.
CaptureIncident records the degraded state, waits while you restart DWM manually,
then captures the new process and closes the trace. BeginIncident and EndIncident
provide a split recovery workflow if a reboot or shell interruption is likely;
that split workflow collects snapshots without WPR unless
-AllowUnboundedSplitTrace is explicitly supplied.

The script never terminates DWM, resets the display driver, clears an event log,
deletes a data collector, or cancels a WPR recording. It also refuses to stop a
WPR session that it did not start.

The output can contain process paths, device identifiers, event messages, and
ETW payloads. The explicit process-tree CommandLine column is omitted unless
-IncludeCommandLines is supplied; raw ETL/EVTX payloads can still contain
commands or secrets. The script never creates process dumps or screen captures.

.EXAMPLE
.\tools\DwmWatch.ps1 StartMonitor

.EXAMPLE
.\tools\DwmWatch.ps1 CaptureIncident
# Restart DWM manually when instructed; the script detects its new PID.

.EXAMPLE
.\tools\DwmWatch.ps1 BeginIncident
# Restart DWM manually after the command finishes.
.\tools\DwmWatch.ps1 EndIncident

.EXAMPLE
.\tools\DwmWatch.ps1 Status

.EXAMPLE
.\tools\DwmWatch.ps1 StopMonitor
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('StartMonitor', 'CaptureIncident', 'BeginIncident', 'EndIncident', 'Status', 'StopMonitor')]
    [string] $Action = 'Status',

    [string] $OutputRoot,

    [ValidateRange(5, 300)]
    [int] $MonitorIntervalSeconds = 30,

    [ValidateRange(128, 4096)]
    [int] $MonitorMaxMegabytes = 1024,

    [ValidateRange(5, 600)]
    [int] $PostRestartSeconds = 10,

    [ValidateRange(0, 30)]
    [int] $PostTraceSeconds = 5,

    [ValidateRange(1, 10)]
    [int] $IncidentWaitMinutes = 5,

    [ValidateRange(256, 8192)]
    [int] $MaxIncidentTraceMegabytes = 2048,

    [ValidateRange(2, 100)]
    [int] $MinimumFreeSpaceGigabytes = 8,

    [ValidateRange(1, 720)]
    [int] $EventLookbackHours = 168,

    [switch] $VerboseTrace,

    [switch] $FileModeTrace,

    [switch] $IncludeCommandLines,

    [switch] $AllowUnboundedSplitTrace,

    [switch] $SkipHashes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:ScriptVersion = '1.0.0'
$script:BoundedIncidentWorkflow = $false
$script:StateFileName = 'state.json'
$script:StateBackupFileName = 'state.previous.json'
$script:SystemDirectory = [Environment]::SystemDirectory
$script:WindowsDirectory = [IO.Directory]::GetParent($script:SystemDirectory).FullName
$script:WprExe = Join-Path $script:SystemDirectory 'wpr.exe'
$script:LogmanExe = Join-Path $script:SystemDirectory 'logman.exe'
$script:WevtutilExe = Join-Path $script:SystemDirectory 'wevtutil.exe'
$script:DxdiagExe = Join-Path $script:SystemDirectory 'dxdiag.exe'
$script:PnputilExe = Join-Path $script:SystemDirectory 'pnputil.exe'
$script:PowercfgExe = Join-Path $script:SystemDirectory 'powercfg.exe'

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $commonData = [Environment]::GetFolderPath([Environment+SpecialFolder]::CommonApplicationData)
    $OutputRoot = Join-Path $commonData 'DwmWatch'
}
$OutputRoot = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($OutputRoot))
$script:StatePath = Join-Path $OutputRoot $script:StateFileName

function Assert-NoReparsePointInExistingPath {
    param([Parameter(Mandatory = $true)][string] $Path)

    $current = [IO.Path]::GetFullPath($Path)
    while (-not [string]::IsNullOrWhiteSpace($current)) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "DwmWatch refuses to use a path containing a junction, symbolic link, or other reparse point: $current"
            }
        }

        $parent = [IO.Directory]::GetParent($current)
        if ($null -eq $parent -or $parent.FullName.Equals($current, [StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        $current = $parent.FullName
    }
}

function Test-OutputRootAclSecure {
    if (-not (Test-Path -LiteralPath $OutputRoot -PathType Container)) {
        return $false
    }

    try {
        $acl = Get-Acl -LiteralPath $OutputRoot
        if (-not $acl.AreAccessRulesProtected) {
            return $false
        }
        $administratorSid = New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')
        $systemSid = New-Object Security.Principal.SecurityIdentifier('S-1-5-18')
        $ownerSid = $acl.Owner
        try {
            $ownerSid = ([Security.Principal.NTAccount]$acl.Owner).Translate([Security.Principal.SecurityIdentifier]).Value
        }
        catch {
            $ownerSid = [string]$acl.Owner
        }
        if ($ownerSid -notin @($administratorSid.Value, $systemSid.Value)) {
            return $false
        }

        $riskyRights = [Security.AccessControl.FileSystemRights]::WriteData -bor
            [Security.AccessControl.FileSystemRights]::AppendData -bor
            [Security.AccessControl.FileSystemRights]::WriteAttributes -bor
            [Security.AccessControl.FileSystemRights]::WriteExtendedAttributes -bor
            [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles -bor
            [Security.AccessControl.FileSystemRights]::Delete -bor
            [Security.AccessControl.FileSystemRights]::ChangePermissions -bor
            [Security.AccessControl.FileSystemRights]::TakeOwnership
        $rules = $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])
        foreach ($rule in $rules) {
            if ($rule.AccessControlType -ne [Security.AccessControl.AccessControlType]::Allow) {
                continue
            }
            $sid = [string]$rule.IdentityReference.Value
            if ($sid -in @($administratorSid.Value, $systemSid.Value)) {
                continue
            }
            if (($rule.FileSystemRights -band $riskyRights) -ne 0) {
                return $false
            }
        }
        return $true
    }
    catch {
        return $false
    }
}

function Set-SecureOutputRootAcl {
    $administratorSid = New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')
    $systemSid = New-Object Security.Principal.SecurityIdentifier('S-1-5-18')
    $currentUserSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $inheritance = [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit
    $propagation = [Security.AccessControl.PropagationFlags]::None
    $allow = [Security.AccessControl.AccessControlType]::Allow
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner($administratorSid)
    [void]$acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($administratorSid, [Security.AccessControl.FileSystemRights]::FullControl, $inheritance, $propagation, $allow)))
    [void]$acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($systemSid, [Security.AccessControl.FileSystemRights]::FullControl, $inheritance, $propagation, $allow)))
    $readRights = [Security.AccessControl.FileSystemRights]::ReadAndExecute -bor [Security.AccessControl.FileSystemRights]::Synchronize
    [void]$acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($currentUserSid, $readRights, $inheritance, $propagation, $allow)))
    Set-Acl -LiteralPath $OutputRoot -AclObject $acl
}

function Initialize-SecureOutputRoot {
    Assert-NoReparsePointInExistingPath -Path $OutputRoot
    if (Test-Path -LiteralPath $OutputRoot) {
        if (-not (Test-Path -LiteralPath $OutputRoot -PathType Container)) {
            throw "The configured OutputRoot is not a directory: $OutputRoot"
        }
        if (-not (Test-OutputRootAclSecure)) {
            throw "The existing OutputRoot is not protected against non-elevated writes. Preserve it and choose a new, dedicated -OutputRoot, or secure its ACL before running elevated: $OutputRoot"
        }
        return
    }

    [void](New-Item -ItemType Directory -Path $OutputRoot)
    Assert-NoReparsePointInExistingPath -Path $OutputRoot
    Set-SecureOutputRootAcl
    if (-not (Test-OutputRootAclSecure)) {
        throw "DwmWatch could not establish a secure ACL on OutputRoot: $OutputRoot"
    }
}

function New-Directory {
    param([Parameter(Mandatory = $true)][string] $Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        [void](New-Item -ItemType Directory -Path $Path)
    }
    Assert-NoReparsePointInExistingPath -Path $Path
}

function Write-Utf8Text {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [AllowEmptyString()][string] $Text
    )

    Set-Content -LiteralPath $Path -Value $Text -Encoding UTF8
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)] $Value,
        [Parameter(Mandatory = $true)][string] $Path,
        [int] $Depth = 10
    )

    $json = $Value | ConvertTo-Json -Depth $Depth
    Write-Utf8Text -Path $Path -Text $json
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Assert-Administrator {
    if (-not (Test-IsAdministrator)) {
        throw 'This action requires an elevated PowerShell window. Right-click PowerShell, choose Run as administrator, and run the command again.'
    }
    Initialize-SecureOutputRoot
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string] $FilePath,
        [string[]] $Arguments = @()
    )

    $outputLines = @()
    $exitCode = -1
    try {
        $outputLines = @(& $FilePath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
    }
    catch {
        $outputLines = @($_ | Out-String)
    }

    [pscustomobject]@{
        FilePath = $FilePath
        Arguments = ($Arguments -join ' ')
        ExitCode = [int]$exitCode
        Output = (($outputLines | Out-String -Width 4096).TrimEnd())
    }
}

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string] $Argument)

    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }
    $escaped = $Argument -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Invoke-NativeCommandWithTimeout {
    param(
        [Parameter(Mandatory = $true)][string] $FilePath,
        [string[]] $Arguments = @(),
        [ValidateRange(1, 300)][int] $TimeoutSeconds = 15
    )

    $startInfo = New-Object Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = (($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' ')
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object Diagnostics.Process
    $process.StartInfo = $startInfo
    $timedOut = $false
    $exitCode = -1
    $output = ''
    try {
        if (-not $process.Start()) {
            throw "Failed to start $FilePath."
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $timedOut = $true
            $process.Kill()
            $process.WaitForExit()
        }
        else {
            $process.WaitForExit()
        }
        $exitCode = $(if ($timedOut) { -2 } else { $process.ExitCode })
        $stdout = $stdoutTask.Result
        $stderr = $stderrTask.Result
        $output = (@($stdout, $stderr) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join [Environment]::NewLine
        if ($timedOut) {
            $output = "Timed out after $TimeoutSeconds seconds; only the process started by DwmWatch was terminated." + [Environment]::NewLine + $output
        }
    }
    catch {
        $output = $_ | Out-String
    }
    finally {
        $process.Dispose()
    }

    [pscustomobject]@{
        FilePath = $FilePath
        Arguments = ($Arguments -join ' ')
        ExitCode = [int]$exitCode
        TimedOut = $timedOut
        Output = $output.TrimEnd()
    }
}

function Write-NativeResult {
    param(
        [Parameter(Mandatory = $true)] $Result,
        [Parameter(Mandatory = $true)][string] $Path
    )

    $lines = @(
        "Command: $($Result.FilePath) $($Result.Arguments)"
        "ExitCode: $($Result.ExitCode)"
    )
    if ($Result.PSObject.Properties['TimedOut']) {
        $lines += "TimedOut: $($Result.TimedOut)"
    }
    $lines += ''
    $lines += $Result.Output
    $text = $lines -join [Environment]::NewLine
    Write-Utf8Text -Path $Path -Text $text
}

function Get-State {
    Assert-NoReparsePointInExistingPath -Path $OutputRoot
    if (-not (Test-Path -LiteralPath $script:StatePath -PathType Leaf)) {
        return $null
    }
    Assert-NoReparsePointInExistingPath -Path $script:StatePath

    try {
        $state = Get-Content -LiteralPath $script:StatePath -Raw | ConvertFrom-Json
        Assert-StatePaths -State $state
        return $state
    }
    catch {
        throw "The DwmWatch state file is unreadable: $($script:StatePath). Preserve it and repair or move it before continuing. $($_.Exception.Message)"
    }
}

function Test-PathWithinRoot {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Root
    )

    $fullPath = [IO.Path]::GetFullPath($Path)
    $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    if ($fullPath.Equals($fullRoot, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $rootPrefix = $fullRoot + [IO.Path]::DirectorySeparatorChar
    return $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)
}

function Assert-StatePaths {
    param([Parameter(Mandatory = $true)] $State)

    if ([int]$State.SchemaVersion -ne 1) {
        throw 'The saved DwmWatch schema version is unsupported.'
    }
    $tokenProperty = $State.PSObject.Properties['OwnershipToken']
    if ($null -eq $tokenProperty -or [string]$tokenProperty.Value -notmatch '^[0-9a-f]{32}$') {
        throw 'The saved DwmWatch ownership token is missing or invalid.'
    }
    $ownershipSuffix = '-' + ([string]$tokenProperty.Value).Substring(0, 8)

    if ([string]::IsNullOrWhiteSpace([string]$State.RunDirectory) -or -not (Test-PathWithinRoot -Path ([string]$State.RunDirectory) -Root $OutputRoot)) {
        throw 'The saved RunDirectory is outside the configured OutputRoot. DwmWatch refuses to use the state file.'
    }
    Assert-NoReparsePointInExistingPath -Path ([string]$State.RunDirectory)

    foreach ($propertyName in @('PendingIncidentDirectory', 'LastIncidentDirectory')) {
        $property = $State.PSObject.Properties[$propertyName]
        if ($null -ne $property -and -not [string]::IsNullOrWhiteSpace([string]$property.Value)) {
            if (-not (Test-PathWithinRoot -Path ([string]$property.Value) -Root ([string]$State.RunDirectory))) {
                throw "The saved $propertyName is outside RunDirectory. DwmWatch refuses to use the state file."
            }
            Assert-NoReparsePointInExistingPath -Path ([string]$property.Value)
        }
    }

    $collectorName = [string]$State.CollectorName
    if (-not [string]::IsNullOrWhiteSpace($collectorName) -and (
        $collectorName -notmatch '^DwmWatch-[0-9]{8}-[0-9]{6}-[0-9]{3}-[0-9a-f]{8}$' -or
        -not $collectorName.EndsWith($ownershipSuffix, [StringComparison]::OrdinalIgnoreCase)
    )) {
        throw 'The saved collector name is not a DwmWatch-owned name.'
    }
    $instanceProperty = $State.PSObject.Properties['WprInstanceName']
    if ($null -ne $instanceProperty) {
        $instanceName = [string]$instanceProperty.Value
        if (-not [string]::IsNullOrWhiteSpace($instanceName) -and (
            $instanceName -notmatch '^DwmWatch-[0-9]{8}-[0-9]{6}-[0-9]{3}-[0-9a-f]{8}$' -or
            -not $instanceName.EndsWith($ownershipSuffix, [StringComparison]::OrdinalIgnoreCase)
        )) {
            throw 'The saved WPR instance name is not a DwmWatch-owned name.'
        }
    }

    $monitorFilesProperty = $State.PSObject.Properties['MonitorFilesAtPause']
    if ($null -ne $monitorFilesProperty) {
        foreach ($record in @($monitorFilesProperty.Value)) {
            if ($null -ne $record -and -not [string]::IsNullOrWhiteSpace([string]$record.Path)) {
                if (-not (Test-PathWithinRoot -Path ([string]$record.Path) -Root ([string]$State.RunDirectory))) {
                    throw 'A saved monitor file is outside RunDirectory. DwmWatch refuses to copy it.'
                }
                Assert-NoReparsePointInExistingPath -Path ([string]$record.Path)
            }
        }
    }
}

function Set-StateProperty {
    param(
        [Parameter(Mandatory = $true)] $State,
        [Parameter(Mandatory = $true)][string] $Name,
        $Value
    )

    $State | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
}

function Save-State {
    param([Parameter(Mandatory = $true)] $State)

    New-Directory -Path $OutputRoot
    if (Test-Path -LiteralPath $script:StatePath) {
        Assert-NoReparsePointInExistingPath -Path $script:StatePath
    }
    Set-StateProperty -State $State -Name 'UpdatedUtc' -Value ([DateTime]::UtcNow.ToString('o'))

    $temporaryPath = Join-Path $OutputRoot ("state.$PID.tmp")
    $json = $State | ConvertTo-Json -Depth 12
    Set-Content -LiteralPath $temporaryPath -Value $json -Encoding UTF8

    if (Test-Path -LiteralPath $script:StatePath -PathType Leaf) {
        $backupPath = Join-Path $OutputRoot $script:StateBackupFileName
        [IO.File]::Replace($temporaryPath, $script:StatePath, $backupPath, $true)
    }
    else {
        [IO.File]::Move($temporaryPath, $script:StatePath)
    }
}

function New-State {
    param(
        [Parameter(Mandatory = $true)][string] $RunDirectory,
        [AllowNull()][string] $CollectorName
    )

    [pscustomobject][ordered]@{
        SchemaVersion = 1
        ScriptVersion = $script:ScriptVersion
        OwnershipToken = [Guid]::NewGuid().ToString('N')
        CreatedUtc = [DateTime]::UtcNow.ToString('o')
        UpdatedUtc = [DateTime]::UtcNow.ToString('o')
        RunDirectory = $RunDirectory
        CollectorName = $CollectorName
        CollectorCreated = $false
        MonitorStatus = 'NotStarted'
        MonitorPausedForIncident = $false
        MonitorFilesAtPause = @()
        PendingIncidentDirectory = $null
        IncidentPhase = $null
        IncidentStartedUtc = $null
        OldDwmPid = $null
        DwmSessionId = $null
        BootTimeAtIncidentUtc = $null
        WprOwned = $false
        WprLifecycle = 'NotStarted'
        WprProfileMode = $null
        WprInstanceName = $null
        IncidentWorkflow = $null
        IncidentIncludeCommandLinesBefore = $false
        IncidentVerboseTrace = $false
        IncidentFileModeTrace = $false
        IncidentMaxTraceMegabytes = $null
        IncidentMinimumFreeSpaceGigabytes = $null
        IncidentPostTraceSeconds = $null
        TraceSafetyLimitReached = $false
        TraceSafetyReason = $null
        TraceSafetyStopAttempted = $false
    }
}

function Add-TimelineEvent {
    param(
        [Parameter(Mandatory = $true)][string] $IncidentDirectory,
        [Parameter(Mandatory = $true)][string] $Event,
        [string] $Details = ''
    )

    $row = [pscustomobject]@{
        LocalTime = (Get-Date).ToString('o')
        UtcTime = [DateTime]::UtcNow.ToString('o')
        Event = $Event
        Details = $Details
    }
    $path = Join-Path $IncidentDirectory 'timeline.csv'
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $row | Export-Csv -LiteralPath $path -NoTypeInformation -Append -Encoding UTF8
    }
    else {
        $row | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding UTF8
    }
}

function Get-SafeValue {
    param([Parameter(Mandatory = $true)][scriptblock] $Expression)

    try {
        return & $Expression
    }
    catch {
        return $null
    }
}

function Get-CurrentSessionId {
    $currentProcess = Get-Process -Id $PID
    return [int]$currentProcess.SessionId
}

function Get-DwmForSession {
    param([Parameter(Mandatory = $true)][int] $SessionId)

    $dwmProcesses = @(Get-Process -Name 'dwm' -ErrorAction SilentlyContinue | Where-Object { $_.SessionId -eq $SessionId })
    if ($dwmProcesses.Count -eq 0) {
        return $null
    }
    return $dwmProcesses | Sort-Object Id -Descending | Select-Object -First 1
}

function Get-CurrentSessionDwm {
    return Get-DwmForSession -SessionId (Get-CurrentSessionId)
}

function Get-LastBootTimeUtc {
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem -OperationTimeoutSec 5
        return ([DateTime]$os.LastBootUpTime).ToUniversalTime().ToString('o')
    }
    catch {
        try {
            $uptimeSeconds = (Get-Counter '\System\System Up Time' -MaxSamples 1).CounterSamples[0].CookedValue
            return ([DateTime]::UtcNow.AddSeconds(-1 * [double]$uptimeSeconds)).ToString('o')
        }
        catch {
            return $null
        }
    }
}

function Get-CounterConfiguration {
    $definitions = @(
        [pscustomobject]@{
            Set = 'Process V2'
            Paths = @(
                '\Process V2(*)\Process ID'
                '\Process V2(*)\Creating Process ID'
                '\Process V2(*)\% Processor Time'
                '\Process V2(*)\Private Bytes'
                '\Process V2(*)\Working Set - Private'
                '\Process V2(*)\Working Set'
                '\Process V2(*)\Handle Count'
                '\Process V2(*)\Thread Count'
                '\Process V2(*)\Page File Bytes'
                '\Process V2(*)\Pool Paged Bytes'
                '\Process V2(*)\Pool Nonpaged Bytes'
                '\Process V2(*)\Page Faults/sec'
                '\Process V2(*)\IO Data Bytes/sec'
                '\Process V2(*)\IO Data Operations/sec'
            )
        }
        [pscustomobject]@{
            Set = 'GPU Process Memory'
            Paths = @(
                '\GPU Process Memory(*)\Dedicated Usage'
                '\GPU Process Memory(*)\Shared Usage'
                '\GPU Process Memory(*)\Local Usage'
                '\GPU Process Memory(*)\Non Local Usage'
                '\GPU Process Memory(*)\Total Committed'
            )
        }
        [pscustomobject]@{
            Set = 'GPU Engine'
            Paths = @(
                '\GPU Engine(*)\Utilization Percentage'
                '\GPU Engine(*)\Running Time'
            )
        }
        [pscustomobject]@{
            Set = 'GPU Adapter Memory'
            Paths = @(
                '\GPU Adapter Memory(*)\Dedicated Usage'
                '\GPU Adapter Memory(*)\Shared Usage'
                '\GPU Adapter Memory(*)\Total Committed'
            )
        }
        [pscustomobject]@{
            Set = 'GPU Local Adapter Memory'
            Paths = @('\GPU Local Adapter Memory(*)\Local Usage')
        }
        [pscustomobject]@{
            Set = 'GPU Non Local Adapter Memory'
            Paths = @('\GPU Non Local Adapter Memory(*)\Non Local Usage')
        }
        [pscustomobject]@{
            Set = 'Memory'
            Paths = @(
                '\Memory\Committed Bytes'
                '\Memory\Commit Limit'
                '\Memory\Available Bytes'
                '\Memory\Pool Paged Bytes'
                '\Memory\Pool Nonpaged Bytes'
                '\Memory\Free System Page Table Entries'
                '\Memory\Pages Input/sec'
                '\Memory\Page Reads/sec'
            )
        }
        [pscustomobject]@{
            Set = 'Processor Information'
            Paths = @(
                '\Processor Information(_Total)\% Processor Time'
                '\Processor Information(_Total)\% DPC Time'
                '\Processor Information(_Total)\% Interrupt Time'
                '\Processor Information(_Total)\DPC Rate'
                '\Processor Information(_Total)\Interrupts/sec'
            )
        }
        [pscustomobject]@{
            Set = 'System'
            Paths = @(
                '\System\System Up Time'
                '\System\Processor Queue Length'
                '\System\Processes'
                '\System\Threads'
                '\System\Context Switches/sec'
            )
        }
        [pscustomobject]@{
            Set = 'Objects'
            Paths = @(
                '\Objects\Processes'
                '\Objects\Threads'
                '\Objects\Events'
                '\Objects\Semaphores'
                '\Objects\Mutexes'
                '\Objects\Sections'
            )
        }
        [pscustomobject]@{
            Set = 'PhysicalDisk'
            Paths = @(
                '\PhysicalDisk(*)\Current Disk Queue Length'
                '\PhysicalDisk(*)\Avg. Disk sec/Transfer'
                '\PhysicalDisk(*)\Disk Transfers/sec'
            )
        }
    )

    $paths = New-Object System.Collections.Generic.List[string]
    $missingSets = New-Object System.Collections.Generic.List[string]
    foreach ($definition in $definitions) {
        try {
            [void](Get-Counter -ListSet $definition.Set -ErrorAction Stop)
            foreach ($path in $definition.Paths) {
                [void]$paths.Add($path)
            }
        }
        catch {
            [void]$missingSets.Add($definition.Set)
        }
    }

    [pscustomobject]@{
        Paths = $paths.ToArray()
        MissingSets = $missingSets.ToArray()
    }
}

function Get-LogmanStatus {
    param([AllowNull()][string] $CollectorName)

    if ([string]::IsNullOrWhiteSpace($CollectorName)) {
        return [pscustomobject]@{
            Exists = $false
            Running = $false
            KnownStopped = $true
            NotFound = $true
            Ambiguous = $false
            State = 'NotConfigured'
            Result = $null
        }
    }

    $result = Invoke-NativeCommandWithTimeout -FilePath $script:LogmanExe -Arguments @('query', $CollectorName) -TimeoutSeconds 15
    $running = $result.ExitCode -eq 0 -and $result.Output -match '(?im)^\s*Status:\s+Running\s*$'
    $stopped = $result.ExitCode -eq 0 -and $result.Output -match '(?im)^\s*Status:\s+(Stopped|Not Running)\s*$'
    $notFound = $result.ExitCode -ne 0 -and $result.Output -match '(?im)data collector set was not found|cannot find|does not exist'
    $ambiguous = -not $running -and -not $stopped -and -not $notFound
    [pscustomobject]@{
        Exists = $running -or $stopped
        Running = $running
        KnownStopped = $stopped -or $notFound
        NotFound = $notFound
        Ambiguous = $ambiguous
        State = $(if ($running) { 'Running' } elseif ($stopped) { 'Stopped' } elseif ($notFound) { 'NotFound' } else { 'Ambiguous' })
        Result = $result
    }
}

function Get-WprStatus {
    param([AllowNull()][string] $InstanceName)

    $arguments = @('-status')
    if (-not [string]::IsNullOrWhiteSpace($InstanceName)) {
        $arguments += @('-instancename', $InstanceName)
    }
    $result = Invoke-NativeCommandWithTimeout -FilePath $script:WprExe -Arguments $arguments -TimeoutSeconds 15
    $notRecording = $result.ExitCode -eq 0 -and $result.Output -match '(?im)^\s*WPR is not recording\s*$'
    [pscustomobject]@{
        Result = $result
        NotRecording = $notRecording
        Recording = ($result.ExitCode -eq 0 -and -not $notRecording)
    }
}

function Flush-WprInstance {
    param([Parameter(Mandatory = $true)][string] $InstanceName)

    return Invoke-NativeCommandWithTimeout -FilePath $script:WprExe -Arguments @(
        '-flush'
        '-instancename'
        $InstanceName
    ) -TimeoutSeconds 15
}

function Initialize-GuiResourceApi {
    if ('DwmWatch.NativeMethods' -as [type]) {
        return
    }

    $source = @'
using System;
using System.Runtime.InteropServices;

namespace DwmWatch
{
    public static class NativeMethods
    {
        [DllImport("user32.dll", SetLastError = true)]
        public static extern uint GetGuiResources(IntPtr hProcess, uint uiFlags);

        [DllImport("kernel32.dll")]
        public static extern void SetLastError(uint dwErrCode);
    }
}
'@
    Add-Type -TypeDefinition $source
}

function Get-ProcessInventory {
    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($process in @(Get-Process | Sort-Object Id)) {
        $threadCount = Get-SafeValue { $process.Threads.Count }
        $row = [pscustomobject][ordered]@{
            Name = $process.ProcessName
            Id = $process.Id
            SessionId = Get-SafeValue { $process.SessionId }
            StartTime = Get-SafeValue { $process.StartTime.ToString('o') }
            TotalProcessorSeconds = Get-SafeValue { [double]$process.TotalProcessorTime.TotalSeconds }
            PrivateBytes = Get-SafeValue { [long]$process.PrivateMemorySize64 }
            WorkingSetBytes = Get-SafeValue { [long]$process.WorkingSet64 }
            VirtualBytes = Get-SafeValue { [long]$process.VirtualMemorySize64 }
            PagedMemoryBytes = Get-SafeValue { [long]$process.PagedMemorySize64 }
            PagedPoolBytes = Get-SafeValue { [long]$process.PagedSystemMemorySize64 }
            NonPagedPoolBytes = Get-SafeValue { [long]$process.NonpagedSystemMemorySize64 }
            Handles = Get-SafeValue { [int]$process.HandleCount }
            Threads = $threadCount
            PriorityClass = Get-SafeValue { $process.PriorityClass.ToString() }
        }
        [void]$rows.Add($row)
    }
    return $rows.ToArray()
}

function Save-SanitizedProcessTree {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $ErrorPath
    )

    try {
        $rows = foreach ($process in @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop)) {
            $row = [ordered]@{
                Name = $process.Name
                ProcessId = $process.ProcessId
                ParentProcessId = $process.ParentProcessId
                SessionId = $process.SessionId
                CreationDate = Get-SafeValue { ([DateTime]$process.CreationDate).ToString('o') }
                ExecutablePath = $process.ExecutablePath
            }
            if ($IncludeCommandLines) {
                $row['CommandLine'] = $process.CommandLine
            }
            [pscustomobject]$row
        }
        $rows | Sort-Object ProcessId | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding UTF8
    }
    catch {
        Write-Utf8Text -Path $ErrorPath -Text ($_ | Out-String)
    }
}

function Get-GuiResourceInventory {
    param([Parameter(Mandatory = $true)][int] $SessionId)

    Initialize-GuiResourceApi
    $rows = New-Object System.Collections.Generic.List[object]

    foreach ($process in @(Get-Process | Where-Object { $_.SessionId -eq $SessionId } | Sort-Object Id)) {
        $gdiObjects = $null
        $userObjects = $null
        $gdiError = $null
        $userError = $null
        try {
            $handle = $process.Handle

            [DwmWatch.NativeMethods]::SetLastError(0)
            $gdiValue = [DwmWatch.NativeMethods]::GetGuiResources($handle, 0)
            $gdiLastError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            if ($gdiValue -eq 0 -and $gdiLastError -ne 0) {
                $gdiError = $gdiLastError
            }
            else {
                $gdiObjects = [long]$gdiValue
            }

            [DwmWatch.NativeMethods]::SetLastError(0)
            $userValue = [DwmWatch.NativeMethods]::GetGuiResources($handle, 1)
            $userLastError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            if ($userValue -eq 0 -and $userLastError -ne 0) {
                $userError = $userLastError
            }
            else {
                $userObjects = [long]$userValue
            }
        }
        catch {
            $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            if ($errorCode -eq 0) {
                $errorCode = -1
            }
            $gdiError = $errorCode
            $userError = $errorCode
        }

        [void]$rows.Add([pscustomobject]@{
            Name = $process.ProcessName
            Id = $process.Id
            SessionId = $SessionId
            GdiObjects = $gdiObjects
            UserObjects = $userObjects
            GdiWin32Error = $gdiError
            UserWin32Error = $userError
        })
    }
    return $rows.ToArray()
}

function Get-CounterRows {
    param(
        [Parameter(Mandatory = $true)][string[]] $CounterPaths,
        [int] $MaxSamples = 3,
        [int] $SampleInterval = 1,
        [Parameter(Mandatory = $true)][string] $ErrorPath
    )

    $counterErrors = @()
    $readings = @()
    try {
        $readings = @(Get-Counter -Counter $CounterPaths -SampleInterval $SampleInterval -MaxSamples $MaxSamples -ErrorAction SilentlyContinue -ErrorVariable counterErrors)
    }
    catch {
        $counterErrors += $_
    }

    if ($counterErrors.Count -gt 0) {
        Write-Utf8Text -Path $ErrorPath -Text (($counterErrors | Out-String -Width 4096).TrimEnd())
    }

    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($reading in $readings) {
        foreach ($sample in @($reading.CounterSamples)) {
            $counterNameProperty = $sample.PSObject.Properties['CounterName']
            $counterName = $(if ($null -ne $counterNameProperty) { $counterNameProperty.Value } else { $null })
            if ([string]::IsNullOrWhiteSpace([string]$counterName)) {
                $lastSeparator = $sample.Path.LastIndexOf('\')
                if ($lastSeparator -ge 0 -and $lastSeparator -lt ($sample.Path.Length - 1)) {
                    $counterName = $sample.Path.Substring($lastSeparator + 1)
                }
            }
            [void]$rows.Add([pscustomobject][ordered]@{
                TimestampLocal = $reading.Timestamp.ToString('o')
                TimestampUtc = $reading.Timestamp.ToUniversalTime().ToString('o')
                Path = $sample.Path
                InstanceName = $sample.InstanceName
                CounterName = $counterName
                CookedValue = [double]$sample.CookedValue
                RawValue = [string]$sample.RawValue
                SecondValue = [string]$sample.SecondValue
                Status = [long]$sample.Status
            })
        }
    }
    return $rows.ToArray()
}

function Get-SnapshotCounterRows {
    param(
        [Parameter(Mandatory = $true)][string[]] $CounterPaths,
        [Parameter(Mandatory = $true)][string] $Directory,
        [Parameter(Mandatory = $true)][int] $DwmPid
    )

    $groups = @(
        [pscustomobject]@{
            Name = 'process'
            Pattern = '^\\Process V2'
            MaxSamples = 1
        }
        [pscustomobject]@{
            Name = 'gpu-memory'
            Pattern = '^\\GPU (Process Memory|Adapter Memory|Local Adapter Memory|Non Local Adapter Memory)'
            MaxSamples = 1
        }
        [pscustomobject]@{
            Name = 'gpu-engine'
            Pattern = '^\\GPU Engine'
            MaxSamples = 2
        }
        [pscustomobject]@{
            Name = 'system'
            Pattern = '.*'
            MaxSamples = 2
        }
    )

    $remaining = @($CounterPaths)
    $allRows = New-Object System.Collections.Generic.List[object]
    foreach ($group in $groups) {
        if ($group.Name -eq 'system') {
            $selected = @($remaining)
        }
        else {
            $selected = @($remaining | Where-Object { $_ -match $group.Pattern })
        }
        if ($selected.Count -eq 0) {
            continue
        }
        $remaining = @($remaining | Where-Object { $selected -notcontains $_ })
        if ($group.Name -eq 'process') {
            $selected = @($selected | ForEach-Object {
                $_ -replace '\(\*\)', ('(dwm:' + $DwmPid + ')')
            })
        }
        $errorPath = Join-Path $Directory ('counter-errors-' + $group.Name + '.txt')
        $sampleCount = $(if ($group.Name -eq 'process') { 2 } else { $group.MaxSamples })
        $rows = @(Get-CounterRows -CounterPaths $selected -MaxSamples $sampleCount -SampleInterval 1 -ErrorPath $errorPath)
        foreach ($row in $rows) {
            [void]$allRows.Add($row)
        }
    }
    return $allRows.ToArray()
}

function Get-LatestCounterValue {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]] $Rows,
        [Parameter(Mandatory = $true)][string] $CounterName,
        [string] $InstancePattern = '.*',
        [switch] $Sum
    )

    $matchingRows = @($Rows | Where-Object {
        $_.CounterName -ieq $CounterName -and $_.InstanceName -match $InstancePattern
    })
    if ($matchingRows.Count -eq 0) {
        return $null
    }

    $latestTimestamp = ($matchingRows | Sort-Object TimestampUtc | Select-Object -Last 1).TimestampUtc
    $latestRows = @($matchingRows | Where-Object { $_.TimestampUtc -eq $latestTimestamp })
    if ($Sum) {
        return [double](($latestRows | Measure-Object -Property CookedValue -Sum).Sum)
    }
    return [double]$latestRows[0].CookedValue
}

function Save-GpuProcessSummary {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]] $CounterRows,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]] $ProcessRows,
        [Parameter(Mandatory = $true)][string] $Path
    )

    $gpuRows = @($CounterRows | Where-Object { $_.InstanceName -match '^pid_[0-9]+_' })
    if ($gpuRows.Count -eq 0) {
        Write-Utf8Text -Path $Path -Text 'No GPU process counter samples were available.'
        return
    }

    $latestRows = @(
        foreach ($counterGroup in @($gpuRows | Group-Object CounterName)) {
            $latestTimestampForCounter = ($counterGroup.Group | Sort-Object TimestampUtc | Select-Object -Last 1).TimestampUtc
            $counterGroup.Group | Where-Object { $_.TimestampUtc -eq $latestTimestampForCounter }
        }
    )
    $processNames = @{}
    foreach ($process in $ProcessRows) {
        $processNames[[string]$process.Id] = $process.Name
    }

    $aggregates = @{}
    foreach ($row in $latestRows) {
        $match = [regex]::Match($row.InstanceName, '^pid_([0-9]+)_')
        if (-not $match.Success) {
            continue
        }
        $processId = $match.Groups[1].Value
        $key = $processId + '|' + $row.CounterName.ToLowerInvariant()
        if (-not $aggregates.ContainsKey($key)) {
            $aggregates[$key] = 0.0
        }
        $aggregates[$key] += [double]$row.CookedValue
    }

    $resultRows = foreach ($key in @($aggregates.Keys | Sort-Object)) {
        $parts = $key.Split('|', 2)
        $processId = $parts[0]
        [pscustomobject]@{
            TimestampUtc = ($latestRows | Where-Object {
                $_.CounterName -ieq $parts[1]
            } | Sort-Object TimestampUtc | Select-Object -Last 1).TimestampUtc
            ProcessId = [int]$processId
            ProcessName = $(if ($processNames.ContainsKey($processId)) { $processNames[$processId] } else { $null })
            Counter = $parts[1]
            SumAcrossAdapterInstances = [double]$aggregates[$key]
        }
    }
    $resultRows | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding UTF8
}

function Get-TrustedNvidiaSmiPath {
    $programFiles = [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFiles)
    $candidates = @(
        (Join-Path $script:SystemDirectory 'nvidia-smi.exe')
        (Join-Path $programFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    ) | Select-Object -Unique

    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            continue
        }
        Assert-NoReparsePointInExistingPath -Path $candidate
        $signature = Get-AuthenticodeSignature -LiteralPath $candidate
        $subject = $(if ($null -ne $signature.SignerCertificate) { [string]$signature.SignerCertificate.Subject } else { '' })
        if ($signature.Status -eq [System.Management.Automation.SignatureStatus]::Valid -and $subject -match 'Microsoft|NVIDIA') {
            return $candidate
        }
    }
    return $null
}

function Save-NvidiaSnapshot {
    param([Parameter(Mandatory = $true)][string] $Directory)

    $nvidiaPath = Get-TrustedNvidiaSmiPath
    if ([string]::IsNullOrWhiteSpace($nvidiaPath)) {
        Write-Utf8Text -Path (Join-Path $Directory 'nvidia-smi-unavailable.txt') -Text 'nvidia-smi.exe was not found.'
        return
    }

    $queryFields = 'timestamp,index,name,uuid,driver_version,driver_model.current,pstate,temperature.gpu,power.draw,memory.total,memory.reserved,memory.used,memory.free,utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder,clocks.current.graphics,clocks.current.memory,pcie.link.gen.current,pcie.link.width.current'
    $commands = @(
        [pscustomobject]@{
            Name = 'nvidia-summary.txt'
            Arguments = @("--query-gpu=$queryFields", '--format=csv,noheader,nounits')
        }
        [pscustomobject]@{
            Name = 'nvidia-query.txt'
            Arguments = @('-q', '-d', 'PERFORMANCE,MEMORY,UTILIZATION,CLOCK,POWER,TEMPERATURE')
        }
        [pscustomobject]@{
            Name = 'nvidia-processes.txt'
            Arguments = @('pmon', '-c', '1')
        }
    )

    foreach ($command in $commands) {
        $result = Invoke-NativeCommandWithTimeout -FilePath $nvidiaPath -Arguments $command.Arguments -TimeoutSeconds 5
        Write-NativeResult -Result $result -Path (Join-Path $Directory $command.Name)
    }
}

function Save-Snapshot {
    param(
        [Parameter(Mandatory = $true)][string] $Directory,
        [Parameter(Mandatory = $true)][string] $Label,
        [Nullable[int]] $TargetSessionId = $null
    )

    New-Directory -Path $Directory
    $captureStartedUtc = [DateTime]::UtcNow.ToString('o')
    $sessionId = $(if ($null -eq $TargetSessionId) { Get-CurrentSessionId } else { [int]$TargetSessionId })
    $dwm = Get-DwmForSession -SessionId $sessionId
    if ($null -eq $dwm) {
        throw "No dwm.exe process was found in target session $sessionId."
    }

    $processRows = @(Get-ProcessInventory)
    $processRows | Export-Csv -LiteralPath (Join-Path $Directory 'processes.csv') -NoTypeInformation -Encoding UTF8
    Save-SanitizedProcessTree -Path (Join-Path $Directory 'process-tree.csv') -ErrorPath (Join-Path $Directory 'process-tree-error.txt')

    $guiRows = @()
    try {
        $guiRows = @(Get-GuiResourceInventory -SessionId $sessionId)
        $guiRows | Export-Csv -LiteralPath (Join-Path $Directory 'gui-resources.csv') -NoTypeInformation -Encoding UTF8
    }
    catch {
        Write-Utf8Text -Path (Join-Path $Directory 'gui-resources-error.txt') -Text ($_ | Out-String)
    }

    $counterConfiguration = Get-CounterConfiguration
    Write-JsonFile -Value $counterConfiguration -Path (Join-Path $Directory 'counter-configuration.json')
    $counterRows = @(Get-SnapshotCounterRows -CounterPaths $counterConfiguration.Paths -Directory $Directory -DwmPid $dwm.Id)
    if ($counterRows.Count -gt 0) {
        $counterRows | Export-Csv -LiteralPath (Join-Path $Directory 'counter-samples.csv') -NoTypeInformation -Encoding UTF8
        Save-GpuProcessSummary -CounterRows $counterRows -ProcessRows $processRows -Path (Join-Path $Directory 'gpu-process-summary.csv')
    }

    Save-NvidiaSnapshot -Directory $Directory

    $dwmGui = $guiRows | Where-Object { $_.Id -eq $dwm.Id } | Select-Object -First 1
    $dwmProcessInstancePattern = '^dwm:' + $dwm.Id + '$'
    $dwmGpuInstancePattern = '^pid_' + $dwm.Id + '_'

    $summary = [pscustomobject][ordered]@{
        Label = $Label
        CaptureStartedUtc = $captureStartedUtc
        CaptureCompletedUtc = [DateTime]::UtcNow.ToString('o')
        SessionId = $sessionId
        DwmPid = $dwm.Id
        DwmStartTime = Get-SafeValue { $dwm.StartTime.ToString('o') }
        LogicalProcessorCount = [Environment]::ProcessorCount
        DwmProcessorPercentOfOneLogicalProcessor = Get-LatestCounterValue -Rows $counterRows -CounterName '% Processor Time' -InstancePattern $dwmProcessInstancePattern
        DwmPrivateBytes = Get-SafeValue { [long]$dwm.PrivateMemorySize64 }
        DwmWorkingSetBytes = Get-SafeValue { [long]$dwm.WorkingSet64 }
        DwmWorkingSetPrivateBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Working Set - Private' -InstancePattern $dwmProcessInstancePattern
        DwmVirtualBytes = Get-SafeValue { [long]$dwm.VirtualMemorySize64 }
        DwmPageFileBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Page File Bytes' -InstancePattern $dwmProcessInstancePattern
        DwmHandles = Get-SafeValue { [int]$dwm.HandleCount }
        DwmThreads = Get-SafeValue { [int]$dwm.Threads.Count }
        DwmPagedPoolBytes = Get-SafeValue { [long]$dwm.PagedSystemMemorySize64 }
        DwmNonPagedPoolBytes = Get-SafeValue { [long]$dwm.NonpagedSystemMemorySize64 }
        DwmGdiObjects = $(if ($null -ne $dwmGui) { $dwmGui.GdiObjects } else { $null })
        DwmUserObjects = $(if ($null -ne $dwmGui) { $dwmGui.UserObjects } else { $null })
        DwmGpuDedicatedBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Dedicated Usage' -InstancePattern $dwmGpuInstancePattern -Sum
        DwmGpuSharedBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Shared Usage' -InstancePattern $dwmGpuInstancePattern -Sum
        DwmGpuLocalBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Local Usage' -InstancePattern $dwmGpuInstancePattern -Sum
        DwmGpuNonLocalBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Non Local Usage' -InstancePattern $dwmGpuInstancePattern -Sum
        DwmGpuTotalCommittedBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Total Committed' -InstancePattern $dwmGpuInstancePattern -Sum
        DwmGpuEngineReportedUtilizationSum = Get-LatestCounterValue -Rows $counterRows -CounterName 'Utilization Percentage' -InstancePattern $dwmGpuInstancePattern -Sum
        SystemCommittedBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Committed Bytes' -InstancePattern '^$'
        SystemCommitLimitBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Commit Limit' -InstancePattern '^$'
        SystemAvailableBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Available Bytes' -InstancePattern '^$'
        SystemPagedPoolBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Pool Paged Bytes' -InstancePattern '^$'
        SystemNonPagedPoolBytes = Get-LatestCounterValue -Rows $counterRows -CounterName 'Pool Nonpaged Bytes' -InstancePattern '^$'
        TotalDpcPercent = Get-LatestCounterValue -Rows $counterRows -CounterName '% DPC Time' -InstancePattern '^_total$'
        TotalInterruptPercent = Get-LatestCounterValue -Rows $counterRows -CounterName '% Interrupt Time' -InstancePattern '^_total$'
        ProcessorQueueLength = Get-LatestCounterValue -Rows $counterRows -CounterName 'Processor Queue Length' -InstancePattern '^$'
        Notes = 'GPU engine percentages are preserved per engine in counter-samples.csv. Their sum is diagnostic only because advertised engines can share hardware.'
    }
    Write-JsonFile -Value $summary -Path (Join-Path $Directory 'summary.json')
    return $summary
}

function Save-SnapshotOnce {
    param(
        [Parameter(Mandatory = $true)][string] $Directory,
        [Parameter(Mandatory = $true)][string] $Label,
        [Parameter(Mandatory = $true)][int] $TargetSessionId
    )

    $summaryPath = Join-Path $Directory 'summary.json'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        return Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    }
    return Save-Snapshot -Directory $Directory -Label $Label -TargetSessionId $TargetSessionId
}

function Get-RegistryCapture {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string[]] $Names
    )

    $values = [ordered]@{}
    if (Test-Path -LiteralPath $Path) {
        $item = Get-ItemProperty -LiteralPath $Path
        foreach ($name in $Names) {
            $property = $item.PSObject.Properties[$name]
            if ($null -ne $property) {
                $values[$name] = $property.Value
            }
            else {
                $values[$name] = $null
            }
        }
    }
    else {
        foreach ($name in $Names) {
            $values[$name] = $null
        }
    }

    [pscustomobject]@{
        Path = $Path
        Exists = (Test-Path -LiteralPath $Path)
        Values = [pscustomobject]$values
    }
}

function Convert-MonitorByteArray {
    param($Value)

    if ($null -eq $Value) {
        return $null
    }
    $bytes = @($Value | Where-Object { $_ -ne 0 })
    if ($bytes.Count -eq 0) {
        return ''
    }
    return [Text.Encoding]::ASCII.GetString([byte[]]$bytes)
}

function Save-CrashArtifactInventory {
    param(
        [Parameter(Mandatory = $true)][string] $Directory,
        [Parameter(Mandatory = $true)][int] $LookbackHours
    )

    New-Directory -Path $Directory
    $cutoffUtc = [DateTime]::UtcNow.AddHours(-1 * $LookbackHours)
    $commonData = [Environment]::GetFolderPath([Environment+SpecialFolder]::CommonApplicationData)
    $localData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
    $sources = [ordered]@{
        LiveKernelReports = Join-Path $script:WindowsDirectory 'LiveKernelReports'
        Minidumps = Join-Path $script:WindowsDirectory 'Minidump'
        FullMemoryDump = Join-Path $script:WindowsDirectory 'MEMORY.DMP'
        WerReportArchive = Join-Path $commonData 'Microsoft\Windows\WER\ReportArchive'
        WerReportQueue = Join-Path $commonData 'Microsoft\Windows\WER\ReportQueue'
        UserCrashDumps = Join-Path $localData 'CrashDumps'
    }
    $rows = New-Object System.Collections.Generic.List[object]
    $statusRows = New-Object System.Collections.Generic.List[object]

    foreach ($sourceName in $sources.Keys) {
        $sourcePath = [string]$sources[$sourceName]
        $exists = Test-Path -LiteralPath $sourcePath
        $captured = 0
        $message = ''
        try {
            if ($exists) {
                $items = @()
                if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
                    $items = @(Get-Item -LiteralPath $sourcePath -Force)
                }
                else {
                    $items = @(Get-ChildItem -LiteralPath $sourcePath -Recurse -File -Force -ErrorAction SilentlyContinue)
                }
                foreach ($item in @($items | Where-Object { $_.LastWriteTimeUtc -ge $cutoffUtc } | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 2000)) {
                    [void]$rows.Add([pscustomobject][ordered]@{
                        Source = $sourceName
                        FullName = $item.FullName
                        Length = $item.Length
                        CreationTimeUtc = $item.CreationTimeUtc.ToString('o')
                        LastWriteTimeUtc = $item.LastWriteTimeUtc.ToString('o')
                        Extension = $item.Extension
                    })
                    $captured++
                }
                if ($items.Count -gt 2000) {
                    $message = 'The output is capped at the 2,000 most recently modified files in this source.'
                }
            }
        }
        catch {
            $message = $_.Exception.Message
        }
        [void]$statusRows.Add([pscustomobject]@{
            Source = $sourceName
            Path = $sourcePath
            Exists = $exists
            RecentFilesCaptured = $captured
            Message = $message
        })
    }

    if ($rows.Count -gt 0) {
        $rows | Export-Csv -LiteralPath (Join-Path $Directory 'recent-crash-artifacts.csv') -NoTypeInformation -Encoding UTF8
    }
    else {
        Write-Utf8Text -Path (Join-Path $Directory 'recent-crash-artifacts-none.txt') -Text "No crash-artifact files modified within the last $LookbackHours hours were found."
    }
    $statusRows | Export-Csv -LiteralPath (Join-Path $Directory 'crash-artifact-sources.csv') -NoTypeInformation -Encoding UTF8
}

function Save-SystemInventory {
    param(
        [Parameter(Mandatory = $true)][string] $Directory,
        [switch] $IncludeDxDiag
    )

    New-Directory -Path $Directory

    $windowsVersion = Get-RegistryCapture -Path 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -Names @(
        'ProductName'
        'DisplayVersion'
        'CurrentBuild'
        'CurrentBuildNumber'
        'UBR'
        'BuildLabEx'
        'EditionID'
        'InstallationType'
    )
    $graphicsDrivers = Get-RegistryCapture -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -Names @(
        'HwSchMode'
        'TdrLevel'
        'TdrDelay'
        'TdrDdiDelay'
        'TdrLimitTime'
        'TdrLimitCount'
        'PlatformSupportMiracast'
    )
    $machineDwm = Get-RegistryCapture -Path 'HKLM:\SOFTWARE\Microsoft\Windows\Dwm' -Names @(
        'OverlayTestMode'
        'ForceEffectMode'
        'DisableDeviceBitmaps'
    )
    $userDwm = Get-RegistryCapture -Path 'HKCU:\Software\Microsoft\Windows\DWM' -Names @(
        'Composition'
        'ColorPrevalence'
        'EnableAeroPeek'
        'AlwaysHibernateThumbnails'
        'OverlayTestMode'
    )
    Write-JsonFile -Value @($windowsVersion, $graphicsDrivers, $machineDwm, $userDwm) -Path (Join-Path $Directory 'graphics-registry.json')

    try {
        Get-CimInstance -ClassName Win32_VideoController |
            Select-Object Name, PNPDeviceID, DriverVersion, DriverDate, VideoProcessor, VideoModeDescription,
                CurrentHorizontalResolution, CurrentVerticalResolution, CurrentRefreshRate, CurrentBitsPerPixel,
                AdapterCompatibility, AdapterRAM, Status |
            Export-Csv -LiteralPath (Join-Path $Directory 'video-controllers.csv') -NoTypeInformation -Encoding UTF8
    }
    catch {
        Write-Utf8Text -Path (Join-Path $Directory 'video-controllers-error.txt') -Text ($_ | Out-String)
    }

    try {
        Get-CimInstance -ClassName Win32_PnPSignedDriver |
            Where-Object { $_.DeviceClass -ieq 'DISPLAY' } |
            Select-Object DeviceName, DeviceID, Manufacturer, DriverProviderName, DriverVersion, DriverDate,
                InfName, IsSigned, Signer |
            Export-Csv -LiteralPath (Join-Path $Directory 'display-drivers.csv') -NoTypeInformation -Encoding UTF8
    }
    catch {
        Write-Utf8Text -Path (Join-Path $Directory 'display-drivers-error.txt') -Text ($_ | Out-String)
    }

    try {
        $monitorRows = foreach ($monitor in @(Get-CimInstance -Namespace 'root\wmi' -ClassName WmiMonitorID)) {
            [pscustomobject]@{
                InstanceName = $monitor.InstanceName
                Active = $monitor.Active
                Manufacturer = Convert-MonitorByteArray $monitor.ManufacturerName
                Model = Convert-MonitorByteArray $monitor.UserFriendlyName
                Serial = Convert-MonitorByteArray $monitor.SerialNumberID
                ProductCodeId = $monitor.ProductCodeID
                WeekOfManufacture = $monitor.WeekOfManufacture
                YearOfManufacture = $monitor.YearOfManufacture
            }
        }
        $monitorRows | Export-Csv -LiteralPath (Join-Path $Directory 'monitors.csv') -NoTypeInformation -Encoding UTF8
    }
    catch {
        Write-Utf8Text -Path (Join-Path $Directory 'monitors-error.txt') -Text ($_ | Out-String)
    }

    $pnpCommands = @(
        [pscustomobject]@{
            FileName = 'pnp-display-devices.txt'
            Arguments = @('/enum-devices', '/class', 'Display', '/drivers', '/services', '/stack')
        }
        [pscustomobject]@{
            FileName = 'pnp-display-drivers.txt'
            Arguments = @('/enum-drivers', '/class', 'Display')
        }
    )
    foreach ($command in $pnpCommands) {
        $result = Invoke-NativeCommandWithTimeout -FilePath $script:PnputilExe -Arguments $command.Arguments -TimeoutSeconds 30
        Write-NativeResult -Result $result -Path (Join-Path $Directory $command.FileName)
    }

    $powerResult = Invoke-NativeCommandWithTimeout -FilePath $script:PowercfgExe -Arguments @('/getactivescheme') -TimeoutSeconds 15
    Write-NativeResult -Result $powerResult -Path (Join-Path $Directory 'active-power-scheme.txt')

    $channelsResult = Invoke-NativeCommand -FilePath $script:WevtutilExe -Arguments @('el')
    $graphicsChannels = @($channelsResult.Output -split [Environment]::NewLine | Where-Object {
        $_ -match 'Dwm|DxgKrnl|GraphicsCapture|IndirectDisplays|Winlogon|WER|WerKernel'
    })
    Write-Utf8Text -Path (Join-Path $Directory 'available-graphics-event-channels.txt') -Text ($graphicsChannels -join [Environment]::NewLine)

    Save-NvidiaSnapshot -Directory $Directory
    Save-CrashArtifactInventory -Directory (Join-Path $Directory 'crash-artifacts') -LookbackHours $EventLookbackHours

    if ($IncludeDxDiag) {
        $dxdiagPath = Join-Path $Directory 'dxdiag.txt'
        try {
            $quote = [char]34
            $argumentLine = '/dontskip /whql:off /t ' + $quote + $dxdiagPath + $quote
            $process = Start-Process -FilePath $script:DxdiagExe -ArgumentList $argumentLine -PassThru -WindowStyle Hidden
            if (-not $process.WaitForExit(120000)) {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                Write-Utf8Text -Path (Join-Path $Directory 'dxdiag-timeout.txt') -Text 'dxdiag exceeded 120 seconds and only the dxdiag process started by DwmWatch was stopped.'
            }
        }
        catch {
            Write-Utf8Text -Path (Join-Path $Directory 'dxdiag-error.txt') -Text ($_ | Out-String)
        }
    }

    $metadata = [pscustomobject][ordered]@{
        CapturedLocal = (Get-Date).ToString('o')
        CapturedUtc = [DateTime]::UtcNow.ToString('o')
        ComputerName = $env:COMPUTERNAME
        PowerShellVersion = $PSVersionTable.PSVersion.ToString()
        PowerShellEdition = $(if ($PSVersionTable.PSObject.Properties['PSEdition']) { $PSVersionTable.PSEdition } else { 'Desktop' })
        IsAdministrator = Test-IsAdministrator
        LogicalProcessorCount = [Environment]::ProcessorCount
        LastBootTimeUtc = Get-LastBootTimeUtc
        CurrentSessionId = Get-CurrentSessionId
        IncludeCommandLines = [bool]$IncludeCommandLines
    }
    Write-JsonFile -Value $metadata -Path (Join-Path $Directory 'capture-metadata.json')
}

function Get-EventLogChannels {
    @(
        'System'
        'Application'
        'Microsoft-Windows-DxgKrnl-Admin'
        'Microsoft-Windows-DxgKrnl-Operational'
        'Microsoft-Windows-Kernel-EventTracing/Admin'
        'Microsoft-Windows-Diagnostics-Performance/Operational'
        'Microsoft-Windows-Winlogon/Operational'
        'Microsoft-Windows-TerminalServices-LocalSessionManager/Operational'
        'Microsoft-Windows-WER-Diag/Operational'
        'Microsoft-Windows-WER-PayloadHealth/Operational'
        'Microsoft-Windows-WerKernel/Operational'
        'Microsoft-Windows-DriverFrameworks-UserMode/Operational'
        'Microsoft-Windows-Kernel-PnP/Configuration'
        'Microsoft-Windows-DeviceSetupManager/Admin'
        'Microsoft-Windows-DeviceSetupManager/Operational'
        'Microsoft-Windows-Kernel-Power/Thermal-Operational'
        'Microsoft-Windows-Win32k/Operational'
        'Microsoft-Windows-Dwm-API/Diagnostic'
        'Microsoft-Windows-Dwm-Compositor/Diagnostic'
        'Microsoft-Windows-Dwm-Core/Diagnostic'
        'Microsoft-Windows-Dwm-Dwm/Diagnostic'
        'Microsoft-Windows-Dwm-Redir/Diagnostic'
        'Microsoft-Windows-Dwm-Udwm/Diagnostic'
        'Microsoft-Windows-DxgKrnl/Diagnostic'
        'Microsoft-Windows-GraphicsCapture-API/Diagnostic'
        'Microsoft-Windows-IndirectDisplays-ClassExtension-Events/Diagnostic'
    )
}

function ConvertTo-SafeFileName {
    param([Parameter(Mandatory = $true)][string] $Name)

    return ($Name -replace '[^A-Za-z0-9_.-]', '-')
}

function Export-RecentEventLogs {
    param(
        [Parameter(Mandatory = $true)][string] $Directory,
        [Parameter(Mandatory = $true)][int] $LookbackHours
    )

    New-Directory -Path $Directory
    $milliseconds = [long]$LookbackHours * 60L * 60L * 1000L
    $xpath = '*[System[TimeCreated[timediff(@SystemTime) <= ' + $milliseconds + ']]]'
    $statusRows = New-Object System.Collections.Generic.List[object]

    foreach ($channel in @(Get-EventLogChannels)) {
        $configuration = Invoke-NativeCommand -FilePath $script:WevtutilExe -Arguments @('gl', $channel)
        $exists = $configuration.ExitCode -eq 0
        $enabled = $exists -and $configuration.Output -match '(?im)^\s*enabled:\s*true\s*$'
        $isCore = $channel -eq 'System' -or $channel -eq 'Application'
        $exported = $false
        $exitCode = $configuration.ExitCode
        $message = ''

        if ($exists -and ($enabled -or $isCore)) {
            $fileName = (ConvertTo-SafeFileName $channel) + '.evtx'
            $destination = Join-Path $Directory $fileName
            $export = Invoke-NativeCommand -FilePath $script:WevtutilExe -Arguments @(
                'epl'
                $channel
                $destination
                ('/q:' + $xpath)
                '/ow:true'
            )
            $exported = $export.ExitCode -eq 0
            $exitCode = $export.ExitCode
            $message = $export.Output
        }
        elseif ($exists) {
            $message = 'Channel exists but is disabled; it was not enabled or exported.'
        }
        else {
            $message = $configuration.Output
        }

        [void]$statusRows.Add([pscustomobject]@{
            Channel = $channel
            Exists = $exists
            Enabled = $enabled
            Exported = $exported
            ExitCode = $exitCode
            Message = $message
        })
    }

    $statusRows | Export-Csv -LiteralPath (Join-Path $Directory 'event-export-status.csv') -NoTypeInformation -Encoding UTF8

    $startTime = (Get-Date).AddHours(-1 * $LookbackHours)
    $keyEvents = New-Object System.Collections.Generic.List[object]
    foreach ($logName in @('System', 'Application')) {
        try {
            $events = @(Get-WinEvent -FilterHashtable @{ LogName = $logName; StartTime = $startTime } -ErrorAction Stop)
            foreach ($event in $events) {
                $interesting = $event.ProviderName -match 'nvlddmkm|Display|DxgKrnl|WHEA|Kernel-Power|Kernel-Boot|DriverFrameworks|Windows Error Reporting|WER-SystemErrorReporting|Application Error|Application Hang|Desktop Window Manager|Dwm|User32|Winlogon|BugCheck'
                if (-not $interesting) {
                    continue
                }
                [void]$keyEvents.Add([pscustomobject][ordered]@{
                    TimeCreated = $(if ($null -ne $event.TimeCreated) { $event.TimeCreated.ToString('o') } else { $null })
                    LogName = $event.LogName
                    ProviderName = $event.ProviderName
                    Id = $event.Id
                    Level = $event.LevelDisplayName
                    RecordId = $event.RecordId
                    ProcessId = $event.ProcessId
                    ThreadId = $event.ThreadId
                    Message = $event.Message
                })
            }
        }
        catch {
            Write-Utf8Text -Path (Join-Path $Directory ((ConvertTo-SafeFileName $logName) + '-summary-error.txt')) -Text ($_ | Out-String)
        }
    }
    if ($keyEvents.Count -gt 0) {
        $keyEvents | Sort-Object TimeCreated | Export-Csv -LiteralPath (Join-Path $Directory 'key-events.csv') -NoTypeInformation -Encoding UTF8
    }
}

function Write-Comparison {
    param([Parameter(Mandatory = $true)][string] $IncidentDirectory)

    $summaryPaths = [ordered]@{
        Before = Join-Path $IncidentDirectory 'before\summary.json'
        AfterImmediate = Join-Path $IncidentDirectory 'after-immediate\summary.json'
        AfterSettled = Join-Path $IncidentDirectory 'after-settled\summary.json'
    }
    $summaries = @{}
    foreach ($key in $summaryPaths.Keys) {
        $path = $summaryPaths[$key]
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $summaries[$key] = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        }
    }
    if (-not $summaries.ContainsKey('Before')) {
        return
    }

    $metrics = @(
        'DwmPid'
        'DwmPrivateBytes'
        'DwmProcessorPercentOfOneLogicalProcessor'
        'DwmWorkingSetBytes'
        'DwmWorkingSetPrivateBytes'
        'DwmVirtualBytes'
        'DwmPageFileBytes'
        'DwmHandles'
        'DwmThreads'
        'DwmPagedPoolBytes'
        'DwmNonPagedPoolBytes'
        'DwmGdiObjects'
        'DwmUserObjects'
        'DwmGpuDedicatedBytes'
        'DwmGpuSharedBytes'
        'DwmGpuLocalBytes'
        'DwmGpuNonLocalBytes'
        'DwmGpuTotalCommittedBytes'
        'DwmGpuEngineReportedUtilizationSum'
        'SystemCommittedBytes'
        'SystemCommitLimitBytes'
        'SystemAvailableBytes'
        'SystemPagedPoolBytes'
        'SystemNonPagedPoolBytes'
        'TotalDpcPercent'
        'TotalInterruptPercent'
        'ProcessorQueueLength'
    )

    $rows = foreach ($metric in $metrics) {
        $beforeProperty = $summaries.Before.PSObject.Properties[$metric]
        $immediateProperty = $(if ($summaries.ContainsKey('AfterImmediate')) { $summaries.AfterImmediate.PSObject.Properties[$metric] } else { $null })
        $settledProperty = $(if ($summaries.ContainsKey('AfterSettled')) { $summaries.AfterSettled.PSObject.Properties[$metric] } else { $null })
        $before = $(if ($null -ne $beforeProperty) { $beforeProperty.Value } else { $null })
        $immediate = $(if ($null -ne $immediateProperty) { $immediateProperty.Value } else { $null })
        $settled = $(if ($null -ne $settledProperty) { $settledProperty.Value } else { $null })

        [pscustomobject]@{
            Metric = $metric
            Before = $before
            AfterImmediate = $immediate
            AfterSettled = $settled
            ImmediateDelta = $(if ($null -ne $before -and $null -ne $immediate) { [double]$immediate - [double]$before } else { $null })
            SettledDelta = $(if ($null -ne $before -and $null -ne $settled) { [double]$settled - [double]$before } else { $null })
        }
    }
    $rows | Export-Csv -LiteralPath (Join-Path $IncidentDirectory 'comparison.csv') -NoTypeInformation -Encoding UTF8
}

function Write-Hashes {
    param([Parameter(Mandatory = $true)][string] $Directory)

    $hashPath = Join-Path $Directory 'sha256.csv'
    $rows = foreach ($file in @(Get-ChildItem -LiteralPath $Directory -Recurse -File | Where-Object { $_.FullName -ne $hashPath })) {
        try {
            $hash = Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256
            [pscustomobject]@{
                RelativePath = $file.FullName.Substring($Directory.Length).TrimStart('\')
                Length = $file.Length
                LastWriteTimeUtc = $file.LastWriteTimeUtc.ToString('o')
                SHA256 = $hash.Hash
            }
        }
        catch {
            [pscustomobject]@{
                RelativePath = $file.FullName.Substring($Directory.Length).TrimStart('\')
                Length = $file.Length
                LastWriteTimeUtc = $file.LastWriteTimeUtc.ToString('o')
                SHA256 = 'ERROR: ' + $_.Exception.Message
            }
        }
    }
    $rows | Export-Csv -LiteralPath $hashPath -NoTypeInformation -Encoding UTF8
}

function New-RunDirectory {
    New-Directory -Path $OutputRoot
    for ($attempt = 0; $attempt -lt 100; $attempt++) {
        $suffix = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
        if ($attempt -gt 0) {
            $suffix = $suffix + '-' + $attempt
        }
        $candidate = Join-Path $OutputRoot ('run-' + $suffix)
        if (-not (Test-Path -LiteralPath $candidate)) {
            New-Directory -Path $candidate
            return $candidate
        }
        Start-Sleep -Milliseconds 10
    }
    throw 'Could not allocate a unique DwmWatch run directory.'
}

function Write-RunReadme {
    param([Parameter(Mandatory = $true)][string] $RunDirectory)

    $content = @'
DwmWatch evidence directory
===========================

This directory contains a bounded PerfMon history and, after an incident,
process/GPU snapshots, event logs, and a short WPR trace.

Normal workflow from an elevated PowerShell window:

  .\tools\DwmWatch.ps1 StartMonitor

When the lag is present, use the bounded one-command workflow:

  .\tools\DwmWatch.ps1 CaptureIncident

Leave that PowerShell window running and restart DWM manually when instructed.
The script detects the new PID and finishes the trace automatically. Default WPR
capture is circular memory mode, which bounds storage and preserves the most
useful final seconds around the DWM transition. -FileModeTrace is an explicit
crash-survival option; while the foreground process is alive, its raw-file size,
free-space margin, and timeout are polled. Abruptly killing PowerShell can bypass
that file-mode polling and can leave either mode's exact named session active
(memory mode remains storage-bounded), so use Status/StopMonitor promptly if
CaptureIncident is interrupted.

Split recovery workflow (snapshots only by default):

  .\tools\DwmWatch.ps1 BeginIncident

Restart DWM manually using the method already tested. The script deliberately
does not kill DWM. After the desktop is fully responsive:

  .\tools\DwmWatch.ps1 EndIncident

The split BeginIncident action does not start WPR unless
-AllowUnboundedSplitTrace is explicitly supplied. That opt-in defaults to
circular memory mode. Combining it with -FileModeTrace has no foreground size or
timeout guard after BeginIncident exits and should be used only deliberately.

Each PerfMon BLG segment is circular and bounded; sealing/resuming creates a new
bounded segment, so StopMonitor after the incident rather than accumulating many
segments indefinitely. Use Status at any time. StopMonitor ends only the exact
PerfMon/WPR names recorded in protected state and leaves definitions/evidence
intact.

Privacy:
ETL and EVTX files can contain usernames, paths, process/image information,
device IDs, IP addresses, registry data, event payloads, command text, and
secrets. The explicit process-tree CommandLine column is omitted unless
-IncludeCommandLines is supplied. No process dump or screenshot is collected.
The protected control/evidence root is administrator-write/current-user-read;
copy only the incident you need into an analysis workspace, and treat that copy
as equally sensitive.

Interpretation:
- Large DWM memory/handle/GPU growth that collapses under the new PID supports
  retained resources.
- Flat counts plus stalled presents or queues in the ETL supports a fence,
  MPO/direct-flip, or per-DWM driver-context problem.
- GPU totals that stay high when DWM's allocations fall support another owner
  or delayed WDDM/driver reclamation.
'@
    Write-Utf8Text -Path (Join-Path $RunDirectory 'README.txt') -Text $content
}

function Invoke-CaptureStep {
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [Parameter(Mandatory = $true)][string] $ErrorDirectory,
        [Parameter(Mandatory = $true)][scriptblock] $Operation
    )

    try {
        return & $Operation
    }
    catch {
        New-Directory -Path $ErrorDirectory
        $safeName = ConvertTo-SafeFileName $Name
        Write-Utf8Text -Path (Join-Path $ErrorDirectory ($safeName + '-error.txt')) -Text ($_ | Out-String)
        Write-Warning "$Name failed; the remaining capture will continue. See the error file."
        return $null
    }
}

function Start-DwmMonitor {
    Assert-Administrator
    $existingState = Get-State
    if ($null -ne $existingState) {
        if (-not [string]::IsNullOrWhiteSpace([string]$existingState.PendingIncidentDirectory)) {
            throw 'An incident is already pending. Run EndIncident before starting another monitor.'
        }
        if ([bool]$existingState.WprOwned) {
            $existingWprStatus = Get-WprStatus -InstanceName ([string]$existingState.WprInstanceName)
            if ($existingWprStatus.Recording) {
                throw "A DwmWatch-owned WPR instance is still active: $($existingState.WprInstanceName). Run StopMonitor before replacing monitor state."
            }
            if (-not $existingWprStatus.NotRecording) {
                throw 'The saved WPR instance status is ambiguous. DwmWatch will not replace the only ownership record; run Status and StopMonitor first.'
            }
            Set-StateProperty -State $existingState -Name 'WprOwned' -Value $false
            Set-StateProperty -State $existingState -Name 'WprLifecycle' -Value 'NotRecording'
            Save-State -State $existingState
        }
        $existingStatus = Get-LogmanStatus -CollectorName $existingState.CollectorName
        if ($existingStatus.Running) {
            Write-Output "DwmWatch is already running: $($existingState.CollectorName)"
            Write-Output "Run directory: $($existingState.RunDirectory)"
            return
        }
        if ($existingStatus.Ambiguous) {
            throw "The existing DwmWatch collector status is ambiguous. Its state will not be replaced: $($existingStatus.Result.Output)"
        }
    }

    $runDirectory = New-RunDirectory
    Write-RunReadme -RunDirectory $runDirectory
    $monitorDirectory = Join-Path $runDirectory 'monitor'
    $baselineDirectory = Join-Path $runDirectory 'baseline'
    New-Directory -Path $monitorDirectory
    New-Directory -Path $baselineDirectory

    $state = New-State -RunDirectory $runDirectory -CollectorName $null
    $collectorName = 'DwmWatch-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + ([string]$state.OwnershipToken).Substring(0, 8)
    Set-StateProperty -State $state -Name 'CollectorName' -Value $collectorName
    Save-State -State $state

    $counterConfiguration = Get-CounterConfiguration
    if ($counterConfiguration.Paths.Count -eq 0) {
        throw 'None of the requested Windows performance counter sets is available.'
    }
    $counterFile = Join-Path $monitorDirectory 'counters.txt'
    Set-Content -LiteralPath $counterFile -Value $counterConfiguration.Paths -Encoding ASCII
    Write-JsonFile -Value $counterConfiguration -Path (Join-Path $monitorDirectory 'counter-configuration.json')

    $hours = [Math]::Floor($MonitorIntervalSeconds / 3600)
    $minutes = [Math]::Floor(($MonitorIntervalSeconds % 3600) / 60)
    $seconds = $MonitorIntervalSeconds % 60
    $interval = '{0:00}:{1:00}:{2:00}' -f $hours, $minutes, $seconds
    $outputBase = Join-Path $monitorDirectory 'DwmWatch'

    $createArguments = @(
        'create'
        'counter'
        $collectorName
        '-cf'
        $counterFile
        '-si'
        $interval
        '-f'
        'bincirc'
        '-max'
        ([string]$MonitorMaxMegabytes)
        '-v'
        'nnnnnn'
        '-o'
        $outputBase
        '-m'
        'start'
        'stop'
    )
    $createResult = Invoke-NativeCommandWithTimeout -FilePath $script:LogmanExe -Arguments $createArguments -TimeoutSeconds 60
    Write-NativeResult -Result $createResult -Path (Join-Path $monitorDirectory 'logman-create.txt')
    if ($createResult.ExitCode -ne 0) {
        Set-StateProperty -State $state -Name 'MonitorStatus' -Value 'CreateFailed'
        Save-State -State $state
        throw "logman could not create the collector. See $(Join-Path $monitorDirectory 'logman-create.txt')."
    }

    Set-StateProperty -State $state -Name 'CollectorCreated' -Value $true
    Save-State -State $state
    $startResult = Invoke-NativeCommandWithTimeout -FilePath $script:LogmanExe -Arguments @('start', $collectorName) -TimeoutSeconds 30
    Write-NativeResult -Result $startResult -Path (Join-Path $monitorDirectory 'logman-start.txt')
    if ($startResult.ExitCode -ne 0) {
        Set-StateProperty -State $state -Name 'MonitorStatus' -Value 'StartFailed'
        Save-State -State $state
        throw "logman created but could not start the collector. See $(Join-Path $monitorDirectory 'logman-start.txt')."
    }

    Set-StateProperty -State $state -Name 'MonitorStatus' -Value 'Running'
    Save-State -State $state

    [void](Invoke-CaptureStep -Name 'baseline-snapshot' -ErrorDirectory $baselineDirectory -Operation {
        Save-Snapshot -Directory (Join-Path $baselineDirectory 'snapshot') -Label 'Healthy baseline'
    })
    [void](Invoke-CaptureStep -Name 'baseline-system-inventory' -ErrorDirectory $baselineDirectory -Operation {
        Save-SystemInventory -Directory (Join-Path $baselineDirectory 'system') -IncludeDxDiag
    })

    Write-Output 'DwmWatch monitoring started.'
    Write-Output "Collector: $collectorName"
    Write-Output "Run directory: $runDirectory"
    Write-Output "Interval: $MonitorIntervalSeconds seconds; circular limit: $MonitorMaxMegabytes MiB"
    if ($counterConfiguration.MissingSets.Count -gt 0) {
        Write-Warning ('Unavailable counter sets were skipped: ' + ($counterConfiguration.MissingSets -join ', '))
    }
    Write-Output 'When the lag is present, run this script again with CaptureIncident.'
}

function Ensure-StateForIncident {
    $state = Get-State
    if ($null -ne $state) {
        return $state
    }

    $runDirectory = New-RunDirectory
    Write-RunReadme -RunDirectory $runDirectory
    $state = New-State -RunDirectory $runDirectory -CollectorName $null
    Set-StateProperty -State $state -Name 'MonitorStatus' -Value 'NotConfigured'
    Save-State -State $state
    return $state
}

function Get-IncidentTraceSafety {
    param([Parameter(Mandatory = $true)][string] $IncidentDirectory)

    $rawDirectory = Join-Path $IncidentDirectory 'wpr-raw'
    $rawBytes = 0L
    if (Test-Path -LiteralPath $rawDirectory -PathType Container) {
        $rawFiles = @(Get-ChildItem -LiteralPath $rawDirectory -Recurse -File -ErrorAction SilentlyContinue)
        if ($rawFiles.Count -gt 0) {
            $rawBytes = [long](($rawFiles | Measure-Object -Property Length -Sum).Sum)
        }
    }

    $availableBytes = $null
    try {
        $directoryItem = Get-Item -LiteralPath $IncidentDirectory
        $drive = New-Object IO.DriveInfo($directoryItem.PSDrive.Root)
        $availableBytes = [long]$drive.AvailableFreeSpace
    }
    catch {
        $availableBytes = $null
    }

    $maximumRawBytes = [long]$MaxIncidentTraceMegabytes * 1MB
    $minimumFreeBytes = [long]$MinimumFreeSpaceGigabytes * 1GB
    $mergeMarginBytes = $rawBytes + 2GB
    $requiredFreeBytes = [Math]::Max($minimumFreeBytes, $mergeMarginBytes)
    $safe = $null -ne $availableBytes -and $rawBytes -lt $maximumRawBytes -and $availableBytes -ge $requiredFreeBytes
    $reason = 'Within configured limits.'
    if ($null -eq $availableBytes) {
        $reason = 'Free space could not be determined.'
    }
    elseif ($rawBytes -ge $maximumRawBytes) {
        $reason = "Raw WPR data reached the configured $MaxIncidentTraceMegabytes MiB limit."
    }
    elseif ($availableBytes -lt $requiredFreeBytes) {
        $reason = "Free space fell below the merge-safety requirement of $requiredFreeBytes bytes."
    }

    [pscustomobject][ordered]@{
        CheckedUtc = [DateTime]::UtcNow.ToString('o')
        Safe = $safe
        Reason = $reason
        RawBytes = $rawBytes
        MaximumRawBytes = $maximumRawBytes
        AvailableBytes = $availableBytes
        RequiredFreeBytes = $requiredFreeBytes
    }
}

function Start-OwnedWpr {
    param(
        [Parameter(Mandatory = $true)] $State,
        [Parameter(Mandatory = $true)][string] $IncidentDirectory
    )

    $rawDirectory = Join-Path $IncidentDirectory 'wpr-raw'
    if ($FileModeTrace) {
        New-Directory -Path $rawDirectory
    }
    $preflight = Get-IncidentTraceSafety -IncidentDirectory $IncidentDirectory
    Write-JsonFile -Value $preflight -Path (Join-Path $IncidentDirectory 'wpr-space-preflight.json')
    if (-not $preflight.Safe) {
        Set-StateProperty -State $State -Name 'WprOwned' -Value $false
        Set-StateProperty -State $State -Name 'WprLifecycle' -Value 'SkippedUnsafeDiskState'
        Save-State -State $State
        Write-Warning ("WPR was skipped: " + $preflight.Reason)
        return
    }

    $globalStatus = Get-WprStatus
    Write-NativeResult -Result $globalStatus.Result -Path (Join-Path $IncidentDirectory 'wpr-status-before.txt')
    if (-not $globalStatus.NotRecording) {
        Set-StateProperty -State $State -Name 'WprOwned' -Value $false
        Set-StateProperty -State $State -Name 'WprProfileMode' -Value 'SkippedExternalOrAmbiguous'
        Save-State -State $State
        Write-Warning 'WPR already appears active or its state is ambiguous. DwmWatch will not alter that recording.'
        return
    }

    $instanceName = 'DwmWatch-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + ([string]$State.OwnershipToken).Substring(0, 8)
    $generalProfile = $(if ($VerboseTrace) { 'GeneralProfile.Verbose' } else { 'GeneralProfile.Light' })
    $gpuProfile = $(if ($VerboseTrace) { 'GPU.Verbose' } else { 'GPU.Light' })
    $desktopProfile = $(if ($VerboseTrace) { 'DesktopComposition.Verbose' } else { 'DesktopComposition.Light' })
    $startArguments = @(
        '-start'
        $generalProfile
        '-start'
        $gpuProfile
        '-start'
        $desktopProfile
    )
    if ($FileModeTrace) {
        $startArguments += @('-filemode', '-recordtempto', $rawDirectory)
    }
    $startArguments += @('-instancename', $instanceName)
    Set-StateProperty -State $State -Name 'WprInstanceName' -Value $instanceName
    Set-StateProperty -State $State -Name 'WprProfileMode' -Value ("{0}-{1}" -f $(if ($VerboseTrace) { 'Verbose' } else { 'Light' }), $(if ($FileModeTrace) { 'File' } else { 'Memory' }))
    Set-StateProperty -State $State -Name 'WprOwned' -Value $true
    Set-StateProperty -State $State -Name 'WprLifecycle' -Value 'Starting'
    Save-State -State $State

    $startResult = Invoke-NativeCommandWithTimeout -FilePath $script:WprExe -Arguments $startArguments -TimeoutSeconds 60
    Write-NativeResult -Result $startResult -Path (Join-Path $IncidentDirectory 'wpr-start.txt')
    if ($FileModeTrace -and (Stop-OwnedWprAtSafetyLimit -State $State -IncidentDirectory $IncidentDirectory)) {
        return
    }
    $instanceStatus = Get-WprStatus -InstanceName $instanceName
    Write-NativeResult -Result $instanceStatus.Result -Path (Join-Path $IncidentDirectory 'wpr-status-after-start.txt')
    if ($instanceStatus.NotRecording) {
        Set-StateProperty -State $State -Name 'WprOwned' -Value $false
        Set-StateProperty -State $State -Name 'WprLifecycle' -Value 'StartFailedOrNotRecording'
        Save-State -State $State
        Write-Warning 'WPR did not start. Counter snapshots and event logs will still be collected.'
        return
    }
    if (-not $instanceStatus.Recording) {
        Set-StateProperty -State $State -Name 'WprOwned' -Value $true
        Set-StateProperty -State $State -Name 'WprLifecycle' -Value 'StartStatusAmbiguous'
        Save-State -State $State
        Write-Warning 'WPR start status is ambiguous. Ownership was retained so EndIncident or StopMonitor can safely recover the exact named instance.'
        return
    }

    Set-StateProperty -State $State -Name 'WprOwned' -Value $true
    Set-StateProperty -State $State -Name 'WprLifecycle' -Value 'Recording'
    Save-State -State $State
    $captureState = Invoke-NativeCommandWithTimeout -FilePath $script:WprExe -Arguments @(
        '-capturestateondemand'
        '-instancename'
        $instanceName
    ) -TimeoutSeconds 30
    Write-NativeResult -Result $captureState -Path (Join-Path $IncidentDirectory 'wpr-capture-state-before.txt')
    if ($FileModeTrace -and (Stop-OwnedWprAtSafetyLimit -State $State -IncidentDirectory $IncidentDirectory)) {
        return
    }
    $marker = Invoke-NativeCommandWithTimeout -FilePath $script:WprExe -Arguments @(
        '-marker'
        'DWM_LAG_DETECTED'
        '-instancename'
        $instanceName
    ) -TimeoutSeconds 15
    Write-NativeResult -Result $marker -Path (Join-Path $IncidentDirectory 'wpr-marker-before.txt')
    if ($FileModeTrace -and (Stop-OwnedWprAtSafetyLimit -State $State -IncidentDirectory $IncidentDirectory)) {
        return
    }
    $flush = Flush-WprInstance -InstanceName $instanceName
    Write-NativeResult -Result $flush -Path (Join-Path $IncidentDirectory 'wpr-flush-before.txt')
    if ($FileModeTrace) {
        [void](Stop-OwnedWprAtSafetyLimit -State $State -IncidentDirectory $IncidentDirectory)
    }
}

function Pause-OwnedMonitor {
    param(
        [Parameter(Mandatory = $true)] $State,
        [Parameter(Mandatory = $true)][string] $IncidentDirectory
    )

    $status = Get-LogmanStatus -CollectorName $State.CollectorName
    if ($status.Ambiguous) {
        Write-NativeResult -Result $status.Result -Path (Join-Path $IncidentDirectory 'logman-query-before-pause-ambiguous.txt')
        Write-Warning 'The continuous collector status is ambiguous. It was not stopped or treated as absent.'
        return
    }
    Set-StateProperty -State $State -Name 'MonitorWasRunningAtIncident' -Value ([bool]$status.Running)
    if (-not $status.Running) {
        Set-StateProperty -State $State -Name 'MonitorPausedForIncident' -Value $false
        Set-StateProperty -State $State -Name 'MonitorStatus' -Value $status.State
        Save-State -State $State
        return
    }

    Set-StateProperty -State $State -Name 'MonitorPausedForIncident' -Value $true
    Set-StateProperty -State $State -Name 'MonitorStatus' -Value 'PausingForIncident'
    Save-State -State $State

    $stopResult = Invoke-NativeCommandWithTimeout -FilePath $script:LogmanExe -Arguments @('stop', [string]$State.CollectorName) -TimeoutSeconds 30
    Write-NativeResult -Result $stopResult -Path (Join-Path $IncidentDirectory 'logman-pause.txt')
    if ($stopResult.ExitCode -ne 0) {
        $afterFailureStatus = Get-LogmanStatus -CollectorName $State.CollectorName
        if ($afterFailureStatus.Running) {
            Set-StateProperty -State $State -Name 'MonitorPausedForIncident' -Value $false
            Set-StateProperty -State $State -Name 'MonitorStatus' -Value 'Running'
            Save-State -State $State
            Write-Warning 'The circular monitor could not be paused and is still running; command output was preserved.'
            return
        }
        if ($afterFailureStatus.Ambiguous) {
            Set-StateProperty -State $State -Name 'MonitorStatus' -Value 'PauseStatusAmbiguous'
            Save-State -State $State
            Write-Warning 'The pause command failed or timed out and collector status is ambiguous. EndIncident will retry recovery.'
            return
        }
    }

    $monitorDirectory = Join-Path ([string]$State.RunDirectory) 'monitor'
    $files = @()
    if (Test-Path -LiteralPath $monitorDirectory -PathType Container) {
        $files = @(Get-ChildItem -LiteralPath $monitorDirectory -Filter '*.blg' -File |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1 |
            ForEach-Object {
            [pscustomobject]@{
                Path = $_.FullName
                Length = $_.Length
                LastWriteTimeUtc = $_.LastWriteTimeUtc.ToString('o')
            }
        })
    }
    Set-StateProperty -State $State -Name 'MonitorFilesAtPause' -Value $files
    Set-StateProperty -State $State -Name 'MonitorPausedForIncident' -Value $true
    Set-StateProperty -State $State -Name 'MonitorStatus' -Value 'PausedForIncident'
    Save-State -State $State

    $resumeResult = Invoke-NativeCommandWithTimeout -FilePath $script:LogmanExe -Arguments @('start', [string]$State.CollectorName) -TimeoutSeconds 30
    Write-NativeResult -Result $resumeResult -Path (Join-Path $IncidentDirectory 'logman-resume-after-seal.txt')
    if ($resumeResult.ExitCode -eq 0) {
        Set-StateProperty -State $State -Name 'MonitorPausedForIncident' -Value $false
        Set-StateProperty -State $State -Name 'MonitorStatus' -Value 'Running'
    }
    else {
        Write-Warning 'The continuous collector was sealed but could not be resumed. EndIncident will retry.'
    }
    Save-State -State $State
}

function Begin-DwmIncident {
    Assert-Administrator
    $state = Ensure-StateForIncident
    if (-not [string]::IsNullOrWhiteSpace([string]$state.PendingIncidentDirectory)) {
        throw "An incident is already pending at $($state.PendingIncidentDirectory). Restart DWM manually if needed, then run EndIncident."
    }
    if ([bool]$state.WprOwned) {
        $priorWprStatus = Get-WprStatus -InstanceName ([string]$state.WprInstanceName)
        if ($priorWprStatus.Recording) {
            throw "A prior DwmWatch-owned WPR instance is still active: $($state.WprInstanceName). Run StopMonitor to preserve and stop it before beginning another incident."
        }
        if (-not $priorWprStatus.NotRecording) {
            throw 'A prior DwmWatch-owned WPR instance has ambiguous status. Run Status and preserve the existing evidence before continuing.'
        }
        Set-StateProperty -State $state -Name 'WprOwned' -Value $false
        Save-State -State $state
    }

    $dwm = Get-CurrentSessionDwm
    if ($null -eq $dwm) {
        throw 'No dwm.exe process was found in the current session.'
    }

    $incidentDirectory = Join-Path ([string]$state.RunDirectory) ('incident-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    New-Directory -Path $incidentDirectory
    Set-StateProperty -State $state -Name 'PendingIncidentDirectory' -Value $incidentDirectory
    Set-StateProperty -State $state -Name 'IncidentPhase' -Value 'CapturingBefore'
    Set-StateProperty -State $state -Name 'IncidentStartedUtc' -Value ([DateTime]::UtcNow.ToString('o'))
    Set-StateProperty -State $state -Name 'OldDwmPid' -Value ([int]$dwm.Id)
    Set-StateProperty -State $state -Name 'DwmSessionId' -Value ([int]$dwm.SessionId)
    Set-StateProperty -State $state -Name 'BootTimeAtIncidentUtc' -Value (Get-LastBootTimeUtc)
    Set-StateProperty -State $state -Name 'WprOwned' -Value $false
    Set-StateProperty -State $state -Name 'WprInstanceName' -Value $null
    Set-StateProperty -State $state -Name 'WprLifecycle' -Value 'NotStarted'
    Set-StateProperty -State $state -Name 'WprProfileMode' -Value $null
    Set-StateProperty -State $state -Name 'InitialDwmPid' -Value $null
    Set-StateProperty -State $state -Name 'DwmChangedDuringBeforeCapture' -Value $false
    Set-StateProperty -State $state -Name 'LastMergedEtlPath' -Value $null
    Set-StateProperty -State $state -Name 'LastMergedEtlBytes' -Value $null
    Set-StateProperty -State $state -Name 'MergedTraceValid' -Value $null
    Set-StateProperty -State $state -Name 'IncidentWorkflow' -Value $(if ($script:BoundedIncidentWorkflow) { 'BoundedCaptureIncident' } else { 'SplitBeginEnd' })
    Set-StateProperty -State $state -Name 'IncidentIncludeCommandLinesBefore' -Value ([bool]$IncludeCommandLines)
    Set-StateProperty -State $state -Name 'IncidentVerboseTrace' -Value ([bool]$VerboseTrace)
    Set-StateProperty -State $state -Name 'IncidentFileModeTrace' -Value ([bool]$FileModeTrace)
    Set-StateProperty -State $state -Name 'IncidentMaxTraceMegabytes' -Value $MaxIncidentTraceMegabytes
    Set-StateProperty -State $state -Name 'IncidentMinimumFreeSpaceGigabytes' -Value $MinimumFreeSpaceGigabytes
    Set-StateProperty -State $state -Name 'IncidentPostTraceSeconds' -Value $PostTraceSeconds
    Set-StateProperty -State $state -Name 'TraceSafetyLimitReached' -Value $false
    Set-StateProperty -State $state -Name 'TraceSafetyReason' -Value $null
    Set-StateProperty -State $state -Name 'TraceSafetyStopAttempted' -Value $false
    Set-StateProperty -State $state -Name 'TraceSafetyStopSucceeded' -Value $null
    Save-State -State $state
    Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'BEGIN_INCIDENT' -Details ("Old DWM PID: $($dwm.Id)")

    $beforeSummary = Invoke-CaptureStep -Name 'before-snapshot' -ErrorDirectory $incidentDirectory -Operation {
        Save-Snapshot -Directory (Join-Path $incidentDirectory 'before') -Label 'Degraded before DWM restart' -TargetSessionId ([int]$state.DwmSessionId)
    }
    if ($null -ne $beforeSummary -and [int]$beforeSummary.DwmPid -ne [int]$state.OldDwmPid) {
        Set-StateProperty -State $state -Name 'InitialDwmPid' -Value ([int]$state.OldDwmPid)
        Set-StateProperty -State $state -Name 'OldDwmPid' -Value ([int]$beforeSummary.DwmPid)
        Set-StateProperty -State $state -Name 'DwmChangedDuringBeforeCapture' -Value $true
        Save-State -State $state
        Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'DWM_CHANGED_DURING_BEFORE_CAPTURE' -Details ("Initial PID changed to captured PID $($beforeSummary.DwmPid)")
        Write-Warning 'DWM changed while the before snapshot was being collected. The captured PID is now the comparison baseline, and the transition is explicitly flagged.'
    }

    # Start WPR only after the potentially slow before-snapshot work. This keeps
    # an interrupted or hung snapshot from leaving an owned trace active.
    if ($script:BoundedIncidentWorkflow -or $AllowUnboundedSplitTrace) {
        Start-OwnedWpr -State $state -IncidentDirectory $incidentDirectory
    }
    else {
        Set-StateProperty -State $state -Name 'WprOwned' -Value $false
        Set-StateProperty -State $state -Name 'WprLifecycle' -Value 'SkippedForUnboundedSplitWorkflow'
        Save-State -State $state
        Write-Utf8Text -Path (Join-Path $incidentDirectory 'wpr-skipped.txt') -Text 'BeginIncident was run without -AllowUnboundedSplitTrace. Temporal snapshots and the continuous PerfMon history are still captured; use CaptureIncident for bounded WPR.'
    }

    Pause-OwnedMonitor -State $state -IncidentDirectory $incidentDirectory

    if ($FileModeTrace -and [bool]$state.WprOwned) {
        [void](Stop-OwnedWprAtSafetyLimit -State $state -IncidentDirectory $incidentDirectory)
    }

    if ([bool]$state.WprOwned) {
        $marker = Invoke-NativeCommandWithTimeout -FilePath $script:WprExe -Arguments @(
            '-marker'
            'DWM_RESTART_IMMINENT'
            '-instancename'
            ([string]$state.WprInstanceName)
        ) -TimeoutSeconds 15
        Write-NativeResult -Result $marker -Path (Join-Path $incidentDirectory 'wpr-marker-restart-imminent.txt')
        $flush = Flush-WprInstance -InstanceName ([string]$state.WprInstanceName)
        Write-NativeResult -Result $flush -Path (Join-Path $incidentDirectory 'wpr-flush-restart-imminent.txt')
    }

    Set-StateProperty -State $state -Name 'IncidentPhase' -Value 'AwaitingDwmRestart'
    Save-State -State $state
    Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'AWAITING_DWM_RESTART' -Details 'DwmWatch will not restart DWM automatically.'

    Write-Output "The degraded state is captured at: $incidentDirectory"
    Write-Output "Old DWM PID: $($state.OldDwmPid); session: $($state.DwmSessionId)"
    Write-Output 'Now restart DWM manually using the method you already tested.'
    if ($script:BoundedIncidentWorkflow) {
        Write-Output 'Keep this window open. DwmWatch will detect the replacement PID and finish automatically.'
    }
    else {
        Write-Output 'After the desktop is responsive, run this script with EndIncident.'
    }
    if ($FileModeTrace -and [bool]$state.WprOwned) {
        Write-Output 'If the machine crashes or reboots, run EndIncident afterward; file-mode raw WPR fragments were directed into the incident folder.'
    }
    elseif ([bool]$state.WprOwned) {
        Write-Output 'The default WPR trace is circular memory mode: bounded and focused on the final seconds, but it will not survive a reboot.'
    }
    if ($AllowUnboundedSplitTrace -and -not $script:BoundedIncidentWorkflow) {
        if ($FileModeTrace) {
            Write-Warning 'The explicitly requested split file-mode WPR trace has no foreground timeout or size poll after this command exits. Run EndIncident promptly.'
        }
        else {
            Write-Warning 'The explicitly requested split memory-mode WPR trace is storage-bounded but has no foreground timeout. Run EndIncident promptly.'
        }
    }
}

function Copy-PausedMonitorFiles {
    param(
        [Parameter(Mandatory = $true)] $State,
        [Parameter(Mandatory = $true)][string] $IncidentDirectory
    )

    $sourceRecords = @($State.MonitorFilesAtPause)
    if ($sourceRecords.Count -eq 0) {
        return
    }
    Write-JsonFile -Value $sourceRecords -Path (Join-Path $IncidentDirectory 'monitor-source-files.json')

    $existingSources = @($sourceRecords | ForEach-Object {
        $sourcePath = [string]$_.Path
        if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
            Assert-NoReparsePointInExistingPath -Path $sourcePath
            Get-Item -LiteralPath $sourcePath
        }
    })
    if ($existingSources.Count -eq 0) {
        Write-Warning 'The paused monitor file list was preserved, but none of those files is currently accessible.'
        return
    }

    $totalBytes = [long](($existingSources | Measure-Object -Property Length -Sum).Sum)
    $availableBytes = $null
    try {
        $directoryItem = Get-Item -LiteralPath $IncidentDirectory
        $drive = New-Object IO.DriveInfo($directoryItem.PSDrive.Root)
        $availableBytes = [long]$drive.AvailableFreeSpace
    }
    catch {
        $availableBytes = $null
    }

    if ($null -ne $availableBytes -and $availableBytes -lt ($totalBytes + 2GB)) {
        $message = "The sealed monitor files require $totalBytes bytes, but the destination does not have the required free-space margin. Their original paths are in monitor-source-files.json."
        Write-Utf8Text -Path (Join-Path $IncidentDirectory 'monitor-copy-skipped.txt') -Text $message
        Write-Warning $message
        return
    }

    $destinationDirectory = Join-Path $IncidentDirectory 'continuous-monitor'
    New-Directory -Path $destinationDirectory
    foreach ($record in $existingSources) {
        $source = [string]$record.FullName
        $destination = Join-Path $destinationDirectory ([IO.Path]::GetFileName($source))
        if (Test-Path -LiteralPath $destination) {
            $destination = Join-Path $destinationDirectory (([IO.Path]::GetFileNameWithoutExtension($source)) + '-' + (Get-Date -Format 'HHmmssfff') + ([IO.Path]::GetExtension($source)))
        }
        Copy-Item -LiteralPath $source -Destination $destination
    }
}

function Repair-PausedMonitorEvidence {
    param(
        [Parameter(Mandatory = $true)] $State,
        [Parameter(Mandatory = $true)][string] $IncidentDirectory
    )

    if (-not [bool]$State.MonitorPausedForIncident -or @($State.MonitorFilesAtPause).Count -gt 0) {
        return
    }
    $monitorDirectory = Join-Path ([string]$State.RunDirectory) 'monitor'
    if (-not (Test-Path -LiteralPath $monitorDirectory -PathType Container)) {
        return
    }

    $status = Get-LogmanStatus -CollectorName $State.CollectorName
    if ($status.Ambiguous) {
        Write-NativeResult -Result $status.Result -Path (Join-Path $IncidentDirectory 'logman-repair-query-ambiguous.txt')
        return
    }
    $files = @(Get-ChildItem -LiteralPath $monitorDirectory -Filter '*.blg' -File | Sort-Object LastWriteTimeUtc -Descending)
    $candidate = $null
    if ($status.Running -and $files.Count -ge 2) {
        $candidate = $files[1]
    }
    elseif (-not $status.Running -and $files.Count -ge 1) {
        $candidate = $files[0]
    }
    if ($null -eq $candidate) {
        return
    }

    $record = [pscustomobject]@{
        Path = $candidate.FullName
        Length = $candidate.Length
        LastWriteTimeUtc = $candidate.LastWriteTimeUtc.ToString('o')
        RecoveredAfterInterruptedPause = $true
    }
    Set-StateProperty -State $State -Name 'MonitorFilesAtPause' -Value @($record)
    Save-State -State $State
    Write-JsonFile -Value $record -Path (Join-Path $IncidentDirectory 'monitor-pause-evidence-recovered.json')
}

function Resume-OwnedMonitor {
    param(
        [Parameter(Mandatory = $true)] $State,
        [Parameter(Mandatory = $true)][string] $IncidentDirectory
    )

    if (-not [bool]$State.MonitorPausedForIncident) {
        return
    }

    $status = Get-LogmanStatus -CollectorName $State.CollectorName
    if ($status.Ambiguous) {
        Write-NativeResult -Result $status.Result -Path (Join-Path $IncidentDirectory 'logman-resume-query-ambiguous.txt')
        Write-Utf8Text -Path (Join-Path $IncidentDirectory 'logman-resume-error.txt') -Text 'Collector status was ambiguous. DwmWatch preserved the resume-needed state and did not issue a start command.'
        Set-StateProperty -State $State -Name 'MonitorStatus' -Value 'ResumeStatusAmbiguous'
        Save-State -State $State
        return
    }
    if ($status.NotFound) {
        Write-Utf8Text -Path (Join-Path $IncidentDirectory 'logman-resume-error.txt') -Text 'The DwmWatch collector definition no longer exists. Evidence was preserved, but monitoring was not resumed.'
        Set-StateProperty -State $State -Name 'MonitorStatus' -Value 'DefinitionMissing'
        Set-StateProperty -State $State -Name 'MonitorPausedForIncident' -Value $false
        Save-State -State $State
        return
    }
    if ($status.Running) {
        Set-StateProperty -State $State -Name 'MonitorStatus' -Value 'Running'
        Set-StateProperty -State $State -Name 'MonitorPausedForIncident' -Value $false
        Save-State -State $State
        return
    }

    $startResult = Invoke-NativeCommandWithTimeout -FilePath $script:LogmanExe -Arguments @('start', [string]$State.CollectorName) -TimeoutSeconds 30
    Write-NativeResult -Result $startResult -Path (Join-Path $IncidentDirectory 'logman-resume.txt')
    if ($startResult.ExitCode -eq 0) {
        Set-StateProperty -State $State -Name 'MonitorStatus' -Value 'Running'
        Set-StateProperty -State $State -Name 'MonitorPausedForIncident' -Value $false
    }
    else {
        Set-StateProperty -State $State -Name 'MonitorStatus' -Value 'ResumeFailed'
    }
    Save-State -State $State
}

function Stop-OwnedWpr {
    param(
        [Parameter(Mandatory = $true)] $State,
        [Parameter(Mandatory = $true)][string] $IncidentDirectory,
        [Parameter(Mandatory = $true)][string] $OutputFileName,
        [switch] $Emergency
    )

    if (-not [bool]$State.WprOwned -or [string]::IsNullOrWhiteSpace([string]$State.WprInstanceName)) {
        return $true
    }

    $instanceName = [string]$State.WprInstanceName
    $attemptId = ConvertTo-SafeFileName ([IO.Path]::GetFileNameWithoutExtension($OutputFileName))
    if (Test-Path -LiteralPath (Join-Path $IncidentDirectory ("wpr-status-before-stop-$attemptId.txt"))) {
        $attemptId = $attemptId + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
    }
    $status = Get-WprStatus -InstanceName $instanceName
    Write-NativeResult -Result $status.Result -Path (Join-Path $IncidentDirectory ("wpr-status-before-stop-$attemptId.txt"))
    if ($status.NotRecording) {
        Write-Utf8Text -Path (Join-Path $IncidentDirectory ("wpr-not-recording-$attemptId.txt")) -Text 'The owned WPR instance is no longer recording. Raw collector ETLs, if any survived a crash, remain in wpr-raw.'
        Set-StateProperty -State $State -Name 'WprOwned' -Value $false
        Set-StateProperty -State $State -Name 'WprLifecycle' -Value 'NotRecording'
        Save-State -State $State
        return $true
    }
    if (-not $status.Recording) {
        Write-Warning 'The owned WPR instance returned an ambiguous status. DwmWatch will still attempt to stop only that exact owned instance.'
    }

    if (-not $Emergency) {
        $captureState = Invoke-NativeCommandWithTimeout -FilePath $script:WprExe -Arguments @(
            '-capturestateondemand'
            '-instancename'
            $instanceName
        ) -TimeoutSeconds 30
        Write-NativeResult -Result $captureState -Path (Join-Path $IncidentDirectory ("wpr-capture-state-after-$attemptId.txt"))
    }

    $outputPath = Join-Path $IncidentDirectory $OutputFileName
    if (Test-Path -LiteralPath $outputPath) {
        $outputPath = Join-Path $IncidentDirectory (([IO.Path]::GetFileNameWithoutExtension($OutputFileName)) + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + ([IO.Path]::GetExtension($OutputFileName)))
    }
    $stopResult = Invoke-NativeCommandWithTimeout -FilePath $script:WprExe -Arguments @(
        '-stop'
        $outputPath
        'DWM lag before and after a manual DWM restart'
        '-skipPdbGen'
        '-instancename'
        $instanceName
    ) -TimeoutSeconds 300
    Write-NativeResult -Result $stopResult -Path (Join-Path $IncidentDirectory ("wpr-stop-$attemptId.txt"))

    $mergedFile = Get-Item -LiteralPath $outputPath -ErrorAction SilentlyContinue
    $mergedTraceValid = $null -ne $mergedFile -and $mergedFile.Length -gt 0
    Set-StateProperty -State $State -Name 'LastMergedEtlPath' -Value $outputPath
    Set-StateProperty -State $State -Name 'LastMergedEtlBytes' -Value $(if ($null -ne $mergedFile) { [long]$mergedFile.Length } else { 0L })
    Set-StateProperty -State $State -Name 'MergedTraceValid' -Value $mergedTraceValid

    $statusAfter = Get-WprStatus -InstanceName $instanceName
    Write-NativeResult -Result $statusAfter.Result -Path (Join-Path $IncidentDirectory ("wpr-status-after-stop-$attemptId.txt"))
    $definitelyStopped = $statusAfter.NotRecording
    if ($stopResult.ExitCode -eq 0 -and $mergedTraceValid) {
        Set-StateProperty -State $State -Name 'WprOwned' -Value $false
        Set-StateProperty -State $State -Name 'WprLifecycle' -Value 'Stopped'
        Save-State -State $State
        return $true
    }
    if ($definitelyStopped) {
        Set-StateProperty -State $State -Name 'WprOwned' -Value $false
        Set-StateProperty -State $State -Name 'WprLifecycle' -Value $(if ($mergedTraceValid) { 'StopReportedFailureButTraceExists' } else { 'StoppedWithoutMergedOutput' })
        Save-State -State $State
        if (-not $mergedTraceValid) {
            Write-Warning 'The owned WPR instance stopped, but the requested merged ETL is missing or empty. Raw fragments and command logs were preserved.'
        }
        return $true
    }

    Write-Warning 'The owned WPR instance could not be stopped. Its raw files and command output were preserved; Status will continue to report it.'
    Set-StateProperty -State $State -Name 'WprLifecycle' -Value 'StopFailedOrAmbiguous'
    Save-State -State $State
    return $false
}

function Stop-OwnedWprAtSafetyLimit {
    param(
        [Parameter(Mandatory = $true)] $State,
        [Parameter(Mandatory = $true)][string] $IncidentDirectory
    )

    if (-not [bool]$State.WprOwned) {
        return $false
    }

    $safety = Get-IncidentTraceSafety -IncidentDirectory $IncidentDirectory
    Write-JsonFile -Value $safety -Path (Join-Path $IncidentDirectory 'wpr-safety-latest.json')
    if ($safety.Safe) {
        return $false
    }

    Set-StateProperty -State $State -Name 'TraceSafetyLimitReached' -Value $true
    Set-StateProperty -State $State -Name 'TraceSafetyReason' -Value ([string]$safety.Reason)
    Set-StateProperty -State $State -Name 'TraceSafetyStopAttempted' -Value $true
    Set-StateProperty -State $State -Name 'IncidentPhase' -Value 'TraceSafetyLimitReached'
    Save-State -State $State
    Write-JsonFile -Value $safety -Path (Join-Path $IncidentDirectory 'wpr-safety-limit.json')
    Add-TimelineEvent -IncidentDirectory $IncidentDirectory -Event 'WPR_SAFETY_LIMIT_REACHED' -Details ([string]$safety.Reason)

    $safetyStopSucceeded = Stop-OwnedWpr -State $State -IncidentDirectory $IncidentDirectory -OutputFileName 'dwm-incident-trace-limit.etl' -Emergency
    Set-StateProperty -State $State -Name 'TraceSafetyStopSucceeded' -Value $safetyStopSucceeded
    Save-State -State $State
    Add-TimelineEvent -IncidentDirectory $IncidentDirectory -Event 'WPR_SAFETY_STOP_ATTEMPTED' -Details ("Succeeded or already absent: $safetyStopSucceeded")
    return $true
}

function End-DwmIncident {
    Assert-Administrator
    $state = Get-State
    if ($null -eq $state -or [string]::IsNullOrWhiteSpace([string]$state.PendingIncidentDirectory)) {
        throw 'No DwmWatch incident is pending. Run BeginIncident while the lag is present.'
    }

    $incidentDirectory = [string]$state.PendingIncidentDirectory
    New-Directory -Path $incidentDirectory
    Set-StateProperty -State $state -Name 'IncidentPhase' -Value 'CapturingAfter'
    Save-State -State $state

    $currentDwm = Get-DwmForSession -SessionId ([int]$state.DwmSessionId)
    $currentDwmPid = $(if ($null -ne $currentDwm) { [int]$currentDwm.Id } else { $null })
    $restartObserved = $null -ne $currentDwmPid -and $currentDwmPid -ne [int]$state.OldDwmPid
    Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'END_INCIDENT_STARTED' -Details ("Target session: $($state.DwmSessionId); current DWM PID: $currentDwmPid; PID changed: $restartObserved")

    if ([bool]$state.WprOwned -and -not [string]::IsNullOrWhiteSpace([string]$state.WprInstanceName)) {
        $ownedStatus = Get-WprStatus -InstanceName ([string]$state.WprInstanceName)
        if ($ownedStatus.Recording) {
            $markerName = $(if ($restartObserved) { 'DWM_PID_CHANGED' } else { 'DWM_END_WITHOUT_RESTART' })
            $marker = Invoke-NativeCommandWithTimeout -FilePath $script:WprExe -Arguments @(
                '-marker'
                $markerName
                '-instancename'
                ([string]$state.WprInstanceName)
            ) -TimeoutSeconds 15
            Write-NativeResult -Result $marker -Path (Join-Path $incidentDirectory 'wpr-marker-after.txt')
            $flush = Flush-WprInstance -InstanceName ([string]$state.WprInstanceName)
            Write-NativeResult -Result $flush -Path (Join-Path $incidentDirectory 'wpr-flush-after.txt')
        }
    }

    $traceSafetyLimitReached = $state.PSObject.Properties['TraceSafetyLimitReached'] -and [bool]$state.TraceSafetyLimitReached
    $wprOutputFileName = $(if ($traceSafetyLimitReached) { 'dwm-incident-trace-limit-retry.etl' } else { 'dwm-incident.etl' })
    $wprStopped = Stop-OwnedWpr -State $state -IncidentDirectory $incidentDirectory -OutputFileName $wprOutputFileName
    if (-not $wprStopped) {
        Start-Sleep -Seconds 1
        $retryFileName = $(if ($traceSafetyLimitReached) { 'dwm-incident-trace-limit-retry2.etl' } else { 'dwm-incident-retry.etl' })
        $wprStopped = Stop-OwnedWpr -State $state -IncidentDirectory $incidentDirectory -OutputFileName $retryFileName
    }
    Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'WPR_STOP_ATTEMPTED' -Details ("Stopped successfully or already absent: $wprStopped; safety limit reached: $traceSafetyLimitReached")

    $bootTimeAfterUtc = Get-LastBootTimeUtc
    $systemRestartObserved = $false
    if (-not [string]::IsNullOrWhiteSpace([string]$state.BootTimeAtIncidentUtc) -and -not [string]::IsNullOrWhiteSpace([string]$bootTimeAfterUtc)) {
        try {
            $beforeBoot = [DateTime]::Parse([string]$state.BootTimeAtIncidentUtc).ToUniversalTime()
            $afterBoot = [DateTime]::Parse([string]$bootTimeAfterUtc).ToUniversalTime()
            $systemRestartObserved = [Math]::Abs(($afterBoot - $beforeBoot).TotalSeconds) -gt 5
        }
        catch {
            $systemRestartObserved = $false
        }
    }
    $postSnapshotSessionId = [int]$state.DwmSessionId
    if ($systemRestartObserved) {
        $postSnapshotSessionId = Get-CurrentSessionId
        $currentDwm = Get-DwmForSession -SessionId $postSnapshotSessionId
        $currentDwmPid = $(if ($null -ne $currentDwm) { [int]$currentDwm.Id } else { $null })
        $restartObserved = $null -ne $currentDwmPid -and $currentDwmPid -ne [int]$state.OldDwmPid
    }
    $dwmRestartObserved = $restartObserved -and -not $systemRestartObserved
    Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'RESTART_CLASSIFIED' -Details ("DWM-only restart: $dwmRestartObserved; system restart: $systemRestartObserved; post-snapshot session: $postSnapshotSessionId")

    $immediateLabel = $(if ($dwmRestartObserved) { 'Immediately after DWM restart' } elseif ($systemRestartObserved) { 'After a system restart' } else { 'End capture without an observed DWM restart' })
    $settledLabel = $(if ($dwmRestartObserved) { "$PostRestartSeconds seconds after DWM restart" } elseif ($systemRestartObserved) { "$PostRestartSeconds seconds after a system restart" } else { "$PostRestartSeconds seconds after end capture without an observed DWM restart" })
    [void](Invoke-CaptureStep -Name 'after-immediate-snapshot' -ErrorDirectory $incidentDirectory -Operation {
        Save-SnapshotOnce -Directory (Join-Path $incidentDirectory 'after-immediate') -Label $immediateLabel -TargetSessionId $postSnapshotSessionId
    })

    Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'POST_RESTART_SETTLE_BEGIN' -Details ("Waiting $PostRestartSeconds seconds")
    Write-Output "Captured the immediate post-restart state. Waiting $PostRestartSeconds seconds for a settled sample..."
    Start-Sleep -Seconds $PostRestartSeconds

    [void](Invoke-CaptureStep -Name 'after-settled-snapshot' -ErrorDirectory $incidentDirectory -Operation {
        Save-SnapshotOnce -Directory (Join-Path $incidentDirectory 'after-settled') -Label $settledLabel -TargetSessionId $postSnapshotSessionId
    })
    Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'POST_RESTART_SETTLE_END'

    [void](Invoke-CaptureStep -Name 'system-inventory' -ErrorDirectory $incidentDirectory -Operation {
        Save-SystemInventory -Directory (Join-Path $incidentDirectory 'system')
    })
    [void](Invoke-CaptureStep -Name 'repair-continuous-monitor-reference' -ErrorDirectory $incidentDirectory -Operation {
        Repair-PausedMonitorEvidence -State $state -IncidentDirectory $incidentDirectory
    })
    [void](Invoke-CaptureStep -Name 'copy-continuous-monitor' -ErrorDirectory $incidentDirectory -Operation {
        Copy-PausedMonitorFiles -State $state -IncidentDirectory $incidentDirectory
    })
    [void](Invoke-CaptureStep -Name 'resume-continuous-monitor' -ErrorDirectory $incidentDirectory -Operation {
        Resume-OwnedMonitor -State $state -IncidentDirectory $incidentDirectory
    })
    [void](Invoke-CaptureStep -Name 'event-log-export' -ErrorDirectory $incidentDirectory -Operation {
        Export-RecentEventLogs -Directory (Join-Path $incidentDirectory 'events') -LookbackHours $EventLookbackHours
    })
    [void](Invoke-CaptureStep -Name 'comparison' -ErrorDirectory $incidentDirectory -Operation {
        Write-Comparison -IncidentDirectory $incidentDirectory
    })

    $includeCommandLinesBefore = $state.PSObject.Properties['IncidentIncludeCommandLinesBefore'] -and [bool]$state.IncidentIncludeCommandLinesBefore
    $traceSafetyReason = $(if ($state.PSObject.Properties['TraceSafetyReason']) { $state.TraceSafetyReason } else { $null })
    $traceSafetyStopAttempted = $state.PSObject.Properties['TraceSafetyStopAttempted'] -and [bool]$state.TraceSafetyStopAttempted
    $traceSafetyStopSucceeded = $(if ($state.PSObject.Properties['TraceSafetyStopSucceeded']) { $state.TraceSafetyStopSucceeded } else { $null })
    $manifest = [pscustomobject][ordered]@{
        ScriptVersion = $script:ScriptVersion
        Workflow = $(if ($state.PSObject.Properties['IncidentWorkflow']) { $state.IncidentWorkflow } else { $null })
        IncidentStartedUtc = $state.IncidentStartedUtc
        IncidentCompletedUtc = [DateTime]::UtcNow.ToString('o')
        TargetDwmSessionId = $state.DwmSessionId
        PostSnapshotSessionId = $postSnapshotSessionId
        InitialDwmPid = $(if ($state.PSObject.Properties['InitialDwmPid'] -and $null -ne $state.InitialDwmPid) { $state.InitialDwmPid } else { $state.OldDwmPid })
        OldDwmPid = $state.OldDwmPid
        NewDwmPid = $currentDwmPid
        DwmPidChanged = $restartObserved
        DwmRestartObserved = $dwmRestartObserved
        BootTimeBeforeUtc = $state.BootTimeAtIncidentUtc
        BootTimeAfterUtc = $bootTimeAfterUtc
        SystemRestartObserved = $systemRestartObserved
        WprProfileMode = $state.WprProfileMode
        VerboseTrace = $(if ($state.PSObject.Properties['IncidentVerboseTrace']) { [bool]$state.IncidentVerboseTrace } else { $false })
        FileModeTrace = $(if ($state.PSObject.Properties['IncidentFileModeTrace']) { [bool]$state.IncidentFileModeTrace } else { $false })
        WprLifecycle = $state.WprLifecycle
        WprStoppedOrAlreadyAbsent = $wprStopped
        MergedTracePath = $(if ($state.PSObject.Properties['LastMergedEtlPath']) { $state.LastMergedEtlPath } else { $null })
        MergedTraceBytes = $(if ($state.PSObject.Properties['LastMergedEtlBytes']) { $state.LastMergedEtlBytes } else { $null })
        MergedTraceValid = $(if ($state.PSObject.Properties['MergedTraceValid']) { $state.MergedTraceValid } else { $null })
        TraceSafetyLimitReached = $traceSafetyLimitReached
        TraceSafetyReason = $traceSafetyReason
        TraceSafetyStopAttempted = $traceSafetyStopAttempted
        TraceSafetyStopSucceeded = $traceSafetyStopSucceeded
        MaxIncidentTraceMegabytes = $(if ($state.PSObject.Properties['IncidentMaxTraceMegabytes']) { $state.IncidentMaxTraceMegabytes } else { $null })
        MinimumFreeSpaceGigabytes = $(if ($state.PSObject.Properties['IncidentMinimumFreeSpaceGigabytes']) { $state.IncidentMinimumFreeSpaceGigabytes } else { $null })
        PostTraceSeconds = $(if ($state.PSObject.Properties['IncidentPostTraceSeconds']) { $state.IncidentPostTraceSeconds } else { $null })
        EventLookbackHours = $EventLookbackHours
        PostRestartSeconds = $PostRestartSeconds
        IncludeCommandLinesBefore = $includeCommandLinesBefore
        IncludeCommandLinesAfter = [bool]$IncludeCommandLines
        IncludeCommandLines = ($includeCommandLinesBefore -or [bool]$IncludeCommandLines)
        PrivacyNotice = 'Treat ETL, EVTX, process paths, device IDs, event messages, and possible embedded command text or secrets as sensitive. No process dump or screenshot was captured.'
    }
    Write-JsonFile -Value $manifest -Path (Join-Path $incidentDirectory 'manifest.json')
    Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'INCIDENT_CAPTURE_COMPLETE'

    if (-not $SkipHashes -and $wprStopped) {
        [void](Invoke-CaptureStep -Name 'sha256' -ErrorDirectory $incidentDirectory -Operation {
            Write-Hashes -Directory $incidentDirectory
        })
    }
    elseif (-not $wprStopped) {
        Write-Utf8Text -Path (Join-Path $incidentDirectory 'hashes-deferred.txt') -Text 'Hashes were not calculated because an owned WPR instance may still be writing raw ETLs.'
    }

    Set-StateProperty -State $state -Name 'LastIncidentDirectory' -Value $incidentDirectory
    Set-StateProperty -State $state -Name 'PendingIncidentDirectory' -Value $null
    Set-StateProperty -State $state -Name 'IncidentPhase' -Value $(if ($wprStopped) { 'Completed' } else { 'CompletedWithWprStillActive' })
    Set-StateProperty -State $state -Name 'MonitorFilesAtPause' -Value @()
    Save-State -State $state

    Write-Output 'DwmWatch incident capture completed.'
    Write-Output "Evidence directory: $incidentDirectory"
    Write-Output "DWM PID changed in session $($state.DwmSessionId): $restartObserved ($($state.OldDwmPid) -> $currentDwmPid)"
    Write-Output "DWM-only restart detected: $dwmRestartObserved"
    Write-Output "System restart detected: $systemRestartObserved"
    if (-not $wprStopped) {
        Write-Warning 'The owned WPR session may still be active. Run Status, then StopMonitor after preserving the incident directory.'
    }
}

function Capture-DwmIncident {
    $incidentPending = $false
    $script:BoundedIncidentWorkflow = $true
    try {
        Begin-DwmIncident
        $state = Get-State
        if ($null -eq $state -or [string]::IsNullOrWhiteSpace([string]$state.PendingIncidentDirectory)) {
            throw 'BeginIncident did not leave a recoverable pending incident.'
        }
        $incidentPending = $true

        $incidentDirectory = [string]$state.PendingIncidentDirectory
        $oldDwmPid = [int]$state.OldDwmPid
        $deadline = (Get-Date).AddMinutes($IncidentWaitMinutes)
        $nextSafetyCheck = Get-Date
        Write-Output "Waiting up to $IncidentWaitMinutes minutes for DWM PID $oldDwmPid in session $($state.DwmSessionId) to be replaced..."
        $restartDetected = $false
        $safetyLimitReached = $state.PSObject.Properties['TraceSafetyLimitReached'] -and [bool]$state.TraceSafetyLimitReached
        $currentDwm = $null
        $fileModeIncident = $state.PSObject.Properties['IncidentFileModeTrace'] -and [bool]$state.IncidentFileModeTrace
        while (-not $safetyLimitReached -and (Get-Date) -lt $deadline) {
            $currentDwm = Get-DwmForSession -SessionId ([int]$state.DwmSessionId)
            if ($null -ne $currentDwm -and $currentDwm.Id -ne $oldDwmPid) {
                $restartDetected = $true
                break
            }

            if ([bool]$state.WprOwned -and $fileModeIncident -and (Get-Date) -ge $nextSafetyCheck) {
                $nextSafetyCheck = (Get-Date).AddSeconds(5)
                if (Stop-OwnedWprAtSafetyLimit -State $state -IncidentDirectory $incidentDirectory) {
                    $safetyLimitReached = $true
                    break
                }
            }
            Start-Sleep -Seconds 1
        }

        if ($restartDetected) {
            Write-Output "Detected the new DWM PID: $($currentDwm.Id). Finishing the capture."
            if ([bool]$state.WprOwned -and $PostTraceSeconds -gt 0) {
                Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'POST_RESTART_TRACE_WINDOW_BEGIN' -Details ("Waiting up to $PostTraceSeconds seconds")
                Write-Output "Keeping the trace open for $PostTraceSeconds seconds of post-restart recovery..."
                for ($second = 0; $second -lt $PostTraceSeconds; $second++) {
                    if ($fileModeIncident -and (Stop-OwnedWprAtSafetyLimit -State $state -IncidentDirectory $incidentDirectory)) {
                        $safetyLimitReached = $true
                        break
                    }
                    Start-Sleep -Seconds 1
                }
                Add-TimelineEvent -IncidentDirectory $incidentDirectory -Event 'POST_RESTART_TRACE_WINDOW_END' -Details ("Safety limit reached: $safetyLimitReached")
            }
        }
        elseif ($safetyLimitReached) {
            Write-Warning 'The WPR trace reached its disk-safety limit before a DWM replacement was detected. The trace was closed and the remaining snapshots will still be collected.'
        }
        else {
            Write-Warning "No replacement DWM PID was detected within $IncidentWaitMinutes minutes. The trace will be closed anyway to protect disk space."
        }
        End-DwmIncident
        $incidentPending = $false
    }
    finally {
        $script:BoundedIncidentWorkflow = $false
        try {
            $emergencyState = Get-State
            $isThisBoundedWorkflow = $null -ne $emergencyState -and
                $emergencyState.PSObject.Properties['IncidentWorkflow'] -and
                [string]$emergencyState.IncidentWorkflow -eq 'BoundedCaptureIncident'
            if (($incidentPending -or $isThisBoundedWorkflow) -and
                $null -ne $emergencyState -and
                -not [string]::IsNullOrWhiteSpace([string]$emergencyState.PendingIncidentDirectory) -and
                [bool]$emergencyState.WprOwned) {
                    $emergencyDirectory = [string]$emergencyState.PendingIncidentDirectory
                    [void](Stop-OwnedWpr -State $emergencyState -IncidentDirectory $emergencyDirectory -OutputFileName 'dwm-incident-emergency-close.etl')
                    Add-TimelineEvent -IncidentDirectory $emergencyDirectory -Event 'CAPTUREINCIDENT_EMERGENCY_WPR_CLOSE'
                    Write-Warning 'CaptureIncident ended unexpectedly. Its owned WPR trace was closed; run EndIncident to finish the recoverable bundle.'
            }
        }
        catch {
            Write-Warning "CaptureIncident ended unexpectedly and could not confirm that its owned WPR trace stopped: $($_.Exception.Message)"
        }
    }
}

function Show-DwmWatchStatus {
    $state = Get-State
    $dwm = Get-CurrentSessionDwm
    $dwmText = $(if ($null -ne $dwm) {
        "PID $($dwm.Id), session $($dwm.SessionId), private $([Math]::Round($dwm.PrivateMemorySize64 / 1MB, 1)) MiB, working set $([Math]::Round($dwm.WorkingSet64 / 1MB, 1)) MiB, handles $($dwm.HandleCount), threads $($dwm.Threads.Count)"
    } else {
        'not found in the current session'
    })

    Write-Output "DWM: $dwmText"
    Write-Output "Output root: $OutputRoot"
    if ($null -eq $state) {
        Write-Output 'DwmWatch state: not configured'
        $wprStatus = Get-WprStatus
        Write-Output "Global WPR status: $($wprStatus.Result.Output)"
        return
    }

    $collectorStatus = Get-LogmanStatus -CollectorName $state.CollectorName
    Write-Output "Run directory: $($state.RunDirectory)"
    Write-Output "Collector: $($state.CollectorName)"
    Write-Output "Collector state: $($collectorStatus.State)"
    Write-Output "Recorded monitor state: $($state.MonitorStatus)"
    Write-Output "Incident phase: $($state.IncidentPhase)"
    Write-Output "Pending incident: $($state.PendingIncidentDirectory)"
    if ($state.PSObject.Properties['LastIncidentDirectory']) {
        Write-Output "Last incident: $($state.LastIncidentDirectory)"
    }
    Write-Output "Script owns WPR: $($state.WprOwned)"
    Write-Output "WPR instance: $($state.WprInstanceName)"

    $wprStatus = Get-WprStatus -InstanceName $(if ([bool]$state.WprOwned) { [string]$state.WprInstanceName } else { $null })
    Write-Output "WPR status: $($wprStatus.Result.Output)"

    if (Test-Path -LiteralPath ([string]$state.RunDirectory) -PathType Container) {
        $evidenceFiles = @(Get-ChildItem -LiteralPath ([string]$state.RunDirectory) -Recurse -File | Where-Object {
            $_.Extension -in @('.blg', '.etl', '.evtx')
        })
        foreach ($file in $evidenceFiles | Sort-Object FullName) {
            Write-Output ("Evidence: {0:N1} MiB  {1}" -f ($file.Length / 1MB), $file.FullName)
        }
        try {
            $runItem = Get-Item -LiteralPath ([string]$state.RunDirectory)
            $drive = New-Object IO.DriveInfo($runItem.PSDrive.Root)
            Write-Output ("Free space: {0:N1} GiB on {1}" -f ($drive.AvailableFreeSpace / 1GB), $runItem.PSDrive.Root)
        }
        catch {
            Write-Warning 'Free disk space could not be queried.'
        }
    }
}

function Stop-DwmMonitor {
    Assert-Administrator
    $state = Get-State
    if ($null -eq $state) {
        Write-Output 'DwmWatch has no saved state; nothing was stopped.'
        return
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$state.PendingIncidentDirectory)) {
        throw 'An incident is pending. Run EndIncident first so its WPR trace and before/after evidence are preserved.'
    }

    $finalDirectory = Join-Path ([string]$state.RunDirectory) ('stopped-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    New-Directory -Path $finalDirectory

    $wprStopped = $true
    if ([bool]$state.WprOwned) {
        $wprStopped = Stop-OwnedWpr -State $state -IncidentDirectory $finalDirectory -OutputFileName 'emergency-owned-wpr.etl' -Emergency
    }
    [void](Invoke-CaptureStep -Name 'final-snapshot' -ErrorDirectory $finalDirectory -Operation {
        Save-Snapshot -Directory (Join-Path $finalDirectory 'snapshot') -Label 'Final monitor snapshot'
    })

    $collectorStatus = Get-LogmanStatus -CollectorName $state.CollectorName
    $collectorStopped = $false
    if ($collectorStatus.Ambiguous) {
        Write-NativeResult -Result $collectorStatus.Result -Path (Join-Path $finalDirectory 'logman-query-ambiguous.txt')
        Set-StateProperty -State $state -Name 'MonitorStatus' -Value 'StopStatusAmbiguous'
        Write-Warning 'The owned collector status is ambiguous. It was not reported stopped; WPR and the remaining evidence will still be handled.'
    }
    elseif ($collectorStatus.Running) {
        $stopResult = Invoke-NativeCommandWithTimeout -FilePath $script:LogmanExe -Arguments @('stop', [string]$state.CollectorName) -TimeoutSeconds 30
        Write-NativeResult -Result $stopResult -Path (Join-Path $finalDirectory 'logman-stop.txt')
        if ($stopResult.ExitCode -eq 0) {
            Set-StateProperty -State $state -Name 'MonitorStatus' -Value 'Stopped'
            $collectorStopped = $true
        }
        else {
            $afterStopStatus = Get-LogmanStatus -CollectorName $state.CollectorName
            Write-NativeResult -Result $afterStopStatus.Result -Path (Join-Path $finalDirectory 'logman-status-after-stop-failure.txt')
            $collectorStopped = $afterStopStatus.KnownStopped
            Set-StateProperty -State $state -Name 'MonitorStatus' -Value $(if ($collectorStopped) { 'StoppedAfterReportedFailure' } else { 'StopFailedOrAmbiguous' })
        }
    }
    else {
        Set-StateProperty -State $state -Name 'MonitorStatus' -Value 'Stopped'
        $collectorStopped = $collectorStatus.KnownStopped
    }

    [void](Invoke-CaptureStep -Name 'event-log-export' -ErrorDirectory $finalDirectory -Operation {
        Export-RecentEventLogs -Directory (Join-Path $finalDirectory 'events') -LookbackHours $EventLookbackHours
    })
    if (-not $SkipHashes -and $collectorStopped -and $wprStopped) {
        [void](Invoke-CaptureStep -Name 'sha256' -ErrorDirectory $finalDirectory -Operation {
            Write-Hashes -Directory $finalDirectory
        })
    }
    Save-State -State $state

    if ($collectorStopped -and $wprStopped) {
        Write-Output 'DwmWatch monitor stopped. Its collector definition and every artifact were left intact.'
    }
    else {
        Write-Warning "DwmWatch could not confirm every owned session stopped (collector: $collectorStopped; WPR: $wprStopped). Run Status and retry StopMonitor; evidence was left intact."
    }
    Write-Output "Final capture: $finalDirectory"
}

$mutex = $null
$mutexAcquired = $false
try {
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        $rootHashBytes = $sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($OutputRoot.ToUpperInvariant()))
        $rootHash = ([BitConverter]::ToString($rootHashBytes, 0, 8)).Replace('-', '')
    }
    finally {
        $sha256.Dispose()
    }
    $mutexName = 'Global\DwmWatch-PowerShell-Capture-' + $rootHash
    $mutex = New-Object Threading.Mutex($false, $mutexName)
    try {
        $mutexAcquired = $mutex.WaitOne([TimeSpan]::FromSeconds(5))
    }
    catch [Threading.AbandonedMutexException] {
        $mutexAcquired = $true
    }
    if (-not $mutexAcquired) {
        throw 'Another DwmWatch command is already running in this Windows session.'
    }

    switch ($Action) {
        'StartMonitor' {
            Start-DwmMonitor
        }
        'CaptureIncident' {
            Capture-DwmIncident
        }
        'BeginIncident' {
            Begin-DwmIncident
        }
        'EndIncident' {
            End-DwmIncident
        }
        'Status' {
            Show-DwmWatchStatus
        }
        'StopMonitor' {
            Stop-DwmMonitor
        }
    }
}
finally {
    if ($mutexAcquired -and $null -ne $mutex) {
        [void]$mutex.ReleaseMutex()
    }
    if ($null -ne $mutex) {
        $mutex.Dispose()
    }
}
