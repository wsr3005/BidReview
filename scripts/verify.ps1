param(
    [switch]$ContinueOnError,
    [string[]]$Gates = @("lint", "test", "typecheck", "build")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "== $Title =="
}

function Get-PackageManager {
    if (Test-Path "pnpm-lock.yaml") { return "pnpm" }
    if (Test-Path "yarn.lock") { return "yarn" }
    if (Test-Path "bun.lockb") { return "bun" }
    return "npm"
}

function Find-ScriptName {
    param(
        [string[]]$Candidates,
        [hashtable]$Scripts
    )
    foreach ($candidate in $Candidates) {
        if ($Scripts.ContainsKey($candidate)) {
            return $candidate
        }
    }
    return $null
}

function Invoke-PackageScript {
    param(
        [string]$PackageManager,
        [string]$ScriptName
    )

    switch ($PackageManager) {
        "pnpm" { & pnpm run $ScriptName; return $LASTEXITCODE }
        "yarn" { & yarn run $ScriptName; return $LASTEXITCODE }
        "bun" { & bun run $ScriptName; return $LASTEXITCODE }
        default { & npm run $ScriptName; return $LASTEXITCODE }
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Push-Location $repoRoot

try {
    $packageJsonPath = Join-Path $repoRoot "package.json"
    if (-not (Test-Path $packageJsonPath)) {
        Write-Warning "No package.json found at $packageJsonPath"
        Write-Host "Nothing to verify. Exiting with success."
        exit 0
    }

    try {
        $packageJson = Get-Content -Raw -Encoding utf8 $packageJsonPath | ConvertFrom-Json
    }
    catch {
        Write-Error "Failed to parse package.json: $($_.Exception.Message)"
        exit 2
    }

    $scripts = @{}
    if ($packageJson.scripts) {
        foreach ($property in $packageJson.scripts.PSObject.Properties) {
            $scripts[$property.Name] = $property.Value
        }
    }

    if ($scripts.Count -eq 0) {
        Write-Warning "No npm scripts were found in package.json."
        Write-Host "Nothing to verify. Exiting with success."
        exit 0
    }

    $gateCandidates = @{
        lint      = @("lint", "check", "eslint")
        test      = @("test", "test:unit", "unit")
        typecheck = @("typecheck", "type-check", "check-types")
        build     = @("build")
    }

    $packageManager = Get-PackageManager
    Write-Host "Package manager: $packageManager"

    $results = New-Object System.Collections.Generic.List[object]
    $hasFailure = $false

    foreach ($gate in $Gates) {
        if (-not $gateCandidates.ContainsKey($gate)) {
            $results.Add([pscustomobject]@{
                    Gate     = $gate
                    Script   = "-"
                    Status   = "skipped"
                    ExitCode = 0
                })
            continue
        }

        $scriptName = Find-ScriptName -Candidates $gateCandidates[$gate] -Scripts $scripts
        if ($null -eq $scriptName) {
            $results.Add([pscustomobject]@{
                    Gate     = $gate
                    Script   = "-"
                    Status   = "skipped"
                    ExitCode = 0
                })
            continue
        }

        Write-Section "$gate -> $scriptName"
        $exitCode = 0
        try {
            $exitCode = Invoke-PackageScript -PackageManager $packageManager -ScriptName $scriptName
            if ($null -eq $exitCode) { $exitCode = 1 }
        }
        catch {
            Write-Host "Command failed: $($_.Exception.Message)"
            $exitCode = 127
        }

        if ($exitCode -eq 0) {
            $status = "pass"
        }
        else {
            $status = "fail"
            $hasFailure = $true
        }

        $results.Add([pscustomobject]@{
                Gate     = $gate
                Script   = $scriptName
                Status   = $status
                ExitCode = $exitCode
            })

        if ($exitCode -ne 0 -and -not $ContinueOnError) {
            break
        }
    }

    Write-Section "Summary"
    $results | Format-Table Gate, Script, Status, ExitCode -AutoSize | Out-String | Write-Host

    if ($hasFailure) { exit 1 }
    exit 0
}
finally {
    Pop-Location
}

