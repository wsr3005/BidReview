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

function Command-Exists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-PythonChecks {
    param(
        [string[]]$Gates,
        [switch]$ContinueOnError
    )

    if (-not (Command-Exists -Name "python")) {
        Write-Error "Python project detected but 'python' command is unavailable."
        exit 2
    }

    $results = New-Object System.Collections.Generic.List[object]
    $hasFailure = $false

    foreach ($gate in $Gates) {
        $status = "skipped"
        $exitCode = 0
        $scriptName = "-"

        switch ($gate) {
            "lint" {
                if (Command-Exists -Name "ruff") {
                    $scriptName = "ruff check ."
                    Write-Section "$gate -> $scriptName"
                    & ruff check .
                    $exitCode = $LASTEXITCODE
                    $status = if ($exitCode -eq 0) { "pass" } else { "fail" }
                }
            }
            "test" {
                if (Test-Path "tests") {
                    $scriptName = "python -m unittest discover -s tests -v"
                    Write-Section "$gate -> $scriptName"
                    & python -m unittest discover -s tests -v
                    $exitCode = $LASTEXITCODE
                    $status = if ($exitCode -eq 0) { "pass" } else { "fail" }
                }
            }
            "typecheck" {
                if (Command-Exists -Name "mypy") {
                    $scriptName = "mypy ."
                    Write-Section "$gate -> $scriptName"
                    & mypy .
                    $exitCode = $LASTEXITCODE
                    $status = if ($exitCode -eq 0) { "pass" } else { "fail" }
                }
            }
            "build" {
                if ((Test-Path "bidagent") -or (Test-Path "tests")) {
                    $scriptName = "python -m compileall bidagent tests"
                    Write-Section "$gate -> $scriptName"
                    & python -m compileall bidagent tests
                    $exitCode = $LASTEXITCODE
                    $status = if ($exitCode -eq 0) { "pass" } else { "fail" }
                }
            }
        }

        $results.Add([pscustomobject]@{
                Gate     = $gate
                Script   = $scriptName
                Status   = $status
                ExitCode = $exitCode
            })

        if ($status -eq "fail") {
            $hasFailure = $true
            if (-not $ContinueOnError) { break }
        }
    }

    Write-Section "Summary"
    $results | Format-Table Gate, Script, Status, ExitCode -AutoSize | Out-String | Write-Host

    if ($hasFailure) { exit 1 }
    exit 0
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Push-Location $repoRoot

try {
    $packageJsonPath = Join-Path $repoRoot "package.json"
    if (-not (Test-Path $packageJsonPath)) {
        $pythonProjectDetected = (Test-Path "pyproject.toml") -or (Test-Path "setup.py") -or (Test-Path "bidagent")
        if ($pythonProjectDetected) {
            Write-Host "Python project detected. Running Python verification gates."
            Invoke-PythonChecks -Gates $Gates -ContinueOnError:$ContinueOnError
        }
        Write-Warning "No package.json or Python project metadata found."
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
