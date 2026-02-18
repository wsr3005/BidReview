param(
    [ValidateSet("t0", "t1", "t2", "rc")]
    [string]$Level = "t1",
    [ValidateSet("auto", "on", "off")]
    [string]$Parallel = "auto",
    [string[]]$Areas = @(),
    [string[]]$Tests = @(),
    [string]$Tender,
    [string]$Bid,
    [string]$OutDir,
    [string]$AiProvider = "deepseek",
    [string]$AiModel = "deepseek-chat",
    [ValidateSet("off", "auto", "tesseract")]
    [string]$OcrMode = "auto",
    [ValidateSet("off", "critical", "all")]
    [string]$GateFailFast = "critical",
    [int]$CanaryMinStreak = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "== $Title =="
}

function Command-Exists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-PythonRunner {
    if (Command-Exists -Name "uv") { return "uv" }
    if (Command-Exists -Name "python") { return "python" }
    return $null
}

function Resolve-ParallelEnabled {
    param(
        [string]$Mode,
        [string]$CurrentLevel
    )
    if ($Mode -eq "on") { return $true }
    if ($Mode -eq "off") { return $false }
    return $CurrentLevel -in @("t2", "rc")
}

function Invoke-PythonRunner {
    param(
        [string]$Runner,
        [string[]]$PyArgs,
        [switch]$AllowNoAi
    )

    $hadEnv = Test-Path Env:BIDAGENT_ALLOW_NO_AI
    $previousValue = $env:BIDAGENT_ALLOW_NO_AI
    if ($AllowNoAi) {
        $env:BIDAGENT_ALLOW_NO_AI = "1"
    }

    try {
        if ($Runner -eq "uv") {
            & uv run python @PyArgs | Out-Host
            return $LASTEXITCODE
        }
        & python @PyArgs | Out-Host
        return $LASTEXITCODE
    }
    finally {
        if ($AllowNoAi) {
            if ($hadEnv) {
                $env:BIDAGENT_ALLOW_NO_AI = $previousValue
            }
            else {
                Remove-Item Env:BIDAGENT_ALLOW_NO_AI -ErrorAction SilentlyContinue
            }
        }
    }
}

function Invoke-BidagentCommand {
    param(
        [string]$Runner,
        [string[]]$CliArgs
    )

    if ($Runner -eq "uv") {
        & uv run bidagent @CliArgs | Out-Host
        return $LASTEXITCODE
    }
    & python -m bidagent @CliArgs | Out-Host
    return $LASTEXITCODE
}

function Get-UniqueOrderedItems {
    param([string[]]$Items)
    $seen = New-Object "System.Collections.Generic.HashSet[string]"
    $ordered = New-Object "System.Collections.Generic.List[string]"
    foreach ($item in $Items) {
        $value = [string]$item
        if ([string]::IsNullOrWhiteSpace($value)) { continue }
        if ($seen.Add($value)) {
            $ordered.Add($value)
        }
    }
    return $ordered.ToArray()
}

function Get-AreaModulesMap {
    return [ordered]@{
        ocr      = @("tests.test_ocr", "tests.test_pipeline_pdf_ocr_smoke", "tests.test_annotate_output")
        rules    = @("tests.test_review", "tests.test_constraints", "tests.test_rule_tier", "tests.test_task_planner")
        llm      = @("tests.test_llm_review", "tests.test_llm_judge", "tests.test_deepseek_reviewer")
        evidence = @("tests.test_evidence_index", "tests.test_evidence_harvester")
        pipeline = @("tests.test_pipeline_review", "tests.test_cli_smoke", "tests.test_eval", "tests.test_goldset_validation")
        output   = @("tests.test_consistency", "tests.test_report", "tests.test_checklist")
    }
}

function Get-ChangedFiles {
    if (-not (Command-Exists -Name "git")) { return @() }
    $files = New-Object "System.Collections.Generic.List[string]"
    try {
        $unstaged = & git diff --name-only -- . | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        $staged = & git diff --name-only --cached -- . | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        $untracked = & git ls-files --others --exclude-standard | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        foreach ($row in @($unstaged + $staged + $untracked)) {
            if (-not [string]::IsNullOrWhiteSpace($row)) {
                $files.Add($row)
            }
        }
    }
    catch {
        return @()
    }
    return Get-UniqueOrderedItems -Items $files.ToArray()
}

function Get-AreasFromChangedFiles {
    param([string[]]$Files)
    $areas = New-Object "System.Collections.Generic.List[string]"
    foreach ($path in $Files) {
        switch -Regex ($path) {
            '^bidagent/(ocr|document|annotators)\.py$' { $areas.Add("ocr"); break }
            '^bidagent/(review|constraints|task_planner)\.py$' { $areas.Add("rules"); break }
            '^bidagent/(llm|llm_judge)\.py$' { $areas.Add("llm"); break }
            '^bidagent/(evidence_index|evidence_harvester)\.py$' { $areas.Add("evidence"); break }
            '^bidagent/(pipeline|cli|eval)\.py$' { $areas.Add("pipeline"); break }
            '^bidagent/consistency\.py$' { $areas.Add("output"); break }
            '^tests/test_(ocr|pipeline_pdf_ocr_smoke|annotate_output)\.py$' { $areas.Add("ocr"); break }
            '^tests/test_(review|constraints|rule_tier|task_planner)\.py$' { $areas.Add("rules"); break }
            '^tests/test_(llm_review|llm_judge|deepseek_reviewer)\.py$' { $areas.Add("llm"); break }
            '^tests/test_(evidence_index|evidence_harvester)\.py$' { $areas.Add("evidence"); break }
            '^tests/test_(pipeline_review|cli_smoke|eval|goldset_validation)\.py$' { $areas.Add("pipeline"); break }
            '^tests/test_(consistency|report|checklist)\.py$' { $areas.Add("output"); break }
        }
    }
    return Get-UniqueOrderedItems -Items $areas.ToArray()
}

function Resolve-T0Modules {
    param(
        [string[]]$RequestedAreas,
        [string[]]$RequestedTests
    )
    if (@($RequestedTests).Count -gt 0) {
        return Get-UniqueOrderedItems -Items $RequestedTests
    }

    $areaMap = Get-AreaModulesMap
    $resolvedAreas = @(
        $RequestedAreas |
        ForEach-Object { [string]$_ } |
        ForEach-Object { $_ -split "," } |
        ForEach-Object { $_.Trim().ToLowerInvariant() } |
        Where-Object { $_ }
    )
    if (@($resolvedAreas).Count -eq 0) {
        $changedFiles = Get-ChangedFiles
        $resolvedAreas = @(Get-AreasFromChangedFiles -Files $changedFiles)
        if (@($resolvedAreas).Count -gt 0) {
            Write-Host ("T0 auto-selected areas from changed files: " + ($resolvedAreas -join ", "))
        }
    }

    if (@($resolvedAreas).Count -eq 0) {
        $resolvedAreas = @("rules", "pipeline")
        Write-Host "T0 fallback areas: rules, pipeline (no explicit area and no changed-file mapping)."
    }

    $invalid = @()
    foreach ($area in $resolvedAreas) {
        if (-not $areaMap.Contains($area)) {
            $invalid += $area
        }
    }
    if (@($invalid).Count -gt 0) {
        throw "Unknown area(s): $($invalid -join ', '). Allowed: $($areaMap.Keys -join ', ')"
    }

    $modules = New-Object "System.Collections.Generic.List[string]"
    foreach ($area in $resolvedAreas) {
        foreach ($module in $areaMap[$area]) {
            $modules.Add($module)
        }
    }
    return Get-UniqueOrderedItems -Items $modules.ToArray()
}

function Invoke-UnittestModules {
    param(
        [string]$Runner,
        [string[]]$Modules,
        [string]$Title,
        [switch]$AllowNoAi
    )
    if (@($Modules).Count -eq 0) {
        throw "No unittest modules specified for: $Title"
    }
    $cmd = if ($Runner -eq "uv") {
        "uv run python -m unittest " + ($Modules -join " ") + " -v"
    }
    else {
        "python -m unittest " + ($Modules -join " ") + " -v"
    }
    Write-Section "$Title -> $cmd"
    $pyArgs = @("-m", "unittest") + @($Modules) + @("-v")
    $exitCode = Invoke-PythonRunner -Runner $Runner -PyArgs $pyArgs -AllowNoAi:$AllowNoAi
    return $exitCode
}

function Invoke-T2Lanes {
    param(
        [string]$Runner,
        [bool]$ParallelEnabled
    )

    $lanes = [ordered]@{
        "Lane-A" = @(
            "tests.test_ocr",
            "tests.test_pipeline_pdf_ocr_smoke",
            "tests.test_annotate_output",
            "tests.test_review",
            "tests.test_constraints",
            "tests.test_rule_tier",
            "tests.test_task_planner"
        )
        "Lane-B" = @(
            "tests.test_llm_review",
            "tests.test_llm_judge",
            "tests.test_evidence_index",
            "tests.test_evidence_harvester",
            "tests.test_deepseek_reviewer"
        )
        "Lane-C" = @(
            "tests.test_pipeline_review",
            "tests.test_cli_smoke",
            "tests.test_eval",
            "tests.test_goldset_validation",
            "tests.test_report",
            "tests.test_checklist",
            "tests.test_consistency"
        )
    }

    if (-not $ParallelEnabled) {
        foreach ($entry in $lanes.GetEnumerator()) {
            $exitCode = Invoke-UnittestModules -Runner $Runner -Modules $entry.Value -Title "T2 $($entry.Key)" -AllowNoAi
            if ($exitCode -ne 0) { return 1 }
        }
        return 0
    }

    Write-Section "T2 parallel lanes"
    $jobs = @()
    foreach ($entry in $lanes.GetEnumerator()) {
        $jobs += Start-Job -Name $entry.Key -ScriptBlock {
            param(
                [string]$LaneName,
                [string]$LaneRunner,
                [string[]]$LaneModules,
                [string]$RepoRoot
            )
            Set-StrictMode -Version Latest
            $ErrorActionPreference = "Continue"
            if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
                $PSNativeCommandUseErrorActionPreference = $false
            }
            Set-Location $RepoRoot
            $env:BIDAGENT_ALLOW_NO_AI = "1"

            if ($LaneRunner -eq "uv") {
                & uv run python -m unittest @LaneModules -v 2>&1 | ForEach-Object { Write-Output $_ }
            }
            else {
                & python -m unittest @LaneModules -v 2>&1 | ForEach-Object { Write-Output $_ }
            }
            $code = if ($null -eq $LASTEXITCODE) { 1 } else { [int]$LASTEXITCODE }
            Write-Output "__L3_MATRIX_EXIT__=$code"
        } -ArgumentList @($entry.Key, $Runner, [string[]]$entry.Value, (Get-Location).Path)
    }

    $jobs | Wait-Job | Out-Null
    $laneFailed = $false

    foreach ($job in $jobs) {
        Write-Section ("T2 " + $job.Name + " output")
        $output = @()
        try {
            $output = @(Receive-Job -Job $job -ErrorAction Continue)
        }
        catch {
            Write-Host ("Lane receive failed: " + $_.Exception.Message)
            $laneFailed = $true
            continue
        }

        $exitCode = 1
        foreach ($item in $output) {
            if ($item -is [string] -and $item -like "__L3_MATRIX_EXIT__=*") {
                $exitCode = [int]($item -replace "^__L3_MATRIX_EXIT__=", "")
                continue
            }
            if ($item -is [string] -and $item -eq "System.Management.Automation.RemoteException") {
                continue
            }
            if ($item -is [System.Management.Automation.RemoteException]) {
                $message = [string]$item.Message
                if (-not [string]::IsNullOrWhiteSpace($message) -and $message -ne "System.Management.Automation.RemoteException") {
                    Write-Host $message
                }
                continue
            }
            if ($item -is [System.Management.Automation.ErrorRecord]) {
                $message = [string]$item.Exception.Message
                if (-not [string]::IsNullOrWhiteSpace($message)) {
                    Write-Host $message
                }
                continue
            }
            Write-Host $item
        }

        if ($job.State -ne "Completed") {
            Write-Host ("Lane state: " + $job.State)
            $laneFailed = $true
            continue
        }
        if ($exitCode -ne 0) {
            Write-Host ("Lane exit code: " + $exitCode)
            $laneFailed = $true
        }
    }

    $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
    if ($laneFailed) { return 1 }
    return 0
}

function Invoke-VerifyScript {
    param([string]$ScriptPath)
    Write-Section "T1 verify -> .\\scripts\\verify.ps1"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $ScriptPath | Out-Host
    return $LASTEXITCODE
}

function Invoke-T2GoldsetValidation {
    param([string]$Runner)
    $cmd = if ($Runner -eq "uv") {
        "uv run python scripts/validate-goldset.py --path docs/goldset/l3-gold.jsonl"
    }
    else {
        "python scripts/validate-goldset.py --path docs/goldset/l3-gold.jsonl"
    }
    Write-Section "T2 goldset -> $cmd"
    return (Invoke-PythonRunner -Runner $Runner -PyArgs @("scripts/validate-goldset.py", "--path", "docs/goldset/l3-gold.jsonl"))
}

function Invoke-RcE2E {
    param(
        [string]$Runner,
        [string]$TenderPath,
        [string]$BidPath,
        [string]$OutPath,
        [string]$Provider,
        [string]$Model,
        [string]$RcOcrMode,
        [string]$FailFastMode,
        [int]$MinStreak
    )

    if ([string]::IsNullOrWhiteSpace($TenderPath) -or [string]::IsNullOrWhiteSpace($BidPath) -or [string]::IsNullOrWhiteSpace($OutPath)) {
        throw "RC requires -Tender, -Bid, and -OutDir."
    }

    Write-Section "RC run pipeline"
    $runArgs = @(
        "run",
        "--tender", $TenderPath,
        "--bid", $BidPath,
        "--out", $OutPath,
        "--ocr-mode", $RcOcrMode,
        "--ai-provider", $Provider,
        "--ai-model", $Model,
        "--release-mode", "auto_final",
        "--gate-fail-fast", $FailFastMode,
        "--canary-min-streak", [string]$MinStreak
    )
    $runExit = Invoke-BidagentCommand -Runner $Runner -CliArgs $runArgs
    if ($runExit -ne 0) { return $runExit }

    Write-Section "RC eval"
    $evalExit = Invoke-BidagentCommand -Runner $Runner -CliArgs @("eval", "--out", $OutPath)
    if ($evalExit -ne 0) { return $evalExit }

    Write-Section "RC gate"
    $gateExit = Invoke-BidagentCommand -Runner $Runner -CliArgs @(
        "gate",
        "--out", $OutPath,
        "--release-mode", "auto_final",
        "--gate-fail-fast", $FailFastMode
    )
    if ($gateExit -ne 0) { return $gateExit }

    $requiredArtifacts = @(
        "requirements.jsonl",
        "review-tasks.jsonl",
        "evidence-packs.jsonl",
        "findings.jsonl",
        "verdicts.jsonl",
        "gate-result.json",
        "release/run-metadata.json",
        "release/canary-result.json",
        "release/release-trace.json"
    )

    Write-Section "RC artifact checks"
    $missing = @()
    foreach ($relativePath in $requiredArtifacts) {
        $target = Join-Path $OutPath $relativePath
        if (Test-Path $target) {
            Write-Host ("ok " + $relativePath)
        }
        else {
            Write-Host ("missing " + $relativePath)
            $missing += $relativePath
        }
    }
    if (@($missing).Count -gt 0) {
        Write-Error ("RC artifact check failed. Missing: " + ($missing -join ", "))
        return 1
    }
    return 0
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Push-Location $repoRoot

try {
    $runner = Get-PythonRunner
    if ($null -eq $runner) {
        Write-Error "Neither 'uv' nor 'python' command is available."
        exit 2
    }

    $parallelEnabled = Resolve-ParallelEnabled -Mode $Parallel -CurrentLevel $Level
    $results = New-Object "System.Collections.Generic.List[object]"

    switch ($Level) {
        "t0" {
            $modules = Resolve-T0Modules -RequestedAreas $Areas -RequestedTests $Tests
            $exitCode = Invoke-UnittestModules -Runner $runner -Modules $modules -Title "T0 quick regression" -AllowNoAi
            $results.Add([pscustomobject]@{
                    Step     = "t0_tests"
                    Status   = if ($exitCode -eq 0) { "pass" } else { "fail" }
                    ExitCode = $exitCode
                })
        }
        "t1" {
            $verifyPath = Join-Path $scriptDir "verify.ps1"
            $exitCode = Invoke-VerifyScript -ScriptPath $verifyPath
            $results.Add([pscustomobject]@{
                    Step     = "t1_verify"
                    Status   = if ($exitCode -eq 0) { "pass" } else { "fail" }
                    ExitCode = $exitCode
                })
        }
        "t2" {
            $lanesExit = Invoke-T2Lanes -Runner $runner -ParallelEnabled:$parallelEnabled
            $results.Add([pscustomobject]@{
                    Step     = "t2_lanes"
                    Status   = if ($lanesExit -eq 0) { "pass" } else { "fail" }
                    ExitCode = $lanesExit
                })
            if ($lanesExit -eq 0) {
                $goldsetExit = Invoke-T2GoldsetValidation -Runner $runner
                $results.Add([pscustomobject]@{
                        Step     = "t2_goldset"
                        Status   = if ($goldsetExit -eq 0) { "pass" } else { "fail" }
                        ExitCode = $goldsetExit
                    })
            }
        }
        "rc" {
            $lanesExit = Invoke-T2Lanes -Runner $runner -ParallelEnabled:$parallelEnabled
            $results.Add([pscustomobject]@{
                    Step     = "rc_t2_lanes"
                    Status   = if ($lanesExit -eq 0) { "pass" } else { "fail" }
                    ExitCode = $lanesExit
                })
            if ($lanesExit -eq 0) {
                $goldsetExit = Invoke-T2GoldsetValidation -Runner $runner
                $results.Add([pscustomobject]@{
                        Step     = "rc_t2_goldset"
                        Status   = if ($goldsetExit -eq 0) { "pass" } else { "fail" }
                        ExitCode = $goldsetExit
                    })
                if ($goldsetExit -eq 0) {
                    $rcExit = Invoke-RcE2E `
                        -Runner $runner `
                        -TenderPath $Tender `
                        -BidPath $Bid `
                        -OutPath $OutDir `
                        -Provider $AiProvider `
                        -Model $AiModel `
                        -RcOcrMode $OcrMode `
                        -FailFastMode $GateFailFast `
                        -MinStreak $CanaryMinStreak
                    $results.Add([pscustomobject]@{
                            Step     = "rc_e2e"
                            Status   = if ($rcExit -eq 0) { "pass" } else { "fail" }
                            ExitCode = $rcExit
                        })
                }
            }
        }
    }

    Write-Section "Summary"
    $results | Format-Table Step, Status, ExitCode -AutoSize | Out-String | Write-Host

    $failed = $results | Where-Object { $_.Status -eq "fail" }
    if (@($failed).Count -gt 0) {
        exit 1
    }
    exit 0
}
finally {
    Pop-Location
}
