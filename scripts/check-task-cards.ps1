param(
    [switch]$FailOnUntracked
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Command-Exists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Command-Exists -Name "git")) {
    Write-Host "git not found; skipping task card discipline check."
    exit 0
}

# Untracked task cards are the common failure mode: created but never committed.
$untracked = @()
try {
    $untracked = & git ls-files --others --exclude-standard docs/tasks | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}
catch {
    Write-Host "Failed to query git status; skipping task card discipline check."
    exit 0
}

# Only enforce on real task cards, not ad-hoc notes.
$taskCards = $untracked | Where-Object { $_ -match '^docs/tasks/TASK-.*\.md$' }

# Allow local/private review notes to remain untracked if desired.
$taskCards = $taskCards | Where-Object { $_ -notmatch 'code-review' }

$taskCardsCount = @($taskCards).Count
if ($taskCardsCount -eq 0) {
    Write-Host "Task card discipline: ok (no untracked TASK cards)."
    exit 0
}

Write-Host "Task card discipline: found untracked TASK cards:"
$taskCards | ForEach-Object { Write-Host ("- " + $_) }

if ($FailOnUntracked) {
    Write-Error "Untracked TASK cards found. Commit them or remove them."
    exit 1
}

Write-Host "Note: run with -FailOnUntracked to enforce as a gate."
exit 0
