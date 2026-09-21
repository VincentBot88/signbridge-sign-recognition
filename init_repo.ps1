# ---------------------------------------------------------------------------
# SignBridge - initialize the git repo locally and stage the first commit.
#
# Run from C:\Users\aweso\SignBridge in PowerShell:
#     .\init_repo.ps1
#
# This script does LOCAL work only. It never contacts GitHub and never pushes
# - it prints the push command at the end for you to run yourself.
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

$GitHubUser = "VincentBot88"
$RepoName   = "signbridge-sign-recognition"

Set-Location $PSScriptRoot

function Get-SizeBytes([string]$Path) {
    $item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    if ($item) { return [int64]$item.Length }
    return [int64]0
}

# --- Preflight -------------------------------------------------------------

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "git is not on PATH. Install it from https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

if (Test-Path .git) {
    Write-Host "A .git folder already exists here - stopping so nothing is clobbered." -ForegroundColor Yellow
    exit 1
}

# git exits non-zero when a config key is unset; relax the preference so that
# does not abort the script (PowerShell 7.4+ treats it as terminating).
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$who   = & git config --global user.name
$email = & git config --global user.email
$ErrorActionPreference = $prevEAP
$global:LASTEXITCODE = 0

if ([string]::IsNullOrWhiteSpace($who) -or [string]::IsNullOrWhiteSpace($email)) {
    Write-Host "git identity is not set. Run these two lines first, then re-run this script:" -ForegroundColor Yellow
    Write-Host '  git config --global user.name  "Vincent Cheng"'
    Write-Host '  git config --global user.email "you@example.com"'
    exit 1
}

foreach ($f in @(".gitignore", "README.md", "requirements.txt")) {
    if (-not (Test-Path -LiteralPath $f)) {
        Write-Host "Missing $f - expected it next to this script." -ForegroundColor Red
        exit 1
    }
}

# --- Stage -----------------------------------------------------------------

git init -b main | Out-Null
git add .

$files = @(git ls-files)
if ($files.Count -eq 0) {
    Write-Host "Nothing staged - check .gitignore." -ForegroundColor Red
    exit 1
}

$total = 0
foreach ($f in $files) { $total += Get-SizeBytes $f }

Write-Host ""
Write-Host ("--- {0} files staged, {1:N1} MB total ---" -f $files.Count, ($total / 1MB)) -ForegroundColor Cyan
git status --short

# GitHub warns above 50 MB per file and rejects above 100 MB.
$big = @($files | Where-Object { (Get-SizeBytes $_) -gt 10MB })
if ($big.Count -gt 0) {
    Write-Host ""
    Write-Host "WARNING - files over 10 MB are staged. Add them to .gitignore unless you meant this:" -ForegroundColor Yellow
    foreach ($f in $big) {
        Write-Host ("  {0,8:N1} MB  {1}" -f ((Get-SizeBytes $f) / 1MB), $f) -ForegroundColor Yellow
    }
}

Write-Host ""
$ans = Read-Host "Commit these files? (y/n)"
if ($ans -ne "y") {
    git reset | Out-Null
    Write-Host "Stopped, nothing committed. The .git folder is still here; 'rmdir /s .git' removes it."
    exit
}

# --- Commit ----------------------------------------------------------------

git commit -q -m "SignBridge sign recognition pipeline (v1-v3 feature representations)"
git tag -a v3 -m "v3: body-relative features, val 0.895"
git remote add origin "https://github.com/$GitHubUser/$RepoName.git"

Write-Host ""
Write-Host "Committed and tagged v3. Remote set to https://github.com/$GitHubUser/$RepoName" -ForegroundColor Green
Write-Host ""
Write-Host "Next:" -ForegroundColor Cyan
Write-Host "  1. Create an EMPTY repo named '$RepoName' at https://github.com/new"
Write-Host "     (owner $GitHubUser, Public, and do NOT add a README, .gitignore or licence)"
Write-Host "  2. Then run:"
Write-Host "       git push -u origin main --follow-tags" -ForegroundColor White
