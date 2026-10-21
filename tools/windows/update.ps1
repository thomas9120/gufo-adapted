# Switch existing Windows installs to windows-main, update, and rebuild Gufo.
[CmdletBinding()]
param(
  [string]$Rocm = "",
  [string]$Vcpkg = "",
  [string]$Ninja = "",
  [ValidateRange(1, 64)][int]$Jobs = 4
)
$ErrorActionPreference = "Stop"
# Reuse setup's read-only tool discovery and running-executable guard.
. "$PSScriptRoot\setup.ps1" -Rocm $Rocm -Vcpkg $Vcpkg -Jobs $Jobs

function Invoke-UpdateGit([string[]]$Arguments) {
  & $tools.Git -C $root @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Git failed (exit $LASTEXITCODE). Fix the reported issue and rerun update-windows.bat." }
}

function Invoke-WindowsUpdate {
  $tools = Get-SetupTools
  if ($Ninja) { $tools.Ninja = (Get-Command $Ninja -CommandType Application -ErrorAction Stop).Source }
  $missing = @(@("Git", "VS", "CMake", "Ninja", "Rocm", "Vcpkg") | Where-Object { -not $tools.$_ })
  if ($missing.Count -or -not (Test-Path -LiteralPath "$($tools.Vcpkg)\vcpkg.exe")) {
    throw "Build prerequisites are missing ($($missing -join ', ')). Run setup-windows.bat first."
  }
  Assert-EngineStopped
  $changes = Invoke-UpdateGit @("status", "--porcelain", "--untracked-files=all")
  if ($changes) { throw "This checkout has local changes or untracked files. Commit, stash or move them before updating." }
  $branch = & $tools.Git -C $root symbolic-ref --quiet --short HEAD
  if ($LASTEXITCODE -ne 0) { throw "Check out a branch before updating; detached HEAD cannot be updated automatically." }
  if ($branch -notin @("windows-port", "windows-main")) {
    throw "Check out windows-port or windows-main before updating; branch '$branch' is left untouched."
  }
  Write-Host "Fetching origin/windows-main..." -ForegroundColor Cyan
  $fetch = @("fetch", "--no-tags", "origin", "+refs/heads/windows-main:refs/remotes/origin/windows-main")
  if ($branch -eq "windows-port") {
    $fetch += "+refs/heads/windows-port:refs/remotes/origin/windows-port"
  }
  Invoke-UpdateGit $fetch
  if ($branch -eq "windows-port") {
    & $tools.Git -C $root merge-base --is-ancestor HEAD refs/remotes/origin/windows-port
    if ($LASTEXITCODE -ne 0) {
      throw "windows-port has unpublished commits or differs from origin/windows-port. Preserve or publish your work before switching."
    }
    & $tools.Git -C $root show-ref --verify --quiet refs/heads/windows-main
    $branchStatus = $LASTEXITCODE
    if ($branchStatus -eq 0) {
      & $tools.Git -C $root merge-base --is-ancestor refs/heads/windows-main refs/remotes/origin/windows-main
      if ($LASTEXITCODE -ne 0) {
        throw "Local windows-main cannot fast-forward to origin/windows-main. Resolve its local history before updating."
      }
      Invoke-UpdateGit @("switch", "windows-main")
    } elseif ($branchStatus -eq 1) {
      Invoke-UpdateGit @("switch", "--no-track", "-c", "windows-main", "origin/windows-main")
    } else {
      throw "Git could not check the local windows-main branch (exit $branchStatus)."
    }
  }
  Write-Host "Updating windows-main from origin/windows-main..." -ForegroundColor Cyan
  Invoke-UpdateGit @("merge", "--ff-only", "--no-autostash", "origin/windows-main")
  Invoke-UpdateGit @("branch", "--set-upstream-to=origin/windows-main", "windows-main")
  Assert-EngineStopped
  $env:PATH = "$(Split-Path $tools.Git);$(Split-Path $tools.CMake);$(Split-Path $tools.Ninja);$env:PATH"
  $shell = Join-Path $PSHOME "powershell.exe"
  if (-not (Test-Path -LiteralPath $shell)) { $shell = Join-Path $PSHOME "pwsh.exe" }
  Write-Host "Rebuilding build\release\gufo.exe..." -ForegroundColor Cyan
  & $shell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\build.ps1" `
    -Preset release -Target gufo -Jobs $Jobs -Rocm $tools.Rocm -Vcpkg $tools.Vcpkg -Ninja $tools.Ninja
  if ($LASTEXITCODE -ne 0) { throw "Build failed. The source update is retained; fix the build issue and rerun update-windows.bat." }
  $engine = Join-Path $root "build\release\gufo.exe"
  & $engine --version
  if ($LASTEXITCODE -ne 0) { throw "The rebuilt executable failed its version check. See docs/WINDOWS.md for runtime prerequisites." }
  Write-Host "`nUpdate complete: $engine" -ForegroundColor Green
  Write-Host "Open launch-gui.bat and select this executable in the GUI."
}

if ($MyInvocation.InvocationName -ne ".") {
  try { Invoke-WindowsUpdate } catch {
    Write-Host "`nUpdate stopped: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
  }
}
