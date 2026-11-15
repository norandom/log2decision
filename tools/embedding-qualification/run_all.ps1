# Runs the scripts of the embedding qualification (task 2.15) and keeps their output.
# Usage, from tools\embedding-qualification:
#   pwsh -File run_all.ps1 -Vectors D:\BuildArtifacts\log2decision\task-2.15\vectors `
#        -Record D:\Source\log2decision\models\science-2.1\record.json `
#        -Logs D:\BuildArtifacts\log2decision\task-2.15\logs
# For an encoder that reads the whole line (task 2.16), with the information view (q5):
#   pwsh -File run_all.ps1 -Vectors <dir> -Record <dir>\record.json -Logs <dir> -Out <dir> `
#        -InputMode whole -InputBytes 1024 -Source 'in float, Python' -Information
param(
    [Parameter(Mandatory)] [string] $Vectors,
    [Parameter(Mandatory)] [string] $Record,
    [Parameter(Mandatory)] [string] $Logs,
    [string] $Out = (Join-Path $PSScriptRoot '..\..\docs\images\2026-11-14-embedding-qualification'),
    [ValidateSet('payload', 'whole')] [string] $InputMode = 'payload',
    [int] $InputBytes = 256,
    [string] $Source = 'from the NPU',
    [switch] $Information
)
$ErrorActionPreference = 'Stop'
if ((Resolve-Path .).Path -match '^[A-Za-z]:\\?$') { throw 'refusing to run from a drive root' }
if ($Logs -match '^[A-Za-z]:\\?$') { throw 'refusing to write logs to a drive root' }
New-Item -ItemType Directory -Force $Logs | Out-Null
$env:PYTHONUTF8 = '1'
# numba (under umap-learn) compiles on first use; keep its cache out of the repository.
$env:NUMBA_CACHE_DIR = Join-Path $Logs '..\numba-cache'
$scripts = @('q0_inputs', 'q1_collapse', 'q2_similarity', 'q3_structure', 'q4_dimensions')
if ($Information) { $scripts += 'q5_information' }
foreach ($script in $scripts) {
    $log = Join-Path $Logs "$($script.Substring(0, 2)).log"
    # Task 2.17 (requirement 15.11): one attempt; a script stopped by the host is a fault
    # to fix, not a run to repeat. With $env:L2D_MEMORY_GIB set, each script runs under
    # that memory limit and records its peak (eq.py).
    uv run python "$script.py" --vectors $Vectors --record $Record --out $Out --input $InputMode --input-bytes $InputBytes --source $Source 2>&1 | Out-File $log -Encoding utf8
    if ($LASTEXITCODE -ne 0) { Get-Content $log -Tail 20; throw "$script failed (exit code $LASTEXITCODE), see $log" }
    Write-Output "$script done, output in $log"
}
