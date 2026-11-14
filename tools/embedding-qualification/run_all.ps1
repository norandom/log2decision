# Runs the five scripts of the embedding qualification (task 2.15) and keeps their output.
# Usage, from tools\embedding-qualification:
#   pwsh -File run_all.ps1 -Vectors D:\BuildArtifacts\log2decision\task-2.15\vectors `
#        -Record D:\Source\log2decision\models\science-2.1\record.json `
#        -Logs D:\BuildArtifacts\log2decision\task-2.15\logs
param(
    [Parameter(Mandatory)] [string] $Vectors,
    [Parameter(Mandatory)] [string] $Record,
    [Parameter(Mandatory)] [string] $Logs,
    [string] $Out = (Join-Path $PSScriptRoot '..\..\docs\images\2026-11-14-embedding-qualification')
)
$ErrorActionPreference = 'Stop'
if ((Resolve-Path .).Path -match '^[A-Za-z]:\\?$') { throw 'refusing to run from a drive root' }
if ($Logs -match '^[A-Za-z]:\\?$') { throw 'refusing to write logs to a drive root' }
New-Item -ItemType Directory -Force $Logs | Out-Null
$env:PYTHONUTF8 = '1'
# numba (under umap-learn) compiles on first use; keep its cache out of the repository.
$env:NUMBA_CACHE_DIR = Join-Path $Logs '..\numba-cache'
foreach ($script in 'q0_inputs', 'q1_collapse', 'q2_similarity', 'q3_structure', 'q4_dimensions') {
    $log = Join-Path $Logs "$($script.Substring(0, 2)).log"
    uv run python "$script.py" --vectors $Vectors --record $Record --out $Out 2>&1 | Out-File $log -Encoding utf8
    if ($LASTEXITCODE -ne 0) { Get-Content $log -Tail 20; throw "$script failed, see $log" }
    Write-Output "$script done, output in $log"
}
