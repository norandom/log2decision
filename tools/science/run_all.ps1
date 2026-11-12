# Runs the science gate (task 2.1) run by run; a run whose record exists is skipped.
# Usage, from tools\science:
#   pwsh -File run_all.ps1 -Data D:\Source\log2decision\data\csic-2010\csic_database.csv `
#        -Out D:\BuildArtifacts\log2decision\science-2.1\runs -Batch main
param(
    [Parameter(Mandatory)] [string] $Data,
    [Parameter(Mandatory)] [string] $Out,
    [ValidateSet('baseline', 'main', 'keep', 'objectives', 'split17', 'penalty', 'keep-unit')] [string] $Batch = 'main',
    [int[]] $Seeds = @(0, 1, 2),
    [int] $Threads = 8
)
$ErrorActionPreference = 'Stop'
if ((Resolve-Path .).Path -match '^[A-Za-z]:\\?$') { throw 'refusing to run from a drive root' }
New-Item -ItemType Directory -Force $Out | Out-Null

function Invoke-Run([string] $Split, [int] $Fold, [int] $Seed, [string] $Routing, [string] $Objective) {
    $name = "$Split-f$Fold-s$Seed-$Routing-$Objective"
    if (Test-Path (Join-Path $Out "$name.json")) { Write-Output "skip $name"; return }
    $batchSize = if ($Objective -like 'supcon*') { 256 } else { 64 }
    uv run python run_gate.py --data $Data --out $Out --split $Split --fold $Fold --seed $Seed `
        --routing $Routing --objective $Objective --batch $batchSize --threads $Threads 2>&1 |
        Where-Object { "$_" -notmatch 'WARNING:root|^\s*$' }
}

switch ($Batch) {
    'baseline' {
        foreach ($fold in 0..4) { uv run python run_gate.py --data $Data --out $Out --split cv5 --fold $fold --baseline }
        uv run python run_gate.py --data $Data --out $Out --split split17 --baseline
    }
    'main' { foreach ($seed in $Seeds) { foreach ($fold in 0..4) { Invoke-Run cv5 $fold $seed strip bce-norm } } }
    'keep' { foreach ($seed in $Seeds) { foreach ($fold in 0..4) { Invoke-Run cv5 $fold $seed keep bce-norm } } }
    'objectives' {
        foreach ($fold in 0..4) { Invoke-Run cv5 $fold 0 strip bce }
    }
    'split17' {
        foreach ($seed in $Seeds) { Invoke-Run split17 0 $seed strip bce-norm }
        Invoke-Run split17 0 0 keep bce-norm
        foreach ($seed in $Seeds) { Invoke-Run split17 0 $seed strip bce-norm-unit }
        Invoke-Run split17 0 0 strip bce
        Invoke-Run split17 0 0 strip supcon
    }
    'keep-unit' {
        foreach ($fold in 0..4) { Invoke-Run cv5 $fold 0 keep bce-norm-unit }
        Invoke-Run split17 0 0 keep bce-norm-unit
    }
    'penalty' { foreach ($seed in $Seeds) { foreach ($fold in 0..4) { Invoke-Run cv5 $fold $seed strip bce-norm-unit } } }
}
