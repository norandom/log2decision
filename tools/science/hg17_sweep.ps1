# Runs steps of the stability gate (task 2.17) one after the other, never two at a time.
# Each step is its own process under a memory limit (hg17_run.py --memory-gib) and takes
# the project's lock for hungry processes. A step that fails stops the sweep: nothing is
# resumed, a run killed by the host (exit code -536870911, 0xE0000001) is a fault to fix.
# Usage, from tools\science:
#   pwsh -NoProfile -File hg17_sweep.ps1 -Folds 0,1,2,3,4 -Seeds 0,1,2 -Variants mask -Steps bytes,line,eval
param(
    # Comma-separated lists (they also arrive as one string through `pwsh -File`)
    [string] $Folds = '0,1,2,3,4',
    [string] $Seeds = '0,1,2',
    [string] $Variants = 'mask',
    [string] $Steps = 'bytes,line,eval',
    [string] $Tag = '',
    [double] $MemoryGiB = 6,
    [switch] $SkipDone,
    [string] $Root = 'D:\BuildArtifacts\log2decision\task-2.17'
)
$ErrorActionPreference = 'Continue'
$foldList = @($Folds -split ',' | ForEach-Object { [int] $_ })
$seedList = @($Seeds -split ',' | ForEach-Object { [int] $_ })
$variantList = @($Variants -split ',')
$stepList = @($Steps -split ',')
if ((Resolve-Path .).Path -match '^[A-Za-z]:\\?$') { throw 'refusing to run from a drive root' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TEMP = Join-Path $Root 'tmp'
$env:TMP = $env:TEMP
$env:L2D_HG17_DIR = $Root
New-Item -ItemType Directory -Force (Join-Path $Root 'logs'), $env:TEMP | Out-Null

function Invoke-Step([string] $Step, [int] $Fold, [int] $Seed, [string] $Variant) {
    $suffix = if ($Tag) { "-$Tag" } else { '' }
    $enc = Join-Path $Root "enc\f$Fold-s$Seed$suffix"
    $done = switch ($Step) {
        'bytes' { Join-Path $enc 'bytes.json' }
        'line' { Join-Path $enc "$Variant\line.json" }
        'eval' { Join-Path $enc "$Variant\eval.json" }
        default { '' }
    }
    if ($SkipDone -and $done -and (Test-Path $done)) { Write-Output "skip $Step f$Fold s$Seed $Variant$suffix (done)"; return }
    $log = Join-Path $Root "logs\$Step-f$Fold-s$Seed-$Variant$suffix.log"
    $argv = @('run', 'python', 'hg17_run.py', $Step, '--fold', $Fold, '--seed', $Seed, '--variant', $Variant, '--memory-gib', $MemoryGiB)
    if ($Tag) { $argv += @('--tag', $Tag) }
    $started = Get-Date
    & uv @argv 2>&1 | Where-Object { $_ -notmatch 'pre-processing' } | Out-File $log -Encoding utf8
    $code = $LASTEXITCODE
    $minutes = ((Get-Date) - $started).TotalMinutes
    if ($code -ne 0) {
        $why = if ($code -eq -536870911) { 'killed by the host for memory (0xE0000001)' } else { "exit code $code" }
        Write-Output ("FAILED {0} f{1} s{2} {3}{4} after {5:N1} min: {6}; see {7}" -f $Step, $Fold, $Seed, $Variant, $suffix, $minutes, $why, $log)
        exit 1
    }
    $last = (Get-Content $log | Where-Object { $_ -match '^RUN ' } | Select-Object -Last 1)
    Write-Output ("done {0} f{1} s{2} {3}{4}: {5:N1} min; {6}" -f $Step, $Fold, $Seed, $Variant, $suffix, $minutes, $last)
}

foreach ($seed in $seedList) {
    foreach ($fold in $foldList) {
        foreach ($step in $stepList) {
            if ($step -eq 'bytes') { Invoke-Step $step $fold $seed $variantList[0]; continue }
            foreach ($variant in $variantList) { Invoke-Step $step $fold $seed $variant }
        }
    }
}
Write-Output 'sweep finished'
