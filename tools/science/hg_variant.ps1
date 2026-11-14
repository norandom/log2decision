# Runs one variant of the hierarchical encoder gate (task 2.16) and keeps its output.
# A run that the host stopped is started again and goes on from its last epoch
# (`checkpoint.pt` in the variant's directory), up to five times.
# Usage, from tools\science:
#   pwsh -NoProfile -File hg_variant.ps1 -Name entropy -Patcher entropy-jump [-Seed 0] [-Threads 6] [-More '--label-weight', '0.5']
param(
    [Parameter(Mandatory)] [string] $Name,
    [Parameter(Mandatory)] [string] $Patcher,
    [int] $Seed = 0,
    [int] $Threads = 6,
    [string[]] $More = @(),
    [string] $Root = 'D:\BuildArtifacts\log2decision\task-2.16'
)
if ((Resolve-Path .).Path -match '^[A-Za-z]:\\?$') { throw 'refusing to run from a drive root' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:L2D_HG_DIR = $Root
New-Item -ItemType Directory -Force (Join-Path $Root 'logs') | Out-Null
$log = Join-Path $Root "logs\variant-$Name-s$Seed.log"
$record = Join-Path $Root "variants\$Name-s$Seed\record.json"
foreach ($attempt in 1..5) {
    uv run python hg_run.py variant --name $Name --patcher $Patcher --seed $Seed --threads $Threads @More 2>&1 |
        Out-File $log -Encoding utf8 -Append
    if (Test-Path $record) { Write-Output "$Name-s$Seed done (attempt $attempt), output in $log"; exit 0 }
    "attempt $attempt ended with exit code $LASTEXITCODE and no record; starting again" | Out-File $log -Encoding utf8 -Append
}
throw "$Name-s$Seed did not finish in five attempts, see $log"
