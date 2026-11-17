param(
    [ValidateSet('all','report','audit')][string]$Stage='all',
    [string]$Root='D:\BuildArtifacts\log2decision\task-2.18-replay',
    [string]$Main='C:\Users\mariu\Source\log2decision',
    [string]$Historical='D:\BuildArtifacts\log2decision\task-2.17'
)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Root=[IO.Path]::GetFullPath($Root)
if(-not $Root.StartsWith('D:\BuildArtifacts\log2decision\',[StringComparison]::OrdinalIgnoreCase)){throw 'Output must be a project artifact subdirectory'}
$run=Join-Path $Root 'run01'
$logs=Join-Path $Root 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$uv=(Get-Command uv.exe).Source
function Admission {
    $target='D:\BuildArtifacts\log2decision\task-1.26-target'
    $size=(Get-ChildItem $target -Recurse -File | Measure-Object Length -Sum).Sum/1GB
    $free=(Get-PSDrive D).Free/1GB
    if($size -ge 30 -or $free -lt 50){throw "Admission refused: target $size GiB; free $free GiB"}
    Write-Host "Admission: target $size GiB; free $free GiB"
}
function Run([string]$exe,[string[]]$arguments,[string]$name) {
    Admission
    & $exe @arguments *> (Join-Path $logs "$name.log")
    if($LASTEXITCODE -ne 0){throw "$name failed; see $(Join-Path $logs "$name.log")"}
    Get-Content (Join-Path $logs "$name.log") -Tail 3
}
Push-Location $repo
try {
    $env:CARGO_TARGET_DIR='D:\BuildArtifacts\log2decision\task-1.26-target'
    $env:CARGO_PROFILE_DEV_DEBUG='0'; $env:CARGO_PROFILE_TEST_DEBUG='0'; $env:CARGO_INCREMENTAL='0'
    if(-not $env:UV_PROJECT_ENVIRONMENT){$env:UV_PROJECT_ENVIRONMENT='D:\BuildArtifacts\log2decision\science-venv'}
    $env:L2D_BUSY='refuse'
    if($Stage -eq 'all') {
        if(Test-Path $run){throw 'Use a fresh output root for a complete replay'}
        Run $uv @('run','--project','tools/science','python','tools/science/transfer218.py','prepare',"$Main\data\csic-2010\csic_database.csv",$Historical,$run) 'prepare'
        Run 'cargo.exe' @('test','-p','l2d-baseline','--test','transfer218','-p','l2d-encoder','--test','npu_hierarchical_gate','--no-run','--message-format=json','-j4') 'build'
        $executables=@{}
        foreach($line in Get-Content (Join-Path $logs 'build.log')) {
            if($line.StartsWith('{')) {
                $message=$line | ConvertFrom-Json
                if($message.reason -eq 'compiler-artifact' -and $message.executable){$executables[$message.target.name]=$message.executable}
            }
        }
        if(-not $executables.transfer218 -or -not $executables.npu_hierarchical_gate){throw 'Missing built measurement executable'}
        $executables | ConvertTo-Json | Set-Content (Join-Path $Root 'executables.json')
        $env:L2D_TRANSFER218=$run
        foreach($fold in 0..4) {
            $env:L2D_TRANSFER_FOLD="$fold"
            Run $executables.transfer218 @('--ignored','--exact','export_markset_comparison','--nocapture','--test-threads=1') "baseline-f$fold"
        }
        Run $uv @('run','--project','tools/science','python','tools/science/transfer218.py','fixtures',"$Main\models\task-2.17",$Historical,$run) 'fixtures'
        $savedPath=$env:PATH
        try {
            $env:PATH="$env:SystemRoot\System32;$env:SystemRoot"
            $env:L2D_RUNTIME_DIR="$Main\runtime\ryzenai"; $env:L2D_MEMORY_LIMIT='4GiB'
            foreach($fold in 0..4) {
                $env:L2D_HG_FIXTURE="$run\f$fold\npu"
                $env:L2D_NPU_WORK_DIR="$Root\npu-work\f$fold"
                Run $executables.npu_hierarchical_gate @('--ignored','--exact','transfer218_export','--nocapture','--test-threads=1') "npu-f$fold"
            }
        } finally {$env:PATH=$savedPath}
    }
    if($Stage -in @('all','audit')) {
        Run $uv @('run','--project','tools/science','python','tools/science/transfer218_verify.py',$run,$Historical) 'encoding-audit'
    }
    if($Stage -in @('all','report')) {
        Run $uv @('run','--project','tools/science','python','tools/science/transfer218_report.py',$run) 'report'
    }
} finally {Pop-Location}
