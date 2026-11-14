# Runs the qualification scripts (q0 to q5) on the vectors of variants of task 2.16, one
# after the other. Each name is a directory below <Root>\vectors (`entropy-s0-int8`);
# logs and pictures go to <Root>\qualification\<name>.
# Usage, from tools\embedding-qualification:
#   pwsh -NoProfile -File run_variants.ps1 -Root D:\BuildArtifacts\log2decision\task-2.16 `
#        -Names fixed-s0-f32, fixed-s0-int8
param(
    [Parameter(Mandatory)] [string] $Root,
    [Parameter(Mandatory)] [string[]] $Names
)
$ErrorActionPreference = 'Stop'
if ((Resolve-Path .).Path -match '^[A-Za-z]:\\?$') { throw 'refusing to run from a drive root' }
foreach ($name in $Names) {
    $vectors = Join-Path $Root "vectors\$name"
    $out = Join-Path $Root "qualification\$name"
    $source = switch -Regex ($name) {
        '-f32$' { 'in float (Python)' }
        '-int8$' { 'in 8 bits (Python reference)' }
        '-npu$' { 'from the NPU' }
        default { throw "the name $name does not end in -f32, -int8 or -npu" }
    }
    & (Join-Path $PSScriptRoot 'run_all.ps1') -Vectors $vectors -Record (Join-Path $vectors 'record.json') `
        -Logs $out -Out $out -InputMode whole -InputBytes 1024 -Source $source -Information
    Write-Output "$name qualified"
}
