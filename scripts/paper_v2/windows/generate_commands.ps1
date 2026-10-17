param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "../../.."))
& $Python tools/paper_v2/run.py commands
exit $LASTEXITCODE
