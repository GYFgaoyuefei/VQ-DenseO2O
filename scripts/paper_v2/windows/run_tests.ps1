param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "../../.."))
& $Python -m pytest tests/paper_v2 -q
exit $LASTEXITCODE
