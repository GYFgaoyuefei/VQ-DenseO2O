param([string]$Python = "python", [string]$Model = "all", [switch]$ConfigOnly)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "../../.."))
$cliArgs = @("tools/paper_v2/run.py", "preflight", "--model", $Model)
if ($ConfigOnly) { $cliArgs += "--config-only" }
& $Python -X utf8 @cliArgs
exit $LASTEXITCODE
