param([string]$RunTag = 'study01', [switch]$IncludeSensitivity)
$ErrorActionPreference = 'Stop'
$repo = if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { Join-Path $PSScriptRoot '../../..' }
$pythonBin = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
if (-not $env:TOMATO_DATA_ROOT -and $env:DATA_ROOT) { $env:TOMATO_DATA_ROOT = $env:DATA_ROOT }
if (-not $env:THESIS_KD_OUTPUT_ROOT -and $env:OUTPUT_ROOT) { $env:THESIS_KD_OUTPUT_ROOT = $env:OUTPUT_ROOT }
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $repo
try {
    $extra = @()
    if ($IncludeSensitivity) { $extra += '--include-sensitivity' }
    & $pythonBin -X utf8 -m thesis_kd.cli export-commands --run-tag $RunTag @extra
    exit $LASTEXITCODE
} finally { Pop-Location }
