$ErrorActionPreference = 'Stop'
$repo = if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { Join-Path $PSScriptRoot '../../..' }
$pythonBin = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
if (-not $env:TOMATO_DATA_ROOT -and $env:DATA_ROOT) { $env:TOMATO_DATA_ROOT = $env:DATA_ROOT }
if (-not $env:THESIS_KD_OUTPUT_ROOT -and $env:OUTPUT_ROOT) { $env:THESIS_KD_OUTPUT_ROOT = $env:OUTPUT_ROOT }
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $repo
try {
    $tempDir = "outputs/thesis_kd_checks/pytest_$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Force -Path 'outputs/thesis_kd_checks' | Out-Null
    & $pythonBin -X utf8 -m pytest -q -p no:cacheprovider --basetemp $tempDir tests/thesis_kd tests/paper_v2/test_config_ablation_audit.py tests/paper_v2/test_config_runner.py tests/paper_v2/test_vq_quality.py tests/paper_v2/test_dc_sta.py @args
    exit $LASTEXITCODE
} finally { Pop-Location }
