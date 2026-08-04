# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

param(
    [switch]$SkipBackend,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SdkRoot = Resolve-Path (Join-Path $ScriptDir '..')
$ModelDownloadDir = Resolve-Path (Join-Path $SdkRoot '..')
$ComposeFile = Join-Path $ModelDownloadDir 'docker/compose.yaml'
$VenvDir = Join-Path $SdkRoot '.venv'
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { 'python' }
$Registry = if ($env:REGISTRY) { $env:REGISTRY } else { '' }
$Tag = if ($env:TAG) { $env:TAG } else { 'latest' }
$EnabledPlugins = if ($env:ENABLED_PLUGINS) { $env:ENABLED_PLUGINS } else { 'all' }
$ModelPath = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "$HOME/models" }
$GetiHost = if ($env:GETI_HOST) { $env:GETI_HOST } else { '' }
$GetiToken = if ($env:GETI_TOKEN) { $env:GETI_TOKEN } else { '' }
$GetiWorkspaceId = if ($env:GETI_WORKSPACE_ID) { $env:GETI_WORKSPACE_ID } else { '' }
$Hls3dPoseCheckpointUrl = if ($env:HLS_3D_POSE_CHECKPOINT_URL) { $env:HLS_3D_POSE_CHECKPOINT_URL } else { '' }
$HlsEcgBaseUrl = if ($env:HLS_ECG_BASE_URL) { $env:HLS_ECG_BASE_URL } else { '' }
$HlsRppgModelUrl = if ($env:HLS_RPPG_MODEL_URL) { $env:HLS_RPPG_MODEL_URL } else { '' }
$HfToken = if ($env:HUGGINGFACEHUB_API_TOKEN) { $env:HUGGINGFACEHUB_API_TOKEN } else { '' }
$HealthUrl = if ($env:HEALTH_URL) { $env:HEALTH_URL } else { 'http://localhost:8200/health' }
$HealthRetries = if ($env:HEALTH_RETRIES) { [int]$env:HEALTH_RETRIES } else { 60 }
$HealthInterval = if ($env:HEALTH_INTERVAL) { [int]$env:HEALTH_INTERVAL } else { 2 }

function Log([string]$Message) {
    Write-Host "[bootstrap] $Message"
}

if (-not $SkipBackend) {
    Log "Starting model-download backend from $ComposeFile"
    $env:REGISTRY = $Registry
    $env:TAG = $Tag
    $env:ENABLED_PLUGINS = $EnabledPlugins
    $env:MODEL_PATH = $ModelPath
    $env:GETI_HOST = $GetiHost
    $env:GETI_TOKEN = $GetiToken
    $env:GETI_WORKSPACE_ID = $GetiWorkspaceId
    $env:HLS_3D_POSE_CHECKPOINT_URL = $Hls3dPoseCheckpointUrl
    $env:HLS_ECG_BASE_URL = $HlsEcgBaseUrl
    $env:HLS_RPPG_MODEL_URL = $HlsRppgModelUrl
    $env:HUGGINGFACEHUB_API_TOKEN = $HfToken
    docker compose -f $ComposeFile up -d --build

    Log "Waiting for backend health at $HealthUrl"
    $ready = $false
    for ($i = 1; $i -le $HealthRetries; $i++) {
        try {
            Invoke-WebRequest -Uri $HealthUrl -Method GET -UseBasicParsing -TimeoutSec 5 | Out-Null
            $ready = $true
            break
        } catch {
            Start-Sleep -Seconds $HealthInterval
        }
    }
    if (-not $ready) {
        throw "Backend did not become healthy at $HealthUrl"
    }
    Log 'Backend is healthy'
} else {
    Log 'Skipping backend startup (--SkipBackend)'
}

if (-not $SkipInstall) {
    Log "Preparing virtual environment at $VenvDir"
    & $PythonBin -m venv $VenvDir

    $VenvPython = Join-Path $VenvDir 'Scripts/python.exe'
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -e $SdkRoot
} else {
    Log 'Skipping SDK install (--SkipInstall)'
}

$VenvPython = Join-Path $VenvDir 'Scripts/python.exe'
Log 'Running SDK health check'
& $VenvPython -c @"
import asyncio
from model_download_sdk.client import ModelDownloadSDK

async def main() -> None:
    client = ModelDownloadSDK()
    try:
        print(await client.health_check())
    finally:
        await client.close()

asyncio.run(main())
"@

Log 'Bootstrap completed'
Log "Try: $VenvDir\Scripts\Activate.ps1; model-download jobs"
