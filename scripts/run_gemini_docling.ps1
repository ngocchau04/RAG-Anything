param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath
)

$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    throw "Missing virtualenv. Expected .venv at repo root."
}

. .\.venv\Scripts\Activate.ps1

if (-not (Test-Path ".\.env")) {
    if (Test-Path ".\env.gemini.example") {
        Copy-Item ".\env.gemini.example" ".\.env"
        Write-Host "Created .env from env.gemini.example. Update LLM_BINDING_API_KEY first."
        exit 1
    }
    throw "Missing .env and env.gemini.example"
}

python .\examples\raganything_example.py $FilePath --parser docling --base-url $env:LLM_BINDING_HOST --api-key $env:LLM_BINDING_API_KEY
