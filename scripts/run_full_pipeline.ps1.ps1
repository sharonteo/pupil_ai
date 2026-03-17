# ------------------------------------------------------------
# NeurOptics Pupillometry – End‑to‑End Clinical AI Pipeline
# One‑Click Orchestration Script
# ------------------------------------------------------------
# This script:
#   1. Generates synthetic pupillometry data
#   2. Trains ML models (Logistic Regression, RF, XGBoost)
#   3. Launches the Streamlit dashboard
# ------------------------------------------------------------

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "`n=== NeurOptics Pupillometry AI Pipeline ===`n" -ForegroundColor Cyan

# ------------------------------------------------------------
# Resolve project root
# ------------------------------------------------------------
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir  = Split-Path -Parent $scriptPath
$projectRoot = Split-Path -Parent $scriptDir

Write-Host "Project root detected at: $projectRoot" -ForegroundColor DarkGray

# ------------------------------------------------------------
# Ensure Python can import the src/ package
# ------------------------------------------------------------
$env:PYTHONPATH = $projectRoot
Write-Host "PYTHONPATH set to: $env:PYTHONPATH" -ForegroundColor DarkGray

# ------------------------------------------------------------
# Step 1 — Generate Synthetic Data
# ------------------------------------------------------------
Write-Host "`n→ Generating synthetic pupillometry dataset..." -ForegroundColor Yellow
python "$projectRoot\src\generate_synthetic_data.py"

# ------------------------------------------------------------
# Step 2 — Train ML Models
# ------------------------------------------------------------
Write-Host "`n→ Training machine learning models..." -ForegroundColor Yellow
python "$projectRoot\src\train_models.py"

# ------------------------------------------------------------
# Step 3 — Launch Streamlit Dashboard
# ------------------------------------------------------------
Write-Host "`n→ Launching Streamlit dashboard..." -ForegroundColor Yellow
streamlit run "$projectRoot\app\streamlit_app.py"

Write-Host "`nPipeline complete. Streamlit app is running." -ForegroundColor Green