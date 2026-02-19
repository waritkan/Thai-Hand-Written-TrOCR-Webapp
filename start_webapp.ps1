# start_webapp.ps1
# เปิด Thai OCR Web Application โดยใช้ venv

$ProjectRoot = $PSScriptRoot

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Starting Thai OCR Web Application" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Activate venv
$VenvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-Not (Test-Path $VenvActivate)) {
    Write-Host "[ERROR] ไม่พบ venv ที่: $VenvActivate" -ForegroundColor Red
    exit 1
}

Write-Host "[1/2] Activating venv..." -ForegroundColor Yellow
& $VenvActivate

# Run app.py
$AppPath = Join-Path $ProjectRoot "webapp\app.py"
if (-Not (Test-Path $AppPath)) {
    Write-Host "[ERROR] ไม่พบ app.py ที่: $AppPath" -ForegroundColor Red
    exit 1
}

Write-Host "[2/2] Starting Flask server..." -ForegroundColor Yellow
Write-Host "       URL: http://localhost:5000" -ForegroundColor Green
Write-Host ""

python $AppPath
