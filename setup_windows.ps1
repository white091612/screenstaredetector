param(
    [switch]$SkipInstall,
    [switch]$StartMonitor
)

$ErrorActionPreference = "Stop"

Write-Host "========================================="
Write-Host " 🔒 Screen Watcher Windows 설치"
Write-Host "========================================="

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $py311 = & py -3.11 --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @("py", "-3.11")
        }

        $py310 = & py -3.10 --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @("py", "-3.10")
        }

        return @("py")
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }

    throw "Python을 찾을 수 없습니다. Python 3.10 또는 3.11을 먼저 설치하세요."
}

function Invoke-Python {
    param(
        [string[]]$Arguments
    )

    $pythonCmd = Get-PythonCommand
    if ($pythonCmd.Length -gt 1) {
        & $pythonCmd[0] $pythonCmd[1..($pythonCmd.Length - 1)] @Arguments
    }
    else {
        & $pythonCmd[0] @Arguments
    }
}

if (-not $SkipInstall) {
    Write-Host ""
    Write-Host "[1/4] Python 확인..."
    $pythonCmd = Get-PythonCommand
    Write-Host ("  → 사용 명령: " + ($pythonCmd -join " "))

    Write-Host ""
    Write-Host "[2/4] 가상환경 생성..."
    if (-not (Test-Path ".\venv")) {
        Invoke-Python -Arguments @("-m", "venv", "venv")
        Write-Host "  → venv 생성 완료"
    }
    else {
        Write-Host "  → 기존 venv 사용"
    }

    Write-Host ""
    Write-Host "[3/4] Python 패키지 설치..."
    & ".\venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
    & ".\venv\Scripts\python.exe" -m pip install cmake
    try {
        & ".\venv\Scripts\python.exe" -m pip install -r requirements.txt
    }
    catch {
        Write-Warning "requirements 설치 중 오류가 발생했습니다."
        Write-Warning "Windows에서는 face_recognition/dlib 설치에 Visual Studio Build Tools(C++)가 필요할 수 있습니다."
        throw
    }

    Write-Host ""
    Write-Host "[4/4] 디렉토리 생성..."
    New-Item -ItemType Directory -Force -Path ".\registered_faces" | Out-Null
    New-Item -ItemType Directory -Force -Path ".\captures" | Out-Null
}

Write-Host ""
Write-Host "========================================="
Write-Host " ✅ 준비 완료"
Write-Host "========================================="
Write-Host ""
Write-Host "사용법:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python .\register_face.py <이름>"
Write-Host "  python .\main.py start"
Write-Host "  python .\main.py start --debug"

if ($StartMonitor) {
    Write-Host ""
    Write-Host "모니터링을 시작합니다..."
    & ".\venv\Scripts\python.exe" ".\main.py" "start"
}