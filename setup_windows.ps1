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

function Test-CMakeAvailable {
    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        $null = & cmake --version 2>$null
        return ($LASTEXITCODE -eq 0)
    }

    return $false
}

function Add-CMakeToCurrentPath {
    $candidatePaths = @(
        "$Env:ProgramFiles\CMake\bin",
        "$Env:ProgramFiles(x86)\CMake\bin"
    )

    foreach ($candidate in $candidatePaths) {
        if (Test-Path $candidate) {
            if (-not ($Env:Path -split ';' | Where-Object { $_ -eq $candidate })) {
                $Env:Path = "$candidate;$Env:Path"
            }
        }
    }
}

function Ensure-CMake {
    Write-Host ""
    Write-Host "[2/5] CMake 확인..."

    Add-CMakeToCurrentPath
    if (Test-CMakeAvailable) {
        Write-Host "  → CMake 확인 완료"
        return
    }

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "  → winget으로 공식 CMake 설치 시도"
        & winget install --id Kitware.CMake -e --accept-source-agreements --accept-package-agreements
        Add-CMakeToCurrentPath
        if (Test-CMakeAvailable) {
            Write-Host "  → CMake 설치 완료"
            return
        }
    }

    throw @"
CMake를 사용할 수 없습니다.

다음 순서로 해결하세요:
1. https://cmake.org/download/ 에서 Windows x64 Installer 설치
2. 설치 중 'Add CMake to the system PATH for all users' 옵션 체크
3. PowerShell을 완전히 닫았다가 다시 열기
4. 다시 실행: powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
"@
}

function Install-FaceRecognitionDependencies {
    Write-Host "  → setuptools/wheel 복구"
    & ".\venv\Scripts\python.exe" -m pip install --upgrade setuptools wheel

    Write-Host "  → face_recognition_models 설치"
    & ".\venv\Scripts\python.exe" -m pip install --no-cache-dir --force-reinstall git+https://github.com/ageitgey/face_recognition_models

    Write-Host "  → face_recognition 설치"
    & ".\venv\Scripts\python.exe" -m pip install --upgrade face_recognition

    Write-Host "  → 설치 검증"
    & ".\venv\Scripts\python.exe" -c "import pkg_resources, face_recognition_models, face_recognition; print('face recognition packages ok')"
}

if (-not $SkipInstall) {
    Write-Host ""
    Write-Host "[1/5] Python 확인..."
    $pythonCmd = Get-PythonCommand
    Write-Host ("  → 사용 명령: " + ($pythonCmd -join " "))

    Ensure-CMake

    Write-Host ""
    Write-Host "[3/5] 가상환경 생성..."
    if (-not (Test-Path ".\venv")) {
        Invoke-Python -Arguments @("-m", "venv", "venv")
        Write-Host "  → venv 생성 완료"
    }
    else {
        Write-Host "  → 기존 venv 사용"
    }

    Write-Host ""
    Write-Host "[4/5] Python 패키지 설치..."
    & ".\venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
    try {
        & ".\venv\Scripts\python.exe" -m pip install -r requirements.txt
        Install-FaceRecognitionDependencies
    }
    catch {
        Write-Warning "requirements 설치 중 오류가 발생했습니다."
        Write-Warning "Windows에서는 face_recognition/dlib 설치에 공식 CMake와 Visual Studio Build Tools(C++)가 필요할 수 있습니다."
        Write-Warning "README의 Windows 문제 해결 섹션을 확인하세요."
        throw
    }

    Write-Host ""
    Write-Host "[5/5] 디렉토리 생성..."
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