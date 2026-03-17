# 🔒 Screen Watcher - 화면 감시 프로그램

노트북 카메라를 통해 화면을 보는 사람을 실시간으로 감시하고,
등록되지 않은 사람이 지정한 방향을 보고 있을 경우 **무소음으로 자동 캡쳐**합니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| 📷 **얼굴 감지** | MediaPipe Face Mesh로 실시간 얼굴 감지 |
| 🧠 **사용자 식별** | face_recognition으로 등록된 사용자 비교 |
| 👁 **시선 추정** | solvePnP 기반 머리 방향(Yaw/Pitch) 추정 |
| 📸 **무소음 캡쳐** | 미등록 사용자 감지 시 10초 간격 자동 캡쳐 |
| 🔧 **백그라운드** | systemd 서비스로 부팅 시 자동 실행 |

## 동작 흐름

```
카메라 프레임 캡쳐
       ↓
얼굴 감지 (MediaPipe Face Mesh)
       ↓
머리 방향 추정 (solvePnP → Yaw/Pitch)
       ↓
지정 방향을 보고 있는가? ──(아니오)──→ 무시
       ↓ (예)
등록된 사용자인가? ──(예)──→ 무시
       ↓ (아니오)
10초 경과? ──(아니오)──→ 대기
       ↓ (예)
📸 카메라 프레임 캡쳐 저장 (무소음)
```

## 설치

### Windows 11

권장 환경:

- Python `3.10` 또는 `3.11`
- PowerShell
- Visual Studio Build Tools C++ 또는 미리 설치된 `dlib` wheel

자동 설치:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

이 스크립트는 먼저 **공식 CMake가 PATH에 있는지** 확인하고, 없으면 `winget`으로 설치를 시도합니다.

수동 설치:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --force-reinstall git+https://github.com/ageitgey/face_recognition_models
python -m pip install -r requirements.txt
python -c "import pkg_resources, face_recognition_models, face_recognition; print('face recognition packages ok')"
```

만약 `face_recognition` 또는 `dlib` 설치에서 실패하면 아래를 먼저 설치하세요.

- Visual Studio 2022 Build Tools
- `Desktop development with C++`
- CMake (반드시 **공식 설치판** + PATH 등록)

### Windows 11에서 `dlib` / `CMake is not installed` 오류가 날 때

이 에러는 대부분 `pip install cmake`로는 해결되지 않고, **Windows PATH에 공식 CMake가 없어서** 발생합니다.

권장 해결 순서:

1. 공식 CMake 설치

```powershell
winget install --id Kitware.CMake -e
```

또는 [cmake.org](https://cmake.org/download/)에서 설치하고,
설치 중 `Add CMake to the system PATH for all users` 옵션을 반드시 체크하세요.

2. PowerShell을 완전히 종료 후 다시 열기

3. 설치 확인

```powershell
cmake --version
```

4. 다시 설치 실행

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

5. 여전히 실패하면 Visual Studio Build Tools 설치

필수 구성요소:

- `Desktop development with C++`
- MSVC v143 이상
- Windows 10/11 SDK

### Windows 11에서 `Please install face_recognition_models` 가 반복될 때

이 경우는 대부분 아래 둘 중 하나입니다.

- `pip`가 현재 venv가 아닌 다른 Python에 설치함
- `face_recognition_models`가 부분 설치되었거나 깨진 캐시를 사용함

반드시 **현재 venv 안에서** 아래 순서로 다시 설치하세요.

```powershell
.\venv\Scripts\Activate.ps1
python -m pip uninstall -y face-recognition-models face_recognition_models face_recognition
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --force-reinstall git+https://github.com/ageitgey/face_recognition_models
python -m pip install --upgrade face_recognition
python -c "import sys; print(sys.executable)"
python -c "import pkg_resources, face_recognition_models, face_recognition; print('face recognition packages ok')"
```

그 다음 다시 실행하세요.

```powershell
python .\register_face.py joseph --dir joseph\
```

### Windows 11에서 `No module named 'pkg_resources'` 가 뜰 때

일부 Windows Python 버전은 venv 생성 시 `setuptools`를 포함하지 않습니다.
`pip install setuptools`가 실패할 수 있으므로, **`ensurepip`으로 먼저 부트스트랩**해야 합니다.

```powershell
.\venv\Scripts\Activate.ps1
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -c "import pkg_resources; print('setuptools OK')"
```

위 명령 이후에도 face_recognition 관련 오류가 나오면:

```powershell
python -m pip install --no-cache-dir --force-reinstall git+https://github.com/ageitgey/face_recognition_models
python -m pip install --upgrade face_recognition
python -c "import pkg_resources, face_recognition_models, face_recognition; print('face recognition packages ok')"
```

### Linux / macOS

자동 설치:

```bash
chmod +x setup.sh
./setup.sh
```

수동 설치:

1. **시스템 의존성** (dlib 빌드에 필요):

```bash
# Ubuntu/Debian
sudo apt-get install cmake build-essential libopenblas-dev liblapack-dev

# macOS
brew install cmake
```

2. **Python 가상환경 및 패키지**:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

현재 기본 설정은 아래 책상 배치를 기준으로 맞춰져 있습니다.

```text
┌─────────────────┐     40cm     ┌──────────┐
│   큰 모니터      │ <---------> │  노트북   │
│   (정면)         │             │ (카메라)  │
└─────────────────┘              └──────────┘
```

## 사용법

### 1단계: 얼굴 등록 (필수)

```bash
# 카메라로 등록 (SPACE로 촬영, ESC로 완료)
# 여러 각도에서 3~5장 촬영 권장
python register_face.py 홍길동

# 이미지 파일로 등록
python register_face.py 홍길동 --image my_photo.jpg

# 폴더의 모든 이미지로 한번에 등록
python register_face.py 홍길동 --dir my_photos/

# 등록된 사용자 목록
python register_face.py --list

# 사용자 삭제
python register_face.py --delete 홍길동
```

### 2단계: 모니터링 시작

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python .\main.py start
python .\main.py start --debug
```

Linux / macOS:

```bash
# 일반 모드
python main.py start

# 디버그 모드 (미리보기 창 표시 - 동작 확인용)
python main.py start --debug

# 백그라운드 실행
nohup python main.py start > /dev/null 2>&1 &
```

### 3단계: 백그라운드 실행 등록 (선택)

#### Windows 11 - 작업 스케줄러

1. `작업 스케줄러` 실행
2. `작업 만들기` 선택
3. `로그온할 때` 트리거 추가
4. 동작은 아래처럼 지정

```text
프로그램/스크립트: powershell.exe
인수 추가: -ExecutionPolicy Bypass -WindowStyle Hidden -File "C:\path\to\screenstaredetector\setup_windows.ps1" -SkipInstall -StartMonitor
시작 위치: C:\path\to\screenstaredetector
```

또는 시작프로그램에 아래 바로가기를 넣어도 됩니다.

```powershell
powershell -ExecutionPolicy Bypass -WindowStyle Hidden -Command "cd 'C:\path\to\screenstaredetector'; .\venv\Scripts\Activate.ps1; python .\main.py start"
```

#### Linux - systemd

```bash
# 서비스 파일 복사
mkdir -p ~/.config/systemd/user/
cp screenwatcher.service ~/.config/systemd/user/

# 서비스 활성화 및 시작
systemctl --user daemon-reload
systemctl --user enable screenwatcher
systemctl --user start screenwatcher

# 상태 확인
systemctl --user status screenwatcher

# 로그 확인
journalctl --user -u screenwatcher -f

# 서비스 중지
systemctl --user stop screenwatcher
```

## 설정 (config.yaml)

| 항목 | 설명 | 기본값 |
|------|------|--------|
| `target_direction` | 감시 방향 (`"screen"`, `"left"`, `"right"`, `"up"`, `"down"`) | `"screen"` |
| `direction_threshold` | 방향 판단 임계값 (도, 낮을수록 엄격) | `15` |
| `camera_offset_angle` | 카메라가 사용자 정면에서 벗어난 각도 (오른쪽=양수) | `30` |
| `face_recognition_tolerance` | 얼굴 비교 허용 오차 (0.0~1.0, 낮을수록 엄격) | `0.6` |
| `recognition_model` | 인식 모델 (`"small"`=빠름, `"large"`=정확) | `"small"` |
| `capture_interval` | 캡쳐 간격 (초) | `10` |
| `camera_index` | 카메라 장치 번호 | `0` |
| `also_capture_screen` | 스크린 캡쳐도 함께 저장 | `false` |
| `show_preview` | 미리보기 창 표시 (디버그용) | `false` |
| `process_interval` | 프레임 처리 간격 (초) | `0.5` |

## 디렉토리 구조

```
screensaver/
├── config.yaml            # 설정 파일
├── main.py                # 메인 진입점
├── register_face.py       # 얼굴 등록 유틸리티
├── requirements.txt       # Python 패키지 목록
├── setup.sh               # 설치 스크립트
├── setup_windows.ps1      # Windows 11 설치/실행 스크립트
├── screenwatcher.service  # systemd 서비스 파일
├── modules/
│   ├── __init__.py
│   ├── camera.py          # 스레드 기반 카메라 모듈
│   ├── gaze_estimator.py  # 시선/머리 방향 추정 모듈
│   ├── face_recognizer.py # 얼굴 인식/비교 모듈
│   ├── capturer.py        # 무소음 캡쳐 모듈
│   └── monitor.py         # 모니터링 통합 모듈
├── registered_faces/      # 등록된 얼굴 인코딩 데이터
└── captures/              # 캡쳐된 이미지 저장소
```

## 트러블슈팅

### dlib 설치 오류

```bash
# cmake 설치 확인
cmake --version

# Ubuntu
sudo apt-get install cmake build-essential libopenblas-dev
```

Windows:

```powershell
cmake --version
```

`cmake --version` 이 실패하면, `pip install cmake`가 아니라 **공식 CMake 설치 + PATH 등록**이 필요합니다.
그래도 실패하면 `Visual Studio Build Tools`의 C++ 구성요소를 설치한 뒤 다시 시도하세요.

### 카메라를 찾을 수 없음

```bash
# 사용 가능한 카메라 확인
ls /dev/video*

# config.yaml에서 camera_index 변경
```

Windows:

- 카메라가 다른 앱(Zoom, Teams, 카메라 앱)에서 점유 중인지 확인
- [config.yaml](config.yaml) 의 `camera_index`를 `0`, `1`, `2` 순서로 바꿔 테스트

### 얼굴 인식 정확도 향상

- 다양한 각도/조명에서 **여러 장** 등록
- `face_recognition_tolerance` 값 조절 (낮을수록 엄격)
- `recognition_model`을 `"large"`로 변경 (느리지만 정확)

### 디버그 모드로 동작 확인

```bash
python main.py start --debug
```

미리보기 창에서:
- 🟡 **노란색 박스**: 감지된 얼굴 + 방향 정보
- 🟢 **초록색 박스**: 등록된 사용자
- 🔴 **빨간색 박스**: 미등록 사용자 (캡쳐 대상)

## 기술 스택

- **OpenCV** — 카메라 제어 및 이미지 처리
- **MediaPipe Face Mesh** — 468개 랜드마크 기반 얼굴 감지 + 시선 추정
- **face_recognition (dlib)** — 얼굴 인코딩 및 비교
- **mss** — 무소음 스크린 캡쳐 (선택)
- **PyYAML** — 설정 파일 관리
