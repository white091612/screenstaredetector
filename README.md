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

### 자동 설치

```bash
chmod +x setup.sh
./setup.sh
```

### 수동 설치

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

```bash
# 일반 모드
python main.py start

# 디버그 모드 (미리보기 창 표시 - 동작 확인용)
python main.py start --debug

# 백그라운드 실행
nohup python main.py start > /dev/null 2>&1 &
```

### 3단계: 백그라운드 서비스 등록 (선택)

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

### 카메라를 찾을 수 없음

```bash
# 사용 가능한 카메라 확인
ls /dev/video*

# config.yaml에서 camera_index 변경
```

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
