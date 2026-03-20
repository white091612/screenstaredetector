# 🔒 Screen Watcher - 화면 감시 프로그램

노트북 카메라를 통해 화면을 보는 사람을 실시간으로 감시하고,
등록되지 않은 사람이 지정한 방향을 보고 있을 경우 **무소음으로 자동 캡쳐 + 화면 잠금**합니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| 📷 **얼굴 감지** | MediaPipe Face Mesh로 실시간 얼굴 감지 |
| 🧠 **사용자 식별** | InsightFace (ArcFace 512d) + cosine similarity + 다수결 판정 |
| 👁 **시선 추정** | solvePnP 기반 머리 방향(Yaw/Pitch) 추정 |
| 👥 **다중 인물** | 전체 프레임 분석 + IoU bbox 매칭으로 안정적인 다중 인물 처리 |
| 📸 **무소음 캡쳐** | 미등록 사용자 감지 시 10초 간격 자동 캡쳐 |
| 🔒 **화면 잠금** | 미등록 사용자 감지 시 Win+L 자동 잠금 |

## 동작 흐름

```
카메라 프레임 캡쳐
       ├─→ MediaPipe Face Mesh ──→ 머리 방향 추정 (Yaw/Pitch) ──→ 방향 판정
       │                                ↓ bbox list
       └─→ InsightFace app.get(frame) ──→ 전체 얼굴 Detection + ArcFace (512d)
                                         ↓ identity list
                                    IoU bbox 매칭 (방향 + 신원 결합)
                                         ↓
                       미등록자가 화면 봄 AND 등록자가 화면 안 봄
                                         ↓
                              📸 캡쳐 + 🔒 화면 잠금
```

## 설치

### 필수 요구사항

- Python **3.10** 또는 **3.11**
- pip

> ⚠️ CMake, Visual Studio Build Tools, dlib 등은 **더 이상 필요하지 않습니다.**
> InsightFace는 사전 빌드된 ONNX 모델을 사용하므로 C++ 컴파일러가 불필요합니다.

### Windows 11

자동 설치:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

수동 설치:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
# insightface가 opencv-python-headless를 설치할 수 있으므로 GUI 버전으로 덮어쓰기
python -m pip install --force-reinstall opencv-python
# 설치 확인
python -c "from insightface.app import FaceAnalysis; import cv2; print('OK')"
```

> **중요:** `pip install -r requirements.txt` 후 반드시 `pip install --force-reinstall opencv-python`을
> 실행하세요. 이 단계를 빠뜨리면 `--debug` 모드에서 `cv2.imshow` 관련 에러가 발생합니다.

### Linux / macOS

자동 설치:

```bash
chmod +x setup.sh
./setup.sh
```

수동 설치:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --force-reinstall opencv-python
python -c "from insightface.app import FaceAnalysis; import cv2; print('OK')"
```

### 첫 실행 시 모델 다운로드

InsightFace `buffalo_l` 모델(~326MB)이 첫 실행 시 자동 다운로드됩니다.

```
다운로드 위치: ~/.insightface/models/buffalo_l/
```

네트워크 환경에 따라 1~5분 소요될 수 있습니다.

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
# 폴더의 모든 이미지로 한번에 등록 (권장)
# 정면 + 좌/우 30° + 다른 조명 포함 5~10장
python register_face.py 홍길동 --dir my_photos/

# 카메라로 등록 (SPACE로 촬영, ESC로 완료)
python register_face.py 홍길동

# 이미지 파일로 등록
python register_face.py 홍길동 --image my_photo.jpg

# 등록된 사용자 목록
python register_face.py --list

# 사용자 삭제
python register_face.py --delete 홍길동
```

> **등록 팁:** 같은 이름으로 다양한 각도(정면, 좌측 30°, 우측 30°)의 사진을 등록하면
> 인식 정확도가 크게 향상됩니다. ArcFace가 pose-invariant이므로 좌/우를 별도 분류할 필요 없습니다.

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
```

## 설정 (config.yaml)

| 항목 | 설명 | 기본값 |
|------|------|--------|
| `target_direction` | 감시 방향 (`"screen"`, `"left"`, `"right"`, `"up"`, `"down"`) | `"screen"` |
| `direction_threshold` | 방향 판단 임계값 (도, 낮을수록 엄격) | `15` |
| `camera_offset_angle` | 카메라가 사용자 정면에서 벗어난 각도 (오른쪽=양수) | `50` |
| `face_recognition_threshold` | Cosine Similarity 임계값 (0.0\~1.0, 높을수록 엄격) | `0.4` |
| `insightface_model` | InsightFace 모델 (`"buffalo_l"`=정확, `"buffalo_sc"`=빠름) | `"buffalo_l"` |
| `insightface_det_size` | Detection 입력 크기 (높을수록 정확, 느림) | `640` |
| `max_faces` | 최대 동시 감지 얼굴 수 | `4` |
| `capture_interval` | 캡쳐 간격 (초) | `10` |
| `lock_screen_on_unknown` | 미등록 사용자 감지 시 화면 잠금 | `true` |
| `lock_cooldown` | 잠금 후 재잠금 대기 시간 (초) | `30` |
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
├── setup.sh               # Linux/macOS 설치 스크립트
├── setup_windows.ps1      # Windows 11 설치/실행 스크립트
├── screenwatcher.service  # systemd 서비스 파일
├── modules/
│   ├── __init__.py
│   ├── camera.py          # 스레드 기반 카메라 모듈
│   ├── gaze_estimator.py  # 시선/머리 방향 추정 모듈
│   ├── face_recognizer.py # InsightFace 얼굴 인식 모듈
│   ├── capturer.py        # 무소음 캡쳐 모듈
│   ├── screen_locker.py   # 화면 잠금 모듈
│   └── monitor.py         # 모니터링 통합 모듈
├── registered_faces/      # 등록된 얼굴 임베딩 데이터
└── captures/              # 캡쳐된 이미지 저장소
```

## 트러블슈팅

### `cv2.imshow` / `The function is not implemented` 에러

`insightface` 또는 `mediapipe`가 `opencv-python-headless`를 설치하면서 GUI 함수가 제거됩니다.

```bash
pip install --force-reinstall opencv-python
```

이 명령으로 GUI 지원이 포함된 full 버전으로 교체하세요.

### 카메라를 찾을 수 없음

```bash
# Linux: 사용 가능한 카메라 확인
ls /dev/video*
```

- Windows: 카메라가 다른 앱(Zoom, Teams)에서 점유 중인지 확인
- `config.yaml`의 `camera_index`를 `0`, `1`, `2` 순서로 바꿔 테스트

### 얼굴 인식 정확도 향상

- 다양한 각도/조명에서 **5~10장** 등록 (정면 + 좌/우 30°)
- `face_recognition_threshold` 값 올리기 (0.5 = 엄격)
- `insightface_det_size`를 `640`으로 유지 (기본값)

### InsightFace 모델 다운로드 실패

```bash
# 모델 수동 다운로드
# https://github.com/deepinsight/insightface/releases/tag/v0.7
# buffalo_l.zip 다운로드 후 ~/.insightface/models/buffalo_l/ 에 압축 해제
```

### 기존 dlib 인코딩 호환 불가

InsightFace 전환 후 기존 `encodings.pkl`은 호환되지 않습니다.
자동으로 감지되어 무시되며, 재등록 안내가 표시됩니다.

```bash
rm registered_faces/encodings.pkl
python register_face.py 홍길동 --dir photos/
```

### 디버그 모드로 동작 확인

```bash
python main.py start --debug
```

미리보기 창에서:
- 🟡 **노란색 박스**: 감지된 얼굴 + 방향 정보
- 🟢 **초록색 박스**: 등록된 사용자
- 🔴 **빨간색 박스**: 미등록 사용자 (캡쳐 대상)

## 기술 스택

- **InsightFace (ArcFace)** — 512차원 얼굴 임베딩 + SCRFD 얼굴 감지
- **ONNX Runtime** — CPU 기반 모델 추론
- **MediaPipe Face Mesh** — 468개 랜드마크 기반 머리 방향 추정
- **OpenCV** — 카메라 제어 및 이미지 처리
- **mss** — 무소음 스크린 캡쳐 (선택)
- **PyYAML** — 설정 파일 관리
