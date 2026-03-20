#!/bin/bash
# =========================================
# Screen Watcher 설치 스크립트
# =========================================

set -e

echo "========================================="
echo " 🔒 Screen Watcher 설치"
echo "========================================="

# Python 가상환경 생성
echo ""
echo "[1/3] Python 가상환경 생성..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  → 가상환경 생성 완료"
else
    echo "  → 기존 가상환경 사용"
fi

source venv/bin/activate

# 패키지 설치
echo ""
echo "[2/3] Python 패키지 설치..."
pip install --upgrade pip wheel
pip install -r requirements.txt

# insightface/mediapipe가 opencv-python-headless를 설치할 수 있으므로
# GUI 지원이 있는 opencv-python으로 강제 재설치
pip install --force-reinstall opencv-python

# 설치 검증
python -c "from insightface.app import FaceAnalysis; import cv2; print('InsightFace + OpenCV OK')"

# 디렉토리 생성
echo ""
echo "[3/3] 디렉토리 생성..."
mkdir -p registered_faces captures

echo ""
echo "========================================="
echo " ✅ 설치 완료!"
echo "========================================="
echo ""
echo "사용법:"
echo "  source venv/bin/activate"
echo "  python register_face.py <이름> --dir <사진폴더>   # 얼굴 등록"
echo "  python main.py start                              # 모니터링 시작"
echo "  python main.py start --debug                      # 디버그 모드"
echo ""
