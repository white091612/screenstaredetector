#!/bin/bash
# =========================================
# Screen Watcher 설치 스크립트
# =========================================

set -e

echo "========================================="
echo " 🔒 Screen Watcher 설치"
echo "========================================="

# 시스템 패키지 확인 (dlib 빌드에 필요)
echo ""
echo "[1/4] 시스템 패키지 확인..."
if command -v apt-get &> /dev/null; then
    echo "  → Debian/Ubuntu 계열 감지"
    sudo apt-get update -qq
    sudo apt-get install -y -qq cmake build-essential libopenblas-dev liblapack-dev
elif command -v yum &> /dev/null; then
    echo "  → RHEL/CentOS 계열 감지"
    sudo yum install -y cmake gcc-c++ openblas-devel lapack-devel
elif command -v brew &> /dev/null; then
    echo "  → macOS 감지"
    brew install cmake
else
    echo "  ⚠ 패키지 매니저를 찾을 수 없습니다."
    echo "    cmake와 C++ 컴파일러가 설치되어 있는지 확인하세요."
fi

# Python 가상환경 생성
echo ""
echo "[2/4] Python 가상환경 생성..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  → 가상환경 생성 완료"
else
    echo "  → 기존 가상환경 사용"
fi

source venv/bin/activate

# 패키지 설치
echo ""
echo "[3/4] Python 패키지 설치..."
pip install --upgrade pip
pip install -r requirements.txt

# 디렉토리 생성
echo ""
echo "[4/4] 디렉토리 생성..."
mkdir -p registered_faces captures

echo ""
echo "========================================="
echo " ✅ 설치 완료!"
echo "========================================="
echo ""
echo "사용법:"
echo "  source venv/bin/activate"
echo "  python register_face.py <이름>        # 얼굴 등록"
echo "  python main.py start                  # 모니터링 시작"
echo "  python main.py start --debug          # 디버그 모드"
echo ""
