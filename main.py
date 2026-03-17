#!/usr/bin/env python3
"""
Screen Watcher - 개인정보 보호 모니터링 프로그램

노트북 카메라를 통해 화면을 보는 사람을 감시하고,
등록되지 않은 사람이 화면을 볼 경우 자동으로 캡쳐합니다.

사용법:
    python main.py start                # 모니터링 시작
    python main.py start --debug        # 디버그 모드 (미리보기 활성)
    python main.py register 홍길동       # 카메라로 얼굴 등록
    python main.py register 홍길동 --image photo.jpg  # 이미지로 등록
    python main.py list                 # 등록된 사용자 목록
    python main.py delete 홍길동         # 사용자 삭제
"""

import argparse
import os
import sys
import signal
import logging
import yaml


def load_config(config_path):
    """설정 파일 로드"""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def setup_logging(config, debug=False):
    """로깅 설정"""
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = config.get("log_file", "screenwatcher.log")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


# ============================================================
# 서브커맨드 핸들러
# ============================================================


def cmd_start(args):
    """모니터링 시작"""
    config = load_config(args.config)
    setup_logging(config, args.debug)

    if args.debug:
        config["show_preview"] = True

    from modules.monitor import Monitor

    monitor = Monitor(config)

    def on_signal(sig, frame):
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    monitor.start()


def cmd_register(args):
    """얼굴 등록"""
    config = load_config(args.config)
    setup_logging(config)

    from modules.face_recognizer import FaceRecognizer

    face_data_dir = config.get("face_data_dir", "./registered_faces")
    recognizer = FaceRecognizer(face_data_dir=face_data_dir)

    if args.image:
        # 이미지 파일에서 등록
        if not os.path.exists(args.image):
            print(f"❌ 파일을 찾을 수 없습니다: {args.image}")
            sys.exit(1)
        try:
            recognizer.register(args.image, args.name)
            print(f"✅ 얼굴 등록 완료: {args.name}")
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
    else:
        # 카메라에서 등록
        import cv2

        camera_index = config.get("camera_index", 0)
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("❌ 카메라를 열 수 없습니다.")
            sys.exit(1)

        print(f"📸 '{args.name}' 얼굴 등록 모드")
        print("   SPACE: 촬영 | ESC: 완료")
        print("   여러 각도에서 3~5장 촬영하면 정확도가 향상됩니다.")

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(
                display,
                f"Registered: {count} | SPACE: capture | ESC: done",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Register Face", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                try:
                    recognizer.register_from_frame(frame, args.name)
                    count += 1
                    print(f"  ✅ {count}번째 촬영 완료")
                except ValueError:
                    print("  ❌ 얼굴을 찾을 수 없습니다. 다시 시도하세요.")

        cap.release()
        cv2.destroyAllWindows()

        if count > 0:
            print(f"\n✅ 총 {count}장 등록 완료: {args.name}")
        else:
            print("\n등록된 얼굴이 없습니다.")


def cmd_list(args):
    """등록된 사용자 목록"""
    config = load_config(args.config)

    from modules.face_recognizer import FaceRecognizer

    face_data_dir = config.get("face_data_dir", "./registered_faces")
    recognizer = FaceRecognizer(face_data_dir=face_data_dir)

    registered = recognizer.list_registered()
    if not registered:
        print("등록된 얼굴이 없습니다.")
    else:
        print("📋 등록된 사용자:")
        for name, count in registered.items():
            print(f"  👤 {name} ({count}개 인코딩)")


def cmd_delete(args):
    """등록된 사용자 삭제"""
    config = load_config(args.config)

    from modules.face_recognizer import FaceRecognizer

    face_data_dir = config.get("face_data_dir", "./registered_faces")
    recognizer = FaceRecognizer(face_data_dir=face_data_dir)

    try:
        recognizer.delete(args.name)
        print(f"✅ 삭제 완료: {args.name}")
    except ValueError as e:
        print(f"❌ {e}")


# ============================================================
# 메인 진입점
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="🔒 Screen Watcher - 개인정보 보호 모니터링 프로그램",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py start                            모니터링 시작
  python main.py start --debug                    디버그 모드 (미리보기 표시)
  python main.py register 홍길동                   카메라로 얼굴 등록
  python main.py register 홍길동 --image photo.jpg 이미지로 등록
  python main.py list                             등록된 사용자 목록
  python main.py delete 홍길동                     사용자 삭제
        """,
    )
    parser.add_argument(
        "--config", default="config.yaml", help="설정 파일 경로 (기본: config.yaml)"
    )

    subparsers = parser.add_subparsers(dest="command", help="실행할 명령")

    # start
    start_p = subparsers.add_parser("start", help="모니터링 시작")
    start_p.add_argument(
        "--debug", action="store_true", help="디버그 모드 (미리보기 창 표시)"
    )

    # register
    reg_p = subparsers.add_parser("register", help="얼굴 등록")
    reg_p.add_argument("name", help="등록할 사용자 이름")
    reg_p.add_argument("--image", "-i", help="이미지 파일 경로 (미입력 시 카메라)")

    # list
    subparsers.add_parser("list", help="등록된 사용자 목록")

    # delete
    del_p = subparsers.add_parser("delete", help="등록된 사용자 삭제")
    del_p.add_argument("name", help="삭제할 사용자 이름")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "start": cmd_start,
        "register": cmd_register,
        "list": cmd_list,
        "delete": cmd_delete,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
