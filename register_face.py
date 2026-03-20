#!/usr/bin/env python3
"""
얼굴 등록 유틸리티

카메라 또는 이미지 파일에서 얼굴을 등록합니다.
여러 각도/조명에서 3~5장을 등록하면 인식 정확도가 향상됩니다.

사용법:
    python register_face.py <이름>                   카메라로 등록
    python register_face.py <이름> --image face.jpg   이미지 파일로 등록
    python register_face.py <이름> --dir photos/      폴더의 모든 이미지로 등록
    python register_face.py --list                    등록된 사용자 목록
    python register_face.py --delete <이름>           사용자 삭제
"""

import argparse
import os
import sys
import glob
import logging

import yaml
import cv2


def load_config(config_path="config.yaml"):
    """설정 파일 로드"""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def register_camera(name, config, camera_index=0):
    """카메라로 여러 장의 사진을 찍어 등록 (정확도 향상)"""
    from modules.face_recognizer import FaceRecognizer

    try:
        recognizer = FaceRecognizer(config)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # 마이그레이션 안내
    if not recognizer.known_embeddings and os.path.exists(recognizer._encoding_file):
        print("⚠ 기존 인코딩이 새 엔진(InsightFace)과 호환되지 않습니다.")
        print("  기존 등록 사진으로 다시 등록해주세요.")
        print()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        sys.exit(1)

    count = 0
    print(f"📸 '{name}' 얼굴 등록 모드")
    print("   SPACE: 촬영 | ESC: 완료")
    print("   정면 + 좌/우 옆모습을 포함하여 5~8장 촬영하면 정확도가 높아집니다.")
    print()

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
                recognizer.register_from_frame(frame, name)
                count += 1
                print(f"  ✅ {count}번째 촬영 완료")
            except ValueError:
                print("  ❌ 얼굴을 찾을 수 없습니다. 다양한 각도를 시도해 보세요.")

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"\n✅ 총 {count}장 등록 완료: {name}")
    else:
        print("\n등록된 얼굴이 없습니다.")


def register_image(name, image_path, config):
    """이미지 파일에서 등록"""
    from modules.face_recognizer import FaceRecognizer

    try:
        recognizer = FaceRecognizer(config)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    # 마이그레이션 안내
    if not recognizer.known_embeddings and os.path.exists(recognizer._encoding_file):
        print("⚠ 기존 인코딩이 새 엔진(InsightFace)과 호환되지 않습니다.")
        print("  기존 등록 사진으로 다시 등록해주세요.")
        print()

    try:
        recognizer.register(image_path, name)
        print(f"✅ 등록 완료: {name} ← {image_path}")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)


def register_directory(name, dir_path, config):
    """디렉토리의 모든 이미지에서 등록"""
    from modules.face_recognizer import FaceRecognizer

    try:
        recognizer = FaceRecognizer(config)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if not os.path.isdir(dir_path):
        print(f"❌ 디렉토리를 찾을 수 없습니다: {dir_path}")
        sys.exit(1)

    # 마이그레이션 안내
    if not recognizer.known_embeddings and os.path.exists(recognizer._encoding_file):
        print("⚠ 기존 인코딩이 새 엔진(InsightFace)과 호환되지 않습니다.")
        print("  기존 등록 사진으로 다시 등록해주세요.")
        print()

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(dir_path, ext)))
        images.extend(glob.glob(os.path.join(dir_path, ext.upper())))

    if not images:
        print(f"❌ 이미지를 찾을 수 없습니다: {dir_path}")
        sys.exit(1)

    print(f"📂 {len(images)}개 이미지에서 '{name}' 등록 시작...")
    count = 0
    for img_path in sorted(set(images)):
        try:
            recognizer.register(img_path, name)
            count += 1
            print(f"  ✅ {os.path.basename(img_path)}")
        except ValueError:
            print(f"  ❌ {os.path.basename(img_path)} (얼굴 미감지)")

    print(f"\n✅ 총 {count}/{len(images)}장 등록 완료: {name}")


def list_registered(config):
    """등록된 사용자 목록"""
    from modules.face_recognizer import FaceRecognizer

    try:
        recognizer = FaceRecognizer(config)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    registered = recognizer.list_registered()
    if not registered:
        print("등록된 얼굴이 없습니다.")
    else:
        print("📋 등록된 사용자:")
        for name, count in registered.items():
            print(f"  👤 {name} ({count}개 임베딩)")


def delete_user(name, config):
    """사용자 삭제"""
    from modules.face_recognizer import FaceRecognizer

    try:
        recognizer = FaceRecognizer(config)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    try:
        recognizer.delete(name)
        print(f"✅ 삭제 완료: {name}")
    except ValueError as e:
        print(f"❌ {e}")


def main():
    parser = argparse.ArgumentParser(
        description="🔒 Screen Watcher - 얼굴 등록 유틸리티",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python register_face.py 홍길동                   카메라로 촬영하여 등록
  python register_face.py 홍길동 --image face.jpg   이미지 파일로 등록
  python register_face.py 홍길동 --dir photos/      폴더의 모든 이미지로 등록
  python register_face.py --list                    등록된 사용자 목록
  python register_face.py --delete 홍길동            사용자 삭제
        """,
    )
    parser.add_argument("name", nargs="?", help="등록할 사용자 이름")
    parser.add_argument("--image", "-i", help="이미지 파일 경로")
    parser.add_argument("--dir", "-d", help="이미지 폴더 경로")
    parser.add_argument("--list", "-l", action="store_true", help="등록된 사용자 목록")
    parser.add_argument("--delete", metavar="NAME", help="사용자 삭제")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")

    args = parser.parse_args()

    # 로깅 설정 (최소)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_config(args.config)

    if args.list:
        list_registered(config)
    elif args.delete:
        delete_user(args.delete, config)
    elif args.name:
        camera_index = config.get("camera_index", 0)
        if args.dir:
            register_directory(args.name, args.dir, config)
        elif args.image:
            register_image(args.name, args.image, config)
        else:
            register_camera(args.name, config, camera_index)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
