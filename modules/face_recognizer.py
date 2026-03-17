"""
얼굴 인식/비교 모듈

face_recognition 라이브러리를 이용하여 얼굴 인코딩을 생성하고
등록된 사용자와 비교합니다.
"""

import importlib.util
import os
import pickle
import logging
import time

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)


def ensure_face_recognition_ready():
    """face_recognition 및 모델 패키지 설치 상태를 확인합니다."""
    missing_packages = []

    if importlib.util.find_spec("face_recognition_models") is None:
        missing_packages.append("face_recognition_models")

    if importlib.util.find_spec("face_recognition") is None:
        missing_packages.append("face_recognition")

    if missing_packages:
        package_text = ", ".join(missing_packages)
        raise RuntimeError(
            "필수 얼굴 인식 패키지가 누락되었습니다: "
            f"{package_text}\n\n"
            "현재 사용 중인 Python에서 아래 명령을 실행하세요:\n"
            '  python -m pip install "setuptools>=65.0.0,<72.0.0" wheel\n'
            "  python -m pip install git+https://github.com/ageitgey/face_recognition_models\n"
            "  python -m pip install face_recognition\n\n"
            "중요: 반드시 현재 실행 중인 동일한 venv에서 `python -m pip` 형식으로 설치하세요."
        )

    if importlib.util.find_spec("pkg_resources") is None:
        raise RuntimeError(
            "`pkg_resources` 모듈이 없습니다.\n"
            "setuptools 72+ 버전에서는 pkg_resources가 제거되었습니다.\n"
            "face_recognition_models가 pkg_resources를 필요로 하므로 구버전을 설치해야 합니다.\n\n"
            "현재 venv에서 아래 명령을 실행하세요:\n"
            '  python -m pip install "setuptools>=65.0.0,<72.0.0"\n'
            "  python -m pip install --no-cache-dir --force-reinstall git+https://github.com/ageitgey/face_recognition_models\n"
            "  python -m pip install --upgrade face_recognition\n"
            '  python -c "import pkg_resources, face_recognition_models, face_recognition; print(\'ok\')"\n'
        )

    try:
        import pkg_resources
        import face_recognition_models
        import face_recognition
    except SystemExit as e:
        raise RuntimeError(
            "face_recognition 또는 face_recognition_models 로딩에 실패했습니다.\n\n"
            "현재 venv에서 아래 명령을 순서대로 다시 실행하세요:\n"
            "  python -m pip uninstall -y face-recognition-models face_recognition_models face_recognition\n"
            '  python -m pip install "setuptools>=65.0.0,<72.0.0" wheel\n'
            "  python -m pip install --no-cache-dir --force-reinstall git+https://github.com/ageitgey/face_recognition_models\n"
            "  python -m pip install --upgrade face_recognition\n"
        ) from e
    except ModuleNotFoundError as e:
        if e.name == "pkg_resources":
            raise RuntimeError(
                "`pkg_resources`를 찾을 수 없습니다.\n"
                "setuptools 72+ 에서 pkg_resources가 제거되었습니다. 구버전을 설치하세요:\n\n"
                '  python -m pip install "setuptools>=65.0.0,<72.0.0"\n'
                "  python -m pip install --no-cache-dir --force-reinstall git+https://github.com/ageitgey/face_recognition_models\n"
                "  python -m pip install --upgrade face_recognition\n"
            ) from e
        raise
    except Exception as e:
        raise RuntimeError(
            "face_recognition 초기화에 실패했습니다.\n"
            f"원인: {e}\n\n"
            "현재 venv에서 아래 명령을 실행하세요:\n"
            '  python -m pip install "setuptools>=65.0.0,<72.0.0" wheel\n'
            "  python -m pip install --no-cache-dir --force-reinstall git+https://github.com/ageitgey/face_recognition_models\n"
            "  python -m pip install --upgrade face_recognition\n"
        ) from e

    return face_recognition


class FaceRecognizer:
    """얼굴 인코딩 기반 사용자 인식"""

    def __init__(
        self, face_data_dir="./registered_faces", tolerance=0.6, model="small"
    ):
        """
        Args:
            face_data_dir: 등록된 얼굴 데이터 저장 디렉토리
            tolerance: 얼굴 비교 허용 오차 (0.0~1.0, 낮을수록 엄격)
            model: 인코딩 모델 ("small"=빠름, "large"=정확)
        """
        self.face_data_dir = face_data_dir
        self.tolerance = tolerance
        self.model = model
        self.face_recognition = ensure_face_recognition_ready()
        self.known_encodings = []
        self.known_names = []
        os.makedirs(face_data_dir, exist_ok=True)
        self._load()

        # MediaPipe Face Detection 인스턴스 재사용 (매번 생성하지 않음)
        self._mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # full-range: ~5m, ±60°
            min_detection_confidence=0.5,
        )

        # 인식 결과 캐시 (bbox 위치 기반)
        self._cache = {}          # {bbox_key: (name_or_None, timestamp)}
        self._cache_ttl = 2.0     # 캐시 유효 시간 (초)

    @property
    def _encoding_file(self):
        return os.path.join(self.face_data_dir, "encodings.pkl")

    def _load(self):
        """저장된 인코딩 데이터를 로드합니다."""
        if os.path.exists(self._encoding_file):
            try:
                with open(self._encoding_file, "rb") as f:
                    data = pickle.load(f)
                self.known_encodings = data.get("encodings", [])
                self.known_names = data.get("names", [])
                unique_names = set(self.known_names)
                logger.info(
                    f"등록된 얼굴 {len(self.known_encodings)}개 로드 "
                    f"({len(unique_names)}명: {', '.join(unique_names)})"
                )
            except Exception as e:
                logger.error(f"인코딩 파일 로드 실패: {e}")
                self.known_encodings = []
                self.known_names = []
        else:
            logger.warning(
                "등록된 얼굴이 없습니다. "
                "'python register_face.py <이름>' 으로 얼굴을 등록하세요."
            )

    def _save(self):
        """인코딩 데이터를 파일에 저장합니다."""
        with open(self._encoding_file, "wb") as f:
            pickle.dump(
                {
                    "encodings": self.known_encodings,
                    "names": self.known_names,
                },
                f,
            )

    # ------------------------------------------------------------------
    # 다단계 얼굴 감지 (옆모습 지원)
    # ------------------------------------------------------------------

    def _detect_faces_robust(self, rgb_image):
        """
        여러 방법으로 얼굴을 감지합니다. (옆모습 포함)

        감지 순서 (성능 최적화):
          1) MediaPipe Face Detection — 가장 빠르고 넓은 각도(±60°)
          2) HOG — 정면 보완
        CNN은 CPU에서 ~3초로 너무 느려 제외함.

        Returns:
            list of (top, right, bottom, left) — face_recognition 형식
        """
        # 1) MediaPipe (fastest, widest angle)
        locations = self._detect_faces_mediapipe(rgb_image)
        if locations:
            return locations

        # 2) HOG (frontal backup)
        locations = self.face_recognition.face_locations(rgb_image, model="hog")
        if locations:
            logger.debug("HOG로 얼굴 감지 성공")
            return locations

        return []

    def _detect_faces_mediapipe(self, rgb_image):
        """
        MediaPipe Face Detection으로 얼굴 위치를 감지합니다.
        클래스 레벨에서 인스턴스를 재사용하여 초기화 오버헤드를 제거합니다.

        Returns:
            list of (top, right, bottom, left) — face_recognition 형식
        """
        h, w = rgb_image.shape[:2]
        locations = []

        results = self._mp_face_detection.process(rgb_image)
        if not results.detections:
            return locations

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = bbox.xmin * w
            y = bbox.ymin * h
            bw = bbox.width * w
            bh = bbox.height * h

            # 바운딩 박스에 15% 여유분 추가 (인코딩 품질 향상)
            pad_x = bw * 0.15
            pad_y = bh * 0.15

            top = max(0, int(y - pad_y))
            right = min(w, int(x + bw + pad_x))
            bottom = min(h, int(y + bh + pad_y))
            left = max(0, int(x - pad_x))

            locations.append((top, right, bottom, left))

        return locations

    def _bbox_cache_key(self, bbox, grid=50):
        """
        bbox를 그리드로 양자화하여 캐시 키를 생성합니다.
        비슷한 위치의 얼굴을 같은 사람으로 간주하여 재인코딩을 방지합니다.
        """
        t, r, b, l = bbox
        return (t // grid, r // grid, b // grid, l // grid)

    def is_registered_user(self, frame, face_bbox):
        """
        지정된 위치의 얼굴이 등록된 사용자인지 확인합니다.
        위치 기반 캐시로 동일 얼굴의 반복 인코딩을 방지합니다.

        Args:
            frame: BGR 이미지 (numpy array)
            face_bbox: (top, right, bottom, left) 형식의 얼굴 위치

        Returns:
            tuple[bool, str | None]: (등록 여부, 사용자 이름)
        """
        if not self.known_encodings:
            return False, None

        # 캐시 확인: 비슷한 위치의 얼굴이 최근에 인식됐다면 재사용
        now = time.time()
        cache_key = self._bbox_cache_key(face_bbox)
        if cache_key in self._cache:
            cached_name, cached_time = self._cache[cache_key]
            if now - cached_time < self._cache_ttl:
                is_reg = cached_name is not None
                logger.debug(
                    f"캐시 사용: {'registered' if is_reg else 'unknown'} "
                    f"({cached_name})"
                )
                return is_reg, cached_name

        # face_recognition은 RGB 이미지를 사용
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            encodings = self.face_recognition.face_encodings(
                rgb_frame,
                known_face_locations=[face_bbox],
                model=self.model,
            )
        except Exception as e:
            logger.debug(f"얼굴 인코딩 실패: {e}")
            return False, None

        if not encodings:
            logger.debug("얼굴 인코딩을 생성할 수 없습니다.")
            return False, None

        encoding = encodings[0]
        distances = self.face_recognition.face_distance(self.known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = distances[best_idx]

        if best_distance <= self.tolerance:
            name = self.known_names[best_idx]
            logger.debug(f"등록된 사용자: {name} (거리: {best_distance:.3f})")
            self._cache[cache_key] = (name, now)
            return True, name

        logger.debug(
            f"미등록 사용자 (최소 거리: {best_distance:.3f}, "
            f"임계값: {self.tolerance})"
        )
        self._cache[cache_key] = (None, now)
        return False, None

    def register(self, image_path, name):
        """
        이미지 파일에서 얼굴을 등록합니다.
        HOG → CNN → MediaPipe 순서로 다단계 감지하여 옆모습도 지원합니다.

        Args:
            image_path: 이미지 파일 경로
            name: 사용자 이름

        Raises:
            ValueError: 이미지에서 얼굴을 찾을 수 없을 때
        """
        image = self.face_recognition.load_image_file(image_path)

        # 다단계 감지로 옆모습도 찾기
        locations = self._detect_faces_robust(image)

        if not locations:
            raise ValueError(f"이미지에서 얼굴을 찾을 수 없습니다: {image_path}")

        # 감지된 위치를 기반으로 인코딩 생성
        encodings = self.face_recognition.face_encodings(
            image, known_face_locations=locations, model="large"
        )

        if not encodings:
            raise ValueError(
                f"얼굴을 감지했지만 인코딩을 생성할 수 없습니다: {image_path}"
            )

        if len(encodings) > 1:
            logger.warning(
                f"이미지에 {len(encodings)}개의 얼굴 감지됨. "
                f"첫 번째 얼굴만 등록합니다."
            )

        self.known_encodings.append(encodings[0])
        self.known_names.append(name)
        self._save()
        logger.info(
            f"얼굴 등록 완료: {name} (총 {len(self.known_encodings)}개 인코딩)"
        )

    def register_from_frame(self, frame, name):
        """
        카메라 프레임에서 직접 얼굴을 등록합니다.
        HOG → CNN → MediaPipe 순서로 다단계 감지하여 옆모습도 지원합니다.

        Args:
            frame: BGR 이미지 (numpy array)
            name: 사용자 이름

        Raises:
            ValueError: 프레임에서 얼굴을 찾을 수 없을 때
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        locations = self._detect_faces_robust(rgb)

        if not locations:
            raise ValueError("프레임에서 얼굴을 찾을 수 없습니다.")

        encodings = self.face_recognition.face_encodings(
            rgb, known_face_locations=locations, model="large"
        )

        if not encodings:
            raise ValueError("얼굴을 감지했지만 인코딩을 생성할 수 없습니다.")

        if len(encodings) > 1:
            logger.warning(
                f"프레임에 {len(encodings)}개의 얼굴 감지됨. "
                f"첫 번째 얼굴만 등록합니다."
            )

        self.known_encodings.append(encodings[0])
        self.known_names.append(name)
        self._save()
        logger.info(f"얼굴 등록 완료: {name}")

    def list_registered(self):
        """
        등록된 사용자 목록을 반환합니다.

        Returns:
            dict[str, int]: {사용자 이름: 인코딩 수}
        """
        from collections import Counter

        return dict(Counter(self.known_names))

    def delete(self, name):
        """
        등록된 사용자를 삭제합니다.

        Args:
            name: 삭제할 사용자 이름

        Raises:
            ValueError: 등록되지 않은 사용자일 때
        """
        indices = [i for i, n in enumerate(self.known_names) if n == name]
        if not indices:
            raise ValueError(f"등록되지 않은 사용자: {name}")

        for idx in sorted(indices, reverse=True):
            del self.known_encodings[idx]
            del self.known_names[idx]

        self._save()
        logger.info(f"얼굴 삭제 완료: {name} ({len(indices)}개 인코딩 제거)")

    def close(self):
        """리소스 해제"""
        if self._mp_face_detection:
            self._mp_face_detection.close()
            self._mp_face_detection = None
        self._cache.clear()
