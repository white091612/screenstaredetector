"""
얼굴 인식/비교 모듈

face_recognition 라이브러리를 이용하여 얼굴 인코딩을 생성하고
등록된 사용자와 비교합니다.
"""

import cv2
import face_recognition
import numpy as np
import os
import pickle
import logging

logger = logging.getLogger(__name__)


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
        self.known_encodings = []
        self.known_names = []
        os.makedirs(face_data_dir, exist_ok=True)
        self._load()

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

    def is_registered_user(self, frame, face_bbox):
        """
        지정된 위치의 얼굴이 등록된 사용자인지 확인합니다.

        Args:
            frame: BGR 이미지 (numpy array)
            face_bbox: (top, right, bottom, left) 형식의 얼굴 위치

        Returns:
            tuple[bool, str | None]: (등록 여부, 사용자 이름)
        """
        if not self.known_encodings:
            return False, None

        # face_recognition은 RGB 이미지를 사용
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            encodings = face_recognition.face_encodings(
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
        distances = face_recognition.face_distance(self.known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = distances[best_idx]

        if best_distance <= self.tolerance:
            name = self.known_names[best_idx]
            logger.debug(f"등록된 사용자: {name} (거리: {best_distance:.3f})")
            return True, name

        logger.debug(
            f"미등록 사용자 (최소 거리: {best_distance:.3f}, "
            f"임계값: {self.tolerance})"
        )
        return False, None

    def register(self, image_path, name):
        """
        이미지 파일에서 얼굴을 등록합니다.

        Args:
            image_path: 이미지 파일 경로
            name: 사용자 이름

        Raises:
            ValueError: 이미지에서 얼굴을 찾을 수 없을 때
        """
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image, model="large")

        if not encodings:
            raise ValueError(f"이미지에서 얼굴을 찾을 수 없습니다: {image_path}")

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

        Args:
            frame: BGR 이미지 (numpy array)
            name: 사용자 이름

        Raises:
            ValueError: 프레임에서 얼굴을 찾을 수 없을 때
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, model="large")

        if not encodings:
            raise ValueError("프레임에서 얼굴을 찾을 수 없습니다.")

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
