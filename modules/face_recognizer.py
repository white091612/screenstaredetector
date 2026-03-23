"""
얼굴 인식/비교 모듈

InsightFace (ArcFace) 기반 얼굴 인코딩 및 등록된 사용자 비교.
512차원 normalized embedding + cosine similarity + 다수결 판정.
"""

import os
import pickle
import logging
import time

import cv2
import numpy as np

# ONNX 스레드 제한은 세션 생성 전에 환경 변수로 설정해야 함
# (FaceAnalysis.prepare()가 내부적으로 세션을 생성하므로 import 전에 설정)
import os as _os
_onnx_threads = _os.environ.get("ONNX_THREADS_OVERRIDE")  # 외부 override 가능

import onnxruntime
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """InsightFace 기반 얼굴 인코딩 사용자 인식"""

    def __init__(self, config):
        """
        Args:
            config: dict — config.yaml 설정 딕셔너리
        """
        self.face_data_dir = config.get("face_data_dir", "./registered_faces")
        self._threshold = config.get("face_recognition_threshold", 0.4)
        os.makedirs(self.face_data_dir, exist_ok=True)

        # ONNX Runtime 스레드 수 제한 (CPU 사용량 감소)
        # 환경 변수로 설정해야 FaceAnalysis 내부 세션 생성 시 적용됨
        onnx_threads = config.get("onnx_threads", 2)
        if _onnx_threads is not None:
            onnx_threads = int(_onnx_threads)
        _os.environ["OMP_NUM_THREADS"] = str(onnx_threads)
        _os.environ["OPENBLAS_NUM_THREADS"] = str(onnx_threads)
        _os.environ["MKL_NUM_THREADS"] = str(onnx_threads)

        onnxruntime.set_default_logger_severity(3)  # WARNING만

        # InsightFace 초기화
        model_name = config.get("insightface_model", "buffalo_l")
        det_size = config.get("insightface_det_size", 640)

        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(
            ctx_id=-1,
            det_size=(det_size, det_size),
            det_thresh=0.5,
        )
        logger.info(
            f"InsightFace 초기화 완료: {model_name}, det_size={det_size}, "
            f"onnx_threads={onnx_threads} (OMP_NUM_THREADS)"
        )

        # 등록 데이터
        self.known_embeddings = []          # list of np.ndarray (512,)
        self.known_names = []               # list of str
        self._known_matrix = np.empty((0, 512))  # (N, 512) 행렬
        self._load()

        # 캐시 TTL (config에서 조절 가능)
        self._cache_ttl = config.get("recognition_cache_ttl", 3.0)

        # bbox 레벨 캐시 (is_registered_user 용)
        self._cache = {}          # {bbox_key: (name_or_None, timestamp)}

        # 프레임 레벨 캐시 (identify_all_faces 용)
        self._frame_cache = None       # list of identity dicts
        self._frame_cache_time = 0

    @property
    def _encoding_file(self):
        return os.path.join(self.face_data_dir, "encodings.pkl")

    # ------------------------------------------------------------------
    # 데이터 관리
    # ------------------------------------------------------------------

    def _rebuild_matrix(self):
        """등록 임베딩을 (N, 512) numpy 행렬로 재구축. cosine similarity 일괄 계산에 사용."""
        if self.known_embeddings:
            self._known_matrix = np.array(self.known_embeddings)  # (N, 512)
        else:
            self._known_matrix = np.empty((0, 512))

    def _load(self):
        """저장된 임베딩 데이터를 로드합니다."""
        if os.path.exists(self._encoding_file):
            try:
                with open(self._encoding_file, "rb") as f:
                    data = pickle.load(f)

                saved_engine = data.get("engine", "dlib")
                embeddings = data.get("embeddings", data.get("encodings", []))

                # 엔진 불일치 감지 (dlib 128d ≠ insightface 512d)
                if saved_engine != "insightface" and embeddings:
                    logger.warning(
                        f"⚠ 저장된 인코딩({saved_engine})이 현재 엔진(insightface)과 "
                        f"호환되지 않습니다. 기존 인코딩을 무시합니다. "
                        f"'python register_face.py <이름> --dir <폴더>'로 재등록하세요."
                    )
                    self.known_embeddings = []
                    self.known_names = []
                else:
                    self.known_embeddings = embeddings
                    self.known_names = data.get("names", [])
                    unique_names = set(self.known_names)
                    logger.info(
                        f"등록된 얼굴 {len(self.known_embeddings)}개 로드 "
                        f"({len(unique_names)}명: {', '.join(unique_names)})"
                    )
            except Exception as e:
                logger.error(f"인코딩 파일 로드 실패: {e}")
                self.known_embeddings = []
                self.known_names = []
        else:
            logger.warning(
                "등록된 얼굴이 없습니다. "
                "'python register_face.py <이름>' 으로 얼굴을 등록하세요."
            )
        self._rebuild_matrix()

    def _save(self):
        """임베딩 데이터를 파일에 저장합니다."""
        with open(self._encoding_file, "wb") as f:
            pickle.dump(
                {
                    "embeddings": self.known_embeddings,
                    "names": self.known_names,
                    "engine": "insightface",
                    "dimensions": 512,
                },
                f,
            )

    # ------------------------------------------------------------------
    # 임베딩 추출
    # ------------------------------------------------------------------

    def _get_face_embedding(self, frame, face_bbox):
        """
        주어진 bbox 영역의 얼굴에 대해 InsightFace 임베딩을 추출합니다.

        Args:
            frame: BGR 이미지 (numpy array)
            face_bbox: (top, right, bottom, left) — gaze_estimator가 준 bbox

        Returns:
            np.ndarray (512,) — normalized embedding, or None
        """
        top, right, bottom, left = face_bbox
        h, w = frame.shape[:2]

        # bbox에 30% 여유분 추가 (InsightFace alignment에 충분한 컨텍스트 제공)
        pad_x = int((right - left) * 0.3)
        pad_y = int((bottom - top) * 0.3)
        crop_top = max(0, top - pad_y)
        crop_bottom = min(h, bottom + pad_y)
        crop_left = max(0, left - pad_x)
        crop_right = min(w, right + pad_x)

        crop = frame[crop_top:crop_bottom, crop_left:crop_right]

        if crop.size == 0:
            return None

        # InsightFace로 크롭 영역에서 얼굴 분석
        faces = self._app.get(crop)

        if not faces:
            # 크롭 실패 시 전체 프레임으로 재시도
            faces = self._app.get(frame)
            if not faces:
                return None
            # 전체 프레임에서 원본 bbox 중심에 가장 가까운 얼굴 선택
            bbox_center = np.array([(left + right) / 2, (top + bottom) / 2])
            best_face = min(
                faces,
                key=lambda f: np.linalg.norm(
                    np.array([(f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2])
                    - bbox_center
                ),
            )
            return best_face.normed_embedding

        if len(faces) == 1:
            return faces[0].normed_embedding

        # 크롭 내 여러 얼굴 감지 시 → 원본 bbox 중심에 가장 가까운 얼굴 선택
        bbox_center = np.array(
            [(left + right) / 2 - crop_left, (top + bottom) / 2 - crop_top]
        )
        best_face = min(
            faces,
            key=lambda f: np.linalg.norm(
                np.array([(f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2])
                - bbox_center
            ),
        )

        return best_face.normed_embedding  # (512,) normalized

    # ------------------------------------------------------------------
    # 다수결 판정 (내부 공통 로직)
    # ------------------------------------------------------------------

    def _majority_vote(self, embedding):
        """
        등록된 모든 임베딩과 cosine similarity 비교 후 다수결 판정.

        Args:
            embedding: np.ndarray (512,) — normalized embedding

        Returns:
            tuple: (name or None, similarity or 0.0)
        """
        if len(self.known_embeddings) == 0:
            return None, 0.0

        # normed_embedding이므로 내적 = cosine similarity
        similarities = self._known_matrix @ embedding  # (N,)

        # 사용자별 다수결 판정
        user_scores = {}  # {name: [sim1, sim2, ...]}
        for i, sim in enumerate(similarities):
            name = self.known_names[i]
            if name not in user_scores:
                user_scores[name] = []
            user_scores[name].append(float(sim))

        best_name = None
        best_avg_sim = -1.0

        for name, sims in user_scores.items():
            matched = [s for s in sims if s >= self._threshold]
            total = len(sims)
            # 다수결: 등록 인코딩의 50% 이상이 threshold를 넘어야 함
            if len(matched) >= max(1, total * 0.5):
                avg_sim = sum(matched) / len(matched)
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_name = name

        max_sim = float(np.max(similarities)) if len(similarities) > 0 else 0.0
        return best_name, best_avg_sim if best_name else max_sim

    # ------------------------------------------------------------------
    # 인식: 개별 얼굴
    # ------------------------------------------------------------------

    def _bbox_cache_key(self, bbox, grid=50):
        """bbox를 그리드로 양자화하여 캐시 키를 생성합니다."""
        t, r, b, l = bbox
        return (t // grid, r // grid, b // grid, l // grid)

    def is_registered_user(self, frame, face_bbox):
        """
        지정된 위치의 얼굴이 등록된 사용자인지 확인합니다.
        인터페이스 동일: (bool, str|None) 반환

        Args:
            frame: BGR 이미지 (numpy array)
            face_bbox: (top, right, bottom, left) 형식의 얼굴 위치

        Returns:
            tuple[bool, str | None]: (등록 여부, 사용자 이름)
        """
        if len(self.known_embeddings) == 0:
            return False, None

        # 1. 캐시 확인
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

        # 2. InsightFace로 임베딩 추출
        embedding = self._get_face_embedding(frame, face_bbox)
        if embedding is None:
            logger.debug("얼굴 임베딩을 추출할 수 없습니다.")
            return False, None

        # 3. 다수결 판정
        best_name, similarity = self._majority_vote(embedding)

        # 4. 결과 캐싱 및 반환
        self._cache[cache_key] = (best_name, now)

        if best_name is not None:
            logger.debug(f"등록된 사용자: {best_name} (유사도: {similarity:.3f})")
            return True, best_name

        logger.debug(
            f"미등록 사용자 (최대 유사도: {similarity:.3f}, "
            f"임계값: {self._threshold})"
        )
        return False, None

    # ------------------------------------------------------------------
    # 인식: 전체 프레임 (다중 인물)
    # ------------------------------------------------------------------

    def identify_all_faces(self, frame, force_fresh=False):
        """
        전체 프레임에서 모든 얼굴을 한번에 감지하고 신원을 확인합니다.
        개별 crop 대신 전체 프레임을 InsightFace에 전달하여
        작은 얼굴도 안정적으로 감지·인식합니다.

        프레임 레벨 캐시: cache_ttl 이내면 이전 결과 반환.
        force_fresh=True면 캐시를 무시하고 새로 분석.

        Args:
            frame: BGR 이미지 (numpy array)
            force_fresh: True면 캐시 무시 (확인 모드 등)

        Returns:
            list of dict: [{"bbox": (x1,y1,x2,y2), "name": str|None, "similarity": float}, ...]
        """
        now = time.time()
        if not force_fresh and self._frame_cache is not None and (now - self._frame_cache_time) < self._cache_ttl:
            return self._frame_cache

        faces = self._app.get(frame)
        results = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            embedding = face.normed_embedding

            name, similarity = self._majority_vote(embedding)

            results.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "name": name,
                "similarity": similarity,
            })

        self._frame_cache = results
        self._frame_cache_time = now
        return results

    # ------------------------------------------------------------------
    # 등록
    # ------------------------------------------------------------------

    def register(self, image_path, name):
        """
        이미지 파일에서 얼굴을 등록합니다.

        Args:
            image_path: 이미지 파일 경로
            name: 사용자 이름

        Raises:
            ValueError: 이미지에서 얼굴을 찾을 수 없을 때
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

        faces = self._app.get(image)

        if not faces:
            raise ValueError(f"이미지에서 얼굴을 찾을 수 없습니다: {image_path}")

        if len(faces) > 1:
            logger.warning(
                f"이미지에 {len(faces)}개의 얼굴 감지됨. "
                f"가장 큰 얼굴만 등록합니다."
            )
            faces.sort(
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )

        embedding = faces[0].normed_embedding  # (512,) normalized

        self.known_embeddings.append(embedding)
        self.known_names.append(name)
        self._rebuild_matrix()
        self._save()
        logger.info(
            f"얼굴 등록 완료: {name} (총 {len(self.known_embeddings)}개 임베딩)"
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
        faces = self._app.get(frame)

        if not faces:
            raise ValueError("프레임에서 얼굴을 찾을 수 없습니다.")

        if len(faces) > 1:
            logger.warning(
                f"프레임에 {len(faces)}개의 얼굴 감지됨. "
                f"가장 큰 얼굴만 등록합니다."
            )
            faces.sort(
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )

        embedding = faces[0].normed_embedding

        self.known_embeddings.append(embedding)
        self.known_names.append(name)
        self._rebuild_matrix()
        self._save()
        logger.info(f"얼굴 등록 완료: {name}")

    # ------------------------------------------------------------------
    # 관리
    # ------------------------------------------------------------------

    def list_registered(self):
        """
        등록된 사용자 목록을 반환합니다.

        Returns:
            dict[str, int]: {사용자 이름: 임베딩 수}
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
            del self.known_embeddings[idx]
            del self.known_names[idx]

        self._rebuild_matrix()
        self._save()
        logger.info(f"얼굴 삭제 완료: {name} ({len(indices)}개 임베딩 제거)")

    def close(self):
        """리소스 해제"""
        self._cache.clear()
        self._frame_cache = None
        self._app = None
