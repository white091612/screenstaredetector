"""
시선/머리 방향 추정 모듈

MediaPipe Face Mesh의 468개 랜드마크와 cv2.solvePnP를 활용하여
머리의 Yaw(좌우), Pitch(상하), Roll(기울기) 각도를 추정합니다.
"""

import cv2
import numpy as np
import logging

try:
    import mediapipe as mp
    _FaceMesh = mp.solutions.face_mesh.FaceMesh
except (AttributeError, ImportError):
    raise ImportError(
        "mediapipe 버전이 호환되지 않습니다.\n"
        "mp.solutions API가 필요합니다. 아래 명령으로 재설치하세요:\n"
        '  pip install "mediapipe>=0.10.9,<0.10.21"'
    )

logger = logging.getLogger(__name__)


class GazeEstimator:
    """MediaPipe Face Mesh 기반 머리 방향 추정"""

    # 머리 자세 추정용 3D 모델 포인트 (일반적인 얼굴 비율)
    MODEL_POINTS_3D = np.array(
        [
            (0.0, 0.0, 0.0),  # 코 끝 (landmark 1)
            (0.0, -330.0, -65.0),  # 턱 (landmark 199)
            (-225.0, 170.0, -135.0),  # 왼쪽 눈 외측 (landmark 33)
            (225.0, 170.0, -135.0),  # 오른쪽 눈 외측 (landmark 263)
            (-150.0, -150.0, -125.0),  # 왼쪽 입꼬리 (landmark 61)
            (150.0, -150.0, -125.0),  # 오른쪽 입꼬리 (landmark 291)
        ],
        dtype=np.float64,
    )

    # 대응하는 MediaPipe Face Mesh 랜드마크 인덱스
    LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

    def __init__(self, direction_threshold=15, max_faces=4, camera_offset_angle=0):
        """
        Args:
            direction_threshold: 방향 판단 임계값 (도). 이 값 이내면 '정면'으로 판단
            max_faces: 동시 감지할 최대 얼굴 수
            camera_offset_angle: 카메라가 사용자 정면에서 벗어난 각도 (도).
                양수 = 카메라가 오른쪽, 음수 = 왼쪽.
                예) 노트북이 큰 모니터 오른쪽 40cm → 약 30°
        """
        self.direction_threshold = direction_threshold
        self.camera_offset_angle = camera_offset_angle
        self.face_mesh = _FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        if camera_offset_angle != 0:
            logger.info(
                f"카메라 오프셋: {camera_offset_angle}° "
                f"({'오른쪽' if camera_offset_angle > 0 else '왼쪽'})"
            )
        logger.info(
            f"시선 추정기 초기화 (임계값: {direction_threshold}°, "
            f"오프셋: {camera_offset_angle}°, 최대 얼굴: {max_faces})"
        )

    def estimate(self, frame):
        """
        프레임에서 얼굴을 감지하고 머리 방향을 추정합니다.

        Args:
            frame: BGR 이미지 (numpy array)

        Returns:
            list[dict]: 각 얼굴 정보
                - bbox: (top, right, bottom, left) — face_recognition 호환 형식
                - yaw: 좌우 회전각 (도)
                - pitch: 상하 회전각 (도)
                - roll: 기울기 (도)
                - direction: 추정 방향 ("screen", "left", "right", "up", "down")
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # MediaPipe 성능 최적화
        results = self.face_mesh.process(rgb)

        faces = []
        if not results.multi_face_landmarks:
            return faces

        # 카메라 내부 파라미터 (근사값)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        for face_landmarks in results.multi_face_landmarks:
            # 2D 이미지 포인트 추출
            image_points = np.array(
                [
                    (
                        face_landmarks.landmark[idx].x * w,
                        face_landmarks.landmark[idx].y * h,
                    )
                    for idx in self.LANDMARK_IDS
                ],
                dtype=np.float64,
            )

            # PnP로 머리 자세 추정
            success, rotation_vec, _ = cv2.solvePnP(
                self.MODEL_POINTS_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                continue

            # 회전 벡터 → 회전 행렬 → 오일러 각도
            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch = angles[0]  # 상하 (X축 회전)
            yaw = angles[1]  # 좌우 (Y축 회전)
            roll = angles[2]  # 기울기 (Z축 회전)

            # RQDecomp3x3 각도 정규화
            # 정면을 보는데 pitch가 ±180° 부근으로 나오는 분해 오류 보정
            if pitch > 90:
                pitch = 180 - pitch
            elif pitch < -90:
                pitch = -180 - pitch

            # 바운딩 박스 계산 (전체 랜드마크 기반 + 여백)
            xs = [lm.x * w for lm in face_landmarks.landmark]
            ys = [lm.y * h for lm in face_landmarks.landmark]
            margin_x = (max(xs) - min(xs)) * 0.15
            margin_y = (max(ys) - min(ys)) * 0.15

            top = max(0, int(min(ys) - margin_y))
            right = min(w, int(max(xs) + margin_x))
            bottom = min(h, int(max(ys) + margin_y))
            left = max(0, int(min(xs) - margin_x))

            direction = self._classify_direction(yaw, pitch)

            faces.append(
                {
                    "bbox": (top, right, bottom, left),
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                    "direction": direction,
                    "looking_at": self._describe_target(yaw),
                }
            )

        return faces

    def _classify_direction(self, yaw, pitch):
        """
        오일러 각도로부터 시선 방향을 분류합니다.

        카메라가 비스듬한 위치에 있는 경우 (camera_offset_angle != 0),
        "screen" 판정 범위가 자동으로 확장됩니다.

        예) camera_offset_angle=30, threshold=15 일 때:
          - 노트북을 볼 때: yaw ≈  0° → screen (0° ± 15°)
          - 큰 모니터 볼 때: yaw ≈ -30° → screen (-30° ± 15°)
          - 실질적 screen 범위: yaw -45° ~ +15°

        Args:
            yaw: 좌우 회전각 (도, 카메라 기준)
            pitch: 상하 회전각 (도)

        Returns:
            str: "screen", "left", "right", "up", "down"
        """
        t = self.direction_threshold
        offset = self.camera_offset_angle

        # pitch가 큰 경우 먼저 상/하 판정
        if abs(pitch) > t and abs(pitch) >= abs(yaw):
            return "up" if pitch < 0 else "down"

        # === "screen" 판정 (듀얼 모니터 대응) ===
        # 카메라 오프셋이 있으면, 두 모니터를 잇는 yaw 범위를 커버
        #   노트북 방향: yaw ≈ 0°
        #   큰 모니터 방향: yaw ≈ -offset° (오프셋이 양수일 때)
        # 범위: (-offset - threshold) ~ (+threshold)
        if offset != 0:
            # 카메라가 오른쪽에 있으면 큰 모니터를 볼 때 yaw가 음수
            if offset > 0:
                yaw_min = -offset - t  # 큰 모니터 너머까지
                yaw_max = t            # 노트북 너머까지
            else:
                yaw_min = -t
                yaw_max = -offset + t
        else:
            # 카메라가 정면: 기존 로직
            yaw_min = -t
            yaw_max = t

        if yaw_min <= yaw <= yaw_max and abs(pitch) <= t:
            # 듀얼 모니터: 큰 모니터 vs 노트북 구분
            if offset != 0:
                if abs(yaw) <= t:
                    return "laptop"   # 카메라 정면 ≈ 노트북
                else:
                    return "screen"   # 오프셋 방향 ≈ 큰 모니터
            return "screen"

        # screen이 아닌 경우 좌/우 판정
        if yaw < yaw_min:
            return "left"
        elif yaw > yaw_max:
            return "right"

        # 나머지 (pitch가 임계값 부근)
        return "up" if pitch < 0 else "down"

    def _describe_target(self, yaw):
        """
        yaw 값으로 사용자가 보고 있을 대상을 추정합니다.
        (디버그/로그용 — OpenCV 호환을 위해 영문 반환)
        """
        offset = self.camera_offset_angle
        t = self.direction_threshold

        if offset == 0:
            return "Monitor" if abs(yaw) <= t else None

        # 큰 모니터 쪽 (yaw ≈ -offset)
        if abs(yaw - (-offset)) <= t:
            return "Main Monitor"
        # 노트북 쪽 (yaw ≈ 0)
        if abs(yaw) <= t:
            return "Laptop"
        # 두 모니터 사이
        if offset > 0 and -offset - t <= yaw <= t:
            return "Screen Area"
        if offset < 0 and -t <= yaw <= -offset + t:
            return "Screen Area"
        return None

    def close(self):
        """리소스 해제"""
        if self.face_mesh:
            self.face_mesh.close()
            logger.info("시선 추정기 종료")
