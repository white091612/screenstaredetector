"""
메인 모니터링 모듈

카메라, 시선 추정, 얼굴 인식, 캡쳐, 화면 잠금 모듈을 통합하여
실시간 화면 감시를 수행합니다.

동작 흐름:
1. 카메라에서 프레임 획득
2. MediaPipe로 얼굴 감지 + 머리 방향 추정
3. 지정 방향을 보는 얼굴에 대해 등록 사용자 여부 확인
4. 미등록 사용자가 지정 방향을 보면 → 타임스탬프 포함 캡쳐 + 화면 잠금
"""

import cv2
import time
import logging

from .camera import Camera
from .gaze_estimator import GazeEstimator
from .face_recognizer import FaceRecognizer
from .capturer import Capturer
from .screen_locker import ScreenLocker

logger = logging.getLogger(__name__)


class Monitor:
    """모든 모듈을 통합하는 메인 모니터링 클래스"""

    def __init__(self, config):
        """
        Args:
            config: dict — config.yaml에서 로드한 설정 딕셔너리
        """
        self.config = config

        # --- 컴포넌트 초기화 ---
        self.camera = Camera(
            camera_index=config.get("camera_index", 0),
            width=config.get("camera_width", 640),
            height=config.get("camera_height", 480),
        )
        self.gaze_estimator = GazeEstimator(
            direction_threshold=config.get("direction_threshold", 15),
            max_faces=config.get("max_faces", 4),
            camera_offset_angle=config.get("camera_offset_angle", 0),
        )
        self.face_recognizer = FaceRecognizer(config)
        self.capturer = Capturer(
            capture_dir=config.get("capture_dir", "./captures"),
            also_capture_screen=config.get("also_capture_screen", False),
        )
        self.screen_locker = ScreenLocker(
            enabled=config.get("lock_screen_on_unknown", True),
            cooldown=config.get("lock_cooldown", 30),
        )

        # --- 설정 ---
        td = config.get("target_direction", "screen")
        if isinstance(td, str):
            self.target_directions = {d.strip() for d in td.split(",")}
        elif isinstance(td, list):
            self.target_directions = set(td)
        else:
            self.target_directions = {"screen"}
        # "screen" 은 항상 "laptop" 도 포함 (듀얼 모니터 지원)
        if "screen" in self.target_directions:
            self.target_directions.add("laptop")
        self.capture_interval = config.get("capture_interval", 10)
        self.process_interval = config.get("process_interval", 0.5)
        self.show_preview = config.get("show_preview", False)

        # --- 상태 ---
        self._running = False
        self._last_capture_time = 0
        self._unknown_count = 0  # 미등록 사용자 감지 누적 횟수

        # 등록된 얼굴이 없으면 경고
        if not self.face_recognizer.known_embeddings:
            logger.warning(
                "⚠ 등록된 얼굴이 없습니다! "
                "모든 감지된 얼굴이 미등록으로 처리됩니다."
            )

    def start(self):
        """
        모니터링을 시작합니다.
        메인 스레드에서 실행되며, Ctrl+C로 중단할 수 있습니다.
        """
        offset = self.config.get("camera_offset_angle", 0)
        logger.info("=" * 55)
        logger.info("  🔒 Screen Watcher 모니터링 시작")
        logger.info(f"  📡 감시 방향  : {', '.join(sorted(self.target_directions))}")
        if offset != 0:
            logger.info(f"  📐 카메라 위치 : 정면에서 {'오른쪽' if offset > 0 else '왼쪽'} {abs(offset)}°")
            t = self.config.get("direction_threshold", 15)
            if offset > 0:
                logger.info(f"  🖥 화면 판정  : yaw {-offset - t}° ~ +{t}° (큰모니터+노트북)")
            else:
                logger.info(f"  🖥 화면 판정  : yaw -{t}° ~ {-offset + t}° (큰모니터+노트북)")
        logger.info(f"  ⏱ 캡쳐 간격  : {self.capture_interval}초")
        lock_enabled = self.config.get("lock_screen_on_unknown", True)
        lock_cooldown = self.config.get("lock_cooldown", 30)
        logger.info(f"  🔒 화면 잠금  : {'활성' if lock_enabled else '비활성'} (쿨다운: {lock_cooldown}초)")
        logger.info(f"  🖥 미리보기   : {'활성' if self.show_preview else '비활성'}")
        logger.info(f"  👤 등록 사용자 : {len(self.face_recognizer.known_embeddings)}개 임베딩")
        logger.info("=" * 55)

        self._running = True
        self.camera.start()

        try:
            while self._running:
                self._process_frame()
                time.sleep(self.process_interval)
        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단됨 (Ctrl+C)")
        finally:
            self.stop()

    def _match_identity(self, gaze_bbox, all_face_identities):
        """
        gaze_estimator의 bbox (top,right,bottom,left)와
        InsightFace의 bbox (x1,y1,x2,y2)를 IoU로 매칭합니다.

        Returns:
            dict or None: {"bbox": ..., "name": ..., "similarity": ...}
        """
        if not all_face_identities:
            return None

        top, right, bottom, left = gaze_bbox
        best_match = None
        best_iou = 0.3  # 최소 IoU 임계값

        for identity in all_face_identities:
            x1, y1, x2, y2 = identity["bbox"]

            # IoU 계산
            inter_left = max(left, x1)
            inter_top = max(top, y1)
            inter_right = min(right, x2)
            inter_bottom = min(bottom, y2)

            if inter_right <= inter_left or inter_bottom <= inter_top:
                continue

            inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
            area1 = (right - left) * (bottom - top)
            area2 = (x2 - x1) * (y2 - y1)
            union_area = area1 + area2 - inter_area

            iou = inter_area / union_area if union_area > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_match = identity

        return best_match

    def _process_frame(self):
        """한 프레임을 처리합니다."""
        frame = self.camera.get_frame()
        if frame is None:
            return

        display_frame = frame.copy() if self.show_preview else None

        # 1단계: 시선 추정 (모든 얼굴의 방향 파악)
        faces = self.gaze_estimator.estimate(frame)

        # 2단계: InsightFace로 전체 프레임 한번에 분석 (모든 얼굴 임베딩 추출)
        all_face_identities = self.face_recognizer.identify_all_faces(frame)

        # 3단계: gaze_estimator bbox와 InsightFace bbox 매칭
        has_registered_looking = False   # 화면 보는 등록자 있는지
        has_unknown_looking = False      # 화면 보는 미등록자 있는지
        has_registered_present = False   # 프레임에 등록자 존재하는지 (방향 무관)

        for face_info in faces:
            bbox = face_info["bbox"]
            yaw = face_info["yaw"]
            pitch = face_info["pitch"]
            direction = face_info["direction"]
            top, right, bottom, left = bbox

            looking_at = face_info.get("looking_at", "")

            # InsightFace 결과에서 가장 가까운 bbox 매칭
            identity = self._match_identity(bbox, all_face_identities)
            is_registered = identity is not None and identity["name"] is not None
            name = identity["name"] if identity else None

            if is_registered:
                has_registered_present = True  # 등록자가 프레임에 존재

            # --- 디버그 미리보기: 기본 정보 ---
            if self.show_preview:
                color = (255, 255, 0)  # 노란색 (기본)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 1)
                info_text = f"{direction} Y:{yaw:.0f} P:{pitch:.0f}"
                if looking_at:
                    info_text += f" [{looking_at}]"
                cv2.putText(
                    display_frame,
                    info_text,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

            # 지정 방향을 보고있는지 확인
            if direction not in self.target_directions:
                continue

            # --- 디버그 미리보기: 인식 결과 ---
            if self.show_preview:
                if is_registered:
                    color = (0, 255, 0)  # 초록 (등록 사용자)
                    label = f"[OK] {name}"
                else:
                    color = (0, 0, 255)  # 빨강 (미등록)
                    label = "[!] UNKNOWN"
                cv2.rectangle(
                    display_frame, (left, top), (right, bottom), color, 2
                )
                cv2.putText(
                    display_frame,
                    label,
                    (left, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            if is_registered:
                has_registered_looking = True
            else:
                has_unknown_looking = True

        # 4단계: 캡쳐/잠금 판정
        # 미등록자가 화면을 보고 있고, 등록자가 화면을 보고 있지 않을 때만 캡쳐
        if has_unknown_looking and not has_registered_looking:
            self._handle_unknown_face(frame)

        # --- 디버그 미리보기 표시 ---
        if self.show_preview and display_frame is not None:
            # 상태 표시줄
            status_items = [
                f"Watch: {','.join(sorted(self.target_directions))}",
                f"Faces: {len(faces)}",
                f"Captures: {self._unknown_count}",
            ]
            status = " | ".join(status_items)
            cv2.putText(
                display_frame,
                status,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
            )
            cv2.imshow("Screen Watcher [DEBUG] - ESC to quit", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC 또는 Q로 종료
                self._running = False

    def _handle_unknown_face(self, frame):
        """미등록 사용자 감지를 처리합니다: 캡쳐 + 화면 잠금"""
        current_time = time.time()
        elapsed = current_time - self._last_capture_time

        if elapsed >= self.capture_interval:
            # 1) 타임스탬프 포함 스크린샷/카메라 캡쳐
            paths = self.capturer.capture(frame)
            self._last_capture_time = current_time
            self._unknown_count += 1
            logger.warning(
                f"⚠ 미등록 사용자 감지! "
                f"(#{self._unknown_count}) 캡쳐 저장: {paths}"
            )

            # 2) 화면 잠금 (Win+L 동작)
            self.screen_locker.lock()

    def stop(self):
        """모니터링을 중지하고 리소스를 해제합니다."""
        self._running = False
        self.camera.stop()
        self.gaze_estimator.close()
        self.face_recognizer.close()
        if self.show_preview:
            cv2.destroyAllWindows()
        logger.info(
            f"Screen Watcher 종료 "
            f"(총 캡쳐: {self._unknown_count}회)"
        )
