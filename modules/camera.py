"""
스레드 기반 카메라 모듈

백그라운드 스레드에서 카메라 프레임을 연속으로 읽어
메인 스레드에서 최신 프레임을 즉시 사용할 수 있게 합니다.
"""

import cv2
import threading
import logging
import time

logger = logging.getLogger(__name__)


class Camera:
    """스레드 기반 카메라 캡쳐 클래스"""

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """카메라를 열고 백그라운드 캡쳐를 시작합니다."""
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"카메라를 열 수 없습니다 (index: {self.camera_index}). "
                f"'ls /dev/video*' 로 사용 가능한 카메라를 확인하세요."
            )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"카메라 시작 (index: {self.camera_index}, "
            f"{self.width}x{self.height})"
        )

    def _capture_loop(self):
        """백그라운드에서 연속 프레임 캡쳐"""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                logger.warning("카메라 프레임 읽기 실패")
                time.sleep(0.1)

    def get_frame(self):
        """최신 프레임을 반환합니다. 프레임이 없으면 None."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def stop(self):
        """카메라를 중지하고 리소스를 해제합니다."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        logger.info("카메라 중지")
