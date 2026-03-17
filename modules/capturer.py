"""
무소음 카메라/스크린 캡쳐 모듈

카메라 프레임과 선택적으로 스크린을 무소음으로 캡쳐하여
지정된 디렉토리에 저장합니다.
"""

import cv2
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Capturer:
    """무소음 카메라/스크린 캡쳐"""

    def __init__(self, capture_dir="./captures", also_capture_screen=False):
        self.capture_dir = capture_dir
        self.also_capture_screen = also_capture_screen
        os.makedirs(capture_dir, exist_ok=True)

        self._sct = None
        if also_capture_screen:
            try:
                from mss import mss

                self._sct = mss()
                logger.info("스크린 캡쳐 활성화")
            except ImportError:
                logger.warning(
                    "mss 패키지가 설치되지 않아 스크린 캡쳐가 비활성화됩니다. "
                    "'pip install mss' 로 설치하세요."
                )
                self.also_capture_screen = False

    def capture(self, frame):
        """
        카메라 프레임을 저장합니다 (무소음).

        Args:
            frame: BGR 이미지 (numpy array)

        Returns:
            list[str]: 저장된 파일 경로 목록
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        paths = []

        # 카메라 프레임 저장
        cam_filename = f"cam_{timestamp}.jpg"
        cam_path = os.path.join(self.capture_dir, cam_filename)
        cv2.imwrite(cam_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(cam_path)
        logger.info(f"카메라 캡쳐 저장: {cam_path}")

        # 스크린 캡쳐 (선택적)
        if self.also_capture_screen and self._sct:
            try:
                screen_filename = f"screen_{timestamp}.png"
                screen_path = os.path.join(self.capture_dir, screen_filename)
                monitor = self._sct.monitors[0]  # 전체 화면
                screenshot = self._sct.grab(monitor)
                from mss.tools import to_png

                to_png(screenshot.rgb, screenshot.size, output=screen_path)
                paths.append(screen_path)
                logger.info(f"스크린 캡쳐 저장: {screen_path}")
            except Exception as e:
                logger.error(f"스크린 캡쳐 실패: {e}")

        return paths
