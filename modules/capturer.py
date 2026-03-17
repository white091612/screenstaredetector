"""
무소음 카메라/스크린 캡쳐 모듈

카메라 프레임과 선택적으로 스크린을 무소음으로 캡쳐하여
지정된 디렉토리에 저장합니다.
"""

import cv2
import numpy as np
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

    def _draw_timestamp(self, image, timestamp_str=None):
        """
        이미지에 타임스탬프를 오버레이합니다.

        Args:
            image: BGR numpy array
            timestamp_str: 표시할 문자열 (없으면 현재 시각)

        Returns:
            numpy array: 타임스탬프가 추가된 이미지
        """
        if timestamp_str is None:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        img = image.copy()
        h, w = img.shape[:2]

        # 텍스트 크기 계산
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, w / 1200)  # 해상도에 비례
        thickness = max(1, int(font_scale * 2))
        (tw, th), baseline = cv2.getTextSize(
            timestamp_str, font, font_scale, thickness
        )

        # 하단 좌측에 반투명 배경 + 흰색 텍스트
        padding = 8
        x, y = padding, h - padding
        # 배경 박스
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (x - padding, y - th - padding),
            (x + tw + padding, y + baseline + padding),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        # 텍스트
        cv2.putText(
            img, timestamp_str, (x, y),
            font, font_scale, (255, 255, 255), thickness,
        )
        return img

    def capture(self, frame):
        """
        카메라 프레임을 저장합니다 (무소음).
        이미지에 캡쳐 시각이 자동으로 표시됩니다.

        Args:
            frame: BGR 이미지 (numpy array)

        Returns:
            list[str]: 저장된 파일 경로 목록
        """
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        display_time = now.strftime("%Y-%m-%d %H:%M:%S")
        paths = []

        # 카메라 프레임에 타임스탬프 오버레이 후 저장
        stamped_frame = self._draw_timestamp(frame, display_time)
        cam_filename = f"cam_{timestamp}.jpg"
        cam_path = os.path.join(self.capture_dir, cam_filename)
        cv2.imwrite(cam_path, stamped_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(cam_path)
        logger.info(f"카메라 캡쳐 저장: {cam_path}")

        # 스크린 캡쳐 (선택적)
        if self.also_capture_screen and self._sct:
            try:
                screen_filename = f"screen_{timestamp}.png"
                screen_path = os.path.join(self.capture_dir, screen_filename)
                monitor = self._sct.monitors[0]  # 전체 화면
                screenshot = self._sct.grab(monitor)

                # mss → numpy → 타임스탬프 오버레이
                screen_img = np.array(screenshot)[:, :, :3]  # BGRA → BGR
                screen_img = self._draw_timestamp(screen_img, display_time)
                cv2.imwrite(screen_path, screen_img)
                paths.append(screen_path)
                logger.info(f"스크린 캡쳐 저장: {screen_path}")
            except Exception as e:
                logger.error(f"스크린 캡쳐 실패: {e}")

        return paths
