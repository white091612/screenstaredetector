"""
화면 잠금 모듈

미등록 사용자가 화면을 쳐다볼 때 OS의 화면 잠금(Win+L)을 실행합니다.
Windows / Linux / macOS를 모두 지원합니다.
"""

import platform
import subprocess
import logging
import time

logger = logging.getLogger(__name__)


class ScreenLocker:
    """OS별 화면 잠금"""

    def __init__(self, enabled=True, cooldown=30):
        """
        Args:
            enabled: 화면 잠금 기능 활성화 여부
            cooldown: 잠금 후 재잠금까지 최소 대기 시간(초).
                      잠금 해제 직후 다시 잠기는 것을 방지합니다.
        """
        self.enabled = enabled
        self.cooldown = cooldown
        self._last_lock_time = 0
        self._system = platform.system()

        if enabled:
            logger.info(
                f"화면 잠금 활성화 (OS: {self._system}, "
                f"쿨다운: {cooldown}초)"
            )

    def lock(self):
        """
        화면을 잠급니다.

        Returns:
            bool: 잠금 성공 여부
        """
        if not self.enabled:
            logger.debug("화면 잠금이 비활성화되어 있습니다.")
            return False

        # 쿨다운 체크
        now = time.time()
        elapsed = now - self._last_lock_time
        if elapsed < self.cooldown:
            remaining = self.cooldown - elapsed
            logger.debug(
                f"화면 잠금 쿨다운 중 (남은 시간: {remaining:.0f}초)"
            )
            return False

        try:
            if self._system == "Windows":
                self._lock_windows()
            elif self._system == "Linux":
                self._lock_linux()
            elif self._system == "Darwin":
                self._lock_macos()
            else:
                logger.error(f"지원하지 않는 OS: {self._system}")
                return False

            self._last_lock_time = time.time()
            logger.warning("🔒 화면 잠금 실행됨!")
            return True

        except Exception as e:
            logger.error(f"화면 잠금 실패: {e}")
            return False

    def _lock_windows(self):
        """Windows: rundll32로 WorkStation 잠금 (Win+L과 동일)"""
        import ctypes
        ctypes.windll.user32.LockWorkStation()

    def _lock_linux(self):
        """Linux: 데스크톱 환경별 잠금 시도"""
        # GNOME / Unity
        try:
            subprocess.run(
                ["gnome-screensaver-command", "-l"],
                check=True,
                timeout=5,
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # GNOME 최신 (loginctl)
        try:
            subprocess.run(
                ["loginctl", "lock-session"],
                check=True,
                timeout=5,
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # KDE
        try:
            subprocess.run(
                ["qdbus", "org.freedesktop.ScreenSaver",
                 "/ScreenSaver", "Lock"],
                check=True,
                timeout=5,
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # xdg-screensaver (범용)
        try:
            subprocess.run(
                ["xdg-screensaver", "lock"],
                check=True,
                timeout=5,
            )
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        raise RuntimeError(
            "화면 잠금 명령을 찾을 수 없습니다. "
            "gnome-screensaver, loginctl, qdbus, xdg-screensaver 중 "
            "하나가 필요합니다."
        )

    def _lock_macos(self):
        """macOS: pmset으로 디스플레이 슬립 → 잠금"""
        subprocess.run(
            ["/System/Library/CoreServices/Menu Extras/"
             "User.menu/Contents/Resources/CGSession", "-suspend"],
            check=True,
            timeout=5,
        )
