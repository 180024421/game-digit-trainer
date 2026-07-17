"""Windows 按窗口标题截取客户区（雷电等）。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from game_digit_trainer.capture import _timestamp_name, save_bgr


def list_window_titles(*, limit: int = 80) -> list[str]:
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    titles: list[str] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lp):  # noqa: ANN001
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if title:
            titles.append(title)
        return True

    user32.EnumWindows(_enum, 0)
    # unique preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return out


def find_hwnd_by_title(substr: str) -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    needle = substr.strip().lower()
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lp):  # noqa: ANN001
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if needle in buf.value.lower():
            found.append(int(hwnd))
            return False
        return True

    user32.EnumWindows(_enum, 0)
    return found[0] if found else None


def capture_window_by_title(dest_dir: Path, title_substr: str = "雷电") -> Path:
    """截取标题包含 substr 的窗口（优先 PyQt grabWindow）。"""
    hwnd = find_hwnd_by_title(title_substr)
    if not hwnd:
        raise RuntimeError(f"未找到标题含「{title_substr}」的窗口。可改关键词或先打开雷电。")

    from PyQt6.QtGui import QGuiApplication

    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("无可用屏幕")
    pix = screen.grabWindow(hwnd)
    if pix.isNull():
        raise RuntimeError("grabWindow 失败（窗口可能最小化或无权限）")
    from game_digit_trainer.capture import qimage_to_bgr

    bgr = qimage_to_bgr(pix.toImage())
    if bgr.size == 0:
        raise RuntimeError("截到空图")
    dest = dest_dir / _timestamp_name("win")
    return save_bgr(dest, bgr)


def capture_window_bgr(title_substr: str = "雷电") -> np.ndarray:
    path = Path(__import__("tempfile").gettempdir())
    p = capture_window_by_title(path, title_substr)
    raw = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    p.unlink(missing_ok=True)
    if img is None:
        raise RuntimeError("窗口截图解码失败")
    return img
