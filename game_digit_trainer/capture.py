from __future__ import annotations

import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _timestamp_name(prefix: str = "capture") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png"


def save_bgr(path: Path, bgr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("编码截图失败")
    path.write_bytes(buf.tobytes())
    return path


def adb_available() -> bool:
    return shutil.which("adb") is not None


def list_adb_devices() -> list[str]:
    if not adb_available():
        return []
    try:
        out = subprocess.check_output(["adb", "devices"], text=True, timeout=8)
    except (subprocess.SubprocessError, OSError):
        return []
    devices: list[str] = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def capture_adb(dest_dir: Path, serial: str | None = None) -> Path:
    """通过 adb exec-out screencap 截取模拟器/真机画面。"""
    devices = list_adb_devices()
    if not devices:
        raise RuntimeError("未检测到 ADB 设备。请打开雷电并开启 ADB（或 adb connect）")
    use = serial or devices[0]
    cmd = ["adb"]
    if use:
        cmd += ["-s", use]
    cmd += ["exec-out", "screencap", "-p"]
    try:
        raw = subprocess.check_output(cmd, timeout=20)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ADB 截图失败: {exc}") from exc
    if not raw:
        raise RuntimeError("ADB 截图为空")
    # Some Windows adb paths corrupt newlines; try decode
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # fallback: write bytes then read
        tmp = Path(tempfile.gettempdir()) / _timestamp_name("adb_raw")
        tmp.write_bytes(raw.replace(b"\r\n", b"\n"))
        img = cv2.imread(str(tmp), cv2.IMREAD_COLOR)
        tmp.unlink(missing_ok=True)
    if img is None:
        raise RuntimeError("无法解码 ADB 截图")
    dest = dest_dir / _timestamp_name("adb")
    return save_bgr(dest, img)


def qimage_to_bgr(qimage) -> np.ndarray:
    """QImage -> BGR uint8."""
    from PyQt6.QtCore import QBuffer, QIODevice

    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    if not qimage.save(buf, "PNG"):
        raise RuntimeError("无法将截图编码为 PNG")
    data = bytes(buf.data())
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("无法解码截图")
    return img


def capture_clipboard_bgr():
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QGuiApplication.instance()
    if app is None:
        raise RuntimeError("无 Qt 应用实例")
    clip = QGuiApplication.clipboard()
    if clip is None:
        raise RuntimeError("无法访问剪贴板")
    qimg = clip.image()
    if qimg.isNull():
        mime = clip.mimeData()
        if mime and mime.hasUrls():
            for url in mime.urls():
                local = url.toLocalFile()
                if local:
                    data = np.fromfile(local, dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
        raise RuntimeError("剪贴板没有图片。可用 Win+Shift+S 截图后再点「粘贴」")
    return qimage_to_bgr(qimg)
