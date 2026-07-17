from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class RegionCaptureOverlay(QWidget):
    """全屏半透明遮罩，拖拽框选屏幕区域后截取。"""

    captured = pyqtSignal(object)  # QImage
    cancelled = pyqtSignal()

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

        # Cover virtual desktop across monitors
        screens = QGuiApplication.screens()
        if screens:
            geo = screens[0].geometry()
            for s in screens[1:]:
                geo = geo.united(s.geometry())
            self.setGeometry(geo)
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._bg = None

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Grab once as backdrop so user sees the screen content dimmed
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            # grab entire virtual desktop
            self._bg = screen.grabWindow(0, self.x(), self.y(), self.width(), self.height())
        self.raise_()
        self.activateWindow()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        if self._bg is not None and not self._bg.isNull():
            painter.drawPixmap(0, 0, self._bg)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        if self._origin and self._current:
            rect = QRect(self._origin, self._current).normalized()
            # clear dim in selection by redrawing bg
            if self._bg is not None and not self._bg.isNull():
                painter.drawPixmap(rect, self._bg, rect)
            pen = QPen(QColor(0, 200, 255), 2)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                rect.left(),
                max(16, rect.top() - 6),
                f"{rect.width()}×{rect.height()}  松开完成 · Esc 取消",
            )
        else:
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(40, 40, "拖拽框选要截取的区域 · Esc 取消")

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._origin is None:
            return
        rect = QRect(self._origin, event.position().toPoint()).normalized()
        self._origin = None
        self._current = None
        if rect.width() < 4 or rect.height() < 4:
            self.update()
            return
        global_top_left = self.mapToGlobal(rect.topLeft())
        screen = QGuiApplication.screenAt(global_top_left)
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.cancelled.emit()
            self.close()
            return
        geo = screen.geometry()
        local_x = global_top_left.x() - geo.x()
        local_y = global_top_left.y() - geo.y()
        grab = screen.grabWindow(0, local_x, local_y, rect.width(), rect.height())
        self.captured.emit(grab.toImage())
        self.close()
