from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel


def numpy_to_pixmap(gray_or_bgr) -> QPixmap:
    import numpy as np

    img = np.ascontiguousarray(gray_or_bgr)
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    else:
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class ImageCanvas(QLabel):
    """可缩放显示图片，支持拖拽框选 ROI，并叠加切字框。"""

    roi_changed = pyqtSignal(object)  # tuple[x,y,w,h] | None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 280)
        self.setStyleSheet("background:#1a1a1a; color:#888; border:1px solid #333;")
        self.setMouseTracking(True)
        self._src: QPixmap | None = None
        self._boxes: list[tuple[int, int, int, int]] = []
        self._roi: tuple[int, int, int, int] | None = None
        self._drag_origin: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._enable_roi = True
        self._draw_stored_roi = True
        self.setText("选择截图后在此预览；按住鼠标拖拽框选数字区域")

    def set_enable_roi(self, enabled: bool) -> None:
        self._enable_roi = enabled

    def clear_image(self) -> None:
        self._src = None
        self._boxes = []
        self._roi = None
        self._drag_origin = None
        self._drag_current = None
        self.setPixmap(QPixmap())
        self.setText("选择截图后在此预览；按住鼠标拖拽框选数字区域")

    def set_image_bgr_or_gray(
        self,
        arr,
        boxes: list[tuple[int, int, int, int]] | None = None,
        *,
        draw_stored_roi: bool = True,
    ) -> None:
        self._src = numpy_to_pixmap(arr)
        self._boxes = list(boxes or [])
        self._draw_stored_roi = draw_stored_roi
        self._repaint_canvas()

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self._roi = roi
        self._repaint_canvas()
        self.roi_changed.emit(roi)

    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def clear_roi(self) -> None:
        self.set_roi(None)

    def _image_rect_on_widget(self) -> QRect | None:
        if self._src is None or self._src.isNull():
            return None
        pw, ph = self.width(), self.height()
        iw, ih = self._src.width(), self._src.height()
        if iw <= 0 or ih <= 0:
            return None
        scale = min(pw / iw, ph / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        x = (pw - dw) // 2
        y = (ph - dh) // 2
        self._scale = scale
        self._offset = QPoint(x, y)
        return QRect(x, y, dw, dh)

    def _widget_to_image(self, pos: QPoint) -> QPoint | None:
        rect = self._image_rect_on_widget()
        if rect is None or self._scale <= 0:
            return None
        if not rect.contains(pos):
            # still map, clamp later
            pass
        ix = int((pos.x() - self._offset.x()) / self._scale)
        iy = int((pos.y() - self._offset.y()) / self._scale)
        return QPoint(ix, iy)

    def _clamp_roi(self, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int] | None:
        if self._src is None:
            return None
        w, h = self._src.width(), self._src.height()
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        xa = max(0, min(xa, w - 1))
        xb = max(0, min(xb, w))
        ya = max(0, min(ya, h - 1))
        yb = max(0, min(yb, h))
        if xb - xa < 2 or yb - ya < 2:
            return None
        return xa, ya, xb - xa, yb - ya

    def _repaint_canvas(self) -> None:
        if self._src is None or self._src.isNull():
            return
        base = self._src.copy()
        painter = QPainter(base)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        if self._roi and self._draw_stored_roi:
            x, y, w, h = self._roi
            pen = QPen(QColor(0, 200, 255), max(2, base.width() // 400))
            painter.setPen(pen)
            painter.setBrush(QColor(0, 180, 255, 40))
            painter.drawRect(x, y, w, h)

        if self._boxes:
            pen = QPen(QColor(50, 255, 120), max(1, base.width() // 500))
            painter.setPen(pen)
            for bx, by, bw, bh in self._boxes:
                painter.drawRect(bx, by, bw, bh)

        if self._drag_origin is not None and self._drag_current is not None and self._src is not None:
            p0 = self._widget_to_image(self._drag_origin)
            p1 = self._widget_to_image(self._drag_current)
            if p0 and p1:
                roi = self._clamp_roi(p0.x(), p0.y(), p1.x(), p1.y())
                if roi:
                    x, y, w, h = roi
                    pen = QPen(QColor(255, 200, 0), max(2, base.width() // 400), Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(QColor(255, 200, 0, 35))
                    painter.drawRect(x, y, w, h)

        painter.end()
        target = self._image_rect_on_widget()
        if target is None:
            self.setPixmap(base)
            return
        scaled = base.scaled(
            target.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._repaint_canvas()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self._enable_roi or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        if self._src is None:
            return
        self._drag_origin = event.position().toPoint()
        self._drag_current = self._drag_origin
        self._repaint_canvas()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is None:
            return super().mouseMoveEvent(event)
        self._drag_current = event.position().toPoint()
        self._repaint_canvas()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is None or event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        p0 = self._widget_to_image(self._drag_origin)
        p1 = self._widget_to_image(event.position().toPoint())
        self._drag_origin = None
        self._drag_current = None
        if p0 and p1:
            roi = self._clamp_roi(p0.x(), p0.y(), p1.x(), p1.y())
            if roi:
                self.set_roi(roi)
            else:
                self._repaint_canvas()
        else:
            self._repaint_canvas()
