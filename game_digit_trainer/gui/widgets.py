from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QWheelEvent
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
    """画布：区域/手动切字 + 滚轮缩放 + 框选后自动放大。"""

    MODE_ROI = "roi"
    MODE_CHAR = "char"

    roi_changed = pyqtSignal(object)  # tuple[x,y,w,h] | None
    boxes_changed = pyqtSignal(list)
    view_changed = pyqtSignal(float)  # zoom factor relative to fit

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 280)
        self.setStyleSheet(
            "background:#0f172a; color:#94a3b8; border:1px solid #334155; border-radius:8px;"
        )
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._src: QPixmap | None = None
        self._boxes: list[tuple[int, int, int, int]] = []
        self._roi: tuple[int, int, int, int] | None = None
        self._drag_origin: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._mode = self.MODE_ROI
        self._draw_stored_roi = True
        self._user_zoom = 1.0  # 1=适应窗口
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_last: QPoint | None = None
        self._auto_zoom_on_roi = True
        self._hint_roi = "区域模式：拖蓝框框住数字 → 自动放大；滚轮继续缩放，右键拖动平移"
        self._hint_char = "手动切字：每个字拖一个绿框；滚轮放大，右键拖动画布"
        self.setText(self._hint_roi)

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        self._mode = mode if mode in (self.MODE_ROI, self.MODE_CHAR) else self.MODE_ROI
        if self._src is None or self._src.isNull():
            self.setText(self._hint_char if self._mode == self.MODE_CHAR else self._hint_roi)
        self._repaint_canvas()

    def set_auto_zoom_on_roi(self, enabled: bool) -> None:
        self._auto_zoom_on_roi = enabled

    def clear_image(self) -> None:
        self._src = None
        self._boxes = []
        self._roi = None
        self._drag_origin = None
        self._drag_current = None
        self.reset_view()
        self.setPixmap(QPixmap())
        self.setText(self._hint_char if self._mode == self.MODE_CHAR else self._hint_roi)

    def set_image_bgr_or_gray(
        self,
        arr,
        boxes: list[tuple[int, int, int, int]] | None = None,
        *,
        draw_stored_roi: bool = True,
        keep_boxes: bool = False,
        reset_view: bool = False,
    ) -> None:
        self._src = numpy_to_pixmap(arr)
        if boxes is not None and not keep_boxes:
            self._boxes = list(boxes)
        self._draw_stored_roi = draw_stored_roi
        if reset_view:
            self.reset_view()
        else:
            self._repaint_canvas()

    def set_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self._boxes = list(boxes)
        self._repaint_canvas()
        self.boxes_changed.emit(list(self._boxes))

    def boxes(self) -> list[tuple[int, int, int, int]]:
        return list(self._boxes)

    def clear_boxes(self) -> None:
        self.set_boxes([])

    def undo_box(self) -> None:
        if self._boxes:
            self._boxes.pop()
            self._repaint_canvas()
            self.boxes_changed.emit(list(self._boxes))

    def set_roi(self, roi: tuple[int, int, int, int] | None, *, auto_zoom: bool | None = None) -> None:
        self._roi = roi
        do_zoom = self._auto_zoom_on_roi if auto_zoom is None else auto_zoom
        if roi and do_zoom:
            self.zoom_to_rect(roi, margin=0.35)
        else:
            self._repaint_canvas()
        self.roi_changed.emit(roi)

    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def clear_roi(self) -> None:
        self._roi = None
        self.reset_view()
        self.roi_changed.emit(None)

    def reset_view(self) -> None:
        self._user_zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._repaint_canvas()
        self.view_changed.emit(self._user_zoom)

    def zoom_to_rect(self, rect: tuple[int, int, int, int], margin: float = 0.3) -> None:
        """把图像坐标矩形放大到视野中央。"""
        if self._src is None or self._src.isNull():
            return
        x, y, w, h = rect
        if w < 2 or h < 2:
            return
        iw, ih = self._src.width(), self._src.height()
        pw, ph = max(1, self.width()), max(1, self.height())
        fit = min(pw / iw, ph / ih)
        # target: rect (+margin) fills ~70% of view
        mx = int(w * margin)
        my = int(h * margin)
        rw = max(1, w + mx * 2)
        rh = max(1, h + my * 2)
        zoom_w = (pw * 0.85) / (rw * fit)
        zoom_h = (ph * 0.85) / (rh * fit)
        self._user_zoom = max(1.0, min(zoom_w, zoom_h, 12.0))
        scale = fit * self._user_zoom
        cx = x + w / 2
        cy = y + h / 2
        # center of widget should map to (cx, cy)
        # widget_x = offset_x + cx * scale  => offset_x = pw/2 - cx*scale
        # with pan: offset_x = (pw - iw*scale)/2 + pan_x
        base_ox = (pw - iw * scale) / 2
        base_oy = (ph - ih * scale) / 2
        self._pan = QPointF(pw / 2 - cx * scale - base_ox, ph / 2 - cy * scale - base_oy)
        self._repaint_canvas()
        self.view_changed.emit(self._user_zoom)

    def zoom_factor(self) -> float:
        return self._user_zoom

    def _fit_scale(self) -> float:
        if self._src is None or self._src.isNull():
            return 1.0
        iw, ih = self._src.width(), self._src.height()
        pw, ph = max(1, self.width()), max(1, self.height())
        return min(pw / iw, ph / ih)

    def _image_rect_on_widget(self) -> QRect | None:
        if self._src is None or self._src.isNull():
            return None
        iw, ih = self._src.width(), self._src.height()
        if iw <= 0 or ih <= 0:
            return None
        fit = self._fit_scale()
        scale = fit * self._user_zoom
        dw, dh = int(iw * scale), int(ih * scale)
        pw, ph = self.width(), self.height()
        x = int((pw - dw) / 2 + self._pan.x())
        y = int((ph - dh) / 2 + self._pan.y())
        self._scale = scale
        self._offset = QPoint(x, y)
        return QRect(x, y, dw, dh)

    def _widget_to_image(self, pos: QPoint) -> QPoint | None:
        if self._image_rect_on_widget() is None or self._scale <= 0:
            return None
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
            pen = QPen(QColor(50, 255, 120), max(2, base.width() // 450))
            painter.setPen(pen)
            for i, (bx, by, bw, bh) in enumerate(self._boxes):
                painter.drawRect(bx, by, bw, bh)
                painter.drawText(bx + 2, max(by + 12, 12), str(i + 1))

        if self._drag_origin is not None and self._drag_current is not None:
            p0 = self._widget_to_image(self._drag_origin)
            p1 = self._widget_to_image(self._drag_current)
            if p0 and p1:
                roi = self._clamp_roi(p0.x(), p0.y(), p1.x(), p1.y())
                if roi:
                    x, y, w, h = roi
                    color = (
                        QColor(80, 255, 120)
                        if self._mode == self.MODE_CHAR
                        else QColor(255, 200, 0)
                    )
                    pen = QPen(color, max(2, base.width() // 400), Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
                    painter.drawRect(x, y, w, h)

        painter.end()
        target = self._image_rect_on_widget()
        if target is None:
            self.setPixmap(base)
            return
        # Draw onto widget-sized pixmap so pan/zoom can go off-center
        canvas = QPixmap(self.size())
        canvas.fill(QColor("#0f172a"))
        scaled = base.scaled(
            target.size(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        qp = QPainter(canvas)
        qp.drawPixmap(target.topLeft(), scaled)
        # zoom badge
        qp.setPen(QColor(255, 255, 255))
        qp.drawText(10, 20, f"{self._user_zoom:.1f}x  滚轮缩放 · 右键拖动画布")
        qp.end()
        self.setPixmap(canvas)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._repaint_canvas()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._src is None or self._src.isNull():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.2 if delta > 0 else 1 / 1.2
        old_zoom = self._user_zoom
        new_zoom = max(1.0, min(old_zoom * factor, 16.0))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        # zoom toward cursor
        pos = event.position().toPoint()
        before = self._widget_to_image(pos)
        self._user_zoom = new_zoom
        self._image_rect_on_widget()
        if before is not None:
            # keep image point under cursor
            # pos.x = offset.x + before.x * scale
            scale = self._scale
            iw, ih = self._src.width(), self._src.height()
            pw, ph = self.width(), self.height()
            base_ox = (pw - iw * scale) / 2
            base_oy = (ph - ih * scale) / 2
            self._pan = QPointF(
                pos.x() - before.x() * scale - base_ox,
                pos.y() - before.y() * scale - base_oy,
            )
        self._repaint_canvas()
        self.view_changed.emit(self._user_zoom)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._src is None:
            return super().mousePressEvent(event)
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._pan_last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self._drag_origin = event.position().toPoint()
        self._drag_current = self._drag_origin
        self._repaint_canvas()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning and self._pan_last is not None:
            cur = event.position().toPoint()
            delta = cur - self._pan_last
            self._pan = QPointF(self._pan.x() + delta.x(), self._pan.y() + delta.y())
            self._pan_last = cur
            self._repaint_canvas()
            return
        if self._drag_origin is None:
            return super().mouseMoveEvent(event)
        self._drag_current = event.position().toPoint()
        self._repaint_canvas()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning and event.button() in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._panning = False
            self._pan_last = None
            self.unsetCursor()
            return
        if self._drag_origin is None or event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        p0 = self._widget_to_image(self._drag_origin)
        p1 = self._widget_to_image(event.position().toPoint())
        self._drag_origin = None
        self._drag_current = None
        if not (p0 and p1):
            self._repaint_canvas()
            return
        box = self._clamp_roi(p0.x(), p0.y(), p1.x(), p1.y())
        if not box:
            self._repaint_canvas()
            return
        if self._mode == self.MODE_CHAR:
            self._boxes.append(box)
            self._boxes.sort(key=lambda b: (b[0], b[1]))
            self._repaint_canvas()
            self.boxes_changed.emit(list(self._boxes))
        else:
            self.set_roi(box)  # triggers auto zoom
