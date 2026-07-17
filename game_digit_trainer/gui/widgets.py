from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QLabel

# hit zones for resize
_HANDLE = ("move", "nw", "n", "ne", "e", "se", "s", "sw", "w")


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
    """画布：区域/手动切字 + 滚轮缩放 + 字框拖移/改大小。"""

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
        self._user_zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_last: QPoint | None = None
        self._auto_zoom_on_roi = True
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._selected_box = -1
        self._edit_idx = -1
        self._edit_kind: str | None = None
        self._edit_start: QPoint | None = None
        self._edit_orig: tuple[int, int, int, int] | None = None
        self._creating = False
        self._predictions: list[tuple[str, float]] = []
        self._hint_roi = "区域模式：拖蓝框框住数字 → 自动放大；滚轮缩放，右键平移"
        self._hint_char = "手动切字：拖新框；点选后可拖移/拖角改大小；滚轮缩放，右键平移"
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
        self._selected_box = -1
        self._edit_idx = -1
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
            self._selected_box = -1
        self._draw_stored_roi = draw_stored_roi
        if reset_view:
            self.reset_view()
        else:
            self._repaint_canvas()

    def set_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self._boxes = list(boxes)
        if self._selected_box >= len(self._boxes):
            self._selected_box = -1
        self._repaint_canvas()
        self.boxes_changed.emit(list(self._boxes))

    def boxes(self) -> list[tuple[int, int, int, int]]:
        return list(self._boxes)

    def clear_boxes(self) -> None:
        self._selected_box = -1
        self._predictions = []
        self.set_boxes([])

    def set_predictions(self, preds: list[tuple[str, float]] | None) -> None:
        """在绿框上方叠预测字符（预览识别）。"""
        self._predictions = list(preds or [])
        self._repaint_canvas()

    def predictions(self) -> list[tuple[str, float]]:
        return list(self._predictions)

    def selected_box_index(self) -> int:
        return self._selected_box

    def split_selected_box_vertical(self) -> bool:
        """把选中字框左右拆成两个（处理粘连）。"""
        i = self._selected_box
        if i < 0 or i >= len(self._boxes):
            return False
        x, y, w, h = self._boxes[i]
        if w < 6:
            return False
        mid = w // 2
        left = (x, y, mid, h)
        right = (x + mid, y, w - mid, h)
        self._boxes[i : i + 1] = [left, right]
        self._boxes.sort(key=lambda b: (b[0], b[1]))
        self._predictions = []
        self._selected_box = -1
        self._repaint_canvas()
        self.boxes_changed.emit(list(self._boxes))
        return True

    def undo_box(self) -> None:
        if self._boxes:
            self._boxes.pop()
            self._selected_box = min(self._selected_box, len(self._boxes) - 1)
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

    def zoom_to_image(self) -> None:
        """新图载入后适度放大，方便立刻框字。"""
        if self._src is None or self._src.isNull():
            return
        if self._roi:
            self.zoom_to_rect(self._roi, margin=0.35)
        else:
            self._user_zoom = 1.4
            self._pan = QPointF(0.0, 0.0)
            self._repaint_canvas()
            self.view_changed.emit(self._user_zoom)

    def zoom_to_rect(self, rect: tuple[int, int, int, int], margin: float = 0.3) -> None:
        if self._src is None or self._src.isNull():
            return
        x, y, w, h = rect
        if w < 2 or h < 2:
            return
        iw, ih = self._src.width(), self._src.height()
        pw, ph = max(1, self.width()), max(1, self.height())
        fit = min(pw / iw, ph / ih)
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

    def _handle_px(self) -> int:
        return max(8, int(10 / max(self._scale, 0.01)))

    def _hit_box(self, img_pt: QPoint) -> tuple[int, str] | None:
        """返回 (index, kind)。优先选中框的角点，再内部。"""
        hs = self._handle_px()
        order = list(range(len(self._boxes)))
        if 0 <= self._selected_box < len(self._boxes):
            order = [self._selected_box] + [i for i in order if i != self._selected_box]
        for i in order:
            x, y, w, h = self._boxes[i]
            pts = {
                "nw": (x, y),
                "ne": (x + w, y),
                "sw": (x, y + h),
                "se": (x + w, y + h),
                "n": (x + w // 2, y),
                "s": (x + w // 2, y + h),
                "w": (x, y + h // 2),
                "e": (x + w, y + h // 2),
            }
            for kind, (px, py) in pts.items():
                if abs(img_pt.x() - px) <= hs and abs(img_pt.y() - py) <= hs:
                    return i, kind
            if x <= img_pt.x() <= x + w and y <= img_pt.y() <= y + h:
                return i, "move"
        return None

    def _cursor_for_kind(self, kind: str | None) -> Qt.CursorShape:
        mapping = {
            "nw": Qt.CursorShape.SizeFDiagCursor,
            "se": Qt.CursorShape.SizeFDiagCursor,
            "ne": Qt.CursorShape.SizeBDiagCursor,
            "sw": Qt.CursorShape.SizeBDiagCursor,
            "n": Qt.CursorShape.SizeVerCursor,
            "s": Qt.CursorShape.SizeVerCursor,
            "e": Qt.CursorShape.SizeHorCursor,
            "w": Qt.CursorShape.SizeHorCursor,
            "move": Qt.CursorShape.SizeAllCursor,
        }
        return mapping.get(kind or "", Qt.CursorShape.CrossCursor)

    def _apply_edit(self, img_pt: QPoint) -> None:
        if self._edit_idx < 0 or not self._edit_orig or not self._edit_start or not self._edit_kind:
            return
        ox, oy, ow, oh = self._edit_orig
        dx = img_pt.x() - self._edit_start.x()
        dy = img_pt.y() - self._edit_start.y()
        x1, y1, x2, y2 = ox, oy, ox + ow, oy + oh
        kind = self._edit_kind
        if kind == "move":
            x1, y1, x2, y2 = ox + dx, oy + dy, ox + ow + dx, oy + oh + dy
        else:
            if "n" in kind:
                y1 = oy + dy
            if "s" in kind:
                y2 = oy + oh + dy
            if "w" in kind:
                x1 = ox + dx
            if "e" in kind:
                x2 = ox + ow + dx
        box = self._clamp_roi(x1, y1, x2, y2)
        if box:
            self._boxes[self._edit_idx] = box

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
            for i, (bx, by, bw, bh) in enumerate(self._boxes):
                selected = i == self._selected_box
                color = QColor(255, 220, 60) if selected else QColor(50, 255, 120)
                pen = QPen(color, max(2, base.width() // 450) + (1 if selected else 0))
                painter.setPen(pen)
                painter.drawRect(bx, by, bw, bh)
                painter.drawText(bx + 2, max(by + 12, 12), str(i + 1))
                if i < len(self._predictions):
                    lab, conf = self._predictions[i]
                    from game_digit_trainer.labels import display_label

                    shown = display_label(lab)
                    painter.setPen(QColor(255, 80, 80) if conf < 0.7 else QColor(255, 255, 80))
                    font = painter.font()
                    font.setBold(True)
                    font.setPixelSize(max(14, min(bw, bh) // 2))
                    painter.setFont(font)
                    painter.drawText(bx + 2, max(by - 4, font.pixelSize() + 2), f"{shown}")
                if selected:
                    hs = max(3, base.width() // 200)
                    painter.setBrush(QColor(255, 220, 60))
                    for px, py in (
                        (bx, by),
                        (bx + bw, by),
                        (bx, by + bh),
                        (bx + bw, by + bh),
                        (bx + bw // 2, by),
                        (bx + bw // 2, by + bh),
                        (bx, by + bh // 2),
                        (bx + bw, by + bh // 2),
                    ):
                        painter.drawRect(px - hs, py - hs, hs * 2, hs * 2)

        if self._creating and self._drag_origin is not None and self._drag_current is not None:
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
        canvas = QPixmap(self.size())
        canvas.fill(QColor("#0f172a"))
        scaled = base.scaled(
            target.size(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        qp = QPainter(canvas)
        qp.drawPixmap(target.topLeft(), scaled)
        qp.setPen(QColor(255, 255, 255))
        tip = f"{self._user_zoom:.1f}x  滚轮缩放 · 右键平移"
        if self._mode == self.MODE_CHAR:
            tip += " · 点选框可拖改"
        qp.drawText(10, 20, tip)
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
        pos = event.position().toPoint()
        before = self._widget_to_image(pos)
        self._user_zoom = new_zoom
        self._image_rect_on_widget()
        if before is not None and self._src is not None:
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

        img_pt = self._widget_to_image(event.position().toPoint())
        if self._mode == self.MODE_CHAR and img_pt is not None:
            hit = self._hit_box(img_pt)
            if hit is not None:
                idx, kind = hit
                self._selected_box = idx
                self._edit_idx = idx
                self._edit_kind = kind
                self._edit_start = img_pt
                self._edit_orig = self._boxes[idx]
                self._creating = False
                self._drag_origin = None
                self.setCursor(self._cursor_for_kind(kind))
                self._repaint_canvas()
                return

        self._creating = True
        self._edit_idx = -1
        self._edit_kind = None
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

        img_pt = self._widget_to_image(event.position().toPoint())
        if self._edit_idx >= 0 and img_pt is not None:
            self._apply_edit(img_pt)
            self._repaint_canvas()
            return

        if self._creating and self._drag_origin is not None:
            self._drag_current = event.position().toPoint()
            self._repaint_canvas()
            return

        # hover cursor
        if self._mode == self.MODE_CHAR and img_pt is not None and not self._panning:
            hit = self._hit_box(img_pt)
            if hit:
                self.setCursor(self._cursor_for_kind(hit[1]))
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning and event.button() in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._panning = False
            self._pan_last = None
            self.unsetCursor()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)

        if self._edit_idx >= 0:
            self._boxes.sort(key=lambda b: (b[0], b[1]))
            # keep selection on same box roughly
            self._edit_idx = -1
            self._edit_kind = None
            self._edit_start = None
            self._edit_orig = None
            self._repaint_canvas()
            self.boxes_changed.emit(list(self._boxes))
            return

        if not self._creating or self._drag_origin is None:
            return
        p0 = self._widget_to_image(self._drag_origin)
        p1 = self._widget_to_image(event.position().toPoint())
        self._drag_origin = None
        self._drag_current = None
        self._creating = False
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
            self._selected_box = self._boxes.index(box) if box in self._boxes else len(self._boxes) - 1
            self._repaint_canvas()
            self.boxes_changed.emit(list(self._boxes))
        else:
            self.set_roi(box)
