from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QLabel, QSizePolicy

# hit zones for resize
_HANDLE = ("move", "nw", "n", "ne", "e", "se", "s", "sw", "w")
_TIP_H = 22  # 顶部提示条高度，图片适配时避开，避免挡住截图顶边


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
    selection_changed = pyqtSignal(int)  # selected box index, -1 if none

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # 禁止 QLabel 用 pixmap 尺寸锁死高度（否则加控件后截图顶部被裁切）
        self.setScaledContents(False)
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
        self._press_pos: QPoint | None = None
        self._pending_blank = False  # 左键空白按下，等待长按平移或拖出框
        self._long_press_timer = QTimer(self)
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.timeout.connect(self._on_long_press_pan)
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
        self._hint_roi = "区域模式：拖蓝框框住数字；长按拖移画面；滚轮缩放"
        self._hint_char = "手动切字：拖新框；点选改大小；长按拖移画面；滚轮缩放"
        self.setText(self._hint_roi)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(640, 400)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(320, 200)

    def _viewport_rect(self) -> QRect:
        """去掉边框与顶部提示条后的可用绘图区。"""
        r = self.contentsRect()
        return QRect(r.x(), r.y() + _TIP_H, r.width(), max(1, r.height() - _TIP_H))

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
        self._selected_box = -1
        self._predictions = []
        self._repaint_canvas()
        self.boxes_changed.emit(list(self._boxes))
        self.selection_changed.emit(-1)

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

    def select_box(self, index: int) -> None:
        if 0 <= index < len(self._boxes):
            self._selected_box = index
        else:
            self._selected_box = -1
        self._repaint_canvas()
        self.selection_changed.emit(self._selected_box)

    def selected_box(self) -> tuple[int, int, int, int] | None:
        i = self._selected_box
        if 0 <= i < len(self._boxes):
            return self._boxes[i]
        return None

    def set_selected_box_size(self, *, width: int | None = None, height: int | None = None) -> bool:
        """用数值精确改选中框宽/高（中心大致保持）。"""
        i = self._selected_box
        if i < 0 or i >= len(self._boxes):
            return False
        x, y, w, h = self._boxes[i]
        nw = max(2, int(width)) if width is not None else w
        nh = max(2, int(height)) if height is not None else h
        # 尽量保持中心
        cx, cy = x + w / 2, y + h / 2
        nx = int(round(cx - nw / 2))
        ny = int(round(cy - nh / 2))
        box = self._clamp_roi(nx, ny, nx + nw, ny + nh)
        if not box:
            return False
        self._boxes[i] = box
        self._predictions = []
        self._repaint_canvas()
        self.boxes_changed.emit(list(self._boxes))
        self.selection_changed.emit(self._selected_box)
        return True

    def nudge_selected_box(self, dx: int = 0, dy: int = 0, dw: int = 0, dh: int = 0) -> bool:
        i = self._selected_box
        if i < 0 or i >= len(self._boxes):
            return False
        x, y, w, h = self._boxes[i]
        box = self._clamp_roi(x + dx, y + dy, x + w + dx + dw, y + h + dy + dh)
        if not box:
            return False
        self._boxes[i] = box
        self._predictions = []
        self._repaint_canvas()
        self.boxes_changed.emit(list(self._boxes))
        self.selection_changed.emit(self._selected_box)
        return True

    def split_selected_box_vertical(self) -> bool:
        """把选中字框左右拆成两个；未选中时自动拆最宽的框。"""
        i = self._selected_box
        if i < 0 or i >= len(self._boxes):
            # 自动挑最宽的粘连候选
            if not self._boxes:
                return False
            i = max(range(len(self._boxes)), key=lambda k: self._boxes[k][2])
            self._selected_box = i
        x, y, w, h = self._boxes[i]
        if w < 6:
            return False
        mid = max(2, w // 2)
        left = (x, y, mid, h)
        right = (x + mid, y, w - mid, h)
        self._boxes[i : i + 1] = [left, right]
        self._boxes.sort(key=lambda b: (b[0], b[1]))
        self._predictions = []
        # 选中拆出的左半，方便继续拆
        try:
            self._selected_box = self._boxes.index(left)
        except ValueError:
            self._selected_box = -1
        self._repaint_canvas()
        self.boxes_changed.emit(list(self._boxes))
        self.selection_changed.emit(self._selected_box)
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
        vr = self._viewport_rect()
        pw, ph = max(1, vr.width()), max(1, vr.height())
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
        base_ox = vr.x() + (pw - iw * scale) / 2
        base_oy = vr.y() + (ph - ih * scale) / 2
        self._pan = QPointF(vr.center().x() - cx * scale - base_ox, vr.center().y() - cy * scale - base_oy)
        self._repaint_canvas()
        self.view_changed.emit(self._user_zoom)

    def zoom_factor(self) -> float:
        return self._user_zoom

    def _fit_scale(self) -> float:
        if self._src is None or self._src.isNull():
            return 1.0
        iw, ih = self._src.width(), self._src.height()
        vr = self._viewport_rect()
        pw, ph = max(1, vr.width()), max(1, vr.height())
        return min(pw / iw, ph / ih)

    def _image_rect_on_widget(self) -> QRect | None:
        if self._src is None or self._src.isNull():
            return None
        iw, ih = self._src.width(), self._src.height()
        if iw <= 0 or ih <= 0:
            return None
        vr = self._viewport_rect()
        fit = self._fit_scale()
        scale = fit * self._user_zoom
        dw, dh = int(iw * scale), int(ih * scale)
        x = int(vr.x() + (vr.width() - dw) / 2 + self._pan.x())
        y = int(vr.y() + (vr.height() - dh) / 2 + self._pan.y())
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
        # 约 10 屏幕像素换算到图像坐标，并限制范围，避免放大后点不中
        return max(4, min(24, int(10 / max(self._scale, 0.05))))

    def _hit_box(self, img_pt: QPoint) -> tuple[int, str] | None:
        """返回 (index, kind)。手柄按屏幕像素判定，框内按图像坐标。"""
        # 先用屏幕坐标测手柄，放大后也好点
        widget_pt = self._image_to_widget(img_pt.x(), img_pt.y())
        hs = 10
        order = list(range(len(self._boxes)))
        if 0 <= self._selected_box < len(self._boxes):
            order = [self._selected_box] + [i for i in order if i != self._selected_box]
        for i in order:
            x, y, w, h = self._boxes[i]
            p0 = self._image_to_widget(x, y)
            p1 = self._image_to_widget(x + w, y + h)
            rect = QRect(p0, p1).normalized()
            pts = {
                "nw": rect.topLeft(),
                "ne": rect.topRight(),
                "sw": rect.bottomLeft(),
                "se": rect.bottomRight(),
                "n": QPoint(rect.center().x(), rect.top()),
                "s": QPoint(rect.center().x(), rect.bottom()),
                "w": QPoint(rect.left(), rect.center().y()),
                "e": QPoint(rect.right(), rect.center().y()),
            }
            for kind, pt in pts.items():
                if abs(widget_pt.x() - pt.x()) <= hs and abs(widget_pt.y() - pt.y()) <= hs:
                    return i, kind
            # 框体：图像坐标
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

    def _image_to_widget(self, ix: float, iy: float) -> QPoint:
        self._image_rect_on_widget()
        return QPoint(
            int(self._offset.x() + ix * self._scale),
            int(self._offset.y() + iy * self._scale),
        )

    def _repaint_canvas(self) -> None:
        if self._src is None or self._src.isNull():
            return
        # 底图不加框（避免放大后线糊掉）；框在屏幕坐标重绘，保证看清选中
        base = self._src.copy()
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
        qp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        qp.drawPixmap(target.topLeft(), scaled)

        # ROI in widget coords
        if self._roi and self._draw_stored_roi:
            x, y, w, h = self._roi
            p0 = self._image_to_widget(x, y)
            p1 = self._image_to_widget(x + w, y + h)
            pen = QPen(QColor(0, 200, 255), 2)
            qp.setPen(pen)
            qp.setBrush(QColor(0, 180, 255, 35))
            qp.drawRect(QRect(p0, p1).normalized())

        # boxes in widget coords — 粗线+填充，选中特别醒目
        for i, (bx, by, bw, bh) in enumerate(self._boxes):
            p0 = self._image_to_widget(bx, by)
            p1 = self._image_to_widget(bx + bw, by + bh)
            rect = QRect(p0, p1).normalized()
            selected = i == self._selected_box
            if selected:
                qp.setPen(QPen(QColor(255, 40, 40), 4))
                qp.setBrush(QColor(255, 220, 0, 90))
            else:
                qp.setPen(QPen(QColor(50, 255, 120), 2))
                qp.setBrush(QColor(50, 255, 120, 35))
            qp.drawRect(rect)
            # index badge
            badge = f"{i + 1}"
            font = qp.font()
            font.setBold(True)
            font.setPixelSize(14 if not selected else 16)
            qp.setFont(font)
            br = qp.fontMetrics().boundingRect(badge).adjusted(-4, -2, 4, 2)
            br.moveTopLeft(rect.topLeft() + QPoint(2, 2))
            qp.fillRect(br, QColor(0, 0, 0, 180) if not selected else QColor(220, 38, 38, 230))
            qp.setPen(QColor(255, 255, 255))
            qp.drawText(br, Qt.AlignmentFlag.AlignCenter, badge)
            if i < len(self._predictions):
                from game_digit_trainer.labels import display_label

                lab, conf = self._predictions[i]
                shown = display_label(lab)
                qp.setPen(QColor(255, 80, 80) if conf < 0.7 else QColor(255, 255, 0))
                font.setPixelSize(18 if selected else 14)
                qp.setFont(font)
                qp.drawText(rect.left() + 2, max(rect.top() - 4, 18), shown)
            if selected:
                # 大号拖拽手柄（屏幕像素）
                hs = 7
                qp.setBrush(QColor(255, 255, 255))
                qp.setPen(QPen(QColor(220, 38, 38), 2))
                for pt in (
                    rect.topLeft(),
                    rect.topRight(),
                    rect.bottomLeft(),
                    rect.bottomRight(),
                    QPoint(rect.center().x(), rect.top()),
                    QPoint(rect.center().x(), rect.bottom()),
                    QPoint(rect.left(), rect.center().y()),
                    QPoint(rect.right(), rect.center().y()),
                ):
                    qp.drawRect(pt.x() - hs, pt.y() - hs, hs * 2, hs * 2)
                # 选中说明条
                tag = f"选中第 {i + 1} 框  {bw}×{bh}px  拖白点改宽高"
                font.setPixelSize(13)
                qp.setFont(font)
                tr = qp.fontMetrics().boundingRect(tag).adjusted(-8, -4, 8, 4)
                tr.moveCenter(QPoint(rect.center().x(), rect.bottom() + 16))
                if tr.bottom() > canvas.height() - 4:
                    tr.moveBottom(rect.top() - 6)
                qp.fillRect(tr, QColor(220, 38, 38, 230))
                qp.setPen(QColor(255, 255, 255))
                qp.drawText(tr, Qt.AlignmentFlag.AlignCenter, tag)

        if self._creating and self._drag_origin is not None and self._drag_current is not None:
            p0 = self._widget_to_image(self._drag_origin)
            p1 = self._widget_to_image(self._drag_current)
            if p0 and p1:
                roi = self._clamp_roi(p0.x(), p0.y(), p1.x(), p1.y())
                if roi:
                    x, y, w, h = roi
                    a = self._image_to_widget(x, y)
                    b = self._image_to_widget(x + w, y + h)
                    color = (
                        QColor(80, 255, 120)
                        if self._mode == self.MODE_CHAR
                        else QColor(255, 200, 0)
                    )
                    qp.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
                    qp.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
                    qp.drawRect(QRect(a, b).normalized())

        tip = f"{self._user_zoom:.1f}x  滚轮缩放 · 长按/右键拖移 · 点绿框选中（红框=当前）"
        if 0 <= self._selected_box < len(self._boxes):
            bx, by, bw, bh = self._boxes[self._selected_box]
            tip += f" · 第{self._selected_box + 1}框 {bw}×{bh}"
        cr = self.contentsRect()
        tip_bg = QRect(cr.x(), cr.y(), cr.width(), _TIP_H)
        qp.fillRect(tip_bg, QColor(15, 23, 42, 230))
        qp.setPen(QColor(226, 232, 240))
        qp.drawText(cr.x() + 8, cr.y() + _TIP_H - 6, tip)
        qp.end()
        # 用与控件同尺寸的 pixmap + 左上对齐，避免 QLabel 居中裁切顶部
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
            iw = self._src.width()
            vr = self._viewport_rect()
            base_ox = vr.x() + (vr.width() - iw * scale) / 2
            base_oy = vr.y() + (vr.height() - self._src.height() * scale) / 2
            self._pan = QPointF(
                pos.x() - before.x() * scale - base_ox,
                pos.y() - before.y() * scale - base_oy,
            )
        self._repaint_canvas()
        self.view_changed.emit(self._user_zoom)
        event.accept()

    def _on_long_press_pan(self) -> None:
        """空白处左键长按 → 进入拖移画面。"""
        if not self._pending_blank:
            return
        self._pending_blank = False
        self._creating = False
        self._drag_origin = None
        self._drag_current = None
        self._panning = True
        self._pan_last = self._press_pos
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self._repaint_canvas()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._src is None:
            return super().mousePressEvent(event)
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._long_press_timer.stop()
            self._pending_blank = False
            self._panning = True
            self._pan_last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        img_pt = self._widget_to_image(event.position().toPoint())
        # 有绿框时：任意模式都可点选/拖改（整行蓝框模式也能选中再拆粘连）
        if self._boxes and img_pt is not None:
            hit = self._hit_box(img_pt)
            if hit is not None:
                self._long_press_timer.stop()
                self._pending_blank = False
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
                self.selection_changed.emit(self._selected_box)
                return

        # 点空白：先等长按（拖画面），若立刻拖动则画框
        if self._selected_box >= 0:
            self._selected_box = -1
            self.selection_changed.emit(-1)
        self._edit_idx = -1
        self._edit_kind = None
        self._creating = False
        self._drag_origin = None
        self._drag_current = None
        self._press_pos = event.position().toPoint()
        self._pending_blank = True
        self._long_press_timer.start(200)
        self._repaint_canvas()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning and self._pan_last is not None:
            cur = event.position().toPoint()
            delta = cur - self._pan_last
            self._pan = QPointF(self._pan.x() + delta.x(), self._pan.y() + delta.y())
            self._pan_last = cur
            self._repaint_canvas()
            return

        # 空白按下后：移动超过阈值 → 画框；未超过则继续等长按
        if self._pending_blank and self._press_pos is not None:
            cur = event.position().toPoint()
            if (cur - self._press_pos).manhattanLength() >= 8:
                self._long_press_timer.stop()
                self._pending_blank = False
                self._creating = True
                self._drag_origin = self._press_pos
                self._drag_current = cur
                self.setCursor(Qt.CursorShape.CrossCursor)
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

        # hover cursor：有绿框时始终可点选
        if self._boxes and img_pt is not None and not self._panning:
            hit = self._hit_box(img_pt)
            if hit:
                self.setCursor(self._cursor_for_kind(hit[1]))
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if not self._panning:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._long_press_timer.stop()
            self._pending_blank = False
            self._panning = False
            self._pan_last = None
            self._press_pos = None
            self.unsetCursor()
            self._repaint_canvas()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)

        self._long_press_timer.stop()
        if self._pending_blank:
            # 短按空白：仅取消选中，不画框
            self._pending_blank = False
            self._press_pos = None
            self._repaint_canvas()
            return

        if self._edit_idx >= 0:
            edited = self._boxes[self._edit_idx] if 0 <= self._edit_idx < len(self._boxes) else None
            self._boxes.sort(key=lambda b: (b[0], b[1]))
            if edited is not None:
                try:
                    self._selected_box = self._boxes.index(edited)
                except ValueError:
                    pass
            self._edit_idx = -1
            self._edit_kind = None
            self._edit_start = None
            self._edit_orig = None
            self._repaint_canvas()
            self.boxes_changed.emit(list(self._boxes))
            self.selection_changed.emit(self._selected_box)
            return

        if not self._creating or self._drag_origin is None:
            return
        p0 = self._widget_to_image(self._drag_origin)
        p1 = self._widget_to_image(event.position().toPoint())
        self._drag_origin = None
        self._drag_current = None
        self._creating = False
        self._press_pos = None
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
            self.selection_changed.emit(self._selected_box)
        else:
            self.set_roi(box)
