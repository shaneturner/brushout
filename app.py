import argparse
import logging
import os
import subprocess
import sys
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPoint, QRectF, QSize, QTimer
from PySide6.QtGui import (
    QAction, QColor, QCursor, QImage, QKeySequence,
    QPainter, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QLabel, QMainWindow,
    QMessageBox, QSlider, QToolBar, QWidget,
)

from model import load_session
from inpaint import inpaint

MAX_UNDO = 10


def _default_dir() -> str:
    """Return Windows user home in WSL, empty string otherwise."""
    try:
        with open("/proc/version") as f:
            if "microsoft" not in f.read().lower():
                return ""
    except OSError:
        return ""
    win_home = os.environ.get("USERPROFILE", "")
    if win_home:
        result = subprocess.run(["wslpath", win_home], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    return "/mnt/c"


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    img_rgb = img.convert("RGB")
    data = img_rgb.tobytes("raw", "RGB")
    qimg = QImage(data, img_rgb.width, img_rgb.height, img_rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def qpixmap_to_pil(pixmap: QPixmap) -> Image.Image:
    qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    width, height = qimg.width(), qimg.height()
    ptr = qimg.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
    return Image.fromarray(arr.copy(), "RGB")


class Canvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self._image_item: QGraphicsPixmapItem | None = None
        self._mask_item: QGraphicsPixmapItem | None = None

        self._image_pil: Image.Image | None = None
        self._mask_arr: np.ndarray | None = None  # uint8 grayscale, 0/255

        self._undo_stack: list[tuple[Image.Image, np.ndarray]] = []

        self._brush_size = 30
        self._drawing = False
        self._erasing = False
        self._last_pos: QPoint | None = None
        self._panning = False
        self._pan_start: QPoint | None = None
        self._space_held = False

        self.setMinimumSize(600, 400)
        self.setBackgroundBrush(QColor(40, 40, 40))

    # ── Image management ─────────────────────────────────────────────────

    def load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        self._image_pil = img
        self._mask_arr = np.zeros((img.height, img.width), dtype=np.uint8)
        self._undo_stack.clear()
        self._rebuild_scene()
        # Defer fitInView so the scene has been laid out before we scale
        QTimer.singleShot(0, lambda: self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio))

    def _rebuild_scene(self):
        self._scene.clear()
        # clear() destroys C++ objects; drop Python refs so they aren't used again
        self._image_item = None
        self._mask_item = None
        if self._image_pil is None:
            return

        pixmap = pil_to_qpixmap(self._image_pil)
        self._image_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._image_item)

        self._refresh_mask_overlay()
        self._scene.setSceneRect(QRectF(pixmap.rect()))

    def _refresh_mask_overlay(self):
        if self._mask_arr is None or self._image_pil is None:
            return
        if self._mask_item is not None and self._mask_item.scene():
            self._scene.removeItem(self._mask_item)
        self._mask_item = None  # C++ object now removed; drop Python ref

        h, w = self._mask_arr.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[self._mask_arr > 0] = [255, 0, 0, 120]  # semi-transparent red

        # Use tobytes() so QImage owns the data and overlay can be freed safely
        qimg = QImage(overlay.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        self._mask_item = QGraphicsPixmapItem(pixmap)
        self._mask_item.setZValue(1)
        self._scene.addItem(self._mask_item)

    def _push_undo(self):
        if self._image_pil is None or self._mask_arr is None:
            return
        self._undo_stack.append((self._image_pil.copy(), self._mask_arr.copy()))
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)

    def undo(self):
        if not self._undo_stack:
            return
        self._image_pil, self._mask_arr = self._undo_stack.pop()
        self._rebuild_scene()

    def clear_mask(self):
        if self._image_pil is None:
            return
        self._mask_arr = np.zeros((self._image_pil.height, self._image_pil.width), dtype=np.uint8)
        self._refresh_mask_overlay()

    # ── Brush drawing ────────────────────────────────────────────────────

    def _scene_pos(self, event) -> QPoint:
        return self.mapToScene(event.position().toPoint()).toPoint()

    def _paint_circle(self, center: QPoint, erase: bool):
        if self._mask_arr is None:
            return
        r = self._brush_size // 2
        h, w = self._mask_arr.shape
        y, x = center.y(), center.x()
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        for py in range(y0, y1):
            for px in range(x0, x1):
                if (px - x) ** 2 + (py - y) ** 2 <= r * r:
                    self._mask_arr[py, px] = 0 if erase else 255

    def _stroke_to(self, pos: QPoint, erase: bool):
        if self._last_pos is None:
            self._paint_circle(pos, erase)
        else:
            # Interpolate between last and current position
            lp = self._last_pos
            steps = max(abs(pos.x() - lp.x()), abs(pos.y() - lp.y())) + 1
            for i in range(steps + 1):
                t = i / steps
                ix = int(lp.x() + t * (pos.x() - lp.x()))
                iy = int(lp.y() + t * (pos.y() - lp.y()))
                self._paint_circle(QPoint(ix, iy), erase)
        self._last_pos = pos
        self._refresh_mask_overlay()

    # ── Mouse events ─────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._image_pil is None:
            return super().mousePressEvent(event)

        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and self._space_held
        ):
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._erasing = False
            self._last_pos = None
            self._push_undo()
            self._stroke_to(self._scene_pos(event), erase=False)
        elif event.button() == Qt.MouseButton.RightButton:
            self._drawing = True
            self._erasing = True
            self._last_pos = None
            self._push_undo()
            self._stroke_to(self._scene_pos(event), erase=True)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self._drawing:
            self._stroke_to(self._scene_pos(event), erase=self._erasing)

    def mouseReleaseEvent(self, event):
        self._drawing = False
        self._erasing = False
        self._last_pos = None
        if self._panning:
            self._panning = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            if not self._panning:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        else:
            super().keyReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        cursor_pos = event.position().toPoint()
        scene_pos = self.mapToScene(cursor_pos)
        self.scale(factor, factor)
        new_vp = self.mapFromScene(scene_pos)
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() + new_vp.x() - cursor_pos.x()
        )
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() + new_vp.y() - cursor_pos.y()
        )

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def brush_size(self) -> int:
        return self._brush_size

    @brush_size.setter
    def brush_size(self, value: int):
        self._brush_size = value

    @property
    def has_image(self) -> bool:
        return self._image_pil is not None

    @property
    def has_mask(self) -> bool:
        return self._mask_arr is not None and self._mask_arr.any()

    def get_image(self) -> Image.Image | None:
        return self._image_pil

    def get_mask(self) -> Image.Image | None:
        if self._mask_arr is None:
            return None
        return Image.fromarray(self._mask_arr, "L")

    def set_image(self, img: Image.Image):
        self._push_undo()
        self._image_pil = img
        self._mask_arr = np.zeros((img.height, img.width), dtype=np.uint8)
        self._rebuild_scene()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Remover")
        self.resize(1100, 750)

        self._session = None
        self._canvas = Canvas(self)
        self.setCentralWidget(self._canvas)

        self._build_toolbar()
        self._status = QLabel("Open an image to get started.")
        self.statusBar().addWidget(self._status)

    def _build_toolbar(self):
        tb = QToolBar("Tools", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        open_act = QAction("Open Image", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._open_image)
        tb.addAction(open_act)

        save_act = QAction("Save", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self._save_image)
        tb.addAction(save_act)

        tb.addSeparator()

        undo_act = QAction("Undo", self)
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        undo_act.triggered.connect(self._canvas.undo)
        tb.addAction(undo_act)

        clear_act = QAction("Clear Mask", self)
        clear_act.triggered.connect(self._canvas.clear_mask)
        tb.addAction(clear_act)

        tb.addSeparator()

        tb.addWidget(QLabel("Brush: "))
        brush_slider = QSlider(Qt.Orientation.Horizontal)
        brush_slider.setRange(5, 120)
        brush_slider.setValue(self._canvas.brush_size)
        brush_slider.setFixedWidth(100)
        brush_slider.valueChanged.connect(lambda v: setattr(self._canvas, "brush_size", v))
        tb.addWidget(brush_slider)

        tb.addSeparator()

        remove_act = QAction("Remove Object", self)
        remove_act.setShortcut("Return")
        remove_act.triggered.connect(self._run_inpaint)
        tb.addAction(remove_act)

    # ── Actions ──────────────────────────────────────────────────────────

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", _default_dir(), "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        self._canvas.load_image(path)
        self._status.setText(f"Loaded: {path}  |  Left-drag to paint mask, Right-drag to erase")

    def _save_image(self):
        if not self._canvas.has_image:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", _default_dir(), "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return
        self._canvas.get_image().save(path)
        self._status.setText(f"Saved: {path}")

    def _run_inpaint(self):
        if not self._canvas.has_image:
            self._status.setText("Open an image first.")
            return
        if not self._canvas.has_mask:
            self._status.setText("Paint a mask over the object to remove first.")
            return

        if self._session is None:
            self._status.setText("Loading model (first run downloads ~100MB)...")
            QApplication.processEvents()
            try:
                self._session = load_session()
            except Exception as e:
                QMessageBox.critical(self, "Model Load Error", str(e))
                return

        self._status.setText("Removing object...")
        QApplication.processEvents()

        try:
            result = inpaint(self._session, self._canvas.get_image(), self._canvas.get_mask())
            self._canvas.set_image(result)
            self._status.setText("Done. Paint another mask to continue, or Save.")
        except Exception as e:
            QMessageBox.critical(self, "Inpainting Error", str(e))
            self._status.setText("Error during inpainting.")


def _setup_logging(log_dir: str, level: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "remover.log")
    err_path = os.path.join(log_dir, "remover.err.log")

    # Truncate on startup so each run starts clean
    open(out_path, "w").close()
    open(err_path, "w").close()

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_path)],
    )
    sys.stdout = open(out_path, "a", buffering=1)
    sys.stderr = open(err_path, "a", buffering=1)
    print(f"--- session start (log level: {level.upper()}) ---")


def main():
    parser = argparse.ArgumentParser(description="Object Remover")
    parser.add_argument(
        "--log",
        metavar="DIR",
        nargs="?",
        const=os.path.expanduser("~/.local/share/remover/logs"),
        help="Write logs to DIR (default: ~/.local/share/remover/logs)",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="warning",
        choices=["debug", "info", "warning", "error"],
        help="Log verbosity: debug, info, warning, error (default: warning)",
    )
    args, qt_args = parser.parse_known_args()

    if args.log:
        _setup_logging(args.log, args.log_level)

    app = QApplication([sys.argv[0]] + qt_args)
    app.setApplicationName("Object Remover")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
