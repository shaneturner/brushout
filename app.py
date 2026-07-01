import argparse
import logging
import os
import subprocess
import sys
import threading
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPoint, QRectF, QSize, QTimer, Signal
from PySide6.QtGui import (
    QAction, QColor, QCursor, QImage, QKeySequence,
    QPainter, QPen, QPixmap, QGuiApplication,
)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QGraphicsEllipseItem, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSlider, QToolBar, QWidget,
)

from model import load_session, ModelType
from inpaint import inpaint

MAX_UNDO = 10


def _default_dir() -> str:
    """Return Windows user home in WSL, empty string otherwise."""
    if not _IN_WSL:
        return ""
    win_home = os.environ.get("USERPROFILE", "")
    if win_home:
        result = subprocess.run(["wslpath", win_home], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    return "/mnt/c"


def _is_wsl() -> bool:
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


_IN_WSL = _is_wsl()


def _drop_path(url) -> str:
    """Convert a dropped QUrl to a filesystem path, handling WSL Windows paths."""
    path = url.toLocalFile()
    # In WSL, Windows Explorer can drop Windows-style paths (C:\...) instead of /mnt/c/...
    if not path:
        path = url.toString()
    if _IN_WSL and len(path) >= 2 and path[1] == ':':
        result = subprocess.run(["wslpath", path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    return path


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
    image_dropped = Signal(str)

    _HINT_WSL = """
        <div style='text-align:center; font-family:sans-serif;'>
          <p style='font-size:13pt; color:#c8c8c8; margin:0 0 6px 0;'>No image loaded</p>
          <p style='font-size:10pt; color:#999; margin:0 0 16px 0;'>
            <span style='color:#ddd;'>Ctrl+O</span> &nbsp;open file dialog
            &nbsp;&nbsp;&nbsp;
            <span style='color:#ddd;'>Ctrl+V</span> &nbsp;paste image
          </p>
          <p style='font-size:10pt; color:#ffbe50; margin:0 0 6px 0;'>
            <b>Running in WSL</b> &mdash; drag &amp; drop from Windows Explorer is not supported
          </p>
          <p style='font-size:10pt; color:#999; line-height:1.7; margin:0;'>
            Use <span style='color:#ddd;'>Ctrl+V</span> to load an image from Windows:<br>
            &bull; &nbsp;Ctrl+C a file in Explorer &rarr; Ctrl+V here<br>
            &bull; &nbsp;Screenshot with <span style='color:#ddd;'>Win+Shift+S</span> &rarr; Ctrl+V here<br>
            &bull; &nbsp;Copy an image in Photos&nbsp;/&nbsp;browser&nbsp;/&nbsp;Paint &rarr; Ctrl+V here
          </p>
        </div>"""

    _HINT_DEFAULT = """
        <div style='text-align:center; font-family:sans-serif;'>
          <p style='font-size:13pt; color:#c8c8c8; margin:0 0 6px 0;'>No image loaded</p>
          <p style='font-size:10pt; color:#999; margin:0;'>
            <span style='color:#ddd;'>Ctrl+O</span> &nbsp;open file dialog
            &nbsp;&nbsp;&nbsp;
            <span style='color:#ddd;'>Ctrl+V</span> &nbsp;paste image
            &nbsp;&nbsp;&nbsp;
            drag &amp; drop a file here
          </p>
        </div>"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
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
        self._cursor_ring: QGraphicsEllipseItem | None = None

        self.setMinimumSize(600, 400)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.viewport().setMouseTracking(True)

        self._hint_label = QLabel(self.viewport())
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setTextFormat(Qt.TextFormat.RichText)
        self._hint_label.setWordWrap(True)
        self._hint_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hint_label.setStyleSheet(
            "background: transparent;"
            "border: 1px dashed #555;"
            "border-radius: 8px;"
        )
        self._hint_label.setText(self._HINT_WSL if _IN_WSL else self._HINT_DEFAULT)
        self._hint_label.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        m = 30
        r = self.viewport().rect()
        self._hint_label.setGeometry(r.adjusted(m, m, -m, -m))

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
        self._cursor_ring = None
        self._hint_label.setVisible(self._image_pil is None)
        if self._image_pil is None:
            self._scene.setSceneRect(QRectF())
            self.resetTransform()
            self._hint_label.raise_()
            return

        pixmap = pil_to_qpixmap(self._image_pil)
        self._image_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._image_item)

        self._refresh_mask_overlay()
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        pen = QPen(QColor(255, 255, 255, 200))
        pen.setCosmetic(True)
        pen.setWidthF(1.5)
        self._cursor_ring = QGraphicsEllipseItem()
        self._cursor_ring.setPen(pen)
        self._cursor_ring.setBrush(Qt.BrushStyle.NoBrush)
        self._cursor_ring.setZValue(3)
        self._cursor_ring.hide()
        self._scene.addItem(self._cursor_ring)

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

    def close_image(self):
        self._image_pil = None
        self._mask_arr = None
        self._undo_stack.clear()
        self._rebuild_scene()

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

    def _move_cursor_ring(self, viewport_pos):
        if self._cursor_ring is None or self._panning or self._space_held:
            if self._cursor_ring:
                self._cursor_ring.hide()
            return
        sp = self.mapToScene(viewport_pos)
        r = self._brush_size / 2.0
        self._cursor_ring.setRect(sp.x() - r, sp.y() - r, r * 2, r * 2)
        self._cursor_ring.show()

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

        self._move_cursor_ring(event.position().toPoint())
        if self._drawing:
            self._stroke_to(self._scene_pos(event), erase=self._erasing)

    def mouseReleaseEvent(self, event):
        self._drawing = False
        self._erasing = False
        self._last_pos = None
        if self._panning:
            self._panning = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def leaveEvent(self, event):
        if self._cursor_ring:
            self._cursor_ring.hide()
        super().leaveEvent(event)

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
        if self._image_pil is None:
            return
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

    # ── Drag and drop ────────────────────────────────────────────────────

    _IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            if any(
                _drop_path(u).lower().endswith(self._IMAGE_EXTS)
                for u in event.mimeData().urls()
            ):
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = _drop_path(url)
            if path.lower().endswith(self._IMAGE_EXTS):
                self.image_dropped.emit(path)
                break
        event.acceptProposedAction()

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def brush_size(self) -> int:
        return self._brush_size

    @brush_size.setter
    def brush_size(self, value: int):
        self._brush_size = value
        cursor_vp = self.mapFromGlobal(self.cursor().pos())
        self._move_cursor_ring(cursor_vp)

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

    def paste_from_clipboard(self) -> str | None:
        """
        Try to load an image from the clipboard.
        Returns a status string on success, None if clipboard had nothing useful.
        """
        clipboard = QGuiApplication.clipboard()
        mime = clipboard.mimeData()

        # 1. Image pixels (screenshot, copy from browser/viewer, etc.)
        if mime.hasImage():
            qimg = clipboard.image()
            if not qimg.isNull():
                qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
                w, h = qimg.width(), qimg.height()
                arr = np.frombuffer(qimg.bits(), dtype=np.uint8).reshape((h, w, 3)).copy()
                self._image_pil = Image.fromarray(arr, "RGB")
                self._mask_arr = np.zeros((h, w), dtype=np.uint8)
                self._undo_stack.clear()
                self._rebuild_scene()
                QTimer.singleShot(0, lambda: self.fitInView(
                    self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
                ))
                return "Pasted image from clipboard."

        # 2. File URLs (text/uri-list) — works on native platforms; may work in WSLg
        if mime.hasUrls():
            for url in mime.urls():
                path = _drop_path(url)
                if path.lower().endswith(self._IMAGE_EXTS) and os.path.isfile(path):
                    self.load_image(path)
                    return f"Loaded: {path}"

        # 3. File path as plain text (e.g. copied from Explorer's address bar)
        if mime.hasText():
            text = mime.text().strip().strip('"')
            if _IN_WSL and len(text) >= 2 and text[1] == ':':
                result = subprocess.run(["wslpath", text], capture_output=True, text=True)
                if result.returncode == 0:
                    text = result.stdout.strip()
            if text.lower().endswith(self._IMAGE_EXTS) and os.path.isfile(text):
                self.load_image(text)
                return f"Loaded: {text}"

        # 4. WSL: read CF_HDROP directly via PowerShell (WSLg doesn't bridge file handles)
        if _IN_WSL:
            try:
                ps = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                     "$f = Get-Clipboard -Format FileDropList; if ($f) { $f | % { $_.FullName } }"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in ps.stdout.splitlines():
                    win_path = line.strip()
                    if not win_path:
                        continue
                    wsl = subprocess.run(["wslpath", win_path], capture_output=True, text=True)
                    if wsl.returncode == 0:
                        linux_path = wsl.stdout.strip()
                        if linux_path.lower().endswith(self._IMAGE_EXTS) and os.path.isfile(linux_path):
                            self.load_image(linux_path)
                            return f"Loaded: {linux_path}"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brushout")
        self.resize(1100, 750)

        self._sessions: dict[ModelType, object] = {}
        self._active_model = ModelType.LAMA
        self._model_btn: QPushButton | None = None
        self._remove_btn: QPushButton | None = None
        self._canvas = Canvas(self)
        self.setCentralWidget(self._canvas)

        self._build_toolbar()
        self._status = QLabel("Open an image to get started.")
        threading.Thread(target=self._preload_models, daemon=True).start()
        self.statusBar().addWidget(self._status)
        self._canvas.image_dropped.connect(self._on_image_dropped)

    def _build_toolbar(self):
        tb = QToolBar("Tools", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        open_act = QAction("Open Image", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._open_image)
        tb.addAction(open_act)

        paste_act = QAction("Paste Image", self)
        paste_act.setShortcut(QKeySequence.StandardKey.Paste)
        paste_act.triggered.connect(self._paste_image)
        tb.addAction(paste_act)

        close_act = QAction("Close Image", self)
        close_act.setShortcut(QKeySequence.StandardKey.Close)
        close_act.triggered.connect(self._close_image)
        tb.addAction(close_act)

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

        self._model_btn = QPushButton(self._model_label())
        self._model_btn.setToolTip("Click to switch inpainting model")
        self._model_btn.clicked.connect(self._toggle_model)
        self._model_btn.setStyleSheet("")
        tb.addWidget(self._model_btn)

        tb.addSeparator()

        self._remove_btn = QPushButton("Remove Object  ↵")
        self._remove_btn.setShortcut("Return")
        self._remove_btn.clicked.connect(self._run_inpaint)
        self._remove_btn.setToolTip("Remove the masked region (Enter)")
        self._remove_btn.setStyleSheet("""
            QPushButton {
                padding: 4px 16px;
                border: none;
                border-radius: 4px;
                background: #3d6b3d;
                color: #ddd;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background: #4a804a; color: #fff; }
            QPushButton:pressed { background: #2f542f; }
            QPushButton:disabled { background: #2a2a2a; color: #666; }
        """)
        tb.addWidget(self._remove_btn)

    # ── Model preloading ─────────────────────────────────────────────────

    def _preload_models(self):
        for model_type in (ModelType.LAMA, ModelType.MIGAN):
            try:
                session = load_session(model_type)
                QTimer.singleShot(0, lambda s=session, t=model_type: self._store_session(t, s))
            except Exception as e:
                print(f"Preload failed for {model_type.value}: {e}")

    def _store_session(self, model_type: ModelType, session):
        self._sessions[model_type] = session

    # ── Model selection ──────────────────────────────────────────────────

    def _model_label(self) -> str:
        return {
            ModelType.LAMA:  "⇄  LaMa",
            ModelType.MIGAN: "⇄  MI-GAN",
        }[self._active_model]

    def _toggle_model(self):
        self._active_model = (
            ModelType.MIGAN if self._active_model == ModelType.LAMA else ModelType.LAMA
        )
        self._model_btn.setText(self._model_label())
        self._status.setText(
            f"Switched to {self._model_label().replace('Model: ', '')}. "
            "Model will load on next removal."
        )

    # ── Actions ──────────────────────────────────────────────────────────

    def _close_image(self):
        self._canvas.close_image()
        self._status.setText("Open an image to get started.")

    def _on_image_dropped(self, path: str):
        self._canvas.load_image(path)
        self._status.setText(f"Loaded: {path}  |  Left-drag to paint mask, Right-drag to erase")

    def _paste_image(self):
        msg = self._canvas.paste_from_clipboard()
        if msg:
            self._status.setText(f"{msg}  |  Left-drag to paint mask, Right-drag to erase")
        else:
            self._status.setText("Nothing to paste — copy an image or a file path first.")

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

        if self._active_model not in self._sessions:
            model_name = self._model_label().replace("Model: ", "")
            self._status.setText(f"Loading {model_name} model...")
            QApplication.processEvents()
            try:
                self._sessions[self._active_model] = load_session(self._active_model)
            except Exception as e:
                QMessageBox.critical(self, "Model Load Error", str(e))
                return

        self._status.setText("Removing object...")
        QApplication.processEvents()

        try:
            session = self._sessions[self._active_model]
            result = inpaint(session, self._canvas.get_image(), self._canvas.get_mask())
            self._canvas.set_image(result)
            self._status.setText("Done. Paint another mask to continue, or Save.")
        except Exception as e:
            QMessageBox.critical(self, "Inpainting Error", str(e))
            self._status.setText("Error during inpainting.")


def _setup_logging(log_dir: str, level: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "brushout.log")
    err_path = os.path.join(log_dir, "brushout.err.log")

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
    parser = argparse.ArgumentParser(description="Brushout — AI-powered object removal")
    parser.add_argument(
        "--log",
        metavar="DIR",
        nargs="?",
        const=os.path.expanduser("~/.local/share/brushout/logs"),
        help="Write logs to DIR (default: ~/.local/share/brushout/logs)",
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
    app.setApplicationName("Brushout")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
