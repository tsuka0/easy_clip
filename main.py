import os
import sys
import numpy as np
import cv2
import torch
from PyQt6 import QtCore, QtGui, QtWidgets

try:
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
except Exception as exc:
    raise RuntimeError(
        "SAM3 import failed. Run install.bat first.\n" + str(exc)
    )


APP_NAME = "easy clip"

DEFAULT_CHECKPOINT = os.path.join("model", "sam3.pt")
DEFAULT_MASK_THRESHOLD = 0.5


def ensure_rgb(image_bgr):
    if image_bgr is None:
        return None
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    if image_bgr.shape[2] == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image, params=None):
    ext = os.path.splitext(path)[1]
    if not ext:
        return False
    if params is None:
        params = []
    ok, enc = cv2.imencode(ext, image, params)
    if not ok:
        return False
    enc.tofile(path)
    return True


def rgb_to_qimage(image_rgb):
    h, w, ch = image_rgb.shape
    bytes_per_line = ch * w
    image = QtGui.QImage(
        image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888
    ).copy()
    image.setDevicePixelRatio(1.0)
    return image


def mask_to_overlay(mask, color=(255, 0, 0), alpha=120):
    if mask is None:
        return None
    if mask.ndim > 2:
        mask = np.squeeze(mask)
        if mask.ndim > 2:
            mask = mask[0]
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = (mask.astype(np.uint8) * alpha)
    return overlay


def rgba_to_qimage(image_rgba):
    h, w, ch = image_rgba.shape
    bytes_per_line = ch * w
    image = QtGui.QImage(
        image_rgba.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGBA8888
    ).copy()
    image.setDevicePixelRatio(1.0)
    return image


class Sam3Engine:
    def __init__(self, checkpoint_path):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            raise RuntimeError("SAM3 は CUDA GPU が必要です。")

        ckpt_path = checkpoint_path if os.path.exists(checkpoint_path) else None
        self.video_predictor = Sam3VideoPredictor(checkpoint_path=ckpt_path, device=device)
        self.device = device

    def start_session(self, resource_path):
        response = self.video_predictor.handle_request(
            {"type": "start_session", "resource_path": resource_path}
        )
        return response["session_id"]

    def add_prompt(self, session_id, frame_idx, text, points, point_labels, box_xyxy, threshold):
        box_xywh = None
        box_labels = None
        if box_xyxy is not None:
            x1, y1, x2, y2 = box_xyxy
            box_xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]]
            box_labels = [1]

        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_idx": int(frame_idx),
            "text": text if text else None,
            "points": points if points else None,
            "point_labels": point_labels if point_labels else None,
            "boxes_xywh": box_xywh,
            "box_labels": box_labels,
        }
        outputs = self.video_predictor.handle_request(request)
        return self._extract_mask(outputs, threshold)

    def propagate(self, session_id):
        for outputs in self.video_predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            if "frame_idx" not in outputs:
                continue
            frame_idx = int(outputs["frame_idx"])
            mask = self._extract_mask(outputs, None)
            yield frame_idx, mask

    def _extract_mask(self, outputs, threshold):
        if outputs is None:
            return None
        mask = None
        if isinstance(outputs, dict):
            if "obj_id_to_mask" in outputs and outputs["obj_id_to_mask"]:
                masks = list(outputs["obj_id_to_mask"].values())
                mask = self._merge_masks(masks)
            elif "masks" in outputs:
                mask = outputs["masks"]
            elif "mask" in outputs:
                mask = outputs["mask"]
        if mask is None:
            return None
        if torch.is_tensor(mask):
            mask = mask.detach().float().cpu().numpy()
        if mask.ndim > 2:
            mask = np.squeeze(mask)
            if mask.ndim > 2:
                mask = mask[0]
        if threshold is None:
            return mask
        return mask >= threshold

    def _merge_masks(self, masks):
        merged = None
        for m in masks:
            if torch.is_tensor(m):
                m = m.detach().float().cpu().numpy()
            if m.ndim > 2:
                m = np.squeeze(m)
                if m.ndim > 2:
                    m = m[0]
            if merged is None:
                merged = m
            else:
                merged = np.maximum(merged, m)
        return merged


class ImageCanvas(QtWidgets.QGraphicsView):
    pointAdded = QtCore.pyqtSignal(float, float, int)
    boxAdded = QtCore.pyqtSignal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.image_item = QtWidgets.QGraphicsPixmapItem()
        self.overlay_item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self.image_item)
        self.overlay_item.setParentItem(self.image_item)

        self.mode = "fg"
        self.dragging_box = False
        self.box_start = None
        self.box_item = None
        self.point_items = []

    def set_mode(self, mode):
        self.mode = mode

    def set_image(self, qimage, fit=True):
        pixmap = QtGui.QPixmap.fromImage(qimage)
        pixmap.setDevicePixelRatio(1.0)
        self.image_item.setOffset(0, 0)
        self.image_item.setPixmap(pixmap)
        self.overlay_item.setPixmap(QtGui.QPixmap())
        self.scene().setSceneRect(self.image_item.boundingRect())
        if fit:
            self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def set_overlay(self, qimage):
        if qimage is None:
            self.overlay_item.setPixmap(QtGui.QPixmap())
            return
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.overlay_item.setPixmap(pixmap)

    def clear_prompts(self):
        for item in self.point_items:
            self.scene().removeItem(item)
        self.point_items = []
        if self.box_item is not None:
            self.scene().removeItem(self.box_item)
            self.box_item = None

    def add_point_item(self, x, y, label):
        radius = 6
        color = QtGui.QColor(0, 200, 0) if label == 1 else QtGui.QColor(200, 60, 60)
        pen = QtGui.QPen(QtGui.QColor(20, 20, 20))
        pen.setWidth(1)
        brush = QtGui.QBrush(color)
        item = QtWidgets.QGraphicsEllipseItem(
            x - radius,
            y - radius,
            radius * 2,
            radius * 2,
        )
        item.setPen(pen)
        item.setBrush(brush)
        item.setParentItem(self.image_item)
        self.point_items.append(item)

    def set_box_item(self, x1, y1, x2, y2):
        if self.box_item is None:
            pen = QtGui.QPen(QtGui.QColor(60, 160, 255))
            pen.setWidth(2)
            self.box_item = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
            self.box_item.setPen(pen)
            self.box_item.setParentItem(self.image_item)
        rect = QtCore.QRectF(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)).normalized()
        self.box_item.setRect(rect)

    def _scene_to_image(self, pos):
        image_pos = self.image_item.mapFromScene(pos)
        return image_pos

    def _image_bounds_contains(self, pos):
        rect = self.image_item.boundingRect()
        return rect.contains(pos)

    def mousePressEvent(self, event):
        if self.image_item.pixmap().isNull():
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            image_pos = self._scene_to_image(pos)
            if not self._image_bounds_contains(image_pos):
                return
            x, y = image_pos.x(), image_pos.y()
            if self.mode in ("fg", "bg"):
                label = 1 if self.mode == "fg" else 0
                self.pointAdded.emit(x, y, label)
            elif self.mode == "box":
                self.dragging_box = True
                self.box_start = (x, y)
                self.set_box_item(x, y, x, y)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging_box and self.box_start is not None:
            pos = self.mapToScene(event.pos())
            image_pos = self._scene_to_image(pos)
            x, y = image_pos.x(), image_pos.y()
            x1, y1 = self.box_start
            self.set_box_item(x1, y1, x, y)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging_box and self.box_start is not None:
            pos = self.mapToScene(event.pos())
            image_pos = self._scene_to_image(pos)
            x, y = image_pos.x(), image_pos.y()
            x1, y1 = self.box_start
            self.dragging_box = False
            self.box_start = None
            self.boxAdded.emit(x1, y1, x, y)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)


class SessionInitWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object, str)

    def __init__(self, engine, resource_path):
        super().__init__()
        self.engine = engine
        self.resource_path = resource_path

    def run(self):
        try:
            session_id = self.engine.start_session(self.resource_path)
            self.finished.emit(session_id, "")
        except Exception as exc:
            self.finished.emit(None, str(exc))


class VideoPropagateWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, object, int)
    finished = QtCore.pyqtSignal(int)

    def __init__(self, engine, session_id, version_id):
        super().__init__()
        self.engine = engine
        self.session_id = session_id
        self.version_id = version_id

    def run(self):
        try:
            for frame_idx, mask in self.engine.propagate(self.session_id):
                self.progress.emit(int(frame_idx), mask, self.version_id)
        finally:
            self.finished.emit(self.version_id)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1200, 720)

        self.engine = Sam3Engine(DEFAULT_CHECKPOINT)
        self.session_id = None
        self.video_cap = None
        self.video_fps = 30
        self.video_frames = 0
        self.session_worker = None
        self.loading_dialog = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self.on_play_tick)
        self.last_frame_size = None
        self.pending_mask_update = False
        self.resource_path = None
        self.session_id = None
        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.setSingleShot(True)
        self.frame_timer.timeout.connect(self.on_frame_timer)
        self.target_frame_index = 0
        self.video_masks = {}
        self.propagate_worker = None
        self.propagate_version = 0
        self.is_propagating = False
        self.queued_frame_index = None

        self.current_image = None
        self.current_mask = None
        self.is_video = False
        self.current_frame_index = 0

        self.points = []
        self.labels = []
        self.box = None
        self.history = []
        self.history_index = -1

        self._build_ui()
        self._apply_theme()

    def set_ui_enabled(self, enabled):
        self.open_button.setEnabled(enabled)
        self.fg_button.setEnabled(enabled)
        self.bg_button.setEnabled(enabled)
        self.box_button.setEnabled(enabled)
        self.text_prompt_input.setEnabled(enabled)
        self.text_prompt_button.setEnabled(enabled)
        self.undo_button.setEnabled(enabled)
        self.redo_button.setEnabled(enabled)
        self.output_combo.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.frame_slider.setEnabled(enabled)
        self.threshold_slider.setEnabled(enabled)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.canvas = ImageCanvas(self)
        self.canvas.pointAdded.connect(self.on_point_added)
        self.canvas.boxAdded.connect(self.on_box_added)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        self.frame_slider.sliderReleased.connect(self.on_slider_released)
        self.frame_label = QtWidgets.QLabel("フレーム: 0")

        preview_layout = QtWidgets.QVBoxLayout()
        preview_layout.addWidget(self.canvas, 1)
        preview_layout.addWidget(self.frame_slider)
        preview_layout.addWidget(self.frame_label)

        preview_widget = QtWidgets.QWidget()
        preview_widget.setLayout(preview_layout)

        self.open_button = QtWidgets.QPushButton("画像/動画を開く")
        self.open_button.clicked.connect(self.open_file)

        self.mode_group = QtWidgets.QButtonGroup(self)
        self.fg_button = QtWidgets.QPushButton("前景ポイント")
        self.bg_button = QtWidgets.QPushButton("背景ポイント")
        self.box_button = QtWidgets.QPushButton("ボックス")
        self.fg_button.setCheckable(True)
        self.bg_button.setCheckable(True)
        self.box_button.setCheckable(True)
        self.fg_button.setChecked(True)
        self.mode_group.addButton(self.fg_button)
        self.mode_group.addButton(self.bg_button)
        self.mode_group.addButton(self.box_button)
        self.fg_button.clicked.connect(lambda: self.set_mode("fg"))
        self.bg_button.clicked.connect(lambda: self.set_mode("bg"))
        self.box_button.clicked.connect(lambda: self.set_mode("box"))

        self.undo_button = QtWidgets.QPushButton("元に戻す")
        self.redo_button = QtWidgets.QPushButton("やり直す")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

        self.text_prompt_input = QtWidgets.QLineEdit()
        self.text_prompt_input.setPlaceholderText("テキストプロンプトを入力")
        self.text_prompt_input.returnPressed.connect(self.on_text_prompt_apply)
        self.text_prompt_button = QtWidgets.QPushButton("テキスト適用")
        self.text_prompt_button.clicked.connect(self.on_text_prompt_apply)

        self.output_combo = QtWidgets.QComboBox()
        self.output_combo.addItems(["緑背景", "青背景", "透過PNG"])

        self.export_button = QtWidgets.QPushButton("出力")
        self.export_button.clicked.connect(self.export_output)

        self.status_label = QtWidgets.QLabel("準備完了")

        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(int(DEFAULT_MASK_THRESHOLD * 100))
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.threshold_label = QtWidgets.QLabel("")
        self.update_threshold_label()

        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(self.open_button)
        control_layout.addSpacing(8)
        control_layout.addWidget(QtWidgets.QLabel("プロンプト"))
        control_layout.addWidget(self.fg_button)
        control_layout.addWidget(self.bg_button)
        control_layout.addWidget(self.box_button)
        control_layout.addSpacing(6)
        control_layout.addWidget(self.text_prompt_input)
        control_layout.addWidget(self.text_prompt_button)
        control_layout.addSpacing(8)
        control_layout.addWidget(QtWidgets.QLabel("マスクしきい値"))
        control_layout.addWidget(self.threshold_slider)
        control_layout.addWidget(self.threshold_label)
        control_layout.addSpacing(8)
        control_layout.addWidget(self.undo_button)
        control_layout.addWidget(self.redo_button)
        control_layout.addSpacing(8)
        control_layout.addWidget(QtWidgets.QLabel("出力形式"))
        control_layout.addWidget(self.output_combo)
        control_layout.addSpacing(8)
        control_layout.addWidget(self.export_button)
        control_layout.addStretch(1)
        control_layout.addWidget(self.status_label)

        control_widget = QtWidgets.QWidget()
        control_widget.setLayout(control_layout)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(control_widget)
        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(splitter)
        central.setLayout(layout)

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow { background-color: #11161d; color: #e6e6e6; }
            QLabel { color: #e6e6e6; }
            QPushButton {
                background-color: #1c232d;
                border: 1px solid #2b3644;
                border-radius: 6px;
                padding: 8px;
                color: #e6e6e6;
            }
            QPushButton:checked {
                background-color: #2b5a8f;
                border: 1px solid #3a76b6;
            }
            QPushButton:hover { background-color: #263141; }
            QComboBox {
                background-color: #1c232d;
                border: 1px solid #2b3644;
                border-radius: 6px;
                padding: 6px;
                color: #e6e6e6;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #1c232d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4b91d1;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            """
        )

    def set_mode(self, mode):
        self.canvas.set_mode(mode)

    def open_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "画像または動画を開く",
            "",
            "画像/動画 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.mp4 *.mov *.avi *.mkv)",
        )
        if not file_path:
            return

        self.clear_state()
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".mp4", ".mov", ".avi", ".mkv"):
            self.load_video(file_path)
        else:
            self.load_image(file_path)

    def load_image(self, path):
        image_bgr = imread_unicode(path)
        if image_bgr is None:
            self.status_label.setText("画像の読み込みに失敗しました。")
            return
        self.is_video = False
        self.current_image = ensure_rgb(image_bgr)
        self.current_mask = None
        self.resource_path = path
        self.session_id = None
        self.canvas.set_image(rgb_to_qimage(self.current_image))
        self.last_frame_size = self.current_image.shape[:2]
        self.frame_slider.setMaximum(0)
        self.frame_label.setText("画像")
        self.push_history()
        self.status_label.setText("画像を読み込みました。")

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.status_label.setText("動画の読み込みに失敗しました。")
            return
        self.is_video = True
        self.video_cap = cap
        self.resource_path = path
        self.session_id = None
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        ret, frame = cap.read()
        if not ret:
            self.status_label.setText("動画の読み込みに失敗しました。")
            return
        self.current_image = ensure_rgb(frame)
        if self.current_image is None:
            self.status_label.setText("動画の読み込みに失敗しました。")
            return
        self.canvas.set_image(rgb_to_qimage(self.current_image))
        self.last_frame_size = self.current_image.shape[:2]
        self.frame_slider.setMaximum(max(self.video_frames - 1, 0))
        self.frame_label.setText(f"フレーム: 0 / {self.video_frames - 1}")
        self.status_label.setText("動画を読み込みました。プロンプト追加時に初期化します。")
        self.session_id = None

    def on_session_init_finished(self, session_id, error):
        if self.loading_dialog is not None:
            self.loading_dialog.close()
            self.loading_dialog = None
        self.session_worker = None

        if error or session_id is None:
            self.status_label.setText("初期化に失敗しました。")
            if error:
                QtWidgets.QMessageBox.warning(self, "初期化エラー", error)
            self.set_ui_enabled(True)
            self.pending_mask_update = False
            return

        self.session_id = session_id
        self.push_history()
        self.status_label.setText("準備完了。プロンプトを追加してください。")
        self.set_ui_enabled(True)
        if self.pending_mask_update:
            self.pending_mask_update = False
            self.update_mask()

    def start_session_init(self):
        if self.session_worker is not None:
            return
        if not self.resource_path:
            self.status_label.setText("初期化に失敗しました。")
            return
        self.status_label.setText("初期化中...")
        self.set_ui_enabled(False)
        self.loading_dialog = QtWidgets.QProgressDialog("初期化中...", None, 0, 0, self)
        self.loading_dialog.setWindowTitle("初期化中")
        self.loading_dialog.setCancelButton(None)
        self.loading_dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.loading_dialog.setMinimumDuration(0)
        self.loading_dialog.show()
        QtWidgets.QApplication.processEvents()
        self.session_worker = SessionInitWorker(self.engine, self.resource_path)
        self.session_worker.finished.connect(self.on_session_init_finished)
        self.session_worker.start()

    def start_propagation(self):
        if self.propagate_worker is not None:
            return
        if self.session_id is None:
            return
        self.propagate_version += 1
        current_version = self.propagate_version
        self.status_label.setText("マスクを動画全体に適用中...")
        self.is_propagating = True
        if self.play_timer.isActive():
            self.play_timer.stop()
        self.frame_timer.stop()
        self.set_ui_enabled(False)
        self.propagate_worker = VideoPropagateWorker(self.engine, self.session_id, current_version)
        self.propagate_worker.progress.connect(self.on_propagate_progress)
        self.propagate_worker.finished.connect(self.on_propagate_finished)
        self.propagate_worker.setPriority(QtCore.QThread.Priority.HighestPriority)
        self.propagate_worker.start()

    def on_propagate_progress(self, frame_idx, mask, version_id):
        if version_id != self.propagate_version:
            return
        if mask is None:
            return
        threshold = self.threshold_slider.value() / 100
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)
        if mask.dtype != np.bool_:
            mask = mask >= threshold
        self.video_masks[frame_idx] = mask

    def on_propagate_finished(self, version_id):
        if version_id != self.propagate_version:
            return
        self.propagate_worker = None
        self.is_propagating = False
        self.set_ui_enabled(True)
        self.status_label.setText("マスク適用完了。")
        if self.queued_frame_index is not None:
            queued = self.queued_frame_index
            self.queued_frame_index = None
            self.load_frame(queued)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self.toggle_play()
            return
        super().keyPressEvent(event)

    def toggle_play(self):
        if not self.is_video or self.video_cap is None:
            return
        if self.is_propagating:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            return
        # Sync capture position to current frame before playing
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        interval_ms = int(1000 / max(self.video_fps, 1))
        self.play_timer.start(max(interval_ms, 15))

    def on_play_tick(self):
        if not self.is_video or self.video_cap is None:
            self.play_timer.stop()
            return
        if self.is_propagating:
            self.play_timer.stop()
            return
        ret, frame = self.video_cap.read()
        if not ret:
            self.play_timer.stop()
            return
        self.current_frame_index += 1
        if self.current_frame_index >= self.video_frames:
            self.play_timer.stop()
            return
        self.current_image = ensure_rgb(frame)
        if self.current_image is None:
            self.play_timer.stop()
            return
        fit = False
        if self.last_frame_size != self.current_image.shape[:2]:
            fit = True
            self.last_frame_size = self.current_image.shape[:2]
        self.canvas.set_image(rgb_to_qimage(self.current_image), fit=fit)
        mask = self.video_masks.get(self.current_frame_index)
        if mask is not None:
            overlay_rgba = mask_to_overlay(mask, color=(255, 80, 80), alpha=130)
            overlay_qimage = rgba_to_qimage(overlay_rgba)
            self.canvas.set_overlay(overlay_qimage)
        else:
            self.canvas.set_overlay(None)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"フレーム: {self.current_frame_index} / {self.video_frames - 1}")

    def on_frame_changed(self, value):
        if not self.is_video or self.video_cap is None:
            return
        if self.is_propagating:
            self.queued_frame_index = value
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
        self.target_frame_index = value
        self.frame_timer.start(140)

    def on_slider_released(self):
        if not self.is_video or self.video_cap is None:
            return
        if self.is_propagating:
            return
        self.frame_timer.stop()
        self.load_frame(self.target_frame_index)

    def on_frame_timer(self):
        if self.is_propagating:
            return
        self.load_frame(self.target_frame_index)

    def load_frame(self, frame_index):
        self.current_frame_index = frame_index
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.video_cap.read()
        if not ret:
            return
        self.current_image = ensure_rgb(frame)
        if self.current_image is None:
            self.status_label.setText("動画の読み込みに失敗しました。")
            return
        fit = False
        if self.last_frame_size != self.current_image.shape[:2]:
            fit = True
            self.last_frame_size = self.current_image.shape[:2]
        self.canvas.set_image(rgb_to_qimage(self.current_image), fit=fit)
        self.current_mask = None
        mask = self.video_masks.get(self.current_frame_index)
        if mask is not None:
            overlay_rgba = mask_to_overlay(mask, color=(255, 80, 80), alpha=130)
            overlay_qimage = rgba_to_qimage(overlay_rgba)
            self.canvas.set_overlay(overlay_qimage)
        else:
            self.canvas.set_overlay(None)
        self.frame_label.setText(f"フレーム: {frame_index} / {self.video_frames - 1}")

    def on_point_added(self, x, y, label):
        self.points.append((x, y))
        self.labels.append(label)
        self.push_history()
        self.update_mask()

    def on_box_added(self, x1, y1, x2, y2):
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            return
        self.box = (x1, y1, x2, y2)
        self.push_history()
        self.update_mask()

    def push_history(self):
        state = (list(self.points), list(self.labels), self.box)
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]
        self.history.append(state)
        self.history_index = len(self.history) - 1
        self.apply_prompts_to_canvas()

    def undo(self):
        if self.history_index <= 0:
            return
        self.history_index -= 1
        self.restore_history()

    def redo(self):
        if self.history_index >= len(self.history) - 1:
            return
        self.history_index += 1
        self.restore_history()

    def restore_history(self):
        points, labels, box = self.history[self.history_index]
        self.points = list(points)
        self.labels = list(labels)
        self.box = box
        self.apply_prompts_to_canvas()
        self.update_mask()

    def clear_state(self):
        self.points = []
        self.labels = []
        self.box = None
        self.history = []
        self.history_index = -1
        if self.video_cap is not None:
            self.video_cap.release()
        self.video_cap = None
        self.session_id = None
        self.is_video = False
        self.canvas.clear_prompts()
        self.canvas.set_overlay(None)
        self.video_masks = {}
        self.propagate_worker = None
        self.is_propagating = False
        self.queued_frame_index = None

    def update_threshold_label(self):
        value = self.threshold_slider.value()
        self.threshold_label.setText(f"{value / 100:.2f}")

    def on_threshold_changed(self, value):
        self.update_threshold_label()
        if self.current_image is not None:
            self.update_mask()

    def on_text_prompt_apply(self):
        if self.current_image is not None:
            self.update_mask()

    def apply_prompts_to_canvas(self):
        self.canvas.clear_prompts()
        for (x, y), label in zip(self.points, self.labels):
            self.canvas.add_point_item(x, y, label)
        if self.box is not None:
            self.canvas.set_box_item(*self.box)

    def update_mask(self):
        if self.current_image is None:
            return
        text_prompt = self.text_prompt_input.text().strip()
        if len(self.points) == 0 and self.box is None and not text_prompt:
            self.canvas.set_overlay(None)
            return

        self.status_label.setText("SAM3 実行中...")
        QtWidgets.QApplication.processEvents()
        threshold = self.threshold_slider.value() / 100

        if self.is_video:
            if self.session_id is None:
                self.pending_mask_update = True
                self.start_session_init()
                return
            mask = self.engine.add_prompt(
                self.session_id,
                self.current_frame_index,
                text_prompt,
                self.points,
                self.labels,
                self.box,
                threshold,
            )
        else:
            if self.session_id is None:
                self.pending_mask_update = True
                self.start_session_init()
                return
            mask = self.engine.add_prompt(
                self.session_id,
                0,
                text_prompt,
                self.points,
                self.labels,
                self.box,
                threshold,
            )

        self.current_mask = mask
        if mask is not None:
            overlay_rgba = mask_to_overlay(mask, color=(255, 80, 80), alpha=130)
            overlay_qimage = rgba_to_qimage(overlay_rgba)
            self.canvas.set_overlay(overlay_qimage)
            self.status_label.setText("マスクを更新しました。")
        else:
            self.canvas.set_overlay(None)
            self.status_label.setText("マスクが生成されませんでした。")

        if self.is_video and mask is not None:
            self.video_masks[self.current_frame_index] = mask
            self.start_propagation()

    def export_output(self):
        if self.current_image is None or self.current_mask is None:
            self.status_label.setText("プロンプトを追加してマスクを作成してください。")
            return

        mode = self.output_combo.currentText()
        if self.is_video:
            self.export_video(mode)
        else:
            self.export_image(mode)

    def export_image(self, mode):
        if mode == "透過PNG":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "PNGを保存",
                "",
                "PNG (*.png)",
            )
        else:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "画像を保存",
                "",
                "PNG (*.png);;JPG (*.jpg)",
            )

        if not path:
            return

        output = self.compose_output(self.current_image, self.current_mask, mode)
        if mode == "透過PNG":
            ok = imwrite_unicode(path, cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA))
        else:
            ok = imwrite_unicode(path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        if not ok:
            self.status_label.setText("画像の保存に失敗しました。")
            return
        self.status_label.setText("画像を出力しました。")

    def export_video(self, mode):
        if self.session_id is None:
            self.pending_mask_update = False
            self.start_session_init()
            self.status_label.setText("初期化中です。完了後に再度出力してください。")
            return

        if mode == "透過PNG":
            out_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "出力先フォルダを選択"
            )
            if not out_dir:
                return
        else:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "動画を保存",
                "",
                "MP4 (*.mp4)",
            )
            if not path:
                return

        self.status_label.setText("マスクを伝播中...")
        QtWidgets.QApplication.processEvents()

        if self.video_cap is None:
            self.status_label.setText("動画の読み込みに失敗しました。")
            return

        height, width = self.current_image.shape[:2]
        writer = None
        if mode != "透過PNG":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, self.video_fps, (width, height))

        threshold = self.threshold_slider.value() / 100
        for frame_idx, mask in self.engine.propagate(self.session_id):
            if mask is None:
                continue
            mask = mask >= threshold
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_cap.read()
            if not ret:
                continue
            rgb = ensure_rgb(frame)
            output = self.compose_output(rgb, mask, mode)
            if mode == "透過PNG":
                filename = os.path.join(out_dir, f"frame_{frame_idx:06d}.png")
                imwrite_unicode(filename, cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA))
            else:
                writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

        if writer is not None:
            writer.release()
        self.status_label.setText("動画を出力しました。")

    def compose_output(self, image_rgb, mask, mode):
        mask_bool = mask.astype(bool)
        if mode == "透過PNG":
            rgba = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = image_rgb
            rgba[..., 3] = (mask_bool * 255).astype(np.uint8)
            return rgba

        if mode == "緑背景":
            bg = np.array([0, 255, 0], dtype=np.uint8)
        else:
            bg = np.array([0, 0, 255], dtype=np.uint8)

        output = image_rgb.copy()
        output[~mask_bool] = bg
        return output


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
