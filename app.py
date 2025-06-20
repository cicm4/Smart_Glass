"""Smart Glasses – ESP32 streaming blink detector.

This GUI mirrors the behaviour of :mod:`on_device_full` but reads video
frames directly from an ESP32 camera module. Each frame is cropped to a
96×48 region, resized to 32×16 grayscale and passed to the image based
blink model. A blink is triggered when the model output exceeds 0.95 and
the previously predicted state was "no blink".
"""

from __future__ import annotations

# ────── stdlib ──────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import sys
import requests
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Macros"))

# ────── third-party ────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QListView,
    QMainWindow,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QSplitter,
    QToolBar,
)

# ────── project ───────────────────────────────────────────────────────
from Model.model import EyeBlinkNet
from Macros.deparse import run as run_macro
from Macros.Starter import MacroModel, MainWindow as MacroEditorWindow
from constants import Paths, Image_Constants, Training_Constants


# ────── camera helpers -------------------------------------------------
def set_resolution(url: str, index: int = 0) -> None:
    """Configure ESP32 camera resolution."""

    try:
        requests.get(f"{url}/control?var=framesize&val={index}", timeout=2)
    except Exception:
        pass


def set_quality(url: str, value: int = 20) -> None:
    """Configure ESP32 JPEG quality."""

    try:
        requests.get(f"{url}/control?var=quality&val={value}", timeout=2)
    except Exception:
        pass


# ────── threaded macro runner -----------------------------------------
class MacroRunnerThread(QThread):
    """Run a macro folder in a background thread."""

    finished = Signal(str)
    error = Signal(str, str)

    def __init__(self, folder: str | Path):
        super().__init__()
        self.folder = str(folder)

    def run(self) -> None:  # pragma: no cover - UI side effect
        try:
            run_macro(self.folder)
            self.finished.emit(self.folder)
        except Exception as ex:  # pylint: disable=broad-except
            self.error.emit(self.folder, str(ex))


# ────── main window ----------------------------------------------------
class MainWindow(QMainWindow):
    """Application window with live preview and blink to macro mapping."""

    THRESH = 0.95

    def __init__(self, url: str) -> None:
        super().__init__()
        self.setWindowTitle("Smart Glasses")

        # Camera setup -------------------------------------------------
        set_resolution(url, 0)  # 160×120
        set_quality(url, 20)
        self.cap = cv2.VideoCapture(f"{url}:81/stream")
        if not self.cap.isOpened():
            raise SystemExit("Could not open MJPEG stream. Check Wi-Fi & URL.")

        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("No frame received from camera.")
        h0, w0 = frame.shape[:2]
        src_w, src_h = 96, 48
        cx = 0.5
        cy = 0.5
        ox = int(w0 * cx) - src_w // 2
        oy = int(h0 * cy) - src_h // 2
        self.ox = max(0, min(ox, w0 - src_w))
        self.oy = max(0, min(oy, h0 - src_h))
        self.src_w, self.src_h = src_w, src_h

        # ML -----------------------------------------------------------
        self.seq_len = Training_Constants.SEQUENCE_LENGTH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EyeBlinkNet(
            input_size=Image_Constants.IM_WIDTH * Image_Constants.IM_HEIGHT
        ).to(self.device).eval()

        if Paths.IMG_WEIGHTS.exists():
            self.model.load_state_dict(torch.load(Paths.IMG_WEIGHTS, map_location=self.device))
        if Paths.IMG_STATS_NPZ.exists():
            stats = np.load(Paths.IMG_STATS_NPZ)
            self.mean, self.std = stats["mean"], stats["std"]
        else:  # fallbacks
            size = Image_Constants.IM_WIDTH * Image_Constants.IM_HEIGHT
            self.mean = np.zeros((self.seq_len, size), dtype=np.float32)
            self.std = np.ones_like(self.mean)

        self.img_buf: list[np.ndarray] = []
        self.prev_pred = 0

        # UI widgets ---------------------------------------------------
        self.view = QListView()
        self.model_list = MacroModel([])
        self.view.setModel(self.model_list)

        self.video_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.text_label = QLabel("No blink", alignment=Qt.AlignmentFlag.AlignCenter)

        left = QWidget()
        lyl = QVBoxLayout(left)
        lyl.addWidget(self.view)

        right = QWidget()
        ryl = QVBoxLayout(right)
        ryl.addWidget(self.video_label)
        ryl.addWidget(self.text_label)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        self.setCentralWidget(splitter)

        self._build_toolbar()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.macro_dir: Path | None = None
        self._macro_thread: QThread | None = None

    # -----------------------------------------------------------------
    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        self.addToolBar(tb)

        choose_act = QAction("Choose Folder", self)
        choose_act.triggered.connect(self.choose_folder)

        run_act = QAction("Run Macro", self)
        run_act.triggered.connect(self.run_macro)

        editor_act = QAction("Macro Editor", self)
        editor_act.triggered.connect(self.open_macro_editor)

        tb.addActions([choose_act, run_act, editor_act])

    # -----------------------------------------------------------------
    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Macro Folder", str(Path.cwd()))
        if folder:
            self.load_macros(Path(folder))

    def load_macros(self, folder: Path) -> None:
        self.macro_dir = folder
        self.model_list._data.clear()
        for sub in folder.iterdir():
            if (sub / "macro.json").exists():
                try:
                    data = json.loads((sub / "macro.json").read_text())
                    name = data.get("name", sub.name)
                except Exception:  # pragma: no cover - malformed macro
                    name = sub.name
                self.model_list._data.append({"name": name, "path": str(sub)})
        self.model_list.layoutChanged.emit()

    # -----------------------------------------------------------------
    def _start_macro(self, folder: str) -> None:
        if self._macro_thread and self._macro_thread.isRunning():
            return

        self.statusBar().showMessage(f"Running macro: {Path(folder).name}")
        self._macro_thread = MacroRunnerThread(folder)
        self._macro_thread.finished.connect(
            lambda f: self.statusBar().showMessage(f"Macro done: {Path(f).name}", 3000)
        )
        self._macro_thread.error.connect(
            lambda f, e: self.statusBar().showMessage(f"Macro error: {e}", 5000)
        )
        self._macro_thread.start()

    def run_macro(self) -> None:
        idx = self.view.currentIndex().row()
        if idx < 0:
            return
        folder = self.model_list._data[idx]["path"]
        self._start_macro(folder)

    def _blink_trigger_macro(self) -> None:
        idx = self.view.currentIndex().row()
        if idx < 0:
            return
        folder = self.model_list._data[idx]["path"]
        self._start_macro(folder)

    # -----------------------------------------------------------------
    def update_frame(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            return

        roi = frame[self.oy : self.oy + self.src_h, self.ox : self.ox + self.src_w]
        if roi.shape[0] != self.src_h or roi.shape[1] != self.src_w:
            roi = cv2.resize(roi, (self.src_w, self.src_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        eye = cv2.resize(
            gray,
            (Image_Constants.IM_WIDTH, Image_Constants.IM_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        feats = eye.flatten().astype(np.float32)
        self.img_buf.append(feats)
        if len(self.img_buf) > self.seq_len:
            self.img_buf.pop(0)

        if len(self.img_buf) == self.seq_len:
            arr = (np.stack(self.img_buf) - self.mean) / self.std
            t = torch.from_numpy(arr)[None].to(self.device)
            with torch.no_grad():
                p = torch.sigmoid(self.model(t)).item()

            pred = int(p > self.THRESH)
            if pred == 1 and self.prev_pred == 0:
                self._blink_trigger_macro()
            self.prev_pred = pred
            self.text_label.setText("Blink" if pred else "No blink")

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # -----------------------------------------------------------------
    def open_macro_editor(self) -> None:
        self.editor = MacroEditorWindow()
        if self.macro_dir is not None:
            self.editor.load_macros(self.macro_dir)
        self.editor.show()

    def closeEvent(self, ev) -> None:  # pragma: no cover - UI hook
        self.cap.release()
        super().closeEvent(ev)


# ────── bootstrap ------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://192.168.4.1", help="ESP32 base URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    app = QApplication(sys.argv)
    win = MainWindow(args.url)
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()

