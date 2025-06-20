"""
Smart Glasses – GUI + Blink-triggered macro runner
Resilient version:  2025-06-16
────────────────────────────────────────────────────────────────────────────
•   Off-loads macro execution to a QThread → UI never freezes
•   Debounces re-entrancy (only one macro can run at a time)
•   Status-bar shows progress / errors instead of hard-crashing
"""

# ────── stdlib ────────────────────────────────────────────────────────────
from __future__ import annotations
import sys, os, json, logging
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Macros"))

# ────── third-party ───────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
from cvzone.FaceMeshModule import FaceMeshDetector
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QListView,
    QMainWindow,
    QFileDialog,
    QComboBox,
    QWidget,
    QVBoxLayout,
    QSplitter,
    QToolBar,
)
# ────── project ───────────────────────────────────────────────────────────
from Model.model import BlinkRatioNet, EyeBlinkNet
from collections import deque
from Macros.deparse import run as run_macro
from Macros.Starter import MacroModel, MainWindow as MacroEditorWindow
from constants import (
    Paths,
    BLINKING_THREASHOLD,
    Image_Constants,
    Training_Constants,
)

# ──────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("main")


# ────── helper functions (metric extraction) ──────────────────────────────
def eye_metrics(face, out_id, in_id, up_id, lo_id, det):
    p_out, p_in = face[out_id], face[in_id]
    p_up, p_lo = face[up_id], face[lo_id]
    ver, _ = det.findDistance(p_up, p_lo)
    hor, _ = det.findDistance(p_out, p_in)
    return ver / (hor + 1e-6), ver, hor


def vertical_ratios(face, pairs, out_id, in_id, det):
    width, _ = det.findDistance(face[out_id], face[in_id])
    feats = []
    for up_id, lo_id in pairs:
        h, _ = det.findDistance(face[up_id], face[lo_id])
        feats.append(h / (width + 1e-6))
    return feats, width


def extract_eye_image(img, face, padding: int = 5):
    """Return a normalised left-eye grayscale patch or None."""
    ids = Image_Constants.LEFT_EYE_IDS
    xs = [face[i][0] for i in ids]
    ys = [face[i][1] for i in ids]
    x0 = max(0, int(min(xs)) - padding)
    y0 = max(0, int(min(ys)) - padding)
    x1 = min(img.shape[1], int(max(xs)) + padding)
    y1 = min(img.shape[0], int(max(ys)) + padding)
    if x0 >= x1 or y0 >= y1:
        return None
    eye = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(
        eye,
        (Image_Constants.IM_WIDTH, Image_Constants.IM_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return eye


# ────── threaded macro runner ─────────────────────────────────────────────
class MacroRunnerThread(QThread):
    finished = Signal(str)           # emits folder path when done
    error = Signal(str, str)         # emits (folder, message)

    def __init__(self, folder: str | Path):
        super().__init__()
        self.folder = str(folder)

    def run(self):
        try:
            run_macro(self.folder)   # blocking call
            self.finished.emit(self.folder)
        except Exception as ex:
            self.error.emit(self.folder, str(ex))


# ────── main window ───────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Glasses")

        # ── ML + vision stack ──────────────────────────────────────────
        self.seq_len = Training_Constants.SEQUENCE_LENGTH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_image_model = False
        self.model_combo = None
        self.load_selected_model(initial=True)

        self.detector = FaceMeshDetector(maxFaces=1)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        self.num_buf = deque(maxlen=self.seq_len)
        self.prev_pred = 0

        # ── UI widgets  (left = macro list, right = camera) ──────────
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

        self.splitter = QSplitter()
        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.setCentralWidget(self.splitter)

        self._build_toolbar()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # ── misc state ───────────────────────────────────────────────
        self.macro_dir: Path | None = None
        self._macro_thread: QThread | None = None

    # ────────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        self.addToolBar(tb)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Numeric", "Eye Image"])
        self.model_combo.currentIndexChanged.connect(self.load_selected_model)

        choose_act = QAction("Choose Folder", self)
        choose_act.triggered.connect(self.choose_folder)

        run_act = QAction("Run Macro", self)
        run_act.triggered.connect(self.run_macro)

        editor_act = QAction("Macro Editor", self)
        editor_act.triggered.connect(self.open_macro_editor)

        tb.addWidget(self.model_combo)
        tb.addActions([choose_act, run_act, editor_act])

    # ────────────────────────────────────────────────────────────────
    # Folder + macro list handling
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Macro Folder", str(Path.cwd())
        )
        if folder:
            self.load_macros(Path(folder))

    def load_macros(self, folder: Path):
        self.macro_dir = folder
        self.model_list._data.clear()

        for sub in folder.iterdir():
            if (sub / "macro.json").exists():
                try:
                    data = json.loads((sub / "macro.json").read_text())
                    name = data.get("name", sub.name)
                except Exception:
                    name = sub.name
                self.model_list._data.append({"name": name, "path": str(sub)})

        self.model_list.layoutChanged.emit()

    # ────────────────────────────────────────────────────────────────
    # Threaded macro run helpers
    def _start_macro(self, folder: str):
        # ignore if a macro is already running
        if self._macro_thread and self._macro_thread.isRunning():
            return

        self.statusBar().showMessage(f"Running macro: {Path(folder).name}")
        self._macro_thread = MacroRunnerThread(folder)
        self._macro_thread.finished.connect(
            lambda f: self.statusBar().showMessage(
                f"Macro done: {Path(f).name}", 3000
            )
        )
        self._macro_thread.error.connect(
            lambda f, e: self.statusBar().showMessage(
                f"Macro error: {e}", 5000
            )
        )
        self._macro_thread.start()

    def run_macro(self):
        idx = self.view.currentIndex().row()
        if idx < 0:
            return
        folder = self.model_list._data[idx]["path"]
        self._start_macro(folder)

    def _blink_trigger_macro(self):
        idx = self.view.currentIndex().row()
        if idx < 0:
            return
        folder = self.model_list._data[idx]["path"]
        self._start_macro(folder)

    # ────────────────────────────────────────────────────────────────
    def load_selected_model(self, index: int | None = None, *, initial: bool = False):
        """Load the chosen blink detection model and its normalisation stats."""
        use_img = False
        if self.model_combo is not None:
            use_img = self.model_combo.currentIndex() == 1
        self.use_image_model = use_img

        if use_img:
            self.model = EyeBlinkNet(
                input_size=Image_Constants.IM_WIDTH * Image_Constants.IM_HEIGHT
            ).to(self.device).eval()
            weights = Paths.IMG_WEIGHTS
            stats_path = Paths.IMG_STATS_NPZ
        else:
            self.model = BlinkRatioNet().to(self.device).eval()
            weights = Paths.NUM_WEIGHTS
            stats_path = Paths.NUM_STATS_NPZ

        if os.path.exists(weights):
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model = torch.jit.script(self.model)
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.mean, self.std = stats["mean"], stats["std"]

        if not initial:
            self.num_buf.clear()
            self.prev_pred = 0

    # ────────────────────────────────────────────────────────────────
    # Blink pipeline + UI update
    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        img, faces = self.detector.findFaceMesh(frame, draw=False)
        if faces:
            face = faces[0]

            if self.use_image_model:
                eye = extract_eye_image(frame, face)
                if eye is None:
                    self.text_label.setText("No face")
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(qimg))
                    return
                feats = eye.flatten().astype(np.float32)
            else:
                L_OUT, L_IN = Image_Constants.LEFT_EYE_OUT_ID, Image_Constants.LEFT_EYE_INSIDE_ID
                L_UP, L_LO = Image_Constants.LEFT_EYE_UP_ID, Image_Constants.LEFT_EYE_LOW_ID
                R_OUT, R_IN = Image_Constants.RIGHT_EYE_OUT_ID, Image_Constants.RIGHT_EYE_INSIDE_ID
                R_UP, R_LO = Image_Constants.RIGHT_EYE_UP_ID, Image_Constants.RIGHT_EYE_LOW_ID
                L_PAIRS, R_PAIRS = (
                    Image_Constants.LEFT_EYE_PAIR_IDS,
                    Image_Constants.RIGHT_EYE_PAIR_IDS,
                )

                ratio_L, _, _ = eye_metrics(face, L_OUT, L_IN, L_UP, L_LO, self.detector)
                ratio_R, _, _ = eye_metrics(face, R_OUT, R_IN, R_UP, R_LO, self.detector)
                verts_L, width_L = vertical_ratios(face, L_PAIRS, L_OUT, L_IN, self.detector)
                verts_R, width_R = vertical_ratios(face, R_PAIRS, R_OUT, R_IN, self.detector)

                feats = np.array(
                    [ratio_L, ratio_R, *verts_L, *verts_R, width_L, width_R],
                    dtype=np.float32,
                )

            self.num_buf.append(feats)

            if len(self.num_buf) == self.seq_len:
                arr = (np.stack(self.num_buf) - self.mean) / self.std
                t = torch.from_numpy(arr)[None].to(self.device)
                with torch.inference_mode():
                    p = torch.sigmoid(self.model(t)).item()

                pred = int(p > BLINKING_THREASHOLD)
                if pred == 1 and self.prev_pred == 0:
                    self._blink_trigger_macro()

                self.prev_pred = pred
                self.text_label.setText("Blink" if pred else "No blink")
        else:
            self.text_label.setText("No face")

        # UI update
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # ────────────────────────────────────────────────────────────────
    def open_macro_editor(self):
        self.editor = MacroEditorWindow()
        if self.macro_dir is not None:
            self.editor.load_macros(self.macro_dir)
        self.editor.show()

    def closeEvent(self, ev):
        self.cap.release()
        super().closeEvent(ev)


# ────── bootstrap ────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec())
