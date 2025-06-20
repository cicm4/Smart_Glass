from __future__ import annotations
# ────────────────────────── imports ────────────────────────────────────────

import cv2
import time
import argparse
from pathlib import Path
import requests
import sys, os, json, logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Macros"))

# ────── third-party ───────────────────────────────────────────────────────
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
from Macros.deparse import run as run_macro
from Macros.Starter import MacroModel, MainWindow as MacroEditorWindow
from constants import (
  Paths,
  BLINKING_THREASHOLD,
  Image_Constants,
  Training_Constants,
)

# ────────────────────────── camera helpers (identical to test.py) ─────────────
def set_resolution(url: str, index: int = 0):
    requests.get(f"{url}/control?var=framesize&val={index}", timeout=2)

def set_quality(url: str, value: int = 20):
    requests.get(f"{url}/control?var=quality&val={value}", timeout=2)

# ──────────────────────────────────────────────────────────────────────────────

url = "http://192.168.4.1"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.4.1", help="ESP32 base URL")
    ap.add_argument("--seconds", type=float, default=10, help="recording length")
    args = ap.parse_args()

    # --- configure camera -------------------------------------------------
    print("[ESP32] set QQVGA (160×120) + Q=20")
    set_resolution(url, 0)       # 0 = QQVGA 160×120
    set_quality(url, 20)         # 10 = best, 63 = worst

    stream = cv2.VideoCapture(f"{args.url}:81/stream")
    if not stream.isOpened():
        raise SystemExit("Could not open MJPEG stream. Check Wi-Fi & URL.")

    # find original frame size once
    ok, frame = stream.read()
    if not ok:
        raise SystemExit("No frame received from camera.")
    h0, w0 = frame.shape[:2]
    cx = 0.5
    cy = 0.5

    # Define ROI size in source image
    SRC_W, SRC_H = 96, 48
    DST_W, DST_H = 32, 16

    ox = int(w0 * cx) - SRC_W // 2
    oy = int(h0 * cy) - SRC_H // 2
    ox = max(0, min(ox, w0 - SRC_W))
    oy = max(0, min(oy, h0 - SRC_H))
    print(f"[INFO] Cropping ROI at ({ox}:{ox+SRC_W}, {oy}:{oy+SRC_H}), resizing to {DST_W}x{DST_H}")
    

    # --- grab + decimate --------------------------------------------------
    target_dt = 1.0 / 20.0        # 50 ms
    next_t = time.perf_counter()
    end_t = next_t + args.seconds
    saved = 0

    while time.perf_counter() < end_t:
        ok, frame = stream.read()
        if not ok:
            continue
        now = time.perf_counter()
        if now < next_t:          # skip until next 50 ms slot
            continue
        next_t += target_dt

        roi = frame[oy:oy+96, ox:ox+96]      # (96,96,3) BGR
        if roi.shape[0] != 96:               # safety if edges clipped
            roi = cv2.resize(roi, (96, 96),
                             interpolation=cv2.INTER_AREA)
            
    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
