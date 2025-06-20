
# ─────────────────────────── imports ──────────────────────────────────────
import os, sys, time, argparse
from pathlib import Path

import cv2 as cv                       # OpenCV for I/O and image ops
import numpy as np
import pandas as pd
from cvzone.FaceMeshModule import FaceMeshDetector

# ────────────────────────── local constants ──────────────────────────────
# Patch size (small to keep dataset files tiny)
EYE_W, EYE_H = 32, 16

# MediaPipe landmark indices that outline the LEFT eye
LEFT_EYE_LANDMARKS = (
     33, 133, 144, 145, 153, 154, 155,
    157, 158, 159, 160, 161, 163, 173
)

# Output directory for generated CSVs
DATA_DIR = Path(__file__).parent / "data"

# ───────────────────────── helper functions ──────────────────────────────
def extract_eye(img, lms, padding: int = 5):
    """
    Crop a (32×16) grayscale patch around the LEFT eye.
    Returns None if the bounding box is invalid (face not detected, etc.).
    """
    xs = [lms[i][0] for i in LEFT_EYE_LANDMARKS]
    ys = [lms[i][1] for i in LEFT_EYE_LANDMARKS]
    x0 = max(0,  int(min(xs)) - padding)
    y0 = max(0,  int(min(ys)) - padding)
    x1 = min(img.shape[1], int(max(xs)) + padding)
    y1 = min(img.shape[0], int(max(ys)) + padding)
    if x0 >= x1 or y0 >= y1:
        return None
    eye = cv.cvtColor(img[y0:y1, x0:x1], cv.COLOR_BGR2GRAY)
    return cv.resize(eye, (EYE_W, EYE_H), interpolation=cv.INTER_AREA)

def load_labels(csv_file: Path, n_frames: int) -> np.ndarray:
    """
    Load blink labels into a per-frame int array (0/1).
    CSV may contain either:
      • manual_blink  … already frame-aligned, or
      • blink_frame   … sparse list of positive frame indices.
    """
    df = pd.read_csv(csv_file)
    if "manual_blink" in df.columns and len(df) >= n_frames:
        return df["manual_blink"].astype(int).to_numpy()[:n_frames]
    if "blink_frame" in df.columns:
        labels = np.zeros(n_frames, dtype=int)
        idx = df["blink_frame"].astype(int).to_numpy()
        labels[idx[idx < n_frames]] = 1
        return labels
    raise ValueError(
        f"{csv_file.name}: must contain 'manual_blink' or 'blink_frame' column."
    )

# ────────────────────────── command-line API ─────────────────────────────
ap = argparse.ArgumentParser(description="Generate blink training CSV from video.")
ap.add_argument("--video",  required=True, help="input video file")
ap.add_argument("--labels", required=True, help="CSV with blink annotations")
ap.add_argument("--out",    default=None,  help="output CSV name (optional)")
args = ap.parse_args()

vid_path  = Path(args.video)
lbl_path  = Path(args.labels)
if not vid_path.exists():
    sys.exit(f"✗ Video not found: {vid_path}")
if not lbl_path.exists():
    sys.exit(f"✗ Label CSV not found: {lbl_path}")

# ─────────────────── video + detector initialisation ─────────────────────
cap = cv.VideoCapture(str(vid_path))
n_total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
labels  = load_labels(lbl_path, n_total)

detector = FaceMeshDetector(maxFaces=1)
DATA_DIR.mkdir(exist_ok=True)
out_csv  = args.out or f"eye_image_data_{vid_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
out_path = DATA_DIR / out_csv

# ─────────────────────────── main loop ───────────────────────────────────
rows, missing = [], 0
t0 = time.time()
print(f"► Processing {vid_path.name}  ({n_total} frames)" )

for idx in range(n_total):
    ok, frame = cap.read()
    if not ok:
        break

    blink_lbl = int(labels[idx])          # 0/1 for this frame
    frame, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        eye_patch = extract_eye(frame, faces[0])
    else:
        eye_patch, missing = None, missing + 1

    if eye_patch is None:                 # keep label-alignment → zeros
        eye_patch = np.zeros((EYE_H, EYE_W), dtype=np.uint8)

    row = {"manual_blink": blink_lbl}
    row.update({f"pixel_{i}": int(v) for i, v in enumerate(eye_patch.ravel())})
    rows.append(row)

    # lightweight progress meter
    if (idx + 1) % 500 == 0 or idx + 1 == n_total:
        pct = 100 * (idx + 1) / n_total
        print(f"\r  {idx + 1:>6}/{n_total}  ({pct:5.1f}%)", end="")

cap.release()
print(f"\n✓ Finished in {time.time() - t0:.1f}s  (missed face: {missing} frames)")

# ─────────────────────────── save CSV ────────────────────────────────────
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"→ Dataset written to  {out_path.resolve()}")