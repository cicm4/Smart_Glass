import cv2
import time
import argparse
from pathlib import Path
import requests

# ────────────────────────── camera helpers (identical to test.py) ─────────────
def set_resolution(url: str, index: int = 0):
    requests.get(f"{url}/control?var=framesize&val={index}", timeout=2)

def set_quality(url: str, value: int = 20):
    requests.get(f"{url}/control?var=quality&val={value}", timeout=2)

# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.4.1", help="ESP32 base URL")
    ap.add_argument("--seconds", type=float, default=10, help="recording length")
    ap.add_argument("--out", default="capture.mp4",
                    help="output .mp4 (or folder/ if ends with /)")
    ap.add_argument("--roi", type=float, nargs=2, metavar=("CX", "CY"),
                    help="ROI centre as % of width/height (0–1). default centre")
    args = ap.parse_args()

    # --- configure camera -------------------------------------------------
    print("[ESP32] set QQVGA (160×120) + Q=20")
    set_resolution(args.url, 0)       # 0 = QQVGA 160×120
    set_quality(args.url, 20)         # 10 = best, 63 = worst

    stream = cv2.VideoCapture(f"{args.url}:81/stream")
    if not stream.isOpened():
        raise SystemExit("Could not open MJPEG stream. Check Wi-Fi & URL.")

    # find original frame size once
    ok, frame = stream.read()
    if not ok:
        raise SystemExit("No frame received from camera.")
    h0, w0 = frame.shape[:2]
    cx = args.roi[0] if args.roi else 0.5
    cy = args.roi[1] if args.roi else 0.5
    # rectangle edges in source image
    HALF = 48                    # 96 // 2
    ox = int(w0 * cx) - HALF
    oy = int(h0 * cy) - HALF
    ox = max(0, min(ox, w0 - 96))
    oy = max(0, min(oy, h0 - 96))
    print(f"[INFO] Cropping ROI at ({ox}:{ox+96}, {oy}:{oy+96})")

    # --- choose output sink ----------------------------------------------
    out_is_folder = str(args.out).endswith(("/", "\\"))
    Path(args.out).mkdir(parents=True, exist_ok=True) if out_is_folder else None

    if not out_is_folder:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(args.out), fourcc, 20.0, (96, 96))

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

        if out_is_folder:
            cv2.imwrite(f"{args.out.rstrip('/')}/f{saved:06}.jpg", roi)
        else:
            vw.write(roi)
        saved += 1
        # *** live preview (optional) ******************************************
        cv2.imshow("96×96 ROI (q to quit)", roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"[DONE] Saved {saved} frames → {args.out}")
    stream.release()
    if not out_is_folder:
        vw.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
