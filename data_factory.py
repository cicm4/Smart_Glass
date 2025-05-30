import time, cv2 as cv, cvzone as cvz, numpy as np, pandas as pd, keyboard
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule   import LivePlot

# --------------- constants ----------------
PATCH_W, PATCH_H = 24, 12                    # eye‑ROI size to save
CSV_NAME = f"blink_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"

# MediaPipe landmark ids used for EAR
L_OUT, L_IN, L_UP, L_LO =  33, 133, 159, 145     # left eye
R_OUT, R_IN, R_UP, R_LO = 362, 263, 386, 374     # right eye
POINTS_USED = [L_OUT, L_IN, L_UP, L_LO, R_OUT, R_IN, R_UP, R_LO]

# --------------- helpers ------------------
def ear(face, out_id, in_id, up_id, lo_id):
    p_out = face[out_id];  p_in = face[in_id]
    p_up  = face[up_id];   p_lo = face[lo_id]
    ver, _ = detector.findDistance(p_up,  p_lo)
    hor, _ = detector.findDistance(p_out, p_in)
    return (ver / hor) * 10, ver, hor

def eye_patch(img, pts):
    x, y, w, h = cv.boundingRect(pts)
    patch = cv.cvtColor(img[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)
    if patch.size == 0:                                  # safety
        return np.zeros((PATCH_H, PATCH_W), np.uint8)
    return cv.resize(patch, (PATCH_W, PATCH_H), cv.INTER_AREA)

# --------------- init ---------------------
cap      = cv.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plot     = LivePlot(640, 360, [1, 4])         # ratio plot

data_rows   = []
blink_count = 0
t0          = time.time()

try:
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img, faces = detector.findFaceMesh(img, draw=False)

        # MANUAL blink state = whether SPACE is held
        manual_blink = keyboard.is_pressed('space')
        if manual_blink and (not blink_count or data_rows[-1]['manual_blink'] == 0):
            blink_count += 1                  # increment on first frame of a new press

        timestamp = time.time() - t0

        if faces:
            face = faces[0]

            # draw landmarks used
            for pid in POINTS_USED:
                cv.circle(img, face[pid], 4, (255, 0, 255), cv.FILLED)

            # ratio for left / right
            ratio_L, vL, hL = ear(face, L_OUT, L_IN, L_UP, L_LO)
            ratio_R, vR, hR = ear(face, R_OUT, R_IN, R_UP, R_LO)
            ratio_avg       = (ratio_L + ratio_R) / 2

            # left‑eye patch pixels
            pts_left = np.array([face[id] for id in [L_OUT, L_IN, L_UP, L_LO]], np.int32)
            patch    = eye_patch(img, pts_left)          # (H,W) greyscale
            pixels   = patch.flatten().tolist()          # 288 ints

            # HUD
            cv.putText(img, f'Blink #: {blink_count}',  (20,35),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv.putText(img, f'EAR L/R: {ratio_L:.2f}/{ratio_R:.2f}',
                       (20,65), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if manual_blink:
                cv.putText(img, 'BLINK (SPACE held)', (20,95),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # plot & show
            plot_img   = plot.update(ratio_avg)
            show_stack = cvz.stackImages([cv.resize(img,(640,360)), plot_img], 2, 1)
            cv.imshow("Blink Recorder", show_stack)

            # ---------- record row ----------
            row = dict(
                timestamp   = timestamp,
                ratio_left  = ratio_L,
                ratio_right = ratio_R,
                ratio_avg   = ratio_avg,
                v_left      = vL,
                h_left      = hL,
                v_right     = vR,
                h_right     = hR,
                blink_count = blink_count,
                manual_blink= int(manual_blink)
            )
            row.update({f'px_{i}': pix for i, pix in enumerate(pixels)})
            data_rows.append(row)

        else:
            # still write a row to keep timeline (pixels = NaNs)
            data_rows.append(dict(timestamp=timestamp,
                                  ratio_left=None, ratio_right=None, ratio_avg=None,
                                  v_left=None, h_left=None, v_right=None, h_right=None,
                                  blink_count=blink_count, manual_blink=int(manual_blink),
                                  **{f'px_{i}': None for i in range(PATCH_W*PATCH_H)}))
            cv.imshow("Blink Recorder", cv.resize(img,(640,360)))

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if data_rows:
        pd.DataFrame(data_rows).to_csv(CSV_NAME, index=False)
        print(f"Saved {len(data_rows)} rows → {CSV_NAME}")
    cap.release()
    cv.destroyAllWindows()
