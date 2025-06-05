# Video Blink Detection Test – NEW INPUT PIPELINE for BlinkDetector
# ---------------------------------------------------------------
import time, cv2 as cv, cvzone as cvz, numpy as np, torch
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule   import LivePlot
from Model.model import BlinkModelMed as model

# ───────── configuration ──────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────

# ---------- helper functions (copied from collector) ----------
def ear(face, out_id, in_id, up_id, lo_id, det):
    p_out, p_in = face[out_id], face[in_id]
    p_up,  p_lo = face[up_id],  face[lo_id]
    ver,_ = det.findDistance(p_up, p_lo)
    hor,_ = det.findDistance(p_out, p_in)
    return (ver / hor) * 10, ver, hor

def eye_patch(img, pts):
    x,y,w,h = cv.boundingRect(pts)
    patch   = cv.cvtColor(img[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)
    if patch.size == 0:
        return np.zeros((PATCH_H, PATCH_W), np.uint8)
    return cv.resize(patch, (PATCH_W, PATCH_H), interpolation=cv.INTER_AREA)
# --------------------------------------------------------------

# ---------- load model & feature stats ------------------------
model = BlinkDetector().to(DEVICE).eval()
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))

stats   = np.load(STATS_NPZ)
MEAN, STD = stats["mean"], stats["std"]          # (7,) each
# --------------------------------------------------------------

# ---------- runtime buffers -----------------------------------
eye_buf, num_buf = [], []        # hold 30 frames
blink_count      = 0
prev_pred        = 0
# --------------------------------------------------------------

# ---------- OpenCV / MediaPipe init ---------------------------
cap      = cv.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plot_y   = LivePlot(640, 360, [1,4])
t0       = time.time()
# --------------------------------------------------------------

while True:
    ok, img = cap.read()
    if not ok:
        break

    img, faces = detector.findFaceMesh(img, draw=False)
    timestamp = time.time() - t0

    if faces:
        face = faces[0]

        # draw landmarks (optional)
        for pid in POINTS_USED:
            cv.circle(img, face[pid], 3, (255,0,255), cv.FILLED)

        # ── numeric features
        ratio_L,vL,hL = ear(face, L_OUT,L_IN,L_UP,L_LO, detector)
        ratio_R,vR,hR = ear(face, R_OUT,R_IN,R_UP,R_LO, detector)
        ratio_avg     = (ratio_L + ratio_R) / 2
        num_feats = np.array([ratio_L,ratio_R,ratio_avg,
                              vL,hL,vR,hR], dtype=np.float32)

        # ── eye patch (left eye)
        pts_left = np.array([face[id] for id in [L_OUT,L_IN,L_UP,L_LO]], np.int32)
        patch    = patch = eye_patch(img, pts_left).T.astype(np.float32) / 255.0  # (24,12) ← matches training

        # add to buffers
        eye_buf.append(patch[None])     # (1,24,12)
        num_buf.append(num_feats)
        if len(eye_buf) > SEQ_LEN: eye_buf.pop(0)
        if len(num_buf) > SEQ_LEN: num_buf.pop(0)

        # run model when window full
        if len(eye_buf) == SEQ_LEN:
            eye_arr = np.stack(eye_buf)                          # (30,1,24,12)
            num_arr = (np.stack(num_buf) - MEAN) / STD           # z-score

            eye_t = torch.from_numpy(eye_arr)[None].to(DEVICE)   # (1,30,1,24,12)
            num_t = torch.from_numpy(num_arr)[None].to(DEVICE)   # (1,30,7)

            with torch.no_grad():
                p = torch.sigmoid(model(eye_t, num_t)).item()

            pred = int(p > THRESH)
            if pred == 1 and prev_pred == 0:      # count rising edge
                blink_count += 1
            prev_pred = pred

            cv.putText(img, f'Model: {"Blink" if pred else "No blink"}  p={p:.2f}',
                       (20,95), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0,0,255) if pred else (0,255,0), 2)

        # HUD
        cv.putText(img, f'Blink #: {blink_count}', (20,35),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.putText(img, f'EAR L/R: {ratio_L:.2f}/{ratio_R:.2f}', (20,65),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # plot & show
        img_plot  = plot_y.update(ratio_avg)
        img       = cv.resize(img, (640,360))
        img_stack = cvz.stackImages([img, img_plot], 2, 1)
        cv.imshow("BlinkDetector", img_stack)
    else:
        cv.imshow("BlinkDetector", cv.resize(img,(640,360)))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
