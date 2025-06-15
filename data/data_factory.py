import time
import cv2 as cv
import cvzone as cvz
import numpy as np
import pandas as pd
import keyboard
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import constants

# CSV output
CSV_NAME = f"blink_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"

# Label window parameters
BLINK_PRE_FRAMES = constants.Data_Gathering_Constants.BLINK_PRE_FRAMES
BLINK_POST_FRAMES = constants.Data_Gathering_Constants.BLINK_POST_FRAMES

# ----- helpers ---------------------------------------------------------------

def eye_ratio(face, out_id, in_id, up_id, lo_id):
    """Return vertical/horizontal eye ratio."""
    p_out, p_in = face[out_id], face[in_id]
    p_up, p_lo = face[up_id], face[lo_id]
    ver, _ = detector.findDistance(p_up, p_lo)
    hor, _ = detector.findDistance(p_out, p_in)
    return ver / hor if hor else 0.0

# ----- init -----------------------------------------------------------------
cap = cv.VideoCapture(0)
# use half resolution to avoid warped frames
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) // 2)
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) // 2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

detector = FaceMeshDetector(maxFaces=1)
plot = LivePlot(640, 360, [0, 0.4])

data_rows = []
blink_count = 0
blink_post_ctr = 0
prev_keypress = False
t0 = time.time()

try:
    while True:
        ok, img = cap.read()
        if not ok:
            break

        img, faces = detector.findFaceMesh(img, draw=False)
        keypress_now = keyboard.is_pressed("space")

        # ---- blink edge detection ----
        if keypress_now and not prev_keypress:
            blink_count += 1
            blink_post_ctr = BLINK_POST_FRAMES
            for i in range(1, BLINK_PRE_FRAMES + 1):
                if len(data_rows) >= i:
                    data_rows[-i]["manual_blink"] = 1

        manual_blink = int(keypress_now or blink_post_ctr > 0)
        if blink_post_ctr:
            blink_post_ctr -= 1

        timestamp = time.time() - t0

        if faces:
            face = faces[0]
            for pid in constants.Image_Constants.ID_ARRAYS:
                cv.circle(img, face[pid], 3, (255, 0, 255), cv.FILLED)

            ratio_L = eye_ratio(
                face,
                constants.Image_Constants.LEFT_EYE_OUT_ID,
                constants.Image_Constants.LEFT_EYE_INSIDE_ID,
                constants.Image_Constants.LEFT_EYE_UP_ID,
                constants.Image_Constants.LEFT_EYE_LOW_ID,
            )
            ratio_R = eye_ratio(
                face,
                constants.Image_Constants.RIGHT_EYE_OUT_ID,
                constants.Image_Constants.RIGHT_EYE_INSIDE_ID,
                constants.Image_Constants.RIGHT_EYE_UP_ID,
                constants.Image_Constants.RIGHT_EYE_LOW_ID,
            )
            ratio_avg = (ratio_L + ratio_R) / 2

            cv.putText(img, f"Blink #: {blink_count}", (20, 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv.putText(img, f"Ratio L/R: {ratio_L:.2f}/{ratio_R:.2f}", (20, 65), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if manual_blink:
                cv.putText(img, "BLINK (label=1)", (20, 95), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            img_plot = plot.update(ratio_avg)
            img_stack = cvz.stackImages([cv.resize(img, (640, 360)), img_plot], 2, 1)
            cv.imshow("Blink Recorder", img_stack)

            data_rows.append(
                dict(
                    timestamp=timestamp,
                    ratio_left=ratio_L,
                    ratio_right=ratio_R,
                    blink_count=blink_count,
                    manual_blink=manual_blink,
                )
            )
        else:
            data_rows.append(
                dict(
                    timestamp=timestamp,
                    ratio_left=None,
                    ratio_right=None,
                    blink_count=blink_count,
                    manual_blink=manual_blink,
                )
            )
            cv.imshow("Blink Recorder", cv.resize(img, (640, 360)))

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        prev_keypress = keypress_now
finally:
    if data_rows:
        pd.DataFrame(data_rows).to_csv(CSV_NAME, index=False)
        print(f"Saved {len(data_rows)} rows -> {CSV_NAME}")
    cap.release()
    cv.destroyAllWindows()
