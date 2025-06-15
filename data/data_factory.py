import time, cv2 as cv, cvzone as cvz, numpy as np, pandas as pd, keyboard
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import constants

# --------------- constants ----------------
WIDTH_IM, HEIGHT_IM = 24, 12  # size of each eye patch
CSV_NAME = f"blink_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"

# Label‑window padding (frames) ---------
BLINK_PRE_FRAMES = 3  # frames before key‑press set to 1
BLINK_POST_FRAMES = 6  # frames after key‑press set to 1

# --------------- helpers ------------------


def eye_aspect_ratio(face, out_id, in_id, up_id, lo_id):

    p_out = face[out_id]
    p_in = face[in_id]
    p_up = face[up_id]
    p_lo = face[lo_id]
    ver = detector.findDistance(p_up, p_lo)[0]
    hor = detector.findDistance(p_out, p_in)[0]
    return (ver / hor) * 10, ver, hor


def eye_patch(img, pts):
    """Return a greyscale crop around the provided eye landmarks."""
    x, y, w, h = cv.boundingRect(pts)
    patch = cv.cvtColor(img[y : y + h, x : x + w], cv.COLOR_BGR2GRAY)
    if patch.size == 0:
        return np.zeros(
            (constants.Image_Constants.IM_HEIGHT, constants.Image_Constants.IM_WIDTH),
            np.uint8,
        )
    return cv.resize(
        patch,
        (constants.Image_Constants.IM_WIDTH, constants.Image_Constants.IM_HEIGHT),
        interpolation=cv.INTER_AREA,
    )


# --------------- init ---------------------
cap = cv.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plot = LivePlot(640, 360, [1, 4])  # ratio plot

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

        # -------- blink edge detection --------
        if keypress_now and not prev_keypress:  # rising edge
            blink_count += 1
            blink_post_ctr = BLINK_POST_FRAMES

            # retroactively label previous frames
            for i in range(1, BLINK_PRE_FRAMES + 1):
                if len(data_rows) >= i:
                    data_rows[-i]["manual_blink"] = 1

        # decide label for this frame
        manual_blink = int(keypress_now or blink_post_ctr > 0)

        # countdown post window
        if blink_post_ctr:
            blink_post_ctr -= 1

        timestamp = time.time() - t0

        if faces:
            face = faces[0]

            # draw landmarks used
            for pid in constants.Image_Constants.ID_ARRAYS:
                cv.circle(img, face[pid], 4, (255, 0, 255), cv.FILLED)

            # ratio for left / right
            ratio_L, vL, hL = eye_aspect_ratio(
                face,
                constants.Image_Constants.LEFT_EYE_OUT_ID,
                constants.Image_Constants.LEFT_EYE_INSIDE_ID,
                constants.Image_Constants.LEFT_EYE_UP_ID,
                constants.Image_Constants.LEFT_EYE_LOW_ID,
            )
            ratio_R, vR, hR = eye_aspect_ratio(
                face,
                constants.Image_Constants.RIGHT_EYE_OUT_ID,
                constants.Image_Constants.RIGHT_EYE_INSIDE_ID,
                constants.Image_Constants.RIGHT_EYE_UP_ID,
                constants.Image_Constants.RIGHT_EYE_LOW_ID,
            )
            ratio_avg = (ratio_L + ratio_R) / 2

            # eye patches
            pts_left = np.array(
                [
                    face[id]
                    for id in [
                        constants.Image_Constants.LEFT_EYE_OUT_ID,
                        constants.Image_Constants.LEFT_EYE_INSIDE_ID,
                        constants.Image_Constants.LEFT_EYE_UP_ID,
                        constants.Image_Constants.LEFT_EYE_LOW_ID,
                    ]
                ],
                np.int32,
            )
            pts_right = np.array(
                [
                    face[id]
                    for id in [
                        constants.Image_Constants.RIGHT_EYE_OUT_ID,
                        constants.Image_Constants.RIGHT_EYE_INSIDE_ID,
                        constants.Image_Constants.RIGHT_EYE_UP_ID,
                        constants.Image_Constants.RIGHT_EYE_LOW_ID,
                    ]
                ],
                np.int32,
            )
            patch_left = eye_patch(img, pts_left)
            patch_right = eye_patch(img, pts_right)
            pixels_left = patch_left.flatten().tolist()
            pixels_right = patch_right.flatten().tolist()

            # HUD
            cv.putText(
                img,
                f"Blink #: {blink_count}",
                (20, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
            )
            cv.putText(
                img,
                f"EAR L/R: {ratio_L:.2f}/{ratio_R:.2f}",
                (20, 65),
                cv.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
            )
            if manual_blink:
                cv.putText(
                    img,
                    "BLINK (label=1)",
                    (20, 95),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # plot & show
            plot_img = plot.update(ratio_avg)
            show_stack = cvz.stackImages([cv.resize(img, (640, 360)), plot_img], 2, 1)
            cv.imshow("Blink Recorder", show_stack)
            cv.imshow("Left Eye", patch_left)
            cv.imshow("Right Eye", patch_right)

            # ---------- record row ----------
            row = dict(
                timestamp=timestamp,
                ratio_left=ratio_L,
                ratio_right=ratio_R,
                ratio_avg=ratio_avg,
                v_left=vL,
                h_left=hL,
                v_right=vR,
                h_right=hR,
                blink_count=blink_count,
                manual_blink=manual_blink,
            )
            row.update({f"px_l_{i}": pix for i, pix in enumerate(pixels_left)})
            row.update({f"px_r_{i}": pix for i, pix in enumerate(pixels_right)})
            data_rows.append(row)

        else:
            # still write a row to keep timeline (pixels = NaNs)
            data_rows.append(
                dict(
                    timestamp=timestamp,
                    ratio_left=None,
                    ratio_right=None,
                    ratio_avg=None,
                    v_left=None,
                    h_left=None,
                    v_right=None,
                    h_right=None,
                    blink_count=blink_count,
                    manual_blink=manual_blink,
                    **{
                        f"px_l_{i}": None
                        for i in range(
                            constants.Image_Constants.IM_WIDTH
                            * constants.Image_Constants.IM_HEIGHT
                        )
                    },
                    **{
                        f"px_r_{i}": None
                        for i in range(
                            constants.Image_Constants.IM_WIDTH
                            * constants.Image_Constants.IM_HEIGHT
                        )
                    },
                )
            )
            cv.imshow("Blink Recorder", cv.resize(img, (640, 360)))
            cv.imshow("Left Eye", np.zeros((HEIGHT_IM, WIDTH_IM), np.uint8))
            cv.imshow("Right Eye", np.zeros((HEIGHT_IM, WIDTH_IM), np.uint8))

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        prev_keypress = keypress_now

finally:
    if data_rows:
        pd.DataFrame(data_rows).to_csv(CSV_NAME, index=False)
        print(f"Saved {len(data_rows)} rows → {CSV_NAME}")
    cap.release()
    cv.destroyAllWindows()
