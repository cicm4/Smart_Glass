# Video Blink Detection Test
# The objective of this test is to verify if the video blink detection works correctly.

import cv2 as cv
import cvzone as cvz
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

import torch
import numpy as np
from blink_lstm_model import BlinkLSTMNet

# Load the trained LSTM model
model = BlinkLSTMNet()
model.load_state_dict(torch.load("blink_lstm_model.pth", map_location="cpu"))
model.eval()

SEQ_LEN = 20
feature_buffer = []

# Initialize the video capture
cap = cv.VideoCapture(0)

# Initialize the FaceMeshDetector
detector = FaceMeshDetector(maxFaces=1)

plot_y = LivePlot(640, 360, [20, 40],)

#list of eye landmarks
eye_id_list = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratio_list = []
counter = 0

blink_count = 0
blink_threshold = 32  # Threshold for blink detection
while True:

  success, img = cap.read()
  img, faces = detector.findFaceMesh(img, draw=False)

  if faces:
    face = faces[0]

    for id in eye_id_list:
      cv.circle(img, face[id], 5, (255, 0, 255), cv.FILLED)
    
    left_up = face[159]
    left_down = face[23]

    left_left = face[130]
    left_right = face[243]

    distance_horizontal,_ = detector.findDistance(left_up, left_down)
    distance_vertical,_ = detector.findDistance(left_left, left_right)

    cv.line(img, left_up, left_down, (0, 200, 0), 3)
    cv.line(img, left_left, left_right, (0, 200, 0), 3)
    
    ratio = (distance_vertical/distance_horizontal) * 10
    ratio_list.append(ratio)
    if len(ratio_list) > 3:
      ratio_list.pop(0)
    
    ratio_avg = sum(ratio_list) / len(ratio_list)

    features = [ratio, ratio_avg, distance_vertical, distance_horizontal]
    feature_buffer.append(features)
    if len(feature_buffer) > SEQ_LEN:
        feature_buffer.pop(0)

    if len(feature_buffer) == SEQ_LEN:
        x_seq = np.array(feature_buffer, dtype=np.float32).reshape(1, SEQ_LEN, 4)
        x_seq = torch.tensor(x_seq)
        with torch.no_grad():
            logits = model(x_seq)
            pred = torch.argmax(logits, dim=1).item()
        cv.putText(img, f'Model: {"Blink" if pred == 1 else "Not Blink"}', (50, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    img_plot = plot_y.update(ratio_avg)
    img = cv.resize(img, (640, 360))
    img_stack = cvz.stackImages([img, img_plot], 2, 1)
    cv.imshow("Image", img_stack)
  else:
    img = cv.resize(img, (640, 360))
    cv.imshow("Image", img)

  #end the loop if 'q' is pressed
  if cv.waitKey(25) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()  