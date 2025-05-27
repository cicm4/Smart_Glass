# Video Blink Detection Test
# The objective of this test is to verify if the video blink detection works correctly.

import cv2 as cv
import cvzone as cvz
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

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

    if ratio_avg > blink_threshold and counter == 0:
      blink_count += 1
      counter = 1
    if counter > 0:
      counter += 1
      if counter > 10:
        counter = 0
    
    cv.putText(img, f'Blink Count: {blink_count}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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