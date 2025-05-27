import os
import numpy as np
import pandas as pd
import time
import cv2 as cv
import cvzone as cvz
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

# Initialize the video capture
cap = cv.VideoCapture(0)

# Initialize the FaceMeshDetector
detector = FaceMeshDetector(maxFaces=1)

plot_y = LivePlot(640, 360, [20, 40],)

# List of eye landmarks
eye_id_list = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratio_list = []
counter = 0

# Create a list to store all data points
data_records = []
timestamp_start = time.time()

# CSV file configuration
csv_filename = f"blink_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"

blink_count = 0
blink_threshold = 32  # Threshold for blink detection

# Add manual blink annotation variables
manual_blink = False
blink_timestamp = 0
blink_display_time = 1.0  # How long to show "BLINK" on screen (seconds)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to get frame")
            break
            
        img, faces = detector.findFaceMesh(img, draw=True)
        
        # Check for space key press (manual blink annotation)
        key = cv.waitKey(25) & 0xFF
        if key == 32:  # 32 is the space key
            manual_blink = True
            blink_timestamp = time.time()
            blink_count += 1
            print("Manual blink annotated!")
        
        # Check if blink display time has passed
        if manual_blink and (time.time() - blink_timestamp > blink_display_time):
            manual_blink = False

        current_time = time.time() - timestamp_start

        if faces:
            face = faces[0]
            timestamp = current_time

            for id in eye_id_list:
                cv.circle(img, face[id], 5, (255, 0, 255), cv.FILLED)
            
            left_up = face[159]
            left_down = face[23]
            left_left = face[130]
            left_right = face[243]

            distance_horizontal, _ = detector.findDistance(left_up, left_down)
            distance_vertical, _ = detector.findDistance(left_left, left_right)

            cv.line(img, left_up, left_down, (0, 200, 0), 3)
            cv.line(img, left_left, left_right, (0, 200, 0), 3)
            
            ratio = (distance_vertical/distance_horizontal) * 10
            ratio_list.append(ratio)
            if len(ratio_list) > 3:
                ratio_list.pop(0)
            
            ratio_avg = sum(ratio_list) / len(ratio_list)
            
            # Store data for this frame
            data_records.append({
                'timestamp': timestamp,
                'ratio': ratio,
                'ratio_avg': ratio_avg,
                'distance_vertical': distance_vertical,
                'distance_horizontal': distance_horizontal,
                'blink_count': blink_count,
                'manual_blink': 1 if manual_blink else 0  # Add manual blink status (1 or 0)
            })
            
            cv.putText(img, f'Blink Count: {blink_count}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(img, f'Ratio: {ratio:.2f}', (50, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(img, f'Avg Ratio: {ratio_avg:.2f}', (50, 130), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display BLINK annotation when active
            if manual_blink:
                cv.putText(img, "BLINK ANNOTATED", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 
                           1.5, (0, 0, 255), 3)

            img_plot = plot_y.update(ratio_avg)
            img = cv.resize(img, (640, 360))
            img_stack = cvz.stackImages([img, img_plot], 2, 1)
            cv.imshow("Image", img_stack)
        else:
            # Even if no face is detected, still record blink data
            data_records.append({
                'timestamp': current_time,
                'ratio': None,
                'ratio_avg': None,
                'distance_vertical': None,
                'distance_horizontal': None,
                'blink_count': blink_count,
                'manual_blink': 1 if manual_blink else 0
            })
            
            img = cv.resize(img, (640, 360))
            # Display BLINK annotation when active
            if manual_blink:
                cv.putText(img, "BLINK ANNOTATED", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 
                           1.5, (0, 0, 255), 3)
            cv.imshow("Image", img)

        # Instructions
        instructions = np.zeros((100, 640, 3), dtype=np.uint8)
        cv.putText(instructions, "Press SPACE to annotate a blink", (20, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(instructions, "Press Q to quit and save data", (20, 70), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.imshow("Instructions", instructions)

        # End the loop if 'q' is pressed
        if key == ord('q'):
            break

finally:
    # Save all collected data to CSV
    if data_records:
        print(f"Saving data to {csv_filename}...")
        df = pd.DataFrame(data_records)
        df.to_csv(csv_filename, index=False)
        print(f"Data saved: {len(data_records)} records")
    
    cap.release()
    cv.destroyAllWindows()