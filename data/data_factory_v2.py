import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2 as cv
import numpy as np
import pandas as pd
import keyboard
from cvzone.FaceMeshModule import FaceMeshDetector
import constants

# Constants
# Use a compact grayscale eye patch to keep file sizes small
EYE_W = constants.Image_Constants.IM_WIDTH
EYE_H = constants.Image_Constants.IM_HEIGHT
CSV_NAME = f"eye_image_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
FPS = 20

# Initialize video capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, FPS)
# Fix camera warping
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) / 2)
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) / 2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
os.makedirs("data", exist_ok=True)

# Left eye landmarks - these will create a bounding box around the eye
LEFT_EYE_LANDMARKS = constants.Image_Constants.LEFT_EYE_IDS

# Initialize face mesh detector
detector = FaceMeshDetector(maxFaces=1)

# Data collection variables
data_rows = []
blink_count = 0
prev_key = False
t0 = time.time()

# Helper function to extract eye image
def extract_eye_image(img, landmarks, padding=5):
    """Return a normalised grayscale eye patch."""
    x_coords = [landmarks[i][0] for i in LEFT_EYE_LANDMARKS]
    y_coords = [landmarks[i][1] for i in LEFT_EYE_LANDMARKS]
    
    # Calculate bounding box with padding
    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = min(img.shape[1], int(max(x_coords)) + padding)
    y_max = min(img.shape[0], int(max(y_coords)) + padding)
    
    # Handle invalid bounding boxes
    if x_min >= x_max or y_min >= y_max:
        return None
        
    # Crop the eye region and convert to grayscale
    eye_img = img[y_min:y_max, x_min:x_max]
    eye_img = cv.cvtColor(eye_img, cv.COLOR_BGR2GRAY)

    # Resize to the target patch size
    eye_img = cv.resize(eye_img, (EYE_W, EYE_H))

    return eye_img

# Display instructions
print("Press SPACE to annotate a blink")
print("Press Q to quit and save data")

try:
    while True:
        ok, img = cap.read()
        if not ok:
            break

        img, faces = detector.findFaceMesh(img, draw=False)
        
        # Check for space key press (manual blink annotation)
        key_now = keyboard.is_pressed("space")
        if key_now and not prev_key:
            blink_count += 1
            manual_blink = 1
            cv.putText(
                img,
                "Blink detected!",
                (50, 100),
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4,
                cv.LINE_AA,
            )
        else:
            manual_blink = int(key_now)

        timestamp = time.time() - t0

        if faces:
            face = faces[0]
            
            # Extract eye image
            eye_img = extract_eye_image(img, face)
            
            # Draw landmarks for visualization
            for pid in LEFT_EYE_LANDMARKS:
                cv.circle(img, face[pid], 3, (255, 0, 255), cv.FILLED)
            
            # Show the extracted eye image
            if eye_img is not None:
                eye_display = cv.resize(eye_img, (100, 100))
                eye_display = cv.cvtColor(eye_display, cv.COLOR_GRAY2BGR)
                img[20:120, 20:120] = eye_display
                
                # Create a row for the CSV
                # First value is blink status
                row = {"manual_blink": manual_blink}
                
                # Flatten the grayscale eye image (32x16 = 512 values)
                flat_img = eye_img.flatten()
                
                # Add each pixel value to the row
                for i, pixel_value in enumerate(flat_img):
                    row[f"pixel_{i}"] = int(pixel_value)
                
                data_rows.append(row)
        # Display current frame count and blink count
        cv.putText(
            img,
            f"Frames: {len(data_rows)} | Blinks: {blink_count}",
            (20, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        
        cv.imshow("Eye Data Collection", img)
        
        # End the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
            
        prev_key = key_now

finally:
    if data_rows:
        os.makedirs("data", exist_ok=True)
        
        print(f"Saving {len(data_rows)} frames to CSV...")
        
        # Save data to CSV
        df = pd.DataFrame(data_rows)
        df.to_csv(os.path.join("data", CSV_NAME), index=False)
        
        print(f"Data saved to {CSV_NAME}")
        print(f"Total blinks recorded: {blink_count}")
        print(f"CSV shape: {df.shape}")
    
    cap.release()
    cv.destroyAllWindows()