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
EYE_IMAGE_SIZE = 50  # 50x50 pixels
CSV_NAME = f"eye_image_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"

# Initialize video capture
cap = cv.VideoCapture(0)
# Fix camera warping
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) / 2)
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) / 2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

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
    """Extract a square eye image based on landmarks with padding"""
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
        
    # Crop the eye region
    eye_img = img[y_min:y_max, x_min:x_max]
    
    # Calculate desired dimensions to keep aspect ratio
    h, w = eye_img.shape[:2]
    if h > w:
        new_h = EYE_IMAGE_SIZE
        new_w = int(w * (EYE_IMAGE_SIZE / h))
    else:
        new_w = EYE_IMAGE_SIZE
        new_h = int(h * (EYE_IMAGE_SIZE / w))
    
    # Resize while maintaining aspect ratio
    eye_img = cv.resize(eye_img, (new_w, new_h))
    
    # Create a black background image
    square_img = np.zeros((EYE_IMAGE_SIZE, EYE_IMAGE_SIZE, 3), dtype=np.uint8)
    
    # Calculate offsets to center the eye image
    x_offset = (EYE_IMAGE_SIZE - new_w) // 2
    y_offset = (EYE_IMAGE_SIZE - new_h) // 2
    
    # Place the resized eye image onto the black background
    square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = eye_img
    
    return square_img

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
        manual_blink = int(key_now and not prev_key)
        if manual_blink:
            blink_count += 1
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
                # Show the eye image
                eye_display = cv.resize(eye_img, (100, 100))  # Make it bigger for display
                img[20:120, 20:120] = eye_display
                
                # Create a row for the CSV
                # First value is blink status
                row = {"manual_blink": manual_blink}
                
                # Flatten the RGB eye image (50x50x3 = 7500 values)
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