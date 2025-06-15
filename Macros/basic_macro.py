import pyautogui
from pathlib import Path

BASE_DIR = Path(__file__).parent
img_path = BASE_DIR / "yt_logo.png"

location = pyautogui.locateOnScreen(
    str(img_path),
    confidence=0.8,     # 0 – 1, lower = more tolerant
    grayscale=True      # often helps by ignoring color shifts
)

if location:
  screen_width, screen_height = pyautogui.size()
  x, y = pyautogui.center(location)
  y_offset = int(screen_height * 0.10)
  pyautogui.click(x, y + y_offset)


print(location or "Logo not found")
