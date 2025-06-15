from __future__ import annotations

import json
from pathlib import Path
import time

import pyautogui


DEFAULT_PAUSE = 0.1


def run(folder: str | Path) -> None:
    """Load ``macro.json`` from ``folder`` and execute the operations."""
    folder = Path(folder)
    data = json.loads((folder / "macro.json").read_text())
    for op in data.get("ops", []):
        kind = op.get("op")
        if kind == "left_click":
            pyautogui.click(button=op.get("button", "left"))
        elif kind == "move":
            pyautogui.moveTo(op.get("x", 0), op.get("y", 0), duration=op.get("duration", 0))
        elif kind == "find_image":
            img = folder / op["image"]
            loc = pyautogui.locateCenterOnScreen(
                str(img),
                confidence=op.get("confidence", 0.8),
                grayscale=op.get("grayscale", True),
            )
            if loc:
                pyautogui.moveTo(loc)
        time.sleep(op.get("pause", DEFAULT_PAUSE))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python deparse.py <macro_folder>")
    else:
        run(sys.argv[1])

