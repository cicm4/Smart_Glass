"""
Robust macro executor for Smart Glasses
────────────────────────────────────────────────────────────────────────────
•   Missing images → logged + skipped, not fatal
•   All operations wrapped in try/except to continue on error
•   Uses pyautogui for GUI automation; FAILSAFE disabled to avoid accidental aborts
"""

from __future__ import annotations
import json, time, logging
from pathlib import Path

import pyautogui

log = logging.getLogger("macro")
pyautogui.FAILSAFE = False


# ────── helpers ───────────────────────────────────────────────────────────
def _safe_locate(img_path: Path, **kw):
    """Locate image on screen; return None if not found or any error occurs."""
    if not img_path.exists():
        log.warning("Image %s not found – skipping op.", img_path.name)
        return None
    try:
        return pyautogui.locateCenterOnScreen(str(img_path), **kw)
    except Exception as e:  # screen-capture errors, etc.
        log.warning("Screen-capture failed: %s – skipping op.", e)
        return None


# ────── public API ────────────────────────────────────────────────────────
def run(folder: str | Path, *, default_pause: float = 0.1) -> None:
    """
    Execute macro defined in <folder>/macro.json.
    Macro JSON format:
    {
        "name": "Pause YouTube",
        "ops": [
            {"op": "left_click"},
            {"op": "find_image", "image": "play.png", "confidence": 0.9},
            {"op": "move", "x": 300, "y": 400, "duration": 0.2},
            {"op": "pause", "time": 1.5}
        ]
    }
    Unknown or invalid operations are logged and ignored.
    """

    folder = Path(folder)
    try:
        data = json.loads((folder / "macro.json").read_text())
    except (OSError, json.JSONDecodeError) as ex:
        raise RuntimeError(f"Cannot load macro in {folder}: {ex}") from ex

    for op in data.get("ops", []):
        kind = op.get("op", "").lower()
        try:
            # ── mouse clicks ─────────────────────────────────────────
            if kind == "left_click":
                pyautogui.click(button="left")
            elif kind == "right_click":
                pyautogui.click(button="right")
            elif kind == "middle_click":
                pyautogui.click(button="middle")

            # ── absolute / relative moves ───────────────────────────
            elif kind == "move":
                pyautogui.moveTo(
                    op.get("x", 0),
                    op.get("y", 0),
                    duration=op.get("duration", 0),
                )
            elif kind == "move_by":
                pyautogui.moveRel(
                    op.get("dx", 0),
                    op.get("dy", 0),
                    duration=op.get("duration", 0),
                )
            elif kind == "move_percent":
                w, h = pyautogui.size()
                px = int(w * op.get("px", 0))
                py = int(h * op.get("py", 0))
                pyautogui.moveTo(px, py, duration=op.get("duration", 0))

            # ── locate image & move ─────────────────────────────────
            elif kind == "find_image":
                loc = _safe_locate(
                    folder / op["image"],
                    confidence=op.get("confidence", 0.8),
                    grayscale=op.get("grayscale", True),
                )
                if loc:
                    pyautogui.moveTo(loc)
                else:
                    log.info("Image %s not on screen – continuing.", op["image"])

            # ── pauses ─────────────────────────────────────────────
            elif kind == "pause":
                time.sleep(op.get("time", 0))

            # ── fallback ────────────────────────────────────────────
            else:
                log.warning("Unknown op '%s' – skipped.", kind)

        except Exception as ex:
            log.error("Op %s failed: %s – continuing macro.", kind, ex)

        # optional inter-op pause
        time.sleep(op.get("pause", default_pause))
