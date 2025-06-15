from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Dict


class Macro:
    """Helper to build and save macro definitions."""

    def __init__(self, name: str = "macro"):
        self.name = name
        self.ops: List[Dict] = []
        self._assets: Dict[str, Path] = {}

    # --- building blocks -------------------------------------------------
    def left_click(self) -> "Macro":
        """Add a left mouse click operation."""
        self.ops.append({"op": "left_click"})
        return self

    def right_click(self) -> "Macro":
        """Add a right mouse click operation."""
        self.ops.append({"op": "right_click"})
        return self

    def middle_click(self) -> "Macro":
        """Add a middle mouse click operation."""
        self.ops.append({"op": "middle_click"})
        return self

    def move(self, x: int, y: int, duration: float = 0.0) -> "Macro":
        """Move the cursor to absolute ``(x, y)`` over ``duration`` seconds."""
        self.ops.append({"op": "move", "x": x, "y": y, "duration": duration})
        return self

    def move_by(self, dx: int, dy: int, duration: float = 0.0) -> "Macro":
        """Move the cursor by ``(dx, dy)`` over ``duration`` seconds."""
        self.ops.append({"op": "move_by", "dx": dx, "dy": dy, "duration": duration})
        return self

    def move_percent(self, px: float, py: float, duration: float = 0.0) -> "Macro":
        """Move the cursor to ``px``/``py`` (0-1) of the screen size."""
        self.ops.append({"op": "move_percent", "px": px, "py": py, "duration": duration})
        return self

    def find_image(self, image: str, confidence: float = 0.8, grayscale: bool = True) -> "Macro":
        img_path = Path(image)
        self.ops.append({
            "op": "find_image",
            "image": img_path.name,
            "confidence": confidence,
            "grayscale": grayscale,
        })
        self._assets[img_path.name] = img_path
        return self

    # --- serialisation ----------------------------------------------------
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name, src in self._assets.items():
            dst = folder / name
            if src.resolve() != dst.resolve():
                shutil.copy(src, dst)
        data = {"name": self.name, "ops": self.ops}
        path = folder / "macro.json"
        path.write_text(json.dumps(data, indent=2))
        return path


