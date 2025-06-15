from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QListView,
    QFileDialog,
    QToolBar,
    QDialog,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QInputDialog,
    QLineEdit,
    QLabel,
    QSplitter,
    QListWidgetItem,
    QWidget,
    QHBoxLayout,
    QAbstractItemView,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex
from pathlib import Path
import json
from parser import Macro
from deparse import run as run_macro


class MacroDialog(QDialog):
    """Dialog to create or edit a macro."""

    def __init__(self, parent=None, macro: Macro | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit Macro")
        self.macro = macro or Macro("Untitled")

        self.name_edit = QLineEdit(self.macro.name)
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.assets: dict[str, Path] = getattr(self.macro, "_assets", {}).copy()
        btn_click = QPushButton("Add Left Click")
        btn_right = QPushButton("Add Right Click")
        btn_middle = QPushButton("Add Middle Click")
        btn_move = QPushButton("Add Mouse Move")
        btn_move_rel = QPushButton("Add Move By")
        btn_move_percent = QPushButton("Add Move Percent")
        btn_find = QPushButton("Add Find Image")
        btn_done = QPushButton("Done")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Macro name:"))
        layout.addWidget(self.name_edit)
        layout.addWidget(self.list)
        layout.addWidget(btn_click)
        layout.addWidget(btn_right)
        layout.addWidget(btn_middle)
        layout.addWidget(btn_move)
        layout.addWidget(btn_move_rel)
        layout.addWidget(btn_move_percent)
        layout.addWidget(btn_find)
        layout.addWidget(btn_done)

        btn_click.clicked.connect(self.add_click)
        btn_right.clicked.connect(self.add_right_click)
        btn_middle.clicked.connect(self.add_middle_click)
        btn_move.clicked.connect(self.add_move)
        btn_move_rel.clicked.connect(self.add_move_rel)
        btn_move_percent.clicked.connect(self.add_move_percent)
        btn_find.clicked.connect(self.add_find)
        btn_done.clicked.connect(self.on_accept)

        for op in self.macro.ops:
            self._add_item(self._desc_from_op(op), op)

    # ------------------------------------------------------------------
    def _desc_from_op(self, op: dict) -> str:
        kind = op.get("op")
        if kind == "left_click":
            return "Left Click"
        if kind == "right_click":
            return "Right Click"
        if kind == "middle_click":
            return "Middle Click"
        if kind == "move":
            return f"Move to ({op.get('x')}, {op.get('y')}) in {op.get('duration', 0)}s"
        if kind == "move_by":
            return f"Move by ({op.get('dx')}, {op.get('dy')}) in {op.get('duration', 0)}s"
        if kind == "move_percent":
            px = op.get('px', 0) * 100
            py = op.get('py', 0) * 100
            return f"Move to {px:.0f}%, {py:.0f}% in {op.get('duration', 0)}s"
        if kind == "find_image":
            return f"Find image {op.get('image')}"
        return kind or "?"

    def _add_item(self, text: str, op: dict) -> None:
        item = QListWidgetItem()
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        lbl = QLabel(text)
        btn = QPushButton("X")
        btn.setFixedWidth(20)
        layout.addWidget(lbl)
        layout.addStretch()
        layout.addWidget(btn)
        item.setSizeHint(widget.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, widget)
        item.setData(Qt.ItemDataRole.UserRole, op)
        btn.clicked.connect(lambda: self.remove_item(item))

    def remove_item(self, item: QListWidgetItem) -> None:
        row = self.list.row(item)
        self.list.takeItem(row)

    # ------------------------------------------------------------------
    def add_click(self) -> None:
        op = {"op": "left_click"}
        self._add_item("Left Click", op)

    def add_right_click(self) -> None:
        op = {"op": "right_click"}
        self._add_item("Right Click", op)

    def add_middle_click(self) -> None:
        op = {"op": "middle_click"}
        self._add_item("Middle Click", op)

    def add_move(self) -> None:
        x, ok = QInputDialog.getInt(self, "Move", "X coordinate:")
        if not ok:
            return
        y, ok = QInputDialog.getInt(self, "Move", "Y coordinate:")
        if not ok:
            return
        speed, ok = QInputDialog.getDouble(
            self, "Move", "Speed (sec):", 0.0, 0.0, 10.0, decimals=2
        )
        if not ok:
            return
        op = {"op": "move", "x": x, "y": y, "duration": speed}
        self._add_item(f"Move to ({x}, {y}) in {speed}s", op)

    def add_move_rel(self) -> None:
        dx, ok = QInputDialog.getInt(self, "Move By", "dx:")
        if not ok:
            return
        dy, ok = QInputDialog.getInt(self, "Move By", "dy:")
        if not ok:
            return
        speed, ok = QInputDialog.getDouble(
            self, "Move By", "Speed (sec):", 0.0, 0.0, 10.0, decimals=2
        )
        if not ok:
            return
        op = {"op": "move_by", "dx": dx, "dy": dy, "duration": speed}
        self._add_item(f"Move by ({dx}, {dy}) in {speed}s", op)

    def add_move_percent(self) -> None:
        px, ok = QInputDialog.getDouble(
            self, "Move Percent", "X % (0-1):", 0.0, 0.0, 1.0, decimals=2
        )
        if not ok:
            return
        py, ok = QInputDialog.getDouble(
            self, "Move Percent", "Y % (0-1):", 0.0, 0.0, 1.0, decimals=2
        )
        if not ok:
            return
        speed, ok = QInputDialog.getDouble(
            self, "Move Percent", "Speed (sec):", 0.0, 0.0, 10.0, decimals=2
        )
        if not ok:
            return
        op = {"op": "move_percent", "px": px, "py": py, "duration": speed}
        self._add_item(f"Move to {px*100:.0f}%, {py*100:.0f}% in {speed}s", op)

    def add_find(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", str(Path.cwd()), "Images (*.png *.jpg *.bmp)"
        )
        if not path:
            return
        op = {"op": "find_image", "image": Path(path).name, "confidence": 0.8, "grayscale": True}
        self.assets[Path(path).name] = Path(path)
        self._add_item(f"Find image {Path(path).name}", op)

    def on_accept(self) -> None:
        ops: list[dict] = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            op = item.data(Qt.ItemDataRole.UserRole)
            if op:
                ops.append(op)
        self.macro.name = self.name_edit.text() or "Untitled"
        self.macro.ops = ops
        self.macro._assets = self.assets
        self.accept()


class MacroModel(QAbstractListModel):
    def __init__(self, macros: list[dict], parent=None):
        super().__init__(parent)
        self._data = macros      # list of dicts

    def rowCount(self, parent=QModelIndex()): return len(self._data)
    def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[idx.row()]["name"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Glasses Macro Creator")
        self.view = QListView()
        self.model = MacroModel([])
        self.view.setModel(self.model)
        self.view.doubleClicked.connect(self.edit_macro)
        self.splitter = QSplitter()
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(QListWidget())  # placeholder for future widgets

        self.setCentralWidget(self.splitter)

        self.macro_dir: Path | None = None
        self._build_toolbar()

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        self.addToolBar(tb)
        folder_act = QAction("Choose Folder", self)
        folder_act.triggered.connect(self.choose_folder)
        new_act = QAction("New Macro", self)
        new_act.triggered.connect(self.new_macro)
        edit_act = QAction("Edit", self)
        edit_act.triggered.connect(self.edit_macro)
        run_act = QAction("Run", self)
        run_act.triggered.connect(self.run_macro)
        save_act = QAction("Save", self)
        save_act.triggered.connect(self.save_profile)
        tb.addActions([folder_act, new_act, edit_act, run_act, save_act])

    def new_macro(self):
        dlg = MacroDialog(self)
        if not dlg.exec():
            return
        if self.macro_dir is None:
            self.choose_folder()
            if self.macro_dir is None:
                return
        macro_folder = self.macro_dir / dlg.macro.name
        dlg.macro.save(macro_folder)
        self.load_macros(self.macro_dir)

    def edit_macro(self):
        idx = self.view.currentIndex().row()
        if idx < 0:
            return
        folder = Path(self.model._data[idx]["path"])
        data = json.loads((folder / "macro.json").read_text())
        macro = Macro(data.get("name", folder.name))
        macro.ops = data.get("ops", [])
        macro._assets = {op.get("image"): folder / op.get("image") for op in macro.ops if op.get("op") == "find_image"}
        dlg = MacroDialog(self, macro)
        if not dlg.exec():
            return
        dlg.macro.save(folder)
        self.load_macros(self.macro_dir)

    def save_profile(self):
        if self.macro_dir is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save profile", ".", "JSON (*.json)")
        if not path:
            return
        data = {"folder": str(self.macro_dir)}
        Path(path).write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Macro Folder", str(Path.cwd()))
        if not folder:
            return
        self.load_macros(Path(folder))

    def load_macros(self, folder: Path):
        self.macro_dir = folder
        self.model._data.clear()
        for sub in folder.iterdir():
            if (sub / "macro.json").exists():
                try:
                    data = json.loads((sub / "macro.json").read_text())
                    name = data.get("name", sub.name)
                except Exception:
                    name = sub.name
                self.model._data.append({"name": name, "path": str(sub)})
        self.model.layoutChanged.emit()

    def run_macro(self):
        idx = self.view.currentIndex().row()
        if idx < 0:
            return
        folder = self.model._data[idx]["path"]
        run_macro(folder)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow(); win.resize(800, 500); win.show()
    sys.exit(app.exec())
