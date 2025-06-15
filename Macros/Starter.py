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
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex
from pathlib import Path
import json
from parser import Macro
from deparse import run as run_macro


class MacroDialog(QDialog):
    """Simple dialog to build a macro by adding operation blocks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Macro")
        self.macro = Macro("Untitled")

        self.name_edit = QLineEdit(self.macro.name)
        self.list = QListWidget()
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

    # ------------------------------------------------------------------
    def add_click(self) -> None:
        self.macro.left_click()
        self.list.addItem("Left Click")

    def add_right_click(self) -> None:
        self.macro.right_click()
        self.list.addItem("Right Click")

    def add_middle_click(self) -> None:
        self.macro.middle_click()
        self.list.addItem("Middle Click")

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
        self.macro.move(x, y, duration=speed)
        self.list.addItem(f"Move to ({x}, {y}) in {speed}s")

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
        self.macro.move_by(dx, dy, duration=speed)
        self.list.addItem(f"Move by ({dx}, {dy}) in {speed}s")

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
        self.macro.move_percent(px, py, duration=speed)
        self.list.addItem(f"Move to {px*100:.0f}%, {py*100:.0f}% in {speed}s")

    def add_find(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", str(Path.cwd()), "Images (*.png *.jpg *.bmp)"
        )
        if not path:
            return
        self.macro.find_image(path)
        self.list.addItem(f"Find image {Path(path).name}")

    def on_accept(self) -> None:
        self.macro.name = self.name_edit.text() or "Untitled"
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
        run_act = QAction("Run", self)
        run_act.triggered.connect(self.run_macro)
        save_act = QAction("Save", self)
        save_act.triggered.connect(self.save_profile)
        tb.addActions([folder_act, new_act, run_act, save_act])

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
