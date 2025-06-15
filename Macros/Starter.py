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
        btn_move = QPushButton("Add Mouse Move")
        btn_find = QPushButton("Add Find Image")
        btn_done = QPushButton("Done")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Macro name:"))
        layout.addWidget(self.name_edit)
        layout.addWidget(self.list)
        layout.addWidget(btn_click)
        layout.addWidget(btn_move)
        layout.addWidget(btn_find)
        layout.addWidget(btn_done)

        btn_click.clicked.connect(self.add_click)
        btn_move.clicked.connect(self.add_move)
        btn_find.clicked.connect(self.add_find)
        btn_done.clicked.connect(self.on_accept)

    # ------------------------------------------------------------------
    def add_click(self) -> None:
        self.macro.left_click()
        self.list.addItem("Left Click")

    def add_move(self) -> None:
        x, ok = QInputDialog.getInt(self, "Move", "X coordinate:")
        if not ok:
            return
        y, ok = QInputDialog.getInt(self, "Move", "Y coordinate:")
        if not ok:
            return
        self.macro.move(x, y)
        self.list.addItem(f"Move to ({x}, {y})")

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
