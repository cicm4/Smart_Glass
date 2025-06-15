from PySide6.QtWidgets import (
    QApplication, QMainWindow, QListView, QStackedWidget,
    QFileDialog, QToolBar, QDialog, QVBoxLayout, QPushButton,
    QListWidget, QInputDialog
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex
from pathlib import Path
from parser import Macro


class MacroDialog(QDialog):
    """Simple dialog to build a macro by adding operation blocks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Macro")
        self.macro = Macro("Untitled")

        self.list = QListWidget()
        btn_click = QPushButton("Add Left Click")
        btn_move = QPushButton("Add Mouse Move")
        btn_find = QPushButton("Add Find Image")
        btn_done = QPushButton("Done")

        layout = QVBoxLayout(self)
        layout.addWidget(self.list)
        layout.addWidget(btn_click)
        layout.addWidget(btn_move)
        layout.addWidget(btn_find)
        layout.addWidget(btn_done)

        btn_click.clicked.connect(self.add_click)
        btn_move.clicked.connect(self.add_move)
        btn_find.clicked.connect(self.add_find)
        btn_done.clicked.connect(self.accept)

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
        self.stack = QStackedWidget()

        self.setCentralWidget(self.view)         # swap to splitter later
        self._build_toolbar()

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        self.addToolBar(tb)
        new_act = QAction("New Macro", self)
        new_act.triggered.connect(self.new_macro)
        save_act = QAction("Save", self)
        save_act.triggered.connect(self.save_profile)
        tb.addActions([new_act, save_act])

    def new_macro(self):
        dlg = MacroDialog(self)
        if not dlg.exec():
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose Macro Folder", str(Path.cwd()))
        if not folder:
            return
        dlg.macro.save(folder)
        self.model._data.append({"name": dlg.macro.name, "path": folder})
        self.model.layoutChanged.emit()

    def save_profile(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save profile", ".", "JSON (*.json)")
        if not path: return
        import json, pathlib
        pathlib.Path(path).write_text(json.dumps(self.model._data, indent=2))

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow(); win.resize(800, 500); win.show()
    sys.exit(app.exec())
