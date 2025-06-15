from PySide6.QtWidgets import (
    QApplication, QMainWindow, QListView, QStackedWidget,
    QFileDialog, QToolBar
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, Signal

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
        new_act = QAction("New Macro", self, triggered=self.new_macro)
        save_act = QAction("Save", self, triggered=self.save_profile)
        tb.addActions([new_act, save_act])

    def new_macro(self):
        self.model._data.append({"name": "Untitled"})
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
