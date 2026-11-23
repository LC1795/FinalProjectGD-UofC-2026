"""Export checked scene layers or individual objects."""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QTreeWidget,
    QTreeWidgetItem,QComboBox,QPushButton,QLabel,QFileDialog,QMessageBox)
from soundform.export.selection_export import export_selection


class ExportPanelMixin:
    def _build_export_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        note = QLabel("Check whole layers or individual objects to export. Coordinates use local site metres.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.export_tree = QTreeWidget()
        self.export_tree.setHeaderLabels(["Layers / objects to export"])
        self.export_tree.setMinimumHeight(140)
        layout.addWidget(self.export_tree)
        row = QHBoxLayout()
        for label, slot in (("All",lambda:self._check_exports(True)),
                            ("None",lambda:self._check_exports(False)),
                            ("Use selection",self._check_selected_exports)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)
        self.export_format = QComboBox()
        for label,suffix in (("Rhino (.3dm)",".3dm"),("Wavefront OBJ (.obj)",".obj"),("STL surfaces (.stl)",".stl")):
            self.export_format.addItem(label,suffix)
        layout.addWidget(self.export_format)
        button = QPushButton("Export checked objects")
        button.clicked.connect(self._export_checked)
        layout.addWidget(button)
        return panel

    def _export_rows(self):
        for i in range(self.export_tree.topLevelItemCount()):
            root = self.export_tree.topLevelItem(i)
            for j in range(root.childCount()):
                yield root.child(j)

    def _refresh_export_tree(self):
        if not hasattr(self,"export_tree"):
            return
        checked = {node.data(0,Qt.UserRole+1) for node in self._export_rows() if node.checkState(0)==Qt.Checked}
        self.export_tree.clear()
        for i in range(self.layers_tree.topLevelItemCount()):
            layer = self.layers_tree.topLevelItem(i)
            root = QTreeWidgetItem(self.export_tree,[layer.text(0)])
            root.setFlags(root.flags()|Qt.ItemIsUserCheckable|Qt.ItemIsAutoTristate)
            root.setCheckState(0,Qt.Unchecked)
            for j in range(layer.childCount()):
                source = layer.child(j)
                data = source.data(0,Qt.UserRole)
                if not data:
                    continue
                key = (layer.text(0),self.viewport.object_visibility_key(data))
                node = QTreeWidgetItem(root,[source.text(0)])
                node.setData(0,Qt.UserRole,data)
                node.setData(0,Qt.UserRole+1,key)
                node.setFlags(node.flags()|Qt.ItemIsUserCheckable)
                node.setCheckState(0,Qt.Checked if key in checked else Qt.Unchecked)
            root.setExpanded(True)

    def _check_exports(self,checked):
        for node in self._export_rows():
            node.setCheckState(0,Qt.Checked if checked else Qt.Unchecked)

    def _check_selected_exports(self):
        keys = set()
        for node in self.layers_tree.selectedItems():
            for data in self._layer_entries(node):
                if data:
                    keys.add(self.viewport.object_visibility_key(data))
        if self._selected_object:
            keys.add(self.viewport.object_visibility_key(self._selected_object))
        for node in self._export_rows():
            data = node.data(0,Qt.UserRole)
            node.setCheckState(0,Qt.Checked if self.viewport.object_visibility_key(data) in keys else Qt.Unchecked)

    def _checked_export_datasets(self):
        items = [node.data(0,Qt.UserRole) for node in self._export_rows() if node.checkState(0)==Qt.Checked]
        return self._selection_datasets(items)

    def _export_checked(self):
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Wait for the current task before exporting.")
            return
        datasets = self._checked_export_datasets()
        if not datasets:
            QMessageBox.information(self,"Nothing checked","Check at least one layer or object to export.")
            return
        suffix = self.export_format.currentData()
        path,_ = QFileDialog.getSaveFileName(self,"Export checked objects","soundform-export"+suffix,
            self.export_format.currentText()+" (*"+suffix+")")
        if not path:
            return
        if Path(path).suffix.lower()!=suffix:
            path = str(Path(path).with_suffix(suffix))
        try:
            result = export_selection(datasets,path)
        except Exception as error:
            QMessageBox.warning(self,"Export failed",str(error))
            return
        self.statusBar().showMessage(f"Exported {result['objects']} object(s) to {result['path']}.")
