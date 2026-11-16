"""Controls for selecting and editing a generated pavilion."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from soundform.geometry.pavilion_edit import part_labels


class PavilionEditor(QWidget):
    selection_changed = Signal()
    apply_requested = Signal()
    delete_requested = Signal()
    undo_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.note = QLabel("Generate a pavilion to edit it.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        form = QFormLayout()
        self.selection = QComboBox()
        self.selection.currentIndexChanged.connect(self.selection_changed)
        form.addRow("Selection", self.selection)
        self.offset = []
        for axis in ("X", "Y", "Z"):
            control = self._number(-500, 500, 0, 0.1, " m")
            self.offset.append(control)
            form.addRow(f"Move {axis}", control)
        self.rotation = self._number(-360, 360, 0, 5, "°")
        self.scale = self._number(0.01, 10, 1, 0.1, "×")
        form.addRow("Rotate about Z", self.rotation)
        form.addRow("Uniform scale", self.scale)
        layout.addLayout(form)
        self.apply = QPushButton("Apply transform")
        self.delete = QPushButton("Delete selection")
        self.undo = QPushButton("Undo edit")
        for button, signal in ((self.apply, self.apply_requested), (self.delete, self.delete_requested),
                               (self.undo, self.undo_requested)):
            button.clicked.connect(signal)
            layout.addWidget(button)
        layout.addStretch()

    @staticmethod
    def _number(low, high, value, step, suffix):
        control = QDoubleSpinBox()
        control.setRange(low, high)
        control.setValue(value)
        control.setSingleStep(step)
        control.setSuffix(suffix)
        return control

    def show_mesh(self, mesh, can_undo=False):
        previous = self.selection.currentData()
        self.selection.blockSignals(True)
        self.selection.clear()
        if mesh is not None:
            self.selection.addItem("Whole pavilion", None)
            labels = sorted(set(part_labels(mesh)))
            for label in labels:
                if label != "Mesh":
                    self.selection.addItem(label, label)
            self.selection.setCurrentIndex(max(0, self.selection.findData(previous)))
        self.selection.blockSignals(False)
        self.note.setText("Select a part to highlight it. Transforms are relative to the current mesh."
                          if mesh is not None else "Generate a pavilion to edit it.")
        for control in (self.selection, *self.offset, self.rotation, self.scale, self.apply, self.delete):
            control.setEnabled(mesh is not None)
        for control in (*self.offset, self.rotation):
            control.setValue(0)
        self.scale.setValue(1)
        self.undo.setEnabled(can_undo)
