"""Prompt, sketch and editable geometry for the experimental pavilion workflow."""

import base64
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QLabel,
    QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from soundform.geometry.proposal import OPERATIONS


MAX_SKETCHES = 5


class ProposalPanel(QWidget):
    generate_requested = Signal()
    dimensions_changed = Signal()

    def __init__(self):
        super().__init__()
        self._sketches = []
        self.proposal = None
        self.applied_threshold = None
        self.controls = []
        layout = QVBoxLayout(self)
        layout.addWidget(self._label("Claude Opus receives the brief, sketches, directional sound summary and a preview of the built proposal."))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Describe your design intent and requirements.")
        self.prompt.setMaximumHeight(105)
        layout.addWidget(self.prompt)
        self.attach_sketches = QPushButton()
        self.attach_sketches.clicked.connect(self._choose_sketches)
        layout.addWidget(self.attach_sketches)
        self.sketch_list = QComboBox()
        self.sketch_list.currentIndexChanged.connect(self._show_sketch)
        layout.addWidget(self.sketch_list)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview)
        self.clear_sketch = QPushButton("Remove selected sketch")
        self.clear_sketch.clicked.connect(self._remove_sketch)
        layout.addWidget(self.clear_sketch)
        self._show_sketch()
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-80, 0)
        self.threshold.setValue(-12)
        self.threshold.setSuffix(" dB")
        form = QFormLayout()
        form.addRow("Uncomfortable above", self.threshold)
        layout.addLayout(form)
        layout.addWidget(self._label("Relative to the loudest window in the analyzed interval."))
        self.generate = QPushButton("Generate proposal")
        self.generate.clicked.connect(self.generate_requested)
        layout.addWidget(self.generate)
        self.result = self._label("")
        layout.addWidget(self.result)
        self.rationale = self._label("")
        layout.addWidget(self.rationale)
        self.dimensions = QVBoxLayout()
        layout.addLayout(self.dimensions)
        layout.addStretch()

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.PlainText)
        return label

    def sketch_images(self) -> tuple[str, ...]:
        return tuple(data_url for _, data_url, _ in self._sketches)

    def _choose_sketches(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Attach concept sketches", "", "Sketch images (*.png *.jpg *.jpeg)"
        )
        problems = []
        attached = {source for source, _, _ in self._sketches}
        last_added = None
        for path in paths:
            source = Path(path).resolve()
            if source in attached:
                continue
            if len(self._sketches) >= MAX_SKETCHES:
                problems.append(f"Only {MAX_SKETCHES} sketches can be attached at once.")
                break
            try:
                with source.open("rb") as stream:
                    data = stream.read(5_000_001)
            except OSError:
                problems.append(f"{source.name} could not be opened.")
                continue
            image = QPixmap()
            png = data.startswith(bytes.fromhex("89504e470d0a1a0a"))
            jpeg = data.startswith(bytes.fromhex("ffd8ff"))
            if len(data) > 5_000_000 or not (png or jpeg) or not image.loadFromData(data):
                problems.append(f"{source.name} must be a readable PNG or JPEG up to 5 MB.")
                continue
            mime = "image/png" if png else "image/jpeg"
            data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            thumbnail = image.scaled(240, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._sketches.append((source, data_url, thumbnail))
            attached.add(source)
            self.sketch_list.addItem(source.name)
            last_added = len(self._sketches) - 1
        if last_added is not None:
            self.sketch_list.setCurrentIndex(last_added)
        self._show_sketch()
        if problems:
            QMessageBox.warning(self, "Some sketches could not be attached", "\n".join(problems))

    def _show_sketch(self):
        index = self.sketch_list.currentIndex()
        has_selection = 0 <= index < len(self._sketches)
        self.sketch_list.setVisible(has_selection)
        self.preview.setVisible(has_selection)
        self.clear_sketch.setVisible(has_selection)
        if has_selection:
            self.preview.setPixmap(self._sketches[index][2])
        else:
            self.preview.clear()
        self.attach_sketches.setText(f"Attach concept sketches ({len(self._sketches)}/{MAX_SKETCHES})")
        self.attach_sketches.setEnabled(len(self._sketches) < MAX_SKETCHES)

    def _remove_sketch(self):
        index = self.sketch_list.currentIndex()
        if index >= 0:
            del self._sketches[index]
            self.sketch_list.removeItem(index)
            self._show_sketch()

    def show_proposal(self, proposal):
        self.proposal = proposal
        self.rationale.setText(proposal["rationale"] if proposal else "")
        while self.dimensions.count():
            self.dimensions.takeAt(0).widget().deleteLater()
        self.controls = []
        if proposal is None:
            self.result.clear()
            self.applied_threshold = None
            return
        if "recipe_version" in proposal:
            group = QGroupBox("Recipe parameters")
            form = QFormLayout(group)
            controls = {}
            for parameter in proposal["parameters"]:
                control = QDoubleSpinBox()
                control.setDecimals(3)
                control.setRange(parameter["min"], parameter["max"])
                control.setSingleStep(parameter["step"])
                control.setValue(parameter["value"])
                if parameter["unit"]:
                    control.setSuffix(f" {parameter['unit']}")
                control.setKeyboardTracking(False)
                control.valueChanged.connect(self.dimensions_changed)
                form.addRow(parameter["label"], control)
                controls[parameter["name"]] = control
            self.controls.append(controls)
            self.dimensions.addWidget(group)
            return
        for index, operation in enumerate(proposal["operations"], start=1):
            kind = operation["type"]
            group = QGroupBox(f"{index}. {kind.replace('_', ' ').title()}")
            form = QFormLayout(group)
            controls = {}
            for name, (label, low, high, unit) in OPERATIONS[kind].items():
                control = QDoubleSpinBox()
                control.setRange(low, high)
                control.setDecimals(2)
                control.setSingleStep(5 if unit == "°" else 0.1)
                if unit:
                    control.setSuffix(f" {unit}")
                control.setValue(operation[name])
                control.setKeyboardTracking(False)
                control.valueChanged.connect(self.dimensions_changed)
                form.addRow(label, control)
                controls[name] = control
            self.controls.append(controls)
            self.dimensions.addWidget(group)

    def edited_proposal(self):
        if "recipe_version" in self.proposal:
            return {**self.proposal, "parameters": [
                {**parameter, "value": self.controls[0][parameter["name"]].value()}
                for parameter in self.proposal["parameters"]
            ]}
        return {
            "rationale": self.proposal["rationale"],
            "operations": [
                {"type": operation["type"], **{name: control.value() for name, control in controls.items()}}
                for operation, controls in zip(self.proposal["operations"], self.controls, strict=True)
            ],
        }
