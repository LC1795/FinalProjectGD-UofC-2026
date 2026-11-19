"""Panels that collapse into a title button on their outer edge."""

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QBoxLayout, QPushButton, QStyle, QStyleOptionButton, QStylePainter, QWidget


class EdgeTab(QPushButton):
    def __init__(self, title, edge):
        super().__init__(title)
        self.edge = edge
        if edge in ("left", "right"):
            self.setFixedWidth(28)
            self.setFixedHeight(super().sizeHint().width())

    def sizeHint(self):
        size = super().sizeHint()
        return QSize(28, size.width()) if self.edge in ("left", "right") else size

    def paintEvent(self, event):
        if self.edge not in ("left", "right"):
            return super().paintEvent(event)
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        painter.translate(self.width(), 0)
        painter.rotate(90)
        option.rect.setRect(0, 0, self.height(), self.width())
        painter.drawControl(QStyle.CE_PushButton, option)


class CollapsiblePanel(QWidget):
    def __init__(self, title, content, edge, parent=None):
        super().__init__(parent)
        self.content = content
        self.edge = edge
        self._expanded_minimum = (content.minimumWidth(), content.minimumHeight())
        direction = QBoxLayout.LeftToRight if edge in ("left", "right") else QBoxLayout.TopToBottom
        layout = QBoxLayout(direction, self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.tab = EdgeTab(title, edge)
        self.tab.setCheckable(True)
        self.tab.setChecked(True)
        self.tab.setToolTip("Click to collapse or reopen this panel.")
        self.tab.toggled.connect(lambda expanded: self.set_collapsed(not expanded))
        if edge in ("right", "bottom"):
            layout.addWidget(content, 1)
            layout.addWidget(self.tab)
        else:
            layout.addWidget(self.tab)
            layout.addWidget(content, 1)

    def set_collapsed(self, collapsed):
        self.content.setVisible(not collapsed)
        self.tab.blockSignals(True)
        self.tab.setChecked(not collapsed)
        self.tab.blockSignals(False)
        if self.edge in ("left", "right"):
            self.setMaximumWidth(self.tab.sizeHint().width() if collapsed else 16777215)
        else:
            self.setMaximumHeight(self.tab.sizeHint().height() if collapsed else 16777215)
        self.updateGeometry()
