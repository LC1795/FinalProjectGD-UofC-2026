"""Resizable point-up hexagonal viewport with six surrounding panel sectors."""

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QRegion, QTransform
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizeGrip, QVBoxLayout, QWidget,
)


def hex_vertices(center, radius):
    return [QPointF(center.x() + radius * math.sin(i * math.pi / 3),
                    center.y() - radius * math.cos(i * math.pi / 3)) for i in range(6)]


def _ray_border(center, vertex, width, height):
    dx, dy = vertex.x() - center.x(), vertex.y() - center.y()
    candidates = []
    if dx:
        candidates.append(((width if dx > 0 else 0) - center.x()) / dx)
    if dy:
        candidates.append(((height if dy > 0 else 0) - center.y()) / dy)
    t = min(value for value in candidates if value > 0)
    return QPointF(center.x() + t * dx, center.y() + t * dy)


def _border_position(point, width, height):
    x, y = point.x(), point.y()
    if abs(y) < 1e-5:
        return x
    if abs(x - width) < 1e-5:
        return width + y
    if abs(y - height) < 1e-5:
        return width + height + width - x
    return 2 * width + height + height - y


def sector_polygon(center, first, second, width, height):
    """Clockwise outer sector between two adjacent radial rays."""
    a, b = (_ray_border(center, point, width, height) for point in (first, second))
    start, end = (_border_position(point, width, height) for point in (a, b))
    perimeter = 2 * (width + height)
    if end <= start:
        end += perimeter
    corners = [(0, QPointF(0, 0)), (width, QPointF(width, 0)),
               (width + height, QPointF(width, height)),
               (2 * width + height, QPointF(0, height))]
    between = []
    for position, point in corners:
        position += perimeter if position <= start else 0
        if start < position < end:
            between.append((position, point))
    return QPolygonF([first, a, *[point for _, point in sorted(between)], b, second])


def content_rectangle(polygon):
    """Find an axis-aligned rectangle inside the convex panel sector."""
    bounds = polygon.boundingRect().adjusted(8, 8, -8, -8)
    best = None
    best_area = 0
    rows = 18
    for top in range(rows - 1):
        for bottom in range(top + 1, rows):
            y0 = bounds.top() + bounds.height() * top / rows
            y1 = bounds.top() + bounds.height() * bottom / rows
            left, right = bounds.left(), bounds.right()
            for y in (y0, y1):
                intersections = []
                for i, a in enumerate(polygon):
                    b = polygon[(i + 1) % len(polygon)]
                    if min(a.y(), b.y()) <= y <= max(a.y(), b.y()) and a.y() != b.y():
                        intersections.append(a.x() + (b.x() - a.x()) * (y - a.y()) / (b.y() - a.y()))
                if len(intersections) >= 2:
                    left, right = max(left, min(intersections) + 8), min(right, max(intersections) - 8)
                else:
                    right = left
            area = max(0, right - left) * (y1 - y0)
            if area > best_area:
                best_area = area
                best = (round(left), round(y0), max(1, round(right - left)), max(1, round(y1 - y0)))
    return best or (round(bounds.left()), round(bounds.top()), 1, 1)


class EdgeTitle(QAbstractButton):
    """Paint and hit-test the title in the direction of its hexagon edge."""

    def __init__(self, title, parent):
        super().__init__(parent)
        self.setText(title)
        self.setCheckable(True)
        self.angle = 0
        self._label_width = 100
        self._label_height = 30

    def set_angle(self, angle):
        self.angle = (angle + 90) % 180 - 90
        self._label_width = self.fontMetrics().horizontalAdvance(self.text()) + 24
        self._label_height = self.fontMetrics().height() + 12
        radians = math.radians(self.angle)
        width = abs(self._label_width * math.cos(radians)) + abs(self._label_height * math.sin(radians))
        height = abs(self._label_width * math.sin(radians)) + abs(self._label_height * math.cos(radians))
        self.resize(math.ceil(width) + 4, math.ceil(height) + 4)
        transform = QTransform()
        transform.translate(self.width() / 2, self.height() / 2)
        transform.rotate(self.angle)
        rectangle = QRectF(-self._label_width / 2, -self._label_height / 2,
                           self._label_width, self._label_height)
        self.setMask(QRegion(transform.map(QPolygonF(rectangle)).toPolygon()))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self.angle)
        rectangle = QRectF(-self._label_width / 2, -self._label_height / 2,
                           self._label_width, self._label_height)
        painter.setPen(QPen(QColor("#889c7c"), 1))
        painter.setBrush(QColor("#adc99a") if self.isChecked() else QColor("#2b312c"))
        painter.drawRoundedRect(rectangle.adjusted(1, 1, -1, -1), 4, 4)
        painter.setPen(QColor("#172016") if self.isChecked() else QColor("#efece3"))
        painter.drawText(rectangle, Qt.AlignCenter, self.text())


class StickyPanel(QWidget):
    """Independent note window whose position and size survive tab toggles."""

    def __init__(self, title, content, parent):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint)
        self.title = title
        self._placed = False
        self._drag_position = None
        self.setWindowTitle(f"Soundform — {title}")
        self.setMinimumSize(300, 240)
        self.resize(650 if title == "Analysis" else 420, 460 if title == "Analysis" else 620)
        self.setObjectName("stickyPanel")
        self.setStyleSheet(
            "#stickyPanel { background: #292c25; border: 1px solid #aab487; }"
            "#stickyTitle { background: #c6cc9d; color: #20251c; border: none; }"
            "#stickyTitle QLabel { color: #20251c; font-weight: bold; border: none; }"
            "#stickyTitle QPushButton { color: #20251c; background: transparent; border: none; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        self.header = QWidget()
        self.header.setObjectName("stickyTitle")
        row = QHBoxLayout(self.header)
        row.setContentsMargins(10, 4, 5, 4)
        label = QLabel(title)
        row.addWidget(label, 1)
        close = QPushButton("×")
        close.setFixedWidth(28)
        close.setToolTip("Hide panel")
        close.clicked.connect(lambda: parent.set_open(title, False))
        row.addWidget(close)
        layout.addWidget(self.header)
        for widget in (self.header, label):
            widget.installEventFilter(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setWidget(content)
        layout.addWidget(self.scroll, 1)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch()
        grip = QSizeGrip(self)
        grip.setToolTip("Drag to resize")
        footer.addWidget(grip)
        layout.addLayout(footer)

    def place_once(self, anchor):
        if self._placed:
            return
        screen = QApplication.screenAt(anchor) or QApplication.primaryScreen()
        bounds = screen.availableGeometry()
        x = max(bounds.left(), min(anchor.x(), bounds.right() - self.width() + 1))
        y = max(bounds.top(), min(anchor.y(), bounds.bottom() - self.height() + 1))
        self.move(x, y)
        self._placed = True

    def eventFilter(self, watched, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.pos()
            return True
        if event.type() == QEvent.MouseMove and self._drag_position is not None:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            return True
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            self._drag_position = None
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        event.ignore()
        self.parentWidget().set_open(self.title, False)


class HexLayout(QWidget):
    def __init__(self, viewport, panels, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self.viewport.setParent(self)
        self.fraction = 0.76
        self._drag_side = None
        self._last_drag_position = None
        self._center_offset = QPointF()
        self._vertices = []
        self._panels = {}
        self.setMouseTracking(True)
        for title, content in panels:
            panel = StickyPanel(title, content, self)
            panel.hide()
            tab = EdgeTitle(title, self)
            tab.toggled.connect(lambda opened, name=title: self.set_open(name, opened))
            self._panels[title] = (panel, tab)
        self._titles = list(self._panels)
        # Only the viewport participates in hexagon-edge dragging.
        for child in [self.viewport, *self.viewport.findChildren(QWidget)]:
            child.installEventFilter(self)

    def set_open(self, title, opened):
        panel, tab = self._panels[title]
        if opened:
            anchor = self.mapToGlobal(tab.geometry().center())
            panel.place_once(anchor)
        panel.setVisible(opened)
        if opened:
            panel.raise_()
        tab.blockSignals(True)
        tab.setChecked(opened)
        tab.blockSignals(False)
        for _, button in self._panels.values():
            button.raise_()
        self.update()

    def is_open(self, title):
        return self._panels[title][1].isChecked()

    def set_collapsed(self, collapsed):
        for title in self._panels:
            self.set_open(title, not collapsed)

    def _center(self, radius):
        width, height = self.width() - 1, self.height() - 1
        half_width = radius * math.sqrt(3) / 2
        base = QPointF(width / 2, height / 2)
        requested = base + self._center_offset
        x = max(half_width + 8, min(requested.x(), width - half_width - 8))
        y = max(radius + 8, min(requested.y(), height - radius - 8))
        center = QPointF(x, y)
        self._center_offset = center - base
        return center

    def _arrange(self):
        width, height = self.width() - 1, self.height() - 1
        radius = max(30, min(height / 2, width / math.sqrt(3)) * self.fraction)
        center = self._center(radius)
        self._vertices = hex_vertices(center, radius)
        polygon = QPolygonF(self._vertices)
        bounds = polygon.boundingRect().toAlignedRect()
        self.viewport.setGeometry(bounds)
        local = QPolygonF([point - QPointF(bounds.topLeft()) for point in self._vertices])
        self.viewport.setMask(QRegion(local.toPolygon()))
        for i, title in enumerate(self._titles):
            panel, tab = self._panels[title]
            edge = {"Cloud": 0, "Layers": 1}.get(title, i)
            first, second = self._vertices[edge], self._vertices[(edge + 1) % 6]
            midpoint = (first + second) / 2
            direction = midpoint - center
            length = math.hypot(direction.x(), direction.y())
            position = midpoint + direction * (25 / length)
            if title in ("Cloud", "AI pavilion", "Layers", "Properties"):
                tangent = second - first
                tangent /= math.hypot(tangent.x(), tangent.y())
                position += tangent * (65 if title in ("Cloud", "Layers") else -65)
            angle = math.degrees(math.atan2(second.y() - first.y(), second.x() - first.x()))
            tab.set_angle(angle)
            tab.move(round(position.x() - tab.width() / 2), round(position.y() - tab.height() / 2))
            tab.raise_()
        self.update()

    def resizeEvent(self, event):
        self._arrange()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#b1c99d"), 3))
        painter.drawPolygon(QPolygonF(self._vertices))

    def _edge_at(self, point):
        for i, first in enumerate(self._vertices):
            second = self._vertices[(i + 1) % 6]
            dx, dy = second.x() - first.x(), second.y() - first.y()
            t = max(0, min(1, ((point.x() - first.x()) * dx + (point.y() - first.y()) * dy) / max(dx * dx + dy * dy, 1)))
            if math.hypot(point.x() - first.x() - t * dx, point.y() - first.y() - t * dy) <= 10:
                return i
        return None

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease):
            position = QPointF(watched.mapTo(self, event.position().toPoint()))
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                side = self._edge_at(position)
                if side is not None:
                    self._drag_side = side
                    self._last_drag_position = position
                    self.grabMouse()
                    return True
            if event.type() == QEvent.MouseMove and self._drag_side is None:
                if self._edge_at(position) is not None:
                    watched.setCursor(Qt.SizeAllCursor)
                else:
                    watched.unsetCursor()
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_side = self._edge_at(event.position())
            if self._drag_side is not None:
                self._last_drag_position = event.position()
                self.grabMouse()
                event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_side is None:
            self.setCursor(Qt.SizeAllCursor if self._edge_at(event.position()) is not None else Qt.ArrowCursor)
            return
        if event.modifiers() & Qt.ShiftModifier:
            delta = event.position() - self._last_drag_position
            self._center_offset += delta
            self._last_drag_position = event.position()
            self._arrange()
            return
        self._last_drag_position = event.position()
        center = sum(self._vertices, QPointF()) / 6
        first, second = self._vertices[self._drag_side], self._vertices[(self._drag_side + 1) % 6]
        normal = (first + second) / 2 - center
        length = math.hypot(normal.x(), normal.y())
        normal /= length
        delta = event.position() - center
        radius = (delta.x() * normal.x() + delta.y() * normal.y()) / math.cos(math.pi / 6)
        maximum = min((self.height() - 1) / 2, (self.width() - 1) / math.sqrt(3))
        self.fraction = max(0.30, min(0.94, radius / max(maximum, 1)))
        self._arrange()

    def mouseReleaseEvent(self, event):
        if self._drag_side is not None:
            self.releaseMouse()
        self._drag_side = None
