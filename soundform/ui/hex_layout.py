"""Resizable point-up hexagonal viewport with six surrounding panel sectors."""

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QRegion, QTransform, QPainterPath
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QComboBox, QButtonGroup, QHBoxLayout, QLabel, QPushButton,
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
        self._attach_at_top = False

    def set_angle(self, angle):
        self.angle = (angle + 90) % 180 - 90
        self._label_width = self.fontMetrics().horizontalAdvance(self.text()) + 24
        self._label_height = self.fontMetrics().height() + 12
        radians = math.radians(self.angle)
        width = abs((self._label_width+16) * math.cos(radians)) + abs(self._label_height * math.sin(radians))
        height = abs((self._label_width+16) * math.sin(radians)) + abs(self._label_height * math.cos(radians))
        self.resize(math.ceil(width) + 4, math.ceil(height) + 4)
        transform = QTransform()
        transform.translate(self.width() / 2, self.height() / 2)
        transform.rotate(self.angle)
        self.setMask(QRegion(transform.map(self._outline()).toFillPolygon().toPolygon()))
        self.update()

    def _outline(self):
        """Chrome-like shoulders, with an open base joining the inner hexagon."""
        half, top, base = self._label_width/2, -self._label_height/2, self._label_height/2
        path = QPainterPath()
        path.moveTo(-half-6,base)
        path.cubicTo(-half,base,-half,base-4,-half,base-7)
        path.lineTo(-half,top+7)
        path.quadTo(-half,top,-half+7,top)
        path.lineTo(half-7,top)
        path.quadTo(half,top,half,top+7)
        path.lineTo(half,base-7)
        path.cubicTo(half,base-4,half,base,half+6,base)
        if self._attach_at_top:
            path = QTransform().scale(1,-1).map(path)
        return path

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width()/2,self.height()/2)
        painter.rotate(self.angle)
        path = self._outline()
        painter.fillPath(path,QColor("#adc99a") if self.isChecked() else QColor("#384235"))
        painter.setPen(QPen(QColor("#889c7c"),1))
        painter.drawPath(path)
        painter.setPen(QColor("#172016") if self.isChecked() else QColor("#efece3"))
        painter.drawText(QRectF(-self._label_width/2,-self._label_height/2,
                               self._label_width,self._label_height),Qt.AlignCenter,self.text())


class LayerSelector(QComboBox):
    """Horizontal layer tab attached by its left edge to the viewport."""
    def paintEvent(self,event):
        painter=QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w,h=self.width()-1,self.height()-1
        path=QPainterPath()
        path.moveTo(0,0)
        path.cubicTo(7,0,7,5,12,5)
        path.lineTo(w-7,5)
        path.quadTo(w,5,w,12)
        path.lineTo(w,h-12)
        path.quadTo(w,h-5,w-7,h-5)
        path.lineTo(12,h-5)
        path.cubicTo(7,h-5,7,h,0,h)
        painter.fillPath(path,QColor("#384235"))
        painter.setPen(QPen(QColor("#889c7c"),1))
        painter.drawPath(path)
        painter.setPen(QColor("#efece3"))
        text=self.fontMetrics().elidedText(self.currentText(),Qt.ElideMiddle,max(1,w-40))
        painter.drawText(QRectF(12,0,w-40,h),Qt.AlignVCenter|Qt.AlignLeft,text)
        painter.setBrush(QColor("#efece3"))
        painter.drawPolygon(QPolygonF([QPointF(w-18,h/2-2),QPointF(w-10,h/2-2),QPointF(w-14,h/2+3)]))


class StickyPanel(QWidget):
    """Resizable note window that reopens at its minimum height."""

    def __init__(self, title, content, parent):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint)
        self.title = title
        self._shutting_down = False
        self._placed = False
        self._drag_position = None
        self.setWindowTitle(f"Soundform — {title}")
        self.setMinimumSize(300, 240)
        self.resize(650 if title == "Analysis" else 420, self.minimumHeight())
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
        if self._shutting_down:
            event.accept()
            return
        event.ignore()
        self.parentWidget().set_open(self.title, False)


class HexLayout(QWidget):
    drawing_tool_requested = Signal(str)
    def __init__(self, viewport, panels, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self.viewport.setParent(self)
        self.fraction = 0.76
        self._drag_side = None
        self._last_drag_position = None
        self._center_offset = QPointF()
        self._vertices = []
        self._outer_vertices = []
        self._band_width = 40
        self._panels = {}
        self.setMouseTracking(True)
        for title, content in panels:
            panel = StickyPanel(title, content, self)
            panel.hide()
            tab = EdgeTitle(title, self)
            tab.toggled.connect(lambda opened, name=title: self.set_open(name, opened))
            self._panels[title] = (panel, tab)
        self._titles = list(self._panels)
        self._view_buttons = {}
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        for label, name in (("Top", "top"), ("Front", "front"),
                            ("Right", "right"), ("Perspective", "perspective")):
            button = EdgeTitle(label, self)
            self._view_group.addButton(button)
            button.setToolTip("Switch to " + label.lower() + " view")
            button.clicked.connect(lambda checked=False, view=name: self.viewport.set_view(view))
            self._view_buttons[name] = button
        self._draw_buttons={}
        for label,name in (("Point","point"),("Line","line"),("Spline","spline")):
            button=EdgeTitle(label,self)
            button.setToolTip("Draw "+name+": left-click to place, right-click/Enter to finish, Escape to cancel")
            button.clicked.connect(lambda checked=False,kind=name:self.drawing_tool_requested.emit(kind if checked else ""))
            self._draw_buttons[name]=button
        self.drawing_layer=LayerSelector(self)
        self.drawing_layer.addItem("Drawing")
        self.drawing_layer.setToolTip("Layer for newly drawn geometry")
        self.drawing_layer.currentTextChanged.connect(lambda _:self._arrange())
        self.viewport.drawing_mode_changed.connect(self._sync_draw_buttons)
        self.viewport.view_changed.connect(self._sync_view_buttons)
        self._sync_view_buttons(self.viewport._view_name)
        # Only the viewport participates in hexagon-edge dragging.
        for child in [self.viewport, *self.viewport.findChildren(QWidget)]:
            child.installEventFilter(self)

    def _sync_draw_buttons(self,mode):
        for name,button in self._draw_buttons.items():
            button.setChecked(name==mode)

    def set_drawing_layers(self,layers):
        current=self.drawing_layer.currentText()
        blocked=self.drawing_layer.blockSignals(True)
        self.drawing_layer.clear()
        self.drawing_layer.addItems(layers)
        self.drawing_layer.setCurrentText(current if current in layers else "Drawing")
        self.drawing_layer.blockSignals(blocked)
        self._arrange()

    def _sync_view_buttons(self, name):
        name = "right" if name == "side" else name
        if name in self._view_buttons:
            self._view_buttons[name].setChecked(True)

    def close_panels(self):
        """Close tool windows when their owning application window exits."""
        for panel, tab in self._panels.values():
            panel._shutting_down = True
            panel.close()
            tab.setChecked(False)

    def set_open(self, title, opened):
        panel, tab = self._panels[title]
        if opened:
            if not panel.isVisible():
                # Start compact on each opening; retain the chosen width and
                # leave both dimensions freely resizable while the note is open.
                panel.resize(panel.width(), max(panel.minimumHeight(), panel.minimumSizeHint().height()))
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
        buttons = [tab for _,tab in self._panels.values()] + list(self._view_buttons.values()) + list(self._draw_buttons.values())
        # Tabs overlap the inner rim by 1 px; leave 2 px beyond their outer edge.
        self._band_width = max(button.fontMetrics().height()+12 for button in buttons)+1
        padding = self._band_width/math.cos(math.pi/6)+8
        radius = max(30, (min(height/2,width/math.sqrt(3))-padding)*self.fraction)
        center = self._center(radius)
        self._vertices = hex_vertices(center, radius)
        self._outer_vertices = hex_vertices(center, radius+self._band_width/math.cos(math.pi/6))
        polygon = QPolygonF(self._vertices)
        bounds = polygon.boundingRect().toAlignedRect()
        self.viewport.setGeometry(bounds)
        local = QPolygonF([point - QPointF(bounds.topLeft()) for point in self._vertices])
        self.viewport.setMask(QRegion(local.toPolygon()))
        # Anchors are fractions of each edge, so spacing scales with the hexagon.
        anchors = {"Cloud": (0, .5), "AI pavilion": (0, .18),
                   "Generate": (0, .82), "Export": (2, .5), "Layers": (4, .72), "Properties": (4, .28),
                   "Analysis": (5, .28), "Audio": (5, .72)}
        for i, title in enumerate(self._titles):
            _, tab = self._panels[title]
            edge, fraction = anchors.get(title, (i % 6, .5))
            self._place_edge_button(tab, edge, fraction, center)
        for fraction, button in zip((.12, .37, .62, .87), self._view_buttons.values()):
            self._place_edge_button(button, 3, fraction, center)
        for fraction,button in zip((.39,.62,.85),self._draw_buttons.values()):
            self._place_edge_button(button,1,fraction,center)
        anchor=self._vertices[1]+(self._vertices[2]-self._vertices[1])*.16
        self.drawing_layer.setFixedSize(max(180,min(300,self.fontMetrics().horizontalAdvance(self.drawing_layer.currentText())+44)),self.fontMetrics().height()+22)
        self.drawing_layer.move(round(anchor.x()-1),round(anchor.y()-self.drawing_layer.height()/2))
        self.drawing_layer.raise_()
        self.update()

    def _place_edge_button(self, button, edge, fraction, center):
        first, second = self._vertices[edge], self._vertices[(edge+1) % 6]
        tangent = second-first
        angle = math.degrees(math.atan2(tangent.y(), tangent.x()))
        upright = (angle+90) % 180-90
        outward = (first+second)/2-center
        length = max(math.hypot(outward.x(),outward.y()), 1)
        # The tab's open base overlaps the viewport rim by one pixel.
        normal = QPointF(-math.sin(math.radians(upright)),math.cos(math.radians(upright)))
        button._attach_at_top = normal.x()*outward.x()+normal.y()*outward.y() > 0
        button.set_angle(angle)
        clearance = button._label_height/2-1
        position = first+tangent*fraction+outward*(clearance/length)
        button.move(round(position.x()-button.width()/2), round(position.y()-button.height()/2))
        button._edge_anchor = (edge, fraction)
        button.raise_()

    def resizeEvent(self, event):
        self._arrange()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        band = QPainterPath()
        band.setFillRule(Qt.OddEvenFill)
        band.addPolygon(QPolygonF(self._outer_vertices))
        band.closeSubpath()
        band.addPolygon(QPolygonF(self._vertices))
        band.closeSubpath()
        painter.fillPath(band,QColor("#2b332b"))
        painter.setPen(QPen(QColor("#687b60"),1))
        painter.drawPolygon(QPolygonF(self._outer_vertices))
        painter.setPen(QPen(QColor("#b1c99d"),1))
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
        if getattr(self.viewport,"_drawing_mode",None):
            return super().eventFilter(watched,event)
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
