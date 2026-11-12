"""Dark, architecture-oriented styling."""

from __future__ import annotations

import matplotlib as mpl

STYLESHEET = """
QMainWindow, QWidget {
    background: #1c1e1c;
    color: #e7e2d6;
    font-family: "Segoe UI";
    font-size: 12px;
}
QScrollArea, QScrollArea > QWidget > QWidget { background: #1c1e1c; border: none; }
QGroupBox {
    border: 1px solid #3c4038;
    border-radius: 6px;
    margin-top: 12px;
    padding: 8px 8px 6px 8px;
    font-weight: 600;
    color: #f0eadf;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLabel { background: transparent; }
QLabel#hint { color: #b7b1a4; }
QLabel#validation { padding: 6px; border-radius: 4px; background: #2a2d2a; }
QPushButton {
    background: #2c302c;
    border: 1px solid #4d5248;
    border-radius: 4px;
    padding: 6px 10px;
}
QPushButton:hover { background: #3a4038; }
QPushButton:disabled { color: #7d786e; background: #262824; }
QPushButton#primary { background: #8d6b3e; color: #1a140c; font-weight: 700; border: none; }
QPushButton#primary:hover { background: #a47d4b; }
QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: #262824;
    border: 1px solid #4d5248;
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: #8d6b3e;
    selection-color: #1a140c;
}
QComboBox QAbstractItemView { background: #262824; selection-background-color: #8d6b3e; }
QSlider::groove:horizontal { height: 4px; background: #3c4038; border-radius: 2px; }
QSlider::handle:horizontal { background: #c4a574; width: 12px; margin: -5px 0; border-radius: 6px; }
QTableWidget { background: #222420; gridline-color: #3c4038; alternate-background-color: #262824; }
QHeaderView::section { background: #2c302c; color: #e7e2d6; border: none; padding: 4px; }
QTabWidget::pane { border: 1px solid #3c4038; }
QTabBar::tab { background: #262824; color: #c8c2b4; padding: 6px 12px; border: 1px solid #3c4038; }
QTabBar::tab:selected { background: #34382f; color: #f3ecdf; }
QStatusBar { background: #161816; color: #d9d3c7; }
QProgressBar { background: #262824; border: none; height: 8px; }
QProgressBar::chunk { background: #c4a574; }
QMenuBar, QMenu { background: #1c1e1c; color: #e7e2d6; }
QMenu::item:selected { background: #3a4038; }
QSplitter::handle { background: #34382f; }
"""


def apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "#1c1e1c",
            "axes.facecolor": "#242724",
            "axes.edgecolor": "#5c6158",
            "axes.labelcolor": "#e7e2d6",
            "axes.titlecolor": "#f3ecdf",
            "xtick.color": "#c8c2b4",
            "ytick.color": "#c8c2b4",
            "text.color": "#e7e2d6",
            "grid.color": "#3a3d38",
            "axes.grid": True,
            "grid.alpha": 0.45,
            "font.size": 9,
        }
    )
