"""Launch Soundform.

Run from the repository root:

    soundform\\.venv\\Scripts\\python.exe -m soundform
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_path() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    _ensure_path()
    from PySide6.QtWidgets import QApplication

    from soundform.ui.main_window import MainWindow
    from soundform.ui.theme import STYLESHEET

    app = QApplication(sys.argv)
    app.setApplicationName("Soundform")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
