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
    from soundform.branding import configure_branding
    configure_branding(app)
    # Keep the application identifier stable for existing settings.
    app.setApplicationName("Soundform")
    from soundform import APP_NAME, __version__
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    # Assign directly after viewport construction as well as at application level.
    # This also covers windows whose native handles were created during setup.
    configure_branding(app)
    window.setWindowIcon(app.windowIcon())
    window.show()
    from soundform.branding import set_taskbar_icon
    set_taskbar_icon(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
