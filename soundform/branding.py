"""Application identity shared by Qt windows and the Windows taskbar."""
from pathlib import Path
import sys

APP_ID = "Soundform.Desktop"
ASSETS = Path(__file__).resolve().parent / "assets"
_native_icons = []


def configure_branding(app):
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    from PySide6.QtGui import QIcon
    icon = ASSETS / "soundform.ico"
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))


def set_taskbar_icon(window):
    """Set both native Windows icon sizes for the actual top-level HWND."""
    if sys.platform != "win32" or not (ASSETS / "soundform.ico").is_file():
        return
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR,
                                 wintypes.UINT, ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.LoadImageW.restype = wintypes.HANDLE
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                   ctypes.c_size_t, ctypes.c_ssize_t]
    user32.SendMessageW.restype = ctypes.c_ssize_t
    # Retain the file-loaded handles for the process lifetime.
    for kind, metric in ((0, 49), (1, 11)):
        size = user32.GetSystemMetrics(metric)
        icon = user32.LoadImageW(None, str(ASSETS / "soundform.ico"), 1, size, size, 0x0010)
        if icon:
            _native_icons.append(icon)
            user32.SendMessageW(int(window.winId()), 0x0080, kind, int(icon))

