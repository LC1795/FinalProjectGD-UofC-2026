from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from soundform.ui.main_window import MainWindow


def test_analysis_redraws_do_not_grow_layout_and_manual_resize_still_works():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    try:
        w.show()
        w.hex_layout.set_open("Analysis", True)
        panel = w.hex_layout._panels["Analysis"][0]
        QTest.qWait(150)
        for index in range(w.analysis_panel.tabs.count()):
            w.analysis_panel.tabs.setCurrentIndex(index)
            QTest.qWait(150)
            canvas = w.analysis_panel.tabs.currentWidget()
            initial = (panel.size(),w.analysis_panel.size(),canvas.minimumSizeHint())
            for _ in range(5):
                canvas.draw_idle()
                QTest.qWait(100)
                assert (panel.size(),w.analysis_panel.size(),canvas.minimumSizeHint()) == initial
            assert not canvas._image.isNull()
        panel.resize(800,600)
        QTest.qWait(200)
        assert panel.width()==800 and panel.height()==600
        assert w.analysis_panel.protection_canvas.width()>320
        w.hex_layout.set_open("Analysis",False)
        w.hex_layout.set_open("Analysis",True)
        QTest.qWait(150)
        initial = w.analysis_panel.size()
        QTest.qWait(400)
        assert w.analysis_panel.size()==initial
        assert panel.height()==panel.minimumHeight()
    finally:
        w.close()
