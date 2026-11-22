import unittest
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from osdagbridge.core.utils.common import (
    KEY_SPAN, KEY_CARRIAGEWAY_WIDTH, KEY_TS_GIRDER_SPACING,
    KEY_TS_DECK_THICKNESS, KEY_TS_DECK_OVERHANG, KEY_TS_OVERALL_WIDTH,
    KEY_MP_CB_SPACING, KEY_INCLUDE_MEDIAN, KEY_FOOTPATH
)
from osdagbridge.core.bridge_types.plate_girder.defaults import BASIC_INPUT_DICT, solve_extend_basic_input_dict
from osdagbridge.desktop.ui.docks.cad_cross_section import CrossSectionCADWidget
from osdagbridge.desktop.ui.docks.cad_top_view import TopViewCADWidget
from osdagbridge.desktop.ui.docks.cad_dual_view import BridgeDualCADWidget

class TestCADInteractiveEditing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def setUp(self):
        self.cross_widget = CrossSectionCADWidget()
        self.cross_widget.resize(1000, 700)
        
        self.top_widget = TopViewCADWidget()
        self.top_widget.resize(1000, 700)
        
        # Trigger painting into an offscreen image to populate dimension_labels
        img = QImage(1000, 700, QImage.Format_ARGB32)
        painter = QPainter(img)
        self.cross_widget.draw_cross_section(painter)
        painter.end()

        img2 = QImage(1000, 700, QImage.Format_ARGB32)
        painter2 = QPainter(img2)
        self.top_widget.draw_top_view(painter2)
        painter2.end()

    def test_cross_section_labels_registered(self):
        labels = {item['name']: item for item in self.cross_widget.dimension_labels}
        
        # Editable labels
        self.assertIn("Carriageway Width", labels)
        self.assertTrue(labels["Carriageway Width"]['editable'])
        self.assertEqual(labels["Carriageway Width"]['key'], KEY_CARRIAGEWAY_WIDTH)

        self.assertIn("Girder Spacing", labels)
        self.assertTrue(labels["Girder Spacing"]['editable'])
        self.assertEqual(labels["Girder Spacing"]['key'], KEY_TS_GIRDER_SPACING)

        self.assertIn("Deck Thickness", labels)
        self.assertTrue(labels["Deck Thickness"]['editable'])
        self.assertEqual(labels["Deck Thickness"]['key'], KEY_TS_DECK_THICKNESS)
        self.assertEqual(labels["Deck Thickness"]['unit'], "mm")

        # Locked labels
        self.assertIn("Overall Bridge Width", labels)
        self.assertFalse(labels["Overall Bridge Width"]['editable'])
        self.assertEqual(labels["Overall Bridge Width"]['key'], KEY_TS_OVERALL_WIDTH)

        self.assertIn("Overhang", labels)
        self.assertFalse(labels["Overhang"]['editable'])
        self.assertEqual(labels["Overhang"]['key'], KEY_TS_DECK_OVERHANG)

    def test_top_view_labels_registered(self):
        labels = {item['name']: item for item in self.top_widget.dimension_labels}

        # Editable labels
        self.assertIn("Span Length", labels)
        self.assertTrue(labels["Span Length"]['editable'])
        self.assertEqual(labels["Span Length"]['key'], KEY_SPAN)

        self.assertIn("Girder Spacing", labels)
        self.assertTrue(labels["Girder Spacing"]['editable'])
        self.assertEqual(labels["Girder Spacing"]['key'], KEY_TS_GIRDER_SPACING)

        # Locked labels
        self.assertIn("Bracing Spacing", labels)
        self.assertFalse(labels["Bracing Spacing"]['editable'])
        self.assertEqual(labels["Bracing Spacing"]['key'], KEY_MP_CB_SPACING)

    def test_cross_section_inline_edit_commit(self):
        labels = {item['name']: item for item in self.cross_widget.dimension_labels}
        cw_label = labels["Carriageway Width"]

        emitted_signals = []
        self.cross_widget.cad_parameter_edited.connect(lambda k, v: emitted_signals.append((k, v)))

        self.cross_widget._start_inline_editing(cw_label)
        self.assertIsNotNone(self.cross_widget._cad_editor)
        self.assertFalse(self.cross_widget._cad_editor.isHidden())

        self.cross_widget._cad_editor.setText("8.50")
        self.cross_widget._commit_inline_editing()

        self.assertTrue(self.cross_widget._cad_editor.isHidden())
        self.assertEqual(len(emitted_signals), 1)
        self.assertEqual(emitted_signals[0], (KEY_CARRIAGEWAY_WIDTH, 8.50))

    def test_dual_cad_widget_signal_forwarding(self):
        dual_widget = BridgeDualCADWidget()
        emitted_signals = []
        dual_widget.cad_parameter_edited.connect(lambda k, v: emitted_signals.append((k, v)))

        # Cross section widget emits
        dual_widget.cross_section_widget.cad_parameter_edited.emit(KEY_TS_DECK_THICKNESS, 220.0)
        # Top view widget emits
        dual_widget.top_view_widget.cad_parameter_edited.emit(KEY_SPAN, 40.0)

        self.assertEqual(len(emitted_signals), 2)
        self.assertEqual(emitted_signals[0], (KEY_TS_DECK_THICKNESS, 220.0))
        self.assertEqual(emitted_signals[1], (KEY_SPAN, 40.0))

    def test_template_page_cad_sync(self):
        from osdagbridge.desktop.ui.template_page import CustomWindow
        from osdagbridge.core.bridge_types.plate_girder.plategirderbridge import PlateGirderBridge
        page = CustomWindow("OsdagBridge", PlateGirderBridge)

        # 1. Edit Carriageway Width
        page.on_cad_parameter_edited(KEY_CARRIAGEWAY_WIDTH, 8.5)
        cw_widget = page.input_dock._w(KEY_CARRIAGEWAY_WIDTH)
        if cw_widget:
            self.assertEqual(float(cw_widget.text()), 8.5)
        self.assertEqual(float(page.input_dict[KEY_CARRIAGEWAY_WIDTH]), 8.5)

        # 2. Edit Span
        page.on_cad_parameter_edited(KEY_SPAN, 42.0)
        span_widget = page.input_dock._w(KEY_SPAN)
        if span_widget:
            self.assertEqual(float(span_widget.text()), 42.0)
        self.assertEqual(float(page.input_dict[KEY_SPAN]), 42.0)

        # 3. Edit Girder Spacing
        page.on_cad_parameter_edited(KEY_TS_GIRDER_SPACING, 2.5)
        self.assertEqual(float(page.input_dict[KEY_TS_GIRDER_SPACING]), 2.5)
        self.assertIn(KEY_TS_DECK_OVERHANG, page.input_dict)

        # 4. Edit Deck Thickness
        page.on_cad_parameter_edited(KEY_TS_DECK_THICKNESS, 230.0)
        self.assertEqual(float(page.input_dict[KEY_TS_DECK_THICKNESS]), 230.0)

if __name__ == "__main__":
    unittest.main()
