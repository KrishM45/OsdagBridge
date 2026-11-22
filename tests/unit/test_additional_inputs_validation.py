import unittest
import sys
from PySide6.QtWidgets import QApplication, QWidget
from osdagbridge.core.utils.common import (
    KEY_SPAN, KEY_CARRIAGEWAY_WIDTH, KEY_INCLUDE_MEDIAN, KEY_FOOTPATH,
    KEY_CB_TYPE, KEY_CB_WIDTH, KEY_CB_HEIGHT, KEY_PROJECT_LOCATION
)
from osdagbridge.core.bridge_types.plate_girder.defaults import BASIC_INPUT_DICT, solve_extend_basic_input_dict
from osdagbridge.desktop.ui.dialogs.additional_input.additional_inputs import AdditionalInputs

class TestAdditionalInputsValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def setUp(self):
        self.dialog = AdditionalInputs()
        test_dict = dict(BASIC_INPUT_DICT)
        test_dict[KEY_SPAN] = 30.0
        test_dict[KEY_CARRIAGEWAY_WIDTH] = 7.5
        test_dict[KEY_INCLUDE_MEDIAN] = "No"
        test_dict[KEY_FOOTPATH] = "None"
        solve_extend_basic_input_dict(test_dict)
        self.dialog.set_input_dictionary(test_dict)

    def test_initial_dirty_state_is_clean(self):
        self.assertFalse(self.dialog._has_unsaved_changes())

    def test_dirty_state_detected_on_change(self):
        self.dialog.working_input_dict[KEY_SPAN] = 45.0
        self.assertTrue(self.dialog._has_unsaved_changes())

    def test_custom_validation_fails_when_empty(self):
        # Set Crash barrier to Custom
        cb_type_widget = self.dialog.findChild(QWidget, KEY_CB_TYPE)
        if cb_type_widget:
            idx = cb_type_widget.findText("Custom")
            if idx >= 0:
                cb_type_widget.setCurrentIndex(idx)
        
        # Clear custom width
        cb_w_widget = self.dialog.findChild(QWidget, KEY_CB_WIDTH)
        if cb_w_widget:
            cb_w_widget.setText("")
        
        is_valid, err_msg, widget = self.dialog._validate_custom_inputs()
        self.assertFalse(is_valid)
        self.assertIn("empty", err_msg.lower())

    def test_custom_validation_passes_when_filled(self):
        cb_type_widget = self.dialog.findChild(QWidget, KEY_CB_TYPE)
        if cb_type_widget:
            idx = cb_type_widget.findText("Custom")
            if idx >= 0:
                cb_type_widget.setCurrentIndex(idx)
        
        for k, v in [(KEY_CB_WIDTH, "500"), (KEY_CB_HEIGHT, "800")]:
            w = self.dialog.findChild(QWidget, k)
            if w:
                w.setText(v)
        
        is_valid, err_msg, widget = self.dialog._validate_custom_inputs()
        self.assertTrue(is_valid)

if __name__ == "__main__":
    unittest.main()
