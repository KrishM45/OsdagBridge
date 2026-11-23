import unittest
import json
from collections import OrderedDict

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from osdagbridge.core.bridge_types.plate_girder.plot_generator import (
    _find_transverse_lines,
    _order_transverse_line,
    _find_girders,
    build_figure_sfd,
    build_figure_bmd,
    build_figure_deflection,
    build_figure_grillage,
)


class MockDataset:
    """Mock xarray.Dataset container with an attrs dict."""
    def __init__(self, attrs=None):
        self.attrs = attrs or {}


class TestTransversePlotting(unittest.TestCase):
    def setUp(self):
        """
        Build a realistic 4-girder, 6-transverse-line grillage geometry.
        Girders: G1 (Z=0), G2 (Z=2.5), G3 (Z=5.0), G4 (Z=7.5)
        Stations: X = 0.0, 6.0, 12.0, 18.0, 24.0, 30.0
        Total nodes: 6 stations x 4 girders = 24 nodes
        """
        self.nodes = {}
        self.members = {}

        # 1. Create nodes: node ID = station_idx * 4 + girder_idx + 1
        node_grid = {}
        for s_idx, x in enumerate([0.0, 6.0, 12.0, 18.0, 24.0, 30.0]):
            for g_idx, z in enumerate([0.0, 2.5, 5.0, 7.5]):
                nid = s_idx * 4 + g_idx + 1
                self.nodes[nid] = [x, 0.0, z]
                node_grid[(s_idx, g_idx)] = nid

        # 2. Create longitudinal girder members (between stations)
        ele_id = 1
        for g_idx in range(4):
            for s_idx in range(5):
                n1 = node_grid[(s_idx, g_idx)]
                n2 = node_grid[(s_idx + 1, g_idx)]
                self.members[ele_id] = [n1, n2]
                ele_id += 1

        # 3. Create transverse members (across girders at each station)
        # Station 0: Start End Diaphragm (3 elements)
        self.start_trans_eles = []
        for g_idx in range(3):
            n1 = node_grid[(0, g_idx)]
            n2 = node_grid[(0, g_idx + 1)]
            self.members[ele_id] = [n1, n2]
            self.start_trans_eles.append(ele_id)
            ele_id += 1

        # Stations 1..4: Intermediate Cross Bracings (4 stations x 3 elements = 12 elements)
        self.cb_trans_eles = []
        for s_idx in range(1, 5):
            station_eles = []
            for g_idx in range(3):
                n1 = node_grid[(s_idx, g_idx)]
                n2 = node_grid[(s_idx + 1, g_idx + 1)] if False else node_grid[(s_idx, g_idx + 1)]
                self.members[ele_id] = [n1, n2]
                station_eles.append(ele_id)
                self.cb_trans_eles.append(ele_id)
                ele_id += 1

        # Station 5: End End Diaphragm (3 elements)
        self.end_trans_eles = []
        for g_idx in range(3):
            n1 = node_grid[(5, g_idx)]
            n2 = node_grid[(5, g_idx + 1)]
            self.members[ele_id] = [n1, n2]
            self.end_trans_eles.append(ele_id)
            ele_id += 1

        # Construct member_elements mapping matching ospgrillage
        self.member_map = {
            "start_edge": {"elements": [self.start_trans_eles]},
            "end_edge": {"elements": [self.end_trans_eles]},
            "transverse_slab": {"elements": [self.cb_trans_eles]},
        }
        self.mock_ds = MockDataset(attrs={"member_elements": json.dumps(self.member_map)})

        # Create realistic xarray Dataset for full figure generation tests
        comps = [
            "Mx_i", "Mx_j", "My_i", "My_j", "Mz_i", "Mz_j",
            "Vx_i", "Vx_j", "Vy_i", "Vy_j", "Vz_i", "Vz_j",
            "x", "y", "z"
        ]
        all_ele_tags = list(self.members.keys())
        all_node_tags = list(self.nodes.keys())

        forces_data = np.zeros((len(all_ele_tags), len(comps)))
        for idx, c in enumerate(comps):
            if "Vy" in c:
                forces_data[:, idx] = 25000.0  # 25 kN
            elif "Vz" in c:
                forces_data[:, idx] = 10000.0  # 10 kN
            elif "Mx" in c:
                forces_data[:, idx] = 45000.0  # 45 kNm
            elif "Mz" in c:
                forces_data[:, idx] = 120000.0 # 120 kNm

        disp_data = np.zeros((len(all_node_tags), len(comps)))
        y_idx = comps.index("y")
        disp_data[:, y_idx] = -0.012  # -12 mm

        self.results_ds = xr.Dataset(
            data_vars={
                "forces": (["Element", "Component"], forces_data),
                "displacements": (["Node", "Component"], disp_data),
            },
            coords={
                "Element": all_ele_tags,
                "Node": all_node_tags,
                "Component": comps,
            },
            attrs={"member_elements": json.dumps(self.member_map)}
        )

    def test_transverse_lines_with_metadata(self):
        """Verify transverse lines identified and ordered using member_elements metadata."""
        lines = _find_transverse_lines(self.nodes, self.members, ds=self.mock_ds)

        # Expected keys: Start, 4 Cross Bracings, End
        expected_keys = [
            "End Diaphragm (Start)",
            "Cross Bracing 1",
            "Cross Bracing 2",
            "Cross Bracing 3",
            "Cross Bracing 4",
            "End Diaphragm (End)",
        ]
        self.assertEqual(list(lines.keys()), expected_keys)

        # Check element counts per line (3 bays between 4 girders)
        for name, elems in lines.items():
            self.assertEqual(len(elems), 3, f"Line {name} should have 3 elements")

        # Verify Start and End diaphragm element IDs match
        self.assertEqual(lines["End Diaphragm (Start)"], self.start_trans_eles)
        self.assertEqual(lines["End Diaphragm (End)"], self.end_trans_eles)

    def test_element_ordering_monotonic_z(self):
        """Verify that elements within each transverse line are ordered continuously from min Z to max Z."""
        lines = _find_transverse_lines(self.nodes, self.members, ds=self.mock_ds)
        for name, elems in lines.items():
            # Check node progression across elements
            for k in range(len(elems) - 1):
                e1 = elems[k]
                e2 = elems[k + 1]
                # Elements should share a node
                shared = set(self.members[e1]).intersection(set(self.members[e2]))
                self.assertEqual(len(shared), 1, f"Adjacent elements in {name} must share exactly 1 node")

            # Check that Z coordinate of midpoints increases monotonically
            mid_zs = [
                (self.nodes[self.members[e][0]][2] + self.nodes[self.members[e][1]][2]) / 2.0
                for e in elems
            ]
            self.assertTrue(
                all(mid_zs[i] < mid_zs[i + 1] for i in range(len(mid_zs) - 1)),
                f"Line {name} midpoints must increase monotonically in Z: {mid_zs}"
            )

    def test_geometric_fallback_without_metadata(self):
        """Verify that missing metadata falls back to geometric identification seamlessly."""
        lines = _find_transverse_lines(self.nodes, self.members, ds=None)

        expected_keys = [
            "End Diaphragm (Start)",
            "Cross Bracing 1",
            "Cross Bracing 2",
            "Cross Bracing 3",
            "Cross Bracing 4",
            "End Diaphragm (End)",
        ]
        self.assertEqual(list(lines.keys()), expected_keys)
        self.assertEqual(lines["End Diaphragm (Start)"], self.start_trans_eles)
        self.assertEqual(lines["End Diaphragm (End)"], self.end_trans_eles)

    def test_malformed_metadata_fallback(self):
        """Verify that corrupt or incomplete metadata safely falls back to geometric detection."""
        bad_ds1 = MockDataset(attrs={"member_elements": "INVALID JSON {"})
        lines1 = _find_transverse_lines(self.nodes, self.members, ds=bad_ds1)
        self.assertIn("End Diaphragm (Start)", lines1)
        self.assertIn("End Diaphragm (End)", lines1)

        bad_ds2 = MockDataset(attrs={"member_elements": "{}"})
        lines2 = _find_transverse_lines(self.nodes, self.members, ds=bad_ds2)
        self.assertIn("End Diaphragm (Start)", lines2)
        self.assertIn("End Diaphragm (End)", lines2)

    def test_skewed_bridge_geometry(self):
        """Verify transverse line detection on a bridge with a 20-degree skew angle."""
        import math
        skew_rad = math.radians(20)
        skew_nodes = {}
        for nid, (x, y, z) in self.nodes.items():
            # Apply skew shift: X_skew = X - Z * tan(skew)
            skew_nodes[nid] = [x - z * math.tan(skew_rad), y, z]

        # Test with metadata
        lines_meta = _find_transverse_lines(skew_nodes, self.members, ds=self.mock_ds)
        self.assertIn("End Diaphragm (Start)", lines_meta)
        self.assertIn("End Diaphragm (End)", lines_meta)
        self.assertEqual(len(lines_meta), 6)

        # Test geometric fallback on skewed geometry
        lines_geom = _find_transverse_lines(skew_nodes, self.members, ds=None)
        self.assertEqual(list(lines_geom.keys()), list(lines_meta.keys()))
        for k in lines_meta:
            self.assertEqual(lines_geom[k], lines_meta[k])

    def test_overhang_trimming_with_edge_dist(self):
        """Verify that cantilever overhang elements outside exterior girders are trimmed when edge_dist > 0."""
        # Add overhang nodes at Z = -1.0 (left overhang) and Z = 8.5 (right overhang) at station 0
        nodes = dict(self.nodes)
        members = dict(self.members)

        # Left overhang element: node 100 (-1.0) -> node 1 (0.0)
        nodes[100] = [0.0, 0.0, -1.0]
        members[901] = [100, 1]

        # Right overhang element: node 4 (7.5) -> node 101 (8.5)
        nodes[101] = [0.0, 0.0, 8.5]
        members[902] = [4, 101]

        # Total elements at start line including overhangs: [901, start_trans_eles..., 902]
        full_start_eles = [901] + self.start_trans_eles + [902]
        member_map = {
            "start_edge": {"elements": [full_start_eles]},
            "end_edge": {"elements": [self.end_trans_eles]},
            "transverse_slab": {"elements": [self.cb_trans_eles]},
        }
        ds = MockDataset(attrs={"member_elements": json.dumps(member_map)})

        # When edge_dist = 0: All elements including overhangs are retained
        lines_untrimmed = _find_transverse_lines(nodes, members, ds=ds, edge_dist=0.0)
        self.assertEqual(len(lines_untrimmed["End Diaphragm (Start)"]), 5)

        # When edge_dist = 1.0: Overhang elements outside [0.0, 7.5] are trimmed
        lines_trimmed = _find_transverse_lines(nodes, members, ds=ds, edge_dist=1.0)
        self.assertEqual(len(lines_trimmed["End Diaphragm (Start)"]), 3)
        self.assertEqual(lines_trimmed["End Diaphragm (Start)"], self.start_trans_eles)

    def test_regression_longitudinal_girders_unaffected(self):
        """Verify that _find_girders continues to work exactly as before without interference."""
        girders = _find_girders(self.nodes, self.members)
        self.assertEqual(len(girders), 4)  # 4 longitudinal girders at Z = 0, 2.5, 5.0, 7.5
        for z_val, elems in girders.items():
            self.assertEqual(len(elems), 5)  # 5 span segments per girder

    def test_transverse_sfd_figure_generation(self):
        """Verify SFD figure and summary generation for a selected transverse line."""
        fig, summary = build_figure_sfd(
            self.results_ds, "Fy", self.nodes, self.members, selected_girder="End Diaphragm (Start)"
        )
        self.assertIsInstance(fig, Figure)
        self.assertIn("End Diaphragm (Start)", summary)
        self.assertIn("max", summary["End Diaphragm (Start)"])
        self.assertIn("min", summary["End Diaphragm (Start)"])
        ax = fig.axes[0]
        self.assertIn("End Diaphragm (Start)", ax.get_title())
        self.assertIn("V$_y$", ax.get_title())

    def test_transverse_bmd_figure_generation_mx_mapping(self):
        """Verify BMD figure generation with automatic mapping from Mz to Mx for transverse members."""
        fig, summary = build_figure_bmd(
            self.results_ds, "Mz", self.nodes, self.members, selected_girder="Cross Bracing 1"
        )
        self.assertIsInstance(fig, Figure)
        self.assertIn("Cross Bracing 1", summary)
        self.assertIn("max", summary["Cross Bracing 1"])
        self.assertIn("min", summary["Cross Bracing 1"])
        ax = fig.axes[0]
        self.assertIn("Cross Bracing 1", ax.get_title())
        self.assertIn("M$_x$", ax.get_title())
        self.assertIn("M$_x$", ax.get_zlabel())

    def test_transverse_deflection_figure_generation(self):
        """Verify Deflection figure and summary generation for a selected transverse member."""
        fig, summary = build_figure_deflection(
            self.results_ds, "Dy", self.nodes, self.members, selected_girder="Cross Bracing 2"
        )
        self.assertIsInstance(fig, Figure)
        self.assertIn("Cross Bracing 2", summary)
        self.assertIn("max", summary["Cross Bracing 2"])
        self.assertIn("min", summary["Cross Bracing 2"])
        ax = fig.axes[0]
        self.assertIn("Cross Bracing 2", ax.get_title())
        self.assertIn("mm", ax.get_title())

    def test_transverse_axial_figure_generation(self):
        """Verify Axial force (Fz -> Vz) figure generation for transverse members."""
        fig, summary = build_figure_sfd(
            self.results_ds, "Fz", self.nodes, self.members, selected_girder="End Diaphragm (End)"
        )
        self.assertIsInstance(fig, Figure)
        self.assertIn("End Diaphragm (End)", summary)
        ax = fig.axes[0]
        self.assertIn("End Diaphragm (End)", ax.get_title())
        self.assertIn("V$_z$", ax.get_title())

    def test_transverse_grillage_figure_generation(self):
        """Verify grillage mesh figure generation with a selected transverse line highlighted."""
        fig = build_figure_grillage(
            self.nodes, self.members, selected_girder="Cross Bracing 1"
        )
        self.assertIsInstance(fig, Figure)

    def test_build_figure_grillage_transverse_highlighting_distinct_stations(self):
        """Verify that End Diaphragm (Start), End Diaphragm (End), and Cross Bracing highlight distinct elements at their respective stations."""
        # 1. End Diaphragm (Start) at X = 0.0
        fig_start = build_figure_grillage(
            self.nodes, self.members, selected_girder="End Diaphragm (Start)", ds=self.mock_ds
        )
        ax_start = fig_start.axes[0]
        red_start = [line.get_data_3d() for line in ax_start.lines if line.get_color() == "#C62828"]
        self.assertGreater(len(red_start), 0)
        for xs, ys, zs in red_start:
            self.assertTrue(all(abs(x - 0.0) < 1e-3 for x in xs))

        # 2. Cross Bracing 1 at X = 6.0
        fig_cb1 = build_figure_grillage(
            self.nodes, self.members, selected_girder="Cross Bracing 1", ds=self.mock_ds
        )
        ax_cb1 = fig_cb1.axes[0]
        red_cb1 = [line.get_data_3d() for line in ax_cb1.lines if line.get_color() == "#C62828"]
        self.assertGreater(len(red_cb1), 0)
        for xs, ys, zs in red_cb1:
            self.assertTrue(all(abs(x - 6.0) < 1e-3 for x in xs))

        # 3. End Diaphragm (End) at X = 30.0
        fig_end = build_figure_grillage(
            self.nodes, self.members, selected_girder="End Diaphragm (End)", ds=self.mock_ds
        )
        ax_end = fig_end.axes[0]
        red_end = [line.get_data_3d() for line in ax_end.lines if line.get_color() == "#C62828"]
        self.assertGreater(len(red_end), 0)
        for xs, ys, zs in red_end:
            self.assertTrue(all(abs(x - 30.0) < 1e-3 for x in xs))

    def test_mpl_plot_widget_grillage_selection_refresh(self):
        """Verify MplPlotWidget.update_plot() refreshes the Grillage view on member dropdown change and preserves camera angles."""
        from PySide6.QtWidgets import QApplication, QWidget, QComboBox, QVBoxLayout
        from osdagbridge.desktop.ui.mpl_plot_widget import MplPlotWidget

        app = QApplication.instance() or QApplication([])

        ds_all = self.results_ds.expand_dims(dim={"Loadcase": ["case1"]})

        class MockDock(QWidget):
            def __init__(self):
                super().__init__()
                self.output_widget = QWidget(self)
                layout = QVBoxLayout(self.output_widget)
                self.combo_member = QComboBox(self.output_widget)
                self.combo_member.setObjectName("analysis.member")
                self.combo_member.addItems([
                    "All", "G1", "End Diaphragm (Start)", "Cross Bracing 1", "Cross Bracing 2", "End Diaphragm (End)"
                ])
                layout.addWidget(self.combo_member)

                self.combo_lc = QComboBox(self.output_widget)
                self.combo_lc.setObjectName("analysis.load_combination")
                self.combo_lc.addItems(["case1"])
                layout.addWidget(self.combo_lc)

        dock = MockDock()
        plot_w = MplPlotWidget()
        plot_w.setup(ds_all, ["case1"], self.nodes, self.members, edge_dist=0.0)
        plot_w.link_output_dock(dock)

        # 1. Switch to Grillage mode and set a test camera orientation
        plot_w._on_grillage_toggled(True)
        self.assertTrue(plot_w._grillage_mode)
        plot_w._fig.axes[0].view_init(elev=42, azim=123)

        # 2. Select End Diaphragm (Start)
        dock.combo_member.setCurrentText("End Diaphragm (Start)")
        ax = plot_w._fig.axes[0]
        red_start = [l.get_data_3d() for l in ax.lines if l.get_color() == "#C62828"]
        self.assertGreater(len(red_start), 0)
        for xs, ys, zs in red_start:
            self.assertTrue(all(abs(x - 0.0) < 1e-3 for x in xs))
        self.assertEqual(ax.elev, 42)
        self.assertEqual(ax.azim, 123)

        # 3. Select Cross Bracing 2 (at X = 12.0)
        dock.combo_member.setCurrentText("Cross Bracing 2")
        ax = plot_w._fig.axes[0]
        red_cb2 = [l.get_data_3d() for l in ax.lines if l.get_color() == "#C62828"]
        self.assertGreater(len(red_cb2), 0)
        for xs, ys, zs in red_cb2:
            self.assertTrue(all(abs(x - 12.0) < 1e-3 for x in xs))
        self.assertEqual(ax.elev, 42)
        self.assertEqual(ax.azim, 123)

        # 4. Select End Diaphragm (End) (at X = 30.0)
        dock.combo_member.setCurrentText("End Diaphragm (End)")
        ax = plot_w._fig.axes[0]
        red_end = [l.get_data_3d() for l in ax.lines if l.get_color() == "#C62828"]
        self.assertGreater(len(red_end), 0)
        for xs, ys, zs in red_end:
            self.assertTrue(all(abs(x - 30.0) < 1e-3 for x in xs))
        self.assertEqual(ax.elev, 42)
        self.assertEqual(ax.azim, 123)

    def test_longitudinal_regression_plotting(self):
        """Verify that longitudinal member plotting (All and G1) continues working without regression."""
        # SFD: All and G1
        fig_all_sfd, sum_all_sfd = build_figure_sfd(
            self.results_ds, "Fy", self.nodes, self.members, selected_girder="All"
        )
        self.assertIn("G1", sum_all_sfd)
        self.assertIn("G2", sum_all_sfd)

        fig_g1_sfd, sum_g1_sfd = build_figure_sfd(
            self.results_ds, "Fy", self.nodes, self.members, selected_girder="G1"
        )
        self.assertIn("G1", sum_g1_sfd)
        self.assertNotIn("G2", sum_g1_sfd)

        # BMD: All and G1
        fig_all_bmd, sum_all_bmd = build_figure_bmd(
            self.results_ds, "Mz", self.nodes, self.members, selected_girder="All"
        )
        self.assertIn("G1", sum_all_bmd)

        fig_g1_bmd, sum_g1_bmd = build_figure_bmd(
            self.results_ds, "Mz", self.nodes, self.members, selected_girder="G1"
        )
        self.assertIn("G1", sum_g1_bmd)
        self.assertNotIn("G2", sum_g1_bmd)

        # Deflection: All and G1
        fig_all_def, sum_all_def = build_figure_deflection(
            self.results_ds, "Dy", self.nodes, self.members, selected_girder="All"
        )
        self.assertIn("G1", sum_all_def)

        fig_g1_def, sum_g1_def = build_figure_deflection(
            self.results_ds, "Dy", self.nodes, self.members, selected_girder="G1"
        )
        self.assertIn("G1", sum_g1_def)
        self.assertNotIn("G2", sum_g1_def)

    def test_output_dock_member_dropdown_population(self):
        """Verify Analysis member dropdown contains All + Girders + Transverse lines, while Design only has Girders."""
        from PySide6.QtWidgets import QApplication
        from osdagbridge.desktop.ui.docks.output_dock import OutputDock
        from osdagbridge.core.utils.common import (
            KEY_ANALYSIS_MEMBER,
            KEY_OUTPUT_DOCK_MEMBER_ID,
            KEY_MP_CB_NO_OF_CROSS_BRACINGS,
            KEY_TS_NO_OF_GIRDERS,
        )

        app = QApplication.instance() or QApplication([])

        class MockBackend:
            def __init__(self, ds, nodes, members):
                self._ds = ds
                self._nodes = nodes
                self._members = members
                self.input_dict = {
                    KEY_TS_NO_OF_GIRDERS: 4,
                    KEY_MP_CB_NO_OF_CROSS_BRACINGS: 4,
                }

            def output_values(self):
                from osdagbridge.core.bridge_types.plate_girder.ui_fields import FrontendData
                return FrontendData().output_values()

            def get_results_dataset(self):
                return self._ds

            def get_nodes_members(self):
                return self._nodes, self._members

            def get_edge_dist(self):
                return 0.0

            def get_results_data(self):
                return None

        backend = MockBackend(self.results_ds, self.nodes, self.members)
        dock = OutputDock(backend=backend)
        dock.refresh_member_dropdown()

        combo_analysis = dock._w(KEY_ANALYSIS_MEMBER)
        combo_design = dock._w(KEY_OUTPUT_DOCK_MEMBER_ID)

        analysis_items = [combo_analysis.itemText(i) for i in range(combo_analysis.count())]
        design_items = [combo_design.itemText(i) for i in range(combo_design.count())]

        # Verify Analysis dropdown
        expected_transverse = [
            "End Diaphragm (Start)",
            "Cross Bracing 1",
            "Cross Bracing 2",
            "Cross Bracing 3",
            "Cross Bracing 4",
            "End Diaphragm (End)",
        ]
        self.assertEqual(analysis_items[0], "All")
        for g in ["G1", "G2", "G3", "G4"]:
            self.assertIn(g, analysis_items)
        for t in expected_transverse:
            self.assertIn(t, analysis_items)

        # Verify Design dropdown does NOT contain transverse members
        for t in expected_transverse:
            self.assertNotIn(t, design_items)
        self.assertNotIn("All", design_items)


if __name__ == "__main__":
    unittest.main()
