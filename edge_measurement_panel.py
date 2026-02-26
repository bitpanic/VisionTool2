from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFormLayout,
    QSpinBox,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QGroupBox,
)
from PyQt5.QtCore import Qt
from edge_profile_plot import EdgeProfilePlot
import csv
from datetime import datetime


class EdgeMeasurementPanel(QWidget):
    """Panel to display edge measurement mode, parameters, metrics, and plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_results = None
        self._current_profile_index = 0
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Mode and parameter summary
        summary_group = QGroupBox("Edge Measurement Summary")
        summary_layout = QFormLayout(summary_group)
        self.mode_label = QLabel("-")
        self.params_label = QLabel("-")
        self.params_label.setWordWrap(True)
        self.params_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_layout.addRow("Mode:", self.mode_label)
        summary_layout.addRow("Parameters:", self.params_label)
        layout.addWidget(summary_group)

        # Metrics group
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QFormLayout(metrics_group)
        self.metrics_labels = {}
        for name in [
            "peak_height_pos",
            "peak_height_neg",
            "peak_to_peak",
            "area_pos",
            "area_neg",
            "area_total",
            "edge_width",
            "cross_section_count",
            "valid_cross_sections",
        ]:
            lbl = QLabel("-")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            metrics_layout.addRow(name + ":", lbl)
            self.metrics_labels[name] = lbl
        layout.addWidget(metrics_group)

        # Cross-section selection (for multi-profile modes)
        cs_group = QGroupBox("Cross-section View")
        cs_layout = QHBoxLayout(cs_group)
        self.cs_index_spin = QSpinBox()
        self.cs_index_spin.setMinimum(0)
        self.cs_index_spin.setMaximum(0)
        self.cs_index_spin.setEnabled(False)
        self.cs_index_spin.valueChanged.connect(self._on_cs_index_changed)
        self.metric_choice_combo = QComboBox()
        self.metric_choice_combo.addItems(["peak_to_peak", "edge_width"])
        cs_layout.addWidget(QLabel("Index:"))
        cs_layout.addWidget(self.cs_index_spin)
        cs_layout.addWidget(QLabel("Best/worst metric:"))
        cs_layout.addWidget(self.metric_choice_combo)
        layout.addWidget(cs_group)

        # Plot widget
        self.plot_widget = EdgeProfilePlot()
        self.plot_widget.setMinimumHeight(160)
        layout.addWidget(self.plot_widget)

        # Export controls
        export_layout = QHBoxLayout()
        self.export_raw_checkbox = QCheckBox("Export raw profile arrays (where available)")
        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.export_button.setEnabled(False)
        export_layout.addWidget(self.export_raw_checkbox)
        export_layout.addStretch()
        export_layout.addWidget(self.export_button)
        layout.addLayout(export_layout)

        layout.addStretch()

    # --- API used by the EdgeMeasurementPlugin ---

    def set_results(self, results):
        """Update the panel with new measurement results.

        Expected structure (dict, minimal core fields):
            mode: str
            params_summary: str
            single_metrics: dict | None
            aggregate_metrics: dict | None
            cross_sections: list[dict] | None
            profile: dict with keys s, I, dI, peaks, width_points | None
        """
        self._last_results = results
        self.export_button.setEnabled(True)

        mode = results.get("mode", "-")
        self.mode_label.setText(mode)
        self.params_label.setText(results.get("params_summary", "-"))

        # Clear metrics
        for lbl in self.metrics_labels.values():
            lbl.setText("-")

        single = results.get("single_metrics") or {}
        agg = results.get("aggregate_metrics") or {}

        # Mode A: single profile metrics
        for key in ["peak_height_pos", "peak_height_neg", "peak_to_peak", "area_pos", "area_neg", "area_total", "edge_width"]:
            if key in single:
                self.metrics_labels[key].setText(f"{single[key]:.4g}")

        # Mode B/C: aggregate metrics + counts
        if agg:
            for key in ["peak_height_pos", "peak_height_neg", "peak_to_peak", "area_pos", "area_neg", "area_total", "edge_width"]:
                if key in agg:
                    val = agg[key].get("mean", None)
                    if val is not None:
                        self.metrics_labels[key].setText(f"{val:.4g}")
            if "cross_section_count" in agg:
                self.metrics_labels["cross_section_count"].setText(str(agg["cross_section_count"]))
            if "valid_cross_sections" in agg:
                self.metrics_labels["valid_cross_sections"].setText(str(agg["valid_cross_sections"]))

        # Cross-section selector
        cs_list = results.get("cross_sections") or []
        if cs_list:
            self.cs_index_spin.setEnabled(True)
            self.cs_index_spin.setMaximum(len(cs_list) - 1)
            self.cs_index_spin.setValue(results.get("default_cs_index", 0))
        else:
            self.cs_index_spin.setEnabled(False)
            self.cs_index_spin.setMaximum(0)
            self.cs_index_spin.setValue(0)

        # Initial profile for plotting
        profile = results.get("profile")
        if profile:
            self._update_plot_from_profile(profile)
        else:
            self.plot_widget.clear()

    def update_profile(self, profile):
        """Update only the plotted profile (used when changing cross-section index)."""
        self._update_plot_from_profile(profile)

    # --- Internal helpers ---

    def _update_plot_from_profile(self, profile):
        s = profile.get("s")
        I = profile.get("I")
        dI = profile.get("dI")
        peaks = profile.get("peaks") or {}
        width_points = profile.get("width_points") or {}
        self.plot_widget.set_data(
            s=s,
            intensity=I,
            derivative=dI,
            peak_pos_index=peaks.get("pos_index"),
            peak_neg_index=peaks.get("neg_index"),
            width_indices=width_points.get("indices"),
        )

    def _on_cs_index_changed(self, index):
        self._current_profile_index = index
        if not self._last_results:
            return
        cs_list = self._last_results.get("cross_sections") or []
        if 0 <= index < len(cs_list):
            cs = cs_list[index]
            profile = cs.get("profile")
            if profile:
                self._update_plot_from_profile(profile)

    def _on_export_clicked(self):
        if not self._last_results:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Edge Measurement CSV",
            "",
            "CSV Files (*.csv)",
        )
        if not file_name:
            return

        results = self._last_results
        mode = results.get("mode", "")
        params_summary = results.get("params_summary", "")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_name = results.get("image_name", "")

        export_raw = self.export_raw_checkbox.isChecked()

        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)

            # Header block
            writer.writerow(["image_name", image_name])
            writer.writerow(["timestamp", timestamp])
            writer.writerow(["mode", mode])
            writer.writerow(["params", params_summary])
            writer.writerow([])

            if mode.startswith("Along"):
                single = results.get("single_metrics") or {}
                writer.writerow(["Metric", "Value"])
                for key, val in single.items():
                    writer.writerow([key, val])
                writer.writerow([])

                if export_raw:
                    profile = results.get("profile") or {}
                    s = profile.get("s") or []
                    I = profile.get("I") or []
                    dI = profile.get("dI") or []
                    writer.writerow(["index", "s", "I", "dI_ds"])
                    for idx in range(min(len(s), len(I), len(dI))):
                        writer.writerow([idx, s[idx], I[idx], dI[idx]])

            else:
                # Cross-section per-row export
                cs_list = results.get("cross_sections") or []
                agg = results.get("aggregate_metrics") or {}
                writer.writerow(
                    [
                        "index",
                        "center_x",
                        "center_y",
                        "x0",
                        "y0",
                        "x1",
                        "y1",
                        "dir_nx",
                        "dir_ny",
                        "peak_height_pos",
                        "peak_height_neg",
                        "peak_to_peak",
                        "area_pos",
                        "area_neg",
                        "area_total",
                        "edge_width",
                    ]
                )
                for cs in cs_list:
                    geom = cs.get("geometry", {})
                    mets = cs.get("metrics", {})
                    center = geom.get("center", (None, None))
                    p0 = geom.get("p0", (None, None))
                    p1 = geom.get("p1", (None, None))
                    direction = geom.get("direction", (None, None))
                    writer.writerow(
                        [
                            cs.get("index"),
                            center[0],
                            center[1],
                            p0[0],
                            p0[1],
                            p1[0],
                            p1[1],
                            direction[0],
                            direction[1],
                            mets.get("peak_height_pos"),
                            mets.get("peak_height_neg"),
                            mets.get("peak_to_peak"),
                            mets.get("area_pos"),
                            mets.get("area_neg"),
                            mets.get("area_total"),
                            mets.get("edge_width"),
                        ]
                    )

                # Aggregate summary block
                writer.writerow([])
                writer.writerow(["Aggregate metric", "mean", "median", "std", "p10", "p90"])
                for key, stats in agg.items():
                    if not isinstance(stats, dict):
                        continue
                    row = [
                        key,
                        stats.get("mean"),
                        stats.get("median"),
                        stats.get("std"),
                        stats.get("p10"),
                        stats.get("p90"),
                    ]
                    writer.writerow(row)

                # Best/worst cross-section indices (by chosen metric)
                best_idx = results.get("best_index")
                worst_idx = results.get("worst_index")
                writer.writerow([])
                writer.writerow(["best_index", best_idx])
                writer.writerow(["worst_index", worst_idx])

