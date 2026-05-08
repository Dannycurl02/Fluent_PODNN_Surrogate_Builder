"""
Validate Page Module
====================
Model list with checkboxes, metrics dashboard per model,
prediction panel (dataset point or custom params), Fluent comparison.

Design:
  - Click a model = focus it (metrics, prediction view) AND check it
  - Check a model = include in predict/compare batch
  - Predict runs all checked models
  - Fluent comparison runs once, caches per-params to disk
"""

import json
import logging
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFrame, QComboBox, QLineEdit,
    QFormLayout, QStackedWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QDoubleSpinBox, QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QSize

from .. import theme
from ..fluent_manager import FluentManager
from ..dataset_manager import DatasetManager
from ..fluent_cache import FluentCache
from ..workers import ValidationWorker

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    import mpl_toolkits.mplot3d  # noqa: F401  registers 3d projection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# Plotting helpers
# ============================================================

def _make_per_sample_r2_figure(sim_ids, r2_values, title="Per-Sample R² (Test Set)"):
    """Scatter of per-sample field-reconstruction R² for a 2D/3D model. One dot
    per test sample, with a horizontal reference line at R²=1 (perfect)."""
    sim_ids = np.asarray(sim_ids)
    r2_values = np.asarray(r2_values, dtype=float)

    fig = Figure(figsize=(7, 5), facecolor=theme.BG_PANEL, layout='constrained')
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.BG_DARK)
    ax.tick_params(colors=theme.TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(theme.BORDER)

    ax.axhline(1.0, linestyle='--', linewidth=1.0, color=theme.TEXT_SECONDARY,
               label='R² = 1 (perfect)')
    ax.scatter(sim_ids, r2_values, c=theme.ORANGE_LIGHT, s=50,
               edgecolor='white', linewidth=0.5,
               label=f'Test samples (n={len(sim_ids)})')

    mean_r2 = float(np.mean(r2_values)) if len(r2_values) else 0.0
    ax.set_title(f"{title}  (mean R² = {mean_r2:.4f})", color=theme.TEXT_PRIMARY)
    ax.set_xlabel("Sample ID", color=theme.TEXT_SECONDARY)
    ax.set_ylabel("R² (per-sample field reconstruction)", color=theme.TEXT_SECONDARY)
    if len(r2_values):
        lo = min(-0.3, float(r2_values.min()) - 0.25)
        hi = max(1.3, float(r2_values.max()) + 0.2)
    else:
        lo, hi = -0.3, 1.3
    ax.set_ylim(lo, hi)
    ax.grid(True, linestyle='--', alpha=0.3, color=theme.BORDER)
    ax.legend(loc='lower right', facecolor=theme.BG_DARK, edgecolor=theme.BORDER,
              labelcolor=theme.TEXT_PRIMARY)
    return fig


def _make_pred_vs_truth_figure(truths, preds, title="Predicted vs Truth"):
    """Scatter of NN predictions against ground truths for a 1D scalar model.
    Adds a y=x diagonal reference and prints R² as the title suffix."""
    truths = np.asarray(truths).flatten()
    preds = np.asarray(preds).flatten()

    fig = Figure(figsize=(7, 5), facecolor=theme.BG_PANEL, layout='constrained')
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.BG_DARK)
    ax.tick_params(colors=theme.TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(theme.BORDER)

    # Diagonal reference (perfect prediction)
    lo = float(min(truths.min(), preds.min()))
    hi = float(max(truths.max(), preds.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            linestyle='--', linewidth=1.2, color=theme.TEXT_SECONDARY, label='y = x')

    ax.scatter(truths, preds, c=theme.ORANGE_LIGHT, s=40, edgecolor='white',
               linewidth=0.5, label=f'Test set (n={len(truths)})')

    ss_res = np.sum((truths - preds) ** 2)
    ss_tot = np.sum((truths - truths.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    ax.set_title(f"{title}  (R² = {r2:.4f})", color=theme.TEXT_PRIMARY)
    ax.set_xlabel("Ground Truth", color=theme.TEXT_SECONDARY)
    ax.set_ylabel("NN Prediction", color=theme.TEXT_SECONDARY)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, linestyle='--', alpha=0.3, color=theme.BORDER)
    ax.legend(loc='best', facecolor=theme.BG_DARK, edgecolor=theme.BORDER,
              labelcolor=theme.TEXT_PRIMARY)
    return fig


def _make_scalar_bar_figure(nn_val, fluent_val, truth_val, label):
    """Bar chart for 1D scalar comparison. Missing values skipped."""
    fig = Figure(figsize=(6, 3.5), facecolor=theme.BG_PANEL)
    ax = fig.add_subplot(111)
    _style_axis(ax)

    categories = []
    values = []
    colors = []
    if nn_val is not None:
        categories.append('NN')
        values.append(nn_val)
        colors.append(theme.ORANGE_LIGHT)
    if fluent_val is not None:
        categories.append('Fluent')
        values.append(fluent_val)
        colors.append(theme.BLUE_INFO)
    if truth_val is not None:
        categories.append('Dataset')
        values.append(truth_val)
        colors.append(theme.GREEN_SUCCESS)

    if not values:
        ax.text(0.5, 0.5, "No values", ha='center', va='center',
                color=theme.TEXT_SECONDARY, transform=ax.transAxes)
    else:
        bars = ax.bar(categories, values, color=colors, edgecolor=theme.BORDER)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4g}", ha='center', va='bottom',
                    color=theme.TEXT_PRIMARY)
    ax.set_ylabel(label, color=theme.TEXT_SECONDARY)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color=theme.BORDER)
    fig.tight_layout()
    return fig


def _is_3d_data(coordinates, eps_rel=1e-3):
    """True if all three coordinate axes have non-trivial span."""
    if coordinates is None:
        return False
    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] < 3:
        return False
    spans = coords.max(axis=0) - coords.min(axis=0)
    max_span = spans.max()
    if max_span == 0:
        return False
    return np.sum(spans > eps_rel * max_span) >= 3


def _downsample(n_points, max_points):
    """Return an array of indices to plot, randomly chosen if n_points > max_points."""
    if max_points <= 0 or n_points <= max_points:
        return np.arange(n_points)
    rng = np.random.default_rng(seed=42)  # deterministic per run
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


def _make_field_figure(nn_values, truth_values, coordinates=None, title="Field",
                        max_points=5000, is_3d=False):
    """
    Horizontal 1x3 figure for 2D/3D field comparison: NN | Truth | Abs Error.
    NN/Truth share a colorbar; Error has its own.
    If only NN is available, shows just that panel.

    is_3d : bool
        Drives whether each panel uses a 3D scatter or a 2D scatter (auto-picking
        the two widest coordinate dims). Caller supplies this from the model's
        metadata `output_type` — geometry-based inference fails for 2D fields
        embedded in 3D space (e.g. tilted mid-plane surfaces).

    Large fields are randomly downsampled to max_points for faster rendering
    and interactivity.
    """
    # Apply random downsampling to all arrays together
    nn_values = np.asarray(nn_values).flatten()
    coordinates = np.asarray(coordinates) if coordinates is not None else None
    truth_values = np.asarray(truth_values).flatten() if truth_values is not None else None

    n_full = len(nn_values)
    if n_full > max_points:
        idx = _downsample(n_full, max_points)
        nn_values = nn_values[idx]
        if coordinates is not None and len(coordinates) == n_full:
            coordinates = coordinates[idx]
        if truth_values is not None and len(truth_values) == n_full:
            truth_values = truth_values[idx]

    # Use constrained_layout so colorbars get their own space automatically
    fig = Figure(figsize=(13, 4.5), facecolor=theme.BG_PANEL, layout='constrained')

    has_truth = truth_values is not None

    if not has_truth:
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        _style_axis(ax, is_3d=is_3d)
        im = _plot_field(ax, nn_values, coordinates, title="NN Prediction", is_3d=is_3d)
        if im is not None:
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
            _style_colorbar(cb)
        return fig

    # 1x3 layout
    if is_3d:
        ax_nn = fig.add_subplot(1, 3, 1, projection='3d')
        ax_truth = fig.add_subplot(1, 3, 2, projection='3d')
        ax_err = fig.add_subplot(1, 3, 3, projection='3d')
    else:
        ax_nn = fig.add_subplot(1, 3, 1)
        ax_truth = fig.add_subplot(1, 3, 2)
        ax_err = fig.add_subplot(1, 3, 3)

    for ax in [ax_nn, ax_truth, ax_err]:
        _style_axis(ax, is_3d=is_3d)

    # Shared color scale for NN/Truth
    vmin = float(min(np.nanmin(nn_values), np.nanmin(truth_values)))
    vmax = float(max(np.nanmax(nn_values), np.nanmax(truth_values)))

    im1 = _plot_field(ax_nn, nn_values, coordinates, title="NN",
                      vmin=vmin, vmax=vmax, is_3d=is_3d)
    im2 = _plot_field(ax_truth, truth_values, coordinates, title="Truth",
                      vmin=vmin, vmax=vmax, is_3d=is_3d)
    # Each of NN and Truth gets its own colorbar so constrained_layout
    # can allocate space without overlapping the next axis.
    if im1 is not None:
        cb1 = fig.colorbar(im1, ax=ax_nn, fraction=0.05, pad=0.04)
        _style_colorbar(cb1)
    if im2 is not None:
        cb2 = fig.colorbar(im2, ax=ax_truth, fraction=0.05, pad=0.04)
        _style_colorbar(cb2)

    err = np.abs(np.asarray(nn_values).flatten() - np.asarray(truth_values).flatten())
    im3 = _plot_field(ax_err, err, coordinates, title="Absolute Error",
                      cmap='Reds', is_3d=is_3d)
    if im3 is not None:
        cb3 = fig.colorbar(im3, ax=ax_err, fraction=0.05, pad=0.04)
        _style_colorbar(cb3)

    return fig


def _style_colorbar(cb):
    cb.ax.tick_params(colors=theme.TEXT_SECONDARY)
    for spine in cb.ax.spines.values():
        spine.set_color(theme.BORDER)


def _make_single_field_figure(values, coordinates=None, title="Field",
                               cmap='viridis', max_points=300, is_3d=False):
    """Single-panel figure for one field (NN, Truth, or Error) in a pop-out window.

    is_3d : bool
        Caller supplies from metadata. See _make_field_figure for why we don't
        infer this from coordinates.
    """
    values = np.asarray(values).flatten()
    coordinates = np.asarray(coordinates) if coordinates is not None else None
    n_full = len(values)
    if n_full > max_points and coordinates is not None and len(coordinates) == n_full:
        idx = _downsample(n_full, max_points)
        values = values[idx]
        coordinates = coordinates[idx]

    fig = Figure(figsize=(8, 6), facecolor=theme.BG_PANEL, layout='constrained')

    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    _style_axis(ax, is_3d=is_3d)

    im = _plot_field(ax, values, coordinates, title=title, cmap=cmap, is_3d=is_3d)
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        _style_colorbar(cb)
    return fig


def _style_axis(ax, is_3d=False):
    if is_3d:
        # 3D axes have different APIs
        ax.set_facecolor(theme.BG_DARK)
        ax.tick_params(colors=theme.TEXT_SECONDARY)
        try:
            ax.xaxis.pane.set_facecolor(theme.BG_DARK)
            ax.yaxis.pane.set_facecolor(theme.BG_DARK)
            ax.zaxis.pane.set_facecolor(theme.BG_DARK)
            ax.xaxis.pane.set_edgecolor(theme.BORDER)
            ax.yaxis.pane.set_edgecolor(theme.BORDER)
            ax.zaxis.pane.set_edgecolor(theme.BORDER)
        except Exception:
            pass
        ax.xaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.yaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.zaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.title.set_color(theme.TEXT_PRIMARY)
    else:
        ax.set_facecolor(theme.BG_DARK)
        ax.tick_params(colors=theme.TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(theme.BORDER)
        ax.xaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.yaxis.label.set_color(theme.TEXT_SECONDARY)
        ax.title.set_color(theme.TEXT_PRIMARY)


def _plot_field(ax, values, coordinates, title, cmap='viridis',
                vmin=None, vmax=None, is_3d=False):
    """
    Scatter plot a field.
    Returns the ScalarMappable for colorbar, or None.
    """
    ax.set_title(title)
    values = np.asarray(values).flatten()
    if coordinates is None or len(coordinates) != len(values):
        ax.plot(values, color=theme.ORANGE_LIGHT)
        ax.set_xlabel("Point index")
        ax.set_ylabel("Value")
        return None

    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] < 2:
        ax.plot(values, color=theme.ORANGE_LIGHT)
        return None

    if is_3d and coords.shape[1] >= 3:
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=values, s=4, cmap=cmap, vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # Enforce 1:1:1 aspect ratio in data space
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        x_mid = (coords[:, 0].max() + coords[:, 0].min()) / 2
        y_mid = (coords[:, 1].max() + coords[:, 1].min()) / 2
        z_mid = (coords[:, 2].max() + coords[:, 2].min()) / 2
        spans = coords.max(axis=0) - coords.min(axis=0)
        half = spans.max() / 2
        if half > 0:
            ax.set_xlim(x_mid - half, x_mid + half)
            ax.set_ylim(y_mid - half, y_mid + half)
            ax.set_zlim(z_mid - half, z_mid + half)
        return scatter

    # 2D: pick the two widest dimensions
    spans = coords.max(axis=0) - coords.min(axis=0)
    dims = np.argsort(-spans)[:2]
    x = coords[:, dims[0]]
    y = coords[:, dims[1]]
    scatter = ax.scatter(x, y, c=values, s=6, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel(f"axis {dims[0]}")
    ax.set_ylabel(f"axis {dims[1]}")
    return scatter


# ============================================================
# Validate page
# ============================================================

class ValidatePage(QWidget):
    """Model validation and prediction page."""

    def __init__(self, project, settings, parent=None):
        super().__init__(parent)
        self.project = project
        self.settings = settings
        self.dm = DatasetManager(project.dataset_dir)
        self.fluent_cache = FluentCache(
            project.fluent_cache_dir, project.fluent_cache_index_file)
        self._worker = None
        self._models_meta = {}      # name -> metadata dict
        self._current_model = None  # focused model name
        self._last_fluent_data = None  # dict of NPZ keys -> arrays from last comparison run

        self._build_ui()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Analyze")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        layout.addSpacing(8)

        # --- Top row: model list (left) + metrics table (right) ---
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        # Left column: model list with its header
        left_col = QVBoxLayout()
        list_row = QHBoxLayout()
        list_label = QLabel("Trained Models")
        list_label.setProperty("secondary", True)
        list_row.addWidget(list_label)
        list_row.addStretch()

        self._export_btn = QPushButton("Export Model")
        self._export_btn.setProperty("flat", True)
        self._export_btn.setFixedWidth(160)
        self._export_btn.clicked.connect(self._export_model)
        list_row.addWidget(self._export_btn)

        self._delete_btn = QPushButton("Delete Model")
        self._delete_btn.setProperty("flat", True)
        self._delete_btn.setFixedWidth(160)
        self._delete_btn.clicked.connect(self._delete_model)
        list_row.addWidget(self._delete_btn)
        left_col.addLayout(list_row)

        self._model_list = QListWidget()
        self._model_list.setSpacing(2)
        self._model_list.itemClicked.connect(self._on_model_clicked)
        left_col.addWidget(self._model_list, 1)
        top_row.addLayout(left_col, 1)

        # Right column: metrics for the focused model
        right_col = QVBoxLayout()
        metrics_label = QLabel("Metrics")
        metrics_label.setProperty("secondary", True)
        right_col.addWidget(metrics_label)

        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(2)
        self._metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        right_col.addWidget(self._metrics_table, 1)
        top_row.addLayout(right_col, 1)

        layout.addLayout(top_row, 1)
        layout.addSpacing(8)

        # --- Middle row: training history (left) | per-model validation plot (right) ---
        self._mid_frame = QFrame()
        self._mid_frame.setProperty("panel", True)
        mid_outer = QHBoxLayout(self._mid_frame)
        mid_outer.setContentsMargins(8, 8, 8, 8)
        mid_outer.setSpacing(8)

        # Left: training loss curve PNG
        loss_col = QVBoxLayout()
        loss_caption = QLabel("Training Loss — Click to expand")
        loss_caption.setProperty("secondary", True)
        loss_caption.setAlignment(Qt.AlignCenter)
        loss_col.addWidget(loss_caption)
        self._loss_png_label = QLabel("Select a model to see its loss curve.")
        self._loss_png_label.setAlignment(Qt.AlignCenter)
        self._loss_png_label.setMinimumHeight(180)
        self._loss_png_label.setStyleSheet(f"background: {theme.BG_DARK}; border: 1px solid {theme.BORDER};")
        loss_col.addWidget(self._loss_png_label, 1)
        mid_outer.addLayout(loss_col, 1)

        # Right: pred-vs-truth (1D) or per-sample R² (2D/3D)
        pvt_col = QVBoxLayout()
        self._pvt_caption = QLabel("Test Set Validation")
        self._pvt_caption.setProperty("secondary", True)
        self._pvt_caption.setAlignment(Qt.AlignCenter)
        pvt_col.addWidget(self._pvt_caption)
        # Container we swap canvases into when a model is focused.
        self._pvt_container = QFrame()
        self._pvt_container.setStyleSheet(f"background: {theme.BG_DARK}; border: 1px solid {theme.BORDER};")
        self._pvt_layout = QVBoxLayout(self._pvt_container)
        self._pvt_layout.setContentsMargins(0, 0, 0, 0)
        self._pvt_placeholder = QLabel("Select a model to compute test-set validation.")
        self._pvt_placeholder.setAlignment(Qt.AlignCenter)
        self._pvt_layout.addWidget(self._pvt_placeholder)
        pvt_col.addWidget(self._pvt_container, 1)
        mid_outer.addLayout(pvt_col, 1)

        self._mid_frame.hide()
        layout.addWidget(self._mid_frame, 1)
        layout.addSpacing(8)

        # --- Dashboard (lower section: mode, params, action buttons, results tabs) ---
        self._dashboard = QFrame()
        self._dashboard.setProperty("panel", True)
        dash_layout = QVBoxLayout(self._dashboard)
        dash_layout.setContentsMargins(16, 16, 16, 16)

        # Prediction mode row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Prediction mode:"))
        self._pred_mode = QComboBox()
        self._pred_mode.addItems(["Dataset Point", "Custom Parameters"])
        self._pred_mode.currentIndexChanged.connect(self._on_pred_mode_changed)
        mode_row.addWidget(self._pred_mode)
        mode_row.addSpacing(16)

        self._dataset_combo = QComboBox()
        self._dataset_combo.setMinimumWidth(180)
        mode_row.addWidget(self._dataset_combo)
        mode_row.addStretch()
        dash_layout.addLayout(mode_row)

        # Custom params form (hidden in dataset-point mode)
        self._custom_scroll = QScrollArea()
        self._custom_scroll.setWidgetResizable(True)
        self._custom_scroll.setMaximumHeight(120)
        self._custom_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._custom_container = QWidget()
        self._custom_form = QFormLayout(self._custom_container)
        self._custom_scroll.setWidget(self._custom_container)
        self._custom_scroll.hide()
        dash_layout.addWidget(self._custom_scroll)

        self._extrap_warning = QLabel("")
        self._extrap_warning.setStyleSheet(f"color: {theme.YELLOW_WARNING}; background: transparent;")
        self._extrap_warning.hide()
        dash_layout.addWidget(self._extrap_warning)

        # Action buttons
        btn_row = QHBoxLayout()
        self._predict_btn = QPushButton("Predict")
        self._predict_btn.setFixedWidth(120)
        self._predict_btn.clicked.connect(self._run_predict)
        btn_row.addWidget(self._predict_btn)

        self._compare_btn = QPushButton("Run Fluent Comparison")
        self._compare_btn.setFixedWidth(180)
        self._compare_btn.clicked.connect(self._run_fluent_comparison)
        btn_row.addWidget(self._compare_btn)

        self._clear_cache_btn = QPushButton("Clear Cache")
        self._clear_cache_btn.setProperty("flat", True)
        self._clear_cache_btn.setFixedWidth(120)
        self._clear_cache_btn.clicked.connect(self._clear_cache)
        btn_row.addWidget(self._clear_cache_btn)

        self._cache_label = QLabel("")
        self._cache_label.setProperty("secondary", True)
        btn_row.addWidget(self._cache_label)

        btn_row.addSpacing(16)
        btn_row.addWidget(QLabel("Plot points:"))
        from PySide6.QtWidgets import QSpinBox
        self._downsample_spin = QSpinBox()
        self._downsample_spin.setRange(100, 1000000)
        self._downsample_spin.setValue(300)
        self._downsample_spin.setButtonSymbols(QSpinBox.NoButtons)
        self._downsample_spin.setFixedWidth(100)
        self._downsample_spin.setToolTip(
            "Maximum number of points to plot. Larger fields are randomly downsampled."
        )
        btn_row.addWidget(self._downsample_spin)
        btn_row.addStretch()
        dash_layout.addLayout(btn_row)

        # Results tab widget (one tab per model after predict)
        # Single result area — replaces a per-model tab strip. Tracks the
        # focused model only (no separate selector).
        self._results_container = QFrame()
        self._results_container_layout = QVBoxLayout(self._results_container)
        self._results_container_layout.setContentsMargins(0, 0, 0, 0)
        self._results_placeholder = QLabel(
            "Click Predict to compute predictions for the focused model."
        )
        self._results_placeholder.setProperty("secondary", True)
        self._results_placeholder.setAlignment(Qt.AlignCenter)
        self._results_container_layout.addWidget(self._results_placeholder)
        dash_layout.addWidget(self._results_container, 1)

        # Dashboard stays visible whenever any model exists; refresh() auto-focuses
        # the first model so the user lands on a populated dashboard.
        # 50/50 vertical split with the top row (model list + metrics).
        self._dashboard.hide()
        layout.addWidget(self._dashboard, 1)
        self._empty_label = QLabel("No trained models yet. Train a model to see metrics here.")
        self._empty_label.setProperty("secondary", True)
        self._empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._empty_label, 1)

    # ------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------

    def refresh(self):
        """Reload model list from disk."""
        self._model_list.clear()
        self._models_meta.clear()

        if not self.project.models_dir.exists():
            self._update_cache_label()
            return

        for model_dir in sorted(self.project.models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            meta_files = list(model_dir.glob("*_metadata.json"))
            if not meta_files:
                continue
            try:
                with open(meta_files[0], 'r') as f:
                    meta = json.load(f)
            except Exception:
                continue

            name = model_dir.name
            test_metrics = meta.get('test_metrics', {})
            r2 = test_metrics.get('r2', 'N/A')
            if isinstance(r2, float):
                r2 = f"{r2:.4f}"
            dv = meta.get('dataset_version', '?')
            date = meta.get('trained_date', '?')
            if isinstance(date, str) and 'T' in date:
                date = date.split('T')[0]
            stale = self.dm.is_model_stale(meta)

            display = f"{name}    R² {r2}    v{dv}    {date}"
            if stale:
                display += "    [STALE]"

            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, name)
            self._model_list.addItem(item)

            self._models_meta[name] = meta

        # Populate dataset point combo
        self._dataset_combo.clear()
        for sid in self.dm.get_completed_ids():
            self._dataset_combo.addItem(f"Sample {sid}", sid)

        self._update_cache_label()

        # Auto-focus the first model so the dashboard is ready on tab open.
        if self._model_list.count() > 0:
            first = self._model_list.item(0)
            self._model_list.setCurrentItem(first)
            first_name = first.data(Qt.UserRole)
            if first_name:
                self._focus_model(first_name)
                self._empty_label.hide()
        else:
            self._mid_frame.hide()
            self._dashboard.hide()
            self._empty_label.show()

    def _update_cache_label(self):
        n = self.fluent_cache.count()
        self._cache_label.setText(f"{n} cached run{'s' if n != 1 else ''}")

    # ------------------------------------------------------------
    # Model list interaction
    # ------------------------------------------------------------

    def _on_model_clicked(self, item):
        """Click focuses a model — Predict targets whichever is focused."""
        name = item.data(Qt.UserRole)
        if name:
            self._focus_model(name)

    def _focus_model(self, name):
        """Show metrics dashboard for this model."""
        self._current_model = name
        meta = self._models_meta.get(name, {})
        self._empty_label.hide()
        self._mid_frame.show()
        self._dashboard.show()
        self._refresh_middle_section(name, meta)
        # Clear stale prediction output — predictions belong to a (model, params)
        # pair and switching focus invalidates them.
        self._show_result_message(
            "Click Predict to compute predictions for the focused model."
        )

        # Metrics
        test_metrics = meta.get('test_metrics', {})
        train_metrics = meta.get('train_metrics', {})
        rows = []
        for key in ['r2', 'rmse', 'mae']:
            test_val = test_metrics.get(key, 'N/A')
            train_val = train_metrics.get(key, 'N/A')
            if isinstance(test_val, float):
                test_val = f"{test_val:.6f}"
            if isinstance(train_val, float):
                train_val = f"{train_val:.6f}"
            rows.append((f"Test {key.upper()}", str(test_val)))
            rows.append((f"Train {key.upper()}", str(train_val)))

        rows.append(("Output", meta.get('output_key', '?')))
        rows.append(("Type", meta.get('output_type', '?')))
        rows.append(("Dataset Version", str(meta.get('dataset_version', '?'))))
        date = meta.get('trained_date', '?')
        if isinstance(date, str) and 'T' in date:
            date = date.split('T')[0]
        rows.append(("Date Trained", date))

        self._metrics_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self._metrics_table.setItem(i, 0, QTableWidgetItem(k))
            self._metrics_table.setItem(i, 1, QTableWidgetItem(v))

        self._build_custom_param_inputs()

    # ------------------------------------------------------------
    # Prediction mode
    # ------------------------------------------------------------

    def _on_pred_mode_changed(self, idx):
        self._dataset_combo.setVisible(idx == 0)
        self._custom_scroll.setVisible(idx == 1)
        self._extrap_warning.hide()

    def _build_custom_param_inputs(self):
        """Build parameter input fields from model_setup.json."""
        while self._custom_form.count():
            item = self._custom_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._param_spins = {}
        if not self.project.model_setup_file.exists():
            return
        try:
            with open(self.project.model_setup_file, 'r') as f:
                setup = json.load(f)
        except Exception:
            return

        from cfdtwin.doe import load_doe_samples
        _, ranges = load_doe_samples(self.project.doe_samples_file)

        for inp in setup.get('model_inputs', []):
            key = f"{inp['name']}|{inp.get('parameter', 'value')}"
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-1e12, 1e12)
            spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
            rng = ranges.get(key, {})
            if rng:
                mid = (rng['min'] + rng['max']) / 2
                spin.setValue(mid)
            label = key.replace('|', ' : ')
            self._custom_form.addRow(label, spin)
            self._param_spins[key] = (spin, rng)

    def _get_current_params(self):
        """Get the current input params as {bc|param: value}, plus the mode."""
        if self._pred_mode.currentIndex() == 0:
            # Dataset point: look up from doe_samples
            sim_id = self._dataset_combo.currentData()
            if sim_id is None:
                return None, None
            from cfdtwin.doe import load_doe_samples
            samples, _ = load_doe_samples(self.project.doe_samples_file)
            if sim_id - 1 >= len(samples):
                return None, None
            return dict(samples[sim_id - 1]), sim_id
        else:
            params = {k: spin.value() for k, (spin, _) in self._param_spins.items()}
            return params, None

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------

    def _run_predict(self):
        name = self._current_model
        if not name:
            QMessageBox.warning(self, "No Model", "Select a model in the list first.")
            return

        params, sim_id = self._get_current_params()
        if params is None:
            QMessageBox.warning(self, "No Params", "Select a dataset point or fill custom parameters.")
            return

        # Extrapolation check for custom mode
        if self._pred_mode.currentIndex() == 1:
            outside = []
            for key, (spin, rng) in self._param_spins.items():
                val = spin.value()
                if rng and (val < rng.get('min', val) or val > rng.get('max', val)):
                    outside.append(key)
            if outside:
                self._extrap_warning.setText(
                    f"Warning: extrapolating outside DOE range for: "
                    f"{', '.join(k.replace('|', ':') for k in outside)}"
                )
                self._extrap_warning.show()
            else:
                self._extrap_warning.hide()

        # Build X vector in the same sorted-key order the trainer uses
        param_names = sorted(params.keys())
        X = np.array([[params[k] for k in param_names]])

        # Check for cached Fluent data for these params
        cached_fluent = self.fluent_cache.lookup(params)
        if cached_fluent is not None:
            logger.info("Found cached Fluent data for these params")
        self._last_fluent_data = cached_fluent

        from cfdtwin.visualization import predict_single_model, predict_dataset_point_single
        from cfdtwin.doe import load_doe_samples

        doe_samples, _ = load_doe_samples(self.project.doe_samples_file)
        model_dir = self.project.models_dir / name
        try:
            if sim_id is not None:
                result = predict_dataset_point_single(
                    model_dir, self.project.dataset_dir, doe_samples, sim_id)
            else:
                result = predict_single_model(model_dir, X)
        except Exception as e:
            logger.error(f"Prediction failed for {name}: {e}")
            self._show_result_message(f"Error: {e}", error=True)
            return

        if result is None:
            self._show_result_message("Prediction returned None", error=True)
            return

        self._show_result(name, result, cached_fluent)

    def _clear_results_container(self):
        while self._results_container_layout.count():
            item = self._results_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_result_message(self, msg, error=False):
        self._clear_results_container()
        label = QLabel(msg)
        label.setAlignment(Qt.AlignCenter)
        if error:
            label.setStyleSheet(f"color: {theme.RED_ERROR}; background: transparent;")
        else:
            label.setProperty("secondary", True)
        label.setWordWrap(True)
        self._results_container_layout.addWidget(label)

    def _show_result(self, name, result, fluent_data):
        """Render the prediction result for a single (focused) model."""
        meta = result['metadata']
        output_type = meta.get('output_type', '1D')
        npz_key = meta.get('npz_key', '')

        nn_pred = np.asarray(result['prediction'])
        truth = result.get('ground_truth')
        if truth is not None:
            truth = np.asarray(truth)

        fluent_vals = None
        if fluent_data and npz_key in fluent_data:
            fluent_vals = np.asarray(fluent_data[npz_key])

        if output_type == '1D':
            widget = self._build_scalar_result_widget(name, result, nn_pred, truth, fluent_vals)
        else:
            widget = self._build_field_result_widget(name, result, nn_pred, truth, fluent_vals)

        self._clear_results_container()
        self._results_container_layout.addWidget(widget)

    def _build_scalar_result_widget(self, name, result, nn_pred, truth, fluent_vals):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        meta = result['metadata']
        field_label = meta.get('field_name', 'value')
        nn_val = float(np.asarray(nn_pred).flatten()[0])
        truth_val = float(truth.flatten()[0]) if truth is not None else None
        fluent_val = float(fluent_vals.flatten()[0]) if fluent_vals is not None else None

        text = f"<b>NN prediction:</b> {nn_val:.6g}"
        if truth_val is not None:
            err = abs(nn_val - truth_val)
            pct = 100 * err / abs(truth_val) if truth_val != 0 else 0
            text += f"<br><b>Dataset truth:</b> {truth_val:.6g}  (abs err {err:.4g}, {pct:.2f}%)"
        if fluent_val is not None:
            err = abs(nn_val - fluent_val)
            pct = 100 * err / abs(fluent_val) if fluent_val != 0 else 0
            text += f"<br><b>Fluent:</b> {fluent_val:.6g}  (abs err {err:.4g}, {pct:.2f}%)"

        summary = QLabel(text)
        summary.setTextFormat(Qt.RichText)
        summary.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; background: transparent; padding: 8px;")
        layout.addWidget(summary)

        if HAS_MATPLOTLIB:
            self._add_plot_results_buttons(layout, [
                ("Bar Chart",
                 lambda: self._open_plot_window(
                     f"{name} — Bar",
                     lambda: _make_scalar_bar_figure(nn_val, fluent_val, truth_val, field_label),
                 )),
                ("Predicted vs Truth (Test Set)",
                 lambda: self._open_pred_vs_truth_window(name)),
            ])

        layout.addStretch()
        return container

    def _add_plot_results_buttons(self, parent_layout, buttons):
        """Add a centered, headlined row of plot-action buttons to a tab layout.

        buttons : list of (label, callback) pairs.
        """
        header = QLabel("Plot Results")
        f = header.font()
        f.setPointSize(13)
        f.setBold(True)
        header.setFont(f)
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; padding-top: 12px;")
        parent_layout.addWidget(header)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch()
        for label, cb in buttons:
            btn = QPushButton(label)
            btn.setFixedHeight(46)
            btn.setMinimumWidth(180)
            btn_font = btn.font()
            btn_font.setPointSize(11)
            btn.setFont(btn_font)
            btn.clicked.connect(cb)
            btn_row.addWidget(btn)
        btn_row.addStretch()
        parent_layout.addLayout(btn_row)

    def _build_field_result_widget(self, name, result, nn_pred, truth, fluent_vals):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        meta = result['metadata']
        output_key = meta.get('output_key', '')
        # Resolve 2D-vs-3D from the output's category in output_parameters.json
        # rather than the trained model's stored output_type — the trainer's
        # n_points-based detection misclassifies sparse cell zones as '2D'.
        # Geometry-based inference also fails (a tilted 2D surface has non-trivial
        # Z span). Category is the authoritative truth.
        is_3d = self._is_3d_from_metadata(meta)

        # Metrics summary
        info_parts = [f"<b>{meta.get('field_name', 'value')}</b> @ {meta.get('location', '?')}"]
        metrics = result.get('metrics')
        if metrics:
            info_parts.append(
                f"R² {metrics['r2']:.4f}  RMSE {metrics['rmse']:.4g}  MAE {metrics['mae']:.4g}"
            )
        if fluent_vals is not None:
            err = np.abs(nn_pred.flatten() - fluent_vals.flatten())
            info_parts.append(f"Fluent MAE {np.mean(err):.4g}  Max err {np.max(err):.4g}")

        # Prefer Fluent as truth for plots if we have both
        compare_vals = fluent_vals if fluent_vals is not None else truth
        coords = self._load_coordinates_for(meta)

        summary = QLabel("<br>".join(info_parts))
        summary.setTextFormat(Qt.RichText)
        summary.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; background: transparent; padding: 8px;")
        layout.addWidget(summary)

        if HAS_MATPLOTLIB:
            def _make_single_fig(values, title_str, cmap='viridis'):
                return _make_single_field_figure(
                    values, coords, title=title_str, cmap=cmap,
                    max_points=self._downsample_spin.value(), is_3d=is_3d,
                )

            buttons = []
            if compare_vals is not None:
                buttons.append((
                    "Triplot (NN | Truth | Error)",
                    lambda: self._open_plot_window(
                        f"{name} — Triplot",
                        lambda: _make_field_figure(
                            nn_pred, compare_vals, coords, title=output_key,
                            max_points=self._downsample_spin.value(), is_3d=is_3d,
                        ),
                    ),
                ))
            buttons.append((
                "NN",
                lambda: self._open_plot_window(
                    f"{name} — NN",
                    lambda: _make_single_fig(nn_pred, "NN Prediction"),
                ),
            ))
            if compare_vals is not None:
                buttons.append((
                    "Truth",
                    lambda: self._open_plot_window(
                        f"{name} — Truth",
                        lambda: _make_single_fig(compare_vals, "Truth"),
                    ),
                ))
                err_vals = np.abs(nn_pred.flatten() - np.asarray(compare_vals).flatten())
                buttons.append((
                    "Error",
                    lambda: self._open_plot_window(
                        f"{name} — Error",
                        lambda: _make_single_fig(err_vals, "Absolute Error", cmap='Reds'),
                    ),
                ))

            self._add_plot_results_buttons(layout, buttons)
        else:
            layout.addWidget(QLabel("matplotlib not available"))

        layout.addStretch()
        return container

    # ------------------------------------------------------------
    # Middle row: loss curve PNG + per-model test-set validation plot
    # ------------------------------------------------------------

    def _refresh_middle_section(self, model_name, meta):
        """Update the loss-curve image and the test-set validation plot for the
        focused model. Both fail gracefully — placeholder text on error."""
        self._update_loss_png(model_name, meta)
        self._update_test_set_plot(model_name, meta)

    def _update_loss_png(self, model_name, meta):
        from PySide6.QtGui import QPixmap
        model_dir = self.project.models_dir / model_name
        # Trainer saves loss curves as <output_model_name>_loss_curve.png.
        # output_model_name lives in metadata; fall back to a glob if missing.
        candidate = model_dir / f"{meta.get('model_name', '')}_loss_curve.png"
        if not candidate.exists():
            pngs = list(model_dir.glob("*_loss_curve.png"))
            candidate = pngs[0] if pngs else None
        if candidate is None or not candidate.exists():
            self._loss_png_label.setPixmap(QPixmap())
            self._loss_png_label.setText("No loss curve PNG found.")
            self._loss_png_path = None
            self._loss_png_label.setCursor(Qt.ArrowCursor)
            self._loss_png_label.mousePressEvent = lambda _ev: None
            return
        pix = QPixmap(str(candidate))
        if pix.isNull():
            self._loss_png_label.setText(f"Couldn't load image: {candidate.name}")
            self._loss_png_path = None
            self._loss_png_label.setCursor(Qt.ArrowCursor)
            self._loss_png_label.mousePressEvent = lambda _ev: None
            return
        # Scale to fit the label width; preserve aspect.
        max_w = max(self._loss_png_label.width() - 8, 200)
        max_h = max(self._loss_png_label.height() - 8, 150)
        self._loss_png_label.setPixmap(pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._loss_png_label.setToolTip("Click to expand")
        self._loss_png_label.setCursor(Qt.PointingHandCursor)
        self._loss_png_path = str(candidate)
        self._loss_png_label.mousePressEvent = (
            lambda _ev, p=str(candidate), n=model_name: self._open_loss_png_window(p, n)
        )

    def _open_loss_png_window(self, png_path, model_name):
        """Pop the loss curve PNG in a resizable window at full resolution."""
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QDialog, QVBoxLayout as _VLayout
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{model_name} — Training Loss")
        dlg.resize(900, 600)
        v = _VLayout(dlg)
        v.setContentsMargins(8, 8, 8, 8)

        pix = QPixmap(png_path)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"background: {theme.BG_DARK};")
        label.setPixmap(pix.scaled(880, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        v.addWidget(label, 1)

        dlg.setModal(False)
        dlg.show()
        if not hasattr(self, '_plot_windows'):
            self._plot_windows = []
        self._plot_windows.append(dlg)
        dlg.finished.connect(lambda _: self._plot_windows.remove(dlg) if dlg in self._plot_windows else None)

    def _update_test_set_plot(self, model_name, meta):
        # Clear existing canvas/widget
        while self._pvt_layout.count():
            item = self._pvt_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not HAS_MATPLOTLIB:
            placeholder = QLabel("matplotlib not available")
            placeholder.setAlignment(Qt.AlignCenter)
            self._pvt_layout.addWidget(placeholder)
            return

        try:
            from cfdtwin.visualization import predict_test_set
            model_dir = self.project.models_dir / model_name
            summary_file = model_dir / "training_summary.json"
            result = predict_test_set(
                model_dir,
                self.project.dataset_dir,
                self.project.doe_samples_file,
                summary_file,
            )
        except Exception as e:
            placeholder = QLabel(f"Test-set validation unavailable:\n{e}")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; padding: 12px;")
            placeholder.setWordWrap(True)
            self._pvt_layout.addWidget(placeholder)
            return

        truths = np.asarray(result['truths'])
        preds = np.asarray(result['predictions'])
        sim_ids = result['sim_ids']

        is_3d = self._is_3d_from_metadata(meta)
        is_field = (meta.get('output_type') in ('2D', '3D')) or (truths.shape[1] > 1) or is_3d

        if is_field:
            # Per-sample R² across the test set (option (i)).
            r2_values = []
            for i in range(truths.shape[0]):
                t = truths[i].flatten()
                p = preds[i].flatten()
                ss_res = float(np.sum((t - p) ** 2))
                ss_tot = float(np.sum((t - t.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
                r2_values.append(r2)
            self._pvt_caption.setText("Per-Sample R² (Test Set) — click to expand")
            popup_title = f"{model_name} — Per-Sample R² (Test Set)"
            def _builder(_sim_ids=list(sim_ids), _r2=list(r2_values), _name=model_name):
                return _make_per_sample_r2_figure(_sim_ids, _r2, title=_name)
        else:
            self._pvt_caption.setText("Predicted vs Ground Truth (Test Set) — click to expand")
            popup_title = f"{model_name} — Predicted vs Truth (Test Set)"
            _truths_flat = truths.flatten()
            _preds_flat = preds.flatten()
            def _builder(_t=_truths_flat, _p=_preds_flat, _name=model_name):
                return _make_pred_vs_truth_figure(_t, _p, title=_name)

        fig = _builder()
        canvas = FigureCanvasQTAgg(fig)
        canvas.setCursor(Qt.PointingHandCursor)
        # Click on the canvas opens the same plot in a resizable popout.
        canvas.mousePressEvent = lambda _ev, t=popup_title, b=_builder: self._open_plot_window(t, b)
        self._pvt_layout.addWidget(canvas)

    def _is_3d_from_metadata(self, meta):
        """Decide if a model's field is 3D using output_parameters.json's category
        (Cell Zone = 3D, Surface = 2D), falling back to the model's stored
        output_type if the category lookup fails."""
        location = meta.get('location')
        if location and self.project.output_parameters_file.exists():
            try:
                with open(self.project.output_parameters_file, 'r') as f:
                    out_data = json.load(f)
                for out in out_data.get('outputs', []):
                    if out.get('name') == location:
                        category = out.get('category', '')
                        if category == 'Cell Zone':
                            return True
                        if category in ('Surface', 'Report Definition'):
                            return False
            except Exception:
                pass
        return meta.get('output_type') == '3D'

    def _open_pred_vs_truth_window(self, model_name):
        """Build and pop out a Predicted-vs-Truth scatter for a 1D scalar model
        across the test set saved in training_summary.json."""
        from cfdtwin.visualization import predict_test_set
        model_dir = self.project.models_dir / model_name
        summary_file = model_dir / "training_summary.json"

        def _build():
            result = predict_test_set(
                model_dir,
                self.project.dataset_dir,
                self.project.doe_samples_file,
                summary_file,
            )
            if result is None:
                raise RuntimeError("predict_test_set returned None")
            return _make_pred_vs_truth_figure(
                result['truths'], result['predictions'],
                title=f"{model_name} — Test Set",
            )

        self._open_plot_window(f"{model_name} — Predicted vs Truth", _build)

    def _open_plot_window(self, title, fig_builder):
        """Open a resizable dialog containing a fresh canvas with the same plot."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout as _VLayout

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(1100, 700)

        layout = _VLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)

        try:
            fig = fig_builder()
        except Exception as e:
            layout.addWidget(QLabel(f"Failed to build plot: {e}"))
            dlg.exec()
            return

        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, dlg)
        toolbar.setStyleSheet(f"background: {theme.BG_PANEL}; color: {theme.TEXT_PRIMARY};")
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)

        # Modeless so you can keep interacting with the main window
        dlg.setModal(False)
        dlg.show()
        # Hold a reference to prevent garbage collection
        if not hasattr(self, '_plot_windows'):
            self._plot_windows = []
        self._plot_windows.append(dlg)
        dlg.finished.connect(lambda _: self._plot_windows.remove(dlg) if dlg in self._plot_windows else None)

    def _load_coordinates_for(self, meta):
        """Load coordinate array for a field output from coordinates.npz."""
        coords_file = self.project.dataset_dir / "coordinates.npz"
        if not coords_file.exists():
            return None
        try:
            data = np.load(coords_file, allow_pickle=True)
            location = meta.get('location', '')
            coord_key = f"{location}|coordinates"
            if coord_key in data.files:
                return data[coord_key]
        except Exception as e:
            logger.warning(f"Could not load coordinates: {e}")
        return None

    # ------------------------------------------------------------
    # Fluent comparison
    # ------------------------------------------------------------

    def _run_fluent_comparison(self):
        fm = FluentManager.instance()
        if not fm.is_available():
            QMessageBox.warning(self, "Not Connected",
                                "Fluent is not connected. Go to Setup and launch Fluent.")
            return

        params, sim_id = self._get_current_params()
        if params is None:
            QMessageBox.warning(self, "No Params", "Select a dataset point or fill custom parameters.")
            return

        # Cache hit?
        cached = self.fluent_cache.lookup(params)
        if cached is not None:
            self._last_fluent_data = cached
            logger.info("Using cached Fluent run for comparison")
            QMessageBox.information(self, "Cached",
                                    "Found a cached Fluent run for these parameters. "
                                    "Click Predict to see the comparison.")
            return

        # Read model_setup
        try:
            with open(self.project.model_setup_file, 'r') as f:
                setup_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot read model_setup.json: {e}")
            return

        fm.set_busy()
        self._compare_btn.setEnabled(False)
        self._compare_btn.setText("Running Fluent...")

        self._pending_params = dict(params)
        self._worker = ValidationWorker(
            solver=fm.solver,
            setup_data=setup_data,
            dataset_dir=self.project.dataset_dir,
            parameters=params,
        )
        self._worker.finished.connect(self._on_comparison_done)
        self._worker.error.connect(self._on_comparison_error)
        self._worker.start()

    def _on_comparison_done(self, results):
        fm = FluentManager.instance()
        fm.set_idle()
        self._compare_btn.setEnabled(True)
        self._compare_btn.setText("Run Fluent Comparison")

        if results:
            self.fluent_cache.store(self._pending_params, results)
            self._last_fluent_data = results
            self._update_cache_label()
            logger.info("Fluent comparison complete and cached")
            QMessageBox.information(self, "Done",
                                    "Fluent comparison complete. Click Predict to view results.")
        self._worker = None

    def _on_comparison_error(self, msg):
        fm = FluentManager.instance()
        fm.set_idle()
        self._compare_btn.setEnabled(True)
        self._compare_btn.setText("Run Fluent Comparison")
        self._worker = None
        QMessageBox.critical(self, "Validation Error", msg)

    def _clear_cache(self):
        reply = QMessageBox.question(
            self, "Clear Cache",
            f"Delete {self.fluent_cache.count()} cached Fluent run(s)?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.fluent_cache.clear()
        self._last_fluent_data = None
        self._update_cache_label()

    # ------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------

    def _export_model(self):
        """Copy the focused model's directory to a user-chosen location."""
        import shutil
        from PySide6.QtWidgets import QFileDialog
        current = self._model_list.currentItem()
        if not current:
            QMessageBox.warning(self, "No Model", "Select a model to export.")
            return
        name = current.data(Qt.UserRole)
        src_dir = self.project.models_dir / name
        if not src_dir.exists():
            QMessageBox.critical(self, "Export Failed", f"Source folder is missing:\n{src_dir}")
            return

        dest_parent = QFileDialog.getExistingDirectory(
            self, "Choose where to save the model folder",
        )
        if not dest_parent:
            return

        dest = Path(dest_parent) / name
        if dest.exists():
            reply = QMessageBox.question(
                self, "Overwrite?",
                f"{dest} already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            try:
                shutil.rmtree(dest)
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Couldn't remove existing folder:\n{e}")
                return

        try:
            shutil.copytree(src_dir, dest)
            logger.info(f"Exported model {name} to {dest}")
            QMessageBox.information(self, "Export Complete", f"Model exported to:\n{dest}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Couldn't copy model:\n{e}")

    def _delete_model(self):
        current = self._model_list.currentItem()
        if not current:
            return
        name = current.data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Delete model '{name}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if self.project.delete_model(name):
            self.refresh()
            self._dashboard.hide()
            self._model_list.setMaximumHeight(16777215)
