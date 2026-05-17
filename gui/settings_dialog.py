"""
Settings Dialog Module
======================
Tabbed dialog: 1D Model, 2D Model, 3D Model.
Model tabs have preset/custom toggle. Settings persist via UserSettings.
"""

import logging
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QFormLayout, QFrame, QLineEdit, QGroupBox,
)
from PySide6.QtCore import Qt

from cfdtwin.nn import PRESETS

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Tabbed settings dialog (Fluent, 1D, 2D, 3D model tabs)."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings

        self.setWindowTitle("Model Settings")
        self.setMinimumSize(520, 480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._build_ui()
        self._load_from_settings()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self._tabs = QTabWidget()
        self._1d_tab = self._build_model_tab('1d')
        self._2d_tab = self._build_model_tab('2d', pod=True)
        self._3d_tab = self._build_model_tab('3d', pod=True)

        self._tabs.addTab(self._1d_tab, "1D Model")
        self._tabs.addTab(self._2d_tab, "2D Model")
        self._tabs.addTab(self._3d_tab, "3D Model")

        layout.addWidget(self._tabs)

        # OK / Cancel
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setProperty("flat", True)
        ok_btn.clicked.connect(self._save_and_accept)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    # --- Model tab (reused for 1d/2d/3d) ---

    def _build_model_tab(self, preset_key, pod=False):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)

        # Mode toggle
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        mode_combo = QComboBox()
        mode_combo.addItems(["Preset", "Custom"])
        mode_row.addWidget(mode_combo)
        mode_row.addStretch()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setProperty("flat", True)
        mode_row.addWidget(reset_btn)
        layout.addLayout(mode_row)

        # Hyperparams form
        form = QFormLayout()
        form.setSpacing(10)

        hidden_edit = QLineEdit()
        hidden_edit.setPlaceholderText("e.g. 128, 128, 64, 32")
        form.addRow("Hidden layers:", hidden_edit)

        dropout_edit = QLineEdit()
        dropout_edit.setPlaceholderText("e.g. 0.1, 0.1, 0.05, 0")
        form.addRow("Dropout:", dropout_edit)

        # Real-time mismatch warning: hidden_layers and dropout must have the
        # same length. SurrogateNN tolerates a shorter dropout list (extras
        # ignored), but a mismatch is almost always a typo, so warn loudly.
        mismatch_label = QLabel("")
        mismatch_label.setStyleSheet("color: #C62828; font-weight: bold;")
        mismatch_label.setWordWrap(True)
        form.addRow("", mismatch_label)

        l2_spin = QDoubleSpinBox()
        l2_spin.setDecimals(6)
        l2_spin.setRange(0, 1)
        l2_spin.setSingleStep(0.0001)
        l2_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        form.addRow("L2 regularization:", l2_spin)

        lr_spin = QDoubleSpinBox()
        lr_spin.setDecimals(6)
        lr_spin.setRange(0.000001, 1)
        lr_spin.setSingleStep(0.0001)
        lr_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        form.addRow("Learning rate:", lr_spin)

        batch_spin = QSpinBox()
        batch_spin.setRange(1, 1024)
        batch_spin.setSpecialValueText("adaptive")
        batch_spin.setButtonSymbols(QSpinBox.NoButtons)
        form.addRow("Batch size:", batch_spin)

        es_patience_spin = QSpinBox()
        es_patience_spin.setRange(1, 1000)
        es_patience_spin.setButtonSymbols(QSpinBox.NoButtons)
        form.addRow("Early stopping patience:", es_patience_spin)

        lr_patience_spin = QSpinBox()
        lr_patience_spin.setRange(1, 1000)
        lr_patience_spin.setButtonSymbols(QSpinBox.NoButtons)
        form.addRow("LR reduce patience:", lr_patience_spin)

        pod_spin = None
        if pod:
            pod_spin = QSpinBox()
            pod_spin.setRange(1, 200)
            pod_spin.setValue(20)
            pod_spin.setButtonSymbols(QSpinBox.NoButtons)
            form.addRow("POD mode count:", pod_spin)

        # Training-loop settings (always editable, NOT tied to Preset/Custom mode).
        # These are training controls rather than architecture choices.
        training_label = QLabel("Training")
        f = training_label.font()
        f.setBold(True)
        training_label.setFont(f)
        layout.addLayout(form)
        layout.addSpacing(6)
        layout.addWidget(training_label)

        train_form = QFormLayout()
        train_form.setSpacing(10)

        epochs_spin = QSpinBox()
        epochs_spin.setRange(10, 10000)
        epochs_spin.setValue(500)
        epochs_spin.setButtonSymbols(QSpinBox.NoButtons)
        train_form.addRow("Max epochs:", epochs_spin)

        test_split_spin = QDoubleSpinBox()
        test_split_spin.setDecimals(2)
        test_split_spin.setRange(0.05, 0.5)
        test_split_spin.setSingleStep(0.05)
        test_split_spin.setValue(0.2)
        test_split_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        train_form.addRow("Test split:", test_split_spin)

        # 'adaptive' for val_split shows when the spinbox value == its minimum,
        # same pattern as batch_size above.
        val_split_spin = QDoubleSpinBox()
        val_split_spin.setDecimals(2)
        val_split_spin.setRange(0.0, 0.5)
        val_split_spin.setSingleStep(0.05)
        val_split_spin.setSpecialValueText("adaptive")
        val_split_spin.setValue(val_split_spin.minimum())  # default = adaptive
        val_split_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        train_form.addRow("Validation split:", val_split_spin)

        layout.addLayout(train_form)
        layout.addStretch()

        # Store widget references
        widgets = {
            'mode': mode_combo,
            'hidden': hidden_edit,
            'dropout': dropout_edit,
            'mismatch_label': mismatch_label,
            'l2': l2_spin,
            'lr': lr_spin,
            'batch': batch_spin,
            'es_patience': es_patience_spin,
            'lr_patience': lr_patience_spin,
            'pod': pod_spin,
            'epochs': epochs_spin,
            'test_split': test_split_spin,
            'val_split': val_split_spin,
        }

        # Recompute the warning whenever either field or the mode changes.
        def update_mismatch_warning():
            if mode_combo.currentIndex() != 1:  # only in Custom mode
                mismatch_label.setText("")
                return
            try:
                h = [int(x.strip()) for x in hidden_edit.text().split(',') if x.strip()]
            except ValueError:
                mismatch_label.setText("Hidden layers: each entry must be an integer.")
                return
            try:
                d = [float(x.strip()) for x in dropout_edit.text().split(',') if x.strip()]
            except ValueError:
                mismatch_label.setText("Dropout: each entry must be a number between 0 and 1.")
                return
            if len(h) != len(d):
                mismatch_label.setText(
                    f"Mismatch: {len(h)} hidden layer(s) but {len(d)} dropout value(s). "
                    "These must match before saving."
                )
            else:
                mismatch_label.setText("")

        hidden_edit.textChanged.connect(update_mismatch_warning)
        dropout_edit.textChanged.connect(update_mismatch_warning)
        mode_combo.currentIndexChanged.connect(lambda _i: update_mismatch_warning())

        # NN architecture widgets are gated by Preset/Custom mode.
        # Training-loop widgets stay editable in both modes.
        param_widgets = [hidden_edit, dropout_edit, l2_spin, lr_spin,
                         batch_spin, es_patience_spin, lr_patience_spin]
        if pod_spin:
            param_widgets.append(pod_spin)

        # Toggle preset/custom
        def on_mode_changed(idx):
            enabled = (idx == 1)  # Custom
            for w in param_widgets:
                w.setEnabled(enabled)

        mode_combo.currentIndexChanged.connect(on_mode_changed)
        # Apply initial state (preset = grayed out)
        on_mode_changed(0)

        # Reset
        def on_reset():
            self._populate_model_widgets(widgets, PRESETS[preset_key], preset_key)
            mode_combo.setCurrentIndex(0)

        reset_btn.clicked.connect(on_reset)

        # Tag for retrieval
        tab.widgets = widgets
        tab.preset_key = preset_key

        return tab

    # --- Load / Save ---

    def _load_from_settings(self):
        # Model tabs
        nn_settings = self.settings.get_nn_settings()
        for tab in [self._1d_tab, self._2d_tab, self._3d_tab]:
            key = tab.preset_key
            saved = nn_settings.get(key, {})
            preset_defaults = PRESETS[key]

            if saved.get('mode', 'preset') == 'custom':
                tab.widgets['mode'].setCurrentIndex(1)
                self._populate_model_widgets(tab.widgets, saved, key)
            else:
                tab.widgets['mode'].setCurrentIndex(0)
                self._populate_model_widgets(tab.widgets, preset_defaults, key)

    def _populate_model_widgets(self, widgets, cfg, preset_key):
        """Fill model tab widgets from a config dict."""
        hl = cfg.get('hidden_layers', PRESETS[preset_key]['hidden_layers'])
        widgets['hidden'].setText(', '.join(str(x) for x in hl))

        dr = cfg.get('dropout', PRESETS[preset_key]['dropout'])
        widgets['dropout'].setText(', '.join(str(x) for x in dr))

        widgets['l2'].setValue(cfg.get('l2', PRESETS[preset_key]['l2']))
        widgets['lr'].setValue(cfg.get('learning_rate', PRESETS[preset_key]['learning_rate']))

        bs = cfg.get('batch_size', PRESETS[preset_key]['batch_size'])
        if bs == 'adaptive':
            widgets['batch'].setValue(widgets['batch'].minimum())  # triggers specialValueText
        else:
            widgets['batch'].setValue(bs)

        widgets['es_patience'].setValue(cfg.get('es_patience', PRESETS[preset_key]['es_patience']))
        widgets['lr_patience'].setValue(cfg.get('lr_patience', PRESETS[preset_key]['lr_patience']))

        if widgets['pod'] is not None:
            widgets['pod'].setValue(cfg.get('pod_modes', 20))

        # Training-loop settings — also persisted in nn_settings[preset_key].
        widgets['epochs'].setValue(int(cfg.get('epochs', 500)))
        widgets['test_split'].setValue(float(cfg.get('test_size', 0.2)))
        vs = cfg.get('val_split', 'adaptive')
        if vs == 'adaptive' or vs is None:
            widgets['val_split'].setValue(widgets['val_split'].minimum())
        else:
            widgets['val_split'].setValue(float(vs))

    def _read_model_widgets(self, tab):
        """Read current values from a model tab into a dict."""
        w = tab.widgets
        is_custom = w['mode'].currentIndex() == 1

        cfg = {'mode': 'custom' if is_custom else 'preset'}

        if is_custom:
            # Parse hidden layers
            try:
                cfg['hidden_layers'] = [int(x.strip()) for x in w['hidden'].text().split(',') if x.strip()]
            except ValueError:
                cfg['hidden_layers'] = PRESETS[tab.preset_key]['hidden_layers']

            # Parse dropout
            try:
                cfg['dropout'] = [float(x.strip()) for x in w['dropout'].text().split(',') if x.strip()]
            except ValueError:
                cfg['dropout'] = PRESETS[tab.preset_key]['dropout']

            cfg['l2'] = w['l2'].value()
            cfg['learning_rate'] = w['lr'].value()

            bs_val = w['batch'].value()
            cfg['batch_size'] = 'adaptive' if bs_val == w['batch'].minimum() else bs_val

            cfg['es_patience'] = w['es_patience'].value()
            cfg['lr_patience'] = w['lr_patience'].value()

        if w['pod'] is not None:
            cfg['pod_modes'] = w['pod'].value()

        # Training-loop settings (always saved regardless of preset/custom mode).
        cfg['epochs'] = w['epochs'].value()
        cfg['test_size'] = w['test_split'].value()
        vs_val = w['val_split'].value()
        cfg['val_split'] = 'adaptive' if vs_val == w['val_split'].minimum() else vs_val

        return cfg

    def _save_and_accept(self):
        # Block save if any Custom-mode tab has hidden_layers / dropout
        # mismatch — same warning the inline label shows, hoisted up so the
        # user can't accidentally bypass it via OK.
        for tab in (self._1d_tab, self._2d_tab, self._3d_tab):
            msg = tab.widgets['mismatch_label'].text()
            if msg:
                from PySide6.QtWidgets import QMessageBox
                self._tabs.setCurrentWidget(tab)
                QMessageBox.warning(self, "Mismatch in Custom config", msg)
                return

        nn = {
            '1d': self._read_model_widgets(self._1d_tab),
            '2d': self._read_model_widgets(self._2d_tab),
            '3d': self._read_model_widgets(self._3d_tab),
        }
        self.settings.save_nn_settings(nn)

        logger.info("Settings saved")
        self.accept()
