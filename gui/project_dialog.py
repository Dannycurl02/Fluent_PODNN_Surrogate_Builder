"""
Project Dialog Module
=====================
Modal dialog shown on launch. Supports Create New, Open Existing,
and Recent Projects. Returns a WorkflowProject or None.
"""

import logging
import os
import shutil
import stat
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QListWidget, QListWidgetItem,
    QFrame, QMessageBox, QWidget,
)
from PySide6.QtCore import Qt, QSize

from . import theme
from cfdtwin._project_system import WorkflowProject, create_project, open_project

logger = logging.getLogger(__name__)


class ProjectDialog(QDialog):
    """
    Project selection dialog shown on app launch.

    Returns a WorkflowProject via self.project after accept(),
    or None if the dialog is rejected.
    """

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.project = None

        self.setWindowTitle("Select Project")
        # 420px was tight under the old text title; the wordmark + the
        # toggle-able Create panel need ~520px to all fit without crushing
        # the recent-projects list. setMinimumSize (not setFixedSize) so
        # high-DPI scaling can grow it further if needed.
        self.setMinimumSize(520, 520)
        self.resize(520, 520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._build_ui()
        self._populate_recent()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Wordmark — replaces the old bold-text title.
        from PySide6.QtGui import QPixmap
        from cfdtwin import __version__ as _cfdtwin_version
        wordmark_path = Path(__file__).resolve().parent / "assets" / "logo_wordmark.png"
        title = QLabel()
        title.setAlignment(Qt.AlignCenter)
        if wordmark_path.exists():
            pix = QPixmap(str(wordmark_path))
            # Scale to a reasonable header height while preserving aspect.
            title.setPixmap(pix.scaledToHeight(56, Qt.SmoothTransformation))
        else:
            # Fall back to text if the asset is missing.
            title.setText("CFDTwin")
            font = title.font()
            font.setPointSize(16)
            font.setBold(True)
            title.setFont(font)
        layout.addWidget(title)

        # Subtle version caption under the wordmark.
        version_label = QLabel(f"v{_cfdtwin_version}")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setProperty("secondary", True)
        layout.addWidget(version_label)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._new_btn = QPushButton("Create New")
        self._open_btn = QPushButton("Open Existing")
        self._new_btn.setFixedHeight(38)
        self._open_btn.setFixedHeight(38)

        btn_row.addWidget(self._new_btn)
        btn_row.addWidget(self._open_btn)
        layout.addLayout(btn_row)

        # --- Create new panel (hidden by default) ---
        self._create_panel = QFrame()
        self._create_panel.setProperty("panel", True)
        cp_layout = QVBoxLayout(self._create_panel)
        cp_layout.setContentsMargins(12, 12, 12, 12)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Project name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("My Surrogate Study")
        name_row.addWidget(self._name_edit)
        cp_layout.addLayout(name_row)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Location:"))
        self._folder_edit = QLineEdit()
        self._folder_edit.setReadOnly(True)
        self._folder_edit.setPlaceholderText("Choose folder...")
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.setProperty("flat", True)
        folder_row.addWidget(self._folder_edit, 1)
        folder_row.addWidget(self._browse_btn)
        cp_layout.addLayout(folder_row)

        self._create_confirm_btn = QPushButton("Create")
        self._create_confirm_btn.setEnabled(False)
        cp_layout.addWidget(self._create_confirm_btn)

        self._create_panel.hide()
        layout.addWidget(self._create_panel)

        # --- Recent projects ---
        recent_label = QLabel("Recent Projects")
        recent_label.setProperty("secondary", True)
        layout.addWidget(recent_label)

        self._recent_list = QListWidget()
        self._recent_list.setSpacing(0)
        self._recent_list.setStyleSheet(f"""
            QListWidget {{
                font-size: 14px;
            }}
            QListWidget::item {{
                color: {theme.TEXT_PRIMARY};
                padding: 10px 12px;
                border-bottom: 1px solid {theme.BORDER};
            }}
            QListWidget::item:hover {{
                background-color: {theme.BG_INPUT};
            }}
            QListWidget::item:selected {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {theme.ORANGE_DARK}, stop:0.6 transparent);
            }}
        """)
        layout.addWidget(self._recent_list, 1)

        # Load / Delete row under the recent list
        recent_btn_row = QHBoxLayout()
        self._load_recent_btn = QPushButton("Load")
        self._load_recent_btn.setEnabled(False)
        self._delete_recent_btn = QPushButton("Delete")
        self._delete_recent_btn.setEnabled(False)
        recent_btn_row.addWidget(self._load_recent_btn)
        recent_btn_row.addWidget(self._delete_recent_btn)
        recent_btn_row.addStretch()
        layout.addLayout(recent_btn_row)

        # --- Connections ---
        self._new_btn.clicked.connect(self._toggle_create_panel)
        self._open_btn.clicked.connect(self._open_existing)
        self._browse_btn.clicked.connect(self._browse_folder)
        self._create_confirm_btn.clicked.connect(self._create_new)
        self._name_edit.textChanged.connect(self._validate_create)
        self._recent_list.itemDoubleClicked.connect(self._open_recent)
        self._recent_list.itemSelectionChanged.connect(self._update_recent_buttons)
        self._load_recent_btn.clicked.connect(self._load_selected_recent)
        self._delete_recent_btn.clicked.connect(self._delete_selected_recent)

    # --- Recent projects ---

    def _populate_recent(self):
        self._recent_list.clear()
        recent = self.settings.get_recent_project_folders()
        if not recent:
            item = QListWidgetItem("No recent projects")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self._recent_list.addItem(item)
            return

        for path in recent:
            # Try to read project name from project_info.json
            info_file = Path(path) / "project_info.json"
            display = Path(path).name
            if info_file.exists():
                try:
                    import json
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    display = info.get('project_name', display)
                except Exception:
                    pass

            item = QListWidgetItem(f"{display}\n{path}")
            item.setData(Qt.UserRole, path)
            item.setSizeHint(QSize(0, 54))
            self._recent_list.addItem(item)

    # --- Create new ---

    def _toggle_create_panel(self):
        visible = self._create_panel.isVisible()
        self._create_panel.setVisible(not visible)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose Project Location")
        if folder:
            self._folder_edit.setText(folder)
            self._validate_create()

    def _validate_create(self):
        name_ok = len(self._name_edit.text().strip()) > 0
        folder_ok = len(self._folder_edit.text().strip()) > 0
        self._create_confirm_btn.setEnabled(name_ok and folder_ok)

    def _create_new(self):
        name = self._name_edit.text().strip()
        folder = Path(self._folder_edit.text().strip()) / name

        if folder.exists() and any(folder.iterdir()):
            reply = QMessageBox.question(
                self, "Folder Exists",
                f"{folder} already exists and is not empty. Use it anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        project = create_project(folder, name)
        if project is None:
            QMessageBox.critical(self, "Error", "Failed to create project.")
            return

        self.settings.add_recent_project_folder(str(folder))
        self.project = project
        logger.info(f"Created project: {name}")
        self.accept()

    # --- Open existing ---

    def _open_existing(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Project Folder")
        if not folder:
            return
        self._try_open(folder)

    def _open_recent(self, item):
        path = item.data(Qt.UserRole)
        if path:
            self._try_open(path)

    def _try_open(self, folder):
        project = open_project(folder)
        if project is None:
            QMessageBox.critical(
                self, "Error",
                f"Could not open project at:\n{folder}\n\nNo project_info.json found.",
            )
            return

        self.settings.add_recent_project_folder(str(folder))
        self.project = project
        logger.info(f"Opened project: {project.info.get('project_name', 'Unknown')}")
        self.accept()

    # --- Recent list buttons ---

    def _selected_recent_path(self):
        """Return the path of the currently selected recent project, or None."""
        items = self._recent_list.selectedItems()
        if not items:
            return None
        path = items[0].data(Qt.UserRole)
        return path  # None for the "No recent projects" placeholder

    def _update_recent_buttons(self):
        has_path = self._selected_recent_path() is not None
        self._load_recent_btn.setEnabled(has_path)
        self._delete_recent_btn.setEnabled(has_path)

    def _load_selected_recent(self):
        path = self._selected_recent_path()
        if path:
            self._try_open(path)

    def _delete_selected_recent(self):
        path = self._selected_recent_path()
        if not path:
            return
        dlg = DeleteCountdownDialog(path, self)
        if dlg.exec() != QDialog.Accepted:
            return
        # Confirmed. Drop from recent list and delete folder from disk.
        self.settings.remove_recent_project_folder(path)
        try:
            shutil.rmtree(path, onerror=_force_remove)
            logger.info(f"Deleted project folder: {path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Delete Failed",
                f"Couldn't delete folder:\n{e}\n\n"
                f"If files are open in another app (e.g. Fluent), close them and try again.",
            )
        self._populate_recent()
        self._update_recent_buttons()


def _force_remove(func, path, exc_info):
    """rmtree onerror handler: clear read-only bit and retry."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


class DeleteCountdownDialog(QDialog):
    """Three-stage Yes-confirmation. Each Yes click advances to the next, harder
    warning. Final Yes accepts."""

    STAGES = [
        ("Delete Project",
         "This will permanently delete the project folder and all of its contents.\n\nAre you sure?"),
        ("Last chance.",
         "Once deleted, simulation results, trained models, and configs cannot be recovered.\n\nReally delete?"),
    ]

    def __init__(self, project_path, parent=None):
        super().__init__(parent)
        self._stage = 0

        self.setWindowTitle("Delete Project")
        self.setFixedWidth(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        self._heading = QLabel()
        f = self._heading.font()
        f.setBold(True)
        self._heading.setFont(f)
        layout.addWidget(self._heading)

        self._path_label = QLabel(str(project_path))
        self._path_label.setProperty("secondary", True)
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)

        self._body = QLabel()
        self._body.setWordWrap(True)
        layout.addWidget(self._body)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._yes_btn = QPushButton("Yes")
        self._cancel_btn.clicked.connect(self.reject)
        self._yes_btn.clicked.connect(self._on_yes)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._yes_btn)
        layout.addLayout(btn_row)

        self._enter_stage(0)

    def _enter_stage(self, stage):
        self._stage = stage
        title, body = self.STAGES[stage]
        self._heading.setText(title)
        self._body.setText(body)

    def _on_yes(self):
        if self._stage + 1 < len(self.STAGES):
            self._enter_stage(self._stage + 1)
        else:
            self.accept()
