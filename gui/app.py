"""
Application Entry Point
=======================
Launches the PySide6 application, shows project dialog, opens main window.
Wires up all pages, settings dialog, fluent manager, step unlocking.
"""

import faulthandler
import logging
import logging.handlers
import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .theme import get_stylesheet
from .main_window import MainWindow
from .project_dialog import ProjectDialog
from .settings_dialog import SettingsDialog
from .fluent_manager import FluentManager
from .pages.setup_page import SetupPage
from .pages.doe_page import DOEPage
from .pages.simulate_page import SimulatePage
from .pages.train_page import TrainPage
from .pages.validate_page import ValidatePage
from .user_settings import UserSettings

logger = logging.getLogger(__name__)

# Runtime files live inside the gui/ folder (portable, easy to find)
_SETTINGS_FILE = Path(__file__).resolve().parent / "user_settings.json"
_LOG_FILE = Path(__file__).resolve().parent / "cfdtwin.log"
_FAULT_FILE = Path(__file__).resolve().parent / "cfdtwin_fault.log"


def _install_crash_handlers():
    """Capture three classes of crash info:
      1. faulthandler — segfaults / interpreter crashes (Python, gRPC, TF, etc.)
      2. sys.excepthook — unhandled Python exceptions (full traceback)
      3. RotatingFileHandler — durable log file the user can grep after the fact
    """
    # Rotating file log (5 MB x 3 backups) so the log doesn't grow unbounded.
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])

    # faulthandler dumps native tracebacks (signal-safe) to a dedicated file —
    # written even when the interpreter is in a bad state and can't run pure Python.
    fault_fp = open(_FAULT_FILE, "a", encoding="utf-8", buffering=1)
    fault_fp.write(f"\n=== faulthandler enabled at startup ===\n")
    faulthandler.enable(file=fault_fp)

    # Catch unhandled Python exceptions and log them with traceback before exit.
    def _on_unhandled(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical(f"UNHANDLED EXCEPTION:\n{tb_str}")
    sys.excepthook = _on_unhandled


def run():
    """Launch the application."""
    _install_crash_handlers()
    logger.info(f"CFDTwin starting. Log file: {_LOG_FILE}")

    app = QApplication(sys.argv)
    app.setStyleSheet(get_stylesheet())

    settings = UserSettings(_SETTINGS_FILE)

    # Show project dialog -- app exits if user closes without selecting
    dialog = ProjectDialog(settings)
    if dialog.exec() != ProjectDialog.Accepted:
        sys.exit(0)

    project = dialog.project

    # Launch main window
    window = MainWindow(settings)
    window.set_project(project)

    # --- Fluent manager -> header status ---
    fm = FluentManager.instance()
    fm.status_changed.connect(window.get_fluent_status_widget().set_status)

    # --- Pages ---
    setup_page = SetupPage(project, settings)
    doe_page = DOEPage(project)
    sim_page = SimulatePage(project)
    train_page = TrainPage(project, settings)
    validate_page = ValidatePage(project, settings)

    # Add pages in order (0=Setup, 1=DOE, 2=Simulate, 3=Train, 4=Validate)
    window.add_page(setup_page)
    window.add_page(doe_page)
    window.add_page(sim_page)
    window.add_page(train_page)
    window.add_page(validate_page)

    # --- Step unlocking signals ---
    def refresh_steps():
        window._refresh_unlocked_steps()

    setup_page.setup_complete.connect(refresh_steps)
    doe_page.doe_changed.connect(refresh_steps)
    sim_page.simulations_changed.connect(refresh_steps)
    training_complete_connected = train_page.training_complete.connect(refresh_steps)

    # Refresh data-dependent pages when switching to them
    def on_step_changed(idx):
        if idx == 1:  # DOE
            doe_page._load_state()
        elif idx == 2:  # Simulate
            sim_page._refresh_status()
        elif idx == 3:  # Train
            train_page.refresh()
        elif idx == 4:  # Validate
            validate_page.refresh()

    window.step_changed.connect(on_step_changed)

    # --- Settings dialog ---
    def open_settings():
        dlg = SettingsDialog(settings, window)
        dlg.exec()

    window.get_settings_button().clicked.connect(open_settings)

    # --- Cleanup on exit ---
    def on_close():
        fm.disconnect()

    app.aboutToQuit.connect(on_close)

    window.show()
    sys.exit(app.exec())
