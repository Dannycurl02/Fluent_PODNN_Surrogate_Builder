"""Thin entry shim for the ``cfdtwin-gui`` script.

Why a separate module: ``gui.app`` imports PySide6 at module top, so a user
who ran ``pip install cfdtwin`` (API-only, no Qt) and then typed
``cfdtwin-gui`` would get an opaque ``ModuleNotFoundError: PySide6``. This
shim catches that case and prints a one-liner pointing at the right install
command before importing the real app.
"""

from __future__ import annotations

import sys


def run() -> None:
    try:
        import PySide6  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "CFDTwin GUI requires PySide6, which isn't installed.\n"
            "\n"
            "Install the GUI extras with:\n"
            "    pip install cfdtwin[gui]\n"
            "\n"
            "Or stick with the API-only install if you only need the Python\n"
            "package (cfdtwin.Project etc.) - no GUI launch required.\n"
        )
        raise SystemExit(1)

    # Defer the real import until we know Qt is available.
    from .app import run as _run
    _run()
