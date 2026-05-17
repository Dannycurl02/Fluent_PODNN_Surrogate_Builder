@echo off
REM === CFDTwin GUI launcher =========================================
REM For repo clones. Double-click to launch the CFDTwin GUI without
REM opening a terminal window (uses pythonw.exe).
REM
REM pip users: just run `cfdtwin-gui` from any terminal instead;
REM this script is not shipped in the PyPI wheel.
REM
REM Requirements:
REM   - Python 3.10+ with cfdtwin installed (`pip install -e .[dev]`
REM     from a clone, which pulls in PySide6).
REM   - pythonw.exe on PATH (default when you install Python from
REM     python.org with "Add Python to PATH" checked).
REM
REM To put a launcher on your Desktop:
REM   1. Right-click this file -> Copy.
REM   2. Right-click your Desktop -> Paste shortcut (or just Paste).
REM   3. (Optional) Right-click the shortcut -> Properties -> Change Icon
REM      -> browse to gui/assets/logo_icon.png in the repo.
REM
REM If launch fails silently, look in gui/cfdtwin.log for the error.
REM ==================================================================

start "" pythonw.exe -m gui
