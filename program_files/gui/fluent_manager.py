"""
Fluent Manager Module
=====================
Singleton managing Fluent solver lifecycle and status.
Emits status_changed signal for the header indicator.
"""

import logging
from enum import Enum

from PySide6.QtCore import QObject, QTimer, Signal

logger = logging.getLogger(__name__)


class FluentStatus(Enum):
    Disconnected = "Disconnected"
    Launching = "Launching"
    Connected = "Connected"
    Busy = "Busy"


class FluentManager(QObject):
    """
    Manages Fluent solver lifecycle.

    Signals
    -------
    status_changed(str)
        Emitted whenever status transitions. Payload is status name string.
    """

    status_changed = Signal(str)

    _instance = None

    @classmethod
    def instance(cls):
        """Return the singleton FluentManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    HEARTBEAT_INTERVAL_MS = 5000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status = FluentStatus.Disconnected
        self._solver = None
        # Heartbeat: detect manual Fluent close / crash and flip the indicator.
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(self.HEARTBEAT_INTERVAL_MS)
        self._heartbeat.timeout.connect(self._check_health)
        self._heartbeat.start()

    # --- Properties ---

    @property
    def status(self):
        return self._status

    @property
    def solver(self):
        return self._solver

    # --- Status transitions ---

    def _set_status(self, new_status):
        if self._status != new_status:
            old = self._status
            self._status = new_status
            logger.debug(f"Fluent status: {old.value} -> {new_status.value}")
            self.status_changed.emit(new_status.value)

    def is_available(self):
        """True only when Connected (not Busy, not Disconnected)."""
        return self._status == FluentStatus.Connected

    # --- Lifecycle ---

    def set_launching(self):
        """Call before starting the launch worker."""
        self._set_status(FluentStatus.Launching)

    def set_connected(self, solver):
        """Call when launch succeeds."""
        self._solver = solver
        self._set_status(FluentStatus.Connected)

    def set_busy(self):
        """Call when starting a blocking operation (simulation, validation)."""
        if self._status == FluentStatus.Connected:
            self._set_status(FluentStatus.Busy)

    def set_idle(self):
        """Call when a blocking operation finishes."""
        if self._status == FluentStatus.Busy:
            self._set_status(FluentStatus.Connected)

    def set_failed(self):
        """Call when launch fails."""
        self._solver = None
        self._set_status(FluentStatus.Disconnected)

    def disconnect(self):
        """Disconnect Fluent session."""
        if self._solver is not None:
            try:
                self._solver.exit()
                logger.info("Fluent session closed")
            except Exception as e:
                logger.warning(f"Error closing Fluent: {e}")
            self._solver = None
        self._set_status(FluentStatus.Disconnected)

    # --- Heartbeat ---

    def _check_health(self):
        """Ping the solver. If it doesn't respond, the user (or a crash) closed
        Fluent — flip the indicator to Disconnected."""
        # Only check when we think we're Connected. During Launching the solver
        # doesn't exist yet; during Busy a sim is using the channel so a probe
        # could conflict; during Disconnected there's nothing to check.
        if self._status != FluentStatus.Connected or self._solver is None:
            return
        try:
            status = self._solver.health_check.check_health()
            healthy = str(status).upper().endswith("SERVING") and "NOT" not in str(status).upper()
        except Exception as e:
            logger.info(f"Fluent health check failed ({e}); marking disconnected")
            healthy = False
        if not healthy:
            self._solver = None
            self._set_status(FluentStatus.Disconnected)
