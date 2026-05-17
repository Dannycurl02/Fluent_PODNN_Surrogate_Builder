"""Tests for gui/fluent_manager.py."""

import os
import sys

import pytest

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PySide6.QtWidgets import QApplication

from gui.fluent_manager import FluentManager, FluentStatus


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def fm(qapp):
    """Fresh FluentManager for each test (bypass singleton)."""
    manager = FluentManager.__new__(FluentManager)
    FluentManager.__init__(manager)
    return manager


class FakeSolver:
    """Mock solver with exit() method."""
    def __init__(self):
        self.exited = False

    def exit(self):
        self.exited = True


class TestStatusTransitions:
    def test_starts_disconnected(self, fm):
        assert fm.status == FluentStatus.Disconnected

    def test_launching(self, fm):
        fm.set_launching()
        assert fm.status == FluentStatus.Launching

    def test_connected(self, fm):
        solver = FakeSolver()
        fm.set_connected(solver)
        assert fm.status == FluentStatus.Connected
        assert fm.solver is solver

    def test_full_lifecycle(self, fm):
        """Disconnected -> Launching -> Connected -> Busy -> Connected -> Disconnected"""
        assert fm.status == FluentStatus.Disconnected

        fm.set_launching()
        assert fm.status == FluentStatus.Launching

        solver = FakeSolver()
        fm.set_connected(solver)
        assert fm.status == FluentStatus.Connected

        fm.set_busy()
        assert fm.status == FluentStatus.Busy

        fm.set_idle()
        assert fm.status == FluentStatus.Connected

        fm.disconnect()
        assert fm.status == FluentStatus.Disconnected
        assert solver.exited

    def test_busy_only_from_connected(self, fm):
        fm.set_launching()
        fm.set_busy()  # should not transition from Launching
        assert fm.status == FluentStatus.Launching

    def test_idle_only_from_busy(self, fm):
        solver = FakeSolver()
        fm.set_connected(solver)
        fm.set_idle()  # should not change from Connected
        assert fm.status == FluentStatus.Connected


class TestIsAvailable:
    def test_not_available_when_disconnected(self, fm):
        assert fm.is_available() is False

    def test_not_available_when_launching(self, fm):
        fm.set_launching()
        assert fm.is_available() is False

    def test_available_when_connected(self, fm):
        fm.set_connected(FakeSolver())
        assert fm.is_available() is True

    def test_not_available_when_busy(self, fm):
        fm.set_connected(FakeSolver())
        fm.set_busy()
        assert fm.is_available() is False


class TestDisconnect:
    def test_disconnect_from_connected(self, fm):
        solver = FakeSolver()
        fm.set_connected(solver)
        fm.disconnect()
        assert fm.status == FluentStatus.Disconnected
        assert fm.solver is None
        assert solver.exited

    def test_disconnect_from_busy(self, fm):
        solver = FakeSolver()
        fm.set_connected(solver)
        fm.set_busy()
        fm.disconnect()
        assert fm.status == FluentStatus.Disconnected

    def test_disconnect_from_disconnected_is_noop(self, fm):
        fm.disconnect()
        assert fm.status == FluentStatus.Disconnected

    def test_disconnect_from_launching(self, fm):
        fm.set_launching()
        fm.disconnect()
        assert fm.status == FluentStatus.Disconnected


class TestSignals:
    def test_status_changed_emits(self, fm, qtbot):
        signals = []
        fm.status_changed.connect(lambda s: signals.append(s))

        fm.set_launching()
        fm.set_connected(FakeSolver())
        fm.set_busy()
        fm.set_idle()
        fm.disconnect()

        assert signals == [
            "Launching", "Connected", "Busy", "Connected", "Disconnected"
        ]

    def test_no_signal_on_same_status(self, fm, qtbot):
        signals = []
        fm.status_changed.connect(lambda s: signals.append(s))

        fm.set_launching()
        fm.set_launching()  # same status, no signal

        assert signals == ["Launching"]

    def test_set_failed_emits(self, fm, qtbot):
        signals = []
        fm.status_changed.connect(lambda s: signals.append(s))

        fm.set_launching()
        fm.set_failed()

        assert signals == ["Launching", "Disconnected"]


class TestLaunchWhenConnected:
    def test_set_connected_twice_is_noop(self, fm):
        """Calling set_connected when already connected updates solver."""
        s1 = FakeSolver()
        s2 = FakeSolver()
        fm.set_connected(s1)
        fm.set_connected(s2)
        assert fm.solver is s2
        assert fm.status == FluentStatus.Connected
