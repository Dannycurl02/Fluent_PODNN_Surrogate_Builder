"""Fixtures for the cfdtwin integration tests.

These tests need a real Fluent. The case file is downloaded once via
PyFluent's example helper; subsequent runs use the local cache."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def mixing_elbow_case():
    """Download mixing_elbow.cas.h5 once per test session, return absolute path.

    Cached by PyFluent under ~/.cache/ansys_fluent_core/examples/.
    """
    try:
        from ansys.fluent.core import examples
    except ImportError:
        pytest.skip("ansys-fluent-core not installed")
    path = examples.download_file("mixing_elbow.cas.h5", "pyfluent/mixing_elbow")
    if not Path(path).exists():
        pytest.skip(f"mixing_elbow.cas.h5 download failed: {path}")
    return path
