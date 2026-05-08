"""Shared fixtures across the test suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_project(tmp_path):
    """Create a fresh, empty Project on a tmp folder.

    Returns the cfdtwin.Project. The folder is wiped after the test.
    """
    import cfdtwin
    return cfdtwin.Project.create(tmp_path / "proj", name="test")


@pytest.fixture
def tmp_project_with_inputs(tmp_project):
    """Project with two declared inputs and a case_file pointer.

    Inputs:
      'inlet|velocity'    range (0.1, 1.0)
      'inlet|temperature' range (290, 320)
    """
    tmp_project.set_case_file("/dev/null")
    tmp_project.set_inputs({
        "inlet|velocity": (0.1, 1.0),
        "inlet|temperature": (290.0, 320.0),
    })
    return tmp_project


@pytest.fixture
def tmp_project_with_outputs(tmp_project_with_inputs):
    """Inputs above + three outputs (1D scalar, 2D surface, 3D cell zone)."""
    tmp_project_with_inputs.set_outputs([
        {"name": "outlet_temp", "category": "Report Definition"},
        {"name": "mid_plane", "category": "Surface",
         "field_variables": ["temperature"]},
        {"name": "fluid", "category": "Cell Zone",
         "field_variables": ["temperature"]},
    ])
    return tmp_project_with_inputs


@pytest.fixture
def fake_metadata():
    """Returns a dict shaped like the trainer's *_metadata.json output."""
    return {
        "model_name": "mid_plane_temperature",
        "output_type": "2D",
        "field_name": "temperature",
        "location": "mid_plane",
        "npz_key": "mid_plane|temperature",
        "n_points": 1234,
        "has_pod": True,
        "preset": "2d",
        "model_architecture": "POD+NN",
        "train_metrics": {"r2": 0.99, "rmse": 1.0, "mae": 0.5},
        "test_metrics": {"r2": 0.95, "rmse": 1.5, "mae": 0.8},
        "trained_date": "2026-05-08T10:00:00",
        "dataset_version": 1,
        "n_train_samples": 50,
        "n_test_samples": 14,
        "n_modes": 10,
        "variance_explained": 0.98,
    }
