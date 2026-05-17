"""Tests for cfdtwin.results dataclasses."""

from pathlib import Path

import numpy as np
import pytest

from cfdtwin.results import (
    Metrics,
    ModelInfo,
    PredictionResult,
    SimulationResult,
    TrainingResult,
)


def test_metrics_from_dict():
    m = Metrics.from_dict({"r2": 0.95, "rmse": 1.2, "mae": 0.8})
    assert m.r2 == 0.95
    assert m.rmse == 1.2
    assert m.mae == 0.8


def test_metrics_from_dict_with_missing_keys_defaults_to_zero():
    m = Metrics.from_dict({})
    assert m.r2 == 0.0 and m.rmse == 0.0 and m.mae == 0.0


def test_model_info_from_metadata(fake_metadata):
    mi = ModelInfo.from_metadata(fake_metadata)
    assert mi.model_name == "mid_plane_temperature"
    assert mi.output_type == "2D"
    assert mi.has_pod is True
    assert mi.test_metrics.r2 == 0.95
    assert mi.n_modes == 10


def test_model_info_handles_missing_optional_fields(fake_metadata):
    fake_metadata.pop("n_modes")
    fake_metadata.pop("variance_explained")
    mi = ModelInfo.from_metadata(fake_metadata)
    assert mi.n_modes is None
    assert mi.variance_explained is None


def test_simulation_result_summary_succeeded():
    sr = SimulationResult(successful=10, failed=0, total=10, elapsed=125.0)
    s = sr.summary()
    assert "10/10" in s
    assert "2m 5s" in s
    assert "failed" not in s.lower()


def test_simulation_result_summary_with_failures():
    sr = SimulationResult(
        successful=8, failed=2, total=10, elapsed=60.0, failed_ids=[3, 7],
    )
    s = sr.summary()
    assert "2 failed" in s
    assert "[3, 7]" in s


def test_simulation_result_summary_with_stop_reason():
    sr = SimulationResult(
        successful=5, failed=0, total=10, elapsed=30.0,
        stopped_reason="User requested stop",
    )
    assert "stopped: User requested stop" in sr.summary()


def test_training_result_summary_and_best_model(fake_metadata, tmp_path):
    m1 = ModelInfo.from_metadata(fake_metadata)
    fake_metadata2 = dict(fake_metadata)
    fake_metadata2["model_name"] = "second"
    fake_metadata2["test_metrics"] = {"r2": 0.99, "rmse": 0.5, "mae": 0.2}
    m2 = ModelInfo.from_metadata(fake_metadata2)

    tr = TrainingResult(
        model_name="my_run",
        models=[m1, m2],
        n_train_samples=50, n_test_samples=14, test_split=0.2,
        epochs=500, test_indices=[1, 2], train_indices=[3, 4],
        models_dir=tmp_path,
    )

    assert tr.n_models == 2
    assert tr.best_model().model_name == "second"  # higher R²
    assert "my_run" in tr.summary()
    assert "2 model(s)" in tr.summary()


def test_training_result_best_model_returns_none_when_empty(tmp_path):
    tr = TrainingResult(
        model_name="empty", models=[],
        n_train_samples=0, n_test_samples=0, test_split=0.2,
        epochs=0, test_indices=[], train_indices=[],
        models_dir=tmp_path,
    )
    assert tr.best_model() is None
    assert tr.n_models == 0


def test_prediction_result_properties(fake_metadata):
    mi = ModelInfo.from_metadata(fake_metadata)
    pr = PredictionResult(
        model_name="m", values=np.zeros((3, 100)),
        coordinates=np.zeros((100, 3)), metadata=mi,
        inputs=[{"a": 1}, {"a": 2}, {"a": 3}],
    )
    assert pr.n_samples == 3
    assert pr.n_points == 100
    assert pr.is_scalar is False


def test_prediction_result_scalar(fake_metadata):
    fake_metadata["output_type"] = "1D"
    mi = ModelInfo.from_metadata(fake_metadata)
    pr = PredictionResult(
        model_name="m", values=np.zeros((1, 1)),
        coordinates=None, metadata=mi, inputs=[{"a": 1}],
    )
    assert pr.is_scalar is True
    assert pr.n_samples == 1
    assert pr.n_points == 1
