"""Project.predict — verifies dispatch + result shape using a mocked predictor."""

import json

import numpy as np
import pytest
from unittest.mock import patch


def _make_model_dir(project, model_name, fake_metadata):
    """Plant a fake trained-model folder with a metadata.json."""
    d = project._wp.models_dir / model_name
    d.mkdir()
    fake_metadata["model_name"] = model_name
    with open(d / f"{model_name}_metadata.json", "w") as f:
        json.dump(fake_metadata, f)
    return d


def test_predict_single_dict_returns_one_row(tmp_project, fake_metadata):
    fake_metadata["output_type"] = "2D"
    _make_model_dir(tmp_project, "m1", fake_metadata)

    with patch("cfdtwin.project._viz.predict_single_model",
               return_value={
                   "prediction": np.full((1, 1234), 42.0),
                   "metadata": fake_metadata,
                   "model_dir": tmp_project._wp.models_dir / "m1",
               }) as mock_pred:
        result = tmp_project.predict("m1", {"a": 1.0, "b": 2.0})

    # Predictor was called with a 1-row X, sorted-key order
    assert mock_pred.call_args[0][1].shape == (1, 2)
    assert result.n_samples == 1
    assert result.n_points == 1234
    assert not result.is_scalar


def test_predict_batch_list(tmp_project, fake_metadata):
    fake_metadata["output_type"] = "1D"
    _make_model_dir(tmp_project, "m1", fake_metadata)

    with patch("cfdtwin.project._viz.predict_single_model",
               return_value={
                   "prediction": np.full((3, 1), 42.0),
                   "metadata": fake_metadata,
                   "model_dir": tmp_project._wp.models_dir / "m1",
               }) as mock_pred:
        result = tmp_project.predict("m1", [
            {"a": 1.0}, {"a": 2.0}, {"a": 3.0},
        ])

    assert mock_pred.call_args[0][1].shape == (3, 1)
    assert result.n_samples == 3
    assert result.is_scalar


def test_predict_unknown_model_raises(tmp_project):
    with pytest.raises(KeyError, match="not found"):
        tmp_project.predict("never_trained", {"a": 1.0})


def test_predict_invalid_params_type_raises(tmp_project, fake_metadata):
    _make_model_dir(tmp_project, "m1", fake_metadata)
    with pytest.raises(TypeError, match="dict or list"):
        tmp_project.predict("m1", 42)


# --- list_models / model_info / delete_model ----------------------------

def test_list_models_empty(tmp_project):
    assert tmp_project.list_models() == []


def test_list_models_finds_metadata(tmp_project, fake_metadata):
    _make_model_dir(tmp_project, "m1", fake_metadata)
    _make_model_dir(tmp_project, "m2", fake_metadata)
    assert tmp_project.list_models() == ["m1", "m2"]


def test_model_info_returns_dataclass(tmp_project, fake_metadata):
    _make_model_dir(tmp_project, "m1", fake_metadata)
    mi = tmp_project.model_info("m1")
    assert mi.model_name == "m1"  # was overwritten by _make_model_dir helper
    assert mi.test_metrics.r2 == 0.95


def test_model_info_missing_raises(tmp_project):
    with pytest.raises(KeyError, match="not found"):
        tmp_project.model_info("nope")


def test_delete_model_removes_folder(tmp_project, fake_metadata):
    _make_model_dir(tmp_project, "m1", fake_metadata)
    assert "m1" in tmp_project.list_models()
    tmp_project.delete_model("m1")
    assert "m1" not in tmp_project.list_models()


def test_delete_model_missing_raises(tmp_project):
    with pytest.raises(KeyError, match="not found"):
        tmp_project.delete_model("nope")


def test_export_model_copies_folder(tmp_project, fake_metadata, tmp_path):
    _make_model_dir(tmp_project, "m1", fake_metadata)
    dest_dir = tmp_path / "exports"
    dest_dir.mkdir()
    out = tmp_project.export_model("m1", dest_dir)
    assert out == dest_dir / "m1"
    assert (out / "m1_metadata.json").exists()


def test_export_model_missing_raises(tmp_project, tmp_path):
    with pytest.raises(KeyError, match="not found"):
        tmp_project.export_model("nope", tmp_path)
