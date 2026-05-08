"""Project.train — verifies arg translation, not actual training.

train_all_models is mocked so tests are fast and don't require sim data
on disk."""

from unittest.mock import patch, MagicMock

import pytest


def _fake_summary(model_name="my_run", n_models=2):
    """Returns the dict shape that train_all_models normally produces."""
    return {
        "case_name": "test",
        "trained_date": "2026-05-08T10:00:00",
        "n_models": n_models,
        "n_train_samples": 50,
        "n_test_samples": 14,
        "test_split": 0.2,
        "epochs": 500,
        "test_indices": list(range(14)),
        "train_indices": list(range(14, 64)),
        "models": [
            {
                "model_name": "outA",
                "output_type": "2D",
                "field_name": "temperature",
                "location": "outA",
                "npz_key": "outA|temperature",
                "n_points": 100,
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
            },
        ],
    }


def _fake_load_data(model_keys=("outA", "outB")):
    """Returns the dict shape that load_training_data produces."""
    import numpy as np
    return {
        "parameters": np.zeros((64, 2)),
        "outputs": {k: np.zeros((64, 100)) for k in model_keys},
        "output_info": {
            k: {"location": k, "field": "temperature", "type": "2D",
                "n_points": 100, "n_modes": 10,
                "npz_key": f"{k}|temperature"} for k in model_keys
        },
        "param_names": ["a", "b"],
    }


def _fake_sim_files(project):
    """Drop a fake sim_*.npz so Project.train's existence check passes."""
    import numpy as np
    f = project._wp.dataset_dir / "sim_0001.npz"
    np.savez(f, **{"outA|temperature": np.zeros(100)})


def test_train_passes_through_basic_args(tmp_project_with_outputs):
    proj = tmp_project_with_outputs
    _fake_sim_files(proj)

    with patch("cfdtwin.project._training.train_all_models",
               return_value=_fake_summary()) as mock_train, \
         patch("cfdtwin.training.load_training_data",
               return_value=_fake_load_data()):
        result = proj.train(
            model_name="my_run", epochs=100, test_size=0.15, random_seed=7,
        )

    mock_train.assert_called_once()
    call_kwargs = mock_train.call_args.kwargs
    assert call_kwargs["model_name"] == "my_run"
    assert call_kwargs["epochs"] == 100
    assert call_kwargs["test_size"] == 0.15
    assert call_kwargs["random_seed"] == 7
    assert result.model_name == "my_run"
    assert result.n_models == 1


def test_train_outputs_list_filters_correctly(tmp_project_with_outputs):
    proj = tmp_project_with_outputs
    _fake_sim_files(proj)

    with patch("cfdtwin.project._training.train_all_models",
               return_value=_fake_summary()) as mock_train, \
         patch("cfdtwin.training.load_training_data",
               return_value=_fake_load_data()):
        proj.train(model_name="r1", outputs=["outA"])
    assert mock_train.call_args.kwargs["output_filter"] == ["outA"]


def test_train_outputs_dict_translates_to_per_output_kwargs(tmp_project_with_outputs):
    proj = tmp_project_with_outputs
    _fake_sim_files(proj)

    outputs_cfg = {
        "outA": {"pod": {"modes": 20}, "nn": {"learning_rate": 5e-4}},
        "outB": {"pod": {"variance": 0.99}},
    }
    with patch("cfdtwin.project._training.train_all_models",
               return_value=_fake_summary()) as mock_train, \
         patch("cfdtwin.training.load_training_data",
               return_value=_fake_load_data()):
        proj.train(model_name="r2", outputs=outputs_cfg)

    kwargs = mock_train.call_args.kwargs
    assert kwargs["per_output_pod"] == {
        "outA": {"modes": 20},
        "outB": {"variance": 0.99},
    }
    assert kwargs["per_output_nn"] == {"outA": {"learning_rate": 5e-4}}
    assert set(kwargs["output_filter"]) == {"outA", "outB"}


def test_train_unknown_output_key_raises(tmp_project_with_outputs):
    proj = tmp_project_with_outputs
    _fake_sim_files(proj)

    with patch("cfdtwin.training.load_training_data",
               return_value=_fake_load_data()):
        with pytest.raises(ValueError, match="Unknown output keys"):
            proj.train(model_name="r3", outputs=["nonexistent"])


def test_train_auto_suffixes_on_collision(tmp_project_with_outputs):
    proj = tmp_project_with_outputs
    _fake_sim_files(proj)
    # Pre-create the model dir so the chosen name collides
    (proj._wp.models_dir / "taken").mkdir()

    with patch("cfdtwin.project._training.train_all_models",
               return_value=_fake_summary()) as mock_train, \
         patch("cfdtwin.training.load_training_data",
               return_value=_fake_load_data()):
        with pytest.warns(UserWarning, match="already exists"):
            result = proj.train(model_name="taken")

    assert result.model_name == "taken_2"
    assert mock_train.call_args.kwargs["model_name"] == "taken_2"


def test_train_errors_with_no_sim_data(tmp_project_with_outputs):
    proj = tmp_project_with_outputs
    with pytest.raises(RuntimeError, match="No simulation data"):
        proj.train(model_name="should_fail")
