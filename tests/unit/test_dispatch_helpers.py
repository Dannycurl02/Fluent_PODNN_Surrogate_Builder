"""Pure-function tests for the three dispatch helpers in cfdtwin.project.

These have zero external dependencies — fast and exhaustive."""

import pytest

from cfdtwin.project import (
    _normalize_outputs,
    _normalize_predict_params,
    _split_per_output_config,
)


# --- _normalize_outputs ----------------------------------------------------

class TestNormalizeOutputs:
    def test_none_yields_all_available(self):
        out = _normalize_outputs(None, ["a", "b", "c"])
        assert out == {"a": {}, "b": {}, "c": {}}

    def test_list_subset_with_empty_configs(self):
        out = _normalize_outputs(["a", "c"], ["a", "b", "c"])
        assert out == {"a": {}, "c": {}}

    def test_dict_passes_through(self):
        cfg = {"a": {"pod": {"modes": 5}}, "b": {}}
        out = _normalize_outputs(cfg, ["a", "b"])
        assert out == cfg

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="outputs must be"):
            _normalize_outputs(42, ["a"])


# --- _normalize_predict_params --------------------------------------------

class TestNormalizePredictParams:
    def test_single_dict_marked_as_single(self):
        result, was_single = _normalize_predict_params({"x": 1.0})
        assert result == [{"x": 1.0}]
        assert was_single is True

    def test_list_of_dicts(self):
        result, was_single = _normalize_predict_params(
            [{"x": 1.0}, {"x": 2.0}]
        )
        assert result == [{"x": 1.0}, {"x": 2.0}]
        assert was_single is False

    def test_tuple_of_dicts_works(self):
        result, was_single = _normalize_predict_params(({"x": 1.0},))
        assert result == [{"x": 1.0}]
        assert was_single is False

    def test_list_with_non_dict_raises(self):
        with pytest.raises(TypeError, match="must be a dict"):
            _normalize_predict_params([{"x": 1}, "not a dict"])

    def test_invalid_top_type_raises(self):
        with pytest.raises(TypeError, match="must be a dict or list"):
            _normalize_predict_params(42)


# --- _split_per_output_config ---------------------------------------------

class TestSplitPerOutputConfig:
    def test_separates_pod_and_nn(self):
        cfg = {
            "a": {"pod": {"modes": 10}, "nn": {"learning_rate": 1e-3}},
            "b": {"nn": {"preset": "1d"}},
            "c": {"pod": {"variance": 0.99}},
            "d": {},
        }
        pod, nn = _split_per_output_config(cfg)
        assert pod == {"a": {"modes": 10}, "c": {"variance": 0.99}}
        assert nn == {"a": {"learning_rate": 1e-3}, "b": {"preset": "1d"}}

    def test_empty_input(self):
        pod, nn = _split_per_output_config({})
        assert pod == {}
        assert nn == {}

    def test_returned_dicts_are_copies(self):
        """Mutating the result must not affect the original config."""
        original = {"a": {"pod": {"modes": 10}}}
        pod, _ = _split_per_output_config(original)
        pod["a"]["modes"] = 999
        assert original["a"]["pod"]["modes"] == 10
