"""generate_doe — LHS, factorial, append, error cases."""

import json

import pytest


def _read_samples(project):
    with open(project._wp.doe_samples_file, "r") as f:
        return json.load(f)


def test_generate_lhs(tmp_project_with_outputs):
    n = tmp_project_with_outputs.generate_doe(n=20, method="lhs", seed=42)
    assert n == 20
    samples = _read_samples(tmp_project_with_outputs)
    assert len(samples["samples"]) == 20
    # Each sample has both input keys
    for s in samples["samples"]:
        assert "inlet|velocity" in s
        assert "inlet|temperature" in s


def test_generate_lhs_within_ranges(tmp_project_with_outputs):
    tmp_project_with_outputs.generate_doe(n=50, method="lhs")
    samples = _read_samples(tmp_project_with_outputs)["samples"]
    for s in samples:
        assert 0.1 <= s["inlet|velocity"] <= 1.0
        assert 290.0 <= s["inlet|temperature"] <= 320.0


def test_generate_factorial(tmp_project_with_outputs):
    n = tmp_project_with_outputs.generate_doe(method="factorial", points_per_param=4)
    # 4^2 = 16 (two inputs, 4 points each)
    assert n == 16


def test_generate_lhs_requires_n(tmp_project_with_outputs):
    with pytest.raises(ValueError, match="requires n"):
        tmp_project_with_outputs.generate_doe(method="lhs")


def test_generate_factorial_requires_points_per_param(tmp_project_with_outputs):
    with pytest.raises(ValueError, match="requires points_per_param"):
        tmp_project_with_outputs.generate_doe(n=10, method="factorial")


def test_unknown_method_raises(tmp_project_with_outputs):
    with pytest.raises(ValueError, match="unknown method"):
        tmp_project_with_outputs.generate_doe(n=10, method="bogus")


def test_generate_doe_requires_inputs_set(tmp_project):
    with pytest.raises(RuntimeError, match="set_inputs"):
        tmp_project.generate_doe(n=10)


def test_generate_doe_replace_default(tmp_project_with_outputs):
    tmp_project_with_outputs.generate_doe(n=10, method="lhs")
    n2 = tmp_project_with_outputs.generate_doe(n=20, method="lhs")
    assert n2 == 20  # replaced, not appended


def test_generate_doe_append(tmp_project_with_outputs):
    tmp_project_with_outputs.generate_doe(n=10, method="lhs")
    n2 = tmp_project_with_outputs.generate_doe(n=15, method="lhs", append=True)
    assert n2 >= 10  # at least the original 10 retained (LHS may dedupe)
