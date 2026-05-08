"""Tests for DOE table logic: generation, redundancy, manual points, persistence."""

import json
import os
import sys

import numpy as np
import pytest

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PySide6.QtWidgets import QApplication

from cfdtwin.doe import (
    generate_lhs_samples,
    generate_factorial_samples,
    save_doe_samples,
    load_doe_samples,
)
from program_files.gui.pages.doe_page import DOESampleModel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# --- Ranges fixture ---

@pytest.fixture
def ranges_2d():
    return {
        'inlet|velocity': {'min': 1.0, 'max': 10.0},
        'wall|temperature': {'min': 300.0, 'max': 500.0},
    }


# ===================================================================
# LHS generation
# ===================================================================

class TestLHS:
    def test_generates_correct_count(self, ranges_2d):
        samples = generate_lhs_samples(ranges_2d, 20)
        assert len(samples) == 20

    def test_values_within_range(self, ranges_2d):
        samples = generate_lhs_samples(ranges_2d, 50)
        for s in samples:
            assert 1.0 <= s['inlet|velocity'] <= 10.0
            assert 300.0 <= s['wall|temperature'] <= 500.0

    def test_respects_existing_points(self, ranges_2d):
        existing = generate_lhs_samples(ranges_2d, 10)
        new = generate_lhs_samples(ranges_2d, 10, existing_samples=existing)
        # New points should not duplicate existing
        for ns in new:
            for es in existing:
                same = all(
                    abs(ns[k] - es[k]) < 1e-6
                    for k in ranges_2d.keys()
                )
                assert not same, "Redundant point found"

    def test_empty_ranges_returns_empty(self):
        assert generate_lhs_samples({}, 10) == []

    def test_all_redundant_returns_empty(self):
        """If we generate 2 points in a tiny range, second generation should filter all."""
        ranges = {'x|val': {'min': 0.0, 'max': 0.0}}
        first = generate_lhs_samples(ranges, 1)
        # All new samples will have value 0.0, same as existing
        second = generate_lhs_samples(ranges, 5, existing_samples=first)
        assert len(second) == 0


# ===================================================================
# Factorial generation
# ===================================================================

class TestFactorial:
    def test_generates_correct_count(self, ranges_2d):
        # 2 params, 3 points each = 9 combinations
        samples = generate_factorial_samples(ranges_2d, 3)
        assert len(samples) == 9

    def test_values_at_linspace(self, ranges_2d):
        samples = generate_factorial_samples(ranges_2d, 3)
        velocities = sorted(set(s['inlet|velocity'] for s in samples))
        assert len(velocities) == 3
        assert pytest.approx(velocities[0]) == 1.0
        assert pytest.approx(velocities[1]) == 5.5
        assert pytest.approx(velocities[2]) == 10.0

    def test_appends_correctly(self, ranges_2d):
        """Factorial with existing points filters redundant."""
        first = generate_factorial_samples(ranges_2d, 2)  # 4 points
        second = generate_factorial_samples(ranges_2d, 2, existing_samples=first)
        assert len(second) == 0  # all redundant

    def test_n_points_less_than_2_returns_empty(self, ranges_2d):
        assert generate_factorial_samples(ranges_2d, 1) == []


# ===================================================================
# Save / Load
# ===================================================================

class TestSaveLoad:
    def test_roundtrip(self, tmp_path, ranges_2d):
        filepath = tmp_path / "doe_samples.json"
        samples = generate_lhs_samples(ranges_2d, 15)

        save_doe_samples(filepath, samples, ranges_2d)
        loaded_samples, loaded_ranges = load_doe_samples(filepath)

        assert len(loaded_samples) == 15
        assert set(loaded_ranges.keys()) == set(ranges_2d.keys())
        for key in ranges_2d:
            assert pytest.approx(loaded_ranges[key]['min']) == ranges_2d[key]['min']
            assert pytest.approx(loaded_ranges[key]['max']) == ranges_2d[key]['max']

    def test_load_nonexistent_returns_empty(self, tmp_path):
        samples, ranges = load_doe_samples(tmp_path / "nope.json")
        assert samples == []
        assert ranges == {}

    def test_values_preserved(self, tmp_path, ranges_2d):
        filepath = tmp_path / "doe.json"
        samples = [{'inlet|velocity': 5.1234, 'wall|temperature': 401.5678}]
        save_doe_samples(filepath, samples, ranges_2d)

        loaded, _ = load_doe_samples(filepath)
        assert pytest.approx(loaded[0]['inlet|velocity']) == 5.1234
        assert pytest.approx(loaded[0]['wall|temperature']) == 401.5678


# ===================================================================
# DOESampleModel (Qt table model)
# ===================================================================

class TestDOESampleModel:
    def test_set_data(self, qapp):
        model = DOESampleModel()
        keys = ['a|x', 'b|y']
        samples = [{'a|x': 1.0, 'b|y': 2.0}, {'a|x': 3.0, 'b|y': 4.0}]
        model.set_data(samples, keys)
        assert model.rowCount() == 2
        assert model.columnCount() == 2

    def test_data_display(self, qapp):
        from PySide6.QtCore import Qt
        model = DOESampleModel()
        model.set_data([{'k': 3.14159}], ['k'])
        val = model.data(model.index(0, 0), Qt.DisplayRole)
        assert '3.14159' in val

    def test_remove_rows(self, qapp):
        model = DOESampleModel()
        samples = [{'x': i} for i in range(5)]
        model.set_data(samples, ['x'])
        assert model.rowCount() == 5

        model.remove_rows([1, 3])  # remove index 1 and 3
        assert model.rowCount() == 3
        remaining = model.get_samples()
        assert [s['x'] for s in remaining] == [0, 2, 4]

    def test_remove_rows_descending_safe(self, qapp):
        """Removing in any order should work correctly."""
        model = DOESampleModel()
        samples = [{'x': i} for i in range(4)]
        model.set_data(samples, ['x'])

        model.remove_rows([0, 2])
        remaining = model.get_samples()
        assert [s['x'] for s in remaining] == [1, 3]

    def test_get_samples_returns_copy(self, qapp):
        model = DOESampleModel()
        original = [{'x': 1.0}]
        model.set_data(original, ['x'])
        got = model.get_samples()
        got.append({'x': 99.0})
        assert model.rowCount() == 1  # original unchanged


# ===================================================================
# Delete with sim file cleanup
# ===================================================================

class TestDeleteWithSimCleanup:
    def test_delete_removes_sim_file(self, tmp_path):
        """Simulates what DOEPage._delete_selected does."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        # Create sim files
        np.savez_compressed(dataset_dir / "sim_0001.npz", data=[1])
        np.savez_compressed(dataset_dir / "sim_0002.npz", data=[2])
        np.savez_compressed(dataset_dir / "sim_0003.npz", data=[3])

        # Delete row index 1 (sim_0002)
        row_idx = 1
        sim_id = row_idx + 1
        sim_file = dataset_dir / f"sim_{sim_id:04d}.npz"
        assert sim_file.exists()
        sim_file.unlink()
        assert not sim_file.exists()

        # Other files still exist
        assert (dataset_dir / "sim_0001.npz").exists()
        assert (dataset_dir / "sim_0003.npz").exists()
