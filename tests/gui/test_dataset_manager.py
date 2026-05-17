"""Tests for gui/dataset_manager.py."""

import json
import numpy as np
from pathlib import Path

import pytest

from gui.dataset_manager import DatasetManager


@pytest.fixture
def dataset_dir(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    return d


@pytest.fixture
def dm(dataset_dir):
    return DatasetManager(dataset_dir)


class TestVersioning:
    def test_version_starts_at_zero(self, dm):
        assert dm.get_version() == 0

    def test_bump_increments(self, dm):
        assert dm.bump_version() == 1
        assert dm.bump_version() == 2
        assert dm.bump_version() == 3
        assert dm.get_version() == 3

    def test_bump_persists_to_disk(self, dm, dataset_dir):
        dm.bump_version()
        dm2 = DatasetManager(dataset_dir)
        assert dm2.get_version() == 1

    def test_bump_creates_dir_if_missing(self, tmp_path):
        d = tmp_path / "new_dataset"
        dm = DatasetManager(d)
        dm.bump_version()
        assert d.exists()
        assert dm.get_version() == 1


class TestSimFiles:
    def test_get_completed_ids_empty(self, dm):
        assert dm.get_completed_ids() == []

    def test_get_completed_ids_scans_correctly(self, dm, dataset_dir):
        # Create some sim files
        np.savez_compressed(dataset_dir / "sim_0001.npz", data=[1])
        np.savez_compressed(dataset_dir / "sim_0003.npz", data=[3])
        np.savez_compressed(dataset_dir / "sim_0010.npz", data=[10])

        ids = dm.get_completed_ids()
        assert ids == [1, 3, 10]

    def test_get_sim_count(self, dm, dataset_dir):
        assert dm.get_sim_count() == 0
        np.savez_compressed(dataset_dir / "sim_0001.npz", data=[1])
        np.savez_compressed(dataset_dir / "sim_0002.npz", data=[2])
        assert dm.get_sim_count() == 2

    def test_remove_simulation_deletes_file_and_bumps(self, dm, dataset_dir):
        np.savez_compressed(dataset_dir / "sim_0005.npz", data=[5])
        assert dm.get_version() == 0

        result = dm.remove_simulation(5)
        assert result is True
        assert not (dataset_dir / "sim_0005.npz").exists()
        assert dm.get_version() == 1

    def test_remove_nonexistent_returns_false(self, dm):
        assert dm.remove_simulation(999) is False

    def test_remove_does_not_bump_if_missing(self, dm):
        dm.remove_simulation(999)
        assert dm.get_version() == 0

    def test_ignores_non_sim_npz_files(self, dm, dataset_dir):
        np.savez_compressed(dataset_dir / "coordinates.npz", data=[0])
        np.savez_compressed(dataset_dir / "sim_0001.npz", data=[1])
        assert dm.get_completed_ids() == [1]
        assert dm.get_sim_count() == 1


class TestStaleness:
    def test_model_stale_when_dataset_newer(self, dm):
        dm.bump_version()
        dm.bump_version()  # version 2
        assert dm.is_model_stale({'dataset_version': 1}) is True

    def test_model_not_stale_when_same_version(self, dm):
        dm.bump_version()
        assert dm.is_model_stale({'dataset_version': 1}) is False

    def test_model_not_stale_when_no_version_file(self, dm):
        # version is 0, model version 0 -> not stale
        assert dm.is_model_stale({'dataset_version': 0}) is False

    def test_model_stale_missing_key(self, dm):
        dm.bump_version()
        # missing dataset_version key defaults to 0
        assert dm.is_model_stale({}) is True
