"""Project.create / Project.open lifecycle."""

import pytest

import cfdtwin


def test_create_writes_project_info(tmp_path):
    p = cfdtwin.Project.create(tmp_path / "x", name="hello")
    assert p.name == "hello"
    assert (tmp_path / "x" / "project_info.json").exists()
    assert (tmp_path / "x" / "dataset").is_dir()
    assert (tmp_path / "x" / "models").is_dir()


def test_create_errors_when_project_exists(tmp_path):
    cfdtwin.Project.create(tmp_path / "x", name="first")
    with pytest.raises(FileExistsError, match="already exists"):
        cfdtwin.Project.create(tmp_path / "x", name="second")


def test_open_loads_existing(tmp_path):
    cfdtwin.Project.create(tmp_path / "x", name="hello")
    p = cfdtwin.Project.open(tmp_path / "x")
    assert p.name == "hello"


def test_open_errors_on_empty_folder(tmp_path):
    (tmp_path / "no_proj").mkdir()
    with pytest.raises(FileNotFoundError, match="No project_info.json"):
        cfdtwin.Project.open(tmp_path / "no_proj")


def test_path_and_name_properties(tmp_path):
    p = cfdtwin.Project.create(tmp_path / "x", name="my_study")
    assert p.path == tmp_path / "x"
    assert p.name == "my_study"
