"""set_case_file / set_inputs / set_outputs."""

import json

import pytest


def _read_setup(project):
    with open(project._wp.model_setup_file, "r") as f:
        return json.load(f)


def _read_outputs(project):
    with open(project._wp.output_parameters_file, "r") as f:
        return json.load(f)


# --- set_case_file --------------------------------------------------------

def test_set_case_file_persists(tmp_project):
    tmp_project.set_case_file("/path/to/case.cas.h5")
    assert tmp_project._wp.get_case_file() == "/path/to/case.cas.h5"


# --- set_inputs (range tuple shape) ---------------------------------------

def test_set_inputs_simple_range_tuple(tmp_project):
    tmp_project.set_inputs({"inlet|velocity": (0.1, 1.0)})
    setup = _read_setup(tmp_project)
    inputs = setup["model_inputs"]
    assert len(inputs) == 1
    assert inputs[0]["name"] == "inlet"
    assert inputs[0]["parameter"] == "velocity"
    assert inputs[0]["range"] == [0.1, 1.0]
    assert inputs[0]["type"] == "Unknown"        # deferred to run-time
    assert inputs[0]["parameter_path"] == "velocity"


# --- set_inputs (rich dict shape) -----------------------------------------

def test_set_inputs_rich_dict(tmp_project):
    tmp_project.set_inputs({
        "inlet|velocity": {
            "range": (0.1, 1.0),
            "bc_type": "velocity-inlet",
            "parameter_path": "vmag.value",
        }
    })
    setup = _read_setup(tmp_project)
    inp = setup["model_inputs"][0]
    assert inp["type"] == "velocity-inlet"
    assert inp["parameter_path"] == "vmag.value"
    assert inp["range"] == [0.1, 1.0]


# --- set_inputs validation -------------------------------------------------

def test_set_inputs_requires_pipe_in_key(tmp_project):
    with pytest.raises(ValueError, match="bc_name|parameter"):
        tmp_project.set_inputs({"no_pipe_here": (0.1, 1.0)})


def test_set_inputs_rejects_non_dict_top(tmp_project):
    with pytest.raises(TypeError, match="inputs must be a dict"):
        tmp_project.set_inputs([("inlet|v", (0.1, 1.0))])


def test_set_inputs_rich_dict_requires_range(tmp_project):
    with pytest.raises(ValueError, match="requires a 'range' key"):
        tmp_project.set_inputs({"inlet|velocity": {"bc_type": "x"}})


def test_set_inputs_rejects_bad_value_type(tmp_project):
    with pytest.raises(TypeError, match="tuple or dict"):
        tmp_project.set_inputs({"inlet|velocity": "not_valid"})


# --- set_outputs ----------------------------------------------------------

def test_set_outputs_three_categories(tmp_project_with_inputs):
    tmp_project_with_inputs.set_outputs([
        {"name": "outlet_temp", "category": "Report Definition"},
        {"name": "mid_plane", "category": "Surface",
         "field_variables": ["temperature", "velocity-magnitude"]},
        {"name": "fluid", "category": "Cell Zone",
         "field_variables": ["temperature"]},
    ])
    out = _read_outputs(tmp_project_with_inputs)
    assert len(out["outputs"]) == 3
    assert out["outputs"][1]["field_variables"] == ["temperature", "velocity-magnitude"]


def test_set_outputs_report_def_keeps_empty_field_vars(tmp_project_with_inputs):
    tmp_project_with_inputs.set_outputs([
        {"name": "x", "category": "Report Definition"},
    ])
    out = _read_outputs(tmp_project_with_inputs)
    assert out["outputs"][0]["field_variables"] == []


def test_set_outputs_invalid_category_raises(tmp_project_with_inputs):
    with pytest.raises(ValueError, match="category must be one of"):
        tmp_project_with_inputs.set_outputs([
            {"name": "x", "category": "Bogus"},
        ])


def test_set_outputs_missing_name_raises(tmp_project_with_inputs):
    with pytest.raises(ValueError, match="missing 'name'"):
        tmp_project_with_inputs.set_outputs([
            {"category": "Surface", "field_variables": ["t"]},
        ])
