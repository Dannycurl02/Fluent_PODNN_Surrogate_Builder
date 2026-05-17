"""Public Project API.

A `Project` represents a folder on disk containing the case-file reference,
input/output specs, DOE samples, simulation outputs, and trained models. It
mirrors the GUI's wizard structure (Setup -> DOE -> Simulate -> Train ->
Analyze) but each stage is one method call.

See docs for the multi-shape arguments (`outputs=`, `predict(params=)`,
per-output config dicts, etc.) — every shape has a worked example there.
"""

from __future__ import annotations

import json
import logging
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import doe as _doe
from . import fluent as _fluent
from . import simulation as _sim
from . import training as _training
from . import visualization as _viz
from . import _project_system
from .results import (
    ModelInfo,
    PredictionResult,
    SimulationResult,
    TrainingResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal dispatch helpers — separate functions so they're easy to unit-test.
# ---------------------------------------------------------------------------

def _normalize_outputs(outputs, available_keys):
    """Coerce the train/predict `outputs=` kwarg into a model_key -> config dict.

    Accepts:
      None             -> all available outputs, defaults
      ["a", "b"]       -> {"a": {}, "b": {}}
      {"a": {...}, ...} -> as-is
    """
    if outputs is None:
        return {k: {} for k in available_keys}
    if isinstance(outputs, list):
        return {k: {} for k in outputs}
    if isinstance(outputs, dict):
        return dict(outputs)
    raise TypeError(
        f"outputs must be None, list, or dict; got {type(outputs).__name__}"
    )


def _normalize_predict_params(params):
    """Coerce predict input into a list of dicts. Returns (list, was_single)."""
    if isinstance(params, dict):
        return [params], True
    if isinstance(params, (list, tuple)):
        for p in params:
            if not isinstance(p, dict):
                raise TypeError("Every entry in predict(params) list must be a dict")
        return list(params), False
    raise TypeError(
        f"predict params must be a dict or list of dicts; got {type(params).__name__}"
    )


def _split_per_output_config(outputs_cfg):
    """Split the unified per-output dict into separate POD and NN dicts for the
    trainer's keyword args. Each value may have keys 'pod' and 'nn'."""
    pod_overrides = {}
    nn_overrides = {}
    for key, cfg in outputs_cfg.items():
        if 'pod' in cfg:
            pod_overrides[key] = dict(cfg['pod'])
        if 'nn' in cfg:
            nn_overrides[key] = dict(cfg['nn'])
    return pod_overrides, nn_overrides


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

class Project:
    """A CFDTwin project — a folder on disk, plus optional Fluent connection.

    Construct via `Project.create(path, name)` or `Project.open(path)` —
    `Project(path)` is reserved for internal use.
    """

    def __init__(self, _wp: _project_system.WorkflowProject):
        self._wp = _wp
        self._solver = None  # set by connect_fluent()

    # --- Lifecycle ---------------------------------------------------------

    @classmethod
    def create(cls, project_path: str | Path, name: str) -> "Project":
        """Create a new project folder. Errors if the folder already contains a project_info.json."""
        wp = _project_system.WorkflowProject(project_path)
        if wp.project_info_file.exists():
            raise FileExistsError(
                f"A project already exists at {wp.project_path}. Use Project.open() to load it."
            )
        if not wp.create(name):
            raise RuntimeError(f"Failed to create project at {wp.project_path}")
        return cls(wp)

    @classmethod
    def open(cls, project_path: str | Path) -> "Project":
        """Open an existing project folder. Errors if no project_info.json is found."""
        wp = _project_system.WorkflowProject(project_path)
        if not wp.load():
            raise FileNotFoundError(
                f"No project_info.json at {wp.project_path}. Use Project.create() for a new project."
            )
        return cls(wp)

    # --- Path properties ---------------------------------------------------

    @property
    def path(self) -> Path:
        return self._wp.project_path

    @property
    def name(self) -> str:
        return self._wp.info.get('project_name', '') if self._wp.info else ''

    # --- Setup (declarative) ----------------------------------------------

    def set_case_file(self, case_file_path: str | Path) -> None:
        """Store the .cas / .cas.h5 file reference path. Doesn't touch Fluent."""
        self._wp.set_case_file(case_file_path)

    def set_inputs(self, inputs: dict) -> None:
        """Declare inputs (no Fluent connection required).

        Three accepted entry shapes — choose one per input:

            # 1. BC range tuple (lo, hi). bc_type and parameter_path resolve at run time.
            project.set_inputs({"inlet|velocity": (0.1, 1.0)})

            # 2. BC rich dict — fully declarative, no Fluent needed at any stage.
            project.set_inputs({
                "inlet|velocity": {
                    "range": (0.1, 1.0),
                    "bc_type": "velocity-inlet",
                    "parameter_path": "vmag.value",
                },
            })

            # 3. Fluent input parameter (named expression with input_parameter=True).
            # Key is the expression name; pass category="Input Parameter" and
            # the unit string. Easiest path: feed back from list_available_inputs().
            project.set_inputs({
                "inlet_vel": {
                    "range": (0.2, 0.8),
                    "category": "Input Parameter",
                    "unit": "m/s",
                },
            })

        BC keys use "<bc_name>|<parameter_display>" (e.g. "inlet|velocity").
        Input-parameter keys are just the expression name.
        """
        if not isinstance(inputs, dict):
            raise TypeError("inputs must be a dict")

        model_inputs = []
        for key, value in inputs.items():
            is_input_param = (
                isinstance(value, dict)
                and value.get('category') == 'Input Parameter'
            )

            if is_input_param:
                if 'range' not in value:
                    raise ValueError(f"Input {key!r}: dict form requires a 'range' key")
                # The named expression IS the parameter — no sub-parameter to pick.
                # Tolerate "name|name" form so list_available_inputs() output can
                # be fed back as-is.
                expr_name = key.split('|', 1)[0]
                model_inputs.append({
                    'name': expr_name,
                    'type': 'Input Parameter',
                    'category': 'Input Parameter',
                    'parameter': expr_name,
                    'parameter_path': expr_name,
                    'unit': value.get('unit', ''),
                    'range': list(value['range']),
                })
                continue

            if '|' not in key:
                raise ValueError(
                    f"Input key {key!r} must be 'bc_name|parameter' (e.g. 'inlet|velocity'), "
                    "or a dict with category='Input Parameter' for Fluent named expressions"
                )
            bc_name, param = key.split('|', 1)
            entry = {
                'name': bc_name,
                'parameter': param,
            }
            if isinstance(value, tuple) and len(value) == 2:
                entry['type'] = 'Unknown'
                entry['parameter_path'] = param.replace(' > ', '.')
                entry['range'] = list(value)
            elif isinstance(value, dict):
                if 'range' not in value:
                    raise ValueError(f"Input {key!r}: dict form requires a 'range' key")
                entry['type'] = value.get('bc_type', 'Unknown')
                entry['parameter_path'] = value.get('parameter_path', param.replace(' > ', '.'))
                entry['range'] = list(value['range'])
            else:
                raise TypeError(
                    f"Input {key!r}: value must be (lo, hi) tuple or dict; got {type(value).__name__}"
                )
            model_inputs.append(entry)

        existing = {}
        if self._wp.model_setup_file.exists():
            with open(self._wp.model_setup_file, 'r') as f:
                existing = json.load(f)
        existing['model_inputs'] = model_inputs
        with open(self._wp.model_setup_file, 'w') as f:
            json.dump(existing, f, indent=2)

    def list_available_inputs(self) -> list[dict]:
        """Discover BCs + input parameters from the connected Fluent session.

        Requires a live Fluent connection (call ``connect_fluent()`` first).
        Each returned dict has at least ``name``, ``type``, ``category``;
        ``Input Parameter`` entries also carry ``unit``, ``current_value``, and
        ``definition``. The dicts can be fed back into ``set_inputs`` after
        adding a ``range``::

            project.connect_fluent()
            for item in project.list_available_inputs():
                print(item['name'], item['category'])
        """
        if self._solver is None:
            raise RuntimeError(
                "No Fluent connection. Call connect_fluent() before list_available_inputs()."
            )
        from . import _project_manager
        return _project_manager.get_available_inputs(self._solver)

    def set_outputs(self, outputs: list[dict]) -> None:
        """Declare outputs to extract from each simulation.

        Each entry is a dict:

            project.set_outputs([
                {"name": "outlet_temp", "category": "Report Definition"},
                {"name": "mid_plane",  "category": "Surface",
                 "field_variables": ["temperature", "velocity-magnitude"]},
                {"name": "fluid",      "category": "Cell Zone",
                 "field_variables": ["temperature"]},
            ])

        category must be one of: "Report Definition", "Surface", "Cell Zone".
        field_variables is required for Surface and Cell Zone, ignored for
        Report Definition (the report's value is the only output).
        """
        if not isinstance(outputs, list):
            raise TypeError("outputs must be a list of dicts")

        valid_categories = {"Report Definition", "Surface", "Cell Zone"}
        cleaned = []
        for out in outputs:
            if not isinstance(out, dict):
                raise TypeError("each output entry must be a dict")
            if 'name' not in out:
                raise ValueError(f"output entry missing 'name': {out}")
            if 'category' not in out or out['category'] not in valid_categories:
                raise ValueError(
                    f"output {out['name']!r}: category must be one of {sorted(valid_categories)}"
                )
            entry = {
                'name': out['name'],
                'category': out['category'],
                'field_variables': list(out.get('field_variables', [])),
            }
            cleaned.append(entry)

        with open(self._wp.output_parameters_file, 'w') as f:
            json.dump({'outputs': cleaned}, f, indent=2)

    # --- DOE ---------------------------------------------------------------

    def generate_doe(
        self,
        n: int | None = None,
        method: str = "lhs",
        points_per_param: int | None = None,
        seed: int = 42,
        append: bool = False,
    ) -> int:
        """Generate DOE samples within the ranges from `set_inputs(...)`.

        Parameters
        ----------
        n : int, optional
            Sample count. Required for method='lhs', ignored for 'factorial'.
        method : str, default 'lhs'
            'lhs' (Latin hypercube) or 'factorial' (full grid).
        points_per_param : int, optional
            Required for method='factorial'. Total samples = n_per_param ** n_inputs.
        seed : int, default 42
            Random seed for LHS reproducibility.
        append : bool, default False
            If True, add to existing samples; otherwise replace.

        Returns
        -------
        int
            Total number of samples now stored.
        """
        if not self._wp.model_setup_file.exists():
            raise RuntimeError("Call set_inputs(...) before generate_doe(...)")

        with open(self._wp.model_setup_file, 'r') as f:
            setup = json.load(f)
        ranges = {}
        for inp in setup.get('model_inputs', []):
            if 'range' not in inp:
                raise RuntimeError(
                    f"Input {inp['name']}|{inp.get('parameter','')} has no range defined"
                )
            key = f"{inp['name']}|{inp.get('parameter', 'value')}"
            ranges[key] = {'min': inp['range'][0], 'max': inp['range'][1]}

        existing_samples = []
        if append and self._wp.doe_samples_file.exists():
            existing_samples, _ = _doe.load_doe_samples(self._wp.doe_samples_file)

        if method == "lhs":
            if n is None:
                raise ValueError("method='lhs' requires n")
            samples = _doe.generate_lhs_samples(ranges, n, existing_samples=existing_samples)
        elif method == "factorial":
            if points_per_param is None:
                raise ValueError("method='factorial' requires points_per_param")
            samples = _doe.generate_factorial_samples(
                ranges, points_per_param, existing_samples=existing_samples,
            )
        else:
            raise ValueError(f"unknown method {method!r}; use 'lhs' or 'factorial'")

        all_samples = existing_samples + samples if append else samples
        _doe.save_doe_samples(self._wp.doe_samples_file, all_samples, ranges)
        return len(all_samples)

    # --- Fluent ------------------------------------------------------------

    def connect_fluent(
        self,
        precision: str | None = None,
        processor_count: int | None = None,
        dimension: int | None = None,
        use_gui: bool | None = None,
    ) -> None:
        """Launch Fluent and load the project's case file.

        Any kwarg passed here overrides the matching field from the project's
        stored ``solver_settings`` (or the built-in defaults if none stored).
        Pass ``precision="single"`` when the case file was saved in single
        precision — Fluent will refuse to read a single-precision case in a
        double-precision session.

        Holds the solver handle on the Project; subsequent calls re-use it.
        """
        if self._solver is not None:
            logger.info("Fluent already connected; skipping launch")
            return
        case_file = self._wp.get_case_file()
        if not case_file or not Path(case_file).exists():
            raise RuntimeError(
                "No valid case file set. Call set_case_file(path) first."
            )
        solver_settings = dict(self._wp.info.get('solver_settings') or {
            'precision': 'double', 'processor_count': 4, 'dimension': 3, 'use_gui': False,
        })
        overrides = {
            'precision': precision,
            'processor_count': processor_count,
            'dimension': dimension,
            'use_gui': use_gui,
        }
        for k, v in overrides.items():
            if v is not None:
                solver_settings[k] = v
        solver = _fluent.launch_fluent(case_file, solver_settings, log_dir=self._wp.logs_dir)
        if solver is None:
            raise RuntimeError("Fluent launch failed; check logs/")
        self._solver = solver

    def disconnect_fluent(self) -> None:
        if self._solver is not None:
            try:
                self._solver.exit()
            except Exception as e:
                logger.warning(f"Error closing Fluent: {e}")
            self._solver = None

    # --- Simulations -------------------------------------------------------

    def run_simulations(
        self,
        iterations: int | None = None,
        reinitialize: bool = True,
        verbose: bool = False,
        on_progress: Callable | None = None,
        on_iteration: Callable | None = None,
    ) -> SimulationResult:
        """Run all incomplete DOE simulations. Auto-connects Fluent if needed.

        Parameters
        ----------
        iterations
            None (default) honors the iteration count baked into the case
            file. Pass an int to override for this batch.
        reinitialize
            Re-run hybrid initialization before each sim. Usually what you want.
        verbose
            If True, prints one "sim X/N <status>" line per simulation as it
            starts and finishes. Ignored if ``on_progress`` is supplied — your
            callback takes over.
        on_progress
            ``callback(idx, total, sim_id, status)`` — fires when each sim
            starts, finishes, or fails.
        on_iteration
            ``callback(sim_id, iter_num, total_iters)`` — fires per Fluent
            iteration. Use sparingly; very chatty for long sim batches.
        """
        if self._solver is None:
            self.connect_fluent()
        if not self._wp.model_setup_file.exists():
            raise RuntimeError("set_inputs/set_outputs not called yet")
        with open(self._wp.model_setup_file, 'r') as f:
            setup_data = json.load(f)
        doe_samples, _ = _doe.load_doe_samples(self._wp.doe_samples_file)
        if not doe_samples:
            raise RuntimeError("No DOE samples; call generate_doe(...) first")

        if verbose and on_progress is None:
            def on_progress(idx, total, sim_id, status):
                print(f"  sim {idx}/{total} (id={sim_id}): {status}", flush=True)

        # Track failed sim_ids by wrapping the user's on_progress callback.
        failed_ids: list[int] = []
        def _track(idx, total, sim_id, status):
            if status == 'failed':
                failed_ids.append(sim_id)
            if on_progress is not None:
                on_progress(idx, total, sim_id, status)

        summary = _sim.run_remaining_simulations(
            solver=self._solver,
            setup_data=setup_data,
            dataset_dir=self._wp.dataset_dir,
            iterations=iterations,
            on_progress=_track,
            on_iteration=on_iteration,
            reinitialize=reinitialize,
            doe_samples=doe_samples,
        )
        return SimulationResult(
            successful=summary['successful'],
            failed=summary['failed'],
            total=summary['total'],
            elapsed=summary['elapsed'],
            stopped_reason=summary.get('stopped_reason'),
            failed_ids=failed_ids,
        )

    # --- Training ----------------------------------------------------------

    def train(
        self,
        model_name: str,
        outputs=None,
        epochs: int = 500,
        test_size: float = 0.2,
        random_seed: int = 42,
        exclude_range: tuple[int, int] | None = None,
        excluded_ids: list[int] | None = None,
        on_progress: Callable | None = None,
        on_epoch: Callable | None = None,
        verbose: int = 1,
    ) -> TrainingResult:
        """Train surrogate models for the requested outputs.

        Parameters
        ----------
        model_name : str
            Folder name under ``project/models/``. Auto-suffixed (``_2``,
            ``_3``, ...) on collision; the final name is in
            ``result.model_name``. Letters/digits/_/- only.
        outputs : None | list[str] | dict, default None
            **None** = train every (location, field_variable) in the project,
            with defaults.

            **list of str** = train just those model_keys
            (``"<location>_<field>"``), with defaults::

                outputs=["mid_plane_temperature", "outlet_temp_value"]

            **dict** = per-output overrides::

                outputs={
                    "mid_plane_temperature": {
                        "pod": {"modes": 20},          # OR {"variance": 0.99}
                        "nn":  {"learning_rate": 5e-4, "hidden_layers": [128, 64]},
                    },
                    "outlet_temp_value": {
                        "nn": {"preset": "1d"},        # switch base preset
                    },
                }
        epochs, test_size, random_seed, exclude_range : training defaults
        excluded_ids : list[int], optional
            Arbitrary 1-indexed sim_ids to withhold from training. Combined
            with `exclude_range` (a sample is dropped if either matches).
        """
        if not self._wp.dataset_dir.exists() or not list(self._wp.dataset_dir.glob("sim_*.npz")):
            raise RuntimeError("No simulation data found; call run_simulations() first")

        # Auto-suffix on collision; final name returned in result.model_name.
        final_name = self._unique_model_name(model_name)
        if final_name != model_name:
            warnings.warn(
                f"Model name {model_name!r} already exists; saving as {final_name!r} instead.",
                UserWarning,
                stacklevel=2,
            )

        # Discover available output keys to filter and to validate user input.
        data = _training.load_training_data(
            self._wp.project_path,
            exclude_range=exclude_range,
            excluded_ids=excluded_ids,
        )
        available_keys = list(data['outputs'].keys())

        outputs_cfg = _normalize_outputs(outputs, available_keys)
        unknown = set(outputs_cfg) - set(available_keys)
        if unknown:
            raise ValueError(
                f"Unknown output keys: {sorted(unknown)}. Available: {sorted(available_keys)}"
            )

        pod_overrides, nn_overrides = _split_per_output_config(outputs_cfg)

        summary = _training.train_all_models(
            dataset_dir=self._wp.project_path,
            model_name=final_name,
            test_size=test_size,
            epochs=epochs,
            exclude_range=exclude_range,
            excluded_ids=excluded_ids,
            on_progress=on_progress,
            on_epoch=on_epoch,
            output_filter=list(outputs_cfg.keys()),
            random_seed=random_seed,
            per_output_pod=pod_overrides,
            per_output_nn=nn_overrides,
        )

        models = [ModelInfo.from_metadata(m) for m in summary.get('models', [])]
        return TrainingResult(
            model_name=final_name,
            models=models,
            n_train_samples=summary['n_train_samples'],
            n_test_samples=summary['n_test_samples'],
            test_split=summary['test_split'],
            epochs=summary['epochs'],
            test_indices=summary.get('test_indices', []),
            train_indices=summary.get('train_indices', []),
            models_dir=self._wp.models_dir / final_name,
        )

    def _unique_model_name(self, base: str) -> str:
        """Append _2, _3, ... until the folder doesn't exist."""
        candidate = base
        n = 2
        while (self._wp.models_dir / candidate).exists():
            candidate = f"{base}_{n}"
            n += 1
        return candidate

    # --- Predict -----------------------------------------------------------

    def predict(self, model_name: str, params) -> PredictionResult:
        """Run a trained model on input parameters.

        Parameters
        ----------
        model_name : str
            Name of the model directory under ``project/models/``.
        params : dict | list[dict]
            **dict** — single sample::

                project.predict("my_run", {"inlet|velocity": 0.5})

            **list of dict** — batch (one prediction per dict)::

                project.predict("my_run", [
                    {"inlet|velocity": 0.5},
                    {"inlet|velocity": 0.7},
                ])
        """
        model_dir = self._wp.models_dir / model_name
        if not model_dir.exists():
            raise KeyError(f"Model {model_name!r} not found in {self._wp.models_dir}")

        params_list, _ = _normalize_predict_params(params)

        # Build X in the same sorted-key order the trainer uses.
        param_names = sorted(params_list[0].keys())
        X = np.array([[p[k] for k in param_names] for p in params_list])

        result = _viz.predict_single_model(model_dir, X)
        if result is None:
            raise RuntimeError(f"Prediction failed for {model_name}")

        meta = ModelInfo.from_metadata(result['metadata'])
        values = np.asarray(result['prediction'])
        if values.ndim == 1:
            values = values.reshape(1, -1)

        coords = None
        if meta.output_type in ('2D', '3D'):
            coords_file = self._wp.dataset_dir / "coordinates.npz"
            if coords_file.exists():
                npz = np.load(coords_file, allow_pickle=True)
                # Coordinates are saved per-location (one set of points serves
                # every field on that surface/zone), keyed "<location>|coordinates".
                coord_key = f"{meta.location}|coordinates"
                if coord_key in npz.files:
                    coords = np.asarray(npz[coord_key])

        return PredictionResult(
            model_name=model_name,
            values=values,
            coordinates=coords,
            metadata=meta,
            inputs=params_list,
        )

    # --- Model management -------------------------------------------------

    def list_models(self) -> list[str]:
        """Return sorted list of model folder names under ``project/models/``."""
        if not self._wp.models_dir.exists():
            return []
        return sorted(
            d.name for d in self._wp.models_dir.iterdir()
            if d.is_dir() and list(d.glob("*_metadata.json"))
        )

    def model_info(self, model_name: str) -> ModelInfo:
        """Return the trained-model metadata. Raises KeyError if missing.

        For models that bundle multiple sub-models (legacy layout — one folder
        held both temperature and velocity), this returns metadata for the
        first metadata.json found. Use a more specific model_name to target
        the per-field model.
        """
        model_dir = self._wp.models_dir / model_name
        if not model_dir.exists():
            raise KeyError(f"Model {model_name!r} not found in {self._wp.models_dir}")
        meta_files = sorted(model_dir.glob("*_metadata.json"))
        if not meta_files:
            raise KeyError(f"No *_metadata.json in {model_dir}")
        with open(meta_files[0], 'r') as f:
            meta = json.load(f)
        return ModelInfo.from_metadata(meta)

    def delete_model(self, model_name: str) -> None:
        """Delete a trained model directory. Raises KeyError if not found."""
        model_dir = self._wp.models_dir / model_name
        if not model_dir.exists():
            raise KeyError(f"Model {model_name!r} not found")
        if not self._wp.delete_model(model_name):
            raise RuntimeError(f"Could not delete model {model_name!r}")

    def export_model(self, model_name: str, dest_path: str | Path) -> Path:
        """Copy a trained model directory to ``dest_path / model_name``.

        Returns the destination path. Existing folder at the destination is
        replaced (after checking that source exists).
        """
        src = self._wp.models_dir / model_name
        if not src.exists():
            raise KeyError(f"Model {model_name!r} not found")
        dest = Path(dest_path) / model_name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return dest
