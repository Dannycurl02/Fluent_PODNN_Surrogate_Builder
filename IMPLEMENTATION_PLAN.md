# cfdtwin Package — Implementation Plan

Phased plan for building the `cfdtwin` Python package alongside the existing
GUI. Each phase is one PR. Settled design decisions are at the bottom.

---

## Phase 1 — Move modules into `cfdtwin/`

Mechanical refactor. GUI keeps working; files just move.

```
cfdtwin/                               ← new top-level package
├── __init__.py                        ← empty (Phase 5 fills it)
├── doe.py                             ← was modules/doe_setup.py
├── simulation.py                      ← was modules/simulation_runner.py
├── pod.py                             ← was modules/pod_reducer.py
├── nn.py                              ← was modules/surrogate_nn.py
├── training.py                        ← was modules/multi_model_trainer.py
├── visualization.py                   ← was modules/multi_model_visualizer.py
├── metrics.py                         ← was modules/metrics.py
├── fluent.py                          ← was modules/fluent_interface.py
├── output_parameters.py               ← unchanged
├── _project_system.py                 ← was modules/project_system.py (internal)
└── _project_manager.py                ← was modules/project_manager.py (internal)
```

GUI: update every `from program_files.modules.X` → `from cfdtwin.X`
(~30 occurrences across 8 files).

Validation: launch the GUI, run the Test100 project end-to-end. Smoke test
only — no behavior should change.

Effort: ~1 hour.

---

## Phase 2 — `pyproject.toml`

Hatchling backend. Declare deps (everything required, settled in API grill 5).
Verify `pip install -e .` works and `python -c "import cfdtwin"` succeeds.

Effort: ~30 min.

---

## Phase 3 — Result classes

```
cfdtwin/results.py:
    @dataclass class SimulationResult: ...
    @dataclass class TrainingResult: ...
    @dataclass class PredictionResult: ...
    @dataclass class ModelInfo: ...
```

Pure dataclasses, no logic. Used as return types for Project methods.

Effort: ~1 hour.

---

## Phase 4 — `Project` class

```
cfdtwin/project.py
```

Methods (settled in API grill, Branch 3):

- `Project.create(path, name)`, `Project.open(path)`
- `set_case_file(path)`, `set_inputs(dict)`, `set_outputs(list)`
- `generate_doe(n=, method=, append=, seed=)`
- `connect_fluent()`
- `run_simulations(...) -> SimulationResult`
- `train(model_name, outputs=..., epochs=..., test_size=..., random_seed=42, ...) -> TrainingResult`
- `predict(model_name, params_or_list) -> PredictionResult`
- `list_models()`, `model_info(name)`, `delete_model(name)`, `export_model(name, dest)`

Three internal dispatch helpers:

- `_normalize_outputs(outputs)` — list/dict/None → dict
- `_normalize_predict_params(params)` — dict/list-of-dicts → list-of-dicts
- `_resolve_pod_config(cfg)` — `{modes:N}` xor `{variance:F}` → PCA `n_components`

Each method is a thin wrapper over the existing module functions. Mostly
validation + dispatch.

Effort: ~3-4 hours including docstrings (with examples for every multi-shape arg).

---

## Phase 5 — `__init__.py`

Lean exports (settled in Branch 4):

```python
from cfdtwin.project import Project
from cfdtwin.results import (
    SimulationResult, TrainingResult, PredictionResult, ModelInfo,
)
__version__ = "0.1.0.dev0"
```

Effort: ~5 min.

---

## Phase 6 — Unit tests

```
tests/
├── conftest.py                         ← shared fixtures
├── unit/
│   ├── test_project_lifecycle.py
│   ├── test_setup.py
│   ├── test_doe.py
│   ├── test_train.py
│   ├── test_predict.py
│   └── test_results.py
└── gui/                                ← existing tests moved here, unchanged
    ├── test_dataset_manager.py
    ├── test_doe_logic.py
    └── test_fluent_manager.py
```

Mocks:
- PyFluent solver: small fake class with `settings.setup.boundary_conditions`,
  `solution.run_calculation.iterate`, etc.
- Tensorflow: monkeypatch `keras.Sequential.fit` and `.predict` to no-op /
  return zero arrays. Speed: ~5 sec per test instead of 5 min.

Coverage target: ~70%.

Effort: ~6-8 hours.

---

## Phase 7 — Integration test

```
tests/integration/
├── conftest.py                         ← downloads mixing_elbow once, caches it
├── test_full_pipeline.py               ← create→setup→DOE→sim→train→predict
└── README.md                           ← "How to run: pytest tests/integration"
```

Uses `pyfluent.examples.download_file("mixing_elbow.cas.h5", ...)` so the test
is self-contained — no large file in the repo. Manual run before each release;
not in CI (Fluent unavailable in GitHub Actions).

Effort: ~3 hours.

---

## Phase 8 — MkDocs site

```
mkdocs.yml
docs/
├── index.md                ← what cfdtwin is, install
├── quickstart.md           ← 10-line example
├── tutorials/
│   ├── full_workflow.md    ← mixing_elbow end-to-end (mirrors integration test)
│   ├── training_tuning.md  ← PCA modes/variance + NN preset/overrides
│   └── fluent_session.md   ← discovering BCs from a live Fluent session
└── api/
    └── (auto-generated via mkdocstrings from docstrings)
```

Style: lean, pragmatic, no fluff (see memory `feedback_docs_style.md`). Every
multi-shape arg has explicit examples for each shape.

Effort: ~4-6 hours.

---

## Phase 9 — GitHub Actions

```
.github/workflows/
├── ci.yml          ← unit tests on push to master + PRs
├── docs.yml        ← build + deploy MkDocs to gh-pages on push to master
└── publish.yml     ← build wheel + publish to PyPI on tag push (v*)
```

PyPI: trusted publishing (no API tokens stored in repo secrets).

Effort: ~2 hours.

---

## Phase 10 — Ship v0.1.0

Tag `v0.1.0`, Action publishes to PyPI, docs live at
`https://dannycurl02.github.io/Fluent_PODNN_Surrogate_Builder/`.

Total estimated effort: 22-28 hours.

---

## Settled design decisions

(References to the API grill in conversation; this section captures the
"why" so the plan stays self-contained.)

- **Layout (Branch 2):** Option A — flat `cfdtwin/` at repo root, GUI imports updated.
- **API granularity (Branch 3):** Flavor A — `Project` class with methods, mirrors GUI mental model.
- **PCA mode truncation (3.1.1):** Either fixed count `{modes: N}` or variance fraction `{variance: F}`. Specifying both is a user error.
- **NN config (3.1.2):** Preset + selective overrides. Mirrors `SurrogateNN.from_preset(..., **overrides)`.
- **Outputs arg (3.1.3):** Accepts list, dict, or None. None = train all outputs in project. List = subset with defaults. Dict = with per-output overrides.
- **Train return (3.1.4):** `TrainingResult` object (not raw dict).
- **Train signature (3.1.5):** Top-level `random_seed`. Folder collision auto-suffixes with warning; final name in `result.model_name`. No `overwrite` kwarg.
- **DOE method dispatch (3.4a):** `method="factorial"` requires `points_per_param`, ignores `n`.
- **DOE append behavior (3.4b):** `append=False` (replace) is default.
- **Setup phase (3.2):** Pure declarative — `set_inputs`/`set_outputs` store specs without Fluent. Validation deferred to `run_simulations`. Live BC discovery covered in a docs appendix.
- **predict input (3.3):** Accepts dict (single sample) or list-of-dicts (batch).
- **predict — one model per call (3.3a).**
- **predict return (3.3β):** `PredictionResult` object with `.values`, `.coordinates`, `.model_name`, `.metadata`.
- **run_simulations connection (3.5a):** Auto-connects if not connected. Persistent / cross-process attach is a future add-on, not v0.1.0.
- **run_simulations return (3.5b):** `SimulationResult` object (consistency with `TrainingResult`).
- **Project lifecycle (3.6):** Explicit `Project.create()` + `Project.open()`. No auto-detect constructor.
- **model_info missing model (3.7a):** Raises `KeyError`.
- **export_model (3.7b):** Folder copy via `shutil.copytree`. No zip.
- **Top-level imports (Branch 4):** Lean — `Project`, the four result classes, `__version__`.
- **Dependencies (Branch 5):** All required, one install. No optional extras.
- **Docs (Branch 6):** MkDocs Material + GitHub Pages, hosted in this repo. Lean style.
- **Tests (Branch 7):** Full unit + integration. Split layout (`tests/unit`, `tests/integration`, `tests/gui`). Mock TF in unit tests. CI runs unit tests on push to master + PRs.
- **GUI relationship (Branch 8):** GUI uses low-level functions (just renamed imports). Refactor to use Project API parked in BACKLOG.
