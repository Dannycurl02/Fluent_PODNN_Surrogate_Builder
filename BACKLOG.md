# Backlog

Two sections: **Pre-v0.1.0 release tasks** (to do before tagging) and
**Long-term fixes & risks** (post-launch).

---

# Pre-v0.1.0 release tasks

## R1. Extensive testing of API + GUI

Verify both surfaces behave correctly end-to-end before tagging v0.1.0.

**API (real Fluent required):**
- `pytest tests/integration` — runs the full pipeline against PyFluent's
  mixing_elbow case
- Each example script runs end-to-end:
  - `python docs/examples/quickstart.py`
  - `python docs/examples/full_workflow.py`
  - `python docs/examples/training_tuning.py` (depends on full_workflow having run)
  - `python docs/examples/discovering_bcs.py`
- Manual edge cases:
  - `predict()` before training → KeyError on missing model
  - `train()` with no sim data → RuntimeError
  - `train(model_name="x")` twice → auto-suffix + warning
  - Per-output config with both `pod.modes` and `pod.variance` → ValueError
  - Predict with mismatched param keys → sensible error

**GUI (manual smoke):**
- Fresh project end-to-end: Setup → DOE → Simulate (small N) → Train → Analyze
- Reopen Test100: every page loads, Analyze auto-focuses first model
- Regression check on every fix from `gui-fixes` branch (Browse button width,
  duplicate dataset path, Fluent disconnect detection, per-iter progress bar,
  Train per-(location, field) row, loss plot linear scale, project delete
  2-stage Yes, Validate→Analyze rename, R² Y-axis padding, click-to-expand
  on middle plots, predict bar follows top-left selection, Export Model)
- Failure modes: kill Fluent mid-sim, click Stop, restart and resume

**CI sanity:** push a no-op commit, confirm the unit-test matrix passes on
3.10 / 3.11 / 3.12.

## R2. Metrics explainer in the docs

The Analyze tab shows two R² values per model — per-sample R² (the middle
plot, one dot per test sample) and whole-model R² (the top-right metrics
table) — and they measure two different things. No docs page explains this
or any of the other metrics (RMSE, MAE).

Add a `docs/concepts/metrics.md` page covering:
- R² basic definition with equation
- **Per-sample R²** — for one prediction, "did the model get the field SHAPE
  right?" Show the equation reducing along the spatial dimension with the
  per-sample mean as baseline. Show what happens when the model collapses to
  predicting a uniform field (R² → 0).
- **Whole-model R²** — per-point R² averaged over spatial points; "does the
  model TRACK parameter-driven variation at each location?" Equation reducing
  along the test-set dimension with the per-point cross-sample mean as
  baseline. Show what happens when the model ignores inputs (R² → 0).
- A failure-mode table showing which metric catches which pathology
  (uniform-field prediction, input-ignoring prediction, etc.).
- A concrete worked example with numbers.
- RMSE and MAE: shorter sections, definitions + when each is more useful
  (RMSE penalizes outliers, MAE is in target units).

Link from:
- `docs/tutorials/training_tuning.md` (when you see low R², which kind?)
- `docs/api/results.md` (the `Metrics` dataclass docstring should reference
  the concepts page)
- Maybe a "What do these mean?" callout next to the metrics table on the
  Analyze page itself (small `(?)` icon → opens the concepts page).

Estimated effort: ~1-2 hours including the equations and figure. Worth doing
before v0.1.0 since the two-R² distinction trips up everyone.

---

## R3. MkDocs palette — exact brand hex

Material's named-color palette doesn't include the CFDTwin brand blues
(`#1583B5` / `#1E5C79`) exactly — `mkdocs.yml` currently uses the closest
named colors (`primary: blue` / `accent: light blue`). Pixel-perfect tuning
to the brand hex needs an `extra_css` override:

- Add `docs/stylesheets/extra.css` with `--md-primary-fg-color`,
  `--md-primary-fg-color--light`, `--md-primary-fg-color--dark`, and
  `--md-accent-fg-color` set to the brand hex.
- Wire it via `extra_css:` in `mkdocs.yml`.
- Verify in both `default` and `slate` schemes.

Effort: ~30 min. Low priority — current colors are close enough that v0.1.0
ships fine without this; do it when the site is live and you can eyeball it.

---

# Long-term fixes & risks

## L1. Fluent Case File Version Drift

**Risk**: High

The project stores a reference path to the `.cas` file but does not track its contents. If the user modifies the case file between sessions (changes mesh, BCs, solver settings, materials) without starting a fresh project, existing datasets and trained models silently become invalid.

**Specific scenarios**:
- User remeshes — point counts change, coordinate reference breaks, all sim data is incompatible
- User renames/removes a BC that is configured as an input — simulation runner will fail or silently skip it
- User changes solver settings (turbulence model, energy equation) — field variables may no longer exist
- User adds report definitions in Fluent (as instructed in the Outputs setup tab) but does **not save the `.cas` file** before running simulations — Fluent won't have the report defs, simulation data will be incomplete, and downstream training/validation will silently use corrupted or missing outputs

**Mitigation ideas**:
- Hash the `.cas` file on first use and store in `project_info.json`. Warn on mismatch at project load.
- After the user configures outputs (especially report defs), prompt them to save the case file in Fluent before proceeding to Simulate.
- Before batch simulation, validate that all configured inputs/outputs still exist in the active Fluent session.
- Track which `.cas` hash each sim was run against. Flag sims that used a different version.
- On app close, prompt "Save Fluent case file before closing?" instead of silently killing the session. Currently the app auto-disconnects Fluent on exit — if the user made changes in Fluent (added report defs, adjusted settings) and hadn't saved, those changes are lost.

---

## L2. Unsupported Fluent Configuration Warnings

**Risk**: High

The surrogate workflow assumes steady-state, fixed-mesh simulations. Several Fluent configurations will silently produce invalid data or crash the batch runner. Users should be warned (or blocked) before they waste compute.

**Known incompatible configurations**:
- Transient simulations — the runner calls `iterate()` assuming steady-state; transient needs `calculate()` with time-stepping
- Adaptive mesh refinement — mesh changes between sims break the coordinate reference system
- Moving/deforming mesh — same coordinate mismatch issue
- Multiphase with phase changes — field variable structure may vary between sims
- Species transport with variable species count

**Mitigation ideas**:
- On Fluent connect, use PyFluent API to query solver type (`solver.settings.setup.general.solver.type`) and warn if transient
- Check for mesh adaption settings and warn if enabled
- Display a "System Limitations" section in the Setup page or project dialog listing what's not supported
- Potentially auto-check via PyFluent and show inline warnings for each detected issue

---

## L3. Test Run Fluent Comparison System

**Risk**: Medium

The Fluent comparison system in the Validate page (`run_fluent_comparison`) reuses the existing solver session to run a fresh simulation for validation. This flow has not been tested end-to-end against a real Fluent case.

**What to verify**:
- Boundary conditions are applied correctly using the same `parameter_path` mapping as the simulation runner
- Hybrid initialization runs without errors
- `extract_field_data` returns data keyed with the same NPZ keys that models expect
- The cache stores and retrieves data correctly across sessions
- Cached data works for all model types (1D report defs, 2D surface fields, 3D volume fields)
- The comparison results display correctly in bar charts and tri-panel field plots

**Mitigation ideas**:
- Manual test with a known case: run a validation sim, compare results against the same DOE point's stored NPZ
- Add a "Validate cached run" button that re-loads cached NPZ and compares keys against model metadata
- Log the NPZ keys stored in cache vs keys expected by each model for debugging mismatches

---

## L5. Move `gui/` into `cfdtwin.gui` namespace

**Risk**: Low (poor citizenship in shared environments)

The GUI ships as a top-level `gui` package in the wheel so the
`cfdtwin-gui` entry point can find it. After `pip install cfdtwin`, the
user's site-packages gets a generic `gui/` folder — likely to collide with
other packages' top-level names.

Fix: move `gui/` → `cfdtwin/gui/`, update every `from gui.X` to
`from cfdtwin.gui.X` (~30 places), update the entry point to
`cfdtwin-gui = "cfdtwin.gui.app:run"`, update the .bat to
`pythonw -m cfdtwin.gui`. Also update `python -m gui` references in README
and BACKLOG L4.

Defer until after v0.1.0 unless another project starts colliding with our
`gui` namespace in user site-packages.

---

## L4. Refactor GUI to use the cfdtwin Project API

**Risk**: Medium (drift over time)

After the cfdtwin package ships (v0.1.0), the GUI continues to call low-level
module functions directly (e.g. `simulation_runner.run_remaining_simulations`).
The new Project class wraps the same calls for library users. Result: two
parallel code paths for the same operations — risk of drift as the API evolves.

**What this work involves**:
- Replace direct module calls in `gui/pages/*.py` with `Project`
  method calls (e.g. `project.run_simulations(...)`).
- Thread the result objects (SimulationResult, TrainingResult, PredictionResult)
  back through Qt signals into the GUI.
- Remove duplicated state management — GUI reads from the Project instance.

**Why deferred**:
- The Project API needs to stabilize in real use first; refactoring against an
  API that may still shift wastes effort.
- ~1-2 days of refactor + retesting risk on the GUI.

**When to do it**: after a few weeks of cfdtwin v0.1.x usage, once the API
hasn't needed breaking changes.
