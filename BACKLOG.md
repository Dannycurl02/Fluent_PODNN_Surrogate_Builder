# Backlog

Long-term fixes and known risks to address in future updates.

---

## 1. Fluent Case File Version Drift

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

## 2. Unsupported Fluent Configuration Warnings

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

## 3. Test Run Fluent Comparison System

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

## 4. Refactor GUI to use the cfdtwin Project API

**Risk**: Medium (drift over time)

After the cfdtwin package ships (v0.1.0), the GUI continues to call low-level
module functions directly (e.g. `simulation_runner.run_remaining_simulations`).
The new Project class wraps the same calls for library users. Result: two
parallel code paths for the same operations — risk of drift as the API evolves.

**What this work involves**:
- Replace direct module calls in `program_files/gui/pages/*.py` with `Project`
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