# Integration tests

These run the full cfdtwin pipeline end-to-end against PyFluent's downloadable
**mixing_elbow** example case. Real Fluent — slow, requires a license, not
runnable in GitHub Actions CI.

Run manually before each release:

    pytest tests/integration -v

The case file is downloaded once into `~/.cache/ansys_fluent_core/examples/`
and cached for subsequent runs.

## What's covered

`test_full_pipeline.py` — single test that walks the entire `Project` API:

1. `Project.create` in a temp directory
2. `set_case_file` + `set_inputs` (rich dict form so no Fluent needed yet) +
   `set_outputs`
3. `generate_doe(n=4, method="lhs")`
4. `connect_fluent()` (this is when Fluent actually launches)
5. `run_simulations()` — verifies all 4 sims complete, returns
   `SimulationResult` with `successful=4`, `failed=0`
6. `train(model_name="elbow_run")` — verifies a `TrainingResult` with
   `n_models > 0` and a sane test R²
7. `predict("elbow_run", {...})` returns a `PredictionResult` with the
   expected `values.shape`

If any stage raises or the result objects are wrong shape, the test fails
loud. Useful as the final smoke test before tagging a release.

## Why these aren't in CI

GitHub Actions doesn't have ANSYS Fluent licenses. The unit tests in
`tests/unit/` mock everything Fluent-related and run on every push.
