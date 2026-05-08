# Quickstart

Smallest end-to-end script. Replace `mixing_elbow.cas.h5` with your own
case file and the input/output names with what your case actually exposes.

> **Run it:** [`examples/quickstart.py`](https://github.com/UARK-NED3/CFDTwin/blob/master/examples/quickstart.py) is the runnable version of this page.

```python
import cfdtwin

project = cfdtwin.Project.create("./my_study", name="study1")
project.set_case_file("mixing_elbow.cas.h5")

project.set_inputs({
    "cold-inlet|velocity": (0.2, 0.6),
    "hot-inlet|velocity":  (0.4, 1.2),
})
project.set_outputs([
    {"name": "outlet", "category": "Surface",
     "field_variables": ["temperature"]},
])

project.generate_doe(n=20, method="lhs")
project.run_simulations()
result = project.train(model_name="run1")

print(result.summary())

pred = project.predict("run1", {
    "cold-inlet|velocity": 0.4,
    "hot-inlet|velocity":  0.8,
})
print(pred.values.shape)         # (1, n_points)
```

## What just happened

1. **`Project.create`** — laid out the project folder structure on disk
   (`my_study/dataset/`, `my_study/models/`, etc.).
2. **`set_inputs` / `set_outputs`** — wrote `model_setup.json` and
   `output_parameters.json`. No Fluent required at this stage.
3. **`generate_doe(n=20, method="lhs")`** — sampled 20 points from the input
   space using Latin hypercube and saved them to `doe_samples.json`.
4. **`run_simulations()`** — launched Fluent, applied each DOE point's BCs,
   ran 100 iterations per sim, extracted the configured outputs, saved
   `sim_0001.npz` ... `sim_0020.npz`.
5. **`train(model_name="run1")`** — trained a POD+NN surrogate (one model
   per output field) and saved everything under `my_study/models/run1/`.
6. **`predict(...)`** — evaluated the surrogate at a new input point. No
   Fluent involved; it loads the trained model from disk.

## Running it again

`generate_doe` replaces existing samples by default (use `append=True` to
add to them). `run_simulations` skips already-completed sim IDs, so you
can interrupt it and restart safely.

If you re-run `train(model_name="run1")` while a `run1/` folder already
exists, cfdtwin auto-suffixes the new run as `run1_2`, `run1_3`, etc., and
emits a warning. The final folder name is in `result.model_name`.
