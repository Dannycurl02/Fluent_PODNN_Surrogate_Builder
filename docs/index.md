<p align="center">
  <img src="assets/logo_wordmark.png" alt="CFDTwin" width="420"/>
</p>

Build neural-network surrogate models from ANSYS Fluent simulations. Same
pipeline as the GUI app, accessible from a script.

## Install

```bash
pip install cfdtwin
```

Requires Python 3.10+ and a working Fluent installation.

## What it does

You point cfdtwin at a Fluent case file and tell it what inputs to vary and
what outputs to extract. It samples the input space (LHS or factorial), runs
the sims, trains a POD+NN surrogate per output, and lets you predict from
new input values without re-running Fluent.

## Quick example

```python
import cfdtwin

project = cfdtwin.Project.create("./my_study", name="elbow_study")
project.set_case_file("mixing_elbow.cas.h5")
project.set_inputs({
    "cold-inlet|momentum > velocity_magnitude": (0.2, 0.6),
    "hot-inlet|momentum > velocity_magnitude":  (0.4, 1.2),
})
project.set_outputs([
    {"name": "outlet", "category": "Surface",
     "field_variables": ["temperature"]},
])
project.generate_doe(n=50, method="lhs")
project.connect_fluent(precision="single")   # mixing_elbow is single-precision
project.run_simulations(verbose=True)
project.train(model_name="elbow_v1")

pred = project.predict("elbow_v1", {
    "cold-inlet|momentum > velocity_magnitude": 0.4,
    "hot-inlet|momentum > velocity_magnitude":  0.8,
})
print(pred.values.shape)
```

## Where to go next

- [Quickstart](quickstart.md) — minimal end-to-end script
- [Full workflow tutorial](tutorials/full_workflow.md) — the example above with explanations
- [Training tuning](tutorials/training_tuning.md) — POD modes, NN config, per-output overrides
- [API reference](api/project.md) — every method, every argument
