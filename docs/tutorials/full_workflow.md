# Full workflow tutorial

Walks the same pipeline as [Quickstart](../quickstart.md) but explains each
step and shows what's on disk at each stage. Uses PyFluent's downloadable
mixing_elbow case so you can reproduce this exactly.

## Get the case file

```python
from ansys.fluent.core import examples
case_file = examples.download_file("mixing_elbow.cas.h5", "pyfluent/mixing_elbow")
print(case_file)
```

The download is cached under `~/.cache/ansys_fluent_core/examples/`.

## Create a project

```python
import cfdtwin
project = cfdtwin.Project.create("./elbow_study", name="elbow_v1")
project.set_case_file(case_file)
```

`elbow_study/project_info.json` now exists with the case-file reference.

## Declare inputs and outputs

```python
project.set_inputs({
    "cold-inlet|momentum > velocity_magnitude": (0.2, 0.6),
    "hot-inlet|momentum > velocity_magnitude":  (0.4, 1.2),
})

project.set_outputs([
    {"name": "outlet", "category": "Surface",
     "field_variables": ["temperature"]},
])
```

Input keys are `"<bc_name>|<parameter_display>"`. The bc names (`cold-inlet`,
`hot-inlet`) come from the Fluent case. See
[Discovering BC names from Fluent](fluent_session.md) if you don't know them.

Categories must be one of:

- **Report Definition** — scalar, e.g. an outlet temperature average
- **Surface** — 2D field on a named surface
- **Cell Zone** — 3D field across a cell zone

`set_outputs` writes `output_parameters.json`.

## Generate DOE samples

```python
project.generate_doe(n=20, method="lhs", seed=42)
```

Samples are written to `doe_samples.json`. LHS is recommended for any
serious training run because it spreads samples evenly across the input
space.

For a small grid sweep instead:

```python
project.generate_doe(method="factorial", points_per_param=4)
# 4 ** 2 = 16 samples (two inputs, four levels each)
```

## Run simulations

```python
project.connect_fluent()
sim = project.run_simulations(iterations=100)
print(sim.summary())
```

Each DOE point is loaded into Fluent, the BCs are applied, the solver
iterates, and outputs are extracted into `dataset/sim_NNNN.npz`. Coordinates
for field outputs are saved once to `dataset/coordinates.npz`.

`run_simulations()` automatically skips already-complete sim IDs, so
interrupting and restarting is safe.

`sim` is a [`SimulationResult`](../api/results.md#cfdtwin.results.SimulationResult)
with `.successful`, `.failed`, `.failed_ids`, `.elapsed`, `.stopped_reason`.

## Train

```python
result = project.train(model_name="elbow_v1", epochs=300)
print(result.summary())
```

This trains a POD+NN model per `(location, field_variable)` pair. For the
case above, that's a single 2D field model on `outlet` predicting
temperature.

`result` is a [`TrainingResult`](../api/results.md#cfdtwin.results.TrainingResult).
`result.models` is a list of [`ModelInfo`](../api/results.md#cfdtwin.results.ModelInfo) — one per trained sub-model.

## Predict

```python
pred = project.predict("elbow_v1", {
    "cold-inlet|momentum > velocity_magnitude": 0.4,
    "hot-inlet|momentum > velocity_magnitude":  0.8,
})
print(pred.values.shape)        # (1, n_points)
print(pred.coordinates.shape)   # (n_points, 3)
```

Field reconstructions need coordinates to be useful (for plotting,
post-processing). cfdtwin loads them automatically from
`dataset/coordinates.npz`.

For batch prediction, pass a list of dicts:

```python
sweep = [
    {"cold-inlet|momentum > velocity_magnitude": 0.3,
     "hot-inlet|momentum > velocity_magnitude":  v}
    for v in [0.5, 0.7, 0.9, 1.1]
]
pred = project.predict("elbow_v1", sweep)
print(pred.values.shape)        # (4, n_points)
```

## What's on disk after all this

```
elbow_study/
├── project_info.json
├── model_setup.json
├── output_parameters.json
├── doe_samples.json
├── dataset/
│   ├── coordinates.npz
│   ├── sim_0001.npz
│   ├── ...
│   └── sim_0020.npz
└── models/
    └── elbow_v1/
        ├── outlet_temperature_nn.h5
        ├── outlet_temperature_nn.npz
        ├── outlet_temperature_pod.npz
        ├── outlet_temperature_metadata.json
        ├── outlet_temperature_loss_curve.png
        └── training_summary.json
```
