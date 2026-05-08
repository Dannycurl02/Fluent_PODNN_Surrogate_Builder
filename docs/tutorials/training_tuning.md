# Training tuning

`project.train()` accepts a per-output config dict so you can tune POD mode
count, NN architecture, and hyperparameters per individual output.

> **Run it:** [`examples/training_tuning.py`](https://github.com/UARK-NED3/CFDTwin/blob/master/examples/training_tuning.py) trains three variants of the mixing_elbow surrogate side-by-side and prints a comparison.

## The three shapes of `outputs=`

### 1. Omit it — train every output with defaults

```python
project.train(model_name="run1")
```

Discovers every `(location, field)` pair available from your simulation
data, trains a model for each, uses default POD modes (auto-detected from
data shape) and the default NN preset for the output type.

### 2. List of strings — train a subset, defaults

```python
project.train(model_name="run1", outputs=[
    "mid_plane_temperature",
    "outlet_temp_value",
])
```

Same defaults as above, but only the named outputs. Keys are
`<location>_<field>`. Look up valid keys with
`project.train(model_name="dryrun")` (which fails with a list when there's
no sim data) or by reading your `output_parameters.json`.

### 3. Dict — per-output overrides

```python
project.train(model_name="run1", outputs={
    "mid_plane_temperature": {
        "pod": {"modes": 20},
        "nn":  {"learning_rate": 5e-4, "hidden_layers": [128, 64]},
    },
    "mid_plane_velocity-magnitude": {
        "pod": {"variance": 0.99},
    },
    "outlet_temp_value": {
        "nn": {"preset": "1d"},
    },
})
```

Each key is the model_key. Each value can have a `pod` block, an `nn` block,
or both. Outputs not listed are skipped — pass an empty dict to include
with defaults: `"foo": {}`.

## POD truncation

POD reduces a high-dimensional field to a small number of modes before
the NN learns to predict them. Two ways to specify the truncation:

**Fixed mode count** — same model size every time:

```python
"mid_plane_temperature": {"pod": {"modes": 20}}
```

**Variance fraction** — retain enough modes to capture this fraction of
the total variance. The number of modes varies between runs:

```python
"mid_plane_temperature": {"pod": {"variance": 0.99}}   # 99% variance
```

Specifying both raises an error. Pick one.

When in doubt, start with **`{"variance": 0.99}`** for a defensible default,
then switch to a fixed `modes` count once you've seen what number it picks.

## NN configuration

The trainer has three presets — `1d`, `2d`, `3d` — each tuning hidden
layers, dropout, learning rate, batch size, and early-stopping patience for
its target output dimensionality. The preset is auto-selected from the
output type.

You override individual fields without giving up the preset:

```python
"mid_plane_temperature": {"nn": {"learning_rate": 5e-4}}
# Uses the 2D preset's hidden layers, dropout, etc., but with a smaller LR.
```

Or switch the base preset:

```python
"outlet_temp_value": {"nn": {"preset": "1d", "learning_rate": 1e-3}}
# Falls back to the 1D preset (since this is a scalar) and overrides LR.
```

Available NN config keys (any preset's defaults, override individually):

- `hidden_layers` — list of int, e.g. `[128, 128, 64]`
- `dropout` — list of float, length matches `hidden_layers`
- `l2` — L2 regularization weight
- `learning_rate` — Adam initial LR
- `batch_size` — int, or `"adaptive"` (scales with dataset size)
- `es_patience`, `es_start_epoch`, `es_min_delta` — early stopping
- `lr_patience`, `lr_factor`, `lr_min` — ReduceLROnPlateau

## Top-level training defaults

```python
project.train(
    model_name="run1",
    outputs=...,
    epochs=500,                 # max epochs (early stopping usually halts sooner)
    test_size=0.2,              # fraction held out for test metrics
    random_seed=42,             # reproducible train/test split + NN init
    exclude_range=(95, 100),    # optionally skip a range of sim IDs
)
```

These are global defaults. `outputs={...}` overrides per output.

## Reading the result

```python
result = project.train(model_name="run1", outputs={...})
print(result.summary())
print(result.best_model().model_name)        # highest test R²
for m in result.models:
    print(m.model_name, m.test_metrics.r2, m.n_modes)
```
