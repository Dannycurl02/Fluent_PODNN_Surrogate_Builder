# cfdtwin examples

Runnable scripts that mirror the [tutorials](../tutorials/). Each one is
self-contained — clone the repo, `pip install -e .`, run.

| Script | What it does |
|---|---|
| `quickstart.py` | Smallest end-to-end pipeline against mixing_elbow |
| `full_workflow.py` | Same as quickstart with comments + result inspection |
| `training_tuning.py` | Per-output PCA + NN config overrides |
| `discovering_bcs.py` | Connect to Fluent and list valid input/output keys |

## Requirements

- Python 3.10+
- `pip install cfdtwin` (or `pip install -e .` from a repo clone)
- Working ANSYS Fluent installation (PyFluent locates it via env vars)

## Running

```bash
python docs/examples/quickstart.py
```

Each script downloads the mixing_elbow case via PyFluent's example helper
on first run and caches it locally. Subsequent runs reuse the cache.

`discovering_bcs.py` is a one-shot exploration script — it launches Fluent,
lists boundary conditions / parameters / surfaces / report defs for the
mixing_elbow case, and exits.
