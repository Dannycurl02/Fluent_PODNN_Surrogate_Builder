# cfdtwin examples

Runnable scripts that exercise the cfdtwin API end-to-end against
PyFluent's `mixing_elbow.cas.h5` case.

> Browsing online? Each script's full source is inlined at
> https://uark-ned3.github.io/CFDTwin/examples/.

| Script | What it does |
|---|---|
| `quickstart.py` | Smallest end-to-end pipeline against mixing_elbow |
| `full_workflow.py` | Same as quickstart with comments + result inspection |
| `training_tuning.py` | Per-output POD + NN config overrides |
| `discovering_bcs.py` | Launch Fluent and list valid input/output keys |

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

`discovering_bcs.py` is one-shot — it launches Fluent, lists BCs /
parameters / surfaces / report defs for the case, and exits.
