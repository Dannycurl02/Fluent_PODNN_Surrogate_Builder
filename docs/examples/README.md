# cfdtwin examples

Scripts that exercise the cfdtwin API end-to-end against PyFluent's
`mixing_elbow.cas.h5` case. Full source for each is inlined at
https://uark-ned3.github.io/CFDTwin/examples/.

| Script | What it does |
|---|---|
| `quickstart.py` | Smallest end-to-end pipeline against mixing_elbow |
| `full_workflow.py` | Quickstart with comments and result inspection |
| `training_tuning.py` | Per-output POD and NN configuration overrides |
| `discovering_bcs.py` | Launch Fluent and list valid input/output keys |

## Requirements

- Python 3.10+
- `pip install cfdtwin`, or `pip install -e .` from a repo clone
- Working ANSYS Fluent installation (PyFluent locates it via environment variables)

## Running

```bash
python docs/examples/quickstart.py
```

Each script downloads the mixing_elbow case via PyFluent's example helper on
first run and caches it locally. Subsequent runs reuse the cache.

`discovering_bcs.py` launches Fluent, lists boundary conditions, parameters,
surfaces, and report definitions for the case, then exits.
