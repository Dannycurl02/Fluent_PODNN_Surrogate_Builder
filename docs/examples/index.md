# Examples

Four runnable scripts that exercise the cfdtwin API end-to-end against
PyFluent's downloadable `mixing_elbow.cas.h5` case. Read them inline below or
grab the source from
[docs/examples/](https://github.com/UARK-NED3/CFDTwin/tree/master/docs/examples)
on GitHub.

> **Requirements:** Python 3.10+, `pip install cfdtwin`, working ANSYS Fluent
> installation. Each script downloads the case via PyFluent's example helper
> on first run (cached locally afterwards).

## quickstart.py

Smallest end-to-end pipeline: project → DOE → 20 sims → train → predict.

```python
--8<-- "docs/examples/quickstart.py"
```

## full_workflow.py

Same shape as quickstart but with verbose prints between every stage, a
batch-prediction sweep, and disk-layout commentary.

```python
--8<-- "docs/examples/full_workflow.py"
```

## training_tuning.py

Trains three variants of the surrogate (defaults, fixed POD modes, variance
cutoff + custom LR) and prints a side-by-side comparison. Reuses the
`elbow_study/` project that `full_workflow.py` creates, so run that first.

```python
--8<-- "docs/examples/training_tuning.py"
```

## discovering_bcs.py

One-shot exploration script — launches Fluent, lists every BC, settable
parameter path, surface, and report definition the case exposes. Useful when
starting from an unfamiliar case file. Also demonstrates the
[`Project.list_available_inputs()`](../api/project.md#cfdtwin.Project.list_available_inputs)
shortcut.

```python
--8<-- "docs/examples/discovering_bcs.py"
```
