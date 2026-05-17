# Examples

Four scripts that exercise the cfdtwin API end-to-end against PyFluent's
`mixing_elbow.cas.h5` case. Source for each is inlined below, or available in
[docs/examples/](https://github.com/UARK-NED3/CFDTwin/tree/master/docs/examples)
on GitHub.

> **Requirements:** Python 3.10+, `pip install cfdtwin`, and a working ANSYS
> Fluent installation. Each script downloads the case via PyFluent's example
> helper on first run; subsequent runs reuse the local cache.

## quickstart.py

Smallest end-to-end pipeline: project → DOE → 20 simulations → train → predict.

```python
--8<-- "docs/examples/quickstart.py"
```

## full_workflow.py

Same pipeline as quickstart with verbose prints between stages, a
batch-prediction sweep, and notes on the resulting disk layout.

```python
--8<-- "docs/examples/full_workflow.py"
```

## training_tuning.py

Trains three surrogate variants (defaults; fixed POD modes; variance cutoff
with a custom learning rate) and prints a side-by-side comparison. Reuses the
`elbow_study/` project that `full_workflow.py` creates, so run that first.

```python
--8<-- "docs/examples/training_tuning.py"
```

## discovering_bcs.py

Launches Fluent and lists every boundary condition, settable parameter path,
surface, and report definition exposed by the case. Useful when starting from
an unfamiliar case file. Also demonstrates the
[`Project.list_available_inputs()`](../api/project.md#cfdtwin.Project.list_available_inputs)
shortcut.

```python
--8<-- "docs/examples/discovering_bcs.py"
```
