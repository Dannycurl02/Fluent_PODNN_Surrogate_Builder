# Discovering BC names from Fluent

`set_inputs(...)` is declarative — you tell cfdtwin which boundary
condition keys to vary, and cfdtwin doesn't validate them against Fluent
until run time. So you have to know the keys upfront.

This page shows how to list them from a live Fluent session if you don't.

> **Run it:** [`docs/examples/discovering_bcs.py`](https://github.com/UARK-NED3/CFDTwin/blob/master/docs/examples/discovering_bcs.py) launches Fluent against mixing_elbow and prints every BC, parameter, surface, and report definition it can find.

## Shortcut: `Project.list_available_inputs()`

If you only care about discovering **inputs**, the Project API wraps the
PyFluent calls below and returns BCs + Fluent input parameters (named
expressions flagged `input_parameter=True`) as one list, already grouped by
category:

```python
import cfdtwin
project = cfdtwin.Project.create("./study", name="elbow")
project.set_case_file("my_case.cas.h5")
project.connect_fluent()

for item in project.list_available_inputs():
    print(item)
# [{'name': 'cold-inlet', 'type': 'Velocity Inlet', 'category': 'Boundary Condition'},
#  {'name': 'inlet_vel', 'type': 'Input Parameter', 'category': 'Input Parameter',
#   'unit': 'm/s', 'current_value': 0.4, 'definition': '0.4 [m/s]'}, ...]
```

Each entry can be fed straight back into `set_inputs` by attaching a `range`:

```python
avail = {x['name']: x for x in project.list_available_inputs()}
ip = avail['inlet_vel']
project.set_inputs({ip['name']: {**ip, 'range': (0.2, 0.8)}})
```

The rest of this page covers the raw PyFluent calls if you need detail the
helper doesn't surface (per-BC sub-parameter paths, surfaces, report defs).

## Connect with PyFluent and load your case

```python
import ansys.fluent.core as pyfluent
solver = pyfluent.launch_fluent(precision="double", processor_count=4,
                                dimension=3, mode="solver")
solver.settings.file.read_case(file_name="my_case.cas.h5")
```

## List boundary conditions

```python
bc_settings = solver.settings.setup.boundary_conditions
for bc_type in bc_settings.get_state():
    bc_group = getattr(bc_settings, bc_type)
    for bc_name in bc_group.get_state():
        print(f"{bc_name}  type={bc_type}")
```

Each `bc_name` is what you'll put before the `|` in `set_inputs` keys.

## List parameters under one BC

For each BC you care about, inspect its writable parameters:

```python
inlet = solver.settings.setup.boundary_conditions.velocity_inlet["cold-inlet"]
print(inlet.get_state())        # full dict; pick the one you want to vary
```

Velocity-inlet typically has `momentum.velocity_magnitude.value`,
`momentum.flow_direction`, `thermal.t.value` (with energy on), etc. The
dotted path you see is the `parameter_path` you'd put in the rich-dict
form of `set_inputs`:

```python
project.set_inputs({
    "cold-inlet|momentum > velocity_magnitude": {
        "range": (0.2, 0.6),
        "bc_type": "velocity-inlet",
        "parameter_path": "momentum.velocity_magnitude.value",
    },
})
```

## List surfaces and cell zones for outputs

```python
surfaces = solver.settings.results.surfaces.get_state()
print(list(surfaces.keys()))

cell_zones = solver.settings.results.cell_zones.get_state()  # if accessible
print(list(cell_zones.keys()))
```

These names go into `set_outputs(...)` `name` fields:

```python
project.set_outputs([
    {"name": "outlet", "category": "Surface",
     "field_variables": ["temperature", "velocity-magnitude"]},
    {"name": "fluid", "category": "Cell Zone",
     "field_variables": ["temperature"]},
])
```

## List report definitions

```python
report_defs = solver.settings.solution.report_definitions
print(list(report_defs.get_state().keys()))
```

For each report def name you can use:

```python
project.set_outputs([
    {"name": "outlet_avg_temp", "category": "Report Definition"},
])
```

## When you're done

```python
solver.exit()
```

You don't need to keep this exploration session running while you work
with cfdtwin — `set_inputs` / `set_outputs` are pure file writes and need
nothing live. Fluent will be re-launched by `project.connect_fluent()`
when it's time to run sims.
