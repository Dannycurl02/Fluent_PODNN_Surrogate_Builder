# Discovering BC names from Fluent

`set_inputs(...)` is declarative — you tell cfdtwin which boundary
condition keys to vary, and cfdtwin doesn't validate them against Fluent
until run time. So you have to know the keys upfront.

This page shows how to list them from a live Fluent session if you don't.

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
