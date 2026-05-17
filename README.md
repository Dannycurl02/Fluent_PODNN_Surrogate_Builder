<p align="center">
  <img src="docs/assets/logo_wordmark.png" alt="CFDTwin" width="420"/>
</p>

Build neural-network surrogate models from ANSYS Fluent simulations.
Two ways to drive the same pipeline:

- **GUI** — wizard-based desktop app, click through Setup → DOE → Simulate → Train → Analyze.
- **API** — Python library, the same pipeline as ten method calls in a script.

**Docs:** https://uark-ned3.github.io/CFDTwin/

**GUI walkthrough (4-part video series):**

1. [Introduction and Case Assignment](https://youtu.be/aIredg034k8)
2. [Case Input and Output Setup](https://youtu.be/XN3sqVYiBiY)
3. [DOE Setup and Batch Simulation](https://youtu.be/fDZwWNaByOo)
4. [NN Training and Model Validation](https://youtu.be/V4EameyTECw)

## Install

Requires Python 3.10+ and a working ANSYS Fluent installation. Three paths
depending on what you want:

| Command                                | What you get                          | Best for                                        |
|----------------------------------------|---------------------------------------|-------------------------------------------------|
| `pip install cfdtwin`                  | API only (no Qt; ~80 MB smaller)      | Headless / HPC / scripted surrogate sweeps     |
| `pip install cfdtwin[gui]`             | API + desktop GUI                     | Engineers who want the wizard, no `git` needed |
| `git clone … && pip install -e .[dev]` | Everything + examples + tests + tools | Contributors, tutorial learners, tinkerers     |

```
pip install cfdtwin              # API only
pip install cfdtwin[gui]         # adds the wizard-based desktop app
git clone https://github.com/UARK-NED3/CFDTwin && cd CFDTwin && pip install -e .[dev]
```

If you `pip install cfdtwin` (no `[gui]`) and then run `cfdtwin-gui`, you'll
get a friendly hint to install the GUI extra — no opaque ImportError.

Runnable examples live in [`docs/examples/`](docs/examples/) — only available
in a clone, or browse them online: https://uark-ned3.github.io/CFDTwin/examples/

## Use the GUI

Three ways to launch, pick whichever you prefer:

```
cfdtwin-gui                          # any terminal, after pip install
python -m gui                        # from a repo clone, no install needed
double-click scripts/launch_gui.bat  # Windows, no terminal pops up
```

To put a launcher on your Desktop (Windows): right-click `scripts/launch_gui.bat`,
**Copy**, then right-click your Desktop and **Paste shortcut**. Optionally
right-click the shortcut → Properties → Change Icon → browse to
`gui/assets/logo_icon.png` for the CFDTwin logo.

On launch, select or create a project. The sidebar steps unlock as prerequisites are met:

1. **Setup** — pick `.cas` file, set Fluent options, define inputs and outputs
2. **DOE** — generate LHS/factorial samples
3. **Simulate** — batch-run Fluent with live progress
4. **Train** — transfer-list filter, live loss curves, per-output NN
5. **Analyze** — metrics dashboard, predictions, Fluent comparison with caching

## Use the API

Same pipeline from a Python script — useful for automation, parameter sweeps,
or building bigger surrogate libraries:

```python
import cfdtwin

project = cfdtwin.Project.create("./elbow_study", name="elbow_v1")
project.set_case_file("mixing_elbow.cas.h5")

project.set_inputs({
    "cold-inlet|momentum > velocity_magnitude": (0.2, 0.6),
    "hot-inlet|momentum > velocity_magnitude":  (0.4, 1.2),
})
project.set_outputs([
    {"name": "outlet", "category": "Surface",
     "field_variables": ["temperature"]},
])

project.generate_doe(n=20, method="lhs")
project.connect_fluent(precision="single")    # mixing_elbow is a single-precision case
project.run_simulations()
project.train(model_name="run1")

pred = project.predict("run1", {
    "cold-inlet|momentum > velocity_magnitude": 0.4,
    "hot-inlet|momentum > velocity_magnitude":  0.8,
})
print(pred.values.shape)
```

Runnable scripts live in [`docs/examples/`](docs/examples/) — quickstart,
full workflow, training tuning, and a "what BCs / parameters does my case
expose?" discovery script.

- [API reference](https://uark-ned3.github.io/CFDTwin/api/project/) — every method, every argument
- [Tutorials](https://uark-ned3.github.io/CFDTwin/tutorials/full_workflow/) — narrative walk-throughs
- [Quickstart](https://uark-ned3.github.io/CFDTwin/quickstart/) — smallest end-to-end script

## License

MIT — see [LICENSE](LICENSE).
