<p align="center">
  <img src="docs/assets/logo_wordmark.png" alt="CFDTwin" width="420"/>
</p>

Build neural-network surrogate models from ANSYS Fluent simulations.
Two interfaces, one pipeline:

- **GUI** — wizard-based desktop app: Setup → DOE → Simulate → Train → Analyze.
- **API** — Python library exposing the same pipeline as `cfdtwin.Project` methods.

**Docs:** https://uark-ned3.github.io/CFDTwin/

**GUI walkthrough (4-part video series):**

1. [Introduction and Case Assignment](https://youtu.be/aIredg034k8)
2. [Case Input and Output Setup](https://youtu.be/XN3sqVYiBiY)
3. [DOE Setup and Batch Simulation](https://youtu.be/fDZwWNaByOo)
4. [NN Training and Model Validation](https://youtu.be/V4EameyTECw)

## Install

Requires Python 3.10+ and a working ANSYS Fluent installation. Three install paths:

| Command                                | What you get                              |
|----------------------------------------|-------------------------------------------|
| `pip install cfdtwin`                  | API only (no Qt)                          |
| `pip install cfdtwin[gui]`             | API and desktop GUI                       |
| `git clone … && pip install -e .[dev]` | API, GUI, examples, and tests             |

```
pip install cfdtwin              # API only
pip install cfdtwin[gui]         # adds the desktop GUI
git clone https://github.com/UARK-NED3/CFDTwin && cd CFDTwin && pip install -e .[dev]
```

Running `cfdtwin-gui` after an API-only install prints an instruction to
install the `[gui]` extra rather than raising ImportError.

## Use the GUI

Three ways to launch:

```
cfdtwin-gui                          # shell command, available after pip install
python -m gui                        # from a repo clone
double-click scripts/launch_gui.bat  # Windows; no terminal window
```

To create a Desktop launcher on Windows: right-click `scripts/launch_gui.bat`
→ **Copy**, then right-click the Desktop → **Paste shortcut**. To set the
CFDTwin icon, right-click the shortcut → Properties → Change Icon → browse
to `gui/assets/logo_icon.png`.

On launch, select or create a project. Sidebar steps unlock as their
prerequisites are met:

1. **Setup** — pick `.cas` file, set Fluent options, define inputs and outputs
2. **DOE** — generate LHS/factorial samples
3. **Simulate** — batch-run Fluent with live progress
4. **Train** — transfer-list filter, live loss curves, per-output NN
5. **Analyze** — metrics dashboard, predictions, Fluent comparison with caching

## Use the API

The same pipeline driven from a Python script, useful for automation and
parameter sweeps:

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

Runnable scripts live in [`docs/examples/`](docs/examples/): quickstart,
full workflow, training tuning, and a discovery script for unfamiliar case
files. Inlined online at
https://uark-ned3.github.io/CFDTwin/examples/.

- [API reference](https://uark-ned3.github.io/CFDTwin/api/project/)
- [Tutorials](https://uark-ned3.github.io/CFDTwin/tutorials/full_workflow/)
- [Quickstart](https://uark-ned3.github.io/CFDTwin/quickstart/)

## License

MIT. See [LICENSE](LICENSE).
