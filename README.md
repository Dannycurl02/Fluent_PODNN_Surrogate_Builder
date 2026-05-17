<p align="center">
  <img src="docs/assets/logo_wordmark.png" alt="CFDTwin" width="420"/>
</p>

Wizard-based desktop app for building neural-network surrogate models from ANSYS Fluent simulations. Load a `.cas` file, sample the parameter space, run the batch sim, train a POD+NN model, and validate — all in one GUI.

## Install

Requires Python 3.10+ and a working ANSYS Fluent installation.

```
pip install -e .[dev]   # drop [dev] if you won't run the test suite
```

## Run the GUI

Three ways, pick whichever you prefer:

```
cfdtwin-gui                          # any terminal, after pip install
python -m gui                        # from the repo root, no install needed
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
5. **Validate** — metrics dashboard, predictions, Fluent comparison with caching
