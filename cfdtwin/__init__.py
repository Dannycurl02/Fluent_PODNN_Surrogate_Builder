"""cfdtwin — neural-network surrogate models for ANSYS Fluent simulations.

The public surface is intentionally small. Build a Project, then call its
methods. Result classes are exported for type hints and isinstance checks.

    import cfdtwin

    project = cfdtwin.Project.open("path/to/project")
    project.set_inputs({"inlet|velocity": (0.1, 1.0)})
    project.set_outputs([{"name": "outlet_temp", "category": "Report Definition"}])
    project.generate_doe(n=50)
    project.run_simulations()
    result = project.train(model_name="my_run")
    print(result.summary())

Lower-level modules (cfdtwin.doe, cfdtwin.training, cfdtwin.simulation, ...)
are public for power users who want fine-grained access.
"""

from cfdtwin.project import Project
from cfdtwin.results import (
    Metrics,
    ModelInfo,
    PredictionResult,
    SimulationResult,
    TrainingResult,
)

__version__ = "0.2.3.dev0"

__all__ = [
    "Project",
    "Metrics",
    "ModelInfo",
    "PredictionResult",
    "SimulationResult",
    "TrainingResult",
    "__version__",
]
