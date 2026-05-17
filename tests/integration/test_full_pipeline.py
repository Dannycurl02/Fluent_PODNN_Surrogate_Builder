"""End-to-end test against the mixing_elbow case.

Walks the whole Project API: create -> setup -> DOE -> Fluent -> sim ->
train -> predict. Asserts the result objects have sane shape at each step.

Slow (real Fluent). Manual run only.
"""

import pytest

import cfdtwin
from cfdtwin.results import (
    PredictionResult,
    SimulationResult,
    TrainingResult,
)


def test_full_pipeline(tmp_path, mixing_elbow_case):
    project = cfdtwin.Project.create(tmp_path / "elbow_run", name="elbow_test")
    assert project.name == "elbow_test"

    project.set_case_file(mixing_elbow_case)

    # Rich-dict form so set_inputs doesn't depend on a live Fluent connection.
    # Mixing elbow has cold-inlet and hot-inlet velocity-magnitude.
    project.set_inputs({
        "cold-inlet|momentum > velocity_magnitude": {
            "range": (0.2, 0.6),
            "bc_type": "velocity-inlet",
            "parameter_path": "momentum.velocity_magnitude.value",
        },
        "hot-inlet|momentum > velocity_magnitude": {
            "range": (0.4, 1.2),
            "bc_type": "velocity-inlet",
            "parameter_path": "momentum.velocity_magnitude.value",
        },
    })

    project.set_outputs([
        {"name": "outlet", "category": "Surface",
         "field_variables": ["temperature"]},
    ])

    n = project.generate_doe(n=4, method="lhs", seed=42)
    assert n == 4

    # mixing_elbow.cas.h5 is a single-precision case.
    project.connect_fluent(precision="single")
    sim = project.run_simulations(iterations=50)
    assert isinstance(sim, SimulationResult)
    assert sim.successful == 4, f"expected 4 successful sims, got {sim.summary()}"
    assert sim.failed == 0

    train = project.train(model_name="elbow_v1", epochs=200)
    assert isinstance(train, TrainingResult)
    assert train.n_models > 0
    best = train.best_model()
    assert best is not None
    # Mixing elbow temperature is well-behaved — expect strong fit on 4 samples
    # to itself (no held-out test guarantees, just smoke).
    assert best.test_metrics.r2 > 0.0

    pred = project.predict(
        train.model_name,
        {
            "cold-inlet|momentum > velocity_magnitude": 0.4,
            "hot-inlet|momentum > velocity_magnitude": 0.8,
        },
    )
    assert isinstance(pred, PredictionResult)
    assert pred.values.shape[0] == 1
    assert pred.values.shape[1] > 0

    project.disconnect_fluent()
