"""Return-value classes for the public Project API.

These are plain dataclasses — no logic beyond a few read-only summaries.
Methods on `Project` build them; users read them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Metrics:
    """Numeric quality metrics for one trained sub-model."""
    r2: float
    rmse: float
    mae: float

    @classmethod
    def from_dict(cls, d: dict) -> Metrics:
        return cls(
            r2=float(d.get("r2", 0.0)),
            rmse=float(d.get("rmse", 0.0)),
            mae=float(d.get("mae", 0.0)),
        )


@dataclass
class ModelInfo:
    """Static info about a single trained sub-model.

    A `Project.train(...)` call can produce several of these — one per
    (output_location, field_variable) pair the trainer consumed. Returned by
    `Project.model_info(name)` and as elements of `TrainingResult.models`.
    """
    model_name: str             # e.g. "mid-plane_temperature"
    output_type: str            # "1D" | "2D" | "3D"
    field_name: str             # e.g. "temperature"
    location: str               # e.g. "mid-plane"
    npz_key: str                # "<location>|<field>"
    n_points: int               # spatial points (1 for scalars)
    has_pod: bool               # POD reducer used?
    preset: str                 # NN preset name ("1d" / "2d" / "3d")
    model_architecture: str     # "SurrogateNN" | "POD+NN"
    train_metrics: Metrics
    test_metrics: Metrics
    trained_date: str           # ISO datetime string from training time
    dataset_version: int
    n_train_samples: int
    n_test_samples: int
    n_modes: int | None = None              # POD modes; None for non-POD
    variance_explained: float | None = None # POD only

    @classmethod
    def from_metadata(cls, meta: dict) -> ModelInfo:
        """Build from the JSON dict the trainer writes to *_metadata.json."""
        return cls(
            model_name=meta["model_name"],
            output_type=meta["output_type"],
            field_name=meta.get("field_name", ""),
            location=meta.get("location", ""),
            npz_key=meta.get("npz_key", ""),
            n_points=int(meta.get("n_points", 1)),
            has_pod=bool(meta.get("has_pod", False)),
            preset=meta.get("preset", ""),
            model_architecture=meta.get("model_architecture", "SurrogateNN"),
            train_metrics=Metrics.from_dict(meta.get("train_metrics", {})),
            test_metrics=Metrics.from_dict(meta.get("test_metrics", {})),
            trained_date=meta.get("trained_date", ""),
            dataset_version=int(meta.get("dataset_version", 0)),
            n_train_samples=int(meta.get("n_train_samples", 0)),
            n_test_samples=int(meta.get("n_test_samples", 0)),
            n_modes=meta.get("n_modes"),
            variance_explained=meta.get("variance_explained"),
        )


@dataclass
class SimulationResult:
    """Summary of a `Project.run_simulations(...)` call."""
    successful: int
    failed: int
    total: int
    elapsed: float                   # seconds
    stopped_reason: str | None = None
    failed_ids: list[int] = field(default_factory=list)

    def summary(self) -> str:
        m, s = divmod(int(self.elapsed), 60)
        line = f"{self.successful}/{self.total} simulations succeeded ({m}m {s}s)"
        if self.failed:
            line += f"; {self.failed} failed (ids: {self.failed_ids})"
        if self.stopped_reason:
            line += f"; stopped: {self.stopped_reason}"
        return line


@dataclass
class TrainingResult:
    """Summary of a `Project.train(...)` call."""
    model_name: str                  # final folder name (may be auto-suffixed)
    models: list[ModelInfo]
    n_train_samples: int
    n_test_samples: int
    test_split: float
    epochs: int                      # max epochs requested (NOT actual)
    test_indices: list[int]
    train_indices: list[int]
    models_dir: Path                 # absolute path to project/models/<model_name>

    @property
    def n_models(self) -> int:
        return len(self.models)

    def best_model(self) -> ModelInfo | None:
        """Return the sub-model with the highest test R². None if no models."""
        if not self.models:
            return None
        return max(self.models, key=lambda m: m.test_metrics.r2)

    def summary(self) -> str:
        lines = [
            f"Training run: {self.model_name}",
            f"  {self.n_models} model(s), "
            f"{self.n_train_samples} train / {self.n_test_samples} test samples",
        ]
        for m in self.models:
            lines.append(
                f"    {m.model_name}: "
                f"test R² {m.test_metrics.r2:.4f}, RMSE {m.test_metrics.rmse:.4g}"
            )
        return "\n".join(lines)


@dataclass
class PredictionResult:
    """Output of `Project.predict(...)`.

    `values` is shape `(n_samples, n_points)` — `n_points == 1` for 1D scalars,
    larger for 2D/3D field models. `coordinates` is `(n_points, 3)` for fields
    or `None` for scalars.
    """
    model_name: str
    values: np.ndarray
    coordinates: np.ndarray | None
    metadata: ModelInfo
    inputs: list[dict[str, float]]   # the param dicts that were predicted

    @property
    def is_scalar(self) -> bool:
        return self.metadata.output_type == "1D"

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    @property
    def n_points(self) -> int:
        return self.values.shape[1] if self.values.ndim > 1 else 1
