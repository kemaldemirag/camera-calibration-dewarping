"""Evaluate a planar perspective transform on held-out grid points.

The U01-T1 baseline deliberately measures only pixel-domain perspective
consistency.  It does not estimate lens distortion, physical scale, or a
customer-specific acceptance result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

GRID_ROWS = 7
GRID_COLUMNS = 5
CORNER_INDICES = (0, 4, 30, 34)


class PerspectiveInputError(ValueError):
    """Raised when an input cannot support the declared U01-T1 procedure."""


@dataclass(frozen=True)
class ViewMetrics:
    """Held-out error metrics for one target view."""

    view: int
    n_test: int
    rms_px: float
    max_px: float


@dataclass(frozen=True)
class PerspectiveSummary:
    """Aggregate metrics across all held-out points."""

    views: int
    n_test: int
    rms_px: float
    p95_px: float
    max_px: float


@dataclass(frozen=True)
class PerspectiveEvaluation:
    """Complete deterministic result for a set of views."""

    view_metrics: tuple[ViewMetrics, ...]
    errors_px: tuple[float, ...]

    @property
    def summary(self) -> PerspectiveSummary:
        errors = np.asarray(self.errors_px, dtype=np.float64)
        if errors.size == 0:
            raise PerspectiveInputError("At least one held-out point is required")
        return PerspectiveSummary(
            views=len(self.view_metrics),
            n_test=int(errors.size),
            rms_px=float(np.sqrt(np.mean(np.square(errors)))),
            p95_px=float(np.percentile(errors, 95)),
            max_px=float(np.max(errors)),
        )


def _grid_points(rows: int = GRID_ROWS, columns: int = GRID_COLUMNS) -> NDArray[np.float64]:
    x, y = np.meshgrid(np.arange(columns), np.arange(rows))
    return np.column_stack((x.reshape(-1), y.reshape(-1))).astype(np.float64)


def _require_points(points: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(points, dtype=np.float64)
    expected_shape = (GRID_ROWS * GRID_COLUMNS, 2)
    if values.shape != expected_shape:
        raise PerspectiveInputError(
            f"Expected {expected_shape[0]} two-dimensional points; got shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise PerspectiveInputError("Point coordinates must all be finite")
    return values


def parse_opencv_points(path: str | Path) -> NDArray[np.float64]:
    """Read one OpenCV YAML-like 7-by-5 corner-coordinate record."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PerspectiveInputError(f"Cannot read {source}: {exc}") from exc

    rows_match = re.search(r"\brows\s*:\s*(\d+)\b", text)
    columns_match = re.search(r"\bcols\s*:\s*(\d+)\b", text)
    data_match = re.search(r"\bdata\s*:\s*\[([^\]]+)\]", text, re.DOTALL)
    if not rows_match or not columns_match or not data_match:
        raise PerspectiveInputError(f"Missing rows, cols, or data field in {source.name}")

    rows = int(rows_match.group(1))
    columns = int(columns_match.group(1))
    if (rows, columns) != (GRID_ROWS, GRID_COLUMNS):
        raise PerspectiveInputError(
            f"Expected a {GRID_ROWS}x{GRID_COLUMNS} grid in {source.name}; got {rows}x{columns}"
        )

    tokens = [part.strip() for part in data_match.group(1).split(",")]
    if len(tokens) != GRID_ROWS * GRID_COLUMNS * 2:
        raise PerspectiveInputError(
            f"Expected {GRID_ROWS * GRID_COLUMNS * 2} coordinate values in {source.name}; "
            f"got {len(tokens)}"
        )
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise PerspectiveInputError(f"A coordinate is not numeric in {source.name}") from exc
    return _require_points(values.reshape(GRID_ROWS * GRID_COLUMNS, 2))


def estimate_homography(
    source_points: NDArray[np.float64], target_points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solve the eight free homography parameters from four point pairs."""

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != (4, 2) or target.shape != (4, 2):
        raise PerspectiveInputError("Exactly four two-dimensional point pairs are required")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise PerspectiveInputError("Homography points must all be finite")

    matrix: list[list[float]] = []
    right_hand_side: list[float] = []
    for (x_value, y_value), (u_value, v_value) in zip(source, target, strict=True):
        matrix.extend(
            [
                [x_value, y_value, 1.0, 0.0, 0.0, 0.0, -u_value * x_value, -u_value * y_value],
                [0.0, 0.0, 0.0, x_value, y_value, 1.0, -v_value * x_value, -v_value * y_value],
            ]
        )
        right_hand_side.extend([u_value, v_value])

    coefficients = np.asarray(matrix, dtype=np.float64)
    if not math.isfinite(float(np.linalg.cond(coefficients))) or np.linalg.cond(coefficients) > 1e14:
        raise PerspectiveInputError("Corner geometry is singular or numerically unstable")
    try:
        solution = np.linalg.solve(coefficients, np.asarray(right_hand_side, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise PerspectiveInputError("Corner geometry does not define a homography") from exc
    return np.append(solution, 1.0).reshape(3, 3)


def project_points(
    points: NDArray[np.float64], homography: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Project planar points and reject points at infinity."""

    values = np.asarray(points, dtype=np.float64)
    transform = np.asarray(homography, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise PerspectiveInputError("Projection input must have shape (N, 2)")
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise PerspectiveInputError("Homography must be a finite 3x3 matrix")

    homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64))) @ transform.T
    denominator = homogeneous[:, 2]
    if np.any(np.isclose(denominator, 0.0, atol=1e-12)):
        raise PerspectiveInputError("Homography projects at least one point to infinity")
    projected = homogeneous[:, :2] / denominator[:, None]
    if not np.all(np.isfinite(projected)):
        raise PerspectiveInputError("Homography projection produced a non-finite coordinate")
    return projected


def evaluate_points(points: NDArray[np.float64], *, view: int = 1) -> tuple[ViewMetrics, tuple[float, ...]]:
    """Evaluate one 7-by-5 observation using four corners for fitting."""

    observed = _require_points(points)
    grid = _grid_points()
    corner_indices = np.asarray(CORNER_INDICES, dtype=np.int64)
    control_mask = np.ones(len(grid), dtype=bool)
    control_mask[corner_indices] = False

    homography = estimate_homography(grid[corner_indices], observed[corner_indices])
    predicted = project_points(grid, homography)
    corner_error = np.max(np.abs(predicted[corner_indices] - observed[corner_indices]))
    if corner_error >= 1e-8:
        raise PerspectiveInputError(
            f"Solved homography does not reproduce its corners (max error {corner_error:.3g} px)"
        )

    errors = np.linalg.norm(predicted[control_mask] - observed[control_mask], axis=1)
    metrics = ViewMetrics(
        view=view,
        n_test=int(errors.size),
        rms_px=float(np.sqrt(np.mean(np.square(errors)))),
        max_px=float(np.max(errors)),
    )
    return metrics, tuple(float(value) for value in errors)


def evaluate_files(
    paths: Iterable[str | Path], *, expected_views: int | None = 7
) -> PerspectiveEvaluation:
    """Evaluate input files in caller order and combine their held-out errors."""

    sources = tuple(Path(path) for path in paths)
    if expected_views is not None and len(sources) != expected_views:
        raise PerspectiveInputError(f"Expected {expected_views} views; got {len(sources)}")
    if not sources:
        raise PerspectiveInputError("At least one view is required")

    metrics: list[ViewMetrics] = []
    errors: list[float] = []
    for view, source in enumerate(sources, start=1):
        result, view_errors = evaluate_points(parse_opencv_points(source), view=view)
        metrics.append(result)
        errors.extend(view_errors)
    return PerspectiveEvaluation(tuple(metrics), tuple(errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifacts(
    evaluation: PerspectiveEvaluation,
    inputs: Sequence[str | Path],
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write the U01-T1 table and traceable summary record."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "perspektif.csv"
    json_path = destination / "ozet.json"

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("view", "n_test", "rms_px", "max_px"))
        writer.writeheader()
        writer.writerows(asdict(row) for row in evaluation.view_metrics)

    source_paths = [Path(path) for path in inputs]
    payload = {
        "schema_version": "1.0",
        "test_id": "U01-T1",
        "status": "TEKRARLANDI",
        "procedure": {
            "grid": {"rows": GRID_ROWS, "columns": GRID_COLUMNS},
            "fit_point_indices": list(CORNER_INDICES),
            "held_out_points_per_view": GRID_ROWS * GRID_COLUMNS - len(CORNER_INDICES),
        },
        "inputs": [
            {"name": path.name, "sha256": _sha256(path)} for path in source_paths
        ],
        "perspective": asdict(evaluation.summary),
        "views": [asdict(row) for row in evaluation.view_metrics],
        "claim_scope": {
            "measured": "pixel-domain perspective consistency on held-out target points",
            "not_measured": [
                "lens distortion profile",
                "millimetre accuracy",
                "object tracking performance",
                "customer-specific tolerance compliance",
            ],
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path, json_path

