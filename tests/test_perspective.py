from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
TEST_TEMP = REPOSITORY / ".test-tmp"
TEST_TEMP.mkdir(exist_ok=True)

from camera_calibration_dewarping.perspective import (  # noqa: E402
    PerspectiveEvaluation,
    PerspectiveInputError,
    evaluate_files,
    evaluate_points,
    parse_opencv_points,
    write_artifacts,
)


def project_grid(transform: np.ndarray) -> np.ndarray:
    x_values, y_values = np.meshgrid(np.arange(5), np.arange(7))
    grid = np.column_stack((x_values.reshape(-1), y_values.reshape(-1))).astype(float)
    homogeneous = np.column_stack((grid, np.ones(35))) @ transform.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def opencv_record(points: np.ndarray, *, rows: int = 7, columns: int = 5) -> str:
    values = ", ".join(format(float(value), ".17g") for value in points.reshape(-1))
    return f"%YAML:1.0\nrows: {rows}\ncols: {columns}\ndata: [ {values} ]\n"


class PerspectiveUnitTests(unittest.TestCase):
    def test_exact_projective_grid_has_only_roundoff_error(self) -> None:
        transform = np.asarray(
            [[2.0, 0.15, 13.0], [0.25, 3.0, -7.0], [0.002, -0.001, 1.0]]
        )
        metrics, errors = evaluate_points(project_grid(transform), view=4)

        self.assertEqual(metrics.view, 4)
        self.assertEqual(metrics.n_test, 31)
        self.assertLess(metrics.rms_px, 1e-11)
        self.assertLess(max(errors), 1e-10)

    def test_control_point_error_does_not_change_corner_fit(self) -> None:
        points = project_grid(np.eye(3))
        points[12] += np.asarray([3.0, 4.0])

        metrics, errors = evaluate_points(points)

        self.assertEqual(metrics.n_test, 31)
        self.assertAlmostEqual(metrics.max_px, 5.0)
        self.assertAlmostEqual(metrics.rms_px, 5.0 / np.sqrt(31))
        self.assertEqual(sum(error > 0.0 for error in errors), 1)

    def test_parser_and_artifact_writer_preserve_provenance(self) -> None:
        points = project_grid(np.eye(3))
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            root = Path(directory)
            source = root / "chess_corners1.dat"
            source.write_text(opencv_record(points), encoding="utf-8")

            parsed = parse_opencv_points(source)
            evaluation = evaluate_files([source], expected_views=1)
            csv_path, json_path = write_artifacts(evaluation, [source], root / "out")
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        np.testing.assert_allclose(parsed, points)
        self.assertTrue(csv_path.name == "perspektif.csv")
        self.assertEqual(payload["test_id"], "U01-T1")
        self.assertEqual(payload["status"], "TEKRARLANDI")
        self.assertEqual(len(payload["inputs"][0]["sha256"]), 64)
        self.assertIn("millimetre accuracy", payload["claim_scope"]["not_measured"])

    def test_summary_combines_all_held_out_errors(self) -> None:
        evaluation = PerspectiveEvaluation(
            view_metrics=(),
            errors_px=(3.0, 4.0),
        )
        summary = evaluation.summary
        self.assertEqual(summary.n_test, 2)
        self.assertAlmostEqual(summary.rms_px, np.sqrt(12.5))
        self.assertEqual(summary.max_px, 4.0)


class PerspectiveNegativeTests(unittest.TestCase):
    def test_rejects_wrong_grid_shape(self) -> None:
        with self.assertRaisesRegex(PerspectiveInputError, "35"):
            evaluate_points(np.zeros((34, 2)))

    def test_rejects_non_finite_coordinate(self) -> None:
        points = project_grid(np.eye(3))
        points[10, 0] = np.nan
        with self.assertRaisesRegex(PerspectiveInputError, "finite"):
            evaluate_points(points)

    def test_rejects_degenerate_corner_geometry(self) -> None:
        points = np.zeros((35, 2), dtype=float)
        with self.assertRaisesRegex(PerspectiveInputError, "singular|define"):
            evaluate_points(points)

    def test_parser_rejects_wrong_dimensions(self) -> None:
        points = project_grid(np.eye(3))
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            source = Path(directory) / "wrong.dat"
            source.write_text(opencv_record(points, rows=5, columns=7), encoding="utf-8")
            with self.assertRaisesRegex(PerspectiveInputError, "7x5"):
                parse_opencv_points(source)

    def test_requires_declared_view_count(self) -> None:
        with self.assertRaisesRegex(PerspectiveInputError, "Expected 7 views"):
            evaluate_files([], expected_views=7)


if __name__ == "__main__":
    unittest.main()

