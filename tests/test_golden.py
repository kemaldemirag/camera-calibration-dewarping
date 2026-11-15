from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "src"))
TEST_TEMP = REPOSITORY / ".test-tmp"
TEST_TEMP.mkdir(exist_ok=True)

from camera_calibration_dewarping.perspective import (  # noqa: E402
    evaluate_files,
    matches_k03_baseline,
    write_artifacts,
)


class PerspectiveGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = WORKSPACE / "003-Proje Kayıtları"
        archived_package = records / "deneyler" / "15.11.2026" / "source-package"
        candidates = sorted((records / "runs").glob("**/veri"))
        candidates.append(archived_package / "veri")
        cls.data = next(
            (
                candidate
                for candidate in candidates
                if all((candidate / f"chess_corners{index}.dat").is_file() for index in range(1, 8))
            ),
            None,
        )
        if cls.data is None:
            raise unittest.SkipTest("Verified K03 OpenCV inputs have not been downloaded")
        cls.inputs = [cls.data / f"chess_corners{index}.dat" for index in range(1, 8)]
        sibling_results = cls.data.parent / "sonuclar"
        cls.results = sibling_results if sibling_results.is_dir() else archived_package / "sonuclar"

    def test_reproduces_reported_aggregate_metrics(self) -> None:
        result = evaluate_files(self.inputs)
        summary = result.summary

        self.assertEqual(summary.views, 7)
        self.assertEqual(summary.n_test, 217)
        self.assertAlmostEqual(summary.rms_px, 0.5811008373426997, places=12)
        self.assertAlmostEqual(summary.p95_px, 1.334420354253144, places=12)
        self.assertAlmostEqual(summary.max_px, 1.7804176308511281, places=12)

    def test_reproduces_each_archived_view(self) -> None:
        result = evaluate_files(self.inputs)
        golden_csv = self.results / "perspektif.csv"
        with golden_csv.open(encoding="utf-8", newline="") as stream:
            expected = list(csv.DictReader(stream))

        self.assertEqual(len(result.view_metrics), len(expected))
        for actual, archived in zip(result.view_metrics, expected, strict=True):
            self.assertEqual(actual.view, int(archived["view"]))
            self.assertEqual(actual.n_test, int(archived["n_test"]))
            self.assertAlmostEqual(actual.rms_px, float(archived["rms_px"]), places=12)
            self.assertAlmostEqual(actual.max_px, float(archived["max_px"]), places=12)

    def test_matches_archived_summary_but_keeps_claim_boundary(self) -> None:
        result = evaluate_files(self.inputs)
        archived = json.loads((self.results / "ozet.json").read_text(encoding="utf-8"))
        self.assertEqual(result.summary.n_test, archived["perspective"]["n_test"])
        self.assertAlmostEqual(result.summary.rms_px, archived["perspective"]["rms_px"], places=12)

    def test_only_locked_inputs_receive_reproduced_status(self) -> None:
        result = evaluate_files(self.inputs)
        self.assertTrue(matches_k03_baseline(result, self.inputs))
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            _, summary_path = write_artifacts(result, self.inputs, directory)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "TEKRARLANDI")
        self.assertTrue(payload["baseline"]["matched"])

    def test_reversed_inputs_cannot_claim_reproduced_status(self) -> None:
        reversed_inputs = list(reversed(self.inputs))
        result = evaluate_files(reversed_inputs)

        self.assertAlmostEqual(result.summary.rms_px, 0.5811008373426997, places=12)
        self.assertFalse(matches_k03_baseline(result, reversed_inputs))
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            _, summary_path = write_artifacts(result, reversed_inputs, directory)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "BEKLİYOR")
        self.assertFalse(payload["baseline"]["matched"])


if __name__ == "__main__":
    unittest.main()

