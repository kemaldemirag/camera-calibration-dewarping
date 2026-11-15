from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "src"))

from camera_calibration_dewarping.perspective import evaluate_files  # noqa: E402


class PerspectiveGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = (
            WORKSPACE
            / "003-Proje Kayıtları"
            / "deneyler"
            / "15.11.2026"
            / "source-package"
        )
        cls.data = cls.package / "veri"
        cls.inputs = [cls.data / f"chess_corners{index}.dat" for index in range(1, 8)]
        if not all(path.is_file() for path in cls.inputs):
            raise unittest.SkipTest("Verified K03 OpenCV inputs have not been downloaded")

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
        golden_csv = self.package / "sonuclar" / "perspektif.csv"
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
        archived = json.loads((self.package / "sonuclar" / "ozet.json").read_text(encoding="utf-8"))
        self.assertEqual(result.summary.n_test, archived["perspective"]["n_test"])
        self.assertAlmostEqual(result.summary.rms_px, archived["perspective"]["rms_px"], places=12)


if __name__ == "__main__":
    unittest.main()

