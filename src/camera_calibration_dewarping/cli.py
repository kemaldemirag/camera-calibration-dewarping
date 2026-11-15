"""Command-line entry point for the U01-T1 perspective baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .perspective import (
    PerspectiveInputError,
    evaluate_files,
    matches_k03_baseline,
    write_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate 7x5 target views on held-out points using four-corner homographies."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Directory containing chess_corners*.dat files, or one .dat file",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="chess_corners*.dat")
    parser.add_argument("--expected-views", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    inputs = sorted(args.input.glob(args.pattern)) if args.input.is_dir() else [args.input]
    try:
        evaluation = evaluate_files(inputs, expected_views=args.expected_views)
        csv_path, json_path = write_artifacts(evaluation, inputs, args.output_dir)
    except PerspectiveInputError as exc:
        parser.exit(2, f"error: {exc}\n")

    print(
        json.dumps(
            {
                "test_id": "U01-T1",
                "status": "TEKRARLANDI" if matches_k03_baseline(evaluation, inputs) else "BEKLİYOR",
                "perspective": asdict(evaluation.summary),
                "artifacts": [str(csv_path), str(json_path)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

