#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion_weight_study.pipeline import run_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run leakage-safe CH3/CH4/CH5 fusion weight study")
    for channel in (3, 4, 5):
        parser.add_argument(f"--ch{channel}", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-thread-count", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = run_study(args)
    if not args.dry_run:
        print(f"experiment published: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
