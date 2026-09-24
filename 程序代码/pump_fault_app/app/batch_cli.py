from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pump_fault_app.batch import load_batch_manifest, run_batch_inference_from_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="水泵六分类故障诊断批量推理入口")
    parser.add_argument("--manifest", required=True, help="V3批量多通道清单 CSV 路径")
    parser.add_argument("--model-directory", help="CatBoost43 V3 部署模型目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_batch_inference_from_manifest(
        load_batch_manifest(Path(args.manifest)),
        model_directory=Path(args.model_directory) if args.model_directory else None,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0 if result.failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
