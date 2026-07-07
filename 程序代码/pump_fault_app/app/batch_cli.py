from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pump_fault_app.batch import (
    BatchInferenceRequest,
    load_batch_manifest,
    run_batch_inference,
    run_batch_inference_from_manifest,
)
from pump_fault_app.export import create_export_output_dir, export_batch_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="水泵六分类故障诊断批量推理入口")
    parser.add_argument("--files", nargs="+", help="待诊断振动文件路径列表")
    parser.add_argument("--manifest", help="批量元数据清单 CSV 路径")
    parser.add_argument("--sampling-rate", type=int, help="原始采样率 Hz")
    parser.add_argument("--rpm", type=float, help="转速 rpm")
    parser.add_argument("--signal-column", help="振动信号列名")
    parser.add_argument("--time-column", help="时间列名")
    parser.add_argument("--device-id", help="设备编号")
    parser.add_argument("--measurement-position", help="测点位置")
    parser.add_argument("--model-bundle", help="正式 BP bundle 路径")
    parser.add_argument("--output-dir", help="批量结果导出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    model_bundle_path = Path(args.model_bundle) if args.model_bundle else None
    if bool(args.files) == bool(args.manifest):
        parser.error("exactly one of --files or --manifest must be provided")

    if args.manifest:
        result = run_batch_inference_from_manifest(
            load_batch_manifest(Path(args.manifest)),
            model_bundle_path=model_bundle_path,
        )
    else:
        if args.sampling_rate is None or args.rpm is None:
            parser.error("--sampling-rate and --rpm are required when using --files")
        result = run_batch_inference(
            BatchInferenceRequest(
                file_paths=tuple(Path(path) for path in args.files),
                sampling_rate_hz=args.sampling_rate,
                rpm=args.rpm,
                signal_column=args.signal_column,
                time_column=args.time_column,
                device_id=args.device_id,
                measurement_position=args.measurement_position,
                model_bundle_path=model_bundle_path,
            )
        )

    if args.output_dir:
        run_output_dir = create_export_output_dir(Path(args.output_dir))
        export_batch_result(result, output_dir=run_output_dir)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
