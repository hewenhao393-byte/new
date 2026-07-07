from __future__ import annotations

import argparse
import json
from pathlib import Path

from pump_fault_app.export import create_export_output_dir, export_diagnosis_summary
from pump_fault_app.inference import FormalInferenceRequest, run_formal_inference
from pump_fault_app.reporting import build_diagnosis_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="水泵六分类故障诊断单文件推理入口")
    parser.add_argument("--file", required=True, help="待诊断振动文件路径")
    parser.add_argument("--sampling-rate", required=True, type=int, help="原始采样率 Hz")
    parser.add_argument("--rpm", required=True, type=float, help="转速 rpm")
    parser.add_argument("--signal-column", help="振动信号列名")
    parser.add_argument("--time-column", help="时间列名")
    parser.add_argument("--device-id", help="设备编号")
    parser.add_argument("--measurement-position", help="测点位置")
    parser.add_argument("--model-bundle", help="正式 BP bundle 路径")
    parser.add_argument("--output-dir", help="单文件结果导出根目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    inference_result = run_formal_inference(
        FormalInferenceRequest(
            file_path=Path(args.file),
            sampling_rate_hz=args.sampling_rate,
            rpm=args.rpm,
            signal_column=args.signal_column,
            time_column=args.time_column,
            device_id=args.device_id,
            measurement_position=args.measurement_position,
            model_bundle_path=Path(args.model_bundle) if args.model_bundle else None,
        )
    )
    summary = build_diagnosis_summary(inference_result)
    if args.output_dir:
        run_output_dir = create_export_output_dir(Path(args.output_dir), prefix="single_run")
        export_diagnosis_summary(summary, output_dir=run_output_dir)
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 0 if summary.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
