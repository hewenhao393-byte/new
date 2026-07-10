from __future__ import annotations

import argparse
import json
from pathlib import Path

from pump_fault_app.app.self_check import run_system_self_check
from pump_fault_app.services import AppSingleRunRequest, export_single_diagnosis_report, run_single_diagnosis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="水泵六分类故障诊断单文件推理入口")
    parser.add_argument("--file", help="待诊断振动文件路径")
    parser.add_argument("--sampling-rate", type=int, help="原始采样率 Hz")
    parser.add_argument("--rpm", type=float, help="转速 rpm")
    parser.add_argument("--signal-column", help="振动信号列名")
    parser.add_argument("--time-column", help="时间列名")
    parser.add_argument("--device-id", help="设备编号")
    parser.add_argument("--measurement-position", help="测点位置")
    parser.add_argument("--model-bundle", help="正式 BP bundle 路径")
    parser.add_argument("--output-dir", help="单文件结果导出根目录")
    parser.add_argument("--output-report", help="单文件 Word 报告输出路径 .docx")
    parser.add_argument("--self-check", action="store_true", help="运行系统自检并输出 JSON 结果")
    parser.add_argument("--demo-config", help="演示样本配置 JSON 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_check:
        payload = run_system_self_check(
            demo_config_path=args.demo_config,
            model_bundle_path=Path(args.model_bundle) if args.model_bundle else None,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["overall_status"] != "failed" else 1

    missing = [name for name in ("file", "sampling_rate", "rpm") if getattr(args, name) is None]
    if missing:
        parser.error(f"the following arguments are required unless --self-check is used: {', '.join('--' + item.replace('_', '-') for item in missing)}")

    result = run_single_diagnosis(
        AppSingleRunRequest(
            file_path=Path(args.file),
            sampling_rate_hz=args.sampling_rate,
            rpm=args.rpm,
            signal_column=args.signal_column,
            time_column=args.time_column,
            device_id=args.device_id,
            measurement_position=args.measurement_position,
            model_bundle_path=Path(args.model_bundle) if args.model_bundle else None,
            export_root=Path(args.output_dir) if args.output_dir else None,
        )
    )
    summary = result.summary
    if args.output_report:
        export_single_diagnosis_report(result, Path(args.output_report), format="docx")
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 0 if summary.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
