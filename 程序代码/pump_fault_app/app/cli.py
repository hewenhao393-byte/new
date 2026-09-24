from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pump_fault_app.app.self_check import run_system_self_check
from pump_fault_app.domain.diagnosis_models import ChannelInput, MultiChannelInferenceRequest
from pump_fault_app.services import AppSingleRunRequest, run_single_diagnosis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="水泵六分类 CatBoost43 多通道推理入口")
    parser.add_argument(
        "--channel",
        action="append",
        nargs="+",
        metavar="VALUE",
        help="可重复1至3次：CH3/CH4/CH5 文件 信号列 [时间列]",
    )
    parser.add_argument("--sampling-rate", type=int, help="原始采样率 Hz")
    parser.add_argument("--rpm", type=float, help="转速 rpm")
    parser.add_argument("--model-directory", help="CatBoost43 V3 部署模型目录")
    parser.add_argument("--self-check", action="store_true", help="运行系统自检并输出 JSON 结果")
    parser.add_argument("--demo-config", help="演示样本配置 JSON 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_check:
        payload = run_system_self_check(
            demo_config_path=args.demo_config,
            model_directory=Path(args.model_directory) if args.model_directory else None,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["overall_status"] != "failed" else 1

    missing = [name for name in ("channel", "sampling_rate", "rpm") if getattr(args, name) is None]
    if missing:
        parser.error(f"the following arguments are required unless --self-check is used: {', '.join('--' + item.replace('_', '-') for item in missing)}")

    channel_inputs = []
    for values in args.channel:
        if len(values) not in (3, 4):
            parser.error("--channel requires CHANNEL FILE SIGNAL_COLUMN [TIME_COLUMN]")
        channel_inputs.append(
            ChannelInput(values[0].upper(), Path(values[1]), values[2], values[3] if len(values) == 4 else None)
        )
    inference_request = MultiChannelInferenceRequest(
        tuple(channel_inputs),
        sampling_rate_hz=args.sampling_rate,
        rpm=args.rpm,
        model_directory=Path(args.model_directory) if args.model_directory else None,
    )
    result = run_single_diagnosis(AppSingleRunRequest(inference_request))
    print(json.dumps(asdict(result.inference_result), ensure_ascii=False, indent=2, default=str))
    return 0 if result.inference_result.status == "diagnosed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
