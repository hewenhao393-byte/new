from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pump_fault_app.app import cli
from pump_fault_app.app.self_check import run_system_self_check
from pump_fault_app.batch import BatchInferenceRequest, load_batch_manifest, run_batch_inference
from pump_fault_app.domain.diagnosis_models import (
    ChannelDiagnosisResult,
    ChannelInput,
    MultiChannelDiagnosisResult,
    MultiChannelInferenceRequest,
)
from pump_fault_app.fusion import fuse_valid_channels
from pump_fault_app.services import (
    AppBatchRunRequest,
    AppSingleRunRequest,
    run_batch_diagnosis,
    run_single_diagnosis,
)


PROBABILITIES = (0.60, 0.10, 0.10, 0.05, 0.10, 0.05)


def _result(channel: str = "CH3") -> MultiChannelDiagnosisResult:
    return fuse_valid_channels(
        (
            ChannelDiagnosisResult(
                channel=channel,
                status="valid",
                predicted_label="正常",
                class_probabilities=PROBABILITIES,
            ),
        )
    )


def _request(path: Path, channel: str = "CH3") -> MultiChannelInferenceRequest:
    return MultiChannelInferenceRequest((ChannelInput(channel, path, "signal"),), 12_000, 1500.0)


def test_single_and_batch_services_call_the_same_v3_entrypoint(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        "pump_fault_app.services.app_service.run_multichannel_inference",
        lambda item: calls.append(item) or _result(item.channels[0].channel),
    )
    request = _request(tmp_path / "a.csv")
    single = run_single_diagnosis(AppSingleRunRequest(request))
    batch = run_batch_diagnosis(AppBatchRunRequest(items=(request,)))
    assert single.inference_result.status == "diagnosed"
    assert batch.batch_result.diagnosed_count == 1
    assert calls == [request, request]


def test_batch_service_runs_only_multichannel_requests(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        "pump_fault_app.batch.service.run_multichannel_inference",
        lambda item: calls.append(item) or _result(item.channels[0].channel),
    )
    requests = (_request(tmp_path / "a.csv", "CH3"), _request(tmp_path / "b.csv", "CH4"))
    result = run_batch_inference(BatchInferenceRequest(requests))
    assert result.total_count == 2
    assert result.success_count == 2
    assert calls == list(requests)


def test_batch_manifest_encodes_one_to_three_channel_inputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "sampling_rate_hz": 12000,
                "rpm": 1500,
                "ch3_file": "a.csv",
                "ch3_signal_column": "s3",
                "ch4_file": "b.csv",
                "ch4_signal_column": "s4",
            }
        ]
    ).to_csv(manifest_path, index=False)
    item = load_batch_manifest(manifest_path).items[0].request
    assert tuple(channel.channel for channel in item.channels) == ("CH3", "CH4")
    assert not hasattr(item, "model_bundle_path")


def test_cli_builds_repeated_channel_request(monkeypatch, tmp_path: Path, capsys) -> None:
    captured = []
    monkeypatch.setattr(
        cli,
        "run_single_diagnosis",
        lambda request: captured.append(request) or SimpleNamespace(inference_result=_result()),
    )
    exit_code = cli.main(
        [
            "--channel", "CH3", str(tmp_path / "a.csv"), "s3",
            "--channel", "CH4", str(tmp_path / "b.csv"), "s4",
            "--sampling-rate", "12000",
            "--rpm", "1500",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "diagnosed"
    assert tuple(item.channel for item in captured[0].inference_request.channels) == ("CH3", "CH4")


def test_self_check_validates_three_deployment_models(tmp_path: Path) -> None:
    payload = run_system_self_check(demo_config_path=tmp_path / "missing.json")
    item = next(row for row in payload["items"] if row["check_name"] == "deployment_model_contract")
    assert item["status"] == "passed"
    assert "CH3, CH4, CH5" in item["message"]
