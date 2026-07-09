from __future__ import annotations

import json
from pathlib import Path

from pump_fault_app.app.cli import main
from pump_fault_app.app.self_check import load_demo_sample_config, run_system_self_check
from tests.test_pump_fault_app_inference import _write_fake_bundle, _write_signal_csv


def test_load_demo_sample_config_reads_existing_json(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.csv"
    sample_path.write_text("time,通道4\n0,0.1\n", encoding="utf-8")
    config_path = tmp_path / "demo_samples.json"
    config_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_name": "demo",
                        "file_path": str(sample_path),
                        "sampling_rate": 12000,
                        "rpm": 1450.0,
                        "signal_column": "通道4",
                        "expected_label": "正常",
                        "description": "演示样本",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = load_demo_sample_config(config_path)

    assert payload["samples"][0]["sample_name"] == "demo"
    assert payload["samples"][0]["file_path"] == str(sample_path)


def test_run_system_self_check_returns_structured_results_and_warns_for_missing_demo_config(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    payload = run_system_self_check(demo_config_path=tmp_path / "missing_demo.json", model_bundle_path=bundle_path)

    assert "items" in payload
    assert any(item["check_name"] == "demo_sample_config" for item in payload["items"])
    demo_item = next(item for item in payload["items"] if item["check_name"] == "demo_sample_config")
    assert demo_item["status"] == "warning"


def test_run_system_self_check_reports_missing_bundle_as_failed(tmp_path: Path) -> None:
    payload = run_system_self_check(demo_config_path=tmp_path / "missing_demo.json", model_bundle_path=tmp_path / "missing_bundle.joblib")

    bundle_item = next(item for item in payload["items"] if item["check_name"] == "model_bundle_exists")
    assert bundle_item["status"] == "failed"
    assert "missing" in bundle_item["message"].lower()


def test_run_system_self_check_can_run_minimal_demo_diagnosis(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)
    signal_path = tmp_path / "demo.csv"
    _write_signal_csv(signal_path, [0.1] * 4800)
    config_path = tmp_path / "demo_samples.json"
    config_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_name": "demo",
                        "file_path": str(signal_path),
                        "sampling_rate": 12000,
                        "rpm": 1450.0,
                        "signal_column": "通道4",
                        "description": "最小演示",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = run_system_self_check(demo_config_path=config_path, model_bundle_path=bundle_path)

    diagnosis_item = next(item for item in payload["items"] if item["check_name"] == "demo_sample_minimal_diagnosis")
    assert diagnosis_item["status"] in {"passed", "warning"}


def test_cli_self_check_can_run(tmp_path: Path, capsys) -> None:
    bundle_path = tmp_path / "bp_bundle.joblib"
    _write_fake_bundle(bundle_path)

    exit_code = main(["--self-check", "--demo-config", str(tmp_path / "missing_demo.json"), "--model-bundle", str(bundle_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert "items" in payload


def test_usage_and_acceptance_docs_exist() -> None:
    usage_doc = Path("/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_usage.md")
    acceptance_doc = Path("/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_acceptance.md")

    assert usage_doc.exists()
    assert acceptance_doc.exists()


def test_demo_and_release_docs_and_scripts_exist() -> None:
    base = Path("/Users/hewenhao/Documents/特征提取/程序代码")
    assert (base / "docs" / "pump_fault_app_demo_guide.md").exists()
    assert (base / "docs" / "pump_fault_app_release_checklist.md").exists()
    assert (base / "scripts" / "run_pump_fault_demo.sh").exists()
    assert (base / "scripts" / "run_pump_fault_self_check.sh").exists()


def test_demo_package_materials_exist() -> None:
    base = Path("/Users/hewenhao/Documents/特征提取/程序代码")
    assert (base / "configs" / "demo_package.example.json").exists()
    assert (base / "configs" / "demo_package.json").exists()
    assert (base / "configs" / "demo_samples.json").exists()
    assert (base / "scripts" / "check_pump_fault_demo_package.sh").exists()
    assert (base / "scripts" / "init_pump_fault_demo_package.sh").exists()
    assert (base / "scripts" / "find_clean_demo_sample.py").exists()
    assert (base / "docs" / "pump_fault_app_screenshot_checklist.md").exists()
    assert (base / "docs" / "pump_fault_app_thesis_figures.md").exists()
    assert (base / "docs" / "pump_fault_app_defense_checklist.md").exists()
    assert (base / "docs" / "pump_fault_app_demo_sample_selection.md").exists()


def test_demo_samples_json_contains_primary_and_optional_backups() -> None:
    config_path = Path("/Users/hewenhao/Documents/特征提取/程序代码/configs/demo_samples.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["primary_demo_sample"]["sample_name"]
    assert payload["primary_demo_sample"]["file_path"]
    assert payload["primary_demo_sample"]["expected_label"]
    assert "backup_demo_samples" in payload
    assert isinstance(payload["backup_demo_samples"], list)
    assert payload["samples"]


def test_demo_package_example_contains_required_fields() -> None:
    config_path = Path("/Users/hewenhao/Documents/特征提取/程序代码/configs/demo_package.example.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["demo_name"]
    assert payload["description"]
    assert payload["model_bundle_path"]
    assert payload["demo_sample_config_path"]
    assert payload["expected_ui_entry"]
    assert payload["expected_cli_entry"]
    assert payload["expected_output_dir"]
    assert payload["expected_report_name"]
    assert payload["required_files"]
    assert payload["optional_files"] is not None
    assert payload["screenshot_output_dir"]


def test_demo_and_self_check_scripts_call_existing_entrypoints() -> None:
    base = Path("/Users/hewenhao/Documents/特征提取/程序代码")
    demo_text = (base / "scripts" / "run_pump_fault_demo.sh").read_text(encoding="utf-8")
    self_check_text = (base / "scripts" / "run_pump_fault_self_check.sh").read_text(encoding="utf-8")
    init_text = (base / "scripts" / "init_pump_fault_demo_package.sh").read_text(encoding="utf-8")
    finder_text = (base / "scripts" / "find_clean_demo_sample.py").read_text(encoding="utf-8")

    assert "pump_fault_app/ui/streamlit_app.py" in demo_text
    assert "--self-check" in demo_text
    assert "check_pump_fault_demo_package.sh" in demo_text
    assert "--self-check" in self_check_text
    assert "pump_fault_app.app.cli" in self_check_text
    assert "demo_samples.example.json" in init_text
    assert "demo_package.example.json" in init_text
    assert "argparse" in finder_text
    assert "__main__" in finder_text


def test_demo_package_check_script_reads_config_and_checks_required_inputs() -> None:
    script_path = Path("/Users/hewenhao/Documents/特征提取/程序代码/scripts/check_pump_fault_demo_package.sh")
    text = script_path.read_text(encoding="utf-8")

    assert "CONFIG_PATH" in text
    assert "required_files" in text
    assert "optional_files" in text
    assert "demo_sample_config_path" in text
    assert "file_path" in text
    assert "screenshot_output_dir" in text


def test_readme_mentions_demo_init_and_recommended_order() -> None:
    readme = Path("/Users/hewenhao/Documents/特征提取/程序代码/README.md").read_text(encoding="utf-8")

    assert "init_pump_fault_demo_package.sh" in readme
    assert "./scripts/run_pump_fault_self_check.sh" in readme
    assert "./scripts/check_pump_fault_demo_package.sh configs/demo_package.json" in readme
    assert "./scripts/run_pump_fault_demo.sh" in readme


def test_readme_mentions_demo_self_check_and_old_flow_warning() -> None:
    readme = Path("/Users/hewenhao/Documents/特征提取/程序代码/README.md").read_text(encoding="utf-8")

    assert "一键演示" in readme
    assert "self-check" in readme
    assert "demo_samples.example.json" in readme
    assert "demo_package.example.json" in readme
    assert "check_pump_fault_demo_package.sh" in readme
    assert "当前主演示样本" in readme
    assert "pump_fault_app_screenshot_checklist.md" in readme
    assert "pump_fault_app_defense_checklist.md" in readme
    assert "旧训练脚本和探索脚本只作为实验追溯资料，不作为软件入口" in readme


def test_docs_preserve_formal_parameters() -> None:
    for path in [
        Path("/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_usage.md"),
        Path("/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_release_checklist.md"),
        Path("/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_demo_guide.md"),
        Path("/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_thesis_figures.md"),
        Path("/Users/hewenhao/Documents/特征提取/程序代码/docs/pump_fault_app_defense_checklist.md"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "12000 Hz" in text
        assert "10~5000 Hz" in text or "10～5000 Hz" in text
        assert "2400" in text and "1200" in text
        assert "db6" in text
        assert "2000~5000 Hz" in text or "2000～5000 Hz" in text
