from __future__ import annotations

from pump_fault_app.app.batch_cli import build_parser


def test_batch_cli_accepts_v3_manifest_and_model_directory() -> None:
    help_text = build_parser().format_help()
    assert "--manifest" in help_text
    assert "--model-directory" in help_text
    assert "model-bundle" not in help_text
