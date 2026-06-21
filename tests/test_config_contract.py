from __future__ import annotations

from config import ModelingConfig, PipelineConfig


def test_pipeline_config_uses_channel4_output_contract(tmp_path) -> None:
    config = PipelineConfig(output_root=tmp_path / "outputs")

    assert config.channel == 4
    assert config.window_size == 4096
    assert config.step_size == 2048
    assert config.output_root.name == "outputs"
    assert config.label_order == (
        "正常",
        "转子不平衡",
        "联轴器不对中",
        "松动",
        "轴承故障",
        "汽蚀",
    )


def test_modeling_config_matches_grouped_selection_defaults() -> None:
    config = ModelingConfig()

    assert config.correlation_threshold == 0.90
    assert config.cv_folds == 5
    assert config.mlp_hidden_layer_sizes == (64, 32)
