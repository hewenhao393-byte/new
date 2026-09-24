"""Engineering-facing description of the frozen formal diagnosis contract."""

from __future__ import annotations

from typing import Any

from pump_fault_app.domain.formal_contract import FORMAL_V3_CONTRACT


_FEATURE_LABELS = {
    "kurtosis": "峭度",
    "skewness": "偏度",
    "crest_factor": "峰值因子",
    "impulse_factor": "脉冲因子",
    "clearance_factor": "裕度因子",
    "shape_factor": "波形因子",
    "rot_2x_1x_ratio": "2X/1X阶次比",
    "rot_3x_1x_ratio": "3X/1X阶次比",
    "harmonic_energy_ratio_1x_5x": "1X～5X谐波能量占比",
    "spectral_entropy": "频谱熵",
    "spectral_flatness": "频谱平坦度",
    **{f"wp_energy_ratio_{index}": f"小波包第{index}频带能量占比" for index in range(8)},
    "env_kurtosis": "包络峭度",
    "env_crest_factor": "包络峰值因子",
}


def build_system_overview() -> dict[str, Any]:
    """Build display data from the immutable formal inference contract."""
    contract = FORMAL_V3_CONTRACT
    labels = tuple("机械松动" if label == "松动" else label for label in contract.label_order)
    features = tuple(_FEATURE_LABELS.get(name, name) for name in contract.feature_names)
    parameters = {
        "统一采样率": f"{contract.target_sampling_rate} Hz",
        "分析频带": f"{contract.filter_low_hz:g}～{contract.filter_high_hz:g} Hz",
        "窗口与步长": f"{contract.window_size}点 / {contract.step_size}点",
        "小波包": f"{contract.wavelet}，{contract.wavelet_level}层分解",
        "包络分析频带": f"{contract.envelope_low_hz:g}～{contract.envelope_high_hz:g} Hz",
    }
    parameter_items = tuple(
        {"label": label, "value": value}
        for label, value in parameters.items()
    )
    feature_groups = (
        {
            "title": "时域冲击特征",
            "features": features[:6],
            "feature_names": contract.feature_names[:6],
            "explanation": "描述振动波形的尖峰、冲击和形态变化，对机械松动及轴承冲击响应较敏感。",
        },
        {
            "title": "转频及倍频特征",
            "features": features[6:9],
            "feature_names": contract.feature_names[6:9],
            "explanation": "描述转频与倍频的相对关系，用于反映不平衡、不对中和松动造成的周期性振动结构变化。",
        },
        {
            "title": "频域能量特征",
            "features": features[9:11],
            "feature_names": contract.feature_names[9:11],
            "explanation": "描述振动能量在频率范围内的集中或分散程度，对汽蚀等宽频随机振动具有解释意义。",
        },
        {
            "title": "包络与小波特征",
            "features": features[11:],
            "feature_names": contract.feature_names[11:],
            "explanation": "描述冲击调制和非平稳振动在不同频带内的能量迁移，对轴承、松动和汽蚀状态具有辅助判别意义。",
        },
    )
    return {
        "title": "系统原理与模型说明",
        "subtitle": "基于43维振动特征与独立通道CatBoost模型的水泵六分类故障诊断方法",
        "pipeline": (
            "振动信号",
            "预处理",
            "滑动窗口",
            "43维特征",
            "通道独立CatBoost模型",
            "概率融合",
            "六类故障",
        ),
        "features": features,
        "formal_feature_names": contract.feature_names,
        "feature_groups": feature_groups,
        "model": {
            "名称": "CH3/CH4/CH5独立CatBoost六分类模型",
            "诊断方式": "窗口级预测 + 多窗口概率融合",
            "窗口级预测": "每个0.4 s有效窗口输入固定顺序的43维特征，由对应通道CatBoost模型分别输出六类状态概率。",
            "多窗口概率融合": "对完整振动记录中全部有效窗口的同类概率取平均，以最高平均概率对应类别作为最终诊断结果。",
        },
        "labels": labels,
        "fault_feature_rows": (
            {
                "故障": "转子不平衡",
                "主要振动表现": "转频主导、周期性明显",
                "主要敏感特征": "2X/1X、3X/1X、1X～5X谐波能量占比",
            },
            {
                "故障": "联轴器不对中",
                "主要振动表现": "2X、3X倍频增强",
                "主要敏感特征": "2X/1X、3X/1X、1X～5X谐波能量占比",
            },
            {
                "故障": "机械松动",
                "主要振动表现": "间隙碰撞、尖峰与多倍频",
                "主要敏感特征": "峭度、峰值因子、脉冲因子、包络特征",
            },
            {
                "故障": "轴承故障",
                "主要振动表现": "周期冲击和调制响应",
                "主要敏感特征": "包络峭度、包络峰值因子、小波包能量占比",
            },
            {
                "故障": "汽蚀",
                "主要振动表现": "宽频随机振动和频带能量迁移",
                "主要敏感特征": "频谱熵、频谱平坦度、小波包能量占比",
            },
        ),
        "parameters": parameters,
        "parameter_rows": (parameter_items[:3], parameter_items[3:]),
    }
