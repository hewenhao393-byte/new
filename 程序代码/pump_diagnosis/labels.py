from __future__ import annotations

LABEL_ORDER = (
    "正常",
    "转子不平衡",
    "联轴器不对中",
    "松动",
    "轴承故障",
    "汽蚀",
)

PREFIX_TO_LABEL = {
    "正常状态": "正常",
    "泵不平衡": "转子不平衡",
    "电机不平衡": "转子不平衡",
    "角向不对中": "联轴器不对中",
    "平行不对中": "联轴器不对中",
    "组合不对中": "联轴器不对中",
    "软脚": "松动",
    "电机地脚松动": "松动",
    "泵地脚松动": "松动",
    "轴承内圈故障": "轴承故障",
    "轴承外圈故障": "轴承故障",
    "轴承滚动体故障": "轴承故障",
    "轴承污染": "轴承故障",
    "泵轴承故障": "轴承故障",
    "吸入口汽蚀": "汽蚀",
    "出口汽蚀": "汽蚀",
}

EXCLUDED_PREFIXES = (
    "正常加噪声",
    "叶轮缺陷",
    "定子绕组短路",
)


def map_fault_to_label(raw_fault_name: str) -> str | None:
    for prefix in EXCLUDED_PREFIXES:
        if raw_fault_name.startswith(prefix):
            return None
    for prefix, label in PREFIX_TO_LABEL.items():
        if raw_fault_name.startswith(prefix):
            return label
    return None
