from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class OutputExistsError(FileExistsError):
    """Raised instead of overwriting an experiment directory."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot_matrix(matrix: pd.DataFrame, path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    axis.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    axis.set_xlabel("Predicted"); axis.set_ylabel("True"); axis.set_title(title)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            axis.text(column, row, str(matrix.iloc[row, column]), ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def publish_results(
    output: Path | str,
    *,
    tables: Mapping[str, pd.DataFrame],
    json_files: Mapping[str, object],
    matrices: Mapping[tuple[str, str, str], pd.DataFrame],
    manifest: Mapping[str, object],
    readme: str,
) -> Path:
    target = Path(output)
    if target.exists():
        raise OutputExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=False, exist_ok=False)
    for relative, frame in tables.items():
        path = staging / relative; path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    for relative, value in json_files.items():
        path = staging / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    for (split, method, level), matrix in matrices.items():
        directory = staging / split; directory.mkdir(parents=True, exist_ok=True)
        stem = f"confusion_{method}_{level}"
        matrix.to_csv(directory / f"{stem}.csv", encoding="utf-8-sig")
        _plot_matrix(matrix, directory / f"{stem}.png", f"{split} / {method} / {level}")
    (staging / "README.md").write_text(readme, encoding="utf-8")
    hashes = {
        str(path.relative_to(staging)): _hash(path)
        for path in sorted(staging.rglob("*")) if path.is_file()
    }
    complete_manifest = dict(manifest)
    complete_manifest.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    complete_manifest["output_sha256"] = hashes
    (staging / "run_manifest.json").write_text(
        json.dumps(complete_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(staging, target)
    return target
