import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from catboost import CatBoostClassifier

from baseline_analysis.evaluation import evaluate_predictions
from .config import CANDIDATES, CHECKPOINTS, FEATURES, LABEL_ORDER
from .selection import choose_unified, summarize_scores


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_probability_table(frame, labels):
    probabilities = frame[list(labels)].to_numpy(float)
    assert probabilities.ndim == 2 and probabilities.shape[1] == len(labels)
    assert np.isfinite(probabilities).all(), "non-finite probability"
    assert (probabilities >= 0).all() and (probabilities <= 1).all(), "probability outside [0,1]"
    sums = probabilities.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-8), "probability rows do not sum to one"
    expected = np.asarray(labels)[probabilities.argmax(axis=1)]
    assert np.array_equal(expected, frame["predicted_label"].to_numpy()), "argmax label mismatch"
    return {"rows": len(frame), "max_sum_error": float(np.max(np.abs(sums - 1))) if len(frame) else 0.0}


def _metric_checks(predictions, metric_path, confusion_path):
    stored = json.loads(Path(metric_path).read_text(encoding="utf-8"))
    actual = evaluate_predictions(predictions.label, predictions.predicted_label, LABEL_ORDER)
    for name in ("accuracy", "macro_f1", "weighted_f1"):
        assert np.isclose(stored[name], actual[name], atol=1e-12), f"metric mismatch: {name}"
    assert stored["confusion_matrix"] == actual["confusion_matrix"], "JSON confusion mismatch"
    csv_matrix = pd.read_csv(confusion_path, index_col=0).to_numpy().tolist()
    assert csv_matrix == actual["confusion_matrix"], "CSV confusion mismatch"


def verify_run(source_root, output_root):
    source = Path(source_root)
    root = Path(output_root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    checks = []

    def passed(name, detail):
        checks.append({"check": name, "passed": True, "detail": str(detail)})

    for path, expected in manifest["input_sha256"].items():
        assert _sha256(path) == expected, f"source hash mismatch: {path}"
    passed("source_hashes", len(manifest["input_sha256"]))

    folds = pd.read_csv(root / "shared_record_folds.csv")
    assert set(folds.fold) == {1, 2, 3}
    assert set(folds.role) == {"fit", "validation"}
    counts = folds.groupby(["fold", "record_id"]).size()
    assert counts.eq(1).all(), "record occurs more than once per fold"
    passed("shared_three_fold_manifest", len(folds))

    scores = pd.read_csv(root / "cv_fold_scores.csv")
    expected_combinations = len(CANDIDATES) * 3 * 3 * len(CHECKPOINTS)
    assert len(scores) == expected_combinations == 189
    assert not scores.duplicated(["candidate", "channel", "fold", "iteration"]).any()
    assert set(scores.candidate) == set(CANDIDATES)
    assert set(scores.channel) == {3, 4, 5}
    assert set(scores.fold) == {1, 2, 3}
    assert set(scores.iteration) == set(CHECKPOINTS)
    assert scores.fold_hash.nunique() == 1 and scores.fold_hash.iloc[0] == manifest["fold_hash"]
    passed("cv_exact_189_scores", len(scores))

    stored_summary = pd.read_csv(root / "cv_candidate_summary.csv")
    recomputed = summarize_scores(scores)
    pd.testing.assert_frame_equal(stored_summary, recomputed, check_exact=False, atol=1e-12, rtol=1e-12)
    chosen = choose_unified(recomputed)
    selected = manifest["selected"]
    assert chosen["candidate"] == selected["candidate"]
    assert int(chosen["iteration"]) == int(selected["iteration"])
    passed("deterministic_selection", f"{chosen['candidate']}@{int(chosen['iteration'])}")

    model_count = 0
    for mode in ("record", "temporal"):
        for channel in (3, 4, 5):
            run_dir = root / "models" / mode / f"ch{channel}"
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            assert metadata["iterations"] == int(selected["iteration"])
            assert metadata["params"] == selected["params"]
            assert metadata["features"] == FEATURES
            model = CatBoostClassifier()
            model.load_model(str(run_dir / "model.cbm"))
            assert model.tree_count_ == int(selected["iteration"])
            assert list(model.feature_names_) == FEATURES
            labels = metadata["classes"]
            assert set(labels) == set(LABEL_ORDER)
            for level in ("window", "record"):
                predictions = pd.read_csv(run_dir / f"{level}_predictions.csv")
                check_probability_table(predictions, labels)
                _metric_checks(
                    predictions,
                    run_dir / f"{level}_metrics.json",
                    run_dir / f"{level}_confusion_matrix.csv",
                )
                with Image.open(run_dir / f"{level}_confusion_matrix.png") as image:
                    image.verify()
            record_predictions = pd.read_csv(run_dir / "record_predictions.csv")
            assert {"window_count", "valid_window_count", "mean_max_class_probability"} <= set(record_predictions)
            assert (record_predictions.valid_window_count <= record_predictions.window_count).all()
            importance = pd.read_csv(run_dir / "feature_importance.csv")
            assert len(importance) == 43 and set(importance.feature) == set(FEATURES)
            assert np.isclose(importance.importance.sum(), 100.0, atol=1e-8)
            for name in ("window_error_source_summary.csv", "record_error_source_summary.csv"):
                assert (run_dir / name).is_file()
            model_count += 1
    passed("six_final_models", model_count)

    assert (root / "final_model_comparison.csv").is_file()
    assert (root / "conclusion.md").is_file()
    for candidate in CANDIDATES:
        with Image.open(root / "figures" / f"cv_{candidate}.png") as image:
            image.verify()
    passed("reports_and_figures", "3 CV curves, 12 confusion matrices, conclusion")

    report = pd.DataFrame(checks)
    report.to_csv(root / "verification_report.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 独立验收摘要",
        "",
        f"- 验收项：{len(report)}",
        f"- 通过：{int(report.passed.sum())}",
        "- 结论：输入哈希、三折CV、统一选参、6个最终模型、两级概率/指标/混淆矩阵均通过重算验收。",
        "",
        report.to_csv(index=False),
    ]
    (root / "verification_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report
