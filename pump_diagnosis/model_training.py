from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import ModelingConfig


def train_and_evaluate_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: list[str],
    config: ModelingConfig,
) -> pd.DataFrame:
    config.output_root.mkdir(parents=True, exist_ok=True)
    results = [
        _fit_and_score_random_forest(train, test, selected_features, config),
        _fit_and_score_svm(train, test, selected_features, config),
        _fit_and_score_mlp(train, test, selected_features, config),
    ]
    summary = pd.DataFrame(results)
    summary.to_csv(config.output_root / "model_metrics.csv", index=False)
    return summary


def _fit_and_score_random_forest(train, test, selected_features, config):
    model = RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        min_samples_leaf=config.rf_min_samples_leaf,
        max_features=config.rf_max_features,
        bootstrap=config.rf_bootstrap,
        class_weight=config.rf_class_weight,
        random_state=config.random_state,
        n_jobs=config.rf_n_jobs,
    )
    return _fit_and_score_model("RandomForest", model, train, test, selected_features, config, len(train))


def _fit_and_score_svm(train, test, selected_features, config):
    sampled = _stratified_sample(train, config.svm_sample_size, config.random_state)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel=config.svm_kernel,
                    C=config.svm_c,
                    gamma=config.svm_gamma,
                    class_weight=config.svm_class_weight,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    return _fit_and_score_model("SVM", model, sampled, test, selected_features, config, len(sampled))


def _fit_and_score_mlp(train, test, selected_features, config):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=config.mlp_hidden_layer_sizes,
                    activation=config.mlp_activation,
                    solver=config.mlp_solver,
                    alpha=config.mlp_alpha,
                    learning_rate_init=config.mlp_learning_rate_init,
                    max_iter=config.mlp_max_iter,
                    early_stopping=False,
                    validation_fraction=config.mlp_validation_fraction,
                    n_iter_no_change=config.mlp_n_iter_no_change,
                    batch_size=config.mlp_batch_size,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    return _fit_and_score_model("MLP", model, train, test, selected_features, config, len(train))


def _fit_and_score_model(model_name, model, train, test, selected_features, config, training_samples):
    x_train = train[selected_features]
    y_train = train["label"]
    x_test = test[selected_features]
    y_test = test["label"]
    start = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - start
    predictions = model.predict(x_test)
    joblib.dump(model, config.output_root / f"{model_name}.joblib")
    labels = list(config.label_order)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        config.output_root / f"{model_name}_confusion_matrix.csv"
    )
    _save_confusion_png(
        config.output_root / f"{model_name}_confusion_matrix.png",
        matrix,
        labels,
        model_name,
        config.plot_dpi,
    )
    return {
        "model": model_name,
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "training_seconds": training_seconds,
        "training_samples": training_samples,
    }


def _stratified_sample(frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= sample_size:
        return frame.copy()
    parts = []
    for label, group in frame.groupby("label"):
        target = max(1, int(round(sample_size * len(group) / len(frame))))
        parts.append(group.sample(n=min(target, len(group)), random_state=random_state))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state)
    elif len(sampled) < sample_size:
        extra = frame.drop(sampled.index, errors="ignore").sample(
            n=sample_size - len(sampled),
            random_state=random_state,
        )
        sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled.reset_index(drop=True)


def _save_confusion_png(path: Path, matrix: np.ndarray, labels: list[str], title: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
