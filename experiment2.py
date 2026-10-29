from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from experiment1 import OUTPUT_DIR as RES1_DIR
from experiment1 import _preprocessor, get_feature_sets, prepare_data


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "res2"
MODEL_NAMES = ["logreg_l1", "logreg_l2", "decision_tree", "random_forest"]
PARAM_GRIDS = {
    "logreg_l1": {"model__C": [0.1, 1.0, 10.0]},
    "logreg_l2": {"model__C": [0.1, 1.0, 10.0]},
    "decision_tree": {
        "model__max_depth": [3, 5, None],
        "model__min_samples_leaf": [2, 5],
    },
    "random_forest": {
        "model__max_depth": [3, None],
        "model__min_samples_leaf": [2, 5],
    },
}
TABLE_COLUMNS = [
    "n_features", "AP", "ROC_AUC", "Brier_Score", "Recall", "Precision",
    "F1", "F2", "train_Brier_mean", "Brier_gap",
]


def load_ratings(res1_dir=RES1_DIR):
    feature_sets = get_feature_sets()
    ratings = {}
    for set_id, (features, _) in feature_sets.items():
        path = Path(res1_dir) / f"rating{set_id}.csv"
        table = pd.read_csv(path)
        ranking = table["feature"].tolist()
        ratings[set_id] = ranking
    return ratings


def get_models():
    return MODEL_NAMES.copy()


def build_pipeline(model_name, selected_features, numeric_features, seed=42):
    if model_name == "logreg_l1":
        model = LogisticRegression(
            l1_ratio=1.0, solver="liblinear", max_iter=5000, random_state=42,
        )
    elif model_name == "logreg_l2":
        model = LogisticRegression(
            l1_ratio=0.0, solver="lbfgs", max_iter=5000, random_state=42,
        )
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(criterion="gini", random_state=42)
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    return Pipeline([
        ("preprocessing", _preprocessor(selected_features, numeric_features)),
        ("model", model),
    ])


def get_param_grid(model_name):
    return PARAM_GRIDS[model_name]


def fit_with_inner_cv(
    X_train, y_train, model_name, selected_features, numeric_features, seed,
):
    pipeline = build_pipeline(model_name, selected_features, numeric_features, seed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = GridSearchCV(
        pipeline,
        get_param_grid(model_name),
        scoring="neg_brier_score",
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score=np.nan,
    )
    search.fit(X_train[selected_features], y_train)
    return search.best_estimator_


def compute_metrics(y_true, p_oof):
    predicted = (p_oof >= 0.5).astype(int)
    return {
        "AP": average_precision_score(y_true, p_oof),
        "ROC_AUC": roc_auc_score(y_true, p_oof),
        "Brier_Score": brier_score_loss(y_true, p_oof),
        "Recall": recall_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "F2": fbeta_score(y_true, predicted, beta=2, zero_division=0),
    }


def evaluate_model_by_n(
    set_id, model_name, n, ranking, X, y, ids, numeric_features, outer_splits,
):
    selected = ranking[:n]
    p_oof = np.full(len(y), np.nan)
    train_brier = []

    with threadpool_limits(limits=1):
        for fold, (train, valid) in enumerate(outer_splits):
            model = fit_with_inner_cv(
                X.iloc[train], y.iloc[train], model_name, selected,
                numeric_features, 42 + fold,
            )
            p_oof[valid] = model.predict_proba(X.iloc[valid][selected])[:, 1]
            p_train = model.predict_proba(X.iloc[train][selected])[:, 1]
            train_brier.append(brier_score_loss(y.iloc[train], p_train))

    row = {"n_features": n, **compute_metrics(y, p_oof)}
    row["train_Brier_mean"] = float(np.mean(train_brier))
    row["Brier_gap"] = row["Brier_Score"] - row["train_Brier_mean"]
    predictions = pd.DataFrame({
        "set_id": set_id,
        "model_name": model_name,
        "n_features": n,
        "patient_id": ids.to_numpy(),
        "y_true": y.to_numpy(),
        "p_oof": p_oof,
    })
    return set_id, model_name, n, row, predictions


def save_model_table(set_id, model_name, rows, output_dir=OUTPUT_DIR):
    table = pd.DataFrame(rows).sort_values("n_features", ascending=False)
    table = table[TABLE_COLUMNS]
    table.to_csv(Path(output_dir) / f"model_{model_name}_set{set_id}.csv", index=False)
    return table


def build_comparison_plots(set_id, tables, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    labels = {
        "logreg_l1": "Логистическая L1",
        "logreg_l2": "Логистическая L2",
        "decision_tree": "Решающее дерево",
        "random_forest": "Случайный лес",
    }
    metrics = [
        ("ROC_AUC", "ROC-AUC", f"roc_auc_by_n_set{set_id}.png"),
        ("AP", "Average Precision", f"ap_by_n_set{set_id}.png"),
        ("Brier_Score", "Brier Score", f"brier_by_n_set{set_id}.png"),
        ("F2", "F2", f"f2_by_n_set{set_id}.png"),
    ]
    for column, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(9, 6))
        for model_name, table in tables.items():
            ordered = table.sort_values("n_features")
            ax.plot(ordered["n_features"], ordered[column], marker="o", markersize=3,
                    label=labels[model_name])
        ax.set(xlabel="Количество признаков", ylabel=ylabel,
               title=f"{ylabel} в зависимости от числа признаков, набор {set_id}")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (model_name, table) in zip(axes.flat, tables.items()):
        ordered = table.sort_values("n_features")
        ax.plot(ordered["n_features"], ordered["Brier_Score"], marker="o",
                markersize=3, label="OOF Brier")
        ax.plot(ordered["n_features"], ordered["train_Brier_mean"], marker="o",
                markersize=3, label="Train Brier")
        ax.set(title=labels[model_name], xlabel="Количество признаков", ylabel="Brier Score")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(f"Разрыв Brier между OOF и train, набор {set_id}")
    fig.tight_layout()
    fig.savefig(output_dir / f"overfitting_brier_set{set_id}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    X, y, ids = prepare_data()
    feature_sets = get_feature_sets(X)
    ratings = load_ratings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_splits = list(outer_cv.split(np.zeros(len(y)), y))
    all_predictions = []

    for set_id, ranking in ratings.items():
        numeric_features = feature_sets[set_id][1]
        tasks = [
            (model_name, n)
            for model_name in get_models()
            for n in range(len(ranking), 0, -1)
        ]
        print(f"Эксперимент 2, набор {set_id}: {len(tasks)} комбинаций", flush=True)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            stream = Parallel(n_jobs=4, return_as="generator_unordered", batch_size=1)(
                delayed(evaluate_model_by_n)(
                    set_id, model_name, n, ranking, X, y, ids,
                    numeric_features, outer_splits,
                )
                for model_name, n in tasks
            )
            results = []
            for result in stream:
                results.append(result)
                if len(results) % 10 == 0 or len(results) == len(tasks):
                    print(f"Набор {set_id}: {len(results)}/{len(tasks)}", flush=True)

        tables = {}
        for model_name in get_models():
            rows = [row for _, name, _, row, _ in results if name == model_name]
            tables[model_name] = save_model_table(set_id, model_name, rows)
        all_predictions.extend(predictions for *_, predictions in results)
        build_comparison_plots(set_id, tables)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(OUTPUT_DIR / "cv_predictions.csv", index=False)


if __name__ == "__main__":
    main()
