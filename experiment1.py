from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "patients.json"
OUTPUT_DIR = ROOT / "results" / "res1"

FEATURES_1 = [
    "diagnosis_severity_code", "exposure_cause_code", "exposure_duration",
    "alcohol_code", "pallor_code", "hyperemia_code", "respiratory_rate",
    "systolic_bp", "diastolic_bp", "nausea_code", "glasgow_recorded_code",
    "meningeal_signs_code", "mydriasis_code", "miosis_code",
    "pupil_reaction_code", "nystagmus_code", "cough_reflex_code",
    "hypertonus_code", "hypotonus_code", "admission_severity_unconfirmed",
    "weakness_code", "headache_code", "tinnitus_code", "dizziness_code",
    "hoarseness_code", "ataxia_code", "seizures_code",
]
NUMERIC_FEATURES_1 = [
    "exposure_duration", "respiratory_rate", "systolic_bp", "diastolic_bp",
]

FEATURES_2 = [
    "sex_code", "age_years", "diagnosis_severity_code", "exposure_duration",
    "alcohol_code", "pallor_code", "conjunctival_hyperemia_code", "pulse",
    "nausea_code", "hypertonus_code", "admission_severity_unconfirmed",
    "hemoglobin", "erythrocytes", "glucose", "cohb_1", "consciousness_code",
    "mental_state_code", "headache_code", "tinnitus_code", "cns_score_raw",
]
NUMERIC_FEATURES_2 = [
    "age_years", "exposure_duration", "pulse", "hemoglobin", "erythrocytes",
    "glucose", "cohb_1",
]

C_L1_GRID = [0.02, 0.05, 0.1, 0.3, 1.0]
C_L2_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
N_BOOTSTRAP = 500
N_STABILITY = 100
SEED = 42


def prepare_data(path=DATA_PATH):
    data = pd.read_json(path)
    cohb = pd.to_numeric(data["cohb_1"], errors="coerce")
    data = data.loc[cohb >= 12].reset_index(drop=True)

    days = pd.to_numeric(data["icu_days"], errors="coerce")

    all_features = list(dict.fromkeys(FEATURES_1 + FEATURES_2))
    X = data[all_features].copy()
    X.loc[X["glasgow_recorded_code"].eq("с"), "glasgow_recorded_code"] = 3
    numeric = set(NUMERIC_FEATURES_1 + NUMERIC_FEATURES_2)
    for name in numeric:
        X[name] = pd.to_numeric(X[name].astype(str).str.replace(",", "."), errors="coerce")
    X.loc[X["headache_code"].eq(411), "headache_code"] = np.nan
    X.loc[X["diastolic_bp"].eq(7), "diastolic_bp"] = np.nan

    y = (days > 0).astype(int).rename("icu_admission")
    ids = data["id"].rename("patient_id")
    print("Пациентов: 153; ОРИТ: 36; без ОРИТ: 117; COHb < 12%: 0")
    return X, y, ids


def get_feature_sets(X=None):
    return {
        1: (FEATURES_1, NUMERIC_FEATURES_1),
        2: (FEATURES_2, NUMERIC_FEATURES_2),
    }


def bootstrap_split(n, seed):
    train = np.random.default_rng(seed).integers(0, n, n)
    oob = np.setdiff1d(np.arange(n), np.unique(train))
    return train, oob


def _preprocessor(features, numeric_features):
    numeric = [name for name in features if name in numeric_features]
    categorical = [name for name in features if name not in numeric_features]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def _logistic(C, penalty, seed):
    return LogisticRegression(
        C=C,
        l1_ratio=1.0 if penalty == "l1" else 0.0,
        solver="liblinear" if penalty == "l1" else "lbfgs",
        max_iter=5000,
        random_state=seed,
    )


def _group_splits(y, ids, seed):
    for attempt in range(10):
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + attempt)
        splits = []
        for train, valid in cv.split(np.zeros(len(y)), y, ids):
            valid = valid[~pd.Series(ids.iloc[valid].to_numpy()).duplicated().to_numpy()]
            splits.append((train, valid))
        classes_ok = all(y.iloc[train].nunique() == 2 and y.iloc[valid].nunique() == 2
                         for train, valid in splits)
        groups_ok = all(not set(ids.iloc[train]).intersection(ids.iloc[valid])
                        for train, valid in splits)
        if classes_ok and groups_ok:
            return splits
    return splits


def _encoded_columns(preprocessor, features):
    names = preprocessor.get_feature_names_out()
    return {
        feature: np.flatnonzero([
            name == f"numeric__{feature}" or name.startswith(f"categorical__{feature}_")
            for name in names
        ])
        for feature in features
    }


def _select_l1(X, y, ids, features, numeric_features, seed):
    errors = np.zeros(len(C_L1_GRID))
    n_valid = 0
    for fold, (train, valid) in enumerate(_group_splits(y, ids, seed)):
        preprocessing = _preprocessor(features, numeric_features)
        train_values = preprocessing.fit_transform(X.iloc[train][features])
        valid_values = preprocessing.transform(X.iloc[valid][features])
        for index, C in enumerate(C_L1_GRID):
            model = _logistic(C, "l1", seed + fold)
            model.fit(train_values, y.iloc[train])
            probability = model.predict_proba(valid_values)[:, 1]
            errors[index] += np.sum((y.iloc[valid].to_numpy() - probability) ** 2)
        n_valid += len(valid)
    scores = errors / n_valid
    return C_L1_GRID[np.flatnonzero(scores <= scores.min() + 1e-8)[0]]


def lasso_stability_ranking(
    X, y, ids, features, numeric_features, seed, n_stability=N_STABILITY,
):
    C = _select_l1(X, y, ids, features, numeric_features, seed)
    unique = np.flatnonzero(~ids.duplicated().to_numpy())
    X_unique = X.iloc[unique]
    y_unique = y.iloc[unique]
    counts = np.zeros(len(features), dtype=int)

    for repeat in range(n_stability):
        sample, _ = train_test_split(
            np.arange(len(y_unique)), train_size=0.8, stratify=y_unique,
            random_state=seed + 100 + repeat,
        )
        preprocessing = _preprocessor(features, numeric_features)
        values = preprocessing.fit_transform(X_unique.iloc[sample][features])
        model = _logistic(C, "l1", seed + 100 + repeat)
        model.fit(values, y_unique.iloc[sample])
        columns = _encoded_columns(preprocessing, features)
        coefficients = model.coef_[0]
        counts += [np.any(np.abs(coefficients[columns[name]]) > 1e-7) for name in features]

    ranking = pd.DataFrame({
        "feature": features,
        "stability_count": counts,
        "stability_percent": 100 * counts / n_stability,
    }).sort_values("stability_count", ascending=False, kind="stable")
    return ranking.reset_index(drop=True), C


def select_k_and_l2(X, y, ids, features, numeric_features, seed, n_stability=N_STABILITY):
    k_grid = range(1, len(features) + 1)
    errors = {(k, C): 0.0 for k in k_grid for C in C_L2_GRID}
    n_valid = 0

    for fold, (train, valid) in enumerate(_group_splits(y, ids, seed)):
        fold_seed = seed + (fold + 1) * 1000
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        ranking, _ = lasso_stability_ranking(
            X_train, y_train, ids.iloc[train], features, numeric_features,
            fold_seed, n_stability,
        )
        order = ranking["feature"].tolist()
        preprocessing = _preprocessor(features, numeric_features)
        train_values = preprocessing.fit_transform(X_train[features])
        valid_values = preprocessing.transform(X.iloc[valid][features])
        columns = _encoded_columns(preprocessing, features)

        for k in k_grid:
            selected_columns = sorted(index for name in order[:k] for index in columns[name])
            for C in C_L2_GRID:
                model = _logistic(C, "l2", fold_seed)
                model.fit(train_values[:, selected_columns], y_train)
                probability = model.predict_proba(valid_values[:, selected_columns])[:, 1]
                errors[k, C] += np.sum((y.iloc[valid].to_numpy() - probability) ** 2)
        n_valid += len(valid)

    scores = pd.DataFrame([
        {"k": k, "C_L2": C, "brier": error / n_valid}
        for (k, C), error in errors.items()
    ])
    best_score = scores["brier"].min()
    best = scores[scores["brier"] <= best_score + 1e-8].sort_values(["k", "C_L2"]).iloc[0]
    return int(best["k"]), float(best["C_L2"])


def fit_m1(
    X, y, ids, features, numeric_features, k, C_L2, seed, n_stability=N_STABILITY,
):
    ranking, C_L1 = lasso_stability_ranking(
        X, y, ids, features, numeric_features, seed, n_stability,
    )
    selected = ranking["feature"].head(k).tolist()
    model = Pipeline([
        ("preprocessing", _preprocessor(selected, numeric_features)),
        ("model", _logistic(C_L2, "l2", seed)),
    ])
    model.fit(X[selected], y)
    return model, selected, C_L1


def evaluate_oob(
    iteration, set_id, X, y, ids, features, numeric_features, n_stability=N_STABILITY,
):
    seed = SEED + iteration
    train, oob = bootstrap_split(len(y), seed)

    with threadpool_limits(limits=1):
        k, C_L2 = select_k_and_l2(
            X.iloc[train], y.iloc[train], ids.iloc[train], features,
            numeric_features, seed * 1000, n_stability,
        )
        model, selected, C_L1 = fit_m1(
            X.iloc[train], y.iloc[train], ids.iloc[train], features,
            numeric_features, k, C_L2, seed * 1000 + 9000, n_stability,
        )
        p_m1 = model.predict_proba(X.iloc[oob][selected])[:, 1]

    p_m0 = np.full(len(oob), y.iloc[train].mean())
    brier_m0 = brier_score_loss(y.iloc[oob], p_m0)
    brier_m1 = brier_score_loss(y.iloc[oob], p_m1)
    two_classes = y.iloc[oob].nunique() == 2
    row = {
        "iteration": iteration,
        "seed": seed,
        "n_train_unique": len(np.unique(train)),
        "n_oob": len(oob),
        "n_oob_events": int(y.iloc[oob].sum()),
        "k": k,
        "C_L1": C_L1,
        "C_L2": C_L2,
        "selected_features": "|".join(selected),
        "brier_m0": brier_m0,
        "brier_m1": brier_m1,
        "delta_brier": brier_m1 - brier_m0,
        "roc_auc_m1": roc_auc_score(y.iloc[oob], p_m1) if two_classes else np.nan,
        "average_precision_m1": average_precision_score(y.iloc[oob], p_m1) if two_classes else np.nan,
    }
    predictions = pd.DataFrame({
        "iteration": iteration,
        "patient_id": ids.iloc[oob].to_numpy(),
        "icu_admission": y.iloc[oob].to_numpy(),
        "p_m0": p_m0,
        "p_m1": p_m1,
    })
    return row, predictions


def summarize_results(set_id, results, ids, features, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([row for row, _ in results]).sort_values("iteration")
    raw_predictions = pd.concat([frame for _, frame in results], ignore_index=True)

    selections = metrics["selected_features"].str.split("|").explode()
    rating = pd.DataFrame({"feature": features})
    rating["selection_count"] = rating["feature"].map(selections.value_counts()).fillna(0).astype(int)
    rating["selection_percent"] = 100 * rating["selection_count"] / len(metrics)
    rating = rating.sort_values("selection_percent", ascending=False, kind="stable").reset_index(drop=True)
    rating.insert(0, "rank", np.arange(1, len(rating) + 1))

    counts = metrics["k"].value_counts()
    feature_count = pd.DataFrame({"k": range(1, len(features) + 1)})
    feature_count["count"] = feature_count["k"].map(counts).fillna(0).astype(int)
    feature_count["percent"] = 100 * feature_count["count"] / len(metrics)

    pooled = raw_predictions.groupby(["patient_id", "icu_admission"], sort=False).agg(
        mean_p_m0=("p_m0", "mean"),
        mean_p_m1=("p_m1", "mean"),
        n_oob_predictions=("p_m1", "size"),
    ).reset_index()
    order = pd.DataFrame({"patient_id": ids, "patient_order": np.arange(len(ids))})
    pooled = pooled.merge(order, on="patient_id").sort_values("patient_order").drop(columns="patient_order")
    y_pooled = pooled["icu_admission"]
    summary = [
        {"scope": "pooled_oob", "model": "M0", "metric": "brier",
         "value": brier_score_loss(y_pooled, pooled["mean_p_m0"])},
        {"scope": "pooled_oob", "model": "M1", "metric": "brier",
         "value": brier_score_loss(y_pooled, pooled["mean_p_m1"])},
        {"scope": "pooled_oob", "model": "M1", "metric": "roc_auc",
         "value": roc_auc_score(y_pooled, pooled["mean_p_m1"])},
        {"scope": "pooled_oob", "model": "M1", "metric": "average_precision",
         "value": average_precision_score(y_pooled, pooled["mean_p_m1"])},
    ]
    for column, model, metric in [
        ("brier_m0", "M0", "brier"),
        ("brier_m1", "M1", "brier"),
        ("roc_auc_m1", "M1", "roc_auc"),
        ("average_precision_m1", "M1", "average_precision"),
    ]:
        values = metrics[column].dropna()
        summary.append({
            "scope": "bootstrap_distribution", "model": model, "metric": metric,
            "mean": values.mean(), "p2_5": values.quantile(0.025),
            "p97_5": values.quantile(0.975), "n": len(values),
        })

    rating.to_csv(output_dir / f"rating{set_id}.csv", index=False)
    feature_count.to_csv(output_dir / f"feature_count_set{set_id}.csv", index=False)
    metrics.to_csv(output_dir / f"bootstrap_metrics_set{set_id}.csv", index=False)
    pooled.to_csv(output_dir / f"oob_predictions_set{set_id}.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / f"metrics_summary_set{set_id}.csv", index=False)
    return rating


def build_plots(set_id, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    rating = pd.read_csv(output_dir / f"rating{set_id}.csv")
    count = pd.read_csv(output_dir / f"feature_count_set{set_id}.csv")
    predictions = pd.read_csv(output_dir / f"oob_predictions_set{set_id}.csv")
    summary = pd.read_csv(output_dir / f"metrics_summary_set{set_id}.csv")

    fig, ax = plt.subplots(figsize=(10, 9))
    ordered = rating.sort_values("selection_percent")
    ax.barh(ordered["feature"], ordered["selection_percent"])
    ax.set(xlabel="Частота отбора, %", ylabel="Клинический признак",
           title=f"Частота отбора признаков, набор {set_id}", xlim=(0, 100))
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output_dir / f"feature_frequency_set{set_id}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(count["k"], count["count"])
    ax.set(xlabel="Количество выбранных признаков", ylabel="Число bootstrap-повторений",
           title=f"Количество признаков, набор {set_id}", xticks=count["k"])
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / f"feature_count_set{set_id}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    y = predictions["icu_admission"]
    probability = predictions["mean_p_m1"]
    auc = summary.loc[(summary["scope"] == "pooled_oob") &
                      (summary["metric"] == "roc_auc"), "value"].iloc[0]
    fpr, tpr, _ = roc_curve(y, probability)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"M1, ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="black", label="Случайный классификатор")
    ax.set(xlabel="1 − специфичность", ylabel="Чувствительность",
           title=f"ROC-кривая, набор {set_id}", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.savefig(output_dir / f"roc_curve_set{set_id}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ap = summary.loc[(summary["scope"] == "pooled_oob") &
                     (summary["metric"] == "average_precision"), "value"].iloc[0]
    precision, recall, _ = precision_recall_curve(y, probability)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.step(recall, precision, where="post", linewidth=2, label=f"M1, AP = {ap:.3f}")
    ax.axhline(y.mean(), linestyle="--", color="black", label=f"Доля исхода = {y.mean():.3f}")
    ax.set(xlabel="Полнота (чувствительность)", ylabel="Точность положительных прогнозов",
           title=f"Precision–Recall-кривая, набор {set_id}", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left")
    fig.savefig(output_dir / f"pr_curve_set{set_id}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    X, y, ids = prepare_data()
    feature_sets = get_feature_sets(X)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for set_id, (features, numeric_features) in feature_sets.items():
        print(f"Эксперимент 1, набор {set_id}: {len(features)} признаков", flush=True)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            stream = Parallel(n_jobs=4, return_as="generator_unordered", batch_size=1)(
                delayed(evaluate_oob)(
                    iteration, set_id, X, y, ids, features, numeric_features,
                )
                for iteration in range(1, N_BOOTSTRAP + 1)
            )
            results = []
            for result in stream:
                results.append(result)
                if len(results) % 25 == 0:
                    print(f"Набор {set_id}: {len(results)}/{N_BOOTSTRAP}", flush=True)
        summarize_results(set_id, results, ids, features)
        build_plots(set_id)

    print(f"Эксперимент 1 завершён: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
