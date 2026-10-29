from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RES2_DIR = Path("results/res2")
RES3_DIR = Path("results/res3")

DEFAULT_WEIGHTS = {
    "Recall": 0.40,
    "ROC_AUC": 0.25,
    "BSS": 0.25,
    "AP": 0.10,
}

WEIGHT_SETS = [
    DEFAULT_WEIGHTS,
    {"Recall": 0.50, "ROC_AUC": 0.20, "BSS": 0.20, "AP": 0.10},
    {"Recall": 0.40, "ROC_AUC": 0.20, "BSS": 0.20, "AP": 0.20},
    {"Recall": 0.40, "ROC_AUC": 0.30, "BSS": 0.20, "AP": 0.10},
    {"Recall": 0.40, "ROC_AUC": 0.20, "BSS": 0.30, "AP": 0.10},
]

MIN_RECALL = 0.6
MIN_ROC_AUC = 0.8


def compute_integral_score(df, weights=DEFAULT_WEIGHTS):
    assert set(weights) == {"Recall", "ROC_AUC", "BSS", "AP"}
    assert all(value >= 0 for value in weights.values())
    assert np.isclose(sum(weights.values()), 1.0)
    return (
        weights["Recall"] * df["Recall_norm"]
        + weights["ROC_AUC"] * df["ROC_AUC_norm"]
        + weights["BSS"] * df["BSS_norm"]
        + weights["AP"] * df["AP_norm"]
    )


def load_model_results(res2_dir, baseline_brier):
    model_names = ["logreg_l1", "logreg_l2", "decision_tree", "random_forest"]
    required = [
        "n_features", "AP", "ROC_AUC", "Brier_Score", "Recall", "Precision",
        "F1", "F2", "train_Brier_mean", "Brier_gap",
    ]
    tables = []
    for set_id in [1, 2]:
        for model_name in model_names:
            path = Path(res2_dir) / f"model_{model_name}_set{set_id}.csv"
            table = pd.read_csv(path)
            assert set(required).issubset(table.columns), f"Нет нужных столбцов в {path}"
            table["model_name"] = model_name
            table["set_id"] = set_id
            tables.append(table)

    df = pd.concat(tables, ignore_index=True)
    df["BSS"] = 1 - df["Brier_Score"] / baseline_brier

    for column in ["Recall", "ROC_AUC", "BSS", "AP"]:
        assert np.isfinite(df[column]).all(), f"Некорректные значения в {column}"
        minimum = df[column].min()
        maximum = df[column].max()
        if maximum == minimum:
            df[f"{column}_norm"] = 0.5
        else:
            df[f"{column}_norm"] = (df[column] - minimum) / (maximum - minimum)

    df["integral_score"] = compute_integral_score(df)
    return df


def apply_quality_thresholds(df, baseline_brier):
    df = df.copy()
    df["eligible"] = (
        (df["Recall"] >= MIN_RECALL)
        & (df["ROC_AUC"] >= MIN_ROC_AUC)
        & (df["Brier_Score"] < baseline_brier)
    )
    return df


def rank_models(df):
    ranked = df.sort_values(
        ["integral_score", "Recall", "ROC_AUC", "Brier_Score"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranked["rank_all"] = np.arange(1, len(ranked) + 1)
    ranked["rank_eligible"] = np.nan
    eligible = ranked["eligible"]
    ranked.loc[eligible, "rank_eligible"] = np.arange(1, eligible.sum() + 1)
    return ranked


def analyze_weight_sensitivity(df, weight_sets: list[dict]):
    scenarios = []
    for number, weights in enumerate(weight_sets, start=1):
        scenario = df.copy()
        scenario["integral_score"] = compute_integral_score(scenario, weights)
        scenario = rank_models(scenario)
        scenario["weight_scenario"] = f"weights_{number}"
        scenario["weight_Recall"] = weights["Recall"]
        scenario["weight_ROC_AUC"] = weights["ROC_AUC"]
        scenario["weight_BSS"] = weights["BSS"]
        scenario["weight_AP"] = weights["AP"]
        scenarios.append(scenario)
    return pd.concat(scenarios, ignore_index=True)


def build_integral_plots(df, output_dir):
    output_dir = Path(output_dir)
    for set_id in [1, 2]:
        fig, ax = plt.subplots(figsize=(9, 6))
        data = df[df["set_id"] == set_id]
        for model_name, table in data.groupby("model_name"):
            table = table.sort_values("n_features")
            ax.plot(
                table["n_features"], table["integral_score"],
                marker="o", markersize=3, label=model_name,
            )
        ax.set(
            xlabel="Количество признаков",
            ylabel="Интегральный индекс",
            title=f"Интегральный индекс, набор {set_id}",
        )
        ax.legend()
        fig.savefig(
            output_dir / f"integral_by_n_set{set_id}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def main():
    RES3_DIR.mkdir(parents=True, exist_ok=True)
    baseline_table = pd.read_csv(RES2_DIR / "baseline_brier.csv")
    assert baseline_table.columns.tolist() == ["baseline_brier"]
    assert len(baseline_table) == 1
    baseline_brier = float(baseline_table.loc[0, "baseline_brier"])
    assert np.isfinite(baseline_brier) and baseline_brier > 0

    df = load_model_results(RES2_DIR, baseline_brier)
    df = apply_quality_thresholds(df, baseline_brier)
    df = rank_models(df)
    sensitivity = analyze_weight_sensitivity(df, WEIGHT_SETS)
    eligible = df[df["eligible"]].copy()
    best_models = eligible.groupby(["set_id", "model_name"], sort=False).head(1)

    df.to_csv(RES3_DIR / "integral_all.csv", index=False)
    eligible.to_csv(RES3_DIR / "integral_eligible.csv", index=False)
    sensitivity.to_csv(RES3_DIR / "weight_sensitivity.csv", index=False)
    best_models.to_csv(RES3_DIR / "best_models.csv", index=False)
    build_integral_plots(df, RES3_DIR)

    best_all = df.iloc[0]
    print(f"baseline_brier: {baseline_brier:.6f}")
    print(f"Всего конфигураций: {len(df)}")
    print(f"Допустимых конфигураций: {len(eligible)}")
    print(
        "Лучший общий вариант: "
        f"набор {best_all['set_id']}, {best_all['model_name']}, "
        f"{best_all['n_features']} признаков, индекс {best_all['integral_score']:.4f}"
    )
    if len(eligible):
        best = eligible.iloc[0]
        print(
            "Лучший допустимый вариант: "
            f"набор {best['set_id']}, {best['model_name']}, "
            f"{best['n_features']} признаков, индекс {best['integral_score']:.4f}"
        )
    else:
        print("Допустимых вариантов нет")
    print(f"Результаты: {RES3_DIR}")


if __name__ == "__main__":
    main()
