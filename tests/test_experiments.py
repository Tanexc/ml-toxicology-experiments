import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import experiment1
import experiment2


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.ids = experiment1.prepare_data()
        cls.feature_sets = experiment1.get_feature_sets(cls.X)

    def test_data_and_feature_sets(self):
        self.assertEqual((len(self.X), self.ids.nunique(), int(self.y.sum())), (153, 153, 36))
        self.assertEqual(len(self.feature_sets[1][0]), 27)
        self.assertEqual(len(self.feature_sets[2][0]), 20)
        self.assertNotIn("cohb_2", self.feature_sets[2][0])
        self.assertIn("cohb_1", self.feature_sets[2][0])
        self.assertFalse({"ecg_text", "color_index", "pirss_raw"}.intersection(
            self.feature_sets[2][0]))

    def test_bootstrap_and_group_splits(self):
        train, oob = experiment1.bootstrap_split(len(self.y), 43)
        self.assertFalse(set(self.ids.iloc[train]).intersection(self.ids.iloc[oob]))
        bootstrap_y = self.y.iloc[train]
        bootstrap_ids = self.ids.iloc[train]
        for fit, valid in experiment1._group_splits(bootstrap_y, bootstrap_ids, 42):
            self.assertFalse(set(bootstrap_ids.iloc[fit]).intersection(bootstrap_ids.iloc[valid]))
            self.assertTrue(bootstrap_ids.iloc[valid].is_unique)

    def test_second_experiment_configuration(self):
        self.assertEqual(experiment2.get_models(), [
            "logreg_l1", "logreg_l2", "decision_tree", "random_forest",
        ])
        self.assertEqual(experiment2.get_param_grid("logreg_l1"), {"model__C": [0.1, 1.0, 10.0]})
        metrics = experiment2.compute_metrics(self.y, np.full(len(self.y), self.y.mean()))
        self.assertEqual(set(metrics), {"AP", "ROC_AUC", "Brier_Score", "Recall", "Precision", "F1", "F2"})
        self.assertTrue(all(0 <= value <= 1 for value in metrics.values()))

    def test_completed_results_when_present(self):
        res1 = Path("results/res1")
        res2 = Path("results/res2")
        if not res1.exists() or not res2.exists():
            self.skipTest("Полные эксперименты ещё не запускались")
        ratings = experiment2.load_ratings(res1)
        self.assertEqual((len(ratings[1]), len(ratings[2])), (27, 20))
        for set_id, size in ((1, 27), (2, 20)):
            metrics = pd.read_csv(res1 / f"bootstrap_metrics_set{set_id}.csv")
            rating = pd.read_csv(res1 / f"rating{set_id}.csv")
            self.assertEqual(len(metrics), 500)
            self.assertEqual(rating["selection_count"].sum(), metrics["k"].sum())
            for model in experiment2.get_models():
                table = pd.read_csv(res2 / f"model_{model}_set{set_id}.csv")
                self.assertEqual(table.columns.tolist(), experiment2.TABLE_COLUMNS)
                self.assertEqual(table["n_features"].tolist(), list(range(size, 0, -1)))
        predictions = pd.read_csv(res2 / "cv_predictions.csv")
        self.assertEqual(len(predictions), 28764)
        self.assertTrue(predictions["p_oof"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
