import unittest
import os
import pandas as pd
import numpy as np

from src.metrics import calculate_shooting_metrics, calculate_player_metrics, calculate_discipline_metrics, calculate_gk_metrics
from src.scoring import calculate_composite_scores
from src.trend import parse_score, calculate_team_form
from src.predict import parse_score as predict_parse_score, predict_fixture, train_player_prediction_model, train_match_prediction_model

class TestPremierLeaguePipeline(unittest.TestCase):

    def setUp(self):
        self.processed_dir = os.path.join("data", "processed")
        self.metrics_dir = os.path.join("data", "metrics")
        self.outputs_dir = "outputs"

    def test_score_parser(self):
        h, a = parse_score("4–2")
        self.assertEqual(h, 4)
        self.assertEqual(a, 2)
        
        h2, a2 = parse_score("0-0")
        self.assertEqual(h2, 0)
        self.assertEqual(a2, 0)

        h3, a3 = parse_score(np.nan)
        self.assertTrue(np.isnan(h3))
        self.assertTrue(np.isnan(a3))

    def test_metrics_files_exist(self):
        files = [
            "composite_scores.csv",
            "team_form_trend.csv",
            "player_ml_predictions.csv",
            "match_ml_predictions.csv"
        ]
        for f in files:
            p = os.path.join(self.metrics_dir, f)
            self.assertTrue(os.path.exists(p), f"File {p} does not exist")

    def test_composite_scores_content(self):
        df = pd.read_csv(os.path.join(self.metrics_dir, "composite_scores.csv"))
        self.assertIn("composite_score", df.columns)
        self.assertIn("pos_group", df.columns)
        self.assertGreater(len(df), 50)
        scores = df['composite_score'].dropna()
        self.assertGreater(len(scores), 0)
        self.assertTrue((scores >= 0).all() and (scores <= 100).all())

    def test_team_form_content(self):
        df = pd.read_csv(os.path.join(self.metrics_dir, "team_form_trend.csv"))
        self.assertIn("Squad", df.columns)
        self.assertIn("pts_last_5", df.columns)
        self.assertIn("form_status", df.columns)
        self.assertEqual(len(df), 20)  # 20 PL clubs

    def test_ml_player_predictions(self):
        df = pd.read_csv(os.path.join(self.metrics_dir, "player_ml_predictions.csv"))
        self.assertIn("predicted_ga_per90", df.columns)
        self.assertIn("efficiency_status", df.columns)
        self.assertGreater(len(df), 100)

    def test_predict_fixture_function(self):
        res = predict_fixture("Liverpool", "Everton", self.processed_dir, self.metrics_dir)
        self.assertEqual(res['home_team'], "Liverpool")
        self.assertEqual(res['away_team'], "Everton")
        self.assertIn("home_win_prob", res)
        self.assertIn("draw_prob", res)
        self.assertIn("away_win_prob", res)
        total_prob = res['home_win_prob'] + res['draw_prob'] + res['away_win_prob']
        self.assertAlmostEqual(total_prob, 100.0, delta=1.0)

if __name__ == "__main__":
    unittest.main()
