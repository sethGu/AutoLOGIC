from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RetainedSelectionContractTests(unittest.TestCase):
    def test_label_informed_clustering_selector_remains_explicit(self):
        path = ROOT / "autologic" / "ensemble" / "cluster_ensemble" / "cluster_auto_ensemble_hidden_evaluator.py"
        text = path.read_text(encoding="utf-8")
        self.assertIn("valid.sort(key=lambda c: c.val_ari, reverse=True)", text)
        self.assertIn("score = float(adjusted_rand_score(y_val, val_labels))", text)

    def test_fig4_reconstruction_still_selects_reported_metrics(self):
        rebuild = (ROOT / "reproduce" / "fig4" / "rebuild_organized_v1_v7.py").read_text(encoding="utf-8")
        best40 = (ROOT / "reproduce" / "fig4" / "build_best40_table.py").read_text(encoding="utf-8")
        self.assertIn('records.sort(key=lambda r: ((r.get("auc")', rebuild)
        self.assertIn('records.sort(key=lambda r: r.get("ari")', rebuild)
        self.assertIn("def better_seed_key", best40)
        self.assertIn("select_best_dataset_groups", best40)


if __name__ == "__main__":
    unittest.main()
