from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAS_DEPS = all(importlib.util.find_spec(name) for name in ['numpy', 'pandas', 'sklearn'])


@unittest.skipUnless(HAS_DEPS, 'Install requirements-demo.txt for demonstration tests.')
class DemoTests(unittest.TestCase):
    def test_demo_produces_valid_calibrated_predictions(self):
        sys.path.insert(0, str(ROOT / 'demo'))
        from run_demo import run_demo
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            report = run_demo(ROOT/'demo'/'data'/'demo_binary.csv', output)
            pred = pd.read_csv(output/'predictions.csv')
            self.assertEqual(len(pred), 48)
            self.assertTrue(pred.probability.between(0, 1).all())
            self.assertGreater(report['stacking_metrics']['roc_auc'], .9)
            self.assertFalse(report['remote_llm_called'])
            self.assertEqual(report['calibration_fit_source'], 'oof_meta_train_cv')
            protocol = json.loads((output/'task_protocol.json').read_text())
            self.assertEqual(protocol['agent']['proposal_backend'], 'bundled_demo_candidates')
            lineage = json.loads((output/'feature_provenance.json').read_text())
            self.assertTrue(lineage)

    def test_test_labels_do_not_affect_stacking_predictions(self):
        sys.path.insert(0, str(ROOT/'autologic'))
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from joblib import parallel_backend
        from threadpoolctl import threadpool_limits
        from utils.ensemble_utils2 import stacking_ensemble_v2
        X, y = make_classification(n_samples=120, n_features=6, random_state=9)
        outputs = []
        with threadpool_limits(limits=1), parallel_backend('threading', n_jobs=1):
            for test_y in [y[90:], 1-y[90:]]:
                result = stacking_ensemble_v2([LogisticRegression()], X[:90], y[:90], X[90:], test_y,
                    n_folds=3, meta_cv_repeats=1, meta_C_grid=(1,), calibration_cv=3,
                    tune_threshold=False, random_state=42)
                outputs.append(result['meta_model'].predict_proba(result['test_meta']))
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
