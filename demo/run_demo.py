"""Offline demonstration of AutoLOGIC's protocol, provenance and calibrated stacking.

Bundled feature/model candidates replace remote LLM generation in this example.
The demonstration is separate from the manuscript's quantitative experiments.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'autologic'))

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from agent import AgenticReasoningAgent
from utils.csv_dataset_loader import load_csv_dataframe
from utils.ensemble_utils2 import stacking_ensemble_v2
from utils.task_protocol import build_feature_provenance, materialize_task_protocol, write_task_protocol


class DemoAgent(AgenticReasoningAgent):
    def describe(self):
        return {**super().describe(), 'proposal_backend': 'bundled_demo_candidates',
                'executor': 'demo_sklearn_executor', 'remote_llm_called': False}


def engineered(frame):
    result = frame.copy()
    result['x2_x3'] = result['x2'] * result['x3']
    return result


def run_demo(data_path: Path, output: Path, seed: int = 42):
    started = time.perf_counter()
    frame, target = load_csv_dataframe(data_path, convert_dtypes=False)
    expected = [f'x{i}' for i in range(6)] + ['target']
    if list(frame.columns) != expected or set(frame[target].unique()) != {0, 1}:
        raise ValueError('Expected six numeric columns x0..x5 and binary target as the last column.')
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Demo data must contain finite numeric values.')
    if output.resolve() == data_path.parent.resolve():
        raise ValueError('Use a separate output directory.')
    X, y = frame.drop(columns=target), frame[target].astype(int)
    train, test = train_test_split(np.arange(len(frame)), test_size=.2, random_state=seed, stratify=y)
    fit, val = train_test_split(train, test_size=.25, random_state=seed, stratify=y.iloc[train])
    args = SimpleNamespace(feat_iterations=1, model_iterations=3, param_iterations=0,
                           enable_feedback=True, enable_optimization=False)
    spec, plans = materialize_task_protocol('classification', data_path.stem, args)
    spec = replace(spec, profile='offline_synthetic_demo')
    plans = replace(plans,
        feature_plan={**plans.feature_plan, 'action': 'evaluate_bundled_interaction'},
        model_plan={**plans.model_plan, 'action': 'evaluate_bundled_estimators'},
        agent_execution={**plans.agent_execution, 'profile': spec.profile,
                         'proposal_backend': 'bundled_demo_candidates', 'executor': 'demo_sklearn_executor'})
    agent = DemoAgent(spec, plans)
    candidates = {
        'logistic_C0.1': make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=seed)),
        'logistic_C1': make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=seed)),
        'random_forest': RandomForestClassifier(n_estimators=24, max_depth=4, min_samples_leaf=3, random_state=seed, n_jobs=1),
    }
    feature_scores = {}
    for name, features in [('raw', X), ('interaction', engineered(X))]:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        model.fit(features.iloc[fit], y.iloc[fit])
        feature_scores[name] = float(roc_auc_score(y.iloc[val], model.predict_proba(features.iloc[val])[:, 1]))
    feature_route = max(feature_scores, key=feature_scores.get)
    chosen = engineered(X) if feature_route == 'interaction' else X
    agent.observe({'stage': 'feature', 'round': 1, 'val_auc': feature_scores[feature_route]})
    model_scores = {}
    for index, (name, model) in enumerate(candidates.items(), 1):
        model.fit(chosen.iloc[fit], y.iloc[fit])
        model_scores[name] = float(roc_auc_score(y.iloc[val], model.predict_proba(chosen.iloc[val])[:, 1]))
        agent.observe({'stage': 'model', 'round': index, 'val_auc': model_scores[name]})
    selected = sorted(model_scores, key=model_scores.get, reverse=True)[:2]
    # The retained route is locked using validation; test labels only enter final metrics.
    with threadpool_limits(limits=1), parallel_backend('threading', n_jobs=1):
        result = stacking_ensemble_v2(
            [candidates[name] for name in selected], chosen.iloc[train], y.iloc[train],
            chosen.iloc[test], y.iloc[test], n_folds=3, meta_cv_repeats=1,
            meta_C_grid=(.1, 1), optimize_metric='roc_auc', tune_threshold=False,
            calibrate_proba=True, calibration_cv=3, random_state=seed,
        )
    agent.observe({'stage': 'output_calibration', 'round': 1, 'status': 'completed'})
    output.mkdir(parents=True, exist_ok=True)
    write_task_protocol(output / 'task_protocol.json', spec, plans, agent)
    program = "df['x2_x3'] = df['x2'] * df['x3']" if feature_route == 'interaction' else ''
    lineage = build_feature_provenance(X.columns, chosen.columns, None, program, 1)
    (output / 'feature_provenance.json').write_text(json.dumps(lineage, indent=2)+'\n', encoding='utf-8')
    (output / 'agent_transitions.json').write_text(json.dumps(agent.transitions(), indent=2)+'\n', encoding='utf-8')
    probability = result['meta_model'].predict_proba(result['test_meta'])[:, 1]
    pd.DataFrame({'row_index': test, 'target': y.iloc[test].to_numpy(), 'probability': probability,
                  'prediction': (probability >= .5).astype(int)}).to_csv(output / 'predictions.csv', index=False)
    report = {
        'mode': 'offline_synthetic_demo', 'remote_llm_called': False,
        'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(), 'seed': seed,
        'split_rows': {'fit': len(fit), 'validation': len(val), 'test': len(test)},
        'feature_validation_auc': feature_scores, 'selected_feature_route': feature_route,
        'model_validation_auc': model_scores, 'selected_models': selected,
        'stacking_metrics': {k: float(v) for k, v in result['stacking_metrics'].items()},
        'calibration_fit_source': result['calibration_fit_source'],
        'settings': {'stacking_folds': 3, 'meta_cv_repeats': 1, 'meta_C_grid': [.1, 1],
                     'calibration': 'sigmoid', 'calibration_folds': 3, 'threshold': .5},
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn']},
        'python': platform.python_version(), 'system': platform.system(),
        'elapsed_seconds': round(time.perf_counter()-started, 3),
    }
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'demo'/'data'/'demo_binary.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs'/'demo')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.data, args.output_dir, args.seed), indent=2))


if __name__ == '__main__':
    main()
