# Reproducible CPU demonstration

Run from the repository root after following [installation instructions](../docs/INSTALLATION.md):

```bash
python demo/run_demo.py
```

The included `data/demo_binary.csv` contains 240 synthetic rows, six numerical
features (`x0` through `x5`), and a binary `target` in the last column. It was
generated with seed 42 by `generate_sample.py`, contains no personal records,
and is released under the repository's Apache-2.0 license. Regenerate it with
`python demo/generate_sample.py`.

The example uses bundled candidates and makes no remote LLM calls. It exercises
AutoLOGIC's task protocol, observation controller, feature provenance, validation
selection, and the existing `stacking_ensemble_v2` implementation with sigmoid
calibration. The candidates are a feature interaction, two logistic-regression
configurations, and a small random forest. Candidate selection uses the 144-row
fit and 48-row validation partitions. After selection, stacking is fitted on
their 192-row union; the independent 48-row test partition is used for reporting.

This example demonstrates the shared binary-classification components. Full LLM
feature/model generation, regression, clustering, neural distillation and the
manuscript benchmark require the full runners and their dependencies. The demo
does not reproduce the manuscript's benchmark values.

## Expected output

The command creates five files under `outputs/demo/`:

- `metrics.json`: split counts, settings, versions, validation scores, final metrics and runtime.
- `predictions.csv`: 48 test rows with probability, prediction and reference target.
- `task_protocol.json`: task specification and four plan channels.
- `feature_provenance.json`: source fields and selected interaction.
- `agent_transitions.json`: feature, model and output-stage observations.

On Windows with Python 3.12.14 and `requirements-demo.txt`, the retained feature
is `x2_x3`, the selected models are `logistic_C0.1` and `logistic_C1`, ROC AUC is
approximately 0.998261, and accuracy is 0.9375. The measured computation took
0.241 seconds on the development desktop; allow a few seconds including Python
startup and imports. Floating-point results can differ slightly across platforms.

The exact checked output is available in `expected_metrics.json`. Runtime and
platform fields are descriptive and are not required to match.

## Other input data

For this small demo, supply a finite numeric CSV with the same columns and a
binary target containing both classes:

```bash
python demo/run_demo.py --data my_example.csv --output-dir outputs/my_example
```

Use the full task runners for tables with other schemas. The last CSV column is
the target; supply the dataset's description alongside it. See the full-run
section of [INSTALLATION.md](../docs/INSTALLATION.md). The generation stages
execute proposed Python code; run full LLM experiments in an isolated environment
containing only the intended input data and required credentials.
