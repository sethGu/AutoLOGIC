# AutoLOGIC manuscript code

This repository contains the compact code set associated with the 40-page manuscript `auto_LOGIC_N.pdf` dated 24 August 2026. The current task runners materialize an explicit task specification and four inspectable plan records before executing the existing feature, model, optimization, and output-control paths. The `legacy_equivalent` profile preserves the numerical computation path used by the retained runners.

This directory is not yet a complete one-command reproduction artifact. Public datasets, historical run records used to assemble the final 40-dataset table, and a publication license are not included. The known evidence-to-code gaps are stated in `docs/REPRODUCIBILITY_STATUS.md`.

## Directory layout

- `autologic/`: current feature-generation, model-generation, stacking, distillation, calibration, regression-interval, and clustering implementations.
- `autologic/utils/task_protocol.py`: task-specification schema, four-plan materialization, and feature-provenance records.
- `configs/`: machine-readable task-specification examples and field documentation.
- `legacy_sage/`: minimal historical SAGE route required by the reported benchmark assembly.
- `compat/`: compatibility modules used by the historical benchmark runners.
- `reproduce/fig4/`: 40-dataset benchmark runners, recovery scripts, result-table reconstruction, and Fig. 4 plotting.
- `reproduce/fig5/`: fixed-pool mechanism, calibration, distillation, and plotting scripts.
- `reproduce/fig6/`: output-control, delivery-fidelity, clustering, stability, and plotting scripts.
- `data/`: data placement and provenance requirements; no raw datasets are committed.
- `scripts/check_release.py`: static compilation, path, secret, and structure checks.

## Environment

Python 3.10 or later is recommended for the cleaned release scripts. The dependency ranges were reconstructed from the latest code tree; they are not an exact environment lock for every historical run.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Set credentials and data locations through environment variables. Credentials must not be written into source files, logs, or manifests.

```text
OPENAI_API_KEY
OPENAI_BASE_URL
AUTOLOGIC_DATA_DIR
AUTOLOGIC_CSV_DATA_DIR
AUTOLOGIC_REGRESSION_DATA_DIR
```

## Entry points

Inspect the current task runners:

```bash
python autologic/ensemble/classification_ensemble/classification_auto_ensemble.py --help
python autologic/ensemble/multiclassification_ensemble/multiclassification_auto_ensemble.py --help
python autologic/ensemble/regression_ensemble/regression_auto_ensemble.py --help
python autologic/ensemble/cluster_ensemble/cluster_auto_ensemble_hidden_evaluator.py --help
```

Run the historical full benchmark route after restoring the documented datasets:

```powershell
pwsh reproduce/fig4/run_full_benchmark.ps1
```

Run static release checks:

```bash
python scripts/check_release.py
```

Regenerate the source checksum manifest after intentional changes:

```bash
python scripts/build_manifest.py
```

## Task specification and run records

Every current task runner accepts an optional `--task-spec <json>` argument. Without that argument, the runner creates a `legacy_equivalent` specification from its active command-line parameters, so the default model and metric computations are unchanged. The generated protocol records contain:

- the supplied task type, dataset, primary metric, output preferences, and bounded runtime constraints;
- feature, model, optimization, and output-control plans;
- failure, retry, and fallback summaries;
- feature-level lineage records containing source fields, static operators when recoverable, and generated-code hashes.

Example:

```bash
python autologic/ensemble/classification_ensemble/classification_auto_ensemble.py \
  --task-spec configs/task_spec.classification.example.json
```

## Known limitations

The following points are intentionally not represented as resolved:

1. Select and add a source-code license.
2. Publish or reconstruct the dataset manifest with source URLs, checksums, preprocessing, and redistribution status.
3. Reconcile the manuscript's label-free clustering-selection statement with the retained label-informed benchmark routes.
4. Reconcile the manuscript's locked-test statement with the retained Fig. 4 result-selection scripts.
5. Publish the historical result records required to regenerate the final 40-dataset table exactly.

The historical Fig. 4 snapshots and table-reconstruction logic are retained as evidence of the reported result assembly; they are not silently rewritten by the compatibility control layer.
