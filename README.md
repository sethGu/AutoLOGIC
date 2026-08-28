# AutoLOGIC

AutoLOGIC is an agentic framework for automated tabular learning across classification, regression, and clustering. It converts a task specification into coordinated feature, model, optimization, and output-control plans, executes the plans with task-specific tools, and records plan revisions from validation and execution observations.

## Agentic workflow

1. `TaskSpec` defines the task, dataset, primary metric, output requirements, and execution constraints.
2. `AgenticReasoningAgent` coordinates four plan channels: feature engineering, model construction, optimization, and output control.
3. Task-specific executors generate and evaluate feature programs, candidate models, and parameter configurations.
4. Validation scores, failures, retries, and fallbacks are returned to the agent as observations.
5. The agent records a retain, revise, or fallback transition for the affected plan channel.
6. The selected route produces stacking, distillation, calibration, prediction-interval, or clustering outputs according to the task.

## Repository structure

- `autologic/agent/`: agent state and observation-revision transitions.
- `autologic/mystage1/`: feature proposal, execution, and evaluation.
- `autologic/ensemble/`: task-specific end-to-end runners.
- `autologic/utils/`: model generation, ensembling, logging, and task protocols.
- `reproduce/fig4/`: benchmark execution, table construction, and Fig. 4 generation.
- `reproduce/fig5/`: mechanism, calibration, distillation, and Fig. 5 generation.
- `reproduce/fig6/`: output-control, clustering, stability, and Fig. 6 generation.
- `legacy_sage/` and `compat/`: benchmark dependencies used by the reproduction scripts.
- `configs/`: task-specification examples.
- `data/`: expected dataset layout.

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Configure data paths and the LLM endpoint through environment variables:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
AUTOLOGIC_DATA_DIR
AUTOLOGIC_CSV_DATA_DIR
AUTOLOGIC_REGRESSION_DATA_DIR
```

## Task runners

```bash
python autologic/ensemble/classification_ensemble/classification_auto_ensemble.py --help
python autologic/ensemble/multiclassification_ensemble/multiclassification_auto_ensemble.py --help
python autologic/ensemble/regression_ensemble/regression_auto_ensemble.py --help
python autologic/ensemble/cluster_ensemble/cluster_auto_ensemble_hidden_evaluator.py --help
```

Each runner accepts `--task-spec <path.json>`. The generated protocol file contains the task specification, four plans, agent configuration, and execution transitions.

```bash
python autologic/ensemble/classification_ensemble/classification_auto_ensemble.py --task-spec configs/task_spec.classification.example.json
```

## Figure reproduction

```powershell
pwsh reproduce/fig4/run_full_benchmark.ps1
```

Fig. 5 and Fig. 6 scripts are organized by panel under `reproduce/fig5/` and `reproduce/fig6/`.

## Source verification

```bash
python scripts/check_release.py
python scripts/build_manifest.py
```
