# Manuscript-to-code alignment

The latest manuscript describes a staged workflow for classification, regression, and clustering. The retained code maps to that workflow as follows.

| Manuscript stage | Retained implementation |
| --- | --- |
| Task and data specification | `autologic/utils/task_protocol.py`, optional `--task-spec` inputs, task-specific entry points under `autologic/ensemble/`, and data loaders under `autologic/mystage1/` and `autologic/utils/` |
| Four coordinated plans | `PlanBundle` materializes feature, model, optimization, and output-control plans from the active task specification and runner parameters |
| Feature proposal and execution | `autologic/mystage1/stage1.py`, `run_llm_code.py`, and `stage1_evaluate.py` |
| Feature screening | Task-specific validation logic in the current ensemble runners |
| Candidate-model proposal | `autologic/utils/model_generate.py` and task-specific prompt construction |
| Candidate refinement and parameter search | Task-specific ensemble runners and helper modules under `autologic/utils/` |
| Route selection before final evaluation | Validation-based selection in classification and regression runners; clustering requires the reconciliation described in `REPRODUCIBILITY_STATUS.md` |
| Stacking | `autologic/utils/ensemble_utils2.py`, `autologic/ensemble/regression_ensemble_utils.py`, and `autologic/ensemble/cluster_ensemble_utils.py` |
| Optional student distillation | Task-specific runners plus Fig. 5 and Fig. 6 delivery scripts |
| Output control | Classification calibration, regression interval construction, and clustering consensus/post-processing in task helpers and `reproduce/fig6/` |
| Process and result records | Versioned JSON/JSONL task protocol, feature provenance, failure/retry/fallback summaries, and table-writing logic in task runners and reproduction scripts |

## Compatibility profile and exact implementation scope

The default `legacy_equivalent` profile is a control and recording layer. It does not change candidate generation, model fitting, validation metrics, clustering candidate ranking, or final benchmark-table selection. A supplied task specification may alter only runner arguments explicitly supported by the selected entry point; unsupported metrics, outputs, or constraints fail before execution.

Feature provenance is recorded after feature execution. Static source fields and operators are extracted when the generated code has directly resolvable assignments; dynamically generated relationships are marked `dynamic_or_unresolved` rather than inferred.

For stacking, ordinary refit-capable base models generate fold-specific out-of-fold predictions. Distilled students in the retained implementation are explicitly marked `prefit_student` and `oof_refit_capable = False`; they are distilled once and reused when included in a stacking configuration. Binary calibration is recorded as `oof_meta_train_cv`, multiclass temperature scaling as `validation_temperature_scaling`, and regression interval calibration as `validation_residuals`.

The label-informed clustering behavior and Fig. 4 test-metric result selection remain unchanged and are documented in `REPRODUCIBILITY_STATUS.md`.

## Figure coverage

- Fig. 4: full benchmark launch, recovery snapshots, 40-dataset table assembly, and subpanel generation are retained under `reproduce/fig4/`.
- Fig. 5: fixed-pool mechanism comparisons, calibration probes, distillation probes, evidence collection, and plotting are retained under `reproduce/fig5/`.
- Fig. 6: output-control runs, delivery-fidelity runs, clustering batches, stability analysis, and plotting are retained under `reproduce/fig6/`.
- An exact standalone Fig. 3 reconstruction script was not identified in the latest code tree. The implementation stages underlying that schematic are retained, but exact figure regeneration is not claimed.
