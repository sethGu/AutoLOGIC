# Reproducibility status and unresolved gaps

## 1. Clustering selection does not have one manuscript-consistent implementation

The manuscript states that clustering labels are reserved for final offline evaluation and are not used to construct or select the clustering route. The retained code contains two materially different behaviors:

- `reproduce/fig6/run_fig6b_output_control.py` uses expected-cluster-count-first internal criteria during construction and selection, then uses labels for final offline ARI/NMI reporting.
- `autologic/ensemble/cluster_ensemble/cluster_auto_ensemble_hidden_evaluator.py`, `legacy_sage/ensemble/cluster_ensemble/cluster_auto_ensemble_cached_labels.py`, and `reproduce/fig6/run_fig6b_live_clustering.py` contain label-informed validation or ARI-based selection behavior.

Therefore, the current code package does not support a blanket claim that every reported clustering result was selected without labels. Before public release, the manuscript claim must be narrowed or the clustering experiments must be rerun with a single label-free selection protocol.

## 2. The final 40-dataset table is assembled from multiple run versions

The final delivery records contain 40 datasets, five seeds per dataset, and 200 dataset-seed records. Thirty datasets use the strict same-setting route. Ten datasets use the available mixed setting `f10_m7_p5` or `f15_m15_p8`:

- Classification: `adult`, `balance-scale`, `bank`, `cd2`
- Clustering: `breast`, `cc3`, `cd1`, `glass`, `ld1`, `seeds`

The classification, multiclassification, and regression script snapshots also differ across the retained v6 and v9 recovery stages. Consequently, the final table must be described as a versioned reconstruction from completed runs, not as the output of one homogeneous end-to-end execution.

The retained reconstruction scripts parse final reported AUC, RMSE, and ARI values and select records across configurations or versions. The compatibility release does not modify this behavior. Therefore, the current package does not establish that the final 40-dataset table was assembled using validation-only route selection followed by one independent test evaluation.

## 3. Data and historical records are not included

No raw dataset is committed in this candidate directory. Exact reproduction also requires the source workbook, historical run tables, selected raw logs, and version manifests used by `build_best40_table.py` and `build_clean_delivery.py`. These materials should be added only after data-source, redistribution, and personal-information checks are complete.

## 4. Verification completed for this candidate

- The manuscript was inspected page by page and its methods, code-availability statement, main figures, and supplementary workflow descriptions were compared with the retained code families.
- User-specific executable, model, data, and output paths were replaced by repository-relative paths or environment variables in the candidate directory.
- Static Python compilation, CLI help checks for selected runners, secret-pattern checks, and file-structure checks are provided by `scripts/check_release.py`.

No new LLM experiment, full 40-dataset rerun, or numerical reproduction was performed while preparing this candidate. Static validity must not be reported as experimental reproduction.

## 5. Result-neutral control and recording additions

Current task runners now materialize task specifications, four plan records, feature-provenance records, and structured failure/retry/fallback summaries. The default profile records the active legacy parameters and does not alter model computations. Calibration sources and the non-refit status of pre-trained student routes are explicitly exposed in returned metadata.
