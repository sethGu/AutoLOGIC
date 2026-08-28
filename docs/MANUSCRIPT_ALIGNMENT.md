# AutoLOGIC implementation map

| Method component | Implementation |
| --- | --- |
| Task specification | `autologic/utils/task_protocol.py` |
| Agent state and revision transitions | `autologic/agent/reasoning_agent.py` |
| Feature proposal and execution | `autologic/mystage1/stage1.py`, `autologic/mystage1/run_llm_code.py` |
| Feature evaluation | `autologic/mystage1/stage1_evaluate.py` and task-specific runners |
| Candidate-model proposal | `autologic/utils/model_generate.py` |
| Parameter optimization | Task-specific runners under `autologic/ensemble/` |
| Stacking | `autologic/utils/ensemble_utils2.py` and task-specific ensemble utilities |
| Student distillation | Classification, multiclassification, and regression runners |
| Classification calibration | Classification and multiclassification runners |
| Regression prediction intervals | Regression runner and Fig. 5/6 scripts |
| Clustering ensemble | `autologic/ensemble/cluster_ensemble/` |
| Structured run records | `autologic/utils/quant_log.py` and `autologic/utils/task_protocol.py` |
| Fig. 4 experiments | `reproduce/fig4/` |
| Fig. 5 experiments | `reproduce/fig5/` |
| Fig. 6 experiments | `reproduce/fig6/` |

The runtime sequence is `propose -> execute -> observe -> revise`. The four plan channels are `feature`, `model`, `optimization`, and `output_control`.
