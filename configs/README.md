# Task specification

Task runners accept `--task-spec <path.json>` with the following fields:

- `task_type`: `classification`, `multiclassification`, `regression`, or `clustering`.
- `dataset`: dataset identifier used by the selected runner.
- `primary_metric`: `auc`, `rmse`, or `ari` according to the task.
- `output_preferences`: requested task outputs.
- `constraints`: feature, model, parameter, timeout, feedback, and optimization settings.
- `profile`: execution profile; the default is `paper_default`.

See `task_spec.classification.example.json` for a complete example.
