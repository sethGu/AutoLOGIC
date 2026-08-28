# Task-specification files

Current task runners accept `--task-spec <path.json>`. The JSON object may contain:

- `task_type`: `classification`, `multiclassification`, `regression`, or `clustering`;
- `dataset`: must match the runner's dataset argument;
- `primary_metric`: must match the implemented task metric (`auc`, `rmse`, or `ari`);
- `output_preferences`: a subset of the outputs supported by that runner;
- `constraints`: supported runner arguments such as bounded feature, model, and parameter iterations;
- `profile`: defaults to `legacy_equivalent`.

Unsupported fields fail before data loading or model execution. Omitting `--task-spec` constructs the same schema from the active command-line arguments without changing their values.
