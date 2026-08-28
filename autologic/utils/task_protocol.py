from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = "1.0"

_PRIMARY_METRICS = {
    "classification": "auc",
    "multiclassification": "auc",
    "regression": "rmse",
    "clustering": "ari",
}

_OUTPUT_PREFERENCES = {
    "classification": ("probabilities", "optional_calibration", "optional_student_route"),
    "multiclassification": ("probabilities", "optional_calibration", "optional_student_route"),
    "regression": ("point_prediction", "optional_intervals", "optional_student_route"),
    "clustering": ("cluster_assignments", "consensus_output", "optional_student_route"),
}

_CONSTRAINT_TO_ARGUMENT = {
    "feature_iterations": "feat_iterations",
    "model_iterations": "model_iterations",
    "parameter_iterations": "param_iterations",
    "stage_timeout_s": "stage_timeout_s",
    "feedback_enabled": "enable_feedback",
    "optimization_enabled": "enable_optimization",
    "max_feature_failures": "max_feature_failures",
    "top_k": "top_k",
    "selection_repeats": "selection_repeats",
}


def _normalise_task_type(task_type: str) -> str:
    value = str(task_type).strip().lower().replace("-", "_")
    aliases = {
        "binary_classification": "classification",
        "multiclass": "multiclassification",
        "multiclass_classification": "multiclassification",
        "cluster": "clustering",
    }
    value = aliases.get(value, value)
    if value not in _PRIMARY_METRICS:
        raise ValueError(f"Unsupported task type: {task_type!r}")
    return value


def _arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _runtime_constraints(args: Any) -> dict[str, Any]:
    values = {
        "feature_iterations": _arg(args, "feat_iterations"),
        "model_iterations": _arg(args, "model_iterations"),
        "parameter_iterations": _arg(args, "param_iterations"),
        "stage_timeout_s": _arg(args, "stage_timeout_s"),
        "feedback_enabled": _arg(args, "enable_feedback"),
        "optimization_enabled": _arg(args, "enable_optimization"),
        "max_feature_failures": _arg(args, "max_feature_failures"),
        "top_k": _arg(args, "top_k"),
        "selection_repeats": _arg(args, "selection_repeats"),
    }
    return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    dataset: str
    primary_metric: str
    output_preferences: tuple[str, ...]
    constraints: dict[str, Any]
    profile: str = "legacy_equivalent"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanBundle:
    feature_plan: dict[str, Any]
    model_plan: dict[str, Any]
    optimization_plan: dict[str, Any]
    output_control_plan: dict[str, Any]
    revision_policy: dict[str, Any]
    compatibility: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_task_spec(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid task-spec JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Task specification must be a JSON object.")
    return payload


def _apply_explicit_constraints(args: Any, constraints: Mapping[str, Any]) -> None:
    unknown = sorted(set(constraints) - set(_CONSTRAINT_TO_ARGUMENT))
    if unknown:
        raise ValueError("Unsupported task-spec constraints: " + ", ".join(unknown))
    for field, value in constraints.items():
        arg_name = _CONSTRAINT_TO_ARGUMENT[field]
        if not hasattr(args, arg_name):
            raise ValueError(f"Constraint {field!r} is not supported by this task runner.")
        setattr(args, arg_name, value)


def build_plan_bundle(spec: TaskSpec, args: Any) -> PlanBundle:
    task = spec.task_type
    feature_plan = {
        "action": "bounded_llm_feature_synthesis",
        "iterations": int(_arg(args, "feat_iterations", 0) or 0),
        "screening_metric": spec.primary_metric,
        "record_feature_lineage": True,
    }
    model_plan = {
        "action": "bounded_llm_candidate_generation",
        "iterations": int(_arg(args, "model_iterations", 0) or 0),
        "task_type": task,
        "prediction_interface": "predict_proba" if "classification" in task else "predict",
    }
    optimization_plan = {
        "action": "validation_guided_local_revision",
        "enabled": bool(_arg(args, "enable_optimization", True)),
        "iterations": int(_arg(args, "param_iterations", 0) or 0),
        "primary_metric": spec.primary_metric,
    }
    if task in {"classification", "multiclassification"}:
        controls = ["stacking", "optional_distillation", "optional_probability_calibration"]
        calibration_fit = "oof_meta_train_cv" if task == "classification" else "validation_temperature_scaling"
    elif task == "regression":
        controls = ["stacking", "optional_distillation", "optional_conformal_interval"]
        calibration_fit = "validation_residuals"
    else:
        controls = ["consensus", "optional_distillation", "cluster_post_processing"]
        calibration_fit = None
    output_control_plan = {
        "action": "evaluate_supported_output_controls",
        "controls": controls,
        "preferences": list(spec.output_preferences),
        "calibration_fit": calibration_fit,
    }
    revision_policy = {
        "failure_capture": True,
        "feedback_enabled": bool(_arg(args, "enable_feedback", True)),
        "bounded_revisions": {
            "feature": int(_arg(args, "feat_iterations", 0) or 0),
            "model": int(_arg(args, "model_iterations", 0) or 0),
            "parameter": int(_arg(args, "param_iterations", 0) or 0),
        },
        "fallback": "best_runnable_or_task_runner_default",
    }
    compatibility = {
        "profile": spec.profile,
        "default_computation_path_changed": False,
        "selection_policy_changed": False,
        "test_evaluation_policy_changed": False,
        "prefit_student_oof_refit": False,
    }
    return PlanBundle(
        feature_plan=feature_plan,
        model_plan=model_plan,
        optimization_plan=optimization_plan,
        output_control_plan=output_control_plan,
        revision_policy=revision_policy,
        compatibility=compatibility,
    )


def materialize_task_protocol(
    task_type: str,
    dataset: str,
    args: Any,
    task_spec_path: Optional[str] = None,
) -> tuple[TaskSpec, PlanBundle]:
    task = _normalise_task_type(task_type)
    primary_metric = _PRIMARY_METRICS[task]
    output_preferences = _OUTPUT_PREFERENCES[task]
    profile = "legacy_equivalent"
    explicit_constraints: Mapping[str, Any] = {}

    if task_spec_path:
        path = Path(task_spec_path).expanduser().resolve()
        payload = _load_task_spec(path)
        requested_task = _normalise_task_type(payload.get("task_type", task))
        if requested_task != task:
            raise ValueError(f"Task specification requests {requested_task!r}; runner implements {task!r}.")
        requested_dataset = str(payload.get("dataset", dataset))
        if requested_dataset != str(dataset):
            raise ValueError(f"Task specification dataset {requested_dataset!r} does not match {dataset!r}.")
        requested_metric = str(payload.get("primary_metric", primary_metric)).lower()
        if requested_metric != primary_metric:
            raise ValueError(
                f"This compatibility runner implements primary metric {primary_metric!r}, not {requested_metric!r}."
            )
        requested_outputs = tuple(payload.get("output_preferences", output_preferences))
        unsupported_outputs = sorted(set(requested_outputs) - set(output_preferences))
        if unsupported_outputs:
            raise ValueError("Unsupported output preferences: " + ", ".join(unsupported_outputs))
        explicit_constraints = payload.get("constraints", {})
        if not isinstance(explicit_constraints, dict):
            raise ValueError("Task-spec constraints must be a JSON object.")
        _apply_explicit_constraints(args, explicit_constraints)
        output_preferences = requested_outputs
        profile = str(payload.get("profile", profile))

    constraints = _runtime_constraints(args)
    spec = TaskSpec(
        task_type=task,
        dataset=str(dataset),
        primary_metric=primary_metric,
        output_preferences=tuple(output_preferences),
        constraints=constraints,
        profile=profile,
    )
    return spec, build_plan_bundle(spec, args)


def write_task_protocol(path: str | Path, spec: TaskSpec, plans: PlanBundle) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"task_spec": spec.to_dict(), "plans": plans.to_dict()}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def log_task_protocol(logger: Any, spec: TaskSpec, plans: PlanBundle) -> None:
    logger.log(
        stage="task_protocol",
        r=0,
        task_spec=spec.to_dict(),
        plans=plans.to_dict(),
    )


def _literal_subscript_key(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Subscript):
        return None
    value = node.slice
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    if hasattr(ast, "Index") and isinstance(value, ast.Index):  # pragma: no cover - Python <3.9
        value = value.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return type(func).__name__


def _assignment_lineage(code: str) -> dict[str, dict[str, list[str]]]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return {}
    records: dict[str, dict[str, list[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets: list[ast.AST]
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        else:
            targets = [node.target]
            value = node.value
        target_names = [name for name in (_literal_subscript_key(target) for target in targets) if name]
        if not target_names:
            continue
        source_fields = sorted(
            {name for name in (_literal_subscript_key(part) for part in ast.walk(value)) if name}
        )
        operators = sorted({_call_name(part) for part in ast.walk(value) if isinstance(part, ast.Call)})
        for target_name in target_names:
            records[target_name] = {"source_fields": source_fields, "operators": operators}
    return records


def build_feature_provenance(
    input_columns: Sequence[Any],
    output_columns: Sequence[Any],
    target_column: Optional[str],
    generated_code: str = "",
    round_num: int = 0,
) -> list[dict[str, Any]]:
    inputs = [str(column) for column in input_columns if str(column) != str(target_column)]
    outputs = [str(column) for column in output_columns if str(column) != str(target_column)]
    input_set = set(inputs)
    assignment_records = _assignment_lineage(generated_code)
    code_hash = hashlib.sha256((generated_code or "").encode("utf-8")).hexdigest() if generated_code else None
    records: list[dict[str, Any]] = []
    for feature in outputs:
        derived = feature not in input_set
        lineage = assignment_records.get(feature, {})
        source_fields = [field for field in lineage.get("source_fields", []) if field in input_set]
        records.append(
            {
                "feature": feature,
                "kind": "derived" if derived else "source",
                "source_fields": source_fields if derived else [feature],
                "operators": lineage.get("operators", []) if derived else ["identity"],
                "code_sha256": code_hash if derived else None,
                "round": int(round_num),
                "lineage_resolution": (
                    "static_assignment" if derived and feature in assignment_records else
                    "dynamic_or_unresolved" if derived else
                    "identity"
                ),
            }
        )
    return records


def log_feature_provenance(
    logger: Any,
    round_num: int,
    input_columns: Iterable[Any],
    output_columns: Iterable[Any],
    target_column: Optional[str],
    generated_code: str,
) -> None:
    records = build_feature_provenance(
        list(input_columns),
        list(output_columns),
        target_column,
        generated_code=generated_code,
        round_num=round_num,
    )
    logger.log(
        stage="feature_provenance",
        r=round_num,
        source_feature_count=sum(record["kind"] == "source" for record in records),
        derived_feature_count=sum(record["kind"] == "derived" for record in records),
        records=records,
    )
