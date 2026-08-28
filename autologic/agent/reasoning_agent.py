from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from utils.task_protocol import PlanBundle, TaskSpec


AGENT_SCHEMA = "autologic.agent.v1"


@dataclass(frozen=True)
class AgentObservation:
    stage: str
    round: int
    status: str
    validation: dict[str, Any]
    failures: dict[str, int]
    retry_count: int
    fallback_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgenticReasoningAgent:
    """Stateful controller for the proposal-execution-observation-revision loop."""

    def __init__(self, task_spec: TaskSpec, plans: PlanBundle) -> None:
        self.task_spec = task_spec
        self.plans = plans
        self._transitions: list[dict[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        return {
            "schema": AGENT_SCHEMA,
            "agent": type(self).__name__,
            "proposal_backend": "task_runner_llm",
            "executor": "task_specific_modeling_executor",
            "loop": ["propose", "execute", "observe", "revise"],
            "plan_channels": ["feature", "model", "optimization", "output_control"],
        }

    def propose(self) -> dict[str, Any]:
        return {
            "task_spec": self.task_spec.to_dict(),
            "plans": self.plans.to_dict(),
        }

    @staticmethod
    def _validation_fields(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key.startswith("val_") or key in {"best_score_so_far", "status"}
        }

    @staticmethod
    def _affected_plan(stage: str) -> str:
        value = stage.lower()
        if "feature" in value:
            return "feature"
        if "param" in value or "optim" in value:
            return "optimization"
        if any(token in value for token in ("ensemble", "calibr", "output", "distill")):
            return "output_control"
        return "model"

    def observe(self, record: Mapping[str, Any]) -> dict[str, Any]:
        events = record.get("events", {}) if isinstance(record.get("events", {}), Mapping) else {}
        failures = record.get("failure_capture", {})
        if not isinstance(failures, Mapping):
            failures = {}
        failure = record.get("failure")
        if not failures and isinstance(failure, Mapping):
            failures = {str(failure.get("category", "execution_error")): 1}
        fallback_count = int(events.get("fallback", 0) or 0)
        if str(record.get("status", "")).lower() == "fallback" or record.get("fallback"):
            fallback_count = max(fallback_count, 1)
        observation = AgentObservation(
            stage=str(record.get("stage", "unknown")),
            round=int(record.get("round", 0) or 0),
            status=str(record.get("status", "completed")),
            validation=self._validation_fields(record),
            failures={str(key): int(value) for key, value in failures.items()},
            retry_count=int(events.get("retry", record.get("retry_count", 0)) or 0),
            fallback_count=fallback_count,
        )
        if observation.fallback_count:
            action = "fallback_to_best_runnable"
        elif observation.failures or observation.retry_count:
            action = "revise_affected_plan"
        else:
            action = "retain_and_continue"
        transition = {
            "schema": AGENT_SCHEMA,
            "observation": observation.to_dict(),
            "revision": {
                "action": action,
                "plan": self._affected_plan(observation.stage),
            },
        }
        self._transitions.append(transition)
        return transition

    def transitions(self) -> list[dict[str, Any]]:
        return list(self._transitions)
