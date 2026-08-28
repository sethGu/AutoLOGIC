import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


RECORD_SCHEMA = "autologic.run_event.v1"
FAILURE_EVENT_NAMES = {
    "compile_error",
    "distill_error",
    "ensemble_eval_error",
    "llm_api_error",
    "runtime_error",
    "timeout",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(x: Any) -> Any:
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass
    if isinstance(x, (datetime,)):
        return x.isoformat()
    if isinstance(x, (set,)):
        return list(x)
    return x


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_jsonable)


@dataclass
class _RoundState:
    record: dict
    started_at: float


class QuantLogger:
    def __init__(
        self,
        task: str,
        dataset: str,
        out_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), "..", "result", "logs")
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        task_dir = os.path.join(out_dir, task)
        os.makedirs(task_dir, exist_ok=True)

        self.task = task
        self.dataset = dataset
        self.run_id = run_id
        self.path = os.path.join(task_dir, f"{dataset}_{run_id}.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")
        self._round: Optional[_RoundState] = None
        self._agent: Any = None

    def bind_agent(self, agent: Any) -> None:
        self._agent = agent

    def close(self) -> None:
        if self._round is not None:
            try:
                self.end_round(status="aborted")
            except Exception:
                self._round = None
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def in_round(self) -> bool:
        return self._round is not None

    def start_round(self, stage: str, r: int, **fields: Any) -> None:
        rec = {
            "record_schema": RECORD_SCHEMA,
            "ts": _now_iso(),
            "task": self.task,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "stage": stage,
            "round": int(r),
            "events": {},
        }
        rec.update(fields)
        self._round = _RoundState(record=rec, started_at=time.time())

    def event(self, name: str, inc: int = 1) -> None:
        if self._round is None:
            return
        events = self._round.record.setdefault("events", {})
        events[name] = int(events.get(name, 0)) + int(inc)
        if name in FAILURE_EVENT_NAMES:
            failures = self._round.record.setdefault("failure_capture", {})
            failures[name] = int(failures.get(name, 0)) + int(inc)

    def end_round(self, **fields: Any) -> None:
        if self._round is None:
            return
        dur_ms = int((time.time() - self._round.started_at) * 1000)
        self._round.record["duration_ms"] = dur_ms
        self._round.record.update(fields)
        decision = self._round.record.get("decision", {})
        events = self._round.record.setdefault("events", {})
        if isinstance(decision, dict) and decision.get("distill_used") and not decision.get("distill_success", True):
            events["fallback"] = int(events.get("fallback", 0)) + 1
        self._round.record["revision_summary"] = {
            "failure_captured": bool(self._round.record.get("failure_capture")),
            "retry_count": int(events.get("retry", self._round.record.get("retry_count", 0)) or 0),
            "fallback_count": int(events.get("fallback", 0) or 0),
        }
        if self._agent is not None:
            try:
                self._round.record["agent_transition"] = self._agent.observe(self._round.record)
            except Exception as exc:
                self._round.record["agent_transition_error"] = type(exc).__name__
        self._fh.write(_json_dumps(self._round.record) + "\n")
        self._fh.flush()
        self._round = None

    def log(self, stage: str, r: int, **fields: Any) -> None:
        rec = {
            "record_schema": RECORD_SCHEMA,
            "ts": _now_iso(),
            "task": self.task,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "stage": stage,
            "round": int(r),
        }
        rec.update(fields)
        if self._agent is not None and stage != "task_protocol":
            try:
                rec["agent_transition"] = self._agent.observe(rec)
            except Exception as exc:
                rec["agent_transition_error"] = type(exc).__name__
        self._fh.write(_json_dumps(rec) + "\n")
        self._fh.flush()

    def record_failure(self, stage: str, r: int, category: str, error: Any, **fields: Any) -> None:
        self.log(
            stage=stage,
            r=r,
            status="failure",
            failure={"category": category, "error": str(error)},
            **fields,
        )

    def record_fallback(self, stage: str, r: int, reason: str, selected: Any, **fields: Any) -> None:
        self.log(
            stage=stage,
            r=r,
            status="fallback",
            fallback={"reason": reason, "selected": selected},
            **fields,
        )
