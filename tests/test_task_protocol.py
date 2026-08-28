from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
AUTOLOGIC_ROOT = ROOT / "autologic"
if str(AUTOLOGIC_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOLOGIC_ROOT))

from utils.task_protocol import (  # noqa: E402
    build_feature_provenance,
    materialize_task_protocol,
    write_task_protocol,
)
from agent import AgenticReasoningAgent  # noqa: E402


def _classification_args():
    return SimpleNamespace(
        feat_iterations=10,
        model_iterations=5,
        param_iterations=5,
        stage_timeout_s=300,
        enable_feedback=True,
        enable_optimization=True,
    )


class TaskProtocolTests(unittest.TestCase):
    def test_default_profile_preserves_active_arguments(self):
        args = _classification_args()
        spec, plans = materialize_task_protocol("classification", "cc1", args)
        self.assertEqual(spec.primary_metric, "auc")
        self.assertEqual(spec.constraints["feature_iterations"], 10)
        self.assertEqual(plans.feature_plan["iterations"], 10)
        self.assertEqual(spec.profile, "paper_default")
        self.assertEqual(plans.agent_execution["agent"], "AgenticReasoningAgent")
        self.assertEqual(
            plans.agent_execution["loop"],
            ["propose", "execute", "observe", "revise"],
        )

    def test_explicit_constraints_update_only_supported_runner_arguments(self):
        args = _classification_args()
        payload = {
            "task_type": "classification",
            "dataset": "cc1",
            "primary_metric": "auc",
            "constraints": {"feature_iterations": 3, "stage_timeout_s": 120},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            spec, plans = materialize_task_protocol("classification", "cc1", args, str(path))
        self.assertEqual(args.feat_iterations, 3)
        self.assertEqual(args.stage_timeout_s, 120)
        self.assertEqual(spec.constraints["feature_iterations"], 3)
        self.assertEqual(plans.feature_plan["iterations"], 3)

    def test_incompatible_metric_fails_before_execution(self):
        args = _classification_args()
        payload = {"task_type": "classification", "dataset": "cc1", "primary_metric": "accuracy"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Runner metric"):
                materialize_task_protocol("classification", "cc1", args, str(path))

    def test_feature_provenance_recovers_static_sources_and_operators(self):
        code = "df['ratio'] = np.log1p(df['amount']) / (df['age'] + 1)"
        records = build_feature_provenance(
            ["amount", "age", "target"],
            ["amount", "age", "ratio", "target"],
            "target",
            generated_code=code,
            round_num=2,
        )
        ratio = next(record for record in records if record["feature"] == "ratio")
        self.assertEqual(ratio["kind"], "derived")
        self.assertEqual(ratio["source_fields"], ["age", "amount"])
        self.assertIn("np.log1p", ratio["operators"])
        self.assertEqual(ratio["lineage_resolution"], "static_assignment")

    def test_protocol_file_is_machine_readable(self):
        args = _classification_args()
        spec, plans = materialize_task_protocol("classification", "cc1", args)
        agent = AgenticReasoningAgent(spec, plans)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_task_protocol(Path(tmp) / "protocol.json", spec, plans, agent=agent)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["task_spec"]["dataset"], "cc1")
        self.assertIn("output_control_plan", payload["plans"])
        self.assertEqual(payload["agent"]["agent"], "AgenticReasoningAgent")
        self.assertEqual(payload["proposal"]["task_spec"]["dataset"], "cc1")

    def test_agent_observation_creates_revision_transition(self):
        args = _classification_args()
        spec, plans = materialize_task_protocol("classification", "cc1", args)
        agent = AgenticReasoningAgent(spec, plans)
        transition = agent.observe(
            {
                "stage": "feature_engineering",
                "round": 2,
                "status": "completed",
                "events": {"retry": 1},
            }
        )
        self.assertEqual(transition["revision"]["action"], "revise_affected_plan")
        self.assertEqual(transition["revision"]["plan"], "feature")


if __name__ == "__main__":
    unittest.main()
