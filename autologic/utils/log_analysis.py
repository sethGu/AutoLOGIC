import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _get_metric(row: dict, key: str) -> Optional[float]:
    v = row.get(key, None)
    if v is None:
        return None
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    except Exception:
        return None


@dataclass
class ConvergenceResult:
    converged_round: Optional[int]
    best_round: Optional[int]
    best_score: Optional[float]


def convergence_round(
    rows: list[dict],
    metric_key: str,
    stable_key: Optional[str] = None,
    eps: float = 1e-3,
    stable_eps: float = 1e-3,
    mode: str = "max",
) -> ConvergenceResult:
    seq = [(r.get("round"), _get_metric(r, metric_key), _get_metric(r, stable_key) if stable_key else None) for r in rows]
    seq = [(rr, s, t) for (rr, s, t) in seq if rr is not None and s is not None]
    seq.sort(key=lambda x: x[0])
    if not seq:
        return ConvergenceResult(converged_round=None, best_round=None, best_score=None)

    best_round = None
    best_score = None
    for rr, s, _ in seq:
        if best_score is None:
            best_score = s
            best_round = rr
            continue
        if mode == "min":
            if s < best_score:
                best_score = s
                best_round = rr
        else:
            if s > best_score:
                best_score = s
                best_round = rr

    converged = None
    for i in range(1, len(seq)):
        r_prev, s_prev, t_prev = seq[i - 1]
        r_cur, s_cur, t_cur = seq[i]
        if abs(s_cur - s_prev) < eps:
            if stable_key is None:
                converged = r_cur
                break
            if t_prev is not None and t_cur is not None and abs(t_cur - t_prev) < stable_eps:
                converged = r_cur
                break

    return ConvergenceResult(converged_round=converged, best_round=best_round, best_score=best_score)


def summarize(path: Path, stage: Optional[str], metric_key: str, stable_key: Optional[str], mode: str) -> None:
    rows = _read_jsonl(path)
    if stage is not None:
        rows = [r for r in rows if r.get("stage") == stage]

    rows_by_round = defaultdict(list)
    for r in rows:
        rr = r.get("round", None)
        if rr is None:
            continue
        rows_by_round[int(rr)].append(r)

    rounds = sorted(rows_by_round.keys())
    if not rounds:
        print("no rows")
        return

    metric_seq = []
    trust_seq = []
    event_seq = []
    for rr in rounds:
        rs = rows_by_round[rr]
        v = None
        for row in rs:
            v = _get_metric(row, metric_key)
            if v is not None:
                break
        metric_seq.append((rr, v))

        tv = None
        if stable_key is not None:
            for row in rs:
                tv = _get_metric(row, stable_key)
                if tv is not None:
                    break
        trust_seq.append((rr, tv))

        events = defaultdict(int)
        for row in rs:
            ev = row.get("events", {}) or {}
            for k, c in ev.items():
                try:
                    events[k] += int(c)
                except Exception:
                    pass
        event_seq.append((rr, dict(events)))

    print(f"path={path}")
    print(f"stage={stage or 'ALL'} metric={metric_key} trust={stable_key or 'None'} rounds={len(rounds)}")

    last = None
    for rr, v in metric_seq:
        if v is None:
            continue
        if last is None:
            print(f"r={rr} {metric_key}={v:.6f} Δ=nan")
        else:
            print(f"r={rr} {metric_key}={v:.6f} Δ={v-last:+.6f}")
        last = v

    if stable_key is not None:
        last = None
        for rr, v in trust_seq:
            if v is None:
                continue
            if last is None:
                print(f"r={rr} {stable_key}={v:.6f} Δ=nan")
            else:
                print(f"r={rr} {stable_key}={v:.6f} Δ={v-last:+.6f}")
            last = v

    cr = convergence_round(rows, metric_key=metric_key, stable_key=stable_key, eps=1e-3, stable_eps=1e-3, mode=mode)
    print(f"best_round={cr.best_round} best_score={cr.best_score}")
    print(f"converged_round={cr.converged_round}")

    total_events = defaultdict(int)
    for _, ev in event_seq:
        for k, c in ev.items():
            total_events[k] += c
    if total_events:
        items = sorted(total_events.items(), key=lambda x: (-x[1], x[0]))
        s = " ".join([f"{k}={v}" for k, v in items])
        print(f"events_total {s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, type=str)
    ap.add_argument("--stage", default=None, type=str)
    ap.add_argument("--metric", required=True, type=str)
    ap.add_argument("--trust", default=None, type=str)
    ap.add_argument("--mode", default="max", choices=["max", "min"])
    args = ap.parse_args()
    summarize(Path(args.path), stage=args.stage, metric_key=args.metric, stable_key=args.trust, mode=args.mode)


if __name__ == "__main__":
    main()
