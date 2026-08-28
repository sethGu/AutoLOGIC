import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent / "fig5_strict_fixed_pool_unit.py"
PYTHON = os.environ.get("AUTOLOGIC_PYTHON_EXE", sys.executable)

TASKS = {
    "classification": ["cc1", "credit-g", "ld1"],
    "regression": ["boston", "concrete", "california"],
    "clustering": ["breast", "glass", "students"],
}
SEEDS = [42, 43, 44, 45, 46]
FIELDNAMES = [
    "task",
    "dataset",
    "seed",
    "status",
    "seconds",
    "returncode",
    "unit_dir",
    "stdout",
    "stderr",
    "error",
]
EXPECTED_CONDITIONS = {"all_select", "wo_feature", "wo_closed_loop", "wo_meta"}


def write_status(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def ledger_cost(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                total += float(json.loads(line).get("cost_usd") or 0)
            except Exception:
                pass
    return total


def strict_results_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    if not rows:
        return False
    conditions = {row.get("condition") for row in rows}
    if not EXPECTED_CONDITIONS.issubset(conditions):
        return False
    return all(row.get("status") == "ok" for row in rows if row.get("condition") in EXPECTED_CONDITIONS)


def api_available(base_url: str, api_key: str, model: str) -> tuple[bool, str]:
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=20)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Return only: ok"}],
            max_completion_tokens=5,
            temperature=0,
        )
        return True, str(resp.choices[0].message.content)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc!r}"


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def completed_keys(rows: list[dict]) -> set[tuple[str, str, int]]:
    return {
        (r["task"], r["dataset"], int(r["seed"]))
        for r in rows
        if r.get("status") == "ok"
    }


def seen_keys(rows: list[dict]) -> set[tuple[str, str, int]]:
    return {
        (r["task"], r["dataset"], int(r["seed"]))
        for r in rows
        if r.get("task") and r.get("dataset") and r.get("seed")
    }


def wait_for_api(args, run_dir: Path, rows: list[dict], status_path: Path, unit_key: tuple[str, str, int]) -> bool:
    task, dataset, seed = unit_key
    while True:
        ok, detail = api_available(args.base_url, args.api_key, args.llm)
        (run_dir / "api_probe_last.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {ok} {detail}\n", encoding="utf-8")
        if ok:
            return True
        if not args.wait_for_api:
            rows.append({"task": task, "dataset": dataset, "seed": seed, "status": "api_unavailable", "seconds": 0, "returncode": "", "unit_dir": "", "stdout": "", "stderr": "", "error": detail})
            write_status(status_path, rows)
            return False
        time.sleep(args.api_retry_sleep_s)


def run_unit(task: str, dataset: str, seed: int, run_dir: Path, args) -> dict:
    unit_dir = run_dir / "units" / task / dataset / f"seed_{seed}"
    if args.archive_existing_unit_dir and unit_dir.exists():
        archive = unit_dir.with_name(f"{unit_dir.name}_archive_{time.strftime('%Y%m%d_%H%M%S')}")
        suffix = 1
        while archive.exists():
            archive = unit_dir.with_name(f"{unit_dir.name}_archive_{time.strftime('%Y%m%d_%H%M%S')}_{suffix}")
            suffix += 1
        shutil.copytree(unit_dir, archive)
    unit_dir.mkdir(parents=True, exist_ok=True)
    stdout = unit_dir / "stdout.log"
    stderr = unit_dir / "stderr.log"
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_BASE_URL": args.base_url,
            "OPENAI_API_KEY": args.api_key,
            "PYTHONIOENCODING": "utf-8",
            "FIG5_TOKEN_LEDGER": str(run_dir / "token_ledger.jsonl"),
            "FIG5_PROMPT_PRICE_PER_M": str(args.prompt_price_per_m),
            "FIG5_COMPLETION_PRICE_PER_M": str(args.completion_price_per_m),
            "FIG5_MAX_LEDGER_COST_USD": str(args.max_cost_usd),
            "FIG5_MAX_COMPLETION_TOKENS": str(args.max_completion_tokens),
            "FIG5_FEATURE_MAX_COMPLETION_TOKENS": str(args.feature_max_completion_tokens),
            "FIG5_MAX_MESSAGES": str(args.max_messages),
            "FIG5_MAX_MESSAGE_CHARS": str(args.max_message_chars),
            "FIG5_OPENAI_TIMEOUT_S": str(args.openai_timeout_s),
            "AUTOLOGIC_MAX_FEATURE_RETRIES": str(args.max_feature_retries),
        }
    )
    cmd = [
        args.python,
        str(UNIT),
        "--task",
        task,
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "-f",
        str(args.feat_iterations),
        "-m",
        str(args.model_iterations),
        "-p",
        str(args.param_iterations),
        "-l",
        args.llm,
        "--base-url",
        args.base_url,
        "--out-dir",
        str(unit_dir),
        "--max-retries-per-model",
        str(args.max_retries_per_model),
    ]
    if args.allow_fallback:
        cmd.append("--allow-fallback")

    started = time.time()
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=out,
            stderr=err,
            timeout=args.unit_timeout_s,
        )
    status = "ok" if proc.returncode == 0 and strict_results_ok(unit_dir / "strict_results.csv") else "failed"
    return {
        "task": task,
        "dataset": dataset,
        "seed": seed,
        "status": status,
        "seconds": round(time.time() - started, 3),
        "returncode": proc.returncode,
        "unit_dir": str(unit_dir),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "error": "",
    }


def run_unit_safe(task: str, dataset: str, seed: int, run_dir: Path, args) -> dict:
    try:
        return run_unit(task, dataset, seed, run_dir, args)
    except subprocess.TimeoutExpired as exc:
        return {"task": task, "dataset": dataset, "seed": seed, "status": "timeout", "seconds": args.unit_timeout_s, "returncode": -999, "unit_dir": "", "stdout": "", "stderr": "", "error": str(exc)}
    except Exception as exc:
        return {"task": task, "dataset": dataset, "seed": seed, "status": "failed", "seconds": 0, "returncode": -1, "unit_dir": "", "stdout": "", "stderr": "", "error": repr(exc)}


def runnable_units(rows: list[dict], args) -> list[tuple[str, str, int]]:
    units = [(task, dataset, seed) for task, datasets in TASKS.items() for dataset in datasets for seed in SEEDS]
    if args.resume_mode == "missing":
        skip = seen_keys(rows)
    elif args.resume_mode == "ok":
        skip = completed_keys(rows)
    else:
        raise ValueError(f"Unsupported resume mode: {args.resume_mode}")
    return [u for u in units if u not in skip]


def run_sequential(units: list[tuple[str, str, int]], run_dir: Path, rows: list[dict], status_path: Path, args) -> None:
    for task, dataset, seed in units:
        if ledger_cost(run_dir / "token_ledger.jsonl") >= args.max_cost_usd:
            rows.append({"task": task, "dataset": dataset, "seed": seed, "status": "budget_stop", "seconds": 0, "returncode": "", "unit_dir": "", "stdout": "", "stderr": "", "error": "budget reached"})
            write_status(status_path, rows)
            break
        if not wait_for_api(args, run_dir, rows, status_path, (task, dataset, seed)):
            return
        rows.append(run_unit_safe(task, dataset, seed, run_dir, args))
        write_status(status_path, rows)


def run_parallel(units: list[tuple[str, str, int]], run_dir: Path, rows: list[dict], status_path: Path, args) -> None:
    next_idx = 0
    futures: dict[concurrent.futures.Future, tuple[str, str, int]] = {}

    def collect_one(executor: concurrent.futures.ThreadPoolExecutor, block: bool) -> bool:
        if not futures:
            return False
        timeout = None if block else 0
        done, _ = concurrent.futures.wait(futures.keys(), timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED)
        if not done:
            return False
        for fut in done:
            unit = futures.pop(fut)
            try:
                row = fut.result()
            except Exception as exc:
                task, dataset, seed = unit
                row = {"task": task, "dataset": dataset, "seed": seed, "status": "failed", "seconds": 0, "returncode": -1, "unit_dir": "", "stdout": "", "stderr": "", "error": repr(exc)}
            rows.append(row)
        write_status(status_path, rows)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        while next_idx < len(units) or futures:
            while next_idx < len(units) and len(futures) < args.workers:
                task, dataset, seed = units[next_idx]
                if ledger_cost(run_dir / "token_ledger.jsonl") >= args.max_cost_usd:
                    rows.append({"task": task, "dataset": dataset, "seed": seed, "status": "budget_stop", "seconds": 0, "returncode": "", "unit_dir": "", "stdout": "", "stderr": "", "error": "budget reached"})
                    write_status(status_path, rows)
                    next_idx = len(units)
                    break
                if not wait_for_api(args, run_dir, rows, status_path, (task, dataset, seed)):
                    next_idx = len(units)
                    break
                fut = executor.submit(run_unit_safe, task, dataset, seed, run_dir, args)
                futures[fut] = (task, dataset, seed)
                next_idx += 1
            collect_one(executor, block=True)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "outputs" / f"fig5_strict_fixed_pool_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = vars(args).copy()
    manifest["api_key"] = "present" if args.api_key else ""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path = run_dir / f"manifest_resume_{time.strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    status_path = run_dir / "run_status.csv"
    rows = load_existing(status_path)
    units = runnable_units(rows, args)
    if args.workers <= 1:
        run_sequential(units, run_dir, rows, status_path, args)
    else:
        run_parallel(units, run_dir, rows, status_path, args)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="")
    p.add_argument("--python", default=PYTHON)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("-l", "--llm", default="gpt-3.5-turbo")
    p.add_argument("-f", "--feat-iterations", default=10, type=int)
    p.add_argument("-m", "--model-iterations", default=10, type=int)
    p.add_argument("-p", "--param-iterations", default=5, type=int)
    p.add_argument("--unit-timeout-s", default=3600, type=int)
    p.add_argument("--max-cost-usd", default=20.0, type=float)
    p.add_argument("--prompt-price-per-m", default=1.5, type=float)
    p.add_argument("--completion-price-per-m", default=1.95, type=float)
    p.add_argument("--max-completion-tokens", default=300, type=int)
    p.add_argument("--feature-max-completion-tokens", default=220, type=int)
    p.add_argument("--max-messages", default=4, type=int)
    p.add_argument("--max-message-chars", default=3000, type=int)
    p.add_argument("--openai-timeout-s", default=90, type=int)
    p.add_argument("--max-feature-retries", default=3, type=int)
    p.add_argument("--max-retries-per-model", default=3, type=int)
    p.add_argument("--api-retry-sleep-s", default=300, type=int)
    p.add_argument("--workers", default=1, type=int)
    p.add_argument("--resume-mode", choices=["ok", "missing"], default="ok")
    p.add_argument("--archive-existing-unit-dir", action="store_true", default=False)
    p.add_argument("--wait-for-api", action="store_true", default=True)
    p.add_argument("--allow-fallback", action="store_true", default=False)
    args = p.parse_args()
    if not args.base_url or not args.api_key:
        raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY are required")
    return args


if __name__ == "__main__":
    main()
