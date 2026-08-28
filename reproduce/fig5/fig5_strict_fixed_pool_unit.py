import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    mean_absolute_error,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Candidate:
    code_id: str
    route: str
    model_index: int
    model: Any
    val_score: float
    val_aux: dict
    val_pred: Optional[np.ndarray] = None
    val_proba: Optional[np.ndarray] = None


def import_module_for_task(task: str):
    if task == "classification":
        module_path = ROOT / "autologic" / "ensemble" / "classification_ensemble" / "classification_auto_ensemble.py"
        sys.path.insert(0, str(ROOT / "autologic"))
        sys.path.insert(0, str(module_path.parent))
    elif task == "regression":
        module_path = ROOT / "autologic" / "ensemble" / "regression_ensemble" / "regression_auto_ensemble.py"
        sys.path.insert(0, str(ROOT / "autologic"))
        sys.path.insert(0, str(module_path.parent))
    elif task == "clustering":
        module_path = ROOT / "autologic" / "ensemble" / "cluster_ensemble" / "cluster_auto_ensemble.py"
        sys.path.insert(0, str(ROOT / "autologic"))
        sys.path.insert(0, str(module_path.parent))
    else:
        raise ValueError(task)

    spec = importlib.util.spec_from_file_location(f"strict_{task}_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_hot_positive(proba: Any) -> np.ndarray:
    arr = np.asarray(proba)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.shape[1] == 1:
        return arr[:, 0].astype(float)
    return arr[:, 1].astype(float)


def regression_rmse(y_true: Any, pred: Any) -> float:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y - p) ** 2)))


def better(task: str, left: float, right: float) -> bool:
    return left < right if task == "regression" else left > right


def metric_name(task: str) -> str:
    return {"classification": "auc", "regression": "rmse", "clustering": "ari"}[task]


def safe_auc(y_true: Any, proba: np.ndarray) -> float:
    y = np.asarray(y_true).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def evaluate_model(task: str, model: Any, x: Any, y: Any) -> tuple[float, dict, Optional[np.ndarray], Optional[np.ndarray]]:
    if task == "classification":
        proba = one_hot_positive(model.predict_proba(x))
        pred = np.asarray(proba >= 0.5).astype(int)
        score = safe_auc(y, proba)
        aux = {"acc": float(accuracy_score(np.asarray(y).ravel(), pred))}
        return score, aux, pred, proba
    if task == "regression":
        pred = np.asarray(model.predict(x)).ravel()
        score = regression_rmse(y, pred)
        aux = {"mae": float(mean_absolute_error(np.asarray(y).ravel(), pred))}
        return score, aux, pred, None
    pred = np.asarray(model.fit_predict(x)).ravel()
    score = float(adjusted_rand_score(np.asarray(y).ravel(), pred))
    aux = {"nmi": float(normalized_mutual_info_score(np.asarray(y).ravel(), pred))}
    return score, aux, pred, None


def compile_classification_or_regression(mod: Any, task: str, code: str, model_index: int, suffix: str):
    if task == "classification":
        class_name = f"myclassifier_{model_index}_{suffix}"
        code = re.sub(r"class\s+myclassifier[_\w]*\s*:", f"class {class_name}:", code, count=1)
        err = mod.code_exec(code)
        if err is not None:
            return None, code, str(err)
        if not hasattr(mod, class_name):
            return None, code, f"compiled code did not define {class_name}"
        return getattr(mod, class_name), code, ""

    class_name = f"myregressor_{model_index}_{suffix}"
    code = re.sub(r"class\s+myregressor[_\w]*\s*:", f"class {class_name}:", code, count=1)
    err = mod.code_exec(code)
    if err is not None:
        return None, code, str(err)
    if not hasattr(mod, class_name):
        return None, code, f"compiled code did not define {class_name}"
    return getattr(mod, class_name), code, ""


def compile_clustering(mod: Any, code: str, model_index: int, suffix: str):
    class_name = f"mycluster_{model_index}_{suffix}"
    code = re.sub(r"class\s+mycluster[_\w]*\s*(\([^)]*\))?\s*:", f"class {class_name}:", code, count=1)
    err, scope = mod.code_exec(code)
    if err is not None or scope is None:
        return None, code, str(err)
    if class_name not in scope:
        return None, code, f"compiled code did not define {class_name}"
    return scope[class_name], code, ""


def instantiate_cluster(mod: Any, cls: Any, n_clusters: int):
    if hasattr(mod, "_instantiate_cluster_model"):
        return mod._instantiate_cluster_model(cls, n_clusters)
    try:
        return cls(n_clusters=n_clusters)
    except TypeError:
        return cls(n_clusters)


def fit_candidate(task: str, mod: Any, cls: Any, route: dict, seed: int, n_clusters: Optional[int] = None):
    model = instantiate_cluster(mod, cls, int(n_clusters)) if task == "clustering" else cls()
    if task != "clustering":
        model.fit(route["x_train"], route["y_train"])
    score, aux, pred, proba = evaluate_model(task, model, route["x_val"], route["y_val"])
    if not np.isfinite(score):
        raise ValueError(f"non-finite validation score: {score}")
    return model, score, aux, pred, proba


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_supervised_data(task: str, mod: Any, dataset: str, seed: int, feat_iterations: int, llm: str, base_url: str, api_key: str):
    if task == "classification":
        df_train_all, df_test, target, desc = mod.load_origin_data(dataset, seed)
        strat = df_train_all[target] if df_train_all[target].nunique() > 1 else None
        df_train, df_val = train_test_split(df_train_all, test_size=0.25, random_state=seed, stratify=strat)
        base = mod.base_model(seed)
    else:
        loc = ROOT / "AutoLogic" / "data" / f"{dataset}.pkl"
        df_train_all, df_test, target, desc = mod.load_origin_data(str(loc), seed)
        df_train, df_val = train_test_split(df_train_all, test_size=0.25, random_state=seed)
        base = mod.base_model(seed)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    raw_train_x, raw_train_y = mod.to_pd(df_train, target)
    raw_val_x, raw_val_y = mod.to_pd(df_val, target)
    raw_test_x, raw_test_y = mod.to_pd(df_test, target)
    routes = {
        "raw": {
            "x_train": raw_train_x,
            "y_train": raw_train_y,
            "x_val": raw_val_x,
            "y_val": raw_val_y,
            "x_test": raw_test_x,
            "y_test": raw_test_y,
        }
    }

    if feat_iterations > 0:
        valtest = pd.concat([df_val, df_test], axis=0, ignore_index=True)
        if task == "classification":
            gen_train, gen_valtest = mod.generate_feat(
                base_classifier=base,
                df_train=df_train,
                df_test=valtest,
                dataset_name=dataset,
                round_num=1,
                llm_model=llm,
                iterations=feat_iterations,
                target_column_name=target,
                dataset_description=desc,
                task_type="classification",
                base_url=base_url,
                api_key=api_key,
            )
        else:
            gen_train, gen_valtest = mod.generate_feat(
                base_model=base,
                df_train=df_train,
                df_test=valtest,
                dataset_name=dataset,
                round_num=1,
                llm_model=llm,
                iterations=feat_iterations,
                target_column_name=target,
                dataset_description=desc,
                task_type="regression",
                base_url=base_url,
                api_key=api_key,
            )
        gen_val = gen_valtest.iloc[: len(df_val)].reset_index(drop=True)
        gen_test = gen_valtest.iloc[len(df_val) :].reset_index(drop=True)
        gen_train = gen_train.reset_index(drop=True)
        gen_train_x, gen_train_y = mod.to_pd(gen_train, target)
        gen_val_x, gen_val_y = mod.to_pd(gen_val, target)
        gen_test_x, gen_test_y = mod.to_pd(gen_test, target)
        routes["generated"] = {
            "x_train": gen_train_x,
            "y_train": gen_train_y,
            "x_val": gen_val_x,
            "y_val": gen_val_y,
            "x_test": gen_test_x,
            "y_test": gen_test_y,
        }

    samples = mod.build_prompt_samples(df_train)
    prompt = mod.get_model_prompt(target_column_name=target, samples=samples) if task == "classification" else mod.get_regression_model_prompt(target_column_name=target, samples=samples)
    return routes, prompt, desc, target


def prepare_clustering_data(mod: Any, dataset: str, seed: int, feat_iterations: int, llm: str, base_url: str, api_key: str):
    loc = Path(os.environ.get("AUTOLOGIC_DATA_DIR", ROOT / "data" / "pkl")) / f"{dataset}.pkl"
    df, n_clusters, target, desc = mod.load_origin_data(str(loc))
    df = df.reset_index(drop=True)
    x_raw, y = mod.to_pd(df, target)
    idx = np.arange(len(df))
    val_idx, test_idx = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None)

    routes = {
        "raw": {
            "x_val": x_raw.iloc[val_idx].reset_index(drop=True),
            "y_val": pd.Series(np.asarray(y)[val_idx]),
            "x_test": x_raw.iloc[test_idx].reset_index(drop=True),
            "y_test": pd.Series(np.asarray(y)[test_idx]),
        }
    }

    if feat_iterations > 0:
        x_gen, y_gen = mod.generate_feat(
            base_model=mod.base_model(n_clusters, seed),
            df=df,
            dataset_name=dataset,
            round_num=1,
            llm_model=llm,
            iterations=feat_iterations,
            target_column_name=target,
            dataset_description=desc,
            task_type="clustering",
            base_url=base_url,
            api_key=api_key,
            logger=None,
        )
        routes["generated"] = {
            "x_val": x_gen.iloc[val_idx].reset_index(drop=True),
            "y_val": pd.Series(np.asarray(y_gen)[val_idx]),
            "x_test": x_gen.iloc[test_idx].reset_index(drop=True),
            "y_test": pd.Series(np.asarray(y_gen)[test_idx]),
        }

    samples = mod.build_prompt_samples(routes["raw"]["x_val"])
    prompt = mod.get_clustering_model_prompt(samples=samples, n_clusters=n_clusters)
    return routes, prompt, desc, target, int(n_clusters)


def make_model_messages(task: str, prompt: str) -> list[dict]:
    if task == "classification":
        role = "top-level machine learning classification expert"
        goal = "maximize AUC"
        class_rule = "complete classifier named `myclassifier` with fit, predict, and predict_proba"
    elif task == "regression":
        role = "top-level regression algorithm expert"
        goal = "minimize RMSE"
        class_rule = "complete regressor named `myregressor` with fit and predict"
    else:
        role = "top-level clustering algorithm expert"
        goal = "maximize ARI"
        class_rule = "complete clustering model named `mycluster` with fit and fit_predict"
    return [
        {
            "role": "system",
            "content": (
                f"You are a {role}. Your task is to generate diverse, executable sklearn-compatible models. "
                f"The primary goal is to {goal} on a validation split. Output Python code only. "
                "Use all available CPU cores when relevant, keep hyperparameters efficient, and avoid deprecated sklearn APIs. "
                f"The code must define a {class_rule}."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def param_messages_for(task: str, code: str, score: float, desc: str, feature_columns: list[str]) -> list[dict]:
    if task == "classification":
        prompt = f"Current validation AUC: {score:.5f}. Optimize hyperparameters only for this classifier. Keep the same class name myclassifier.\n```python\n{code}\n```"
        system = "You tune classifier hyperparameters. Output executable Python code only."
    elif task == "regression":
        prompt = f"Current validation RMSE: {score:.5f}. Optimize hyperparameters only for this regressor. Keep the same class name myregressor.\n```python\n{code}\n```"
        system = "You tune regressor hyperparameters. Output executable Python code only."
    else:
        prompt = f"Current validation ARI: {score:.5f}. Optimize hyperparameters only for this clustering model. Keep the same class name mycluster.\n```python\n{code}\n```"
        system = "You tune clustering hyperparameters. Output executable Python code only."
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def generate_candidate_pool(task: str, mod: Any, routes: dict, prompt: str, desc: str, args, n_clusters: Optional[int], event_log: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    messages = make_model_messages(task, prompt)
    best_score = math.inf if task == "regression" else -math.inf
    best_code = ""
    model_count = 0
    attempts = 0
    max_attempts = max(args.model_iterations * args.max_retries_per_model, args.model_iterations)

    while model_count < args.model_iterations and attempts < max_attempts:
        attempts += 1
        try:
            if task == "clustering":
                code = mod.generate_model(args.llm, messages, args.base_url, args.api_key)
            else:
                code = mod.generate_model_2(args.llm, messages, args.base_url, args.api_key)["code"]
            code = mod.clean_llm_code(code)
        except BaseException as exc:
            append_jsonl(event_log, {"stage": "model_generate_error", "attempt": attempts, "error": repr(exc)})
            time.sleep(min(20, 3 * attempts))
            continue

        model_index = model_count + 1
        compiled = compile_clustering(mod, code, model_index, "base") if task == "clustering" else compile_classification_or_regression(mod, task, code, model_index, "base")
        cls, renamed_code, err = compiled
        if cls is None:
            append_jsonl(event_log, {"stage": "compile_error", "model_index": model_index, "error": err, "code_head": code[:800]})
            messages += [{"role": "assistant", "content": code}, {"role": "user", "content": f"Compilation failed: {err}. Fix the code. Output code only."}]
            continue

        model_count += 1
        route_results = []
        for route_name, route in routes.items():
            try:
                model, score, aux, pred, proba = fit_candidate(task, mod, cls, route, args.seed, n_clusters=n_clusters)
                cand = Candidate(f"m{model_index}_base", route_name, model_index, model, score, aux, pred, proba)
                candidates.append(cand)
                route_results.append((route_name, score))
            except Exception as exc:
                append_jsonl(event_log, {"stage": "fit_error", "code_id": f"m{model_index}_base", "route": route_name, "error": repr(exc)})

        local_best = min([x[1] for x in route_results], default=math.inf) if task == "regression" else max([x[1] for x in route_results], default=-math.inf)
        if np.isfinite(local_best) and better(task, local_best, best_score):
            best_score = local_best
            best_code = renamed_code

        param_messages = param_messages_for(task, renamed_code, local_best if np.isfinite(local_best) else best_score, desc, [])
        for p_iter in range(args.param_iterations):
            try:
                if task == "clustering":
                    p_code = mod.generate_model(args.llm, param_messages, args.base_url, args.api_key)
                else:
                    p_code = mod.generate_model_2(args.llm, param_messages, args.base_url, args.api_key)["code"]
                p_code = mod.clean_llm_code(p_code)
            except BaseException as exc:
                append_jsonl(event_log, {"stage": "param_generate_error", "model_index": model_index, "p_iter": p_iter + 1, "error": repr(exc)})
                time.sleep(min(20, 2 * (p_iter + 1)))
                continue
            compiled = compile_clustering(mod, p_code, model_index, f"p{p_iter + 1}") if task == "clustering" else compile_classification_or_regression(mod, task, p_code, model_index, f"p{p_iter + 1}")
            p_cls, p_renamed_code, p_err = compiled
            if p_cls is None:
                append_jsonl(event_log, {"stage": "param_compile_error", "model_index": model_index, "p_iter": p_iter + 1, "error": p_err})
                param_messages += [{"role": "assistant", "content": p_code}, {"role": "user", "content": f"Compilation failed: {p_err}. Fix it. Output code only."}]
                continue
            p_scores = []
            for route_name, route in routes.items():
                try:
                    model, score, aux, pred, proba = fit_candidate(task, mod, p_cls, route, args.seed, n_clusters=n_clusters)
                    cand = Candidate(f"m{model_index}_p{p_iter + 1}", route_name, model_index, model, score, aux, pred, proba)
                    candidates.append(cand)
                    p_scores.append(score)
                except Exception as exc:
                    append_jsonl(event_log, {"stage": "param_fit_error", "code_id": f"m{model_index}_p{p_iter + 1}", "route": route_name, "error": repr(exc)})
            p_best = min(p_scores, default=math.inf) if task == "regression" else max(p_scores, default=-math.inf)
            if np.isfinite(p_best) and better(task, p_best, best_score):
                best_score = p_best
                best_code = p_renamed_code
            param_messages += [{"role": "assistant", "content": p_renamed_code}, {"role": "user", "content": f"Current validation {metric_name(task)}: {p_best:.5f}; best: {best_score:.5f}. Improve further by hyperparameters only."}]

        if args.enable_feedback and best_code:
            messages += [
                {"role": "assistant", "content": best_code},
                {"role": "user", "content": f"The model executed. Best validation {metric_name(task)} so far is {best_score:.5f}. Propose a different model type or structure. Output code only."},
            ]

    return candidates


def add_fallback_candidates(task: str, mod: Any, routes: dict, seed: int, n_clusters: Optional[int], event_log: Path) -> list[Candidate]:
    if task == "classification":
        model_factories = [
            lambda: RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1),
            lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    elif task == "regression":
        model_factories = [
            lambda: RandomForestRegressor(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1),
            lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        ]
    else:
        model_factories = [lambda m=m: copy.deepcopy(m) for m in mod.fallback_models(int(n_clusters), seed)]

    out: list[Candidate] = []
    for model_index, factory in enumerate(model_factories, start=1):
        for route_name, route in routes.items():
            try:
                model = factory()
                if task != "clustering":
                    model.fit(route["x_train"], route["y_train"])
                score, aux, pred, proba = evaluate_model(task, model, route["x_val"], route["y_val"])
                out.append(Candidate(f"fallback{model_index}", route_name, model_index, model, score, aux, pred, proba))
            except Exception as exc:
                append_jsonl(event_log, {"stage": "fallback_error", "model_index": model_index, "route": route_name, "error": repr(exc)})
    append_jsonl(event_log, {"stage": "fallback_added", "candidate_count": len(out)})
    return out


def select_option(task: str, candidates: list[Candidate], allow_routes: set[str], allow_closed_loop: bool, allow_meta: bool) -> dict:
    pool = [c for c in candidates if c.route in allow_routes and (allow_closed_loop or c.model_index <= 1)]
    if not pool:
        return {"kind": "none", "val_score": None, "members": []}

    options = [{"kind": "single", "val_score": c.val_score, "members": [c]} for c in pool]
    if allow_meta and task in {"classification", "regression"} and len(pool) >= 2:
        ranked = sorted(pool, key=lambda c: c.val_score, reverse=(task != "regression"))
        for k in range(2, min(5, len(ranked)) + 1):
            members = ranked[:k]
            if task == "classification":
                val_proba = np.mean([m.val_proba for m in members if m.val_proba is not None], axis=0)
                score = safe_auc(members[0].val_aux.get("_y_val", []), val_proba) if "_y_val" in members[0].val_aux else float("nan")
                if not np.isfinite(score):
                    y_val = getattr(members[0].model, "_strict_y_val", None)
                    score = safe_auc(y_val, val_proba) if y_val is not None else float("nan")
            else:
                y_val = getattr(members[0].model, "_strict_y_val", None)
                val_pred = np.mean([m.val_pred for m in members if m.val_pred is not None], axis=0)
                score = regression_rmse(y_val, val_pred) if y_val is not None else float("nan")
            if np.isfinite(score):
                options.append({"kind": f"ensemble_top{k}", "val_score": float(score), "members": members})

    return min(options, key=lambda o: o["val_score"]) if task == "regression" else max(options, key=lambda o: o["val_score"])


def attach_y_val_to_models(candidates: list[Candidate], routes: dict) -> None:
    for c in candidates:
        try:
            c.model._strict_y_val = np.asarray(routes[c.route]["y_val"]).ravel()
            c.val_aux["_y_val"] = np.asarray(routes[c.route]["y_val"]).ravel().tolist()
        except Exception:
            pass


def evaluate_selected(task: str, option: dict, routes: dict) -> dict:
    if not option["members"]:
        return {"status": "failed", "error": "no selected members"}
    members: list[Candidate] = option["members"]
    route = members[0].route
    x_test = routes[route]["x_test"]
    y_test = routes[route]["y_test"]

    if task == "classification":
        probs = [one_hot_positive(m.model.predict_proba(routes[m.route]["x_test"])) for m in members]
        proba = np.mean(probs, axis=0)
        pred = (proba >= 0.5).astype(int)
        return {
            "status": "ok",
            "metric": "auc",
            "test_score": safe_auc(y_test, proba),
            "acc": float(accuracy_score(np.asarray(y_test).ravel(), pred)),
        }
    if task == "regression":
        preds = [np.asarray(m.model.predict(routes[m.route]["x_test"])).ravel() for m in members]
        pred = np.mean(preds, axis=0)
        return {
            "status": "ok",
            "metric": "rmse",
            "test_score": regression_rmse(y_test, pred),
            "mae": float(mean_absolute_error(np.asarray(y_test).ravel(), pred)),
        }
    # Clustering options are single-model only here, to avoid selecting a consensus on test labels.
    model = copy.deepcopy(members[0].model)
    pred = np.asarray(model.fit_predict(x_test)).ravel()
    return {
        "status": "ok",
        "metric": "ari",
        "test_score": float(adjusted_rand_score(np.asarray(y_test).ravel(), pred)),
        "nmi": float(normalized_mutual_info_score(np.asarray(y_test).ravel(), pred)),
    }


def run_unit(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    mod = import_module_for_task(args.task)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_log = out_dir / "events.jsonl"

    if args.task == "clustering":
        routes, prompt, desc, target, n_clusters = prepare_clustering_data(mod, args.dataset, args.seed, args.feat_iterations, args.llm, args.base_url, args.api_key)
    else:
        routes, prompt, desc, target = prepare_supervised_data(args.task, mod, args.dataset, args.seed, args.feat_iterations, args.llm, args.base_url, args.api_key)
        n_clusters = None

    candidates = generate_candidate_pool(args.task, mod, routes, prompt, desc, args, n_clusters, event_log)
    if not candidates and args.allow_fallback:
        candidates = add_fallback_candidates(args.task, mod, routes, args.seed, n_clusters, event_log)
    attach_y_val_to_models(candidates, routes)

    conditions = {
        "all_select": {"routes": set(routes), "closed_loop": True, "meta": True},
        "wo_feature": {"routes": {"raw"}, "closed_loop": True, "meta": True},
        "wo_closed_loop": {"routes": set(routes), "closed_loop": False, "meta": True},
        "wo_meta": {"routes": set(routes), "closed_loop": True, "meta": False},
    }
    rows = []
    for cond, cfg in conditions.items():
        selected = select_option(args.task, candidates, cfg["routes"], cfg["closed_loop"], cfg["meta"])
        result = evaluate_selected(args.task, selected, routes)
        rows.append(
            {
                "task": args.task,
                "dataset": args.dataset,
                "seed": args.seed,
                "condition": cond,
                "status": result.get("status"),
                "metric": result.get("metric", metric_name(args.task)),
                "test_score": result.get("test_score"),
                "acc": result.get("acc"),
                "mae": result.get("mae"),
                "nmi": result.get("nmi"),
                "val_score": selected.get("val_score"),
                "selected_kind": selected.get("kind"),
                "selected_routes": ";".join(sorted({m.route for m in selected.get("members", [])})),
                "selected_members": ";".join(m.code_id for m in selected.get("members", [])),
                "candidate_count": len(candidates),
                "route_count": len(routes),
                "error": result.get("error", ""),
            }
        )

    csv_path = out_dir / "strict_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "strict_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"rows": rows, "csv": str(csv_path)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["classification", "regression", "clustering"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("-f", "--feat-iterations", default=10, type=int)
    p.add_argument("-m", "--model-iterations", default=10, type=int)
    p.add_argument("-p", "--param-iterations", default=5, type=int)
    p.add_argument("-l", "--llm", default="gpt-3.5-turbo")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-retries-per-model", default=3, type=int)
    p.add_argument("--enable-feedback", action="store_true", default=True)
    p.add_argument("--allow-fallback", action="store_true", default=False)
    args = p.parse_args()
    if not args.base_url or not args.api_key:
        raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY are required")
    return args


if __name__ == "__main__":
    started = time.time()
    parsed = parse_args()
    try:
        payload = run_unit(parsed)
        payload["seconds"] = round(time.time() - started, 3)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except BaseException as exc:
        out = Path(parsed.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        err = {"status": "failed", "error_type": type(exc).__name__, "error": repr(exc), "seconds": round(time.time() - started, 3)}
        (out / "unit_error.json").write_text(json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
