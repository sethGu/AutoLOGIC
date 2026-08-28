from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs" / "fig6c_delivery_fidelity"
OUT = ROOT / "outputs" / "fig6c_plots"
MAIN_SUB = OUT / "main_panel_c_subfigures"
EXT = OUT / "extended_data"


DATASET_ORDER = ["cc1", "credit-g", "ld1", "boston", "concrete", "california"]
CLASS_DATASETS = ["cc1", "credit-g", "ld1"]
REG_DATASETS = ["boston", "concrete", "california"]
DISPLAY = {
    "cc1": "CC1",
    "credit-g": "Credit-g",
    "ld1": "LD1",
    "boston": "Boston",
    "concrete": "Concrete",
    "california": "California",
}

COL_TEACHER = "#202936"
COL_CLASS = "#2176AE"
COL_REG = "#C65A2E"
COL_CLASS_LIGHT = "#DDECF5"
COL_REG_LIGHT = "#F4E2DA"
COL_GRID = "#E6E8EB"
COL_TEXT = "#1B1F24"
COL_MUTED = "#6B7280"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 160,
            "savefig.dpi": 450,
        }
    )


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN_SUB.mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)


def read_data():
    metrics = pd.read_csv(SOURCE / "per_run_metrics.csv")
    cpred = pd.read_csv(SOURCE / "classification_predictions.csv")
    rpred = pd.read_csv(SOURCE / "regression_predictions.csv")
    batch = pd.read_csv(SOURCE / "batch_command_records.csv")
    for df in [metrics, cpred, rpred, batch]:
        for col in df.columns:
            if col not in {"task", "dataset", "status", "stdout_log", "stderr_log", "error"}:
                df[col] = pd.to_numeric(df[col], errors="ignore")
    metrics = metrics[metrics["status"].eq("success")].copy()
    return metrics, cpred, rpred, batch


def save_fig(fig: plt.Figure, stem: Path) -> None:
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(stem.with_suffix(ext), bbox_inches="tight", facecolor="white")


def interp_seed_curve(group: pd.DataFrame, task: str, n_grid: int = 151):
    grid = np.linspace(0, 1, n_grid)
    curves_t = []
    curves_s = []
    curves_d = []
    for _, g in group.groupby("seed"):
        if task == "classification":
            t = g["teacher_prob"].to_numpy(float)
            s = g["student_prob"].to_numpy(float)
        else:
            t0 = g["teacher_pred"].to_numpy(float)
            s0 = g["student_pred"].to_numpy(float)
            mu = float(np.mean(t0))
            sd = float(np.std(t0))
            if sd < 1e-12:
                sd = 1.0
            t = (t0 - mu) / sd
            s = (s0 - mu) / sd
        order = np.argsort(t)
        t = t[order]
        s = s[order]
        xp = np.linspace(0, 1, len(t))
        ti = np.interp(grid, xp, t)
        si = np.interp(grid, xp, s)
        curves_t.append(ti)
        curves_s.append(si)
        curves_d.append(np.abs(si - ti))
    return grid, np.vstack(curves_t), np.vstack(curves_s), np.vstack(curves_d)


def mean_sd(values: pd.Series):
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def dataset_metric_summary(metrics: pd.DataFrame, dataset: str, task: str):
    g = metrics[metrics["dataset"].eq(dataset)]
    if task == "classification":
        fid = g["fidelity_prob_pearson"].mean()
        delta = g["auc_delta_student_minus_teacher"].mean()
        delta_label = f"dAUC {delta:+.3f}"
    else:
        fid = g["fidelity_pred_pearson"].mean()
        delta = g["r2_delta_student_minus_teacher"].mean()
        delta_label = f"dR2 {delta:+.3f}"
    speed = 1.0 / g["latency_ratio_student_over_teacher"].mean()
    size = 1.0 / g["artifact_size_ratio_student_over_teacher"].mean()
    return fid, delta_label, speed, size


def add_deviation_strip(ax: plt.Axes, grid: np.ndarray, dev: np.ndarray, color: str) -> None:
    y0, y1 = ax.get_ylim()
    height = (y1 - y0) * 0.045
    base = y0 + (y1 - y0) * 0.02
    cmap = LinearSegmentedColormap.from_list("dev", ["#FFFFFF", color])
    vmax = float(np.nanpercentile(dev, 95))
    vmax = vmax if vmax > 0 else 1.0
    ax.imshow(
        dev[np.newaxis, :],
        extent=[0, 100, base, base + height],
        aspect="auto",
        interpolation="bicubic",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        alpha=0.95,
        zorder=0,
    )
    ax.text(1.5, base + height * 1.18, "|diff|", color=COL_MUTED, fontsize=5.6, ha="left", va="bottom")


def plot_rank_panel(ax: plt.Axes, dataset: str, task: str, metrics: pd.DataFrame, pred: pd.DataFrame) -> None:
    g = pred[pred["dataset"].eq(dataset)].copy()
    grid, ct, cs, cd = interp_seed_curve(g, task)
    x = grid * 100.0
    tm = ct.mean(axis=0)
    sm = cs.mean(axis=0)
    tlo, thi = np.percentile(ct, [20, 80], axis=0)
    slo, shi = np.percentile(cs, [20, 80], axis=0)
    dm = cd.mean(axis=0)
    color = COL_CLASS if task == "classification" else COL_REG
    light = COL_CLASS_LIGHT if task == "classification" else COL_REG_LIGHT

    ax.axvspan(90, 100, color="#EEF0F3", alpha=0.9, lw=0, zorder=0)
    ax.fill_between(x, tlo, thi, color="#BFC5CE", alpha=0.22, lw=0, zorder=1)
    ax.fill_between(x, slo, shi, color=light, alpha=0.86, lw=0, zorder=1)
    ax.fill_between(x, tm, sm, color=color, alpha=0.10, lw=0, zorder=2)
    ax.plot(x, tm, color=COL_TEACHER, lw=1.35, zorder=3)
    ax.plot(x, sm, color=color, lw=1.35, zorder=4)

    if task == "classification":
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.5, 1.0])
        ylabel = "Prob."
    else:
        allv = np.concatenate([ct.ravel(), cs.ravel()])
        lo, hi = np.percentile(allv, [1, 99])
        pad = max(0.2, (hi - lo) * 0.10)
        ax.set_ylim(lo - pad, hi + pad)
        ylabel = "Std. pred."
    add_deviation_strip(ax, x, dm, color)

    fid, delta_label, speed, size = dataset_metric_summary(metrics, dataset, task)
    ax.text(
        0.03,
        0.96,
        f"r {fid:.2f} | {delta_label}\n{speed:.1f}x spd | {size:.1f}x sz",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=COL_TEXT,
        linespacing=1.15,
    )
    ax.text(95, ax.get_ylim()[1], "top 10%", ha="center", va="top", fontsize=5.8, color=COL_MUTED)
    ax.set_title(DISPLAY[dataset], loc="left", pad=2, color=COL_TEXT, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.grid(axis="y", color=COL_GRID, lw=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=COL_MUTED)
    ax.spines["left"].set_color("#B8BEC7")
    ax.spines["bottom"].set_color("#B8BEC7")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Teacher rank (%)")


def main_panel(metrics: pd.DataFrame, cpred: pd.DataFrame, rpred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.35), constrained_layout=True)
    for ax, ds in zip(axes[0], CLASS_DATASETS):
        plot_rank_panel(ax, ds, "classification", metrics, cpred)
    for ax, ds in zip(axes[1], REG_DATASETS):
        plot_rank_panel(ax, ds, "regression", metrics, rpred)
    for j in range(3):
        axes[0, j].set_xlabel("")
        axes[0, j].set_xticklabels([])
    for i in range(2):
        for j in [1, 2]:
            axes[i, j].set_ylabel("")
            axes[i, j].set_yticklabels([])
    handles = [
        Line2D([0], [0], color=COL_TEACHER, lw=1.7, label="Teacher"),
        Line2D([0], [0], color=COL_CLASS, lw=1.7, label="Student, cls."),
        Line2D([0], [0], color=COL_REG, lw=1.7, label="Student, reg."),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.075))
    save_fig(fig, OUT / "fig6_panel_c_main_2x3_ranked_output_preservation")
    plt.close(fig)

    for ds in CLASS_DATASETS:
        fig, ax = plt.subplots(figsize=(2.55, 2.15))
        plot_rank_panel(ax, ds, "classification", metrics, cpred)
        save_fig(fig, MAIN_SUB / f"panel_c_{ds.replace('-', '_')}")
        plt.close(fig)
    for ds in REG_DATASETS:
        fig, ax = plt.subplots(figsize=(2.55, 2.15))
        plot_rank_panel(ax, ds, "regression", metrics, rpred)
        save_fig(fig, MAIN_SUB / f"panel_c_{ds}")
        plt.close(fig)


def plot_per_seed_retention(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), constrained_layout=True)
    specs = [
        (axes[0], "classification", CLASS_DATASETS, "auc_delta_student_minus_teacher", "dAUC", COL_CLASS, (-0.105, 0.025)),
        (axes[1], "regression", REG_DATASETS, "r2_delta_student_minus_teacher", "dR2", COL_REG, (-0.085, 0.015)),
    ]
    rng = np.random.default_rng(12)
    for ax, task, datasets, col, xlabel, color, xlim in specs:
        yy = np.arange(len(datasets))
        for i, ds in enumerate(datasets):
            g = metrics[(metrics["task"].eq(task)) & (metrics["dataset"].eq(ds))]
            vals = g[col].to_numpy(float)
            jitter = rng.normal(0, 0.045, size=len(vals))
            ax.scatter(vals, i + jitter, s=18, color=color, edgecolor="white", linewidth=0.45, alpha=0.86, zorder=3)
            mean = vals.mean()
            sd = vals.std(ddof=1)
            ax.errorbar(mean, i, xerr=sd, fmt="o", ms=4.2, color=COL_TEACHER, ecolor=COL_TEACHER, elinewidth=1.0, capsize=2.5, zorder=4)
        ax.axvline(0, color="#7D8590", lw=0.9, ls="--", zorder=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([DISPLAY[d] for d in datasets])
        ax.set_xlabel(xlabel + " (student - teacher)")
        ax.set_xlim(*xlim)
        ax.invert_yaxis()
        ax.grid(axis="x", color=COL_GRID, lw=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("Performance retention: " + ("classification" if task == "classification" else "regression"), loc="left", fontweight="bold")
    save_fig(fig, EXT / "edfig_c1_per_seed_performance_retention")
    plt.close(fig)


def plot_delivery_map(metrics: pd.DataFrame) -> None:
    m = metrics.copy()
    m["speedup"] = 1.0 / m["latency_ratio_student_over_teacher"].astype(float)
    m["size_reduction"] = 1.0 / m["artifact_size_ratio_student_over_teacher"].astype(float)
    m["perf_loss"] = 0.0
    cls = m["task"].eq("classification")
    reg = m["task"].eq("regression")
    m.loc[cls, "perf_loss"] = np.maximum(0, -m.loc[cls, "auc_delta_student_minus_teacher"].astype(float)) * 100.0
    m.loc[reg, "perf_loss"] = np.maximum(0, -m.loc[reg, "r2_delta_student_minus_teacher"].astype(float)) * 100.0
    fig, ax = plt.subplots(figsize=(4.6, 3.45), constrained_layout=True)
    markers = {
        "cc1": "o",
        "credit-g": "s",
        "ld1": "^",
        "boston": "o",
        "concrete": "s",
        "california": "^",
    }
    label_offsets = {
        "cc1": (0.95, 0.88),
        "credit-g": (1.05, 0.82),
        "ld1": (0.97, 1.10),
        "boston": (1.03, 0.82),
        "concrete": (0.83, 1.08),
        "california": (0.86, 1.09),
    }
    for ds in DATASET_ORDER:
        g = m[m["dataset"].eq(ds)]
        task = g["task"].iloc[0]
        color = COL_CLASS if task == "classification" else COL_REG
        sizes = 26 + 8.0 * g["perf_loss"].to_numpy(float)
        ax.scatter(
            g["size_reduction"],
            g["speedup"],
            s=sizes,
            marker=markers[ds],
            color=color,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.82,
            label=DISPLAY[ds],
        )
        ox, oy = label_offsets[ds]
        ax.text(g["size_reduction"].mean() * ox, g["speedup"].mean() * oy, DISPLAY[ds], fontsize=6.0, color=COL_TEXT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1, color="#9AA2AE", lw=0.8, ls="--")
    ax.axhline(1, color="#9AA2AE", lw=0.8, ls="--")
    ax.set_xlabel("Size reduction (teacher / student)")
    ax.set_ylabel("Speed-up (teacher / student)")
    ax.set_title("Delivery compression map", loc="left", fontweight="bold")
    ax.grid(which="both", color=COL_GRID, lw=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    proxy = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_CLASS, markeredgecolor="white", label="Classification", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_REG, markeredgecolor="white", label="Regression", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#999999", label="Larger = more perf. loss", markersize=8),
    ]
    ax.legend(handles=proxy, frameon=False, loc="lower right")
    save_fig(fig, EXT / "edfig_c2_delivery_compression_map")
    plt.close(fig)


def plot_fidelity_matrix(metrics: pd.DataFrame) -> None:
    rows = []
    for ds in DATASET_ORDER:
        g = metrics[metrics["dataset"].eq(ds)]
        task = g["task"].iloc[0]
        if task == "classification":
            rows.append((ds, "Corr", g["fidelity_prob_pearson"].to_numpy(float), COL_CLASS, (0.74, 1.01)))
            rows.append((ds, "Top10", g["fidelity_top10_overlap"].to_numpy(float), COL_CLASS, (0.45, 1.01)))
            rows.append((ds, "Err", g["fidelity_prob_mae"].to_numpy(float), COL_CLASS, (0.0, 0.32)))
        else:
            rows.append((ds, "Corr", g["fidelity_pred_pearson"].to_numpy(float), COL_REG, (0.74, 1.01)))
            rows.append((ds, "Top10", g["teacher_student_top10_pred_overlap"].to_numpy(float), COL_REG, (0.45, 1.01)))
            rows.append((ds, "Err", g["fidelity_pred_nrmse_y_std"].to_numpy(float), COL_REG, (0.0, 0.32)))

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 3.8), constrained_layout=True)
    metrics_names = ["Corr", "Top10", "Err"]
    rng = np.random.default_rng(4)
    for ax, met in zip(axes, metrics_names):
        ax.set_title(met + (" (low)" if met == "Err" else ""), loc="left", fontweight="bold")
        for i, ds in enumerate(DATASET_ORDER):
            item = [r for r in rows if r[0] == ds and r[1] == met][0]
            vals, color, xlim = item[2], item[3], item[4]
            y = len(DATASET_ORDER) - 1 - i
            jit = rng.normal(0, 0.045, len(vals))
            ax.scatter(vals, y + jit, s=15, color=color, edgecolor="white", linewidth=0.4, alpha=0.82)
            ax.errorbar(vals.mean(), y, xerr=vals.std(ddof=1), fmt="o", ms=4, color=COL_TEACHER, ecolor=COL_TEACHER, capsize=2.2, lw=1.0)
            ax.set_xlim(*xlim)
        ax.set_yticks(np.arange(len(DATASET_ORDER)))
        ax.set_yticklabels([DISPLAY[d] for d in DATASET_ORDER[::-1]] if ax is axes[0] else [])
        ax.grid(axis="x", color=COL_GRID, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if met == "Err":
            ax.invert_xaxis()
            ax.set_xlabel("lower is better")
        else:
            ax.set_xlabel("higher is better")
    save_fig(fig, EXT / "edfig_c3_fidelity_metric_matrix")
    plt.close(fig)


def prepare_density_data(dataset: str, task: str, cpred: pd.DataFrame, rpred: pd.DataFrame):
    if task == "classification":
        g = cpred[cpred["dataset"].eq(dataset)].copy()
        return g["teacher_prob"].to_numpy(float), g["student_prob"].to_numpy(float), (0, 1), (0, 1)
    g = rpred[rpred["dataset"].eq(dataset)].copy()
    xs = []
    ys = []
    for _, gg in g.groupby("seed"):
        t = gg["teacher_pred"].to_numpy(float)
        s = gg["student_pred"].to_numpy(float)
        mu = t.mean()
        sd = t.std()
        if sd < 1e-12:
            sd = 1.0
        xs.append((t - mu) / sd)
        ys.append((s - mu) / sd)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    lim = np.percentile(np.concatenate([x, y]), [0.5, 99.5])
    pad = (lim[1] - lim[0]) * 0.08
    return x, y, (lim[0] - pad, lim[1] + pad), (lim[0] - pad, lim[1] + pad)


def plot_density(metrics: pd.DataFrame, cpred: pd.DataFrame, rpred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.45), constrained_layout=True)
    for ax, ds in zip(axes.ravel(), DATASET_ORDER):
        task = "classification" if ds in CLASS_DATASETS else "regression"
        x, y, xlim, ylim = prepare_density_data(ds, task, cpred, rpred)
        cmap = LinearSegmentedColormap.from_list("den", ["#F7F8FA", COL_CLASS if task == "classification" else COL_REG, COL_TEACHER])
        ax.hexbin(x, y, gridsize=38, mincnt=1, cmap=cmap, linewidths=0, alpha=0.96)
        lo = min(xlim[0], ylim[0])
        hi = max(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], color="white", lw=2.0, zorder=3)
        ax.plot([lo, hi], [lo, hi], color=COL_TEACHER, lw=0.8, ls="--", zorder=4)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(DISPLAY[ds], loc="left", fontweight="bold")
        g = metrics[metrics["dataset"].eq(ds)]
        r = g["fidelity_prob_pearson"].mean() if task == "classification" else g["fidelity_pred_pearson"].mean()
        ax.text(0.04, 0.92, f"r {r:.2f}", transform=ax.transAxes, fontsize=6.4, color=COL_TEXT)
        ax.set_xlabel("Teacher")
        ax.set_ylabel("Student")
        ax.grid(color=COL_GRID, lw=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    save_fig(fig, EXT / "edfig_c4_teacher_student_density")
    plt.close(fig)


def plot_run_trace(metrics: pd.DataFrame, batch: pd.DataFrame) -> None:
    m = metrics.copy()
    m["seed"] = m["seed"].astype(int)
    seeds = sorted(m["seed"].unique())
    datasets = DATASET_ORDER
    tokens = np.full((len(datasets), len(seeds)), np.nan)
    minutes = np.full_like(tokens, np.nan, dtype=float)
    calls = np.full_like(tokens, np.nan, dtype=float)
    for i, ds in enumerate(datasets):
        for j, seed in enumerate(seeds):
            g = m[(m["dataset"].eq(ds)) & (m["seed"].eq(seed))]
            if len(g) == 1:
                tokens[i, j] = float(g["llm_total_tokens"].iloc[0]) / 1000.0
                minutes[i, j] = float(g["seconds"].iloc[0]) / 60.0
                calls[i, j] = float(g["llm_calls"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.15), constrained_layout=True)
    for ax, mat, title, cmap, fmt in [
        (axes[0], tokens, "LLM tokens (k)", "YlGnBu", "{:.0f}"),
        (axes[1], minutes, "Runtime (min)", "OrRd", "{:.1f}"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=5.8, color="#17202A")
        ax.set_xticks(np.arange(len(seeds)))
        ax.set_xticklabels(seeds)
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels([DISPLAY[d] for d in datasets] if ax is axes[0] else [])
        ax.set_xlabel("Seed")
        ax.set_title(title, loc="left", fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(seeds), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=5.8, length=2)
    total_tokens = m["llm_total_tokens"].sum()
    total_calls = m["llm_calls"].sum()
    total_runs = len(m)
    fig.text(
        0.01,
        0.01,
        f"{total_runs}/30 success | {int(total_calls)} calls | {total_tokens/1e6:.2f}M tokens",
        fontsize=6.5,
        color=COL_MUTED,
    )
    save_fig(fig, EXT / "edfig_c5_run_trace_resource_heatmap")
    plt.close(fig)


def write_plot_manifest(metrics: pd.DataFrame) -> None:
    manifest = {
        "source_dir": str(SOURCE),
        "output_dir": str(OUT),
        "n_success_runs": int(len(metrics)),
        "datasets": DATASET_ORDER,
        "seeds": sorted([int(x) for x in metrics["seed"].unique()]),
        "outputs": [
            "fig6_panel_c_main_2x3_ranked_output_preservation.png/pdf/svg",
            "main_panel_c_subfigures/panel_c_*.png/pdf/svg",
            "extended_data/edfig_c1_per_seed_performance_retention.png/pdf/svg",
            "extended_data/edfig_c2_delivery_compression_map.png/pdf/svg",
            "extended_data/edfig_c3_fidelity_metric_matrix.png/pdf/svg",
            "extended_data/edfig_c4_teacher_student_density.png/pdf/svg",
            "extended_data/edfig_c5_run_trace_resource_heatmap.png/pdf/svg",
        ],
    }
    (OUT / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_cols = [
        "task",
        "dataset",
        "teacher_auc",
        "student_auc",
        "auc_delta_student_minus_teacher",
        "teacher_rmse",
        "student_rmse",
        "r2_delta_student_minus_teacher",
        "fidelity_prob_pearson",
        "fidelity_pred_pearson",
        "latency_ratio_student_over_teacher",
        "artifact_size_ratio_student_over_teacher",
    ]
    metrics[[c for c in summary_cols if c in metrics.columns]].to_csv(OUT / "plot_source_metrics.csv", index=False)


def main() -> None:
    global SOURCE, OUT, MAIN_SUB, EXT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    SOURCE = args.source.expanduser().resolve()
    OUT = args.output.expanduser().resolve()
    MAIN_SUB = OUT / "main_panel_c_subfigures"
    EXT = OUT / "extended_data"
    setup_style()
    ensure_dirs()
    metrics, cpred, rpred, batch = read_data()
    main_panel(metrics, cpred, rpred)
    plot_per_seed_retention(metrics)
    plot_delivery_map(metrics)
    plot_fidelity_matrix(metrics)
    plot_density(metrics, cpred, rpred)
    plot_run_trace(metrics, batch)
    write_plot_manifest(metrics)


if __name__ == "__main__":
    main()
