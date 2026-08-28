from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Wedge


ROOT = Path(__file__).resolve().parent
CLASS_REG_DIR = ROOT / "full_incremental_42_46_20260517_v2"
CLUSTER_DIR = ROOT / "clustering_full_breast_glass_students_42_46_strong_20260517"
OUT = ROOT / "panel_d_png_v2_nonredundant_20260517"
OUT_SUB = OUT / "subfigures"

DATASET_ORDER = ["cc1", "credit-g", "ld1", "boston", "concrete", "california", "breast", "glass", "students"]
CLASS_ORDER = ["cc1", "credit-g", "ld1"]
REG_ORDER = ["boston", "concrete", "california"]
CLUSTER_ORDER = ["breast", "glass", "students"]
DISPLAY = {
    "cc1": "CC1",
    "credit-g": "Cr-g",
    "ld1": "LD1",
    "boston": "Bos",
    "concrete": "Con",
    "california": "Cal",
    "breast": "Br",
    "glass": "Gla",
    "students": "Stu",
}
TASK_COLOR = {"classification": "#1F78B4", "regression": "#C7542D", "clustering": "#2E8B57"}
METRIC_COLOR = {"Fid": "#2775B6", "Ret": "#3A9B73", "Ctrl": "#7C6AC4", "Deliv": "#D77933"}
COL_TEXT = "#16202E"
COL_MUTED = "#657083"
COL_GRID = "#E4E8EE"
COL_SPINE = "#AEB6C2"
COL_BAD = "#D65A5A"
COL_WARN = "#E6B857"
COL_GOOD = "#60A982"
COL_NA = "#D7DDE5"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.6,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 420,
        }
    )


def save_png(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor="white")


def style_axis(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=COL_GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(COL_SPINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=COL_MUTED, width=0.8, length=3)
    ax.title.set_color(COL_TEXT)
    ax.xaxis.label.set_color(COL_TEXT)
    ax.yaxis.label.set_color(COL_TEXT)


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cr = pd.read_csv(CLASS_REG_DIR / "panel_d_dataset_seed_enhanced.csv")
    cr = cr[cr["task"].isin(["classification", "regression"])].copy()
    cr["seed"] = cr["seed"].astype(int)

    selected = pd.read_csv(CLUSTER_DIR / "clustering_selected_by_train_fidelity.csv")
    selected["seed"] = selected["seed"].astype(int)
    per_seed = pd.read_csv(CLUSTER_DIR / "clustering_per_seed.csv")
    per_seed["seed"] = per_seed["seed"].astype(int)
    selected = selected.merge(
        per_seed[
            [
                "dataset",
                "seed",
                "config",
                "feature_setting",
                "ari",
                "nmi",
                "ari_percent",
                "nmi_percent",
                "retention_ratio",
            ]
        ],
        on=["dataset", "seed", "config", "feature_setting"],
        how="left",
    )
    return cr, selected, per_seed


def value_tables(cr: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fidelity = cr[["task", "dataset", "seed", "fidelity_primary"]].rename(columns={"fidelity_primary": "value"})
    cluster_fid = selected[["dataset", "seed", "test_knn_overlap"]].copy()
    cluster_fid["task"] = "clustering"
    cluster_fid = cluster_fid.rename(columns={"test_knn_overlap": "value"})
    fidelity = pd.concat([fidelity, cluster_fid[["task", "dataset", "seed", "value"]]], ignore_index=True)

    retention = cr[["task", "dataset", "seed", "performance_retention"]].rename(columns={"performance_retention": "value"})
    cluster_ret = selected[["dataset", "seed", "retention_ratio"]].copy()
    cluster_ret["task"] = "clustering"
    cluster_ret = cluster_ret.rename(columns={"retention_ratio": "value"})
    retention = pd.concat([retention, cluster_ret[["task", "dataset", "seed", "value"]]], ignore_index=True)

    control = cr[["task", "dataset", "seed", "output_control_gain"]].rename(columns={"output_control_gain": "value"})

    delivery = cr[["task", "dataset", "seed", "latency_ratio", "artifact_size_ratio", "delivery_gain_latency", "delivery_gain_artifact"]].copy()
    delivery["speed_x"] = 1.0 / delivery["latency_ratio"].astype(float)
    delivery["size_x"] = 1.0 / delivery["artifact_size_ratio"].astype(float)
    delivery["delivery_score"] = delivery[["delivery_gain_latency", "delivery_gain_artifact"]].mean(axis=1)
    return fidelity, retention, control, delivery


def summary_table(cr: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    fidelity, retention, control, delivery = value_tables(cr, selected)
    rows = []
    for ds in DATASET_ORDER:
        if ds in CLASS_ORDER:
            task = "classification"
        elif ds in REG_ORDER:
            task = "regression"
        else:
            task = "clustering"
        rows.append(
            {
                "dataset": ds,
                "task": task,
                "fidelity": fidelity[fidelity["dataset"].eq(ds)]["value"].mean(),
                "retention": retention[retention["dataset"].eq(ds)]["value"].mean(),
                "control": control[control["dataset"].eq(ds)]["value"].mean(),
                "delivery_score": delivery[delivery["dataset"].eq(ds)]["delivery_score"].mean(),
                "speed_x": delivery[delivery["dataset"].eq(ds)]["speed_x"].mean(),
                "size_x": delivery[delivery["dataset"].eq(ds)]["size_x"].mean(),
            }
        )
    return pd.DataFrame(rows)


def score_fidelity(v: float) -> float:
    return float(np.clip(v, 0, 1)) if np.isfinite(v) else np.nan


def score_retention(v: float) -> float:
    return float(np.clip(v, 0, 1)) if np.isfinite(v) else np.nan


def score_control(v: float) -> float:
    if not np.isfinite(v):
        return np.nan
    return float(np.clip((v + 0.05) / 0.05, 0, 1))


def score_delivery(v: float) -> float:
    return float(np.clip(v, 0, 1)) if np.isfinite(v) else np.nan


def add_arc_glyph(ax: plt.Axes, x: float, y: float, values: dict[str, float], label: str, task: str) -> None:
    radii = [0.37, 0.29, 0.21, 0.13]
    width = 0.055
    score_funcs = {"Fid": score_fidelity, "Ret": score_retention, "Ctrl": score_control, "Deliv": score_delivery}
    for key, radius in zip(["Fid", "Ret", "Ctrl", "Deliv"], radii):
        raw = values.get(key, np.nan)
        score = score_funcs[key](raw)
        ax.add_patch(Wedge((x, y), radius, 0, 360, width=width, facecolor=COL_NA, edgecolor="white", lw=0.5))
        if np.isfinite(score):
            ax.add_patch(
                Wedge(
                    (x, y),
                    radius,
                    90,
                    90 - 360 * score,
                    width=width,
                    facecolor=METRIC_COLOR[key],
                    edgecolor="white",
                    lw=0.5,
                )
            )
    ax.add_patch(plt.Circle((x, y), 0.075, color=TASK_COLOR[task], ec="white", lw=0.7, zorder=5))
    ax.text(x, y - 0.52, label, ha="center", va="top", fontsize=7.4, fontweight="bold", color=COL_TEXT)


def plot_glyph_atlas(ax: plt.Axes, summary: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    layout = [
        ("classification", CLASS_ORDER, 2.55),
        ("regression", REG_ORDER, 1.35),
        ("clustering", CLUSTER_ORDER, 0.15),
    ]
    for task, datasets, y in layout:
        ax.text(-0.55, y, {"classification": "Cls", "regression": "Reg", "clustering": "Clu"}[task], ha="right", va="center", fontsize=7.5, color=COL_MUTED)
        for x, ds in zip([0.0, 1.15, 2.30], datasets):
            r = summary[summary["dataset"].eq(ds)].iloc[0]
            add_arc_glyph(
                ax,
                x,
                y,
                {"Fid": r["fidelity"], "Ret": r["retention"], "Ctrl": r["control"], "Deliv": r["delivery_score"]},
                DISPLAY[ds],
                task,
            )
    for i, key in enumerate(["Fid", "Ret", "Ctrl", "Deliv"]):
        ax.add_patch(Wedge((-0.15 + 0.58 * i, 3.25), 0.08, 0, 360, width=0.04, facecolor=METRIC_COLOR[key], edgecolor="none"))
        ax.text(-0.03 + 0.58 * i, 3.25, key, ha="left", va="center", fontsize=6.8, color=COL_TEXT)
    ax.set_xlim(-0.75, 2.85)
    ax.set_ylim(-0.55, 3.45)
    ax.set_title("a  metric glyph atlas", loc="left", fontweight="bold")


def plot_fidelity_violin(ax: plt.Axes, fidelity: pd.DataFrame) -> None:
    data = [fidelity[fidelity["dataset"].eq(ds)]["value"].dropna().to_numpy(float) for ds in DATASET_ORDER]
    parts = ax.violinplot(data, positions=np.arange(len(DATASET_ORDER)), vert=False, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, ds in zip(parts["bodies"], DATASET_ORDER):
        task = fidelity[fidelity["dataset"].eq(ds)]["task"].iloc[0]
        body.set_facecolor(TASK_COLOR[task])
        body.set_edgecolor("white")
        body.set_alpha(0.74)
        body.set_linewidth(0.6)
    for i, ds in enumerate(DATASET_ORDER):
        vals = fidelity[fidelity["dataset"].eq(ds)]["value"].dropna().to_numpy(float)
        med = np.median(vals)
        ax.add_patch(FancyBboxPatch((med - 0.006, i - 0.23), 0.012, 0.46, boxstyle="round,pad=0.0,rounding_size=0.007", color=COL_TEXT, ec="none", zorder=4))
    ax.axvspan(0.9, 1.02, color="#E6F2EC", zorder=0)
    ax.axvspan(0.0, 0.8, color="#F8ECEC", zorder=0)
    ax.axvline(0.90, color="#8A93A1", ls="--", lw=0.9)
    ax.axvline(0.80, color="#B6BEC9", ls=":", lw=0.9)
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels([DISPLAY[d] for d in DATASET_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0.55, 1.02)
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.set_xlabel("Decision fidelity")
    ax.set_title("b  fidelity spread", loc="left", fontweight="bold")
    style_axis(ax)


def plot_retention_margin(ax: plt.Axes, retention: pd.DataFrame) -> None:
    means = retention.groupby(["dataset", "task"], as_index=False)["value"].mean()
    means["margin"] = means["value"] - 0.90
    means["dataset"] = pd.Categorical(means["dataset"], DATASET_ORDER, ordered=True)
    means = means.sort_values("dataset")
    colors = [TASK_COLOR[t] if m >= 0 else COL_BAD for t, m in zip(means["task"], means["margin"])]
    ax.barh(np.arange(len(means)), means["margin"], color=colors, alpha=0.88, edgecolor="white", linewidth=0.7)
    for i, row in means.iterrows():
        xpos = row["margin"] + (0.012 if row["margin"] >= 0 else -0.012)
        ha = "left" if row["margin"] >= 0 else "right"
        ax.text(xpos, list(means.index).index(i), f"{row['value']:.2f}", va="center", ha=ha, fontsize=6.8, color=COL_TEXT)
    ax.axvline(0, color="#7E8794", lw=0.9, ls="--")
    ax.set_yticks(np.arange(len(means)))
    ax.set_yticklabels([DISPLAY[d] for d in means["dataset"].astype(str)])
    ax.invert_yaxis()
    ax.set_xlim(-0.23, 0.26)
    ax.set_xticks([-0.2, 0, 0.2])
    ax.set_xlabel("Retention margin vs 0.90")
    ax.set_title("c  retention boundary", loc="left", fontweight="bold")
    style_axis(ax)


def plot_control_bars(ax: plt.Axes, control: pd.DataFrame) -> None:
    means = control.groupby(["dataset", "task"], as_index=False)["value"].mean()
    means["dataset"] = pd.Categorical(means["dataset"], CLASS_ORDER + REG_ORDER, ordered=True)
    means = means.dropna(subset=["dataset"]).sort_values("dataset")
    finite = means["value"].notna().to_numpy()
    colors = [
        TASK_COLOR[t] if v >= -0.01 else (COL_WARN if v >= -0.03 else COL_BAD)
        for t, v in zip(means.loc[finite, "task"], means.loc[finite, "value"])
    ]
    ax.barh(np.arange(len(means))[finite], means.loc[finite, "value"], color=colors, alpha=0.88, edgecolor="white", linewidth=0.7)
    for i, row in means.reset_index(drop=True).iterrows():
        if not np.isfinite(row["value"]):
            ax.text(-0.055, i, "NA", va="center", ha="left", fontsize=7.0, color="#8D96A4")
            continue
        xpos = row["value"] + (0.002 if row["value"] >= 0 else -0.002)
        ha = "left" if row["value"] >= 0 else "right"
        ax.text(xpos, i, f"{row['value']:+.3f}", va="center", ha=ha, fontsize=6.8, color=COL_TEXT)
    ax.axvline(0, color="#7E8794", lw=0.9, ls="--")
    ax.set_yticks(np.arange(len(means)))
    ax.set_yticklabels([DISPLAY[d] for d in means["dataset"].astype(str)])
    ax.invert_yaxis()
    ax.set_xlim(-0.06, 0.015)
    ax.set_xticks([-0.05, -0.025, 0])
    ax.set_xlabel("Output-control change")
    ax.set_title("d  control deltas", loc="left", fontweight="bold")
    style_axis(ax)


def capsule(ax: plt.Axes, x0: float, y: float, width: float, height: float, fill: float, color: str, label: str) -> None:
    base = FancyBboxPatch((x0, y - height / 2), width, height, boxstyle=f"round,pad=0.0,rounding_size={height/2}", facecolor="#EEF1F5", edgecolor="none")
    ax.add_patch(base)
    fw = max(0.0001, width * np.clip(fill, 0, 1))
    filled = FancyBboxPatch((x0, y - height / 2), fw, height, boxstyle=f"round,pad=0.0,rounding_size={height/2}", facecolor=color, edgecolor="none", alpha=0.88)
    ax.add_patch(filled)
    ax.text(x0 + width + 0.04, y, label, va="center", ha="left", fontsize=6.6, color=COL_TEXT)


def plot_delivery_capsules(ax: plt.Axes, delivery: pd.DataFrame) -> None:
    means = delivery.groupby(["dataset", "task"], as_index=False)[["speed_x", "size_x"]].mean()
    means["dataset"] = pd.Categorical(means["dataset"], CLASS_ORDER + REG_ORDER, ordered=True)
    means = means.sort_values("dataset")
    x0 = 0.28
    width = 0.56
    for i, row in means.reset_index(drop=True).iterrows():
        color = TASK_COLOR[row["task"]]
        y = i
        ax.text(0.0, y, DISPLAY[str(row["dataset"])], ha="left", va="center", fontsize=7.4, color=COL_TEXT)
        capsule(ax, x0, y - 0.14, width, 0.12, np.log10(row["size_x"]) / np.log10(150), color, f"S {row['size_x']:.1f}x")
        capsule(ax, x0, y + 0.14, width, 0.12, np.log10(row["speed_x"]) / np.log10(22), color, f"T {row['speed_x']:.1f}x")
    ax.set_xlim(-0.02, 1.32)
    ax.set_ylim(-0.65, len(means) - 0.35)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("e  delivery capsules", loc="left", fontweight="bold")


def plot_cluster_bars(ax: plt.Axes, selected: pd.DataFrame) -> None:
    metrics = [("ARI", "test_assignment_ari"), ("NMI", "test_assignment_nmi"), ("kNN", "test_knn_overlap")]
    x = np.arange(len(CLUSTER_ORDER))
    bar_w = 0.23
    colors = ["#4D8FC5", "#7B6EC4", "#52A36F"]
    for j, (lab, col) in enumerate(metrics):
        vals = [selected[selected["dataset"].eq(ds)][col].mean() for ds in CLUSTER_ORDER]
        ax.bar(x + (j - 1) * bar_w, vals, width=bar_w, color=colors[j], edgecolor="white", linewidth=0.7, label=lab, alpha=0.9)
        for xi, val in zip(x + (j - 1) * bar_w, vals):
            ax.text(xi, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=6.4, color=COL_TEXT)
    ax.axhspan(0.9, 1.02, color="#E6F2EC", zorder=0)
    ax.axhspan(0.0, 0.8, color="#F8ECEC", zorder=0)
    ax.axhline(0.90, color="#8A93A1", lw=0.9, ls="--")
    ax.axhline(0.80, color="#B6BEC9", lw=0.9, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in CLUSTER_ORDER])
    ax.set_ylim(0.45, 1.03)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_ylabel("Teacher-student")
    ax.legend(frameon=False, ncol=3, loc="upper right", handlelength=0.8, columnspacing=0.8)
    ax.set_title("f  clustering boundary", loc="left", fontweight="bold")
    style_axis(ax, axis="y")


def make_all() -> None:
    setup_style()
    OUT_SUB.mkdir(parents=True, exist_ok=True)
    cr, selected, _per_seed = read_data()
    fidelity, retention, control, delivery = value_tables(cr, selected)
    summary = summary_table(cr, selected)

    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.6), constrained_layout=True)
    plot_glyph_atlas(axes[0, 0], summary)
    plot_fidelity_violin(axes[0, 1], fidelity)
    plot_retention_margin(axes[0, 2], retention)
    plot_control_bars(axes[1, 0], control)
    plot_delivery_capsules(axes[1, 1], delivery)
    plot_cluster_bars(axes[1, 2], selected)
    save_png(fig, OUT / "fig6_panel_d_v2_nonredundant")
    plt.close(fig)

    subplots = [
        ("panel_d_v2_a_metric_glyph_atlas", plot_glyph_atlas, (summary,), (3.2, 2.7)),
        ("panel_d_v2_b_fidelity_spread", plot_fidelity_violin, (fidelity,), (3.2, 2.7)),
        ("panel_d_v2_c_retention_boundary", plot_retention_margin, (retention,), (3.2, 2.7)),
        ("panel_d_v2_d_control_deltas", plot_control_bars, (control,), (3.2, 2.3)),
        ("panel_d_v2_e_delivery_capsules", plot_delivery_capsules, (delivery,), (3.2, 2.3)),
        ("panel_d_v2_f_clustering_boundary", plot_cluster_bars, (selected,), (3.2, 2.3)),
    ]
    for name, func, args, size in subplots:
        sfig, sax = plt.subplots(figsize=size, constrained_layout=True)
        func(sax, *args)
        save_png(sfig, OUT_SUB / name)
        plt.close(sfig)


if __name__ == "__main__":
    make_all()
