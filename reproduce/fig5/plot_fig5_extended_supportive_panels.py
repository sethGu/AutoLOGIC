from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "detailed_results" / "fig5_extended_data_curated_supportive_20260525"
TABLES = PACKAGE / "tables"
OUT = PACKAGE / "panels_png"

DPI = 520
WIDTH_PX = 3804
HEIGHT_PX = 1578
FIGSIZE = (WIDTH_PX / DPI, HEIGHT_PX / DPI)

FONT = "DejaVu Sans"
TICK = 8.5
LABEL = 9.2
TITLE = 10.2

BLUE = "#3b8bc0"
ORANGE = "#cf6f4a"
NAVY = "#1f2937"
GRAY = "#667085"
LIGHT_GRID = "#e4e7eb"
SPINE = "#b8c1cc"
TIE = "#cfd6df"
RED = "#bd3f55"
GREEN = "#5da5a4"
PALE_BLUE = "#dbeaf4"
PALE_ORANGE = "#f3dfd4"

DATASET_LABEL = {
    "ALL": "All",
    "cc1": "CC1",
    "credit-g": "Cr-g",
    "ld1": "LD1",
    "boston": "Bos",
    "california": "Cal",
    "concrete": "Con",
    "all": "All",
    "supervised": "Sup.",
    "classification": "Cls",
    "regression": "Reg",
}


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": FONT,
            "font.size": TICK,
            "axes.titlesize": TITLE,
            "axes.labelsize": LABEL,
            "xtick.labelsize": TICK,
            "ytick.labelsize": TICK,
            "legend.fontsize": TICK,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.linewidth": 0.75,
            "axes.edgecolor": SPINE,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "axes.labelcolor": "black",
            "axes.titleweight": "bold",
        }
    )


def finish(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return path


def style_axis(ax, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=LIGHT_GRID, linewidth=0.65)
        ax.set_axisbelow(True)
    ax.tick_params(width=0.75, length=3.2, color=GRAY)


def panel_label(ax, text: str) -> None:
    # Extended Data subpanels are titled by the final composite layout.
    return None


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def plot_distillation() -> Path:
    df = read_table("ED_Fig5S_02_distillation_retention_deployability.csv")
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        left=0.088,
        right=0.985,
        bottom=0.16,
        top=0.985,
        wspace=0.32,
        hspace=0.28,
        width_ratios=[1.0, 1.05, 1.05],
    )

    ax_ret = fig.add_subplot(gs[:, 0])
    ax_size_cls = fig.add_subplot(gs[0, 1])
    ax_speed_cls = fig.add_subplot(gs[0, 2])
    ax_size_reg = fig.add_subplot(gs[1, 1])
    ax_speed_reg = fig.add_subplot(gs[1, 2])

    order = ["ALL", "cc1", "credit-g", "ld1", "ALL", "boston", "california", "concrete"]
    tasks = ["classification"] * 4 + ["regression"] * 4
    y = np.arange(len(order))[::-1]
    vals = []
    colors = []
    labels = []
    for ds, task in zip(order, tasks):
        row = df[(df["dataset"] == ds) & (df["task"] == task)].iloc[0]
        if task == "classification":
            vals.append(row["auc_retention_ratio"])
            colors.append(BLUE)
            labels.append(DATASET_LABEL[ds])
        else:
            vals.append(row["teacher_student_prediction_corr"])
            colors.append(ORANGE)
            labels.append(DATASET_LABEL[ds])
    vals = np.array(vals, dtype=float)
    ax_ret.axvspan(0.95, 1.13, color=PALE_BLUE, alpha=0.55, zorder=0)
    ax_ret.axvline(1.0, color=GRAY, linestyle="--", linewidth=0.9)
    for yi, val, color in zip(y, vals, colors):
        ax_ret.plot([0.95, val], [yi, yi], color=color, linewidth=1.3, alpha=0.75)
        ax_ret.scatter([val], [yi], s=36, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax_ret.scatter([vals[0]], [y[0]], s=70, color=NAVY, edgecolor="white", linewidth=0.5, zorder=4)
    ax_ret.scatter([vals[4]], [y[4]], s=70, color=NAVY, edgecolor="white", linewidth=0.5, zorder=4)
    ax_ret.set_yticks(y)
    ax_ret.set_yticklabels(labels)
    ax_ret.set_xlim(0.92, 1.13)
    ax_ret.set_xlabel("Retention / fidelity")
    panel_label(ax_ret, "Teacher signal retained")
    style_axis(ax_ret, "x")
    ax_ret.text(0.925, 6.5, "AUC ret.", ha="left", va="center", fontsize=TICK, color=BLUE, fontweight="bold")
    ax_ret.text(0.925, 2.5, "Pred. r", ha="left", va="center", fontsize=TICK, color=ORANGE, fontweight="bold")

    def delivery_axis(ax, task: str, col: str, title: str, xlabel: str, show_xlabel: bool = True) -> None:
        sub = df[(df["task"] == task) & (df["dataset"] != "ALL")].copy()
        sub["label"] = sub["dataset"].map(DATASET_LABEL)
        sub = sub.set_index("dataset").loc[
            ["cc1", "credit-g", "ld1"] if task == "classification" else ["boston", "concrete", "california"]
        ].reset_index()
        yy = np.arange(len(sub))[::-1]
        color = BLUE if task == "classification" else ORANGE
        ax.set_xscale("log")
        vals = sub[col].astype(float).values
        all_row = df[(df["task"] == task) & (df["dataset"] == "ALL")].iloc[0]
        max_val = max(float(np.nanmax(vals)), float(all_row[col]))
        ax.set_xlim(0.8, max_val * 2.2)

        def place_value_label(x, y, text, bold=False):
            # Keep labels away from markers in compressed panels.
            if x > max_val * 0.45:
                tx, ha = x / 1.42, "right"
            else:
                tx, ha = x * 1.34, "left"
            ax.text(
                tx,
                y,
                text,
                va="center",
                ha=ha,
                fontsize=TICK,
                fontweight="bold" if bold else "normal",
                color=NAVY,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.08),
                zorder=6,
            )

        for yi, (_, row) in zip(yy, sub.iterrows()):
            val = row[col]
            ax.plot([1, val], [yi, yi], color=color, linewidth=1.25, alpha=0.62)
            ax.scatter([val], [yi], s=44, color=color, marker="o" if task == "classification" else "s", edgecolor="white", linewidth=0.45, zorder=3)
            place_value_label(val, yi, f"{val:.1f}x")
        ax.scatter([all_row[col]], [len(sub) + 0.18], s=80, color=NAVY, marker="D", edgecolor="white", linewidth=0.5, zorder=4)
        place_value_label(all_row[col], len(sub) + 0.18, f"All {all_row[col]:.1f}x", bold=True)
        ax.set_yticks(yy)
        ax.set_yticklabels(sub["label"])
        ax.set_ylim(-0.6, len(sub) + 0.75)
        ax.set_xlabel(xlabel if show_xlabel else "")
        if not show_xlabel:
            ax.tick_params(labelbottom=False)
        panel_label(ax, title)
        style_axis(ax, "x")

    delivery_axis(ax_size_cls, "classification", "artifact_size_reduction_x", "Classification size", "Artifact reduction x", show_xlabel=False)
    delivery_axis(ax_speed_cls, "classification", "inference_speedup_x", "Classification speed", "Inference speed-up x", show_xlabel=False)
    delivery_axis(ax_size_reg, "regression", "artifact_size_reduction_x", "Regression size", "Artifact reduction x")
    delivery_axis(ax_speed_reg, "regression", "inference_speedup_x", "Regression speed", "Inference speed-up x")

    return finish(fig, "ed_fig5s_02_distillation_retention_deployability")


def plot_calibration() -> Path:
    df = read_table("ED_Fig5S_03_calibration_supportive_metrics.csv")
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.075, right=0.985, bottom=0.16, top=0.985, wspace=0.28, hspace=0.30)
    ax_ece = fig.add_subplot(gs[0, 0])
    ax_brier = fig.add_subplot(gs[1, 0])
    ax_picp = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1])

    cls = df[df["task"] == "classification"].copy()
    cls["label"] = cls["dataset"].map(DATASET_LABEL)
    cls = cls.set_index("dataset").loc[["ALL", "cc1", "ld1"]].reset_index()
    y = np.arange(len(cls))[::-1]

    def dumbbell(ax, raw_col: str, cal_col: str, delta_col: str, title: str, xmax: float, show_xlabel: bool = True) -> None:
        for yi, (_, row) in zip(y, cls.iterrows()):
            raw = row[raw_col]
            cal = row[cal_col]
            ax.plot([raw, cal], [yi, yi], color=BLUE, linewidth=1.5, alpha=0.62)
            ax.annotate("", xy=(cal, yi), xytext=(raw, yi), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
            ax.scatter([raw], [yi], s=48, facecolor="white", edgecolor=GRAY, linewidth=0.8, zorder=3)
            ax.scatter([cal], [yi], s=48, facecolor=BLUE, edgecolor="white", linewidth=0.5, zorder=4)
            ax.text(
                xmax * 0.88,
                yi,
                f"{row[delta_col]:+.4f}",
                ha="left",
                va="center",
                fontsize=TICK,
                fontweight="bold",
                color=BLUE,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.2),
            )
        ax.set_yticks(y)
        ax.set_yticklabels(cls["label"])
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Lower is better" if show_xlabel else "")
        if not show_xlabel:
            ax.tick_params(labelbottom=False)
        panel_label(ax, title)
        style_axis(ax, "x")
        ax.scatter([], [], s=42, facecolor="white", edgecolor=GRAY, label="raw")
        ax.scatter([], [], s=42, facecolor=BLUE, edgecolor="white", label="cal.")
        ax.legend(frameon=False, loc="lower right", ncol=2, handletextpad=0.3, columnspacing=0.8)

    dumbbell(ax_ece, "mean_ece_raw", "mean_ece_cal", "ece_delta_cal_minus_raw", "ECE shift", 0.070, show_xlabel=False)
    dumbbell(ax_brier, "mean_brier_raw", "mean_brier_cal", "brier_delta_cal_minus_raw", "Brier shift", 0.220)
    leg = ax_brier.get_legend()
    if leg is not None:
        leg.remove()

    reg = df[df["task"] == "regression"].copy()
    reg["label"] = reg["dataset"].map(DATASET_LABEL)
    reg = reg.set_index("dataset").loc[["ALL", "boston", "concrete", "california"]].reset_index()
    yy = np.arange(len(reg))[::-1]
    ax_picp.axvline(0.9, color=GRAY, linestyle="--", linewidth=0.9)
    for yi, (_, row) in zip(yy, reg.iterrows()):
        raw = row["mean_picp_raw_gaussian"]
        cal = row["mean_picp_conformal"]
        ax_picp.plot([raw, cal], [yi, yi], color=ORANGE, linewidth=1.5, alpha=0.70)
        ax_picp.annotate("", xy=(cal, yi), xytext=(raw, yi), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
        ax_picp.scatter([raw], [yi], s=48, facecolor="white", edgecolor=GRAY, linewidth=0.8, zorder=3)
        ax_picp.scatter([cal], [yi], s=48, facecolor=ORANGE, edgecolor="white", linewidth=0.5, zorder=4)
    ax_picp.set_yticks(yy)
    ax_picp.set_yticklabels(reg["label"])
    ax_picp.set_xlim(0.54, 0.94)
    ax_picp.set_xlabel("")
    ax_picp.tick_params(labelbottom=False)
    panel_label(ax_picp, "PICP to target")
    style_axis(ax_picp, "x")
    ax_picp.text(0.895, len(reg) - 0.25, "target", fontsize=TICK, color=GRAY, ha="right", va="center")

    bars = reg["coverage_error_reduction"].astype(float).values
    ax_err.barh(yy, bars, color=[NAVY, ORANGE, ORANGE, ORANGE], alpha=0.92, height=0.55)
    for yi, val in zip(yy, bars):
        ax_err.text(val + 0.006, yi, f"{val:.3f}", va="center", ha="left", fontsize=TICK, fontweight="bold", color=NAVY)
    ax_err.set_yticks(yy)
    ax_err.set_yticklabels(reg["label"])
    ax_err.set_xlim(0, 0.34)
    ax_err.set_xlabel("Coverage-error reduction")
    panel_label(ax_err, "Reliability gain")
    style_axis(ax_err, "x")

    return finish(fig, "ed_fig5s_03_calibration_supportive_metrics")


def plot_meta() -> Path:
    df = read_table("ED_Fig5S_04_meta_supportive_wtl.csv")
    df = df.set_index("scope").loc[["all", "supervised", "classification", "regression"]].reset_index()
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = gridspec.GridSpec(1, 2, figure=fig, left=0.08, right=0.985, bottom=0.19, top=0.965, wspace=0.32, width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_margin = fig.add_subplot(gs[0, 1])

    y = np.arange(len(df))[::-1]
    colors = {"win": BLUE, "tie": TIE, "loss": RED}
    for yi, (_, row) in zip(y, df.iterrows()):
        total = row["non_missing"]
        left = 0.0
        for key in ["win", "tie", "loss"]:
            val = row[key]
            width = val / total
            ax.barh(yi, width, left=left, height=0.58, color=colors[key], edgecolor="white", linewidth=0.8)
            if val > 0:
                ax.text(left + width / 2, yi, f"{int(val)}", ha="center", va="center", color="white" if key != "tie" else NAVY, fontsize=TICK, fontweight="bold")
            left += width
        ax.text(1.025, yi, f"{int(row['win'])}W/{int(row['tie'])}T/{int(row['loss'])}L", ha="left", va="center", fontsize=TICK, fontweight="bold", color=NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels(df["scope"].map(DATASET_LABEL))
    ax.set_xlim(0, 1.28)
    ax.set_xlabel("Share of non-missing comparisons")
    panel_label(ax, "Meta W/T/L")
    style_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.legend(
        [Rectangle((0, 0), 1, 1, color=colors["win"]), Rectangle((0, 0), 1, 1, color=colors["tie"]), Rectangle((0, 0), 1, 1, color=colors["loss"])],
        ["win", "tie", "loss"],
        frameon=False,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.62, 1.08),
        handlelength=1.1,
        columnspacing=0.9,
    )

    margins = df["win_loss_margin"].astype(float).values
    ax_margin.axvline(0, color=GRAY, linestyle="--", linewidth=0.9)
    ax_margin.barh(y, margins, color=[NAVY, BLUE, BLUE, ORANGE], height=0.55, alpha=0.92)
    for yi, val in zip(y, margins):
        ax_margin.text(val + 0.25, yi, f"+{int(val)}", va="center", ha="left", fontsize=TICK, fontweight="bold", color=NAVY)
    ax_margin.set_yticks(y)
    ax_margin.set_yticklabels(df["scope"].map(DATASET_LABEL))
    ax_margin.set_xlim(0, max(margins) + 2.5)
    ax_margin.set_xlabel("Win-loss margin")
    panel_label(ax_margin, "Positive margin")
    style_axis(ax_margin, "x")
    ax_margin.spines["left"].set_visible(False)

    return finish(fig, "ed_fig5s_04_meta_supportive_wtl")


def main() -> None:
    setup()
    paths = [plot_distillation(), plot_calibration(), plot_meta()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
