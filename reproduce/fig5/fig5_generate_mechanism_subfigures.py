from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "detailed_results" / "fig5_master_summary_20260522"
MECH = ROOT / "detailed_results" / "fig5_mechanism_evidence_20260522"
DISTILL_PROBE = ROOT / "detailed_results" / "fig5_distill_latency_complexity_probe_20260522_002320"
CAL_PROBE = ROOT / "detailed_results" / "fig5_calibration_probe_20260522_002454"
STRICT = ROOT / "detailed_results" / "fig5_strict_fixed_pool_20260521_011927" / "analysis"
OUT = MASTER / "fig5_mechanism_subfigures_jpg"

DPI = 300
FONT = 11
LETTER = 12

BLUE = "#2f5597"
TEAL = "#147fa3"
LIGHT = "#dcebf7"
LIGHT2 = "#edf4fb"
ORANGE = "#c7551d"
RED = "#b73a4a"
GRAY = "#707783"
DARK = "#111111"
GRID = "#d9dee7"
GREEN = "#5da5a4"
PURPLE = "#7666a5"

TASK_COLORS = {"classification": TEAL, "regression": ORANGE, "meta": BLUE, "performance": GRAY}
CLS_DATA = {"cc1": "CC1", "credit-g": "Cr-g", "ld1": "LD1"}
REG_DATA = {"boston": "Bos", "concrete": "Con", "california": "Cal"}
MARKERS = {"cc1": "o", "credit-g": "s", "ld1": "^", "boston": "o", "concrete": "s", "california": "^"}


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FONT,
            "axes.titlesize": FONT,
            "axes.labelsize": FONT,
            "xtick.labelsize": FONT,
            "ytick.labelsize": FONT,
            "legend.fontsize": FONT,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.linewidth": 0.9,
            "axes.edgecolor": DARK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fig_px(width, height):
    return plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jpg"
    for text in fig.findobj(mpl.text.Text):
        text.set_fontsize(FONT)
        text.set_fontfamily("Arial")
    fig.savefig(path, format="jpg", dpi=DPI, facecolor="white")
    plt.close(fig)
    return path


def add_card(ax, xy, wh, fc=LIGHT2, ec=DARK, lw=1.3, radius=0.04, shadow=True, z=1):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    if shadow:
        patch.set_path_effects([pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.18), pe.Normal()])
    ax.add_patch(patch)
    return patch


def add_box(ax, x, y, w, h, label, fc=BLUE, tc="white", ec=DARK):
    add_card(ax, (x, y), (w, h), fc=fc, ec=ec, radius=0.035, shadow=True, z=3)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=tc, weight="bold", zorder=4)


def add_arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.7, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=14,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=2,
        )
    )


def clean_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.85)
        ax.set_axisbelow(True)


def title_band(ax, text, color=BLUE):
    ax.add_patch(Rectangle((0, 0.925), 1, 0.075, transform=ax.transAxes, color=color, clip_on=False, zorder=5))
    ax.text(0.015, 0.963, text, transform=ax.transAxes, ha="left", va="center", color="white", weight="bold", zorder=6)


def read_data():
    return {
        "key": pd.read_csv(MASTER / "fig5_master_key_claims.csv"),
        "distill_existing": pd.read_csv(MECH / "distill_existing_raw.csv"),
        "distill_existing_summary": pd.read_csv(MECH / "distill_existing_summary.csv"),
        "distill_probe": pd.read_csv(DISTILL_PROBE / "distill_latency_complexity_raw.csv"),
        "distill_probe_summary": pd.read_csv(DISTILL_PROBE / "distill_latency_complexity_summary.csv"),
        "cal_probe": pd.read_csv(CAL_PROBE / "calibration_probe_raw.csv"),
        "cal_summary": pd.read_csv(CAL_PROBE / "calibration_probe_summary.csv"),
        "meta": pd.read_csv(MECH / "meta_validation_selected_summary.csv"),
        "perf": pd.read_csv(STRICT / "strict_final_wtl_summary.csv"),
    }


def make_a1_protocol():
    fig = fig_px(1660, 720)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title_band(ax, "Locked validation protocol")

    xs = [0.065, 0.26, 0.455, 0.65, 0.81]
    labels = ["Candidate\npool", "Val\nselection", "Distill", "Calibrate", "Locked\ntest"]
    subtitles = ["generated\n+ fallback", "same\npool", "teacher ->\nstudent", "ECE /\nPICP", "reported\nonce"]
    colors = [TEAL, BLUE, "#4c78a8", GREEN, ORANGE]
    for i, (x, label, sub, col) in enumerate(zip(xs, labels, subtitles, colors)):
        add_box(ax, x, 0.56, 0.125, 0.17, label, fc=col)
        ax.text(x + 0.0625, 0.49, sub, ha="center", va="top", color=GRAY)
        if i < len(xs) - 1:
            add_arrow(ax, x + 0.125, 0.645, xs[i + 1], 0.645)

    add_card(ax, (0.045, 0.12), (0.89, 0.18), fc="white", ec="#aab4c1", radius=0.035, shadow=True, z=1)
    callouts = [
        ("No oracle", 0.13, BLUE),
        ("Val only", 0.34, TEAL),
        ("T-S distill", 0.56, GREEN),
        ("Calib. report", 0.79, ORANGE),
    ]
    for text, x, col in callouts:
        ax.scatter([x - 0.055], [0.21], s=185, color=col, edgecolor=DARK, linewidth=0.9, zorder=3)
        ax.text(x, 0.21, text, ha="left", va="center", weight="bold")
    ax.text(0.5, 0.055, "All selection decisions are made before the test split is read.", ha="center", va="center", color=GRAY)
    return save(fig, "fig5-panel-a-1")


def make_a2_evidence_matrix(data):
    key = data["key"]
    fig = fig_px(1660, 720)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title_band(ax, "Mechanism evidence map")

    rows = [
        ("Distill", "retention", "AUC 1.04x | r 0.99", TEAL),
        ("Deploy", "size / latency", "15-118x smaller | 9-20x faster", GREEN),
        ("Calib.", "ECE / PICP", "37% lower ECE | PICP 0.899", ORANGE),
        ("Meta", "Val-selected W/T/L", "13W / 20T / 3L", BLUE),
        ("Perf.", "supplement", "12W / 13T / 11L", GRAY),
    ]
    x0, y0, w, h = 0.04, 0.12, 0.92, 0.72
    add_card(ax, (x0, y0), (w, h), fc="white", ec="#aab4c1", radius=0.025, shadow=True)
    cols = [0.08, 0.30, 0.56]
    headers = ["Module", "Evidence unit", "Best-supported readout"]
    for x, header in zip(cols, headers):
        ax.text(x0 + x * w, y0 + h - 0.075, header, ha="left", va="center", weight="bold")
    for i, (module, ev, readout, col) in enumerate(rows):
        y = y0 + h - 0.17 - i * 0.12
        ax.plot([x0 + 0.02, x0 + w - 0.02], [y + 0.055, y + 0.055], color=GRID, lw=0.8)
        ax.scatter([x0 + cols[0] * w - 0.025], [y], s=180, color=col, edgecolor=DARK, linewidth=0.8)
        ax.text(x0 + cols[0] * w, y, module, ha="left", va="center", weight="bold")
        ax.text(x0 + cols[1] * w, y, ev, ha="left", va="center", color=GRAY)
        ax.text(x0 + cols[2] * w, y, readout, ha="left", va="center")
    ax.text(0.5, 0.045, "Performance is kept as context; mechanism gains carry the main Fig.5 claim.", ha="center", va="center", color=GRAY)
    return save(fig, "fig5-panel-a-2")


def make_b1_retention(data):
    df = data["distill_existing"]
    cls = df[(df.task == "classification") & (df.condition == "all")].copy()
    reg = df[(df.task == "regression") & (df.condition == "all")].copy()
    vals = [
        ("Cls AUC\nret.", cls["auc_retention_ratio"].dropna().values, 1.0, TEAL),
        ("Reg corr", reg["prediction_corr_teacher_student"].dropna().values, 0.95, ORANGE),
        ("Reg RMSE\nratio", reg["rmse_ratio_student_over_teacher"].dropna().values, 1.0, PURPLE),
    ]
    fig = fig_px(1106, 900)
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.08, right=0.98, bottom=0.14, top=0.87, wspace=0.38)
    for i, (label, arr, ref, color) in enumerate(vals):
        ax = fig.add_subplot(gs[0, i])
        parts = ax.violinplot(arr, positions=[0], widths=0.72, showmeans=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(DARK)
            pc.set_alpha(0.22)
        rng = np.random.default_rng(100 + i)
        jitter = rng.normal(0, 0.055, size=len(arr))
        ax.scatter(jitter, arr, s=12, color=color, alpha=0.42, edgecolor="none")
        q1, med, q3 = np.nanpercentile(arr, [25, 50, 75])
        ax.plot([-0.25, 0.25], [med, med], color=DARK, lw=1.5)
        ax.add_patch(Rectangle((-0.18, q1), 0.36, q3 - q1, facecolor="white", edgecolor=DARK, linewidth=1.0, alpha=0.9))
        ax.axhline(ref, color=GRAY, linestyle="--", lw=1.0)
        ax.text(0, np.nanmax(arr) + 0.03 * (np.nanmax(arr) - np.nanmin(arr)), f"mean {np.nanmean(arr):.2f}", ha="center", va="bottom", weight="bold")
        ax.set_xticks([0])
        ax.set_xticklabels([label])
        clean_axes(ax)
        if i == 0:
            ax.set_ylabel("Student / teacher signal")
        else:
            ax.set_ylabel("")
    fig.suptitle("Teacher-to-student retention", x=0.08, y=0.965, ha="left", weight="bold", fontsize=FONT)
    return save(fig, "fig5-panel-b-1")


def deploy_map(ax, df, task, color_col, title, cbar_label):
    sub = df[df.task == task].copy()
    sub["speedup"] = 1.0 / sub["latency_ratio_student_over_teacher"]
    sub["shrink"] = 1.0 / sub["artifact_kb_ratio_student_over_teacher"]
    norm = mpl.colors.Normalize(vmin=np.nanmin(sub[color_col]), vmax=np.nanmax(sub[color_col]))
    cmap = LinearSegmentedColormap.from_list("ret", ["#d7e8f2", TEAL if task == "classification" else ORANGE])
    for ds, g in sub.groupby("dataset"):
        label = CLS_DATA.get(ds, REG_DATA.get(ds, ds))
        sc = ax.scatter(
            g["speedup"],
            g["shrink"],
            c=g[color_col],
            cmap=cmap,
            norm=norm,
            s=72,
            marker=MARKERS.get(ds, "o"),
            edgecolor=DARK,
            linewidth=0.7,
            alpha=0.92,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Speed-up (x)")
    ax.set_ylabel("Size reduction (x)")
    clean_axes(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98), borderpad=0.1, handletextpad=0.3)
    ax.set_title(title, weight="bold", loc="left")
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.015)
    cb.set_label(cbar_label)
    mean_speed = np.nanmean(sub["speedup"])
    mean_shrink = np.nanmean(sub["shrink"])
    ax.scatter([mean_speed], [mean_shrink], s=260, marker="*", color="#f1c232", edgecolor=DARK, linewidth=1.0, zorder=6)
    ax.annotate(
        f"mean\n{mean_speed:.1f}x spd\n{mean_shrink:.1f}x size",
        xy=(mean_speed, mean_shrink),
        xytext=(0.58, 0.20),
        textcoords=ax.transAxes,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#aab4c1", lw=0.8, alpha=0.92),
        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9),
        zorder=7,
    )


def make_b2_cls_deploy(data):
    fig = fig_px(1106, 900)
    ax = fig.add_axes([0.14, 0.20, 0.68, 0.70])
    deploy_map(ax, data["distill_probe"], "classification", "auc_retention_ratio", "Classification deployability", "AUC ret.")
    return save(fig, "fig5-panel-b-2")


def make_b3_reg_deploy(data):
    fig = fig_px(1106, 900)
    ax = fig.add_axes([0.14, 0.20, 0.68, 0.70])
    deploy_map(ax, data["distill_probe"], "regression", "prediction_corr_teacher_student", "Regression deployability", "Pred. r")
    return save(fig, "fig5-panel-b-3")


def make_c1_class_cal_shift(data):
    raw = data["cal_probe"]
    cls = raw[raw.task == "classification"].copy()
    metrics = [("ECE", "ece_raw", "ece_cal"), ("Brier", "brier_raw", "brier_cal"), ("NLL", "nll_raw", "nll_cal")]
    fig = fig_px(1660, 720)
    ax = fig.add_axes([0.11, 0.18, 0.84, 0.70])
    y_positions = np.arange(len(metrics))[::-1]
    colors = [TEAL, GREEN, GRAY]
    for y, (label, raw_col, cal_col), col in zip(y_positions, metrics, colors):
        values = cls[[raw_col, cal_col]].dropna()
        raw_mean = values[raw_col].mean()
        cal_mean = values[cal_col].mean()
        ax.plot([raw_mean, cal_mean], [y, y], color=col, lw=4.0, solid_capstyle="round", alpha=0.38)
        ax.scatter([raw_mean], [y], s=150, facecolor="white", edgecolor=DARK, linewidth=1.0, zorder=3)
        ax.scatter([cal_mean], [y], s=150, facecolor=col, edgecolor=DARK, linewidth=1.0, zorder=4)
        ax.annotate("", xy=(cal_mean, y), xytext=(raw_mean, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7))
        rng = np.random.default_rng(20 + int(y))
        jitter = rng.normal(0, 0.045, len(values))
        ax.scatter(values[raw_col], y + jitter, s=18, facecolor="white", edgecolor=col, linewidth=0.6, alpha=0.6)
        ax.scatter(values[cal_col], y + jitter, s=18, facecolor=col, edgecolor="none", alpha=0.6)
        delta = cal_mean - raw_mean
        ax.text(0.612, y, f"{delta:+.4f}", va="center", ha="left", weight="bold", color=col if delta < 0 else RED)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m[0] for m in metrics])
    ax.set_xlabel("Lower is better")
    ax.set_xlim(0.0, 0.72)
    ax.set_title("Classification calibration shift", loc="left", weight="bold")
    clean_axes(ax)
    ax.scatter([], [], s=90, facecolor="white", edgecolor=DARK, label="raw")
    ax.scatter([], [], s=90, facecolor=TEAL, edgecolor=DARK, label="calibrated")
    ax.legend(frameon=False, loc="center", ncol=2, bbox_to_anchor=(0.58, 0.74), borderaxespad=0.0)
    return save(fig, "fig5-panel-c-1")


def make_c2_class_delta_heatmap(data):
    raw = data["cal_probe"]
    cls = raw[raw.task == "classification"].copy()
    rows = []
    for ds, label in CLS_DATA.items():
        g = cls[cls.dataset == ds]
        rows.append(
            [
                label,
                g["ece_delta_cal_minus_raw"].mean(),
                g["brier_delta_cal_minus_raw"].mean(),
                g["nll_delta_cal_minus_raw"].mean(),
            ]
        )
    mat = np.array([r[1:] for r in rows], dtype=float)
    fig = fig_px(1660, 720)
    ax = fig.add_axes([0.12, 0.16, 0.78, 0.72])
    cmap = LinearSegmentedColormap.from_list("delta", [BLUE, "white", RED])
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    im = ax.imshow(mat, cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", color=DARK, weight="bold")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["ECE", "Brier", "NLL"])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_title("Dataset-level calibration deltas", loc="left", weight="bold")
    ax.set_xlabel("calibrated - raw")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Lower is better")
    return save(fig, "fig5-panel-c-2")


def make_c3_reg_coverage(data):
    raw = data["cal_probe"]
    reg = raw[raw.task == "regression"].copy()
    fig = fig_px(1660, 720)
    ax = fig.add_axes([0.10, 0.17, 0.86, 0.72])
    labels = [REG_DATA[d] for d in REG_DATA]
    xs = np.arange(len(labels))
    for i, ds in enumerate(REG_DATA):
        g = reg[reg.dataset == ds]
        raw_vals = g["picp_raw_gaussian"].dropna()
        cal_vals = g["picp_conformal"].dropna()
        rng = np.random.default_rng(40 + i)
        ax.scatter(np.full(len(raw_vals), xs[i] - 0.12) + rng.normal(0, 0.015, len(raw_vals)), raw_vals, s=26, color="white", edgecolor=GRAY, linewidth=0.7, alpha=0.75)
        ax.scatter(np.full(len(cal_vals), xs[i] + 0.12) + rng.normal(0, 0.015, len(cal_vals)), cal_vals, s=26, color=ORANGE, edgecolor=DARK, linewidth=0.5, alpha=0.8)
        ax.plot([xs[i] - 0.12, xs[i] + 0.12], [raw_vals.mean(), cal_vals.mean()], color=ORANGE, lw=2.0)
        ax.scatter([xs[i] - 0.12], [raw_vals.mean()], s=120, facecolor="white", edgecolor=DARK, zorder=3)
        ax.scatter([xs[i] + 0.12], [cal_vals.mean()], s=120, facecolor=ORANGE, edgecolor=DARK, zorder=3)
    ax.axhline(0.90, color=DARK, linestyle="--", lw=1.2)
    ax.text(2.05, 0.905, "target 0.90", ha="right", va="bottom", weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Observed coverage (PICP)")
    ax.set_xlim(-0.35, 2.25)
    ax.set_ylim(0.48, 1.02)
    ax.set_title("Regression interval coverage", loc="left", weight="bold")
    clean_axes(ax)
    ax.scatter([], [], s=80, facecolor="white", edgecolor=GRAY, label="raw")
    ax.scatter([], [], s=80, facecolor=ORANGE, edgecolor=DARK, label="conformal")
    ax.legend(frameon=False, loc="lower right", ncol=2)
    return save(fig, "fig5-panel-c-3")


def make_c4_reg_width_error(data):
    raw = data["cal_probe"]
    reg = raw[raw.task == "regression"].copy()
    fig = fig_px(1660, 720)
    ax = fig.add_axes([0.11, 0.17, 0.84, 0.72])
    for ds, label in REG_DATA.items():
        g = reg[reg.dataset == ds]
        x1 = g["pinaw_raw_gaussian"].mean()
        y1 = g["coverage_error_raw_abs"].mean()
        x2 = g["pinaw_conformal"].mean()
        y2 = g["coverage_error_conformal_abs"].mean()
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=2.0, color=ORANGE, alpha=0.85))
        ax.scatter([x1], [y1], s=140, facecolor="white", edgecolor=DARK, linewidth=1.0, zorder=3)
        ax.scatter([x2], [y2], s=140, facecolor=ORANGE, edgecolor=DARK, linewidth=1.0, zorder=4)
        ax.text(x2 + 0.008, y2 + 0.004, label, ha="left", va="bottom", weight="bold")
    ax.set_xlabel("Normalized interval width (PINAW)")
    ax.set_ylabel("|PICP - target|")
    ax.set_title("Width-error trade-off", loc="left", weight="bold")
    clean_axes(ax)
    ax.scatter([], [], s=80, facecolor="white", edgecolor=DARK, label="raw")
    ax.scatter([], [], s=80, facecolor=ORANGE, edgecolor=DARK, label="conformal")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    return save(fig, "fig5-panel-c-4")


def make_d1_meta_wtl(data):
    meta = data["meta"].copy()
    rows = [
        ("Cls", 5, 6, 2, TEAL),
        ("Reg", 8, 2, 1, ORANGE),
        ("Sup.", 13, 8, 3, BLUE),
    ]
    fig = fig_px(1660, 760)
    ax = fig.add_axes([0.14, 0.18, 0.80, 0.70])
    y = np.arange(len(rows))[::-1]
    colors = {"win": BLUE, "tie": "#cfd6df", "loss": RED}
    for yi, (label, win, tie, loss, col) in zip(y, rows):
        total = win + tie + loss
        left = 0
        for name, val in [("win", win), ("tie", tie), ("loss", loss)]:
            ax.barh(yi, val / total, left=left, color=colors[name], edgecolor="white", height=0.58)
            if val:
                ax.text(left + val / total / 2, yi, str(val), ha="center", va="center", color="white" if name != "tie" else DARK, weight="bold")
            left += val / total
        ax.text(1.02, yi, f"{win}W/{tie}T/{loss}L", ha="left", va="center", weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1.28)
    ax.set_xlabel("Share of non-missing seed-dataset comparisons")
    ax.set_title("Meta: validation-selected ensemble vs best single", loc="left", weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        [Rectangle((0, 0), 1, 1, color=colors["win"]), Rectangle((0, 0), 1, 1, color=colors["tie"]), Rectangle((0, 0), 1, 1, color=colors["loss"])],
        ["win", "tie", "loss"],
        frameon=False,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.50, 1.04),
        borderaxespad=0.0,
    )
    return save(fig, "fig5-panel-d-1")


def make_d2_perf_matrix(data):
    perf = data["perf"].copy()
    scopes = ["classification", "regression", "clustering", "all"]
    comparisons = ["wo_feature", "wo_closed_loop", "wo_meta"]
    label_scope = {"classification": "Cls", "regression": "Reg", "clustering": "Clu", "all": "All"}
    label_comp = {"wo_feature": "w/o Feat.", "wo_closed_loop": "w/o Loop", "wo_meta": "w/o Meta"}
    mat = np.zeros((len(comparisons), len(scopes)))
    txt = [["" for _ in scopes] for _ in comparisons]
    for i, comp in enumerate(comparisons):
        for j, scope in enumerate(scopes):
            row = perf[(perf.scope == scope) & (perf.comparison == comp)]
            if row.empty:
                mat[i, j] = np.nan
                txt[i][j] = "NA"
                continue
            r = row.iloc[0]
            non = max(1, int(r.win) + int(r.tie) + int(r.loss))
            mat[i, j] = (int(r.win) - int(r.loss)) / non
            txt[i][j] = f"{int(r.win)}/{int(r.tie)}/{int(r.loss)}"
    fig = fig_px(1660, 760)
    ax = fig.add_axes([0.16, 0.18, 0.72, 0.68])
    cmap = LinearSegmentedColormap.from_list("perf", [RED, "white", BLUE])
    im = ax.imshow(mat, cmap=cmap, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto", alpha=0.92)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, txt[i][j], ha="center", va="center", weight="bold")
    ax.set_xticks(np.arange(len(scopes)))
    ax.set_xticklabels([label_scope[s] for s in scopes])
    ax.set_yticks(np.arange(len(comparisons)))
    ax.set_yticklabels([label_comp[c] for c in comparisons])
    ax.set_title("Strict performance context", loc="left", weight="bold")
    ax.set_xlabel("W/T/L in each cell")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(scopes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(comparisons), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cb.set_label("W-L share")
    return save(fig, "fig5-panel-d-2")


def main():
    setup_style()
    data = read_data()
    paths = [
        make_a1_protocol(),
        make_a2_evidence_matrix(data),
        make_b1_retention(data),
        make_b2_cls_deploy(data),
        make_b3_reg_deploy(data),
        make_c1_class_cal_shift(data),
        make_c2_class_delta_heatmap(data),
        make_c3_reg_coverage(data),
        make_c4_reg_width_error(data),
        make_d1_meta_wtl(data),
        make_d2_perf_matrix(data),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
