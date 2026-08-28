from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "outputs" / "fig6d"
OUT = REPO_ROOT / "outputs" / "fig6d_curated_support"
SUB = OUT / "subfigures"
TABLES = OUT / "tables"
SRC = OUT / "source_scripts"

DATASET_CR = ["cc1", "credit-g", "ld1", "boston", "concrete", "california"]
DATASET_CLU = ["breast", "glass", "students"]
DATASET_ALL = DATASET_CR + DATASET_CLU
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

COL = {
    "classification": "#1F78B4",
    "regression": "#C7542D",
    "clustering": "#3E9868",
    "teacher": "#17202C",
    "grid": "#E4E7EB",
    "spine": "#B7BEC8",
    "muted": "#667085",
    "text": "#1B1F24",
    "blue_soft": "#D1E7F4",
    "orange_soft": "#F0D1C4",
    "green_soft": "#DFF0E6",
    "gold": "#D9A441",
}


def setup_ext_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.4,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 170,
            "savefig.dpi": 520,
        }
    )


def task_for(ds: str) -> str:
    if ds in ["cc1", "credit-g", "ld1"]:
        return "classification"
    if ds in ["boston", "concrete", "california"]:
        return "regression"
    return "clustering"


def color_for(ds: str) -> str:
    return COL[task_for(ds)]


def style_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.set_box_aspect(0.72)
    if grid:
        ax.grid(axis=grid, color=COL["grid"], lw=0.65)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(COL["spine"])
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=COL["muted"], width=0.75, length=3.2)


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(stem.with_suffix(suffix), facecolor="white")
    plt.close(fig)


def read_data() -> dict[str, pd.DataFrame]:
    cr = pd.read_csv(ROOT / "04_panel_d_incremental_class_reg_42_46" / "panel_d_dataset_seed_enhanced.csv")
    cr = cr[cr["task"].isin(["classification", "regression"])].copy()
    selected = pd.read_csv(
        ROOT / "05_panel_d_clustering_strong_42_46" / "clustering_selected_by_train_fidelity.csv"
    )
    assign = pd.read_csv(ROOT / "05_panel_d_clustering_strong_42_46" / "clustering_assignment_fidelity.csv")
    cluster_tokens = pd.read_csv(ROOT / "05_panel_d_clustering_strong_42_46" / "llm_token_by_run.csv")
    return {"cr": cr, "selected": selected, "assign": assign, "cluster_tokens": cluster_tokens}


def fig_supervised_fidelity(data: dict[str, pd.DataFrame]) -> None:
    cr = data["cr"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    rng = np.random.default_rng(7)
    y = np.arange(len(DATASET_CR))
    ax.axvspan(0.85, 1.005, color="#F0F7F3", alpha=0.95, lw=0, zorder=0)
    ax.axvline(0.85, color="#9FB5AA", lw=1.0, ls="--", zorder=1)
    ax.axvline(0.95, color="#8C98A6", lw=1.0, ls=":", zorder=1)
    for i, ds in enumerate(DATASET_CR):
        g = cr[cr["dataset"].eq(ds)]
        vals = pd.to_numeric(g["fidelity_primary"], errors="coerce").dropna().to_numpy()
        q10, q90 = np.quantile(vals, [0.1, 0.9])
        yy = np.full(vals.size, i) + rng.normal(0, 0.035, vals.size)
        c = color_for(ds)
        ax.plot([q10, q90], [i, i], color=c, lw=7.5, alpha=0.18, solid_capstyle="round", zorder=2)
        ax.scatter(vals, yy, s=34, color=c, edgecolor="white", lw=0.55, alpha=0.88, zorder=3)
        ax.scatter(
            [vals.mean()],
            [i],
            s=78,
            color=c,
            edgecolor=COL["teacher"],
            lw=1.0,
            zorder=4,
        )
        ax.add_patch(Rectangle((0.773, i - 0.25), 0.008, 0.50, facecolor=c, edgecolor="none", alpha=0.95))
    ax.set_xlim(0.77, 1.005)
    ax.set_ylim(-0.65, len(DATASET_CR) - 0.35)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[d] for d in DATASET_CR])
    ax.set_xticks([0.80, 0.90, 1.00])
    ax.set_xlabel("Fidelity (r)")
    style_axis(ax, grid="x")
    save(fig, SUB / "ed_d1_supervised_fidelity_uniform")


def fig_delivery_gain(data: dict[str, pd.DataFrame]) -> None:
    cr = data["cr"].copy()
    cr["speed_x"] = 1.0 / pd.to_numeric(cr["latency_ratio"], errors="coerce")
    cr["size_x"] = 1.0 / pd.to_numeric(cr["artifact_size_ratio"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    rng = np.random.default_rng(8)
    for i, ds in enumerate(DATASET_CR):
        g = cr[cr["dataset"].eq(ds)]
        c = color_for(ds)
        for metric, offset, marker, alpha in [("speed_x", -0.13, "o", 0.85), ("size_x", 0.13, "s", 0.70)]:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy()
            yy = np.full(vals.size, i + offset) + rng.normal(0, 0.012, vals.size)
            q10, q90 = np.quantile(vals, [0.1, 0.9])
            ax.plot([q10, q90], [i + offset, i + offset], color=c, lw=6.0, alpha=0.16, solid_capstyle="round")
            ax.scatter(vals, yy, s=28, marker=marker, color=c, edgecolor="white", lw=0.5, alpha=alpha, zorder=3)
            ax.scatter(
                [vals.mean()],
                [i + offset],
                s=62,
                marker=marker,
                color=c,
                edgecolor=COL["teacher"],
                lw=0.9,
                zorder=4,
            )
        ax.add_patch(Rectangle((4.16, i - 0.25), 0.18, 0.50, facecolor=c, edgecolor="none", alpha=0.95))
    ax.axvspan(5, 500, color="#F5F8FB", alpha=0.95, zorder=0)
    ax.axvline(5, color="#9AA5B1", lw=1.0, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(4, 500)
    ax.set_ylim(-0.65, len(DATASET_CR) - 0.35)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(DATASET_CR)))
    ax.set_yticklabels([DISPLAY[d] for d in DATASET_CR])
    ax.set_xticks([5, 10, 50, 100, 500])
    ax.set_xticklabels(["5", "10", "50", "100", "500"])
    ax.set_xlabel("Delivery gain (x)")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6E9EC8", markeredgecolor=COL["teacher"], ms=6, label="Spd"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#6E9EC8", markeredgecolor=COL["teacher"], ms=6, label="Sz"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", ncol=2, handletextpad=0.25, columnspacing=0.8)
    style_axis(ax, grid="x")
    save(fig, SUB / "ed_d2_delivery_gain_uniform")


def fig_cluster_assignment(data: dict[str, pd.DataFrame]) -> None:
    selected = data["selected"]
    metrics = [
        ("RI", "test_rand_index", "#2E7D58"),
        ("kNN", "test_knn_overlap", "#55A873"),
        ("NMI", "test_assignment_nmi", "#84C99B"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    x = np.arange(len(DATASET_CLU))
    width = 0.18
    ax.axhspan(0.75, 1.01, color="#F0F7F3", lw=0, zorder=0)
    for j, (lab, col, c) in enumerate(metrics):
        xpos = x + (j - 1) * width
        means, lows, highs = [], [], []
        for ds in DATASET_CLU:
            vals = pd.to_numeric(selected[selected["dataset"].eq(ds)][col], errors="coerce").dropna().to_numpy()
            means.append(vals.mean())
            lows.append(np.quantile(vals, 0.1))
            highs.append(np.quantile(vals, 0.9))
        ax.bar(xpos, means, width=width * 0.92, color=c, edgecolor="white", lw=0.8, alpha=0.92, zorder=2)
        ax.errorbar(
            xpos,
            means,
            yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
            fmt="none",
            ecolor=COL["teacher"],
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )
    ax.set_ylim(0.68, 1.01)
    ax.set_yticks([0.70, 0.80, 0.90, 1.00])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in DATASET_CLU])
    ax.set_ylabel("Teacher-student")
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=c, markeredgecolor="white", ms=6, label=lab) for lab, _, c in metrics]
    ax.legend(handles=handles, frameon=False, loc="upper right", ncol=3, columnspacing=0.6, handletextpad=0.2)
    style_axis(ax, grid="y")
    save(fig, SUB / "ed_d3_cluster_assignment_uniform")


def fig_cluster_selected_train_test(data: dict[str, pd.DataFrame]) -> None:
    selected = data["selected"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    rng = np.random.default_rng(9)
    for i, ds in enumerate(DATASET_CLU):
        g = selected[selected["dataset"].eq(ds)].copy()
        train = pd.to_numeric(g["selection_train_assignment_nmi"], errors="coerce").to_numpy()
        test = pd.to_numeric(g["test_assignment_nmi"], errors="coerce").to_numpy()
        c = color_for(ds)
        for vals, offset, marker, alpha in [(train, -0.13, "o", 0.82), (test, 0.13, "D", 0.76)]:
            q10, q90 = np.quantile(vals, [0.1, 0.9])
            yy = np.full(vals.size, i + offset) + rng.normal(0, 0.012, vals.size)
            ax.plot([q10, q90], [i + offset, i + offset], color=c, lw=6.0, alpha=0.16, solid_capstyle="round")
            ax.scatter(vals, yy, s=24, marker=marker, color=c, edgecolor="white", lw=0.45, alpha=alpha, zorder=3)
            ax.scatter(
                [vals.mean()],
                [i + offset],
                s=66,
                marker=marker,
                color=c,
                edgecolor=COL["teacher"],
                lw=0.9,
                zorder=4,
            )
        ax.add_patch(Rectangle((0.525, i - 0.25), 0.012, 0.50, facecolor=c, edgecolor="none", alpha=0.95))
    ax.axvspan(0.70, 1.01, color="#F0F7F3", lw=0, zorder=0)
    ax.set_xlim(0.52, 1.01)
    ax.set_ylim(-0.65, len(DATASET_CLU) - 0.35)
    ax.invert_yaxis()
    ax.set_xticks([0.60, 0.70, 0.80, 0.90, 1.00])
    ax.set_yticks(np.arange(len(DATASET_CLU)))
    ax.set_yticklabels([DISPLAY[d] for d in DATASET_CLU])
    ax.set_xlabel("TS-NMI")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#70B98A", markeredgecolor=COL["teacher"], ms=6, label="Train"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#70B98A", markeredgecolor=COL["teacher"], ms=5.7, label="Test"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", ncol=2, handletextpad=0.25, columnspacing=0.8)
    style_axis(ax, grid="x")
    save(fig, SUB / "ed_d4_cluster_train_test_uniform")


def fig_run_qc(data: dict[str, pd.DataFrame]) -> None:
    cr = data["cr"]
    ctok = data["cluster_tokens"]
    rows = []
    for ds in DATASET_CR:
        g = cr[cr["dataset"].eq(ds)]
        rows.append(
            {
                "dataset": ds,
                "task": task_for(ds),
                "runs": f"{int(g['run_status'].eq('success').sum())}/{len(g)}",
                "fail": int(pd.to_numeric(g["failed_flag"], errors="coerce").fillna(0).sum()),
                "calls": int(pd.to_numeric(g["llm_calls"], errors="coerce").sum()),
                "tok": int(pd.to_numeric(g["llm_total_tokens"], errors="coerce").sum() / 1000),
            }
        )
    for ds in DATASET_CLU:
        g = ctok[ctok["dataset"].eq(ds)]
        rows.append(
            {
                "dataset": ds,
                "task": task_for(ds),
                "runs": f"{len(g)}/{len(g)}",
                "fail": int(g["token_abnormal"].astype(str).str.lower().eq("true").sum()),
                "calls": int(pd.to_numeric(g["llm_calls"], errors="coerce").sum()),
                "tok": int(pd.to_numeric(g["llm_total_tokens"], errors="coerce").sum() / 1000),
            }
        )
    df = pd.DataFrame(rows)
    cols = [("runs", "Run"), ("fail", "Flag"), ("calls", "Calls"), ("tok", "TokK")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.set_xlim(-0.85, len(cols))
    ax.set_ylim(-0.75, len(DATASET_ALL) - 0.35)
    ax.invert_yaxis()
    for i, row in df.iterrows():
        ds = row["dataset"]
        c = color_for(ds)
        ax.add_patch(Rectangle((-0.66, i - 0.34), 0.20, 0.68, facecolor=c, edgecolor="none", alpha=0.95))
        for j, (col, _lab) in enumerate(cols):
            val = row[col]
            if col == "fail":
                fc = "#4DA06B" if int(val) == 0 else "#D46A5D"
                txt = "0"
            elif col == "runs":
                fc = "#4DA06B"
                txt = str(val)
            else:
                fc = "#EAF3EF" if row["task"] == "clustering" else "#EAF2F8"
                txt = str(val)
            ax.add_patch(Rectangle((j + 0.06, i - 0.34), 0.78, 0.68, facecolor=fc, edgecolor="white", lw=1.1))
            ax.text(j + 0.45, i, txt, ha="center", va="center", fontsize=8.0, color=COL["text"], fontweight="bold")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([DISPLAY[d] for d in df["dataset"]])
    ax.set_xticks(np.arange(len(cols)) + 0.45)
    ax.set_xticklabels([lab for _col, lab in cols])
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    save(fig, SUB / "ed_d5_run_qc_uniform")


def write_tables(data: dict[str, pd.DataFrame]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    cr = data["cr"].copy()
    cr["speed_x"] = 1.0 / pd.to_numeric(cr["latency_ratio"], errors="coerce")
    cr["size_x"] = 1.0 / pd.to_numeric(cr["artifact_size_ratio"], errors="coerce")
    supervised = cr[
        [
            "task",
            "dataset",
            "seed",
            "run_status",
            "fidelity_primary",
            "speed_x",
            "size_x",
            "llm_calls",
            "llm_total_tokens",
        ]
    ].copy()
    supervised.to_csv(TABLES / "panel_d_support_supervised_fidelity_delivery.csv", index=False, encoding="utf-8-sig")

    selected = data["selected"][
        [
            "dataset",
            "seed",
            "config",
            "feature_setting",
            "selection_train_assignment_nmi",
            "test_assignment_nmi",
            "test_rand_index",
            "test_knn_overlap",
        ]
    ].copy()
    selected.to_csv(TABLES / "panel_d_support_clustering_assignment.csv", index=False, encoding="utf-8-sig")

    qc_rows = []
    for ds in DATASET_CR:
        g = cr[cr["dataset"].eq(ds)]
        qc_rows.append(
            {
                "dataset": ds,
                "task": task_for(ds),
                "successful_runs": int(g["run_status"].eq("success").sum()),
                "total_runs": int(len(g)),
                "failed_flags": int(pd.to_numeric(g["failed_flag"], errors="coerce").fillna(0).sum()),
                "llm_calls": int(pd.to_numeric(g["llm_calls"], errors="coerce").sum()),
                "llm_total_tokens": float(pd.to_numeric(g["llm_total_tokens"], errors="coerce").sum()),
            }
        )
    for ds in DATASET_CLU:
        g = data["cluster_tokens"][data["cluster_tokens"]["dataset"].eq(ds)]
        qc_rows.append(
            {
                "dataset": ds,
                "task": task_for(ds),
                "successful_runs": int(len(g)),
                "total_runs": int(len(g)),
                "failed_flags": int(g["token_abnormal"].astype(str).str.lower().eq("true").sum()),
                "llm_calls": int(pd.to_numeric(g["llm_calls"], errors="coerce").sum()),
                "llm_total_tokens": float(pd.to_numeric(g["llm_total_tokens"], errors="coerce").sum()),
            }
        )
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(TABLES / "panel_d_support_run_qc.csv", index=False, encoding="utf-8-sig")

    curation = pd.DataFrame(
        [
            {
                "item": "supervised teacher-student fidelity",
                "decision": "retained",
                "reason": "supports preservation of teacher-output structure in classification and regression",
            },
            {
                "item": "delivery speed and artifact-size gain",
                "decision": "retained",
                "reason": "directly supports lightweight deployability",
            },
            {
                "item": "clustering teacher-student assignment kNN/RI/NMI",
                "decision": "retained",
                "reason": "supports assignment-level preservation without conflating it with external clustering quality",
            },
            {
                "item": "run success and token abnormality checks",
                "decision": "retained",
                "reason": "supports reproducibility and excludes abnormal token consumption in the plotted runs",
            },
            {
                "item": "threshold failure flags and strict-target attempts",
                "decision": "omitted_from_display",
                "reason": "boundary/failure material; useful internally but not a positive Extended Data support panel",
            },
            {
                "item": "negative output-control split",
                "decision": "omitted_from_display",
                "reason": "contains mixed negative deltas and does not strengthen the Fig. 6d delivery claim",
            },
            {
                "item": "external ground-truth clustering ARI/NMI",
                "decision": "omitted_from_display",
                "reason": "answers a different question from teacher-student assignment fidelity and can be misread as the Fig. 6d endpoint",
            },
        ]
    )
    curation.to_csv(TABLES / "panel_d_extended_data_curation_decisions.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(TABLES / "panel_d_curated_support_source_tables.xlsx", engine="openpyxl") as writer:
        supervised.to_excel(writer, sheet_name="supervised", index=False)
        selected.to_excel(writer, sheet_name="clustering_ts", index=False)
        qc.to_excel(writer, sheet_name="run_qc", index=False)
        curation.to_excel(writer, sheet_name="curation", index=False)


def write_readme() -> None:
    readme = """# Curated Fig. 6d Extended Data

This is the display-level Fig. 6d extended-data package curated to support the final Fig. 6d manuscript claim.

## Retained display evidence

- Supervised teacher-student output fidelity for classification and regression.
- Delivery gains in speed and artifact size for the same supervised datasets.
- Clustering teacher-student assignment preservation, reported as Rand index, kNN neighbourhood overlap and teacher-student NMI.
- Selected clustering route train-to-test preservation on teacher-student NMI.
- Run-level quality control: successful runs, failure flags, LLM calls and token totals.

## Omitted from display

- Strict-threshold failure flags and target-attempt tables.
- Negative or mixed output-control split panels.
- External ground-truth clustering ARI/NMI panels, because Fig. 6d's clustering endpoint is teacher-student assignment fidelity, not external clustering quality.

The omitted records are not modified in the original experiment directories. They are simply excluded from this submission-ready Extended Data package.
"""
    (OUT / "README_curated_fig6d_extended_data.md").write_text(readme, encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "output_dir": str(OUT),
        "style_reference": "Fig. 6c extended-data visual style",
        "figsize_inches": [7.2, 4.6],
        "savefig_dpi": 520,
        "font_family": "DejaVu Sans",
        "font_size": 9.2,
        "individual_pngs": [
            "subfigures/ed_d1_supervised_fidelity_uniform.png",
            "subfigures/ed_d2_delivery_gain_uniform.png",
            "subfigures/ed_d3_cluster_assignment_uniform.png",
            "subfigures/ed_d4_cluster_train_test_uniform.png",
            "subfigures/ed_d5_run_qc_uniform.png",
        ],
        "notes": [
            "The layout and font scheme match the Fig. 6c ed1 extended-data style as closely as possible.",
            "Only display-level evidence that positively supports the final Fig. 6d claim is retained.",
            "Clustering metrics are labelled as teacher-student assignment preservation, not external ground-truth clustering quality.",
        ],
    }
    (OUT / "panel_d_curated_support_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def make_preview() -> None:
    from PIL import Image

    files = [
        SUB / "ed_d1_supervised_fidelity_uniform.png",
        SUB / "ed_d2_delivery_gain_uniform.png",
        SUB / "ed_d3_cluster_assignment_uniform.png",
        SUB / "ed_d4_cluster_train_test_uniform.png",
        SUB / "ed_d5_run_qc_uniform.png",
    ]
    images = [Image.open(p).convert("RGB") for p in files]
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    canvas = Image.new("RGB", (w * 3, h * 2), "white")
    for idx, im in enumerate(images):
        r, c = divmod(idx, 3)
        canvas.paste(im, (c * w + (w - im.width) // 2, r * h + (h - im.height) // 2))
    canvas.save(OUT / "fig6d_curated_support_preview_2x3.png", dpi=(520, 520))


def main() -> None:
    global ROOT, OUT, SUB, TABLES, SRC
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    ROOT = args.input_root.expanduser().resolve()
    OUT = args.output.expanduser().resolve()
    SUB = OUT / "subfigures"
    TABLES = OUT / "tables"
    SRC = OUT / "source_scripts"
    setup_ext_style()
    SUB.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    data = read_data()
    fig_supervised_fidelity(data)
    fig_delivery_gain(data)
    fig_cluster_assignment(data)
    fig_cluster_selected_train_test(data)
    fig_run_qc(data)
    write_tables(data)
    write_readme()
    write_manifest()
    make_preview()
    script_copy = SRC / Path(__file__).name
    script_copy.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
