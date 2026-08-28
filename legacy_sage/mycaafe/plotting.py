import matplotlib as mp
import seaborn as sb


def set_labels(
    axes,
    title=None,
    xlabel=None,
    ylabel=None,
    x_labels=None,
    y_labels=None,
    x_tick_params=None,
    y_tick_params=None,
    legend_title=None,
):

    axes.set_title("" if not title else title, fontsize="xx-large")
    axes.set_xlabel("" if not xlabel else xlabel, fontsize="x-large")
    axes.set_ylabel("" if not ylabel else ylabel, fontsize="x-large")
    if not x_tick_params:
        x_tick_params = {}
    if not y_tick_params:
        y_tick_params = {}
    axes.tick_params(axis="x", labelsize="x-large", **x_tick_params)
    axes.tick_params(axis="y", labelsize="x-large", **y_tick_params)
    if x_labels is not None:
        axes.set_xticklabels(x_labels)
    if y_labels is not None:
        axes.set_yticklabels(y_labels)
    legend = axes.get_legend()
    if legend:
        legend_title = legend_title or legend.get_title().get_text()
        legend.set_title(legend_title, prop=dict(size="x-large"))
        for text in legend.get_texts():
            text.set_fontsize("x-large")


def set_scales(axes, xscale=None, yscale=None):
    if isinstance(xscale, str):
        axes.set_xscale(xscale)
    elif isinstance(xscale, tuple):
        axes.set_xscale(xscale[0], **xscale[1])
    if isinstance(yscale, str):
        axes.set_yscale(yscale)
    elif isinstance(yscale, tuple):
        axes.set_yscale(yscale[0], **yscale[1])


def set_limits(axes, xlim=None, ylim=None):
    if isinstance(xlim, tuple):
        axes.set_xlim(xlim)
    if isinstance(xlim, dict):
        axes.set_xlim(**xlim)
    if isinstance(ylim, tuple):
        axes.set_ylim(ylim)
    if isinstance(ylim, dict):
        axes.set_ylim(**ylim)


def draw_stripplot(
    df,
    x,
    y,
    hue,
    xscale="linear",
    xbound=None,
    hue_order=None,
    xlabel=None,
    ylabel=None,
    y_labels=None,
    title=None,
    legend_title=None,
    legend_loc="best",
    legend_labels=None,
    colormap="colorblind",
    size=None,
):
    with sb.axes_style(
        "whitegrid",
        rc={
            "grid.linestyle": "dotted",
            "font.family": "serif",
            "font.serif": "Times New Roman",
        },
    ), sb.plotting_context("paper"):
        # print(sb.axes_style())
        # Initialize the figure
        strip_fig, axes = mp.pyplot.subplots(
            dpi=120, figsize=size or (10, len(df.index.unique()))
        )
        set_scales(axes, xscale=xscale)
        if xbound is not None:
            axes.set_autoscalex_on(False)
            axes.set_xbound(*xbound)
            # axes.invert_xaxis()
        sb.despine(bottom=True, left=True)

        # Show each observation with a scatterplot
        sb.stripplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            hue_order=hue_order,
            palette=colormap,
            dodge=True,
            jitter=False,
            alpha=0.15,
            zorder=1,
            linewidth=0.2,
        )

        # Show the conditional means
        sb.pointplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            hue_order=hue_order,
            palette=colormap,
            dodge=0.5,
            join=False,
            markers="d",
            scale=1.1,
            ci=None,
        )

        # Improve the legend
        handles, labels = axes.get_legend_handles_labels()
        dist = int(len(labels) / 2)
        handles, labels = handles[dist:], labels[dist:]
        if legend_labels is not None:
            if isinstance(legend_labels, list):
                labels = legend_labels
            else:
                labels = map(legend_labels, labels)
        print("Legend", handles, labels)

        axes.legend(
            handles,
            labels,
            title=legend_title or hue,
            handletextpad=0,
            columnspacing=1,
            loc=legend_loc,
            ncol=1,
            frameon=True,
        )
        # draw vertical line from (70,100) to (70, 250)
        for i in range(0, 14):
            axes.plot([0, 1], [i + 0.5, i + 0.5], "--", lw=0.5, color="gray")
        axes.plot([0, 1], [13.5, 13.5], "--", lw=1.5, color="gray")
        axes.plot([0, 1], [10.5, 10.5], "--", lw=1.5, color="gray")

        # axes.get_legend().remove()
        set_labels(axes, title=title, xlabel=xlabel, ylabel=ylabel, y_labels=y_labels)
        return strip_fig
