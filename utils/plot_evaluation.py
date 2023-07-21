import os
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle
from tensorboard_setup import read_tensorboard


def self_plot(
    data,
    fname=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    legend_loc="best",
    color_list=None,
    xlim=None,
    ylim=None,
    xtick=None,
    ytick=None,
    yline=None,
    xline=None,
    ncol=1,
    figsize_scalar=1,
    category="plot",
):
    """
    Plot single figure containing several curves.
    """
    default_cfg = dict()
    default_cfg["fig_size"] = (12, 9)
    default_cfg["dpi"] = 300
    default_cfg["pad"] = 0.5

    default_cfg["tick_size"] = 8
    default_cfg["tick_label_font"] = "Times New Roman"
    default_cfg["legend_font"] = {
        "family": "Times New Roman",
        "size": "8",
        "weight": "normal",
    }
    default_cfg["label_font"] = {
        "family": "Times New Roman",
        "size": "9",
        "weight": "normal",
    }

    default_cfg["img_fmt"] = "png"

    # pre-process
    assert isinstance(data, (dict, list, tuple))

    if isinstance(data, dict):
        data = [data]
    num_data = len(data)

    fig_size = (
        default_cfg["fig_size"] * figsize_scalar,
        default_cfg["fig_size"] * figsize_scalar,
    )
    _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

    # color list
    if (color_list is None) or len(color_list) < num_data:
        tableau_colors = cycle(mcolors.TABLEAU_COLORS)
        color_list = [next(tableau_colors) for _ in range(num_data)]

    # plot figure
    for (i, d) in enumerate(data):
        if category == "plot":
            plt.plot(d["x"], d["y"], color=color_list[i])
        elif category == "scatter":
            plt.scatter(d["x"], d["y"], color=color_list[0], s=1)
        else:
            raise NotImplemented

    # legend
    plt.tick_params(labelsize=default_cfg["tick_size"])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]

    if legend is not None:
        plt.legend(legend, loc=legend_loc, ncol=ncol, prop=default_cfg["legend_font"])

    #  label
    plt.xlabel(xlabel, default_cfg["label_font"])
    plt.ylabel(ylabel, default_cfg["label_font"])

    if yline is not None:
        plt.axhline(yline, ls=":", c="grey")
    if xline is not None:
        plt.axvline(xline, ls=":", c="grey")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xtick is not None:
        plt.xticks(xtick)
    if ytick is not None:
        plt.yticks(ytick)
    plt.tight_layout(pad=default_cfg["pad"])

    if fname is None:
        pass
    else:
        plt.savefig(fname)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def plot_all(path):
    data = read_tensorboard(path)
    figure_path = os.path.join(path, "figure")
    os.makedirs(figure_path, exist_ok=True)
    for (key, values) in data.items():
        x_label, y_label = str_edit(key)
        self_plot(
            values,
            os.path.join(figure_path, x_label + "-" + y_label + ".tiff"),
            xlabel=x_label,
            ylabel=y_label,
            color_list=["orange"],
        )
    plt.show()


def str_edit(str_):
    str_ = str_.replace("\\", "/")
    if "/" in str_:
        str_ = str_.split("/")
        str_ = str_[-1]

    str_total = str_
    x_label = None
    y_label = None
    if "-" in str_:
        str_ = str_.split("-")
        if len(str_) == 2:
            x_label = str_[1]
            y_label = str_[0]
            y_label = re.sub(r"\d\.(\s*)", "", y_label, 1)
    if x_label is None:
        x_label = "Iteration Steps"
        y_label = str_total
    return x_label, y_label
