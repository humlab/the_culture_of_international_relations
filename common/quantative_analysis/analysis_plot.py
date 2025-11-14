from collections.abc import Sequence
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

from common import color_utility, config


def quantity_plot(
    data: pd.DataFrame,
    pivot: pd.DataFrame,
    chart_type: config.KindOfChart,
    plot_style: str,
    overlay: bool = True,
    figsize: tuple[int | float, int | float] = (1000, 600),
    xlabel: str = "",
    ylabel: str = "",
    xticks: list | None = None,
    yticks=None,
    xticklabels: list[str] | None = None,
    yticklabels=None,
    xlim=None,
    ylim=None,
    dpi: int = 48,
    colors: Sequence[str] = color_utility.DEFAULT_PALETTE,
    **kwargs,
):  # pylint: disable=W0613

    if plot_style in plt.style.available:
        plt.style.use(plot_style)  # type: ignore ; noqa

    figsize = (figsize[0] / dpi, figsize[1] / dpi)

    kind: str = f'{chart_type.kind}{"h" if chart_type.horizontal else ""}'

    ax = pivot.plot(kind=kind, stacked=chart_type.stacked, figsize=figsize, color=colors, **kwargs)  # type: ignore ; noqa

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.set_title(None)  # TODO: Add as argument

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Put a legend to the right of the current axis
    # TODO: Add `loc` as argument
    if kwargs.get("legend"):
        legend = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        legend.get_frame().set_linewidth(0.0)

    if ax.get_xticklabels() is not None:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    return ax.get_figure()


def create_party_name_map(parties, palette=color_utility.DEFAULT_PALETTE):

    def rev_dict(d) -> dict[Any, Any]:
        return {v: k for k, v in d.items()}

    df: pd.DataFrame = parties.rename(columns={"short_name": "party_short_name", "country": "party_country"})
    df["party"] = df.index

    rd = df[~df.group_no.isin([0, 8])][["party", "party_short_name", "party_name", "party_country"]].to_dict()

    party_name_map = {k: rev_dict(rd[k]) for k in rd}

    party_color_map = {party: palette[i % len(palette)] for i, party in enumerate(parties.index)}

    return party_name_map, party_color_map


def vstepper(vmax: int) -> int:
    for step, value in [(1, 15), (5, 30), (10, 250), (25, 500), (100, 2000)]:
        if vmax < value:
            return step
    return 500


def prepare_plot_kwargs(
    data: pd.DataFrame,
    chart_type: config.KindOfChart,
    normalize_values: bool,
    period_group: dict[str, Any],
    vmax: float | None = None,
) -> dict[str, tuple[int, int]]:

    kwargs: dict[str, tuple[int, int]] = {"figsize": (1000, 500)}

    label: str = "Number of treaties" if not normalize_values else "Share%"

    c, v = ("x", "y") if not chart_type.horizontal else ("y", "x")

    kwargs.setdefault(f"{v}label", label)  # type: ignore ; noqa

    if period_group["type"] == "range":
        ticklabels: list[str] = [str(x) for x in period_group["periods"]]
    else:
        ticklabels = [f"{x[0]} to {x[1]}" for x in period_group["periods"]]

    kwargs.setdefault(f"{c}ticks", list(data.index))  # type: ignore ; noqa

    if vmax is None:
        vmax = data.sum(axis=1).max() if chart_type.stacked else data.max().max()

    vstep: int = vstepper(vmax)  # type: ignore ; noqa

    if not normalize_values:
        kwargs.setdefault(f"{v}ticks", list(range(0, int(vmax) + vstep, vstep)))  # type: ignore ; noqa

    if ticklabels is not None:
        kwargs.setdefault(f"{c}ticklabels", ticklabels)  # type: ignore ; noqa

    if normalize_values:
        kwargs.setdefault(f"{v}lim", [0, 100])  # type: ignore ; noqa

    return kwargs
