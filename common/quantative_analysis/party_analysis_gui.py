import datetime
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pprint import pprint as pp
from typing import Any

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from loguru import logger

from common import color_utility, config, treaty_utility, utility, widgets_config
from common.treaty_state import TreatyState

from . import analysis_data, analysis_plot

OTHER_CATEGORY_OPTIONS: dict[str, str] = {
    "Other category": "other_category",
    "All category": "all_category",
    "Nothing": "",
}


def display_quantity_by_party(
    *,
    period_group_index: int = 0,
    chart_type_name: str,
    party_name: str = "",
    parties: list[str],
    wti_index: TreatyState,
    year_limit: Sequence[int] | None,
    treaty_filter: str = "",
    extra_category: str = "",
    normalize_values: bool = False,
    plot_style: str = "classic",
    top_n_parties: int = 5,
    overlay: bool = True,
    progress=utility.noop,
    print_args: bool = False,
    treaty_sources: list[str] | None = None,
    vmax: float | None = None,
    legend: bool = True,
    output_filename: str | None = None,
):
    try:

        chart_type: config.KindOfChart = config.CHART_TYPE_MAP[chart_type_name]

        static_color_map: color_utility.StaticColorMap = color_utility.get_static_color_map()

        if print_args or (chart_type_name == "print_args"):
            args = utility.filter_dict(locals(), set(["progress", "print_args"]), filter_out=True)
            args["wti_index"] = None
            pp(args)

        progress()

        period_group: dict[str, Any] = config.DEFAULT_PERIOD_GROUPS[period_group_index]
        chart_type: config.KindOfChart = config.CHART_TYPE_MAP[chart_type_name]

        parties = list(parties or [])

        if period_group["type"] == "range":
            period_group = treaty_utility.trim_period_group(period_group, year_limit)
        else:
            year_limit = None

        data: pd.DataFrame = analysis_data.QuantityByParty.get_treaties_statistics(
            wti_index,
            period_group=period_group,
            party_name=party_name,
            parties=parties,
            treaty_filter=treaty_filter,
            extra_category=extra_category,
            n_top=top_n_parties,
            year_limit=year_limit,
            treaty_sources=treaty_sources,
        )

        progress()

        if data.shape[0] == 0:
            print("No data for selection")
            return

        pivot = pd.pivot_table(data, index=["Period"], values=["Count"], columns=["Party"], fill_value=0)
        pivot.columns = [x[-1] for x in pivot.columns]

        if normalize_values is True:
            pivot = pivot.div(0.01 * pivot.sum(1), axis=0)
            data["Count"] = data.groupby(["Period"]).transform(lambda x: 100.0 * (x / x.sum()))

        progress()

        if chart_type.name.startswith("plot"):

            columns: Sequence[str] = pivot.columns.tolist()

            pivot: pd.DataFrame = pivot.reset_index()[columns]
            colors: list[str] = static_color_map.get_palette(columns)

            kwargs: dict[str, Any] = analysis_plot.prepare_plot_kwargs(
                pivot, chart_type, normalize_values, period_group, vmax=vmax
            )
            kwargs.update({"overlay": overlay, "colors": colors})
            kwargs.update({"legend": legend})

            progress()

            kwargs.update(ylim=(0, vmax))
            ax = analysis_plot.quantity_plot(data, pivot, chart_type, plot_style, **kwargs)

            if output_filename:
                basename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                ax.savefig(f"{basename}_{output_filename}.eps", format="eps", dpi=300)

        elif chart_type.name == "table":
            display(data)
        else:
            display(pivot)

        progress()

    except Exception as ex:  # pylint: disable=broad-except
        logger.error(ex)
        # raise
    finally:  # pylint: disable=lost-exception
        progress(0)


@dataclass
class PartyAnalysisGUI:
    year_limit: widgets.IntRangeSlider
    sources: widgets.SelectMultiple
    period_group_index: widgets.Dropdown
    party_name: widgets.Dropdown
    normalize_values: widgets.ToggleButton
    chart_type_name: widgets.Dropdown

    plot_style: widgets.Dropdown
    top_n_parties: widgets.IntRangeSlider
    party_preset: widgets.Dropdown
    parties: widgets.SelectMultiple
    treaty_filter: widgets.Dropdown
    extra_category: widgets.Dropdown

    progress: widgets.IntProgress
    info: widgets.Label


def display_gui(wti_index, print_args=False):

    def lw(width="100px", left="0"):
        return widgets.Layout(width=width, left=left)

    def period_group_window(period_group_index):
        """Returns (min_year, max_year) for the period group.

        Periods are either a list of years, or a list of tuples (from-year, to-year)
        """
        period_group = config.DEFAULT_PERIOD_GROUPS[period_group_index]

        periods = period_group["periods"]

        if period_group["type"] == "divisions":
            periods = list(itertools.chain(*periods))

        return min(periods), max(periods)

    treaty_source_options = wti_index.unique_sources
    party_preset_options = wti_index.get_party_preset_options()

    period_group_index_widget = widgets_config.period_group_widget(index_as_value=True)

    min_year, max_year = period_group_window(period_group_index_widget.value)

    gui = PartyAnalysisGUI(
        year_limit=widgets_config.rangeslider(
            "Window", min_year, max_year, [min_year, max_year], layout=lw("900px"), continuous_update=False
        ),
        sources=widgets_config.select_multiple(
            description="Sources",
            options=treaty_source_options,
            values=treaty_source_options,
            disabled=False,
            layout=lw("200px"),
        ),
        period_group_index=period_group_index_widget,
        party_name=widgets_config.party_name_widget(),
        normalize_values=widgets_config.toggle("Display %", False, icon="", layout=lw("100px")),
        chart_type_name=widgets_config.dropdown(
            "Output", config.CHART_TYPE_NAME_OPTIONS, "plot_stacked_bar", layout=lw("200px")
        ),
        plot_style=widgets_config.plot_style_widget(),
        top_n_parties=widgets_config.slider("Top #", 0, 10, 0, continuous_update=False, layout=lw(width="200px")),
        party_preset=widgets_config.dropdown("Presets", party_preset_options, None, layout=lw(width="200px")),
        parties=widgets_config.parties_widget(
            options=wti_index.get_countries_list(excludes=["ALL", "ALL OTHER"]),
            value=["FRANCE"],
            rows=8,
        ),
        treaty_filter=widgets_config.dropdown(
            "Filter", config.TREATY_FILTER_OPTIONS, "is_cultural", layout=lw(width="200px")
        ),
        extra_category=widgets_config.dropdown("Include", OTHER_CATEGORY_OPTIONS, "", layout=lw(width="200px")),
        # overlay_option = widgets_config.toggle('Overlay', True, icon='', layout=lw()),
        progress=widgets_config.progress(0, 5, 1, 0, layout=lw("95%")),
        info=widgets.Label(value="", layout=lw("95%")),
    )

    def stepper(step=None):
        gui.progress.value = gui.progress.value + 1 if step is None else step

    itw = widgets.interactive(
        display_quantity_by_party,
        period_group_index=period_group_index_widget,
        year_limit=gui.year_limit,
        party_name=gui.party_name,
        parties=gui.parties,
        treaty_filter=gui.treaty_filter,
        extra_category=gui.extra_category,
        normalize_values=gui.normalize_values,
        chart_type_name=gui.chart_type_name,
        plot_style=gui.plot_style,
        top_n_parties=gui.top_n_parties,
        overlay=widgets.fixed(False),  # overlay_option,
        progress=widgets.fixed(stepper),
        wti_index=widgets.fixed(wti_index),
        print_args=widgets.fixed(print_args),
        treaty_sources=gui.sources,
        vmax=widgets.fixed(None),
        legend=widgets.fixed(True),
        output_filename=widgets.fixed(None),
    )

    def on_party_preset_change(change):  # pylint: disable=W0613

        if gui.party_preset.value is None:
            return

        try:
            gui.parties.unobserve(on_parties_change, names="value")
            gui.top_n_parties.unobserve(on_top_n_parties_change, names="value")
            if "ALL" in gui.party_preset.value:
                gui.parties.value = gui.parties.options
            else:
                gui.parties.value = gui.party_preset.value

            if gui.top_n_parties.value > 0: 
                gui.top_n_parties.value = 0
        except Exception as ex:  # pylint: disable=W0703
            logger.info(ex)
        finally:
            gui.parties.observe(on_parties_change, names="value")
            gui.top_n_parties.observe(on_top_n_parties_change, names="value")

    def on_parties_change(change):  # pylint: disable=W0613
        try:
            if gui.top_n_parties.value != 0:
                gui.top_n_parties.unobserve(on_top_n_parties_change, names="value")
                gui.top_n_parties.value = 0
                gui.top_n_parties.observe(on_top_n_parties_change, names="value")
        except Exception as ex:  # pylint: disable=W0703
            logger.info(ex)

    def on_top_n_parties_change(change):  # pylint: disable=W0613
        try:
            if gui.top_n_parties.value > 0:
                gui.parties.unobserve(on_parties_change, names="value")
                gui.parties.disabled = True
                gui.party_preset.disabled = True
                if len(gui.parties.value) > 0:
                    gui.parties.value = []
            else:
                gui.parties.observe(on_parties_change, names="value")
                gui.parties.disabled = False
                gui.party_preset.disabled = False
        except Exception as ex:  # pylint: disable=W0703
            logger.info(ex)

    def set_years_window(period_group_index):
        try:
            min_year, max_year = period_group_window(period_group_index)
            gui.year_limit.min, gui.year_limit.max = min_year, max_year
            gui.year_limit.value = (min_year, max_year)
            period_group = config.DEFAULT_PERIOD_GROUPS[period_group_index]
            gui.year_limit.disabled = period_group["type"] != "range"
        except Exception as ex:  # pylint: disable=W0703
            logger.info(ex)

    def on_period_change(change):
        period_group_index = change["new"]
        set_years_window(period_group_index)

    gui.parties.observe(on_parties_change, names="value")
    gui.period_group_index.observe(on_period_change, names="value")
    gui.party_preset.observe(on_party_preset_change, names="value")
    gui.top_n_parties.observe(on_top_n_parties_change, names="value")

    set_years_window(gui.period_group_index.value)

    boxes = widgets.HBox(
        [
            widgets.VBox([gui.period_group_index, gui.party_name, gui.top_n_parties, gui.party_preset]),
            widgets.VBox([gui.parties]),
            widgets.VBox(
                [
                    widgets.HBox(
                        [
                            widgets.VBox([gui.treaty_filter, gui.extra_category, gui.sources]),
                            widgets.VBox([gui.chart_type_name, gui.plot_style]),
                            widgets.VBox([gui.normalize_values]),
                        ]
                    ),
                    gui.progress,
                    gui.info,
                ]
            ),
        ]
    )

    display(widgets.VBox([boxes, gui.year_limit, itw.children[-1]]))
    itw.update()
