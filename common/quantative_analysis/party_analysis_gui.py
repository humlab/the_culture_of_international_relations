import datetime
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pprint import pprint as pp
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from loguru import logger

from common import config, utility
from common.gui import widgets_config
from common.treaty_state import TreatyState, trim_period_group
from common.utils import color_utility

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
            period_group = trim_period_group(period_group, year_limit)
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
            fig = ax.figure
            display(fig)
            plt.close(fig)

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

    # Helpers
    def lw(width="100px", left="0"):
        return widgets.Layout(width=width, left=left)

    def period_group_window(period_group_index):
        period_group = config.DEFAULT_PERIOD_GROUPS[period_group_index]
        periods = period_group["periods"]
        if period_group["type"] == "divisions":
            periods = list(itertools.chain(*periods))
        return min(periods), max(periods)

    # Initial state
    treaty_source_options = wti_index.unique_sources
    party_preset_options = wti_index.get_party_preset_options()

    period_group_index_widget = widgets_config.period_group_widget(index_as_value=True)
    min_year, max_year = period_group_window(period_group_index_widget.value)

    # GUI object
    gui = PartyAnalysisGUI(
        year_limit=widgets_config.rangeslider(
            "Window",
            min_year,
            max_year,
            [min_year, max_year],
            layout=lw("900px"),
            continuous_update=False,
        ),
        sources=widgets_config.select_multiple(
            "Sources",
            treaty_source_options,
            treaty_source_options,
            layout=lw("200px"),
        ),
        period_group_index=period_group_index_widget,
        party_name=widgets_config.party_name_widget(),
        normalize_values=widgets_config.toggle("Display %", False, layout=lw("100px")),
        chart_type_name=widgets_config.dropdown(
            "Output", config.CHART_TYPE_NAME_OPTIONS, "plot_stacked_bar", layout=lw("200px")
        ),
        plot_style=widgets_config.plot_style_widget(),
        top_n_parties=widgets_config.slider("Top #", 0, 10, 0, continuous_update=False, layout=lw("200px")),
        party_preset=widgets_config.dropdown("Presets", party_preset_options, None, layout=lw("200px")),
        parties=widgets_config.parties_widget(
            options=wti_index.get_countries_list(excludes=["ALL", "ALL OTHER"]),
            value=["FRANCE"],
            rows=8,
        ),
        treaty_filter=widgets_config.dropdown(
            "Filter", config.TREATY_FILTER_OPTIONS, "is_cultural", layout=lw("200px")
        ),
        extra_category=widgets_config.dropdown("Include", OTHER_CATEGORY_OPTIONS, "", layout=lw("200px")),
        progress=widgets.IntProgress(min=0, max=5, value=0, layout=lw("95%")),
        info=widgets.Label("", layout=lw("95%")),
    )

    run_button = widgets.Button(
        description="Run",
        button_style="primary",
        icon="play",
        layout=lw("120px"),
    )

    output_widget = widgets.Output()

    # Progress callback
    def stepper(step=None):
        gui.progress.value = gui.progress.value + 1 if step is None else step

    # Execution logic
    @output_widget.capture(clear_output=True)
    def run():
        gui.progress.value = 0
        run_button.disabled = True
        gui.info.value = ""

        try:
            display_quantity_by_party(
                period_group_index=gui.period_group_index.value,
                year_limit=gui.year_limit.value,
                party_name=gui.party_name.value,
                parties=gui.parties.value,
                treaty_filter=gui.treaty_filter.value,
                extra_category=gui.extra_category.value,
                normalize_values=gui.normalize_values.value,
                chart_type_name=gui.chart_type_name.value,
                plot_style=gui.plot_style.value,
                top_n_parties=gui.top_n_parties.value,
                overlay=False,
                progress=stepper,
                wti_index=wti_index,
                print_args=print_args,
                treaty_sources=gui.sources.value,
                vmax=None,
                legend=True,
                output_filename=None,
            )
        except Exception as e:
            print("Error:", e)
        finally:
            run_button.disabled = False

    run_button.on_click(lambda _: run())

    # Interaction logic
    def on_party_preset_change(change):
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
        finally:
            gui.parties.observe(on_parties_change, names="value")
            gui.top_n_parties.observe(on_top_n_parties_change, names="value")

    def on_parties_change(change):
        if gui.top_n_parties.value != 0:
            gui.top_n_parties.value = 0

    def on_top_n_parties_change(change):
        if gui.top_n_parties.value > 0:
            gui.parties.disabled = True
            gui.party_preset.disabled = True
            gui.parties.value = []
        else:
            gui.parties.disabled = False
            gui.party_preset.disabled = False

    def on_period_change(change):
        min_year, max_year = period_group_window(change["new"])
        gui.year_limit.min = min_year
        gui.year_limit.max = max_year
        gui.year_limit.value = (min_year, max_year)
        gui.year_limit.disabled = config.DEFAULT_PERIOD_GROUPS[change["new"]]["type"] != "range"

    # Observers
    gui.parties.observe(on_parties_change, names="value")
    gui.party_preset.observe(on_party_preset_change, names="value")
    gui.top_n_parties.observe(on_top_n_parties_change, names="value")
    gui.period_group_index.observe(on_period_change, names="value")

    for w in [
        gui.year_limit,
        gui.party_name,
        gui.treaty_filter,
        gui.extra_category,
        gui.normalize_values,
        gui.chart_type_name,
        gui.plot_style,
        gui.sources,
    ]:
        w.observe(lambda change: run(), names="value")

    # Layout
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
                    run_button,
                    gui.progress,
                    gui.info,
                ]
            ),
        ]
    )

    display(widgets.VBox([boxes, gui.year_limit, output_widget]))

    # Initial run
    run()
