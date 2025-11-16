import sys
import warnings
from collections.abc import Sequence
from pprint import pprint as pp
from typing import Any

import ipywidgets as widgets
import pandas as pd
from bokeh.io import push_notebook
from IPython.display import display
from loguru import logger

from common import color_utility, config, utility, widgets_config
from common.network import layout as network_layout
from common.network import networkx_utility
from common.network_analysis.network_analysis import (
    adjust_node_label_offset,
    create_party_network,
    setup_node_size,
    slice_network_datasource,
)
from common.network_analysis.network_analysis_plot import get_palette, plot_network
from common.treaty_state import TreatyState

sys.path = sys.path if ".." in sys.path else sys.path + [".."]

# pylint: disable=consider-using-f-string

NETWORK_LAYOUT_OPTIONS: dict[str, Any] = {x.name: x.key for x in network_layout.layout_setups}
warnings.filterwarnings("ignore")

NETWORK_PLOT_OPTS: dict[str, Any] = {
    "x_axis_type": None,
    "y_axis_type": None,
    "background_fill_color": "white",
    "line_opts": {"color": "green", "alpha": 0.5},
    "node_opts": {"color": None, "level": "overlay", "alpha": 1.0},
}

NODE_SIZE_OPTIONS: dict[str, Any] = {
    "(default)": None,
    "Degree centrality": "degree",
    "Closeness centrality": "closeness",
    "Betweenness centrality": "betweenness",
    "Eigenvector centrality": "eigenvector",
}

PALETTE_OPTIONS: dict[str, str] = {
    palette_name: palette_name
    for palette_name in color_utility.DEFAULT_ALL_PALETTES
    if any(len(x) > 7 for x in color_utility.DEFAULT_ALL_PALETTES[palette_name].values())
}

OUTPUT_OPTIONS: dict[str, str] = {"Network": "network", "Table": "table", "Arguments": "print_args"}

COMMUNITY_OPTIONS: dict[str, Any] = {"(default)": None, "Louvain": "louvain"}
SLICE_TYPE_OPTIONS, SLICE_TYPE_DEFAULT = {"Sequence": 1, "Year": 2}, 1


def display_partial_party_network(plot_data, slice_range_type, slice_range, slice_changed_callback):
    if not plot_data or not plot_data.edges:
        return

    nodes, edges = slice_network_datasource(plot_data.edges, plot_data.nodes, slice_range, slice_range_type)

    plot_data.edges_source.data.update(edges)
    plot_data.nodes_source.data.update(nodes)

    plot_data.slice_range_type = slice_range_type
    plot_data.slice_range = slice_range

    sign_dates = plot_data.edges_source.data["signed"]

    if len(sign_dates) == 0:
        return

    min_year, max_year = min(sign_dates).year, max(sign_dates).year

    slice_changed_callback(min_year, max_year)

    print(
        "Slice range by {}: {}-{}, edge count: {} ({})".format(
            "id" if slice_range_type == 1 else "Year",
            slice_range[0],
            slice_range[1],
            len(plot_data.edges_source.data["signed"]),
            len(plot_data.edges["signed"]),
        )
    )

    push_notebook(handle=plot_data.handle)


def display_party_network(
    wti_index: TreatyState,
    *,
    parties: Sequence[str] | None = None,
    period_group_index: int = 0,
    treaty_filter: str = "",
    palette_name: str,
    plot_data: Any = None,
    topic_group: dict[str, Any] | None = None,
    recode_is_cultural: bool = False,
    layout_algorithm: str = "",
    C: float = 1.0,
    K: float = 0.10,
    p1: float = 0.10,
    output: str = "network",
    party_name: str = "party",
    node_size_range: list[int] | None = None,
    width: int = 900,
    height: int = 900,
    node_size=None,
    node_partition=None,
    treaty_sources=None,
    year_limit=None,
    progress=utility.noop,
    done_callback=None,
):
    try:
        node_size_range = node_size_range or [40, 60]
        if output == "print_args":
            args = utility.filter_dict(locals(), {"progress", "done_callback"}, filter_out=True)
            args["wti_index"] = None
            args["plot_data"] = None
            args["output"] = "network"
            pp(args)
            return

        plot_data = plot_data or utility.SimpleStruct(
            handle=None, nodes=None, edges=None, slice_range_type=2, slice_range=year_limit
        )
        weight_threshold: float = 0.0

        palette: list[str] = get_palette(palette_name)

        progress(1)
        period_group: dict[str, Any] = config.DEFAULT_PERIOD_GROUPS[period_group_index]
        kwargs: dict[str, Any] = {
            "period_group": period_group,
            "treaty_filter": treaty_filter,
            "recode_is_cultural": recode_is_cultural,
            "year_limit": year_limit,
            "treaty_sources": treaty_sources,
        }
        parties = list(parties or [])
        party_data: pd.DataFrame | None = wti_index.get_party_network(party_name, topic_group, parties, **kwargs)

        if party_data is None or party_data.shape[0] == 0:
            print("No data for selection")
            return

        if topic_group is not None:

            party_data = party_data.loc[(party_data.topic.isin(topic_group.keys()))]

            group_keys = topic_group.keys()
            line_palette = color_utility.DEFAULT_LINE_PALETTE
            line_palette_map = {k: i % len(line_palette) for i, k in enumerate(group_keys)}
            party_data["line_color"] = party_data.category.apply(lambda x: line_palette[line_palette_map[x]])

        else:
            party_data["category"] = party_data.topic

        party_data["edge_label"] = party_data.signed.apply(lambda x: x.year).astype(str) + "/" + party_data.category

        progress(2)

        # if not multigraph:
        #    data = data.groupby(['party', 'party_other']).size().reset_index().rename(columns={0: 'weight'})

        if party_data is None or party_data.shape[0] == 0:
            print("No data for selection")
            return

        G = create_party_network(party_data, K, node_partition, palette)  # , multigraph)

        progress(3)

        if output == "network":

            if weight_threshold > 0:
                G = networkx_utility.get_subgraph(G, threshold=weight_threshold)

            layout, _ = network_layout.layout_network(G, layout_algorithm, **{"scale": 1.0, "K": K, "C": C, "p": p1})

            progress(4)

            edges: dict[str, list[Any]] = networkx_utility.get_positioned_edges2(G, layout, sort_attr="signed")
            nodes: dict[str, list[Any]] = networkx_utility.get_positioned_nodes(G, layout)

            edges = {k: list(edges[k]) for k in edges}
            nodes = {k: list(nodes[k]) for k in nodes}

            node_size = setup_node_size(nodes, node_size, node_size_range)

            x_offset, y_offset = adjust_node_label_offset(nodes, node_size)

            plot_opts: dict[str, Any] = utility.extend(
                NETWORK_PLOT_OPTS,
                figsize=(width, height),
                node_size=node_size,
                node_label_opts={"y_offset": y_offset, "x_offset": x_offset},
                edge_label_opts={},
            )

            progress(5)

            data: dict[str, Any] = plot_network(nodes=nodes, edges=edges, node_label="name", edge_label="edge_label", **plot_opts)  # type: ignore

            plot_data.update(**data)

            progress(6)

            # bp.show(p)

            if done_callback is not None:
                done_callback(None)

        elif output == "table":
            party_data.columns = [{"party": "source", "party_other": "target"}.get(x, x) for x in party_data.columns]
            display(party_data)

    except Exception as ex:
        logger.error(ex)
        raise
    finally:
        progress(0)


def display_network_analyis_gui(wti_index: TreatyState, plot_data):

    box = widgets.Layout()

    def lw(w="200px"):
        return widgets.Layout(width=w)

    treaty_source_options: list[str] = wti_index.unique_sources

    party_preset_widget = widgets_config.dropdown("Presets", config.PARTY_PRESET_OPTIONS, None, layout=lw())

    # period_group=widgets_config.period_group_widget(layout=lw()),
    period_group_index_widget = widgets_config.period_group_widget(index_as_value=True, layout=lw())

    topic_group_widget = widgets_config.topic_groups_widget2(layout=lw())
    party_name_widget = widgets_config.party_name_widget(layout=lw())

    treaty_filter_widget = widgets_config.dropdown("Filter", config.TREATY_FILTER_OPTIONS, "is_cultural", layout=lw())
    recode_is_cultural_widget = widgets_config.toggle("Recode 7CORR", True, layout=lw(w="200px"))

    parties_widget = widgets_config.parties_widget(
        options=wti_index.get_countries_list(), value=["FRANCE"], layout=lw()
    )

    palette_widget = widgets_config.dropdown("Color", PALETTE_OPTIONS, None, layout=lw())
    node_size_widget = widgets_config.dropdown("Node size", NODE_SIZE_OPTIONS, None, layout=lw())
    node_size_range_widget = widgets_config.rangeslider("Range", 5, 100, [20, 49], step=1, layout=box)

    C_widget = widgets_config.slider("C", 0, 100, 1, step=1, layout=box, continuous_update=False)
    K_widget = widgets_config.sliderf("K", 0.01, 1.0, 0.01, 0.10, layout=box, continuous_update=False)
    p_widget = widgets_config.sliderf("p", 0.01, 2.0, 0.01, 1.10, layout=box, continuous_update=False)
    fig_width_widget = widgets_config.slider("Width", 600, 1600, 900, step=100, layout=box, continuous_update=False)
    fig_height_widget = widgets_config.slider("Height", 600, 1600, 700, step=100, layout=box, continuous_update=False)
    output_widget = widgets_config.dropdown("Output", OUTPUT_OPTIONS, "network", layout=lw())
    layout_algorithm_widget = widgets_config.dropdown("Layout", NETWORK_LAYOUT_OPTIONS, "nx_spring_layout", layout=lw())
    progress_widget = widgets_config.progress(0, 4, 1, 0, layout=lw("300px"))
    node_partition_widget = widgets_config.dropdown("Partition", COMMUNITY_OPTIONS, None, layout=lw())
    simple_mode_widget = widgets_config.toggle("Simple", False, tooltip="Simple view", layout=lw(w="200px"))
    treaty_sources_widget = widgets_config.select_multiple(
        description="Sources", options=treaty_source_options, values=["UNTS"], disabled=False, layout=lw("200px")
    )

    slice_range_type_widget = widgets_config.dropdown("Unit", SLICE_TYPE_OPTIONS, SLICE_TYPE_DEFAULT, layout=lw())
    time_travel_range_widget = widgets_config.rangeslider(
        "Time travel", 0, 100, [0, 100], layout=lw("60%"), continuous_update=False
    )
    time_travel_label_widget = widgets.Label(value="")

    label_map: dict[str, dict[str, str]] = {
        "nx_spring_layout": {"K": "k", "C": "_", "p": "_"},
        "nx_spectral_layout": {"K": "_", "C": "_", "p": "_"},
        "nx_circular_layout": {"K": "_", "C": "_", "p": "_"},
        "nx_shell_layout": {"K": "_", "C": "_", "p": "_"},
        "nx_kamada_kawai_layout": {"K": "_", "C": "_", "p": "_"},
        "other": {"K": "K", "C": "_", "p": "_"},
    }

    def on_layout_type_change(change):
        layout = change["new"]
        opts: dict[str, str] = label_map.get(layout, label_map.get("other", {}))
        gv: list[str] = ["graphviz_neato", "graphviz_dot", "graphviz_circo", "graphviz_fdp", "graphviz_sfdp"]
        C_widget.disabled = True
        C_widget.description = " " if C_widget.disabled else opts.get("C")
        K_widget.disabled = layout not in ["nx_spring_layout"] + gv
        K_widget.description = " " if K_widget.disabled else opts.get("K", " ")
        p_widget.disabled = True
        p_widget.description = " " if p_widget.disabled else opts.get("p", " ")

    def on_simple_mode_value_change(change):
        display_mode = "none" if change["new"] is True else ""
        node_partition_widget.layout.display = display_mode
        node_size_widget.layout.display = display_mode
        node_size_range_widget.layout.display = display_mode
        layout_algorithm_widget.layout.display = display_mode
        C_widget.layout.display = display_mode
        K_widget.layout.display = display_mode
        p_widget.layout.display = display_mode
        fig_width_widget.layout.display = display_mode
        fig_height_widget.layout.display = display_mode
        palette_widget.layout.display = display_mode
        if change["new"] is True:
            on_layout_type_change({"new": layout_algorithm_widget.value})

    def progress_callback(step=0):
        progress_widget.value = step

    def update_slice_range(slice_range_type):
        """Called whenever display of new plot_data is done."""
        if not plot_data or not plot_data.edges:
            return

        slice_range_type = slice_range_type or plot_data.slice_range_type
        sign_dates = plot_data.edges["signed"]

        if slice_range_type == 1:
            range_min, range_max = 0, len(sign_dates)
        else:
            range_min, range_max = min(sign_dates).year, max(sign_dates).year

        plot_data.update(slice_range_type=slice_range_type, slice_range=(range_min, range_max))

        time_travel_range_widget.min = 0
        time_travel_range_widget.max = 10000
        time_travel_range_widget.min = range_min
        time_travel_range_widget.max = range_max
        time_travel_range_widget.value = (range_min, range_max)

    def slice_changed_callback(min_year, max_year):
        time_travel_range_widget.description = f"{min_year}-{max_year}"

    def on_slice_range_type_change(change):
        update_slice_range(change["new"])

    def on_party_preset_change(change):  # pylint: disable=W0613
        if party_preset_widget.value is None:
            return
        parties_widget.value = (
            parties_widget.options if "ALL" in party_preset_widget.value else party_preset_widget.value
        )

    slice_range_type_widget.observe(on_slice_range_type_change, names="value")
    simple_mode_widget.observe(on_simple_mode_value_change, names="value")
    layout_algorithm_widget.observe(on_layout_type_change, names="value")
    party_preset_widget.observe(on_party_preset_change, names="value")

    simple_mode_widget.value = True

    wn = widgets.interactive(
        display_party_network,
        parties=parties_widget,
        period_group_index=period_group_index_widget,
        treaty_filter=treaty_filter_widget,
        plot_data=widgets.fixed(plot_data),
        topic_group=topic_group_widget,
        recode_is_cultural=recode_is_cultural_widget,
        layout_algorithm=layout_algorithm_widget,
        C=C_widget,
        K=K_widget,
        p1=p_widget,
        output=output_widget,
        party_name=party_name_widget,
        node_size_range=node_size_range_widget,
        palette_name=palette_widget,
        width=fig_width_widget,
        height=fig_height_widget,
        node_size=node_size_widget,
        node_partition=node_partition_widget,
        year_limit=widgets.fixed(None),
        wti_index=widgets.fixed(wti_index),
        treaty_sources=treaty_sources_widget,
        progress=widgets.fixed(progress_callback),
        done_callback=widgets.fixed(update_slice_range),
    )

    boxes = widgets.HBox(
        [
            widgets.VBox(
                [
                    period_group_index_widget,
                    topic_group_widget,
                    party_name_widget,
                    treaty_filter_widget,
                    party_preset_widget,
                ]
            ),
            widgets.VBox([parties_widget, treaty_sources_widget]),
            widgets.VBox(
                [
                    widgets.HBox([recode_is_cultural_widget, simple_mode_widget, progress_widget]),
                    widgets.HBox(
                        [
                            widgets.VBox(
                                [
                                    output_widget,
                                    layout_algorithm_widget,
                                    palette_widget,
                                    node_size_widget,
                                    node_partition_widget,
                                ]
                            ),
                            widgets.VBox(
                                [
                                    K_widget,
                                    C_widget,
                                    p_widget,
                                    fig_width_widget,
                                    fig_height_widget,
                                    node_size_range_widget,
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )

    display(widgets.VBox([boxes, wn.children[-1]]))

    wn.update()

    iw_time_travel = widgets.interactive(
        display_partial_party_network,
        plot_data=widgets.fixed(plot_data),
        slice_range_type=slice_range_type_widget,
        slice_range=time_travel_range_widget,
        slice_changed_callback=widgets.fixed(slice_changed_callback),
    )

    time_travel_box = widgets.VBox(
        [
            widgets.HBox([time_travel_label_widget, time_travel_range_widget, slice_range_type_widget]),
            iw_time_travel.children[-1],
        ]
    )

    display(time_travel_box)

    slice_range_type_widget.value = 2
