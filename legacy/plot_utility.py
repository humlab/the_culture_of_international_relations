from collections.abc import Sequence
from typing import Any

import bokeh.models as bm
import bokeh.palettes
import networkx as nx
import pandas as pd
from bokeh.plotting import figure

from common.network.networkx_utility import get_bipartite_node_set

from .network_utility import NetworkMetricHelper, NetworkUtility

# import wordcloud


if "__file__" in globals():
    import os
    import sys

    curdir = os.path.abspath(os.path.dirname(__file__))
    if curdir not in sys.path:
        sys.path.append(curdir)


if "extend" not in globals():

    def extend(a, b):
        return a.update(b) or a


DFLT_NODE_OPTS: dict[str, str | float] = {
    "color": "green",
    "level": "overlay",
    "alpha": 1.0,
}
DFLT_EDGE_OPTS: dict[str, str | float] = {"color": "black", "alpha": 0.2}

layout_algorithms = {
    "Fruchterman-Reingold": nx.spring_layout,
    "Eigenvectors of Laplacian": nx.spectral_layout,
    "Circular": nx.circular_layout,
    "Shell": nx.shell_layout,
    "Kamada-Kawai": nx.kamada_kawai_layout,
}


class PlotNetworkUtility:

    @staticmethod
    def layout_args(layout_algorithm, network: nx.Graph, scale: float):
        args: dict[str, Any] = {}
        if layout_algorithm == "Shell" and nx.is_bipartite(network):
            nodes, other_nodes = get_bipartite_node_set(network, bipartite=0)
            args = {"nlist": [nodes, other_nodes]}

        if layout_algorithm == "Fruchterman-Reingold":
            k: float = scale  # / math.sqrt(network.number_of_nodes())
            args = {
                "dim": 2,
                "k": k,
                "iterations": 20,
                "weight": "weight",
                "scale": 0.5,
            }

        if layout_algorithm == "Kamada-Kawai":
            args = {"dim": 2, "weight": "weight", "scale": 1.0}

        return args

    @staticmethod
    def get_layout_algorithm(layout_algorithm):
        if layout_algorithm not in layout_algorithms:
            raise ValueError(f"Unknown algorithm {layout_algorithm}")
        return layout_algorithms[layout_algorithm]

    @staticmethod
    def project_series_to_range(series: pd.Series, low: float, high: float) -> pd.Series:
        norm_series: pd.Series = series / series.max()
        return norm_series.apply(lambda x: low + (high - low) * x)

    @staticmethod
    def plot_network(
        network: nx.Graph,
        layout_algorithm: str | None = None,
        scale: float = 1.0,
        threshold: float = 0.0,
        node_description: str | None = None,  # pylint: disable=unused-argument
        node_proportions: pd.Series | None = None,
        weight_scale: float = 5.0,  # pylint: disable=unused-argument
        normalize_weights: bool = True,  # pylint: disable=unused-argument
        node_opts: dict[str, str | float] | None = None,
        line_opts: dict[str, str | float] | None = None,
        element_id: str = "nx_id3",  # pylint: disable=unused-argument
        figsize: tuple[int, int] = (900, 900),
    ) -> figure:
        if threshold > 0:
            values = nx.get_edge_attributes(network, "weight").values()
            max_weight: float = max(1.0, max(values))

            print(f"Max weigth: {max_weight}")
            print(f"Mean weigth: {sum(values) / len(values)}")

            filter_edges = [(u, v) for u, v, d in network.edges(data=True) if d["weight"] >= (threshold * max_weight)]

            sub_network: nx.Graph = network.edge_subgraph(filter_edges)
        else:
            sub_network = network

        args: dict[str, Any] = PlotNetworkUtility.layout_args(layout_algorithm, sub_network, scale)
        layout = (PlotNetworkUtility.get_layout_algorithm(layout_algorithm))(sub_network, **args)
        # lines_source: bm.ColumnDataSource = NetworkUtility.get_edges_source(
        #     sub_network, layout, scale=weight_scale, normalize=normalize_weights
        # )
        nodes_source: bm.ColumnDataSource = NetworkUtility.create_nodes_data_source(sub_network, layout)

        nodes_community: Sequence[int] = NetworkMetricHelper.compute_partition(sub_network)
        community_colors: list[str] = NetworkMetricHelper.partition_colors(
            nodes_community, bokeh.palettes.Category20[20]
        )

        nodes_source.add(nodes_community, "community")
        nodes_source.add(community_colors, "community_color")

        nodes_size: int | str = 5
        if node_proportions is not None:
            # NOTE!!! By pd index - not iloc!!
            assert node_proportions is not None

            nodes_weight: pd.Series = node_proportions.loc[list(sub_network.nodes)]
            nodes_weight = PlotNetworkUtility.project_series_to_range(nodes_weight, 20, 60)
            nodes_size = "size"
            nodes_source.add(nodes_weight, nodes_size)  # type: ignore ; noqa; pylint: disable=E1101

        node_opts = DFLT_NODE_OPTS | (node_opts or {})
        line_opts = DFLT_EDGE_OPTS | (line_opts or {})

        p = figure(
            width=figsize[0],
            height=figsize[1],
            x_axis_type=None,
            y_axis_type=None,
        )
        r_nodes = p.circle("x", "y", radius=nodes_size, source=nodes_source, **node_opts)  # type: ignore

        text_opts: dict[str, str] = {
            "x": "x",
            "y": "y",
            "text": "name",
            "level": "overlay",
            "text_align": "center",
            "text_baseline": "middle",
        }

        r_nodes.glyph.fill_color = "lightgreen"  # 'community_color'

        p.add_layout(bm.LabelSet(source=nodes_source, text_color="black", **text_opts))  # type: ignore

        return p
