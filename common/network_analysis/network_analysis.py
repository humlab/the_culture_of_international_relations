import datetime
from enum import Enum

import community
import networkx as nx
import numpy as np
import pandas as pd

from common.network.networkx_utility import df_to_nx_edge_format
from common.utility import clamp_values


def create_party_network(df_party_data, K, node_partition, palette):  # , multigraph=True):

    edges_data = df_to_nx_edge_format(df_party_data)

    graph: nx.MultiGraph = nx.MultiGraph(K=K)
    graph.add_edges_from(edges_data)

    if node_partition is not None:
        partition = community.best_partition(graph)
        partition_color = {n: palette[p % len(palette)] for n, p in partition.items()}
        nx.set_node_attributes(graph, partition, "community")
        nx.set_node_attributes(graph, partition_color, "fill_color")
    else:
        # nx.set_node_attributes(G, 0, 'community')
        nx.set_node_attributes(graph, palette[0], "fill_color")

    nx.set_node_attributes(graph, dict(graph.degree()), name="degree")
    try:
        nx.set_node_attributes(graph, dict(nx.betweenness_centrality(graph, weight=None)), name="betweenness")
    except:  # type: ignore ; pylint: disable=bare-except ; noqa
        ...

    nx.set_node_attributes(graph, dict(nx.closeness_centrality(graph)), name="closeness")

    return graph


def slice_network_datasource(edges, nodes, slice_range, slice_range_type) -> tuple[dict, dict]:
    """Retrieves slice of network data defined by range of sequence's index (low, high)

    Parameters
    ----------
    slice_range : (int,int)
        Specifies interval of slice

    slice_range_type : SliceRangeType
        Specifies type of range: 1=index/sequence, 2=year

    edges_data : dict of edges data (lists or constants)
        Presentation model of edge data. Note that some elements can be either a list or a constant
        keys ['source', 'target', 'xs', 'ys', 'party', 'party_other', 'signed', 'topic', 'headnote', 'weight', 'category', 'line_color']

    nodes_data : dict of nodes data (lists or constants)
        Presentation model of nodes data. Note that some elements can be either a list or a constant
        keys ['fill_color', 'degree', 'betweenness', 'closeness', 'x', 'y', 'name', 'node_id']

    Returns
    -------
    Subset of network data where edges are sliced by index or year, and nodes filtered out from sliced edges

    """
    df_edges = pd.DataFrame(edges)

    if slice_range_type == 1:  # sequence
        df_edges = df_edges.loc[slice_range[0] : slice_range[1]]
    else:
        lower = np.datetime64(f"{slice_range[0]}-01-01")
        upper = np.datetime64(f"{slice_range[1] + 1}-01-01")
        df_edges = df_edges[(df_edges.signed >= lower) & (df_edges.signed < upper)]

    nodes_indices: set[str] = set(df_edges.source.unique()) | set(df_edges.target.unique())
    df_nodes: pd.DataFrame = pd.DataFrame(nodes).set_index("node_id").loc[nodes_indices].reset_index()

    sliced_edges: dict = df_edges.to_dict(orient="list")
    sliced_nodes: dict = df_nodes.to_dict(orient="list")

    return sliced_nodes, sliced_edges


def setup_node_size(nodes, node_size, node_size_range):

    if node_size is None:
        node_size = node_size_range[0]

    if node_size in nodes.keys() and node_size_range is not None:
        nodes["clamped_size"] = clamp_values(nodes[node_size], node_size_range)
        node_size = "clamped_size"

    return node_size


def adjust_node_label_offset(nodes, node_size):

    label_x_offset = 0
    label_y_offset = "y_offset" if node_size in nodes.keys() else node_size + 5
    if label_y_offset == "y_offset":
        nodes["y_offset"] = [y + r for (y, r) in zip(nodes["y"], [r / 2.0 + 5 for r in nodes[node_size]])]
    return label_x_offset, label_y_offset
