import networkx as nx
import pandas as pd
import community
import datetime

from enum import Enum
from common.network.networkx_utility import pandas_to_network_edges
from common.utility import clamp_values

def create_party_network(data, K, node_partition, palette):  # , multigraph=True):

    edges_data = pandas_to_network_edges(data)

    G = nx.MultiGraph(K=K)
    G.add_edges_from(edges_data)

    if node_partition is not None:
        partition = community.best_partition(G)
        partition_color = { n: palette[p % len(palette)] for n, p in partition.items() }
        nx.set_node_attributes(G, partition, 'community')
        nx.set_node_attributes(G, partition_color, 'fill_color')
    else:
        # nx.set_node_attributes(G, 0, 'community')
        nx.set_node_attributes(G, palette[0], 'fill_color')

    nx.set_node_attributes(G, dict(G.degree()), name='degree')
    nx.set_node_attributes(G, dict(nx.betweenness_centrality(G, weight=None)), name='betweenness')
    nx.set_node_attributes(G, dict(nx.closeness_centrality(G)), name='closeness')

    return G

def slice_network_datasorce(edges, nodes, slice_range, slice_range_type):
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
        df_edges = df_edges.loc[slice_range[0]:slice_range[1]]
    else:
        lower, upper = datetime.date(slice_range[0],1,1), datetime.date(slice_range[1],1,1)
        df_edges = df_edges[(df_edges.signed >= lower) & (df_edges.signed < upper)]

    nodes_indices = set(df_edges.source.unique()) | set(df_edges.target.unique())
    df_nodes = pd.DataFrame(nodes).set_index('node_id').loc[nodes_indices].reset_index()

    sliced_edges = df_edges.to_dict(orient='list')
    sliced_nodes = df_nodes.to_dict(orient='list')

    return sliced_nodes, sliced_edges

def setup_node_size(nodes, node_size, node_size_range):

    if node_size in nodes.keys() and node_size_range is not None:
        nodes['clamped_size'] = clamp_values(nodes[node_size], node_size_range)
        node_size = 'clamped_size'

    return node_size

def adjust_node_label_offset(nodes, node_size):

    label_x_offset = 0
    label_y_offset = 'y_offset' if node_size in nodes.keys() else node_size + 5
    if label_y_offset == 'y_offset':
        nodes['y_offset'] = [ y + r for (y, r) in zip(nodes['y'], [ r / 2.0 + 5 for r in nodes[node_size] ]) ]
    return label_x_offset, label_y_offset
