""" NetworkX utility functions """
import networkx as nx
from common.utility import clamp_values, extend

def _compile_edges_coordinates(network, layout):
    """Compiles coordinates for edges in a networkx graph based on node coordinates.

        Parameters
        ----------
        network : nx.Graph
            The networkx graph.

        layout : dict of node to [x,y] items
            A dictionary that contains coordinates for all nodes (which are keys).

        Returns
        -------
        list of tuples (node, node, [[x1,y1], [x2,y2]]) for all edges in the graph

    """
    data = [ (u, v, d['weight'], [layout[u][0], layout[v][0]], [layout[u][1], layout[v][1]])
                for u, v, d in network.edges(data=True) ]

    return zip(*data)

def get_positioned_edges(network, layout, weight_scale=1.0, normalize_weights=False):
    """Returns edges assigned position from (nodes' position in) given layout

        Parameters
        ----------
        network : nx.Graph
            The networkx graph.

        layout : dict of node to [x,y] items
            A dictionary that contains coordinates for all nodes (which are keys).

        weight_scale : float
            Scale factor for weights.

        normalize_weights : bool
            Specifies if data should be normalized such that max(weight) is 1.

        Returns
        -------
            Dict containing edges, source and target nodes with coordinates and weights

    """
    u, v, weights, xs, ys = _compile_edges_coordinates(network, layout)
    norm = max(weights) if normalize_weights else 1.0
    weights = [ weight_scale * x / norm for x in weights ]
    edges = dict(source=u, target=v, xs=xs, ys=ys, weights=weights)
    return edges

def get_sub_network(G, threshold):
    """Creates a subgraph of G of all edges having a weight equal to or above h.

        Parameters
        ----------
        network : nx.Graph
            The networkx graph.

        threshold : float
            A dictionary that contains coordinates for all nodes (which are keys).

        Returns
        -------
            A networkx sub-greaph

    """
    max_weight = max(1.0, max(nx.get_edge_attributes(G, 'weight').values()))
    filter_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= (threshold * max_weight)]
    tng = G.edge_subgraph(filter_edges)
    return tng

def get_positioned_nodes(network, layout, nodes=None):
    """Returns nodes assigned position from given layout.

        Parameters
        ----------
        network : nx.Graph
            The networkx graph.

        layout : dict of node + point pairs i.e. (node, [x,y])
            A dictionary that contains coordinates for all nodes.

        nodes : optional list of str
            Subset of nodes to return.

        Returns
        -------
            Positioned nodes (x, y, nodes, node_id) and any additional found node attributes

    """
    layout_items = layout.items() if nodes is None else [ x for x in layout.items() if x[0] in nodes ]
    nodes, nodes_coordinates = zip(*sorted(layout_items))
    xs, ys = list(zip(*nodes_coordinates))
    list_of_attribs = [ network.nodes[k] for k in nodes ]
    attrib_lists = dict(zip(list_of_attribs[0], zip(*[d.values() for d in list_of_attribs])))
    attrib_lists.update(dict(x=xs, y=ys, name=nodes, node_id=nodes))
    return attrib_lists

def create_network(df, source_field='source', target_field='target', weight='weight'):
    """Creates a new networkx graph from values in a dataframe.

        Parameters
        ----------
        df : DataFrame
            A pandas dataframe that contains edges i.e. source/target/weight columns.

        source_field : str
            Name of column that contains source nodes.

        target_field : str
            Name of column that contains target nodes.

        weight : str
            Name of column that contains edges' weights.

        Returns
        -------
            A networkx Graph.

    """
    G = nx.Graph()
    nodes = list(set(list(df[source_field].values) + list(df[target_field].values)))
    edges = [ (x, y, { weight: z }) for x, y, z in [ tuple(x) for x in df[[source_field, target_field, weight]].values]]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def get_positioned_edges_as_dict(G, layout, weight_scale, normalize_weights):
    """Returns edges assigned position from (nodes' position in) in given layout

        Parameters
        ----------
        network : nx.Graph
            The networkx graph.

        layout : dict of node to [x,y] items
            A dictionary that contains coordinates for all nodes (which are keys).

        weight_scale : float
            Scale factor for weights.

        normalize_weights : bool
            Specifies if data should be normalized such that max(weight) is 1.

        Returns
        -------
            TODO

    """
    edges = get_positioned_edges(G, layout, weight_scale, normalize_weights)
    edges = { k: list(edges[k]) for k in edges}
    return edges

def get_positioned_nodes_as_dict(G, layout, node_size, node_size_range):

    nodes = get_positioned_nodes(G, layout)

    if node_size in nodes.keys() and node_size_range is not None:
        nodes['clamped_size'] = clamp_values(nodes[node_size], node_size_range)
        node_size = 'clamped_size'

    label_y_offset = 'y_offset' if node_size in nodes.keys() else node_size + 8
    if label_y_offset == 'y_offset':
        nodes['y_offset'] = [ y + r for (y, r) in zip(nodes['y'], [ r / 2.0 + 8 for r in nodes[node_size] ]) ]

    nodes = { k: list(nodes[k]) for k in nodes}

    return nodes

def create_bipartite_network(df, source_field='source', target_field='target', weight='weight'):

    G = nx.Graph()
    G.add_nodes_from(set(df[source_field].values), bipartite=0)
    G.add_nodes_from(set(df[target_field].values), bipartite=1)
    edges = list(zip(df[source_field].values, df[target_field].values, df[weight].apply(lambda x: dict(weight=x))))
    G.add_edges_from(edges)
    return G

def get_bipartite_node_set(network, bipartite=0):

    nodes = set(n for n, d in network.nodes(data=True) if d['bipartite'] == bipartite)
    others = set(network) - nodes
    return list(nodes), list(others)

def create_network_from_xyw_list(values):
    """A simple wrapper for networkx factory function add_weighted_edges_from

        Parameters
        ----------
        values : Edges represented in nx style as a list of (source, target, attributes) triplets

        Returns
        -------
            A new nx.Graph
    """
    G = nx.Graph()
    G.add_weighted_edges_from(values)
    return G

def network_edges_to_dicts(G, layout):
    """TODO

        Parameters
        ----------
        G : nx.Graph

        layout : dict of node to [x,y] items
            A dictionary that contains coordinates for all nodes (which are keys).

        Returns
        -------
            TODO
    """
    LD = [ extend(dict(source=u, target=v, xs=[layout[u][0], layout[v][0]], ys=[layout[u][1], layout[v][1]]), d) for u, v, d in G.edges(data=True) ]
    LD.sort(key=lambda x: x['signed'])
    edges = dict(zip(LD[0], zip(*[d.values() for d in LD])))
    return edges

def pandas_to_network_edges(data):
    """Transform a dataframe's edge data into nx style i.e. as a list of (source, target, attributes) triplets

        The source and target nodes are assumed to be the first to columns.
        Any other column are stored as attributes in an attr dictionary

        Parameters
        ----------
        data : DataFrame
            A pandas dataframe that contains edges i.e. source/target/weight columns.

        Returns
        -------
            Edges represented in nx style as a list of (source, target, attributes) triplets

    """
    return [ (x[0], x[1], { y: x[j] for j, y in enumerate(data.columns)}) for i, x in data.iterrows() ]
