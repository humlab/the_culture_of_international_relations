import networkx as nx
from common.utility import clamp_values, extend

def get_edge_layout_data(network, layout):
    """Creates layout data, i.e. nodes coordinates for a networkx graph's edges.
    
        Parameters
        ----------
        network : nx.Graph
            The networkx graph.
            
        layout : dict of (node-id, [x,y])
            A dictionary that contains coordinates for all nodes.
            
        Returns
        -------
        list of tuples (node, node, [[x1,y1], [x2,y2]]) for all edges in the graph
        
    """
    data = [ (u, v, d['weight'], [layout[u][0], layout[v][0]], [layout[u][1], layout[v][1]])
                for u, v, d in network.edges(data=True) ]

    return zip(*data)

def get_edges(network, layout, scale=1.0, normalize=False):

    u, v, weights, xs, ys = get_edge_layout_data(network, layout)
    norm = max(weights) if normalize else 1.0
    weights = [ scale * x / norm for x in  weights ]
    edges = dict(source=u, target=v, xs=xs, ys=ys, weights=weights)
    return edges

def filter_by_weight(G, threshold):
    ''' Note: threshold is percent of max weight '''
    max_weight = max(1.0, max(nx.get_edge_attributes(G, 'weight').values()))
    filter_edges = [(u, v) for u, v, d in G.edges(data=True) \
                    if d['weight'] >= (threshold * max_weight)]
    tng = G.edge_subgraph(filter_edges)
    return tng

def get_node_attributes(network, layout, node_list = None):

    layout_items = layout.items() if node_list is None else [ x for x in layout.items() if x[0] in node_list ]

    nodes, nodes_coordinates = zip(*sorted(layout_items))

    xs, ys = list(zip(*nodes_coordinates))

    list_of_attribs = [ network.nodes[k] for k in nodes ]

    attrib_lists = dict(zip(list_of_attribs[0],zip(*[d.values() for d in list_of_attribs])))

    attrib_lists.update(dict(x=xs, y=ys, name=nodes, node_id=nodes))

    return attrib_lists

def create_network(df, source_field='source', target_field='target', weight='weight'):

    G = nx.Graph()
    nodes = list(set(list(df[source_field].values) + list(df[target_field].values)))
    edges = [ (x, y, { weight: z }) for x, y, z in [ tuple(x) for x in df[[source_field, target_field, weight]].values]]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def get_edge_data(G, layout, weight_scale, normalize_weights):
    edges = get_edges(G, layout, scale=weight_scale, normalize=normalize_weights)
    edges = { k: list(edges[k]) for k in edges}
    return edges

def get_node_data(G, layout, node_size, node_size_range):
    
    nodes = get_node_attributes(G, layout)

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
    edges = list(zip(df[source_field].values,df[target_field].values,df[weight]\
                     .apply(lambda x: dict(weight=x))))
    G.add_edges_from(edges)
    return G

def get_bipartite_node_set(network, bipartite=0):
    nodes = set(n for n,d in network.nodes(data=True) if d['bipartite']==bipartite) 
    others = set(network) - nodes
    return list(nodes), list(others)

def create_network_from_xyw_list(values, threshold=0.0):
    G = nx.Graph()
    G.add_weighted_edges_from(values)
    return G

def network_edges_to_dicts(G, layout):
    LD = [ extend(dict(source=u,target=v,xs=[layout[u][0], layout[v][0]], ys=[layout[u][1], layout[v][1]]), d) for u, v, d in G.edges(data=True) ]
    LD.sort(key=lambda x: x['signed'])
    edges = dict(zip(LD[0],zip(*[d.values() for d in LD])))
    return edges

def pandas_to_network_edges(data):
    return [ (x[0], x[1], { y: x[j] for j, y in enumerate(data.columns)}) for i, x in data.iterrows() ]
