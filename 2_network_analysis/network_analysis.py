import networkx as nx
import community

from common.network.networkx_utility import pandas_to_network_edges
from common.utility import clamp_values

def create_party_network(data, K, node_partition, palette): #, multigraph=True):

    #if multigraph:
    
    edges_data = pandas_to_network_edges(data)

    G = nx.MultiGraph(K=K)
    G.add_edges_from(edges_data)
    #else:
    #    edges_data = [ tuple(x) for x in data.values ]
    #    print(edges_data)
    #    G = nx.Graph(K=K)
    #    G.add_weighted_edges_from(edges_data)

    if node_partition is not None:
        partition = community.best_partition(G)
        partition_color = { n: palette[p % len(palette)] for n, p in partition.items() }
        nx.set_node_attributes(G, partition, 'community')
        nx.set_node_attributes(G, partition_color, 'fill_color')
    else:
        #nx.set_node_attributes(G, 0, 'community')
        nx.set_node_attributes(G, palette[0], 'fill_color')

    nx.set_node_attributes(G, dict(G.degree()), name='degree')
    nx.set_node_attributes(G, dict(nx.betweenness_centrality(G, weight=None)), name='betweenness')
    nx.set_node_attributes(G, dict(nx.closeness_centrality(G)), name='closeness')
    
    # if not multigraph:
    #    nx.set_node_attributes(G, dict(nx.eigenvector_centrality(G, weight=None)), name='eigenvector')
        
    # nx.set_node_attributes(G, dict(nx.communicability_betweenness_centrality(G)), name='communicability_betweenness')
    
    return G

def setup_node_size(nodes, node_size, node_size_range):

    if node_size in nodes.keys() and node_size_range is not None:
        nodes['clamped_size'] = clamp_values(nodes[node_size], node_size_range)
        node_size = 'clamped_size'
    return node_size
    
def setup_label_y_offset(nodes, node_size):

    label_y_offset = 'y_offset' if node_size in nodes.keys() else node_size + 5
    if label_y_offset == 'y_offset':
        nodes['y_offset'] = [ y + r for (y, r) in zip(nodes['y'], [ r / 2.0 + 5 for r in nodes[node_size] ]) ]
    return label_y_offset
