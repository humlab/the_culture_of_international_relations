# %load ./network_utility.py
import math
import community # pip3 install python-louvain packages
from networkx.algorithms import bipartite
import networkx as nx
import bokeh.models as bm
import bokeh.palettes

from itertools import product

if 'extend' not in globals():
    extend = lambda a,b: a.update(b) or a
    
if 'filter_kwargs' not in globals():
    import inspect
    filter_kwargs = lambda f, args: { k:args[k] for k in args.keys() if k in inspect.getargspec(f).args }

DISTANCE_METRICS = {
    # 'Bray-Curtis': 'braycurtis',
    # 'Canberra': 'canberra',
    # 'Chebyshev': 'chebyshev',
    # 'Manhattan': 'cityblock',
    'Correlation': 'correlation',
    'Cosine': 'cosine',
    'Euclidean': 'euclidean',
    # 'Mahalanobis': 'mahalanobis',
    # 'Minkowski': 'minkowski',
    'Normalized Euclidean': 'seuclidean',
    'Squared Euclidean': 'sqeuclidean',
    'Kullback-Leibler': 'kullback–leibler',
    'Kullback-Leibler (SciPy)': 'scipy.stats.entropy'
}

class NetworkMetricHelper:
    
    @staticmethod
    def compute_centrality(network):
        centrality = nx.algorithms.centrality.betweenness_centrality(network)   
        _, nodes_centrality = zip(*sorted(centrality.items()))
        max_centrality = max(nodes_centrality)
        centrality_vector = [7 + 10 * t / max_centrality for t in nodes_centrality]
        return centrality_vector

    @staticmethod
    def compute_partition(network):
        partition = community.best_partition(network)
        p_, nodes_community = zip(*sorted(partition.items()))
        return nodes_community
    
    @staticmethod
    def partition_colors(nodes_community, color_palette=None):
        if color_palette is None:
            color_palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
                             '#ffff33','#a65628', '#b3cde3','#ccebc5','#decbe4','#fed9a6',
                             '#ffffcc','#e5d8bd','#fddaec','#1b9e77','#d95f02','#7570b3','#e7298a',
                             '#66a61e','#e6ab02','#a6761d','#666666']
        community_colors = [ color_palette[x % len(color_palette)] for x in nodes_community ]
        return community_colors
    
    @staticmethod
    def compute_alpha_vector(value_vector):
        max_value = max(value_vector)
        alphas = list(map(lambda h: 0.1 + 0.6 * (h / max_value), value_vector))
        return alphas

class NetworkUtility:
    
    @staticmethod
    def get_edge_layout_data(network, layout):

        data = [ (u, v, d['weight'], [layout[u][0], layout[v][0]], [layout[u][1], layout[v][1]])
                    for u, v, d in network.edges(data=True) ]

        return zip(*data)

    @staticmethod
    def get_edges(network, layout, scale=1.0, normalize=False):

        u, v, weights, xs, ys = NetworkUtility.get_edge_layout_data(network, layout)
        norm = max(weights) if normalize else 1.0
        weights = [ scale * x / norm for x in  weights ]
        edges = dict(source=u, target=v, xs=xs, ys=ys, weights=weights)
        return edges
   
    @staticmethod
    def get_nodes(network, layout, node_list = None):
        layout_items = layout.items() if node_list is None else [ x for x in layout.items() if x[0] in node_list ]
        nodes, nodes_coordinates = zip(*sorted(layout_items))
        xs, ys = list(zip(*nodes_coordinates))
        return dict(x=xs, y=ys, name=nodes, node_id=nodes)
    
    @staticmethod
    def create_network(df, source_field='source', target_field='target', weight='weight'):

        G = nx.Graph()
        nodes = list(set(list(df[source_field].values) + list(df[target_field].values)))
        edges = [ (x, y, { weight: z })
                 for x, y, z in [ tuple(x) for x in df[[source_field, target_field, weight]].values]]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    @staticmethod
    def create_bipartite_network(df, source_field='source', target_field='target', weight='weight'):

        G = nx.Graph()
        G.add_nodes_from(set(df[source_field].values), bipartite=0)
        G.add_nodes_from(set(df[target_field].values), bipartite=1)
        edges = list(zip(df[source_field].values,df[target_field].values,df[weight]\
                         .apply(lambda x: dict(weight=x))))
        G.add_edges_from(edges)
        return G

    @staticmethod
    def get_bipartite_node_set(network, bipartite=0):
        nodes = set(n for n,d in network.nodes(data=True) if d['bipartite']==bipartite) 
        others = set(network) - nodes
        return list(nodes), list(others)

    @staticmethod
    def create_network_from_xyw_list(values, threshold=0.0):
        G = nx.Graph()
        G.add_weighted_edges_from(values)
        return G


try:
    import graph_tool.all as gt
except:
    print('warning: graph_tool not installed!')
          
class GraphToolUtility():
    
    @staticmethod
    def get_prop_type(value, key=None):
        """
        Performs typing and value conversion for the graph_tool PropertyMap class.
        If a key is provided, it also ensures the key is in a format that can be
        used with the PropertyMap. Returns a tuple, (type name, value, key)
        """
        #if isinstance(key, unicode):
        #    # Encode the key as ASCII
        #    key = key.encode('ascii', errors='replace')

        # Deal with the value
        if isinstance(value, bool):
            tname = 'bool'

        elif isinstance(value, int):
            tname = 'float'
            value = float(value)

        elif isinstance(value, float):
            tname = 'float'

        #elif isinstance(value, unicode):
        #    tname = 'string'
        #    value = value.encode('ascii', errors='replace')

        elif isinstance(value, dict):
            tname = 'object'

        else:
            tname = 'string'
            value = str(value)

        return tname, value, key


    @staticmethod
    def nx2gt(nxG):
        """
        Converts a networkx graph to a graph-tool graph.
        """
        # Phase 0: Create a directed or undirected graph-tool Graph
        gtG = gt.Graph(directed=nxG.is_directed())

        # Add the Graph properties as "internal properties"
        for key, value in nxG.graph.items():
            # Convert the value and key into a type for graph-tool
            tname, value, key = GraphToolUtility.get_prop_type(value, key)

            prop = gtG.new_graph_property(tname) # Create the PropertyMap
            gtG.graph_properties[key] = prop     # Set the PropertyMap
            gtG.graph_properties[key] = value    # Set the actual value

        # Phase 1: Add the vertex and edge property maps
        # Go through all nodes and edges and add seen properties
        # Add the node properties first
        nprops = set() # cache keys to only add properties once
        for node in nxG.nodes():
            data = nxG.nodes[node]
            # Go through all the properties if not seen and add them.
            for key, val in data.items():
                if key in nprops: continue # Skip properties already added

                # Convert the value and key into a type for graph-tool
                tname, _, key  = GraphToolUtility.get_prop_type(val, key)

                prop = gtG.new_vertex_property(tname) # Create the PropertyMap
                gtG.vertex_properties[key] = prop     # Set the PropertyMap

                # Add the key to the already seen properties
                nprops.add(key)

        # Also add the node id: in NetworkX a node can be any hashable type, but
        # in graph-tool node are defined as indices. So we capture any strings
        # in a special PropertyMap called 'id' -- modify as needed!
        gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

        # Add the edge properties second
        eprops = set() # cache keys to only add properties once
        for src, dst, data in nxG.edges.data():

            # Go through all the edge properties if not seen and add them.
            for key, val in data.items():
                if key in eprops: continue # Skip properties already added

                # Convert the value and key into a type for graph-tool
                tname, _, key = GraphToolUtility.get_prop_type(val, key)

                prop = gtG.new_edge_property(tname) # Create the PropertyMap
                gtG.edge_properties[key] = prop     # Set the PropertyMap

                # Add the key to the already seen properties
                eprops.add(key)

        # Phase 2: Actually add all the nodes and vertices with their properties
        # Add the nodes
        vertices = {} # vertex mapping for tracking edges later
        for node in nxG.nodes():
            data = nxG.nodes[node]
            # Create the vertex and annotate for our edges later
            v = gtG.add_vertex()
            vertices[node] = v

            # Set the vertex properties, not forgetting the id property
            data['id'] = str(node)
            for key, value in data.items():
                gtG.vp[key][v] = value # vp is short for vertex_properties

        # Add the edges
        for src, dst, data in nxG.edges.data():

            # Look up the vertex structs from our vertices mapping and add edge.
            e = gtG.add_edge(vertices[src], vertices[dst])

            # Add the edge properties
            for key, value in data.items():
                gtG.ep[key][e] = value # ep is short for edge_properties

        # Done, finally!
        return gtG
