import wordcloud
import inspect
import matplotlib.pyplot as plt
from network_utility import NetworkMetricHelper, NetworkUtility
from widgets_utility import WidgetUtility
import networkx as nx
import bokeh.palettes
import bokeh.models as bm
from bokeh.plotting import figure

if 'extend' not in globals():
    extend = lambda a,b: a.update(b) or a
    
if 'filter_kwargs' not in globals():
    filter_kwargs = lambda f, args: { k:args[k] for k in args.keys() if k in inspect.getargspec(f).args }
    
class WordcloudUtility:
    
    def plot_wordcloud(df_data, token='token', weight='weight', **args):
        token_weights = dict({ tuple(x) for x in df_data[[token, weight]].values })
        image = wordcloud.WordCloud(**args,)
        image.fit_words(token_weights)
        plt.figure(figsize=(12, 12)) #, dpi=100)
        plt.imshow(image, interpolation='bilinear')
        plt.axis("off")
        # plt.set_facecolor('w')
        # plt.tight_layout()
        plt.show()
        

    # plot_correlation_network
TOOLS = "pan,wheel_zoom,box_zoom,reset,previewsave"

DFLT_NODE_OPTS = dict(color='green', level='overlay', alpha=1.0)
DFLT_EDGE_OPTS = dict(color='black', alpha=0.2)
DFLT_TEXT_OPTS = dict(x='x', y='y', text='name', level='overlay', text_align='center', text_baseline='middle')

layout_function_name = {
    'spring_layout': 'Fruchterman-Reingold',
    'spectral_layout': 'Eigenvectors of Laplacian',
    'circular_layout': 'Circular',
    'shell_layout': 'Shell',
    'kamada_kawai_layout': 'Kamada-Kawai'
}

layout_functions = {
    'spring_layout': nx.spring_layout,
    'spectral_layout':  nx.spectral_layout,
    'circular_layout': nx.circular_layout,
    'shell_layout': nx.shell_layout,
    'kamada_kawai_layout': nx.kamada_kawai_layout
}

def layout_args(layout_algorithm, network, scale=0.5, k=10):
    
    if layout_algorithm == 'shell_layout':
        if nx.is_bipartite(network):
            nodes, other_nodes = get_bipartite_node_set(network, bipartite=0)   
            args = dict(nlist=[nodes, other_nodes])

    if layout_algorithm == 'spring_layout':
        #k = scale / math.sqrt(network.number_of_nodes())
        args = dict(dim=2, k=k, iterations=100, weight='weight', scale=scale)

    if layout_algorithm == 'kamada_kawai_layout':
        args = dict(dim=2, weight='weight', scale=scale)

    return dict()

def adjust_edge_endpoint(p, q, d):
    
    dx, dy = q[0] - p[0], q[1] - p[1]
    alpha = math.atan2(dy, dx)
    w = (q[0] - d * math.cos(alpha), q[1] - d * math.sin(alpha))
    return w

def adjust_edge_lengths(edges, nodes):

    node2id = { x: i for i, x in enumerate(nodes['name']) }
    
    for i in range(0, len(edges['xs'])):
        
        p1 = (edges['xs'][i][0], edges['ys'][i][0])
        p2 = (edges['xs'][i][1], edges['ys'][i][1])
        
        source_id = node2id[edges['source'][i]]
        target_id = node2id[edges['target'][i]]
        
        x1, y1 = adjust_edge_endpoint(p1, p2, nodes['size'][source_id])
        x2, y2 = adjust_edge_endpoint(p2, p1, nodes['size'][target_id])
        
        edges['xs'][i][0] = x1
        edges['xs'][i][1] = x2
        edges['ys'][i][0] = y1
        edges['ys'][i][1] = y2
        
def get_layout_function(layout_name):

    layout_function = layout_functions.get(layout_name, None)
    
    def f(G, **args):
        
        args = filter_kwargs(layout_function, args)
        return layout_function(G, **args)
    
    return f

class PlotNetworkUtility:
    
    @staticmethod
    def layout_args(layout_algorithm, network, scale=0.5, k=10):
        if layout_algorithm == 'shell_layout':
            if nx.is_bipartite(network):
                nodes, other_nodes = get_bipartite_node_set(network, bipartite=0)   
                args = dict(nlist=[nodes, other_nodes])

        if layout_algorithm == 'spring_layout':
            #k = scale / math.sqrt(network.number_of_nodes())
            args = dict(dim=2, k=k, iterations=100, weight='weight', scale=scale)

        if layout_algorithm == 'kamada_kawai_layout':
            args = dict(dim=2, weight='weight', scale=scale)

        return dict()
    
    @staticmethod
    def get_layout_algorithm(layout_algorithm):
        if layout_algorithm not in layout_algorithms:
            raise Exception("Unknown algorithm {}".format(layout_algorithm))
        return layout_algorithms.get(layout_algorithm, None)
    
    @staticmethod
    def project_series_to_range(series, low, high):
        norm_series = series / series.max()
        return norm_series.apply(lambda x: low + (high - low) * x)
    
    @staticmethod
    def project_to_range(value, low, high):
        return low + (high - low) * value

    @staticmethod
    def plot_network(
        network,
        layout,
        scale=1.0,
        threshold=0.0,
        node_description=None,
        node_size_source=5,
        node_size_range=[20,40],
        weight_scale=5.0,
        normalize_weights=True,
        node_opts=None,
        line_opts=None,
        text_opts=None,
        element_id='nx_id3',
        figsize=(900, 900),
        tools=None,
        palette=bokeh.palettes.Set3[12],
        **figkwargs
    ):
        if threshold > 0:
            max_weight = max(1.0, max(nx.get_edge_attributes(network, 'weight').values()))
            filter_edges = [(u, v) for u, v, d in network.edges(data=True) \
                            if d['weight'] >= (threshold * max_weight)]
            sub_network = network.edge_subgraph(filter_edges)
        else:
            sub_network = network

        #args = PlotNetworkUtility.layout_args(layout_algorithm, sub_network, scale)
        #layout = (PlotNetworkUtility.get_layout_algorithm(layout_algorithm))(sub_network, **args)
        
        edges = NetworkUtility.get_edges(
            sub_network, layout, scale=weight_scale, normalize=normalize_weights
        )

        # if bezier:
        #    x0, x1, y0, y1 = list(zip(*[ tuple(edges['xs'][i])+tuple(edges['ys'][i]) for i in range(len(edges['xs'])) ]))
        #    edges.update(dict(x0=x0, x1=x1, y0=y0, y1=y1))
        
        nodes = NetworkUtility.get_nodes(sub_network, layout)

        edges_source = bm.ColumnDataSource(edges)
        nodes_source = bm.ColumnDataSource(nodes)
    
        nodes_community = NetworkMetricHelper.compute_partition(sub_network)
        community_colors = NetworkMetricHelper.partition_colors(nodes_community, palette)

        nodes_source.add(nodes_community, 'community')
        nodes_source.add(community_colors, 'community_color')

        node_size_specifier = 'size'
        if isinstance(node_size_source or 5, int):
            node_size_specifier = node_size_source
            label_y_offset = node_size_specifier
        else:
            # Assume list of (node, size)....
            nodes_weight_map = { n: s for (n, s) in node_size_source if n in sub_network }
            max_weight = max(nodes_weight_map.values())
            nodes_weight = [ 
                PlotNetworkUtility.project_to_range(nodes_weight_map[n]/max_weight, node_size_range[0], node_size_range[1])
                    for n in nodes_source.data['node_id']
            ]
            nodes_source.add(nodes_weight, node_size_specifier)

            y_offset = [ y+r for (y,r) in zip(nodes_source.data['y'], [ x / 2.0 + 8 for x in nodes_weight ]) ]
            nodes_source.add(y_offset, 'y_offset')
            label_y_offset = 'y_offset'
            
            #adjust_edge_lengths(edges, nodes_source.data)
            
        node_opts = extend(DFLT_NODE_OPTS, node_opts or {})
        line_opts = extend(DFLT_EDGE_OPTS, line_opts or {})
        label_opts = extend(DFLT_TEXT_OPTS, {
            'y_offset': label_y_offset,
            'text_color': 'black',
            'text_baseline': 'bottom'
        })
        label_opts = extend(label_opts, text_opts or {})

        p = figure(
            plot_width=figsize[0],
            plot_height=figsize[1],
            tools=tools or TOOLS,
            **figkwargs
        )
        
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        # r_lines = p.bezier(x0='x0', y0='y0', x1='x1', y1='y1', line_width='weights', source=edges_source, **line_opts)
        r_lines = p.multi_line('xs', 'ys', line_width='weights', source=edges_source, **line_opts)
        r_nodes = p.circle('x', 'y', size=node_size_specifier, source=nodes_source, **node_opts)
        r_nodes.glyph.fill_color = 'community_color'

        if node_description is not None:
            p.add_tools(bm.HoverTool(renderers=[r_nodes], tooltips=None, callback=WidgetUtility.\
                glyph_hover_callback(nodes_source, 'node_id', text_ids=node_description.index, \
                                     text=node_description, element_id=element_id))
            )
            
        p.add_layout(bm.LabelSet(source=nodes_source, **label_opts))

        return p
