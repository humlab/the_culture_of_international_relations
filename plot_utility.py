import wordcloud
import inspect
import matplotlib.pyplot as plt
from network_utility import NetworkMetricHelper, NetworkUtility, GraphToolUtility
from widgets_utility import WidgetUtility
import networkx as nx
import bokeh.palettes
import bokeh.models as bm
from bokeh.plotting import figure
import graph_tool.draw as gt_draw
import graph_tool.all as gt
        
if 'extend' not in globals():
    extend = lambda a,b: a.update(b) or a
    
def filter_kwargs(f, args):
    try:
        return { k:args[k] for k in args.keys() if k in inspect.getargspec(f).args }
    except:
        return args
    
# plot_correlation_network
TOOLS = "pan,wheel_zoom,box_zoom,reset,previewsave"

DFLT_NODE_OPTS = dict(color='green', level='overlay', alpha=1.0)
DFLT_EDGE_OPTS = dict(color='black', alpha=0.2)
DFLT_TEXT_OPTS = dict(x='x', y='y', text='name', level='overlay', text_align='center', text_baseline='middle')

layout_function_name = {
    'nx (fruchterman_reingold)': 'nx_spring_layout',
    'nx (Spectral)': 'nx_spectral_layout',
    'nx (Circular)': 'nx_circular_layout',
    'nx (Shell)': 'nx_shell_layout',
    'nx (Kamada-Kawai)': 'nx_kamada_kawai_layout',
    'graph-tool (arf)': 'graphtool_arf',
    'graph-tool (sfdp)': 'graphtool_sfdp',
    'graph-tool (fruchterman_reingold)': 'graphtool_fruchterman_reingold',
    'graphviz (neato)': 'graphviz_neato',
    'graphviz (dot)': 'graphviz_dot',
    'graphviz (circo)': 'graphviz_circo',
    'graphviz (fdp)': 'graphviz_fdp',
    'graphviz (sfdp)': 'graphviz_sfdp',
}

pydot_layout = nx.nx_pydot.pydot_layout

layout_functions = {
    'nx_spring_layout': nx.spring_layout,
    'nx_spectral_layout':  nx.spectral_layout,
    'nx_circular_layout': nx.circular_layout,
    'nx_shell_layout': nx.shell_layout,
    'nx_kamada_kawai_layout': nx.kamada_kawai_layout,
    'graphtool_arf': gt_draw.arf_layout,
    'graphtool_sfdp': gt_draw.sfdp_layout,
    'graphtool_fruchterman_reingold': gt_draw.fruchterman_reingold_layout,
    'graphviz_neato': pydot_layout,
    'graphviz_dot': pydot_layout,
    'graphviz_circo': pydot_layout,
    'graphviz_fdp': pydot_layout,
    'graphviz_sfdp': pydot_layout,
}
[('graphviz ({})'.format(x), 'graphviz_{}'.format(x))
                     for x in  ['neato', 'dot', 'circo', 'fdp', 'sfdp']
                    ]

def ifextend(p, d, **kwargs):
    if p: d.update(**kwargs)

def cpif(source, target, name):
    if name in source: target.update({name: source[name]})

def layout_args(G, layout_algorithm, weight='weight', **kwargs):
    
    args = {}
    
    K = kwargs.get('K', 0.1)
    C = kwargs.get('C', 1.0)
    p = kwargs.get('p', 0.1)
    N = len(G)
    
    if layout_algorithm == 'nx_shell_layout':
        if nx.is_bipartite(G):
            nodes, other_nodes = NetworkUtility.get_bipartite_node_set(G, bipartite=0)   
            return nx.shell_layout(nlist=[nodes, other_nodes])

    if layout_algorithm == 'nx_spring_layout':
        
        args.update(k=K, weight='weight', scale=1.0)
        
        cpif(kwargs, args, 'iterations')

    if layout_algorithm == 'nx_kamada_kawai_layout':
        
        args.update(weight='weight', scale=1.0) 
        
    if layout_algorithm.startswith('graphtool'):
            
        if layout_algorithm.endswith('sfdp'):
            args.update(eweight=weight, K=K, C=C/100.0, gamma=p)
            
        if layout_algorithm.endswith('arf'):
            args.update(weight=weight, d=K, a=C)
            
        if layout_algorithm.endswith('fruchterman_reingold'):
            args.update(weight=weight, a=(2.0*N*K), r=2.0*C)
                
    if layout_algorithm.startswith('graphviz'):

        G.graph['K'] = K
        G.graph['overlap'] = False
        engine = layout_algorithm.split('_')[1]
        args.update(prog=engine)
        
    return args

def layout_graphtool_network(G, layout_algorithm, **kwargs):
    G_gt = GraphToolUtility.nx2gt(G)
    G_gt.set_directed(False)
    weight = G_gt.edge_properties['weight']
    args = layout_args(G, layout_algorithm, weight=weight, **kwargs)
    layout_gt = (get_layout_function(layout_algorithm))(G_gt, **args)
    layout = { G_gt.vertex_properties['id'][i]: layout_gt[i] for i in G_gt.vertices() }
    return layout, G_gt, layout_gt

def layout_network(G, layout_algorithm, **kwargs):

    if layout_algorithm.startswith('graphtool'):
        layout, _, _ = layout_graphtool_network(G, layout_algorithm, **kwargs)
        return layout
    
    args = layout_args(G, layout_algorithm, weight='weight', **kwargs)
    layout = (get_layout_function(layout_algorithm))(G, **args)

    if layout_algorithm.startswith('graphviz'):
        layout = nx_normalize_layout(layout)
        
    return layout
                    
def project_series_to_range(series, low, high):
    norm_series = series / series.max()
    return norm_series.apply(lambda x: low + (high - low) * x)

def project_to_range(value, low, high):
    return low + (high - low) * value

def clamp_values(values, low_high):
    mw = max(values)
    return [ project_to_range(w / mw, low_high[0], low_high[1]) for w in values ]
    
def get_layout_algorithm(layout_algorithm):
    if layout_algorithm not in layout_algorithms:
        raise Exception("Unknown algorithm {}".format(layout_algorithm))
    return layout_algorithms.get(layout_algorithm, None)
    
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

def nx_normalize_layout(layout):
    max_xy = max([ max(x,y) for x,y in layout.values()])
    layout = { n: (layout[n][0]/max_xy, layout[n][1]/max_xy) for n in layout.keys() }
    return layout

class BokehPlotNetworkUtility:
    
    @staticmethod
    def plot(
        network,
        layout,
        scale=1.0,
        threshold=0.0,
        node_description=None,
        node_size=5,
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
            network = filter_by_weight(network, threshold)

        edges = NetworkUtility.get_edges(network, layout, scale=weight_scale, normalize=normalize_weights)
        nodes = NetworkUtility.get_node_attributes(network, layout)

        if node_size in nodes.keys() and node_size_range is not None:
            nodes['clamped_size'] = clamp_values(nodes[node_size], node_size_range)
            node_size = 'clamped_size'
            
        label_y_offset = 'y_offset' if node_size in nodes.keys() else node_size + 8
        if label_y_offset == 'y_offset':
            nodes['y_offset'] = [ y + r for (y, r) in zip(nodes['y'], [ r / 2.0 + 8 for r in nodes[node_size] ]) ]
            
        edges = { k: list(edges[k]) for k in edges}
        nodes = { k: list(nodes[k]) for k in nodes}

        edges_source = bm.ColumnDataSource(edges)
        nodes_source = bm.ColumnDataSource(nodes)

        node_opts = extend(DFLT_NODE_OPTS, node_opts or {})
        line_opts = extend(DFLT_EDGE_OPTS, line_opts or {})

        p = figure(plot_width=figsize[0], plot_height=figsize[1], tools=tools or TOOLS, **figkwargs)

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        r_lines = p.multi_line('xs', 'ys', line_width='weights', source=edges_source, **line_opts)
        r_nodes = p.circle('x', 'y', size=node_size, source=nodes_source, **node_opts)
        
        if 'fill_color' in nodes.keys():
            r_nodes.glyph.fill_color = 'fill_color'

        if node_description is not None:
            p.add_tools(bm.HoverTool(renderers=[r_nodes], tooltips=None, callback=WidgetUtility.\
                glyph_hover_callback(nodes_source, 'node_id', text_ids=node_description.index, \
                                     text=node_description, element_id=element_id))
            )

        label_opts = extend(DFLT_TEXT_OPTS, dict(y_offset=label_y_offset, text_color='black', text_baseline='bottom'))
        label_opts = extend(label_opts, text_opts or {})
        
        p.add_layout(bm.LabelSet(source=nodes_source, **label_opts))

        return p
    
class NetworkxPlotUtility():
    
    @staticmethod   
    def plot(G):
        import graphviz, pydotplus
        # Not used:
        def apply_styles(graph, styles):
            graph.graph_attr.update(
                ('graph' in styles and styles['graph']) or {}
            )
            graph.node_attr.update(
                ('nodes' in styles and styles['nodes']) or {}
            )
            graph.edge_attr.update(
                ('edges' in styles and styles['edges']) or {}
            )
            return graph
        styles = {
            'graph': {
                'label': 'Graph',
                'fontsize': '16',
                'fontcolor': 'white',
                'bgcolor': '#333333',
                'rankdir': 'BT',
            },
            'nodes': {
                'fontname': 'Helvetica',
                'shape': 'hexagon',
                'fontcolor': 'white',
                'color': 'white',
                'style': 'filled',
                'fillcolor': '#006699',
            },
            'edges': {
                'style': 'dashed',
                'color': 'white',
                'arrowhead': 'open',
                'fontname': 'Courier',
                'fontsize': '12',
                'fontcolor': 'white',
            }
        }
        P=nx.nx_pydot.to_pydot(G)
        P.format = 'svg'
        #if root is not None :
        #    P.set("root",make_str(root))
        D=P.create_dot(prog='circo')
        if D=="":
            return
        Q=pydotplus.graph_from_dot_data(D)
        #Q = apply_styles(Q, styles)
        from IPython.display import Image
        I = Image(Q.create_png())
        return I
    
    
import graph_tool.draw as gt_draw
import graph_tool.all as gt
import numpy as np

class GraphToolPlotUtility():
    
    @staticmethod   
    def plot(G_gt, layout_gt, n_range, palette):
        
        v_text = G_gt.vertex_properties['id']
        #v_degrees_p = G_gt.degree_property_map('out')
        #v_degrees_p.a = np.sqrt(v_degrees_p.a)+2
        v_degrees_p = G_gt.vertex_properties['degree']
        v_size_p = gt.prop_to_size(v_degrees_p, n_range[0], n_range[1])
        v_fill_color  = G_gt.vertex_properties['fill_color']
        e_weights = G_gt.edge_properties['weight']
        e_size_p = gt.prop_to_size(e_weights, 1.0, 4.0)
        #state = gt.minimize_blockmodel_dl(G_gt)
        #state.draw(
        #c = gt.all.closeness(G_gt)

        v_blocks = gt.minimize_blockmodel_dl(G_gt).get_blocks()
        plot_color = G_gt.new_vertex_property('vector<double>')
        G_gt.vertex_properties['plot_color'] = plot_color

        for v_i, v in enumerate(G_gt.vertices()):
            scolor = palette[v_blocks[v_i]]
            plot_color[v] = tuple(int(scolor[i:i+2], 16) for i in (1, 3, 5)) + (1,)

        gt_draw.graph_draw(
            G_gt,
            #vorder=c,
            pos=layout_gt,
            output_size=(1000, 1000),
            #vertex_text_offset=[-1,1],
            vertex_text_position=0.0,
            vertex_text=v_text,
            vertex_color=[1,1,1,0],
            vertex_fill_color=v_fill_color,
            vertex_size=v_size_p,
            vertex_font_family='helvetica',
            vertex_text_color='black',
            edge_pen_width=e_size_p,
            inline=True
        )
        
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
        
