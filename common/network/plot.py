import os

os.sys.path = os.sys.path if '.' in os.sys.path else os.sys.path + ['.']

import common.network.networkx_utility as network_utility
from common.utility import extend
from common.widgets_utility import WidgetUtility
import bokeh.palettes
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.plotting import figure
        
TOOLS = "pan,wheel_zoom,box_zoom,reset,previewsave"

DFLT_NODE_OPTS = dict(color='green', level='overlay', alpha=1.0)
DFLT_EDGE_OPTS = dict(color='black', alpha=0.2)
DFLT_TEXT_OPTS = dict(x='x', y='y', text='name', level='overlay', text_align='center', text_baseline='middle')

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

    edges = network_utility.get_edges(network, layout, scale=weight_scale, normalize=normalize_weights)
    nodes = network_utility.get_node_attributes(network, layout)

    if node_size in nodes.keys() and node_size_range is not None:
        nodes['clamped_size'] = clamp_values(nodes[node_size], node_size_range)
        node_size = 'clamped_size'
        
    label_y_offset = 'y_offset' if node_size in nodes.keys() else node_size + 8
    if label_y_offset == 'y_offset':
        nodes['y_offset'] = [ y + r for (y, r) in zip(nodes['y'], [ r / 2.0 + 8 for r in nodes[node_size] ]) ]
        
    edges = { k: list(edges[k]) for k in edges}
    nodes = { k: list(nodes[k]) for k in nodes}

    edges_source = ColumnDataSource(edges)
    nodes_source = ColumnDataSource(nodes)

    node_opts = extend(DFLT_NODE_OPTS, node_opts or {})
    line_opts = extend(DFLT_EDGE_OPTS, line_opts or {})

    p = figure(plot_width=figsize[0], plot_height=figsize[1], tools=tools or TOOLS, **figkwargs)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    r_lines = p.multi_line('xs', 'ys', line_width='weights', source=edges_source, **line_opts)
    r_nodes = p.circle('x', 'y', size=node_size, source=nodes_source, **node_opts)
    
    if 'fill_color' in nodes.keys():
        r_nodes.glyph.fill_color = 'fill_color'

    if node_description is not None and hover_callback is not None:
        p.add_tools(HoverTool(renderers=[r_nodes], tooltips=None, callback=WidgetUtility.\
            glyph_hover_callback(nodes_source, 'node_id', text_ids=node_description.index, \
                                 text=node_description, element_id=element_id))
        )

    label_opts = extend(DFLT_TEXT_OPTS, dict(y_offset=label_y_offset, text_color='black', text_baseline='bottom'))
    label_opts = extend(label_opts, text_opts or {})
    
    p.add_layout(LabelSet(source=nodes_source, **label_opts))

    return p
