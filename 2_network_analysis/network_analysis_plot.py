import types
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import show
import bokeh.palettes

def get_palette(palette_name):
    if palette_name not in bokeh.palettes.all_palettes.keys():
        return bokeh.palettes.RdYlBu[11]
    key = max(bokeh.palettes.all_palettes[palette_name].keys())
    return bokeh.palettes.all_palettes[palette_name][key]

def get_line_color():
    line_palette = bokeh.palettes.Set1[8]
    group_keys = category_group_settings[category_map_name].keys()
    line_palette_map = { k: i % len(line_palette) for i, k in enumerate(group_keys) }
    return data.category.apply(lambda x: line_palette[line_palette_map[x]])

TOOLS = "pan,wheel_zoom,box_zoom,reset,previewsave"

DFLT_NODE_OPTS = dict(color='green', level='overlay', alpha=1.0)
DFLT_EDGE_OPTS = dict(color='black', alpha=0.2)
DFLT_TEXT_OPTS = dict(x='x', y='y', text='name', level='overlay', text_align='center', text_baseline='middle')

def plot_network(
    nodes,
    edges,
    node_description=None,
    node_size=5,
    node_opts=None,
    line_opts=None,
    text_opts=None,
    element_id='nx_id3',
    figsize=(900, 900),
    tools=None,
    **figkwargs
):

    edges_source = ColumnDataSource(edges)
    nodes_source = ColumnDataSource(nodes)

    node_opts = extend(DFLT_NODE_OPTS, node_opts or {})
    line_opts = extend(DFLT_EDGE_OPTS, line_opts or {})

    p = figure(plot_width=figsize[0], plot_height=figsize[1], tools=tools or TOOLS, **figkwargs)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    
    if 'line_color' in edges.keys():
        line_opts = extend(line_opts, { 'line_color': 'line_color', 'alpha': 1.0})

    r_lines = p.multi_line('xs', 'ys', line_width='weight', source=edges_source, **line_opts)
    r_nodes = p.circle('x', 'y', size=node_size, source=nodes_source, **node_opts)

    if 'fill_color' in nodes.keys():
        r_nodes.glyph.fill_color = 'fill_color'

    if node_description is not None:
        p.add_tools(HoverTool(renderers=[r_nodes], tooltips=None, callback=WidgetUtility.\
            glyph_hover_callback(nodes_source, 'node_id', text_ids=node_description.index, \
                                 text=node_description, element_id=element_id))
        )

    label_opts = extend(DFLT_TEXT_OPTS, text_opts or {})

    p.add_layout(bm.LabelSet(source=nodes_source, **label_opts))

    handle = bp.show(p, notebook_handle=True)
    return types.SimpleNamespace(
        handle=handle,
        edges_source=edges_source,
        nodes_source=nodes_source,
        nodes=nodes,
        edges=edges,
    )
