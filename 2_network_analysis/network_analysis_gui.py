import os
import warnings
import ipywidgets as widgets

os.sys.path = os.sys.path if '..' in os.sys.path else os.sys.path + ['..']

from common import widgets_config

from bokeh.palettes import all_palettes  # pylint: disable=E0611
from bokeh.io import push_notebook
from common.widgets_utility import WidgetUtility
from common.network.layout import layout_setups
from IPython.display import display
from network_analysis import slice_network_datasorce

NETWORK_LAYOUT_OPTIONS = {x.name: x.key for x in layout_setups}

warnings.filterwarnings('ignore')

LABEL_TEXT_OPTS = dict(
    x_offset=0,  # y_offset=5,
    level='overlay',
    text_align='center',
    text_baseline='bottom',
    render_mode='canvas',
    text_font="Tahoma",
    text_font_size="9pt",
    text_color='black'
)

NETWORK_PLOT_OPTS = dict(
    x_axis_type=None,
    y_axis_type=None,
    background_fill_color='white',
    line_opts=dict(color='green', alpha=0.5),
    node_opts=dict(color=None, level='overlay', alpha=1.0),
)

NODE_SIZE_OPTIONS = {
    '(default)': None,
    'Degree centrality': 'degree',
    'Closeness centrality': 'closeness',
    'Betweenness centrality': 'betweenness',
    'Eigenvector centrality': 'eigenvector'
}

PALETTE_OPTIONS = {
    palette_name: palette_name
    for palette_name in all_palettes.keys()
    if any([len(x) > 7 for x in all_palettes[palette_name].values()])
}

OUTPUT_OPTIONS = {
    'Network': 'network',
    'Table': 'table'
}

COMMUNITY_OPTIONS = {
    '(default)': None,
    'Louvain': 'louvain'

}
SLICE_TYPE_OPTIONS, SLICE_TYPE_DEFAULT = {
    'Sequence': 1,
    'Year': 2
}, 1

def display_partial_party_network(plot_data, slice_range, slice_range_type, callback):

    slice_nodes, slice_edges = slice_network_datasorce(plot_data.edges, plot_data.nodes, slice_range, slice_range_type)

    plot_data.edges_source.data.update(slice_edges)
    plot_data.nodes_source.data.update(slice_nodes)
    plot_data.slice_range = slice_range
    plot_data.slice_range_type = slice_range_type
    
    min_year, max_year = min(plot_data.edges_source.data['signed']).year, max(plot_data.edges_source.data['signed']).year

    callback(min_year, max_year)

    push_notebook(handle=plot_data.handle)

def display_network_analyis_gui(wti_index, display_function, plot_data):

    box = widgets.Layout()

    def lw(w='170px'):
        return widgets.Layout(width=w)

    zn = WidgetUtility(

        period_group=widgets_config.period_group_widget(layout=lw()),
        topic_group=widgets_config.topic_groups_widget2(layout=lw()),
        party_name=widgets_config.party_name_widget(layout=lw()),
        treaty_filter=widgets_config.treaty_filter_widget(layout=lw()),

        parties=widgets_config.parties_widget(options=wti_index.get_countries_list(), value=['FRANCE'], layout=lw()),

        palette=widgets_config.dropdown('Color', PALETTE_OPTIONS, None, layout=lw()),
        node_size=widgets_config.dropdown('Node size', NODE_SIZE_OPTIONS, None, layout=lw()),
        node_size_range=widgets_config.rangeslider('Range', 5, 100, [20, 49], step=1, layout=box),
        
        C=widgets_config.slider('C', 0, 100, 1, step=1, layout=box),
        K=widgets_config.sliderf('K', 0.01, 1.0, 0.01, 0.10, layout=box),
        p=widgets_config.sliderf('p', 0.01, 2.0, 0.01, 1.10, layout=box),
        fig_width=widgets_config.slider('Width', 600, 1600, 900, step=100, layout=box),
        fig_height=widgets_config.slider('Height', 600, 1600, 700, step=100, layout=box),
        output=widgets_config.dropdown('Output', OUTPUT_OPTIONS, 'network', layout=lw()),
        layout_algorithm=widgets_config.dropdown('Layout', NETWORK_LAYOUT_OPTIONS, 'nx_spring_layout', layout=lw()),
        progress=widgets_config.progress(0, 4, 1, 0, layout=lw("300px")),
        refresh=widgets_config.toggle('Refresh', False, tooltip='Update plot'),
        node_partition=widgets_config.dropdown('Partition', COMMUNITY_OPTIONS, None, layout=lw()),
        simple_mode=widgets_config.toggle('Simple', False, tooltip='Simple view'),
        
        slice_range_type=widgets_config.dropdown('Slice Type', SLICE_TYPE_OPTIONS, SLICE_TYPE_DEFAULT, layout=lw()),
        time_travel_range=widgets_config.rangeslider('Time travel', 0, 100, [0, 100], layout=lw('60%')),
        time_travel_label=widgets.Label(value="")
    )

    def on_simple_mode_value_change(change):
        display_mode = 'none' if change['new'] is True else ''
        zn.node_partition.layout.display = display_mode
        zn.node_size.layout.display = display_mode
        zn.node_size_range.layout.display = display_mode
        zn.layout_algorithm.layout.display = display_mode
        zn.C.layout.display = display_mode
        zn.K.layout.display = display_mode
        zn.p.layout.display = display_mode
        zn.fig_width.layout.display = display_mode
        zn.fig_height.layout.display = display_mode
        zn.palette.layout.display = display_mode

    zn.simple_mode.observe(on_simple_mode_value_change, names='value')
    zn.simple_mode.value = True

    wn = widgets.interactive(
        display_function,
        parties=zn.parties,
        period_group=zn.period_group,
        treaty_filter=zn.treaty_filter,
        topic_group=zn.topic_group,
        layout_algorithm=zn.layout_algorithm,
        C=zn.C,
        K=zn.K,
        p1=zn.p,
        output=zn.output,
        party_name=zn.party_name,
        node_size_range=zn.node_size_range,
        refresh=zn.refresh,
        palette_name=zn.palette,
        width=zn.fig_width,
        height=zn.fig_height,
        node_size=zn.node_size,
        node_partition=zn.node_partition,
        wti_index=widgets.fixed(wti_index),
        gui=widgets.fixed(zn)
        # plot_data=zn.node_size #widgets.fixed(plot_data)
    )

    boxes = widgets.HBox([
        widgets.VBox([
            zn.period_group,
            zn.topic_group,
            zn.party_name,
            zn.treaty_filter
        ]),
        widgets.VBox([zn.parties]),
        widgets.VBox([
            widgets.HBox([zn.progress, zn.refresh, zn.simple_mode]),
            widgets.HBox([
                widgets.VBox([
                    zn.output,
                    zn.layout_algorithm, zn.palette, zn.node_size,
                    zn.node_partition
                ]),
                widgets.VBox([
                    zn.K, zn.C, zn.p, zn.fig_width, zn.fig_height, zn.node_size_range
                ])
            ])
        ])
    ])

    display(widgets.VBox([boxes, wn.children[-1]]))

    wn.update()

    def slice_callback(min_year, max_year):
        zn.time_travel_range.description = '{}-{}'.format(min_year, max_year)
    
    def on_slice_range_type_change(change):
        slice_range_type = change['new']
        if slice_range_type == 1:  # sequence
            zn.time_travel_range.min = plot_data.eg
            zn.time_travel_range.max = 10
                    
    zn.slice_range_type.observe(on_slice_range_type_change, names='value')

    iw_time_travel = widgets.interactive(
        display_partial_party_network,
        slice_range=zn.time_travel_range,
        slice_range_type=zn.slice_range_type,
        plot_data=widgets.fixed(plot_data),
        callback=widgets.fixed(slice_callback)
    )
                    
    time_travel_box = widgets.VBox([
        widgets.VBox([zn.time_travel_label, zn.time_travel_range, zn.slice_range_type]),
        iw_time_travel.children[-1]])

    display(time_travel_box)
