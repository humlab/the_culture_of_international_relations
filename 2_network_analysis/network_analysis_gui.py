import os
import warnings
import ipywidgets as widgets
import pandas as pd
import bokeh.palettes

os.sys.path = os.sys.path if '..' in os.sys.path else os.sys.path + ['..']

from common import widgets_config

from common.widgets_utility import WidgetUtility
from common.network.layout import layout_setups
from common.utility import extend

network_layout_options = { x.name : x.key for x in layout_setups }

warnings.filterwarnings('ignore')

label_text_opts=dict(
    x_offset=0, #y_offset=5,
    level='overlay',
    text_align='center',
    text_baseline='bottom',
    render_mode='canvas',
    text_font="Tahoma",
    text_font_size="9pt",
    text_color='black'
)

network_plot_opts = dict(
    x_axis_type=None,
    y_axis_type=None,
    background_fill_color='white',
    line_opts=dict(color='green', alpha=0.5 ),
    node_opts=dict(color=None, level='overlay', alpha=1.0),
)

node_size_options = {
    '(default)': None,
    'Degree centrality': 'degree',
    'Closeness centrality': 'closeness',
    'Betweenness centrality': 'betweenness',
    'Eigenvector centrality': 'eigenvector'            
}

palette_options = {
    palette_name: palette_name
        for palette_name in bokeh.palettes.all_palettes.keys()
            if any([ len(x) > 7 for x in bokeh.palettes.all_palettes[palette_name].values()])
}

output_options = {
    'Network': 'network',
    'Table': 'table'
}

community_options = {
    '(default)': None,
    'Louvain': 'louvain'
}

def display_partial_party_network(update_partial_function, range):

    global plot_data
    nodes, edges = update_partial_function(range, plot_data)

    plot_data.edges_source.data.update(edges)    
    plot_data.nodes_source.data.update(nodes)
        
    min_year, max_year = min(plot_data.edges_source.data['signed']).year, max(plot_data.edges_source.data['signed']).year
        
    zn.time_travel_range.description = '{}-{}'.format(min_year, max_year)
        
    bokeh.io.push_notebook(handle=plot_data.handle)

def display_network_analyis_gui(wti_index, display_function, update_partial_function):
    global plot_data

    box = widgets.Layout(width='170px', height='18px')
    
    lw = lambda w: widgets.Layout(width=w)
    lwh = lambda w,h: widgets.Layout(width=w,height=h)
    
    zn = WidgetUtility(
        
        period_group=widgets_config.period_group_widget(layout=lw('170px')),
        topic_group=widgets_config.topic_groups_widget2(layout=lw('170px')),
        party_name=widgets_config.party_name_widget(layout=lw('170px')),
        treaty_filter=widgets_config.treaty_filter_widget(layout=lw('170px')),

        parties=widgets_config.parties_widget(options=wti_index.get_countries_list(),value=['FRANCE'], layout=lw('170px')),
        
        palette=widgets_config.dropdown('Color', palette_options, None, layout=lw('170px')),
        node_size=widgets_config.dropdown('Node size', node_size_options, None, layout=lw('170px')),
        node_size_range=widgets_config.rangeslider('Range', 5, 100, [20, 49], step=1, layout=box),
        
        C=widgets_config.slider('C', 0, 100, 1, step=1, layout=box),
        K=widgets_config.sliderf('K', 0.01, 1.0, 0.01, 0.10, layout=box),
        p=widgets_config.sliderf('p', 0.01, 2.0, 0.01, 1.10, layout=box),
        fig_width=widgets_config.slider('Width', 600, 1600, 1000, step=100, layout=box),
        fig_height=widgets_config.slider('Height', 600, 1600, 700, step=100, layout=box),
        output=widgets_config.dropdown('Output', output_options, 'network', layout=lw('170px')),
        layout_algorithm=widgets_config.dropdown('Layout', network_layout_options, 'nx_spring_layout', layout=lw('170px')),
        progress=widgets_config.progress(0, 4, 1, 0, layout=lw("300px")),
        refresh=widgets_config.toggle('Refresh', False, tooltip='Update plot'),
        node_partition=widgets_config.dropdown('Partition', community_options, None, layout=lw('170px') ),
        simple_mode=widgets_config.toggle('Simple', False, tooltip='Simple view'),
        time_travel_range=widgets_config.rangeslider('Time travel', 0, 100, [0, 100], layout=lw('80%')),
        time_travel_label=widgets.Label(value="")
    ) 

    def on_value_change(change):
        display_mode = 'none' if change['new'] == True else ''
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

    zn.simple_mode.observe(on_value_change, names='value')
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
        #plot_data=zn.node_size #widgets.fixed(plot_data)
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
                    zn.K, zn.C, zn.p, zn.fig_width, zn.fig_height, zn.node_size_range,
                ])
            ])
        ])
    ])

    display(widgets.VBox([boxes, wn.children[-1]]))

    wn.update()

    iw_time_travel = widgets.interactive(
        display_partial_party_network,
        update_partial_function=widgets.fixed(update_partial_function),
        range=zn.time_travel_range  #,
        #plot_data=widgets.fixed(plot_data)
    )
    
    time_travel_box = widgets.VBox([
        widgets.VBox([zn.time_travel_label, zn.time_travel_range]),
        iw_time_travel.children[-1]])

    display(time_travel_box)
