
import os
import warnings
import ipywidgets as widgets
import pandas as pd

os.sys.path = os.sys.path if '..' in os.sys.path else os.sys.path + ['..']

import common.widgets_config as wc

from common.widgets_utility import BaseWidgetUtility
from topic_groups_config import category_group_maps

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

def display_network_analyis_gui(state, callback):

    zn = BaseWidgetUtility(
        period_group=wc.period_group_widget(),
        category_map_name=wc.topic_groups_widgets(),
        parties=wc.parties_widget(options=state.get_countries_list(),value=['FRANCE']),
        party_name=wc.party_name_widget(),
        node_size=widgets.Dropdown(
            description='Node size:',
            options={
                '(default)': None,
                'Degree centrality': 'degree',
                'Closeness centrality': 'closeness',
                'Betweenness centrality': 'betweenness',
                'Eigenvector centrality': 'eigenvector'            
                #'communicability_betweenness': 'communicability_betweenness'
            },
            value=None,
            layout=widgets.Layout(width='220px')
        ),
        palette=widgets.Dropdown(
            description='Color:',
            options={
                palette_name: palette_name
                        for palette_name in bokeh.palettes.all_palettes.keys()
                            if any([ len(x) > 7 for x in bokeh.palettes.all_palettes[palette_name].values()])
            },
            layout=widgets.Layout(width='220px')
        ),
        C=widgets.IntSlider(
            description='C', min=0, max=100, step=1, value=1,
            continuous_update=False, layout=widgets.Layout(width='240px', height='30px')  # , orientation='vertical'
        ),
        K=widgets.FloatSlider(
            description='K', min=0.01, max=1.0, step=0.01, value=0.10,
            continuous_update=False, layout=widgets.Layout(width='240px', height='30px')  # , orientation='vertical'
        ),
        p=widgets.FloatSlider(
            description='p', min=0.01, max=2.0, step=0.01, value=1.10,
            continuous_update=False, layout=widgets.Layout(width='240px', height='30px')  # , orientation='vertical', 
        ),
        node_size_range=widgets.IntRangeSlider(
            description='Node size range',
            value=[20, 40], min=5, max=100, step=1,
            continuous_update=False,
            layout=widgets.Layout(width='240px', height='30px'),  # , orientation='vertical'
            style={'font-size': '9pt' }
        ),
        fig_width=widgets.IntSlider(
            description='Width', min=600, max=1600, step=100, value=1000,
            continuous_update=False, layout=widgets.Layout(width='240px', height='30px')  # , orientation='vertical'
        ),
        fig_height=widgets.IntSlider(
            description='Height', min=600, max=1600, step=100, value=700,
            continuous_update=False, layout=widgets.Layout(width='240px', height='30px')  # , orientation='vertical'
        ),
        only_is_cultural=widgets.ToggleButton(
            description='Only Cultural', value=True,
            tooltip='Display only "is_cultural" treaties', layout=widgets.Layout(width='100px')
        ),
        output=widgets.Dropdown(
            description='Output:',
            options={ 'Plot': 'bokeh_plot', 'List': 'table' }, # ,'Framework': 'framework_plot' },
            value='bokeh_plot',
            layout=widgets.Layout(width='220px')
        ),
        layout_algorithm=widgets.Dropdown(
            description='Layout',
            options=layout_function_name,
            value='graphtool_sfdp',
            layout=widgets.Layout(width='220px')
        ),
        progress=wf.create_int_progress_widget(min=0, max=4, step=1, value=0, layout=widgets.Layout(width="50%")),
        refresh=widgets.ToggleButton(
            description='Refresh', value=False,
            tooltip='Update plot', layout=widgets.Layout(width='100px')
        ),
        node_partition=widgets.Dropdown(
            description='Partition:',
            options={
                '(default)': None,
                'Louvain': 'louvain',
            },
            value=None,
            layout=widgets.Layout(width='220px')
        ),
        simple_mode=widgets.Checkbox(
            value=False,
            description='Simple',
            disabled=False,
            layout=widgets.Layout(width='150px')
        ),
        time_travel_range=widgets.IntRangeSlider(
            value=[0, 100],
            min=0,
            max=100,
            step=1,
            description='Time travel',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=widgets.Layout(width='80%')
        ),
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
        display_party_network,
        parties=zn.parties,
        period=zn.period,
        only_is_cultural=zn.only_is_cultural,
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
        category_map_name=zn.category_map_name
    )

    boxes = widgets.HBox([
        widgets.VBox([widgets.HBox([zn.parties, zn.only_is_cultural]), widgets.HBox([zn.period, zn.refresh]),
                      widgets.HBox([zn.category_map_name]),
                      widgets.HBox([zn.simple_mode, zn.progress])
                     ]),
        widgets.VBox([zn.layout_algorithm, zn.party_name, zn.output, zn.palette]),
        widgets.VBox([zn.K, zn.C, zn.p, zn.node_partition]),
        widgets.VBox([zn.fig_width, zn.fig_height, zn.node_size, zn.node_size_range]),
    ])

    display(widgets.VBox([boxes, wn.children[-1]]))

    wn.update()

    def display_partial_party_network(range=[1,100]):
        global handle_data    
        edge_count = len(handle_data.edges['source'])
        edges = { k: handle_data.edges[k][range[0]:range[1]] for k in handle_data.edges } 
        nodes_indices = set(edges['source'] + edges['target'])
        df = pd.DataFrame({k:handle_data.nodes[k] for k in handle_data.nodes if isinstance(handle_data.nodes[k], list) }).set_index('id')
        nodes = df.loc[nodes_indices].reset_index().to_dict(orient='list')
        nodes = extend(nodes, {k:handle_data.nodes[k] for k in handle_data.nodes if not isinstance(handle_data.nodes[k], list) })
        handle_data.edges_source.data.update(edges)    
        handle_data.nodes_source.data.update(nodes)
        min_year, max_year = min(handle_data.edges_source.data['signed']).year, max(handle_data.edges_source.data['signed']).year
        zn.time_travel_range.description = '{}-{}'.format(min_year, max_year)
        bokeh.io.push_notebook(handle=handle_data.handle)

    iw_time_travel = widgets.interactive(
        display_partial_party_network,
        range=zn.time_travel_range
    )
    time_travel_box = widgets.VBox([widgets.VBox([zn.time_travel_label, zn.time_travel_range]), iw_time_travel.children[-1]])

    display(time_travel_box)
