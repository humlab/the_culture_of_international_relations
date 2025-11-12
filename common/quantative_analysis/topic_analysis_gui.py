 
import ipywidgets as widgets
import pandas as pd
import common.config as config
import common.utility as utility
import common.treaty_utility as treaty_utility
import common.widgets_config as widgets_config
import common.color_utility as color_utility
import analysis_data
import analysis_plot
import logging
import types
import datetime
from common.treaty_state import TreatyState

from IPython.display import display
from pprint import pprint as pp

logger = utility.getLogger('tq_by_topic', level=logging.WARNING)

def display_topic_quantity(
    *,
    wti_index: TreatyState,
    chart_type_name: str,
    period_group=0,
    topic_group=None,
    party_group=None,
    recode_is_cultural=False,
    normalize_values=False,
    extra_other_category=False,
    plot_style='classic',
    target_quantity="topic",
    treaty_sources=None,
    progress=utility.noop,
    vmax=None,
    legend=True,
    output_filename=None
):
    try:
        progress()
        
        # Note: Using `output_filename`-argument instead
        # save_plot = False
        # if chart_type_name.endswith('_save'):
        #     save_plot = True
        #     chart_type_name = chart_type_name[:-5]
            
        chart_type = config.CHART_TYPE_MAP[chart_type_name]

        data: pd.DataFrame | None= analysis_data.QuantityByTopic.get_treaty_topic_quantity_stat(
            wti_index,
            period_group,
            topic_group,
            party_group,
            recode_is_cultural,
            extra_other_category,
            target_quantity=target_quantity,
            treaty_sources=treaty_sources
        )

        if data is None or data.shape[0] == 0:
            return

        progress()

        pivot = pd.pivot_table(data, index=['Period'], values=["Count"], columns=['Category'], fill_value=0)
        pivot.columns = [ x[-1] for x in pivot.columns ]

        if normalize_values is True:
            pivot = pivot.div(0.01 * pivot.sum(1), axis=0)

        if chart_type.name.startswith('plot'):

            columns = pivot.columns
            pivot = pivot.reset_index()[columns]

            kwargs = analysis_plot.prepare_plot_kwargs(pivot, chart_type, normalize_values, period_group, vmax=vmax)
            progress()
            #kwargs.update(dict(overlay=False, colors=colors))
            kwargs.update(dict(title=party_group['label']))
            kwargs.update(dict(legend=legend))
            
            ax = analysis_plot.quantity_plot(data, pivot, chart_type, plot_style, **kwargs)
            
            if output_filename:
                basename = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                ax.savefig('{}_{}.{}'.format(basename, output_filename, 'eps'), format='eps', dpi=300)

        elif chart_type.name == 'table':
            display(data)
        else:
            display(pivot)

        progress()

    except Exception as ex:
        logger.error(ex)
        #raise
    finally:
        progress(0)

def party_group_label(parties):
    return ', '.join(parties[:6]) + ('' if len(parties) <= 6 else ' +{} parties'.format(len(parties)-6))

def display_topic_quantity_groups(
    period_group_index,
    topic_group_name,
    recode_is_cultural=False,
    normalize_values=False,
    extra_other_category=None,
    chart_type_name=None,
    plot_style='classic',
    parties=None,
    chart_per_category=False,
    target_quantity="topic",
    treaty_sources=None,
    wti_index=None,
    progress=utility.noop,
    print_args=False,
    vmax=None,
    legend=True,
    output_filename=None
):

    if print_args or (chart_type_name == 'print_args'):
        args = utility.filter_dict(locals(), [ 'progress', 'print_args' ], filter_out=True)
        args['wti_index'] = None
        pp(args)

    period_group = config.DEFAULT_PERIOD_GROUPS[period_group_index]
    topic_group = config.TOPIC_GROUP_MAPS[topic_group_name]
    topic_groups = [ topic_group ]
    wti_parties = [ x for x in parties if x not in ['ALL OTHER' ] ]

    if target_quantity in ['party', 'region']:
        party_groups = [ { 'label': topic_group_name, 'parties': wti_parties } ]
        if chart_per_category:
            topic_groups = [ { k: { k: topic_group[k] }} for k in topic_group.keys() ]
    else:
        if not chart_per_category:
            if 'ALL' in parties:
                party_groups = [ { 'label': 'ALL', 'parties': None } ]
            else:
                party_groups = [ { 'label': party_group_label(parties), 'parties': wti_parties } ]
        else:
            party_groups = [ { 'label': x, 'parties': [ x ] if x != 'ALL OTHER' else parties } for x in parties ]

    for p_g in party_groups:
        for t_c in topic_groups:
            display_topic_quantity(
                period_group=period_group,
                party_group=p_g,
                topic_group=t_c,
                recode_is_cultural=recode_is_cultural,
                normalize_values=normalize_values,
                extra_other_category=extra_other_category,
                chart_type_name=chart_type_name,
                plot_style=plot_style,
                target_quantity=target_quantity,
                treaty_sources=treaty_sources,
                wti_index=wti_index,
                progress=progress,
                vmax=vmax,
                legend=legend,
                output_filename=output_filename
            )

def display_gui(wti_index, print_args=False):

    def lw(width='120px'):
        return widgets.Layout(width=width)

    party_preset_options = wti_index.get_party_preset_options()

    treaty_source_options = wti_index.unique_sources

    gui = types.SimpleNamespace(
        treaty_sources = widgets_config.select_multiple(
            description='Sources', options=treaty_source_options,
            values=treaty_source_options, disabled=False, layout=lw('180px')),
        period_group_index = widgets_config.period_group_widget(index_as_value=True),
        topic_group_name = widgets_config.topic_groups_widget(value='7CULTURE'),
        # treaty_filter = widgets_config.treaty_filter_widget()
        recode_is_cultural = widgets_config.recode_7corr_widget(layout=lw('120px')),
        normalize_values = widgets_config.toggle('Display %', False, icon='', layout=lw('100px')),
        chart_type_name = widgets_config.dropdown('Output', config.CHART_TYPE_NAME_OPTIONS, "plot_stacked_bar", layout=lw('200px')),
        plot_style = widgets_config.plot_style_widget(),
        parties = widgets_config.parties_widget(options=[ x for x in wti_index.get_countries_list() if x != 'ALL OTHER' ], value=['FRANCE']),
        party_preset = widgets_config.dropdown('Presets', party_preset_options, None, layout=lw(width='200px')),
        chart_per_category = widgets_config.toggle('Chart per Qty', False,
            tooltip='Display one chart per selected quantity category', layout=lw()),
        extra_other_category = widgets_config.toggle('Add OTHER topics', False, layout=lw()),
        target_quantity = widgets_config.dropdown('Quantity', ['topic', 'party', 'source', 'continent', 'group'], 'topic', layout=lw(width='200px')),
        progress = widgets_config.progress(0, 5, 1, 0, layout=lw())
    )


    def stepper(step=None):
        gui.progress.value = gui.progress.value + 1 if step is None else step

    def on_party_preset_change(change):  # pylint: disable=W0613

        if gui.party_preset.value is None:
            return

        if 'ALL' in gui.party_preset.value:
            gui.parties.value = gui.parties.options
        else:
            gui.parties.value = gui.party_preset.value

        #if top_n_parties.value > 0:
        #    top_n_parties.value = 0

    gui.party_preset.observe(on_party_preset_change, names='value')

    itw = widgets.interactive(
        display_topic_quantity_groups,
        period_group_index=gui.period_group_index,
        topic_group_name=gui.topic_group_name,
        parties=gui.parties,
        recode_is_cultural=gui.recode_is_cultural,
        normalize_values=gui.normalize_values,
        extra_other_category=gui.extra_other_category,
        chart_type_name=gui.chart_type_name,
        plot_style=gui.plot_style,
        chart_per_category=gui.chart_per_category,
        target_quantity=gui.target_quantity,
        progress=widgets.fixed(stepper),
        wti_index=widgets.fixed(wti_index),
        treaty_sources=gui.treaty_sources,
        print_args=widgets.fixed(print_args)
    )

    boxes = widgets.HBox(
        [
            widgets.VBox([ gui.period_group_index, gui.topic_group_name, gui.target_quantity, gui.party_preset]),
            widgets.VBox([ gui.treaty_sources, gui.parties ]),
            widgets.VBox([ gui.recode_is_cultural, gui.extra_other_category, gui.chart_per_category]),
            widgets.VBox([ gui.chart_type_name, gui.plot_style ]),
            widgets.VBox([ gui.normalize_values, gui.progress ])
        ]
    )
    display(widgets.VBox([boxes, itw.children[-1]]))
    itw.update()