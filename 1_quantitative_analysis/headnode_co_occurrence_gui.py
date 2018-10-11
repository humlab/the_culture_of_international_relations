
import ipywidgets as widgets
from common.widgets_utility import WidgetUtility
import common.widgets_config as widgets_config

OUTPUT_OPTIONS = {
    'Table': 'table',
    'Table, grid': 'qgrid',
    'Table, unstacked': 'unstack',
    'Plot bar': 'plot_bar',
    'Plot stacked bar': 'plot_stacked_bar',
    'Plot scatter': 'plot_scatter',
    'Plot line': 'plot_line',
    'Plot area': 'plot_area',
    'Plot stacked area': 'plot_stacked_area'
}

EXTRA_GROUPBY_OPTIONS = {
    '': None,
    'Topic': [ 'topic_category' ],
}

def display_gui(wti_index, display_function):
    
    def lw(width='180px', left='0'):
        return widgets.Layout(width=width)
    
    def parties_options():
        return ['ALL'] + sorted([ x for x in wti_index.get_countries_list() if x not in ['ALL', 'ALL OTHER']])
    
    tw = WidgetUtility(
        period_group=widgets_config.period_group_widget(layout=lw()),
        topic_group=widgets_config.topic_groups_widget2(layout=lw()),
        party_name=widgets_config.party_name_widget(layout=lw()),
        treaty_filter=widgets_config.treaty_filter_widget(layout=lw()),
        parties=widgets_config.parties_widget(options=parties_options(), value=['ALL'], rows=10),
        
        recode_7corr=widgets_config.toggle('Recode 7CULT', True, tooltip='Recode all treaties with cultural=yes as 7CORR', layout=lw(width='110px')),
        use_lemma=widgets_config.toggle('Use LEMMA', False,  tooltip='Use WordNet lemma', layout=lw(width='110px')),
        remove_stopwords=widgets_config.toggle('Remove STOP', True, tooltip='Do not include stopwords', layout=lw(width='110px')),
        compute_co_occurance=widgets_config.toggle('Cooccurrence', True,  tooltip='Compute Cooccurrence', layout=lw(width='110px')),
        
        extra_groupbys=widgets_config.dropdown('Groupbys', EXTRA_GROUPBY_OPTIONS, None, layout=lw()),
        min_word_size=widgets_config.itext(0, 5, 2, description='Min word:', layout=lw()),
        output_format=widgets_config.dropdown('Output', OUTPUT_OPTIONS, 'plot_stacked_bar', layout=lw()),
        plot_style=widgets_config.plot_style_widget(layout=lw()),
        n_top=widgets_config.slider('Top/grp #:', 2, 100, 25, step=10),
        n_min_count=widgets_config.slider('Min count:', 1, 10, 5, step=1, tooltip='Filter out words with count less than specified value'),
        progress=widgets_config.progress(0, 8, 1, 0, layout=lw(width='95%'))
    )

    def progress_set(x):
        tw.progress.value = x
        
    iw = widgets.interactive(
        display_function,
        period_group=tw.period_group,
        topic_group=tw.topic_group,
        recode_7corr=tw.recode_7corr,
        treaty_filter=tw.treaty_filter,
        parties=tw.parties,
        extra_groupbys=tw.extra_groupbys,
        n_min_count=tw.n_min_count,
        n_top=tw.n_top,
        min_word_size=tw.min_word_size,
        use_lemma=tw.use_lemma,
        compute_co_occurance=tw.compute_co_occurance,
        remove_stopwords=tw.remove_stopwords,
        output_format=tw.output_format,
        progress=widgets.fixed(progress_set),
        wti_index=widgets.fixed(wti_index)
        # plot_style=tw.plot_style
    )

    boxes = widgets.HBox(
        [
            widgets.VBox([ tw.period_group, tw.parties ]),
            widgets.VBox([ tw.topic_group, tw.extra_groupbys, tw.n_top, tw.min_word_size, tw.n_min_count]),
            widgets.VBox([ tw.recode_7corr, tw.use_lemma, tw.remove_stopwords, tw.compute_co_occurance ]),
            widgets.VBox([ tw.treaty_filter]),
            widgets.VBox([ tw.output_format, tw.progress ])
        ]
    )
    display(widgets.VBox([boxes, iw.children[-1]]))
    iw.update()
    