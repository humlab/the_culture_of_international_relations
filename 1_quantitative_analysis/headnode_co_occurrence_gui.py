
import ipywidgets as widgets
from IPython.display import display
import common.widgets_config as widgets_config
import common.config as config

OUTPUT_OPTIONS = {
    'Table': 'table',
    'Table, grid': 'qgrid',
    'Table, unstacked': 'unstack',
    'Plot bar': 'plot_bar',
    'Plot stacked bar': 'plot_stacked_bar',
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

    period_group_index_widget = widgets_config.period_group_widget(index_as_value=True)
    topic_group_name_widget = widgets_config.topic_groups_widget(layout=lw())
    parties_widget = widgets_config.parties_widget(options=parties_options(), value=['ALL'], rows=10)

    recode_7corr_widget = widgets_config.toggle('Recode 7CULT', True, tooltip='Recode all treaties with cultural=yes as 7CORR', layout=lw(width='110px'))
    use_lemma_widget = widgets_config.toggle('Use LEMMA', False,  tooltip='Use WordNet lemma', layout=lw(width='110px'))
    remove_stopwords_widget = widgets_config.toggle('Remove STOP', True, tooltip='Do not include stopwords', layout=lw(width='110px'))
    compute_co_occurance_widget = widgets_config.toggle('Cooccurrence', True,  tooltip='Compute Cooccurrence', layout=lw(width='110px'))

    extra_groupbys_widget = widgets_config.dropdown('Groupbys', EXTRA_GROUPBY_OPTIONS, None, layout=lw())
    min_word_size_widget = widgets_config.itext(0, 5, 2, description='Min word', layout=lw())
    # plot_style_widget = widgets_config.plot_style_widget(layout=lw())
    n_top_widget = widgets_config.slider('Top/grp #', 2, 100, 25, step=10, layout=lw(width='200px'))
    n_min_count_widget = widgets_config.slider('Min count', 1, 10, 5, step=1, tooltip='Filter out words with count less than specified value', layout=lw(width='200px'))
    
    treaty_filter_widget = widgets_config.dropdown('Filter', config.TREATY_FILTER_OPTIONS, 'is_cultural', layout=lw())
    output_format_widget = widgets_config.dropdown('Output', OUTPUT_OPTIONS, 'plot_stacked_bar', layout=lw())
    progress_widget = widgets_config.progress(0, 8, 1, 0, layout=lw(width='95%'))

    def progress(x=None):
        progress_widget.value = progress_widget.value + 1 if x is None else x

    iw = widgets.interactive(
        display_function,
        period_group_index=period_group_index_widget,
        topic_group_name=topic_group_name_widget,
        recode_7corr=recode_7corr_widget,
        treaty_filter=treaty_filter_widget,
        parties=parties_widget,
        extra_groupbys=extra_groupbys_widget,
        n_min_count=n_min_count_widget,
        n_top=n_top_widget,
        min_word_size=min_word_size_widget,
        use_lemma=use_lemma_widget,
        compute_co_occurance=compute_co_occurance_widget,
        remove_stopwords=remove_stopwords_widget,
        output_format=output_format_widget,
        progress=widgets.fixed(progress),
        wti_index=widgets.fixed(wti_index)
        # plot_style=plot_style
    )

    boxes = widgets.HBox(
        [
            widgets.VBox([ period_group_index_widget, parties_widget ]),
            widgets.VBox([ topic_group_name_widget, extra_groupbys_widget, n_top_widget, min_word_size_widget, n_min_count_widget]),
            widgets.VBox([ recode_7corr_widget, use_lemma_widget, remove_stopwords_widget, compute_co_occurance_widget ]),
            widgets.VBox([ treaty_filter_widget, output_format_widget, progress_widget]),
        ]
    )
    display(widgets.VBox([boxes, iw.children[-1]]))
    iw.update()
