import sys

import ipywidgets as widgets
from IPython.display import display


sys.path = sys.path if '..' in sys.path else sys.path + ['..']

import common.widgets_config as widgets_config
import common.config as config
import common.utility as utility
import headnote_corpus

from pprint import pprint as pp

logger = utility.getLogger(name='title_analysis')

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

def compute_co_occurrance(treaty_tokens):

    """Computes and returns a co-occurencence matrix

        Parameters
        ----------
        treaty_tokens : DataFrame
            Subset of treaty tokens for which we want to compute co-occurrence
        """
    co_occurrance = treaty_tokens.merge(treaty_tokens, how='inner', left_index=True, right_index=True)

    co_occurrance = co_occurrance.loc[(co_occurrance['token_x'] < co_occurrance['token_y'])]

    co_occurrance['token'] = co_occurrance.apply(lambda row: ' - '.join([row['token_x'].upper(), row['token_y'].upper()]), axis=1)
    co_occurrance['lemma'] = co_occurrance.apply(lambda row: ' - '.join([row['lemma_x'].upper(), row['lemma_y'].upper()]), axis=1)

    co_occurrance = co_occurrance.assign(is_stopword=False, sequence_id=0)[['sequence_id', 'token', 'lemma', 'is_stopword']]

    return co_occurrance

def display_headnote_toplist(
    period_group_index=None,
    topic_group_name=None,
    recode_is_cultural=False,
    treaty_filter=None,
    parties=None,
    extra_groupbys=None,
    use_lemma=False,
    compute_co_occurance=False,
    remove_stopwords=True,
    min_word_size=2,
    n_min_count=1,
    output_format='table',
    n_top=50,
    progress=utility.noop,
    wti_index=None,
    print_args=False
):
    """Display headnote word co-occurrences

    Parameters
    ----------

    Returns
    -------

    """
    try:
        if print_args:
            args = utility.filter_dict(locals(), [ 'progress', 'print_args' ], filter_out=True)
            args['wti_index'] = None
            pp(args)

        corpus = headnote_corpus.HeadnoteTokenCorpus(treaties=wti_index.treaties)

        period_group = config.DEFAULT_PERIOD_GROUPS[period_group_index]
        topic_group = config.TOPIC_GROUP_MAPS[topic_group_name]

        progress()

        """ Get subset of treaties filtered by topic, period, and is-cultural filter """
        treaties = wti_index.get_categorized_treaties(
            topic_category=topic_group,
            period_group=period_group,
            treaty_filter=treaty_filter,
            recode_is_cultural=recode_is_cultural
        )

        if parties is not None and not 'ALL' in parties:
            treaties = treaties.loc[(treaties.party1.isin(parties))|(treaties.party2.isin(parties))]

        if treaties.shape[0] == 0:
            print('No data for selection')
            return

        progress()

        token_or_lemma = 'token' if not use_lemma else 'lemma'

        if compute_co_occurance:
            treaty_tokens = corpus.get_tokens_for(treaties.index)
            treaty_tokens = treaty_tokens\
                .loc[treaty_tokens[token_or_lemma].str.len() >= min_word_size]\
                .reset_index()\
                .drop(['is_stopword', 'sequence_id'], axis=1)\
                .set_index('treaty_id')
            treaty_tokens = compute_co_occurrance(treaty_tokens)

        else:
            #FIXME: Filter based on token-length
            treaty_tokens = corpus.tokens
            if remove_stopwords is True:
                treaty_tokens = treaty_tokens.loc[treaty_tokens.is_stopword==False]
            treaty_tokens = treaty_tokens.reset_index().set_index('treaty_id')

        progress()

        treaty_tokens = treaty_tokens.merge(treaties[['signed_year']], how='inner', left_index=True, right_index=True)

        progress()

        groupbys  = ([ period_group['column'] ] if not period_group is None else []) +\
                    (extra_groupbys or []) +\
                    [ token_or_lemma ]

        result = treaty_tokens.groupby(groupbys).size().reset_index().rename(columns={0: 'Count'})

        progress()

        ''' Filter out the n_top most frequent words from each group '''
        result = result\
                    .groupby(groupbys[-1])\
                    .apply(lambda x: x.nlargest(n_top, 'Count'))\
                    .reset_index(level=0, drop=True)

        if n_min_count > 1:
            result = result.loc[result.Count >= n_min_count]

        progress()

        result = result.sort_values(groupbys[:-1] + ['Count'], ascending=len(groupbys[:-1])*[True] + [False])

        progress()

        if output_format in ('table', 'qgrid'):
            result.columns = [ remove_snake_case(x) for x in result.columns ]
            if output_format == 'table':
                display(result)
            else:
                qgrid.show_grid(result, show_toolbar=True)
        elif output_format == 'unstack':
            result = result.set_index(groupbys).unstack(level=0).fillna(0).astype('int32')
            result.columns = [ x[1] for x in result.columns ]
            display(result)
        elif output_format.startswith('plot'):
            parts = output_format.split('_')
            kind = parts[-1]
            stacked = 'stacked' in parts
            result = result.set_index(list(reversed(groupbys))).unstack(level=0).fillna(0).astype('int32')
            result.columns = [ x[1] for x in result.columns ]
            result.plot(kind=kind, stacked=stacked, figsize=(16,8))

        progress(0)

    except Exception as ex:
        raise
        logger.error(ex)
    finally:
        progress(0)

def display_gui(wti_index, print_args=False):

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
        display_headnote_toplist,
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
        wti_index=widgets.fixed(wti_index),
        print_args=widgets.fixed(print_args)
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
