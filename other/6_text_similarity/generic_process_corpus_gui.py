import types
import ipywidgets as widgets
import textacy_corpus_utility as textacy_utility
import common.utility as utility

logger = utility.getLogger('corpus_text_analysis')

def process_corpus(corpus, terms_opts, process_function, process_opts):

    def get_merged_words(scores, low, high):
        ws = set([])
        for x, wl in scores.items():
            if low <= x <= high:
                ws.update(wl)
        return ws

    normalize = terms_opts['normalize'] or 'orth'

    extra_stop_words = set(terms_opts.get('extra_stop_words', []))
    if terms_opts.get('min_freq', 1) > 1:
        assert 'word_count_scores' in terms_opts
        extra_stop_words.update(get_merged_words(terms_opts['word_count_scores'], 1, terms_opts.get('min_freq')))

    if terms_opts.get('max_doc_freq') < 100:
        assert 'word_document_count_scores' in terms_opts
        extra_stop_words.update(get_merged_words(terms_opts['word_document_count_scores'], terms_opts.get('max_doc_freq'), 100))

    #if terms_opts.get('mask_gpe', False):
    #    extra_stop_words.update(['_gpe_'])

    extract_args = dict(
        args=dict(
            ngrams=terms_opts['ngrams'],
            named_entities=terms_opts['named_entities'],
            normalize=terms_opts['normalize'],
            as_strings=True
        ),
        kwargs=dict(
            min_freq=terms_opts['min_freq'],
            include_pos=terms_opts['include_pos'],
            filter_stops=terms_opts['filter_stops'],
            filter_punct=terms_opts['filter_punct']
        ),
        extra_stop_words=extra_stop_words,
        substitutions=(terms_opts.get('gpe_substitutions', []) if terms_opts.get('mask_gpe', False) else None),
    )

    process_function(corpus, process_opts, extract_args)


def process_corpus_gui(container, wti_index, process_function, **opts):

    lw = lambda width: widgets.Layout(width=width)

    def frequent_words(corpus, normalize, include_pos, n_top=100):

        if include_pos is None or include_pos == ('', ):
            include_pos = []

        return [
            x[0] for x in textacy_utility.get_most_frequent_words(
                corpus, n_top, normalize=normalize, include_pos=include_pos
            )
        ] + [ '_gpe_' ]

    #logger.info('Preparing corpus statistics...')

    corpus = container.textacy_corpus

    gpe_substitutions = { }
    if opts.get('gpe_filename', None) is not None:
        logger.info('...loading term substitution mappings...')
        gpe_substitutions = { x: '_gpe_' for x in textacy_utility.load_term_substitutions(filepath=opts['gpe_filename'], vocab=None) }

    pos_tags = opts.get('tagset').groupby(['POS'])['DESCRIPTION'].apply(list).apply(lambda x: ', '.join(x[:1])).to_dict()

    if False:  # display_pos_legend:
        pos_options = sorted([(k + ' (' + v + ')', k) for k,v in pos_tags.items() ])
    else:
        pos_options = sorted([(k, k) for k,v in pos_tags.items() ])

    ngrams_options = { '1': [1], '1,2': [1,2], '1,2,3': [1,2,3]}
    default_normalize = 'lemma'
    gui = types.SimpleNamespace(
        progress=widgets.IntProgress(value=0, min=0, max=5, step=1, description='', layout=lw('90%')),
        min_freq=widgets.IntSlider(description='Min word freq', min=0, max=10, value=2, step=1, layout=lw('240px')),
        max_doc_freq=widgets.IntSlider(description='Min doc. %', min=75, max=100, value=100, step=1, layout=lw('240px')),
        mask_gpe=widgets.ToggleButton(value=False, description='Mask GPE',  tooltip='Replace geographical entites with `_gpe_`', icon='check', layout=lw('115px')),
        ngrams=widgets.Dropdown(description='n-grams', options=ngrams_options, value=[1], layout=lw('180px')),
        min_word=widgets.Dropdown(description='Min length', options=[1,2,3,4], value=1, layout=lw('180px')),
        normalize=widgets.Dropdown(description='Normalize', options=[ None, 'lemma', 'lower' ], value=default_normalize, layout=lw('180px')),
        filter_stops=widgets.ToggleButton(value=False, description='Filter stops',  tooltip='Filter out stopwords', icon='check', layout=lw('115px')),
        filter_punct=widgets.ToggleButton(value=False, description='Filter punct',  tooltip='Filter out punctuations', icon='check', layout=lw('115px')),
        named_entities=widgets.ToggleButton(value=False, description='Merge entities',  tooltip='Merge entities', icon='check', disabled=True, layout=lw('115px')),
        apply_idf=widgets.ToggleButton(value=False, description='Apply IDF',  tooltip='Apply IDF (skikit-learn) or TF-IDF (gensim)', icon='check'),
        include_pos=widgets.SelectMultiple(description='POS', options=pos_options, value=list(), rows=10, layout=lw('180px')),
        stop_words=widgets.SelectMultiple(description='STOP', options=[], value=list(), rows=10, layout=widgets.Layout(width='220px')),
        output=widgets.Output(),
        compute=widgets.Button(description='Compute', button_style='Success', layout=lw('115px'))
    )

    #logger.info('...word counts...')
    #word_count_scores = opts.get('word_count_scores', None) or dict(
    #    lemma=textacy_utility.generate_word_count_score(corpus, 'lemma', gui.min_freq.max),
    #    lower=textacy_utility.generate_word_count_score(corpus, 'lower', gui.min_freq.max),
    #    orth=textacy_utility.generate_word_count_score(corpus, 'orth', gui.min_freq.max)
    #)

    #logger.info('...word document count...')
    #word_document_count_scores = opts.get('word_document_count_scores', None) or dict(
    #    lemma=textacy_utility.generate_word_document_count_score(corpus, 'lemma', gui.max_doc_freq.min),
    #    lower=textacy_utility.generate_word_document_count_score(corpus, 'lower', gui.max_doc_freq.min),
    #    orth=textacy_utility.generate_word_document_count_score(corpus, 'orth', gui.max_doc_freq.min)
    #)

    #logger.info('...done!')

    def pos_change_handler(*args):
        with gui.output:
            gui.compute.disabled = True
            selected = set(gui.stop_words.value)
            gui.stop_words.options = frequent_words(corpus, gui.normalize.value, gui.include_pos.value)
            selected = selected & set(gui.stop_words.options)
            gui.stop_words.value = list(selected)
            gui.compute.disabled = False

    pos_change_handler()

    gui.include_pos.observe(pos_change_handler, 'value')

    def tick(x=None, max_value=None):
        if max_value is not None:
            gui.progress.max = max_value
        gui.progress.value = gui.progress.value + 1 if x is None else x

    def buzy(is_buzy):
        gui.compute.disabled = is_buzy
        #gui.spinner.layout.visibility = 'visible' if is_buzy else 'hidden'

    def process_corpus_handler(*args):

        gui.output.clear_output()
        buzy(True)

        with gui.output:

            try:

                terms_opts = dict(
                    min_freq=gui.min_freq.value,
                    max_doc_freq=gui.max_doc_freq.value,
                    mask_gpe=gui.mask_gpe.value,
                    ngrams=gui.ngrams.value,
                    min_word=gui.min_word.value,
                    normalize=gui.normalize.value,
                    filter_stops=gui.filter_stops.value,
                    filter_punct=gui.filter_punct.value,
                    named_entities=gui.named_entities.value,
                    include_pos=gui.include_pos.value,
                    extra_stop_words=gui.stop_words.value,
                    gpe_substitutions=gpe_substitutions,
                    word_count_scores=container.get_word_count(gui.normalize.value),
                    word_document_count_scores=container.get_word_document_count(gui.normalize.value)
                )

                process_opts = dict(
                    container=container,
                    gui=gui,
                    tick=tick
                )
                process_opts.update(opts)

                process_corpus(corpus, terms_opts, process_function, process_opts)

                # display(result)

            except Exception as ex:
                logger.error(ex)
                raise
            finally:
                buzy(False)

    gui.compute.on_click(process_corpus_handler)

    gui.boxes = widgets.VBox([
        gui.progress,
        widgets.HBox([
            widgets.VBox([
                widgets.HBox([gui.normalize]),
                widgets.HBox([gui.ngrams]),
                widgets.HBox([gui.min_word]),
                gui.min_freq,
                gui.max_doc_freq
            ]),
            widgets.VBox([
                gui.include_pos
            ]),
            widgets.VBox([
                gui.stop_words
            ]),
            widgets.VBox([
                gui.filter_stops,
                gui.mask_gpe,
                gui.filter_punct,
                gui.named_entities,
                gui.compute
            ])
        ]),
        widgets.HBox([
            gui.output
        ])
    ])

    display(gui.boxes)

    return gui
