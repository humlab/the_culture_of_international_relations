import types
import ipywidgets as widgets
import pandas as pd
import logging
import common.utility as utility
import topic_model_utility
import topic_model
import textacy_corpus_utility as textacy_utility

logger = utility.getLogger('corpus_text_analysis')

gensim_logger = logging.getLogger('gensim')
gensim_logger.setLevel(logging.INFO)

def compile_vectorizer_args(gui, corpus):
    
    args = dict(apply_idf=gui.apply_idf.value)
    
    return args

def compile_topic_modeller_args(gui, corpus):
    
    args = dict(n_topics=gui.n_topics.value, max_iter=gui.max_iter.value, learning_method='online', n_jobs=1)
    
    return args

def compile_tokenizer_args(gui, corpus, **opts):
    
    gpe_substitutions = {}
    
    if gui.mask_gpe.value is True:
        assert opts.get('gpe_filename', None) is not None
        gpe_substitutions = { x: '_gpe_' for x in textacy_utility.get_gpe_names(filename=opts['gpe_filename'], corpus=corpus) }
        
    args = dict(
        args=dict(
            ngrams=gui.ngrams.value,
            named_entities=gui.named_entities.value,
            normalize=gui.normalize.value,
            as_strings=True
        ),
        kwargs=dict(
            min_freq=gui.min_freq.value,
            include_pos=gui.include_pos.value,
            filter_stops=gui.filter_stops.value,
            filter_punct=True
        ),
        extra_stop_words=set(gui.stop_words.value),
        substitutions=gpe_substitutions,
        min_freq=gui.min_freq.value,
        max_doc_freq=gui.max_doc_freq.value
    )
    
    return args

def compute_topic_model(data_folder, corpus, method, n_topics, gui, n_topic_window=0, tick=utility.noop, **opts):
    
    result = None
    
    try:
        
        tick(1)

        gensim_logger = logging.getLogger('gensim')
        gensim_logger.setLevel(logging.INFO if gui.show_trace.value else logging.WARNING)

        vectorizer_args     = compile_vectorizer_args(gui, corpus)
        tokenizer_args      = compile_tokenizer_args(gui, corpus, **opts)
        topic_modeller_args = compile_topic_modeller_args(gui, corpus)

        window_coherences = [ ]

        for x_topics in range(max(n_topics - n_topic_window, 2), n_topics + n_topic_window + 1):

            topic_modeller_args['n_topics'] = x_topics

            logger.info('Computing model with {} topics...'.format(x_topics))
            
            data = topic_model.compute(
                corpus=corpus,
                tick=tick,
                method=method,
                vec_args=vectorizer_args,
                term_args=tokenizer_args,
                tm_args=topic_modeller_args,
                tfidf_weiging=gui.apply_idf.value
            )

            if x_topics == n_topics:
                result = data

            p_score = data.perplexity_score
            c_score = data.coherence_score
            logger.info('#topics: {}, coherence_score {} perplexity {}'.format(x_topics, c_score, p_score))

            if n_topic_window > 0:
                window_coherences.append({'n_topics': x_topics, 'perplexity_score': p_score, 'coherence_score': c_score})

            #if not gui.show_trace.value:
            #    gui.output.clear_output()

        if n_topic_window > 0:
            #df = pd.DataFrame(window_coherences)
            #df['n_topics'] = df.n_topics.astype(int)
            #df = df.set_index('n_topics')
            #model_result.coherence_scores = df
            result.coherence_scores = pd.DataFrame(window_coherences).set_index('n_topics')
            
            #df.to_excel(utility.path_add_timestamp('perplexity.xlsx'))
            #df['perplexity_score'].plot.line()
            
    except Exception as ex:
        logger.error(ex)
        raise
    finally:
        return result
    
def display_topic_model_gui(data_folder, state, corpus, **opts):
    
    def spinner_widget(filename="images/spinner-02.gif", width=40, height=40):
        with open(filename, "rb") as image_file:
            image = image_file.read()
        return widgets.Image(value=image, format='gif', width=width, height=height, layout={'visibility': 'hidden'})

    pos_options = [ x for x in opts['tagset'].POS.unique() if x not in ['PUNCT', '', 'DET', 'X', 'SPACE', 'PART', 'CONJ', 'SYM', 'INTJ', 'PRON']]
    engine_options = [
        ('MALLET LDA', 'gensim_mallet-lda'),
        ('gensim LDA', 'gensim_lda'),
        ('gensim LSI', 'gensim_lsi'),
        ('gensim HDP', 'gensim_hdp'),
        ('gensim DTM', 'gensim_dtm'),
        ('scikit LDA', 'sklearn_lda'),
        ('scikit NMF', 'sklearn_nmf'),
        ('scikit LSA', 'sklearn_lsa'),
        ('STTM   LDA', 'gensim_sttm-lda'),
        ('STTM   BTM', 'gensim_sttm-btm'),
        ('STTM   PTM', 'gensim_sttm-ptm'),
        ('STTM  SATM', 'gensim_sttm-satm'),
        ('STTM   DMM', 'gensim_sttm-dmm'),
        ('STTM  WATM', 'gensim_sttm-watm'),
    ]
    normalize_options = { 'None': False, 'Use lemma': 'lemma', 'Lowercase': 'lower'}
    ngrams_options = { '1': [1], '1, 2': [1, 2], '1,2,3': [1, 2, 3] }
    default_include_pos = [ 'NOUN', 'PROPN' ]
    frequent_words = [ x[0] for x in textacy_utility.get_most_frequent_words(corpus, 100, normalize='lemma', include_pos=default_include_pos) ] + [ '_gpe_' ]
    named_entities_disabled = len(corpus) == 0 or len(corpus[0].spacy_doc.ents) == 0
    gui = types.SimpleNamespace(
        progress=widgets.IntProgress(value=0, min=0, max=5, step=1, description='', layout=widgets.Layout(width='90%')),
        n_topics=widgets.IntSlider(description='#topics', min=2, max=100, value=20, step=1, layout=widgets.Layout(width='240px')),
        min_freq=widgets.IntSlider(description='Min word freq', min=0, max=10, value=2, step=1, layout=widgets.Layout(width='240px')),
        max_doc_freq=widgets.IntSlider(description='Min doc %', min=75, max=100, value=100, step=1, layout=widgets.Layout(width='240px')),
        max_iter=widgets.IntSlider(description='Max iterations', min=100, max=6000, value=2000, step=10, layout=widgets.Layout(width='240px')),
        ngrams=widgets.Dropdown(description='n-grams', options=ngrams_options, value=[1], layout=widgets.Layout(width='200px')),
        normalize=widgets.Dropdown(description='Normalize', options=normalize_options, value='lemma', layout=widgets.Layout(width='200px')),
        filter_stops=widgets.ToggleButton(value=True, description='Remove stopword',  tooltip='Filter out stopwords', icon='check'),
        named_entities=widgets.ToggleButton(value=False, description='Merge entities',  tooltip='Merge entities', icon='check', disabled=named_entities_disabled),
        apply_idf=widgets.ToggleButton(value=False, description='Apply IDF',  tooltip='Apply IDF (skikit-learn) or TF-IDF (gensim)', icon='check'),
        mask_gpe=widgets.ToggleButton(value=False, description='Mask GPE',  tooltip='Mask GPE', icon='check'),
        include_pos=widgets.SelectMultiple(description='POS', options=pos_options, value=default_include_pos, rows=7, layout=widgets.Layout(width='160px')),
        stop_words=widgets.SelectMultiple(description='STOP', options=frequent_words, value=list([]), rows=7, layout=widgets.Layout(width='220px')),
        method=widgets.Dropdown(description='Engine', options=engine_options, value='gensim_lda', layout=widgets.Layout(width='200px')),
        compute=widgets.Button(description='Compute', button_style='Success', layout=widgets.Layout(width='115px',background_color='blue')),
        show_trace=widgets.Checkbox(value=False, description='Show trace', disabled=False), #layout=widgets.Layout(width='80px')),
        boxes=None,
        output=widgets.Output(layout={'border': '1px solid black'}), # , 'height': '400px'
        spinner=spinner_widget()
    )
    
    gui.boxes = widgets.VBox([
        gui.progress,
        widgets.HBox([
            widgets.VBox([
                gui.n_topics,
                gui.min_freq,
                gui.max_doc_freq,
                gui.max_iter
            ]),
            widgets.VBox([
                gui.filter_stops,
                gui.named_entities,
                gui.apply_idf,
                gui.mask_gpe
            ]),
            gui.include_pos,
            gui.stop_words,
            widgets.VBox([
                gui.normalize,
                gui.ngrams,
                gui.method,
                widgets.HBox([gui.spinner, gui.compute], layout=widgets.Layout(align_items='flex-end'))
            ], layout=widgets.Layout(align_items='flex-end')),
            gui.show_trace
        ]),
        widgets.VBox([gui.output]),
    ])
    
    def tick(x=None):
        gui.progress.value = gui.progress.value + 1 if x is None else x
        
    def buzy(is_buzy):
        gui.compute.disabled = is_buzy
        gui.spinner.layout.visibility = 'visible' if is_buzy else 'hidden'
            
    def compute_topic_model_handler(*args):
        gui.output.clear_output()
        buzy(True)

        with gui.output:
            
            try:
                state.data = compute_topic_model(data_folder, corpus, gui.method.value, gui.n_topics.value, gui, tick=tick, **opts)

                topics = topic_model_utility.get_topics_unstacked(
                    state.topic_model,
                    n_tokens=100,
                    id2term=state.id2term,
                    topic_ids=state.relevant_topics
                )

                display(topics)
        
            except Exception as ex:
                logger.error(ex)
                state.data = None
                raise
            finally:
                buzy(False)
        
    gui.compute.on_click(compute_topic_model_handler)
    
    def pos_change_handler(*args):
        with gui.output:
            gui.compute.disabled = True
            selected = set(gui.stop_words.value)
            frequent_words = [ x[0] for x in textacy_utility.get_most_frequent_words(corpus, 100, normalize=gui.normalize.value, include_pos=gui.include_pos.value) ]
            gui.stop_words.options = frequent_words
            selected = selected & set(gui.stop_words.options)
            gui.stop_words.value = list(selected)
            gui.compute.disabled = False
        
    gui.include_pos.observe(pos_change_handler, 'value')
    
    def method_change_handler(*args):
        with gui.output:
            
            gui.compute.disabled = True
            method = gui.method.value
            
            gui.apply_idf.disabled = False
            gui.apply_idf.description = 'Apply TF-IDF' if method.startswith('gensim') else 'Apply IDF'
            
            gui.ngrams.disabled = False
            if 'MALLET' in method:
                gui.ngrams.value = [1]
                gui.ngrams.disabled = True
                gui.apply_idf.description = 'TF-IDF N/A'
                gui.apply_idf.disabled = True
                
            gui.n_topics.disabled = False
            if 'HDP' in method:
                gui.n_topics.value = gui.n_topics.max
                gui.n_topics.disabled = True
            
            gui.compute.disabled = False
        
    gui.method.observe(method_change_handler, 'value')
    method_change_handler()
    display(gui.boxes)
    return gui

        