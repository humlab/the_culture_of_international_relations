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

def compile_tokenizer_args(gui, corpus):
       
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
        mask_gpe=gui.mask_gpe.value,
        min_freq=gui.min_freq.value,
        max_doc_freq=gui.max_doc_freq.value
    )
    
    return args

def compute_topic_model(corpus, method, n_topics, gui, n_topic_window=0):
    try:
        def tick(x=None):
            gui.progress.value = gui.progress.value + 1 if x is None else x

        tick(1)

        with gui.output:
            vectorizer_args     = compile_vectorizer_args(gui, corpus)
            tokenizer_args      = compile_tokenizer_args(gui, corpus)
            topic_modeller_args = compile_topic_modeller_args(gui, corpus)
            #print(vectorizer_args)
            #print(tokenizer_args)
            #print(topic_modeller_args)

        gui.payload = { 'models': [] }

        for x_topics in range(max(n_topics - n_topic_window, 2), n_topics + n_topic_window + 1):

            with gui.output:
                gui.spinner.layout.visibility = 'visible'
                topic_modeller_args['n_topics'] = x_topics

                logger.info('Computing model with {} topics...'.format(x_topics))
                candidate_model = topic_model.compute(
                    corpus=corpus,
                    tick=tick,
                    method=method,
                    vec_args=vectorizer_args,
                    term_args=tokenizer_args,
                    tm_args=topic_modeller_args
                )

                gui.payload['models'].append(candidate_model)

                if x_topics == n_topics:
                    gui.model = candidate_model

                if n_topic_window > 0:
                    logger.info('#topics: {}, coherence_score {} perplexity {}'.format(x_topics, candidate_model.coherence_score, candidate_model.perplexity_score))

                gui.spinner.layout.visibility = 'hidden'

            #gui.output.clear_output()

            logger.info('#topics: {}, coherence_score {} perplexity {}'.format(n_topics, gui.model.coherence_score, gui.model.perplexity_score))

        with gui.output:
            relevant_topic_ids = gui.model.compiled_data.relevant_topic_ids
            topics = topic_model_utility.get_topics_unstacked(gui.model.tm_model, n_tokens=100, id2term=gui.model.tm_id2term, topic_ids=relevant_topic_ids)
            display(topics)

        if n_topic_window > 0:

            models = gui.payload['models']
            df = pd.DataFrame([ {
                'n_topics': int(model.tm_model.num_topics),
                'perplexity_score': model.perplexity_score,
                'coherence_score': model.coherence_score
              } for model in models ])
            df['n_topics'] = df.n_topics.astype(int)
            df = df.set_index('n_topics')
            df.to_excel(utility.path_add_timestamp('perplexity.xlsx'))
            df['perplexity_score'].plot.line()
            
    except Exception as ex:
        with gui.output:
            logger.error(ex)

def display_topic_model_gui(corpus, tagset):
    
    def spinner_widget(filename="images/spinner-02.gif", width=40, height=40):
        with open(filename, "rb") as image_file:
            image = image_file.read()
        return widgets.Image(value=image, format='gif', width=width, height=height, layout={'visibility': 'hidden'})

    pos_options = [ x for x in tagset.POS.unique() if x not in ['PUNCT', '', 'DET', 'X', 'SPACE', 'PART', 'CONJ', 'SYM', 'INTJ', 'PRON']]
    engine_options = [
        ('MALLET LDA', 'gensim_malletlda'),
        ('gensim LDA', 'gensim_lda'),
        ('gensim LSI', 'gensim_lsi'),
        ('gensim HDP', 'gensim_hdp'),
        ('gensim DTM', 'gensim_dtm'),
        ('scikit LDA', 'sklearn_lda'),
        ('scikit NMF', 'sklearn_nmf'),
        ('scikit LSA', 'sklearn_lsa')        
    ]
    normalize_options = { 'None': False, 'Use lemma': 'lemma', 'Lowercase': 'lower'}
    ngrams_options = { '1': [1], '1, 2': [1, 2], '1,2,3': [1, 2, 3] }
    default_include_pos = [ 'NOUN', 'PROPN' ]
    frequent_words = [ x[0] for x in textacy_utility.get_most_frequent_words(corpus, 100, normalize='lemma', include_pos=default_include_pos) ]
    named_entities_disabled = len(corpus) == 0 or len(corpus[0].spacy_doc.ents) == 0
    gui = types.SimpleNamespace(
        progress=widgets.IntProgress(value=0, min=0, max=5, step=1, description='', layout=widgets.Layout(width='90%')),
        n_topics=widgets.IntSlider(description='#topics', min=5, max=100, value=20, step=1, layout=widgets.Layout(width='240px')),
        min_freq=widgets.IntSlider(description='Min word freq', min=0, max=10, value=2, step=1, layout=widgets.Layout(width='240px')),
        max_doc_freq=widgets.FloatSlider(description='Min doc freq', min=0.75, max=1.0, value=1.0, step=0.01, layout=widgets.Layout(width='240px')),
        max_iter=widgets.IntSlider(description='Max iterations', min=100, max=6000, value=2000, step=10, layout=widgets.Layout(width='240px')),
        ngrams=widgets.Dropdown(description='n-grams', options=ngrams_options, value=[1], layout=widgets.Layout(width='200px')),
        normalize=widgets.Dropdown(description='Normalize', options=normalize_options, value='lemma', layout=widgets.Layout(width='200px')),
        filter_stops=widgets.ToggleButton(value=True, description='Remove stopword',  tooltip='Filter out stopwords', icon='check'),
        #filter_nums=widgets.ToggleButton(value=True, description='Remove nums',  tooltip='Filter out stopwords', icon='check'),
        named_entities=widgets.ToggleButton(value=False, description='Merge entities',  tooltip='Merge entities', icon='check', disabled=named_entities_disabled),
        #drop_determiners=widgets.ToggleButton(value=False, description='Drop DET',  tooltip='Drop determiners', icon='check', disabled=True),
        apply_idf=widgets.ToggleButton(value=False, description='Apply IDF',  tooltip='Apply TF-IDF', icon='check'),
        mask_gpe=widgets.ToggleButton(value=False, description='Mask GPE',  tooltip='Mask GPE', icon='check'),
        include_pos=widgets.SelectMultiple(description='POS', options=pos_options, value=default_include_pos, rows=7, layout=widgets.Layout(width='160px')),
        stop_words=widgets.SelectMultiple(description='STOP', options=frequent_words, value=list([]), rows=7, layout=widgets.Layout(width='220px')),
        method=widgets.Dropdown(description='Engine', options=engine_options, value='gensim_lda', layout=widgets.Layout(width='200px')),
        compute=widgets.Button(description='Compute', button_style='Success', layout=widgets.Layout(width='115px',background_color='blue')),
        boxes=None,
        output=widgets.Output(layout={'border': '1px solid black'}), # , 'height': '400px'
        model=None,
        payload={},
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
                #gui.filter_nums,
                gui.named_entities,
                #gui.drop_determiners,
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
            ], layout=widgets.Layout(align_items='flex-end'))
        ]),
        widgets.VBox([gui.output]),
    ])
    
    def compute_topic_model_handler(*args):
        try:
            gui.output.clear_output()
            gui.compute.disabled = True
            with gui.output:
                compute_topic_model(corpus, gui.method.value, gui.n_topics.value, gui)
        finally:
            gui.compute.disabled = False
            
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
    
    display(gui.boxes)
    return gui
