import types
import ipywidgets as widgets
import pandas as pd
import logging
import common.utility as utility
import topic_model_utility
import topic_model
import textacy_corpus_utility as textacy_utility

logger = utility.getLogger('corpus_text_analysis')
    
def compute_topic_model(corpus, gui, *args):

    def tick(x=None):
        gui.progress.value = gui.progress.value + 1 if x is None else x
    
    tick(1)
    gui.output.clear_output()

    with gui.output:
        vec_args = dict(apply_idf=gui.apply_idf.value)
        term_args = dict(
            args=dict(
                ngrams=gui.ngrams.value,
                named_entities=gui.named_entities.value,
                normalize=gui.normalize.value,
                as_strings=True
            ),
            kwargs=dict(
                filter_nums=gui.filter_nums.value,
                drop_determiners=gui.drop_determiners.value,
                min_freq=gui.min_freq.value,
                include_pos=gui.include_pos.value,
                filter_stops=gui.filter_stops.value,
                filter_punct=True
            ),
            extra_stop_words=gui.stop_words.value
        )
        tm_args = dict(
            n_topics=gui.n_topics.value,
            max_iter=gui.max_iter.value,
            learning_method='online', 
            n_jobs=1
        )
        method = gui.method.value
        gui.model = topic_model.compute(
            corpus=corpus,
            tick=tick,
            method=method,
            vec_args=vec_args,
            term_args=term_args,
            tm_args=tm_args
        )
    
    gui.output.clear_output()
    with gui.output:
        topics = topic_model_utility.get_lda_topics(gui.model.tm_model, n_tokens=20)
        display(topics)
        
def display_topic_model_gui(corpus, tagset):
    
    pos_options = [ x for x in tagset.POS.unique() if x not in ['PUNCT', '', 'DET', 'X', 'SPACE', 'PART', 'CONJ', 'SYM', 'INTJ', 'PRON']]
    engine_options = { 'MALLET LDA': 'gensim_malletlda', 'gensim LDA': 'gensim_lda', 'gensim LSI': 'gensim_lsi' } #, 'sklearn_lda': 'sklearn_lda'}
    normalize_options = { 'None': False, 'Use lemma': 'lemma', 'Lowercase': 'lower'}
    ngrams_options = { '1': [1], '1, 2': [1, 2], '1,2,3': [1, 2, 3] }
    default_include_pos = [ 'NOUN', 'PROPN' ]
    frequent_words = [ x[0] for x in textacy_utility.get_most_frequent_words(corpus, 100, normalize='lemma', include_pos=default_include_pos) ]
    
    gui = types.SimpleNamespace(
        progress=widgets.IntProgress(value=0, min=0, max=5, step=1, description='', layout=widgets.Layout(width='90%')),
        n_topics=widgets.IntSlider(description='#topics', min=5, max=50, value=20, step=1, layout=widgets.Layout(width='240px')),
        min_freq=widgets.IntSlider(description='Min word freq', min=0, max=10, value=2, step=1, layout=widgets.Layout(width='240px')),
        max_iter=widgets.IntSlider(description='Max iterations', min=100, max=2000, value=600, step=10, layout=widgets.Layout(width='240px')),
        ngrams=widgets.Dropdown(description='n-grams', options=ngrams_options, value=[1], layout=widgets.Layout(width='200px')),
        normalize=widgets.Dropdown(description='Normalize', options=normalize_options, value='lemma', layout=widgets.Layout(width='200px')),
        filter_stops=widgets.ToggleButton(value=True, description='Remove stopword',  tooltip='Filter out stopwords', icon='check'),
        filter_nums=widgets.ToggleButton(value=True, description='Remove nums',  tooltip='Filter out stopwords', icon='check'),
        named_entities=widgets.ToggleButton(value=False, description='Merge entities',  tooltip='Merge entities', icon='check'),
        drop_determiners=widgets.ToggleButton(value=True, description='Drop determiners',  tooltip='Drop determiners', icon='check'),
        apply_idf=widgets.ToggleButton(value=False, description='Apply IDF',  tooltip='Apply TF-IDF', icon='check'),
        include_pos=widgets.SelectMultiple(description='POS', options=pos_options, value=default_include_pos, rows=7, layout=widgets.Layout(width='160px')),
        stop_words=widgets.SelectMultiple(description='STOP', options=frequent_words, value=list([]), rows=7, layout=widgets.Layout(width='220px')),
        method=widgets.Dropdown(description='Engine', options=engine_options, value='gensim_lda', layout=widgets.Layout(width='200px')),
        compute=widgets.Button(description='Compute' , layout=widgets.Layout(width='200px',background_color='blue')),
        boxes=None,
        output = widgets.Output(),
        model=None
    )
    
    gui.boxes = widgets.VBox([
        gui.progress,
        widgets.HBox([
            widgets.VBox([
                gui.n_topics,
                gui.min_freq,
                gui.max_iter
            ]),
            widgets.VBox([
                gui.filter_stops,
                gui.filter_nums,
                gui.named_entities,
                gui.drop_determiners,
                gui.apply_idf
            ]),
            gui.include_pos,
            gui.stop_words,
            widgets.VBox([
                gui.normalize,
                gui.ngrams,
                gui.method,
                gui.compute
            ])
        ]),
        widgets.VBox([gui.output]),
    ])
    def compute_topic_model_handler(*args):
        try:
            gui.compute.disabled = True
            compute_topic_model(corpus, gui, *args)
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
