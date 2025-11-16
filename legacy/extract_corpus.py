# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: 'Python 3.7.6 64-bit (''tcoir2'': pipenv)'
#     language: python
#     name: python37664bittcoir2pipenv6e54fafa2278416988eb069a5b65da20
# ---

# %% [markdown]
# ## The Culture of International Relations - Text Analysis
# ### <span style='color: green'>SETUP </span> Prepare and Setup Notebook <span style='float: right; color: red'>MANDATORY</span>

# %% code_folding=[]

import os, collections
from typing import Any
import nltk, textacy 
import pandas as pd
import ipywidgets as widgets
import types

import matplotlib.pyplot as plt
from common import utility, widgets_config, config, utility, treaty_state
from common.corpus import corpus_utility
from common.gui.load_wti_index_gui import current_wti_index, load_wti_index_with_gui
from common.gui import textacy_corpus_gui
from common.gui.utility import get_treaty_time_groupings
from loguru import logger
from IPython.display import display

utility.setup_default_pd_display()

PATTERN = '*.txt'
PERIOD_GROUP = 'years_1945-1972'
DF_TAGSET: pd.DataFrame = pd.read_csv('../data/tagset.csv', sep='\t').fillna('')
load_wti_index_with_gui(data_folder=config.DATA_FOLDER)
TREATY_TIME_GROUPINGS: dict[str, dict[str, Any]] = get_treaty_time_groupings()

# %matplotlib inline
# set_matplotlib_formats('svg')   
#bokeh.plotting.output_notebook()

current_corpus_container = corpus_utility.CorpusContainer.container
current_corpus = corpus_utility.CorpusContainer.corpus


# %% [markdown]
# ## <span style='color: green'>PREPARE </span> Load and Prepare Corpus <span style='float: right; color: red'>MANDATORY</span>
#

# %%
container = current_corpus_container()
textacy_corpus_gui.display_corpus_load_gui(config.DATA_FOLDER, current_wti_index(), container)


# %% [markdown]
# ## <span style='color: green'>PREPARE/DESCRIBE </span> Find Key Terms <span style='float: right; color: green'>OPTIONAL</span>
# - [TextRank]	Mihalcea, R., & Tarau, P. (2004, July). TextRank: Bringing order into texts. Association for Computational Linguistics.
# - [SingleRank]	Hasan, K. S., & Ng, V. (2010, August). Conundrums in unsupervised keyphrase extraction: making sense of the state-of-the-art. In Proceedings of the 23rd International Conference on Computational Linguistics: Posters (pp. 365-373). Association for Computational Linguistics.
# - [RAKE]	Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010). Automatic Keyword Extraction from Individual Documents. In M. W. Berry & J. Kogan (Eds.), Text Mining: Theory and Applications: John Wiley & Son
# https://github.com/csurfer/rake-nltk
# https://github.com/aneesha/RAKE
# https://github.com/vgrabovets/multi_rake
#
#

# %% [markdown]
# #### <span style='color: green'>PREPARE/DESCRIBE </span>RAKE <span style='float: right; color: green'>WORK IN PROGRESS</span>
#
# https://github.com/JRC1995/RAKE-Keyword-Extraction
# https://github.com/JRC1995/TextRank-Keyword-Extraction

# %% code_folding=[0]
# Document Key Terms
from common.gui.utility import get_treaty_dropdown_options
from rake_nltk import Rake, Metric
import string

def textacy_rake(doc, language='english', normalize='lemma', n_keyterms=20, stopwords=None, metric=Metric.DEGREE_TO_FREQUENCY_RATIO):
    punctuations = string.punctuation + "\""
    r = Rake(
        stopwords=stopwords,  # NLTK stopwords if None
        punctuations=punctuations, # NLTK by default
        language=language,
        ranking_metric=metric,
        max_length=100000,
        min_length=1
    )
    text = ' '.join([ x.lemma_ for x in doc.spacy_doc ] if normalize == 'lemma' else [ x.lower_ for x in doc.spacy_doc ])
    r.extract_keywords_from_text(doc.text)
    keyterms = [ (y, x) for (x, y) in r.get_ranked_phrases_with_scores() ]
    return keyterms[:n_keyterms]


def display_rake_gui(corpus, language):
    
    lw = lambda width: widgets.Layout(width=width)
    
    document_options = get_treaty_dropdown_options(current_wti_index(), corpus)
    metric_options = [
        ('Degree / Frequency', Metric.DEGREE_TO_FREQUENCY_RATIO),
        ('Degree', Metric.WORD_DEGREE),
        ('Frequency', Metric.WORD_FREQUENCY)
    ]
    gui = types.SimpleNamespace(
        position=1,
        progress=widgets.IntProgress(min=0, max=1, step=1, layout=lw(width='95%')),
        output=widgets.Output(layout={'border': '1px solid black'}),
        n_keyterms=widgets.IntSlider(description='#words', min=10, max=500, value=10, step=1, layout=lw(width='340px')),
        document_id=widgets.Dropdown(description='Treaty', options=document_options, value=document_options[1][1], layout=lw(width='40%')),
        metric=widgets.Dropdown(description='Metric', options=metric_options, value=Metric.DEGREE_TO_FREQUENCY_RATIO, layout=lw(width='300px')),
        normalize=widgets.Dropdown(description='Normalize', options=[ 'lemma', 'lower' ], value='lemma', layout=lw(width='160px')),
        left=widgets.Button(description='<<', button_style='Success', layout=lw('40px')),
        right=widgets.Button(description='>>', button_style='Success', layout=lw('40px')),
    )
    
    def compute_textacy_rake(corpus, treaty_id, language, normalize, n_keyterms, metric):
        gui.output.clear_output()
        with gui.output:
            doc = corpus_utility.get_treaty_doc(corpus, treaty_id)
            phrases = textacy_rake(doc, language=language, normalize=normalize, n_keyterms=n_keyterms, stopwords=None, metric=metric)
            df = pd.DataFrame(phrases, columns=['phrase', 'score'])
            display(df.set_index('phrase'))
            return df
    
    itw = widgets.interactive(
        compute_textacy_rake,
        corpus=widgets.fixed(corpus),
        treaty_id=gui.document_id,
        language=widgets.fixed(language),
        normalize=gui.normalize,
        n_keyterms=gui.n_keyterms,
        metric=gui.metric
    )
    
    def back_handler(*args):
        if gui.position == 0:
            return
        gui.output.clear_output()
        gui.position = (gui.position - 1) % len(document_options)
        gui.document_id.value = document_options[gui.position][1]
        #itw.update()
        
    def forward_handler(*args):
        gui.output.clear_output()
        gui.position = (gui.position + 1) % len(document_options)
        gui.document_id.value = document_options[gui.position][1]
    
    gui.left.on_click(back_handler)
    gui.right.on_click(forward_handler)
    
    display(widgets.VBox([
        gui.progress,
        widgets.HBox([gui.document_id, gui.left, gui.right, gui.metric, gui.normalize]),
        widgets.HBox([gui.n_keyterms]),
        gui.output
    ]))

    itw.update()

try:
    display_rake_gui(current_corpus(), language='english')
except Exception as ex:
    logger.error(ex)


# %% [markdown]
# #### <span style='color: green'>PREPARE/DESCRIBE </span>TextRank/SingleRank <span style='float: right; color: green'>OPTIONAL</span>
#
# https://github.com/JRC1995/TextRank-Keyword-Extraction
#   

# %% code_folding=[0]

def display_document_key_terms_gui(corpus, wti_index):
    
    lw = lambda width: widgets.Layout(width=width)
    
    methods = { 'RAKE': textacy_rake, 'SingleRank': textacy.keyterms.singlerank, 'TextRank': textacy.keyterms.textrank }
    document_options = [('All Treaties', None)] + get_treaty_dropdown_options(wti_index, corpus)    
    
    gui = types.SimpleNamespace(
        position=1,
        progress=widgets.IntProgress(min=0, max=1, step=1, layout=lw(width='95%')),
        output=widgets.Output(layout={'border': '1px solid black'}),
        n_keyterms=widgets.IntSlider(description='#words', min=10, max=500, value=100, step=1, layout=lw('240px')),
        document_id=widgets.Dropdown(description='Treaty', options=document_options, value=document_options[0][1], layout=lw(width='40%')),        
        method=widgets.Dropdown(description='Algorithm', options=[ 'RAKE', 'TextRank', 'SingleRank' ], value='TextRank', layout=lw('180px')),
        normalize=widgets.Dropdown(description='Normalize', options=[ 'lemma', 'lower' ], value='lemma', layout=lw('160px')),
        left=widgets.Button(description='<<', button_style='Success', layout=lw('40px')),
        right=widgets.Button(description='>>', button_style='Success', layout=lw('40px')),
    )
    
    def get_keyterms(method, doc, normalize, n_keyterms):
        keyterms = methods[method](doc, normalize=normalize, n_keyterms=n_keyterms)
        terms = ', '.join([ x for x, y in keyterms ])
        gui.progress.value += 1
        return terms
    
    def get_document_key_terms(corpus, method='TextRank', document_id=None, normalize='lemma', n_keyterms=10):
        treaty_ids = [ document_id ] if document_id is not None else [ doc.metadata['treaty_id'] for doc in corpus ]
        gui.progress.value = 0
        gui.progress.max = len(treaty_ids)
        keyterms = [
            get_keyterms(method, corpus_utility.get_treaty_doct(corpus, treaty_id), normalize, n_keyterms) for treaty_id in treaty_ids
        ]
        df = pd.DataFrame({ 'treaty_id': treaty_ids, 'keyterms': keyterms}).set_index('treaty_id')
        
        # Add parties and groups
        df_extended = pd.merge(df, wti_index.treaties, left_index=True, right_index=True, how='inner')
        group_map = wti_index.get_parties()['group_name'].to_dict()        
        df_extended['group1'] = df_extended['party1'].map(group_map)
        df_extended['group2'] = df_extended['party2'].map(group_map)        
        columns = ['signed_year', 'party1', 'group1', 'party2', 'group2', 'keyterms']
        
        gui.progress.value = 0
        return df_extended[columns]

    def display_document_key_terms(corpus, method='TextRank', document_id=None, normalize='lemma', n_keyterms=10):
        gui.output.clear_output()
        with gui.output:
            df = get_document_key_terms(corpus, method, document_id, normalize, n_keyterms)
            if len(df) == 1:
                print(df.iloc[0]['keyterms'])
            else:
                display(df)
            
    itw = widgets.interactive(
        display_document_key_terms,
        corpus=widgets.fixed(corpus),
        method=gui.method,
        document_id=gui.document_id,
        normalize=gui.normalize,
        n_keyterms=gui.n_keyterms,
    )

    display(widgets.VBox([
        gui.progress,
        widgets.HBox([gui.document_id, gui.left, gui.right, gui.method, gui.normalize, gui.n_keyterms]),
        gui.output
    ]))
    
    def back_handler(*args):
        if gui.position == 0:
            return
        gui.output.clear_output()
        gui.position = (gui.position - 1) % len(document_options)
        gui.document_id.value = document_options[gui.position][1]
        #itw.update()
        
    def forward_handler(*args):
        gui.output.clear_output()
        gui.position = (gui.position + 1) % len(document_options)
        gui.document_id.value = document_options[gui.position][1]
    
    gui.left.on_click(back_handler)
    gui.right.on_click(forward_handler)

    itw.update()

display_document_key_terms_gui(current_corpus(), current_wti_index())



# %% [markdown]
# ## <span style='color: green'>PREPARE/DESCRIBE </span> Clean Up the Text <span style='float: right; color: green'>TRY IT</span>

# %% code_folding=[]
import common.gui.utility
from common.corpus import corpus_utility

# %config InlineBackend.print_figure_kwargs = {'bbox_inches':'tight'}

def plot_xy_data(data, title='', xlabel='', ylabel='', **kwargs):
    x, y = list(data[0]), list(data[1])
    labels = x
    plt.figure(figsize=(8, 9 / 1.618))
    plt.plot(x, y, 'ro', **kwargs)
    plt.xticks(x, labels, rotation='75')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def display_cleaned_up_text(container, gui, display_type, treaty_id, gpe_substitutions, word_count_scores, word_document_count_scores, **opts): 
       
    try:
        def get_merged_words(scores, low, high):
            ws = set([])
            for x, wl in scores.items():
                if low <= x <= high:
                    ws.update(wl)
            return ws

        corpus = container.textacy_corpus
                
        doc = corpus_utility.get_treaty_doc(corpus, treaty_id)

        if doc is None:
            return
        
        gui.output_text.clear_output()
        gui.output_statistics.clear_output()

        if display_type.startswith('source_text'):

            source_files = {
                'source_text_raw': { 'filename': container.source_path, 'description': 'Raw text from PDF: Automatic text extraction using pdfminer Python package. ' },
                'source_text_edited': { 'filename': container.source_path, 'description': 'Manually edited text: List of references, index, notes and page headers etc. removed.' },
                'source_text_preprocessed': { 'filename': container.prepped_source_path, 'description': 'Preprocessed text: Normalized whitespaces. Unicode fixes. Urls, emails and phonenumbers removed. Accents removed.' }
            }        

            source_filename = source_files[display_type]['filename']
            description =  source_files[display_type]['description']
            text = utility.zip_get_text(source_filename, doc.metadata['filename'])

            with gui.output_text:
                print('[ ' + description.upper() + ' ]')
                print(text)
            return

        with gui.output_text:

            normalize = opts['normalize'] or 'orth'
            
            extra_stop_words = set([])            
            
            if opts['min_freq'] > 1:
                extra_stop_words.update(get_merged_words(word_count_scores[normalize], 1, opts['min_freq']))
            
            if opts['max_doc_freq'] < 100:                
                extra_stop_words.update(get_merged_words(word_document_count_scores[normalize], opts['max_doc_freq'], 100))            
            
            extract_args = dict(
                args=dict(
                    ngrams=opts['ngrams'],
                    named_entities=opts['named_entities'],
                    normalize=opts['normalize'],
                    as_strings=True
                ),
                kwargs=dict(
                    min_freq=opts['min_freq'],
                    include_pos=opts['include_pos'],
                    filter_stops=opts['filter_stops'],
                    filter_punct=opts['filter_punct']
                ),
                min_length=opts['min_word'],
                extra_stop_words=extra_stop_words,
                substitutions=(gpe_substitutions if opts['mask_gpe'] else None),
            )
            
            terms = [ x for x in corpus_utility.extract_document_terms(doc, extract_args)]

            if len(terms) == 0:
                print("No text. Please change selection.")

        if display_type in ['sanitized_text', 'statistics' ]:
                
            if display_type == 'sanitized_text':
                with gui.output_text:
                    print(' '.join([ t.replace(' ', '_') for t in terms ]))
                    return

            if display_type == 'statistics':

                wf = nltk.FreqDist(terms)

                with gui.output_text:

                    df = pd.DataFrame(wf.most_common(25), columns=['token','count'])
                    print('Token count: {} Vocab count: {}'.format(wf.N(), wf.B()))
                    display(df)

                with gui.output_statistics:

                    data = list(zip(*wf.most_common(25)))
                    plot_xy_data(data, title='Word distribution', xlabel='Word', ylabel='Word count')

                    wf = nltk.FreqDist([len(x) for x in terms])
                    data = list(zip(*wf.most_common(25)))
                    plot_xy_data(data, title='Word length distribution', xlabel='Word length', ylabel='Word count')

    except Exception as ex:
        with gui.output_text:
            logger.error(ex)
            
def display_cleanup_text_gui(container, wti_index):
    
    lw = lambda width: widgets.Layout(width=width)
    
    logger.info('Preparing corpus statistics...')
    
    corpus = container.textacy_corpus
    
    document_options = get_treaty_dropdown_options(wti_index, corpus)
        
    logger.info('...loading term substitution mappings...')
    gpe_filename = os.path.join(config.DATA_FOLDER, 'gpe_substitutions.txt')
    gpe_substitutions = { x: '_gpe_' for x in corpus_utility.get_gpe_names(filename=gpe_filename, corpus=corpus) }

    #pos_options = [ x for x in DF_TAGSET.POS.unique() if x not in ['PUNCT', '', 'DET', 'X', 'SPACE', 'PART', 'CONJ', 'SYM', 'INTJ', 'PRON']]  # groupby(['POS'])['DESCRIPTION'].apply(list).apply(lambda x: ', '.join(x)).to_dict()
    pos_tags = DF_TAGSET.groupby(['POS'])['DESCRIPTION'].apply(list).apply(lambda x: ', '.join(x[:1])).to_dict()
    pos_options = [('(All)', None)] + sorted([(k + ' (' + v + ')', k) for k,v in pos_tags.items() ])
    display_options = {
        'Source text (raw)': 'source_text_raw',
        'Source text (edited)': 'source_text_edited',
        'Source text (processed)': 'source_text_preprocessed',
        'Sanitized text': 'sanitized_text',
        'Statistics': 'statistics'
    }
    ngrams_options = { '1': [1], '1,2': [1,2], '1,2,3': [1,2,3]}
    gui = types.SimpleNamespace(
        position=1,
        document_id=widgets.Dropdown(description='Treaty', options=document_options, value=document_options[1][1], layout=lw('400px')),
        progress=widgets.IntProgress(value=0, min=0, max=5, step=1, description='', layout=widgets.Layout(width='90%')),
        min_freq=widgets.IntSlider(description='Min word freq', min=0, max=10, value=2, step=1, layout=widgets.Layout(width='400px')),
        max_doc_freq=widgets.IntSlider(description='Max doc. %', min=0, max=100, value=100, step=1, layout=widgets.Layout(width='400px')),
        mask_gpe=widgets.ToggleButton(value=False, description='Mask GPE',  tooltip='Replace geographical entites with `_gpe_`', icon='check'),
        ngrams=widgets.Dropdown(description='n-grams', options=ngrams_options, value=[1], layout=widgets.Layout(width='180px')),
        min_word=widgets.Dropdown(description='Min length', options=[1,2,3,4], value=1, layout=widgets.Layout(width='180px')),
        normalize=widgets.Dropdown(description='Normalize', options=[ None, 'lemma', 'lower' ], value='lower', layout=widgets.Layout(width='180px')),
        filter_stops=widgets.ToggleButton(value=False, description='Filter stops',  tooltip='Filter out stopwords', icon='check'),
        filter_punct=widgets.ToggleButton(value=False, description='Filter punct',  tooltip='Filter out punctuations', icon='check'),
        named_entities=widgets.ToggleButton(value=False, description='Merge entities',  tooltip='Merge entities', icon='check'),
        include_pos=widgets.SelectMultiple(description='POS', options=pos_options, value=list(), rows=10, layout=widgets.Layout(width='400px')),
        display_type=widgets.Dropdown(description='Show', value='statistics', options=display_options, layout=widgets.Layout(width='180px')),
        left=widgets.Button(description='<', button_style='Success', layout=lw('40px')),
        right=widgets.Button(description='>', button_style='Success', layout=lw('40px')),
        output_text=widgets.Output(layout={'height': '500px'}),
        output_statistics = widgets.Output(),
        boxes=None
    )
    
    logger.info('...word counts...')
    word_count_scores = dict(
        lemma=corpus_utility.generate_word_count_score(corpus, 'lemma', gui.min_freq.max),
        lower=corpus_utility.generate_word_count_score(corpus, 'lower', gui.min_freq.max),
        orth=corpus_utility.generate_word_count_score(corpus, 'orth', gui.min_freq.max)
    )
    logger.info('...word document count...')
    word_document_count_scores = dict(
        lemma=corpus_utility.generate_word_document_count_score(corpus, 'lemma', gui.max_doc_freq.min),
        lower=corpus_utility.generate_word_document_count_score(corpus, 'lower', gui.max_doc_freq.min),
        orth=corpus_utility.generate_word_document_count_score(corpus, 'orth', gui.max_doc_freq.min)
    )

    logger.info('...done!')
    opts = dict(
        min_freq=gui.min_freq,
        max_doc_freq=gui.max_doc_freq,
        mask_gpe=gui.mask_gpe,
        ngrams=gui.ngrams,
        min_word=gui.min_word,
        normalize=gui.normalize,
        filter_stops=gui.filter_stops,
        filter_punct=gui.filter_punct,
        named_entities=gui.named_entities,
        include_pos=gui.include_pos
    )
    uix = widgets.interactive(
        display_cleaned_up_text,
        container=widgets.fixed(container),
        gui=widgets.fixed(gui),
        display_type=gui.display_type,
        treaty_id=gui.document_id,
        gpe_substitutions=widgets.fixed(gpe_substitutions),
        word_count_scores=widgets.fixed(word_count_scores),
        word_document_count_scores=widgets.fixed(word_document_count_scores),
        **opts
    )
    
    gui.boxes = widgets.VBox([
        gui.progress,
        widgets.HBox([
            widgets.VBox([
                widgets.HBox([gui.document_id]),
                widgets.HBox([gui.left, gui.right], layout=widgets.Layout(align_items='flex-end')),
                widgets.HBox([gui.display_type, gui.normalize]),
                widgets.HBox([gui.ngrams, gui.min_word]),
                gui.min_freq,
                gui.max_doc_freq
            ]),
            widgets.VBox([
                gui.include_pos
            ]),
            widgets.VBox([
                gui.filter_stops,
                gui.mask_gpe,
                gui.filter_punct,
                gui.named_entities
            ])
        ]),
        widgets.HBox([
            gui.output_text, gui.output_statistics
        ]),
        uix.children[-1]
    ])
    
    display(gui.boxes)
    
    def back_handler(*args):
        if gui.position == 0:
            return
        #gui.output.clear_output()
        gui.position = (gui.position - 1) % len(document_options)
        gui.document_id.value = document_options[gui.position][1]
        #itw.update()
        
    def forward_handler(*args):
        #gui.output.clear_output()
        gui.position = (gui.position + 1) % len(document_options)
        gui.document_id.value = document_options[gui.position][1]
    
    gui.left.on_click(back_handler)
    gui.right.on_click(forward_handler)
    
    uix.update()
    return gui, uix

try:
    xgui, xuix = display_cleanup_text_gui(current_corpus_container(), treaty_state.current_wti_index())
except Exception as ex:
    raise
    logger.error(ex)


# %% [markdown]
# ### <span style='color: green;'>DESCRIBE</span> Most Discriminating Terms<span style='color: blue; float: right'>OPTIONAL</span>
# References
# King, Gary, Patrick Lam, and Margaret Roberts. “Computer-Assisted Keyword and Document Set Discovery from Unstructured Text.” (2014). http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.1445&rep=rep1&type=pdf.
#
# Displays the *most discriminating words* between two sets of treaties. Each treaty group can be filtered by country and period (signed year). In this way, the same group of countries can be studied for different time periods, or different groups of countries can be studied for the same time period. If "Closed region" is checked then **both** parties must be to the selected set of countries, from each region. In this way, one can for instance compare treaties signed between countries within the WTI group "Communists", against treaties signed within "Western Europe". 
#
# <b>#terms</b> The number of most discriminating terms to return for each group.<br>
# <b>#top</b> Only terms with a frequency within the top #top terms out of all terms<br>
# <b>Closed region</b> If checked, then <u>both</u> treaty parties must be within selected region

# %%
from common.gui import most_discriminating_terms_gui
try:
    most_discriminating_terms_gui.display_gui(current_wti_index(), current_corpus())
except Exception as ex:
    logger.error(ex)

# %% [markdown]
# ## <span style='color: green;'>DESCRIBE</span> Corpus Statistics<span style='color: blue; float: right'>OPTIONAL</span>

# %% [markdown]
# ### <span style='color: green;'>DESCRIBE</span> List of Most Frequent Words<span style='color: blue; float: right'>OPTIONAL</span>

# %%
import spacy

def compute_list_of_most_frequent_words(
    corpus,
    gui,
    group_by_column='signed_year',
    parties=None,
    target='lemma',
    weighting='count',
    include_pos=None,
    stop_words=None,
    display_score=False
):
    stop_words = stop_words or set()
    
    def include(token):
        flag = True
        if not include_pos is None:
             flag = flag and token.pos_ in include_pos
        flag = flag and token.lemma_ not in stop_words
        return flag
    
    gui.progress.max = len(corpus)
    
    df_freqs = pd.DataFrame({ 'treaty_id': [], 'signed_year': [], 'token': [], 'score': [] })
    
    parties_set = set(parties or [])
    
    docs = corpus if len(parties_set) == 0 \
        else ( x for x in corpus if len(set((x.metadata['party1'], x.metadata['party2'])) & parties_set) > 0 )
                                                   
    for doc in docs:
        
        doc_freqs = corpus_utility.textacy_doc_to_bow(doc, target=target, weighting=weighting, as_strings=True, include=include)
        
        df = pd.DataFrame({
            'treaty_id': doc.metadata['treaty_id'],
            'signed_year': int(doc.metadata['signed_year']),
            'token': list(doc_freqs.keys()),
            'score': list(doc_freqs.values())
        })
        
        df_freqs = pd.concat([df_freqs, df], ignore_index=True)
        gui.progress.value = gui.progress.value + 1
        
    df_freqs['signed_year'] = df_freqs.signed_year.astype(int)
    
    for key, group in TREATY_TIME_GROUPINGS.items():
        if key in df_freqs.columns:
            continue
        df_freqs[key] = (group['fx'])(df_freqs)
        
    df_freqs['term'] = df_freqs.token # if True else df_freqs.token
    
    df_freqs = df_freqs.groupby([group_by_column, 'term']).sum().reset_index()[[group_by_column, 'term', 'score']]
    
    if display_score is True:
        df_freqs['term'] = df_freqs.term + '*' + (df_freqs.score.apply('{:,.3f}'.format) if weighting == 'freq' else df_freqs.score.astype(str))
        
    df_freqs['position'] = df_freqs.sort_values(by=[group_by_column, 'score'], ascending=False).groupby([group_by_column]).cumcount() + 1
    
    gui.progress.value = 0
    
    return df_freqs
    
def display_list_of_most_frequent_words(gui, df):
    if gui.output_type.value == 'table':
        display(df)
    elif gui.output_type.value == 'rank':
        group_by_column = gui.group_by_column.value
        df = df[df.position <= gui.n_tokens.value]
        df_unstacked_freqs = df[[group_by_column, 'position', 'term']].set_index([group_by_column, 'position']).unstack()
        display(df_unstacked_freqs)
    else:
        filename = '../data/word_trend_data.xlsx'
        df.to_excel(filename)
        print('Excel written: ' + filename)
        
def word_frequency_gui(wti_index, corpus, compute_callback, display_callback):
    
    lw = lambda w: widgets.Layout(width=w)
    
    include_pos_tags = [ 'ADJ', 'VERB', 'NUM', 'ADV', 'NOUN', 'PROPN' ]
    weighting_options = { 'Count': 'count', 'Frequency': 'freq' }
    normalize_options = { '':  False, 'Lemma': 'lemma', 'Lower': 'lower' }
    #pos_tags = DF_TAGSET[DF_TAGSET.POS.isin(include_pos_tags)].groupby(['POS'])['DESCRIPTION'].apply(list).apply(lambda x: ', '.join(x[:1])).to_dict()
    #pos_options = { k + ' (' + v + ')': k for k,v in pos_tags.items() }
    pos_options = include_pos_tags
    
    counter = collections.Counter(corpus.word_freqs(normalize='lemma', weighting='count', as_strings=True))
    default_include_pos = ['NOUN', 'PROPN']
    frequent_words = [ x[0] for x in corpus_utility.get_most_frequent_words(corpus, 100, include_pos=default_include_pos) ]

    group_by_options = { TREATY_TIME_GROUPINGS[k]['title']: k for k in TREATY_TIME_GROUPINGS }
    output_type_options = [ ( 'List', 'table' ), ( 'Rank', 'rank' ), ( 'Excel', 'excel' ), ]
    ngrams_options = { '-': None, '1': [1], '1,2': [1,2], '1,2,3': [1,2,3]}
    party_preset_options = wti_index.get_party_preset_options()
    parties_options = [ x for x in wti_index.get_countries_list() if x != 'ALL OTHER' ]
    gui = types.SimpleNamespace(
        progress=widgets.IntProgress(value=0, min=0, max=5, step=1, description='', layout=lw('98%')),
        parties=widgets.SelectMultiple(description='Parties', options=parties_options, value=[], rows=7, layout=lw('200px')),
        party_preset=widgets_config.dropdown('Presets', party_preset_options, None, layout=lw('200px')),
        ngrams=widgets.Dropdown(description='n-grams', options=ngrams_options, value=None, layout=lw('200px')),
        min_word=widgets.Dropdown(description='Min length', options=[1,2,3,4], value=1, layout=lw('200px')),
        normalize=widgets.Dropdown(description='Normalize', options=normalize_options, value='lemma', layout=lw('200px')),
        weighting=widgets.Dropdown(description='Weighting', options=weighting_options, value='freq', layout=lw('200px')),
        include_pos=widgets.SelectMultiple(description='POS', options=pos_options, value=default_include_pos, rows=7, layout=lw('150px')),
        stop_words=widgets.SelectMultiple(description='STOP', options=frequent_words, value=list([]), rows=7, layout=lw('200px')),
        group_by_column=widgets.Dropdown(description='Group by', value='signed_year', options=group_by_options, layout=lw('200px')),
        output_type=widgets.Dropdown(description='Output', value='rank', options=output_type_options, layout=lw('200px')),
        n_tokens=widgets.IntSlider(description='#tokens', value=25, min=3, max=500, layout=lw('250px')),
        compute=widgets.Button(description='Compute', button_style='Success', layout=lw('120px')),
        display_score=widgets.ToggleButton(description='Display score', icon='check', value=False, layout=lw('120px')),
        output=widgets.Output(layout={'border': '1px solid black'})
    )
    
    boxes = widgets.VBox([
        gui.progress,
        widgets.HBox([
            widgets.VBox([
                gui.normalize,
                gui.ngrams,
                gui.weighting,
                gui.group_by_column,
                gui.output_type,
            ]),
            widgets.VBox([
                gui.parties,
                gui.party_preset,
            ]),
            gui.include_pos,
            gui.stop_words,
            widgets.VBox([
                gui.n_tokens,
                gui.display_score,
                gui.compute,
            ], layout=widgets.Layout(align_items='flex-end')),
        ]),
        gui.output
    ])
    
    display(boxes)
    
    def on_party_preset_change(change):  # pylint: disable=W0613
        if gui.party_preset.value is None:
            return
        gui.parties.value = gui.parties.options if 'ALL' in gui.party_preset.value else gui.party_preset.value
            
    gui.party_preset.observe(on_party_preset_change, names='value')
    
    def pos_change_handler(*args):
        with gui.output:
            gui.compute.disabled = True
            selected = set(gui.stop_words.value)
            frequent_words = [
                x[0] for x in corpus_utility.get_most_frequent_words(
                    corpus,
                    100,
                    normalize=gui.normalize.value,
                    include_pos=gui.include_pos.value,
                    weighting=gui.weighting.value
                )
            ]
            gui.stop_words.options = frequent_words
            selected = selected & set(gui.stop_words.options)
            gui.stop_words.value = list(selected)
            gui.compute.disabled = False
        
    gui.include_pos.observe(pos_change_handler, 'value')    
    gui.weighting.observe(pos_change_handler, 'value')    
    
    def compute_callback_handler(*_args):
        gui.output.clear_output()
        with gui.output:
            try:
                gui.compute.disabled = True
                df_freqs = compute_callback(
                    corpus=corpus,
                    gui=gui,
                    target=gui.normalize.value,
                    group_by_column=gui.group_by_column.value,
                    parties=gui.parties.value,
                    weighting=gui.weighting.value,
                    include_pos=gui.include_pos.value,
                    stop_words=set(gui.stop_words.value),
                    display_score=gui.display_score.value
                )
                display_callback(gui, df_freqs)
            finally:
                gui.compute.disabled = False

    gui.compute.on_click(compute_callback_handler)
    return gui
                
try:
    word_frequency_gui(
        current_wti_index(),
        current_corpus(),
        compute_callback=compute_list_of_most_frequent_words,
        display_callback=display_list_of_most_frequent_words
    )
except Exception as ex:
    logger.error(ex)
    


# %% [markdown]
# ### <span style='color: green;'>DESCRIBE</span> Corpus and Document Sizes<span style='color: blue; float: right'>OPTIONAL</span>

# %%



# %%
