import collections
import os
import types

import ipywidgets as widgets
import nltk
import pandas as pd
from IPython.display import display
from loguru import logger
from matplotlib import pyplot as plt

from common import config, utility
from common.corpus import textacy_corpus_utility as textacy_utility
from common.gui import gui_utility

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
                
        doc = textacy_utility.get_document_by_id(corpus, treaty_id)

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
            text = utility.zip_get_text(source_filename, doc._.meta['filename'])

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
            
            terms = [ x for x in textacy_utility.extract_document_terms(doc, extract_args)]

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
    
    document_options = gui_utility.get_treaty_dropdown_options(wti_index, corpus)
        
    logger.info('...loading term substitution mappings...')
    gpe_filename = os.path.join(config.DATA_FOLDER, 'gpe_substitutions.txt')
    gpe_substitutions = textacy_utility.load_term_substitutions(filepath=gpe_filename)

    #pos_options = [ x for x in DF_TAGSET.POS.unique() if x not in ['PUNCT', '', 'DET', 'X', 'SPACE', 'PART', 'CONJ', 'SYM', 'INTJ', 'PRON']]  # groupby(['POS'])['DESCRIPTION'].apply(list).apply(lambda x: ', '.join(x)).to_dict()
    pos_tags = config.get_tag_set().groupby(['POS'])['DESCRIPTION'].apply(list).apply(lambda x: ', '.join(x[:1])).to_dict()
    pos_options = [('(All)', None)] + sorted([(k + ' (' + v + ')', k) for k,v in pos_tags.items() ])
    display_options: dict[str, str] = {
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
    word_count_scores: dict[str, dict[int, set[str]]] = dict(
        lemma=textacy_utility.generate_word_count_score(corpus, 'lemma', gui.min_freq.max),
        lower=textacy_utility.generate_word_count_score(corpus, 'lower', gui.min_freq.max),
        orth=textacy_utility.generate_word_count_score(corpus, 'orth', gui.min_freq.max)
    )
    logger.info('...word document count...')
    word_document_count_scores: dict[str, dict[int, set[str]]] = dict(
        lemma=textacy_utility.generate_word_document_count_score(corpus, 'lemma', gui.max_doc_freq.min),
        lower=textacy_utility.generate_word_document_count_score(corpus, 'lower', gui.max_doc_freq.min),
        orth=textacy_utility.generate_word_document_count_score(corpus, 'orth', gui.max_doc_freq.min)
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
