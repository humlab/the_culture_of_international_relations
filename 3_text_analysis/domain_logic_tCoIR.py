
import os
import common.utility as utility
import pandas as pd
import textacy_corpus_utility
import text_corpus
import itertools
import re
import fnmatch

# import domain.tCoIR.treaty_state as treaty_repository
import common.treaty_state as treaty_repository
import textacy

# FIXME VARYING ASPECTS

logger = utility.getLogger('corpus_text_analysis')

DATA_FOLDER = '../data'

CORPUS_NAME_PATTERN = 'tCoIR_*.txt.zip'
CORPUS_TEXT_FILES_PATTERN = '*.txt'

WTI_INDEX_FOLDER = DATA_FOLDER # os.path.join(DATA_FOLDER, 'wti_index')

DOCUMENT_FILTERS = [
#         {
#             'type': 'multiselect',
#             'description': 'Party #1',
#             'field': 'party1'
#             # FIXME: Not implemented:
#             # 'filter_query': '(party1=={0}) | (party2=={0})'
#         },
#         {
#             'type': 'multiselect',
#             'description': 'Party #2',
#             'field': 'party2'
#             # FIXME: Not implemented:
#             # 'filter_query': '(party1=={0}) | (party2=={0})'
#         },
#         {
#             'type': 'multiselect',
#             'description': 'Topic',
#             'field': 'topic'
#         },
#         {
#             'type': 'multiselect',
#             'description': 'Year',
#             'field': 'signed_year',
#             'query': 'signed_year > 0'
#         }
]


GROUP_BY_OPTIONS = [
    ('Year', ['signed_year']),
    ('Party1', ['party1']),
    ('Party1, Year', ['party1', 'signed_year']),
    ('Party2, Year', ['party2', 'signed_year']),
    ('Group1, Year', ['group1', 'signed_year']),
    ('Group2, Year', ['group2', 'signed_year']),
]
    #group_by_options = { 'Year': 'year', 'Pope': 'pope', 'Genre': 'genre' } #TREATY_TIME_GROUPINGS[k]['title']: k for k in TREATY_TIME_GROUPINGS }

WTI_INDEX = None

def get_wti_index():
    global WTI_INDEX
    if WTI_INDEX is None:
        WTI_INDEX = treaty_repository.load_wti_index(WTI_INDEX_FOLDER)
    return WTI_INDEX

def get_parties():
    parties = get_wti_index().get_parties()
    return parties

def get_treaties(lang='en', period_group='years_1945-1972'): # , treaty_filter='is_cultural', parties=None)

    columns = [ 'party1', 'party2', 'topic', 'topic1', 'signed_year']
    treaties = get_wti_index().get_treaties(language=lang, period_group=period_group)[columns]

    group_map = get_parties()['group_name'].to_dict()
    treaties['group1'] = treaties['party1'].map(group_map)
    treaties['group2'] = treaties['party2'].map(group_map)

    return treaties

def get_extended_treaties(lang='en'):
    treaties = get_treaties(lang=lang)
    return treaties

def get_corpus_documents(corpus):
    metadata = [ utility.extend({}, doc._.meta, _get_pos_statistics(doc)) for doc in corpus ]
    df = pd.DataFrame(metadata)[['treaty_id', 'filename', 'signed_year', 'party1', 'party2', 'topic1', 'is_cultural'] + POS_NAMES]
    df['title'] = df.treaty_id
    df['lang'] = df.filename.str.extract(r'\w{4,6}\_(\w\w)')
    df['words'] = df[POS_NAMES].apply(sum, axis=1)
    return df

def get_treaty_dropdown_options(wti_index, corpus):

    def format_treaty_name(x):

        return '{}: {} {} {} {}'.format(x.name, x['signed_year'], x['topic'], x['party1'], x['party2'])

    documents = wti_index.treaties.loc[textacy_utility.get_corpus_documents(corpus).treaty_id]

    options = [ (v, k) for k, v in documents.apply(format_treaty_name, axis=1).to_dict().items() ]
    options = sorted(options, key=lambda x: x[0])

    return options

def get_document_stream(source, lang, document_index=None, id_extractor=None):

    assert document_index is not None

    if 'document_id' not in document_index.columns:
        document_index['document_id'] = document_index.index

    id_extractor = lambda filename: filename.split('_')[0]
    lang_pattern = re.compile("^(\w*)\_" + lang + "([\_\-]corr)?\.txt$")
    item_filter  = lambda x: lang_pattern.match(x) # and id_extractor(x) in document_index.index
        
    if isinstance(source, str):
        print('Opening archive: {}'.format(source))
        reader = text_corpus.CompressedFileReader(source, pattern=lang_pattern, itemfilter=item_filter)
    else:
        reader = source

    id_map = {
        filename : id_extractor(filename) for filename in reader.filenames if item_filter(filename)
    }
    
    if len(set(document_index.index) - set(id_map.values())) > 0:
        logger.warning('Treaties not found in archive: ' +
                           ', '.join(list(set(document_index.index) - set(id_map.values()))))
            
    columns = ['signed_year', 'party1', 'party2']
    
    df = document_index[columns]
    
    for filename, text in reader:
        
        document_id = id_map.get(filename, None)
        
        if document_id not in df.index:
            continue
            
        metadata = df.loc[document_id].to_dict()
        
        metadata['filename'] = filename
        metadata['document_id'] = document_id
        metadata['treaty_id'] = document_id
        
        yield filename, document_id, text, metadata
    
def compile_documents_by_filename(filenames):

    treaties = get_treaties()
    treaty_map = {
        treaty_id: filename for (treaty_id, filename) in map(lambda x: (x.split('_')[0], x), filenames)
    }
    treaties = treaties[treaties.index.isin(treaty_map.keys())]
    treaties.index.rename('index',inplace=True)
    treaties['document_id'] = treaties.index
    treaties['treaty_id'] = treaties.index
    treaties['local_number'] = treaties.index
    treaties['year'] = treaties.signed_year
    treaties['filename'] = treaties.treaty_id.apply(lambda x: treaty_map[x])

    return treaties

def compile_documents(corpus, corpus_index=None):

    if len(corpus) == 0:
        return None

    if isinstance(corpus, textacy.corpus.Corpus):
        filenames = [ doc._.meta['filename'] for doc in corpus]
    else:
        filenames = corpus.filenames

    df = compile_documents_by_filename(filenames)

    return df

# FIXME VARYING ASPECTs: What attributes to extend
def add_domain_attributes(df, document_index):

    treaties = get_treaties()
    group_map = get_parties()['group_name'].to_dict()

    df_extended = pd.merge(df, treaties, left_index=True, right_index=True, how='inner')
    return df_extended

def load_corpus_index(source_name):
    return None

