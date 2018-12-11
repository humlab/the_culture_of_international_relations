
import zipfile
import pandas as pd
import common.utility as utility
import logging
import textacy
import logging
import collections

from spacy.language import Language

import treaty_corpus

logger = utility.getLogger('corpus_text_analysis')

LANGUAGE_MODEL_MAP = { 'en': 'en_core_web_sm', 'fr': 'fr_core_web_sm', 'it': 'it_core_web_sm', 'de': 'de_core_web_sm' }

def preprocess_text(source_filename, target_filename, tick=utility.noop):
    '''
    Pre-process of zipped archive that contains text documents
    
    Returns
    -------
    Zip-archive
    '''
    
    filenames = utility.zip_get_filenames(source_filename)
    texts = ( (filename, utility.zip_get_text(source_filename, filename)) for filename in filenames )
    logger.info('Preparing text corpus...')
    tick(0, len(filenames))
    with zipfile.ZipFile(target_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename, text in texts:
            text = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
            text = textacy.preprocess.normalize_whitespace(text)   
            text = textacy.preprocess.fix_bad_unicode(text)   
            text = textacy.preprocess.replace_currency_symbols(text)
            text = textacy.preprocess.unpack_contractions(text)
            text = textacy.preprocess.remove_accents(text)
            zf.writestr(filename, text)
            tick()
    tick(0)

DEFAULT_TERM_PARAMS = dict(
    args=dict(ngrams=1, named_entities=True, normalize='lemma', as_strings=True),
    kwargs=dict(filter_stops=True, filter_punct=True, filter_nums=True, min_freq=1, drop_determiners=True, include_pos=('NOUN', 'PROPN', ))
)

FIXED_STOPWORDS = []

def textacy_filter_terms(doc, term_args, chunk_size=None, min_length=2):
    
    kwargs = utility.extend({}, DEFAULT_TERM_PARAMS['kwargs'], term_args['kwargs'])
    args = utility.extend({}, DEFAULT_TERM_PARAMS['args'], term_args['args'])
    
    terms = (x for x in doc.to_terms_list(
        args['ngrams'],
        args['named_entities'],
        args['normalize'],
        args['as_strings'],
        **kwargs
    ) if len(x) >= min_length and x not in FIXED_STOPWORDS)
    return terms

def get_treaty_doc(corpus, treaty_id):
    for doc in corpus.get(lambda x: x.metadata['treaty_id'] == treaty_id, limit=1):
        return doc
    return None

def get_document_stream(corpus_path, lang, treaties):
    
    if 'treaty_id' not in treaties.columns:
        treaties['treaty_id'] = treaties.index
        
    documents = treaty_corpus.TreatyCompressedFileReader(corpus_path, lang, list(treaties.index))
    
    for treaty_id, language, filename, text in documents:
        assert language == lang
        metadata = treaties.loc[treaty_id]
        yield filename, text, metadata
        
def create_textacy_corpus(documents, nlp, tick=utility.noop):
    corpus = textacy.Corpus(nlp)
    for filename, text, metadata in documents:
        corpus.add_text(text, utility.extend(dict(filename=filename), metadata))
        tick()
    return corpus

def generate_corpus_filename(source_path, language, nlp_args=None, preprocess_args=None, compression='bz2'):
    nlp_args = nlp_args or {}
    preprocess_args = preprocess_args or {}
    disabled_pipes = nlp_args.get('disable', [])
    suffix = '_{}_{}{}'.format(
        language,
        '_'.join([ k for k in preprocess_args if preprocess_args[k] ]),
        '_disable({})'.format(','.join(disabled_pipes)) if len(disabled_pipes) > 0 else ''
    )
    filename = utility.path_add_suffix(source_path, suffix, new_extension='.pkl')
    if compression is not None:
        filename += ('.' + compression)
    return filename

def setup_nlp_language_model(language, **nlp_args):
    
    def remove_whitespace_entities(doc):
        doc.ents = [ e for e in doc.ents if not e.text.isspace() ]
        return doc

    logger.info('Loading model: %s...', language)
    
    Language.factories['remove_whitespace_entities'] = lambda nlp, **cfg: remove_whitespace_entities
    
    nlp = textacy.load_spacy(LANGUAGE_MODEL_MAP[language], **nlp_args)
    pipeline = lambda: [ x[0] for x in nlp.pipeline ]
    
    logger.info('Using pipeline: ' + ' '.join(pipeline()))
    
    return nlp

def propagate_document_attributes(corpus):
    for doc in corpus:
        doc.spacy_doc.user_data['title'] = doc.metadata['treaty_id']
        doc.spacy_doc.user_data['treaty_id'] = doc.metadata['treaty_id']
    
def get_corpus_documents(corpus):
    metadata = [ doc.metadata for doc in corpus ]
    df = pd.DataFrame(metadata)[['treaty_id', 'filename', 'signed_year', 'party1', 'party2', 'topic1', 'is_cultural']]
    df['title'] = df.treaty_id
    df['lang'] = df.filename.str.extract(r'\w{4,6}\_(\w\w)')  #.apply(lambda x: x.split('_')[1][:2])
    return df


POS_TO_COUNT = {
    'SYM': 0, 'PART': 0, 'ADV': 0, 'NOUN': 0, 'CCONJ': 0, 'ADJ': 0, 'DET': 0, 'ADP': 0, 'INTJ': 0, 'VERB': 0, 'NUM': 0, 'PRON': 0, 'PROPN': 0
}

POS_NAMES = list(sorted(POS_TO_COUNT.keys()))

def _get_pos_statistics(doc):
    pos_iter = ( x.pos_ for x in doc if x.pos not in [96, 0, 100] )
    pos_counts = dict(collections.Counter(pos_iter))
    stats = utility.extend(dict(POS_TO_COUNT), pos_counts)
    return stats

def get_corpus_documents(corpus):
    metadata = [ utility.extend({}, doc.metadata, _get_pos_statistics(doc)) for doc in corpus ]
    df = pd.DataFrame(metadata)[['treaty_id', 'filename', 'signed_year', 'party1', 'party2', 'topic1', 'is_cultural']+POS_NAMES]
    df['title'] = df.treaty_id
    df['lang'] = df.filename.str.extract(r'\w{4,6}\_(\w\w)')  #.apply(lambda x: x.split('_')[1][:2])
    df['words'] = df[POS_NAMES].apply(sum, axis=1)
    return df
