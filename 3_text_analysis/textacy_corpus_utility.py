import re
import zipfile
import pandas as pd
import logging
import textacy
import logging
import collections

import spacy

from spacy.language import Language
from spacy import attrs

import common.utility as utility
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
    args=dict(
        ngrams=None,
        named_entities=True,
        normalize='lemma',
        as_strings=True
    ),
    kwargs=dict(
        filter_stops=True,
        filter_punct=True,
        filter_nums=True,
        min_freq=1,
        drop_determiners=True,
        include_pos=('NOUN', 'PROPN', )
    )
)

def textacy_filter_terms(doc, term_args, chunk_size=None, min_length=2):
    
    def fix_peculiarities(w):
        if '\n' in w:
            w = w.replace('\n', ' ')
        return w
    
    kwargs = utility.extend({}, DEFAULT_TERM_PARAMS['kwargs'], term_args['kwargs'])
    args = utility.extend({}, DEFAULT_TERM_PARAMS['args'], term_args['args'])
    extra_stop_words = set(term_args.get('extra_stop_words', None) or [])
    terms = (
        fix_peculiarities(x) for x in doc.to_terms_list(
            args['ngrams'],
            args['named_entities'],
            args['normalize'],
            args['as_strings'],
            **kwargs
        ) if len(x) >= min_length and x not in extra_stop_words
    )
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
    disabled_pipes = nlp_args.get('disable', ())
    suffix = '_{}_{}{}'.format(
        language,
        '_'.join([ k for k in preprocess_args if preprocess_args[k] ]),
        '_disable({})'.format(','.join(disabled_pipes)) if len(disabled_pipes) > 0 else ''
    )
    filename = utility.path_add_suffix(source_path, suffix, new_extension='.pkl')
    if compression is not None:
        filename += ('.' + compression)
    return filename

def keep_hyphen_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

    return spacy.tokenizer.Tokenizer(nlp.vocab, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer, token_match=None)

def setup_nlp_language_model(language, **nlp_args):
    
    if (len(nlp_args.get('disable', [])) == 0):
        nlp_args.pop('disable')
        
    def remove_whitespace_entities(doc):
        doc.ents = [ e for e in doc.ents if not e.text.isspace() ]
        return doc

    logger.info('Loading model: %s...', language)
    
    Language.factories['remove_whitespace_entities'] = lambda nlp, **cfg: remove_whitespace_entities
    
    nlp = textacy.load_spacy(LANGUAGE_MODEL_MAP[language], **nlp_args)
    nlp.tokenizer = keep_hyphen_tokenizer(nlp)
    
    pipeline = lambda: [ x[0] for x in nlp.pipeline ]
    
    logger.info('Using pipeline: ' + ' '.join(pipeline()))
    
    return nlp

def propagate_document_attributes(corpus):
    for doc in corpus:
        doc.spacy_doc.user_data['title'] = doc.metadata['treaty_id']
        doc.spacy_doc.user_data['treaty_id'] = doc.metadata['treaty_id']

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

def textacy_doc_to_bow(doc, target='lemma', weighting='count', as_strings=False, include=None):

    spacy_doc = doc.spacy_doc
    
    weighing_keys = { 'count', 'freq' }
    target_keys = { 'lemma': attrs.LEMMA, 'lower': attrs.LOWER, 'orth': attrs.ORTH }
    
    default_exclude = lambda x: x.is_stop or x.is_punct or x.is_space
    exclude = default_exclude if include is None else lambda x: default_exclude(x) or not include(x)
    
    assert weighting in weighing_keys
    assert target in target_keys

    target_weights = spacy_doc.count_by(target_keys[target], exclude=exclude)
    
    if weighting == 'freq':
        n_tokens = sum(target_weights.values())
        target_weights = { id_: weight / n_tokens for id_, weight in target_weights.items() }

    if as_strings:
        bow = { doc.spacy_stringstore[word_id]: count for word_id, count in target_weights.items() }
    else:
        bow = { word_id: count for word_id, count in target_weights.items() }
        
    return bow

def get_most_frequent_words(corpus, n_top, normalize='lemma', include_pos=None, weighting='count'):
    include_pos = include_pos or [ 'VERB', 'NOUN', 'PROPN' ]
    include = lambda x: x.pos_ in include_pos
    word_counts = collections.Counter()
    for doc in corpus:
        bow = textacy_doc_to_bow(doc, target=normalize, weighting=weighting, as_strings=True, include=include)
        word_counts.update(bow)
    return word_counts.most_common(n_top)

def store_tokens_to_file(corpus, filename):
    import itertools
    doc_tokens = lambda d: (
        dict(
            i=t.i,
            token=t.lower_,
            lemma=t.lemma_,
            pos=t.pos_,
            signed_year=d.metadata['signed_year'],
            treaty_id=d.metadata['treaty_id']
        ) for t in d )
    tokens = pd.DataFrame(list(itertools.chain.from_iterable(doc_tokens(d) for d in corpus )))
    
    if filename.endswith('.xlxs'):
        tokens.to_excel(filename)
    else:
        tokens['token'] = tokens.token.str.replace('\t', ' ')
        tokens['token'] = tokens.token.str.replace('\n', ' ')
        tokens['token'] = tokens.token.str.replace('"', ' ')
        tokens['lemma'] = tokens.token.str.replace('\t', ' ')
        tokens['lemma'] = tokens.token.str.replace('\n', ' ')
        tokens['lemma'] = tokens.token.str.replace('"', ' ')
        tokens.to_csv(filename, sep='\t')
    
