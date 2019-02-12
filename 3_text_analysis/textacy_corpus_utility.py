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

from common.utility import deprecated

logger = utility.getLogger('corpus_text_analysis')

LANGUAGE_MODEL_MAP = { 'en': 'en_core_web_sm', 'fr': 'fr_core_news_sm', 'it': 'it_core_web_sm', 'de': 'de_core_web_sm' }
    
import itertools

def count_documents_by_pivot(corpus, attribute):
    ''' Return a list of document counts per group defined by attribute
    Assumes documents are sorted by attribute!
    '''
    fx_key = lambda doc: doc.metadata[attribute]
    return [ len(list(g)) for _, g in itertools.groupby(corpus, fx_key) ]

def generate_word_count_score(corpus, normalize, count):
    wc = corpus.word_freqs(normalize=normalize, weighting='count', as_strings=True)
    d = { i: set([]) for i in range(1, count+1)}
    for k, v in wc.items():
        if v <= count:
            d[v].add(k)
    return d

def generate_word_document_count_score(corpus, normalize, threshold=75):
    wc = corpus.word_doc_freqs(normalize=normalize, weighting='freq', smooth_idf=True, as_strings=True)
    d = { i: set([]) for i in range(threshold, 101)}
    for k, v in wc.items():
        slot = int(round(v,2)*100)
        if slot >= threshold:
            d[slot].add(k)
    return d

class CorpusContainer():
    """Singleton class for current (last) computed or loaded corpus
    """
    corpus_container = None
    
    class CorpusNotLoaded(Exception):
        pass
    
    def __init__(self):
        self.language = None
        self.source_path = None
        self.prepped_source_path = None
        self.textacy_corpus_path = None
        self.textacy_corpus = None
        self.nlp = None
        self.word_count_scores = None
    
    def get_word_count(self, normalize):
        key = 'word_count_' + normalize
        self.word_count_scores = self.word_count_scores or { }
        if key not in self.word_count_scores:
            self.word_count_scores[key] = generate_word_count_score(self.textacy_corpus, normalize, 100)
        return self.word_count_scores[key]
    
    def get_word_document_count(self, normalize):
        key = 'word_document_count_' + normalize
        self.word_count_scores = self.word_count_scores or { }
        if key not in self.word_count_scores:
            self.word_count_scores[key] = generate_word_document_count_score(self.textacy_corpus, normalize, 75)
        return self.word_count_scores[key]
        
    @staticmethod
    def container():

        CorpusContainer.corpus_container = CorpusContainer.corpus_container or CorpusContainer()
        
        return CorpusContainer.corpus_container

    @staticmethod
    def corpus():
        
        class CorpusNotLoaded(Exception):
            pass
        
        if CorpusContainer.container().textacy_corpus is None:
            raise CorpusNotLoaded('Corpus not loaded or computed')
        
        return CorpusContainer.container().textacy_corpus

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
    
def get_gpe_names(filename, corpus=None):
    
    with open(filename) as f:
        content = f.readlines()
        
    lemmas = set([ x.strip() for x in content ])
    
    tokens = set([])
    
    if corpus is not None:
        for doc in corpus:
            candidates = set([ x.lower_ for x in doc if x.lemma_ in lemmas ])
            tokens = tokens.union(candidates)
        
    tokens = tokens.union(lemmas)
    
    return tokens
    

def infrequent_words(corpus, normalize='lemma', weighting='count', threshold=0, as_strings=False):
    
    '''Returns set of infrequent words i.e. words having total count less than given threshold'''
    
    if weighting == 'count' and threshold <= 1:
        return set([])
    
    word_counts = corpus.word_freqs(normalize=normalize, weighting=weighting, as_strings=as_strings)
    words = set([ w for w in word_counts if word_counts[w] < threshold ])
    
    return words

def frequent_document_words(corpus, normalize='lemma', weighting='freq', dfs_threshold=80, as_strings=True):
    '''Returns set of words that occurrs freuently in many documents, candidate stopwords'''
    document_freqs = corpus.word_doc_freqs(normalize=normalize, weighting=weighting, smooth_idf=True, as_strings=True)
    frequent_document_words = set([ w for w, f in document_freqs.items() if int(round(f,2)*100) >= dfs_threshold ])
    return frequent_document_words
   
def extract_document_terms(doc, extract_args):
    """ Extracts documents and terms from a corpus
    
    Parameters
    ----------
    corpus : textacy Corpus
        Corpus in textacy format.
        
    extract_args : dict
        Dict that contains args that specifies the filter and transforms
        extract_args['args'] positional arguments for textacy.Doc.to_terms_list
        extract_args['kwargs'] Keyword arguments for textacy.Doc.to_terms_list
        extract_args['substitutions'] Dict (map) with term substitution
        extract_args['extra_stop_words'] List of additional stopwords to use
        
    Returns
    -------
    iterable of documents (which is iterable of terms)
        Documents where terms have ben filtered and transformed according to args.
        
    """
    kwargs = extract_args.get('kwargs', {})
    args = extract_args.get('args', {})
    
    extra_stop_words = set(extract_args.get('extra_stop_words', None) or [])
    substitutions = extract_args.get('substitutions', None)    
    min_length = extract_args.get('min_length', 2)
    
    ngrams = args.get('ngrams', None)
    named_entities = args.get('named_entities', False)
    normalize = args.get('normalize', 'lemma')
    as_strings = args.get('as_strings', True)
 
    def tranform_token(w, substitutions=None):
        if '\n' in w:
            w = w.replace('\n', '_')
        if substitutions is not None and w in substitutions:
            w = substitutions[w]
        return w

    terms = ( z for z in (
        tranform_token(w, substitutions)
            for w in doc.to_terms_list(ngrams, named_entities, normalize, as_strings, **kwargs)
                if len(w) >= min_length # and w not in extra_stop_words
    ) if z not in extra_stop_words)
        
    return terms

def extract_corpus_terms(corpus, extract_args):
    
    """ Extracts documents and terms from a corpus
    
    Parameters
    ----------
    corpus : textacy Corpus
        Corpus in textacy format.
        
    extract_args : dict
        Dict that contains args that specifies the filter and transforms
        extract_args['args'] positional arguments for textacy.Doc.to_terms_list
        extract_args['kwargs'] Keyword arguments for textacy.Doc.to_terms_list
        extract_args['extra_stop_words'] List of additional stopwords to use
        extract_args['substitutions'] Dict (map) with term substitution
        DEPRECATED extract_args['mask_gpe'] Boolean flag indicating if GPE should be substituted
        extract_args['min_freq'] Integer value specifying min global word count.
        extract_args['max_doc_freq'] Float value between 0 and 1 indicating threshold
          for documentword frequency, Words that occur in more than `max_doc_freq`
          documents will be filtered out.
        
    None
    ----
        extract_args.min_freq and extract_args.min_freq is the same value but used differently
        kwargs.min_freq is passed directly as args to `textacy_doc.to_terms_list`
        tokens below extract_args.min_freq threshold are added to the `extra_stop_words` list
    Returns
    -------
    iterable of documents (which is iterable of terms)
        Documents where terms have ben filtered and transformed according to args.
        
    """
    
    kwargs = dict(extract_args.get('kwargs', {}))
    args = dict(extract_args.get('args', {}))
    normalize = args.get('normalize', 'lemma')    
    substitutions = extract_args.get('substitutions', {})
    extra_stop_words = set(extract_args.get('extra_stop_words', None) or [])
    chunk_size = extract_args.get('chunk_size', None)
    min_length = extract_args.get('min_length', 2)
    
    #mask_gpe = extract_args.get('mask_gpe', False)
    #if mask_gpe is True:
    #    gpe_names = { x: '_gpe_' for x in get_gpe_names(corpus) }
    #    substitutions = utility.extend(substitutions, gpe_names)
    
    min_freq = extract_args.get('min_freq', 1)
    
    if min_freq > 1:
        words = infrequent_words(corpus, normalize=normalize, weighting='count', threshold=min_freq, as_strings=True)
        extra_stop_words = extra_stop_words.union(words)
        logger.info('Ignoring {} low-frequent words!'.format(len(words)))

    max_doc_freq = extract_args.get('max_doc_freq', 100)
    
    if max_doc_freq < 100 :
        words = frequent_document_words(corpus, normalize=normalize, weighting='freq', dfs_threshold=max_doc_freq, as_strings=True)
        extra_stop_words = extra_stop_words.union(words)
        logger.info('Ignoring {} high-frequent words!'.format(len(words)))
    
    extract_args = {
        'args': args,
        'kwargs': kwargs,
        'substitutions': substitutions,
        'extra_stop_words': extra_stop_words,
        'chunk_size': None
    }
    
    terms = ( extract_document_terms(doc, extract_args) for doc in corpus )
        
    return terms

def get_treaty_doc(corpus, treaty_id):
    for doc in corpus.get(lambda x: x.metadata['treaty_id'] == treaty_id, limit=1):
        return doc
    return None

# FIXME: Move to 
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

def generate_corpus_filename(source_path, language, nlp_args=None, preprocess_args=None, compression='bz2', period_group=''):
    nlp_args = nlp_args or {}
    preprocess_args = preprocess_args or {}
    disabled_pipes = nlp_args.get('disable', ())
    suffix = '_{}_{}{}_{}'.format(
        language,
        '_'.join([ k for k in preprocess_args if preprocess_args[k] ]),
        '_disable({})'.format(','.join(disabled_pipes)) if len(disabled_pipes) > 0 else '',
        (period_group or '')
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
    model_name = LANGUAGE_MODEL_MAP[language]
    if not model_name.endswith('lg'):
        logger.warning('Selected model is not the largest availiable.')
    nlp = textacy.load_spacy(model_name, **nlp_args)
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
