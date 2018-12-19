import os
import pandas as pd
import glob
import nltk
import gensim
import zipfile
import fnmatch
import logging
import re
import typing.re

from gensim.corpora.textcorpus import TextCorpus

logger = logging.getLogger(__name__)

HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)
TREATY_FILENAME = re.compile(r"^([a-zA-Z0-9]*)\_(en|fr|de|it).*\.txt$")

#@deprecated('Moved to TreatyCompressedFileReader.language_filename_pattern')
def language_filename_pattern(language):
    return re.compile("^(\w*)\_" + language + "([\_\-]corr)?\.txt$") 
    
def dehyphen(text):
    result = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
    return result

def list_treaty_archive_files(archivename, pattern):
    px = lambda x: pattern.match(x) if isinstance(pattern, typing.re.Pattern) else fnmatch.fnmatch(x, pattern)
    with zipfile.ZipFile(archivename) as zf:
        return [ name for name in zf.namelist() if px(name) ]
        
#@deprecated('Moved to TreatyCompressedFileReader.get_treaty_filename_lookups')
def get_treaty_filename_lookup(archivename, language):
    pattern = language_filename_pattern(language)
    filenames = list_treaty_archive_files(archivename, pattern)
    treaty_lookup = { x.split('_')[0]: x for x in filenames }
    return treaty_lookup
    
class CompressedFileReader:
    
    def __init__(self, path, pattern='*.txt', itemfilter=None):
        self.path = path
        self.filename_pattern = pattern
        self.archive_filenames = list_treaty_archive_files(path, pattern)
        filenames = None
        if itemfilter is not None:
            if isinstance(itemfilter, list):
                filenames = [ x for x in itemfilter if x in self.archive_filenames ]
            elif callable(itemfilter):
                filenames = [ x for x in self.archive_filenames if itemfilter(self.archive_filenames, x) ]
            else:
                assert False
        self.filenames = filenames or self.archive_filenames
        self.iterator = None

    def __iter__(self):
        self.iterator = None
        return self
    
    def __next__(self):
        if self.iterator is None:
            self.iterator = self.get_iterator()
        return next(self.iterator)
    
    def get_file(self, filename):
        
        if filename not in self.filenames:
            yield  os.path.basename(filename), None
            
        with zipfile.ZipFile(self.path) as zip_file:
            yield os.path.basename(filename), self._read_content(zip_file, filename)
                    
    def get_iterator(self):
        with zipfile.ZipFile(self.path) as zip_file:
            for filename in self.filenames:
                yield os.path.basename(filename), self._read_content(zip_file, filename)
                    
    def _read_content(self, zip_file, filename):
        with zip_file.open(filename, 'rU') as text_file:
            content = text_file.read()
            content = gensim.utils.to_unicode(content, 'utf8', errors='ignore')
            content = dehyphen(content)
            return content
        
class TreatyCompressedFileReader(CompressedFileReader):

    def __init__(self, path, language, treaty_ids):
        
        self.path        = path
        self.language    = language
        
        pattern          = re.compile("^(\w*)\_" + language + "([\_\-]corr)?\.txt$") 
        treaty_lookup    = { x.split('_')[0]: x for x in self._ls_archive(path, pattern) }
        
        self.treaty_ids  = [ x for x in treaty_ids if x in treaty_lookup ]
        self.filenames   = [ treaty_lookup[x] for x in self.treaty_ids ]
        
        if len(set(treaty_ids) - set(self.treaty_ids)) > 0:
            logger.warning('Treaties not found in archive: ' + ', '.join(list(set(treaty_ids) - set(self.treaty_ids))))
            
        CompressedFileReader.__init__(self, path, pattern=pattern, itemfilter=self.filenames)
        
    def _ls_archive(self, path, pattern):
        
        px = lambda x: pattern.match(x) \
            if isinstance(pattern, typing.re.Pattern) \
            else fnmatch.fnmatch(x, pattern)
        
        with zipfile.ZipFile(path) as f:
            return [ x for x in f.namelist() if px(x) ]
    
    def __next__(self):
        
        filename, content = super(TreatyCompressedFileReader, self).__next__()
        
        m = TREATY_FILENAME.match(filename)
        
        treaty_id = m.groups(1)[0]
        language = m.groups(1)[1]
        
        return treaty_id, language, filename, content

def get_document_stream(corpus_path, lang, treaties):

    if 'treaty_id' not in treaties.columns:
        treaties['treaty_id'] = treaties.index

    documents = TreatyCompressedFileReader(corpus_path, lang, list(treaties.index))

    for treaty_id, language, filename, text in documents:
        assert language == lang
        metadata = treaties.loc[treaty_id]
        yield filename, text, metadata
    
class TreatyCorpus(TextCorpus):

    def __init__(self,
                 stream,
                 dictionary=None,
                 metadata=False,
                 character_filters=None,
                 tokenizer=None,
                 token_filters=None,
                 bigram_transform=False
    ):
        self.stream = stream
        self.filenames = None
        self.documents = None
        self.length = None
        
        #if 'filenames' in content_iterator.__dict__:
        #    self.filenames = content_iterator.filenames
        #    self.document_names = self._compile_documents()
        #    self.length = len(self.filenames)
            
        token_filters = [
           (lambda tokens: [ x.lower() for x in tokens ]),
           (lambda tokens: [ x for x in tokens if any(map(lambda x: x.isalpha(), x)) ])
        ] + (token_filters or [])
        
        #if bigram_transform is True:
        #    train_corpus = TreatyCorpus(content_iterator, token_filters=[ x.lower() for x in tokens ])
        #    phrases = gensim.models.phrases.Phrases(train_corpus)
        #    bigram = gensim.models.phrases.Phraser(phrases)
        #    token_filters.append(
        #        lambda tokens: bigram[tokens]
        #    )           
        
        super(TreatyCorpus, self).__init__(
            input=True,
            dictionary=dictionary,
            metadata=metadata,
            character_filters=character_filters,
            tokenizer=tokenizer,
            token_filters=token_filters
        )
        
    def getstream(self):
        """Generate documents from the underlying plain text collection (of one or more files).
        Yields
        ------
        str
            Document read from plain-text file.
        Notes
        -----
        After generator end - initialize self.length attribute.
        """
        
        document_infos = []
        for treaty_id, language, filename, content in self.stream:
            yield content
            document_infos.append({
                'document_name': filename,
                'treaty_id': treaty_id,
                'language': language
            })
            
        self.length = len(document_infos)
        self.documents = pd.DataFrame(document_infos)
        #self.filenames = list(self.documents.document_name)
                 
    def get_texts(self):
        '''
        This is mandatory method from gensim.corpora.TextCorpus. Returns stream of documents.
        '''
        for document in self.getstream():
            yield self.preprocess_text(document)
            
    def preprocess_text(self, text):
            """Apply `self.character_filters`, `self.tokenizer`, `self.token_filters` to a single text document.
            
            Parameters
            ---------
            text : str
                Document read from plain-text file.
                
            Returns
            ------
            list of str
                List of tokens extracted from `text`.
                
            """
            for character_filter in self.character_filters:
                text = character_filter(text)

            tokens = self.tokenizer(text)
            for token_filter in self.token_filters:
                tokens = token_filter(tokens)

            return tokens

    def __get_document_info(self, filename):
        parts = TREATY_FILENAME.match(filename)
        if not parts:
            return {
            'document_name': filename,
            'treaty_id': None,
            'language': None
        }
        return {
            'document_name': filename,
            'treaty_id': parts.groups(0)[0],
            'language': parts.groups(0)[1]
        }

    def ___compile_documents(self):
        
        document_data = map(self.get_document_info, self.filenames)

        documents = pd.DataFrame(list(document_data))
        documents.index.names = ['document_id']
        dupes = documents.groupby(['treaty_id', 'language']).size().loc[lambda x: x > 1]
        
        if len(dupes) > 0:
            logger.critical('Warning! Duplicate treaties found in corpus: {}'.format(' '.join(list(dupes.index))))
            
        return documents
        
class MmCorpusStatisticsService():
    
    def __init__(self, corpus, dictionary, language):
        self.corpus = corpus
        self.dictionary = dictionary
        self.stopwords = nltk.corpus.stopwords.words(language[1])
        _ = dictionary[0]
        
    def get_total_token_frequencies(self):
        dictionary = self.corpus.dictionary
        freqencies = np.zeros(len(dictionary.id2token))
        document_stats = []
        for document in corpus:
            for i, f in document:
                freqencies[i] += f
        return freqencies

    def get_document_token_frequencies(self):
        from itertools import chain
        '''
        Returns a DataFrame with per document token frequencies i.e. "melts" doc-term matrix
        '''
        data = ((document_id, x[0], x[1]) for document_id, values in enumerate(self.corpus) for x in values )
        pd = pd.DataFrame(list(zip(*data)), columns=['document_id', 'token_id', 'count'])
        pd = pd.merge(self.corpus.document_names, left_on='document_id', right_index=True)

        return pd

    def compute_word_frequencies(self, remove_stopwords):
        id2token = self.dictionary.id2token
        term_freqencies = np.zeros(len(id2token))
        document_stats = []
        for document in self.corpus:
            for i, f in document:
                term_freqencies[i] += f
        stopwords = set(self.stopwords).intersection(set(id2token.values()))
        df = pd.DataFrame({
            'token_id': list(id2token.keys()),
            'token': list(id2token.values()),
            'frequency': term_freqencies,
            'dfs':  list(self.dictionary.dfs.values())
        })
        df['is_stopword'] = df.token.apply(lambda x: x in stopwords)
        if remove_stopwords is True:
            df = df.loc[(df.is_stopword==False)]
        df['frequency'] = df.frequency.astype(np.int64)
        df = df[['token_id', 'token', 'frequency', 'dfs', 'is_stopword']].sort_values('frequency', ascending=False)
        return df.set_index('token_id')

    def compute_document_stats(self):
        id2token = self.dictionary.id2token
        stopwords = set(self.stopwords).intersection(set(id2token.values()))
        df = pd.DataFrame({
            'document_id': self.corpus.index,
            'document_name': self.corpus.document_names.document_name,
            'treaty_id': self.corpus.document_names.treaty_id,
            'size': [ sum(list(zip(*document))[1]) for document in self.corpus],
            'stopwords': [ sum([ v for (i,v) in document if id2token[i] in self.stopwords]) for document in self.corpus],
        }).set_index('document_name')
        df[['size', 'stopwords']] = df[['size', 'stopwords']].astype('int')
        return df

    def compute_word_stats(self):
        df = self.compute_document_stats()[['size', 'stopwords']]
        df_agg = df.agg(['count', 'mean', 'std', 'min', 'median', 'max', 'sum']).reset_index()
        legend_map = {
            'count': 'Documents',
            'mean': 'Mean words',
            'std': 'Std',
            'min': 'Min',
            'median': 'Median',
            'max': 'Max',
            'sum': 'Sum words'
        }
        df_agg['index'] = df_agg['index'].apply(lambda x: legend_map[x]).astype('str')
        df_agg = df_agg.set_index('index')
        df_agg[df_agg.columns] = df_agg[df_agg.columns].astype('int')
        return df_agg.reset_index()
    
#@staticmethod

class ExtMmCorpus(gensim.corpora.MmCorpus):
    """Extension of MmCorpus that allow TF normalization based on document length.
    """

    @staticmethod
    def norm_tf_by_D(doc):
        D = sum([x[1] for x in doc])
        return doc if D == 0 else map(lambda tf: (tf[0], tf[1]/D), doc)

    def __init__(self, fname):
        gensim.corpora.MmCorpus.__init__(self, fname)
        
    def __iter__(self):
        for doc in gensim.corpora.MmCorpus.__iter__(self):
            yield self.norm_tf_by_D(doc)

    def __getitem__(self, docno):
        return self.norm_tf_by_D(gensim.corpora.MmCorpus.__getitem__(self, docno))

class TreatyCorpusSaveLoad():

    def __init__(self, source_folder, lang):
        
        self.mm_filename = os.path.join(source_folder, 'corpus_{}.mm'.format(lang))
        self.dict_filename = os.path.join(source_folder, 'corpus_{}.dict.gz'.format(lang))
        self.document_index = os.path.join(source_folder, 'corpus_{}_documents.csv'.format(lang))
        
    def store_as_mm_corpus(self, treaty_corpus):
        
        gensim.corpora.MmCorpus.serialize(self.mm_filename, treaty_corpus, id2word=treaty_corpus.dictionary.id2token)
        treaty_corpus.dictionary.save(self.dict_filename)
        treaty_corpus.document_names.to_csv(self.document_index, sep='\t')

    def load_mm_corpus(self, normalize_by_D=False):
    
        corpus_type = ExtMmCorpus if normalize_by_D else gensim.corpora.MmCorpus
        corpus = corpus_type(self.mm_filename)
        corpus.dictionary = gensim.corpora.Dictionary.load(self.dict_filename)
        corpus.document_names = pd.read_csv(self.document_index, sep='\t').set_index('document_id')  

        return corpus
    
    def exists(self):
        return os.path.isfile(self.mm_filename) and \
            os.path.isfile(self.dict_filename) and \
            os.path.isfile(self.document_index)

