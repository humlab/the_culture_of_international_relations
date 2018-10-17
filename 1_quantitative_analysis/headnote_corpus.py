import nltk
import pandas as pd
from common.utility import uniquify

class HeadnoteTokenCorpus():
    
    def __init__(self, treaties, tokenize=None, stopwords=None, lemmatize=None, min_word_length=2, ignore_word_count=True):
        
        self.min_word_length = min_word_length
        
        tokenize = tokenize or nltk.tokenize.word_tokenize
        lemmatize = lemmatize or nltk.stem.WordNetLemmatizer().lemmatize
        stopwords = stopwords or nltk.corpus.stopwords.words('english')
        
        transforms = [
            lambda ws: ( x for x in ws if len(x) >= min_word_length ),
            lambda ws: ( x for x in ws if any(ch.isalpha() for ch in x))
        ]
        
        if ignore_word_count:
            transforms = transforms + [ lambda ws: uniquify(ws) ]
            
        headnote_tokens = treaties.headnote.str.lower().apply(tokenize)
        for f in transforms:
            headnote_tokens = headnote_tokens.apply(f)
    
        self.headnote_tokens = pd.DataFrame({'tokens': headnote_tokens})
        
        tokens = pd.DataFrame(self.headnote_tokens.tokens.tolist(), index=self.headnote_tokens.index)\
            .stack()\
            .reset_index()\
            .rename(columns={'level_1': 'sequence_id', 0: 'token'})
        
        self.vocabulary = pd.DataFrame({ 'token': tokens.token.unique()})
        self.vocabulary['lemma'] = self.vocabulary.token.apply(lemmatize)
        self.vocabulary['is_stopword'] = self.vocabulary.token.apply(lambda x: x in stopwords)
        self.vocabulary = self.vocabulary.set_index('token')
        
        self.tokens = tokens.merge(self.vocabulary, left_on='token', right_index=True, how='inner').set_index(['treaty_id', 'sequence_id'])

    
