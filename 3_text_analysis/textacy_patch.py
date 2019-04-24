import spacy
import collections
from spacy import attrs

def to_bag_of_words(document, normalize="lemma", weighting="count", as_strings=False):
    """
    Patched to not remove stopwords.
    """
    if weighting not in {"count", "freq", "binary"}:
        raise ValueError('weighting "{}" is invalid'.format(weighting))
    count_by = (
        attrs.LEMMA
        if normalize == "lemma"
        else attrs.LOWER
        if normalize == "lower"
        else attrs.ORTH
    )        
    word_to_weight = document.spacy_doc.count_by(count_by)
    if weighting == "freq":
        n_tokens = document.n_tokens
        word_to_weight = {
            id_: weight / n_tokens for id_, weight in word_to_weight.items()
        }
    elif weighting == "binary":
        word_to_weight = {word: 1 for word in word_to_weight.keys()}

    bow = {}
    if as_strings is False:
        for id_, count in word_to_weight.items():
            lexeme = document.spacy_vocab[id_]
            #if lexeme.is_stop or lexeme.is_punct or lexeme.is_space:                    
            if lexeme.is_punct or lexeme.is_space:
                continue
            bow[id_] = count
    else:
        for id_, count in word_to_weight.items():
            lexeme = document.spacy_vocab[id_]
            #if lexeme.is_stop or lexeme.is_punct or lexeme.is_space:                    
            if lexeme.is_punct or lexeme.is_space:
                continue
            bow[document.spacy_stringstore[id_]] = count
    return bow


def word_doc_freqs(corpus, normalize="lemma", weighting="count", smooth_idf=True, as_strings=False):
    """
    Patched to not remove stopwords.
    """
    word_doc_counts = collections.Counter()
    for doc in corpus:
        word_doc_counts.update(
            #doc.to_bag_of_words(
            to_bag_of_words(doc,
                normalize=normalize, weighting="binary", as_strings=as_strings
            )
        )
    if weighting == "count":
        word_doc_counts = dict(word_doc_counts)
    elif weighting == "freq":
        n_docs = corpus.n_docs
        word_doc_counts = {
            word: count / n_docs for word, count in word_doc_counts.items()
        }
    elif weighting == "idf":
        n_docs = corpus.n_docs
        if smooth_idf is True:
            word_doc_counts = {
                word: math.log(1 + n_docs / count)
                for word, count in word_doc_counts.items()
            }
        else:
            word_doc_counts = {
                word: math.log(n_docs / count)
                for word, count in word_doc_counts.items()
            }
    elif weighting == "binary":
        word_doc_counts = {word: 1 for word in word_doc_counts.keys()}
    return word_doc_counts

