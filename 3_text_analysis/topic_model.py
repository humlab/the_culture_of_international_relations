import types
import textacy
from gensim import corpora, models, matutils

import common.utility as utility
import textacy_corpus_utility as textacy_utility
import topic_model_utility

# OBS OBS! https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)

def compute(corpus, tick=utility.noop, method='sklearn_lda', vec_args=None, term_args=None, tm_args=None, **args):
    
    tick()
    vec_args = utility.extend({}, DEFAULT_VECTORIZE_PARAMS, vec_args)
    
    terms_iter = lambda: (textacy_utility.textacy_filter_terms(doc, term_args) for doc in corpus)
    tick()
    
    vectorizer = textacy.Vectorizer(**vec_args)
    doc_term_matrix = vectorizer.fit_transform(terms_iter())

    if method.startswith('sklearn'):
        
        tm_model = textacy.TopicModel(method.split('_')[1], **tm_args)
        tm_model.fit(doc_term_matrix)
        tick()
        doc_topic_matrix = tm_model.transform(doc_term_matrix)
        tick()
        tm_id2word = vectorizer.id_to_term
        tm_corpus = matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)
        compiled_data = None # FIXME
        
    elif method.startswith('gensim_'):
        
        algorithm = method.split('_')[1].upper()
        doc_topic_matrix = None # ?
        tm_id2word = corpora.Dictionary(terms_iter())
        tm_corpus = [ tm_id2word.doc2bow(text) for text in terms_iter() ]
        
        algorithms = {
            'LSI': {
                'engine': models.LsiModel,
                'options': {
                    'corpus': tm_corpus, 
                    'num_topics':  tm_args.get('n_topics', 0),
                    'id2word':  tm_id2word,
                    'power_iters': 2,
                    'onepass': True
                }
            },
            'LDA': {
                'engine': models.LdaModel,
                'options': {
                    'corpus': tm_corpus, 
                    'num_topics':  tm_args.get('n_topics', 0),
                    'id2word':  tm_id2word,
                    'iterations': tm_args.get('max_iter', 0),
                    'passes': 20,
                    'alpha': 'auto'
                }
            }
        }
        tm_model = algorithms[algorithm]['engine'](**algorithms[algorithm]['options'])
        documents = textacy_utility.get_corpus_documents(corpus)
        compiled_data = topic_model_utility.compile_metadata(tm_model, tm_corpus, tm_id2word, documents)
    
    tm_data = types.SimpleNamespace(
        tm_model=tm_model,
        tm_id2term=tm_id2word,
        tm_corpus=tm_corpus,
        doc_term_matrix=doc_term_matrix,
        doc_topic_matrix=doc_topic_matrix,
        vectorizer=vectorizer,
        compiled_data=compiled_data,
        options=dict(method=method, vec_args=vec_args, term_args=term_args, tm_args=tm_args, **args)
    )
    
    tick(0)
    
    return tm_data

import pickle

def store_model(model, filename):
    
    data = types.SimpleNamespace(
        tm_model=model.tm_model,
        tm_id2term=model.tm_id2term,
        tm_corpus=model.tm_corpus,
        doc_term_matrix=None, #doc_term_matrix,
        doc_topic_matrix=None, #doc_topic_matrix,
        vectorizer=None, #vectorizer,
        compiled_data=model.compiled_data
    )

    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
