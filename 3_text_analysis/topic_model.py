import types
import textacy
import pandas as pd
import gensim

import common.utility as utility
import textacy_corpus_utility as textacy_utility
import topic_model_utility
import mallet_service
    
# OBS OBS! https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)

# FIXME: Bug somewhere...
def n_gram_detector(doc_iter, n_gram_size=2, min_count=5, threshold=100):
    
    for n_span in range(2, n_gram_size+1):
        print('Applying {}_gram detector'.format(n_span))
        n_grams = gensim.models.Phrases(doc_iter(), min_count=min_count, threshold=threshold)
        ngram_modifier = gensim.models.phrases.Phraser(n_grams)
        ngram_doc_iter = lambda: ( ngram_modifier[doc] for doc in doc_iter() )
        doc_iter = ngram_doc_iter
        
    return doc_iter
    
def compute(corpus, tick=utility.noop, method='sklearn_lda', vec_args=None, term_args=None, tm_args=None, **args):
    
    tick()
    
    vec_args = utility.extend({}, DEFAULT_VECTORIZE_PARAMS, vec_args)
    
    def terms_iter():
        
        n_gram_size = term_args.get('arg', {}).get('ngrams', 1)
        
        doc_iter = ( textacy_utility.textacy_filter_terms(doc, term_args) for doc in corpus )
            
        if n_gram_size > 1:
            doc_iter = (token.replace(' ', '_') for token in doc_iter)
            # doc_iter = n_gram_detector(doc_iter, n_gram_size=2, min_count=5, threshold=100)
            
        return doc_iter
    
    tick()
            
    perplexity_score = None
    coherence_score = None
    vectorizer = None
    doc_topic_matrix = None
    doc_term_matrix = None
    documents = textacy_utility.get_corpus_documents(corpus)

    if method.startswith('sklearn'):
        
        vectorizer = textacy.Vectorizer(**vec_args)
        doc_term_matrix = vectorizer.fit_transform(terms_iter())

        tm_model = textacy.TopicModel(method.split('_')[1], **tm_args)
        tm_model.fit(doc_term_matrix)
        
        tick()
        
        doc_topic_matrix = tm_model.transform(doc_term_matrix)
        
        tick()
        
        tm_id2word = vectorizer.id_to_term
        tm_corpus = gensim.matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)
        
        # FIXME!!!
        perplexity_score = None
        coherence_score = None
        
    elif method.startswith('gensim_'):
        
        algorithm = method.split('_')[1].upper()
        tm_id2word = gensim.corpora.Dictionary(terms_iter())
        tm_corpus = [ tm_id2word.doc2bow(text) for text in terms_iter() ]
        
        #tfidf_model = gensim.models.tfidfmodel.TfidfModel(tm_corpus)
        #tm_corpus = [ tfidf_model[d] for d in tm_corpus ]
        
        algorithms = {
            'LSI': {
                'engine': gensim.models.LsiModel,
                'options': {
                    'corpus': tm_corpus, 
                    'num_topics':  tm_args.get('n_topics', 0),
                    'id2word':  tm_id2word,
                    'power_iters': 2,
                    'onepass': True
                }
            },
            'LDA': {
                'engine': gensim.models.LdaModel,
                'options': {
                    'corpus': tm_corpus, 
                    'num_topics':  tm_args.get('n_topics', 0),
                    'id2word':  tm_id2word,
                    'iterations': tm_args.get('max_iter', 0),
                    'passes': tm_args.get('passes', 40),
                    #'eval_every': 5,
                    'update_every': 0,
                    'alpha': 'auto',
                    'eta': 'auto'
                }
            },
            'HDP': {
                'engine': gensim.models.HdpModel,
                'options': {
                    'corpus': tm_corpus, 
                    'T':  tm_args.get('n_topics', 0),
                    'id2word':  tm_id2word,
                    #'iterations': tm_args.get('max_iter', 0),
                    #'passes': tm_args.get('passes', 20),
                    #'alpha': 'auto'
                }
            },
            'DTM': {
                'engine': gensim.models.LdaSeqModel,
                'options': {
                    'corpus': tm_corpus, 
                    'num_topics':  tm_args.get('n_topics', 0),
                    'id2word':  tm_id2word,
                    'time_slice': textacy_utility.count_documents_by_pivot(corpus, 'signed_year')
                    # 'initialize': 'gensim/own/ldamodel',
                    # 'lda_model': model # if initialize='gensim'
                    # 'lda_inference_max_iter': tm_args.get('max_iter', 0),
                    # 'passes': tm_args.get('passes', 20),
                    # 'alpha': 'auto'
                }
            },
            'MALLETLDA': {
                'engine': mallet_service.LdaMalletService,
                'options': {
                    'corpus': tm_corpus, 
                    'id2word':  tm_id2word,
                    'default_mallet_home': '/usr/local/share/mallet-2.0.8/',
                    
                    'num_topics':  tm_args.get('n_topics', 0),
                    'iterations': tm_args.get('max_iter', 0),
                    'passes': tm_args.get('passes', 20),
                    
                    'prefix': './data/',
                    'workers': 4,
                    'optimize_interval': 10
                }
            },
        }
        
        tm_model = algorithms[algorithm]['engine'](**algorithms[algorithm]['options'])
        
        if hasattr(tm_model, 'log_perplexity'):
            perplexity_score = tm_model.log_perplexity(tm_corpus, len(tm_corpus))
        
        try:
            coherence_model_lda =  gensim.models.CoherenceModel(model=tm_model, texts=terms_iter(), dictionary=tm_id2word, coherence='c_v')
            coherence_score = coherence_model_lda.get_coherence()
        except Exception as ex:
            logger.error(ex)
            coherence_score = None
            
    compiled_data = topic_model_utility.compile_metadata(
        tm_model,
        tm_corpus,
        tm_id2word,
        documents,
        vectorizer=vectorizer,
        doc_topic_matrix=doc_topic_matrix,
        n_tokens=200
    )
    
    tm_data = types.SimpleNamespace(
        tm_model=tm_model,
        tm_id2term=tm_id2word,
        tm_corpus=tm_corpus,
        #doc_term_matrix=doc_term_matrix,
        #doc_topic_matrix=doc_topic_matrix,
        #vectorizer=vectorizer,
        compiled_data=compiled_data,
        perplexity_score=perplexity_score,
        coherence_score=coherence_score,
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

def compute_topic_proportions(document_topic_weights, doc_length_series):

    '''
    Topic proportions are computed in the same as in LDAvis.

    Computes topic proportions over entire corpus.
    The document topic weight (pivot) matrice is multiplies by the length of each document
      i.e. each weight are multiplies ny the document's length.
    The topic frequency is then computed by summing up all values over each topic
    This value i then normalized by total sum of matrice

    theta matrix: with each row containing the probability distribution
      over topics for a document, with as many rows as there are documents in the
      corpus, and as many columns as there are topics in the model.

    doc_length integer vector containing token count for each document in the corpus

    '''
    # compute counts of tokens across K topics (length-K vector):
    # (this determines the areas of the default topic circles when no term is highlighted)
    # topic.frequency <- colSums(theta * doc.length)
    # topic.proportion <- topic.frequency/sum(topic.frequency)

    theta = pd.pivot_table(
        document_topic_weights,
        values='weight',
        index=['document_id'],
        columns=['topic_id']
    ) #.set_index('document_id')

    theta_mult_doc_length = theta.mul(doc_length_series.words, axis=0)

    topic_frequency = theta_mult_doc_length.sum()
    topic_proportion = topic_frequency / topic_frequency.sum()

    return topic_proportion
    