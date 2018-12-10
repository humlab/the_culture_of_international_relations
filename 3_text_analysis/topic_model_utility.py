import types
import pandas as pd
import logging
import common.utility as utility
import gensim

logger = utility.getLogger('corpus_text_analysis')

def get_doc_topic_weights(doc_topic_matrix, threshold=0.05):
    topic_ids = range(0,doc_topic_matrix.shape[1])
    for document_id in range(0,doc_topic_matrix.shape[1]):
        topic_weights = doc_topic_matrix[document_id, :]
        for topic_id in topic_ids:
            if topic_weights[topic_id] >= threshold:
                yield (document_id, topic_id, topic_weights[topic_id])

def get_df_doc_topic_weights(doc_topic_matrix, threshold=0.05):
    it = get_doc_topic_weights(doc_topic_matrix, threshold)
    df = pd.DataFrame(list(it), columns=['document_id', 'topic_id', 'weight']).set_index('document_id')
    return df

def compile_dictionary(model):
    logger.info('Compiling dictionary...')
    token_ids, tokens = list(zip(*model.id2word.items()))
    dfs = model.id2word.dfs.values() if model.id2word.dfs is not None else [0] * len(tokens)
    dictionary = pd.DataFrame({
        'token_id': token_ids,
        'token': tokens,
        'dfs': list(dfs)
    }).set_index('token_id')[['token', 'dfs']]
    return dictionary

def compile_topic_token_weights(tm, dictionary, num_words=200):
    logger.info('Compiling topic-tokens weights...')

    df_topic_weights = pd.DataFrame(
        [ (topic_id, token, weight)
            for topic_id, tokens in (tm.show_topics(tm.num_topics, num_words=num_words, formatted=False))
                for token, weight in tokens if weight > 0.0 ],
        columns=['topic_id', 'token', 'weight']
    )

    df = pd.merge(
        df_topic_weights.set_index('token'),
        dictionary.reset_index().set_index('token'),
        how='inner',
        left_index=True,
        right_index=True
    )
    return df.reset_index()[['topic_id', 'token_id', 'token', 'weight']]

def compile_topic_token_overview(topic_token_weights, alpha=None, n_words=200):
    """
    Group by topic_id and concatenate n_words words within group sorted by weight descending.
    There must be a better way of doing this...
    """
    logger.info('Compiling topic-tokens overview...')

    df = topic_token_weights.groupby('topic_id')\
        .apply(lambda x: sorted(list(zip(x["token"], x["weight"])), key=lambda z: z[1], reverse=True))\
        .apply(lambda x: ' '.join([z[0] for z in x][:n_words])).reset_index()
    df['alpha'] = df.topic_id.apply(lambda topic_id: alpha[topic_id]) if alpha is not None else 0.0
    df.columns = ['topic_id', 'tokens', 'alpha']

    return df.set_index('topic_id')

def compile_document_topics(model, corpus, documents, minimum_probability=0.001):

    def document_topics_iter(model, corpus, minimum_probability):

        if isinstance(model, gensim.models.LsiModel):
            data_iter = model[corpus]
        else:
            data_iter = model.get_document_topics(corpus, minimum_probability=minimum_probability)\
                if hasattr(model, 'get_document_topics')\
                else model.load_document_topics()

        for document_id, topic_weights in enumerate(data_iter):
            for (topic_id, weight) in ((topic_id, weight) for (topic_id, weight) in topic_weights if weight >= minimum_probability):
                yield (document_id, topic_id, weight)
    '''
    Get document topic weights for all documents in corpus
    Note!  minimum_probability=None filters less probable topics, set to 0 to retrieve all topcs

    If gensim model then use 'get_document_topics', else 'load_document_topics' for mallet model
    '''
    logger.info('Compiling document topics...')
    logger.info('  Creating data iterator...')
    data = document_topics_iter(model, corpus, minimum_probability)
    logger.info('  Creating frame from iterator...')
    df_doc_topics = pd.DataFrame(data, columns=[ 'document_id', 'topic_id', 'weight' ]).set_index('document_id')
    logger.info('  Merging data...')
    df = pd.merge(documents, df_doc_topics, how='inner', left_index=True, right_index=True)
    logger.info('  DONE!')
    return df

def compile_metadata(model, corpus, id2term, documents):
    '''
    Compile metadata associated to given model and corpus
    '''
    dictionary = compile_dictionary(model)
    topic_token_weights = compile_topic_token_weights(model, dictionary, num_words=200)
    alpha = model.alpha if 'alpha' in model.__dict__ else None
    topic_token_overview = compile_topic_token_overview(topic_token_weights, alpha)
    document_topic_weights = compile_document_topics(model, corpus, documents, minimum_probability=0.001)
    year_period = (documents.signed_year.min(), documents.signed_year.max())
    
    return types.SimpleNamespace(
        dictionary=dictionary,
        documents=documents,
        topic_token_weights=topic_token_weights,
        topic_token_overview=topic_token_overview,
        document_topic_weights=document_topic_weights,
        year_period=year_period
    )

def get_topic_titles(topic_token_weights, topic_id=None, n_words=100):
    df_temp = topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id==topic_id)]
    df = df_temp\
            .sort_values('weight', ascending=False)\
            .groupby('topic_id')\
            .apply(lambda x: ' '.join(x.token[:n_words].str.title()))
    return df

def get_topic_title(topic_token_weights, topic_id, n_words=100):
    return get_topic_titles(topic_token_weights, topic_id, n_words=n_words).iloc[0]

def get_topic_tokens(topic_token_weights, topic_id=None, n_words=100):
    df_temp = topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
    df = df_temp.sort_values('weight', ascending=False)[:n_words]
    return df

def get_lda_topics(model, n_tokens=20):
    return pd.DataFrame({
        'Topic#{:02d}'.format(topic_id+1) : [ word[0] for word in model.show_topic(topic_id, topn=n_tokens) ]
            for topic_id in range(model.num_topics)
    })
