
import textacy_corpus_utility as textacy_utility

def get_treaty_dropdown_options(wti_index, corpus):
    
    def format_treaty_name(x):
        return '{}: {} {} {} {}'.format(x.name, x['signed_year'], x['topic'], x['party1'], x['party2'])
    
    documents = wti_index.treaties.loc[textacy_utility.get_corpus_documents(corpus).treaty_id]

    options = [ (v, k) for k, v in documents.apply(format_treaty_name, axis=1).to_dict().items() ]
    options = sorted(options, key=lambda x: x[0])

    return options
