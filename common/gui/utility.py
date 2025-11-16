import pandas as pd
from textacy.corpus import Corpus

from common import treaty_utility


def get_treaty_dropdown_options(wti_index: pd.DataFrame, corpus: Corpus):

    def format_treaty_name(x):
        return f'{x.name}: {x["signed_year"]} {x["topic"]} {x["party1"]} {x["party2"]}'

    treaties: pd.DataFrame = treaty_utility.get_corpus_documents(corpus)
    documents: pd.Series = wti_index.treaties.loc[treaties.treaty_id]

    options = [(v, k) for k, v in documents.apply(format_treaty_name, axis=1).to_dict().items()]
    options = sorted(options, key=lambda x: x[0])

    return options
