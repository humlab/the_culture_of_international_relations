import pandas as pd

from common.domain_logic_config import current_domain as domain_logic


def get_treaty_dropdown_options(wti_index, corpus):

    def format_treaty_name(x):
        return f'{x.name}: {x["signed_year"]} {x["topic"]} {x["party1"]} {x["party2"]}'

    treaties: pd.DataFrame = domain_logic.get_corpus_documents(corpus)
    df = wti_index.treaties.loc[treaties.treaty_id]

    options = [(v, k) for k, v in df.apply(format_treaty_name, axis=1).to_dict().items()]
    options = sorted(options, key=lambda x: x[0])

    return options
