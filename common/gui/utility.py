from typing import Any

import pandas as pd
from textacy.corpus import Corpus

from common.corpus.textacy_corpus_utility import get_corpus_documents
from common.utility import trunc_year_by


def get_treaty_dropdown_options(wti_index: pd.DataFrame, corpus: Corpus):

    def format_treaty_name(x):
        return f'{x.name}: {x["signed_year"]} {x["topic"]} {x["party1"]} {x["party2"]}'

    treaties: pd.DataFrame = get_corpus_documents(corpus)
    documents: pd.Series = wti_index.treaties.loc[treaties.treaty_id]

    options = [(v, k) for k, v in documents.apply(format_treaty_name, axis=1).to_dict().items()]
    options = sorted(options, key=lambda x: x[0])

    return options


def get_treaty_time_groupings() -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {
        "treaty_id": {
            "column": "treaty_id",
            "divisor": None,
            "title": "Treaty",
            "fx": None,
        },
        "signed_year": {
            "column": "signed_year",
            "divisor": 1,
            "title": "Year",
            "fx": None,
        },
        "signed_lustrum": {
            "column": "signed_lustrum",
            "divisor": 5,
            "title": "Lustrum",
            "fx": lambda df: trunc_year_by(df.signed_year, 5),
        },
        "signed_decade": {
            "column": "signed_decade",
            "divisor": 10,
            "title": "Decade",
            "fx": lambda df: trunc_year_by(df.signed_year, 10),
        },
    }
    return groups
