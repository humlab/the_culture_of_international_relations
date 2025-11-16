import collections
import os
import re
from typing import Any

import pandas as pd
from loguru import logger
from textacy.corpus import Corpus

from common import config, utility
from common.corpus.utility import CompressedFileReader
from common.gui.load_wti_index_gui import current_wti_index

DATA_FOLDER: str = config.DATA_FOLDER


def get_tagset() -> pd.DataFrame | None:
    filepath: str = os.path.join(DATA_FOLDER, "tagset.csv")
    if os.path.isfile(filepath):
        return pd.read_csv(filepath, sep="\t").fillna("")
    return None


def get_parties() -> pd.DataFrame:
    parties: pd.DataFrame = current_wti_index().get_parties()
    return parties


POS_TO_COUNT: dict[str, int] = {
    "SYM": 0,
    "PART": 0,
    "ADV": 0,
    "NOUN": 0,
    "CCONJ": 0,
    "ADJ": 0,
    "DET": 0,
    "ADP": 0,
    "INTJ": 0,
    "VERB": 0,
    "NUM": 0,
    "PRON": 0,
    "PROPN": 0,
}

POS_NAMES: list[str] = list(sorted(POS_TO_COUNT.keys()))


def _get_pos_statistics(doc) -> dict[str, int]:
    pos_iter = (x.pos_ for x in doc if x.pos_ not in ["NUM", "PUNCT", "SPACE"])
    pos_counts: dict[str, int] = dict(collections.Counter(pos_iter))
    stats: dict[str, int] = utility.extend(dict(POS_TO_COUNT), pos_counts)
    return stats


def get_corpus_documents(corpus: Corpus) -> pd.DataFrame:
    metadata = [utility.extend({}, doc._.meta, _get_pos_statistics(doc)) for doc in corpus.docs]
    df: pd.DataFrame = pd.DataFrame(metadata)[["treaty_id", "filename", "signed_year", "party1", "party2"] + POS_NAMES]
    df["title"] = df.treaty_id
    df["lang"] = df.filename.str.extract(r"\w{4,6}\_(\w\w)")
    df["words"] = df[POS_NAMES].apply(sum, axis=1)
    return df


def get_treaty_dropdown_options(wti_index: pd.DataFrame, corpus: Corpus):

    def format_treaty_name(x) -> str:

        return f'{x.name}: {x["signed_year"]} {x["topic"]} {x["party1"]} {x["party2"]}'

    documents: pd.Series = wti_index.treaties.loc[get_corpus_documents(corpus).treaty_id]

    options = [(v, k) for k, v in documents.apply(format_treaty_name, axis=1).to_dict().items()]
    options = sorted(options, key=lambda x: x[0])

    return options


def trim_period_group(period_group, year_limit):
    pg = dict(period_group)
    low, high = year_limit
    if pg["type"] == "range":
        pg["periods"] = [x for x in pg["periods"] if low <= x <= high]
    else:
        pg["periods"] = [(max(low, x), min(high, y)) for x, y in pg["periods"] if not (high < x and low > y)]
    return pg


def period_group_years(period_group: dict[str, Any]):
    if period_group["type"] == "range":
        return period_group["periods"]
    period_years: list[list[int]] = [list(range(x[0], x[1] + 1)) for x in period_group["periods"]]
    return utility.flatten(period_years)


class QueryUtility:

    @staticmethod
    def parties_mask(parties):
        return lambda df: (df.party1.isin(parties)) | (df.party2.isin(parties))

    @staticmethod
    def period_group_mask(pg):
        return lambda df: df.signed_year.isin(period_group_years(pg))

    @staticmethod
    def years_mask(ys):
        return lambda df: (
            ((ys[0] <= df.signed_year) & (df.signed_year <= ys[1]))
            if isinstance(ys, tuple) and len(ys) == 2
            else df.signed_year.isin(ys)
        )

    @staticmethod
    def query_treaties(treaties, filter_masks) -> pd.DataFrame:
        #  period_group=None, treaty_filter='', recode_is_cultural=False, parties=None, year_limit=None):
        if not isinstance(filter_masks, list):
            filter_masks = [filter_masks]
        for mask in filter_masks:
            if mask is True:
                pass
            elif isinstance(mask, str):
                treaties = treaties.query(mask)
            elif callable(mask):
                treaties = treaties.loc[mask(treaties)]
            else:
                treaties = treaties.loc[mask]
        return treaties
