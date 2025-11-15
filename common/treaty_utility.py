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
SUBSTITUTION_FILENAME: str = os.path.join(DATA_FOLDER, "term_substitutions.txt")

CORPUS_NAME_PATTERN: str = "tCoIR_*.txt.zip"
CORPUS_TEXT_FILES_PATTERN: str = "*.txt"

WTI_INDEX_FOLDER: str = DATA_FOLDER  # os.path.join(DATA_FOLDER, 'wti_index')

GROUP_BY_OPTIONS: list[tuple[str, list[str]]] = [
    ("Year", ["signed_year"]),
    ("Party1", ["party1"]),
    ("Party1, Year", ["party1", "signed_year"]),
    ("Party2, Year", ["party2", "signed_year"]),
    ("Group1, Year", ["group1", "signed_year"]),
    ("Group2, Year", ["group2", "signed_year"]),
]


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


def get_document_stream(source: CompressedFileReader | str, lang: str, document_index: pd.DataFrame | None = None):

    assert document_index is not None

    if "document_id" not in document_index.columns:
        document_index["document_id"] = document_index.index

    def id_extractor(filename: str) -> str:
        match: re.Match | None = re.match(r"^(\w*)\_" + lang + r"([\_\-]corr)?\.txt$", filename)
        if match:
            return match.group(1)
        return ""

    lang_pattern: re.Pattern = re.compile(rf"^(\w*)_{lang}([_-]corr)?\.txt$")

    def item_filter(x):
        return lang_pattern.match(x)  # and id_extractor(x) in document_index.index

    if isinstance(source, str):
        print(f"Opening archive: {source}")
        reader: CompressedFileReader = CompressedFileReader(source, pattern=lang_pattern, itemfilter=item_filter)
    else:
        reader = source

    id_map: dict[str, str] = {
        filename: id_extractor(filename) for filename in reader.filenames if item_filter(filename)
    }

    if len(set(document_index.index) - set(id_map.values())) > 0:
        logger.warning(
            "Treaties not found in archive: " + ", ".join(list(set(document_index.index) - set(id_map.values())))
        )

    columns: list[str] = ["signed_year", "party1", "party2"]

    df: pd.DataFrame = document_index[columns]

    for filename, text in reader:

        document_id: str | None = id_map.get(filename)

        if document_id not in df.index:
            continue

        metadata = df.loc[document_id].to_dict()

        metadata["filename"] = filename
        metadata["document_id"] = document_id
        metadata["treaty_id"] = document_id

        yield filename, document_id, text, metadata


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

    # @staticmethod
    # def is_cultural_mask():
    #     def fx(df):
    #         df.loc[df.is_cultural, "topic1"] = "7CORR"
    #         return True

    #     return fx

    # @staticmethod
    # def is_cultural_mask():
    #     return lambda df: (df.is_cultural)

    @staticmethod
    def years_mask(ys):
        return lambda df: (
            ((ys[0] <= df.signed_year) & (df.signed_year <= ys[1]))
            if isinstance(ys, tuple) and len(ys) == 2
            else df.signed_year.isin(ys)
        )

    @staticmethod
    def topic_7cult_mask():
        return lambda df: (df.topic1 == "7CULT")

    @staticmethod
    def query_treaties(treaties, filter_masks):
        """NOT USED but could perhaps replace functions above"""
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
