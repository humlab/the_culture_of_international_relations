import time
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import ipywidgets as widgets
import numpy as np
import pandas as pd
import spacy
import spacy.tokens
from IPython.display import display
from loguru import logger
from spacy.attrs import LEMMA, LOWER, ORTH  # pylint: disable=no-name-in-module
from textacy.corpus import Corpus

from common import utility, widgets_config
from common.corpus import textacy_corpus_utility as textacy_utility

utility.setup_default_pd_display()

# def count_words_by(doc, target='lemma', include=None):

#     spacy_store = doc.vocab.strings

#     default_exclude = lambda x: x.is_stop or x.is_punct or x.is_space
#     exclude = default_exclude if include is None else lambda x: x.is_stop or x.is_punct or x.is_space or not include(x)

#     assert target in target_keys

#     word_counts = doc.count_by(target_keys[target], exclude=exclude)

#     bow = {
#         spacy_store[word_id]: count  for word_id, count in word_counts.items()
#     }

#     if target == 'lemma':
#         lower_cased_word_counts = collections.Counter()
#         for k, v in bow.items():
#             lower_cased_word_counts.update({ k.lower(): v })
#         bow = lower_cased_word_counts

#     return bow


def corpus_size_by_grouping(corpus, treaty_time_groups, group_by_column):

    data = ((doc._.meta["treaty_id"], doc._.meta["signed_year"], doc._.n_tokens) for doc in corpus)
    df_sizes = pd.DataFrame(data, columns=["treaty_id", "signed_year", "n_tokens"])
    if group_by_column not in df_sizes.columns:
        df_sizes[group_by_column] = (treaty_time_groups[group_by_column]["fx"])(df_sizes)
    df_sizes = df_sizes.groupby(group_by_column)["n_tokens"].sum()
    return df_sizes


def compute_list_of_most_frequent_words(
    corpus: Corpus,
    gui,
    treaty_time_groups: dict,
    group_by_column: str = "signed_year",
    parties=None,
    target: str = "lemma",
    weighting: str = "count",
    include_pos: set[str] | None = None,
    stop_words: set[str] | None = None,
    display_score=False,
) -> pd.DataFrame:

    stop_words = set(stop_words or set())
    include_pos = set(include_pos) if include_pos is not None and len(include_pos) > 0 else None
    target_keys = {"lemma": LEMMA, "lower": LOWER, "orth": ORTH}

    def exclude(x) -> bool:

        if x.is_stop or x.is_punct or x.is_space:
            return True

        if include_pos is not None and x.pos_ not in include_pos:
            return True

        if x.lemma_ in stop_words:
            return True

        return len(x.lemma_) < 2

    df_counts = pd.DataFrame({"treaty_id": [], "signed_year": [], "word_id": [], "word": [], "word_count": []})

    parties_set = set(parties or [])

    docs: Iterable[spacy.tokens.Doc] = (
        corpus.docs
        if len(parties_set) == 0
        else (x for x in corpus.docs if len(set((x._.meta["party1"], x._.meta["party2"])) & parties_set) > 0)
    )

    gui.progress.max = len(corpus)

    for doc in docs:

        # f doc._.meta['signed_year'] != 1956:
        #   continue

        spacy_store = doc.vocab.strings

        word_counts = doc.count_by(target_keys[target], exclude=exclude)

        df = pd.DataFrame(
            {
                "treaty_id": doc._.meta["treaty_id"],
                "signed_year": int(doc._.meta["signed_year"]),
                "word_id": [np.uint64(key) for key in word_counts],
                "word": [spacy_store[key].lower() for key in word_counts],
                "word_count": list(word_counts.values()),
            }
        )

        df_counts = df_counts.append(df)  # type: ignore

        gui.progress.value = gui.progress.value + 1

    df_counts["signed_year"] = df_counts.signed_year.astype(int)
    df_counts["word_count"] = df_counts.word_count.astype(int)

    if group_by_column not in df_counts.columns:
        df_counts[group_by_column] = (treaty_time_groups[group_by_column]["fx"])(df_counts)

    if group_by_column != "treaty_id":

        df_counts = df_counts.groupby([group_by_column, "word"]).sum()

    df_counts = df_counts.reset_index()[
        [group_by_column, "word", "word_count"] + (["signed_year"] if group_by_column != "signed_year" else [])
    ].set_index([group_by_column, "word"])

    # df_sizes = corpus_size_by_grouping(corpus, group_by_column)
    df_sizes = df_counts.groupby([group_by_column]).sum()["word_count"]
    # df_sizes.to_excel('df_sizes.xlsx')
    # df_counts['word_frequency'] = df_counts.word_count.astype(np.float64).div(df_sizes)
    df_counts["word_frequency"] = 100.0 * (df_counts.word_count.astype(np.float64) / df_sizes.astype(np.float64))

    df_counts = df_counts.reset_index()
    # df_counts.to_excel('df_counts.xlsx')

    if display_score is True:
        df_counts["word"] = (
            df_counts.word
            + "*"
            + (
                df_counts.word_frequency.apply("{:,.3f}".format)  # pylint: disable=consider-using-f-string
                if weighting == "freq"
                else df_counts.word_count.astype(str)
            )
        )

    df_counts["position"] = (
        df_counts.sort_values(by=[group_by_column, "word_frequency"], ascending=False)
        .groupby([group_by_column])
        .cumcount()
        + 1
    )

    df_counts: pd.DataFrame = df_counts[[group_by_column, "word", "word_count", "word_frequency", "position"]]
    gui.progress.value = 0

    return df_counts


def sanitize_name(x) -> str:
    sane_x = "_".join([c if (c.isalpha() or c.isdigit()) else "_" for c in (x or "")]).strip()
    return sane_x.lower()


def display_list_of_most_frequent_words(gui, df):

    display("N.B. *All* documents in the selected text corpus are used in this computation.")
    display(
        "No additional filters apart from selections made in Parties are applied i.e. WTI index selection made above has no effect in this notebook!"
    )

    if gui.output_type.value == "table":

        display(df.head(500))

    elif gui.output_type.value == "rank":

        group_by_column = gui.group_by_column.value
        df = df[df.position <= 50]
        df_unstacked_freqs = (
            df[[group_by_column, "position", "word"]].set_index([group_by_column, "position"]).unstack()
        )
        display(df_unstacked_freqs)

    else:
        sanitized_suffix: str = sanitize_name(gui.file_suffix.value)
        sanitized_suffix = "_" + sanitized_suffix if len(sanitized_suffix) > 0 else ""
        # if gui.party_preset.value is not None:
        #    sanitized_suffix = sanitize_name(gui.party_preset.value[0])
        print("Writing excel...")
        filename: str = (
            f"{time.strftime('%Y%m%d_%H%M%S')}_word_trends_{gui.normalize.value}_{'_'.join(gui.include_pos.value)}{sanitized_suffix}.xlsx"
        )
        df.to_excel(filename)
        print("Excel written: " + filename)


def word_frequency_gui(wti_index, corpus) -> SimpleNamespace:

    treaty_time_groups = wti_index.get_treaty_time_groupings()

    def lw(w):
        return widgets.Layout(width=w)

    include_pos_tags: list[str] = ["ADJ", "VERB", "ADV", "NOUN", "PROPN"]
    weighting_options: dict[str, str] = {"Count": "count", "Frequency": "freq"}
    normalize_options: dict[str, str | bool] = {"": False, "Lemma": "lemma", "Lower": "lower"}
    pos_options: list[str] = include_pos_tags
    default_include_pos: list[str] = ["NOUN", "PROPN"]
    frequent_words: list[str] = [
        x[0].lower() for x in textacy_utility.get_most_frequent_words(corpus, 100, include_pos=default_include_pos)
    ]

    group_by_options: dict[str, str] = {treaty_time_groups[k]["title"]: k for k in treaty_time_groups}
    output_type_options: list[tuple[str, str]] = [
        ("Sample", "table"),
        ("Rank", "rank"),
        ("Excel", "excel"),
    ]
    ngrams_options: dict[str, Any] = {"-": None, "1": [1], "1,2": [1, 2], "1,2,3": [1, 2, 3]}
    party_preset_options: dict[str, Any] = wti_index.get_party_preset_options()
    parties_options: list[str] = [x for x in wti_index.get_countries_list() if x != "ALL OTHER"]

    gui = SimpleNamespace(
        progress=widgets.IntProgress(value=0, min=0, max=5, step=1, description="", layout=lw("98%")),
        parties=widgets.SelectMultiple(
            description="Parties", options=parties_options, value=[], rows=7, layout=lw("200px")
        ),
        party_preset=widgets_config.dropdown("Presets", party_preset_options, None, layout=lw("200px")),
        ngrams=widgets.Dropdown(description="n-grams", options=ngrams_options, value=None, layout=lw("200px")),
        min_word=widgets.Dropdown(description="Min length", options=[1, 2, 3, 4], value=1, layout=lw("200px")),
        normalize=widgets.Dropdown(
            description="Normalize", options=normalize_options, value="lemma", layout=lw("200px")
        ),
        weighting=widgets.Dropdown(
            description="Weighting", options=weighting_options, value="freq", layout=lw("200px")
        ),
        include_pos=widgets.SelectMultiple(
            description="POS", options=pos_options, value=default_include_pos, rows=7, layout=lw("150px")
        ),
        stop_words=widgets.SelectMultiple(
            description="STOP", options=frequent_words, value=list([]), rows=7, layout=lw("200px")
        ),
        group_by_column=widgets.Dropdown(
            description="Group by", value="signed_year", options=group_by_options, layout=lw("200px")
        ),
        output_type=widgets.Dropdown(
            description="Output", value="rank", options=output_type_options, layout=lw("200px")
        ),
        compute=widgets.Button(description="Compute", button_style="Success", layout=lw("120px")),
        display_score=widgets.ToggleButton(description="Display score", icon="check", value=False, layout=lw("120px")),
        output=widgets.Output(layout={"border": "1px solid black"}),
        file_suffix=widgets.Text(
            value="",
            placeholder="(optional id)",
            description="ID:",
            disabled=False,
            layout=lw("200px"),
            tooltip="Optional plain text id that will be added to filename.",
        ),
    )

    boxes = widgets.VBox(
        [
            gui.progress,
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            gui.normalize,
                            gui.ngrams,
                            gui.weighting,
                            gui.group_by_column,
                            gui.output_type,
                        ]
                    ),
                    widgets.VBox(
                        [
                            gui.parties,
                            gui.party_preset,
                        ]
                    ),
                    gui.include_pos,
                    widgets.VBox(
                        [
                            gui.stop_words,
                            gui.file_suffix,
                        ]
                    ),
                    widgets.VBox(
                        [
                            gui.display_score,
                            gui.compute,
                        ],
                        layout=widgets.Layout(align_items="flex-end"),
                    ),
                ]
            ),
            gui.output,
        ]
    )

    display(boxes)

    def on_party_preset_change(change):  # pylint: disable=W0613
        if gui.party_preset.value is None:
            return
        gui.parties.value = gui.parties.options if "ALL" in gui.party_preset.value else gui.party_preset.value

    gui.party_preset.observe(on_party_preset_change, names="value")

    def pos_change_handler(*args) -> None:  # pylint: disable=unused-argument
        with gui.output:
            gui.compute.disabled = True
            selected: set[str] = set(gui.stop_words.value)
            frequent_words: list[str] = [
                x[0].lower()
                for x in textacy_utility.get_most_frequent_words(
                    corpus,
                    100,
                    normalize=gui.normalize.value,
                    include_pos=gui.include_pos.value,
                    # weighting=gui.weighting.value
                )
            ]

            gui.stop_words.options = frequent_words
            selected = selected & set(gui.stop_words.options)
            gui.stop_words.value = list(selected)
            gui.compute.disabled = False

    gui.include_pos.observe(pos_change_handler, "value")
    gui.weighting.observe(pos_change_handler, "value")

    def compute_callback_handler(*_args):
        gui.output.clear_output()
        with gui.output:
            try:
                gui.compute.disabled = True
                df_counts: pd.DataFrame = compute_list_of_most_frequent_words(
                    corpus=corpus,
                    gui=gui,
                    target=gui.normalize.value,
                    treaty_time_groups=treaty_time_groups,
                    group_by_column=gui.group_by_column.value,
                    parties=gui.parties.value,
                    weighting=gui.weighting.value,
                    include_pos=gui.include_pos.value,
                    stop_words=set(gui.stop_words.value),
                    display_score=gui.display_score.value,
                )
                display_list_of_most_frequent_words(gui, df_counts)
            except Exception as ex:
                logger.error(ex)
                raise
            finally:
                gui.progress.value = 0
                gui.compute.disabled = False

    gui.compute.on_click(compute_callback_handler)
    return gui
