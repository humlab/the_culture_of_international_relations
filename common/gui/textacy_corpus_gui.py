import glob
import os
import types
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from loguru import logger
from textacy.corpus import Corpus
from textacy.extract.basics import entities as extract_entities
from textacy.spacier.utils import merge_spans

from common import config, domain_logic, utility, widgets_config
from common.corpus import textacy_corpus_utility as textacy_utility
from common.corpus.utility import CompressedFileReader
from common.treaty_state import ConfigValue, TreatyState


def generate_textacy_corpus(
    data_folder: str,  # noqa pylint: disable=unused-argument
    wti_index: TreatyState,
    container: textacy_utility.CorpusContainer,
    source_path: str,
    language: str,
    merge_entities: bool,
    overwrite: bool = False,
    period_group: str = "years_1935-1972",
    treaty_filter: str = "",
    parties: list[str] | None = None,
    disabled_pipes: Sequence[str] | None = None,
    tick: Callable[[int, int], None] = utility.noop,
    treaty_sources: list[str] | None = None,
):

    for key in container.__dict__:
        container.__dict__[key] = None

    nlp_args = {"disable": disabled_pipes or []}

    container.source_path = source_path
    container.language = language
    container.textacy_corpus = None
    container.prepped_source_path = utility.path_add_suffix(source_path, "_preprocessed")

    if overwrite or not os.path.isfile(container.prepped_source_path):
        textacy_utility.preprocess_text(container.source_path, container.prepped_source_path, tick=tick)

    container.textacy_corpus_path = textacy_utility.generate_corpus_filename(
        container.prepped_source_path,
        container.language,
        nlp_args=nlp_args,
        period_group=period_group,
    )

    container.nlp = textacy_utility.setup_nlp_language_model(container.language, **nlp_args)

    if overwrite or not os.path.isfile(container.textacy_corpus_path):

        logger.info("Working: Computing new corpus " + container.textacy_corpus_path + "...")
        treaties: pd.DataFrame = wti_index.get_treaties(
            language=container.language,
            period_group=period_group,
            treaty_filter=treaty_filter,
            parties=parties,
            treaty_sources=treaty_sources,
        )
        reader: CompressedFileReader = CompressedFileReader(container.prepped_source_path)

        stream = domain_logic.get_document_stream(reader, container.language, document_index=treaties)

        logger.info("Working: Stream created...")

        tick(0, len(treaties))
        container.textacy_corpus = textacy_utility.create_textacy_corpus(stream, container.nlp, tick)
        container.textacy_corpus.save(container.textacy_corpus_path)
        tick(0, 0)

    else:
        logger.info("Working: Loading corpus " + container.textacy_corpus_path + "...")
        tick(1, 2)
        container.textacy_corpus = Corpus.load(container.nlp, container.textacy_corpus_path)
        logger.info(f"Loaded corpus with {len(container.textacy_corpus)} documents.")

        tick(0, 0)

    if merge_entities:
        logger.info("Working: Merging named entities...")
        for doc in container.textacy_corpus.docs:
            named_entities: Iterable = extract_entities(doc)
            merge_spans(named_entities, doc)
    else:
        logger.info("Named entities not merged")

    logger.info("Done!")


def display_corpus_load_gui(data_folder: str, wti_index: TreatyState, container: textacy_utility.CorpusContainer):

    def lw(w):
        return widgets.Layout(width=w)

    treaty_source_options: list[str] = wti_index.unique_sources
    treaty_default_source_options: list[str] = ["LTS", "UNTS", "UNXX"]
    language_mapping: dict[str, str] = ConfigValue("data.treaty_index.language.mapping").resolve()
    language_options: dict[str, str] = {v.title(): k for k, v in language_mapping.items() if k in ["en", "fr"]}

    period_group_options: dict[str, Any] = {v["title"]: k for k, v in config.PERIOD_GROUPS_ID_MAP.items()}

    corpus_files: list[str] = list(
        sorted(
            glob.glob(os.path.join(data_folder, "treaty_text_corpora_??_*[!_preprocessed].zip")),
            key=lambda x: os.stat(x).st_mtime,
        )
    )

    if len(corpus_files) == 0:
        print("No prepared corpus")
        return None

    corpus_files = [os.path.basename(x) for x in corpus_files]

    gui = types.SimpleNamespace(
        progress=widgets.IntProgress(value=0, min=0, max=5, step=1, description="", layout=lw("90%")),
        output=widgets.Output(layout={"border": "1px solid black"}),
        source_path=widgets_config.dropdown(
            description="Corpus", options=corpus_files, value=corpus_files[-1], layout=lw("400px")
        ),
        sources=widgets_config.select_multiple(
            description="Sources",
            options=treaty_source_options,
            values=treaty_default_source_options,
            disabled=False,
            layout=lw("180px"),
        ),
        language=widgets_config.dropdown(
            description="Language", options=language_options, value="en", layout=lw("180px")
        ),
        period_group=widgets_config.dropdown(
            "Period", period_group_options, "years_1935-1972", disabled=False, layout=lw("180px")
        ),
        merge_entities=widgets_config.toggle("Merge NER", False, icon="", layout=lw("100px")),
        overwrite=widgets_config.toggle(
            "Force", False, icon="", layout=lw("100px"), tooltip="Force generation of new corpus (even if exists)"
        ),
        compute_pos=widgets_config.toggle(
            "POS", True, icon="", layout=lw("100px"), disabled=True, tooltip="Enable Part-of-Speech tagging"
        ),
        compute_ner=widgets_config.toggle(
            "NER", False, icon="", layout=lw("100px"), disabled=False, tooltip="Enable NER tagging"
        ),
        compute_dep=widgets_config.toggle(
            "DEP", False, icon="", layout=lw("100px"), disabled=True, tooltip="Enable dependency parsing"
        ),
        compute=widgets.Button(description="Compute", button_style="Success", layout=lw("100px")),
    )

    display(
        widgets.VBox(
            [
                gui.progress,
                widgets.HBox(
                    [
                        gui.source_path,
                        widgets.VBox([gui.language, gui.period_group, gui.sources]),
                        widgets.VBox([gui.merge_entities, gui.overwrite]),
                        widgets.VBox([gui.compute_pos, gui.compute_ner, gui.compute_dep]),
                        gui.compute,
                    ]
                ),
                gui.output,
            ]
        )
    )

    def tick(step: int | None = None, max_step: int | None = None) -> types.NoneType:
        if max_step is not None:
            gui.progress.max = max_step
        gui.progress.value = gui.progress.value + 1 if step is None else step

    def compute_callback(*_args):
        gui.output.clear_output()
        with gui.output:
            disabled_pipes = (
                (() if gui.compute_pos.value else ("tagger",))
                + (() if gui.compute_dep.value else ("parser",))
                + (() if gui.compute_ner.value else ("ner",))
            )
            generate_textacy_corpus(
                data_folder=data_folder,
                wti_index=wti_index,
                container=container,
                source_path=os.path.join(data_folder, gui.source_path.value),
                language=gui.language.value,
                merge_entities=gui.merge_entities.value,
                overwrite=gui.overwrite.value,
                period_group=gui.period_group.value,
                parties=None,
                disabled_pipes=tuple(disabled_pipes),
                tick=tick,
                treaty_sources=gui.sources.value,
            )

    gui.compute.on_click(compute_callback)

    return gui
