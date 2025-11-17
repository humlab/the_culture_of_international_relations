import os
from collections.abc import Callable, Iterable, Sequence

import pandas as pd
from loguru import logger
from textacy.corpus import Corpus
from textacy.extract.basics import entities as extract_entities
from textacy.spacier.utils import merge_spans

from common.corpus import corpus_utility
from common.corpus.container import CorpusContainer
from common.corpus.utility import CompressedFileReader, get_document_stream
from common.treaty_state import TreatyState
from common.utility import noop, path_add_suffix


def generate_corpus(
    data_folder: str,  # noqa pylint: disable=unused-argument
    wti_index: TreatyState,
    container: CorpusContainer,
    source_path: str,
    language: str,
    merge_entities: bool,
    overwrite: bool = False,
    period_group: str = "years_1935-1972",
    treaty_filter: str = "",
    parties: list[str] | None = None,
    disabled_pipes: Sequence[str] | None = None,
    tick: Callable[[int, int], None] = noop,
    treaty_sources: list[str] | None = None,
):

    for key in container.__dict__:
        container.__dict__[key] = None

    nlp_args = {"disable": disabled_pipes or []}

    container.source_path = source_path
    container.language = language
    container.textacy_corpus = None
    container.prepped_source_path = path_add_suffix(source_path, "_preprocessed")

    if overwrite or not os.path.isfile(container.prepped_source_path):
        corpus_utility.preprocess_text(container.source_path, container.prepped_source_path, tick=tick)

    container.textacy_corpus_path = corpus_utility.generate_corpus_filename(
        container.prepped_source_path,
        container.language,
        nlp_args=nlp_args,
        period_group=period_group,
    )

    container.nlp = corpus_utility.setup_nlp_language_model(container.language, **nlp_args)

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

        stream = get_document_stream(reader, container.language, document_index=treaties)

        logger.info("Working: Stream created...")

        tick(0, len(treaties))
        container.textacy_corpus = corpus_utility.create_textacy_corpus(stream, container.nlp, tick)
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
