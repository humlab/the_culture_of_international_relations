# pylint: disable=no-name-in-module

import pandas as pd
from spacy.language import Language
from textacy.corpus import Corpus


class CorpusContainer:
    """Singleton class for current (last) computed or loaded corpus"""

    corpus_container = None

    class CorpusNotLoaded(Exception):
        pass

    def __init__(self):
        self.language: str | None = None
        self.source_path: str | None = None
        self.prepped_source_path: str | None = None
        self.textacy_corpus_path: str | None = None
        self.textacy_corpus: Corpus | None = None
        self.nlp: Language | None = None
        self.word_count_scores: dict[str, dict[int, set[str]]] | None = None
        self.document_index: pd.DataFrame | None = None

    # def get_word_count(self, normalize: str) -> dict[int, set[str]]:
    #     """Not used currently."""
    #     key: str = "word_count_" + normalize
    #     self.word_count_scores = self.word_count_scores or {}
    #     if key not in self.word_count_scores:
    #         assert self.textacy_corpus is not None
    #         self.word_count_scores[key] = generate_word_count_score(self.textacy_corpus, normalize, 100)
    #     return self.word_count_scores[key]

    # def get_word_document_count(self, normalize: str) -> dict[int, set[str]]:
    #     """Not used currently."""
    #     key: str = "word_document_count_" + normalize
    #     self.word_count_scores = self.word_count_scores or {}
    #     if key not in self.word_count_scores:
    #         assert self.textacy_corpus is not None
    #         self.word_count_scores[key] = generate_word_document_count_score(self.textacy_corpus, normalize, 75)
    #     return self.word_count_scores[key]

    @staticmethod
    def container() -> "CorpusContainer":

        CorpusContainer.corpus_container = CorpusContainer.corpus_container or CorpusContainer()

        return CorpusContainer.corpus_container

    @staticmethod
    def corpus() -> Corpus | None:

        class CorpusNotLoaded(Exception):
            pass

        if CorpusContainer.container().textacy_corpus is None:
            raise CorpusNotLoaded("Corpus not loaded or computed")

        return CorpusContainer.container().textacy_corpus
