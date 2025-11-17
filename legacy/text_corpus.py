import fnmatch
import logging
import os
import re
from typing import Any, Callable, Generator, Self
import zipfile

import gensim
import nltk
import numpy as np
import pandas as pd
from gensim.corpora.textcorpus import TextCorpus

logger = logging.getLogger(__name__)

HYPHEN_REGEXP: re.Pattern = re.compile(r"\b(\w+)-\s*\r?\n\s*(\w+)\b", re.UNICODE)


def dehyphen(text: str) -> str:
    result: str = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
    return result


def list_archive_files(archivename: str, pattern: str | re.Pattern) -> list[str]:
    def px(x: str) -> bool:
        if isinstance(pattern, str):
            return fnmatch.fnmatch(x, pattern)
        return  bool(pattern.match(x))

    with zipfile.ZipFile(archivename) as zf:
        return [name for name in zf.namelist() if px(name)]


class CompressedFileReader:

    def __init__(self, path: str, pattern: str = "*.txt", itemfilter: str | None = None) -> None:
        self.path: str = path
        self.filename_pattern: str = pattern
        self.archive_filenames: list[str] = list_archive_files(path, pattern)
        filenames: list[str] | None = None
        if itemfilter is not None:
            if isinstance(itemfilter, list):
                filenames = [x for x in itemfilter if x in self.archive_filenames]
            elif callable(itemfilter):
                filenames = [x for x in self.archive_filenames if itemfilter(x)]
            else:
                assert False
        self.filenames: list[str] = filenames or self.archive_filenames
        self.iterator: Generator[tuple[str, str], Any, None] | None = None

    def __iter__(self) -> Self:
        self.iterator = None
        return self

    def __next__(self) -> tuple[Any | str, str]:
        if self.iterator is None:
            self.iterator = self.get_iterator()
        return next(self.iterator)

    def get_file(self, filename: str):

        if filename not in self.filenames:
            yield os.path.basename(filename), None

        with zipfile.ZipFile(self.path) as zip_file:
            yield os.path.basename(filename), self._read_content(zip_file, filename)

    def get_iterator(self) -> Generator[tuple[str, str], Any, None]:
        with zipfile.ZipFile(self.path) as zip_file:
            for filename in self.filenames:
                yield os.path.basename(filename), self._read_content(zip_file, filename)

    def _read_content(self, zip_file: zipfile.ZipFile, filename: str) -> str:
        with zip_file.open(filename, "r") as text_file:
            byte_str: bytes = text_file.read()
            content: str = gensim.utils.to_unicode(byte_str, "utf8", errors="ignore")
            content = dehyphen(content)
            return content


class GenericTextCorpus(TextCorpus):

    def __init__(
        self,
        stream: Generator[tuple[str, str], Any, None],
        dictionary=None,
        metadata: bool=False,
        character_filters=None,
        tokenizer=None,
        token_filters: list[Callable[[list[str]], list[Any]]]|None=None,
        bigram_transform: bool = False,  # pylint: disable=unused-argument
    ) -> None: 
        self.stream = stream
        self.filenames: list[str] | None = None
        self.documents: pd.DataFrame | None = None
        self.length: int | None = None

        # if 'filenames' in content_iterator.__dict__:
        #    self.filenames = content_iterator.filenames
        #    self.document_names = self._compile_documents()
        #    self.length = len(self.filenames)

        token_filters = self.default_token_filters() + (token_filters or [])

        # if bigram_transform is True:
        #    train_corpus = GenericTextCorpus(content_iterator, token_filters=[ x.lower() for x in tokens ])
        #    phrases = gensim.models.phrases.Phrases(train_corpus)
        #    bigram = gensim.models.phrases.Phraser(phrases)
        #    token_filters.append(
        #        lambda tokens: bigram[tokens]
        #    )

        super().__init__(
            input=True,
            dictionary=dictionary,
            metadata=metadata,
            character_filters=character_filters,
            tokenizer=tokenizer,
            token_filters=token_filters,
        )

    def default_token_filters(self) -> list[Callable[[list[str]], list[Any]]]:
        return [
            (lambda tokens: [x.lower() for x in tokens]),
            (lambda tokens: [x for x in tokens if any(map(lambda x: x.isalpha(), x))]),
        ]

    def getstream(self):
        """Generate documents from the underlying plain text collection (of one or more files).
        Yields
        ------
        str
            Document read from plain-text file.
        Notes
        -----
        After generator end - initialize self.length attribute.
        """

        document_infos: list[dict[str, str]] = []
        for filename, content in self.stream:
            yield content
            document_infos.append({"document_name": filename})

        self.length = len(document_infos)
        self.documents = pd.DataFrame(document_infos)
        self.filenames = list(self.documents.document_name.values)

    def get_texts(self) -> Generator[list[str], None, None]:
        """
        This is mandatory method from gensim.corpora.TextCorpus. Returns stream of documents.
        """
        for document in self.getstream():
            yield self.preprocess_text(document)

    def preprocess_text(self, text: str) -> list[str]:
        """Apply `self.character_filters`, `self.tokenizer`, `self.token_filters` to a single text document.

        Parameters
        ---------
        text : str
            Document read from plain-text file.

        Returns
        ------
        list of str
            List of tokens extracted from `text`.

        """
        for character_filter in self.character_filters:
            text = character_filter(text)

        tokens: list[str] = self.tokenizer(text)
        for token_filter in self.token_filters:
            tokens = token_filter(tokens)

        return tokens

    # def __get_document_info(self, filename):
    #     return {
    #         'document_name': filename,
    #     }

    # def ___compile_documents(self):

    #     document_data = map(self.get_document_info, self.filenames)

    #     documents = pd.DataFrame(list(document_data))
    #     documents.index.names = ['document_id']

    #     return documents


class SimplePreparedTextCorpus(GenericTextCorpus):
    """Reads content in stream and returns tokenized text. No other processing."""

    def __init__(self, source, lowercase=False):

        self.reader = CompressedFileReader(source)
        self.filenames = self.reader.filenames
        self.lowercase = lowercase
        source = self.reader
        super(SimplePreparedTextCorpus, self).__init__(source)

    def default_token_filters(self):
        return []

    def preprocess_text(self, text):
        if self.lowercase:
            text = text.lower()
        return self.tokenizer(text)


class MmCorpusStatisticsService:

    def __init__(self, corpus, dictionary, language):
        self.corpus = corpus
        self.dictionary = dictionary
        self.stopwords = nltk.corpus.stopwords.words(language[1])
        _ = dictionary[0]

    def get_total_token_frequencies(self):
        dictionary = self.corpus.dictionary
        freqencies = np.zeros(len(dictionary.id2token))
        for document in self.corpus:
            for i, f in document:
                freqencies[i] += f
        return freqencies

    def get_document_token_frequencies(self):
        """
        Returns a DataFrame with per document token frequencies i.e. "melts" doc-term matrix
        """
        data = ((document_id, x[0], x[1]) for document_id, values in enumerate(self.corpus) for x in values)
        df: pd.DataFrame = pd.DataFrame(list(zip(*data)), columns=["document_id", "token_id", "count"])
        df = pd.merge(df, self.corpus.document_names, left_on="document_id", right_index=True)

        return df

    def compute_word_frequencies(self, remove_stopwords):
        id2token = self.dictionary.id2token
        term_freqencies = np.zeros(len(id2token))
        for document in self.corpus:
            for i, f in document:
                term_freqencies[i] += f
        stopwords = set(self.stopwords).intersection(set(id2token.values()))
        df = pd.DataFrame(
            {
                "token_id": list(id2token.keys()),
                "token": list(id2token.values()),
                "frequency": term_freqencies,
                "dfs": list(self.dictionary.dfs.values()),
            }
        )
        df["is_stopword"] = df.token.apply(lambda x: x in stopwords)
        if remove_stopwords is True:
            df = df.loc[(df.is_stopword == False)]
        df["frequency"] = df.frequency.astype(np.int64)
        df = df[["token_id", "token", "frequency", "dfs", "is_stopword"]].sort_values("frequency", ascending=False)
        return df.set_index("token_id")

    def compute_document_stats(self):
        id2token = self.dictionary.id2token
        # stopwords = set(self.stopwords).intersection(set(id2token.values()))
        df: pd.DataFrame = pd.DataFrame(
            {
                "document_id": self.corpus.index,
                "document_name": self.corpus.document_names.document_name,
                "treaty_id": self.corpus.document_names.treaty_id,
                "size": [sum(list(zip(*document))[1]) for document in self.corpus],
                "stopwords": [
                    sum([v for (i, v) in document if id2token[i] in self.stopwords]) for document in self.corpus
                ],
            }
        ).set_index("document_name")
        df[["size", "stopwords"]] = df[["size", "stopwords"]].astype("int")
        return df

    def compute_word_stats(self):
        df = self.compute_document_stats()[["size", "stopwords"]]
        df_agg = df.agg(["count", "mean", "std", "min", "median", "max", "sum"]).reset_index()
        legend_map = {
            "count": "Documents",
            "mean": "Mean words",
            "std": "Std",
            "min": "Min",
            "median": "Median",
            "max": "Max",
            "sum": "Sum words",
        }
        df_agg["index"] = df_agg["index"].apply(lambda x: legend_map[x]).astype("str")
        df_agg = df_agg.set_index("index")
        df_agg[df_agg.columns] = df_agg[df_agg.columns].astype("int")
        return df_agg.reset_index()


# @staticmethod


class ExtMmCorpus(gensim.corpora.MmCorpus):
    """Extension of MmCorpus that allow TF normalization based on document length."""

    @staticmethod
    def norm_tf_by_D(doc):
        D = sum([x[1] for x in doc])
        return doc if D == 0 else map(lambda tf: (tf[0], tf[1] / D), doc)

    def __init__(self, fname):
        gensim.corpora.MmCorpus.__init__(self, fname)

    def __iter__(self):
        for doc in gensim.corpora.MmCorpus.__iter__(self):
            yield self.norm_tf_by_D(doc)

    def __getitem__(self, docno):
        return self.norm_tf_by_D(gensim.corpora.MmCorpus.__getitem__(self, docno))


class GenericCorpusSaveLoad:

    def __init__(self, source_folder, lang):

        self.mm_filename = os.path.join(source_folder, "corpus_{}.mm".format(lang))
        self.dict_filename = os.path.join(source_folder, "corpus_{}.dict.gz".format(lang))
        self.document_index = os.path.join(source_folder, "corpus_{}_documents.csv".format(lang))

    def store_as_mm_corpus(self, corpus):

        gensim.corpora.MmCorpus.serialize(self.mm_filename, corpus, id2word=corpus.dictionary.id2token)
        corpus.dictionary.save(self.dict_filename)
        corpus.document_names.to_csv(self.document_index, sep="\t")

    def load_mm_corpus(self, normalize_by_D=False):

        corpus_type = ExtMmCorpus if normalize_by_D else gensim.corpora.MmCorpus
        corpus = corpus_type(self.mm_filename)
        corpus.dictionary = gensim.corpora.Dictionary.load(self.dict_filename)
        corpus.document_names = pd.read_csv(self.document_index, sep="\t").set_index("document_id")

        return corpus

    def exists(self):
        return (
            os.path.isfile(self.mm_filename)
            and os.path.isfile(self.dict_filename)
            and os.path.isfile(self.document_index)
        )
