from collections.abc import Sequence

import nltk
import pandas as pd
from loguru import logger

from common import utility

uniquify = utility.uniquify


class HeadnoteTokenCorpus:

    def __init__(
        self, treaties, tokenize=None, stopwords=None, lemmatize=None, min_word_length=2, ignore_word_count=True
    ):

        logger.info("Please wait, preparing headnote corpus...")

        self.min_word_length = min_word_length

        tokenize = tokenize or nltk.tokenize.word_tokenize
        lemmatize = lemmatize or nltk.stem.WordNetLemmatizer().lemmatize
        stopwords = stopwords or nltk.corpus.stopwords.words("english")

        transforms = [
            lambda ws: (x for x in ws if len(x) >= min_word_length),
            lambda ws: (x for x in ws if any(ch.isalpha() for ch in x)),
        ]

        if ignore_word_count:
            transforms = transforms + [uniquify]

        headnote_tokens = treaties.headnote.str.lower().apply(tokenize)
        for f in transforms:
            headnote_tokens = headnote_tokens.apply(f)

        self.headnote_tokens = pd.DataFrame({"tokens": headnote_tokens})

        tokens: pd.DataFrame = (
            pd.DataFrame(self.headnote_tokens.tokens.tolist(), index=self.headnote_tokens.index)
            .stack()
            .reset_index()
            .rename(columns={"level_1": "sequence_id", 0: "token"})
        )

        self.vocabulary: pd.DataFrame = pd.DataFrame({"token": tokens.token.unique()})
        self.vocabulary["lemma"] = self.vocabulary.token.apply(lemmatize)
        self.vocabulary["is_stopword"] = self.vocabulary.token.apply(lambda x: x in stopwords)
        self.vocabulary = self.vocabulary.set_index("token")

        self.tokens: pd.DataFrame = tokens.merge(
            self.vocabulary, left_on="token", right_index=True, how="inner"
        ).set_index(["treaty_id", "sequence_id"])

        logger.info("Done!")

    def get_tokens_for(self, treaty_ids: Sequence[int] | pd.Index) -> pd.DataFrame:
        """Returns tokens for given list of treaty ids

        Parameters:
        ----------
         treaty_ids : str list (or Pandas Index)
             List of treaty ids.
        Returns:
        --------
          Subset of self.tokens for given treaty ids, with stopwords removed.
        """
        tokens: pd.DataFrame = self.tokens

        if treaty_ids is not None:
            tokens = tokens[tokens.index.get_level_values(0).isin(treaty_ids)]

        tokens = tokens.loc[~tokens.is_stopword]

        return tokens
