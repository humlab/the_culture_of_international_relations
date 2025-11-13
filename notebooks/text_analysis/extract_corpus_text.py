# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## tCoIR - Text Analysis
# ### <span style='color: green'>SETUP </span> Prepare and Setup Notebook <span style='float: right; color: red'>MANDATORY</span>

# %% code_folding=[]

import os

from loguru import logger

from common import config, treaty_state
from common.corpus import textacy_corpus_utility as textacy_utility
from notebooks.text_analysis.src.extract_text_gui import display_generate_tokenized_corpus_gui

treaty_state.load_wti_index_with_gui(data_folder=config.DATA_FOLDER)

# %matplotlib inline

current_corpus_container = lambda: textacy_utility.CorpusContainer.container()
current_corpus = lambda: textacy_utility.CorpusContainer.corpus()


# %% [markdown]
# ## <span style='color: green'>PREPARE </span> Load and Prepare Corpus <span style='float: right; color: red'>MANDATORY</span>
#

# %%
from common.gui import textacy_corpus_gui

try:
    container = current_corpus_container()
    textacy_corpus_gui.display_corpus_load_gui(config.DATA_FOLDER, treaty_state.current_wti_index(), container)
except Exception as ex:
    raise


# %% [markdown]
# ## <span style='color: green'>PREPARE </span> Extract Text From Corpus <span style='float: right; color: green'>TRY IT</span>

# %% code_folding=[]


try:
    subst_filename = os.path.join(config.DATA_FOLDER, 'term_substitutions.txt')
    corpus = current_corpus_container().textacy_corpus
    corpus_path =  current_corpus_container().prepped_source_path
    if corpus is None:
        logger.info('Please load corpus!')
    else:
        display_generate_tokenized_corpus_gui(corpus, corpus_path)
except Exception as ex:
    raise


# %%
