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
#     display_name: the-culture-of-international-relations
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## The Culture of International Relations - Text Analysis
# ### <span style='color: green'>SETUP </span> Prepare and Setup Notebook <span style='float: right; color: red'>MANDATORY</span>

# %% code_folding=[]

from loguru import logger

from common import config, setup_config
from common.corpus import textacy_corpus_utility as textacy_utility
from common.gui import textacy_corpus_gui
from common.gui import word_in_doc_frequencies_gui as widfgui
from common.gui.load_wti_index_gui import load_wti_index_with_gui, current_wti_index

await setup_config()

load_wti_index_with_gui(data_folder=config.DATA_FOLDER)

# %matplotlib inline

current_corpus_container = lambda: textacy_utility.CorpusContainer.container()
current_corpus = lambda: textacy_utility.CorpusContainer.corpus()

container: textacy_utility.CorpusContainer = current_corpus_container()
textacy_corpus_gui.display_corpus_load_gui(config.DATA_FOLDER, current_wti_index(), container);


# %% [markdown]
# ## Compute word in document per year frequencies
#
# N.B. *All* documents in the selected text corpus are used in this computation (WTI index selection above has no effect in this notebook)!
#

# %%

try:
    widfgui.word_doc_frequencies_gui(current_corpus())
except Exception as ex:
    logger.error(ex)
