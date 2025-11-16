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
from common import config, setup_config
from common.corpus.corpus_utility import CorpusContainer
from common.gui import load_corpus_gui, word_frequencies_gui
from common.gui.load_wti_index_gui import current_wti_index, load_wti_index_with_gui

await setup_config()

load_wti_index_with_gui(data_folder=config.DATA_FOLDER)

# %matplotlib inline

current_corpus_container = lambda: CorpusContainer.container()
current_corpus = lambda: CorpusContainer.corpus()

# %% [markdown]
# ## <span style='color: green'>PREPARE </span> Load and Prepare Corpus <span style='float: right; color: red'>MANDATORY</span>
#

# %%
container: CorpusContainer = current_corpus_container()
load_corpus_gui.display_corpus_load_gui(config.DATA_FOLDER, current_wti_index(), container)

# %% [markdown]
# ### <span style='color: green;'>DESCRIBE</span> List of Most Frequent Words<span style='color: blue; float: right'>OPTIONAL</span>

# %%
word_frequencies_gui.word_frequency_gui(current_wti_index(), current_corpus())

# %%
