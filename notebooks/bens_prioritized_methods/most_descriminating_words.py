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
from common import utility, config
from common.corpus.textacy_corpus_utility import CorpusContainer
from common import setup_config
from common.gui.load_wti_index_gui import load_wti_index_with_gui, current_wti_index
from loguru import logger

await setup_config()

utility.setup_default_pd_display()

load_wti_index_with_gui(data_folder=config.DATA_FOLDER)

# %matplotlib inline

current_corpus_container = lambda: CorpusContainer.container()
current_corpus = lambda: CorpusContainer.corpus()

# %% [markdown]
# ## <span style='color: green'>PREPARE </span> Load and Prepare Corpus <span style='float: right; color: red'>MANDATORY</span>
#

# %%
from common.gui import textacy_corpus_gui

try:
    container: CorpusContainer = current_corpus_container()
    textacy_corpus_gui.display_corpus_load_gui(config.DATA_FOLDER, current_wti_index(), container)
except Exception as ex:
    logger.error(ex)

# %% [markdown]
# ### <span style='color: green;'>DESCRIBE</span> Most Discriminating Terms<span style='color: blue; float: right'>OPTIONAL</span>
# References
# King, Gary, Patrick Lam, and Margaret Roberts. “Computer-Assisted Keyword and Document Set Discovery from Unstructured Text.” (2014). http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.1445&rep=rep1&type=pdf.
#
# Displays the *most discriminating words* between two sets of treaties. Each treaty group can be filtered by country and period (signed year). In this way, the same group of countries can be studied for different time periods, or different groups of countries can be studied for the same time period. If "Closed region" is checked then **both** parties must be to the selected set of countries, from each region. In this way, one can for instance compare treaties signed between countries within the WTI group "Communists", against treaties signed within "Western Europe". 
#
# <b>#terms</b> The number of most discriminating terms to return for each group.<br>
# <b>#top</b> Only terms with a frequency within the top #top terms out of all terms<br>
# <b>Closed region</b> If checked, then <u>both</u> treaty parties must be within selected region

# %%
from common.gui import most_discriminating_terms_gui
try:
    most_discriminating_terms_gui.display_gui(current_wti_index(), current_corpus())
except Exception as ex:
    logger.error(ex)
