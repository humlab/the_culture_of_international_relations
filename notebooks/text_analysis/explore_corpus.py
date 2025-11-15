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
# ## The Culture of International Relations - Text Analysis
# ### <span style='color: green'>SETUP </span> Prepare and Setup Notebook <span style='float: right; color: red'>MANDATORY</span>

# %% code_folding=[]

import os

from loguru import logger

from common import config, utility
from common.corpus import textacy_corpus_utility as textacy_utility
from common.gui import textacy_corpus_gui
from common.gui.load_wti_index_gui import current_wti_index, load_wti_index_with_gui
from notebooks.text_analysis.src.cleanup_gui import display_cleanup_text_gui
from common.gui import most_discriminating_terms_gui
from common import setup_config

await setup_config()

utility.setup_default_pd_display()

PATTERN = '*.txt'
PERIOD_GROUP = 'years_1945-1972'

load_wti_index_with_gui(data_folder=config.DATA_FOLDER)

# %matplotlib inline

current_corpus_container = lambda: textacy_utility.CorpusContainer.container()  # type: ignore
current_corpus = lambda: textacy_utility.CorpusContainer.corpus()  # type: ignore


# %% [markdown]
# ## <span style='color: green'>PREPARE </span> Load and Prepare Corpus <span style='float: right; color: red'>MANDATORY</span>
#
#

# %%
container: textacy_utility.CorpusContainer = current_corpus_container()
textacy_corpus_gui.display_corpus_load_gui(config.DATA_FOLDER, current_wti_index(), container)


# %% [markdown]
# ## <span style='color: green'>PREPARE/DESCRIBE </span> Find Key Terms <span style='float: right; color: green'>OPTIONAL</span>
# - [TextRank]	Mihalcea, R., & Tarau, P. (2004, July). TextRank: Bringing order into texts. Association for Computational Linguistics.
# - [SingleRank]	Hasan, K. S., & Ng, V. (2010, August). Conundrums in unsupervised keyphrase extraction: making sense of the state-of-the-art. In Proceedings of the 23rd International Conference on Computational Linguistics: Posters (pp. 365-373). Association for Computational Linguistics.
# - [RAKE]	Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010). Automatic Keyword Extraction from Individual Documents. In M. W. Berry & J. Kogan (Eds.), Text Mining: Theory and Applications: John Wiley & Son
# https://github.com/csurfer/rake-nltk
# https://github.com/aneesha/RAKE
# https://github.com/vgrabovets/multi_rake
#
#

# %% [markdown]
# #### <span style='color: green'>PREPARE/DESCRIBE </span>RAKE <span style='float: right; color: green'>WORK IN PROGRESS</span>
#
# https://github.com/JRC1995/RAKE-Keyword-Extraction
# https://github.com/JRC1995/TextRank-Keyword-Extraction
#
#

# %% code_folding=[0]
from notebooks.text_analysis.src import rake_gui

rake_gui.display_rake_gui(current_corpus(), language="english")


# %% [markdown]
# #### <span style='color: green'>PREPARE/DESCRIBE </span>TextRank/SingleRank <span style='float: right; color: green'>OPTIONAL</span>
#
# https://github.com/JRC1995/TextRank-Keyword-Extraction
#   

# %% code_folding=[0]
from notebooks.text_analysis.src.rake_key_terms_gui import display_document_key_terms_gui

display_document_key_terms_gui(current_corpus(), current_wti_index())


# %% [markdown]
# ## <span style='color: green'>PREPARE/DESCRIBE </span> Clean Up the Text <span style='float: right; color: green'>TRY IT</span>

# %% code_folding=[]
xgui, xuix = display_cleanup_text_gui(current_corpus_container(), current_wti_index())


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
most_discriminating_terms_gui.display_gui(current_wti_index(), current_corpus())

# %% [markdown]
# ## <span style='color: green;'>DESCRIBE</span> Corpus Statistics<span style='color: blue; float: right'>OPTIONAL</span>

# %% [markdown]
# ### <span style='color: green;'>DESCRIBE</span> List of Most Frequent Words<span style='color: blue; float: right'>OPTIONAL</span>

# %%
from common.gui import word_frequencies_gui

word_frequencies_gui.word_frequency_gui(current_wti_index(), current_corpus())

# %% [markdown]
# ### <span style='color: green;'>DESCRIBE</span> Corpus and Document Sizes<span style='color: blue; float: right'>OPTIONAL</span>

# %%
from notebooks.text_analysis.src.corpus_statistics_gui import (
    compute_corpus_statistics,
    corpus_statistics_gui,
    display_corpus_statistics,
)

gui = corpus_statistics_gui(
    config.DATA_FOLDER,
    current_wti_index(),
    current_corpus_container(),
    compute_callback=compute_corpus_statistics,
    display_callback=display_corpus_statistics,
)

# %%
# Create and export region vs region MDT files as Excel Spreadsheets

import spacy


def create_mdt(group1, group2, include_pos, closed_region):
    corpus = current_corpus()
    assert corpus is not None, "Please load corpus!"

    most_discriminating_terms_gui.compute_most_discriminating_terms(
        current_wti_index(),
        corpus,
        group1=config.get_region_parties(*group1),
        group2=config.get_region_parties(*group2),
        top_n_terms=100,
        max_n_terms=2000,
        include_pos=include_pos,
        period1=(1945, 1972),
        period2=(1945, 1972),
        closed_region=closed_region,
        normalize=spacy.attrs.LEMMA,  # type: ignore
        output_filename=os.path.join(
            config.DATA_FOLDER,
            "MDT_{}_vs_{}_({})_{}.xlsx".format(
                "+".join(["R{}".format(x) for x in group1]),
                "+".join(["R{}".format(x) for x in group2]),
                ",".join(include_pos),
                "CLOSED" if closed_region else "OPEN",
            ),
        ),
    )


include_pos: list[str] = ["ADJ", "VERB", "NOUN"]

create_mdt((1,), (2, 3), include_pos, True)
create_mdt((2,), (1, 3), include_pos, True)
create_mdt((3,), (1, 2), include_pos, True)
create_mdt((1,), (2, 3), include_pos, False)
create_mdt((2,), (1, 3), include_pos, False)
create_mdt((3,), (1, 2), include_pos, False)
