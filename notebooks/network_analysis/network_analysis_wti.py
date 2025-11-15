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
# ### The Culture of International Relations
#
# #### About this project
# Cultural treaties are the bi-lateral and multilateral agreements among states that promote and regulate cooperation and exchange in the fields of life generally call cultural or intellectual. Although it was only invented in the early twentieth century, this treaty type came to be the fourth most common bilateral treaty in the period 1900-1980 (Poast et al., 2010). In this project, we seek to use several (mostly European) states’ cultural treaties as a historical source with which to explore the emergence of a global concept of culture in the twentieth century. Specifically, the project will investigate the hypothesis that the culture concept, in contrast to earlier ideas of civilization, played a key role in the consolidation of the post-World War II international order.
#
# The central questions that interest me here can be divided into two groups: 
# - First, what is the story of the cultural treaty, as a specific tool of international relations, in the twentieth century? What was the historical curve of cultural treaty-making? For example, in which political or ideological constellations do we find (the most) use of cultural treaties? Among which countries, in which historical periods? What networks of relations were thereby created, reinforced, or challenged? 
# - Second, what is the "culture" addressed in these treaties? That is, what do the two signatories seem to mean by "culture" in these documents, and what does that tell us about the role that concept played in the international system? How can quantitative work on this dataset advance research questions about the history of concepts?
#
# In this notebook, we deal with these treaties in three ways:
# 1) quantitative analysis of "metadata" about all bilateral cultural treaties signed betweeen 1919 and 1972, as found in the World Treaty Index or WTI (Poast et al., 2010).
#     For more on how exactly we define a "cultural treaty" here, and on other principles of selection, see... [add this, using text now in "WTI quality assurance"].
# 2) network analysis of the system of international relationships created by these treaties (using data from WTI, as above).
# 3) Text analysis of the complete texts of selected treaties. 
#
# After some set-up sections, the discussion of the material begins at "Part 1," below.

# %% [markdown]
# ### <span style='color:blue'>**Mandatory Prepare Step**</span>: Setup Notebook
# Setup HTML appearance.

# %%
from bokeh.plotting import output_notebook

from common import utility, config, treaty_state, setup_config
from common.network_analysis import network_analysis_gui
from common.gui.load_wti_index_gui import load_wti_index_with_gui, current_wti_index

await setup_config()

load_wti_index_with_gui(config.DATA_FOLDER)

current_wti_index().check_party()

output_notebook()


# %% [markdown]
# ### Party Network
#

# %%
# Visualize treaties

plot_data = utility.SimpleStruct(handle=None, nodes=None, edges=None, slice_range_type=2, slice_range=(1900, 2000))

network_analysis_gui.display_network_analyis_gui(treaty_state.current_wti_index(), plot_data)
