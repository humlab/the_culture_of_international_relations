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
# ### The Rise of the Cultural Treaty (article draft, Fall 2018)
#
# #### Introduction
# Cultural treaties are the bi-lateral and multilateral agreements among states that promote and regulate cooperation and exchange in the fields of life generally call cultural or intellectual. Although it was only invented in the early twentieth century, this treaty type came to be the fourth most common bilateral treaty in the period 1900-1980 (Poast et al., 2010). In this project (funded by the Swedish Foundation for Humanities and Social Sciences), we seek to use the world's cultural treaties as a historical source with which to explore the emergence of a global concept of culture in the twentieth century. Specifically, the project will investigate the hypothesis that the culture concept, in contrast to earlier ideas of civilization, played a key role in the consolidation of the post-World War II international order.
#
# The central questions that this project explores can be divided into two groups:
# - First, what is the story of the cultural treaty, as a specific tool of international relations, in the twentieth century? What was the historical curve of cultural treaty-making? For example, in which political or ideological constellations do we find (the most) use of cultural treaties? Among which countries, in which historical periods? What networks of relations were thereby created, reinforced, or challenged?
# - Second, what is the "culture" addressed in these treaties? That is, what do the two signatories seem to mean by "culture" in these documents, and what does that tell us about the role that concept played in the international system? How can quantitative work on this dataset advance research questions about the history of concepts?
#
# I begin here with the first set of questions. It seems worthwhile to begin by getting an historical overview of the emergence and use of this treaty-type. (I have been inspired in this approach by Keene 2012.) This article thus offers a quantiative overview of the rise of the cultural treaty, as a new type of bilateral agreement, in the twentieth century. It outlines the statistical trends that can be observed, and examines and analyzes several of these. The basis of this study is a quantitative analysis of the metadata about all bilateral cultural treaties signed betweeen 1919 and 1972. The data for this investigation comes from the [World Treaty Index](http://worldtreatyindex.com/) (Poast 2010).
#
# A first question I address in the article has to do with the definition of a "cultural treaty." Here, we use the coding of the WTI. This, I have confirmed, corresponds closely to the categories developed by United Nations officials (see UNESCO 1962). The issue of the definition of this treaty type was the subject of debate in the interwar period. I address that debate in a separate, future article (Martin, "The Birth of the Cultural Treaty in Europe's Age of Crisis", under review).
#
# With the definitional question settled, let us turn to the analysis.
#
# After some set-up sections, the discussion of the material begins at "Part 1," below.

# %% code_folding=[0] language="html"
# <style>
# .jupyter-widgets { font-size: 8pt; }
# .widget-label { font-size: 8pt;}
# .widget-vbox .widget-label { width: 60px; }
# .widget-dropdown > select { font-size: 8pt; }
# .widget-select > select { font-size: 8pt; }
# .widget-toggle-buttons .widget-toggle-button {
#     font-size: 8pt;
#     width: 100px;
# }
# </style>

# %% [markdown]
# ### <span style='color:blue'>**Mandatory Prepare Step**</span>: Setup Notebook and Load and Process Treaty Master Index
# The following code cell to be executed once for each user session.

# %% code_folding=[]
import warnings

from bokeh.plotting import output_notebook

from common import config, setup_config
from common.gui.load_wti_index_gui import load_wti_index
from common.network_analysis import network_analysis_gui
from common.quantative_analysis import party_analysis_gui, topic_analysis_gui

await setup_config()

output_notebook()

warnings.filterwarnings("ignore")

# %matplotlib inline

# wti_index = treaty_state.load_wti_index('../data', filename='Treaties_Master_List_Treaties.csv', is_cultural_yesno_column='is_cultural_yesno_plus')
# wti_index = treaty_state.load_wti_index('../data', filename='Treaties_Master_List_Treaties.csv', is_cultural_yesno_column='is_cultural_yesno_org')
wti_index = load_wti_index(
    config.DATA_FOLDER, filename="Treaties_Master_List_Treaties.csv", is_cultural_yesno_column="is_cultural_yesno_gen"
)

# %% [markdown]
# ### <span style='color:blue'>**Chart 1: Cultural Treaties (broadly defined), 1919-1972**</span>
# This chart shows the number of new treaties signed per year in each of the sub-categories classified un the WTI as "cultural" (that is, under the heading 7Culture). This category, like the others used in the WTI, corresponds closely to those used by the United Nations and its institutions. “The term bilateral general cultural agreements,” according to a 1962 UNESCO publication, “is used to denote cultural conventions between two States or governments, dealing with all or several aspects of international relations in the field of education, science and culture” (UNESCO 1962, 65). Somewhat confusingly, this grouping includes one sub-category called 7CULT. This category covers both "general cultural agreements" as well as "specialized agreements" addressing such things as "cultural institutes; artistic exhibitions; protection of literary and artistic property,” regulating exchanges among the class of persons that includes "Artists (general); writers; musicians (composers, conductors, singers and instrumentalists); painters and sculptors; actors; film actors; dancers; artistes; architects; librarians; museum archivists; archaeologists" (UNESCO 1962, 18).
#
# This chart shows that the use over time of all the treaty categories included under the heading 7Culture. Three developments stand out in particular:
# * a) cultural treaties came into regular use from 1935
# * b) a first burst in their use took place in 1956
# * c) The use of cultural treaties takes off in the 1960s, reaching a peak in 1966.
# * d) Treaties in the category 7CULT predominate over the other types.
#

# %%
args = {
    "chart_per_category": False,  # Overlay charts
    "chart_type_name": "plot_stacked_bar",  # Kind of charts (see config.CHART_TYPES, attribute name for valid keys)
    "include_other_category": False,  # Add 'ALL' or 'ALL_OTHER' category
    "normalize_values": False,  # Normalize each values in each category to 100%
    "parties": ("ALL",),  # Parties to include
    "period_group_index": 0,  # Period group item index in config.DEFAULT_PERIOD_GROUPS (first item has index 0)
    "plot_style": "seaborn-v0_8-pastel",  # Matplotlib plot style (see config.yml for valid values)
    "recode_is_cultural": False,  # Flag: recode is_cultural='Yes' to topic '7CORR'
    "target_quantity": "topic",  # Target entity type: 'topic' or 'party'
    "topic_group_name": "7CULTURE",  # Topic group name (see config.DEFAULT_TOPIC_GROUPS)
    "wti_index": wti_index,  # WTI index to use, assigned in SETUP CELL above
}
topic_analysis_gui.display_topic_quantity_groups(**args)

# %% [markdown]
# ### <span style='color:blue'>**Chart 2: Cultural Treaties (narrowly defined), 1919-1972**</span>
# Removing the other sub-categories allows us to focus on the matter of the cultural treaty (which, as we saw, includes both general agreements and more narrowly "cultural" ones). This chart shows my count of the cultural treaty, for which I have made some corrections and adjustments to the treaties included under 7CULT, thus creating the new "corrected" heading 7CORR.
# This chart shows the number of new treaties signed in this category per year, 1919-1972.
#

# %%
args = {
    "chart_per_category": False,  # Overlay charts
    "chart_type_name": "plot_stacked_bar",  # Kind of charts (see config.CHART_TYPES, attribute name for valid keys)
    "include_other_category": False,  # Add 'ALL' or 'ALL_OTHER' category
    "normalize_values": False,  # Normalize each values in each category to 100%
    "parties": ("ALL",),  # Parties to include
    "period_group_index": 0,  # Period group item index in config.DEFAULT_PERIOD_GROUPS (first item has index 0)
    "plot_style": "seaborn-v0_8-pastel",  # Matplotlib plot style (see config.yml for valid values)
    "recode_is_cultural": True,  # Flag: recode is_cultural='Yes' to topic '7CORR'
    "target_quantity": "topic",  # Target entity type: 'topic' or 'party'
    "topic_group_name": "7CORR",  # Topic group name (see config.DEFAULT_TOPIC_GROUPS)
    "wti_index": wti_index,  # WTI index to use, assigned in SETUP CELL above
}
topic_analysis_gui.display_topic_quantity_groups(**args)

# %% [markdown]
# ### <span style='color:blue'>**Cultural Treaties compared to other types of agreements (Charts 3 and 4)**</span>
#
# Chart 3 shows us 7CORR compared to all other treaties, by year, 1945-1972. Chart 4 shows the same data by percentage. These show that cultural treaty was never a great portion of total treaty-making; but also that the increases in cultural treaty-making in the late 1950s and mid-1960s cannot simply be attributed to the growth in treaty-making over all. Cultural treaty-making represented around 2% of the total number of treaties signed per year in the 1940s, and rose to 6.9 % of all new bilateral agreements signed in 1966.
#

# %%
args = {
    "chart_per_category": False,  # Overlay charts
    "chart_type_name": "plot_stacked_bar",  # Kind of charts (see config.CHART_TYPES, attribute name for valid keys)
    "include_other_category": True,  # Add 'ALL' or 'ALL_OTHER' category
    "normalize_values": False,  # Normalize each values in each category to 100%
    "parties": ("ALL",),  # Parties to include
    "period_group_index": 3,  # Period group item index in config.DEFAULT_PERIOD_GROUPS (first item has index 0)
    "plot_style": "seaborn-v0_8-pastel",  # Matplotlib plot style (see config.yml for valid values)
    "recode_is_cultural": True,  # Flag: recode is_cultural='Yes' to topic '7CORR'
    "target_quantity": "topic",  # Target entity type: 'topic' or 'party'
    "topic_group_name": "7CORR",  # Topic group name (see config.DEFAULT_TOPIC_GROUPS)
    "wti_index": wti_index,  # WTI index to use, assigned in SETUP CELL above
}
topic_analysis_gui.display_topic_quantity_groups(**args)

# %% [markdown]
# ### <span style='color:blue'>**Chart 4: Cultural Treaties as a percentage of all treaty-making, per year, 1945-1972**</span>
#

# %%
args = {
    "chart_per_category": False,  # Overlay charts
    "chart_type_name": "plot_stacked_bar",  # Kind of charts (see config.CHART_TYPES, attribute name for valid keys)
    "include_other_category": True,  # Add 'ALL' or 'ALL_OTHER' category
    "normalize_values": True,  # Normalize each values in each category to 100%
    "parties": ("ALL",),  # Parties to include
    "period_group_index": 3,  # Period group item index in config.DEFAULT_PERIOD_GROUPS (first item has index 0)
    "plot_style": "seaborn-v0_8-pastel",  # Matplotlib plot style (see config.yml for valid values)
    "recode_is_cultural": True,  # Flag: recode is_cultural='Yes' to topic '7CORR'
    "target_quantity": "topic",  # Target entity type: 'topic' or 'party'
    "topic_group_name": "7CORR",  # Topic group name (see config.DEFAULT_TOPIC_GROUPS)
    "wti_index": wti_index,  # WTI index to use, assigned in SETUP CELL above
}
topic_analysis_gui.display_topic_quantity_groups(**args)

# %% [markdown]
# ### <span style='color:blue'>**Cultural Treaties vs "Diplomatic" agreements (Chart 5)**</span>
#
# Another way to explore the use of the cultural treaty is to compare it to the frequency of use of other treaty types. One might suspect, for example, that the growth in the use of cultural treaties simply reflected an increase in the number of states, in particular the emergence of many new postcolonial states around 1960. Chart 5 tests this suspicion by comparing the number of new cultural treaties to a the treaties of group of agreements typically signed with new states: these are the "diplomatic" agreements (WTI category 1), a category that includes treaties establishing diplomatic relations as well as peace treaties, treaties of friendship and mutual assistance, and territorial and reparations settlements. We see that the burst of new treaties of recognition following World War II was not accompanied by any major growth in the use of cultural treaties. After that, CT-making echoed increases in diplomatic treaty making. For example, the peak of cultural treaty-making in 1961, followed by a steep decline in 1962, clearly echoed, with a short delay, the burst of diplomatic agreements that took place in 1960, followed by a steep dropoff in 1961. From 1966-1969, however, cultural treaty-making outpaced the signing of new diplomatic agreements.
#
# After WWII, then, the cultural treaty existed already, but diplomats chose only rarely to use it in the first postwar decade, although there were many new states and, one might think, a lot of relationships in need of being reestablished. Something seems to have happened beginning in 1957--something beyond the mere apperance of new states, that is--that made the cultural treaty seem newly attractive and relevant to diplomats.
#

# %%
args = {
    "chart_per_category": False,  # Overlay charts
    "chart_type_name": "plot_line",  # Kind of charts (see config.CHART_TYPES, attribute name for valid keys)
    "include_other_category": False,  # Add 'ALL' or 'ALL_OTHER' category
    "normalize_values": False,  # Normalize each values in each category to 100%
    "parties": ("ALL",),  # Parties to include
    "period_group_index": 3,  # Period group item index in config.DEFAULT_PERIOD_GROUPS (first item has index 0)
    "plot_style": "seaborn-v0_8-pastel",  # Matplotlib plot style (see config.yml for valid values)
    "recode_is_cultural": True,  # Flag: recode is_cultural='Yes' to topic '7CORR'
    "target_quantity": "topic",  # Target entity type: 'topic' or 'party'
    "topic_group_name": "7CORR + 1DIPLOMACY",  # Topic group name (see config.DEFAULT_TOPIC_GROUPS)
    "wti_index": wti_index,  # WTI index to use, assigned in SETUP CELL above
}
topic_analysis_gui.display_topic_quantity_groups(**args)

# %% [markdown]
# ### <span style='color:blue'>**Which Countries? Top 3 signatories to Cultural Treaties by period (Chart 6)**</span>
#
# Which states entered into cultural treaties most often? This chart shows the top three countries that signed the most cultural treaties by period, where the periods are 1919-1944, 1945-1955, 1956-1966, and 1967-1972. Rather than assuming which are the great powers here, this graph allows us to see which countries in fact took the lead in the use of this particular technology of international relations. We find that a particular set of countries were leading actors in the use of the cultural treaty.
# * France was among the top three in every period.
# * During the interwar period and during World War II, Latin American countries were among the most frequent signatories of such agreements. The top three postions of Brazil and Bolivia here are largely accounted for by those states' busy treaty-making, almost exclusively with fellow Latin American states, during the war.
# * Cultural treaty-making in the decade following the war was led by three Southern European states: France, Italy, and Spain.
# * The period of "take off" in cultural treaty-making was dominated by new powers: France was joined by the Soviet Union and Egypt.
# * The Soviet Union and France maintained a leading role in the late 1960s and early 1970s, joined now by another Eastern bloc state, the German Democratic Republic.
#
# This raises questions about why these particular nations were so active. A large volume of treaty-making may normally be a sign of a state's great power status. But a large volume of cultural treaty-making seems to reflect something else. Let us compare the activity in this sector of the post-war era's other great powers.
#
# QUESTION:
# Can we do running 5-year averages, or compare to alternative time blocs (decades 1940-50, 51-60, 61-72)?
#

# %%
args = {
    "parties": (),  # Parties to include
    "treaty_filter": "is_cultural",  # Treaty filter (see config.TREATY_FILTER_OPTIONS)
    "party_name": "party_name",  # Party name to use (see config.PARTY_NAME_OPTIONS)
    "top_n_parties": 3,  # Used? Filter on top parties per category
    "extra_category": "",  # Add 'ALL' or 'ALL_OTHER' category
    "plot_style": "seaborn-v0_8-pastel",  # Matplotlib plot style (see config.yml for valid values)
    "wti_index": wti_index,  # WTI index to use, assigned in SETUP CELL above
    "period_group_index": 2,  # Period item index in config.DEFAULT_PERIOD_GROUPS (first item has index 0)
    "normalize_values": False,  # Normalize each values in each category to 100%
    "overlay": True,  # Used? Display one chart per category
    "chart_type_name": "plot_stacked_bar",  # Kind of charts (see config.CHART_TYPES, attribute name for valid keys)
    "year_limit": (1919, 1972),  # Signed year filter
}

party_analysis_gui.display_quantity_by_party(**args)

# %% [markdown]
# ### <span style='color:blue'>**Which Countries? Use of cultural treaties by the P5 (Chart 7)**</span>
#
# This chart shows the number of new cultural agreements signed by the permanent members of the United Nations Security Council, the so-called P5. This shows that two countries on the P5 accounted for the overwhelming majority of cultural agreements signed: France and the Soviet Union.
# This graph shows, too, part of what happened in 1956 to account for the burst in new cultural agreements: the USSR discovered the genre. Likewise, the burst of 1966 seems to be acccounted for largely by France.
#
# The Anglo-American bloc made quite modest use of the cultural treaty. The Soviets and the French made extensive use. Accounting for the first observation does not seem terribly difficult: the United States and British governments were evidently skeptical about the value of such agreements and chose to enter into only a few. But what is the common element that accounts for the Soviet and French states' parallel investment in this diplomatic tool?
#
# CHECK: What can we say about China here? Is this data on ROC or PRC? Compare and do sanity check of "China" vs "Taiwan" in WTI excel file, confirming source.

# %%
args = {
    "chart_per_category": False,  # Overlay charts
    "chart_type_name": "plot_line",  # Kind of charts (see config.CHART_TYPES, attribute name for valid keys)
    "include_other_category": False,  # Add 'ALL' or 'ALL_OTHER' category
    "normalize_values": False,  # Normalize each values in each category to 100%
    "parties": (
        "CHINA",
        "FRANCE",
        "USSR",
        "UK",
        "USA",
    ),  # Parties to include
    "period_group_index": 3,  # Period group item index in config.DEFAULT_PERIOD_GROUPS (first item has index 0)
    "plot_style": "seaborn-v0_8-pastel",  # Matplotlib plot style (see config.yml for valid values)
    "recode_is_cultural": True,  # Flag: recode is_cultural='Yes' to topic '7CORR'
    "target_quantity": "party",  # Target entity type: 'topic' or 'party'
    "topic_group_name": "7CORR",  # Topic group name (see config.DEFAULT_TOPIC_GROUPS)
    "wti_index": wti_index,  # WTI index to use, assigned in SETUP CELL above
}
topic_analysis_gui.display_topic_quantity_groups(**args)

# %% [markdown]
# ### <span style='color:blue'>**Cultural Treaty Networks: The Soviets (Figure 8)**</span>
#
# Having determined that France and the Soviet Union were the leading users of the cultural treaty, we can use this data to further explore _how_ they used such treaties, by visualizing the WTI data as networks.
# In the case of the USSR, the year to focus on is obviously 1956. What motivated the decision to suddenly sign thirteen cultural agreements in one year, having never signed a single one prior to this point? The Soviets of course had no lack of experience in pursuing cultural diplomacy. But some set of factors led the regime to turn to this diplomatic tool at this point. A first step to exploring this question is, naturally, to see with which countries the Soviets entered into cultural agreements.
#
# In the image below we see the Soviet Union's cultural treaties signed in 1956 visualized as a network. 1956 was the year that the regime decided for the first time to invest in the genre. This investment was evidently aimed almost exclusively at consolidating ties with countries that were firmly in the socialist orbit: the central and eastern European communist dictatorships already under Soviet control and several socialist states in Asia. And, for some reason, Norway.
#
# Observing that Syria was part of this group sends us looking for information about this. (The USSR apparently signed a broader pact with Syria that year, but this is not in WTI.) The closeness of the two regimes in 1956 triggered a major international crisis. *More on this here*
#
# A final note: this little excursus on Syrian-Soviet relations is, if I may say so, evidence of this digital method working as I hoped it would: looking with digital tools at a global data set allows us to identify items of interest--particular developments of relevance to the broader history of cultural diplomacy--that would surely have escaped my notice otherwise.

# %% code_folding=[]
args = {
    "C": 1,
    "K": 0.1,
    "height": 700,
    "layout_algorithm": "nx_spring_layout",
    "node_partition": None,
    "node_size": None,
    "node_size_range": (20, 49),
    "output": "network",
    "p1": 1.1,
    "palette_name": None,
    "parties": ("FRANCE", "USSR"),
    "party_name": "party_name",
    "period_group_index": 3,
    "plot_data": None,
    "recode_is_cultural": True,
    "topic_group": {"7CORR": "7CORR"},
    "treaty_filter": "is_cultural",
    "year_limit": (1955, 1956),
    "width": 900,
    "wti_index": wti_index,
}

network_analysis_gui.display_party_network(**args)

# %% [markdown]
# ### <span style='color:blue'>**Cultural Treaty Networks: France, part 1 (Figure 9)**</span>
#
# Let us turn now to France. The first graph below shows the network created by France's cultural treaties from 1945 to 1958.
#
# 1958 was a year when France signed no cultural treaties, which can serve as a cut off point before the new phase of treaty-making linked to decolonization (which is the subject of the next visualization).
#
# #### QUESTION: cannot get this to show me France's treaty network; it says "no data" but that's not true!

# %%
args = {
    "C": 1,
    "K": 0.1,
    "height": 700,
    "layout_algorithm": "nx_spring_layout",
    "node_partition": None,
    "node_size": None,
    "node_size_range": (20, 49),
    "output": "network",
    "p1": 1.1,
    "palette_name": None,
    "parties": ("USSR", "France"),
    "party_name": "party_name",
    "period_group_index": 3,
    "plot_data": None,
    "recode_is_cultural": True,
    "topic_group": {"7CORR": "7CORR"},
    "treaty_filter": "is_cultural",
    "year_limit": (1945, 1958),
    "width": 900,
    "wti_index": wti_index,
}

network_analysis_gui.display_party_network(**args)

# %% [markdown]
# ### <span style='color:blue'>**Cultural Treaty Networks: France, part 2 (Figure 10)**</span>
#
# This next graph shows the network created by France's cultural treaties from 1960 to 1970, at the height of African decolonization.

# %%
args = {
    "C": 1,
    "K": 0.1,
    "height": 700,
    "layout_algorithm": "nx_spring_layout",
    "node_partition": None,
    "node_size": None,
    "node_size_range": (20, 49),
    "output": "network",
    "p1": 1.1,
    "palette_name": None,
    "parties": ("France"),
    "party_name": "party_name",
    "period_group_index": 3,
    "plot_data": None,
    "recode_is_cultural": True,
    "topic_group": {"7CORR": "7CORR"},
    "treaty_filter": "is_cultural",
    "year_limit": (1960, 1970),
    "width": 900,
    "wti_index": wti_index,
}

network_analysis_gui.display_party_network(**args)

# %% [markdown]
# ### <span style='color:blue'>**Comparing Networks of different Treaty Types: (Figure 11)**</span>
#
# One might suspect that cultural treaties served as little more than window dressing for bilateral agreements that were really about something else. One way to test that theory with this data is to compare a given country's treaty networks by different treaty types. Below are visualizations of France's economic agreements in the period 1960-1970. How does this compare to the network created by France's cultural treaties signed in the same period?
#
# #### Getting "no data" here, too...not sure why.

# %% [markdown]
# import common.utility as utility
# import network_analysis_gui
# from bokeh.plotting import output_notebook
#
# output_notebook()
#
# args = {
#     'C': 1,
#     'K': 0.1,
#     'height': 700,
#     'layout_algorithm': 'nx_spring_layout',
#     'node_partition': None,
#     'node_size': None,
#     'node_size_range': (20, 49),
#     'output': 'network',
#     'p1': 1.1,
#     'palette_name': None,
#     'parties': ('France'),
#     'party_name': 'party_name',
#     'period_group_index': 3,
#     'plot_data': None,
#     'recode_is_cultural': False,
#     'topic_group': {'ECONOMIC': ['3CLAIM', '3COMMO', '3CUSTO', '3ECON', '3INDUS', '3INVES', '3MOSTF', '3PATEN', '3PAYMT', '3PROD', '3TAXAT', '3TECH', '3TOUR', '3TRADE', '3TRAPA']},
#     'treaty_filter': 'is_cultural',
#     'year_limit': (1945, 1958),
#     'width': 900,
#     'wti_index': wti_index
# }
#
# network_analysis_gui.display_party_network(**args)

# %% [markdown]
# ### <span style='color:blue'>**Conclusion**</span>
#
# Much remains to be done with cultural treaties, using these methods, and using other (more traditional) approaches. But the above analysis of the treaty metadata allows us to draw a few provisional conclusions.
#
#
#
#
#
#

# %% [markdown]
# IGNORE BELOW HERE, get rid of eventually...
#
#
# ### Chart: Treaty Quantities by Selected Parties
# This chart displays the number of treaties per time period for each party or group of parties. The time period can be year or one of a predefined set of time period divisions. The default time period division is 1919-1944, 1945-1955, 1956-1966, and 1967-1972, the alternative division has 1940-1944 as an additional period. The third division is a single period between 1945 and 1972 (inclusive)
#
# Several parties can be selected using CTRL + click on left mouse button. Shift-Up/Down can also be used to select consecutive parties. Some predefined party sets exist in the "Preset" dropdown, and, when selected, the corresponding parties is selected in the "Parties" widget. Use the "Top #n" slider to display parties with most treaties for each period. Note that a value greater than 0 will disable the "Parties" widget since these to selections are mutually exclusive. Set slider back to 0 to enable the "Parties" widget.
#
# The treaty count is based on the following rules:
# - Treaties outside of selected division are discarded
# - When "Is Cultural" is selected than the only treaties included are those having "is cultural" flag in WTI index set to "Yes"
# - When "Topic is 7CULT" is selected than all treaties having topic equal to 7CULT are included using original topic value according to WTI i.e. no recoding!
# - Candidate for removal: When "No filter" is selected then no topic filter is applied
#
# The topic filter also restricts the extra category count added when "Other category" or "All category" are selected. When "Other" is selected, than the "complement" count of the sum of all countries not included is added as a category, and the "All" category adds the count of all treaties. Note again, that both these counts are restricted by time period (no treaty outside period), and selected topic filter.
#
# Also note that when several countries are selected, than the count for a given country is the count of all treaties that country signed for each time period (filtered by topic selection). Each of the treaties that are signed *between* two of the selected parties will hence add +1 for each of the two parties.
#

# %% [markdown]
# Example on how to create a chart, in code, using a pre-defined parameter setup:
# <span style='font-size: 8pt'>
# ```python
# args = {
#     'overlay': True,
#     'top_n_parties': 0,
#     'plot_style': 'seaborn-pastel',
#     'chart_type_name': 'plot_stacked_bar',
#     'normalize_values': False,
#     'extra_category': '',
#     'treaty_filter': 'is_cultural',
#     'year_limit': (1945, 1972),
#     'parties': ('FRANCE', 'SWEDEN'),
#     'period_group_index': 3,
#     'party_name': 'party_name'
# }
# display_quantity_by_party(**args)
# ```
# </span>
#

# %% code_folding=[1]
args = {
    "parties": ("FINLAN", "FRANCE"),  # Parties to include
    "treaty_filter": "is_cultural",  # Treaty filter (see config.TREATY_FILTER_OPTIONS)
    "party_name": "party_name",  # Party name to use (see config.PARTY_NAME_OPTIONS)
    "top_n_parties": 0,  # Used? Filter on top parties per category
    "extra_category": "",  # Add 'ALL' or 'ALL_OTHER' category
    "plot_style": "seaborn-v0_8-pastel",  # Matplotlib plot style (see config.yml for valid values)
    "wti_index": wti_index,  # WTI index to use, assigned in SETUP CELL above
    "period_group_index": 3,  # Period item index in config.DEFAULT_PERIOD_GROUPS (first item has index 0)
    "normalize_values": False,  # Normalize each values in each category to 100%
    "overlay": True,  # Used? Display one chart per category
    "chart_type_name": "plot_stacked_bar",  # Kind of charts (see config.CHART_TYPES, attribute name for valid keys)
    "year_limit": (1945, 1972),  # Signed year filter
}

party_analysis_gui.display_quantity_by_party(**args)

# %% [markdown]
# ###  Chart: Treaty Quantities by Selected Topics [CHANGE!]
# This chart displays the number of treaties per selected category and time period. The category can be wither topic or party. Only treaties signed by one of the selected parties are included in the count (unless either of ALL OTHER or ALL is selected, see below).
#
# See previous chart for a description of "period" and "parties" options (the same rules apply in this chart).
#
# Topics **not** specified in the selected "Category" are as default **excluded** from the counts. If the "Add OTHER topics" toggle is set then a new "OTHER" category is added as a summed up count for topics outside the selected topic category.
#
# Note that, as default when the "Recode 7CORR" is set, all treaties marked as "is_cultural" = "Yes" in the WTI index are assigned a new topic code "7CORR". The remaining treaties ´having topic 7CULT are not recoded (will keep this code).
#
# When the "Chart per Qty" toggle is set, then a seperate chart is created for each category (of selected Quantity). Please not that this might take some time if, if the category selection contains many items. Extra charts will also be created for **excluded** category items (i.e. parties not selected or topics not included in topic category) if either of the "ALL OTHER" or "ALL" party items are selected. In both cases the category filter is applies as for any other chart.
#
# The topic categories are defined in file **.\common\config.py** in the "category_group_settings" and is common to all notebooks. A topic category has the following the structure
# ```python
# category_group_settings = {
#     'category-group-name': {
#         'category-item-1': [list-of-topic-codes-in-category-1],
#         'category-item-2': [list-of-topic-codes-in-category-2],
#         ...
#         'category-item-n': [list-of-topic-codes-in-category-n]
#     },
# ```
# The **category-group-name**s will be visible in the "Category" dropdown list, and it is hence important for these names to be unique. It can be an arbitrary name. The **category-item**s is the name that will be visible on the chart legend as label for associated list of topic-codes. The **list-of-topic-codes**, finally, must be a list of valid topic codes in the WTI-index (+ 7CORR). Note that these lists within the same category-group should be mutually exclusive i.e. the same code should not be added to more than one group (will give an unpredicted result). Also note that correct braces **must** be used, and strings **must** be surrounded by "'", and all topic codes must also be in uppercase. Follow the same syntax as in the existing specifications.

# %%
args = {
    "chart_per_category": False,  # Overlay charts
    "chart_type_name": "plot_stacked_bar",  # Kind of charts (see config.CHART_TYPES, attribute name for valid keys)
    "include_other_category": False,  # Add 'ALL' or 'ALL_OTHER' category
    "normalize_values": False,  # Normalize each values in each category to 100%
    "parties": ("FRANCE",),  # Parties to include
    "period_group_index": 3,  # Period group item index in config.DEFAULT_PERIOD_GROUPS (first item has index 0)
    "plot_style": "seaborn-v0_8-pastel",  # Matplotlib plot style (see config.yml for valid values)
    "recode_is_cultural": True,  # Flag: recode is_cultural='Yes' to topic '7CORR'
    "target_quantity": "topic",  # Target entity type: 'topic' or 'party'
    "topic_group_name": "7CULTURE",  # Topic group name (see config.DEFAULT_TOPIC_GROUPS)
    "wti_index": wti_index,  # WTI index to use, assigned in SETUP CELL above
}
topic_analysis_gui.display_topic_quantity_groups(**args)

# %% [markdown]
# Describe Network visualizations...

# %%
args = {
    "C": 1,
    "K": 0.1,
    "height": 700,
    "layout_algorithm": "nx_spring_layout",
    "node_partition": None,
    "node_size": None,
    "node_size_range": (20, 49),
    "output": "network",
    "p1": 1.1,
    "palette_name": None,
    "parties": ("FRANCE", "ITALY", "UK", "GERM"),
    "party_name": "party_name",
    "period_group_index": 3,
    "plot_data": None,
    "recode_is_cultural": True,
    "topic_group": {"7CORR": "7CORR"},
    "treaty_filter": "is_cultural",
    "year_limit": (1945, 1950),
    "width": 900,
    "wti_index": wti_index,
}

network_analysis_gui.display_party_network(**args)

# %%
