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
# ### Brief Instructions on Jupyter Notebooks
# Please see [this tutorial](https://www.youtube.com/watch?v=h9S4kN4l5Is) for an introduction on what Jupyter notebooks are and how to use them. There are lots of other Jupyter tutorials on YouTube (and elsewhere) as well. In short, a notebook is a document with embedded executable code presented in a simple and easy to use web interface. Most important things to note are:
# - Click on the menu Help -> User Interface Tour for an overview of the Jupyter Notebook App user interface.
# - The **code cells** contains the script code (Python in this case, but can be other languages are also suported) and are the sections marked by **In [x]** in the left margin. It is marked as **In []** if it hasn't been executed, and as **In [n]** when it has been executed(n is an integer). A cell marked as **In [\*]** is either executing, or waiting to be executed (i.e. other cells are executing).
# - The **current cell** is highlighted with a blue (or green if in "edit" mode) border. You make a cell current by clicking on it,
# - Code cells aren't executed automatically. Instead you execute the current cell by either pressing **shift+enter** or the **play** button in the toolbar. The output (or result) of a cell's execution is presented directly below the cell prefixed by **Out[n]**.
# - The next cell will automatically be selected (made current) after a cell has been executed. Repeatadly pressing **shift+enter** or the play button hence executes the cells in sequence.
# - You can run the entire notebook in a single step by clicking on the menu Cell -> Run All. Note that this can take some time to finish. You can see how cells are executed in sequence via the indicator in the margin (i.e. "In [\*]" changes to "In [n]" where n is an integer).
# - The cells can be edited if they are double-clicked, in which case the cell border turns green. Use the ESC key to escape edit mode (or click on any other cell).
#
# To restart the kernel (i.e. the computational engine assigned to your session), click on the menu Kernel -> Restart. 
#

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
# The following code cell to be executed once for each user session. The step loads utility Python code stored in separate files, and imports dependencies to external libraries. The code also loads the WTI master index (and some related data files), and prepares the data for subsequent use.
#
# The treaty data is processed as follows:
# - All the treaty data are loaded.Extract year treaty was signed as seperate fields
# - Add new fields for specified signed period divisions
# - Fields 'group1' and 'group2' are ignored (many missing values). Instead group are fetched via party code from encoding found in the "groups" table.

# %% [markdown]
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
#

# %% [markdown]
# ###  Chart: Treaty Quantities by Selected Topics
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
