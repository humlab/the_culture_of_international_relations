
import os
import re
import glob
import logging
import fnmatch
import datetime
import wordcloud
import warnings
import pandas as pd
import numpy as np
import networkx as nx
import bokeh.plotting as bp
import bokeh.palettes
import bokeh.models as bm
import bokeh.io
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib
import zipfile
import nltk.tokenize
import nltk.corpus
import gensim.models

from pivottablejs import pivot_ui
from math import sqrt
from bokeh.io import push_notebook


from IPython.display import display, HTML #, clear_output, IFrame
from IPython.core.interactiveshell import InteractiveShell

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,previewsave"

InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')
bp.output_notebook()

%run utility
#%autosave 120
%config IPCompleter.greedy=True

matplotlib_plot_styles =[
    'ggplot',
    'bmh',
    'seaborn-notebook',
    'seaborn-whitegrid',
    '_classic_test',
    'seaborn',
    'fivethirtyeight',
    'seaborn-white',
    'seaborn-dark',
    'seaborn-talk',
    'seaborn-colorblind',
    'seaborn-ticks',
    'seaborn-poster',
    'seaborn-pastel',
    'fast',
    'seaborn-darkgrid',
    'seaborn-bright',
    'Solarize_Light2',
    'seaborn-dark-palette',
    'grayscale',
    'seaborn-muted',
    'dark_background',
    'seaborn-deep',
    'seaborn-paper',
    'classic'
]

output_formats = {
    'Plot vertical bar': 'plot_bar',
    'Plot horisontal bar': 'plot_barh',
    'Plot vertical bar, stacked': 'plot_bar_stacked',
    'Plot horisontal bar, stacked': 'plot_barh_stacked',
    'Plot line': 'plot_line',
    'Plot stacked line': 'plot_line_stacked',
    # 'Chart ': 'chart',
    'Table': 'table',
    'Pivot': 'pivot'
}

toggle_style = dict(icon='', layout=widgets.Layout(width='100px', left='0'))
drop_style = dict(layout=widgets.Layout(width='260px'))
