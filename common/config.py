
default_topic_groups = {
    '7CULT, 7SCIEN, and 7EDUC': {
        '7CULT': ['7CULT'],
        '7SCIEN': ['7SCIEN'],
        '7EDUC': ['7EDUC']
    },
    '7CULT, 7SCI, and 7EDUC+4EDUC': {
        '7CULT': ['7CULT'],
        '7SCIEN': ['7SCIEN'],
        '7EDUC+4EDUC': ['7EDUC', '4EDUC']
    },
    '7CULT + 1AMITY': {
        '7CULT': ['7CULT'],
        '1AMITY': ['1AMITY']
    },
    '7CULT + 1ALLY': {
        '7CULT': ['7CULT'],
        '1ALLY': ['1ALLY']
    },
    '7CULT + 1DIPLOMACY': {
        '7CULT': ['7CULT'],
        'DIPLOMACY': ['1ALLY', '1AMITY', '1ARMCO', '1CHART', '1DISPU', '1ESTAB', '1HEAD', '1OCCUP', '1OPTC', '1PEACE', '1RECOG', '1REPAR', '1STATU', '1TERRI', '1TRUST']
    },
    '7CULT + 2WELFARE': {
        '7CULT': ['7CULT'],
        '2WELFARE': [ '2HEW', '2HUMAN','2LABOR', '2NARK', '2REFUG', '2SANIT', '2SECUR', '2WOMEN' ]
    },
    '7CULT + 3ECONOMIC': {
        '7CULT': ['7CULT'],
        'ECONOMIC': ['3CLAIM', '3COMMO', '3CUSTO', '3ECON', '3INDUS', '3INVES', '3MOSTF', '3PATEN', '3PAYMT', '3PROD', '3TAXAT', '3TECH', '3TOUR','3TRADE','3TRAPA']
    },
    '7CULT + 4AID': {
        '7CULT': ['7CULT'],
        '4AID': ['4AGRIC','4AID', '4ATOM', '4EDUC', '4LOAN', '4MEDIC', '4MILIT', '4PCOR', '4RESOU', '4TECA', '4UNICE']
    },
    '7CULT + 5TRANSPORT': {
        '7CULT': ['7CULT'],
        '5TRANSPORT': ['5AIR', '5LAND', '5TRANS', '5WATER']
    },
    '7CULT + 6COMMUNICATIONS': {
        '7CULT': ['7CULT'],
        '6COMMUNICATIONS': ['6COMMU', '6MEDIA', '6POST', '6TELCO']
    },
    '7CULTURE': {
        '7CULT': ['7CULT'],
        '7EDUC': ['7EDUC'],
        '7RELIG': ['7RELIG'],
        '7SCIEN': ['7SCIEN'],
        '7SEMIN': ['7SEMIN'],
        '7SPACE': ['7SPACE']
    },
    '7CULT': {
        '7CULT': ['7CULT']
    },
    '7CULT + 8RESOURCES': {
        '7CULT': ['7CULT'],
        '8RESOURCES': ['8AGRIC', '8CATTL', '8ENERG', '8ENVIR', '8FISH', '8METAL', '8WATER', '8WOOD']
    },
    '7CULT + 9ADMINISTRATION': {
        '7CULT': ['7CULT'],
        '9ADMINISTRATION': ['9ADMIN', '9BOUND', '9CITIZ', '9CONSU', '9LEGAL', '9MILIT', '9MILMI', '9PRIVI', '9VISAS', '9XTRAD' ]
    },
}

topic_group_maps = { 
    group_name: { v: k for k in default_topic_groups[group_name].keys() for v in default_topic_groups[group_name][k]  }
        for group_name in default_topic_groups.keys()
}


default_graph_tools = "pan,wheel_zoom,box_zoom,reset,hover,previewsave"

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
    'Table': 'table',
    'Pivot': 'pivot'
}

import collections

class BunchOfStuff:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        
KindOfChart = collections.namedtuple('KindOfChart', 'description name kind stacked horizontal')

chart_types = [
    KindOfChart(description='Bar', name='plot_bar', kind='bar', stacked=False, horizontal=False),
    KindOfChart(description='Line', name='plot_line', kind='line', stacked=False, horizontal=False),
    KindOfChart(description='Bar (stacked)', name='plot_bar_stacked', kind='bar', stacked=True, horizontal=False),
    KindOfChart(description='Line (stacked)', name='plot_line_stacked', kind='line', stacked=True, horizontal=False),
    KindOfChart(description='Bar (horizontal)', name='plot_barh', kind='bar', stacked=False, horizontal=True),
    KindOfChart(description='Bar (horizontal, stacked)', name='plot_barh_stacked', kind='bar', stacked=True, horizontal=True),
    KindOfChart(description='Table', name='table', kind=None, stacked=False, horizontal=False),
    KindOfChart(description='Pivot', name='pivot', kind=None, stacked=False, horizontal=False)
]
chart_type_map = { x.name: x for x in chart_types }
output_charts = ([x.description for x in chart_types], chart_types)

parties_of_interest = ['FRANCE', 'GERMU', 'ITALY', 'GERMAN', 'UK', 'GERME', 'GERMW', 'INDIA', 'GERMA' ]
    
