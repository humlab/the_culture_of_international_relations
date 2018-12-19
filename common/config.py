import re
import numpy as np

LANGUAGE_MAP = { 'en': 'english', 'fr': 'french', 'it': 'other', 'de': 'other' }

AGGREGATES = { 'mean': np.mean, 'sum': np.sum, 'max': np.max, 'std': np.std }

HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

DEFAULT_PERIOD_GROUPS = [
    {
        'id': 'years_1919-1972',
        'title': '1919-1972',
        'column': 'signed_year',
        'type': 'range',
        'periods': list(range(1919, 1973))
    },
    {
        'id': 'default_division',
        'title': 'Default division',
        'column': 'signed_period',
        'type': 'divisions',
        'periods': [ (1919, 1939), (1940, 1944), (1945, 1955), (1956, 1966), (1967, 1972) ]
    },
    {
        'id': 'alt_default_division',
        'title': 'Alt. division',
        'column': 'signed_period_alt',
        'type': 'divisions',
        'periods': [ (1919, 1944), (1945, 1955), (1956, 1966), (1967, 1972) ]
    },
    {
        'id': 'years_1945-1972',
        'title': '1945-1972',
        'column': 'signed_year',
        'type': 'range',
        'periods': list(range(1945, 1973))
    }
]

PERIOD_GROUPS_ID_MAP = { x['id']: x for x in DEFAULT_PERIOD_GROUPS }

DEFAULT_TOPIC_GROUPS = {
    '7CULT, 7SCIEN, and 7EDUC': {
        '7CULT': ['7CULT', '7CORR'],
        '7SCIEN': ['7SCIEN'],
        '7EDUC': ['7EDUC']
    },
    '7CULT, 7SCI, and 7EDUC+4EDUC': {
        '7CULT': ['7CULT', '7CORR'],
        '7SCIEN': ['7SCIEN'],
        '7EDUC+4EDUC': ['7EDUC', '4EDUC']
    },
    '7CULT + 1AMITY': {
        '7CULT': ['7CULT', '7CORR'],
        '1AMITY': ['1AMITY']
    },
    '7CULT + 1ALLY': {
        '7CULT': ['7CULT', '7CORR'],
        '1ALLY': ['1ALLY']
    },
    '7CORR + 1DIPLOMACY': {
        '7CORR': ['7CORR'],
        'DIPLOMACY': ['1ALLY', '1AMITY', '1ARMCO', '1CHART', '1DISPU', '1ESTAB', '1HEAD', '1OCCUP', '1OPTC', '1PEACE', '1RECOG', '1REPAR', '1STATU', '1TERRI', '1TRUST']
    },    
    '7CULT + 1DIPLOMACY': {
        '7CULT': ['7CULT', '7CORR'],
        'DIPLOMACY': ['1ALLY', '1AMITY', '1ARMCO', '1CHART', '1DISPU', '1ESTAB', '1HEAD', '1OCCUP', '1OPTC', '1PEACE', '1RECOG', '1REPAR', '1STATU', '1TERRI', '1TRUST']
    },
    '7CULT + 2WELFARE': {
        '7CULT': ['7CULT', '7CORR'],
        '2WELFARE': [ '2HEW', '2HUMAN', '2LABOR', '2NARK', '2REFUG', '2SANIT', '2SECUR', '2WOMEN' ]
    },
    '7CULT + 3ECONOMIC': {
        '7CULT': ['7CULT', '7CORR'],
        'ECONOMIC': ['3CLAIM', '3COMMO', '3CUSTO', '3ECON', '3INDUS', '3INVES', '3MOSTF', '3PATEN', '3PAYMT', '3PROD', '3TAXAT', '3TECH', '3TOUR', '3TRADE', '3TRAPA']
    },
    '7CULT + 4AID': {
        '7CULT': ['7CULT', '7CORR'],
        '4AID': ['4AGRIC', '4AID', '4ATOM', '4EDUC', '4LOAN', '4MEDIC', '4MILIT', '4PCOR', '4RESOU', '4TECA', '4UNICE']
    },
    '7CULT + 5TRANSPORT': {
        '7CULT': ['7CULT', '7CORR'],
        '5TRANSPORT': ['5AIR', '5LAND', '5TRANS', '5WATER']
    },
    '7CULT + 6COMMUNICATIONS': {
        '7CULT': ['7CULT', '7CORR'],
        '6COMMUNICATIONS': ['6COMMU', '6MEDIA', '6POST', '6TELCO']
    },
    '7CULTURE': {
        '7CORR': ['7CORR'],
        '7CULT': ['7CULT'],
        '7EDUC': ['7EDUC'],
        '7RELIG': ['7RELIG'],
        '7SCIEN': ['7SCIEN'],
        '7SEMIN': ['7SEMIN'],
        '7SPACE': ['7SPACE']
    },
    '7CORR': {
        '7CORR': ['7CORR']
    },
    '7CULT': {
        '7CULT': ['7CULT', '7CORR']
    },
    '7CULT + 8RESOURCES': {
        '7CULT': ['7CULT', '7CORR'],
        '8RESOURCES': ['8AGRIC', '8CATTL', '8ENERG', '8ENVIR', '8FISH', '8METAL', '8WATER', '8WOOD']
    },
    '7CULT + 9ADMINISTRATION': {
        '7CULT': ['7CULT', '7CORR'],
        '9ADMINISTRATION': ['9ADMIN', '9BOUND', '9CITIZ', '9CONSU', '9LEGAL', '9MILIT', '9MILMI', '9PRIVI', '9VISAS', '9XTRAD' ]
    },
}

TOPIC_GROUP_MAPS = {
    group_name: {
        v: k for k in DEFAULT_TOPIC_GROUPS[group_name].keys() for v in DEFAULT_TOPIC_GROUPS[group_name][k]
    } for group_name in DEFAULT_TOPIC_GROUPS.keys()
}

default_graph_tools = "pan,wheel_zoom,box_zoom,reset,hover,previewsave"

MATPLOTLIB_PLOT_STYLES = [
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

CHART_TYPES = [
    KindOfChart(description='Area', name='plot_area', kind='area', stacked=False, horizontal=False),
    KindOfChart(description='Area (stacked)', name='plot_stacked_area', kind='area', stacked=True, horizontal=False),
    KindOfChart(description='Bar', name='plot_bar', kind='bar', stacked=False, horizontal=False),
    KindOfChart(description='Line', name='plot_line', kind='line', stacked=False, horizontal=False),
    KindOfChart(description='Bar (stacked)', name='plot_stacked_bar', kind='bar', stacked=True, horizontal=False),
    KindOfChart(description='Line (stacked)', name='plot_stacked_line', kind='line', stacked=True, horizontal=False),
    KindOfChart(description='Bar (horizontal)', name='plot_barh', kind='bar', stacked=False, horizontal=True),
    KindOfChart(description='Bar (horizontal, stacked)', name='plot_stacked_barh', kind='bar', stacked=True, horizontal=True),
    # KindOfChart(description='Scatter', name='plot_scatter', kind='scatter', stacked=False, horizontal=False),
    # KindOfChart(description='Histogram', name='plot_hist', kind='hist', stacked=False, horizontal=False),
    KindOfChart(description='Table', name='table', kind=None, stacked=False, horizontal=False),
    KindOfChart(description='Pivot', name='pivot', kind=None, stacked=False, horizontal=False)
]

CHART_TYPE_MAP = { x.name: x for x in CHART_TYPES }
CHART_TYPE_OPTIONS = { x.name: x.name for x in CHART_TYPES }
CHART_TYPE_NAME_OPTIONS = [ (x.description, x.name) for x in CHART_TYPES ]

# output_charts = ([x.description for x in CHART_TYPES], CHART_TYPES)

parties_of_interest = ['FRANCE', 'GERMU', 'ITALY', 'GERMAN', 'UK', 'GERME', 'GERMW', 'INDIA', 'GERMA' ]

TREATY_FILTER_OPTIONS = {
    'Is Cultural': 'is_cultural',
    'Topic is 7CULT': 'is_7cult',
    'No filter': ''
}

TREATY_FILTER_TOOLTIPS = [
    'Include ONLY treaties marked as "is cultural"',
    'Include all treaties where topic is 7CULT (disregard "is cultural" flag)',
    'Include ALL treaties (no topic filter)'
]

PARTY_NAME_OPTIONS = {
    'WTI Code': 'party',
    'WTI Name': 'party_name',
    'WTI Short': 'party_short_name',
    'Country': 'party_country'
}

PARTY_PRESET_OPTIONS = {
    'PartyOf5': parties_of_interest,
    'Germany (all)': [ 'GERMU', 'GERMAN', 'GERME', 'GERMW', 'GERMA' ],
    'Select all': [ 'SELECT_ALL', 'ALL_PARTIES', 'ALL' ],
    'Unselect all': [ ]
}

FOX_STOPWORDS = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'because', 'become', 'becomes', 'became', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'herself', 'here', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'newer', 'newest', 'next', 'no', 'non', 'not', 'nobody', 'noone', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'sees', 'seem', 'seemed', 'seeming', 'seems', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'uses', 'used', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours']

