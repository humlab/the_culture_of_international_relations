
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
    'Plot stacked line': 'plot_line_stacked',
    # 'Chart ': 'chart',
    'Table': 'table',
    'Pivot': 'pivot'
}

period_divisions = [
    [ (1919, 1939), (1940, 1944), (1945, 1955), (1956, 1966), (1967, 1972) ],
    [ (1919, 1944), (1945, 1955), (1956, 1966), (1967, 1972) ]
]

parties_of_interest = ['FRANCE', 'GERMU', 'ITALY', 'GERMAN', 'UK', 'GERME', 'GERMW', 'INDIA', 'GERMA' ]


period_group_options = {
    'Year': 'signed_year',
    'Default division': 'signed_period',
    'Alt. division': 'signed_period_alt'
}

default_party_options = {
    'Top #n parties': None,
    'PartyOf5': parties_of_interest,
    'France': [ 'FRANCE' ],
    'France vs UK': [ 'FRANCE', 'UK' ],
    'France vs ALL': [ 'FRANCE', None ],
    'Italy': [ 'ITALY' ],
    'UK': [ 'UK' ],
    'India': [ 'INDIA' ],
    'Germany, after 1991': [ 'GERMU' ],
    'Germany, before 1945': [ 'GERMAN' ],
    'East Germany': [ 'GERME' ],
    'West Germany': [ 'GERMW' ],
    'Germany, allied occupation': [ 'GERMA' ],
    'Germany (all)': [ 'GERMU', 'GERMAN', 'GERME', 'GERMW', 'GERMA' ],
    'China': [ 'CHINA' ]
}

category_group_settings = {
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

category_group_maps = { 
    category_name: { v: k for k in category_group_settings[category_name].keys() for v in category_group_settings[category_name][k]  }
        for category_name in category_group_settings.keys()
}