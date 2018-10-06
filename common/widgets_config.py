# from __future__ import print_function
import os
import ipywidgets as widgets
import logging

#if __package__:
#    print('Package named {!r}; __name__ is {!r}'.format(__package__, __name__))
    
from common import extend    
from common.treaty_state import default_period_specification
from common.config import topic_group_maps, matplotlib_plot_styles, chart_types

def kwargser(d):
    args = dict(d)
    kwargs = args['kwargs']
    del args['kwargs']
    args.update(kwargs)
    return args

def toggle(description, value, **kwargs):
    return widgets.ToggleButton(**kwargser(locals()))
    
def dropdown(description, options, value, **kwargs):
    return widgets.Dropdown(**kwargser(locals()))

def slider(description, min, max, value, **kwargs):
    return widgets.IntSlider(**kwargser(locals()))

def rangeslider(description, min, max, value, **kwargs):
    return widgets.IntRangeSlider(**kwargser(locals()))

def sliderf(description, min, max, step, value, **kwargs):
    return widgets.FloatSlider(**kwargser(locals()))

def progress(min, max, step, value, **kwargs):
    return widgets.IntProgress(**kwargser(locals()))

def treaty_filter_widget(**kwopts):
    default_opts = dict(
        options={ 'Is Cultural': 'is_cultural', 'Topic is 7CULT': 'is_7cult', 'No filter': '' },
        description='Topic filter:',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=[
            'Include ONLY treaties marked as "is cultural"',
            'Include all treaties where topic is 7CULT (disregard "is cultural" flag)',
            'Include ALL treaties (no topic filter)'
        ],
        value='is_cultural',
        layout=widgets.Layout(width='200px')
    )
    return widgets.ToggleButtons(**extend(default_opts, kwopts))
    
def period_group_widget(**kwopts):
    default_opts = dict(
        options= {
            x['title']: x for x in default_period_specification
        },
        value=default_period_specification[-1],
        description='Divisions',
        layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))
    
def party_name_widget(**kwopts):
    default_opts = dict(
        options={
            'WTI Code': 'party',
            'WTI Name': 'party_name',
            'WTI Short': 'party_short_name',
            'Country': 'party_country'
        },
        value='party_name',
        description='Name',
        layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))
    
def normalize_widget(**kwopts):
    default_opts=dict(
        description='Display %',
        icon='',
        layout=widgets.Layout(width='100px', left='0')
    )
    return widgets.ToggleButton(**extend(default_opts, kwopts))
    
def parties_widget(**kwopts):
    default_opts=dict(
        options=[],
        value=None,
        rows=12,
        description='Parties',
        disabled=False,
        layout=widgets.Layout(width='180px')
    )
    return widgets.SelectMultiple(**extend(default_opts, kwopts))
        
def topic_groups_widget(**kwopts):
    default_opts=dict(
        options=topic_group_maps.keys(),
        description='Category:',
        layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))

def topic_groups_widget2(**kwopts):
    default_opts=dict(
        options=topic_group_maps,
        value=topic_group_maps['7CULTURE'],
        description='Category:',
        layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))

def plot_style_widget(**kwopts):
    default_opts=dict(
        options=[ x for x in matplotlib_plot_styles if 'seaborn' in x ],
        value='seaborn-pastel',
        description='Style:',
        layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))

def chart_type_widget(**kwopts):
    default_opts=dict(
            description='Output',
            options=[(x.description, x) for x in chart_types],
            value=chart_types[0],
            layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))

def recode_7corr_widget(**kwopts):
    default_opts=dict(
        description='Recode 7CORR',
        tooltip='Recode all treaties with cultural=yes as 7CORR',
        value=True,
        layout=widgets.Layout(width='120px')
    )
    return widgets.ToggleButton(**extend(default_opts, kwopts))

def add_other_category_widget(**kwopts):
    default_opts=dict(
        description='Add OTHER topics',
        tooltip='Add summed up category "OTHER" for all other topic (and for selected parties)',
        layout=widgets.Layout(width='120px'),
        value=False
    )
    return widgets.ToggleButton(**extend(default_opts, kwopts))
