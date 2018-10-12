import ipywidgets as widgets
import common.widgets_config as widgets_config
from common.widgets_utility import WidgetUtility
import common.config as config
from common.utility import getLogger
from IPython.display import display

logger = getLogger('qa')

PARTY_PRESET_OPTIONS = {
    'PartyOf5': config.parties_of_interest,
    'Germany (all)': [ 'GERMU', 'GERMAN', 'GERME', 'GERMW', 'GERMA' ]
}

OTHER_CATEGORY_OPTIONS = {
    'Other category': 'other_category',
    'All category': 'all_category',
    'Nothing': ''
}

def quantitative_analysis_party_main(wti_index, fn):
    
    def lw(width='100px', left='0'):
        return widgets.Layout(width=width, left=left)
    
    def period_group_window(period_group_index):
        '''Returns (min_year, max_year) for the period group.
        
        Periods are either a list of years, or a list of tuples (from-year, to-year)
        '''
        period_group = config.DEFAULT_PERIOD_GROUPS[period_group_index]

        periods = period_group['periods']
        
        if period_group['column'] != 'signed_year':
            periods = list(map(max, periods))
            
        return min(periods), max(periods)
        
    period_group_widget = widgets_config.period_group_widget(index_as_value=True)
    
    min_year, max_year = period_group_window(period_group_widget.value)

    year_limit_widget = widgets_config.rangeslider('Window', min_year, max_year, [min_year, max_year], layout=lw('90%'), continuous_update=False)
    
    party_name_widget = widgets_config.party_name_widget()
    normalize_values_widget = widgets_config.normalize_widget()
    chart_type_widget = widgets_config.chart_type_widget()
    plot_style_widget = widgets_config.plot_style_widget()
    top_n_parties_widget = widgets_config.slider('Top #', 0, 10, 0, continuous_update=False, layout=lw(width='200px'))
    party_preset_widget = widgets_config.dropdown('Presets', PARTY_PRESET_OPTIONS, None, layout=lw(width='200px'))
    parties_widget = widgets_config.parties_widget(options=wti_index.get_countries_list(excludes=['ALL_OTHER']), value=['FRANCE'])
    treaty_filter_widget = widgets_config.treaty_filter_widget()
    extra_category_widget = widgets_config.toggles('Include', OTHER_CATEGORY_OPTIONS, '', layout=lw(width='150px'))
    overlay_option_widget = widgets_config.toggle('Overlay', True, icon='', layout=lw())
    # holoviews_backend = widgets_config.toggle('Holoviews', True, icon='', layout=lw())

    itw = widgets.interactive(
        fn,
        period_group=period_group_widget,
        year_limit=year_limit_widget,
        party_name=party_name_widget,
        parties=parties_widget,
        treaty_filter=treaty_filter_widget,
        extra_category=extra_category_widget,
        normalize_values=normalize_values_widget,
        chart_type=chart_type_widget,
        plot_style=plot_style_widget,
        top_n_parties=top_n_parties_widget,
        overlay=overlay_option_widget
        # holoviews_backend=holoviews_backend
    )

    def on_party_preset_change(change):
        if len(party_preset_widget.value or []) == 0:
            return
        if top_n_parties_widget.value > 0:
            top_n_parties_widget.value = 0
        parties_widget.value = party_preset_widget.value

    party_preset_widget.observe(on_party_preset_change, names='value')

    def on_parties_change(change):
        try:
            top_n_parties_widget.unobserve(on_top_n_parties_change, names='value')
            if top_n_parties_widget.value != 0:
                top_n_parties_widget.value = 0
            top_n_parties_widget.observe(on_top_n_parties_change, names='value')
        except Exception as ex:
            logger.info(ex)
            
    def on_top_n_parties_change(change):
        try:
            if top_n_parties_widget.value > 0:
                parties_widget.unobserve(on_parties_change, names='value')
                parties_widget.disabled = True
                if len(parties_widget.value) > 0:
                    parties_widget.value = []
            else:
                parties_widget.observe(on_parties_change, names='value')
                parties_widget.disabled = False
        except Exception as ex:
            logger.info(ex)

    def set_years_window(period_group):
        min_year, max_year = period_group_window(period_group)
        year_limit_widget.min, year_limit_widget.max = min_year, max_year
        year_limit_widget.value = (min_year, max_year)
        
    def on_period_change(change):
        set_years_window(change['new'])
        
    parties_widget.observe(on_parties_change, names='value')
    period_group_widget.observe(on_period_change, names='value')
    
    set_years_window(period_group_widget.value)
    
    boxes =  widgets.HBox([
        widgets.VBox([ period_group_widget, party_name_widget, top_n_parties_widget, party_preset_widget]),
        widgets.VBox([ parties_widget ]),
        widgets.VBox([ treaty_filter_widget, extra_category_widget]),
        widgets.VBox([ chart_type_widget, plot_style_widget]),
        widgets.VBox([ normalize_values_widget, overlay_option_widget ]), #, holoviews_backend ])
    ])
    display(widgets.VBox([boxes, year_limit_widget, itw.children[-1]]))
    itw.update()