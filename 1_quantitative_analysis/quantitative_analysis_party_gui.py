import ipywidgets as widgets
import common.widgets_config as wc
from common.widgets_utility import WidgetUtility
import common.config as config
from common.utility import getLogger
from IPython.display import display

logger = getLogger('qa')

def quantitative_analysis_party_main(state, fn):
    
    def period_group_year_window(period_group):
        
        periods = period_group['periods']\
            if period['column'] == 'signed_years' else list(map(max, periods))
        
        return min(periods), max(periods)
        
    period_group_widget = wc.period_group_widget()
    min_year, max_year = period_group_year_window(period_group_widget.value)
    
    year_window_widget = wc.rangeslider('Window', min_year, max_year, [min_year, max_year], layout=lw('90%'), continuous_update=False)
    
    tw = WidgetUtility(
        party_name=wc.party_name_widget(),
        normalize_values=wc.normalize_widget(),
        chart_type=wc.chart_type_widget(),
        plot_style=wc.plot_style_widget(),   
        top_n_parties=widgets.IntSlider(
            value=0, min=0, max=10, step=1,
            description='Top #',
            continuous_update=False,
            layout=widgets.Layout(width='200px')
        ),
        party_preset=widgets.Dropdown(
            options={
                'PartyOf5': config.parties_of_interest,
                'Germany (all)': [ 'GERMU', 'GERMAN', 'GERME', 'GERMW', 'GERMA' ]
            },
            value=None,
            description='Presets',
            layout=widgets.Layout(width='200px')
        ),
        parties=wc.parties_widget(options=state.get_countries_list(),value=['FRANCE']),
        treaty_filter=wc.treaty_filter_widget(),
        extra_category=widgets.ToggleButtons(
            options={ 'Other category': 'other_category', 'All category': 'all_category', 'Nothing': '' },
            description='Include:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            value='',
            layout=widgets.Layout(width='200px')
        ),
        overlay_option=widgets.ToggleButton(
            description='Overlay',
            icon='',
            value=True,
            layout=widgets.Layout(width='100px', left='0')
        ),
        holoviews_backend=widgets.ToggleButton(
            description='Holoviews',
            icon='',
            value=True,
            layout=widgets.Layout(width='100px', left='0')
        ),

    )

    itw = widgets.interactive(
        fn,
        period_group=period_group_widget,
        year_window=year_window_widget,
        party_name=tw.party_name,
        parties=tw.parties,
        treaty_filter=tw.treaty_filter,
        extra_category=tw.extra_category,
        normalize_values=tw.normalize_values,
        chart_type=tw.chart_type,
        plot_style=tw.plot_style,
        top_n_parties=tw.top_n_parties,
        overlay=tw.overlay_option
        # holoviews_backend=tw.holoviews_backend
    )

    def on_party_preset_change(change):
        if len(tw.party_preset.value or []) == 0:
            return
        if tw.top_n_parties.value > 0:
            tw.top_n_parties.value = 0
        tw.parties.value = tw.party_preset.value

    tw.party_preset.observe(on_party_preset_change, names='value')

    def on_parties_change(change):
        try:
            tw.top_n_parties.unobserve(on_top_n_parties_change, names='value')
            if tw.top_n_parties.value != 0:
                tw.top_n_parties.value = 0
            tw.top_n_parties.observe(on_top_n_parties_change, names='value')
        except Exception as ex:
            logger.info(ex)
            
    def on_top_n_parties_change(change):
        try:
            if tw.top_n_parties.value > 0:
                tw.parties.unobserve(on_parties_change, names='value')
                tw.parties.disabled = True
                if len(tw.parties.value) > 0:
                    tw.parties.value = []
            else:
                tw.parties.observe(on_parties_change, names='value')
                tw.parties.disabled = False
        except Exception as ex:
            logger.info(ex)

    def on_period_change(change):
        min_year, max_year = period_group_year_window(change['new'])
        year_window_widget.min, year_window_widget.max = min_year, max_year
        year_window_widget.value = (min_year, max_year)
        
    tw.parties.observe(on_parties_change, names='value')
    tw.period_group.observe(on_period_change, names='value')
    
    on_period_change(tw.period_group.value)
    
    boxes =  widgets.HBox([
        widgets.VBox([ tw.period_group, tw.party_name, tw.top_n_parties, tw.party_preset]),
        widgets.VBox([ tw.parties ]),
        widgets.VBox([ tw.treaty_filter, tw.extra_category]),
        widgets.VBox([ tw.chart_type, tw.plot_style]),
        widgets.VBox([ tw.normalize_values, tw.overlay_option ]), #, tw.holoviews_backend ])
    ])
    display(widgets.VBox([boxes, year_window_widget, itw.children[-1]]))
    itw.update()