import ipywidgets as widgets
import common.widgets_config as wc
from common.widgets_utility import WidgetUtility
import common.config as config

def quantitative_analysis_party_main(state, fn):

    tw = WidgetUtility(
        period_group=wc.period_group_widget(),
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
        )
    )

    itw = widgets.interactive(
        fn,
        period_group=tw.period_group,
        party_name=tw.party_name,
        parties=tw.parties,
        treaty_filter=tw.treaty_filter,
        extra_category=tw.extra_category,
        normalize_values=tw.normalize_values,
        chart_type=tw.chart_type,
        plot_style=tw.plot_style,
        top_n_parties=tw.top_n_parties,
        overlay=tw.overlay_option,
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

    tw.parties.observe(on_parties_change, names='value')
    tw.top_n_parties.observe(on_top_n_parties_change, names='value')

    column_boxes =  widgets.HBox([
        widgets.VBox([ tw.period_group, tw.party_name, tw.top_n_parties, tw.party_preset]),
        widgets.VBox([ tw.parties ]),
        widgets.VBox([ tw.treaty_filter, tw.extra_category]),
        widgets.VBox([ tw.chart_type, tw.plot_style]),
        widgets.VBox([ tw.normalize_values, tw.overlay_option ]) #, tw.holoviews_backend ])
    ])
    display(widgets.VBox([column_boxes, itw.children[-1]]))
    itw.update()