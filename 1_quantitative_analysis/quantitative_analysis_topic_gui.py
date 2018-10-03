import ipywidgets as widgets
import common.widgets_config as wc
from common.widgets_utility import BaseWidgetUtility
import common.config as config

def quantitative_analysis_topic_main(state, callback):

    period_group = wc.period_group_widget()
    topic_category_name = wc.topic_groups_widget(value='7CULTURE')
    # treaty_filter = wc.treaty_filter_widget()
    recode_is_cultural = wc.recode_7corr_widget()
    normalize_values = wc.normalize_widget()
    chart_type = wc.chart_type_widget()
    plot_style = wc.plot_style_widget()
    parties = wc.parties_widget(options=state.get_countries_list(), value=['FRANCE'])
    chart_per_party = widgets.ToggleButton(
        description='Chart per party', 
        tooltip='Display one chart per party',
        layout=widgets.Layout(width='120px')
    )
    extra_other_category = wc.add_other_category_widget()
           
    itw = widgets.interactive(
        callback,
        period_group=period_group,
        topic_category_name=topic_category_name,
        parties=parties,
        recode_is_cultural=recode_is_cultural,
        normalize_values=normalize_values,
        extra_other_category=extra_other_category,
        chart_type=chart_type,
        plot_style=plot_style,
        chart_per_party=chart_per_party
    )

    boxes = widgets.HBox(
        [
            widgets.VBox([ period_group, topic_category_name]),        
            widgets.VBox([ parties]),
            widgets.VBox([ recode_is_cultural, extra_other_category]),
            widgets.VBox([ chart_type, plot_style ]),
            widgets.VBox([ normalize_values, chart_per_party ])
        ]
    )
    display(widgets.VBox([boxes, itw.children[-1]]))
    itw.update()