import ipywidgets as widgets
import common.widgets_config as wc
import common.config as config
from IPython.display import display

def quantitative_analysis_topic_main(wti_index, callback):

    width = widgets.Layout(width='120px')
    period_group = wc.period_group_widget()
    topic_category_name = wc.topic_groups_widget(value='7CULTURE')
    # treaty_filter = wc.treaty_filter_widget()
    recode_is_cultural = wc.recode_7corr_widget()
    normalize_values = wc.normalize_widget()
    chart_type = wc.chart_type_widget()
    plot_style = wc.plot_style_widget()
    parties = wc.parties_widget(options=[ x for x in wti_index.get_countries_list() if x != 'ALL OTHER' ], value=['FRANCE'])
    chart_per_category = widgets.ToggleButton(description='Chart per Qty', tooltip='Display one chart per selected quantity category', layout=width)
    extra_other_category = wc.add_other_category_widget()
    target_quantity = widgets.Dropdown(options=['topic', 'party'], value='topic', description='Quantity', layout=width)
    progress = widgets.IntProgress(min=0, max=5, step=1, layout=width)
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
        chart_per_category=chart_per_category,
        target_quantity=target_quantity,
        progress=widgets.fixed(progress)
    )

    boxes = widgets.HBox(
        [
            widgets.VBox([ period_group, topic_category_name, target_quantity]),        
            widgets.VBox([ parties ]),
            widgets.VBox([ recode_is_cultural, extra_other_category, chart_per_category]),
            widgets.VBox([ chart_type, plot_style ]),
            widgets.VBox([ normalize_values, progress ])
        ]
    )
    display(widgets.VBox([boxes, itw.children[-1]]))
    itw.update()