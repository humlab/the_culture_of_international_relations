import ipywidgets as widgets
import common.widgets_config as wc
import common.config as config
from IPython.display import display

def display_gui(wti_index, display_function):

    def lw(width='120px'):
        return widgets.Layout(width=width)

    period_group_index_widget = wc.period_group_widget(index_as_value=True)
    topic_group_name_widget = wc.topic_groups_widget(value='7CULTURE')
    # treaty_filter_widget = wc.treaty_filter_widget()
    recode_is_cultural_widget = wc.recode_7corr_widget(layout=lw('120px'))
    normalize_values_widget = wc.toggle('Display %', False, icon='', layout=lw('100px'))
    chart_type_name_widget = wc.dropdown('Output', config.CHART_TYPE_NAME_OPTIONS, "plot_stacked_bar", layout=lw('200px'))
    plot_style_widget = wc.plot_style_widget()
    parties_widget = wc.parties_widget(options=[ x for x in wti_index.get_countries_list() if x != 'ALL OTHER' ], value=['FRANCE'])
    chart_per_category_widget = wc.toggle('Chart per Qty', False,
        tooltip='Display one chart per selected quantity category', layout=lw())
    extra_other_category_widget = wc.toggle('Add OTHER topics', False, layout=lw())
    target_quantity_widget = wc.dropdown('Quantity', ['topic', 'party'], 'topic', layout=lw())
    progress_widget = wc.progress(0, 5, 1, 0, layout=lw())

    def stepper(step=None):
        progress_widget.value = progress_widget.value + 1 if step is None else step

    itw = widgets.interactive(
        display_function,
        period_group_index=period_group_index_widget,
        topic_group_name=topic_group_name_widget,
        parties=parties_widget,
        recode_is_cultural=recode_is_cultural_widget,
        normalize_values=normalize_values_widget,
        extra_other_category=extra_other_category_widget,
        chart_type_name=chart_type_name_widget,
        plot_style=plot_style_widget,
        chart_per_category=chart_per_category_widget,
        target_quantity=target_quantity_widget,
        progress=widgets.fixed(stepper)
    )

    boxes = widgets.HBox(
        [
            widgets.VBox([ period_group_index_widget, topic_group_name_widget, target_quantity_widget]),
            widgets.VBox([ parties_widget ]),
            widgets.VBox([ recode_is_cultural_widget, extra_other_category_widget, chart_per_category_widget]),
            widgets.VBox([ chart_type_name_widget, plot_style_widget ]),
            widgets.VBox([ normalize_values_widget, progress_widget ])
        ]
    )
    display(widgets.VBox([boxes, itw.children[-1]]))
    itw.update()