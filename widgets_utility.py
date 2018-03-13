# from __future__ import print_function
import ipywidgets as widgets
from bokeh.models import ColumnDataSource, CustomJS

import logging
logger = logging.getLogger(__name__)

BUTTON_STYLE = dict(description_width='initial', button_color='lightgreen')

if 'extend' not in globals():
    extend = lambda a, b: a.update(b) or a

def create_js_callback(axis, attribute, source):
    return CustomJS(args=dict(source=source), code="""
        var data = source.data;
        var start = cb_obj.start;
        var end = cb_obj.end;
        data['""" + axis + """'] = [start + (end - start) / 2];
        data['""" + attribute + """'] = [end - start];
        source.change.emit();
    """)

class WidgetUtility:

    @staticmethod
    def glyph_hover_js_code(element_id, id_name, text_name, glyph_name='glyph', glyph_data='glyph_data'):
        return """
            var indices = cb_data.index['1d'].indices;
            var current_id = -1;
            if (indices.length > 0) {
                var index = indices[0];
                var id = parseInt(""" + glyph_name + """.data.""" + id_name + """[index]);
                if (id !== current_id) {
                    current_id = id;
                    var text = """ + glyph_data + """.data.""" + text_name + """[id];
                    $('.""" + element_id + """').html('ID ' + id.toString() + ': ' + text);
                }
        }
        """

    @staticmethod
    def glyph_hover_callback(glyph_source, glyph_id, text_ids, text, element_id):
        source = ColumnDataSource(dict(text_id=text_ids, text=text))
        code = WidgetUtility.glyph_hover_js_code(element_id, glyph_id, 'text', glyph_name='glyph', glyph_data='glyph_data')
        callback = CustomJS(args={'glyph': glyph_source, 'glyph_data': source}, code=code)
        return callback


class BaseWidgetUtility():

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        #self.__dict__.update(kwargs)

    def forward(self, description=None):
        if 'progress' in self.__dict__.keys():
            self.progress.value = self.progress.value + 1
            if description is not None:
                self.progress.description = description

    def create_int_progress_widget(self, **kwargs):
        return widgets.IntProgress(**kwargs)

    def create_select_widget(self, label='', values=[], default=None, **kwargs):
        opts = dict(
            options=values,
            value=default if default is not None and default in values else values[0] if len(values or []) > 0 else None,
            description=label,
            disabled=False
        )
        opts = extend(opts, kwargs)
        return widgets.Dropdown(**opts)

    def create_int_slider(self, description, **args):
        args = extend(dict(min=0, max=0, step=1, value=0, disabled=False, continuous_update=False), args)
        return widgets.IntSlider(description=description, **args)

    def create_int_range_slider(self, **args):
        return widgets.IntRangeSlider(**args)
    
    def create_float_slider(self, description, **args):
        args = extend(dict(min=0.0, max=0.0, step=0.1, value=0.0, disabled=False, continuous_update=False), args)
        return widgets.FloatSlider(description=description, **args)

    def select_aggregate_fn_widget(self, options=['mean', 'sum', 'std', 'min', 'max'], default='mean'):
        return self.create_select_widget('Aggregate', options, default)

    def select_year_widget(self, options=None):
        return self.create_select_widget('Year', options, default=None)

    def layout_algorithm_widget(self, options, default=None):
        return self.create_select_widget('Layout', options, default=default)

    def topic_id_slider(self, n_topics):
        return self.create_int_slider(description='Topic ID', min=0, max=n_topics - 1, step=1, value=0)

    def word_count_slider(self, min=1, max=500):
        return self.create_int_slider(description='Word count', min=min, max=max, step=1, value=50)

    def topic_count_slider(self, n_topics):
        return self.create_int_slider(description='Topic count', min=0, max=n_topics - 1, step=1, value=3)

    def create_button(self, description, style=BUTTON_STYLE, callback=None):
        button = widgets.Button(description=description, style=style)
        if callback is not None:
            button.on_click(callback)
        return button

    def create_text_input_widget(self, **opts):
        return widgets.Text(**opts)

    def create_text_area_input_widget(self, **opts):
        return widgets.Textarea(**opts)

    def create_text_widget(self, element_id=None, default_value=''):
        value = "<span class='{}'>{}</span>".format(element_id, default_value) if element_id is not None else ''
        return widgets.HTML(value=value, placeholder='', description='')

    def create_prev_button(self, callback):
        return self.create_button(description="<<", callback=callback)

    def create_next_button(self, callback):
        return self.create_button(description=">>", callback=callback)

    def create_next_id_button(self, name, count):
        that = self
        def f(b):
            control = getattr(that, name, None)
            if control is not None:
                control.value = (control.value + 1) % count
        return self.create_button(description=">>", callback=f)
        
    def create_prev_id_button(self, name, count):
        that = self
        def f(b):
            control = getattr(that, name, None)
            if control is not None:
                control.value = (control.value - 1) % count     
        return self.create_button(description="<<", callback=f)
            
    def next_topic_id_clicked(self, b): self.topic_id.value = (self.topic_id.value + 1) % self.n_topics
    def prev_topic_id_clicked(self, b): self.topic_id.value = (self.topic_id.value - 1) % self.n_topics

class TopicWidgets(BaseWidgetUtility):

    def __init__(self, n_topics, years=None, word_count=None, text_id=None):

        self.n_topics = n_topics
        self.text_id = text_id
        self.text = self.create_text_widget(text_id)
        self.year = self.select_year_widget(years) if not years is None else None
        self.topic_id = self.topic_id_slider(n_topics)

        self.word_count = self.word_count_slider(1, 500) if word_count is not None else None

        self.prev_topic_id = self.create_prev_button(self.prev_topic_id_clicked)
        self.next_topic_id = self.create_next_button(self.next_topic_id_clicked)


class TopTopicWidgets(BaseWidgetUtility):

    def __init__(self, n_topics=0, years=None, aggregates=None, text_id='text_id', layout_algorithms=None):

        self.n_topics = n_topics
        self.text_id = text_id
        self.text = self.create_text_widget(text_id) if text_id is not None else None
        self.year = self.select_year_widget(years) if years is not None else None

        self.topics_count = self.topic_count_slider(n_topics) if n_topics > 0 else None

        self.aggregate = self.select_aggregate_fn_widget(aggregates, default='mean') if aggregates is not None else None
        self.layout_algorithm = self.layout_algorithm_widget(layout_algorithms, default='Fruchterman-Reingold') \
            if layout_algorithms is not None else None

wf = BaseWidgetUtility()
