# plot_topic_relevance_by_year

def make_range_complete(str_values):
    # Create a complete range from min/max range in case values are missing
    # Assume the values are integers stored as strings
    values = list(map(int, str_values))
    values = range(min(values), max(values) + 1)
    return list(map(str, values))

import bokeh.transform

def setup_glyph_coloring(df):
    max_weight = df.weight.max()
    #colors = list(reversed(bokeh.palettes.Greens[9]))
    colors = ['#ffffff', '#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
    mapper = bokeh.models.LinearColorMapper(palette=colors, low=0.0, high=1.0) # low=df.weight.min(), high=max_weight)
    color_transform = bokeh.transform.transform('weight', mapper)
    color_bar = bokeh.models.ColorBar(color_mapper=mapper, location=(0, 0),
                         ticker=bokeh.models.BasicTicker(desired_num_ticks=len(colors)),
                         formatter=bokeh.models.PrintfTickFormatter(format=" %5.2f"))
    return color_transform, color_bar

def setup_int_ticker(str_values, major_label_orientation=None):
    '''
    Note: label ranges (from dataframe) are integers stored as string-values.
    '''
    ticker = SingleIntervalTicker(interval=1, num_minor_ticks=None)
    axis = LinearAxis(ticker=ticker)
    axis.major_label_orientation = major_label_orientation
    axis.axis_line_color = None
    axis.major_tick_line_color = None
    axis.major_label_text_font_size = LABEL_FONT_SIZE
    axis.major_label_text_align = "right"
    if len(str_values) > 10:
        label_overrides = { i: v if int(v) % 10 == 0 or i == 0 else '' for i, v in enumerate(str_values) }
        label_overrides[len(str_values)] = ''
        axis.major_label_overrides = label_overrides
    return axis

def plot_topic_relevance_by_year(df, xs, ys, flip_axis, glyph, titles, text_id):

    line_height = 7
    if flip_axis is True:
        xs, ys = ys, xs
        line_height = 10

    x_range = make_range_complete(df[xs].unique())
    y_range = make_range_complete(df[ys].unique())

    ''' Setup coloring and color bar '''
    color_transform, color_bar = setup_glyph_coloring(df)

    source = bokeh.models.ColumnDataSource(df)

    opts = dict(x_axis_type=None, y_axis_type=None, tools=TOOLS, toolbar_location="right")
    plot_height = max(len(y_range) * line_height, 500)

    p = bokeh.plotting.figure(title="Topic heatmap", toolbar_location="right", x_range=x_range,
            y_range=y_range, x_axis_location="above", plot_width=1000, plot_height=plot_height, **opts)

    args = dict(x=xs, y=ys, source=source, alpha=1.0, hover_color='red')

    if glyph == 'Circle':
        cr = p.circle(color=color_transform, **args)
    else:
        cr = p.rect(width=1, height=1, line_color=None, fill_color=color_transform, **args)

    p.x_range.range_padding = 0
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None

    # FIXME: Must separate between x=years, and x=documents. No skip of labesl if documents
    p.add_layout(setup_int_ticker(x_range, major_label_orientation=1.0), 'above')
    p.add_layout(setup_int_ticker(y_range, major_label_orientation=0.0), 'left')

    p.add_layout(color_bar, 'right')

    p.add_tools(bokeh.models.HoverTool(tooltips=None, callback=widgets_utility.WidgetUtility.glyph_hover_callback(
        source, 'topic_id', titles.index, titles, text_id), renderers=[cr]))

    return p

def display_doc_topic_heatmap(tm_data, key='max', year=None, glyph='Circle'):
    try:
        container = tm_data.compiled_data
        titles = LdaDataCompiler.get_topic_titles(container.topic_token_weights, n_words=100)
        df, pivot_column = state.get_topic_weight_by_year_or_document(key=key, year=year)
        df[pivot_column] = df[pivot_column].astype(str)
        df['topic_id'] = df.topic_id.astype(str)

        p = plot_topic_relevance_by_year(df, xs=pivot_column, ys='topic_id', glyph=glyph,titles=titles, text_id='topic_relevance')

        #os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        #p.output_backend = "svg"
        #export_svgs(p, filename="plot.svg")
        p.output_backend = "webgl"

        bokeh.plotting.show(p)
    except Exception as ex:
        raise
        logger.error(ex)

def doc_topic_heatmap_gui(tm_data):

     def text_widget(element_id=None, default_value='', style='', line_height='20px'):
        value = "<span class='{}' style='line-height: {};{}'>{}</span>".format(element_id, line_height, style, default_value) if element_id is not None else ''
        return widgets.HTML(value=value, placeholder='', description='', layout=widgets.Layout(height='150px'))

    text_id = 'topic_relevance'
    #text_widget = widgets_utility.wf.create_text_widget(text_id)
    text_widget = text_widget(text_id)
    glyph = widgets.Dropdown(options=['Circle', 'Square'], value='Square', description='Glyph', layout=widgets.Layout(width="180px"))
    year = wf.create_select_widget('Year', options=state.years)
    aggregate = wf.create_select_widget('Aggregate', list(AGGREGATES.keys()), 'max')

    iw = widgets.interactive(
        display_doc_topic_heatmap,
        tm_data=widgets.fixed(tm_data),
        glyph=glyph,
        key=aggregate,
        year=year
    )

    display(widgets.VBox([
        widgets.HBox([zo.aggregate, zo.glyph, zo.year]),
        zo.text,
        iw.children[-1]
    ]))

    iw.update()

doc_topic_heatmap_gui(get_current_model())
