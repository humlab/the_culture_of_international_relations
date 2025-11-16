import types

import bokeh.plotting
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from textacy.corpus import Corpus

from common import treaty_utility, widgets_config
from common.corpus import textacy_corpus_utility as textacy_utility

# FROM NOTEBOOK:
# def compute_corpus_statistics(
#     data_folder,
#     wti_index,
#     container,
#     gui,
#     group_by_column='signed_year',
#     parties=None,
#     target='lemma',
#     include_pos=None,
#     stop_words=None
# ):

#     corpus = container.textacy_corpus

#     value_columns = list(textacy_utility.POS_NAMES) if (len(include_pos or [])) == 0 else list(include_pos)

#     documents = textacy_utility.get_corpus_documents(corpus)

#     if len(parties or []) > 0:
#         documents = documents[documents.party1.isin(parties)|documents.party2.isin(parties)]

#     documents['signed_lustrum'] = (documents.signed_year - documents.signed_year.mod(5)).astype(int)
#     documents['signed_decade'] = (documents.signed_year - documents.signed_year.mod(10)).astype(int)
#     documents['total'] = documents[value_columns].apply(sum, axis=1)

#     #documents = documents.groupby(group_by_column).agg(sum) #.reset_index()
#     aggregates = { x: ['sum'] for x in value_columns }
#     aggregates['total'] = ['sum', 'mean', 'min', 'max', 'size' ]
#     #if group_by_column != 'treaty_id':
#     documents = documents.groupby(group_by_column).agg(aggregates)
#     documents.columns = [ ('Total, ' + x[1].lower()) if x[0] == 'total' else x[0] for x in documents.columns ]
#     columns = sorted(value_columns) + sorted([ x for x in documents.columns if x.startswith('Total')])
#     return documents[columns]

# def corpus_statistics_gui(data_folder, wti_index, container, compute_callback, display_callback):

#     lw = lambda w: widgets.Layout(width=w)

#     corpus = container.textacy_corpus

#     include_pos_tags =  list(textacy_utility.POS_NAMES)
#     pos_options = include_pos_tags

#     counter = collections.Counter(corpus.word_freqs(normalize='lemma', weighting='count', as_strings=True))
#     frequent_words = [ x[0] for x in textacy_utility.get_most_frequent_words(corpus, 100) ]

#     group_by_options = { TREATY_TIME_GROUPINGS[k]['title']: k for k in TREATY_TIME_GROUPINGS }
#     # output_type_options = [ ( 'Table', 'table' ), ( 'Pivot', 'pivot' ), ( 'Excel', 'excel' ), ]
#     ngrams_options = { '1': [1], '1,2': [1,2], '1,2,3': [1,2,3]}
#     party_preset_options = wti_index.get_party_preset_options()
#     parties_options = [ x for x in wti_index.get_countries_list() if x != 'ALL OTHER' ]
#     gui = types.SimpleNamespace(
#         parties=widgets.SelectMultiple(description='Parties', options=parties_options, value=[], rows=7, layout=lw('180px')),
#         party_preset=widgets_config.dropdown('Presets', party_preset_options, None, layout=lw('200px')),
#         target=widgets.Dropdown(description='Normalize', options={ '':  False, 'Lemma': 'lemma', 'Lower': 'lower' }, value='lemma', layout=lw('200px')),
#         include_pos=widgets.SelectMultiple(description='POS', options=pos_options, value=list([]), rows=7, layout=widgets.Layout(width='180px')),
#         group_by_column=widgets.Dropdown(description='Group by', value='signed_year', options=group_by_options, layout=lw('200px')),
#         #output_type=widgets.Dropdown(description='Output', value='table', options=output_type_options, layout=widgets.Layout(width='200px')),
#         compute=widgets.Button(description='Compute', layout=lw('120px')),
#         output=widgets.Output(layout={'border': '1px solid black'})
#     )

#     boxes = widgets.VBox([
#         widgets.HBox([
#             widgets.VBox([
#                 gui.group_by_column,
#                 #gui.output_type,
#                 gui.party_preset,
#             ]),
#             widgets.VBox([
#                 gui.parties,
#             ]),
#             gui.include_pos,
#             widgets.VBox([
#                 gui.compute
#             ]),
#         ]),
#         gui.output
#     ])

#     display(boxes)

#     def on_party_preset_change(change):  # pylint: disable=W0613
#         if gui.party_preset.value is None:
#             return
#         gui.parties.value = gui.parties.options if 'ALL' in gui.party_preset.value else gui.party_preset.value

#     gui.party_preset.observe(on_party_preset_change, names='value')

#     def compute_callback_handler(*_args):
#         gui.output.clear_output()
#         with gui.output:
#             df_freqs = compute_callback(
#                 data_folder=data_folder,
#                 wti_index=wti_index,
#                 container=container,
#                 gui=gui,
#                 target=gui.target.value,
#                 group_by_column=gui.group_by_column.value,
#                 parties=gui.parties.value,
#                 include_pos=gui.include_pos.value,
#             )
#             display_callback(gui, df_freqs)

#     gui.compute.on_click(compute_callback_handler)
#     return gui

# def plot_simple(xs, ys, **figopts):
#     source = bokeh.models.ColumnDataSource(dict(x=xs, y=ys))
#     figopts = utility.extend(dict(title='', toolbar_location="right"), figopts)
#     p = bokeh.plotting.figure(**figopts)
#     glyph = p.line(source=source, x='x', y='y', line_color="#b3de69")
#     return p

# def display_corpus_statistics(gui, df):
#     display(df)
#     #with gui.output:
#     #    plotopts=dict(plot_width=1000, plot_height=500, title='', tools='box_zoom,wheel_zoom,pan,reset')
#     #    p = plot_simple(X.index, X['Total, mean'], **plotopts)
#     #    bokeh.plotting.show(p)

# try:
#     gui = corpus_statistics_gui(config.DATA_FOLDER, treaty_state.current_wti_index(), current_corpus_container(), compute_callback=compute_corpus_statistics, display_callback=display_corpus_statistics)
# except Exception as ex:
#     logger.error(ex)


def compute_corpus_statistics(
    # data_folder: str,
    # wti_index: TreatyState,
    *,
    parties: list[str],
    container: textacy_utility.CorpusContainer,
    include_pos: list[str],
    # gui,
    group_by_column: str = "signed_year",
    # target: str = "lemma",
    # stop_words: list[str] | None = None,
) -> pd.DataFrame:

    assert container.textacy_corpus is not None, "Corpus not generated"

    corpus: Corpus = container.textacy_corpus

    value_columns: list[str] = (
        list(textacy_utility.POS_NAMES) if (len(include_pos or [])) == 0 else list(include_pos or [])
    )

    documents: pd.DataFrame = treaty_utility.get_corpus_documents(corpus)

    if len(parties or []) > 0:
        documents = documents[documents.party1.isin(parties) | documents.party2.isin(parties)]

    documents["signed_lustrum"] = (documents.signed_year - documents.signed_year.mod(5)).astype(int)
    documents["signed_decade"] = (documents.signed_year - documents.signed_year.mod(10)).astype(int)
    documents["total"] = documents[value_columns].apply(sum, axis=1)

    # documents = documents.groupby(group_by_column).agg(sum) #.reset_index()
    aggregates: dict[str, list[str]] = {x: ["sum"] for x in value_columns}
    aggregates["total"] = ["sum", "mean", "min", "max", "size"]
    # if group_by_column != 'treaty_id':
    documents = documents.groupby(group_by_column).agg(aggregates)  # type: ignore
    documents.columns = [("Total, " + x[1].lower()) if x[0] == "total" else x[0] for x in documents.columns]
    columns: list[str] = sorted(value_columns) + sorted([x for x in documents.columns if x.startswith("Total")])
    return documents[columns]


def corpus_statistics_gui(data_folder, wti_index, container, compute_callback, display_callback):

    lw = lambda w: widgets.Layout(width=w)

    # corpus = container.textacy_corpus

    include_pos_tags = list(textacy_utility.POS_NAMES)
    pos_options = include_pos_tags

    # counter = collections.Counter(corpus.word_counts(normalize="lemma", weighting="count", as_strings=True))
    # frequent_words = [x[0] for x in textacy_utility.get_most_frequent_words(corpus, 100)]

    treaty_time_groupings = wti_index.get_treaty_time_groupings()
    group_by_options = {treaty_time_groupings[k]["title"]: k for k in treaty_time_groupings}
    # output_type_options = [ ( 'Table', 'table' ), ( 'Pivot', 'pivot' ), ( 'Excel', 'excel' ), ]
    # ngrams_options = {"1": [1], "1,2": [1, 2], "1,2,3": [1, 2, 3]}
    party_preset_options = wti_index.get_party_preset_options()
    parties_options = [x for x in wti_index.get_countries_list() if x != "ALL OTHER"]
    gui = types.SimpleNamespace(
        parties=widgets.SelectMultiple(
            description="Parties", options=parties_options, value=[], rows=7, layout=lw("180px")
        ),
        party_preset=widgets_config.dropdown("Presets", party_preset_options, None, layout=lw("200px")),
        target=widgets.Dropdown(
            description="Normalize",
            options={"": False, "Lemma": "lemma", "Lower": "lower"},
            value="lemma",
            layout=lw("200px"),
        ),
        include_pos=widgets.SelectMultiple(
            description="POS", options=pos_options, value=list([]), rows=7, layout=widgets.Layout(width="180px")
        ),
        group_by_column=widgets.Dropdown(
            description="Group by", value="signed_year", options=group_by_options, layout=lw("200px")
        ),
        # output_type=widgets.Dropdown(description='Output', value='table', options=output_type_options, layout=widgets.Layout(width='200px')),
        compute=widgets.Button(description="Compute", layout=lw("120px")),
        output=widgets.Output(layout={"border": "1px solid black"}),
    )

    boxes = widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            gui.group_by_column,
                            # gui.output_type,
                            gui.party_preset,
                        ]
                    ),
                    widgets.VBox(
                        [
                            gui.parties,
                        ]
                    ),
                    gui.include_pos,
                    widgets.VBox([gui.compute]),
                ]
            ),
            gui.output,
        ]
    )

    display(boxes)

    def on_party_preset_change(change):  # pylint: disable=W0613
        if gui.party_preset.value is None:
            return
        gui.parties.value = gui.parties.options if "ALL" in gui.party_preset.value else gui.party_preset.value

    gui.party_preset.observe(on_party_preset_change, names="value")

    def compute_callback_handler(*_args):
        gui.output.clear_output()
        with gui.output:
            df_freqs = compute_callback(
                data_folder=data_folder,
                wti_index=wti_index,
                container=container,
                gui=gui,
                target=gui.target.value,
                group_by_column=gui.group_by_column.value,
                parties=gui.parties.value,
                include_pos=gui.include_pos.value,
            )
            display_callback(gui, df_freqs)

    gui.compute.on_click(compute_callback_handler)
    return gui


def plot_simple(xs, ys, **figopts) -> bokeh.plotting.figure:
    # source = bokeh.models.ColumnDataSource(dict(x=xs, y=ys))
    opts: dict[str, str] = {"title": "", "toolbar_location": "right"} | figopts
    p = bokeh.plotting.figure(**opts)  # type: ignore
    # glyph = p.line(source=source, x="x", y="y", line_color="#b3de69")
    return p


def display_corpus_statistics(gui, df):
    display(df)
    # with gui.output:
    #    plotopts=dict(plot_width=1000, plot_height=500, title='', tools='box_zoom,wheel_zoom,pan,reset')
    #    p = plot_simple(X.index, X['Total, mean'], **plotopts)
    #    bokeh.plotting.show(p)
