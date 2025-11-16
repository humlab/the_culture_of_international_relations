import string
import types

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from rake_nltk import Metric, Rake 

from common import treaty_state
from common.corpus import textacy_corpus_utility as textacy_utility
from common.gui import utility as gui_utility


def textacy_rake(
    doc, language="english", normalize="lemma", n_keyterms=20, stopwords=None, metric=Metric.DEGREE_TO_FREQUENCY_RATIO
):
    punctuations = string.punctuation + '"'
    r = Rake(
        stopwords=stopwords,  # NLTK stopwords if None
        punctuations=punctuations,  # NLTK by default
        language=language,
        ranking_metric=metric,
        max_length=100000,
        min_length=1,
    )
    text = " ".join(
        [x.lemma_.lower().strip("_") for x in doc] if normalize == "lemma" else [x.lower_ for x in doc.spacy_doc]
    )
    r.extract_keywords_from_text(doc.text)
    keyterms = [(y, x) for (x, y) in r.get_ranked_phrases_with_scores()]
    return keyterms[:n_keyterms]


def display_rake_gui(corpus, language):

    lw = lambda width: widgets.Layout(width=width)

    document_options = gui_utility.get_treaty_dropdown_options(current_wti_index(), corpus)
    metric_options = [
        ("Degree / Frequency", Metric.DEGREE_TO_FREQUENCY_RATIO),
        ("Degree", Metric.WORD_DEGREE),
        ("Frequency", Metric.WORD_FREQUENCY),
    ]
    gui = types.SimpleNamespace(
        position=1,
        progress=widgets.IntProgress(min=0, max=1, step=1, layout=lw(width="95%")),
        output=widgets.Output(layout={"border": "1px solid black"}),
        n_keyterms=widgets.IntSlider(description="#words", min=10, max=500, value=10, step=1, layout=lw(width="340px")),
        document_id=widgets.Dropdown(
            description="Treaty", options=document_options, value=document_options[1][1], layout=lw(width="40%")
        ),
        metric=widgets.Dropdown(
            description="Metric",
            options=metric_options,
            value=Metric.DEGREE_TO_FREQUENCY_RATIO,
            layout=lw(width="300px"),
        ),
        normalize=widgets.Dropdown(
            description="Normalize", options=["lemma", "lower"], value="lemma", layout=lw(width="160px")
        ),
        left=widgets.Button(description="<<", button_style="Success", layout=lw("40px")),
        right=widgets.Button(description=">>", button_style="Success", layout=lw("40px")),
    )

    def compute_textacy_rake(corpus, treaty_id, language, normalize, n_keyterms, metric):
        gui.output.clear_output()
        with gui.output:
            doc = textacy_utility.get_document_by_id(corpus, treaty_id)
            phrases = textacy_rake(
                doc, language=language, normalize=normalize, n_keyterms=n_keyterms, stopwords=None, metric=metric
            )
            df = pd.DataFrame(phrases, columns=["phrase", "score"])
            display(df.set_index("phrase"))
            return df

    itw = widgets.interactive(
        compute_textacy_rake,
        corpus=widgets.fixed(corpus),
        treaty_id=gui.document_id,
        language=widgets.fixed(language),
        normalize=gui.normalize,
        n_keyterms=gui.n_keyterms,
        metric=gui.metric,
    )

    def back_handler(*args):  # pylint: disable=unused-argument
        if gui.position == 0:
            return
        gui.output.clear_output()
        gui.position = (gui.position - 1) % len(document_options)
        gui.document_id.value = document_options[gui.position][1]
        # itw.update()

    def forward_handler(*args):  # pylint: disable=unused-argument
        gui.output.clear_output()
        gui.position = (gui.position + 1) % len(document_options)
        gui.document_id.value = document_options[gui.position][1]

    gui.left.on_click(back_handler)
    gui.right.on_click(forward_handler)

    display(
        widgets.VBox(
            [
                gui.progress,
                widgets.HBox([gui.document_id, gui.left, gui.right, gui.metric, gui.normalize]),
                widgets.HBox([gui.n_keyterms]),
                gui.output,
            ]
        )
    )

    itw.update()
