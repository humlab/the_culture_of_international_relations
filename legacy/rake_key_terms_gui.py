import types

import gui_utility
import ipywidgets as widgets
import pandas as pd
import textacy.keyterms
from common.corpus import corpus_utility
from IPython.display import display


def display_document_key_terms_gui(corpus, wti_index):

    lw = lambda width: widgets.Layout(width=width)

    methods = {"RAKE": textacy_rake, "SingleRank": textacy.keyterms.singlerank, "TextRank": textacy.keyterms.textrank}
    document_options = [("All Treaties", None)] + gui_utility.get_treaty_dropdown_options(wti_index, corpus)

    gui = types.SimpleNamespace(
        position=1,
        progress=widgets.IntProgress(min=0, max=1, step=1, layout=lw(width="95%")),
        output=widgets.Output(layout={"border": "1px solid black"}),
        n_keyterms=widgets.IntSlider(description="#words", min=10, max=500, value=100, step=1, layout=lw("240px")),
        document_id=widgets.Dropdown(
            description="Treaty", options=document_options, value=document_options[0][1], layout=lw(width="40%")
        ),
        method=widgets.Dropdown(
            description="Algorithm", options=["RAKE", "TextRank", "SingleRank"], value="TextRank", layout=lw("180px")
        ),
        normalize=widgets.Dropdown(
            description="Normalize", options=["lemma", "lower"], value="lemma", layout=lw("160px")
        ),
        left=widgets.Button(description="<<", button_style="Success", layout=lw("40px")),
        right=widgets.Button(description=">>", button_style="Success", layout=lw("40px")),
    )

    def get_keyterms(method, doc, normalize, n_keyterms):
        keyterms = methods[method](doc, normalize=normalize, n_keyterms=n_keyterms)
        terms = ", ".join([x for x, y in keyterms])
        gui.progress.value += 1
        return terms

    def get_document_key_terms(corpus, method="TextRank", document_id=None, normalize="lemma", n_keyterms=10):
        treaty_ids = [document_id] if document_id is not None else [doc.metadata["treaty_id"] for doc in corpus]
        gui.progress.value = 0
        gui.progress.max = len(treaty_ids)
        keyterms = [
            get_keyterms(method, corpus_utility.get_treaty_doc(corpus, treaty_id), normalize, n_keyterms)
            for treaty_id in treaty_ids
        ]
        df = pd.DataFrame({"treaty_id": treaty_ids, "keyterms": keyterms}).set_index("treaty_id")

        # Add parties and groups
        df_extended = pd.merge(df, wti_index.treaties, left_index=True, right_index=True, how="inner")
        group_map = wti_index.get_parties()["group_name"].to_dict()
        df_extended["group1"] = df_extended["party1"].map(group_map)
        df_extended["group2"] = df_extended["party2"].map(group_map)
        columns = ["signed_year", "party1", "group1", "party2", "group2", "keyterms"]

        gui.progress.value = 0
        return df_extended[columns]

    def display_document_key_terms(corpus, method="TextRank", document_id=None, normalize="lemma", n_keyterms=10):
        gui.output.clear_output()
        with gui.output:
            df = get_document_key_terms(corpus, method, document_id, normalize, n_keyterms)
            if len(df) == 1:
                print(df.iloc[0]["keyterms"])
            else:
                display(df)

    itw = widgets.interactive(
        display_document_key_terms,
        corpus=widgets.fixed(corpus),
        method=gui.method,
        document_id=gui.document_id,
        normalize=gui.normalize,
        n_keyterms=gui.n_keyterms,
    )

    display(
        widgets.VBox(
            [
                gui.progress,
                widgets.HBox([gui.document_id, gui.left, gui.right, gui.method, gui.normalize, gui.n_keyterms]),
                gui.output,
            ]
        )
    )

    def back_handler(*args):
        if gui.position == 0:
            return
        gui.output.clear_output()
        gui.position = (gui.position - 1) % len(document_options)
        gui.document_id.value = document_options[gui.position][1]
        # itw.update()

    def forward_handler(*args):
        gui.output.clear_output()
        gui.position = (gui.position + 1) % len(document_options)
        gui.document_id.value = document_options[gui.position][1]

    gui.left.on_click(back_handler)
    gui.right.on_click(forward_handler)

    itw.update()
