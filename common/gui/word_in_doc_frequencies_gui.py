import types

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from loguru import logger
from textacy.corpus import Corpus


def compute_doc_frequencies(corpus: Corpus, token_to_find: str, normalize: str = "lemma") -> pd.DataFrame:
    logger.info(f"Current corpus has {len(corpus)} documents")
    f = {"lemma": lambda x: x.lemma_, "lower": lambda x: x.lower_, "orth": lambda x: x.orth_}[normalize]
    token_to_find = token_to_find.lower() if normalize != "orth" else token_to_find
    data: dict[int, dict[str, int]] = {}
    for doc in corpus.docs:
        signed_year: int = doc._.meta["signed_year"]
        if signed_year not in data:
            data[signed_year] = {"signed_year": signed_year, "match_count": 0, "total_count": 0}
        data[signed_year]["total_count"] += 1
        if token_to_find in [f(x) for x in doc]:
            data[signed_year]["match_count"] += 1
    df: pd.DataFrame = pd.DataFrame(list(data.values())).set_index("signed_year")
    df["doc_frequency"] = (df.match_count / df.total_count) * 100
    df = df[["total_count", "match_count", "doc_frequency"]]
    return df


def word_doc_frequencies_gui(corpus: Corpus) -> None:
    normalize_options: dict[str, str] = {"Text": "orth", "Lemma": "lemma", "Lower": "lower"}

    gui = types.SimpleNamespace(
        normalize=widgets.Dropdown(
            description="Normalize", options=normalize_options, value="lemma", layout=widgets.Layout(width="200px")
        ),
        compute=widgets.Button(description="Compute", button_style="Success", layout=widgets.Layout(width="120px")),
        token=widgets.Text(description="Word"),
        output=widgets.Output(layout={"border": "1px solid black"}),
    )

    boxes = widgets.VBox([widgets.HBox([gui.normalize, gui.token, gui.compute]), gui.output])

    display(boxes)

    def compute_callback_handler(*_args):
        gui.output.clear_output()
        with gui.output:
            try:
                gui.compute.disabled = True
                df_counts: pd.DataFrame = compute_doc_frequencies(
                    corpus=corpus, token_to_find=gui.token.value, normalize=gui.normalize.value
                )
                display(df_counts)
            except Exception as ex:
                logger.error(ex)
                raise
            finally:
                gui.compute.disabled = False

    gui.compute.on_click(compute_callback_handler)
