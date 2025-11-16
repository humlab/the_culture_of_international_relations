import types
import zipfile

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from loguru import logger

from common import config, utility
from common.corpus import corpus_utility


def chunks(l, n):
    """Returns list l in n-sized chunks"""
    if (n or 0) == 0:
        yield l
    else:
        for i in range(0, len(l), n):
            yield l[i : i + n]


def tokenize_docs(docs, **opts):
    document_id = 0
    normalize = opts["normalize"] or "orth"
    term_substitutions = opts.get("substitutions", {})
    word_counts = opts.get("word_counts", {})
    word_document_counts = opts.get("word_document_counts", {})
    extra_stop_words = set([])

    if opts["min_freq"] > 1:
        stop_words: set[str] = utility.extract_counter_items_within_threshold(
            word_counts[normalize], 1, opts["min_freq"]
        )
        extra_stop_words.update(stop_words)

    if opts["max_doc_freq"] < 100:
        stop_words = utility.extract_counter_items_within_threshold(
            word_document_counts[normalize], opts["max_doc_freq"], 100
        )
        extra_stop_words.update(stop_words)

    extract_args = {
        "args": {
            "ngrams": opts["ngrams"],
            "named_entities": opts["named_entities"],
            "normalize": opts["normalize"],
            "as_strings": True,
        },
        "kwargs": {
            "min_freq": opts["min_freq"],
            "include_pos": opts["include_pos"],
            "filter_stops": opts["filter_stops"],
            "filter_punct": opts["filter_punct"],
        },
        "extra_stop_words": extra_stop_words,
        "substitutions": (term_substitutions if opts.get("substitute_terms", False) else None),
    }

    for document_name, doc in docs:

        terms: list[str] = list(corpus_utility.extract_document_terms(doc, extract_args))

        chunk_size = opts.get("chunk_size", 0)
        chunk_index = 0
        for tokens in chunks(terms, chunk_size):
            yield document_id, document_name, chunk_index, tokens
            chunk_index += 1

        document_id += 1


def store_tokenized_corpus(
    tokenized_docs, corpus_source_filepath, **opts  # pylint: disable=unused-argument
) -> tuple[str, pd.DataFrame]:

    filepath: str = utility.path_add_timestamp(corpus_source_filepath)
    filepath = utility.path_add_suffix(filepath, ".tokenized")

    file_stats = []
    process_count: int = 0

    # TODO: Enable store of all documents line-by-line in a single file
    with zipfile.ZipFile(filepath, "w") as zf:

        for document_id, document_name, chunk_index, tokens in tokenized_docs:

            text = " ".join([t.replace(" ", "_") for t in tokens])
            store_name: str = utility.path_add_sequence(document_name, chunk_index, 4)

            zf.writestr(store_name, text, zipfile.ZIP_DEFLATED)

            file_stats.append((document_id, document_name, chunk_index, len(tokens)))

            if process_count % 100 == 0:
                logger.info(f"Stored {process_count} files...")

            process_count += 1

    df_summary = pd.DataFrame(file_stats, columns=["document_id", "document_name", "chunk_index", "n_tokens"])

    return filepath, df_summary


def display_generate_tokenized_corpus_gui(corpus, corpus_source_filepath, subst_filename=None):

    # filenames = [doc.user_data["textacy"]["meta"]["filename"] for doc in corpus]
    term_substitutions = {}

    if subst_filename is not None:
        logger.info("Loading term substitution mappings...")
        term_substitutions = corpus_utility.load_term_substitutions(
            subst_filename, default_term="_masked_", delim=";", vocab=corpus.spacy_vocab
        )

    pos_tags: dict[str, str] = (
        config.get_tag_set().groupby(["POS"])["DESCRIPTION"].apply(list).apply(lambda x: ", ".join(x[:1])).to_dict()
    )
    pos_options: list[tuple[str, str] | tuple[str, None]] = [("(All)", None)] + sorted(
        [(k + " (" + v + ")", k) for k, v in pos_tags.items()]
    )
    ngrams_options: dict[str, list[int]] = {"1": [1], "1,2": [1, 2], "1,2,3": [1, 2, 3]}

    lw = lambda width: widgets.Layout(width=width)
    gui = types.SimpleNamespace(
        progress=widgets.IntProgress(value=0, min=0, max=5, step=1, description="", layout=widgets.Layout(width="90%")),
        min_freq=widgets.IntSlider(
            description="Min word freq", min=0, max=10, value=2, step=1, layout=widgets.Layout(width="400px")
        ),
        max_doc_freq=widgets.IntSlider(
            description="Min doc. %", min=75, max=100, value=100, step=1, layout=widgets.Layout(width="400px")
        ),
        substitute_terms=widgets.ToggleButton(
            value=False, description="Mask GPE", tooltip="Replace geographical entites with `_gpe_`", icon="check"
        ),
        ngrams=widgets.Dropdown(
            description="n-grams", options=ngrams_options, value=[1], layout=widgets.Layout(width="180px")
        ),
        min_word=widgets.Dropdown(
            description="Min length", options=[1, 2, 3, 4], value=1, layout=widgets.Layout(width="180px")
        ),
        chunk_size=widgets.Dropdown(
            description="Chunk size",
            options=[("None", 0), ("500", 500), ("1000", 1000), ("2000", 2000)],
            value=0,
            layout=widgets.Layout(width="180px"),
        ),
        normalize=widgets.Dropdown(
            description="Normalize",
            options=[None, "lemma", "lower"],
            value="lower",
            layout=widgets.Layout(width="180px"),
        ),
        filter_stops=widgets.ToggleButton(
            value=False, description="Filter stops", tooltip="Filter out stopwords", icon="check"
        ),
        filter_punct=widgets.ToggleButton(
            value=False, description="Filter punct", tooltip="Filter out punctuations", icon="check"
        ),
        named_entities=widgets.ToggleButton(
            value=False, description="Merge entities", tooltip="Merge entities", icon="check"
        ),
        include_pos=widgets.SelectMultiple(
            description="POS", options=pos_options, value=[], rows=10, layout=widgets.Layout(width="400px")
        ),
        compute=widgets.Button(description="Compute", button_style="Success", layout=lw("100px")),
        output=widgets.Output(layout={"border": "1px solid black"}),
    )

    logger.info("Preparing corpus statistics...")
    logger.info("...word counts...")
    word_counts: dict[str, dict[int, set[str]]] = {
        k: corpus_utility.generate_word_count_score(corpus, k, gui.min_freq.max) for k in ["lemma", "lower", "orth"]
    }

    logger.info("...word document count...")
    word_document_counts: dict[str, dict[int, set[str]]] = {
        k: corpus_utility.generate_word_document_count_score(corpus, k, gui.max_doc_freq.min)
        for k in ["lemma", "lower", "orth"]
    }

    logger.info("...done!")

    gui.boxes = widgets.VBox(
        [
            gui.progress,
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            widgets.HBox([gui.normalize, gui.chunk_size]),
                            widgets.HBox([gui.ngrams, gui.min_word]),
                            gui.min_freq,
                            gui.max_doc_freq,
                        ]
                    ),
                    widgets.VBox([gui.include_pos]),
                    widgets.VBox([gui.filter_stops, gui.substitute_terms, gui.filter_punct, gui.named_entities]),
                    widgets.VBox([gui.compute]),
                ]
            ),
            gui.output,
        ]
    )

    display(gui.boxes)

    def compute_callback(*_args):
        gui.compute.disabled = True
        filepath = ""
        opts = {
            "min_freq": gui.min_freq.value,
            "max_doc_freq": gui.max_doc_freq.value,
            "substitute_terms": gui.substitute_terms.value,
            "ngrams": gui.ngrams.value,
            "min_word": gui.min_word.value,
            "normalize": gui.normalize.value,
            "filter_stops": gui.filter_stops.value,
            "filter_punct": gui.filter_punct.value,
            "named_entities": gui.named_entities.value,
            "include_pos": gui.include_pos.value,
            "chunk_size": gui.chunk_size.value,
            "term_substitutions": term_substitutions,
            "word_counts": word_counts,
            "word_document_counts": word_document_counts,
        }

        with gui.output:
            docs = ((doc.user_data["textacy"]["meta"]["filename"], doc) for doc in corpus)
            tokenized_docs = tokenize_docs(docs, **opts)
            filepath, df_summary = store_tokenized_corpus(tokenized_docs, corpus_source_filepath, **opts)

        gui.output.clear_output()

        with gui.output:
            logger.info("Process DONE!")
            logger.info(f"Result stored in '{filepath}'")
            display(df_summary)

        gui.compute.disabled = False

    gui.compute.on_click(compute_callback)

    return gui
