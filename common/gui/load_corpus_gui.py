from ast import Index
import glob
import os
import types
from typing import Any
from venv import logger

import ipywidgets as widgets
from IPython.display import display

from common import config
from common.corpus.container import CorpusContainer
from common.corpus.generate_corpus import generate_corpus
from common.gui import widgets_config
from common.treaty_state import ConfigValue, TreatyState


def display_corpus_load_gui(data_folder: str, wti_index: TreatyState, container: CorpusContainer):

    def lw(w):
        return widgets.Layout(width=w)

    treaty_source_options: list[str] = wti_index.unique_sources
    treaty_default_source_options: list[str] = ["LTS", "UNTS", "UNXX"]
    language_mapping: dict[str, str] = ConfigValue("data.treaty_index.language.columns").resolve()
    language_options: dict[str, str] = {v.title(): k for k, v in language_mapping.items() if k in ["en", "fr"]}

    period_group_options: dict[str, Any] = {v["title"]: k for k, v in config.PERIOD_GROUPS_ID_MAP.items()}

    corpus_files: list[str] = list(
        sorted(
            glob.glob(os.path.join(data_folder, "treaty_text_corpora_??_*[!_preprocessed].zip")),
            key=lambda x: os.stat(x).st_mtime,
        )
    )

    if len(corpus_files) == 0:
        print("No prepared corpus")
        return None

    corpus_files = [os.path.basename(x) for x in corpus_files]

    gui = types.SimpleNamespace(
        progress=widgets.IntProgress(value=0, min=0, max=5, step=1, description="", layout=lw("90%")),
        output=widgets.Output(layout={"border": "1px solid black"}),
        source_path=widgets_config.dropdown(
            description="Corpus", options=corpus_files, value=corpus_files[-1], layout=lw("400px")
        ),
        sources=widgets_config.select_multiple(
            description="Sources",
            options=treaty_source_options,
            values=treaty_default_source_options,
            disabled=False,
            layout=lw("180px"),
        ),
        language=widgets_config.dropdown(
            description="Language", options=language_options, value="en", layout=lw("180px")
        ),
        period_group=widgets_config.dropdown(
            "Period", period_group_options, "years_1935-1972", disabled=False, layout=lw("180px")
        ),
        merge_entities=widgets_config.toggle("Merge NER", False, icon="", layout=lw("100px")),
        overwrite=widgets_config.toggle(
            "Force", False, icon="", layout=lw("100px"), tooltip="Force generation of new corpus (even if exists)"
        ),
        compute_pos=widgets_config.toggle(
            "POS", True, icon="", layout=lw("100px"), disabled=True, tooltip="Enable Part-of-Speech tagging"
        ),
        compute_ner=widgets_config.toggle(
            "NER", False, icon="", layout=lw("100px"), disabled=False, tooltip="Enable NER tagging"
        ),
        compute_dep=widgets_config.toggle(
            "DEP", False, icon="", layout=lw("100px"), disabled=True, tooltip="Enable dependency parsing"
        ),
        compute=widgets.Button(description="Compute", button_style="Success", layout=lw("100px")),
    )

    display(
        widgets.VBox(
            [
                gui.progress,
                widgets.HBox(
                    [
                        gui.source_path,
                        widgets.VBox([gui.language, gui.period_group, gui.sources]),
                        widgets.VBox([gui.merge_entities, gui.overwrite]),
                        widgets.VBox([gui.compute_pos, gui.compute_ner, gui.compute_dep]),
                        gui.compute,
                    ]
                ),
                gui.output,
            ]
        )
    )

    def tick(step: int | None = None, max_step: int | None = None) -> types.NoneType:
        if max_step is not None:
            gui.progress.max = max_step
        gui.progress.value = gui.progress.value + 1 if step is None else step

    def compute_callback(*_args):

        try:
            gui.output.clear_output()
            with gui.output:
                disabled_pipes = (
                    (() if gui.compute_pos.value else ("tagger",))
                    + (() if gui.compute_dep.value else ("parser",))
                    + (() if gui.compute_ner.value else ("ner",))
                )
                if f'_{gui.language.value}' not in gui.source_path.value:
                    logger.warning(
                        f"selected corpus may not match selected language '{gui.language.value}'"
                    )

                generate_corpus(
                    data_folder=data_folder,
                    wti_index=wti_index,
                    container=container,
                    source_path=os.path.join(data_folder, gui.source_path.value),
                    language=gui.language.value,
                    merge_entities=gui.merge_entities.value,
                    overwrite=gui.overwrite.value,
                    period_group=gui.period_group.value,
                    parties=None,
                    disabled_pipes=tuple(disabled_pipes),
                    tick=tick,
                    treaty_sources=gui.sources.value,
                )
        except IndexError as ie:
            logger.error(f"Did you select correct language? {ie}")

    gui.compute.on_click(compute_callback)

    return gui
