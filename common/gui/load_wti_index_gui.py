import types
from dataclasses import dataclass

import ipywidgets as widgets
from IPython.display import display
from loguru import logger

from common import config
from common.treaty_state import TreatyState, get_treaties_skip_column_names


@dataclass
class WTI_IndexContainer:
    value: TreatyState | None = None


WTI_INDEX_CONTAINER = WTI_IndexContainer()

WTI_OPTIONS: list[tuple[str, str]] = [
    ("WTI 7CULT Old", "is_cultural_yesno_org"),
    ("WTI 7CULT+", "is_cultural_yesno_plus"),
    ("WTI 7CULTGen", "is_cultural_yesno_gen"),
]

WTI_INFO: dict[str, str] = {x[1]: x[0] for x in WTI_OPTIONS}


def load_wti_index(
    data_folder: str | None = None,
    skip_columns: list[str] | None = None,
    period_groups=None,
    filename: str | None = None,
    is_cultural_yesno_column: str = "is_cultural_yesno_org",
) -> TreatyState | None:
    try:
        skip_columns = skip_columns or get_treaties_skip_column_names()
        data_folder = data_folder or config.DATA_FOLDER
        period_groups = period_groups or config.DEFAULT_PERIOD_GROUPS

        state = TreatyState(
            data_folder,
            skip_columns,
            period_groups,
            filename=filename,
            is_cultural_yesno_column=is_cultural_yesno_column,
        )

        print(
            f'WTI index loaded! Current index "{WTI_INFO[is_cultural_yesno_column]}" has {len(state.treaties[state.treaties.is_cultural])} treaties ({len(state.treaties.loc[(state.treaties.is_cultural) & (state.treaties.english == "en")])} in English).'
        )

        return state
    except Exception as ex:  # pylint: disable=broad-exception-caught
        logger.error(ex)

    return None


def current_wti_index() -> TreatyState:
    if WTI_INDEX_CONTAINER.value is None:
        raise AttributeError(
            "WTI Index not loaded. Please call load_wti_index or load_wti_index_with_gui prior to calling this method"
        )
    return WTI_INDEX_CONTAINER.value


def load_wti_index_with_gui(data_folder: str | None = None) -> None:

    global WTI_INDEX_CONTAINER  # pylint: disable=W0603,global-variable-not-assigned

    data_folder = data_folder or config.DATA_FOLDER

    def lw(w):
        return widgets.Layout(width=w)

    gui = types.SimpleNamespace(
        output=widgets.Output(),
        wti=widgets.Dropdown(
            description="Load index",
            options=WTI_OPTIONS,
            value="is_cultural_yesno_gen",
            layout=lw("300px"),
        ),
    )

    display(widgets.VBox([gui.wti, gui.output]))

    def compute_callback(*_args):
        gui.output.clear_output()
        with gui.output:
            WTI_INDEX_CONTAINER.value = load_wti_index(data_folder, is_cultural_yesno_column=gui.wti.value)

    gui.wti.observe(compute_callback, names="value")
    compute_callback()
