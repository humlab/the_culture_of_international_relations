from typing import Any

import pandas as pd
from bokeh.io.notebook import CommsHandle
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.palettes import RdYlBu, Set1, all_palettes  # pylint: disable=E0611
from bokeh.plotting import figure, show

from common import config
from common.utility import extend
from common.widgets_config import glyph_hover_callback2


def get_palette(palette_name: str) -> list[str]:
    if palette_name not in all_palettes:
        return RdYlBu[11]
    key: int = max(all_palettes[palette_name].keys())
    return all_palettes[palette_name][key]


def get_line_color(data: pd.DataFrame, topic_group_name: str) -> pd.Series:
    line_palette = Set1[8]
    group_keys = config.DEFAULT_TOPIC_GROUPS[topic_group_name].keys()
    line_palette_map: dict[str, int] = {k: i % len(line_palette) for i, k in enumerate(group_keys)}
    return data.category.apply(lambda x: line_palette[line_palette_map[x]])


TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

DFLT_NODE_OPTS: dict[str, Any] = {"color": "green", "level": "overlay", "alpha": 1.0}

DFLT_EDGE_OPTS: dict[str, Any] = {"color": "black", "alpha": 0.2}

DFLT_LABEL_OPTS: dict[str, str] = {
    "level": "overlay",
    "text_align": "center",
    "text_baseline": "middle",
    "text_font": "Tahoma",
    "text_font_size": "9pt",
    "text_color": "black",
}


def plot_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_description: pd.Series | None = None,
    node_size: int = 5,
    node_opts: dict[str, Any] | None = None,
    line_opts: dict[str, Any] | None = None,
    element_id: str = "nx_id3",
    figsize: tuple[int, int] = (900, 900),
    node_label: str = "name",
    node_label_opts: dict[str, Any] | None = None,
    edge_label: str = "name",
    edge_label_opts: dict[str, Any] | None = None,
    tools: str | None = None,
    **figkwargs,
) -> dict[str, Any]:

    edges_source: ColumnDataSource = ColumnDataSource(edges)
    nodes_source: ColumnDataSource = ColumnDataSource(nodes)

    node_opts = DFLT_NODE_OPTS | (node_opts or {})
    line_opts = DFLT_EDGE_OPTS | (line_opts or {})

    p: figure = figure(width=figsize[0], height=figsize[1], tools=tools or TOOLS, **figkwargs)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    if "line_color" in edges:
        line_opts = extend(line_opts, {"line_color": "line_color", "alpha": 1.0})

    _ = p.multi_line("xs", "ys", line_width="weight", source=edges_source, **line_opts)
    r_nodes = p.circle("x", "y", radius=node_size, source=nodes_source, **node_opts)

    if "fill_color" in nodes:
        r_nodes.glyph.fill_color = "fill_color"

    if node_description is not None:
        p.add_tools(
            HoverTool(
                renderers=[r_nodes],
                tooltips=None,
                callback=glyph_hover_callback2(
                    nodes_source,
                    "node_id",
                    text_ids=node_description.index,  # type: ignore
                    text_source=node_description,  # type: ignore
                    element_id=element_id,
                ),
            )
        )

    if node_label is not None and node_label in nodes:
        label_opts: dict[str, Any] = {} | DFLT_LABEL_OPTS | (node_label_opts or {})
        p.add_layout(LabelSet(source=nodes_source, x="x", y="y", text=node_label, **label_opts))

    if edge_label is not None and edge_label in edges:
        label_opts: dict[str, Any] = {} | DFLT_LABEL_OPTS | (edge_label_opts or {})
        p.add_layout(LabelSet(source=edges_source, x="m_x", y="m_y", text=edge_label, **label_opts))

    handle: CommsHandle | None = show(p, notebook_handle=True)

    return {
        "handle": handle,
        "edges": edges,
        "nodes": nodes,
        "edges_source": edges_source,
        "nodes_source": nodes_source,
    }
