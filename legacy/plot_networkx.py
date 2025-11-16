import networkx as nx
import pydotplus
from IPython.display import Image

# def apply_styles(graph: nx.Graph, styles: dict):
#     graph.graph_attr.update(("graph" in styles and styles["graph"]) or {})
#     graph.node_attr.update(("nodes" in styles and styles["nodes"]) or {})
#     graph.edge_attr.update(("edges" in styles and styles["edges"]) or {})
#     return graph


# styles = {
#     "graph": {
#         "label": "Graph",
#         "fontsize": "16",
#         "fontcolor": "white",
#         "bgcolor": "#333333",
#         "rankdir": "BT",
#     },
#     "nodes": {
#         "fontname": "Helvetica",
#         "shape": "hexagon",
#         "fontcolor": "white",
#         "color": "white",
#         "style": "filled",
#         "fillcolor": "#006699",
#     },
#     "edges": {
#         "style": "dashed",
#         "color": "white",
#         "arrowhead": "open",
#         "fontname": "Courier",
#         "fontsize": "12",
#         "fontcolor": "white",
#     },
# }


def plot(G, **kwargs) -> None | Image:  # pylint: disable=W0613

    pd_graph = nx.nx_pydot.to_pydot(G)
    pd_graph.format = "svg"
    # if root is not None :
    #    P.set("root",make_str(root))
    data = pd_graph.create_dot(prog="circo")
    if data == "":
        return None
    graph = pydotplus.graph_from_dot_data(data)
    # Q = apply_styles(Q, styles)
    image = Image(graph.create_png())
    return image
