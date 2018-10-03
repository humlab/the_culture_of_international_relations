*N = Number of nodes in graph*

<table>
    <tr><th>Layout algorithm</th><th>C</th><th>K</th><th>p</th><th>Note</th></tr>
    <tr><td>NetworkX ([fruchterman_reingold](https://networkx.github.io/documentation/networkx-1.11/reference/generated/networkx.drawing.layout.fruchterman_reingold_layout.html))</td><td></td><td></td><td>Optimal distance between nodes.</td><td>Fruchterman-Reingold force-directed algorithm.</td></tr>
    <tr><td>NetworkX ([Spectral](https://networkx.github.io/documentation/networkx-1.11/reference/generated/networkx.drawing.nx_pylab.draw_spectral.html#networkx.drawing.nx_pylab.draw_spectral))</td><td></td><td></td><td></td><td>Position nodes using the eigenvectors of the graph Laplacian.</td></tr>
    <tr><td>NetworkX (Circular)</td><td></td><td></td><td></td><td>Position nodes on a circle.</td></tr>
    <tr><td>NetworkX ([Shell](https://networkx.github.io/documentation/networkx-1.11/reference/generated/networkx.drawing.nx_pylab.draw_shell.html#networkx.drawing.nx_pylab.draw_shell))</td><td></td><td></td><td></td><td>Position nodes in a bipartite graph in concentric circles.</td></tr>
    <tr><td>NetworkX (Kamada-Kawai)</td><td></td><td></td><td></td></tr>
    <tr><td>graph-tool ([arf](https://graph-tool.skewed.de/static/doc/draw.html#graph_tool.draw.arf_layout))</td><td>Attracting force between adjacent vertices (a).</td><td>Opposing force between vertices (d).</td><td></td></tr>
    <tr><td>graph-tool ([sfdp](https://graph-tool.skewed.de/static/doc/draw.html#graph_tool.draw.sfdp_layout))</td><td>Relative strength of repulsive forces (C/100).</td><td>Optimal edge length.</td><td>Strength of the attractive force between connected components (gamma).</td></tr>
    <tr><td>graph-tool ([fruchterman_reingold](https://graph-tool.skewed.de/static/doc/draw.html#graph_tool.draw.fruchterman_reingold_layout))</td><td>Repulsive force between vertices (a=2*N*K).</td><td>Attracting force between adjacent vertices (r = 2\*C).</td><td></td><td>Fruchterman-Reingold force-directed algorithm.</td></tr>
    <tr><td>graphviz ([neato](https://graphviz.gitlab.io/_pages/pdf/neatoguide.pdf))</td><td></td><td>K=K</td><td></td></tr>
    <tr><td>graphviz ([dot](https://graphviz.gitlab.io/_pages/pdf/dotguide.pdf))</td><td></td><td>K=K</td><td></td></tr>
    <tr><td>graphviz (circo)</td><td></td><td>K=K</td><td></td></tr>
    <tr><td>graphviz (fdp)</td><td></td><td>K=K</td><td></td></tr>
    <tr><td>graphviz (sfdp)</td><td></td><td>K=K</td><td></td></tr>
</table>
