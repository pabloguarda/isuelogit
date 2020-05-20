import sys

import matplotlib.pyplot as plt

import networkx as nx
from networkx.utils import is_string_like

folder_plots = "TransportAI/plots/"

def draw_networkx_digraph_edge_labels(G, pos,
                              edge_labels=None,
                              label_pos=0.5,
                              font_size=10,
                              font_color='k',
                              font_family='sans-serif',
                              font_weight='normal',
                              alpha=None,
                              bbox=None,
                              ax=None,
                              rotate=True,
                              **kwds):
    """Modify networkX to Draw edge labels so it properly draw and put labels for DIgraph with bidrectional edges.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    alpha : float or None
       The text transparency (default=None)

    edge_labels : dictionary
       Edge labels in a dictionary keyed by edge two-tuple of text
       labels (default=None). Only labels for the keys in the dictionary
       are drawn.

    label_pos : float
       Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int
       Font size for text labels (default=12)

    font_color : string
       Font color string (default='k' black)

    font_weight : string
       Font weight (default='normal')

    font_family : string
       Font family (default='sans-serif')

    bbox : Matplotlib bbox
       Specify text box shape and colors.

    clip_on : bool
       Turn on clipping at axis boundaries (default=True)

    Returns
    -------
    dict
        `dict` of labels keyed on the edges

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_labels()
    """

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    offset = kwds.pop("offset", 0.0)

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        _norm = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        (x, y) = (x1 * label_pos + x2 * (1.0 - label_pos),
                  y1 * label_pos + y2 * (1.0 - label_pos))
        x += offset * (y2 - y1) / _norm
        y += offset * -(x2 - x1) / _norm

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < - 90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(np.array((angle,)),
                                                        xy.reshape((1, 2)))[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle='round',
                        ec=(1.0, 1.0, 1.0),
                        fc=(1.0, 1.0, 1.0),
                        )
        if not is_string_like(label):
            label = str(label)  # this makes "1" and 1 labeled the same

        # set optional alignment
        horizontalalignment = kwds.get('horizontalalignment', 'center')
        verticalalignment = kwds.get('verticalalignment', 'center')

        t = ax.text(x, y,
                    label,
                    size=font_size,
                    color=font_color,
                    family=font_family,
                    weight=font_weight,
                    alpha=alpha,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    rotation=trans_angle,
                    transform=ax.transData,
                    bbox=bbox,
                    zorder=1,
                    clip_on=True,
                    )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)

    return text_items

def show_network(G):
    '''Visualization of network.
    :arg G: graph object

    '''
    fig = plt.subplots()
    # fig.set_tight_layout(False) #Avoid warning using matplot

    pos = nx.get_node_attributes(G, 'pos')

    if len(pos) == 0:
        pos = nx.spring_layout(G)

    nx.draw(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw(G, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1')

    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw_networkx_labels(G, pos)

    plt.show()

def show_multiDiNetwork(G0, show_labels = False):

    #https://stackoverflow.com/questions/60067022/multidigraph-edges-from-networkx-draw-with-connectionstyle

    def new_add_edge(G, a, b):
        if (a, b) in G.edges:
            max_rad = max(x[2]['rad'] for x in G.edges(data=True) if sorted(x[:2]) == sorted([a,b]))
        else:
            max_rad = 0
        G.add_edge(a, b, rad=max_rad+0.1)

    G = nx.MultiDiGraph()

    edges = list(G0.edges)

    for edge in edges:
        new_add_edge(G, edge[0], edge[1])

    plt.figure(figsize=(10,10))

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)

    labels = nx.get_edge_attributes(G0, 'weight')

    if show_labels == True:
        for label_key in list(labels.keys()):
            labels[label_key] = str(list(label_key))+ ' = ' + str(labels[label_key])

            if labels.get(label_key[1],label_key[0]) is not None:
                labels[label_key] += ' , ' + str((label_key[1],label_key[0])) + ' = ' + str(labels.get(label_key[1],label_key[0]))


    # nx.draw(G, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1')

    for edge in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0],edge[1])], connectionstyle=f'arc3, rad = {edge[2]["rad"]}')

        if show_labels == True:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos= 0.5)

    plt.show()
