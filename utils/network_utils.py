"""
Module for network functions we will be using
"""

from networkx.algorithms.tree import maximum_spanning_edges
import networkx as nx


def get_max_spanning_tree(G):
    """
    This gets the maximum weighted spanning tree for the input network
    :param G: The graph whose tree you want to get the mwst for
    :return: the max weighted spanning tree networkx network object
    """
    G_copy = G.copy()

    mst_edges = maximum_spanning_edges(G_copy)
    mst = nx.Graph()
    for edge in mst_edges:
        mst.add_edge(edge[0], edge[1], weight=1)

    return mst
