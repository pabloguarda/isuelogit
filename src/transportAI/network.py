""" Graphical representation of the infrastructure"""


"""
Engineer designs the network
-Connect the nodes in the network
- Create a train fleet,
- Define a train schedule, etc.
The role of the engineer is making the network works as a system.
"""

import numpy as np
from .infrastructure import Infrastructure
import networkx as nx
import random

# class Engineer:
#     def __init__(self,x,y):
#         '''
#         :argument
#         '''
#         pass

class Network:
    def __init__(self,infrastructure: list):
        '''
        :argument
        '''
        self.infrastructure = []
        self._nodes = []
        
        @property
        def infrastructure(self):
            return self.infrastructure
        
        @infrastructure.setter
        def infrastructure(self, value):
            self.infrastructure = value

        @property
        def nodes(self):
            return self.nodes

        @nodes.setter
        def nodes(self, value):
            self.nodes = value


    def random_positions(self, n):

        positions = range(0,n)

        return positions

    def create_nodes(self, infrastructure: list):

        positions = self.random_positions(n = len(self))

        for element in self.infrastructure:
            self.nodes.append(Node(label = element.label, position = positions))

        return self.nodes()

class Position:
    def __init__(self,x,y):
        '''
        :argument
        '''
        self.x = x
        self.y = y
        self.pos = (x,y)

class Node:

    def __init__(self, label, pos: Position):
        '''
        :argument label:
        :argument pos: tuple with length 2 (x,y)
        '''
        self.label = label
        self.pos = pos

class Link:
    def __init__(self, index, origin_node: Node, destination_node: Node, capacity: int):
        '''
        :argument label: index of the arc. There might be more than one arc between two nodes
        :argument flow: flow in the edge
        :argument cost: cost function of the arc, which generally depend only on the flow in the edge
        '''
        #self.label = label

class Route:
    def __init__(self, origin, destination, links: list, traveltime = -1):
        self._links = links
        self._destination = destination
        self._origin = origin
        self._traveltime = traveltime  #Initialization. Travel time must be greater than 0

    @property
    def traveltime(self):
        return self._traveltime

    @traveltime.setter
    def traveltime(self, value):
        self._traveltime = value

    def compute_travel_time(self,links):
        self.traveltime = np.sum([link.traveltime for link in links])

def create_network(W):
    '''Create a graph object compatible with network x package

    :arg W: weight or adjacency matrix

    Notes: edges with 0 cost and loops are ignored.

    '''
    graph = nx.DiGraph()
    n, m = W.shape
    assert n == m, "Adjacency matrix must be square"
    graph.add_nodes_from(range(n))
    for (i, j) in zip(*W.nonzero()):
        if i != j:
            graph.add_edge(i, j, weight=W[i, j])


    #Add attribute for the heuristic cost associated to each node
    nx.set_node_attributes(graph, 0,'heuristic')

    return graph

def set_random_nodes_coordinates(G, attribute_label, factor = 1):

    if factor != 1:
        pos = {k:factor*v for k,v in nx.random_layout(G.copy(), dim=2).items()}

    nx.set_node_attributes(G,pos, attribute_label)

    return G

def get_edges_euclidean_distances(G, nodes_coordinate_label = 'pos'):

    pos_nodes = nx.get_node_attributes(G,nodes_coordinate_label)
    len_edges = {}
    for edge in G.edges():
        len_edges[edge] = np.linalg.norm(np.array(pos_nodes[edge[0]]) - np.array(pos_nodes[edge[1]]))

    return len_edges

def get_euclidean_distances_between_nodes(G, nodes_coordinate_label = 'pos'):

    pos_nodes = nx.get_node_attributes(G, nodes_coordinate_label)
    nodes_G = G.nodes()
    nodes_euclidean_distances = np.zeros([len(nodes_G),len(nodes_G)])

    for node_i in G.nodes():
        for node_j in G.nodes():
            nodes_euclidean_distances[node_i,node_j] = np.linalg.norm(np.array(pos_nodes[node_i]) - np.array(pos_nodes[node_j]))

    return np.asmatrix(nodes_euclidean_distances)

def random_edge_weights(A, limits, type = int):
    '''
    Assign random integer weights to non-zero cells in adjacency matrix A

    :arg limits: tuple with lower and upper bound for the random integer numbers
    '''

    # for (u, v) in G.edges():
    #     G.edges[u, v]['weight'] = random.randint(0, 10)

    for (i, j) in zip(*A.nonzero()):

        if type is int:
            A[(i,j)] = random.randint(*limits)
        else:
            A[(i, j)] = random.random(*limits)

    return A

def random_parallel_edges(A, limits):

    return random_edge_weights(A, limits, type = int)

def create_MultiDiGraph_network(DG):
    ''' Receive a Digraph and return MultiDiGraph by
    randomly creating additional edges between nodes
    with a existing edge

    '''

    #Get adjacency matrix
    A0 = nx.convert_matrix.to_numpy_matrix(DG)

    # Randomly generate extra edges
    UM = np.random.randint(low = 0, high = 2, size = A0.shape)
    A1 = np.multiply(UM,random_edge_weights(A0, (np.min(A0), np.max(A0))))

    # Generate MultiGraphNetwork
    MG = nx.compose(nx.from_numpy_matrix(A1),nx.DiGraph(nx.from_numpy_matrix(A1)))

    return MG

