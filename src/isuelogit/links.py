from __future__ import annotations


import networkx as nx
import random
import numpy as np

from isuelogit.nodes import Node

from isuelogit.geographer import LinkPosition

class Link:

    def __init__(self, key: tuple, init_node: Node, term_node: Node, position: LinkPosition = None):
        '''
        :argument key: tuple with the origin node, destination node, and index from 0 to N-1 when there are N parallel links
        :argument init_nodes: head node
        :argument term_nodes: tail node
        '''
        
        # TODO: similar to path object, a link object should store the corresponding origin and destination node
        
        self._key = key
        
        # If the id is read from xan external file, then the id will store that value
        self._id = str()

        # List of candidate PeMS station ids
        self._pems_stations_ids = []

        # INRIX segment id (int)
        self._inrix_id = np.nan
        
        # INRIX data associated to inrix id
        self._inrix_features = dict.fromkeys(
            ['speed_avg','speed_sd','speed_cv','speed_ref_avg','speed_ref_sd','speed_hist_avg','speed_hist_sd',
             'traveltime_cv','traveltime_avg','traveltime_sd','traveltime_var','road_closure_avg'])

        # List of incidents (list of dictionaries with data on incidents)
        self._incidents_list = []

        # List of bus stops (list of dictionaries with data on bus stops)
        self._bus_stops_list = []

        # List of bus stops (list of dictionaries with data on intersections)
        self._streets_intersections_list = []

        # Current link flow (e.g. equilibrium or predicted)
        self._x = float(0)

        # If simulated mode with synthetic data is used, then the observed count is filled out with the synthetic count
        self._count = np.nan

        # Link capacity (it is also stored in the bpr object)
        self._capacity = int(0) # int
        #origin_node: Node, destination_node: Node

        # Dictionary with attributes labels as dict keys, and their values as dict value
        self._Z_dict = {}
        self._Y_dict = {}
        self._performance_function = None # It can be BPR() or other function
        self._bpr = None  # BPR()
        self._traveltime = 0 # current travel time
        self._true_traveltime = None
        self._true_counts = None
        self._utility = 0
        self._init = key[0]
        self._term = key[1]

        # Type of links ('LWRLK','PQULK','DMDLK','DMOLK')
        self._link_type = 'LWRLK'
        self.Z_dict['link_type'] = 'LWRLK'
        
        # Nodes
        self._init_node = init_node
        self._term_node = term_node
        self._nodes = (self.init_node, self.term_node)

        # Direction (tuple N/S - W/E, e.g. ('N', 'W') which is northwest
        self._direction= None

        # Distance in coordinate system
        self._crs_distance= None

        # Confidence or strength on the N/S or W/E direction for the street
        self._direction_confidence = (0,0)

        # Pos
        self._position = None

        # if position is not None:
        #     self._position = position
        #
        # # self._position = None
        # if self.nodes[0].position is not None and self.nodes[1].position is not None:
        #
        #     # print('pass here')
        #     self.set_position_from_nodes(nodes = self.nodes)

    def get_position_from_nodes(self, nodes):

        init_node, term_node = nodes[0], nodes[1]

        assert init_node.crs == term_node.crs, 'crs systems of nodes in link are different'

        return LinkPosition(*init_node.position.get_xy(), *term_node.position.get_xy(), crs=init_node.crs)

    @property
    def position(self):
        return self.get_position_from_nodes(self.nodes)

    # @position.setter
    # def position(self, value):
    #     self._position = value

    # def set_position_from_nodes(self, nodes):
    #
    #    init_node, term_node = nodes[0], nodes[1]
    #
    #    assert init_node.crs == term_node.crs, 'crs systems of nodes in link are different'
    #
    #    self.position = LinkPosition(*init_node.position.get_xy(), *term_node.position.get_xy(), crs = init_node.crs)


    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value
        
    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        self._direction = value
        
    @property
    def direction_confidence(self):
        return self._direction_confidence

    @direction_confidence.setter
    def direction_confidence(self, value):
        self._direction_confidence = value
        
    @property
    def crs_distance(self):
        return self._crs_distance

    @crs_distance.setter
    def crs_distance(self, value):
        self._crs_distance = value
        
    @property
    def link_type(self):
        return self.Z_dict['link_type']

    @link_type.setter
    def link_type(self, value):
        self._link_type = value
        self.Z_dict['link_type'] = self._link_type

    @property
    def init_node(self):
        return self._init_node

    @init_node.setter
    def init_node(self, value):
        self._init_node = value

    @property
    def term_node(self):
        return self._term_node

    @term_node.setter
    def term_node(self, value):
        self._term_node = value
        
    @property
    def init(self):
        return self._init

    @init.setter
    def init(self, value):
        self._init = value
        
    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        self._term = value

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = value
    
    @property
    def traveltime(self):
        return self._traveltime

    @traveltime.setter
    def traveltime(self, value):
        self._traveltime = value

        self.Y_dict.update({'tt': self._traveltime})

    @property
    def true_traveltime(self):
        return self._true_traveltime

    @true_traveltime.setter
    def true_traveltime(self, value):
        self._true_traveltime = value
        
    @property
    def true_counts(self):
        return self._true_counts

    @true_counts.setter
    def true_counts(self, value):
        self._true_counts = value


    def set_traveltime_from_x(self, x):
        self.traveltime = self.bpr.bpr_function_x(x)


    def get_traveltime_from_x(self, x):
        return self.bpr.bpr_function_x(x)

    @property
    def bpr(self):
        return self._bpr

    @bpr.setter
    def bpr(self, value):
        self._bpr = value

    @property
    def performance_function(self):
        return self._performance_function

    @performance_function.setter
    def performance_function(self, value):
        self._performance_function = value
        if isinstance(self._performance_function,BPR):
            self._bpr = self._performance_function

    @property
    def pems_stations_ids(self):
        return self._pems_stations_ids

    @pems_stations_ids.setter
    def pems_stations_ids(self, value):
        self._pems_stations_ids = value

    @property
    def inrix_id(self):
        return self._inrix_id

    @inrix_id.setter
    def inrix_id(self, value):
        self._inrix_id = value
        
    @property
    def inrix_features(self):
        return self._inrix_features

    @inrix_features.setter
    def inrix_features(self, value):
        self._inrix_features = value

    @property
    def incidents_list(self):
        return self._incidents_list

    @incidents_list.setter
    def incidents_list(self, value):
        self._incidents_list = value
        
    @property
    def bus_stops_list(self):
        return self._bus_stops_list

    @bus_stops_list.setter
    def bus_stops_list(self, value):
        self._bus_stops_list = value
        
    @property
    def streets_intersections_list(self):
        return self._streets_intersections_list

    @streets_intersections_list.setter
    def streets_intersections_list(self, value):
        self._streets_intersections_list = value

    @property
    def Z_dict(self):
        return self._Z_dict

    @Z_dict.setter
    def Z_dict(self, value):
        self._Z_dict = value

    @property
    def Y_dict(self):
        return self._Y_dict

    @Y_dict.setter
    def Y_dict(self, value):
        self._Y_dict = value

    def utility(self, theta: {}):
        v = 0
        # for key in self.Z_dict.keys():
        #     v += theta[key]*self.Z_dict[key]

        #Intersect keys from theta vector and exogenous features from links
        keys = set(theta.keys()).intersection(set(self.Z_dict.keys()))

        for key in keys:
            v += theta[key]*self.Z_dict[key]

        return v + theta['tt']*self.traveltime

    # @utility.setter
    # def utility(self, value):
    #     self._utility = value

class BPR:

    def __init__(self, alpha: float, beta: float, tf: float, k: float):
        """ BPR function that maps link predicted_counts into travel times

        :arg alpha: shape parameter (b)
        :arg beta: exponent parameter
        :arg tf: free flow travel time
        :arg k: capacity [in flow units]
        """

        self._alpha = float(alpha)
        self._beta = float(beta)
        self._tf = float(tf)
        self._k = float(k)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = float(value)

    @property
    def tf(self):
        return self._tf

    @tf.setter
    def tf(self, value):
        self._tf = float(value)

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = float(value)

    def get_bpr_attrs_dict(self):
        return {'alpha': self.alpha, 'beta': self.beta, 'k': self.k, 'tf': self.tf}

    def bpr_function_x(self, x):
        """
        Return
        - :arg traveltime: Output of bpr function given a flow value (x) (in travel time units)
        """

        traveltime = self.tf * (1 + self.alpha * (x / self.k) ** self.beta)

        return traveltime

    def bpr_integral_x(self, x):
        """
        Return
        :arg integral: value of integral of the BPR function given a flow value (x)

        """
        # integral = self.tf * (1 + self.alpha * 1/(self.k ** self.beta) * x ** (self.beta + 1) / (self.beta + 1))

        # For numerical stability, equivalently:
        integral = self.tf * (1 + self.alpha * self.k * (x/self.k) ** (self.beta + 1) / (self.beta + 1))

        return integral


def generate_links_keys(G):
    links_keys = []
    if isinstance(G, nx.MultiDiGraph):
        links_keys = list(G.edges(keys=True))

        for i in range(len(links_keys)):
            links_keys[i] = (links_keys[i][0], links_keys[i][1], str(links_keys[i][2]))

    else:
        links_keys = [(i, j, '0') for i, j in list(G.edges())]

    # return 2
    return links_keys



def generate_links_nodes_dicts(links_keys: [tuple],
                               nodes_dict = {}):
    # links_keys = self.get_links_keys(self.G)

    links = []

    for link_key in links_keys:

        if link_key[0] not in nodes_dict.keys():
            nodes_dict[link_key[0]] = Node(key=link_key[0])

        if link_key[1] not in nodes_dict.keys():
            nodes_dict[link_key[1]] = Node(key=link_key[1])

        node_init = nodes_dict[link_key[0]]
        node_term = nodes_dict[link_key[1]]

        links.append(Link(key=link_key,
                          init_node=node_init,
                          term_node=node_term))

    links_dict = dict(zip(links_keys, links))

    return links_dict, nodes_dict


