from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytypes import Proportion, Links, Matrix

# from transportAI.mytypes import Proportion

""" Graphical representation of the infrastructure"""


"""
Engineer designs the network
-Connect the nodes in the network
- Create a train fleet,
- Define a train schedule, etc.
The role of the engineer is making the network works as a system.
"""

import numpy as np
# from .infrastructure import Infrastructure
import networkx as nx
import random

import sys
import time
import pandas as pd

from paths import k_simple_paths_nx
# import transportAI.links
from paths import Path
import printer
from links import Link, generate_links_keys, generate_links_dict, BPR
from nodes import Node


import copy
import csv

import collections
# from transportAI.nodes import *

# from transportAI.pathgeneration import

# class Engineer:
#     def __init__(self,x,y):
#         '''
#         :argument
#         '''
#         pass

class TNetwork(nx.Graph):

    # RELEVANT MATRICES

    # A: Node-node adjacency matrix
    # D: Incidence link-path matrix
    # G: Networkx Multidigraph
    # M: Incidence OD pair-path matrix
    # Q: Demand between OD pairs (OD Matrix)

    # # Dictionaries of network attributes and matrices
    # A = {i:Ni.A for i,Ni in zip(N.keys(),N.values())}
    # D = {i:Ni.D for i,Ni in zip(N.keys(),N.values())}
    # G = {i:Ni.G for i,Ni in zip(N.keys(),N.values())}
    # M = {i:Ni.M for i,Ni in zip(N.keys(),N.values())}
    # Q = {i:Ni.Q for i,Ni in zip(N.keys(),N.values())}

    def __init__(self, A, G, links = None):
        '''
        :param links:
        :argument
        # Matrix A: Node to node incidence matrix
        '''
        # super().__init__(**attr)
        # self._infrastructure = []
        self._G = G # nx.Graph()
        self._nodes = [Node]

        #name of the network
        self._key = ''

        # Options
        self._setup_options = {}

        #Setup links and nodes (mandatory)
        self.setup_links_nodes(links) #If links are not provided, they are created automatically

        self._paths = [] # List with path objects
        self._paths_od = {}  # Dictionary with a list of path objects by od pair (key)

        self._A = A
        self._V = np.array([[]])
        self._D = np.array([[]])
        self._M = np.array([[]])
        self._Q = np.array([[]])
        self._q = np.array([[]])
        
        # If error is introducted to the matrix, this variables store the noisy matrix
        self._Q_true = np.array([[]])
        self._q_true = np.array([[]])

        # Type of network (multidinetwork or dinetwork)
        self._network_type = MultiDiTNetwork if (A > 1).any() else DiTNetwork

        # Choice set matrix (it is convenient to store it because it is expensive to generate)
        self._C = np.ndarray

        self._ods = []

        self._x_dict = {}
        self._x = None

        self._f_dict = {}
        self._f = None

        # Endogenous attributes for every link of the network
        self._Y_dict = {}  # Dict
        self._Y = np.ndarray  # Matrix

        #Exogenous attributes for every link of the network
        self._Z_dict = {} # Dict
        self._Z = np.ndarray # Matrix
        
        # Endogenous attributes for every link of the network
        self._Y_f_dict = {}  # Dict
        self._Y_f = np.ndarray  # Matrix
        
        #Exogenous attributes for every path of the network
        self._Z_f_dict = {} # Dict
        self._Z_f = np.ndarray # Matrix

        # Matrix with exogenous component of path utilities (|P| x |Z|)
        self._V_f_Z = np.ndarray

        # Store list of fixed effects
        self._k_fixed_effects = []



    # @property
    # def infrastructure(self):
    #     return self._infrastructure
    #
    # @infrastructure.setter
    # def infrastructure(self, value):
    #     self._infrastructure = value


    @property
    def setup_options(self):
        return self._setup_options

    @setup_options.setter
    def setup_options(self, value):
        self._setup_options = value

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    def get_n_nodes(self):
        return self.A.shape[0]

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, value):
        self._G = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def D(self):
        return self._D

    @D.setter
    def D(self, value):
        self._D = value

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        self._M = value
        
    @property
    def network_type(self):
        return self._network_type

    @network_type.setter
    def network_type(self, value):
        self._network_type = value

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value
        
    @property
    def Q_true(self):
        return self._Q_true

    @Q_true.setter
    def Q_true(self, value):
        self._Q_true = value


    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        self._q = value
        
    @property
    def q_true(self):
        return self._q_true

    @q_true.setter
    def q_true(self, value):
        self._q_true = value


    @property
    def x_dict(self):
        return self._x_dict

    @x_dict.setter
    def x_dict(self, value):
        self._x_dict = value

    @property
    def f_dict(self):
        return self._f_dict

    @f_dict.setter
    def f_dict(self, value):
        self._f_dict = value

    @property
    def f(self):
        if self._f is None:
            self._f = np.array(list(self.f_dict.values()))

        return self._f

    @f.setter
    def f(self, value):
        self._f = value

    @property
    def x(self):
        if self._x is None:
            self._x = np.array(list(self.x_dict.values()))

        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def Y_dict(self):
        return self._Y_dict

    @Y_dict.setter
    def Y_dict(self, value):
        self._Y_dict = value

    @property
    def Z_dict(self):
        return self._Z_dict

    @Z_dict.setter
    def Z_dict(self, value):
        self._Z_dict = value

    @property
    def Y_f(self):
        return self._Y_f

    @Y_f.setter
    def Y_f(self, value):
        self._Y_f = value

    @property
    def Z_f(self):
        return self._Z_f

    @Z_f.setter
    def Z_f(self, value):
        self._Z_f = value
        
    @property
    def Y_f_dict(self):
        return self._Y_f_dict

    @Y_f_dict.setter
    def Y_f_dict(self, value):
        self._Y_f_dict = value
        
    @property
    def Z_f_dict(self):
        return self._Z_f_dict

    @Z_f_dict.setter
    def Z_f_dict(self, value):
        self._Z_f_dict = value
    
    @property
    def V_f_Z(self):
        return self._V_f_Z

    @V_f_Z.setter
    def V_f_Z(self, value):
        self._V_f_Z = value


    @property
    def k_fixed_effects(self):
        return self._k_fixed_effects

    @k_fixed_effects.setter
    def k_fixed_effects(self, value):
        self._k_fixed_effects = value

    def set_V_f_Z(self, paths, k_Z, theta):
        listZ = []
        for path in paths:
            listZ.append([float(path.Z_dict[key]) * theta[key] for key in k_Z])

        self.V_f_Z = np.sum(np.asarray(listZ), axis=1)

    def get_matrix_from_dict_attrs_values(self, W_dict: dict):
        # Return Matrix Y or Z using Y or Z_dict
        listW = []
        for i in W_dict.keys():
            listW.append([float(x) for x in W_dict[i].values()])

        return np.asarray(listW).T

    def get_design_matrix(self, k_Y, k_Z):

        if len(k_Z)>0:
            Y_x = self.get_matrix_from_dict_attrs_values({k_y: self.Y[k_y] for k_y in k_Y})
            Z_x = self.get_matrix_from_dict_attrs_values({k_z: self.Z[k_z] for k_z in k_Z})
            YZ_x = np.column_stack([Y_x, Z_x])

        else:
            Y_x = self.get_matrix_from_dict_attrs_values({k_y: self.Y[k_y] for k_y in k_Y})
            YZ_x = np.column_stack([Y_x])

        return YZ_x


    @property
    def ods(self):
        return self._ods

    @ods.setter
    def ods(self, value):
        self._ods = value

    @property
    def paths(self):
        return self._paths

    @paths.setter
    def paths(self, value):
        self._paths = value

    @property
    def paths_od(self):
        return self._paths_od

    @paths_od.setter
    def paths_od(self, value):
        self._paths_od = value


    @property
    def links(self):
        return self._links

    @links.setter
    def links(self, value):
        self._links = value

    def get_n_links(self):
        return len(self.links)

    @property
    def links_dict(self):
        return self._links_dict

    @links_dict.setter
    def links_dict(self, value):
        self._links_dict = value

    @property
    def links_keys(self):
        return self._links_keys

    @links_keys.setter
    def links_keys(self, value):
        self._links_keys = value

    def get_observed_links(self, links: [] = None):

        """ Return list of links that have observed counts"""

        if links is None:
            return [link for link in self.links if not np.isnan(link.observed_count)]

        else:
            return [link for link in links if not np.isnan(link.observed_count)]


    def get_regular_links(self, links: [] = None):
        """
        Return list of links that exclude OD connectors

        If no list of links are provided, the internal list of links is used
        """

        if links is None:
            links = self.links

        regular_links = []

        link_types_list = []
        for link in links:
            link_type = link.link_type
            link_types_list.append(link_type)

            if link_type == 'LWRLK':
                regular_links.append(link)

        return regular_links

    def get_non_regular_links(self, links: [] = None):
        """
        Return list of links that exclude OD connectors

        If no list of links are provided, the internal list of links is used
        """

        if links is None:
            links = self.links

        non_regular_links = []

        link_types_list = []

        for link in links:
            link_type = link.link_type
            link_types_list.append(link_type)

            if link_type != 'LWRLK':
                non_regular_links.append(link)

        return non_regular_links


    # @staticmethod
    # def generate_paths_keys(paths_od: []) -> [tuple]:
    #
    #     paths_keys = []
    #
    #     for od in paths_od.keys():
    #         for path in paths_od[od]:
    #             paths_keys = (path.origin, path.destination,
    #
    #     # return 2
    #     return paths_keys
    
    # @staticmethod
    # def generate_paths_dict(link_keys: [tuple]):
    #     # paths_keys = self.get_paths_keys(self.G)
    #     return {link_label: transportAI.Link(label = link_label) for link_label in link_keys}


    def get_paths_from_paths_od(self, paths_od):

        paths_list = []

        for od,paths in paths_od.items():
            # This solves the problem to append paths when there is only one path per OD
            paths_list.extend(list(paths))

        return paths_list

    def get_paths_od_from_paths(self, paths):

        paths_od = {}
        ods = set()

        for path in paths:

            origin = path.origin
            destination = path.destination

            path = Path(origin,destination, path.links)

            if (origin,destination) in ods:
                paths_od[(origin,destination)].append(path)
            else:
                # print('path in different OD')
                paths_od[(origin, destination)] = [path]

            ods.update([(origin, destination)])

        return paths_od

    def setup_links_nodes(self, links= None) -> None:

        if links is None:

            # This is the case for the generation of toy networks. TODO: I should create the links externally and then provide them when creating the network.

            self.links_keys = generate_links_keys(G=self.G)  # list(self._G.edges())
            self.links_dict = generate_links_dict(link_keys=self._links_keys)  # list(self._G.edges())
            self.links = list(self._links_dict.values())  # List with link objects

        else:
            self.links_dict = self.copy_links_data(links = links)
            # print('here now')
            # print(links[0].bpr.bpr_function_x(0))

        # Store a list of the nodes among links in the network
        self.nodes = [node for link in self.links for node in link.nodes]

    def match_paths_with_existing_links(self, paths) -> None:

        # Match paths to the copy of the links objects inside the path objects so paths and existing links become associated




        self.paths = []

        total_links = len(self._links_dict)
        total_paths =len(paths)

        print('Matching ' + str(total_paths) + ' paths into ' + str(total_links) + ' links')

        for path, counter in zip(paths, range(total_paths)) :

            printer.printProgressBar(counter, total_paths, prefix='Progress:', suffix='',length=20)

            # print(len(paths))
            # print('here')

            links_path_copy = []

            for link in path.links:
                # print(len(path.links))
                # path_copy.append(link)
                links_path_copy.append(self._links_dict[link.key])

            path = Path(origin = path.init, destination = path.destination, links = links_path_copy)

            self.paths.append(path)

        # print(self.paths)

        # self._paths_od = paths_od
        # self.set_paths_from_paths_od(paths_od=self._paths_od)

    def set_Y_attr_links(self, y, label=key):
        self.Y_dict[label] = y

    @staticmethod
    def randomDiNetwork(n_nodes):
        A = np.random.randint(0, 2, [n_nodes, n_nodes])
        np.fill_diagonal(A, 0) # No cycles

        return DiTNetwork(A)

    @staticmethod
    def ods_fromQ(remove_zeros,Q = None):

        # OD pairs
        ods = []

        assert Q is not None, 'No Q matrix was provided'

        if remove_zeros:
            # Do not account for ODs with no trips, then D and M are smaller
            for (i, j) in zip(*Q.nonzero()):
                ods.append((i, j))
        else:
            for i, j in np.ndenumerate(Q):
                ods.append(i)

        return ods

    @staticmethod
    # @timeit
    def generate_M(paths_od: {tuple:Path}, paths = None):
        """Matrix M: Path-OD pair incidence matrix"""

        print('Generating matrix M')

        t0 = time.time()
        if paths is None:
            paths = []
            for pair in paths_od.keys():
                paths.extend(paths_od[pair])

        ods, n_ods, npaths = paths_od.keys(), len(paths_od.values()), len(paths)

        M = np.zeros([n_ods, npaths], dtype=np.int64)
        path_j = 0

        counter = 0
        for od, od_i in zip(ods, range(n_ods)):

            printer.printProgressBar(counter, n_ods, prefix='Progress(M):', suffix='', length=20)

            for path in paths_od[od]:
                M[od_i, path_j] = 1
                path_j += 1

            counter+=1

        assert M.shape[0] > 0, 'No matrix M generated'

        print('Matrix M ' + str(M.shape) + ' generated in ' + str(round(time.time() - t0, 1)) + '[s]')

        return M

    def generate_D(self, paths_od: {tuple:Path}, links: Links, paths = None):
        """Matrix D: Path-link incidence matrix"""

        t0 = time.time()

        print('Generating matrix D ')

        if paths is None:
            paths = []
            for pair in paths_od.keys():
                paths.extend(paths_od[pair])

        D = np.zeros([len(links), len(paths)], dtype=np.int64)

        total_paths = len(paths)

        for path, i in zip(paths, range(total_paths)):

            printer.printProgressBar(i, total_paths, prefix='Progress(D):', suffix='', length=20)

            links_path_list = path.links
            for link in links_path_list:
                # TODO: make this indexing operation faster as here is the bottleneck for reading D in Ohio
                #  This may involve to set the link id in a smart way so it represents the corresponding column in the
                #  row associated to the path that should be equal to 1

                D[links.index(link), i] = 1

        assert D.shape[0]>0, 'No matrix D generated'

        print('Matrix D ' + str(D.shape) + ' generated in ' + str(round(time.time() - t0,1)) + '[s]')

        return D

    def generate_V(self, A, links: Links, theta: dict):

        """ Matrix with link utilities with the same shape than the adjacency matrix """

        V = copy.deepcopy(A)

        for link in links:
            V[(link.init_node.key,link.term_node.key)] = link.utility(theta)

        return V

    def generate_edges_weights_dict_from_utility_matrix(self, V: Matrix):

        # edges_weights_dict = dict(zip(dict(G.links).keys(), np.random.randint(0, 20, len(list(G.links)))))

        # To avoid problems with link with negative utilities, we deviates them by the link with the most negative values such that all have utilities greater or equal than 0.

        V = V+abs(np.min(V))


        edges_weights_dict = {}

        for index, vx in np.ndenumerate(V):
            edges_weights_dict[index] = vx

        return edges_weights_dict

    @staticmethod
    def generate_Q(Nt: TNetwork, min_q: float, max_q: float, cutoff: int, n_paths: int, sparsity: Proportion = 0):

        print('Generating matrix Q')


        t0 = time.time()

        # G = nx.DiGraph(Nt.A)

        Q_mask = np.random.randint(min_q, max_q, Nt.A.shape)
        Q = np.zeros(Q_mask.shape)

        # Q = np.random.randint(2, 10, (100,100))



        # Set terms to 0 if there is no a path in between nodes on the graph produced by A

        nonzero_entries_Q_mask = list(Q_mask.nonzero())
        random.shuffle(nonzero_entries_Q_mask)

        total_entries_Q_mask = Q.shape[0]**2
        expected_non_zero_Q_entries = int(total_entries_Q_mask*(1-sparsity))

        print('The expected number of matrix entries to fill out is ' + str(expected_non_zero_Q_entries) + '. Sparsity: ' + "{:.0%}". format(sparsity))

        # total = len(nonzero_Q_entries)
        counter = 0

        for (i, j) in zip(*tuple(nonzero_entries_Q_mask)):

            # Q = tai.config.sim_options['custom_networks']['A']['N2']
            # print((i,j))

            printer.printProgressBar(counter, expected_non_zero_Q_entries, prefix='Progress:', suffix='',length=20)

            # Very ineficient as it requires to enumerate all paths
            # if len(list(nx.all_simple_paths(G, source=i, target=j, cutoff=cutoff))) == 0:

            k_paths_od = k_simple_paths_nx(k = n_paths, source=i, target=j, cutoff=cutoff, G = Nt.G, links = Nt.links_dict)

            if len(list(k_paths_od)) == n_paths:

                Q[(i, j)] = Q_mask[(i, j)]

                counter += 1

                if counter > expected_non_zero_Q_entries:
                    break
            else:
                # print('No paths with less than ' + str(cutoff) + ' links were found in o-d pair ' + str((i,j)))
                pass

        non_zero_entries_final_Q = np.count_nonzero(Q != 0) #/ float(Q.size)

        sparsity_final_Q = 1-non_zero_entries_final_Q/float(Q.size)

        #  Fill the Q matrix with zeros according to the degree of sparsity. As higher is the sparsity, faster is the generation of the Q matrix because there is less computation of shortest paths below

        # example: https://stackoverflow.com/questions/40058912/randomly-controlling-the-percentage-of-non-zero-values-in-a-matrix-using-python

        # idx = np.flatnonzero(Q)
        # N = np.count_nonzero(Q != 0) - int(round((1-sparsity) * Q.size))
        # np.put(Q, np.random.choice(idx, size=N, replace=False), 0)

        assert Q.shape[0] > 0, 'Matrix Q could not be generated'

        print(str(non_zero_entries_final_Q) + ' entries were filled out. Sparsity: ' + '{:.0%}'.format(sparsity_final_Q))

        print('Matrix Q ' + str(Q.shape) + ' generated in ' + str(round(time.time() - t0,1)) + '[s]')

        return Q

    @staticmethod
    def random_disturbance_Q(Q, sd = 0):
        '''Add a random disturbance but only for od pairs with trips'''

        Q_original = Q.copy()

        non_zeros_entries = 0
        # print(var)
        if sd == 'Poisson':
            for (i, j) in zip(*Q.nonzero()):
                Q[(i, j)] = np.random.poisson(lam=Q[(i, j)])
                non_zeros_entries += 1
                if Q[(i, j)] == 0:
                    Q[(i, j)] += 1e-7 #To avoid bugs when zeros are removed from Q matrix for other methods

        elif sd > 0:

            # # Lognormal
            # for (i, j) in zip(*Q.nonzero()):
            #     non_zeros_entries += 1
            #     Q[(i, j)] += np.random.lognormal(mean = 0, sigma = np.log(np.sqrt(var)))
            #
            #Truncated normal
            for (i, j) in zip(*Q.nonzero()):
                non_zeros_entries += 1
                Q[(i, j)] += np.random.normal(loc=0, scale=sd)

            # We truncate entries so they are positive by small number to avoid bugs when zeros are removed from Q matrix for other methods
            Q[Q< 0] = 1e-7
        # Compute difference between cell values in original and new demand matrix that were non zeros
        print('Mean of nonzero entries in the original demand matrix is ', "{0:.1f}".format(Q_original[np.nonzero(Q)].mean()))
        print('The mean absolute difference between the noisy and original demand matrices is ', "{0:.1f}".format(np.sum(np.abs(Q_original-Q))/non_zeros_entries))
        print('The approximated percentage change is',
              "{0:.1%}".format(np.sum(np.abs(Q_original - Q)) / (non_zeros_entries*Q_original [np.nonzero(Q_original )].mean())))

        return Q

    # def random_positions(self, n):
    #
    #     positions = range(0,n)
    #
    #     return positions

    # @classmethod
    # def create_nodes(infrastructure: list):
    #
    #     positions = random_positions(n = len(self))
    #
    #     for element in infrastructure:
    #         nodes.append(Node(label = element.label, position = positions))
    #
    #     return nodes()

    def copy_link_BPR_network(self, links: {}):
        '''
            :argument bpr_classes: different set of values for bpr functions
        '''

        # i) Assign randomly BPR function to the links in each network -> travel time:
        for i, link in links.items():
            self.links_dict[i].performance_function = copy.deepcopy(links[i].performance_function)

    def set_random_link_BPR_network(self, bpr_classes: {}):
        '''
            :argument bpr_classes: different set of values for bpr functions
        '''

        # i) Assign randomly BPR function to the links in each network -> travel time:
        for link in self.links_dict.values():
            bpr_class = random.choice(list(bpr_classes.values()))

            link.performance_function = BPR(alpha=bpr_class['alpha'], beta=bpr_class['beta'], tf=bpr_class['tf'], k=bpr_class['k'])


    def set_random_link_Z_attributes_network(self, Z_attrs_classes: {}, R_labels):

        '''
            :argument n_k: Number of random attributes to be generated.
            it moderate the level of sparsity as all parameters values are initilized in zero.
        '''

        # i) Pseudorandom attributes (currently are number of streets intersectinos and cost)
        for link in self.links_dict.values():
            for z_attr in Z_attrs_classes.keys():
                link.Z_dict[z_attr] = random.choice(list(Z_attrs_classes[z_attr].values()))

        # ii) Random attributes ('r1','r2', ...,'rk') -> sparsity
        n_l = len(self.links_dict.keys())
        n_R = len(R_labels)
        Z = np.random.random((n_l, n_R))

        counter = 0
        for i, link in self.links_dict.items():
            # Add elements to existing dictionary
            self.links_dict[i].Z_dict = {**self.links_dict[i].Z_dict, **dict(zip(R_labels, Z[counter, :]))}
            counter += 1

    def set_fixed_effects_attributes(self, fixed_effects, observed_links: str = None, links_keys: [tuple] = None):

        '''
            :argument fixed_effects: by q (direction matters) or nodes (half because it does not matter the direction of the links)

            notes: fixed effect at the od or nodes pair level are not identifiable if using data from one time period only
        '''

        coverage = fixed_effects['coverage']

        # i) Links

        if fixed_effects['links']:

            if observed_links == 'random':

                observed_link_idxs = []

                for link, i in zip(self.links, np.arange(len(self.links))):

                    if not np.isnan(link.observed_count):
                        # print(link.key)
                        observed_link_idxs.append(i)

                n_coverage = int(np.floor(len(observed_link_idxs) * coverage))

                idxs = np.random.choice(np.arange(len(observed_link_idxs)), size=n_coverage, replace=False)

                selected_links = [self.links[observed_link_idxs[idx]] for idx in idxs]

            elif observed_links == 'custom':

                selected_links = [self.links_dict[link_key] for link_key in links_keys]

            else:

                n_coverage = int(np.floor(len(self.links) * coverage))

                idxs = np.random.choice(np.arange(len(self.links)), size=n_coverage, replace=False)

                selected_links = [self.links[idx] for idx in idxs]


            for selected_link in selected_links:

                attr_lbl = 'l' + str(selected_link.key[0]) + '-' + str(selected_link.key[1])

                for link_key, link in self.links_dict.items():
                    if link_key[0] == link.key[0] and link_key[1] == selected_link.key[1]:
                        link.Z_dict[attr_lbl] = 1
                    else:
                        link.Z_dict[attr_lbl] = 0

                self.k_fixed_effects.append(attr_lbl)


        # ii) OD matrix

        if fixed_effects['Q'] or fixed_effects['nodes']:

            n_coverage = int(np.floor(len(self.A.nonzero()[0]) * coverage))

            idxs = np.random.choice(np.arange(len(self.A.nonzero()[0])), size=n_coverage, replace=False)

            selected_idxs = [(self.A.nonzero()[0][idx], self.A.nonzero()[1][idx]) for idx in idxs]



            # Store list of fixed effects
            self.k_fixed_effects = []

            if fixed_effects['Q']:

                for (i,j) in selected_idxs:
                    attr_lbl = 'q'+ str(i+1) + '-' + 'q'+ str(j+1)

                    for link_i, link in self.links_dict.items():
                        if link_i[0] == i and link_i[1] == j:
                            link.Z_dict[attr_lbl] = 1
                        else:
                            link.Z_dict[attr_lbl] = 0

                    self.k_fixed_effects.append(attr_lbl)

            # iii) Pair of nodes

            if fixed_effects['nodes']:
                for (i, j) in selected_idxs:

                    attr_lbl = ''
                    if i > j:
                        attr_lbl = 'n' + str(i+1) + ',' + 'n' + str(j+1)
                    if i < j:
                        attr_lbl = 'n' + str(j + 1) + ',' + 'n' + str(i + 1)

                    for link_i, link in self.links_dict.items():
                        if link_i[0] == i and link_i[1] == j or link_i[0] == j and link_i[1] == i :
                            link.Z_dict[attr_lbl] = 1
                        else:
                            link.Z_dict[attr_lbl] = 0

                    self.k_fixed_effects.append(attr_lbl)




    def copy_Z_attributes_dict_links(self, links_dict: {}):

        Z_labels = list(links_dict.values())[0].Z_dict.keys()

        for attr in Z_labels:
            for i, link in links_dict.items():

                self.links_dict[i].Z_dict[attr] = links_dict[i].Z_dict[attr]


    def set_Z_attributes_dict_network(self, links_dict: {}):
        '''Add the link attributes to a general dictionary indexed by the attribute names and with key the values of that attribute for every link in the network

        :argument links: dictionary of links objects. Each link contains the attributes values so it requires that a method such that set_random_link_attributes_network is executed before
        '''

        # Index the network dictionary with the attributes names of any link
        Z_labels = list(links_dict.values())[0].Z_dict.keys()
        self.Z_dict = {}

        for attr in Z_labels:
            self.Z_dict[attr] = {}
            for i, link in links_dict.items():
                self.Z_dict[attr][i] = link.Z_dict[attr]


    def reset_link_counts(self):

        for link in self.links:
            link.observed_count = np.nan

    def store_link_counts(self, xct):

        for link_key, count in xct.items():

            if not np.isnan(count):
                self.links_dict[link_key].observed_count = count




    def copy_links_data(self, links: []):
        '''

        '''

        # for link in links:
        #     test = copy.copy(link)

        # self.links = [copy.deepcopy(link) for link in links]

        self.links = [copy.copy(link) for link in links]

        self.links_dict = {link.key: link for link in self.links}

        self.copy_Z_attributes_dict_links(links_dict = self.links_dict)

        self.set_Z_attributes_dict_network(links_dict=self.links_dict)

        self.copy_link_BPR_network(links = self.links_dict)

        return self.links_dict

class MultiDiTNetwork(TNetwork):

    @staticmethod
    def multigraph(A, links = None):
        """:argument A: node to node adjacency matrix
        return nx.DiGraph()

        """
        n, m = A.shape
        assert n == m, "Node-to-node adjacency matrix must be square"

        G = nx.MultiDiGraph()

        # Labels does not matter
        G.add_nodes_from(np.arange(0,A.shape[0]))
        # G = nx.relabel_nodes(G, mapping)

        # Set node labels starting from one
        # nx.set_node_attributes(G,dict(zip(list(G.nodes()),list(G.nodes()))))

        if links is None:
            for (i, j) in zip(*A.nonzero()):
                for k in range(A[(i, j)]):
                    G.add_edge(i, j)

        else:
            for link in links:
                G.add_edge(link.key[0], link.key[1])

        return G

    def __init__(self, A, links = None):
        '''
        :param links:
        :argument
        '''

        super().__init__(A=A, G=self.multigraph(A, links), links = links)
        # self._nodes = list(self._G.nodes())

class DiTNetwork(TNetwork):

    @staticmethod
    def digraph(A, links = None):
        """:argument A: node to node adjacency matrix
        return nx.DiGraph()

        """
        n, m = A.shape
        assert n == m, "Node-to-node adjacency matrix must be square"

        G = nx.DiGraph()

        G.add_nodes_from(np.arange(0,A.shape[0]))

        if links is None:
            for (i, j) in zip(*A.nonzero()): # Add links between OD pairs with trips to ensure there is at least a route available
                if i != j:  # No loops allowed
                    G.add_edge(i, j)

        else:
            for link in links:
                G.add_edge(link.key[0], link.key[1])

        return G

    def __init__(self, A, links = None):
        '''
        :param links:
        :argument
        '''

        super().__init__(A=A, G=self.digraph(A, links), links =  links)


def create_graph(W):
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

def multiday_network(N, n_days, label,remove_zeros_Q, q_range, R_labels, cutoff_paths, od_paths, Z_attrs_classes, bpr_classes, fixed_effects, randomness):

    N_multiday = {}

    for day in range(0,n_days):
        N_multiday[day] = transportAI.modeller.setup_networks(N={day:N}, label= label, R_labels=R_labels
                                         , randomness= randomness
                                         , q_range= q_range
                                         , remove_zeros_Q=remove_zeros_Q
                                         , Z_attrs_classes=Z_attrs_classes
                                         , bpr_classes=bpr_classes, cutoff_paths=cutoff_paths, n_paths= od_paths
                                         , fixed_effects = fixed_effects).get(day)


    return N_multiday


def get_euclidean_distances_links(G, nodes_coordinate_label = 'pos'):

    pos_nodes = nx.get_node_attributes(G,nodes_coordinate_label)
    len_edges = {}
    for edge in G.edges():
        len_edges[edge] = np.linalg.norm(np.array(pos_nodes[edge[0]]) - np.array(pos_nodes[edge[1]]))

    return len_edges

def set_random_link_attributes(G):

    ''' TODO: customize attribute ranges'''

    # - Distance [m]  - based on nodes coordinates
    distances_links = get_euclidean_distances_links(G, nodes_coordinate_label='pos')
    nx.set_edge_attributes(G, distances_links, name='distance')

    # - Cost ($)
    cost_links = dict(zip(dict(G.links).keys(), np.random.uniform(0, 100, len(list(G.links)))))
    nx.set_edge_attributes(G, values=cost_links, name='cost')

    # - Intersections
    intesections_links = dict(zip(dict(G.links).keys(), np.random.randint(0, 100, len(list(G.links)))))
    nx.set_edge_attributes(G, values=intesections_links, name='intersections')

    # # - Speed (km/hr) - Set to constant meanwhile
    # constant_speed = 20
    # speed_links = dict(zip(dict(G.links).keys(),np.repeat(constant_speed, len(G.links()))))
    # nx.set_edge_attributes(G, values=speed_links, name='speed')
    #
    # # - Travel time (mins) - based on distance between links and a constant speed
    # travel_time_links = dict(zip(dict(G.links).keys(),
    #                              60/1000 * np.array(list(nx.get_edge_attributes(G, 'distance').values()))/np.array(list(nx.get_edge_attributes(G, 'speed').values())),
    #                                       ))
    # nx.set_edge_attributes(G, values=travel_time_links, name='travel_time')

    return G

def links_path(path):
    links_list = []
    for i in range(len(path) - 1):
        links_list.append((path[i], path[i + 1], path[i + 2]))

    return links_list

def denseQ(Q: np.matrix, remove_zeros: bool):

    q = [] # np.zeros([len(np.nonzero(Q)[0]),1])

    if remove_zeros:
        for i, j in zip(*Q.nonzero()):
            q.append(Q[(i,j)])

    else:
        for i, j in np.ndenumerate(Q):
            q.append(Q[i])

    q = np.array(q)[:,np.newaxis]

    assert q.shape[1] == 1, "od vector is not a column vector"

    return q


# def adjacency_dictionaries(G: nx.graph, Q: np.array):
#
#     # Matrix A: Node to node incidence matrix
#     A = adjacency_A(G).todense()
#
#     # OD pairs with trips
#     odpairs = []
#     for (i, j) in zip(*Q.nonzero()):
#         odpairs.append((i,j))
#
#     # Path generation
#     paths_odpair = {}
#     n_paths = 0
#     for pair in odpairs:
#         paths_odpair[pair] = list(nx.all_simple_paths(G, source =  pair[0], target = pair[1]))
#         n_paths += len(paths_odpair[pair])
#
#     # Matrix M: Path-OD pair incidence matrix
#     n_odpairs = len(odpairs)
#     M = {} # np.zeros([n_odpairs, n_paths])
#     odpair_i = 0
#     path_j = 0
#     for odpair in paths_odpair.keys():
#         M =
#         for path in paths_odpair[odpair]:
#             M[odpair_i,path_j] = 1
#             path_j += 1
#         odpair_i += 1
#
#     # Matrix D: Path-link incidence matrix
#     G.edges()
#
#     paths_odpair[(0,2)]
#
#     return A,D,M



def adjacency_D(G: nx.graph):
    pass

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


