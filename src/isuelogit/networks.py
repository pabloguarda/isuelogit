from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytypes import Proportion, Links, Matrix, LogitParameters, Dict, List, Feature, Optional, ColumnVector

import numpy as np
import pandas as pd
import networkx as nx
import random
import copy
import time

from paths import Path, get_paths_od_from_paths, get_paths_from_paths_od
from etl import LinkData
import printer
from links import Link, generate_links_keys, generate_links_nodes_dicts, BPR
from utils import Options,get_design_matrix
# from factory import generate_A_Q_custom_networks

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

    def __init__(self,
                 A,
                 G: nx.Graph(),
                 links = None):
        '''
        :param links:
        :argument
        # Matrix A: Node to node incidence matrix
        '''

        self._G = G # nx.Graph()
        # self._nodes = [Node]
        self._nodes_dict = {}

        #name of the network
        self._key = ''

        # Options
        self._setup_options = {}

        self._paths = [] # List with path objects
        self._paths_od = {}  # Dictionary with a list of path objects by od pair (key)

        self._A = A
        self._V = np.array([[]])
        self._D = np.array([[]])
        self._M = np.array([[]])

        # OD demand class
        self._OD = OD()

        # Type of network (multidinetwork or dinetwork)
        self._network_type = MultiDiTNetwork if (A > 1).any() else DiTNetwork

        # Choice set matrix (it is convenient to store it because it is expensive to generate)
        self._C = np.ndarray

        # self._ods = []

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

        #Setup links and nodes (mandatory)
        self.setup_links_nodes(links) #If links are not provided, they are created automatically

        # # Link data
        # self.link_data = LinkData(self.links)


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
        return list(self.nodes_dict.values())

    # @nodes.setter
    # def nodes(self, value):
    #     self._nodes = value

    @property
    def nodes_dict(self):
        return self._nodes_dict

    @nodes_dict.setter
    def nodes_dict(self, value):
        self._nodes_dict = value

    def get_n_nodes(self):
        return len(self._nodes_dict.values())

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
        return self.OD.Q
        # return self._Q

    # @Q.setter
    # def Q(self, value):
    #
    #     self.OD.Q = value
    #     # self._Q = value
        
    @property
    def Q_true(self):
        return self.OD.Q_true

    # @Q_true.setter
    # def Q_true(self, value):
    #     self.OD.Q_true = value

    @property
    def q(self):
        return self.OD.q
        # return self._q

    # @q.setter
    # def q(self, value):
    #     # self._q = value
    #     self.OD.q = value
        
    @property
    def q_true(self):
        return self.OD.q_true

    @property
    def OD(self):
        return self._OD

    # @q_true.setter
    # def q_true(self, value):
    #     self.OD.q_true = value

    @property
    def x_dict(self):
        return self.link_data.x

    # @x_dict.setter
    # def x_dict(self, value):
    #     self._x_dict = value

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
    def Y_data(self) -> pd.DataFrame:

        return self.link_data.Y_data

    @property
    def Z_data(self) -> pd.DataFrame:

        return self.link_data.Z_data

    # @property
    # def Y_dict(self,
    #            features = None):
    #
    #     return self.link_data.Y_dict(features)

    # @Y_dict.setter
    # def Y_dict(self, value):
    #     self._Y_dict = value

    # @property
    # def Z_dict(self,
    #            features = None):
    #
    #     return self.link_data.Z_dict(features)

    # @Z_dict.setter
    # def Z_dict(self, value):
    #     self._Z_dict = value


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

    def get_paths_specific_utility(self):
        utilities = []
        for path in self.paths:
            utilities.append(path.specific_utility)
            # print(path.utility({'tt':-1}))
        return np.array(utilities)[:,np.newaxis]

    def get_paths_Z_utility(self, theta, features_Z = None):

        # Path utilities associated to exogenous attributes

        Z_utilities = []
        if len(features_Z) > 0:
            for path in self.paths:
                Z_utilities.append(path.Z_utility(theta, features_Z))

            return np.array(Z_utilities)[:,np.newaxis]

        else:
            return 0

    def get_paths_Y_utility(self, theta):

        # Path utilities associated to endogenous attributes (travel time) is the only that changes over iterations

        Y_utilities = []

        for path in self.paths:
            Y_utilities.append(path.Y_utility(theta))

            return np.array(Y_utilities)[:, np.newaxis]

        else:
            return 0

    def get_paths_utility(self, theta, features_Z = None):

        utilities = []

        for path in self.paths:
            utilities.append(path.utility(theta,features_Z))

        return np.array(utilities)[:,np.newaxis]

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

    def design_matrix(self,
                      features_Y,
                      features_Z):

        return get_design_matrix(Y=self.Y_data[features_Y], Z=self.Z_data[features_Z])

        # Y = self.network.Y_data,
        # Z = self.network.Z_data[self.utility_function.features_Z])
        #
        # if len(features_Z)>0:
        #     Y_x = self.get_matrix_from_dict_attrs_values({k_y: self.Y[k_y] for k_y in features_Y})
        #     Z_x = self.get_matrix_from_dict_attrs_values({k_z: self.Z[k_z] for k_z in features_Z})
        #     YZ_x = np.column_stack([Y_x, Z_x])
        #
        # else:
        #     Y_x = self.get_matrix_from_dict_attrs_values({k_y: self.Y[k_y] for k_y in features_Y})
        #     YZ_x = np.column_stack([Y_x])
        #
        # return YZ_x


    @property
    def ods(self):

        return self.OD.ods
        # return self._ods

    # @ods.setter
    # def ods(self, value):
    #     self.OD.ods = value
    #     # self._ods = value

    @property
    def paths(self):
        # return self._paths
        return get_paths_from_paths_od(self.paths_od)

    # @paths.setter
    # def paths(self, value):
    #     self._paths = value

    @property
    def paths_od(self):
        return self._paths_od

    @paths_od.setter
    def paths_od(self, value):
        self._paths_od = value


    @property
    def links(self):
        # return self._links
        return list(self.links_dict.values())

    @links.setter
    def links(self, value):

        links_keys = [link.key for link in value]

        self.links_dict = dict(zip(links_keys, value))


    def get_n_links(self):
        return len(self.links)


    @property
    def links_dict(self):
        return self._links_dict

    @links_dict.setter
    def links_dict(self, value):

        self._links_dict = value

        links = None

        if value is not None:
            links = list(self.links_dict.values())

    @property
    def link_data(self):
        # # Update LinkData object
        # self.link_data = \

        return LinkData(links=self.links)

    # @link_data.setter
    # def link_data(self, value):
    #     # # Update LinkData object
    #     self.link_data =
    #
    #     return LinkData(links=self.links)

    @property
    def links_keys(self):
        return list(self.links_dict.keys())
        # return self._links_keys

    # @links_keys.setter
    # def links_keys(self, value):
    #     self._links_keys = value

    def get_observed_links(self, links: [] = None):

        """ Return list of links that have observed counts"""

        return get_observed_links(links = self.links)


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
    #     return {link_label: isuelogit.Link(label = link_label) for link_label in link_keys}



    def setup_links_nodes(self,
                          links= None) -> None:

        if links is None:

            # This is the case for the generation of toy networks. TODO: I should create the links externally and then provide them when creating the network.

            links_keys = generate_links_keys(G=self.G)  # list(self._G.edges())
            self.links_dict, self.nodes_dict = generate_links_nodes_dicts(
                nodes_dict = self.nodes_dict, links_keys=links_keys)  # list(self._G.edges())
            # self.links = list(self._links_dict.values())  # List with link objects

        else:
            # self.links = links
            self.links_dict = {link.key: link for link in links}
            # self.links_dict = self.copy_links_data(links = links)



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

    def set_Y_attr_links(self,
                         Y: Dict[Feature, Dict[tuple, float]],
                         feature: str = None):

        if feature is not None:
            features = [feature]

        if feature is None:
            features = list(Y.keys())

        for i in features:
            for key, value in Y[i].items():
                self.links_dict[key].Y_dict[i] = value

                if i == 'tt':
                    self.links_dict[key].traveltime = value

    @staticmethod
    def randomDiNetwork(n_nodes):
        A = np.random.randint(0, 2, [n_nodes, n_nodes])
        np.fill_diagonal(A, 0) # No cycles

        return DiTNetwork(A)

    @staticmethod
    def generate_D(paths_od: {tuple: Path}, links: Links, paths=None):
        """Matrix D: Path-link incidence matrix"""

        t0 = time.time()

        # print('Generating matrix D ')

        if paths is None:
            paths = []
            for pair in paths_od.keys():
                paths.extend(paths_od[pair])

        D = np.zeros([len(links), len(paths)], dtype=np.int64)

        total_paths = len(paths)

        # print('\n')

        for path, i in zip(paths, range(total_paths)):

            printer.printProgressBar(i, total_paths-1, prefix='Progress(D):', suffix='', length=20)

            links_path_list = path.links
            for link in links_path_list:
                # TODO: make this indexing operation faster as here is the bottleneck for reading D in Ohio
                #  This may involve to set the link id in a smart way so it represents the corresponding column in the
                #  row associated to the path that should be equal to 1

                D[links.index(link), i] = 1

        assert D.shape[0] > 0, 'No matrix D generated'

        print('Matrix D ' + str(D.shape) + ' generated in ' + str(round(time.time() - t0, 1)) + '[s]')

        return D

    @staticmethod
    def generate_V(A, links: Links, theta):

        """ Matrix with link utilities with the same shape than the adjacency matrix """

        V = copy.deepcopy(A)

        for link in links:
            V[(link.init_node.key, link.term_node.key)] = link.utility(theta)

        return V

    @staticmethod
    # @timeit
    def generate_M(paths_od: {tuple: Path},
                   paths=None,
                   ods_paths_idxs=False):
        """Matrix M: Path-OD pair incidence matrix"""

        # print('Generating matrix M')

        t0 = time.time()
        if paths is None:
            paths = []
            for pair in paths_od.keys():
                paths.extend(paths_od[pair])

        ods, n_ods, npaths = paths_od.keys(), len(paths_od.values()), len(paths)

        M = np.zeros([n_ods, npaths], dtype=np.int64)

        # Create dictionary Dict[od pair (tuple), columns idx of paths in that OD]
        ods_paths_ids = {}

        path_j = 0

        counter = 0
        # print('\n')
        for od, od_i in zip(ods, range(n_ods)):

            printer.printProgressBar(counter, n_ods-1, prefix='Progress(M):', suffix='', length=20, eraseBar = True)


            ods_paths_ids[od] = []

            for _ in paths_od[od]:
                ods_paths_ids[od].append(path_j)
                M[od_i, path_j] = 1
                path_j += 1

            counter += 1

        assert M.shape[0] > 0, 'No matrix M generated'

        print('Matrix M ' + str(M.shape) + ' generated in ' + str(round(time.time() - t0, 1)) + '[s]')

        if ods_paths_idxs:
            return M, ods_paths_ids
        else:
            return M

    @staticmethod
    def generate_C(M):
        """Wide to long format
        Choice_set_matrix_from_M
        The new matrix has one rows per alternative

        # This is the availability matrix or choice set matrix. Note that it is very expensive to
        compute when using matrix operation Iq.T.dot(Iq) so a more programatic method was preferred
        """

        t0 = time.time()

        assert M.shape[0] > 0, 'Matrix C was not generated because M matrix is empty'

        # print('Generating choice set matrix')

        wide_matrix = M.astype(int)

        if wide_matrix.ndim == 1:
            wide_matrix = wide_matrix.reshape(1, wide_matrix.shape[0])

        C = np.repeat(wide_matrix, repeats=np.sum(wide_matrix, axis=1), axis=0)

        print('Matrix C ' + str(C.shape) + ' generated in ' + str(round(time.time() - t0, 1)) + '[s]')

        return C



    def generate_edges_weights(self, V: Matrix) -> Dict:

        '''

        arguments:
            V (Matrix): Utility Matrix

        returns:
            edge weights in utility units
        '''

        # edges_weights_dict = dict(zip(dict(G.links).keys(), np.random.randint(0, 20, len(list(G.links)))))

        # To avoid problems with link with negative utilities, we deviates them by the link with the most negative values such that all have utilities greater or equal than 0.

        V = V+abs(np.min(V))


        edges_weights_dict = {}

        for index, vx in np.ndenumerate(V):
            edges_weights_dict[index] = vx

        return edges_weights_dict

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



    def copy_Z_attributes_dict_links(self, links_dict: {}):

        Z_labels = list(links_dict.values())[0].Z_dict.keys()

        for attr in Z_labels:
            for i, link in links_dict.items():

                self.links_dict[i].Z_dict[attr] = links_dict[i].Z_dict[attr]


    def reset_link_counts(self):

        for link in self.links:
            link.count = np.nan

    def load_traffic_counts(self, counts: Dict[str, float]):

        self.reset_link_counts()

        for link_key, count in counts.items():

            if not np.isnan(count):
                self.links_dict[link_key].count = float(count)

    @property
    def observed_counts_vector(self) -> ColumnVector:
        return self.link_data.observed_counts_vector

    def load_linkflows(self, x: Dict[str, float]):

        # self.reset_link_counts()

        for link_key, value in x.items():

            # if not np.isnan(value):
            self.links_dict[link_key].x = value

    @property
    def linkflows_vector(self) -> ColumnVector:
        return self.link_data.predicted_counts_vector

    def load_traveltimes(self, traveltimes: Dict[str, float]):
        self.set_Y_attr_links({'tt': traveltimes})

    def load_features_data(self,
                           linkdata: pd.DataFrame,
                           features: Optional[List[Feature]]= None,
                           link_key: Optional[str] = None):

        if link_key is None:
             link_key = 'link_key'

        if features is None:
            features = list(linkdata.columns)

            if link_key in features:
                features.pop(features.index(link_key))

        for key, link in self.links_dict.items():
            # Add elements to existing dictionary
            link_row = linkdata[linkdata[link_key].astype(str) == str(key)]
            link_features_dict = link.Z_dict
            for feature in features:
                link_features_dict[feature] = link_row[feature].values[0]
                # link.Z_dict[feature] = link_row[feature].values[0]


    def get_features_data(self, features):
        pass

        # linkdata = LinkData(link_key='key',
        #                     count_key='counts',
        #                     dataframe=link_features_df)

    def load_bpr_data(self,
                      linkdata: pd.DataFrame,
                      parameters_keys = ['tf', 'k', 'alpha', 'beta'],
                      link_key: Optional[str] = None):

        pass


    def set_bpr_functions(self,
                          bprdata: pd.DataFrame,
                          parameters_keys= ['alpha', 'beta','tf', 'k'],
                          link_key: Optional[str] = None) -> None:

        if link_key is None:
             link_key = 'link_key'

        links_keys = list(bprdata[link_key].values)

        for key in links_keys:

            alpha, beta, tf, k \
                = tuple(bprdata[bprdata[link_key] == key][parameters_keys].values.flatten())

            self.links_dict[key].performance_function = BPR(alpha=alpha,
                                                            beta=beta,
                                                            tf=tf,
                                                            k=k)

            # Add parameters to Z_Dict
            self.links_dict[key].Z_dict['alpha'] = alpha
            self.links_dict[key].Z_dict['beta'] = beta
            self.links_dict[key].Z_dict['tf'] = tf
            self.links_dict[key].Z_dict['k'] = k

            # Initialize link travel time
            self.links_dict[key].set_traveltime_from_x(x=0)


    def load_OD(self, Q=None, scale = 1):

        # print('\nLoading OD matrix')

        if Q is None:
            # Generate random OD matrix
            raise NotImplementedError

        self.OD.Q_true = Q.copy()

        self.OD.Q = scale*self.OD.Q_true.copy()

        # Store od pairs
        self.OD.ods = self.OD.ods_fromQ(Q=self.Q)

        total_trips = float(np.sum(self.OD.Q))

        print(str(round(total_trips,1)) + ' trips were loaded among '  + str(len(self.ods)) + ' o-d pairs')

    def scale_OD(self, scale = 1):

        # print('\nLoading OD matrix')

        self.OD.scale_OD(scale)

        print('OD was scaled with factor',scale)

    def update_incidence_matrices(self, paths_od = None):

        if paths_od is None:
            paths_od = self.paths_od

        print('Updating incidence matrices')

        # printer.blockPrint()

        self.D = self.generate_D(paths_od=paths_od, links=self.links)
        self.M = self.generate_M(paths_od=paths_od)
        self.C = self.generate_C(self.M)

        # printer.enablePrint()

    def load_paths(self,
                   paths = None,
                   paths_od = None,
                   update_incidence_matrices = True):

        # assert paths is not None, 'No paths have been provided'

        # self.paths = paths

        # if paths is None:
        #     paths = get_paths_from_paths_od(paths_od)

        if paths_od is None:
            paths_od = get_paths_od_from_paths(paths)

        self.paths_od = paths_od

        print(str(len(self.paths)) + ' paths were loaded in the network')

        if update_incidence_matrices:
            self.update_incidence_matrices(paths_od = paths_od)




    def copy_links_data(self, links: []):
        '''

        '''

        # for link in links:
        #     test = copy.copy(link)

        # self.links = [copy.deepcopy(link) for link in links]

        # self.links = [copy.copy(link) for link in links]
        self.links = links

        self.links_dict = {link.key: link for link in self.links}

        self.copy_Z_attributes_dict_links(links_dict = self.links_dict)

        # self.link_data.set_Z_attributes_dict_network(links_dict=self.links_dict)

        # self.copy_link_BPR_network(links = self.links_dict)

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


class OD:
    def __init__(self,
                 Q = None,
                 **kwargs):

        self.set_default_options()
        self.update_options(**kwargs)

        # if Q is None:
        #     Q = self.generate_Q()

        self._Q = Q
        self._q = None
        self.ods = []

        # Without scaling
        self._Q_true = None

        self.scale = 1

        # If error is introducted to the matrix, this variables store the noisy matrix
        self.Q_true = np.array([[]])

    def update_options(self,**kwargs):
        self.options =  self.options.get_updated_options(new_options = kwargs)

    def set_default_options(self):

        self.options = Options()

        # True M and D are smaller which speed up significantly the code
        self.options['remove_zeros_Q'] = True

    def ods_fromQ(self,
                  Q: np.matrix = None,
                  remove_zeros: bool = None):

        if Q is None:
            Q = self.Q

        if remove_zeros is None:
            remove_zeros = self.options['remove_zeros_Q']

        return ods_fromQ(Q =Q, remove_zeros=remove_zeros)

    def nonzero_ods_fromQ(self,
                  Q: np.matrix = None):

        if Q is None:
            Q = self.Q

        return list(map(tuple,list(np.transpose(np.nonzero(Q)))))

    def denseQ(self,
               Q: np.matrix,
               remove_zeros: bool = None):

        if Q is None:
            Q = self.options['Q']

        if remove_zeros is None:
            remove_zeros = self.options['remove_zeros_Q']

        q = denseQ(Q,remove_zeros)

        return q

    def scale_OD(self, scale = 1):

        # print('\nScaling OD matrix')

        self.scale = scale

        self.Q = self.scale*self.Q_true.copy()

    def update_Q_from_q(self,
                        q: np.array,
                        Q: np.Matrix,
                        removed_zeros: bool = True):

        new_Q = Q.copy()

        if removed_zeros:
            counter = 0
            for (i, j) in zip(*Q.nonzero()):
                new_Q[(i, j)] = q[counter]
                counter+=1
        else:
            counter = 0
            for i, j in np.ndenumerate(Q):
                new_Q[(i, j)] = q[counter]
                counter += 1

        self.Q = new_Q

        # return

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value
        self._q = denseQ(Q = self.Q)

    @property
    def Q_true(self):
        return self._Q_true

    @Q_true.setter
    def Q_true(self, value):
        self._Q_true = value
        self._q_true = denseQ(Q=self.Q_true)

    @property
    def q(self):
        return self._q
        # return denseQ(Q = self.Q)

    # @q.setter
    # def q(self, value):
    #     self._q = value

    @property
    def q_true(self):
        return self._q_true
        # return denseQ(Q = self.Q_true)

    # @q.setter
    # def q_true(self, value):
    #     self._q_true = value

    def sample_ods_by_demand(self, proportion, k = 1):

        ods_sorted = list(np.dstack(np.unravel_index(np.argsort(-self.Q.ravel()), self.Q.shape)))[0]

        if self.options['remove_zeros_Q']:
            n = len(self.q)

        n_samples = int(np.floor(proportion * n))

        # k select the kth set of ODs with largest demand
        max_k = int(n/n_samples)

        if k > max_k-1:
            k = k % max_k

        start = k*n_samples
        end = (k+1)*n_samples

        ods_sample = [tuple(ods_sorted[idx]) for idx in np.arange(start,end)]

        return ods_sample

    def random_ods(self, percentage):

        ods = []

        if self.options['remove_zeros_Q']:
            ods = self.nonzero_ods_fromQ()
        else:
            ods = self.ods_fromQ()

        n = len(ods)

        n_ods_sample = int(np.floor(percentage * n ))

        ods_sample = [ods[idx] for idx in np.random.choice(np.arange(len(ods)), n_ods_sample, replace=False)]

        return ods_sample


def get_observed_links(links: [] = None):

    """ Return list of links that have observed counts"""
    return [link for link in links if not np.isnan(link.count)]

def denseQ(Q: np.matrix,
           remove_zeros: bool = True):

    q = []  # np.zeros([len(np.nonzero(Q)[0]),1])

    if remove_zeros:
        for i, j in zip(*Q.nonzero()):
            q.append(Q[(i, j)])

    else:
        for i, j in np.ndenumerate(Q):
            q.append(Q[i])

    q = np.array(q)[:, np.newaxis]

    assert q.shape[1] == 1, "od vector is not a column vector"

    return q


def ods_fromQ(
        Q: np.matrix,
        remove_zeros: bool = None):

    # OD pairs
    ods = []

    if remove_zeros:
        # Do not account for ODs with no trips, then D and M are smaller
        for (i, j) in zip(*Q.nonzero()):
            ods.append((i, j))
    else:
        for i, j in np.ndenumerate(Q):
            ods.append(i)

    return ods






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


