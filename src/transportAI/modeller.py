"""
Modeller can estimate models from observed data.
"""

import pandas as pd
import numpy as np
from networks import TNetwork, MultiDiTNetwork, DiTNetwork
import networks
from links import Link, BPR
from nodes import Node
from paths import path_generation_nx
from geographer import LinkPosition, NodePosition
import writer
import reader
import estimation
import random
import time
# import config

from mytypes import Matrix, ColumnVector, Links, LogitFeatures, LogitParameters, Paths, Options

def tnetwork_factory(labels: {}, factory_options: {}, A: {}, links: Links = None, **kwargs)-> TNetwork:
    """ Create transportation network objects with links objects

     :arg labels: name(s) of created network(s)
     """

    for key, value in kwargs.items():
        factory_options[key] = value

    N = {}
    # i = 'N1'
    for i, A_i in A.items():

        print('\n' +'Creating ' + str(labels[i]) +' network\n')

        multinetwork = (A[i] > 1).any()
        dinetwork = (A[i] <= 1).any()

        if multinetwork:
            N[i] = MultiDiTNetwork(A=A[i], links = links)
        elif dinetwork:
            N[i] = DiTNetwork(A=A[i], links = links)
            # print(N[i].links[0].bpr)
            # print('here')

        # N[i].setup_options = setup_options
        N[i].key = labels[i]

        print('Nodes: ' + str(N[i].get_n_nodes()) + ', Links: '+ str(N[i].get_n_links()))

    return N

    # - Validation (TODO: Create tests with this matrices)

    # q1 = np.array([10])  # 1-2
    # N['N1'].M = np.array([1, 1, 1])[np.newaxis, :]
    # N['N1'].D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # q2 = np.array([10, 20, 30])  # 1-4, 2-4, 3-4
    # N['N2'].M = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]])
    # N['N2'].D = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])

    # q3 = np.array([10, 20])  # 1-3, 2-3
    # N['N3'].M = np.array([[1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1]])
    # N['N3'].D = np.array([[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0],
    #                [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]]
    #               )

def random_network_factory(N_keys: [], factory_options: {}, labels={}, **kwargs):

    """ Create a dictionary of network objects with a random adjacency matrices and their corresponding set of links """

    A = {}

    for key, value in kwargs.items():
        factory_options[key] = value

    nodes_range, q_range, cut_off = factory_options['nodes_range'], factory_options['q_range'], factory_options[
        'cutoff_paths']

    # print(nodes_range)
    # print(nodes_range[0])
    # print(nodes_range[-1])
    for i in N_keys:
        A[i] = TNetwork.randomDiNetwork(n_nodes=random.randint(nodes_range[0], nodes_range[-1])).A

    return tnetwork_factory(A=A, factory_options=factory_options, labels=labels)

def tntp_network_factory(subfoldername, folderpath, options, config, **kwargs):

    subfoldersnames = config.tntp_networks

    assert subfoldername in subfoldersnames, 'Invalid network name'

    for key, value in kwargs.items():
        options[key] = value

    # print(options['writing'])

    # Write dat files
    writer.write_tntp_github_to_dat(folderpath, subfoldername)

    # Read input dat file
    A_real, links_attrs = reader.read_tntp_network(folderpath=folderpath,
                                                   subfolder=subfoldername)

    # Create network object
    Nt_label = subfoldername

    Nt = tnetwork_factory(factory_options=options,
                          A={Nt_label: A_real},
                          labels={Nt_label: Nt_label}
                          )[Nt_label]

    links = Nt.links_dict
    # Set BPR functions and attributes values associated each link
    for index, row in links_attrs.iterrows():
        # TODO: Review if the parameters from the data have a consistent meaning with the way the BPR function is written in my code
        link_label = (int(row['init_node']), int(row['term_node']), '0')

        # BPR functions
        links[link_label].performance_function = BPR(alpha=row['b'], beta=row['power'],
                                                                       tf=row['free_flow_time'], k=row['capacity'])

        # Attributes
        links[link_label].Z_dict['speed'] = pd.to_numeric(row['speed'], errors='coerce',
                                                          downcast='float')  # It ssemes to be the speed limit
        links[link_label].Z_dict['toll'] = pd.to_numeric(row['toll'], errors='coerce', downcast='float')
        links[link_label].Z_dict['length'] = pd.to_numeric(row['length'], errors='coerce', downcast='float')

        # Replace Nan values with nas
        for i in ['speed', 'toll', 'length']:
            if np.isnan(links[link_label].Z_dict[i]):
                links[link_label].Z_dict[i] = float('nan') #float(0)
        # Identical/Similar to free flow travel time


    # Write paths if required
    # print('here')
    # print(len(Nt.paths_od.keys()))

        # print('here 2')
        # print(len(Nt.paths_od.keys()))

        # # Generate D and M matrix
        # Nt.M = Nt.generate_M(paths_od=Nt.paths_od, paths=Nt.paths)
        # Nt.D = Nt.generate_D(paths_od=Nt.paths_od, links=Nt.links, paths=Nt.paths)
        #
        # # Generate abd store choice set matrix
        # Nt.C = estimation.choice_set_matrix_from_M(Nt.M)


    # reader.network_reader(Nt, options)
    # writer.network_writer(Nt, options)

    return Nt

def colombus_network_factory(folder, label, options, **kwargs):

    # # Write dat files
    # writer.write_tntp_github_to_dat(folder, subfolder)

    # print(read_paths)

    # folder = tai.config.paths['Colombus_network']

    for key, value in kwargs.items():
        options[key] = value

    # Read files
    A, links_df, nodes_df = reader.read_colombus_network(
        folderpath=folder)

    #Create link objects and set BPR functions and attributes values associated each link
    links = {}

    for index, row in links_df.iterrows():
        # TODO: Review if the parameters from the data have a consistent meaning with the way the BPR function is written in my code

        link_key = (int(row['init_node_key']), int(row['term_node_key']), '0')

        # Adding gis information via nodes object store in each link
        init_node_row = nodes_df[nodes_df['key'] == link_key[0]]
        term_node_row = nodes_df[nodes_df['key'] == link_key[1]]


        # TODO: double check with Bin about the matching of the network elements and the shape file information. He told me that the ids did not perfectly match and that I should look at the jupyter notebook

        # x_cord_origin, y_cord_origin = tuple(list(init_node_row[['x', 'y']].values[0]))
        # x_cord_term, y_cord_term = tuple(list(term_node_row[['x', 'y']].values[0]))
        #
        # node_init = Node(key=link_key[0], position=NodePosition(x_cord_origin, y_cord_origin, crs='xy'))
        # node_term = Node(key=link_key[1], position=NodePosition(x_cord_term, y_cord_term, crs='xy'))

        node_init = Node(key=link_key[0])
        node_term = Node(key=link_key[1])

        links[link_key] = Link(key=link_key, init_node=node_init, term_node=node_term)

        # Store original ids from nodes and links
        links[link_key].init_node.id = str(init_node_row['id'].values[0])
        links[link_key].term_node.id = str(term_node_row['id'].values[0])
        # note that some ids include a large tab before the number comes up ('   1), I may remove those spaces
        links[link_key].id = row['id']

        # Attributes
        links[link_key].Z_dict['capacity_car'] = pd.to_numeric(row['capacity_car'], errors='coerce', downcast='float')
        links[link_key].Z_dict['capacity_truck'] = pd.to_numeric(row['capacity_truck'], errors='coerce', downcast='float')
        links[link_key].Z_dict['ff_speed_car'] = pd.to_numeric(row['ff_speed_car'], errors='coerce',downcast='float')
        links[link_key].Z_dict['ff_speed_truck'] = pd.to_numeric(row['ff_speed_truck'], errors='coerce', downcast='float')
        links[link_key].Z_dict['rhoj_car'] = pd.to_numeric(row['rhoj_car'], errors='coerce', downcast='float')
        links[link_key].Z_dict['rhoj_truck'] = pd.to_numeric(row['rhoj_truck'], errors='coerce', downcast='float')
        links[link_key].Z_dict['length'] = pd.to_numeric(row['length'], errors='coerce', downcast='float')
        links[link_key].Z_dict['lane'] = pd.to_numeric(row['lane'], errors='coerce', downcast='integer')
        links[link_key].Z_dict['conversion_factor'] = pd.to_numeric(row['conversion_factor'], errors='coerce', downcast='float')

        # We assume that the free flow speed and capacity are those associated to cars and not trucks
        links[link_key].Z_dict['capacity'] = links[link_key].Z_dict['capacity_car']
        links[link_key].Z_dict['ff_speed'] = links[link_key].Z_dict['ff_speed_car']

        # BPR functions
        # Parameters of BPR function are assumed to be (alpha, beta) = (0.15, 4). Source: https://en.wikipedia.org/wiki/Route_assignment

        links[link_key].performance_function \
            = BPR(alpha=0.15, beta=4
                                    , tf=links[link_key].Z_dict['ff_speed']
                                    , k=links[link_key].Z_dict['capacity'])

    # Create network object
    Nt = tnetwork_factory(factory_options=options, A={label: A}
                          , labels={label: label}, links = list(links.values()))[label]

    # TODO: Optimize the generation of the path link incidence matrix as it is taking too long for Ohio. Sparse matrix and a better index method to index links may help.
    # Nt.D = Nt.generate_D(paths_od=Nt.paths_od, links = Nt.links, paths = Nt.paths)

    return Nt

def fresno_network_factory(folder, label, options, **kwargs):

    for key, value in kwargs.items():
        options[key] = value

    # Read files
    A, links_df, nodes_df = reader.read_fresno_network(
        folderpath=folder)

    # Create link objects and set BPR functions and attributes values associated each link
    links = {}

    for index, row in links_df.iterrows():
        # TODO: Review if the parameters from the data have a consistent meaning with the way the BPR function is written in my code

        # print(row)
        link_key = (int(row['init_node_key']), int(row['term_node_key']), '0')

        # Adding gis information via nodes object store in each link
        init_node_row = nodes_df[nodes_df['key'] == link_key[0]]
        term_node_row = nodes_df[nodes_df['key'] == link_key[1]]

        x_cord_origin,y_cord_origin = tuple(list(init_node_row[['x','y']].values[0]))
        x_cord_term, y_cord_term = tuple(list(term_node_row[['x','y']].values[0]))

        node_init = Node(key = link_key[0], position = NodePosition(x_cord_origin,y_cord_origin, crs = 'xy'))
        node_term = Node(key=link_key[1], position = NodePosition(x_cord_term, y_cord_term, crs = 'xy'))


        # links[link_key] = Link(key= link_key, init_node = node_init, term_node = node_term)

        links[link_key] = Link(key=link_key, init_node=node_init, term_node = node_term)

        #Store original ids from nodes and links
        links[link_key].init_node.id = str(init_node_row['id'].values[0])
        links[link_key].term_node.id = str(term_node_row['id'].values[0])
        # note that some ids include a large tab before the number comes up ('   1), I may remove those spaces
        links[link_key].id = row['id']

        # Attributes
        links[link_key].link_type = row['type'].strip()
        links[link_key].Z_dict['capacity'] = pd.to_numeric(row['capacity'], errors='coerce', downcast='float')
        links[link_key].Z_dict['ff_speed'] = pd.to_numeric(row['ff_speed'], errors='coerce',downcast='float')
        links[link_key].Z_dict['rhoj'] = pd.to_numeric(row['rhoj'], errors='coerce', downcast='float')
        links[link_key].Z_dict['length'] = pd.to_numeric(row['length'], errors='coerce', downcast='float')
        links[link_key].Z_dict['lane'] = pd.to_numeric(row['lane'], errors='coerce', downcast='integer')

        # Weighting by 60 will leave travel time with minutes units, because speeds are originally in per hour units
        if options['tt_units'] == 'minutes':
            tt_factor = 60

        if options['tt_units'] == 'seconds':
            tt_factor = 60*60

        links[link_key].Z_dict['ff_traveltime'] = tt_factor*links[link_key].Z_dict['length']/links[link_key].Z_dict['ff_speed']

        # BPR functions
        # Parameters of BPR function are assumed to be (alpha, beta) = (0.15, 4). Source: https://en.wikipedia.org/wiki/Route_assignment
        links[link_key].performance_function \
            = BPR(alpha=options['bpr_parameters']['alpha'], beta=options['bpr_parameters']['beta']
                                    , tf=links[link_key].Z_dict['ff_traveltime']
                                    , k=links[link_key].Z_dict['capacity'])

    # Create network object
    # TODO: Run shortest path with Igraph and Graphx for dinetworks. This can speed up this execution. Alternatively, the paths could be read and generated from path file

    Nt = tnetwork_factory(factory_options=options, A={label: A}
                          , labels={label: label}, links = list(links.values()))[label]

    # TODO: Optimize the generation of the path link incidence matrix as it is taking too long for Ohio. Sparse matrix and a better index method to index links may help.
    # Nt.D = Nt.generate_D(paths_od=Nt.paths_od, links = Nt.links, paths = Nt.paths)

    return Nt

def sacramento_network_factory(folder, label, options, **kwargs):

    # TODO: Take advantage of nodes information

    for key, value in kwargs.items():
        options[key] = value

    # Read files
    A, links_df, nodes_df = reader.read_sacramento_network(
        folderpath=folder)

    # Create link objects and set BPR functions and attributes values associated each link
    links = {}

    for index, row in links_df.iterrows():
        # TODO: Review if the parameters from the data have a consistent meaning with the way the BPR function is written in my code

        # print(row)
        link_key = (int(row['init_node_key']), int(row['term_node_key']), '0')

        # Adding gis information via nodes object store in each link
        init_node_row = nodes_df[nodes_df['key'] == link_key[0]]
        term_node_row = nodes_df[nodes_df['key'] == link_key[1]]

        x_cord_origin,y_cord_origin = tuple(list(init_node_row[['x','y']].values[0]))
        x_cord_term, y_cord_term = tuple(list(term_node_row[['x','y']].values[0]))

        node_init = Node(key = link_key[0], position = NodePosition(x_cord_origin,y_cord_origin, crs = 'xy'))
        node_term = Node(key=link_key[1], position = NodePosition(x_cord_term, y_cord_term, crs = 'xy'))


        # links[link_key] = Link(key= link_key, init_node = node_init, term_node = node_term)

        links[link_key] = Link(key=link_key, init_node=node_init, term_node = node_term)

        #Store original ids from nodes and links
        links[link_key].init_node.id = str(init_node_row['id'].values[0])
        links[link_key].term_node.id = str(term_node_row['id'].values[0])
        # note that some ids include a large tab before the number comes up ('   1), I may remove those spaces
        links[link_key].id = row['id']

        # Attributes
        links[link_key].Z_dict['capacity'] = pd.to_numeric(row['capacity'], errors='coerce', downcast='float')
        links[link_key].Z_dict['ff_speed'] = pd.to_numeric(row['ff_speed'], errors='coerce',downcast='float')
        links[link_key].Z_dict['rhoj'] = pd.to_numeric(row['rhoj'], errors='coerce', downcast='float')
        links[link_key].Z_dict['length'] = pd.to_numeric(row['length'], errors='coerce', downcast='float')
        links[link_key].Z_dict['lane'] = pd.to_numeric(row['lane'], errors='coerce', downcast='integer')

        # BPR functions
        # Parameters of BPR function are assumed to be (alpha, beta) = (0.15, 4). Source: https://en.wikipedia.org/wiki/Route_assignment
        links[link_key].performance_function \
            = BPR(alpha=0.15, beta=4
                                    , tf=links[link_key].Z_dict['ff_speed']
                                    , k=links[link_key].Z_dict['capacity'])

    # Create network object
    # TODO: Run shortest path with Igraph and Graphx for dinetworks. This can speed up this execution. Alternatively, the paths could be read and generated from path file

    Nt = tnetwork_factory(factory_options=options, A={label: A}
                          , labels={label: label}, links = list(links.values()))[label]

    # TODO: Optimize the generation of the path link incidence matrix as it is taking too long for Ohio. Sparse matrix and a better index method to index links may help.
    # Nt.D = Nt.generate_D(paths_od=Nt.paths_od, links = Nt.links, paths = Nt.paths)

    return Nt

def setup_network(Nt: TNetwork, setup_options: Options, Q: Matrix = None, M: Matrix = None, D: Matrix = None, paths = None, message = True, **kwargs):
    '''
    Create network with random attributes and Q matrix if required. The mandatory input if a Tnetwork object

    # - Set values of attributes in matrix Z, including those sparse (n_R) or not.

    # Incident matrices are created
    
    This function allows to deal with reading data from files which are generated by this package 

    :argument randomness: dictionary of booleans with keys ['Q', 'BPR', 'Z']
    :argument Nt: dictionary of networks

    #TODO: I may combine this method with the custom_factory to avoid dupplication

    TODO: network object should store all the input in a dictionary called 'setup_options'

    '''
    # generation = dict(tai.config.sim_options, generation = {'Q': True, 'bpr': False, 'Z': True})
    # kwargs = generation

    # Additional options will override global options
    for key, value in kwargs.items():
        # print(key)
        setup_options[key] = value
        # print("%s == %s" % (key, value))

    if message:
        print('\n' + 'Setting up network ' + str(Nt.key) + '\n')

    # print('here')

    # N = {}
    min_q, max_q = setup_options['q_range']
    q_sparsity = setup_options['q_sparsity']
    cutoff_paths = setup_options['cutoff_paths']
    R_labels = setup_options['R_labels']
    Z_attrs_classes = setup_options['Z_attrs_classes']
    fixed_effects = setup_options['fixed_effects']
    bpr_classes = setup_options['bpr_classes']
    remove_zeros_Q = setup_options['remove_zeros_Q']
    n_paths = setup_options['n_initial_paths']

    # try:
    #     randomness = setup_options['randomness']
    # except:
    #     randomness = kwargs['randomness']

    # This method may be used with a single network. In that case, a dictionary has to be created

    # if not isinstance(Nt, dict):
    #     Nt = {Nt.label: Nt}
    # N = {N.label:N if not isinstance(N,dict) else break}

    # print('here 2')


    # Q = N_i.random_disturbance_Q(Q, var=randomness['var_Q'])

    #     # if i == 'N1':
    #
    #
    # except:
    #     Q = N_i.random_disturbance_Q(Q)

    # if len(labels)> 0:
    # Nt.label = label[i]

    # N_copy[i] = custom_network_factory(A= {i:N_i.A}, Q={i:Q}, labels = {i: label + ' ' + str(i)}
    #                                    , cutoff_paths=cutoff_paths, remove_zeros_Q= remove_zeros_Q, n_paths= n_paths).get(i)

    # Store setup options
    Nt.setup_options = setup_options

    #Call reader
    reader.read_internal_network_files(Nt, setup_options)

    if setup_options['generation']['Q'] and Q is None:

        # TODO: provide paths per od to generate Q, otherwise is very expensive to generate this matrix. If paths are not provided, then path generation should be performed.

        Nt.Q = Nt.generate_Q(Nt=Nt, min_q=min_q, max_q=max_q, cutoff=cutoff_paths, n_paths = n_paths, sparsity = q_sparsity)

    if Q is not None:
        Nt.Q = Q

    # assert Nt.Q.shape[0] > 0, 'Invalid matrix Q was provided or it could not be generated'
    # print('Matrix Q was successfuly generated')

    # Dense od vector
    Nt.q = networks.denseQ(Nt.Q, remove_zeros=remove_zeros_Q)

    # Store od pairs
    Nt.ods = Nt.ods_fromQ(Q=Nt.Q, remove_zeros=remove_zeros_Q)

    print(str(len(Nt.ods)) + ' o-d pairs')


    # Link level attributes

    # - Bpr functions
    if setup_options['generation']['bpr']:
        Nt.set_random_link_BPR_network(bpr_classes=bpr_classes)

    else:
        # It assumes that the set of existing links in the network already have bpr functions
        Nt.copy_link_BPR_network(links=Nt.links_dict)

    # # - Existing exogenous attributes
    # Nt.copy_Z_attributes_dict_links(links=Nt.links_dict)

    if setup_options['generation']['Z']:
        # If randomness is used, the Z attributes will be replaced by random values
        Nt.set_random_link_Z_attributes_network(
            Z_attrs_classes=Z_attrs_classes
            , R_labels=R_labels
        )

    # Fixed effects
    if setup_options['generation']['fixed_effects']:
        Nt.set_fixed_effects_attributes(fixed_effects=fixed_effects)

    # Update dictionary with attributes values at the network level
    Nt.set_Z_attributes_dict_network(links_dict=Nt.links_dict)

    # # Setup links and paths
    # self.setup_links(links)
    # print(setup_options['path_generation'])

    # Write paths if required
    # if setup_options['writing']['paths']:
    #     transportAI.writer.write_paths(N[i].paths, N[i].label)
    #
    #     # Links in the network should be updated so paths and links are associated as wished.

    if setup_options['generation']['paths'] and paths is None:
        # print('No paths provided... Generating paths ...')
        # print('Generating paths')

        if 'theta' in setup_options.keys():

            # Matrix with link utilities
            Nt.V = Nt.generate_V(A=Nt.A, links=Nt.links, theta=setup_options['theta'])

            # Key to have the minus sign so we look the route that lead to the lowest disutility
            edge_utilities = Nt.generate_edges_weights_dict_from_utility_matrix(V=Nt.V)

            Nt.paths, Nt.paths_od = path_generation_nx(A=Nt.A
                                                       , ods=Nt.ods
                                                       , links=Nt.links_dict
                                                       , cutoff=cutoff_paths, n_paths=n_paths
                                                       , edge_weights = edge_utilities
                                                       )


        else:
            # This uses the number of links for shortest paths only
            Nt.paths, Nt.paths_od = path_generation_nx(A=Nt.A
                                                       , ods=Nt.ods
                                                       , links=Nt.links_dict
                                                       , cutoff=cutoff_paths, n_paths=n_paths)

    if paths is not None:
        # TODO: Matching may be useful for clone method
        # Nt.match_paths_with_existing_links(paths)
        Nt.paths = paths
        Nt.paths_od = Nt.get_paths_od_from_paths(Nt.paths)

        # print(len(Nt.paths_od.keys()))
        # print(len(Nt.paths))
        #


    # Network matrices

    # - Path-OD incidence matrix D


    if setup_options['generation']['M'] and M is None:
        # print(Nt.paths_od)
        # print(Nt.paths)
        Nt.M = Nt.generate_M(paths_od=Nt.paths_od)
        # print('here m')

    if M is not None:
        Nt.M = M

    # assert Nt.M.shape[0] > 0, 'Invalid matrix M was provided or it could not be generated'

    # - Path-link incidence matrix D
    if setup_options['generation']['D'] and D is None:
        # print(len(Nt.paths))
        # print(len(Nt.paths_od))

        Nt.D = Nt.generate_D(paths_od=Nt.paths_od, links=Nt.links)

    if D is not None:
        Nt.D = D

    # assert Nt.D.shape[0] > 0, 'Invalid matrix D was provided or it could not be generated'

    # Choice set matrix
    if setup_options['generation']['C']:
        Nt.C = estimation.choice_set_matrix_from_M(Nt.M)

    # assert Nt.C.shape[0] > 0, 'Choice set matrix could not be generated'

    #Call writer
    writer.write_internal_network_files(Nt, setup_options)

    # valid_network = None

    # while valid_network is None:
    #     try:
    #
    #         # Compute SUE
    #         results_sue = {i: transportAI.equilibrium.sue_logit(q=transportAI.denseQ(Q = N_i.Q, remove_zeros=remove_zeros_Q)
    #                                                             , M=N_i.M
    #                                                             , D= N_i.D
    #                                                             , paths = N_i.paths
    #                                                             , links=N_i.links_dict
    #                                                             , Z_dict = N_i.Z_dict
    #                                                             , theta= theta
    #                                                             , cp_solver = cp_solver
    #                                                             )
    #                            for i, N_i in N_copy.items()}
    #     except:
    #         exceptions[i] += 1
    #         # print('error')
    #         pass
    #
    #     else:
    #         # print('ok')
    #         valid_network = True
    #
    #         # Store travel time, link and path flows in Network objects
    #         for i in N_copy.keys():
    #             N_copy[i].set_Y_attr_links(y=results_sue[i]['tt_x'], label='tt')
    #             N_copy[i].x_dict = results_sue[i]['x']
    #             N_copy[i].f_dict = results_sue[i]['f']

    # # Endogenous attributes at path level (e.g. traveltime: tt)
    # Y_links = {i:np.hstack(list(results_sue[i]['tt_x'].values())) for i in N.keys()}
    # Y_routes = {i: Y_links[i]@ N[i].D for i in N.keys()}
    #
    # # # + Exogenous attributes from link to path level (rows)
    # Z_links = {i:transportAI.logit.get_matrix_from_dict_attrs_values(N[i].Z_dict) for i in N.keys()}
    # Z_routes = {i:(Z_links[i].T @ N[i].D).T for i in N.keys()}

    return Nt  # results_sue, exceptions, Y_links, Y_routes, Z_links, Z_routes

def setup_tntp_network(Nt: TNetwork, setup_options: Options, **kwargs):

    """ This function allows to deal with reading data from tntp files which are not generated by this package """

    print('\n' + 'Setting up ' + str(Nt.key) + ' network \n')

    for key, value in kwargs.items():
        # print(key)
        setup_options[key] = value

    Q = None

    if setup_options['reading']['Q'] and not setup_options['reading']['sparse_Q']:

        # print('here reading ')

        Q = reader.read_tntp_od(folderpath= setup_options['folder'], subfolder = setup_options['subfolder'])
        
        # Do not need to read again using internal reader
        setup_options['reading']['Q'] = False

    # The matrix Q was read and store in network object using tntp_factory method
    Nt = setup_network(Nt, Q = Q, message = False, setup_options = setup_options)

    return Nt

def setup_colombus_network(Nt: TNetwork, setup_options: Options, **kwargs):
    """ This function allows to deal with reading data from colombus files which are not generated by this package """

    for key, value in kwargs.items():
        # print(key)
        setup_options[key] = value

    print('\n' + 'Setting up network ' + str(Nt.key) + '\n')

    Q = None

    # TODO: reading should be done later only from internal files. After the reading of external files is done, a translation method should write them in the internal format

    # Create paths using dictionary of existing links and information read from txt
    if setup_options['reading']['paths']:
        reader.read_colombus_paths(Nt = Nt, filepath= setup_options['folder'] + '/ODE_outputs/path_table')

        # Do not need to read again using internal reader
        setup_options['reading']['paths'] = False

    if setup_options['reading']['Q'] and not setup_options['reading']['sparse_Q']:
        Q = reader.read_colombus_od(A = Nt.A, nodes = Nt.nodes, folderpath=setup_options['folder'])

        # Do not need to read again using internal reader
        setup_options['reading']['Q'] = False


    # The matrix Q was read and store in network object using colombus_factory method
    Nt = setup_network(Nt, Q=Q, message = False, setup_options=setup_options)

    return Nt

def setup_fresno_network(Nt: TNetwork, setup_options: Options, **kwargs):
    """ This function allows to deal with reading data from colombus files which are not generated by this package """

    for key, value in kwargs.items():
        # print(key)
        setup_options[key] = value

    print('\n' + 'Setting up network ' + str(Nt.key) + '\n')

    Q = None

    # TODO: reading should be done later only from internal files. After the reading of external files is done, a translation method should write them in the internal format

    if setup_options['reading']['Q'] and not setup_options['reading']['sparse_Q']:
        # Q = reader.read_fresno_od(A = Nt.A, folderpath=setup_options['folder'], nodes = Nt.nodes)
        Q = reader.read_fresno_dynamic_od(A=Nt.A, folderpath=setup_options['folder'], nodes=Nt.nodes, periods = setup_options['od_periods'])

        # Do not need to read again using internal reader
        setup_options['reading']['Q'] = False

    setup_options['folder'] = setup_options['folder'] + '/SR41'

    # The matrix Q was read and store in network object using colombus_factory method
    Nt = setup_network(Nt, Q=Q, message = False, setup_options=setup_options)

    return Nt

def setup_sacramento_network(Nt: TNetwork, setup_options: Options, **kwargs):
    """ This function allows to deal with reading data from colombus files which are not generated by this package """

    for key, value in kwargs.items():
        # print(key)
        setup_options[key] = value

    print('\n' + 'Setting up network ' + str(Nt.key) + '\n')

    Q = None

    # TODO: reading should be done later only from internal files. After the reading of external files is done, a translation method should write them in the internal format

    if setup_options['reading']['Q'] and not setup_options['reading']['sparse_Q']:
        Q = reader.read_sacramento_od(A = Nt.A, folder=setup_options['folder'], nodes = Nt.nodes)

        # Do not need to read again using internal reader
        setup_options['reading']['Q'] = False

    setup_options['folder'] = setup_options['folder'] + '/sac'

    # The matrix Q was read and store in network object using colombus_factory method
    Nt = setup_network(Nt, Q=Q, message = False, setup_options=setup_options)

    return Nt

def clone_network(N: TNetwork, **kwargs) -> TNetwork:
    '''

    :param N: Single network
    :param kwargs:
    :return:
    '''

    # TODO: Fix clone method under new modifications of setup and factory methods

    clone_options = N.setup_options
    # setup_options ["label", "remove_zeros_Q", "q_range", "R_labels", "cutoff_paths", "n_paths"
    # , "Z_attrs_classes", "bpr_classes", "randomness", "fixed_effects"]

    for key, value in kwargs.items():
        clone_options[key] = value
        # print("%s == %s" % (key, value))

    # Default for randomness
    clone_options['randomness'] = {'Q': False, 'BPR': False, 'Z': False}

    # Q = copy.deepcopy(N.Q)
    # A = copy.deepcopy(N.A)
    # N.links[0].bpr.bpr_function_x(0)
    # print(N.links[0].bpr.bpr_function_x(0))

    N_copy = tnetwork_factory(A={N.key: N.A}, Q={N.key: N.Q}
                              , M={N.key: N.M}, D={N.key: N.D}
                              , links=N.links, paths_od={N.key: N.paths_od}
                              , labels={N.key: clone_options['label']}
                              , factory_options=clone_options)

    # print(N_copy[N.label].links[0].bpr)

    N_copy = setup_network(Nt=N_copy, **clone_options)

    return N_copy[N.key]

class Modeller:

    def __init__(self, name = None, date = None):
        self._name = name
        self._date = date

    def create_od_from_system(self, trips):
        od = pd.pivot_table(trips[['ostation', 'dstation']], index=['ostation'], columns=['dstation'], aggfunc=[len])
        od = od.replace(np.nan, 0)

        return od

    def create_od_from_agents(self, trips):

        df = pd.DataFrame(columns = ['o','d'])

        for i in range(len(trips)):
            trip = trips[i]
            df = df.append({'o':trip.init, 'd':trip.destination}, ignore_index = True)

        od = pd.pivot_table(df[['o', 'd']], index=['o'], columns=['d'], aggfunc=[len])
        od = od.replace(np.nan, 0)

        np.sum(od.to_numpy())  # Checking the total number of trips is consistent with the original dataset

        return od
    # def build_od(self, trips):
    #     """:argument trips: list of trips"""