"""Reader reads data of different formats and return objects that are suitable to be read by others"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mytypes import Nodes, Paths, Matrix,TNetwork, DiTNetwork, MultiDiTNetwork

from bs4 import BeautifulSoup
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import requests
import re
import urllib
import io

import os
import openmatrix as omx
import time
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import scipy.sparse
import csv
import sys

csv.field_size_limit(sys.maxsize)  # to support reading large matrices

import printer

from paths import Path, get_paths_od_from_paths
from networks import MultiDiTNetwork

import config

def read_colombus_paths(network:TNetwork,
                        filepath: str) -> Paths:
    """
    Read path data from Colombus network
    It assumes that a dictionary of existing links is available in the input network
    Each line in the text file has a sequence of nodes separated by an empty space


    :returns
    List of paths and a dictionary with a list of path per OD


    """

    print('Reading paths from external file')

    paths = []
    paths_od = {}
    ods = set()
    t0 = time.time()

    with open(filepath, 'r', newline='') as csvfile:
        path_reader = csv.reader(csvfile, delimiter=' ')
        total_paths = len(list(path_reader))
        print('Total paths to be read: ' + str(total_paths))

    if network.network_type is MultiDiTNetwork:
        raise NotImplementedError

    # Create hash table (dictionary) to speed up mapping from original to internal ids (keys)
    links_ids_to_keys = {(int(link.init_node.id), int(link.term_node.id), '0'): link.key for link in network.links}

    with open(filepath, 'r', newline='') as csvfile:
        path_reader = csv.reader(csvfile, delimiter=' ')

        for path_line, counter in zip(path_reader, range(total_paths)):

            printer.printProgressBar(counter, total_paths, prefix='Progress (paths):', suffix='',
                                     length=20)

            links = []

            for i in range(len(path_line) - 1):
                # links = links.append(Link(label=(path_line[i], path_line[i + 1])))

                # Key of link but with original ids
                link_key_id = (int(path_line[i]), int(path_line[i + 1]), '0')

                # Key of link based on internal id
                link_key = links_ids_to_keys[link_key_id]

                # if network.network_type is DiTNetwork:
                links.append(network.links_dict[link_key])

                # links.append(self._links_dict[(path_line[i], path_line[i + 1], '0')])

            origin = links[0].init
            destination = links[-1].term

            path = Path(origin, destination, links)

            if (origin, destination) in ods:
                paths_od[(origin, destination)].append(path)
            else:
                paths_od[(origin, destination)] = [path]

            ods.update([(origin, destination)])

            paths.append(path)

    assert len(paths) >= total_paths, str(total_paths-len(paths)) + ' were not succesfully read'
    print(str(total_paths) + ' were read in ' + str(np.round(time.time() - t0, 1)) + ' [s]')

    network.paths = paths
    network.paths_od = paths_od

    # return paths, paths_od


def read_internal_paths(network: TNetwork) -> Paths:
    """ It assumes that a dictionary of existing links is available and the path generated and rematched to those links

    Each line in the text file contains a sequence of nodes separated by a comma

    """

    paths = []

    filepath = config.dirs['read_network_data'] + 'paths/paths-' + network.key + '.csv'

    if network.network_type is MultiDiTNetwork:
        raise NotImplementedError

    with open(filepath, 'r', newline='') as csvfile:
        path_reader = csv.reader(csvfile, delimiter=' ')
        total_paths = len(list(path_reader))
        # print('Total paths to be read: ' + str(total_paths))

    t0 = time.time()

    with open(filepath, 'r', newline='') as csvfile:

        # print('reading')

        path_reader = csv.reader(csvfile, delimiter=',')

        # Generate a list of links from each line depicting a certain path

        for path_line, counter in zip(path_reader, range(total_paths)):

            printer.printProgressBar(counter, total_paths-1, prefix='Progress:', suffix='',length=20)

            links = []

            for i in range(len(path_line) - 1):
                # links = links.append(Link(label=(path_line[i], path_line[i + 1])))
                link_label = (int(path_line[i]), int(path_line[i + 1]), '0')
                # print(link_label)

                # Searching for existing links in network object that match the current path

                # if network.network_type is DiTNetwork:
                links.append(network.links_dict[link_label])

                # if network.network_type is MultiDiTNetwork:
                #     raise NotImplementedError
                # links.append(self._links_dict[(path_line[i], path_line[i + 1], '0')])

            # Create a path object

            origin = links[0].init
            destination = links[-1].term

            # print([origin,destination])

            path = Path(origin, destination, links)

            # print(path.get_nodes_labels())

            paths.append(path)

    assert len(paths) > 0, 'Paths were not succesfully read'
    print(str(total_paths) + ' paths were read in ' + str(np.round(time.time() - t0, 1)) + '[s]          ')
    
    return paths

    # # Do not need to read again
    # options['reading']['paths'] = False
    #
    # print(len(paths))
    # print(len(paths_od))

    # print('Paths were succesfully read')


def read_internal_C(network: TNetwork, 
                    sparse_format: bool = False) -> Matrix:
    
    format_label = 'sparse' if sparse_format else 'dense'

    # print('Reading C in ' + format_label + ' format')

    t0 = time.time()

    C_rows = []

    if sparse_format:
        filepath = config.dirs['read_network_data'] + 'C/C-sparse-' + network.key + '.npz'
        C = np.array(scipy.sparse.load_npz(filepath).todense())

    else:
        filepath = config.dirs['read_network_data'] + 'C/C-' + network.key + '.csv'

        with open(filepath, 'r', newline='') as csvfile:
            rows_reader = csv.reader(csvfile, delimiter=' ')
            total_rows = len(list(rows_reader))
            print('Total rows to be read: ' + str(total_rows))

        with open(filepath, 'r', newline='') as csvfile:
            rows_reader = csv.reader(csvfile, delimiter=',')

            for C_row, counter in zip(rows_reader, range(rows_reader)):
                printer.printProgressBar(counter, total_rows, prefix='Progress (C):', suffix='', length=20)

                C_rows.append(list(map(float, C_row)))

        C = np.array(C_rows)

    assert C.shape[0] > 0, 'Matrix C was not succesfully read using ' + format_label + ' format'
    print('Matrix C ' + str(C.shape) + ' read in ' + str(
        round(time.time() - t0, 1)) + '[s] with ' + format_label + ' format')
    
    return C


def read_internal_D(network: TNetwork, 
                    sparse_format: bool = False) -> Matrix:
    format_label = 'sparse' if sparse_format else 'dense'

    # print('Reading D in ' + format_label + ' format')

    t0 = time.time()

    D_rows = []

    if sparse_format:
        filepath = config.dirs['read_network_data'] + 'D/D-sparse-' + network.key + '.npz'
        D = np.array(scipy.sparse.load_npz(filepath).todense())

    else:
        filepath = config.dirs['read_network_data'] + 'D/D-' + network.key + '.csv'
        with open(filepath, 'r', newline='') as csvfile:
            rows_reader = csv.reader(csvfile, delimiter=' ')
            total_rows = len(list(rows_reader))
            print('Total rows to be read: ' + str(total_rows))

        with open(filepath, 'r', newline='') as csvfile:
            rows_reader = csv.reader(csvfile, delimiter=',')

            for D_row, counter in zip(rows_reader, range(total_rows)):
                printer.printProgressBar(counter, total_rows, prefix='Progress (D):', suffix='', length=20)

                D_rows.append(list(map(float, D_row)))

        D = np.array(D_rows)

    assert D.shape[0] > 0, 'Matrix D was not succesfully read using ' + format_label + ' format'
    print('Matrix D ' + str(D.shape) + ' read in ' + str(
        round(time.time() - t0, 1)) + '[s] with ' + format_label + ' format')
    
    return D


def read_internal_M(network: TNetwork, 
                    sparse_format: bool = False) -> Matrix:
    # TODO: distributed storage per OD pair for huge matrices

    format_label = 'sparse' if sparse_format else 'dense'

    # print('Reading M in ' + format_label + ' format')

    t0 = time.time()

    M_rows = []

    if sparse_format:
        filepath = config.dirs['read_network_data'] + 'M/M-sparse-' + network.key + '.npz'
        M = np.array(scipy.sparse.load_npz(filepath).todense())


    else:
        filepath = config.dirs['read_network_data'] + 'M/M-' + network.key + '.csv'
        with open(filepath, 'r', newline='') as csvfile:
            rows_reader = csv.reader(csvfile, delimiter=' ')
            total_rows = len(list(rows_reader))
            print('Total rows to be read: ' + str(total_rows))

        with open(filepath, 'r', newline='') as csvfile:
            rows_reader = csv.reader(csvfile, delimiter=',')

            for M_row, counter in zip(rows_reader, range(total_rows)):
                printer.printProgressBar(counter, total_rows, prefix='Progress (M):', suffix='', length=20)

                M_rows.append(list(map(float, M_row)))

        M = np.array(M_rows)

    assert M.shape[0] > 0, 'Matrix M was not succesfully read using ' + format_label + ' format'

    print('Matrix M ' + str(M.shape) + ' read in ' + str(
        round(time.time() - t0, 1)) + '[s] with ' + format_label + ' format')
    
    return M


def read_internal_Q(network: TNetwork, 
                    sparse_format: bool = False) -> Matrix:
    format_label = 'sparse' if sparse_format else 'dense'

    # print('Reading Q in ' + format_label + ' format')

    t0 = time.time()

    Q_rows = []

    if sparse_format:
        filepath = config.dirs['read_network_data'] + 'Q/Q-sparse-' + network.key + '.npz'
        Q = np.array(scipy.sparse.load_npz(filepath).todense())

    else:
        filepath = config.dirs['read_network_data'] + 'Q/Q-' + network.key + '.csv'
        with open(filepath, 'r', newline='') as csvfile:
            rows_reader = csv.reader(csvfile, delimiter=' ')
            total_rows = len(list(rows_reader))
            print('Total rows to be read: ' + str(total_rows))

        with open(filepath, 'r', newline='') as csvfile:
            rows_reader = csv.reader(csvfile, delimiter=',')

            for Q_row, counter in zip(rows_reader, range(rows_reader)):
                printer.printProgressBar(counter, total_rows, prefix='Progress (Q):', suffix='', length=20)

                Q_rows.append(list(map(float, Q_row)))

        Q = np.array(Q_rows)

    assert Q.shape[0] > 0, 'Matrix Q was not succesfully read using ' + format_label + ' format'
    print('Matrix Q ' + str(Q.shape) + ' read in ' + str(
        round(time.time() - t0, 1)) + '[s] with ' + format_label + ' format')
    
    return Q


def read_internal_network_files(network: TNetwork,
                                options,
                                **kwargs) -> None:
    """ Wrapper function for reading"""

    # TODO: Reading sparse matrix si quite slow. An upgrade writing matrix in sparse format is needed
    # TODO: Break this method in parts as it is done for the writer
    # print(options['reading'])



    if options['reading']['C'] or options['reading']['sparse_C']:

        sparse_format = False

        if options['reading']['sparse_C']:
            sparse_format = True

        network.C = read_internal_C(network=network, sparse_format=sparse_format)

    if options['reading']['D'] or options['reading']['sparse_D']:

        sparse_format = False

        if options['reading']['sparse_D']:
            sparse_format = True

        network.D = read_internal_D(network=network, sparse_format=sparse_format)

    if options['reading']['M'] or options['reading']['sparse_M']:

        sparse_format = False

        if options['reading']['sparse_M']:
            sparse_format = True

        network.M = read_internal_M(network=network, sparse_format=sparse_format)

    if options['reading']['Q'] or options['reading']['sparse_Q']:

        sparse_format = False

        if options['reading']['sparse_Q']:
            sparse_format = True

        Q = read_internal_Q(network=network, sparse_format=sparse_format)
        network.load_OD(Q = Q)

def get_files_tntp_repo(network_name):

    supported_networks = ['Anaheim', 'Austin', 'Barcelona', 'Berlin-Center', 'Berlin-Friedrichshain',
                          'Berlin-Mitte-Center', 'Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center',
                          'Berlin-Prenzlauerberg-Center', 'Berlin-Tiergarten', 'Birmingham-England', 'Braess-Example',
                          'chicago-regional', 'Chicago-Sketch', 'Eastern-Massachusetts', 'GoldCoast, Australia',
                          'Hessen-Asymmetric', 'Philadelphia', 'SiouxFalls', 'Sydney, Australia', 'Terrassa-Asymmetric',
                          'Winnipeg', 'Winnipeg-Asymmetric']

    assert network_name in supported_networks, 'network is not supported'

    # https://stackoverflow.com/questions/60924860/python-get-list-of-csv-files-in-public-github-repository

    url = "https://github.com/bstabler/TransportationNetworks/tree/master/" + network_name
    r = requests.get(url)

    # Extract text: html_doc => str
    html_doc = r.text

    # Parse the HTML: soup => bs4.BeautifulSoup
    soup = BeautifulSoup(html_doc, "html.parser")

    # Find all 'a' tags (which define hyperlinks): a_tags => bs4.element.ResultSet
    a_tags = soup.find_all('a')

    # Store a list of urls ending in .csv: urls => list
    urls = ['https://raw.githubusercontent.com' + re.sub('/blob', '', link.get('href'))
            for link in a_tags if '.tntp' in link.get('href')]

    # Store a list of Data Frame names to be assigned to the list: df_list_names => list
    # files_list = [url.split('/')[url.count('/')] for url in urls]

    # # Store an empty list of dataframes: df_list => list
    # df_list = [pd.read_csv(url, sep=',') for url in urls]
    #
    # # Name the dataframes in the list, coerce to a dictionary: df_dict => dict
    # df_dict = dict(zip(df_list_names, df_list))

    return urls


def read_tntp_linkdata(network_name: str,
                       folderpath: str = None,
                       local_files = False):

    if local_files is False:

        urls = get_files_tntp_repo(network_name)

        links_filename = [_ for _ in  urls if '_net' in _][0]


    if local_files is True:
        links_filename = folderpath + network_name + '/' + network_name + '_net.tntp'


    links_df = pd.read_csv(links_filename, skiprows=8, sep='\t')

    trimmed = [s.strip().lower() for s in links_df.columns]
    links_df.columns = trimmed

    # And drop the silly first andlast columns
    links_df.drop(['~', ';'], axis=1, inplace=True)

    # Reduce the number of the node by 1 for internal consistency
    links_df['init_node'] -= 1
    links_df['term_node'] -= 1

    return links_df

def read_tntp_network(folderpath: str, subfoldername: str) -> (Matrix, pd.Dataframe):
    """

    This method return an adjacency matrix, od matrix and link level information based on the ".omx" and "_net.tntp" files
    available for each network in  https://github.com/bstabler/TransportationNetworks. It is an adaptation from the script available
    at https://github.com/bstabler/TransportationNetworks/tree/master/_scripts

    """

    # root = os.path.dirname(os.path.abspath('.'))

    # We list all folders available, most of which are TNTP instances
    folders = [x for x in os.listdir(folderpath)[1:] if os.path.isdir(os.path.join(folderpath, x))]

    #
    # selected_folders = ['Berlin-Mitte-Center']
    #
    # # If we want to import all matrices in place
    # for f in selected_folders:
    #     mod = os.path.join(root, f)
    #     mod_files = os.listdir(mod)
    #
    #     for i in mod_files:
    #         print(f.upper())
    #         if 'TRIPS' in i.upper() and i.lower()[-5:]=='.tntp':
    #             print('trips')
    #             source_file = os.path.join(mod, i)
    #             import_matrix(source_file)

    # Importing the networks into a Pandas dataframe consists of a single line of code
    # but we can also make sure all headers are lower case and without trailing spaces

    # net_files = {}
    # # netfile = os.path.join(root, 'SiouxFalls','SiouxFalls_net.tntp')
    # net_files['Anaheim'] = os.path.join(root, 'Anaheim', 'Anaheim_net.tntp')
    # net_files['Berlin-Center'] = os.path.join(root, 'Berlin-Center', 'berlin-center_net.tntp')
    # net_files['SiouxFalls'] = os.path.join(root, 'SiouxFalls', 'SiouxFalls_net.tntp')
    # net_files['Austin'] = os.path.join(root, 'Austin', 'Austin_net.tntp')
    # netfile = os.path.join(root, selected_folders[0],'berlin-mitte-center_net.tntp')

    # od_files = {}
    # od_files['SiouxFalls'] = os.path.join(root, 'SiouxFalls', 'SiouxFalls_trips.tntp')

    od_filename = [_ for _ in os.listdir(os.path.join(folderpath, subfoldername)) if 'trips' in _ and _.endswith('tntp')]

    prefix_filenames = od_filename[0].partition('_')[0]

    # nets = {i: pd.read_csv(j, skiprows=8, sep='\t') for i, j in net_files.items()}

    # links_attrs = links_attrss['Berlin-Center']
    links_df = pd.read_csv(folderpath + subfoldername + '/' + prefix_filenames + '_net.tntp', skiprows=8, sep='\t')  # links_dfs['SiouxFalls']
    # links_df = links_dfs['Austin']

    trimmed = [s.strip().lower() for s in links_df.columns]
    links_df.columns = trimmed

    # And drop the silly first andlast columns
    links_df.drop(['~', ';'], axis=1, inplace=True)

    import matplotlib.pyplot as plt

    # Reduce the number of the node by 1 for internal consistency
    links_df['init_node'] -= 1
    links_df['term_node'] -= 1

    # Create link id column
    # links_df['link_key']

    # Create adjacency matrix
    # dimension_A = len(links_df['init_node'].append(links_df['term_node']).unique())
    dimension_A = links_df['init_node'].append(links_df['term_node']).unique().max() + 1

    A = np.zeros([dimension_A, dimension_A])

    for index, row in links_df.iterrows():
        A[(int(row['init_node']), int(row['term_node']))] = 1

    return A, links_df


def read_colombus_network(folderpath: str) -> (Matrix, pd.DataFrame, pd.DataFrame):

    #TODO: read new files upload by Bin

    # folder = isl.Config().paths['Colombus_network']

    # # ===================================================================
    # # SHAPE FILE COLOMBUS NETWORK
    # # ===================================================================
    #
    # columbus_gdf = gpd.read_file(folder + '/network/MORPC2018_Link_v1.shp')
    #
    # # columbus_gdf.plot()
    # # plt.show()
    #
    # columbus_gdf.keys()

    # ===================================================================
    # (SNAP) Graph
    # ===================================================================

    # pd.read_csv(folder + '/graph/Snap_graph', delimiter=)

    edge_list = []

    # Each edge have an id
    edge_dict = {}

    with open(folderpath + '/graph/Snap_graph', 'r', newline='') as csvfile:
        graph_reader = csv.reader(csvfile, delimiter=' ')

        counter = 0
        for row in graph_reader:
            if counter > 0:
                edge_dict[row[0]] = (row[1], row[2])
            counter += 1

    colombus_graph = nx.from_edgelist(list(edge_dict.values()))

    # nx.read_edgelist(edge_list)

    # nx.plot(edge_list)

    # Draw takes a lot of time because the large number of edges and nodes
    # nx.draw_networkx(colombus_graph)

    # Number of nodes and edges in the network
    # print('nodes : ' + str(len(list(colombus_graph.nodes()))))
    # print('edges : ' + str(len(list(colombus_graph.edges()))))

    # # Pick two random nodes
    # print(*random.choices(list(colombus_graph.nodes()), k=2))

    # # Generate of the shortest path between a pair of random points
    # print(nx.shortest_path(colombus_graph, *random.choices(list(colombus_graph.nodes()), k=2)))

    # ===================================================================
    # Traffic counts (every 15 min intervals)
    # ===================================================================

    counts_edges_by_time = {}

    count_files = ['count_5-12AM_7hours_car.csv', 'count_1-7PM_6hours_car.csv', 'count_5-12AM_7hours_truck.csv',
                   'count_1-7PM_6hours_truck.csv']

    with open(folderpath + '/counts/' + count_files[1], 'r', newline='') as csvfile:
        counts_reader = csv.reader(csvfile, delimiter=',')

        counter = 0
        time_intervals = []
        for row in counts_reader:
            if counter == 0:
                time_intervals = list(map(lambda x: x.replace(':', '.'), row[1:]))
                # edge_counts_by_time = dict(zip(row[1:], [[]] * len(row)))

                # This methods make a reference copy and it messes up things
                # counts_edges_by_time = dict(zip(time_intervals, [{}] * len(time_intervals)))

                for time_interval in time_intervals:
                    counts_edges_by_time[time_interval] = {}

            # counter += 1
            if counter > 0:
                # for key, i in zip(keys, range(keys)):
                # Replace empty values with nan
                counts_edge_row = [np.nan if i == "" else i for i in row[1:]]
                for j in range(len(time_intervals)):
                    counts_edges_by_time[time_intervals[j]][row[0]] = float(counts_edge_row[j])

            counter += 1

    # Convert to pandas dataframe

    # time_intervals = list(counts_edges_by_time.keys())
    # links_ids = list(counts_edges_by_time[random.choice(time_intervals)].keys())

    counts_edges_df = pd.DataFrame()

    for time_interval in time_intervals:
        counts_edges_time_df = pd.DataFrame(
            {'link_id': list(counts_edges_by_time[time_interval].keys()), 'time_interval': time_interval,
             'count': list(counts_edges_by_time[time_interval].values())})
        counts_edges_df = counts_edges_df.append(counts_edges_time_df)

    # ===================================================================
    # CONVERT NODES ORIGINAL IDS INTO SEQUENTIAL INTERNAL IDS ('KEYS') FROM 1 to N
    # ===================================================================
    # ODE_outputs/MNM_input_node
    filepath = folderpath + '/ODE_outputs/MNM_input_node'

    nodes_df = pd.read_csv(filepath, skiprows=1, delimiter=" ",
                           names=['id', 'type', 'conversion_factor']
                           , dtype={'id': 'int', 'type': 'string', 'conversion_factor': 'float'}
                           )
    # Create label for internal use
    nodes_df['key'] = range(0, len(nodes_df))
    nodes_df['key'] = nodes_df['key'].astype('int')

    # ===================================================================
    # LINK CHARACTERISTICS
    # ===================================================================

    # ODE_outputs/MNM_input_link

    # Given the file structure, it seems convenient to store link information in pandas df

    # Original header
    # ID Type LEN(mile) FFS_car(mile/h) Cap_car(v/hour) RHOJ_car(v/miles) Lane FFS_truck(mile/h) Cap_truck(v/hour) RHOJ_truck(v/miles) Convert_factor(1)
    header_links_df = ['id', 'type', 'length', 'ff_speed_car', 'capacity_car', 'rhoj_car', 'lane', 'ff_speed_truck', 'capacity_truck', 'rhoj_truck', 'conversion_factor']

    # https://medium.com/analytics-vidhya/make-the-most-out-of-your-pandas-read-csv-1531c71893b5
    links_df = pd.read_csv(folderpath + '/ODE_outputs/MNM_input_link', delimiter=" ", names=header_links_df, skiprows=1, dtype ={'id': str, 'type': str})

    # Add a column with the origin and destination node of each link
    edges_df = pd.DataFrame({'id': edge_dict.keys(), 'od_pair': edge_dict.values()})
    edges_df['init_node_id'] = edges_df['od_pair'].apply(lambda x: int(x[0]))
    edges_df['term_node_id'] = edges_df['od_pair'].apply(lambda x: int(x[1]))

    links_df = pd.merge(links_df, edges_df, how='left')

    # Remap original ids of init and end nodes in link file into sequential ids
    links_df = pd.merge(links_df, nodes_df[['id', 'key']], left_on='init_node_id', right_on='id').drop('id_y', axis=1)\
        .rename(columns={'key': 'init_node_key', 'id_x': 'id'})

    links_df = pd.merge(links_df, nodes_df[['id', 'key']], left_on='term_node_id', right_on='id').drop('id_y', axis=1)\
        .rename(columns={'key': 'term_node_key', 'id_x': 'id'})

    # ===================================================================
    # ADJACENCY MATRIX
    # ===================================================================

    # dimension_A = len(links_df.init_node_id.append(links_df.term_node_id).unique())
    dimension_A = len(nodes_df)

    A = np.zeros([dimension_A, dimension_A])

    for index, row in links_df.iterrows():
        A[(int(row['init_node_key']), int(row['term_node_key']))] = 1
    # ===================================================================
    # CONVERT NODES ORIGINAL IDS INTO SEQUENTIAL IDS FROM 1 to N
    # ===================================================================
    # ODE_outputs/MNM_input_node

    return A, links_df, nodes_df


def read_fresno_network(folderpath: str) -> (Matrix, pd.DataFrame, pd.DataFrame):
    # ===================================================================
    # NETWORK SUMMARY (SR41.net)
    # ===================================================================
    # import isuelogit as isl
    # folder = isl.Config().paths['Fresno_network']
    # filepath = folder + '/SR41.net'
    #
    # n_nodes, n_links, n_origins, n_destination, n_ods = 0, 0, 0, 0, 0
    #
    # with open(filepath, 'r', newline='') as csvfile:
    #
    #     net_reader = csv.reader(csvfile, delimiter=' ')
    #
    #     n_row = 0
    #
    #     for row in net_reader:
    #
    #         if n_row == 0:
    #             n_nodes = row[-1]
    #             print('Nodes: ' + str(n_nodes))
    #
    #         if n_row == 1:
    #             n_links = row[-1]
    #             print('Links: ' + str(n_links))
    #
    #         if n_row == 2:
    #             n_origins = row[-1]
    #             print('Origins: ' + str(n_origins))
    #
    #         if n_row == 3:
    #             n_destinations = row[-1]
    #             print('Destinations: ' + str(n_destinations))
    #
    #         if n_row == 4:
    #             n_ods = row[-1]
    #             print('OD Pairs: ' + str(n_ods))
    #
    #         n_row += 1

    # ===================================================================
    # NODES (SR41.nod)
    # ===================================================================

    filepath = folderpath + '/SR41.nod'

    nodes_df = pd.read_csv(filepath, skiprows=1, delimiter="\t"
                           , names=['id', 'type', 'x', 'y']
                           , dtype={'id': 'int', 'type': 'string', 'x': 'int', 'y': 'int'}
                           )

    # Sequential key
    nodes_df['key'] = range(0, len(nodes_df))
    nodes_df['key'] = nodes_df['key'].astype('int')

    # ===================================================================
    # LINKS (sre.lin)
    # ===================================================================

    # ODE_outputs/MNM_input_link

    # Given the file structure, it seems convenient to store link information in pandas df

    # Original header
    # ID Type Name From To LEN(mi) FFS(mi/h)	Cap(v/h) RHOJ(v/mi)	Lane
    header_links_df = ['id', 'link_type', 'name', 'init_node_id', 'term_node_id', 'length', 'ff_speed', 'capacity', 'rhoj','lane']

    # links_df['init_node']

    links_df = pd.read_csv(folderpath + '/SR41.lin', delimiter="\t"
                           , names=header_links_df, skiprows=1
                           , dtype={'id': 'int', 'link_type': 'string'
            , 'init_node_id': 'int', 'term_node_id': 'int'}
                           )

    # Remap original ids of init and end nodes in link file into internal key (which is sequential)

    links_df = pd.merge(links_df, nodes_df[['id', 'key']], left_on='init_node_id', right_on='id').drop('id_y', axis=1)\
        .rename(columns={'key': 'init_node_key', 'id_x': 'id'})
    links_df = pd.merge(links_df, nodes_df[['id', 'key']], left_on='term_node_id', right_on='id').drop('id_y',axis=1)\
        .rename(columns={'key': 'term_node_key', 'id_x': 'id'})

    # # Create link internal key (assuming that it is a Dinetwork
    # links_df['key'] =  links_df.apply(lambda row: (row['init_node_key'],row['term_node_key']))
    # # links_df['key'] = '(' + links_df['init_node_key'].astype(str) + ',' + links_df['term_node_key'].astype(str) + ',0)'

    # ===================================================================
    # ADJACENCY MATRIX (more reliable than networkX supporting methods)
    # ===================================================================

    # dimension_A = len(links_df.init_node_key.append(links_df.term_node_key).unique())
    dimension_A = len(nodes_df)

    #
    # dimension_A = max()

    # A = np.zeros([dimension_A, dimension_A])
    #
    # for index, row in links_df.iterrows():
    #     A[(int(row['init_node_key']), int(row['term_node_key']))] = 1

    return links_df, nodes_df


def read_sacramento_network(folderpath: str) -> (Matrix, pd.DataFrame, pd.DataFrame):
    # TODO: Fix issue with the dimensions of the adjacency matrix

    # ===================================================================
    # NETWORK SUMMARY (pfe.net)
    # ===================================================================
    # import isuelogit as isl
    # folder = isl.Config().paths['Sacramento_network']
    # filepath = folder + '/pfe.net'
    #
    # n_nodes, n_links, n_origins, n_destination, n_ods = 0, 0, 0, 0, 0
    #
    # with open(filepath, 'r', newline='') as csvfile:
    #
    #     net_reader = csv.reader(csvfile, delimiter=' ')
    #
    #     n_row = 0
    #
    #     for row in net_reader:
    #
    #         if n_row == 0:
    #             n_nodes = row[-1]
    #             print('Nodes: ' + str(n_nodes))
    #
    #         if n_row == 1:
    #             n_links = row[-1]
    #             print('Links: ' + str(n_links))
    #
    #         if n_row == 2:
    #             n_origins = row[-1]
    #             print('Origins: ' + str(n_origins))
    #
    #         if n_row == 3:
    #             n_destinations = row[-1]
    #             print('Destinations: ' + str(n_destinations))
    #
    #         if n_row == 4:
    #             n_ods = row[-1]
    #             print('OD Pairs: ' + str(n_ods))
    #
    #         n_row += 1

    # ===================================================================
    # NODES (pfe.nod)
    # ===================================================================
    # ODE_outputs/MNM_input_node

    filepath = folderpath + '/pfe.nod'

    nodes_df = pd.read_csv(filepath, skiprows=1, delimiter="\t"
                           , names=['id', 'node_type', 'x', 'y']
                           , dtype={'id': 'int', 'node_type': 'string', 'x': 'int', 'y': 'int'}
                           )

    # Sequential key
    nodes_df['key'] = range(0, len(nodes_df))
    nodes_df['key'] = nodes_df['key'].astype('int')

    # ===================================================================
    # LINKS (pfe.lin)
    # ===================================================================

    filepath = folderpath + '/pfe.lin'

    # Given the file structure, it seems convenient to store link information in pandas df

    # Original header
    # ID Type Name From To LEN(mi) FFS(mi/h)	Cap(v/h) RHOJ(v/mi)	Lane
    header_links_df = ['id', 'link_type', 'name', 'init_node_id', 'term_node_id',
                       'length', 'ff_speed', 'capacity', 'rhoj', 'lane']

    # links_df['init_node']

    links_df = pd.read_csv(filepath, delimiter="\t"
                           , names=header_links_df, skiprows=1
                           , dtype={'id': 'int', 'link_type': 'string'
            , 'init_node_id': 'int', 'term_node_id': 'int'}
                           )

    # Remap original ids of init and end nodes in link file into internal key (which is sequential)

    links_df = pd.merge(links_df, nodes_df[['id', 'key']], left_on='init_node_id',
                        right_on='id').drop('id_y', axis=1).rename(
        columns={'key': 'init_node_key', 'id_x': 'id'})
    links_df = pd.merge(links_df, nodes_df[['id', 'key']], left_on='term_node_id',
                        right_on='id').drop('id_y', axis=1).rename(
        columns={'key': 'term_node_key', 'id_x': 'id'})

    # # Create link internal key (assuming that it is a Dinetwork
    # links_df['key'] =  links_df.apply(lambda row: (row['init_node_key'],row['term_node_key']))
    # # links_df['key'] = '(' + links_df['init_node_key'].astype(str) + ',' + links_df['term_node_key'].astype(str) + ',0)'

    # ===================================================================
    # ADJACENCY MATRIX (more reliable than networkX supporting methods)
    # ===================================================================

    # dimension_A = len(links_df.init_node_key.append(links_df.term_node_key).unique())
    dimension_A = len(nodes_df)

    # links_df.init_node_key.max()
    # links_df.term_node_key.max()

    # dimension_A = max()

    A = np.zeros([dimension_A, dimension_A])
    # np.sum(A)

    for index, row in links_df.iterrows():
        A[(int(row['init_node_key']), int(row['term_node_key']))] = 1

    return A, links_df, nodes_df


def read_tntp_od(network_name: str,
                 folderpath: str = None,
                 local_files = False) -> Matrix:
    """

    This method return an adjacency matrix, od matrix and link level information based on the ".omx" and "_net.tntp" files
    available for each network in  https://github.com/bstabler/TransportationNetworks. It is an adaptation from the script available
    at https://github.com/bstabler/TransportationNetworks/tree/master/_scripts

    """

    print('Reading Q from external file')

    t0 = time.time()

    # Function to import OMX matrices
    def import_matrix(filepath, network_name):
        if local_files:
            f = open(filepath, 'r')
            all_rows = f.read()
        else:
            f = urllib.request.urlopen(filepath)
            all_rows = f.read().decode(f.headers.get_content_charset())

        blocks = all_rows.split('Origin')[1:]
        matrix = {}
        for k in range(len(blocks)):
            orig = blocks[k].split('\n')
            dests = orig[1:]
            orig = int(orig[0])

            d = [eval('{' + a.replace(';', ',').replace(' ', '') + '}') for a in dests]
            destinations = {}
            for i in d:
                destinations = {**destinations, **i}
            matrix[orig] = destinations
        zones = max(matrix.keys())
        mat = np.zeros((zones, zones))
        for i in range(zones):
            for j in range(zones):
                # We map values to a index i-1, as Numpy is base 0
                mat[i, j] = matrix.get(i + 1, {}).get(j + 1,
                                                      0)  # This fails for Braess example using raw data. I modified it

        index = np.arange(zones) + 1

        write_folderpath = config.dirs['output_folder'] + 'network-data/Q/'

        if not os.path.exists(write_folderpath):
            os.makedirs(write_folderpath)

        write_filepath = write_folderpath + network_name + '_demand' + '.omx'

        myfile = omx.open_file(write_filepath, 'w')
        myfile['matrix'] = mat
        myfile.create_mapping('taz', index)

        numpy_matrix = np.array(myfile['matrix'])

        myfile.close()

        os.remove(write_filepath)

        return numpy_matrix

    if local_files is False:
        urls = get_files_tntp_repo(network_name)
        filepath = [_ for _ in urls if '_trips' in _][0]

    else:
        file_list = os.listdir(os.path.join(folderpath, network_name))
        filepath = folderpath + network_name + '/' + [_ for _ in file_list if 'trips' in _ and _.endswith('tntp')][0]

    # prefix_filenames = od_filename[0].partition('_')[0]

    Q = import_matrix(filepath=filepath, network_name=network_name)

    assert Q.shape[0] > 0, 'Matrix Q was not succesfully read'
    print('Matrix Q ' + str(Q.shape) + ' read in ' + str(round(time.time() - t0, 1)) + '[s]')

    return Q

def read_csv_github(url, username, token, **kwargs):
    '''
    https://medium.com/towards-entrepreneurship/importing-a-csv-file-from-github-in-a-jupyter-notebook-e2c28e7e74a5

    Args:
        url:  # Make sure the url is the raw version of the file on GitHub
        token: Personal Access Token (PAO) from your GitHub account
        username: Username of your GitHub account

    Returns:

    '''
    # Creates a re-usable session object with your creds in-built

    github_session = requests.Session()
    github_session.auth = (username, token)

    # Downloading the csv file from your GitHub
    download = github_session.get(url).content

    # Reading the downloaded content and making it a pandas dataframe

    df = pd.read_csv(io.StringIO(download.decode('utf-8')), **kwargs)

    # Printing out the first 5 rows of the dataframe to make sure everything is good

    # print(df.head())

    return df


def read_colombus_od(folderpath: str, A: Matrix, nodes: Nodes) -> Matrix:
    print('Reading Q from external file')

    t0 = time.time()

    # ===================================================================
    # Trips
    # ===================================================================

    # ODE_outputs/MNM_input_demand

    # There are od trips for cars and truck
    trips_mode_link = {}

    with open(folderpath + '/trips/df_trip_final.csv', 'r', newline='') as csvfile:
        od_reader = csv.reader(csvfile, delimiter=',')

        counter = 0

        for row in od_reader:
            if counter == 0:
                pass
            if counter > 0:
                # for key, i in zip(keys, range(keys)):
                trips_mode_link[(str(row[0]), str(row[1]))] = {'cars': float(row[2]), 'trucks': float(row[3]),
                                                               'total': float(row[2]) + float(row[3])}
                # edge_counts_by_time[key].append(row[i])
                # edge_counts_by_time.append([row[1], row[2]])
            counter += 1

    # ===================================================================
    # OD demand
    # ===================================================================

    # ODE_outputs/MNM_input_demand

    # Mapping between original id and internal keys
    nodes_dict = {node.id: node.key for node in nodes}

    nodes_df = pd.DataFrame({'id': list(nodes_dict.keys()), 'key': list(nodes_dict.values())}
                            )

    # Each line describes the od pair and then 28 time intervals for cars first and then trucks
    trips_mode_time_od = {}
    trips_od = {}

    with open(folderpath + '/ODE_outputs/MNM_input_demand', 'r', newline='') as csvfile:
        od_reader = csv.reader(csvfile, delimiter=' ')

        counter = 0
        time_intervals = []
        for row in od_reader:
            if counter == 0:
                pass
            if counter > 0:
                # for key, i in zip(keys, range(keys)):
                trips_row = row[2:]
                trips_cars_row = np.array(list(map(float, trips_row[2:(2 + int(len(trips_row) / 2))])))
                trips_trucks_row = np.array(list(map(float, trips_row[int(len(trips_row) / 2):])))
                trips_mode_time_od[(str(row[0]), str(row[1]))] = {'cars': trips_cars_row, 'trucks': trips_trucks_row}

                # TODO: confirm if the od trips are already in equivalent cars units
                trips_od[(str(row[0]), str(row[1]))] = np.sum(trips_cars_row + trips_trucks_row)

            counter += 1

    # TODO: the total number of OD pair is not a perfect squared. Probably nodes with 0 trips are not written.
    #  To save computational resources, so I should use a dense vector representation

    # - Pandas dataframe

    trips_od_df = pd.DataFrame({'od_pair': pd.Series(trips_od.keys()), 'trips': trips_od.values()})

    trips_od_df['origin_node'] = trips_od_df['od_pair'].apply(lambda x: x[0])
    trips_od_df['destination_node'] = trips_od_df['od_pair'].apply(lambda x: x[1])
    # 
    # trips_od_df.head()

    # Remap original ids of origin and end nodes in OD dataframe into internal keys
    trips_od_df = pd.merge(trips_od_df, nodes_df[['id', 'key']], left_on='origin_node',
                           right_on='id').drop(
        'id', axis=1).rename(columns={'key': 'origin_node_key'})
    trips_od_df = pd.merge(trips_od_df, nodes_df[['id', 'key']], left_on='destination_node',
                           right_on='id').drop(
        'id', axis=1).rename(columns={'key': 'destination_node_key'})

    # ===================================================================
    # OD MATRIX
    # ===================================================================

    # OD matrix has the same dimensions than the adjacency matrix. Trips information is obtained from trips_od_df

    Q = np.zeros(A.shape)

    for index, row in trips_od_df.iterrows():
        Q[(int(row['origin_node_key']), int(row['destination_node_key']))] = row['trips']

    print('Matrix Q ' + str(Q.shape) + ' read in ' + str(round(time.time() - t0, 1)) + '[s]')

    return Q


def read_fresno_od(folderpath, A, nodes) -> Matrix:
    """

    :return:
    Matrix with OD demand

    """

    t0 = time.time()

    # ===================================================================
    # Reading OD demand
    # ===================================================================

    # The OD can be read from sr41.sum or sr41.odp. The summary file it is easy to parse and the OD information start from line 79 (78 is header). I transformed it with sublime text to a csv file as the line information was irregular.

    # original header:  Org  Dest   Demands   SimVehs   DIST(m)   TT(min)   TD(min)  SPD(mph)  ETD(min)

    # od_df = ['origin', 'destination', 'demand', 'sim_vehs', 'dist', 'tt', 'td', 'speed', 'etd']

    # The SR41.csv was created by my own to extract only the data that summarizes the total OD trips between nodes
    filepath = folderpath + '/SR41.csv'
    # 
    # od_df = pd.read_csv(filepath, delimiter="\t"
    #                        , names = od_df, skiprows=79
    #                        , dtype={'origin': 'float', 'destination': 'float'
    #                        , 'demand': 'float'}
    #                        )

    od_dict = {}

    # Q_rows = len(set([int(node.id) for node in nodes]))

    # Q = np.zeros((Q_rows,Q_rows))

    with open(filepath, 'r', newline='') as csvfile:

        net_reader = csv.reader(csvfile, delimiter=',')

        n_row = 0

        for row in net_reader:

            if n_row >= 1:
                origin = int(row[0])
                destination = int(row[1])
                demand = float(row[2])

                od_dict[(origin, destination)] = demand

            n_row += 1

    # Convert dictionary to pandas dataframe

    od_df = pd.DataFrame({'od_pair': od_dict.keys(), 'demand': od_dict.values()})
    od_df['origin_node'] = od_df['od_pair'].apply(lambda x: x[0]).astype(str)
    od_df['destination_node'] = od_df['od_pair'].apply(lambda x: x[1]).astype(str)

    # Mapping between original id and internal keys
    nodes_dict = {node.id: node.key for node in nodes}

    nodes_df = pd.DataFrame({'id': list(nodes_dict.keys()), 'key': list(nodes_dict.values())}
                            )

    # Remap original ids of origin and end nodes in OD dataframe into internal keys
    od_df = pd.merge(od_df, nodes_df[['id', 'key']], left_on='origin_node', right_on='id').drop('id', axis=1).rename(
        columns={'key': 'origin_key'})
    od_df = pd.merge(od_df, nodes_df[['id', 'key']], left_on='destination_node', right_on='id').drop(
        'id', axis=1).rename(columns={'key': 'destination_key'})

    # ===================================================================
    # OD MATRIX
    # ===================================================================

    # OD matrix has the same dimensions than the adjacency matrix. Trips information is obtained from trips_od_df

    Q = np.zeros(A.shape)

    for index, row in od_df.iterrows():
        Q[(int(row['origin_key']), int(row['destination_key']))] = row['demand']

    print('Matrix Q ' + str(Q.shape) + ' read in ' + str(round(time.time() - t0, 1)) + '[s]')

    return Q


def read_fresno_dynamic_od(filepath, network, periods: []) -> Matrix:
    """

    :argument periods: list with periods (integers 1..6) that will be aggregated to get the OD

    :return:
    Matrix with OD demand

    """

    t0 = time.time()

    nodes = network.nodes
    A = network.A

    # ===================================================================
    # Reading OD demand from SR41.dmd
    # ===================================================================

    # This file contains the number of trips in a 15 minute (900[s]/6[periods]) resolution between every node. According to Sean the starting hour is 4pm

    # filepath = folderpath + '/SR41.csv'

    # filepath = folderpath + '/SR41/SR41.dmd'
    # filepath = folderpath + '/SR41.dmd'

    assert not any([period not in [1,2,3,4,5,6] for period in periods]), 'invalid period for the OD matrix'


    # The format of the file is strightforward

    od_dict = {}

    with open(filepath, 'r', newline='') as csvfile:

        net_reader = csv.reader(csvfile)

        n_row = 0
        origin = 0
        destination = 0

        for row in net_reader:

            if 'Origin' in row[0]:

                origin = int(row[0].split(':')[1].strip())

            elif 'Dest' in row[0]:
                destination = int(row[0].split(':')[1].strip())
                # print(destination)

            elif row[0] == '':
                # Here the file ends with a blank line
                break

            else:
                demand_periods = [elem.strip() for elem in row[0].split(' ') if elem != '']
                # for k in range(0,len(demand_periods)):
                #     demand_period = demand_periods[k]
                #     od_dict[origin][destination][k + 1] = float(demand_period)
                sum_demand_periods = 0
                for k in periods:
                    # print(float(demand_periods[k]))
                    sum_demand_periods += float(demand_periods[k-1])

                od_dict[(origin,destination)] = float(sum_demand_periods)


    # Convert dictionary to pandas dataframe

    od_df = pd.DataFrame({'od_pair': od_dict.keys(), 'demand': od_dict.values()})
    od_df['origin_node'] = od_df['od_pair'].apply(lambda x: x[0]).astype(str)
    od_df['destination_node'] = od_df['od_pair'].apply(lambda x: x[1]).astype(str)

    # Mapping between original id and internal keys
    nodes_dict = {node.id: node.key for node in nodes}

    nodes_df = pd.DataFrame({'id': list(nodes_dict.keys()), 'key': list(nodes_dict.values())})

    # Remap original ids of origin and end nodes in OD dataframe into internal keys
    od_df = pd.merge(od_df, nodes_df[['id', 'key']], left_on='origin_node', right_on='id').drop('id', axis=1).rename(
        columns={'key': 'origin_key'})
    od_df = pd.merge(od_df, nodes_df[['id', 'key']], left_on='destination_node', right_on='id').drop(
        'id', axis=1).rename(columns={'key': 'destination_key'})

    # ===================================================================
    # OD MATRIX
    # ===================================================================

    # OD matrix has the same dimensions than the adjacency matrix. Trips information is obtained from trips_od_df

    Q = np.zeros(network.A.shape)

    for index, row in od_df.iterrows():
        Q[(int(row['origin_key']), int(row['destination_key']))] = row['demand']

    print('Matrix Q ' + str(Q.shape) + ' read in ' + str(round(time.time() - t0, 1)) + '[s]')

    network.load_OD(Q = Q)

    return Q


def read_sacramento_od(folder, A, nodes) -> Matrix:
    """

       :return:
       Matrix with OD demand

       """

    t0 = time.time()
    # folder = isl.Config().paths['Fresno_Sac_networks']

    # ===================================================================
    # Reading OD demand
    # ===================================================================

    # The OD can be read from sr41.sum or sr41.odp. The summary file it is easy to parse and the OD information start from line 79 (78 is header).
    # I transformed it with sublime text to a csv file as the line information was irregular.

    # original header:  Org  Dest   Demands   SimVehs   DIST(m)   TT(min)   TD(min)  SPD(mph)  ETD(min)
    # od_df = ['origin', 'destination', 'demand', 'sim_vehs', 'dist', 'tt', 'td', 'speed', 'etd']

    filepath = folder + '/pfe.csv'
    #
    # od_df = pd.read_csv(filepath, delimiter="\t"
    #                        , names = od_df, skiprows=79
    #                        , dtype={'origin': 'float', 'destination': 'float'
    #                        , 'demand': 'float'}
    #                        )

    od_dict = {}

    # Q_rows = len(set([int(node.id) for node in nodes]))

    # Q = np.zeros((Q_rows,Q_rows))

    with open(filepath, 'r', newline='') as csvfile:

        net_reader = csv.reader(csvfile, delimiter=',')

        n_row = 0

        for row in net_reader:

            if n_row >= 1:
                origin = int(row[0])
                destination = int(row[1])
                demand = float(row[2])

                od_dict[(origin, destination)] = demand

            n_row += 1

    # Convert dictionary to pandas dataframe

    od_df = pd.DataFrame({'od_pair': od_dict.keys(), 'demand': od_dict.values()})
    od_df['origin_node'] = od_df['od_pair'].apply(lambda x: x[0]).astype(str)
    od_df['destination_node'] = od_df['od_pair'].apply(lambda x: x[1]).astype(str)

    # Mapping between original id and internal keys

    nodes_dict = {node.id: node.key for node in nodes}

    nodes_df = pd.DataFrame({'id': list(nodes_dict.keys()), 'key': list(nodes_dict.values())}
                            )

    # Remap original ids of origin and end nodes in OD dataframe into internal keys
    od_df = pd.merge(od_df, nodes_df[['id', 'key']], left_on='origin_node', right_on='id').drop('id', axis=1).rename(
        columns={'key': 'origin_key'})
    od_df = pd.merge(od_df, nodes_df[['id', 'key']], left_on='destination_node', right_on='id').drop(
        'id', axis=1).rename(columns={'key': 'destination_key'})

    # ===================================================================
    # OD MATRIX
    # ===================================================================

    # OD matrix has the same dimensions than the adjacency matrix. Trips information is obtained from trips_od_df

    Q = np.zeros(A.shape)

    for index, row in od_df.iterrows():
        Q[(int(row['origin_key']), int(row['destination_key']))] = row['demand']

    print('Matrix Q ' + str(Q.shape) + ' read in ' + str(round(time.time() - t0, 1)) + '[s]')

    return Q


def read_southern_california_network() -> None:  # -> TNetwork(nx.Graph):

    """ Network provided by MPO from 6 cities in California """

    # ===================================================================
    # DATA
    # ===================================================================

    data_folder_path = "/Users/pablo/GoogleDrive/university/cmu/2-research/datasets/private/bin-networks/SCAG-DTA"

    # Transportation analysis zone (TAZ)

    # Page 16-2 in the Report indicates the generalized function used which includes time and cost (HOT penalty). This means that we may have a way to validate our estimates. In particular, the cost conversion factor may be interpreted as a value of time.

    # ===================================================================
    # SHAPE FILE
    # ===================================================================

    southern_ca_gdf = gpd.read_file(data_folder_path + '/network/20R16BY_links.shp')

    # Subset with Los Angeles network
    la_gdf = southern_ca_gdf[southern_ca_gdf['CITY'] == 'Los Angeles']
    # la_gdf.plot()
    # plt.show()

    # Irvine
    irvine_gdf = southern_ca_gdf[southern_ca_gdf['CITY'] == 'Irvine']

    # ===================================================================
    # Graph
    # ===================================================================

    # TODO: It is not clear which are the origin and destination node and whether it is reliable to convert the shapefile into a graph

    # ===================================================================
    # Traffic counts (every 15 min intervals)
    # ===================================================================

    # ===================================================================
    # Trips
    # ===================================================================

    # ===================================================================
    # OD demand
    # ===================================================================

    # ===================================================================
    # LINK CHARACTERISTICS
    # ===================================================================

    # ODE_outputs/MNM_input_link

    # Given the file structure, it seems convenient to store link information in pandas df

    # Original header
    # ID Type LEN(mile) FFS_car(mile/h) Cap_car(v/hour) RHOJ_car(v/miles) Lane FFS_truck(mile/h) Cap_truck(v/hour) RHOJ_truck(v/miles) Convert_factor(1)
    header_links_df = ['id', 'type', 'length', 'ff_speed_car', 'capacity_car', 'rhoj_car', 'lane', 'ff_speed_truck',
                       'capacity_truck', 'rhoj_truck', 'conversion_factor']

    # https://medium.com/analytics-vidhya/make-the-most-out-of-your-pandas-read-csv-1531c71893b5
    links_df = pd.read_csv(data_folder_path + '/ODE_outputs/MNM_input_link', delimiter=" ", names=header_links_df,
                           skiprows=1
                           , dtype={'id': str, 'type': str}
                           )

    # ===================================================================
    # PREPARING DATA FOR ODLE
    # ===================================================================

    # TODO: Write file with TNTP format so I can be read easily
    # https://github.com/bstabler/TransportationNetworks/blob/master/SiouxFalls/SiouxFalls_trips.tntp

    raise NotImplementedError
