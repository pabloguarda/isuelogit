""" Path module"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import collections
import numpy as np
from links import Link, generate_links_keys, generate_links_dict
from nodes import Node
import printer
import time
import config
import igraph as ig

if TYPE_CHECKING:
    from networks import TNetwork
    from mytypes import Matrix


class Path:
    def __init__(self, origin, destination, links: [Link] = None, nodes: [Node] = None):
        # self._label = () # origin, destination, and id to distinguish between paths in the same origin pair

        # Create path object from links information
        if links is not None:
            self._links = links

            self._nodes = self.get_nodes_from_links(links)

        # Create path object from nodes information
        if nodes is not None:
            raise NotImplementedError

        self._origin = origin
        self._destination = destination
        self._key = (self._origin, self._destination)
        self._f = 0
        self._Z_dict = {}
        self._key = dict()

    @property
    def links(self):
        return self._links

    @links.setter
    def links(self, value):
        self._links = value

        # Update list of nodes
        self._nodes = self.get_nodes_from_links(self._links)

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    def get_nodes_labels(self):
        return [node.key for node in self.nodes]

    def get_nodes_from_links(self, links: [Link]):

        nodes = []

        if len(list(links)) == 1:
            nodes.append(links[0].init_node)
            nodes.append(links[0].term_node)

        if len(links) > 1:
            for link, i in zip(links,range(len(links))):
                nodes.append(link.init_node)

                if i == (len(links)-1):
                    nodes.append(link.term_node)

        return nodes

    # def get_links_from_nodes(self, nodes: [nodes]):
    #
    #     nodes = []
    #
    #     if len(list(links)) == 1:
    #         nodes.append(Node(links[0].origin))
    #         nodes.append(Node(links[0].destination))
    #
    #     if len(links) > 1:
    #         for link, i in zip(links,range(len(links))):
    #             nodes.append(Node(link.origin))
    #
    #             if i == len(links):
    #                 nodes.append(Node(link.destination))
    #
    #     return nodes


    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = value
        
    @property
    def destination(self):
        return self._destination

    @destination.setter
    def destination(self, value):
        self._destination = value

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self._f = value
    
    @property
    def traveltime(self):
        return self._traveltime

    @traveltime.setter
    def traveltime(self, value):
        self._traveltime = value

    @property
    def traveltime(self):
        return np.sum([link.traveltime for link in self.links])

    @property
    def Z_dict(self):
        
        # All attributes in Z_dict dictionaries must be numeric, otherwise the summation will raise errors

        # if not bool(self._Z_dict): #dictionary is empty
        Z_dict_path = {}

        for z_attr in self.links[0].Z_dict.keys():
            Z_dict_path[z_attr] = np.sum([link.Z_dict[z_attr] for link in self.links])


        return Z_dict_path

    @Z_dict.setter
    def Z_dict(self, value):
        self._Z_dict = value

    def utility(self, theta):

        Z_dict = self.Z_dict

        v = 0

        for key in Z_dict.keys():
            v += theta[key]*Z_dict[key]

        return v + theta['tt']*self.traveltime


def _shortest_simple_paths_dinetwork_nx(G, source, target, weight):
    ''' Modified version of nx._all_simple_paths_graph'''

    # TODO: edge costs should be utilities, travel times or just unit lengths (weight = None)

    # G = nx.DiGraph(A)


    if source not in G:
        raise nx.NodeNotFound('source node %s not in graph' % source)

    if target not in G:
        raise nx.NodeNotFound('target node %s not in graph' % target)

    if weight is None:
        length_func = len
        shortest_path_func = nx.simple_paths._bidirectional_shortest_path
    else:
        def length_func(path):
            return sum(G.adj[u][v][weight] for (u, v) in zip(path, path[1:]))

        shortest_path_func = nx.simple_paths._bidirectional_dijkstra

    listA = list()
    listB = nx.simple_paths.PathBuffer()
    prev_path = None
    while True:
        if not prev_path:
            length, path = shortest_path_func(G, source, target, weight=weight)
            listB.push(length, path)
        else:
            ignore_nodes = set()
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    length, spur = shortest_path_func(G, root[-1], target,
                                                      ignore_nodes=ignore_nodes,
                                                      ignore_edges=ignore_edges,
                                                      weight=weight)
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except nx.NetworkXNoPath:
                    pass
                ignore_nodes.add(root[-1])

        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
        else:
            break

def _all_simple_paths_dinetwork(G, source, targets, cutoff):
    ''' Modified version of nx._all_simple_paths_graph'''

    visited = collections.OrderedDict.fromkeys([source])
    # G = nx.DiGraph(A)
    stack = [iter(G[source])]
    # stack = [((v) for u, v, k in transportAI.network.Network.edges_from_source(source=source, links=N.links))]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child in targets:
                yield list(visited) + [child]
            visited[child] = None
            if targets - set(visited.keys()):  # expand stack until find all targets
                stack.append(iter(G[child]))
            else:
                visited.popitem()  # maybe other ways to child
        else:  # len(visited) == cutoff:
            for target in (targets & (set(children) | {child})) - set(visited.keys()):
                yield list(visited) + [target]
            stack.pop()
            visited.popitem()


# @staticmethod
def edges_from_source(source, links):

    edges_list = []
    for link_i in links.keys():
        if link_i[0] == source:
            edges_list.append(link_i)

    return edges_list


def _all_simple_paths_multidinetwork(A, source, targets, cutoff):
    ''' Modified version of nx._all_simple_paths_multigraph'''

    visited = collections.OrderedDict.fromkeys([source])  # list(visited)
    # N = MultiDiTNetwork(A=A)


    # Generate dictionary of links
    G = nx.MultiDiGraph()

    # Labels does not matter
    G.add_nodes_from(np.arange(0, A.shape[0]))

    for (i, j) in zip(*A.nonzero()):
        for k in range(A[(i, j)]):
            G.add_edge(i, j)

    link_keys = generate_links_keys(G)
    link_dict = generate_links_dict(link_keys)

    # link_list = [Link(key=link_key, init_node=Node(link_key[0]), term_node=Node(link_key[1])) for link_key in link_keys]

    stack = [((u, v, k) for u, v, k in edges_from_source(source=source, links=link_dict))]

    while stack:
        children = stack[-1]  # list(stack[-1])
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
            # predecessor_nodes.popitem()
            # predecessor_links.popitem()
        elif len(visited) < cutoff:
            if child[1] in visited:
                continue
            if child[1] in targets:
                # print(list(visited) + [child])
                # yield [child]
                yield {'path': list(visited.keys()) + [child[1]] , 'pred_link': list(visited.values()) + [(child[0],child[1], child[2])]}
                    #list(predecessor_links) + [child[2]]}
                #, 'pred_node': list(predecessor_nodes) + [child[0]]
            visited[child[1]] = (child[0],child[1], child[2]) # visited[child[1]] = None
            # predecessor_nodes[child[0]] = None
            # predecessor_links[child[2]] = None
            if targets - set(visited.keys()): # Target has not been reached
                # stack.append((v for u, v in G.edges(child[1])))
                stack.append(((u, v, k) for u, v, k in
                              edges_from_source(source=child[1], links=link_dict)))
            else:
                visited.popitem()
                # predecessor_nodes.popitem()
                # predecessor_links.popitem()
        else:  # len(visited) == cutoff:
            for target in targets - set(visited.keys()):
                # count = ([child[1]] + list(children)).count(target)
                count = ([child] + list(children)).count(target)
                for i in range(count):
                    yield {'path': list(visited.keys()) + [target],
                           'pred_link': list(visited.values()) + [(child[0],child[1], child[2])]}
                    # yield 'z'
                    # print(list(visited) + [target])
                    # yield list(visited) + [target]
            stack.pop()
            visited.popitem()

def k_simple_paths_nx(source, target, G, links: {}, k, cutoff = None, weight = None):
    """Return a list of path objects using adjacency matrices and links objects """

    A = nx.adjacency_matrix(G).todense()

    multinetwork = (A > 1).any()
    dinetwork = (A <= 1).any()
    paths = []

    if multinetwork:
        # paths_labels = list(_all_simple_paths_multidinetwork(A = A, source = source, targets = {target}, cutoff = cutoff))

        # https://stackoverflow.com/questions/53583341/k-shortest-paths-using-networkx-package-in-python
        paths_generator = _all_simple_paths_multidinetwork(A = A, source = source, targets = {target}, cutoff = cutoff)

        paths_labels = []
        counter = 0
        while counter < k:
            try:
                paths_labels.append(next(paths_generator))
            except StopIteration as e:
                counter = k
                pass
            else:
                counter += 1

        # paths_labels = [next(paths_generator,[]) for count in range(k)]

        # Get for links in network
        for path, i in zip(paths_labels, range(len(paths_labels))):
            links_path_list = path['pred_link'][1:]
            path_links = []
            path_origin = path['path'][0]
            path_destination = path['path'][-1]
            for link_label in links_path_list:
                path_links.append(links[link_label])

            paths.append(Path(origin = path_origin, destination = path_destination, links = path_links))

    elif dinetwork:
        # paths_labels = list(_all_simple_paths_dinetwork(A = A, source=source, targets={target}, cutoff=cutoff))
        # paths_generator = _all_simple_paths_dinetwork(A=A, source=source, targets={target}, cutoff=cutoff)

        paths_generator = _shortest_simple_paths_dinetwork_nx(G=G, source=source, target = target, weight = weight)

        # paths_labels = [next(paths_generator,[]) for count in range(k)]
        paths_labels = []
        counter = 0
        while counter < k:
            try:
                paths_labels.append(next(paths_generator))
            except:
                # print('Internal exception')
                counter = k
                pass
            else:
                counter += 1


        for path, i in zip(paths_labels, range(len(paths_labels))):
            links_path_list = [(path[i], path[i + 1], '0') for i in range(len(path) - 1)]
            path_links = []
            path_origin = path[0]
            path_destination = path[-1]

            for link_label in links_path_list:
                path_links.append(links[link_label])

            paths.append(Path(origin=path_origin, destination=path_destination, links=path_links))

    return paths

def path_generation_nx(A: Matrix, ods: [tuple], links:{tuple: Link}, cutoff: int, n_paths: int, edge_weights: dict = None, silent_mode = False):

        # TODO: Review weight label used to compute the shortest paths. Confirm if all simple paths are being generated or only a subset

        # for our specific usecase. Read on label correcting algorithms or even A*

        print('Generating ' + str(n_paths) + ' paths per od')


        t0 = time.time()

        # Path generation
        paths_od = {}
        # n_paths = 0
        paths = []

        # od = (19,7)
        # od = ods[0]

        # total_ods = len(ods)

        # print(total_ods)

        ods_no_paths = []


        #Create graph for validation

        multinetwork = (A > 1).any()
        dinetwork = (A <= 1).any()
        paths = []

        if dinetwork:
            G = nx.DiGraph()

        if multinetwork:
            G = nx.MultiDiGraph()

        # Labels does not matter
        G.add_nodes_from(np.arange(0, A.shape[0]))

        # Assign link utilities as weight if provided
        for (i, j) in zip(*A.nonzero()):
            for k in range(int(A[(i, j)])):
                G.add_edge(i, j)

        weight = None

        if edge_weights is not None:
            weight = 'utility'
            nx.set_edge_attributes(G, name=weight, values=edge_weights)

        for od, counter in zip(ods, range(len(ods))):
            # print(od)

            printer.enablePrint()
            printer.printProgressBar(counter, len(ods), prefix='Progress:', suffix='', length=20)

            if silent_mode is True:
                printer.blockPrint()


            paths_od[od] = list(k_simple_paths_nx(source=od[0], target=od[1], cutoff=cutoff, G = G, links = links, k = n_paths, weight = weight))

            #TODO: implement shortest path with igraph
            # g = Graph.from_networkx(nwx)
            # https: // igraph.org / python / doc / tutorial / generation.html

            # if n_paths is not None and n_paths < len(paths_od):
            #     paths_od[od] = random.sample(paths_od_temp, min(len(paths_od_temp), n_paths))
            # else:
            #     paths_od[od] = paths_od_temp

            # assert len(paths_od[od]) > n_paths, "Less than " + str(n_paths) + " paths between OD pair " + str(od[0])+ '-' +str(od[1])

            cutoff_increase_factor = config.Config('default').sim_options['cutoff_paths_increase_factor']
            max_attempts = config.Config('default').sim_options['max_attempts_path_generation']

        # nx.all_simple_paths()
        #     nx.all_shortest_paths()
            if len(paths_od[od]) < n_paths:

                attempt = 1

                ods_no_paths.append(od)

                # print(str(len(ods_no_paths)) + ' ods')

                # assert nx.has_path(G, source=od[0], target=od[1]), 'The od pair ' + str(od) + ' are not connected, thus, there are no paths'

                # Increase labels of OD pair by 1
                assert nx.has_path(G, source=od[0], target=od[1]), 'The od pair ' + str(tuple(map(lambda x: x+1,list(od)))) + ' is not connected, thus, there are no paths'

                new_cutoff = cutoff
                # max_paths = n_paths

                n_current_paths = len(paths_od[od])

                # print('With a cutoff of ' + str(
                #     new_cutoff) + ' links, ' + str(n_current_paths) + ' path(s) found in od-pair ' + str(od))

                # Increase labels of OD pair by 1
                print('With a cutoff of ' + str(
                    new_cutoff) + ' links, ' + str(n_current_paths) + ' path(s) found in od-pair ' + str(tuple(map(lambda x: x+1,list(od)))))

                # It makes sense to increase the cutoff only until the maximum number of links in the network has been reached

                total_links_reached = False
                n_links = len(links)

                while n_current_paths < n_paths and attempt < max_attempts and not total_links_reached:
                    # max_paths = len(list(nx.all_simple_paths(G, source=od[0], target=od[1])))
                    # nx.has_path()

                    # print('There are only ' + str(max_paths) + ' paths in od-pair ' + str(od))

                    new_cutoff = cutoff * cutoff_increase_factor * attempt

                    # print('To find more paths, the cutoff bound is increased to ' + str(new_cutoff) + ' links')

                    paths_od[od] = list(k_simple_paths_nx(source=od[0], target=od[1], G=G, links=links, k=n_paths, cutoff = new_cutoff))
                    attempt += 1

                # assert len(paths_od[od]) >= n_paths, 'No enough paths were found again'

                    # maximum links condition
                    if new_cutoff > n_links:
                        total_links_reached = True

                if len(paths_od[od]) > n_current_paths:
                    additional_paths = len(paths_od[od]) - n_current_paths
                    print(str(additional_paths) + ' simple path(s) found in ' + str(attempt) + ' attempts')
                    # print('No enough paths were found again')

                elif len(paths_od[od]) == n_current_paths:
                    print('No new paths were found by increasing the cutoff to ' + str(new_cutoff) + ' links')

                # assert len(paths_od[od]) > 0, "No paths between OD pair " + str(od[0]+1)+ '-' +str(od[1]+1)

            paths.extend(paths_od[od])
            # n_paths += len(paths_od[od])

        # # Store paths
        # self.paths = paths
        #
        # # Store paths per od
        # self.paths_od = paths_od

        # paths_od = A
        # n_paths = 2
        # paths = list(A.keys())
        # if n_paths is not None and n_paths < len(paths):
        #     self.paths = random.sample(paths, min(len(paths), n_paths))
        #     random_keys_paths_od = random.sample(list(paths_od), min(len(list(paths_od)), n_paths))
        #     paths_od = {k: paths_od[k] for k in random_keys_paths_od}



        # if len(paths) == 0:
        #     raise ValueError
        #
        # else:

        # print(len(ods_no_paths))

        print(str(len(paths)) + ' paths were generated among ' + str(len(ods)) + ' od pairs')
        print('Computation time: ' + str(np.round(time.time()- t0,1)) + ' [s]')

        if silent_mode is True:
            printer.enablePrint()

        return paths, paths_od #



# def all_paths_from_OD(G,Q):
#
#     G.nodes()
#     G.edges()
#
#
#     paths = list(nx.all_simple_paths(G, source = 0, target = 1))
#
#     for path in map(nx.utils.pairwise, paths):
#         print(list(path))
#
#     list(test_all_paths(G))
#
#     return paths