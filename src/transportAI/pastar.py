import numpy as np
import networkx as nx
import cvxpy as cp

def astar_path_heuristic_nodes(G, heuristic_costs, source, target, weight = 'weight'):

    def astar_heuristic(a, b):
        return heuristic_costs[a]

    astar_path = nx.astar_path(G=G, source=source, target=target, heuristic=astar_heuristic, weight = weight)

    return astar_path

def get_chosen_edges(paths):
    y_G = {}
    for observed_path, key in zip(paths.values(), range(len(paths))):
        y_G[key] = [(i, j) for i, j in zip(observed_path[:-1], observed_path[1:])]

    return y_G

def get_heuristic_goals(observed_paths: dict):
    return {i:path[-1] for i,path in observed_paths.items()}

def get_goal_dependent_heuristic_attribute_matrix(X, observed_paths: dict):

    n_nodes = X.shape[0]
    H = {i:np.asmatrix(np.transpose(np.repeat(np.array(X[:, path[-1]]),n_nodes, axis = 1))) for i,path in observed_paths.items()}

    return H


def path_length(G, path: list, attribute: str):
    return sum(dict(G.edges)[(i,j)][attribute]  for i,j in zip(path[:-1],path[1:]))

def paths_lengths(G, paths: dict, attribute: str):
    return {k: path_length(G, v, attribute) for k,v in paths.items()}



def get_edge_attributes_labels(G):
    return list(G.edges().values())[0].keys()

def max_admissible_heuristic(G, target: str):

    max_hcost_nodes = {}
    shortest_paths = dict(nx.all_pairs_dijkstra(G))

    for node in list(G.nodes):
        max_hcost_nodes[node] = shortest_paths[node][0][target]

    return max_hcost_nodes

def neighbors_path(G, path: list):

    neighbors_optimal_nodes = {node:list(neighbors) for node,neighbors in zip(path,list(map(G.neighbors,path)))}

    return neighbors_optimal_nodes

def set_heuristic_costs_nodes(G):

    h_cost_nodes = list(map(max, zip(list(nx.get_node_attributes(G, 'h_bound_optimal').values()),
                                     list(nx.get_node_attributes(G, 'h_bound_neighbor').values()))))
    h_cost_nodes = dict(zip(dict(G.nodes).keys(), h_cost_nodes))
    nx.set_node_attributes(G, values=h_cost_nodes, name='h')

    return G

def set_heuristic_costs_edges(G):
    '''Require that the node weight have already assigned'''

    if len(nx.get_node_attributes(G,'h')) == 0:
        set_heuristic_costs_nodes(G)

    h_cost_nodes = nx.get_node_attributes(G, 'h')
    h_cost_edges = {}

    for edge in dict(G.edges).keys():
        h_cost_edges[edge] = h_cost_nodes[edge[1]]

    nx.set_edge_attributes(G, values= h_cost_edges, name='h')

    return G

def heuristic_bounds(G,observed_path: list):

    target = observed_path[-1]

    nx.set_node_attributes(G, 0, 'f_bound_neighbor')
    nx.set_node_attributes(G, 0, 'h_bound_neighbor')
    nx.set_node_attributes(G, 0, 'h_bound_optimal')

    for observed_node in observed_path:
        # index = int(np.where(optimal_node == optimal_path)[0])
        # optimal_path[index:]
        #
        # optimal_path[3:]

        # cost_optimal_path_from_optimal_node =  optimal_path[]#nx.shortest_path(G,weight = 'weight', source = optimal_node, target = target)
        cost_observed_path_from_observed_node = nx.shortest_path_length(G, weight = 'weight', source = observed_node, target = target)

        for neighbor in list(G.neighbors(observed_node)):

            if neighbor not in observed_path:
                G.nodes(data=True)[neighbor]['f_bound_neighbor'] = max(G.nodes(data=True)[neighbor]['f_bound_neighbor'], cost_observed_path_from_observed_node)
                G.nodes(data=True)[neighbor]['h_bound_neighbor'] = max(0,G.nodes(data=True)[neighbor]['f_bound_neighbor']-dict(G.edges)[(observed_node,neighbor)]['weight'])
            else:
                G.nodes(data=True)[neighbor]['h_bound_optimal'] = nx.shortest_path_length(G, weight = 'weight', source = neighbor, target = target)

    # set_heuristic_costs_nodes(G)
    # set_heuristic_costs_edges(G)

    return G

def path_generator(G,n_paths: int, attribute = 'weight'):

    nodes_G = len(list(G.nodes))

    # Compute shortest path between every OD pairs
    # all_shortest_paths = dict(nx.all_pairs_shortest_path(G))
    all_shortest_paths = dict(nx.all_pairs_dijkstra_path(G))

    #Generate pair of distinct random numbers
    random_od_pairs = {i:np.random.choice(a=np.arange(nodes_G), size=2, replace=False) for i in range(n_paths)}

    #Sample of shortest paths
    random_shortest_paths = {i:all_shortest_paths[v[0]][v[1]] for (k,v),i in zip(random_od_pairs.items(), range(len(random_od_pairs)))}

    return random_shortest_paths





