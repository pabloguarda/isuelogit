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


def compute_goal_dependent_heuristic_utility(G, observed_paths: dict, H_G: dict, theta_H: dict, K_H: dict):
    ''':argument '''

    utility_G = {}  # Node to node utility matrix which is dependent on the goal in the observed path

    for i, path in observed_paths.items():
        utility_G[i] = np.zeros([len(G.nodes()), len(G.nodes())])
        for k_H in K_H:
            utility_G[i] += theta_H[k_H] * np.array(H_G[k_H][i])

    return utility_G


def recursive_logit_estimation(C, X, K_X, y, H, K_H, g):
    '''

    :argument C: dictionary with matrices encoding the choice sets (set) associated to each (expanded) node in observed path i
    :argument X: dictionary with edge-to-edge matrices with network attributes values
    :argument K_X: subset of attributes from X chosen to fit discrete choice model
    :argument y: dictionary with list of chosen edges in path i
    :argument H: Nested dictionary (one per goal node) with edge-to-edge matrices of attribute values which are goal dependent (heuristic costs that varies with the goal)
    :argument K_H: dictionary with subset of attributes from H chosen to fit discrete choice model
    :argument g: dictionary with goal (destination) in path i
    '''

    # List with all attributes
    K = K_X + K_H

    # Dictionary with all attribute values together
    XH = X
    XH.update(H)

    # Estimated parameters to be optimized (learned)
    cp_theta = {i: cp.Variable(1) for i in K}

    # Number of paths
    n_paths = len(C.items())

    # Dictionary with list for nodes connected (alternatives) with each observed (expanded) node in each observed path i (key)
    nodes_alternatives = {i: [y_j[0] for y_j in y_i] for i, y_i in zip(range(len(y)), y.values())}

    # Dictionary with list of nodes expanded (chosen) in each observed path i (key)
    nodes_chosen = {i: [y_j[1] for y_j in y_i] for i, y_i in zip(range(len(y)), y.values())}

    # Nested dictionary of the attribute's (bottom level) values of the nodes(alternatives)
    # connected to each (expanded) node in the each observed path (top level)
    X_c = {}
    for i, c_matrix_path in C.items():
        X_c[i] = {attribute: get_list_attribute_vectors_choice_sets(c_matrix_path, X[attribute]) for attribute in K_X}

    # Nested dictionary for heuristic attributes which are goal dependent
    H_c = {}
    for i, c_matrix_path in C.items():
        H_c[i] = {attribute: get_list_attribute_vectors_choice_sets(c_matrix_path, H[attribute][i]) for attribute in
                  K_H}

    # Loglikelihood function obtained from iterating across choice sets
    Z = []

    for i in range(n_paths):

        # List storing the contribution from each choice (expansion) set to the likelihood
        Z_i = []

        for j, k in zip(nodes_alternatives[i], nodes_chosen[i]):
            Z_chosen_attr = []
            Z_logsum_attr = []

            for attribute in K_X:
                Z_chosen_attr.append(X[attribute][j, k] * cp_theta[attribute])
                Z_logsum_attr.append(X_c[i][attribute][j] * cp_theta[attribute])

            for attribute in K_H:
                Z_chosen_attr.append(H[attribute][i][j, k] * cp_theta[attribute])
                Z_logsum_attr.append(H_c[i][attribute][j] * cp_theta[attribute])

            Z_i.append(cp.sum(Z_chosen_attr) - cp.log_sum_exp(cp.sum(Z_logsum_attr)))

        Z.append(cp.sum(Z_i))

    cp_objective_logit = cp.Maximize(cp.sum(Z))

    cp_problem_logit = cp.Problem(cp_objective_logit, constraints=[])  # Excluding heuristic constraints

    cp_problem_logit.solve()

    return {key: val.value for key, val in cp_theta.items()}


def logit_path_predictions(G, observed_paths: dict, theta_logit: dict):
    G_copy = G.copy()

    # Edge attributes component in utility
    edge_utilities = compute_edge_utility(G, theta=theta_logit)
    edge_weights = {edge: -u for edge, u in edge_utilities.items()}

    # All utilities are positive so a-star can run properly.
    min_edge_weight = min(list(edge_weights.values()))
    if min_edge_weight < 0:
        edge_weights = {i: w + abs(min_edge_weight) for i, w in edge_weights.items()}

    nx.set_edge_attributes(G_copy, values=edge_weights, name='weights_prediction')

    # nx.get_edge_attributes(G_copy, 'utility_prediction')

    predicted_paths = {}

    for key, observed_path in observed_paths.items():
        predicted_paths[key] = nx.astar_path(G=G_copy, source=observed_path[0], target=observed_path[-1],
                                             weight='weights_prediction')

    predicted_paths_length = paths_lengths(G_copy, predicted_paths,
                                           'utility')  # Utility acts as a proxy of the path length (negative)

    return {'predicted_path': predicted_paths, 'length': predicted_paths_length}


def pastar_path_predictions(G, observed_paths: dict, H: dict, theta_X: dict, theta_H: dict, endogenous_heuristic=False):
    predicted_paths = {}
    predicted_paths_length = {}
    U_H = 0

    # Keep track of numbers of iterations made by astar for each path
    n_iterations = {}

    if len(theta_H) == 0:
        predictions = logit_path_predictions(G=G, observed_paths=observed_paths
                                             , theta_logit=theta_X)
        predicted_paths = predictions['predicted_path']
        predicted_paths_length = predictions['length']

    else:
        # Edge attributes component in utility
        edge_utilities = compute_edge_utility(G, theta=theta_X)
        edge_weights = {edge: -u for edge, u in edge_utilities.items()}

        # All utilities are positive so a-star can run properly.
        min_edge_weight = min(list(edge_weights.values()))
        if min_edge_weight < 0:
            edge_weights = {i: w + abs(min_edge_weight) for i, w in edge_weights.items()}

        # Heuristic components in utility
        K_H = list(H.keys())

        U_H = compute_goal_dependent_heuristic_utility(G, observed_paths=observed_paths, H_G=H, theta_H=theta_H,
                                                       K_H=K_H)

        for i, observed_path in observed_paths.items():

            G_copy = G.copy()
            edge_heuristic_weights = edge_weights.copy()
            n_iterations[i] = 0

            heuristic_weights_path = -U_H[i]
            min_heuristic = np.min(heuristic_weights_path)

            if min_heuristic < 0:
                heuristic_weights_path = heuristic_weights_path + abs(min_heuristic)

            # TODO: The loop below introduces correlation between alternatives which violates key assumption in Multinomial Logit Model
            # Correlation arises from the fact that multiple edges may have the same or similar heuristic cost.
            # As expected, there is a significant increse in accuracy

            if endogenous_heuristic is True:
                for edge in edge_heuristic_weights.keys():
                    edge_heuristic_weights[edge] += heuristic_weights_path[edge]

            nx.set_edge_attributes(G_copy, values=edge_heuristic_weights, name='weights_prediction')

            def astar_heuristic(a, b):
                # a is the neighboor and the heuristc_weights matrix have all rows equal and the column (:,a) gives distance o goal
                # print(heuristic_weights_path[(0,a)])
                n_iterations[i] += 1
                return heuristic_weights_path[(0, a)]

            predicted_paths[i] = nx.astar_path(G=G_copy, source=observed_path[0], target=observed_path[-1],
                                               weight='weights_prediction', heuristic=astar_heuristic)

        predicted_paths_length = paths_lengths(G, predicted_paths,
                                               'utility')  # Utility acts as a proxy of the path length (negative)

    return {'predicted_path': predicted_paths, 'length': predicted_paths_length, 'n_iterations_astar': n_iterations}


def accuracy_pastar_predictions(G, predicted_paths: dict, observed_paths: dict):
    # paths_lengths(G, paths = predicted_paths, attribute = 'utility')

    edge_acc = {}
    for key, observed_path in observed_paths.items():
        edge_acc[key] = sum(el in observed_path for el in predicted_paths[key]) / len(observed_path)

    path_acc = dict(zip(edge_acc.keys(), np.where(np.array(list(edge_acc.values())) < 1, 0, 1)))

    x_correct_edges = np.round(np.mean(np.array(list(edge_acc.values()))), 4)
    x_correct_paths = np.round(np.mean(np.array(list(path_acc.values()))), 4)

    # utility_diff = abs(sum(paths_lengths(G, paths= predicted_paths, attribute='utility').values())
    #                   -abs(sum(paths_lengths(G, paths= observed_paths, attribute='utility').values()))
    #                   )

    return {'acc_edges': x_correct_edges, 'acc_paths': x_correct_paths}


def generate_choice_set_matrix_from_observed_path(G, observed_path):
    ''' Receive a list with the nodes in an observed path
        Return a matrix that encodes the choice sets at each node of the observed path
    '''

    nNodes = len(np.unique(list(G.nodes)))
    expanded_nodes = observed_path[:-1]  # All nodes except target node
    # next_nodes = dict(zip(optimal_path[:-1], optimal_path[1:]))
    connected_nodes = neighbors_path(G=G, path=observed_path)

    choice_set_matrix = np.zeros([nNodes, nNodes])

    for expanded_node in expanded_nodes:
        choice_set_matrix[expanded_node, connected_nodes[expanded_node]] = 1
        # nRow += 1

    return choice_set_matrix
    # return avail_matrix


def compute_edge_utility(G, theta: dict):
    attributes = list(theta.keys())

    utility = np.zeros(len(G.edges))

    for attribute in attributes:
        utility += theta[attribute] * np.array(list(nx.get_edge_attributes(G, attribute).values()))

    return dict(zip(G.edges, utility))






