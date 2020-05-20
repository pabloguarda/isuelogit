from inverse_astar_search.pastar import neighbors_path, paths_lengths # These dependencies can be removed

import numpy as np
import cvxpy as cp
import networkx as nx

def generate_choice_set_matrix_from_observed_path(G, observed_path):
    ''' Receive a list with the nodes in an observed path
        Return a matrix that encodes the choice sets at each node of the observed path
    '''

    nNodes = len(np.unique(list(G.nodes)))
    expanded_nodes = observed_path[:-1] #All nodes except target node
    # next_nodes = dict(zip(optimal_path[:-1], optimal_path[1:]))
    connected_nodes = neighbors_path(G = G, path = observed_path)

    choice_set_matrix = np.zeros([nNodes,nNodes])

    for expanded_node in expanded_nodes:
        choice_set_matrix[expanded_node, connected_nodes[expanded_node]] = 1
        # nRow += 1

    return choice_set_matrix
    # return avail_matrix

def get_list_attribute_vectors_choice_sets(choice_set_matrix, Xk):
    ''' Get a list with attribute values (vector, i.e. Matrix 1D) for each alternative in the choice set (only with entries different than 0)'''
    Xk_avail = []

    for i in range(choice_set_matrix.shape[0]):
        Xk_avail.append(Xk[i,np.where(choice_set_matrix[i, :] == 1)[0]])

    return Xk_avail

def widetolong(wide_matrix):
    """Wide to long format
    The new matrix has one rows per alternative
    """

    wide_matrix = wide_matrix.astype(int)

    if wide_matrix.ndim == 1:
        wide_matrix = wide_matrix.reshape(1, wide_matrix.shape[0])

    long_matrix = np.repeat(wide_matrix, repeats=np.sum(wide_matrix, axis=1), axis=0)

    return long_matrix

def compute_edge_utility(G, theta: dict):
    attributes = list(theta.keys())

    utility = np.zeros(len(G.edges))

    for attribute in attributes:
        utility += theta[attribute] * np.array(list(nx.get_edge_attributes(G, attribute).values()))

    return dict(zip(G.edges, utility))

def compute_goal_dependent_heuristic_utility(G, observed_paths:dict, H_G: dict, theta_H: dict, K_H: dict):
    ''':argument '''

    utility_G = {} #Node to node utility matrix which is dependent on the goal in the observed path

    for i, path in observed_paths.items():
        utility_G[i] = np.zeros([len(G.nodes()),len(G.nodes())])
        for k_H in K_H:
            utility_G[i] += theta_H[k_H] * np.array(H_G[k_H][i])

    return utility_G


def logit_estimation(C, X, K_X, y, H, K_H, g):
    '''

    :argument C: dictionary with matrices encoding the choice sets (set) associated to each (expanded) node in observed path i
    :argument X: dictionary with edge-to-edge matrices with network attributes values
    :argument K_X: subset of attributes from X chosen to fit discrete choice model
    :argument y: dictionary with list of chosen edges in path i
    :argument H: Nested dictionary (one per goal node) with edge-to-edge matrices of attribute values which are goal dependent (heuristic costs that varies with the goal)
    :argument K_H: dictionary with subset of attributes from H chosen to fit discrete choice model
    :argument g: dictionary with goal (destination) in path i
    '''

    #List with all attributes
    K = K_X + K_H

    # Dictionary with all attribute values together
    XH = X
    XH.update(H)

    #Estimated parameters to be optimized (learned)
    cp_theta = {i:cp.Variable(1) for i in K}

    # Number of paths
    n_paths = len(C.items())

    #Dictionary with list for nodes connected (alternatives) with each observed (expanded) node in each observed path i (key)
    nodes_alternatives = {i:[y_j[0] for y_j in y_i] for i,y_i in zip(range(len(y)),y.values())}

    # Dictionary with list of nodes expanded (chosen) in each observed path i (key)
    nodes_chosen = {i: [y_j[1] for y_j in y_i] for i, y_i in zip(range(len(y)), y.values())}

    # Nested dictionary of the attribute's (bottom level) values of the nodes(alternatives)
    # connected to each (expanded) node in the each observed path (top level)
    X_c = {}
    for i, c_matrix_path in C.items():
        X_c[i] = {attribute:get_list_attribute_vectors_choice_sets(c_matrix_path, X[attribute]) for attribute in K_X}

    # Nested dictionary for heuristic attributes which are goal dependent
    H_c = {}
    for i, c_matrix_path in C.items():
        H_c[i] = {attribute: get_list_attribute_vectors_choice_sets(c_matrix_path, H[attribute][i]) for attribute in K_H}

    # Loglikelihood function obtained from iterating across choice sets
    Z = []

    for i in range(n_paths):

        #List storing the contribution from each choice (expansion) set to the likelihood
        Z_i = []

        for j,k in zip(nodes_alternatives[i],nodes_chosen[i]):
            Z_chosen_attr = []
            Z_logsum_attr = []

            for attribute in K_X:
                Z_chosen_attr.append(X[attribute][j,k] * cp_theta[attribute])
                Z_logsum_attr.append(X_c[i][attribute][j] * cp_theta[attribute])

            for attribute in K_H:
                Z_chosen_attr.append(H[attribute][i][j,k] * cp_theta[attribute])
                Z_logsum_attr.append(H_c[i][attribute][j] * cp_theta[attribute])

            Z_i.append(cp.sum(Z_chosen_attr) - cp.log_sum_exp(cp.sum(Z_logsum_attr)))

        Z.append(cp.sum(Z_i))


    cp_objective_logit = cp.Maximize(cp.sum(Z))

    cp_problem_logit = cp.Problem(cp_objective_logit, constraints = []) #Excluding heuristic constraints

    cp_problem_logit.solve()

    return {key:val.value for key,val in cp_theta.items()}

# def non_negative_costs_edges():

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
        predicted_paths[key] = nx.astar_path(G=G_copy, source=observed_path[0], target=observed_path[-1], weight='weights_prediction')

    predicted_paths_length = paths_lengths(G_copy,predicted_paths,'utility') #Utility acts as a proxy of the path length (negative)

    return {'predicted_path': predicted_paths, 'length': predicted_paths_length}

def pastar_path_predictions(G, observed_paths: dict, H: dict, theta_X: dict, theta_H: dict, endogenous_heuristic = False):

    predicted_paths = {}
    predicted_paths_length = {}
    U_H = 0

    # Keep track of numbers of iterations made by astar for each path
    n_iterations = {}

    if len(theta_H) == 0:
        predictions = logit_path_predictions(G = G, observed_paths = observed_paths
                                                                         , theta_logit = theta_X)
        predicted_paths = predictions['predicted_path']
        predicted_paths_length = predictions['length']

    else:
        # Edge attributes component in utility
        edge_utilities = compute_edge_utility(G, theta= theta_X)
        edge_weights = {edge:-u for edge,u in edge_utilities.items()}

        # All utilities are positive so a-star can run properly.
        min_edge_weight = min(list(edge_weights.values()))
        if min_edge_weight < 0:
            edge_weights = {i: w + abs(min_edge_weight) for i, w in edge_weights.items()}

        # Heuristic components in utility
        K_H = list(H.keys())

        U_H = compute_goal_dependent_heuristic_utility(G, observed_paths=observed_paths, H_G=H, theta_H=theta_H, K_H=K_H)

        for i,observed_path in observed_paths.items():

            G_copy = G.copy()
            edge_heuristic_weights = edge_weights.copy()
            n_iterations[i] = 0

            heuristic_weights_path = -U_H[i]
            min_heuristic = np.min(heuristic_weights_path)

            if min_heuristic < 0:
                heuristic_weights_path = heuristic_weights_path + abs(min_heuristic)

            #TODO: The loop below introduces correlation between alternatives which violates key assumption in Multinomial Logit Model
            # Correlation arises from the fact that multiple edges may have the same or similar heuristic cost.
            # As expected, there is a significant increse in accuracy

            if endogenous_heuristic is True:
                for edge in edge_heuristic_weights.keys():
                    edge_heuristic_weights[edge] += heuristic_weights_path[edge]

            nx.set_edge_attributes(G_copy, values= edge_heuristic_weights, name= 'weights_prediction')

            def astar_heuristic(a, b):
                # a is the neighboor and the heuristc_weights matrix have all rows equal and the column (:,a) gives distance o goal
                # print(heuristic_weights_path[(0,a)])
                n_iterations[i] += 1
                return heuristic_weights_path[(0,a)]

            predicted_paths[i] = nx.astar_path(G=G_copy, source=observed_path[0], target=observed_path[-1],
                                             weight='weights_prediction', heuristic=astar_heuristic)

        predicted_paths_length = paths_lengths(G, predicted_paths,'utility')  # Utility acts as a proxy of the path length (negative)

    return {'predicted_path': predicted_paths, 'length': predicted_paths_length, 'n_iterations_astar': n_iterations}


def accuracy_pastar_predictions(G, predicted_paths: dict, observed_paths:dict):

    # paths_lengths(G, paths = predicted_paths, attribute = 'utility')

    edge_acc = {}
    for key,observed_path in observed_paths.items():
        edge_acc[key] = sum(el in observed_path for el in predicted_paths[key])/len(observed_path)

    path_acc = dict(zip(edge_acc.keys(),np.where(np.array(list(edge_acc.values())) < 1, 0, 1)))

    x_correct_edges = np.round(np.mean(np.array(list(edge_acc.values()))),4)
    x_correct_paths = np.round(np.mean(np.array(list(path_acc.values()))),4)

    # utility_diff = abs(sum(paths_lengths(G, paths= predicted_paths, attribute='utility').values())
    #                   -abs(sum(paths_lengths(G, paths= observed_paths, attribute='utility').values()))
    #                   )

    return {'acc_edges': x_correct_edges, 'acc_paths': x_correct_paths}