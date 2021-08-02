# =============================================================================
# Imports
# =============================================================================

# Internal modules
from inverse_astar_search.network import create_network, set_random_nodes_coordinates, get_edges_euclidean_distances\
    , get_euclidean_distances_between_nodes

from inverse_astar_search.pastar import heuristic_bounds, astar_path_heuristic_nodes\
    , set_heuristic_costs_nodes, set_heuristic_costs_edges, get_edge_attributes_labels,path_generator\
    , get_chosen_edges, path_length, paths_lengths, neighbors_path, get_heuristic_goals, get_goal_dependent_heuristic_attribute_matrix

from inverse_astar_search.logit import generate_choice_set_matrix_from_observed_path\
    , recursive_logit_estimation, compute_edge_utility, logit_path_predictions, pastar_path_predictions\
    , accuracy_pastar_predictions

# External modules
import numpy as np
import networkx as nx
import cvxpy as cp

# =============================================================================
# Network Factory
# =============================================================================

def set_random_edge_attributes(G):

    # - Distance [m]  - based on nodes coordinates
    distances_edges = get_edges_euclidean_distances(G, nodes_coordinate_label='pos')
    nx.set_edge_attributes(G, distances_edges, name='distance')

    # - Cost ($)
    cost_edges = dict(zip(dict(G.edges).keys(), np.random.randint(0, 20, len(list(G.edges)))))
    nx.set_edge_attributes(G, values=cost_edges, name='cost')

    # - Speed (km/hr) - Set to constant meanwhile
    constant_speed = 20
    speed_edges = dict(zip(dict(G.edges).keys(),np.repeat(constant_speed, len(G.edges()))))
    nx.set_edge_attributes(G, values=speed_edges, name='speed')

    # - Travel time (mins) - based on distance between edges and a constant speed
    travel_time_edges = dict(zip(dict(G.edges).keys(),
                                 60/1000 * np.array(list(nx.get_edge_attributes(G, 'distance').values()))/np.array(list(nx.get_edge_attributes(G, 'speed').values())),
                                          ))
    nx.set_edge_attributes(G, values=travel_time_edges, name='travel_time')

    return G

def create_network_data(n_nodes, n_sample_paths: int, theta_logit: dict, attributes_thresholds: dict = None):

    nodes_G = n_nodes

    # Adjacency matrix
    A = np.random.randint(0, 2, [nodes_G, nodes_G])

    # Create networkX graph
    G = create_network(A)

    # Node attributes

    # - Coordinates - selected at random and using factor = 1000 so they are in 'metres' (number between 0 and 1000)
    G = set_random_nodes_coordinates(G, attribute_label = 'pos', factor=1000)
    nx.get_node_attributes(G, name='pos')

    # Edges attributes (distance, cost, travel time)
    G = set_random_edge_attributes(G)

    # Utility at edges
    utility_edges = compute_edge_utility(G, theta=theta_logit)
    nx.set_edge_attributes(G, utility_edges, name='utility')

    # Edge weight equals utility
    weight_edges = {key: -val for key, val in nx.get_edge_attributes(G, 'utility').items()}
    nx.set_edge_attributes(G, values=weight_edges, name='weight')

    # Simulate observed paths - by sampling from the set of all shortest path in the network
    observed_paths = path_generator(G = G, n_paths =n_sample_paths, attribute='utility')

    return G, observed_paths

# Preference parameters
theta_logit_true_G_training = {'travel_time': -4, 'cost': -2} #Signs are not required to be negative
vot = theta_logit_true_G_training['travel_time'] / theta_logit_true_G_training['cost']

G_training, observed_paths_training = create_network_data(n_nodes = 20, n_sample_paths = 40, theta_logit = theta_logit_true_G_training)
# n_nodes = 20; n_sample_paths = 100; theta= theta_G_training

# =============================================================================
# Logit Estimation
# =============================================================================

# Matrix with choice sets generated at each node
choice_sets_paths_G_training = {key: generate_choice_set_matrix_from_observed_path(G = G_training, observed_path = observed_path) for key,observed_path  in observed_paths_training.items()}

# Edge Attributes
edge_attributes_G_training = ['travel_time','cost'] # get_edge_attributes_labels(G_training) # Candidate attributes

# - Dictionary with matrix of attributes between every OD pair
X_G_training = {attribute:nx.adjacency_matrix(G_training, weight = attribute).todense() for attribute in edge_attributes_G_training}

# - Chosen edges (edge between expanded nodes)
y_G_training = get_chosen_edges(paths = observed_paths_training)

# Heuristic attributes
heuristic_attributes_G_training = ['h_g']

#Goals
g_G_training = get_heuristic_goals(observed_paths_training)

#Dictionary with matrix of heuristic attributes between every OD pair
H_G_training = {}

# Admissible travel time (normalizing by maximum speed across edges)
H_G_training['h_g'] = get_goal_dependent_heuristic_attribute_matrix(get_euclidean_distances_between_nodes(G = G_training), observed_paths = observed_paths_training) # Euclidian (aerial) distance to goal

#Logit estimates

# - Without heuristic (standard logit)
edge_attributes_standard_logit_G = edge_attributes_G_training # [i for i in edge_attributes_G_training if i != 'h']
theta_logit_standard_G_training = recursive_logit_estimation(C= choice_sets_paths_G_training
                                         , X = X_G_training, K_X = edge_attributes_standard_logit_G, y = y_G_training
                                         , H = [], K_H = [], g = [])

# print(theta_logit_standard_G_training)
# print(theta_logit_true_G_training)

# - With (endogenous) heuristic
theta_logit_heuristic_G_training = recursive_logit_estimation(C= choice_sets_paths_G_training
                                         , X = X_G_training, K_X = edge_attributes_G_training, y = y_G_training
                                         , H = H_G_training, K_H = heuristic_attributes_G_training, g = g_G_training
                                         )

theta_logit_X_heuristic_G_training = {key:theta_logit_heuristic_G_training[key] for key in edge_attributes_G_training}
theta_logit_H_heuristic_G_training = {key:theta_logit_heuristic_G_training[key] for key in heuristic_attributes_G_training}

# print(theta_logit_H_heuristic_G_training)
# print(theta_logit_X_heuristic_G_training)
# print(theta_logit_true_G_training)


# =============================================================================
# Prediction in training data
# =============================================================================

#Path predictions based on logit estimates


predictions_training = {}

#- Observed paths
# print(*observed_paths.values(), sep='\n')
observed_paths_training
paths_lengths(G_training, paths = observed_paths_training, attribute = 'utility')

# - Standard logit model (Exclusion of heuristic cost in estimation and prediction)
# prediction_logit_standard_training = logit_path_predictions(G = G_training, observed_paths = observed_paths_training
#                                                             , theta_logit = theta_logit_standard_G_training)

predictions_training['logit_standard'] = pastar_path_predictions(G = G_training, observed_paths = observed_paths_training
                                                             , theta_X = theta_logit_standard_G_training
                                                             , theta_H = []
                                                             , H = H_G_training)


# - Adding endogenous heuristic cost (included in estimation)
predictions_training['endogenous_heuristic'] = pastar_path_predictions(G = G_training, observed_paths = observed_paths_training
                                                                         , theta_X = theta_logit_X_heuristic_G_training
                                                                         , theta_H = theta_logit_H_heuristic_G_training
                                                                         , H = H_G_training
                                                                         , endogenous_heuristic = True)


# - Ignoring heuristic cost from estimation but included for prediction

admissible_aerial_heuristic = {'h_g':theta_logit_standard_G_training['travel_time']/(1000*20/60)}

predictions_training['exogenous_heuristic'] = pastar_path_predictions(G = G_training, observed_paths = observed_paths_training #{list(observed_paths_training.keys())[6]: list(observed_paths_training.values())[6]}
                                                                 , theta_X = theta_logit_standard_G_training
                                                                 , theta_H = admissible_aerial_heuristic
                                                                 , H = H_G_training)
# =============================================================================
# Generalization power (testing data)
# =============================================================================

# Create testing data (using same preference parameters, but different network and sample of paths)
theta_logit_true_G_testing = theta_logit_true_G_training
G_testing, observed_paths_testing = create_network_data(n_nodes = 30, n_sample_paths = 100, theta_logit = theta_logit_true_G_testing)

# Predictions

predictions_testing = {}

#Dictionary with matrix of heuristic attributes between every OD pair
H_G_testing = {}

# Euclidian (aerial) distance to goal
H_G_testing['h_g'] = get_goal_dependent_heuristic_attribute_matrix(get_euclidean_distances_between_nodes(G = G_testing), observed_paths = observed_paths_testing)

#Predictions with pastar
predictions_testing['logit_standard'] = pastar_path_predictions(G = G_testing, observed_paths = observed_paths_testing
                                                             , theta_X = theta_logit_standard_G_training
                                                            , theta_H=[]
                                                            , H=H_G_training)

predictions_testing['endogenous_heuristic'] = pastar_path_predictions(G = G_testing, observed_paths = observed_paths_testing
                                                              , theta_X = theta_logit_X_heuristic_G_training
                                                              , theta_H = theta_logit_H_heuristic_G_training
                                                              , H = H_G_testing
                                                              , endogenous_heuristic=True
                                                                        )

predictions_testing['exogenous_heuristic'] = pastar_path_predictions(G = G_testing, observed_paths = observed_paths_testing
                                                                 , theta_X = theta_logit_standard_G_training
                                                                 , theta_H = {'h_g':0*admissible_aerial_heuristic['h_g']} #Admissible but less efficient
                                                                 , H = H_G_testing)

# =============================================================================
# Summary tables
# =============================================================================

# a) Accuracy training sample

# Observed paths (accuracy 100%)
print('Observed paths')
print(accuracy_pastar_predictions(G = G_training, predicted_paths = observed_paths_training, observed_paths= observed_paths_training))

#Standard logit model
print('Standard logit model')
print(accuracy_pastar_predictions(G = G_training, predicted_paths = predictions_training['logit_standard']['predicted_path'], observed_paths= observed_paths_training))

#Logit model and heuristic (pastar)
print('Pastar with endogenous heuristic')
print(accuracy_pastar_predictions(G = G_training, predicted_paths = predictions_training['endogenous_heuristic']['predicted_path'], observed_paths= observed_paths_training))

#Pastar with no heuristic
print('Pastar with exogeneous heuristic')
print(accuracy_pastar_predictions(G = G_training, predicted_paths = predictions_training['exogenous_heuristic']['predicted_path'], observed_paths= observed_paths_training))

# b) Accuracy testing sample

# Observed paths (accuracy 100%)
print('Observed paths')
print(accuracy_pastar_predictions(G = G_testing, predicted_paths = observed_paths_testing, observed_paths= observed_paths_testing))

#Standard logit model
print('Standard logit model')
print(accuracy_pastar_predictions(G = G_testing, predicted_paths = predictions_testing['logit_standard']['predicted_path'], observed_paths= observed_paths_testing))

#Logit model and heuristic (pastar)
print('Pastar with endogenous heuristic')
print(accuracy_pastar_predictions(G = G_testing, predicted_paths = predictions_testing['endogenous_heuristic']['predicted_path'], observed_paths= observed_paths_testing))

#Pastar with no heuristic
print('Pastar with exogenous heuristic')
print(accuracy_pastar_predictions(G = G_testing, predicted_paths = predictions_testing['exogenous_heuristic']['predicted_path'], observed_paths= observed_paths_testing))

#Table path predictions

# list(prediction_logit_heuristic['predicted_path'].values())
# list(prediction_logit_standard['predicted_path'].values())

# =============================================================================
# Summary plots
# =============================================================================

# Packages
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True


# a) Trade-off accuracy versus efficiency for a range of admissible or not admissible heuristics

admissibility_factors = np.arange(0,10,0.5)
predictions_vs_admissibility_factor = {}
accuracy_vs_admissibility_factor = {}
efficiency_vs_admissibility_factor = {}

# i) Training
for admissibility_factor in admissibility_factors:
    value_heuristic = admissibility_factor * admissible_aerial_heuristic['h_g']
    admissibility_factor = np.round(admissibility_factor,4)

    aerial_heuristic_time_utility_units = {'h_g': value_heuristic}

    predictions_vs_admissibility_factor = pastar_path_predictions(G = G_training, observed_paths = observed_paths_training
                                  , theta_X = theta_logit_standard_G_training
                                  , theta_H = aerial_heuristic_time_utility_units
                                  , H = H_G_training)

    accuracy_vs_admissibility_factor[admissibility_factor] =  accuracy_pastar_predictions(G = G_training
                                       , predicted_paths = predictions_vs_admissibility_factor['predicted_path']
                                       , observed_paths= observed_paths_training)

    efficiency_vs_admissibility_factor[admissibility_factor] = np.sum(list(predictions_vs_admissibility_factor['n_iterations_astar'].values()))

# print(accuracy_vs_admissibility_factor)
# print(efficiency_vs_admissibility_factor)

## - Plot Accuracy (Iterations) versus admissibility

accuracy_edges_list = [acc['acc_edges'] for acc in list(accuracy_vs_admissibility_factor.values())]
accuracy_paths_list = [acc['acc_paths'] for acc in list(accuracy_vs_admissibility_factor.values())]
admisibility_factors_list = list(accuracy_vs_admissibility_factor.keys())

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(admisibility_factors_list, accuracy_edges_list, label='Edges')
plt.plot(admisibility_factors_list, accuracy_paths_list, label='Paths')

#plt.plot(val_loss_list, label='validation')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Aerial Heuristic x Factor', fontsize=12)

plt.tight_layout()
plt.xticks(np.arange(np.min(admisibility_factors_list),np.max(admisibility_factors_list), 1))
plt.yticks(np.append(np.arange(np.round(np.min(accuracy_paths_list),2), 1, 0.05),1))

plt.legend()
plt.savefig('examples/figures/accuracy_admissibility_training.pdf')
plt.show()

## - Plot Efficiency (Iterations) versus admissibility
n_iterations_dijkstra = efficiency_vs_admissibility_factor[0]
efficiency_list = [round(1-n_iterations/n_iterations_dijkstra,4) for n_iterations in list(efficiency_vs_admissibility_factor.values())]
admisibility_factors_list = list(accuracy_vs_admissibility_factor.keys())

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(admisibility_factors_list, efficiency_list)

plt.ylabel('Efficiency gain (%)', fontsize=12)
plt.xlabel('Aerial Heuristic x Factor', fontsize=12)

plt.tight_layout()
plt.xticks(np.arange(np.min(admisibility_factors_list),np.max(admisibility_factors_list), 1))
plt.savefig('examples/figures/efficiency_admissibility_training.pdf')
plt.show()

# - Combined plot of accuracy and efficiency

# Double axis plots
fig, ax = plt.subplots()

plt.xlabel('Aerial Heuristic x Factor', fontsize=12)
plt.plot(admisibility_factors_list, accuracy_edges_list, '-r', label='Edges')
plt.plot(admisibility_factors_list, accuracy_paths_list, '--r', label='Paths')
plt.yticks(np.arange(np.round(min(accuracy_paths_list),1), 1.05, 0.1))
plt.ylabel('Accuracy (%)', fontsize=12)
ax.yaxis.key.set_color('red')
ax.legend(loc="upper center")

# Admissibility Line
plt.axvline(1, 0, 1.05, label='Admisibility boundary', color='red', linestyle = 'dotted' )

# Get second axis
ax2 = ax.twinx()
plt.plot(admisibility_factors_list, efficiency_list, '-b')
plt.yticks(np.arange(0, np.round(np.max(efficiency_list),1) + 0.06, 0.05))

plt.ylabel('Efficiency Gain (%)', fontsize=12)
plt.axhline(efficiency_list[2], 0, 9, label= 'Efficiency boundary', color='blue', linestyle = 'dotted' )
ax2.yaxis.key.set_color('blue')

plt.xticks(np.arange(0, max(admisibility_factors_list)+1, 1))
plt.tight_layout()
plt.savefig('examples/figures/accuracy_efficiency_admissibility_training.pdf', format='pdf')
plt.show()

# ii) Testing

for admissibility_factor in admissibility_factors:
    value_heuristic = admissibility_factor * admissible_aerial_heuristic['h_g']
    admissibility_factor = np.round(admissibility_factor,4)

    aerial_heuristic_time_utility_units = {'h_g': value_heuristic}

    predictions_vs_admissibility_factor = pastar_path_predictions(G = G_testing, observed_paths = observed_paths_testing
                                  , theta_X = theta_logit_standard_G_training
                                  , theta_H = aerial_heuristic_time_utility_units
                                  , H = H_G_testing)

    accuracy_vs_admissibility_factor[admissibility_factor] =  accuracy_pastar_predictions(G = G_testing
                                       , predicted_paths = predictions_vs_admissibility_factor['predicted_path']
                                       , observed_paths= observed_paths_testing)

    efficiency_vs_admissibility_factor[admissibility_factor] = np.sum(list(predictions_vs_admissibility_factor['n_iterations_astar'].values()))

# print(accuracy_vs_admissibility_factor)
# print(efficiency_vs_admissibility_factor)

## - Plot Accuracy versus admissibility

accuracy_edges_list = [acc['acc_edges'] for acc in list(accuracy_vs_admissibility_factor.values())]
accuracy_paths_list = [acc['acc_paths'] for acc in list(accuracy_vs_admissibility_factor.values())]
admisibility_factors_list = list(accuracy_vs_admissibility_factor.keys())

## - Plot Effiency (Iterations) versus admissibility
n_iterations_dijkstra = efficiency_vs_admissibility_factor[0]
efficiency_list = [round(1-n_iterations/n_iterations_dijkstra,4) for n_iterations in list(efficiency_vs_admissibility_factor.values())]
admisibility_factors_list = list(accuracy_vs_admissibility_factor.keys())

# - Combined plot of accuracy and efficiency

# Double axis plots
fig, ax = plt.subplots()

plt.xlabel('Aerial Heuristic x Factor', fontsize=12)
# axes = plt.gca()
# axes.set_ylim([0.5,np.max(accuracy_paths_list)*1.2])
plt.plot(admisibility_factors_list, accuracy_edges_list, '-r', label='Edges')
plt.plot(admisibility_factors_list, accuracy_paths_list, '--r', label='Paths')
plt.yticks(np.arange(np.round(min(accuracy_paths_list),1), 1.05, 0.1))
plt.ylabel('Accuracy (%)', fontsize=12)
ax.yaxis.key.set_color('red')
ax.legend(loc="upper center")

# Admissibility Lines
plt.axvline(1, 0, 1.05, label='Admisibility boundary', color='red', linestyle = 'dotted' )

# Get second axis
ax2 = ax.twinx()
plt.plot(admisibility_factors_list, efficiency_list, '-b')
plt.yticks(np.arange(0, np.round(np.max(efficiency_list),1) + 0.06, 0.05))

plt.ylabel('Efficiency Gain (%)', fontsize=12)
plt.axhline(efficiency_list[2], 0, 9, label= 'Efficiency boundary', color='blue', linestyle = 'dotted' )
ax2.yaxis.key.set_color('blue')

plt.xticks(np.arange(0, max(admisibility_factors_list)+1, 1))
plt.tight_layout()
plt.savefig('examples/figures/accuracy_efficiency_admissibility_testing.pdf', format='pdf')
plt.show()

# b) Analysis of consistency of parameter estimates in logit model with and without (endogenous) heuristic

def consistency_analysis_pastar_logit(n_replicates, G, n_paths, K_X, K_H):

    # Dictionary storing logit estimates
    thetas_G = {}
    thetas_logit_standard_G = {}
    thetas_logit_standard_G['travel_time'] = []
    thetas_logit_standard_G['cost'] = []
    thetas_endogenous_heuristic_G = {}
    thetas_endogenous_heuristic_G['travel_time'] = []
    thetas_endogenous_heuristic_G['cost'] = []
    observed_paths_replicates = {}

    # - Dictionary with matrix of attributes between every OD pair
    X_G = {attribute:nx.adjacency_matrix(G, weight = attribute).todense() for attribute in K_X}

    for i in range(n_replicates):

        observed_paths = path_generator(G=G, n_paths=n_paths, attribute='utility')

        observed_paths_replicates[i] = observed_paths

        # Matrix with choice sets generated at each node
        choice_sets_paths_G = {key: generate_choice_set_matrix_from_observed_path(G = G, observed_path = observed_path) for key,observed_path  in observed_paths.items()}

        # - Chosen edges (edge between expanded nodes)
        y_G = get_chosen_edges(paths = observed_paths)

        #Goals
        g_G = get_heuristic_goals(observed_paths)

        #Dictionary with matrix of heuristic attributes between every OD pair
        H_G = {}

        # Admissible travel time (normalizing by maximum speed across edges)
        H_G['h_g'] = get_goal_dependent_heuristic_attribute_matrix(get_euclidean_distances_between_nodes(G = G), observed_paths = observed_paths) # Euclidian (aerial) distance to goal

        #Logit estimates

        # - Standard Logit
        theta_logit_standard_G = recursive_logit_estimation(C= choice_sets_paths_G
                                                 , X = X_G, K_X = K_X, y = y_G
                                                 , H = [], K_H = [], g = [])

        thetas_logit_standard_G['travel_time'] += [theta_logit_standard_G['travel_time']]
        thetas_logit_standard_G['cost'] += [theta_logit_standard_G['cost']]

        # - With endogenous heuristic (standard logit)
        theta_endogenous_heuristic_G = recursive_logit_estimation(C= choice_sets_paths_G
                                                 , X = X_G, K_X = K_X, y = y_G
                                                 , H = H_G, K_H = K_H, g = g_G)

        thetas_endogenous_heuristic_G['travel_time'] += [theta_endogenous_heuristic_G['travel_time']]
        thetas_endogenous_heuristic_G['cost'] += [theta_endogenous_heuristic_G['cost']]


    thetas_G['logit_standard'] = thetas_logit_standard_G
    thetas_G['endogenous_heuristic'] = thetas_endogenous_heuristic_G

    return thetas_G, observed_paths_replicates

consistency_analysis = {}
observed_paths_replicates = {}

n_replicates = 100
n_paths = 40

consistency_analysis['training'], observed_paths_replicates['training'] \
    = consistency_analysis_pastar_logit(n_replicates = n_replicates, G = G_training, n_paths = n_paths
                                        , K_X = ['travel_time', 'cost'], K_H = ['h_g'])

consistency_analysis['testing'], observed_paths_replicates['testing'] \
    = consistency_analysis_pastar_logit(n_replicates = n_replicates, G = G_testing, n_paths = n_paths
                                        , K_X = ['travel_time', 'cost'], K_H = ['h_g'])


true_ratio_cost_time = theta_logit_true_G_training['travel_time'] / theta_logit_true_G_training['cost']

#Create pandas dataframe

import pandas as pd

consistency_analysis_df1 = pd.DataFrame({'cost': consistency_analysis['training']['endogenous_heuristic']['cost']
                 ,'travel_time': consistency_analysis['training']['endogenous_heuristic']['travel_time']
                      , 'model' : 'endogenous_heuristic'
                   })

consistency_analysis_df2 = pd.DataFrame({'cost': consistency_analysis['training']['logit_standard']['cost']
                 ,'travel_time': consistency_analysis['training']['logit_standard']['travel_time']
                    , 'model' : 'logit_standard'
                    })

consistency_analysis_df = consistency_analysis_df1.append(consistency_analysis_df2)
consistency_analysis_df['ratios_theta_time_cost'] = consistency_analysis_df['travel_time']/consistency_analysis_df['cost']

# Doing this analysis on training or testing is enough

#- Logit standard
ratios_theta_time_cost = consistency_analysis_df['ratios_theta_time_cost'][consistency_analysis_df.model == 'logit_standard']

# print(np.mean(ratio_theta_cost_time_list))
# print(theta_logit_true_G_training['travel_time']/theta_logit_true_G_training['cost'])

import seaborn as sns

fig, ax = plt.subplots()

sns.distplot(ratios_theta_time_cost, hist=True, kde=True
             # ,bins=int(7)
             , color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.axvline(np.mean(ratios_theta_time_cost), 0,1, color = 'blue', linestyle = 'dotted',label = 'Estimate' )
plt.axvline(true_ratio_cost_time, 0,1, color = 'red', linestyle = 'dotted',label = 'Truth' )
plt.legend(loc="upper right")

ax.set_xlabel(r'$\frac{\theta_t}{\theta_c}$', fontsize = 20)

plt.tight_layout()

plt.savefig('examples/figures/consistency_logit_standard_hist_training.pdf', format='pdf')

plt.show()

# - Endogenous heuristic
ratios_theta_time_cost = consistency_analysis_df['ratios_theta_time_cost'][consistency_analysis_df.model == 'endogenous_heuristic']
# print(np.mean(ratio_theta_cost_time_list))
# print(theta_logit_true_G_training['travel_time']/theta_logit_true_G_training['cost'])

fig, ax = plt.subplots()

sns.distplot(ratios_theta_time_cost, hist=True, kde=True
             ,bins=int(7)
             , color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.axvline(np.mean(ratios_theta_time_cost), 0,1, color = 'blue', linestyle = 'dotted',label = 'Estimate' )
plt.axvline(true_ratio_cost_time, 0,1, color = 'red', linestyle = 'dotted',label = 'Truth' )
plt.legend(loc="upper right")

ax.set_xlabel(r'$\frac{\theta_t}{\theta_c}$', fontsize = 20)

plt.tight_layout()
plt.savefig('examples/figures/consistency_endogenous_heuristic_hist_training.pdf', format='pdf')
plt.show()

#- Error bars
models = ['logit_standard', 'endogenous_heuristic']
mean_ratio_models = {}
sd_ratio_models = {}

ratios_theta_time_cost = {'logit_standard': consistency_analysis_df['ratios_theta_time_cost'][consistency_analysis_df.model == 'logit_standard']
    , 'endogenous_heuristic': consistency_analysis_df['ratios_theta_time_cost'][consistency_analysis_df.model == 'endogenous_heuristic']
                          }
for model in models:
    for accuracy_replicate in ratios_theta_time_cost:
        mean_ratio_models[model] =  np.mean(ratios_theta_time_cost[model])
        sd_ratio_models[model] = np.sqrt(np.var(ratios_theta_time_cost[model]))

x = ['truth'] + list(consistency_analysis['training'].keys()) # mean_acc_models.keys()
y =  [true_ratio_cost_time] + list(mean_ratio_models.values())
e = [0] + list(sd_ratio_models.values())

fig, ax = plt.subplots()

plt.errorbar(x, y, e, linestyle='None', marker='o')
plt.tight_layout()
plt.savefig('examples/figures/consistency_models_errorbars_training.pdf', format='pdf')

plt.show()

# c) Accuracy (edges and path) of standard logit, endogenous heuristic and exogeneous heuristic

predictions_paths_replicates = {} # Training, testing

# i) Training Data

# Make predictions
def predictions_comparison_pastar(observed_paths_replicates, G
                                     , theta_logit_standard, theta_logit_X_heuristic, theta_logit_H_heuristic):

    predictions_replicates = {}
    prediction_logit_standard_replicate = {}
    prediction_logit_endogenous_heuristic_replicate = {}
    prediction_logit_exogenous_heuristic_replicate = {}

    accuracy = {}

    for i,observed_paths_replicate in observed_paths_replicates.items():

        prediction_logit_standard_replicate[i] = pastar_path_predictions(G = G, observed_paths = observed_paths_replicate
                                                                     , theta_X = theta_logit_standard
                                                                    , theta_H= []
                                                                    , H= [])['predicted_path']

        H = {}
        H['h_g'] = get_goal_dependent_heuristic_attribute_matrix(
            get_euclidean_distances_between_nodes(G=G),
            observed_paths=observed_paths_replicate)  # Euclidian (aerial) distance to goal

        prediction_logit_endogenous_heuristic_replicate[i] = pastar_path_predictions(G = G, observed_paths = observed_paths_replicate
                                                                      , theta_X = theta_logit_X_heuristic
                                                                      , theta_H = theta_logit_H_heuristic
                                                                      , H = H
                                                                      , endogenous_heuristic=True
                                                                            )['predicted_path']

        admissible_aerial_heuristic = {'h_g':theta_logit_standard['travel_time']/(1000*20/60)}

        prediction_logit_exogenous_heuristic_replicate[i] = pastar_path_predictions(G = G, observed_paths = observed_paths_replicate
                                                                         , theta_X = theta_logit_standard
                                                                         , theta_H = {'h_g':0*admissible_aerial_heuristic['h_g']} #Admissible but less efficient
                                                                         , H = H)['predicted_path']

    predictions_replicates['logit_standard'] = prediction_logit_standard_replicate
    predictions_replicates['endogenous_heuristic'] = prediction_logit_endogenous_heuristic_replicate
    predictions_replicates['exogenous_heuristic'] = prediction_logit_exogenous_heuristic_replicate

    return predictions_replicates, observed_paths_replicates

predictions_paths_replicates['training'], observed_paths_replicates['training'] \
    = predictions_comparison_pastar(observed_paths_replicates = observed_paths_replicates['training']
                                 , G = G_training
                               , theta_logit_standard = theta_logit_standard_G_training
                               , theta_logit_X_heuristic = theta_logit_X_heuristic_G_training
                               , theta_logit_H_heuristic = theta_logit_H_heuristic_G_training
                               )

predictions_paths_replicates['testing'], observed_paths_replicates['testing'] \
    = predictions_comparison_pastar(observed_paths_replicates = observed_paths_replicates['testing']
                                 , G = G_testing
                               , theta_logit_standard = theta_logit_standard_G_training
                               , theta_logit_X_heuristic = theta_logit_X_heuristic_G_training
                               , theta_logit_H_heuristic = theta_logit_H_heuristic_G_training
                               )


## Obtain accuracies
accuracy_predictions_replicates = {}
models = ['logit_standard', 'endogenous_heuristic', 'exogenous_heuristic']

for model in models:
    accuracy_predictions_replicates_model = {}
    accuracy_predictions_replicates_model_edges = {}
    accuracy_predictions_replicates_model_paths = {}

    for i in observed_paths_replicates['training'].keys():
        accuracy_predictions_replicates_model[i] = accuracy_pastar_predictions(G = G_training
                                      , predicted_paths = predictions_paths_replicates['training'][model][i]
                                      , observed_paths= observed_paths_replicates['training'][i])

        accuracy_predictions_replicates_model_edges[i] = accuracy_predictions_replicates_model[i]['acc_edges']
        accuracy_predictions_replicates_model_paths[i] = accuracy_predictions_replicates_model[i]['acc_paths']

    accuracy_predictions_replicates[model] = {'acc_edges': accuracy_predictions_replicates_model_edges
        , 'acc_paths': accuracy_predictions_replicates_model_paths }

# Panda dataframe
df1 = pd.DataFrame({'data' : 'training', 'model': 'logit_standard', 'acc_paths': accuracy_predictions_replicates['logit_standard']['acc_paths'],
    'acc_edges': accuracy_predictions_replicates['logit_standard']['acc_edges']})

df2 = pd.DataFrame({'data' : 'training', 'model': 'endogenous_heuristic', 'acc_paths': accuracy_predictions_replicates['endogenous_heuristic']['acc_paths'],
    'acc_edges': accuracy_predictions_replicates['endogenous_heuristic']['acc_edges']})

df3 = pd.DataFrame({'data' : 'training', 'model': 'exogenous_heuristic', 'acc_paths': accuracy_predictions_replicates['exogenous_heuristic']['acc_paths'],
    'acc_edges': accuracy_predictions_replicates['exogenous_heuristic']['acc_edges']})

df_accuracy_training = df1.append(df2).append(df3)

# Plot mean and standard deviation of accuracy by model in paths and edges
x = models # mean_acc_models.keys()
y = np.array(df_accuracy_training.groupby(['model'])['acc_paths'].mean()[models])
e = np.array(df_accuracy_training.groupby(['model'])['acc_paths'].std()[models])

plt.errorbar(x, y, e, linestyle='None', marker='o')

plt.savefig('examples/figures/accuracy_models_errorbars_training.pdf', format='pdf')

plt.show()

# ii) Testing Data

predictions_paths_replicates['testing'], observed_paths_replicates['testing'] \
    = predictions_comparison_pastar(observed_paths_replicates = observed_paths_replicates['testing']
                                 , G = G_testing
                               , theta_logit_standard = theta_logit_standard_G_training
                               , theta_logit_X_heuristic = theta_logit_X_heuristic_G_training
                               , theta_logit_H_heuristic = theta_logit_H_heuristic_G_training
                               )
## Obtain accuracies
accuracy_predictions_replicates = {}
models = ['logit_standard', 'endogenous_heuristic', 'exogenous_heuristic']

for model in models:
    accuracy_predictions_replicates_model = {}
    accuracy_predictions_replicates_model_edges = {}
    accuracy_predictions_replicates_model_paths = {}

    for i in observed_paths_replicates['testing'].keys():
        accuracy_predictions_replicates_model[i] = accuracy_pastar_predictions(G = G_testing
                                      , predicted_paths = predictions_paths_replicates['testing'][model][i]
                                      , observed_paths= observed_paths_replicates['testing'][i])

        accuracy_predictions_replicates_model_edges[i] = accuracy_predictions_replicates_model[i]['acc_edges']
        accuracy_predictions_replicates_model_paths[i] = accuracy_predictions_replicates_model[i]['acc_paths']

    accuracy_predictions_replicates[model] = {'acc_edges': accuracy_predictions_replicates_model_edges
        , 'acc_paths': accuracy_predictions_replicates_model_paths }

# Panda dataframe
df1 = pd.DataFrame({'data' : 'testing', 'model': 'logit_standard', 'acc_paths': accuracy_predictions_replicates['logit_standard']['acc_paths'],
    'acc_edges': accuracy_predictions_replicates['logit_standard']['acc_edges']})

df2 = pd.DataFrame({'data' : 'testing', 'model': 'endogenous_heuristic', 'acc_paths': accuracy_predictions_replicates['endogenous_heuristic']['acc_paths'],
    'acc_edges': accuracy_predictions_replicates['endogenous_heuristic']['acc_edges']})

df3 = pd.DataFrame({'data' : 'testing', 'model': 'exogenous_heuristic', 'acc_paths': accuracy_predictions_replicates['exogenous_heuristic']['acc_paths'],
    'acc_edges': accuracy_predictions_replicates['exogenous_heuristic']['acc_edges']})

df_accuracy_testing = df1.append(df2).append(df3)

# Plots Mean and standard deviation of accuracy by model in paths and edges
x = models # mean_acc_models.keys()
y = np.array(df_accuracy_testing.groupby(['model'])['acc_paths'].mean()[models])
e = np.array(df_accuracy_testing.groupby(['model'])['acc_paths'].std()[models])

plt.errorbar(x, y, e, linestyle='None', marker='o')

plt.savefig('examples/figures/accuracy_models_errorbars_testing.pdf', format='pdf')

plt.show()


# iii) Training and testing data

df_accuracy_models = df_accuracy_training.append(df_accuracy_testing)

df_accuracy_models = df_accuracy_training[['model','acc_paths']].rename(columns = {'acc_paths':'training'})
df_accuracy_models['testing'] = df_accuracy_testing[['acc_paths']]
df_accuracy_models = df_accuracy_models.set_index('model')

gp = df_accuracy_models.groupby(level=('model'))
means = gp.mean().reindex(models)
errors = gp.std().reindex(models)

models_labels = ['Recursive \n Logit', 'Endogenous \n Heuristic ', 'Exogenous \n Heuristic ']

fig, ax = plt.subplots()

plt.locator_params(axis='x', nbins=4)

x_scatter = 0.1

plt.errorbar(
    np.arange(3)-x_scatter, means['training'], yerr=errors['training'], fmt='o',
    label="Training", color = 'red')

# Add some some random scatter in x
plt.errorbar(
    np.arange(3) + x_scatter, means['testing'], yerr=errors['testing'], fmt='o', label='Testing', color = 'blue')

ax.set_xticklabels(['']+models_labels)
plt.xticks(fontsize=12)
ax.legend(loc="lower right")

plt.ylabel('Accuracy (%)', fontsize=12)

plt.savefig('examples/figures/accuracy_models_errorbars_testing_and_training.pdf', format='pdf')
plt.show()

# #Visualize network
# show_multiDiNetwork(G_training)
# show_multiDiNetwork(G_testing)

#
# # =============================================================================
# # Single path analysis of heuristic cost
# # =============================================================================

# TODO: needs to be implemented properly. Add as penalties
#  in objective function

# # - Heuristic cost at edge level:
#
# G = heuristic_bounds(G=G, observed_path=observed_paths[0])
# G = set_heuristic_costs_nodes(G)
# G = set_heuristic_costs_edges(G)
#
# # # Observed path (real observation)
# # print((observed_paths[0], path_length(G, path=observed_paths[0], attribute='distance')))
#
# # nx.get_node_attributes(G, 'f_bound_neighbor')
# # nx.get_node_attributes(G, 'h_bound_neighbor')
# # nx.get_node_attributes(G, 'h_bound_optimal')

# #TODO: Generalize heuristic assignment when having multiple paths.
#
# # Pick first random path
# print((observed_paths[0], path_length(G_training, path=observed_paths[0], attribute='distance')))
#
# # Path predicted with dijsktra (h = 0)
# dijkstra_path_prediction = nx.astar_path(G = G_training
#                            ,source = observed_paths[0][0], target = observed_paths[0][-1]
#                            ,weight = 'distance')
#
# print((dijkstra_path_prediction,path_length(G_training, dijkstra_path_prediction,attribute = 'distance')))
#
# # Path predicted with A*star and the computed heuristic costs
# astar_heuristic_path_prediction = astar_path_heuristic_nodes(G = G_training,heuristic_costs= nx.get_node_attributes(G_training, 'h')
#                            ,source = observed_paths[0][0], target = observed_paths[0][-1]
#                            ,weight = 'distance')
#
# print((astar_heuristic_path_prediction,path_length(G_training, astar_heuristic_path_prediction, attribute = 'distance')))
