# =============================================================================
# 1) SETUP
# =============================================================================
import os

# Set seed for reproducibility and consistency between experiments
import numpy as np
import random

np.random.seed(2021)
random.seed(2021)

#=============================================================================
# 1a) MODULES
#==============================================================================

# Internal modules
import transportAI as tai

# External modules

# import transportAI.modeller

import sys
import pandas as pd
from sortedcontainers import SortedSet
import os
import copy
import time
# import datetime

from scipy import stats

import matplotlib
from matplotlib import rc
# matplotlib.rcParams['text.usetex'] = False
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

import matplotlib.pyplot as plt

import seaborn as sns

# Memory usage
import tracemalloc
# https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python



# =============================================================================
# 1b) CONFIGURATION
# =============================================================================

#Note: The strategy does not work with 'Eastern-Massachusetts' under an uncongested network. Do not why

config = tai.config.Config(network_key = 'N3')
# config = tai.config.Config(network_key = 'Yang')
# config = tai.config.Config(network_key = 'Wang')
# config = tai.config.Config(network_key = 'LoChan')


config.sim_options['prop_validation_sample'] = 0
config.sim_options['regularization'] = False

# No scaling is performed to compute equilibrium as if normalizing by std, otherwise the solution change significantly.
config.estimation_options['standardization_regularized'] = {'mean': True, 'sd': True}
config.estimation_options['standardization_norefined'] = {'mean': False, 'sd': False} #Standardization by std, change results a lot.
config.estimation_options['standardization_refined'] = {'mean': False, 'sd': False}
# * It seems scaling helps to speed up convergence

# Features in utility function
k_Y = ['tt']
k_Z = config.estimation_options['k_Z']
# config.estimation_options['k_Z'] = ['wt']
# config.estimation_options['k_Z'] = ['n2,n1']

# If any of the two following is set to be equal to none, then  k_Z will be used later and thus, it includes all features
k_Z_simulation = None #config.estimation_options['k_Z'] #
k_Z_estimation = None #k_Z #config.estimation_options['k_Z']

# Fixed effect by link, nodes or OD zone
config.sim_options['fixed_effects'] = {'Q': False, 'nodes': False, 'links': True, 'links_observed': True, 'coverage': 0}
theta_true_fixed_effects = 1e3
theta_0_fixed_effects = 0
observed_links_fixed_effects = None #'random'

# Feature selection based on t-test from no refined step and ignoring fixed effects (I must NOT do post-selection inference as it violates basic assumptions)
config.estimation_options['ttest_selection_norefined'] = False #True

# We relax the critical value to remove features that are highly "no significant". A simil of regularization
config.estimation_options['alpha_selection_norefined'] = 3 #if it g is higher than 1, it choose the k minimum values

# Computation of t-test with top percentage of observations in terms of SSE
config.estimation_options['pct_lowest_sse_norefined'] = 100
config.estimation_options['pct_lowest_sse_refined'] = 100

# synthetic counts
config.sim_options['n_paths_synthetic_counts'] = 2 #None #3  # If none, the

# initial path set is used to generate counts
config.sim_options['sue_iters'] = 40 #itereates to generate synthetic counts

# Number of paths in the initial path set
config.estimation_options['n_initial_paths'] = 6

# Coverage of OD pairs to sample new paths
config.estimation_options['ods_coverage_column_generation'] = 0.5
config.estimation_options['n_paths_column_generation'] = 5 #2

# Number of path selected after column generation
config.estimation_options['k_path_set_selection'] = 2

# TODO: Analyze the impact of this parameter on recovery. If it equals 1, the performance is bad by construction
config.estimation_options['dissimilarity_weight'] = 0.5#0.5 works well


# accuracy for relative gap
config.estimation_options['accuracy_eq'] = 1e-4

# Optimization methods used in no refined and refined stages
# config.estimation_options['outeropt_method_norefined'] = 'gauss-newton' #'adam'
# config.estimation_options['outeropt_method_refined'] = 'ngd' #'lm' #gauss-newton
config.estimation_options['outeropt_method_norefined'] = 'ngd' #'adam'
config.estimation_options['outeropt_method_refined'] = 'gauss-newton' #'lm' #gauss-newton
# Note: lm works bad for inference, understand why but it may related with scaling?

# Size of batch for paths used to compute gradient (the improvement in speed is impressive)
config.estimation_options['paths_batch_size'] = 0 # > 1000 to get descent results
config.estimation_options['links_batch_size'] = 0

# Learning rate for first order optimization
config.estimation_options['eta_norefined'] = 1e-1
config.estimation_options['eta_refined'] = 1e-2


# Parameters for simulation with no noise
# config.set_simulated_counts(max_link_coverage= 1, sd_x = 0, sd_Q = 0)

# Std of 0.2 is tolerable in terms of consistency of statistical inference
# config.set_simulated_counts(max_link_coverage = 1, sd_x = 0.4, sd_Q = 0.5, scale_Q =1)
# config.set_simulated_counts(max_link_coverage = 1, sd_x = 0.1, sd_Q = 0.1, scale_Q =1)

# Leave coverage in 0.3 so around 5 links are considered for estimation
config.set_simulated_counts(max_link_coverage = 1, sd_x = 0, sd_Q = 0, scale_Q =1)

# For yang network
# config.set_simulated_counts(max_link_coverage = 0.2, snr_x = None, sd_x = 0, sd_Q = 0, scale_Q =1)

# Uncongested mode
# config.set_uncongested_mode(True)
config.set_uncongested_mode(False)

# Under this mode, the true path is used as the path set to learn the logit parameters
config.set_known_pathset_mode(True)
#Note: when the path set is unknown the parameter for travel time tends to be small, so the bias is in one direction on.y.

# Note: Column generation is a good cure against noise is to perform column generation. But the greater improvement comes
# from computing eequilibrium properly

# Out of sample prediction mode
# config.set_outofsample_prediction_mode(theta = {'tt': -2, 'wt': -3, 'c': -7}, outofsample_prediction = True, mean_count = 100)

# Experiments

# - Monotonocity
# config.set_monotonicity_experiment(theta_grid = np.arange(-15, 15, 0.1), uncongested_mode = False)

# - Pseudo-convexity
# config.set_pseudoconvexity_experiment(theta_grid = np.arange(-15, 15, 0.1), uncongested_mode = False)

# - Inductive bias
config.set_inductive_bias_experiment(uncongested_mode = False)

# - Inference
# config.set_inference_experiment(uncongested_mode = False, sd_x = 0.03)

# - Distorted OD matrix
# config.set_od_bias_yang_experiment(uncongested_mode = False)


# Random initilization of theta parameter and performed before scaling Q matrix

# config.estimation_options['theta_search'] = 'random' # Do not use boolean, options are 'grid','None', 'random'
# config.estimation_options['q_random_search'] = True # Include od demand matrix factor variation for random search
# config.estimation_options['n_draws_random_search'] = 20 # To avoid a wrong scaling factor, at least 20 random draws needs to be performed
# config.estimation_options['scaling_Q'] = True #True

# - Number of attributes that will be set to 0, which moderate sparsity: with 20 at least, we observe benefits of regularize
config.sim_options['n_R'] = 0  # 2 #5 #10 #20 #50

#Labels of sparse attributes
config.sim_options['R_labels'] = ['k' + str(i) for i in np.arange(0, config.sim_options['n_R'])]

# # Initial theta for optimization
# config.theta_0['tt'] = 0

# Key internal objects for analysis and visualization
artist = tai.visualization.Artist(folder_plots = config.plots_options['folder_plots'], dim_subplots=config.plots_options['dim_subplots'])

# New options added for the custom case
config.sim_options['q_sparsity'] = 0

# # Set theta true and exogenous attributes
config.theta_true['Z'] = {'c': config.theta_true['Z']['c']}
k_Z = ['c']
config.estimation_options['k_Z'] = k_Z
k_Z_simulation = k_Z

config.theta_true['Z'] = {}
k_Z = []
config.estimation_options['k_Z'] = k_Z
k_Z_simulation = k_Z

# =============================================================================
# 1c) LOG-FILE
# =============================================================================

# Record starting date and time of the simulation
# https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python

if config.experiment_options['experiment_mode'] is None:
    config.set_log_file(networkname = config.sim_options['current_network'].lower())

# =============================================================================
# 2) NETWORKS FACTORY
# =============================================================================
# Dictionary with training and testing networks
N = {}
N['train'] = {}

# Label of networks selected (to avoid commeting out everytime)
current_network = config.sim_options['current_network']

# =============================================================================
# 2.1) CREATION OF TOY NETWORKS
# =============================================================================

# i) CUSTOM NETWORK CREATION

A = config.sim_options['custom_networks']['A']
Q = config.sim_options['custom_networks']['Q']

N_labels = {i:'Train ' + 'N' + str(j) for i,j in zip(A.keys(), np.arange(1, config.sim_options['n_custom_networks'] + 1))}

#Add label for Yang, Wang and Lo&Chan networks
N_labels['Yang'] = 'Train Yang'
N_labels['Yang2'] = 'Train Yang2'
N_labels['LoChan'] = 'Train LoChan'
N_labels['Wang'] = 'Train Wang'

# Attribute values are defined at random though yet
N_custom = tai.modeller.tnetwork_factory(A = A, labels = N_labels
                                         , factory_options= config.sim_options.copy()
                                         )

N['train'] = {**N_custom}
#
# i = 'N4'
for i in [*list(N_custom.keys())]:
    N['train'][i] = tai.modeller.setup_network(Nt=N['train'][i], Q = Q[i]
                                               , setup_options = {**config.sim_options,**config.estimation_options}
                                               , writing=dict(config.sim_options['writing'],
                                                                  **{'paths': True, 'Q': True, 'C': True, 'M': True, 'D': True})
                                               # , generation = dict(config.sim_options, generation = {'Q': False, 'bpr': True, 'Z': True})
                                               , generation=dict(config.sim_options['generation'],
                                                                 **{'paths': True, 'bpr': True, 'Z': True,
                                                                    'C': True, 'D': True, 'M': True, 'Q': False})
                                               )

# ii) RANDOM NETWORK CREATION

# n_custom_networks = len(N['train'])
N_keys = ['N' + str(i) for i in list(config.sim_options['n_custom_networks'] + np.arange(1, config.sim_options['n_random_networks'] + 1))]
# N_keys = ['RN' + str(i) for i in list(n_custom_networks + np.arange(1,n_random_networks+1))]
N_labels = {i:'Train ' + str(i) for i in N_keys}

N_random = {i:tai.modeller.random_network_factory(N_keys = [i]
                                                  , labels = {i:N_labels[i]}
                                                  , factory_options = config.sim_options.copy()
                                                  )[i] for i,j in zip(N_keys, config.sim_options['nodes_range'])}



N['train'] = {**N_custom,**N_random}

for i in [*list(N_random.keys())]:
    N['train'][i] = tai.modeller.setup_network(Nt=N['train'][i]
                                               , setup_options = {**config.sim_options,**config.estimation_options}
                                               # , generation = dict(config.sim_options, generation = {'Q': False, 'bpr': True, 'Z': True})
                                               # , writing=dict(config.sim_options['writing'],
                                               #                **{'paths': True, 'Q': True, 'C': True, 'M': True,
                                               #                   'D': True})
                                               , generation=dict(config.sim_options['generation'],
                                                                 **{'paths': True, 'bpr': True, 'Z': True,
                                                                    'C': True, 'D': True, 'M': True, 'Q': True})
                                               )

# N['train']['N4'].D
# N['train']['N4'].A

# v) Network 5
# Select subset
# N['train'] = {('N' + str(i)):N['train']['N' + str(i)] for i in np.arange(1,5)}



# =============================================================================
# 1.8) VISUALIZATION
# =============================================================================
plot = tai.Artist(folder_plots = config.plots_options['folder_plots'], dim_subplots= config.plots_options['dim_subplots'])

# plot.plot_custom_networks(N = {i:N['train'][i] for i in ['N1','N2','N3','N4']},
#                           show_edge_labels = True, subfolder = "networks", filename = "custom_networks")

# Plot yang network
# plot.draw_MultiDiNetwork(G=N['train']['Yang'].G, nodes_pos={0: (0, 1), 1: (1, 1), 2: (2,1)
#                                                     , 3: (0, 0), 4: (1, 0), 5: (2,0)
#                                                     , 6: (0, -1), 7: (1, -1), 8: (2,-1)
#                                                     }, show_edge_labels=False)
#
# plt.title('Yang')
# plt.show()

# plot.draw_MultiDiNetwork(G=N['train']['Lo&Chan'].G, nodes_pos={0: (0, 1), 1: (1, 1), 2: (2,1)
#                                                     , 3: (0, 0), 4: (1, 0), 5: (2,0)
#                                                     }, show_edge_labels=False)
#
# plt.title('Lo and Chan')
# plt.show()
#
# plot.draw_MultiDiNetwork(G=N['train']['Wang'].G, nodes_pos={0: (0, 1), 1: (1, 1), 2: (1,0)
#                                                     , 3: (0, 0)
#                                                     }, show_edge_labels=False)
#
# plt.title('Wang')
# plt.show()



# plot.draw_MultiDiNetwork(G = N['train']['N6'].G, show_edge_labels=False)
# plt.show()

# # Plot real network
# plot.draw_MultiDiNetwork(G = N['train'][subfolder].G, node_size=100, font_size=8, edge_curvature= 0.1, show_edge_labels= False)
# plt.show()
#
# Plot all networks together
# plot.plot_all_networks(N = N['train'], show_edge_labels = False, subfolder = "networks", filename = "all_networks")



# =============================================================================
# c) BEHAVIORAL PARAMETERS AND UTILITY FUNCTIONS
# =============================================================================

# Initialize theta vector with travellers preferences for every attribute
# theta_true_Z = dict(zip(Z_lb, np.zeros(len(Z_lb))))
for k in config.sim_options['R_labels']:
    if k not in config.theta_true['Z'].keys():
        config.theta_true['Z'][k] = 0 #-0.001

        # Update theta_0 vector
        config.theta_0[k] = 0

theta_true = {}
for i in N['train'].keys():
    theta_true[i] = {**config.theta_true['Y'], **config.theta_true['Z']}

# TODO: Normalize preference vector to 1 for analysis purposes:
# for i in N['train'].keys():
#     theta_true[i] =     # {**theta_true_Y, **theta_true_Z}

# theta_Z = {k: v for k,v in theta.items() if k != 'tt'}

# Labels in sparse vector are added in list of labels of exogenous attributes
config.estimation_options['k_Z'] = [*config.estimation_options['k_Z'], *config.sim_options['R_labels']]

# Fixed effects parameters (dependent on the network)

if observed_links_fixed_effects is not None and config.sim_options['fixed_effects']['coverage']>0:

    # Create fixed effect for every link

    for i in N['train'].keys():
        Z_lb = list(N['train'][i].Z_dict.keys())  # Extract column names in the network
        for k in N['train'][i].k_fixed_effects:
            theta_true[i][k] = theta_true_fixed_effects # -float(np.random.uniform(1,2,1))
            config.theta_0[k] = theta_0_fixed_effects
            k_Z = [*config.estimation_options['k_Z'], k]



    if len(N['train'][current_network].k_fixed_effects) > 0:
        print('\nFixed effects created:', N['train'][current_network].k_fixed_effects)



# # Store path utilities associated to exogenous attributes
# for i in N['train'].keys():
#     N['train'][i].set_V_Z(paths = N['train'][i].paths, theta = theta_true)





# =============================================================================
# 1.7) SUMMARY OF NETWORKS CHARACTERISTICS
# =============================================================================
networks_table = {'nodes':[],'links':[], 'paths': [], 'ods': []}

# Network description
for i in N['train'].keys():
    networks_table['ods'].append(len(tai.networks.denseQ(N['train'][i].Q, remove_zeros=True)))
    networks_table['nodes'].append(N['train'][i].A.shape[0])
    networks_table['links'].append(len(N['train'][i].links))
    networks_table['paths'].append(len(N['train'][i].paths))

networks_df = pd.DataFrame()
networks_df['N'] = np.array(list(N['train'].keys()))

for var in ['nodes','links', 'ods','paths']:
    networks_df[var] = networks_table[var]

# Print Latex Table
print(networks_df.to_latex(index=False))

# =============================================================================
# 1.7) UNCONGESTED/CONGESTED MODE (MUST BE BEFORE SYNTHETIC COUNTS)
# =============================================================================

if config.sim_options['uncongested_mode'] is True:

    print('Algorithm is performed assuming an uncongested network')

    # TODO: implement good copy and clone methods so the code does not need to be run from scratch everytime the bpr functions are set to zero
    # To account for the uncongested case, the BPR parameters are set to be equal to zero
    for link in N['train'][current_network].links:
        link.bpr.alpha = 0
        link.bpr.beta = 0

    # for link in N['train'][i].links:
    #     print(link.bpr.alpha)
    #     # print(link.bpr.beta)

else:
    print('Algorithm is performed assuming a congested network')

    # Given that it is very expensive to compute path probabitilies and the resuls are good already, it seems fine to perform only one iteration for outer problem
    # iters_est = config.sim_options['ngd_iters']  #5

# =============================================================================
# 1.8) SYNTHETIC COUNTS
# =============================================================================

if config.sim_options['simulated_counts'] is True:

    if k_Z_simulation is None:
        k_Z_simulation = config.estimation_options['k_Z']

    # Generate synthetic traffic counts
    xc_simulated, xc_withdraw = tai.estimation.generate_link_counts_equilibrium(
        Nt=N['train'][current_network]  # tai.modeller.clone_network(N['train'][i], label = 'clone')
        , theta = theta_true[current_network]
        , k_Y = k_Y, k_Z = k_Z_simulation
        , eq_params = {'iters': config.sim_options['max_sue_iters'], 'accuracy_eq': config.estimation_options['accuracy_eq'], 'method': 'line_search', 'iters_ls': 50}
        , coverage = config.sim_options['max_link_coverage']
        , noise_params = config.sim_options['noise_params']
        , n_paths = config.sim_options['n_paths_synthetic_counts']
        , uncongested_mode=config.sim_options['uncongested_mode']
    )

    N['train'][current_network].reset_link_counts()
    N['train'][current_network].store_link_counts(xct=xc_simulated)

    xc = xc_simulated

    print('Synthetic observed links counts:')

    dict_observed_link_counts = {link_key: np.round(count,1) for link_key, count in xc_simulated.items() if not np.isnan(count)}

    print(pd.DataFrame({'link_key': dict_observed_link_counts.keys(), 'counts': dict_observed_link_counts.values()}).to_string())



# =============================================================================
# 3) DATA DESCRIPTION AND CURATION
# =============================================================================

# =============================================================================
# 3a) LINK COUNTS AND TRAVERSING PATHS
# =============================================================================

#Initial coverage
x_bar = np.array(list(xc.values()))[:, np.newaxis]

# If no path are traversing some link observations, they are set to nan values
xc = tai.estimation.masked_link_counts_after_path_coverage(N['train'][current_network], xct = xc)

x_bar_remasked = np.array(list(xc.values()))[:, np.newaxis]

true_coverage = np.count_nonzero(~np.isnan(x_bar_remasked))/x_bar.shape[0]

# After accounting for path coverage (not all links may be traversed)

total_true_counts_observations = np.count_nonzero(~np.isnan(np.array(list(xc.values()))))

print('Adjusted link coverage:', "{:.1%}". format(true_coverage))
print('Adjusted total link observations: ' + str(total_true_counts_observations))

# print('dif in coverage', np.count_nonzero(~np.isnan(x_bar))-np.count_nonzero(~np.isnan( x_bar_remasked)))

# =============================================================================
# 3b) SUMMARY STATS FOR TRAFFIC COUNTS
# =============================================================================

# Report link count information

total_counts_observations = np.count_nonzero(~np.isnan(np.array(list(xc.values()))))

# if config.sim_options['prop_validation_sample'] > 0:
#     total_counts_observations +=  np.count_nonzero(~np.isnan(np.array(list(xc_validation.values()))))
#
# total_links = np.array(list(xc.values())).shape[0]
#
# print('\nlink counts observations: ' + str(total_counts_observations))
# print('link coverage: ' + "{:.1%}". format(round(total_counts_observations/total_links,4)))

if config.sim_options['simulated_counts'] is True:
 print('Std set to simulate link observations: ' + str(config.sim_options['noise_params']['sd_x']))


# =============================================================================
# 3c) DESCRIPTIVE STATISTICS
# =============================================================================
summary_table_links_df = tai.descriptive_statistics.summary_table_links(links = N['train'][current_network].get_observed_links()
                                                                        , Z_attrs = ['wt', 'tt', 'c']
                                                                        # , Z_labels = ['incidents', 'income [1K USD]', 'high_inc', 'speed_avg [mi/hr]', 'tt_sd', 'tt_sd_adj', 'tt_var','stops', 'ints']
                                                                        )

with pd.option_context('display.float_format', '{:0.1f}'.format):
    print(summary_table_links_df.to_string())


# Write log file
tai.writer.write_csv_to_log_folder(df = summary_table_links_df, filename = 'summary_table_links_df'
                          , log_file = config.log_file)

# =============================================================================
# 4) EXPERIMENTS
# =============================================================================


# =============================================================================
# 4a) ANALYSIS OF MONOTONOCITY
# ==============================================================================

if config.experiment_options['monotonicity_experiment']:

    config.experiment_options['theta_0'] = config.theta_0
    config.experiment_options['theta_true'] = theta_true[current_network]

    config.theta_true['Z'] = {}
    k_Z = []
    config.estimation_options['k_Z'] = []
    k_Z_simulation = config.estimation_options['k_Z']

    config.set_experiments_log_files(networkname='small-networks')


    networks = ['N3','Wang','LoChan', 'Yang']

    for network in networks:

        # config.set_log_file(networkname=network.lower())



        # Generate synthetic traffic counts
        if k_Z_simulation is None:
            k_Z_simulation = config.estimation_options['k_Z']

        # Generate synthetic traffic counts
        xc_simulated, xc_withdraw = tai.estimation.generate_link_counts_equilibrium(
            Nt=N['train'][current_network]  # tai.modeller.clone_network(N['train'][i], label = 'clone')
            , theta=theta_true[current_network]
            , k_Y=k_Y, k_Z=k_Z_simulation
            , eq_params={'iters': config.sim_options['max_sue_iters'],
                         'accuracy_eq': config.estimation_options['accuracy_eq'], 'method': 'line_search', 'iters_ls': 50}
            , uncongested_mode=config.experiment_options['uncongested_mode']
            , coverage=config.sim_options['max_link_coverage']
            , noise_params=config.sim_options['noise_params']
            , n_paths=config.sim_options['n_paths_synthetic_counts']
        )

        N['train'][current_network].reset_link_counts()
        N['train'][current_network].store_link_counts(xct=xc_simulated)

        xc = xc_simulated

        traffic_count_links_df \
            = tai.estimation.monotonocity_traffic_count_functions(Nt=N['train'][network]
                                                      , k_Y=k_Y, k_Z=config.estimation_options['k_Z']
                                                      , x_bar=np.array(list(xc_simulated.values()))[:, np.newaxis]
                                                      , theta_attr_grid= config.experiment_options['theta_grid']
                                                      , attr_label='tt'
                                                      , theta=theta_true[network]
                                                      # , paths_column_generation=config.experiment_options['paths_column_generation']
                                                      , inneropt_params={'iters': config.sim_options['max_sue_iters'], 'accuracy_eq': config.experiment_options['accuracy_eq'], 'uncongested_mode': config.experiment_options['uncongested_mode']
                                                                         }

                                                      )



        plot1 = tai.Artist(folder_plots=config.log_file['folderpath'])


        # Select a random number of links
        n_links = 4
        unique_links = list(SortedSet(list(traffic_count_links_df.link)))
        idx_links = unique_links

        if len(unique_links) > n_links:
            idx_n_links = sorted(random.sample(list(np.arange(0,len(unique_links))),n_links))
            idx_links = [unique_links[i] for i in idx_n_links]


        traffic_count_links_subset = traffic_count_links_df[traffic_count_links_df['link'].isin(idx_links)]

        fig = plot1.monotonocity_traffic_count_functions(filename='monotonocity_' + network
                                    , subfolder=""
                                    , x_bar = np.array(list(xc_simulated.values()))[:, np.newaxis]
                                    , traffic_count_links_df =traffic_count_links_subset)

        # tai.writer.write_figure_to_log_folder(fig=fig
        #                                      , filename='monotonocity_traffic_count_functions.pdf',
        #                                      log_file=config.log_file)

        # Summary report
        tai.writer.write_experiment_options_report(filename = 'experiment_options_' + network, config = config)

    sys.exit()



# =============================================================================
# 4b) ANALYSIS OF QUASICONVEXITY
# ==============================================================================

# TODO: this can be speed up but allowing the gradient calculation be performed in a subset of the attribute
#  , or in particular, only the one for grid search

if config.experiment_options['pseudoconvexity_experiment']:

    config.theta_true['Z'] = {}
    k_Z = []
    config.estimation_options['k_Z'] = []
    k_Z_simulation = config.estimation_options['k_Z']

    networks = ['N3','Wang','LoChan', 'Yang']

    f_vals, grad_f_vals, hessian_f_vals = {}, {}, {}

    config.set_experiments_log_files(networkname='small-networks')

    for network in networks:

        # Generate synthetic traffic counts
        if k_Z_simulation is None:
            k_Z_simulation = config.estimation_options['k_Z']

        # Generate synthetic traffic counts
        xc_simulated, xc_withdraw = tai.estimation.generate_link_counts_equilibrium(
            Nt=N['train'][network]  # tai.modeller.clone_network(N['train'][i], label = 'clone')
            , theta=theta_true[network]
            , k_Y=k_Y, k_Z=k_Z_simulation
            , eq_params={'iters': config.experiment_options['max_sue_iters'],
                         'accuracy_eq': config.experiment_options['accuracy_eq'], 'method': 'line_search',
                         'iters_ls': 50}
            , uncongested_mode=config.experiment_options['uncongested_mode']
            , coverage=config.sim_options['max_link_coverage']
            , noise_params=config.sim_options['noise_params']
            , n_paths=config.sim_options['n_paths_synthetic_counts']
        )

        N['train'][network].reset_link_counts()
        N['train'][network].store_link_counts(xct=xc_simulated)

        xc = xc_simulated

        # N['train'][network].links[0].bpr.alpha
        # N['train'][network].links[0].bpr.k
        # N['train'][network].links[0].tf
    
        # tai.printer.blockPrint()
        theta_attr_grid, f_vals[network], grad_f_vals[network], hessian_f_vals[network] \
            = tai.estimation.grid_search_optimization(Nt=N['train'][network]
                                                      , k_Y=k_Y, k_Z=config.estimation_options['k_Z']
                                                      , x_bar=np.array(list(xc_simulated.values()))[:, np.newaxis]
                                                      # , q0 = N['train']['Yang2'].q
                                                      , q0=N['train'][network].q
                                                      , theta_attr_grid=config.experiment_options['theta_grid']
                                                      , attr_label='tt'
                                                      , theta=theta_true[network]
                                                      , gradients=True, hessians=True
                                                      # , paths_column_generation=config.experiment_options['paths_column_generation']
                                                      ,
                                                      inneropt_params={'iters': config.experiment_options['max_sue_iters'],
                                                                       'accuracy_eq': config.experiment_options[
                                                                           'accuracy_eq'], 'uncongested_mode': config.experiment_options['uncongested_mode']
                                                                       }
                                                      )
    
        # Create pandas dataframe
        grid_search_experiment_df = pd.DataFrame({'theta_attr_grid': theta_attr_grid,
                                                  'f_vals': np.array(f_vals[network]).flatten(),
                                                  'grad_f_vals': np.array(grad_f_vals[network]).flatten(),
                                                  'hessian_f_vals': np.array(hessian_f_vals[network]).flatten()})
    
        # Write csv file
        tai.writer.write_csv_to_log_folder(df=grid_search_experiment_df,
                                           filename='summary_report'
                                           , log_file=config.log_file
                                           , float_format='%.1f'
                                           )
    
        # tai.printer.enablePrint()
    
        # Plot
        # plot1 = tai.Artist(folder_plots=config.plots_options['folder_plots'], dim_subplots=(2, 2))
        plot1 = tai.Artist(folder_plots=config.log_file['folderpath'], dim_subplots=(2, 2))
    
        plot1.pseudoconvexity_loss_function_small_networks(filename='pseudo_convexity_loss_function_' + network
                                            , subfolder=""
                                            , f_vals=f_vals[network], grad_f_vals=grad_f_vals[network], hessian_f_vals=hessian_f_vals[network]
                                            , x_range=theta_attr_grid  # np.arange(-3,3, 0.5)
                                            , theta_true=theta_true[network]['tt'])
    
        plt.show()
    
        # Write report in log file
        options_df = pd.DataFrame({'group': [], 'option': [], 'value': []})
    
        for key, value in config.experiment_options.items():
            options_df = options_df.append({'group': 'experiment_options', 'option': key, 'value': value},
                                           ignore_index=True)
    
        tai.writer.write_csv_to_log_folder(df=options_df,
                                           filename='experiment_options_' + network
                                           , log_file=config.log_file
                                           , float_format='%.1f'
                                           )

    # plot1.pseudoconvexity_loss_function_small_networks(
    #     filename='pseudo_convexity_loss_function_small_networks'
    #     , subfolder=""
    #     , f_vals=f_vals, grad_f_vals=grad_f_vals, hessian_f_vals=hessian_f_vals
    #     , x_range=theta_attr_grid  # np.arange(-3,3, 0.5)
    #     , colors = ['blue','red', 'black', 'green']
    #     , theta_true=theta_true[network]['tt'])

    plot1.pseudoconvexity_loss_function_small_networks_lite(
        filename='pseudo_convexity_loss_function_small_networks'
        , subfolder=""
        , f_vals=f_vals
        , x_range=theta_attr_grid  # np.arange(-3,3, 0.5)
        , colors = ['blue','red', 'black', 'green']
        , labels = ['Toy', 'Wang','Lo', 'Yang']
        , theta_true=theta_true[network]['tt'])



    sys.exit()


# ==============================================================================
# 4c) OPTIMIZATION METHODS
# ==============================================================================

# - Conduct hypothesis testing by starting with NGD or Gauss-newton and viceversa.
# - Show the convergence of all networks in the same plot
# - Make sure to write tables with the inference results


if config.experiment_options['inductive_bias_experiment']:

    config.set_experiments_log_files(networkname='small-networks')

    #At theta = -10, the newton method alone gets stuck. L-M requires tuning of the dumping parameter to perform better than G-N
    # Update with wrong reference OD matrix
    # N['train']['Yang'].q = N['train']['Yang2'].q

    config.theta_0 = config.experiment_options['theta_0']

    config.experiment_options['theta_true'] = theta_true

    # Leave coverage in 0.3 so around 5 links are considered for estimation
    config.set_simulated_counts(max_link_coverage=1, sd_x=0, sd_Q=0, scale_Q=1)

    # config.set_od_bias_yang_experiment(theta_0=config.theta_0, theta_true = theta_true[current_network])

    # config.estimation_options['outeropt_method_norefined'] = 'ngd'
    # config.estimation_options['outeropt_method_refined'] = 'gauss-newton'

    config.estimation_options['outeropt_method_norefined'] = 'gauss-newton'
    config.estimation_options['outeropt_method_refined'] = 'ngd'

    # Learning rate for first order optimization
    config.estimation_options['eta_norefined'] = 2e-0
    # config.estimation_options['eta_norefined'] = 3e-0
    config.estimation_options['eta_refined'] = 2e-0

    config.estimation_options['bilevel_iters_norefined'] = 10  # 10
    config.estimation_options['bilevel_iters_refined'] = 10  # 5

    # A nice example is to set the eta for NGD in 1 and start from -4.2. It shows the quasi-optimality and Newton converging to the true value then




    # Features includes in utility function for estimation
    if k_Z_estimation is None:
        k_Z_estimation = config.estimation_options['k_Z']

    # q0s = {'true_od': N['train'][current_network].q, 'distorted_od': N['train']['Yang2'].q}

    results_norefined_bilevelopt = {}
    results_refined_bilevelopt= {}
    results_norefined_refined_df = {}
    parameter_inference_refined_table = {}
    model_inference_refined_table = {}

    networks = ['N3', 'Wang', 'LoChan', 'Yang']

    methods1 = ['ngd', 'gauss-newton']
    methods2 = list(reversed(methods1))

    for methods in [methods1, methods2]:

        config.set_experiments_log_files(networkname='small-networks')

        for network in networks:

            # Generate traffic counts
            if k_Z_simulation is None:
                k_Z_simulation = config.estimation_options['k_Z']

            # Generate synthetic traffic counts
            xc_simulated, xc_withdraw = tai.estimation.generate_link_counts_equilibrium(
                Nt=N['train'][network]  # tai.modeller.clone_network(N['train'][i], label = 'clone')
                , theta=theta_true[network]
                , k_Y=k_Y, k_Z=k_Z_simulation
                , uncongested_mode=config.experiment_options['uncongested_mode']
                , eq_params={'iters': config.experiment_options['max_sue_iters'],
                             'accuracy_eq': config.estimation_options['accuracy_eq'], 'method': 'line_search',
                             'iters_ls': 50}
                , coverage=config.sim_options['max_link_coverage']
                , noise_params=config.sim_options['noise_params']
                , n_paths=config.sim_options['n_paths_synthetic_counts']
            )

            # Store synthetic counts into link objects of network object
            N['train'][network].reset_link_counts()
            N['train'][network].store_link_counts(xct=xc_simulated)

            xc = xc_simulated

            bilevel_estimation_norefined = tai.estimation.Estimation(config.theta_0)

            q_norefined_bilevelopt, theta_norefined_bilevelopt, objective_norefined_bilevelopt, result_eq_norefined_bilevelopt, results_norefined_bilevelopt[network] \
                = bilevel_estimation_norefined.odtheta_estimation_bilevel(
                # Nt= tai.modeller.clone_network(N['train'][i], label = N['train'][i].label),
                Nt=N['train'][network],
                k_Y=k_Y, k_Z=k_Z_estimation,
                Zt={1: N['train'][network].Z_dict},
                q0=N['train'][network].q,
                q_bar=N['train'][network].q,
                xct={1: np.array(list(xc.values()))},
                theta0=config.theta_0,
                standardization=config.estimation_options['standardization_norefined'],
                outeropt_params={
                    'od_estimation': False,
                    'method': methods[0],
                    'batch_size': config.estimation_options['links_batch_size'],
                    'paths_batch_size': config.estimation_options['paths_batch_size'],
                    'iters_scaling': int(0e0),
                    'iters': config.estimation_options['iters_norefined'],  # 10
                    'eta_scaling': 1e-1,
                    'eta': config.experiment_options['eta_norefined'],  # works well for simulated networks
                    # 'eta': 1e-4, # works well for Fresno real network
                    'gamma': 0,
                    'v_lm': 1, 'lambda_lm': 0,
                    'beta_1': 0.9, 'beta_2': 0.99
                },
                inneropt_params={'iters': config.experiment_options['max_sue_iters_norefined'],
                                 'accuracy_eq': config.estimation_options['accuracy_eq']
                    , 'method': 'line_search', 'iters_ls': 10
                    , 'k_path_set_selection': config.estimation_options['k_path_set_selection']
                    , 'dissimilarity_weight': config.estimation_options['dissimilarity_weight']
                    , 'uncongested_mode': config.experiment_options['uncongested_mode']
                                 },
                bilevelopt_params={'iters': config.estimation_options['bilevel_iters_norefined']},  # {'iters': 10},
                n_paths_column_generation=config.estimation_options['n_paths_column_generation']
                # plot_options = {'y': 'objective'}
            )

            config.estimation_results['theta_norefined'] = theta_norefined_bilevelopt
            config.estimation_results['best_loss_norefined'] = objective_norefined_bilevelopt

            # Statistical inference
            print('\nInference with no refined solution')

            parameter_inference_norefined_table, model_inference_norefined_table \
                = tai.estimation.hypothesis_tests(theta_h0=0
                                                  , theta=theta_norefined_bilevelopt
                                                  , YZ_x=tai.estimation.get_design_matrix(
                    Y={'tt': result_eq_norefined_bilevelopt['tt_x']}
                    , Z=N['train'][network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
                                                  , xc=np.array(list(xc.values()))[:, np.newaxis]
                                                  , q=tai.networks.denseQ(Q=N['train'][network].Q
                                                                          , remove_zeros=
                                                                          N['train'][network].setup_options[
                                                                              'remove_zeros_Q'])
                                                  , Ix=N['train'][network].D, Iq=N['train'][network].M
                                                  , C=N['train'][network].C
                                                  , pct_lowest_sse=config.estimation_options['pct_lowest_sse_norefined']
                                                  , alpha=0.05)

            with pd.option_context('display.float_format', '{:0.3f}'.format):
                print('\nSummary of logit parameters: \n', parameter_inference_norefined_table.to_string(index=False))
                # tai.writer.write_csv_to_log_folder(df=parameter_inference_norefined_table, filename='parameter_inference_norefined_table'
                #                                    , log_file=config.log_file)

                print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
                # tai.writer.write_csv_to_log_folder(df=model_inference_norefined_table,
                #                                    filename='model_inference_norefined_table'
                #                                    , log_file=config.log_file)



            # Fine scale solution (the initial objective can be different because we know let's more iterations to be performed to achieve equilibrium)

            bilevel_estimation_refined = tai.estimation.Estimation(theta_norefined_bilevelopt)

            q_refined_bilevelopt, theta_refined_bilevelopt, objective_refined_bilevelopt, result_eq_refined_bilevelopt\
                , results_refined_bilevelopt[network] \
                = bilevel_estimation_refined.odtheta_estimation_bilevel(Nt=N['train'][network],
                                                            k_Y=k_Y, k_Z=k_Z_estimation,
                                                            Zt={1: N['train'][network].Z_dict},
                                                            # q0=N['train'][network].q,
                                                            # q_bar=N['train']['Yang2'].q,
                                                            q_bar=N['train'][network].q,
                                                            q0=N['train'][network].q,
                                                            xct={1: np.array(list(xc.values()))},
                                                            theta0=theta_norefined_bilevelopt,
                                                            # theta0= dict.fromkeys(k_Y+config.estimation_options['k_Z'],0),
                                                            standardization=config.estimation_options[
                                                                'standardization_refined'],
                                                            outeropt_params={
                                                                'method': methods[1]
                                                                , 'iters_scaling': int(0e0)
                                                                , 'iters': config.estimation_options['iters_refined']
                                                                # int(2e1)
                                                                , 'eta_scaling': 1e-2
                                                                , 'eta': config.experiment_options['eta_refined']  # 1e-6
                                                                , 'gamma': 0
                                                                , 'v_lm': 1e3, 'lambda_lm': 1e1
                                                                , 'beta_1': 0.9, 'beta_2': 0.99
                                                                , 'batch_size': config.estimation_options['links_batch_size']
                                                                , 'paths_batch_size': config.estimation_options[
                                                                    'paths_batch_size']
                                                            },
                                                            inneropt_params={
                                                                'iters': config.experiment_options['max_sue_iters_refined'],
                                                                'accuracy_eq': config.estimation_options['accuracy_eq'],
                                                                'method': 'line_search', 'iters_ls': 20
                                                                , 'uncongested_mode': config.experiment_options[
                                                                    'uncongested_mode']
                                                            },
                                                            # {'iters': 100, 'accuracy_eq': config.estimation_options['accuracy_eq']},
                                                            bilevelopt_params={
                                                                'iters': config.estimation_options['bilevel_iters_refined']},
                                                            # {'iters': 10}
                                                            # , plot_options = {'y': 'objective'}
                                                            n_paths_column_generation=config.estimation_options[
                                                                'n_paths_column_generation']
                                                            )

            config.estimation_results['theta_refined'] = theta_refined_bilevelopt
            config.estimation_results['best_loss_refined'] = objective_refined_bilevelopt

            # Statistical inference
            print('\nInference with refined solution')

            parameter_inference_refined_table, model_inference_refined_table \
                = tai.estimation.hypothesis_tests(theta_h0=0
                                                  , theta=theta_refined_bilevelopt
                                                  , YZ_x=tai.estimation.get_design_matrix(
                    Y={'tt': result_eq_refined_bilevelopt['tt_x']}
                    , Z=N['train'][network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
                                                  , xc=np.array(list(xc.values()))[:, np.newaxis]
                                                  , q=tai.networks.denseQ(Q=N['train'][network].Q
                                                                          , remove_zeros=
                                                                          N['train'][network].setup_options[
                                                                              'remove_zeros_Q'])
                                                  , Ix=N['train'][network].D, Iq=N['train'][network].M
                                                  , C=N['train'][network].C
                                                  , pct_lowest_sse=config.estimation_options['pct_lowest_sse_refined']
                                                  , alpha=0.05)

            with pd.option_context('display.float_format', '{:0.3f}'.format):
                # pd.set_option('display.max_rows', 500)
                # pd.set_option('display.max_columns', 500)
                # pd.set_option('display.width', 150)
                print('\nSummary of logit parameters: \n', parameter_inference_refined_table.to_string(index=False))
                # tai.writer.write_csv_to_log_folder(df=parameter_inference_refined_table, filename='parameter_inference_refined_table'
                #                                    , log_file=config.log_file)

                print('\nSummary of model: \n', model_inference_refined_table.to_string(index=False))
                # tai.writer.write_csv_to_log_folder(df=model_inference_refined_table, filename='model_inference_refined_table'
                #                                    , log_file=config.log_file)

                # - Generate pandas dataframe prior plotting

            # Store estimates
            # T-tests, confidence intervals and parameter estimates

            parameter_inference_norefined_table.insert(0, 'stage', 'norefined')
            parameter_inference_refined_table.insert(0, 'stage', 'refined')
            parameter_inference_table = parameter_inference_norefined_table.append(parameter_inference_refined_table)

            # print('\nSummary of logit parameters: \n', parameter_inference_norefined_table.to_string(index=False))
            tai.writer.write_csv_to_experiment_log_folder(df=parameter_inference_table, filename='parameter_inference_table_' + network
                                               , log_file=config.log_file)

            # F-test and model summary statistics
            model_inference_norefined_table.insert(0, 'stage', 'norefined')
            model_inference_refined_table.insert(0, 'stage', 'refined')

            model_inference_table = model_inference_norefined_table.append(model_inference_refined_table)

            # print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
            tai.writer.write_csv_to_experiment_log_folder(df=model_inference_table,
                                               filename='model_inference_table_' + network
                                               , log_file=config.log_file)

            # Descriptive statistics

            results_norefined_refined_df[network] = tai.descriptive_statistics \
                .get_loss_and_estimates_over_iterations(results_norefined=results_norefined_bilevelopt[network]
                                                        , results_refined=results_refined_bilevelopt[network])


            # print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
            tai.writer.write_csv_to_experiment_log_folder(df=results_norefined_refined_df[network],filename='estimates_and_losses_over_iterations_df_'+network
                                               , log_file=config.log_file
                                               , float_format='%.3f'
                                               )

            # # Summary report
            # tai.writer.write_experiment_report(filename='summary_report'
            #                                    , config=config
            #                                    , decimals=3
            #                                    # , float_format = 2
            #                                    )

            # Summary report
            tai.writer.write_experiment_options_report(filename='experiment_options_'+network, config=config)


        # Joint bilevel optimization convergence plot

        plot1 = tai.Artist(folder_plots=config.plots_options['folder_plots'], dim_subplots=(2, 2))

        fig = plot1.bilevel_optimization_convergence_small_networks(
            results_df= results_norefined_refined_df
            , filename='loss-vs-vot-over-iterations_' + config.sim_options['current_network']
            , methods=[methods[0],methods[1]]
            , subfolder="experiments/inference"
            , theta_true=theta_true[current_network]
            , colors=['blue','red', 'black', 'green']
            , labels=['Toy', 'Wang', 'Lo', 'Yang']
        )
        tai.writer.write_figure_to_log_folder(fig=fig
                                              , filename='bilevel_optimization_convergence_' + methods[0] + '_' + methods[1]+ '.pdf', log_file=config.log_file)


    sys.exit()


# ==============================================================================
# 4d) INFERENCE
# ==============================================================================

# - Conduct hypothesis testing by starting with NGD or Gauss-newton and viceversa.
# - Show the convergence of all networks in the same plot
# - Make sure to write tables with the inference results


if config.experiment_options['inference_experiment']:

    config.set_experiments_log_files(networkname='small-networks')

    #At theta = -10, the newton method alone gets stuck. L-M requires tuning of the dumping parameter to perform better than G-N
    # Update with wrong reference OD matrix
    # N['train']['Yang'].q = N['train']['Yang2'].q

    config.theta_0 = config.experiment_options['theta_0']

    config.experiment_options['theta_true'] = theta_true

    # Leave coverage in 0.3 so around 5 links are considered for estimation
    # config.set_simulated_counts(max_link_coverage=1, sd_x=0, sd_Q=0, scale_Q=1)

    # config.set_od_bias_yang_experiment(theta_0=config.theta_0, theta_true = theta_true[current_network])

    # config.estimation_options['outeropt_method_norefined'] = 'ngd'
    # config.estimation_options['outeropt_method_refined'] = 'gauss-newton'

    config.estimation_options['outeropt_method_norefined'] = 'gauss-newton'
    config.estimation_options['outeropt_method_refined'] = 'ngd'

    # Learning rate for first order optimization
    config.estimation_options['eta_norefined'] = 2e-0
    # config.estimation_options['eta_norefined'] = 3e-0
    config.estimation_options['eta_refined'] = 2e-0

    config.estimation_options['bilevel_iters_norefined'] = 10  # 10
    config.estimation_options['bilevel_iters_refined'] = 10  # 5

    # A nice example is to set the eta for NGD in 1 and start from -4.2. It shows the quasi-optimality and Newton converging to the true value then


    # Features includes in utility function for estimation
    if k_Z_estimation is None:
        k_Z_estimation = config.estimation_options['k_Z']

    # q0s = {'true_od': N['train'][current_network].q, 'distorted_od': N['train']['Yang2'].q}

    results_norefined_bilevelopt = {}
    results_refined_bilevelopt= {}
    results_norefined_refined_df = {}
    parameter_inference_refined_table = {}
    model_inference_refined_table = {}

    networks = ['N3', 'Wang', 'LoChan', 'Yang']

    methods1 = ['ngd', 'gauss-newton']
    methods2 = list(reversed(methods1))

    xc_simulated = {}
    xc = {}
    for network in networks:
        # Generate synthetic traffic counts
        xc_simulated[network], xc_withdraw = tai.estimation.generate_link_counts_equilibrium(
            Nt=N['train'][network]  # tai.modeller.clone_network(N['train'][i], label = 'clone')
            , theta=theta_true[network]
            , k_Y=k_Y, k_Z=k_Z_simulation
            , uncongested_mode=config.experiment_options['uncongested_mode']
            , eq_params={'iters': config.experiment_options['max_sue_iters'],
                         'accuracy_eq': config.estimation_options['accuracy_eq'], 'method': 'line_search',
                         'iters_ls': 50}
            , coverage=config.sim_options['max_link_coverage']
            , noise_params=config.experiment_options['noise_params']
            , n_paths=config.sim_options['n_paths_synthetic_counts']
        )

        # Store synthetic counts into link objects of network object
        N['train'][network].reset_link_counts()
        N['train'][network].store_link_counts(xct=xc_simulated[network])

        xc[network] = xc_simulated[network]

    for methods in [methods1, methods2]:

        config.set_experiments_log_files(networkname='small-networks')

        for network in networks:

            # Generate traffic counts
            if k_Z_simulation is None:
                k_Z_simulation = config.estimation_options['k_Z']

            bilevel_estimation_norefined = tai.estimation.Estimation(config.theta_0)

            q_norefined_bilevelopt, theta_norefined_bilevelopt, objective_norefined_bilevelopt, result_eq_norefined_bilevelopt, results_norefined_bilevelopt[network] \
                = bilevel_estimation_norefined.odtheta_estimation_bilevel(
                # Nt= tai.modeller.clone_network(N['train'][i], label = N['train'][i].label),
                Nt=N['train'][network],
                k_Y=k_Y, k_Z=k_Z_estimation,
                Zt={1: N['train'][network].Z_dict},
                q0=N['train'][network].q,
                q_bar=N['train'][network].q,
                xct={1: np.array(list(xc[network].values()))},
                theta0=config.theta_0,
                standardization=config.estimation_options['standardization_norefined'],
                outeropt_params={
                    'od_estimation': False,
                    'method': methods[0],
                    'batch_size': config.estimation_options['links_batch_size'],
                    'paths_batch_size': config.estimation_options['paths_batch_size'],
                    'iters_scaling': int(0e0),
                    'iters': config.estimation_options['iters_norefined'],  # 10
                    'eta_scaling': 1e-1,
                    'eta': config.estimation_options['eta_norefined'],  # works well for simulated networks
                    # 'eta': 1e-4, # works well for Fresno real network
                    'gamma': 0,
                    'v_lm': 1, 'lambda_lm': 0,
                    'beta_1': 0.9, 'beta_2': 0.99
                },
                inneropt_params={'iters': config.experiment_options['max_sue_iters_norefined'],
                                 'accuracy_eq': config.estimation_options['accuracy_eq']
                    , 'method': 'line_search', 'iters_ls': 10
                    , 'k_path_set_selection': config.estimation_options['k_path_set_selection']
                    , 'dissimilarity_weight': config.estimation_options['dissimilarity_weight']
                    , 'uncongested_mode': config.experiment_options['uncongested_mode']
                                 },
                bilevelopt_params={'iters': config.estimation_options['bilevel_iters_norefined']},  # {'iters': 10},
                n_paths_column_generation=config.estimation_options['n_paths_column_generation']
                # plot_options = {'y': 'objective'}
            )

            config.estimation_results['theta_norefined'] = theta_norefined_bilevelopt
            config.estimation_results['best_loss_norefined'] = objective_norefined_bilevelopt

            # Statistical inference
            print('\nInference with no refined solution')

            parameter_inference_norefined_table, model_inference_norefined_table \
                = tai.estimation.hypothesis_tests(theta_h0=0
                                                  , theta=theta_norefined_bilevelopt
                                                  , YZ_x=tai.estimation.get_design_matrix(
                    Y={'tt': result_eq_norefined_bilevelopt['tt_x']}
                    , Z=N['train'][network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
                                                  , xc=np.array(list(xc[network].values()))[:, np.newaxis]
                                                  , q=tai.networks.denseQ(Q=N['train'][network].Q
                                                                          , remove_zeros=
                                                                          N['train'][network].setup_options[
                                                                              'remove_zeros_Q'])
                                                  , Ix=N['train'][network].D, Iq=N['train'][network].M
                                                  , C=N['train'][network].C
                                                  , pct_lowest_sse=config.estimation_options['pct_lowest_sse_norefined']
                                                  , alpha=0.05)

            with pd.option_context('display.float_format', '{:0.3f}'.format):
                print('\nSummary of logit parameters: \n', parameter_inference_norefined_table.to_string(index=False))
                # tai.writer.write_csv_to_log_folder(df=parameter_inference_norefined_table, filename='parameter_inference_norefined_table'
                #                                    , log_file=config.log_file)

                print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
                # tai.writer.write_csv_to_log_folder(df=model_inference_norefined_table,
                #                                    filename='model_inference_norefined_table'
                #                                    , log_file=config.log_file)



            # Fine scale solution (the initial objective can be different because we know let's more iterations to be performed to achieve equilibrium)

            bilevel_estimation_refined = tai.estimation.Estimation(theta_norefined_bilevelopt)

            q_refined_bilevelopt, theta_refined_bilevelopt, objective_refined_bilevelopt, result_eq_refined_bilevelopt\
                , results_refined_bilevelopt[network] \
                = bilevel_estimation_refined.odtheta_estimation_bilevel(Nt=N['train'][network],
                                                            k_Y=k_Y, k_Z=k_Z_estimation,
                                                            Zt={1: N['train'][network].Z_dict},
                                                            # q0=N['train'][network].q,
                                                            # q_bar=N['train']['Yang2'].q,
                                                            q_bar=N['train'][network].q,
                                                            q0=N['train'][network].q,
                                                            xct={1: np.array(list(xc[network].values()))},
                                                            theta0=theta_norefined_bilevelopt,
                                                            # theta0= dict.fromkeys(k_Y+config.estimation_options['k_Z'],0),
                                                            standardization=config.estimation_options[
                                                                'standardization_refined'],
                                                            outeropt_params={
                                                                'method': methods[1]
                                                                , 'iters_scaling': int(0e0)
                                                                , 'iters': config.estimation_options['iters_refined']
                                                                # int(2e1)
                                                                , 'eta_scaling': 1e-2
                                                                , 'eta': config.estimation_options['eta_refined']  # 1e-6
                                                                , 'gamma': 0
                                                                , 'v_lm': 1e3, 'lambda_lm': 1e1
                                                                , 'beta_1': 0.9, 'beta_2': 0.99
                                                                , 'batch_size': config.estimation_options['links_batch_size']
                                                                , 'paths_batch_size': config.estimation_options[
                                                                    'paths_batch_size']
                                                            },
                                                            inneropt_params={
                                                                'iters': config.experiment_options['max_sue_iters_refined'],
                                                                'accuracy_eq': config.estimation_options['accuracy_eq'],
                                                                'method': 'line_search', 'iters_ls': 20
                                                                , 'uncongested_mode': config.experiment_options[
                                                                    'uncongested_mode']
                                                            },
                                                            # {'iters': 100, 'accuracy_eq': config.estimation_options['accuracy_eq']},
                                                            bilevelopt_params={
                                                                'iters': config.estimation_options['bilevel_iters_refined']},
                                                            # {'iters': 10}
                                                            # , plot_options = {'y': 'objective'}
                                                            n_paths_column_generation=config.estimation_options[
                                                                'n_paths_column_generation']
                                                            )

            config.estimation_results['theta_refined'] = theta_refined_bilevelopt
            config.estimation_results['best_loss_refined'] = objective_refined_bilevelopt

            # Statistical inference
            print('\nInference with refined solution')

            parameter_inference_refined_table, model_inference_refined_table \
                = tai.estimation.hypothesis_tests(theta_h0=0
                                                  , theta=theta_refined_bilevelopt
                                                  , YZ_x=tai.estimation.get_design_matrix(
                    Y={'tt': result_eq_refined_bilevelopt['tt_x']}
                    , Z=N['train'][network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
                                                  , xc=np.array(list(xc[network].values()))[:, np.newaxis]
                                                  , q=tai.networks.denseQ(Q=N['train'][network].Q
                                                                          , remove_zeros=
                                                                          N['train'][network].setup_options[
                                                                              'remove_zeros_Q'])
                                                  , Ix=N['train'][network].D, Iq=N['train'][network].M
                                                  , C=N['train'][network].C
                                                  , pct_lowest_sse=config.estimation_options['pct_lowest_sse_refined']
                                                  , alpha=0.05)

            with pd.option_context('display.float_format', '{:0.3f}'.format):
                # pd.set_option('display.max_rows', 500)
                # pd.set_option('display.max_columns', 500)
                # pd.set_option('display.width', 150)
                print('\nSummary of logit parameters: \n', parameter_inference_refined_table.to_string(index=False))
                # tai.writer.write_csv_to_log_folder(df=parameter_inference_refined_table, filename='parameter_inference_refined_table'
                #                                    , log_file=config.log_file)

                print('\nSummary of model: \n', model_inference_refined_table.to_string(index=False))
                # tai.writer.write_csv_to_log_folder(df=model_inference_refined_table, filename='model_inference_refined_table'
                #                                    , log_file=config.log_file)

                # - Generate pandas dataframe prior plotting

            # Store estimates
            # T-tests, confidence intervals and parameter estimates

            parameter_inference_norefined_table.insert(0, 'stage', 'norefined')
            parameter_inference_refined_table.insert(0, 'stage', 'refined')
            parameter_inference_table = parameter_inference_norefined_table.append(parameter_inference_refined_table)

            # print('\nSummary of logit parameters: \n', parameter_inference_norefined_table.to_string(index=False))
            tai.writer.write_csv_to_experiment_log_folder(df=parameter_inference_table, filename='parameter_inference_table_' + network
                                               , log_file=config.log_file)

            # F-test and model summary statistics
            model_inference_norefined_table.insert(0, 'stage', 'norefined')
            model_inference_refined_table.insert(0, 'stage', 'refined')

            model_inference_table = model_inference_norefined_table.append(model_inference_refined_table)

            # print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
            tai.writer.write_csv_to_experiment_log_folder(df=model_inference_table,
                                               filename='model_inference_table_' + network
                                               , log_file=config.log_file)

            # Descriptive statistics

            results_norefined_refined_df[network] = tai.descriptive_statistics \
                .get_loss_and_estimates_over_iterations(results_norefined=results_norefined_bilevelopt[network]
                                                        , results_refined=results_refined_bilevelopt[network])

            # # Summary report
            # tai.writer.write_experiment_report(filename='summary_report'
            #                                    , config=config
            #                                    , decimals=3
            #                                    # , float_format = 2
            #                                    )

            # Summary report
            tai.writer.write_experiment_options_report(filename='experiment_options_'+network, config=config)


        # Joint bilevel optimization convergence plot

        plot1 = tai.Artist(folder_plots=config.plots_options['folder_plots'], dim_subplots=(2, 2))

        fig = plot1.bilevel_optimization_convergence_small_networks(
            results_df= results_norefined_refined_df
            , filename='loss-vs-vot-over-iterations_' + config.sim_options['current_network']
            , methods=[methods[0],methods[1]]
            , subfolder="experiments/inference"
            , theta_true=theta_true[current_network]
            , colors=['blue','red', 'black', 'green']
            , labels=['Toy', 'Wang', 'Lo', 'Yang']
        )
        tai.writer.write_figure_to_log_folder(fig=fig
                                              , filename='bilevel_optimization_convergence_' + methods[0] + '_' + methods[1]+ '.pdf', log_file=config.log_file)


    sys.exit()


# ==============================================================================
# 4e) BIAS REFERENCE OD MATRIX (YANG)
# ==============================================================================

if config.experiment_options['Yang_biased_reference_od_experiment']:

    current_network = 'Yang'

    # config.set_log_file(networkname=current_network)

    config.set_experiments_log_files(networkname='small-networks')

    #At theta = -10, the newton method alone gets stuck. L-M requires tuning of the dumping parameter to perform better than G-N
    # Update with wrong reference OD matrix
    # N['train']['Yang'].q = N['train']['Yang2'].q

    config.experiment_options['theta_true'] = theta_true[current_network]

    # config.experiment_options['theta_true']['tt'] = -1.5
    config.theta_0 = config.experiment_options['theta_0']

    # Leave coverage in 0.3 so around 5 links are considered for estimation
    config.set_simulated_counts(max_link_coverage=0.3, sd_x=0.01, sd_Q=0, scale_Q=1)

    # Generate traffic counts
    if k_Z_simulation is None:
        k_Z_simulation = config.estimation_options['k_Z']

    # Generate synthetic traffic counts
    xc_simulated, xc_withdraw = tai.estimation.generate_link_counts_equilibrium(
        Nt=N['train'][current_network]  # tai.mod eller.clone_network(N['train'][i], label = 'clone')
        , theta=config.experiment_options['theta_true']
        , k_Y=k_Y, k_Z=k_Z_simulation
        , eq_params={'iters': config.sim_options['max_sue_iters'],
                     'accuracy_eq': config.estimation_options['accuracy_eq'], 'method': 'line_search', 'iters_ls': 50}
        , coverage=config.sim_options['max_link_coverage']
        , noise_params=config.sim_options['noise_params']
        , uncongested_mode= config.sim_options['uncongested_mode']
        , n_paths=config.sim_options['n_paths_synthetic_counts']
    )

    N['train'][current_network].reset_link_counts()
    N['train'][current_network].store_link_counts(xct=xc_simulated)

    xc = xc_simulated

    # config.set_od_bias_yang_experiment(theta_0=config.theta_0, theta_true = theta_true[current_network])

    # config.estimation_options['outeropt_method_norefined'] = 'gauss-newton' #'adam'
    # config.estimation_options['outeropt_method_norefined'] = 'lm'
    # config.estimation_options['outeropt_method_norefined'] = 'adagrad'
    config.estimation_options['outeropt_method_norefined'] = 'ngd' #'adam'
    config.estimation_options['outeropt_method_refined'] = 'gauss-newton'
    # config.estimation_options['outeropt_method_refined'] = 'lm' #'lm' #gauss-newton
    # config.estimation_options['outeropt_method_refined'] = 'ngd' #'adam'

    # Learning rate for first order optimization
    config.estimation_options['eta_norefined'] = 2e-0
    # config.estimation_options['eta_norefined'] = 3e-0
    config.estimation_options['eta_refined'] = 1e-3

    config.estimation_options['bilevel_iters_norefined'] = 10  # 10
    config.estimation_options['bilevel_iters_refined'] = 10  # 5

    # A nice example is to set the eta for NGD in 1 and start from -4.2. It shows the quasi-optimality and Newton converging to the true value then

    # Store synthetic counts into link objects of network object
    # N['train'][current_network].store_link_counts(xct=xc_validation)
    N['train'][current_network].store_link_counts(xct=xc)


    # Features includes in utility function for estimation
    if k_Z_estimation is None:
        k_Z_estimation = config.estimation_options['k_Z']

    q0s = {'true_od': N['train'][current_network].q, 'distorted_od': N['train']['Yang2'].q}

    results_norefined_bilevelopt = {}
    results_refined_bilevelopt= {}
    results_norefined_refined_df = {}
    parameter_inference_refined_table = {}
    model_inference_refined_table = {}

    for scenario, q0 in q0s.items():

        bilevel_estimation_norefined = tai.estimation.Estimation(config.theta_0)

        q_norefined_bilevelopt, theta_norefined_bilevelopt, objective_norefined_bilevelopt, result_eq_norefined_bilevelopt, results_norefined_bilevelopt[scenario] \
            = bilevel_estimation_norefined.odtheta_estimation_bilevel(
            # Nt= tai.modeller.clone_network(N['train'][i], label = N['train'][i].label),
            Nt=N['train'][current_network],
            k_Y=k_Y, k_Z=k_Z_estimation,
            Zt={1: N['train'][current_network].Z_dict},
            q0=q0,
            # q0=N['train'][current_network].q,
            # q0=N['train']['Yang2'].q,
            # q0=np.ones([N['train'][current_network].q.size,1]),
            # q_bar = N['train']['Yang2'].q,
            q_bar=N['train'][current_network].q,
            xct={1: np.array(list(xc.values()))},
            theta0=config.theta_0,
            standardization=config.estimation_options['standardization_norefined'],
            outeropt_params={
                'od_estimation': False,
                'method': config.estimation_options['outeropt_method_norefined'],
                'batch_size': config.estimation_options['links_batch_size'],
                'paths_batch_size': config.estimation_options['paths_batch_size'],
                'iters_scaling': int(0e0),
                'iters': config.estimation_options['iters_norefined'],  # 10
                'eta_scaling': 1e-1,
                'eta': config.experiment_options['eta_norefined'],  # works well for simulated networks
                # 'eta': 1e-4, # works well for Fresno real network
                'gamma': 0,
                'v_lm': 1, 'lambda_lm': 0,
                'beta_1': 0.9, 'beta_2': 0.99
            },
            inneropt_params={'iters': config.experiment_options['max_sue_iters_norefined'],
                             'accuracy_eq': config.estimation_options['accuracy_eq']
                , 'method': 'line_search', 'iters_ls': 10
                , 'k_path_set_selection': config.estimation_options['k_path_set_selection']
                , 'dissimilarity_weight': config.estimation_options['dissimilarity_weight']
                , 'uncongested_mode': config.experiment_options['uncongested_mode']
                             },
            bilevelopt_params={'iters': config.estimation_options['bilevel_iters_norefined']},  # {'iters': 10},
            n_paths_column_generation=config.estimation_options['n_paths_column_generation']
            # plot_options = {'y': 'objective'}
        )

        config.estimation_results['theta_norefined'] = theta_norefined_bilevelopt
        config.estimation_results['best_loss_norefined'] = objective_norefined_bilevelopt

        bilevel_estimation_refined = tai.estimation.Estimation(theta_norefined_bilevelopt)

        # Fine scale solution (the initial objective can be different because we know let's more iterations to be performed to achieve equilibrium)
        q_refined_bilevelopt, theta_refined_bilevelopt, objective_refined_bilevelopt, result_eq_refined_bilevelopt\
            , results_refined_bilevelopt[scenario] \
            = bilevel_estimation_refined.odtheta_estimation_bilevel(Nt=N['train'][current_network],
                                                        k_Y=k_Y, k_Z=k_Z_estimation,
                                                        Zt={1: N['train'][current_network].Z_dict},
                                                        # q0=N['train'][current_network].q,
                                                        # q_bar=N['train']['Yang2'].q,
                                                        q_bar=N['train'][current_network].q,
                                                        q0= q_norefined_bilevelopt,
                                                        xct={1: np.array(list(xc.values()))},
                                                        theta0=theta_norefined_bilevelopt,
                                                        # theta0= dict.fromkeys(k_Y+config.estimation_options['k_Z'],0),
                                                        standardization=config.estimation_options[
                                                            'standardization_refined'],
                                                        outeropt_params={
                                                            'method': config.estimation_options['outeropt_method_refined']
                                                            , 'iters_scaling': int(0e0)
                                                            , 'iters': config.estimation_options['iters_refined']
                                                            # int(2e1)
                                                            , 'eta_scaling': 1e-2
                                                            , 'eta': config.experiment_options['eta_refined']  # 1e-6
                                                            , 'gamma': 0
                                                            , 'v_lm': 1e3, 'lambda_lm': 1e1
                                                            , 'beta_1': 0.9, 'beta_2': 0.99
                                                            , 'batch_size': config.estimation_options['links_batch_size']
                                                            , 'paths_batch_size': config.estimation_options[
                                                                'paths_batch_size']
                                                        },
                                                        inneropt_params={
                                                            'iters': config.experiment_options['max_sue_iters_refined'],
                                                            'accuracy_eq': config.estimation_options['accuracy_eq'],
                                                            'method': 'line_search', 'iters_ls': 20
                                                            , 'uncongested_mode': config.experiment_options[
                                                                'uncongested_mode']
                                                        },
                                                        # {'iters': 100, 'accuracy_eq': config.estimation_options['accuracy_eq']},
                                                        bilevelopt_params={
                                                            'iters': config.estimation_options['bilevel_iters_refined']},
                                                        # {'iters': 10}
                                                        # , plot_options = {'y': 'objective'}
                                                        n_paths_column_generation=config.estimation_options[
                                                            'n_paths_column_generation']
                                                        )

        config.estimation_results['theta_refined'] = theta_refined_bilevelopt
        config.estimation_results['best_loss_refined'] = objective_refined_bilevelopt

        # Statistical inference
        print('\nInference with refined solution')

        parameter_inference_refined_table, model_inference_refined_table \
            = tai.estimation.hypothesis_tests(theta_h0=0
                                              , theta=theta_refined_bilevelopt
                                              , YZ_x=tai.estimation.get_design_matrix(
                Y={'tt': result_eq_refined_bilevelopt['tt_x']}
                , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
                                              , xc=np.array(list(xc.values()))[:, np.newaxis]
                                              , q=tai.networks.denseQ(Q=N['train'][current_network].Q
                                                                      , remove_zeros=
                                                                      N['train'][current_network].setup_options[
                                                                          'remove_zeros_Q'])
                                              , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
                                              , C=N['train'][current_network].C
                                              , pct_lowest_sse=config.estimation_options['pct_lowest_sse_refined']
                                              , alpha=0.05)

        with pd.option_context('display.float_format', '{:0.3f}'.format):
            # pd.set_option('display.max_rows', 500)
            # pd.set_option('display.max_columns', 500)
            # pd.set_option('display.width', 150)
            print('\nSummary of logit parameters: \n', parameter_inference_refined_table.to_string(index=False))
            # tai.writer.write_csv_to_log_folder(df=parameter_inference_refined_table, filename='parameter_inference_refined_table'
            #                                    , log_file=config.log_file)

            print('\nSummary of model: \n', model_inference_refined_table.to_string(index=False))
            # tai.writer.write_csv_to_log_folder(df=model_inference_refined_table, filename='model_inference_refined_table'
            #                                    , log_file=config.log_file)

            # - Generate pandas dataframe prior plotting

        # Store estimates
        # T-tests, confidence intervals and parameter estimates
        # parameter_inference_norefined_table.insert(0, 'stage', 'norefined')
        parameter_inference_refined_table.insert(0, 'stage', 'refined')
        # parameter_inference_table = parameter_inference_norefined_table.append(parameter_inference_refined_table)
        parameter_inference_table = parameter_inference_refined_table

        # print('\nSummary of logit parameters: \n', parameter_inference_norefined_table.to_string(index=False))
        tai.writer.write_csv_to_log_folder(df=parameter_inference_table, filename='parameter_inference_table_' + scenario
                                           , log_file=config.log_file)

        # F-test and model summary statistics
        # model_inference_norefined_table.insert(0, 'stage', 'norefined')
        model_inference_refined_table.insert(0, 'stage', 'refined')

        # model_inference_table = model_inference_norefined_table.append(model_inference_refined_table)
        model_inference_table = model_inference_refined_table

        # print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
        tai.writer.write_csv_to_log_folder(df=model_inference_table,
                                           filename='model_inference_table_' + scenario
                                           , log_file=config.log_file)

        # Losses over iterations
        results_norefined_refined_df[scenario] = tai.descriptive_statistics \
            .get_loss_and_estimates_over_iterations(results_norefined=results_norefined_bilevelopt[scenario]
                                                    , results_refined=results_refined_bilevelopt[scenario])

        # print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
        tai.writer.write_csv_to_log_folder(df=results_norefined_refined_df[scenario],
                                           filename='estimates_and_losses_over_iterations_df_' + scenario
                                           , log_file=config.log_file
                                           , float_format='%.3f'
                                           )

        # # Summary report
        # tai.writer.write_experiment_report(filename='summary_report'
        #                                    , config=config
        #                                    , decimals=3
        #                                    # , float_format = 2
        #                                    )


    # Joint bilevel optimization convergence plot

    plot1 = tai.Artist(folder_plots=config.plots_options['folder_plots'], dim_subplots=(2, 2))

    fig = plot1.bilevel_optimization_convergence_yang(
        results_df= results_norefined_refined_df
        , filename='loss-vs-vot-over-iterations_' + config.sim_options['current_network']
        , methods=[config.estimation_options['outeropt_method_norefined'],
                   config.estimation_options['outeropt_method_refined']]
        , subfolder="experiments/inference"
        , theta_true=theta_true[current_network]
        , colors = ['blue','red']
        , labels = ['True O-D', 'Distorted O-D']
    )

    tai.writer.write_figure_to_log_folder(fig=fig
                                          , filename='bilevel_optimization_convergence_distorted_od_yang.pdf', log_file=config.log_file)

    # Summary report
    tai.writer.write_experiment_options_report(filename='experiment_options', config=config)


    sys.exit()



# =============================================================================
# 4a) RANDOM ERROR OD MATRIX (Non-deterministic OD) -> For Sioux Falls and Yang only
# ==============================================================================


# =============================================================================
# 5) BILEVEL OPTIMIZATION
# =============================================================================

if config.sim_options['prop_validation_sample'] > 0:

    # Get a training and testing sample
    xc, xc_validation = tai.estimation.generate_training_validation_samples(
        xct = xc
        , prop_validation = config.sim_options['prop_validation_sample']
    )

else:
    xc = xc
    xc_validation = xc

# Store synthetic counts into link objects of network object
N['train'][current_network].store_link_counts(xct = xc_validation)
N['train'][current_network].store_link_counts(xct = xc)


# =============================================================================
# 3.2) FIXED EFFECTS
# =============================================================================

#TODO: Simulation is not accounting for observed link effects because synthetic counts were created before

if observed_links_fixed_effects == 'random' and config.sim_options['fixed_effects']['coverage'] > 0:
    N['train'][current_network].set_fixed_effects_attributes(config.sim_options['fixed_effects'], observed_links = observed_links_fixed_effects)

    for k in N['train'][current_network].k_fixed_effects:
        theta_true[i][k] = theta_true_fixed_effects # -float(np.random.uniform(1,2,1))
        config.theta_0[k] = theta_0_fixed_effects
        config.estimation_options['k_Z'] = [*config.estimation_options['k_Z'], k]


    # Update dictionary with attributes values at the network level
    N['train'][current_network].set_Z_attributes_dict_network(links_dict=N['train'][current_network].links_dict)

    if len(N['train'][current_network].k_fixed_effects) > 0:
        print('\nFixed effects created within observed links only:', N['train'][current_network].k_fixed_effects)


# =============================================================================
# 3.2) HEURISTICS FOR SCALING OF OD MATRIX AND SEARCH OF INITIAL LOGIT ESTIMATE
# =============================================================================

if config.estimation_options['theta_search'] is not None:

    # # Generate synthetic traffic counts
    # xc_simulated = tai.estimation.generate_link_counts_equilibrium(
    #     Nt=N['train'][current_network]  # tai.modeller.clone_network(N['train'][i], label = 'clone')
    #     , theta=theta_true[current_network]
    #     , k_Y=k_Y, k_Z=config.estimation_options['k_Z']
    #     , eq_params={'iters': config.sim_options['max_sue_iters'], 'accuracy_eq': config.estimation_options['accuracy_eq']}
    #     , coverage=config.sim_options['max_link_coverage']
    #     , noise_params=config.sim_options['noise_params']
    # )

    # Grid and random search are performed under the assumption of an uncongested network to speed up the search

    if config.estimation_options['theta_search'] == 'grid':




        theta_attr_grid, f_vals, grad_f_vals, hessian_f_vals \
            = tai.estimation.grid_search_optimization(Nt=N['train'][current_network]
                                                      , k_Y=k_Y, k_Z=config.estimation_options['k_Z']
                                                      , x_bar=np.array(list(xc.values()))[:, np.newaxis]
                                                      # , theta_attr_grid= np.arange(-10, 10, 0.5)
                                                      # , theta_attr_grid= [1,0,-1,-10,-10000]
                                                      , theta_attr_grid=[1, 0, -1e-3, -1e-1, -5e-1, -1, -2]
                                                      , attr_label='tt'
                                                      , theta=config.theta_0
                                                      , gradients=False, hessians=False
                                                      , inneropt_params=
                                                      {'iters': 0*config.estimation_options['max_sue_iters_refined'], 'accuracy_eq': config.estimation_options['accuracy_eq']}
                                                      )

    # print('grid for theta_t', theta_attr_grid)
    # print('losses ', f_vals)
    # print('gradients ', grad_f_vals)

        # Plot
        plot1 = tai.Artist(folder_plots = config.plots_options['folder_plots'], dim_subplots=(2, 2))



        plot1.pseudoconvexity_loss_function(filename ='quasiconvexity_l2norm_' + config.sim_options['current_network']
                                            , subfolder = "experiments/quasiconvexity"
                                            , f_vals = f_vals, grad_f_vals = grad_f_vals, hessian_f_vals = hessian_f_vals
                                            , x_range = theta_attr_grid  #np.arange(-3,3, 0.5)
                                            , theta_true = theta_true[current_network]['tt'])


        # Initial point to perform the scaling using the best value for grid search of the travel time parameter
        min_loss = float('inf')
        min_theta_gs = 0

        for theta_gs, loss in zip(theta_attr_grid,f_vals):
            if loss < min_loss:
                min_loss = loss
                min_theta_gs = theta_gs

        # print('best theta is ', min_theta_gs)
        # print('best theta: ', str("{0:.0E}".format(min_theta_gs)))
        print('best theta is ', str({key: round(val, 3) for key, val in min_theta_gs.items()}))

        config.theta_0['tt'] = min_theta_gs

    if config.estimation_options['theta_search'] == 'random':

        # Bound for values of the theta vector entries. Note that if it is too loose, the sparse attributes generates problems.
        config.bounds_theta_0 = {key: (-0.2, 0.2) for key, val in config.theta_0.items()}

        # No random search is performed along the scale of the od matrix is the q_random_search option is false

        if config.estimation_options['q_random_search']:
            config.bounds_q = (0, 2)

        else:
            config.bounds_q = (1, 1)

        thetas_rs, q_scales_rs, f_vals \
            = tai.estimation.random_search_optimization(Nt=N['train'][current_network]
                                                      , k_Y=k_Y, k_Z=config.estimation_options['k_Z']
                                                      , x_bar=np.array(list(xc.values()))[:, np.newaxis]
                                                      , n_draws = config.estimation_options['n_draws_random_search']
                                                      , theta_bounds = config.bounds_theta_0
                                                      , q_bounds = config.bounds_q #config.bounds_q
                                                      , inneropt_params={
                                                        'iters': config.estimation_options['max_sue_iters_norefined']
                                                        , 'accuracy_eq': config.estimation_options['accuracy_eq'], 'method': 'line_search', 'iters_ls': 10}
                                                      , silent_mode = True
                                                      , uncongested_mode = False
                                                      )

        min_loss = float('inf')
        min_theta_rs = 0
        min_q_scale_rs = 0

        for theta_rs, q_scale_rs, loss in zip(thetas_rs,q_scales_rs, f_vals):
            if loss < min_loss:
                min_loss = loss
                min_theta_rs = theta_rs
                min_q_scale_rs = q_scale_rs

        # print('best theta is: ', str({key: "{0:.1E}".format(val) for key, val in min_theta_rs.items()}))
        print('best theta is ', str({key: round(val, 3) for key, val in min_theta_rs.items()}))

        # print('best q scale is: ', str({key: "{0:.1E}".format(val) for key, val in min_q_scale_rs.items()}))
        print('best q scale is ', str({key: round(val, 3) for key, val in min_q_scale_rs.items()}))


        #Update the parameter for the initial theta values
        for attr in min_theta_rs.keys():
            config.theta_0[attr] = min_theta_rs[attr]

    # plt.show()

#TODO: scaling of Q matrix in the simulated setting is not working well. It is not 1 the best scaling and it does affect significantly
# statistical inference

if config.estimation_options['scaling_Q']:

    # If the scaling factor is too little, the gradients become 0 apparently.

    # Create grid
    scale_grid_q = [1e-1, 5e-1, 1e0, 1.5, 2e0, 3e0]

    # Add best scale found by random search into grid
    if config.estimation_options['theta_search'] == 'random':
        scale_grid_q.append(list(min_q_scale_rs.values())[0])

    loss_scaling = tai.estimation.scale_Q(x_bar = np.array(list(xc.values()))[:, np.newaxis]
                                          , Nt = N['train'][current_network], k_Y = k_Y, k_Z = config.estimation_options['k_Z']
                                          , theta_0 = config.theta_0
                                          # , scale_grid = [1e-3,1e-2,1e-1]
                                          # , scale_grid=[10e-1]
                                          , scale_grid = scale_grid_q
                                          , n_paths = None #config.estimation_options['n_paths_column_generation']
                                          # , scale_grid = [9e-1,10e-1,11e-1]
                                          , silent_mode = True
                                          , inneropt_params = {'iters': config.estimation_options['max_sue_iters_norefined'], 'accuracy_eq': config.estimation_options['accuracy_eq']
                                                                , 'method': 'line_search', 'iters_ls': 10}
                                          )

    # Search for best scaling

    min_loss = float('inf')
    min_scale = 1
    for scale,loss in loss_scaling.items():
        if loss < min_loss:
            min_loss = loss
            min_scale = scale

    # Perform scaling that minimizes the loss
    N['train'][current_network].Q = min_scale*N['train'][current_network].Q
    N['train'][current_network].q = tai.networks.denseQ(Q=N['train'][current_network].Q, remove_zeros= N['train'][current_network].setup_options['remove_zeros_Q'])

    print('Q matrix was rescaled with a ' + str(min_scale) + ' factor')

# =============================================================================
# 3.2) BENCHMARK PREDICTIONS
# =============================================================================

# Naive prediction using mean counts
config.estimation_results['mean_counts_prediction_loss'], config.estimation_results['mean_count_benchmark_model'] = tai.estimation.mean_count_l2norm(x_bar =  np.array(list(xc.values()))[:, np.newaxis], mean_x = config.estimation_results['mean_count_benchmark_model'])

print('\nObjective function under mean count prediction: ' + '{:,}'.format(round(config.estimation_results['mean_counts_prediction_loss'],1)))

# Naive prediction using uncongested network
config.estimation_results['equilikely_prediction_loss'], x_eq_equilikely \
    = tai.estimation.loss_predicted_counts_uncongested_network(
    x_bar = np.array(list(xc.values()))[:, np.newaxis], Nt = N['train'][current_network]
    , k_Y = k_Y, k_Z = config.estimation_options['k_Z'], theta_0 = dict.fromkeys(config.theta_0, 0))

print('Objective function under equilikely route choices: ' + '{:,}'.format(round(config.estimation_results['equilikely_prediction_loss'],1)))


# =============================================================================
# 3d) REGULARIZATION
# =============================================================================

# i) Regularization with first order optimization which is faster and scaling path level values

if config.sim_options['regularization']:

    # Evaluate objective function for the grid of values for lamabda

    # # Keep this here for efficiency instead of putting it in gap function
    # Yt[i] = (get_matrix_from_dict_attrs_values({k_y: Yt[i][k_y] for k_y in k_Y}).T @ Dt[i]).T
    # Zt[i] = (get_matrix_from_dict_attrs_values({k_z: Zt[i][k_z] for k_z in k_Z}).T @ Dt[i]).T
    #
    # if scale['mean'] or scale['std']:
    #     # Scaling by attribute
    #     Yt[i] = preprocessing.scale(Yt[i], with_mean=scale['mean'], with_std=scale['std'], axis=0)
    #     Zt[i] = preprocessing.scale(Zt[i], with_mean=scale['mean'], with_std=scale['std'], axis=1)

    # Use lasso thresholding operator

    # Perform regularization before no refined optimization and with a first optimization method as
    #  it is faster. In addition, features should be scaled so, the regularization is done properly. I may use a lower
    # amount of MSA iterations as only an approximated solution is needed

    xc_training = xc

    lasso_standardization = {'mean': True, 'sd': True}

    theta_regularized_bilevelopt, objective_regularized_bilevelopt, result_eq_regularized_bilevelopt, results_regularized_bilevelopt \
        = tai.estimation.odtheta_estimation_bilevel(
        # Nt= tai.modeller.clone_network(N['train'][i], label = N['train'][i].label),
        Nt=N['train'][current_network],
        k_Y=k_Y, k_Z=config.estimation_options['k_Z'],
        Zt={1: N['train'][current_network].Z_dict},
        q0=N['train'][current_network].q,
        # q0 = N['train'][current_network].q,
        xct={1: np.array(list(xc_training.values()))},
        theta0=config.theta_0,
        # If change to positive number, a higher number of iterations is required but it works well
        # theta0 = theta_true[i],
        standardization = config.estimation_options['standardization_regularized'],
        outeropt_params={
            # 'method': 'gauss-newton',
            # 'method': 'lm',
            # 'method': 'gd',
            'method': 'ngd',
            # 'method': 'adagrad',
            # 'method': 'adam',
            'batch_size': config.estimation_options['links_batch_size'],
            'paths_batch_size': config.estimation_options['paths_batch_size'],
            'iters_scaling': int(0e0),
            'iters': config.estimation_options['iters_regularized'],  # 10
            'eta_scaling': 1e-1,
            'eta': config.estimation_options['eta_regularized'],  # works well for simulated networks
            # 'eta': 1e-4, # works well for Fresno real network
            'gamma': 0,
            'v_lm': 1, 'lambda_lm': 0,
            'beta_1': 0.8, 'beta_2': 0.8
        },
        inneropt_params={'iters': config.estimation_options['max_sue_iters_regularized'], 'accuracy_eq': config.estimation_options['accuracy_eq']},
        bilevelopt_params={'iters': config.estimation_options['bilevel_iters_regularized']},  # {'iters': 10},
        # n_paths_column_generation=config.estimation_options['n_paths_column_generation']
        # plot_options = {'y': 'objective'}
    )

    # Regularization is made using the soft thresholding operator

    # Grid for lambda

    grid_lambda = [0, 1e-3, 1e-2, 5e-2, 1e-1, 1, 1e2, 1e3]

    tai.estimation.lasso_regularization(Nt = N['train'][current_network], grid_lambda = grid_lambda
                                        , theta_estimatef = theta_regularized_bilevelopt
                                        , k_Y= k_Y, k_Z = config.estimation_options['k_Z']
                                        , eq_params = {'iters': config.estimation_options['max_sue_iters_regularized'], 'accuracy_eq': config.estimation_options['accuracy_eq']}
                                        , x_bar = np.array(list(xc_validation.values()))[:,np.newaxis]
                                        , standardization = lasso_standardization)

    here = 0



# =============================================================================
# 3d) ESTIMATION
# =============================================================================


# ii) NO REFINED OPTIMIZATION AND INFERENCE WITH FIRST ORDER OPTIMIZATION METHODS

# Features includes in utility function for estimation
if k_Z_estimation is None:
    k_Z_estimation = config.estimation_options['k_Z']


bilevel_estimation_norefined = tai.estimation.Estimation(config.theta_0)


q_norefined_bilevelopt, theta_norefined_bilevelopt, objective_norefined_bilevelopt, result_eq_norefined_bilevelopt, results_norefined_bilevelopt \
    = bilevel_estimation_norefined.odtheta_estimation_bilevel(
    # Nt= tai.modeller.clone_network(N['train'][i], label = N['train'][i].label),
    Nt=N['train'][current_network],
    k_Y=k_Y, k_Z=k_Z_estimation,
    Zt={1: N['train'][current_network].Z_dict},
    q0=N['train'][current_network].q,
    # q0= N['train']['Yang2'].q,
    # q0=np.ones([N['train'][current_network].q.size,1]),
    # q_bar = N['train']['Yang2'].q,
    q_bar = N['train'][current_network].q,
    xct={1: np.array(list(xc.values()))},
    theta0=config.theta_0,  # If change to positive number, a higher number of iterations is required but it works well
    # theta0 = theta_true[i],
    # theta0 = dict.fromkeys(config.theta_0.keys(), 1),
    standardization=config.estimation_options['standardization_norefined'],
    outeropt_params={
        'od_estimation': False,
        # 'method': 'gauss-newton',
        # 'method': 'lm',
        # 'method': 'gd',
        'method': config.estimation_options['outeropt_method_norefined'],
        # 'method': 'adagrad',
        # 'method': 'adam',
        'batch_size': config.estimation_options['links_batch_size'],
        'paths_batch_size': config.estimation_options['paths_batch_size'],
        'iters_scaling': int(0e0),
        'iters': config.estimation_options['iters_norefined'],  # 10
        'eta_scaling': 1e-1,
        'eta': config.estimation_options['eta_norefined'],  # works well for simulated networks
        # 'eta': 1e-4, # works well for Fresno real network
        'gamma': 0,
        'v_lm': 1, 'lambda_lm': 0,
        'beta_1': 0.9, 'beta_2': 0.99
    },
    inneropt_params={'iters': config.estimation_options['max_sue_iters_norefined'],
                     'accuracy_eq': config.estimation_options['accuracy_eq']
        , 'method': 'line_search', 'iters_ls': 10
        , 'k_path_set_selection': config.estimation_options['k_path_set_selection']
        , 'dissimilarity_weight': config.estimation_options['dissimilarity_weight']},
    bilevelopt_params={'iters': config.estimation_options['bilevel_iters_norefined']},  # {'iters': 10},
    n_paths_column_generation=config.estimation_options['n_paths_column_generation']
    # plot_options = {'y': 'objective'}
)

config.estimation_results['theta_norefined'] = theta_norefined_bilevelopt
config.estimation_results['best_loss_norefined'] = objective_norefined_bilevelopt
best_x_eq_norefined = np.array(list(result_eq_norefined_bilevelopt['x'].values()))[:, np.newaxis]

# Statistical inference
print('\nInference with no refined solution')

parameter_inference_norefined_table, model_inference_norefined_table \
    = tai.estimation.hypothesis_tests(theta_h0=0
                                      , theta=theta_norefined_bilevelopt
                                      , x_eq = best_x_eq_norefined
                                      , YZ_x=tai.estimation.get_design_matrix(
        Y={'tt': result_eq_norefined_bilevelopt['tt_x']}
        , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
                                      , xc=np.array(list(xc.values()))[:, np.newaxis]
                                      , q=tai.networks.denseQ(Q=N['train'][current_network].Q
                                                              , remove_zeros=N['train'][current_network].setup_options[
            'remove_zeros_Q'])
                                      , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
                                      , C=N['train'][current_network].C
                                      , pct_lowest_sse=config.estimation_options['pct_lowest_sse_norefined']
                                      , alpha=0.05)

with pd.option_context('display.float_format', '{:0.3f}'.format):
    print('\nSummary of logit parameters: \n', parameter_inference_norefined_table.to_string(index=False))
    # tai.writer.write_csv_to_log_folder(df=parameter_inference_norefined_table, filename='parameter_inference_norefined_table'
    #                                    , log_file=config.log_file)

    print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
    # tai.writer.write_csv_to_log_folder(df=model_inference_norefined_table,
    #                                    filename='model_inference_norefined_table'
    #                                    , log_file=config.log_file)

# print(parameter_inference_norefined_table)

if config.estimation_options['ttest_selection_norefined']:

    if config.estimation_options['alpha_selection_norefined'] < len(theta_norefined_bilevelopt):

        # An alternative to regularization
        ttest_norefined = np.array(parameter_inference_norefined_table['t-test'])
        ttest_norefined_dict = dict(
            zip(k_Y + k_Z_estimation, list(map(float, parameter_inference_norefined_table['t-test']))))

        n = np.count_nonzero(~np.isnan(np.array(list(xc.values()))[:, np.newaxis]))
        p = len(k_Y + k_Z_estimation)

        critical_alpha = config.estimation_options['alpha_selection_norefined']
        critical_tvalue = stats.t.ppf(1 - critical_alpha / 2, df=n - p)

        if config.estimation_options['alpha_selection_norefined'] >= 1:
            # It picks the alpha minimum(s) ignoring fixed effects

            ttest_lists = []
            for attr, ttest, idx in zip(ttest_norefined_dict.keys(), ttest_norefined.flatten(), np.arange(p)):
                if attr not in N['train'][current_network].k_fixed_effects:
                    ttest_lists.append(ttest)

            critical_tvalue = float(
                -np.sort(-abs(np.sort(ttest_lists)))[config.estimation_options['alpha_selection_norefined'] - 1])

            print('\nSelecting top ' + str(config.estimation_options[
                                               'alpha_selection_norefined']) + ' features based on t-values and excluding fixed effects')

        else:
            print('Selecting features based on critical t-value ' + str(critical_tvalue))

            # print('\ncritical_tvalue:', critical_tvalue)

        # Loop over endogenous and exogenous attributes

        for attr, t_test in ttest_norefined_dict.items():

            if attr not in N['train'][current_network].k_fixed_effects:

                if abs(t_test) < critical_tvalue - 1e-3:
                    if attr in k_Y:
                        k_Y.remove(attr)

                    if attr in k_Z_estimation:
                        k_Z_estimation.remove(attr)

        print('k_Y:', k_Y)
        print('k_Z:', k_Z_estimation)

# print(confint_theta)
# print(width_confint_theta)

# for iter in np.arange(len(list(results_bilevelopt.values()))):
#     print('\n iter: ' + str(iter) )
#     print('theta : ' + str(np.round(results_bilevelopt[iter]['theta'],2)))
#     print('objective: ' + str(np.round(results_bilevelopt[iter]['objective'],2)))


# iii) REFINED OPTIMIZATION AND INFERENCE WITH SECOND ORDER OPTIMIZATION METHODS

# k_Z=['wt', 'c']

if not config.estimation_options['outofsample_prediction_mode']:


    bilevel_estimation_refined = tai.estimation.Estimation(theta_norefined_bilevelopt)

    # Fine scale solution (the initial objective can be different because we know let's more iterations to be performed to achieve equilibrium)
    q_refined_bilevelopt, theta_refined_bilevelopt, objective_refined_bilevelopt, result_eq_refined_bilevelopt, results_refined_bilevelopt \
        = bilevel_estimation_refined .odtheta_estimation_bilevel(Nt=N['train'][current_network],
                                                    k_Y=k_Y, k_Z=k_Z_estimation,
                                                    Zt={1: N['train'][current_network].Z_dict},
                                                    # q0=N['train'][current_network].q,
                                                    # q_bar=N['train']['Yang2'].q,
                                                    q_bar=N['train'][current_network].q,
                                                    q0=q_norefined_bilevelopt,
                                                    xct={1: np.array(list(xc_validation.values()))},
                                                    theta0=theta_norefined_bilevelopt,
                                                    # theta0= dict.fromkeys(k_Y+config.estimation_options['k_Z'],0),
                                                    standardization=config.estimation_options[
                                                        'standardization_refined'],
                                                    outeropt_params={
                                                        # 'method': 'gauss-newton'
                                                        #  'method': 'lm-revised'
                                                        'method': config.estimation_options['outeropt_method_refined']
                                                        # 'method': 'newton'
                                                        # 'method': 'gd'
                                                        # 'method': 'ngd'
                                                        # 'method': 'adagrad'
                                                        , 'iters_scaling': int(0e0)
                                                        , 'iters': config.estimation_options['iters_refined']
                                                        # int(2e1)
                                                        , 'eta_scaling': 1e-2
                                                        , 'eta': config.estimation_options['eta_refined']  # 1e-6
                                                        , 'gamma': 0
                                                        , 'v_lm': 1e3, 'lambda_lm': 1e1
                                                        , 'beta_1': 0.9, 'beta_2': 0.99
                                                        , 'batch_size': config.estimation_options['links_batch_size']
                                                        , 'paths_batch_size': config.estimation_options[
                                                            'paths_batch_size']
                                                    },
                                                    inneropt_params={
                                                        'iters': config.estimation_options['max_sue_iters_refined'],
                                                        'accuracy_eq': config.estimation_options['accuracy_eq'],
                                                        'method': 'line_search', 'iters_ls': 20},
                                                    # {'iters': 100, 'accuracy_eq': config.estimation_options['accuracy_eq']},
                                                    bilevelopt_params={
                                                        'iters': config.estimation_options['bilevel_iters_refined']},
                                                    # {'iters': 10}
                                                    # , plot_options = {'y': 'objective'}
                                                    n_paths_column_generation=config.estimation_options[
                                                        'n_paths_column_generation']
                                                    )

    config.estimation_results['theta_refined'] = theta_refined_bilevelopt
    config.estimation_results['best_loss_refined'] = objective_refined_bilevelopt
    best_x_eq_refined = np.array(list(result_eq_refined_bilevelopt['x'].values()))[:, np.newaxis]

    # Statistical inference
    print('\nInference with refined solution')

    parameter_inference_refined_table, model_inference_refined_table \
        = tai.estimation.hypothesis_tests(theta_h0=0
                                          , theta=theta_refined_bilevelopt
                                          , x_eq = best_x_eq_refined
                                          , YZ_x=tai.estimation.get_design_matrix(
            Y={'tt': result_eq_refined_bilevelopt['tt_x']}
            , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
                                          , xc=np.array(list(xc.values()))[:, np.newaxis]
                                          , q=tai.networks.denseQ(Q=N['train'][current_network].Q
                                                                  , remove_zeros=
                                                                  N['train'][current_network].setup_options[
                                                                      'remove_zeros_Q'])
                                          , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
                                          , C=N['train'][current_network].C
                                          , pct_lowest_sse=config.estimation_options['pct_lowest_sse_refined']
                                          , alpha=0.05)

    with pd.option_context('display.float_format', '{:0.3f}'.format):
        # pd.set_option('display.max_rows', 500)
        # pd.set_option('display.max_columns', 500)
        # pd.set_option('display.width', 150)
        print('\nSummary of logit parameters: \n', parameter_inference_refined_table.to_string(index=False))
        # tai.writer.write_csv_to_log_folder(df=parameter_inference_refined_table, filename='parameter_inference_refined_table'
        #                                    , log_file=config.log_file)

        print('\nSummary of model: \n', model_inference_refined_table.to_string(index=False))
        # tai.writer.write_csv_to_log_folder(df=model_inference_refined_table, filename='model_inference_refined_table'
        #                                    , log_file=config.log_file)

else:

    theta_refined_bilevelopt, objective_refined_bilevelopt, result_eq_refined_bilevelopt, results_refined_bilevelopt = \
        copy.deepcopy(theta_norefined_bilevelopt), copy.deepcopy(objective_norefined_bilevelopt), copy.deepcopy(
            result_eq_norefined_bilevelopt), copy.deepcopy(results_norefined_bilevelopt)

    config.estimation_results['theta_refined'] = copy.deepcopy(theta_refined_bilevelopt)
    config.estimation_results['best_loss_refined'] = copy.deepcopy(objective_refined_bilevelopt)

    parameter_inference_refined_table, model_inference_refined_table = \
        copy.deepcopy(parameter_inference_norefined_table), copy.deepcopy(model_inference_norefined_table)

# print('Inference with refined solution')
# print('ttests :'  + str(ttest_refined))
# print('p-values '  + str(pval_refined))
# print('confidence intervals :' + str(confint_theta_refined))

# Distribution of errors across link counts

# best_x_eq_norefined = np.array(list(
#     results_norefined_bilevelopt[config.estimation_options['bilevel_iters_norefined']]['equilibrium']['x'].values()))[:, np.newaxis]

# best_x_eq_refined = np.array(
#     list(results_refined_bilevelopt[config.estimation_options['bilevel_iters_refined']]['equilibrium']['x'].values()))[:, np.newaxis]

# print('Loss by link', tai.estimation.loss_function_by_link(x_bar = np.array(list(xc.values()))[:,np.newaxis], x_eq = best_x_eq_norefined))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(tai.estimation.error_by_link(x_bar=np.array(list(xc.values()))[:, np.newaxis], x_eq=best_x_eq_norefined))
axs[1].hist(tai.estimation.error_by_link(x_bar=np.array(list(xc.values()))[:, np.newaxis], x_eq=best_x_eq_refined))

plt.show()

tai.writer.write_figure_to_log_folder(fig=fig
                                      , filename='distribution_predicted_count_error.pdf', log_file=config.log_file)

# - Generate pandas dataframe prior plotting
results_norefined_refined_df = tai.descriptive_statistics \
    .get_loss_and_estimates_over_iterations(results_norefined=results_norefined_bilevelopt
                                            , results_refined=results_refined_bilevelopt)

# Plot
plot1 = tai.Artist(folder_plots=config.plots_options['folder_plots'], dim_subplots=(2, 2))

fig = plot1.bilevel_optimization_convergence(
    results_norefined_df=results_norefined_refined_df[results_norefined_refined_df['stage'] == 'norefined']
    , results_refined_df=results_norefined_refined_df[results_norefined_refined_df['stage'] == 'refined']
    , simulated_data=config.sim_options['simulated_counts']
    , filename='loss-vs-vot-over-iterations_' + config.sim_options['current_network']
    , methods=[config.estimation_options['outeropt_method_norefined'],
               config.estimation_options['outeropt_method_refined']]
    , subfolder="experiments/inference"
    , theta_true = theta_true[current_network]
)

tai.writer.write_figure_to_log_folder(fig=fig
                                      , filename='bilevel_optimization_convergence.pdf', log_file=config.log_file)

# # OD estimation over iterations
#
# # Plot
# plot2 = tai.Artist(folder_plots=config.plots_options['folder_plots'], dim_subplots=(1, 2))
#
# # - Create pandas dataframe
# columns_df = ['stage'] + ['iter'] + ['objective']
#
# df_bilevel_norefined = pd.DataFrame(columns=columns_df)
# df_bilevel_refined = pd.DataFrame(columns=columns_df)
#
# # Create pandas dataframe using each row of the dictionary returned by the bilevel methodç
#
# q_true = N['train'][current_network].q_true
#
# # No refined iterations
# for iter in np.arange(1, len(results_norefined_bilevelopt) + 1):
#     q_estimate = np.sum((results_norefined_bilevelopt[iter]['q']-q_true)**2)
#     df_bilevel_norefined.loc[iter] = ['norefined'] + [iter] + [q_estimate]
#
# # Refined iterations
# for iter in np.arange(1, len(results_refined_bilevelopt) + 1):
#     q_estimate = np.sum((results_refined_bilevelopt[iter]['q']-q_true)**2)
#     df_bilevel_refined.loc[iter] = ['refined'] + [iter] + [q_estimate]
#
# # Adjust the iteration numbers
# df_bilevel_refined['iter'] = (df_bilevel_refined['iter'] + df_bilevel_norefined['iter'].max()).astype(int)
#
#
# fig = plot2.q_estimation_convergence(results_norefined_df = df_bilevel_norefined
#                                , results_refined_df = df_bilevel_refined
#                                , methods=[config.estimation_options['outeropt_method_norefined'],
#                                           config.estimation_options['outeropt_method_refined']]
#                                , filename = 'q_estimation_convergence.pdf', subfolder="experiments/inference")
#
# tai.writer.write_figure_to_log_folder(fig=fig
#                                       , filename='q_estimation_convergence.pdf', log_file=config.log_file)

# =============================================================================
# 6) LOG FILE
# =============================================================================

# =============================================================================
# 6a) Summary with most relevant options, prediction error, initial parameters, etc
# =============================================================================

tai.writer.write_estimation_report(filename='summary_report'
                                   , config=config
                                   , decimals=3
                                   # , float_format = 2
                                   )

# =============================================================================
# 6b) General options (sim_options, estimation_options)
# =============================================================================

# Update vector with exogenous covariates
# config.estimation_options['k_Z'] = config.estimation_options['k_Z']

# general_dict = {'type': 'sim_option', 'key': 'selected_year', 'value': 2019}
options_df = pd.DataFrame({'group': [], 'option': [], 'value': []})

for key, value in config.sim_options.items():
    options_df = options_df.append(pd.DataFrame({'group': ['sim_options'], 'option': [key], 'value': [value]}),
                                   ignore_index=True)

for key, value in config.estimation_options.items():
    options_df = options_df.append({'group': 'estimation_options', 'option': key, 'value': value}, ignore_index=True)

for key, value in config.gis_options.items():
    options_df = options_df.append({'group': 'gis', 'option': key, 'value': value}, ignore_index=True)

tai.writer.write_csv_to_log_folder(df=options_df,
                                   filename='global_options'
                                   , log_file=config.log_file
                                   , float_format='%.1f'
                                   )

# =============================================================================
# 6c) Analysis of predicted counts and travel time over iterations
# =============================================================================

predicted_link_counts_over_iterations_df \
    = tai.descriptive_statistics.get_predicted_link_counts_over_iterations_df(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt
    , Nt=N['train'][current_network])

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
tai.writer.write_csv_to_log_folder(df=predicted_link_counts_over_iterations_df,
                                   filename='predicted_link_counts_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.1f'
                                   )

gap_predicted_link_counts_over_iterations_df \
    = tai.descriptive_statistics.get_gap_predicted_link_counts_over_iterations_df(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt
    , Nt=N['train'][current_network])

tai.writer.write_csv_to_log_folder(df=gap_predicted_link_counts_over_iterations_df,
                                   filename='gap_predicted_link_counts_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.1f'
                                   )

# Travel times
predicted_link_traveltime_over_iterations_df \
    = tai.descriptive_statistics.get_predicted_traveltimes_over_iterations_df(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt
    , Nt=N['train'][current_network])

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
tai.writer.write_csv_to_log_folder(df=predicted_link_traveltime_over_iterations_df,
                                   filename='predicted_link_traveltimes_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.2f'
                                   )

# =============================================================================
# 6b) Analysis of parameter estimates and loss over iterations
# =============================================================================

# Log file
loss_and_estimates_over_iterations_df \
    = tai.descriptive_statistics.get_loss_and_estimates_over_iterations(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt)

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
tai.writer.write_csv_to_log_folder(df=loss_and_estimates_over_iterations_df,
                                   filename='loss_and_estimates_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.3f'
                                   )

gap_estimates_over_iterations_df \
    = tai.descriptive_statistics.get_gap_estimates_over_iterations(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt
    , theta_true=theta_true[current_network])

tai.writer.write_csv_to_log_folder(df=gap_estimates_over_iterations_df,
                                   filename='gap_estimates_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.3f'
                                   )

# =============================================================================
# 6c) Best parameter estimates and inference at the end of norefined and refined stages
# =============================================================================

# T-tests, confidence intervals and parameter estimates
parameter_inference_norefined_table.insert(0, 'stage', 'norefined')
parameter_inference_refined_table.insert(0, 'stage', 'refined')
parameter_inference_table = parameter_inference_norefined_table.append(parameter_inference_refined_table)

# print('\nSummary of logit parameters: \n', parameter_inference_norefined_table.to_string(index=False))
tai.writer.write_csv_to_log_folder(df=parameter_inference_table, filename='parameter_inference_table'
                                   , log_file=config.log_file)

# F-test and model summary statistics
model_inference_norefined_table.insert(0, 'stage', 'norefined')
model_inference_refined_table.insert(0, 'stage', 'refined')

model_inference_table = model_inference_norefined_table.append(model_inference_refined_table)

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
tai.writer.write_csv_to_log_folder(df=model_inference_table,
                                   filename='model_inference_table'
                                   , log_file=config.log_file)

sys.exit()