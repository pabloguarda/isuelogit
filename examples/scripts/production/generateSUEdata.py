# =============================================================================
# 1) SETUP
# =============================================================================
import os

# Set seed for reproducibility and consistency between experiments
import numpy as np
np.random.seed(2021)

#=============================================================================
# 1a) MODULES
#==============================================================================

# Internal modules
import isuelogit as isl

# import isuelogit.modeller

# External modules
import sys
import pandas as pd
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
import pylab

import seaborn as sns

# Memory usage
import tracemalloc
# https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python



# =============================================================================
# 1b) CONFIGURATION
# =============================================================================

#Note: The strategy does not work with 'Eastern-Massachusetts' under an uncongested network. Do not why

config = isl.config.Config(network_key = 'SiouxFalls')
# config = isl.config.Config(network_key = 'Eastern-Massachusetts')
# config = isl.config.Config(network_key = 'Berlin-Friedrichshain')
# config = isl.config.Config(network_key = 'Berlin-Mitte-Center')
# config = isl.config.Config(network_key = 'Barcelona')

config.sim_options['prop_validation_sample'] = 0
config.sim_options['regularization'] = False

# No scaling is performed to compute equilibrium as if normalizing by std, otherwise the solution change significantly.
config.estimation_options['standardization_regularized'] = {'mean': True, 'sd': True}
config.estimation_options['standardization_norefined'] = {'mean': False, 'sd': False} #Standardization by std, change results a lot.
config.estimation_options['standardization_refined'] = {'mean': False, 'sd': False}
# * It seems scaling helps to speed up convergence

# Features in utility function
k_Y = ['tt']
k_Z = []
# features = config.estimation_options['features']
# config.estimation_options['features'] = ['wt']
# config.estimation_options['features'] = ['n2,n1']

# If any of the two following is set to be equal to none, then  features will be used later and thus, it includes all features
k_Z_simulation = None #config.estimation_options['features'] #
k_Z_estimation = None #features #config.estimation_options['features']

# Fixed effect by link, nodes or OD zone
config.sim_options['fixed_effects'] = {'Q': False, 'nodes': False, 'links': True, 'links_observed': True, 'coverage': 0}
theta_true_fixed_effects = 1e3
theta_0_fixed_effects = 0
observed_links_fixed_effects = None #'random'

# Feature selection based on t-test from no refined step and ignoring fixed effects (I must NOT do post-selection inference as it violates basic assumptions)
# config.estimation_options['ttest_selection_norefined'] = False
config.estimation_options['ttest_selection_norefined'] = True

# We relax the critical value to remove features that are highly "no significant". A simil of regularization
config.estimation_options['alpha_selection_norefined'] = 0.05 #if it g is higher than 1, it choose the k minimum values

# synthetic counts
config.sim_options['n_paths_synthetic_counts'] = 2 #None #3  # If none, the

# initial path set is used to generate counts
config.sim_options['sue_iters'] = 40 #itereates to generate synthetic counts

# Number of paths in the initial path set
config.estimation_options['n_initial_paths'] = 2

# Coverage of OD pairs to sample new paths
config.estimation_options['ods_coverage_column_generation'] = 0.5
config.estimation_options['n_paths_column_generation'] = 5 #2

# Number of path selected after column generation
config.estimation_options['k_path_set_selection'] = 2

# TODO: Analyze the impact of this parameter on recovery. If it equals 1, the performance is bad by construction
config.estimation_options['dissimilarity_weight'] = 0 #0.5 works well


# accuracy for relative gap
config.estimation_options['accuracy_eq'] = 1e-4

# Bilevel iters
config.estimation_options['bilevel_iters_norefined'] = 10  # 10
config.estimation_options['bilevel_iters_refined'] = 10 # 5


# Parameters for simulation with no noise
config.set_simulated_counts(max_link_coverage = 1, snr_x = None, sd_x = 0, sd_Q = 0.1, scale_Q =1)

# Uncongested mode (Disable when running experiments or bpr parameters will be set to 0)
config.set_uncongested_mode(True)

# Under this mode, the true path is used as the path set to learn the logit parameters
config.set_known_pathset_mode(True)
#Note: when the path set is unknown the parameter for travel time tends to be small, so the bias is in one direction on.y.

# Note: Column generation is a good cure against noise is to perform column generation. But the greater improvement comes
# from computing eequilibrium properly

# Out of sample prediction mode
# config.set_outofsample_prediction_mode(theta = {'tt': -2, 'wt': -3, 'c': -7}, outofsample_prediction = True, mean_count = 100)

# EXPERIMENTS

# - Number of attributes that will be set to 0, which moderate sparsity: with 20 at least, we observe benefits of regularize
config.sim_options['n_R'] = 3#3  # 2 #5 #10 #20 #50

#Labels of sparse attributes
config.sim_options['R_labels'] = ['k' + str(i) for i in np.arange(0, config.sim_options['n_R'])]

# # Initial theta for optimization
# config.theta_0['tt'] = 0

# Key internal objects for analysis and visualization
artist = isl.visualization.Artist(folder_plots = config.plots_options['folder_plots'], dim_subplots=config.plots_options['dim_subplots'])



# =============================================================================
# 1c) LOG-FILE
# =============================================================================

# Record starting date and time of the simulation
# https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python

# if config.experiment_options['experiment_mode'] is not None:
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
# a) CREATION OF TNTP NETWORKS
# =============================================================================

# FROM TNTP REPO
if config.sim_options['current_network'] in config.tntp_networks:

    tntp_network = isl.modeller.build_tntp_network(folderpath=config.paths['folder_tntp_networks'], subfoldername=config.sim_options['current_network'], options=config.sim_options.copy(), config = config)

    N['train'].update({tntp_network.key: tntp_network})

# =============================================================================
# b) SETUP INCIDENCE MATRICES AND LINK ATTRIBUTES IN NETWORKS
# =============================================================================

# + TNTP networks
if config.sim_options['current_network'] in config.tntp_networks:

    N_tntp = N['train'][config.sim_options['current_network']]

    N_tntp = isl.modeller \
        .setup_tntp_network_matrices\
        (
            network=N_tntp
            , setup_options={**config.sim_options, **config.estimation_options}

            # (i) First time to write network matrices consistently

            # , generation=dict(config.sim_options['generation'],
            #            **{'paths': True, 'bpr': False, 'Z': True
            #                , 'C': True, 'D': True, 'M': True,
            #               'Q': False})

            , generation=dict(config.sim_options['generation'],
                       **{'paths': True, 'bpr': False, 'Z': True
                           , 'C': False, 'D': False, 'M': False,
                          'Q': False})

            , reading=dict(config.sim_options['reading'],
                           **{'paths': True
                               , 'C': True, 'D': True, 'M': True, 'Q': True
                               , 'sparse_C': True, 'sparse_D': True
                               , 'sparse_M': True, 'sparse_Q': False
                              }
                           )

            # , reading=dict(config.sim_options['reading'],
            #                **{'paths': False
            #                    , 'C': False, 'D': False, 'M': False, 'Q': True
            #                    , 'sparse_C': True, 'sparse_D': True
            #                    , 'sparse_M': True, 'sparse_Q': False
            #                   }
            #                )
            # , writing=dict(config.sim_options['writing'],
            #                **{'paths': True
            #                    , 'C': True, 'D': True, 'M': True, 'Q': True
            #                    , 'sparse_C': True, 'sparse_D': True
            #                    , 'sparse_M': True , 'sparse_Q': False
            #                   }
            #                )

            # Theta used to generate initial path set
            , theta= config.theta_0 #{**config.theta_true['Y'], **config.theta_true['Z']}
            , folder=config.paths['folder_tntp_networks']
            , subfolder=N_tntp.key
        )


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
config.estimation_options['features'] = [*config.estimation_options['features'], *config.sim_options['R_labels']]

# =============================================================================
# 1.7) SUMMARY OF NETWORKS CHARACTERISTICS
# =============================================================================
networks_table = {'nodes':[],'links':[], 'paths': [], 'ods': []}

# Network description
for i in N['train'].keys():
    networks_table['ods'].append(len(isl.networks.denseQ(N['train'][i].Q, remove_zeros=True)))
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

else:
    print('Algorithm is performed assuming a congested network')

    # Given that it is very expensive to compute path probabitilies and the resuls are good already, it seems fine to perform only one iteration for outer problem
    # iters_est = config.sim_options['ngd_iters']  #5

# =============================================================================
# 1.8) SYNTHETIC COUNTS WITH NOISY OD
# =============================================================================

replicates = 1000

od_df = pd.DataFrame({'od':N['train'][current_network].ods_fromQ(remove_zeros = False, Q = N['train'][current_network].Q)})
links_flows_df = pd.DataFrame({'link': N['train'][current_network].links_dict.keys()})
path_flows_df = pd.DataFrame({'path': [path.get_nodes_keys() for path in N['train'][current_network].paths]})
links_traveltimes_df = pd.DataFrame({'link': N['train'][current_network].links_dict.keys()})


for key,val in theta_true[current_network].items():
    theta_true[current_network][key] = 0

theta_true[current_network]['tt'] = -1

# Sparsity
sparsity = 0.8

# Increase scale of OD matrix to generate congestion and thus variability in travel time
N['train'][current_network].Q = 3*N['train'][current_network].Q

if config.sim_options['simulated_counts'] is True:

    if k_Z_simulation is None:
        k_Z_simulation = config.estimation_options['features']

    # Generate sparsity vector with ids for OD
    sparsity_idxs = list(set(np.flatnonzero(N['train'][current_network].Q)))
    sparsity_idxs = np.random.choice(sparsity_idxs, size=int(round(sparsity * N['train'][current_network].Q.size)),replace=False)

    for replicate in range(replicates):

        print('replicate: ', replicate)



        # Generate synthetic traffic counts
        xc_simulated, xc_withdraw = isl.estimation.generate_link_counts_equilibrium(
            Nt=N['train'][current_network]  # isl.modeller.clone_network(N['train'][i], label = 'clone')
            , theta = theta_true[current_network]
            , k_Y = k_Y, k_Z = k_Z_simulation
            , eq_params = {'iters': config.sim_options['max_sue_iters'], 'accuracy_eq': config.estimation_options['accuracy_eq'], 'method': 'line_search', 'iters_ls': 100}
            # , eq_params={'iters': config.sim_options['max_sue_iters'],
            #              'accuracy_eq': config.estimation_options['accuracy_eq'], 'method': 'msa',
            #              'iters_ls': 1}
            , coverage = config.sim_options['max_link_coverage']
            , noise_params = config.sim_options['noise_params']
            , uncongested_mode=config.sim_options['uncongested_mode']
            , n_paths = config.sim_options['n_paths_synthetic_counts']
            , sparsity_idxs = sparsity_idxs
        )

        N['train'][current_network].reset_link_counts()
        N['train'][current_network].load_traffic_counts(counts=xc_simulated)

        xc = xc_simulated

        print('Synthetic observed links counts:')

        dict_observed_link_counts = {link_key: np.round(count,1) for link_key, count in xc_simulated.items() if not np.isnan(count)}

        print(pd.DataFrame({'link_key': dict_observed_link_counts.keys(), 'counts': dict_observed_link_counts.values()}).to_string())

        # print([round(float(link.traveltime),2) for link in N['train'][current_network].links])

        # N['train'][current_network].links[0].bpr.bpr_function_x(1000)

        # N['train'][current_network].links[0].bpr.bpr_function_x.


        # Keep zeros in diagonal of OD matrix for visualization purposes
        od_df['r'+str(replicate)] = isl.networks.denseQ(Q=N['train'][current_network].Q_noisy, remove_zeros=False).flatten()
        links_flows_df['r'+str(replicate)] =  xc_simulated.values()
        path_flows_df['r'+str(replicate)] = N['train'][current_network].path_flows

        links_traveltimes_df['r'+str(replicate)] = [round(float(link.traveltime),2) for link in N['train'][current_network].links]



    od_df.to_csv(config.paths['output_folder'] + 'network-data/SUE' + '/' + 'demand' + '.csv', sep=',', encoding='utf-8', index=False)

    links_traveltimes_df.to_csv(config.paths['output_folder'] + 'network-data/SUE' + '/' + 'travel_times' + '.csv', sep=',', encoding='utf-8', index=False)

    links_flows_df.to_csv(config.paths['output_folder'] + 'network-data/SUE' + '/' + 'link_flows' + '.csv', sep=',', encoding='utf-8', index=False)

    path_flows_df.to_csv(config.paths['output_folder'] + 'network-data/SUE' + '/' + 'path_flows' + '.csv', sep=',', encoding='utf-8', index=False)




# =============================================================================
# 3) DATA DESCRIPTION AND CURATION
# =============================================================================

# =============================================================================
# 3a) LINK COUNTS AND TRAVERSING PATHS
# =============================================================================

#Initial coverage
x_bar = np.array(list(xc.values()))[:, np.newaxis]

# If no path are traversing some link observations, they are set to nan values
xc = isl.estimation.masked_link_counts_after_path_coverage(N['train'][current_network], xct = xc)

N['train'][current_network].reset_link_counts()
N['train'][current_network].load_traffic_counts(counts=xc)

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
# total_links = np.array(list(counts.values())).shape[0]
#
# print('\nlink counts observations: ' + str(total_counts_observations))
# print('link coverage: ' + "{:.1%}". format(round(total_counts_observations/total_links,4)))

if config.sim_options['simulated_counts'] is True:
 print('Std set to simulate link observations: ' + str(config.sim_options['noise_params']['sd_x']))


# =============================================================================
# 3c) DESCRIPTIVE STATISTICS
# =============================================================================
summary_table_links_df = isl.descriptive_statistics.summary_table_links(links = N['train'][current_network].get_observed_links()
                                                                        , Z_attrs = ['wt', 'tt', 'c']
                                                                        # , Z_labels = ['incidents', 'income [1K USD]', 'high_inc', 'speed_avg [mi/hr]', 'tt_sd', 'tt_sd_adj', 'tt_var','stops', 'ints']
                                                                        )

with pd.option_context('display.float_format', '{:0.1f}'.format):
    print(summary_table_links_df.to_string())


# Write log file
isl.writer.write_csv_to_log_folder(df = summary_table_links_df, filename = 'summary_table_links_df'
                          , log_file = config.log_file)