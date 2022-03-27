# =============================================================================
# 1) SETUP
# =============================================================================

# Set seed for reproducibility and consistency between experiments
import numpy as np
import random

np.random.seed(2020)
random.seed(2020)

#=============================================================================
# 1a) MODULES
#==============================================================================

# Internal modules
import isuelogit as isl

# External modules
import sys
import pandas as pd
from sklearn import preprocessing

import matplotlib.pyplot as plt

# =============================================================================
# 1b) CONFIGURATION
# =============================================================================

config = isl.config.Config(network_key = 'N3')

# Key internal objects for analysis and visualization
artist = isl.visualization.Artist(folder_plots=config.plots_options['folder_plots'],
                                  dim_subplots=config.plots_options['dim_subplots'])

# =============================================================================
# 1c) LOG-FILE
# =============================================================================

# Record starting date and time of the simulation
# https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python

if config.experiment_options['experiment_mode'] is None:
    config.set_log_file(networkname=config.sim_options['current_network'].lower())

# =============================================================================
# 2) NETWORKS FACTORY
# =============================================================================

# Dictionary to store network objects for random networks
network_name = 'N1'

# =============================================================================
# a) CREATION OF RANDOM NETWORKS
# =============================================================================

# ii) RANDOM NETWORK CREATION

# n_random_networks = 4
# network_names =  ['N' + str(i) for i in
#           list(np.arange(1, n_random_networks + 1))]

# Create Network Generator
network_generator = isl.factory.NetworkGenerator()

# Create transportation network with randomly generated adjacency matrix
random_network = network_generator.build_random_network(network_name= network_name,
                                                        nodes_range = (8,8))



# =============================================================================
# c) BEHAVIORAL PARAMETERS AND UTILITY FUNCTIONS
# =============================================================================

utility_parameters = isl.estimation.Parameters(features_Y=['tt'],
                                               features_Z= [],
                                               # features_Z= ['c'],
                                               true_values={'tt': -1, 'c': -2},
                                               # initial_values =  {'tt': -1, 'c': -6},
                                               initial_values =  {'tt': 0}
                                               )

utility_function = isl.estimation.UtilityFunction(utility_parameters)

# =============================================================================
# b) EXOGENOUS LINK ATTRIBUTES
# =============================================================================

# Set Link Performance functions and link level attributes

# Create data generator to generate synthetic link attributes
linkdata_generator = isl.factory.LinkDataGenerator()

# Generate synthetic link attributes
link_features_df = linkdata_generator.simulate_features(
    links = random_network.links,
    features_Z= ['c','w'],
    option = 'discrete',
    range=(0, 1),
    normalization=False
)

# - Number of attributes that will be set to 0, which moderate sparsity: with 20 at least, we observe benefits of regularize
n_sparse_features = 3 #10 #20 #50
sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

sparse_features_df = linkdata_generator.simulate_features(
    links = random_network.links,
    features_Z= sparse_features_labels,
    option = 'continuous',
    range = (-1,1),
    normalization = False
)

#Merge data with existing dataframe
link_features_df = link_features_df.merge(sparse_features_df,
                                          left_on = 'key',
                                          right_on = 'key')

#Load features data
random_network.load_features_data(linkdata = link_features_df)

# =============================================================================
# b) ENDOGENOUS LINK ATTRIBUTES
# =============================================================================


# Set random link performance functions
bpr_parameters_df = linkdata_generator.generate_random_bpr_parameters(
    links_keys = random_network.links_keys)

# bpr_parameters_df['tf'] = preprocessing.scale(bpr_parameters_df['tf'],
#                                               with_mean=False,
#                                               with_std=False,
#                                               axis=0)

random_network.set_bpr_functions(bprdata = bpr_parameters_df)

# =============================================================================
# c) OD
# =============================================================================

# Create OD generator (for random networks only)
od_generator = isl.factory.ODGenerator()

Q = od_generator.generate_Q(network = random_network,
                            min_q = 100, #100
                            max_q = 4000, #4000
                            cutoff = 1,
                            sparsity_Q = 0.1)

# Load O-D matrix
random_network.load_OD(Q  = Q)

# =============================================================================
# d) PATHS
# =============================================================================

# Create path generator
paths_generator = isl.factory.PathsGenerator()

# Generate and Load paths in network
paths_generator.load_k_shortest_paths(network = random_network, k=3)


# =============================================================================
# d) INCIDENT MATRICES
# =============================================================================

# Create incidence matrices using the provided OD matrices and the parameters for path and attributes generation
random_network = network_generator.setup_incidence_matrices(network= random_network,
                                                            setup_options = {**config.sim_options,**config.estimation_options},
                                                            writing= dict(config.sim_options['writing'],
                                                        **{'paths': True, 'Q': True, 'C': True, 'M': True, 'D': True}),
                                                            # , generation = dict(config.sim_options, generation = {'Q': False, 'bpr': True, 'Z': True})
                                                            generation=dict(config.sim_options['generation'],
                                                          **{'C': True, 'D': True, 'M': True})
                                                            )



# =============================================================================
# 4) EXPERIMENTS
# =============================================================================

# =============================================================================
# 4c) CONVERGENCE
# ==============================================================================

# Plot the convergence plot but  with curves for both the uncongested and congested case

# config.experiment_options['convergence_experiment'] = True
config.experiment_options['convergence_experiment'] = False

if config.experiment_options['convergence_experiment']:
    # Outer level optimizer

    utility_parameters = isl.estimation.Parameters(features_Y=['tt'],
                                                   features_Z=['c'],
                                                   true_values={'tt': -1e-0, 'c': -6e-0})

    utility_function = isl.estimation.UtilityFunction(utility_parameters)

    n_sparse_features = 2  # 10 #20 #50
    sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

    utility_function.add_sparse_features(Z=sparse_features_labels)
    outer_optimizer_norefined = isl.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=1e-2
    )

    outer_optimizer_refined = isl.estimation.OuterOptimizer(
        method='lm',
        # vdown_lm=9,
        # vup_lm = 10,
        # lambda_lm = 1e3,
        iters=1
    )

    # outer_optimizer_refined = isl.estimation.LUE_OuterOptimizer(
    #     method='ngd',
    #     eta =4e-2,
    #     iters=1
    # )

    convergence_experiment = isl.experiments.ConvergenceExperiment(
        seed=2026,
        config=config,
        name='Convergence Experiment',
        datetime=None,
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        utility_function=utility_function,
        linkdata_generator=isl.factory.LinkDataGenerator(
            noise_params = {'mu_x': 0, 'sd_x': 0.01}),
        equilibrator=isl.equilibrium.LUE_Equilibrator(
            max_iters=100,
            method='fw',
            iters_fw=100,
            accuracy = 1e-10,
        ),
        network=random_network)

    convergence_experiment.run(bilevel_iterations = 10,
                               # range_initial_values = (-1e-1,1e-1)
                               )

    sys.exit()

# ==============================================================================
# 4e) CONGESTION EXPERIMENT
# ==============================================================================

# config.experiment_options['congestion_experiment'] = True
config.experiment_options['congestion_experiment'] = False

if config.experiment_options['congestion_experiment']:

    # # OD level
    # random_network.load_OD(Q=random_network.Q)

    utility_parameters = isl.estimation.Parameters(features_Y=['tt'],
                                                   features_Z=['c'],
                                                   true_values={'tt': -1, 'c': -2})

    utility_function = isl.estimation.UtilityFunction(utility_parameters)

    n_sparse_features = 2  # 10 #20 #50
    sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

    utility_function.add_sparse_features(Z=sparse_features_labels)

    outer_optimizer_norefined = isl.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=1e-1
    )

    outer_optimizer_refined = isl.estimation.OuterOptimizer(
        method='lm',
        # lambda_lm = 1e-1,
        iters=1,
    )

    congestion_experiment = isl.experiments.ODExperiment(
        seed=2020,
        config=config,
        name='Congestion OD Experiment',
        datetime=None,
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        # outer_optimizers=[outer_optimizer_norefined],
        utility_function=utility_function,
        linkdata_generator=isl.factory.LinkDataGenerator(
            noise_params={'mu_x': 0, 'sd_x': 0.03}),
        equilibrator=isl.equilibrium.LUE_Equilibrator(
            max_iters=100,
            method='fw',
            iters_fw=100,
            accuracy = 1e-10,
            uncongested_mode = False,
            exogenous_traveltimes = True,
        ),
        bilevel_iters=10,
        network=random_network)

    congestion_experiment.run(
        replicates = 40,
        range_initial_values = (-1,1),
        # levels = [0.25, 0.5, 1.0],
        # levels=[0.05, 0.10, 0.15],
        levels=[0.2, 0.4, 0.6],
        type = 'congestion')


# ==============================================================================
# 4d) CONSISTENCY (RELEVANT ATTRIBUTES ONLY)
# ==============================================================================
# A reasonable amount or no noise will be added

# config.experiment_options['consistency_experiment'] = True
config.experiment_options['consistency_experiment'] = False

if config.experiment_options['consistency_experiment']:

    # OD level
    random_network.load_OD(Q=1e-1*Q)

    # Outer level optimizer

    utility_parameters = isl.estimation.Parameters(features_Y=['tt'],
                                                   features_Z=['c'],
                                                   true_values={'tt': -1e0, 'c': -2e0})

    utility_function = isl.estimation.UtilityFunction(utility_parameters)

    n_sparse_features = 2 # 10 #20 #50
    sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

    utility_function.add_sparse_features(Z=sparse_features_labels)

    outer_optimizer_norefined_1 = isl.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=1e-1
    )

    outer_optimizer_norefined_2 = isl.estimation.OuterOptimizer(
        method='lm',
        iters=1,
        # lambda_lm = 1e4,
        # vup_lm = 9e4,
        # vdown_lm=10e4,
    )

    outer_optimizer_refined = isl.estimation.OuterOptimizer(
        method='lm',
        iters=1,
        # lambda_lm=1e4,
        # vup_lm=9e4,
        # vdown_lm=10e4,
    )

    consistency_experiment = isl.experiments.ConsistencyExperiment(
        seed=2022,
        config=config,
        name='Consistency Experiment',
        datetime=None,
        outer_optimizers=[outer_optimizer_norefined_1,
                          outer_optimizer_norefined_2,
                          outer_optimizer_refined],
        utility_function=utility_function,
        linkdata_generator=isl.factory.LinkDataGenerator(
            noise_params = {'mu_x': 0, 'sd_x': 0.03}),
        equilibrator=isl.equilibrium.LUE_Equilibrator(
            max_iters=100,
            method='fw',
            iters_fw=100,
            accuracy = 1e-10,
            uncongested_mode = False,
            exogenous_traveltimes = True
        ),
        network=random_network)

    consistency_experiment.run(bilevel_iterations = 10,
                               range_initial_values = (-1e0, 1e0),
                               replicates = 30
                               )

    sys.exit()

# ==============================================================================
# 4d) INCLUSION OF IRRELEVANT ATTRIBUTES
# ==============================================================================
# A reasonable amount or no noise will be added

# config.experiment_options['irrelevant_attributes_experiment'] = True
config.experiment_options['irrelevant_attributes_experiment'] = False

if config.experiment_options['irrelevant_attributes_experiment']:

    # OD level
    random_network.load_OD(Q=1e-1*Q)

    utility_parameters = isl.estimation.Parameters(features_Y=['tt'],
                                                   features_Z=['c'],
                                                   true_values={'tt': -1e0, 'c': -2e0})

    utility_function = isl.estimation.UtilityFunction(utility_parameters)

    n_sparse_features = 2 # 10 #20 #50
    sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

    utility_function.add_sparse_features(Z=sparse_features_labels)

    outer_optimizer_no_refined = isl.estimation.OuterOptimizer(
        method='ngd',
        iters = 1,
        eta=1e-1
    )

    outer_optimizer_refined_1 = isl.estimation.OuterOptimizer(
        method='lm',
        iters=10,
        # lambda_lm=1e-1,
        # lambda_lm = 1e4,
        # vup_lm = 9e4,
        # vdown_lm=10e4,
    )

    outer_optimizer_refined_2 = isl.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=1e-1,
    )

    irrelevant_attributes_experiment = isl.experiments.IrrelevantAttributesExperiment(
        seed=2022,
        config=config,
        name='Irrelevant Atttributes Experiment',
        datetime=None,
        outer_optimizers=[outer_optimizer_no_refined,
                          outer_optimizer_refined_1,
                          outer_optimizer_refined_2],
        utility_function=utility_function,
        linkdata_generator=isl.factory.LinkDataGenerator(
            noise_params = {'mu_x': 0, 'sd_x': 0.03}),
        equilibrator=isl.equilibrium.LUE_Equilibrator(
            max_iters=100,
            method='fw',
            iters_fw=100,
            accuracy = 1e-10,
            uncongested_mode = False,
            exogenous_traveltimes = True
        ),
        network=random_network)

    irrelevant_attributes_experiment.run(bilevel_iterations = 10,
                                         range_initial_values = (-1e-0,1e-0),
                                         replicates = 20
                                         )

    sys.exit()


# ==============================================================================
# 4e) ERROR IN LINK COUNT MEASUREMENTS
# ==============================================================================

config.experiment_options['noisy_counts_experiment'] = True
# config.experiment_options['noisy_counts_experiment'] = False

if config.experiment_options['noisy_counts_experiment']:

    # OD level
    random_network.load_OD(Q=2e-1*Q)

    utility_parameters = isl.estimation.Parameters(features_Y=['tt'],
                                                   features_Z=['c'],
                                                   true_values={'tt': -1, 'c': -2})

    utility_function = isl.estimation.UtilityFunction(utility_parameters)

    n_sparse_features = 2  # 10 #20 #50
    sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

    utility_function.add_sparse_features(Z=sparse_features_labels)
    outer_optimizer_norefined = isl.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=1e-1
    )

    outer_optimizer_refined = isl.estimation.OuterOptimizer(
        method='lm',
        iters=10,
    )

    noisy_counts_experiment = isl.experiments.CountsExperiment(
        seed=2021,
        config=config,
        name='Noisy Counts Experiment',
        datetime=None,
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        # outer_optimizers=[outer_optimizer_norefined],
        utility_function=utility_function,
        linkdata_generator=isl.factory.LinkDataGenerator(),
        equilibrator=isl.equilibrium.LUE_Equilibrator(
            max_iters=100,
            method='fw',
            iters_fw=100,
            accuracy = 1e-10,
            uncongested_mode = False,
            exogenous_traveltimes = True
        ),
        bilevel_iters=10,
        network=random_network)

    noisy_counts_experiment.run(
        replicates = 50,
        # range_initial_values = (-1e0,1e0),
        levels = [0.05,0.10,0.15],
        # levels=[0.1, 0.2, 0.3],
        type = 'noise')


    sys.exit()

# ==============================================================================
# 4e) SENSOR COVERAGE
# ==============================================================================

# config.experiment_options['sensor_coverage_experiment'] = True
config.experiment_options['sensor_coverage_experiment'] = False

if config.experiment_options['sensor_coverage_experiment']:

    # OD level
    random_network.load_OD(Q=1e-1*random_network.Q)

    utility_parameters = isl.estimation.Parameters(features_Y=['tt'],
                                                   features_Z=['c'],
                                                   true_values={'tt': -1, 'c': -2})

    utility_function = isl.estimation.UtilityFunction(utility_parameters)

    n_sparse_features = 2  # 10 #20 #50
    sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

    utility_function.add_sparse_features(Z=sparse_features_labels)

    outer_optimizer_norefined = isl.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=1e-1
    )

    outer_optimizer_refined = isl.estimation.OuterOptimizer(
        method='lm',
        iters=10,
    )

    sensor_coverage_experiment = isl.experiments.CountsExperiment(
        seed=2018,
        config=config,
        name='Sensor Coverage Experiment',
        datetime=None,
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        # outer_optimizers=[outer_optimizer_norefined],
        utility_function=utility_function,
        linkdata_generator=isl.factory.LinkDataGenerator(
            noise_params={'mu_x': 0, 'sd_x': 0.03}),
        equilibrator=isl.equilibrium.LUE_Equilibrator(
            max_iters=100,
            method='fw',
            accuracy = 1e-10,
            iters_fw=100,
            # uncongested_mode = False,
            exogenous_traveltimes=True,
        ),
        bilevel_iters = 10,
        network=random_network)

    sensor_coverage_experiment.run(
        replicates = 20,
        # range_initial_values = (-1,1),
        levels = [0.25,0.5,0.75],
        type = 'coverage')


    sys.exit()

# ==============================================================================
# 4e) ERROR IN OD MATRIX
# ==============================================================================

config.experiment_options['noisy_od_experiment'] = True
# config.experiment_options['noisy_od_experiment'] = False

if config.experiment_options['noisy_od_experiment']:

    # OD level
    random_network.load_OD(Q=1e-1*random_network.Q)

    utility_parameters = isl.estimation.Parameters(features_Y=['tt'],
                                                   features_Z=['c'],
                                                   true_values={'tt': -1, 'c': -2})

    utility_function = isl.estimation.UtilityFunction(utility_parameters)

    n_sparse_features = 2  # 10 #20 #50
    sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

    utility_function.add_sparse_features(Z=sparse_features_labels)

    outer_optimizer_norefined = isl.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=1e-1
    )

    outer_optimizer_refined = isl.estimation.OuterOptimizer(
        method='lm',
        iters=10,
    )

    noisy_od_experiment = isl.experiments.ODExperiment(
        seed=2021,
        config=config,
        name='Noisy OD Experiment',
        datetime=None,
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        # outer_optimizers=[outer_optimizer_norefined],
        utility_function=utility_function,
        linkdata_generator=isl.factory.LinkDataGenerator(
            noise_params={'mu_x': 0, 'sd_x': 0.03}),
        equilibrator=isl.equilibrium.LUE_Equilibrator(
            max_iters=40,
            method='fw',
            iters_fw=10,
            accuracy = 1e-10,
            uncongested_mode = False,
            exogenous_traveltimes=True
        ),
        bilevel_iters=10,
        network=random_network)

    noisy_od_experiment.run(
        replicates = 40,
        range_initial_values = (-1,1),
        levels = [0.05, 0.10, 0.15],
        # levels=[0.25, 0.5, 0.75],
        type = 'noise')

# ==============================================================================
# 4e) SCALE OF  OD MATRIX
# ==============================================================================

config.experiment_options['ill_scaled_od_experiment'] = True
# config.experiment_options['ill_scaled_od_experiment'] = False

if config.experiment_options['ill_scaled_od_experiment']:

    # OD level
    random_network.load_OD(Q=1e-1*random_network.Q)

    utility_parameters = isl.estimation.Parameters(features_Y=['tt'],
                                                   features_Z=['c'],
                                                   true_values={'tt': -1, 'c': -2})

    utility_function = isl.estimation.UtilityFunction(utility_parameters)

    n_sparse_features = 2  # 10 #20 #50
    sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

    utility_function.add_sparse_features(Z=sparse_features_labels)

    outer_optimizer_norefined = isl.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=1e-1
    )

    outer_optimizer_refined = isl.estimation.OuterOptimizer(
        method='lm',
        iters=10,
    )

    ill_scaled_od_experiment = isl.experiments.ODExperiment(
        seed=2021,
        config=config,
        name='Noisy OD Experiment',
        datetime=None,
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        # outer_optimizers=[outer_optimizer_norefined],
        utility_function=utility_function,
        linkdata_generator=isl.factory.LinkDataGenerator(
            noise_params={'mu_x': 0, 'sd_x': 0.03}),
        equilibrator=isl.equilibrium.LUE_Equilibrator(
            max_iters=100,
            method='fw',
            iters_fw=100,
            accuracy = 1e-10,
            uncongested_mode = False,
            exogenous_traveltimes=True
        ),
        bilevel_iters=10,
        network=random_network)

    ill_scaled_od_experiment.run(
        replicates = 50,
        # range_initial_values = (-1,1),
        levels=[0.90, 0.95, 1.05, 1.1],
        # levels=[0.96, 0.98, 1.02, 1.04],
        type = 'scale')
    
# =============================================================================
# 6) LOG FILE
# =============================================================================

# =============================================================================
# 6a) Summary with most relevant options, prediction error, initial parameters, etc
# =============================================================================

isl.writer.write_estimation_report(filename='summary_report'
                                   , config=config
                                   , decimals=3
                                   # , float_format = 2
                                   )

# =============================================================================
# 6b) General options (sim_options, estimation_options)
# =============================================================================

# Update vector with exogenous covariates
# config.estimation_options['features'] = config.estimation_options['features']

# general_dict = {'type': 'sim_option', 'key': 'selected_year', 'value': 2019}
options_df = pd.DataFrame({'group': [], 'option': [], 'value': []})

for key, value in config.sim_options.items():
    options_df = options_df.append(pd.DataFrame({'group': ['sim_options'], 'option': [key], 'value': [value]}),
                                   ignore_index=True)

for key, value in config.estimation_options.items():
    options_df = options_df.append({'group': 'estimation_options', 'option': key, 'value': value}, ignore_index=True)

for key, value in config.gis_options.items():
    options_df = options_df.append({'group': 'gis', 'option': key, 'value': value}, ignore_index=True)

isl.writer.write_csv_to_log_folder(df=options_df,
                                   filename='global_options'
                                   , log_file=config.log_file
                                   , float_format='%.1f'
                                   )

# =============================================================================
# 6c) Analysis of predicted counts and travel time over iterations
# =============================================================================

predicted_link_counts_over_iterations_df \
    = isl.descriptive_statistics.get_predicted_link_counts_over_iterations_df(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt
    , network=small_networks[current_network])

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
isl.writer.write_csv_to_log_folder(df=predicted_link_counts_over_iterations_df,
                                   filename='predicted_link_counts_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.1f'
                                   )

gap_predicted_link_counts_over_iterations_df \
    = isl.descriptive_statistics.get_gap_predicted_link_counts_over_iterations_df(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt
    , network=small_networks[current_network])

isl.writer.write_csv_to_log_folder(df=gap_predicted_link_counts_over_iterations_df,
                                   filename='gap_predicted_link_counts_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.1f'
                                   )

# Travel times
predicted_link_traveltime_over_iterations_df \
    = isl.descriptive_statistics.get_predicted_traveltimes_over_iterations_df(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt
    , network=small_networks[current_network])

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
isl.writer.write_csv_to_log_folder(df=predicted_link_traveltime_over_iterations_df,
                                   filename='predicted_link_traveltimes_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.2f'
                                   )

# =============================================================================
# 6b) Analysis of parameter estimates and loss over iterations
# =============================================================================

# Log file
loss_and_estimates_over_iterations_df \
    = isl.descriptive_statistics.get_loss_and_estimates_over_iterations(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt)

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
isl.writer.write_csv_to_log_folder(df=loss_and_estimates_over_iterations_df,
                                   filename='loss_and_estimates_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.3f'
                                   )

gap_estimates_over_iterations_df \
    = isl.descriptive_statistics.get_gap_estimates_over_iterations(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt
    , theta_true=theta_true[current_network])

isl.writer.write_csv_to_log_folder(df=gap_estimates_over_iterations_df,
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
isl.writer.write_csv_to_log_folder(df=parameter_inference_table, filename='parameter_inference_table'
                                   , log_file=config.log_file)

# F-test and model summary statistics
model_inference_norefined_table.insert(0, 'stage', 'norefined')
model_inference_refined_table.insert(0, 'stage', 'refined')

model_inference_table = model_inference_norefined_table.append(model_inference_refined_table)

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
isl.writer.write_csv_to_log_folder(df=model_inference_table,
                                   filename='model_inference_table'
                                   , log_file=config.log_file)

sys.exit()
