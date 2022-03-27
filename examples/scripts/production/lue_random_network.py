# =============================================================================
# 1) SETUP
# =============================================================================
# External modules
import sys
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Internal modules
import isuelogit as isl

# =============================================================================
# 2) NETWORK FACTORY
# =============================================================================

# Dictionary to store network objects for random networks
network_name = 'N1'

isl.config.set_main_dir(dir = os.getcwd())

# Reporter of estimation results
estimation_reporter = isl.writer.Reporter(foldername=network_name, seed = 2022)

# Create Network Generator
network_generator = isl.factory.NetworkGenerator()

# Create transportation network with randomly generated adjacency matrix
random_network = network_generator.build_random_network(network_name= network_name,
                                                        nodes_range = (10,10))

# =============================================================================
# b) EXOGENOUS LINK ATTRIBUTES
# =============================================================================

# Set Link Performance functions and link level attributes

# Create data generator to generate synthetic link attributes
linkdata_generator = isl.factory.LinkDataGenerator()

# Generate synthetic link attributes
link_features_df = linkdata_generator.simulate_features(
    links = random_network.links,
    features_Z= ['c','s'],
    option = 'continuous',
    range = (0,1))

# - Number of attributes that will be set to 0, which moderate sparsity: with 20 at least, we observe benefits of regularize
n_sparse_features = 2 #10 #20 #50
sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

sparse_features_df = linkdata_generator.simulate_features(
    links = random_network.links,
    features_Z= sparse_features_labels,
    option = 'continuous',
    range = (-1,1))

#Merge data with existing dataframe
link_features_df = link_features_df.merge(sparse_features_df,
                                          left_on = 'link_key',
                                          right_on = 'link_key')

#Load features data
random_network.load_features_data(linkdata = link_features_df)

# =============================================================================
# b) ENDOGENOUS LINK ATTRIBUTES
# =============================================================================

# Set random link performance functions
bpr_parameters_df = linkdata_generator.generate_random_bpr_parameters(
    links_keys = random_network.links_keys)
random_network.set_bpr_functions(bprdata = bpr_parameters_df)

# =============================================================================
# c) BEHAVIORAL PARAMETERS AND UTILITY FUNCTIONS
# =============================================================================

utility_function = isl.estimation.UtilityFunction(features_Y=['tt'],
                                               # features_Z= [],
                                               features_Z= ['c'],
                                               true_values={'tt': -1, 'c': -6},
                                               # initial_values =  {'tt': -1, 'c': -6},
                                               # initial_values =  {'tt': 0}
                                               )


# Add sparse features
utility_function.add_sparse_features(Z = sparse_features_labels)

# =============================================================================
# c) OD
# =============================================================================

# Create OD generator (for random networks only)
od_generator = isl.factory.ODGenerator()

Q = od_generator.generate_Q(network = random_network,
                            min_q = 0, max_q = 100,
                            cutoff = 1,
                            sparsity_Q = 0.2)

# Load O-D matrix
random_network.load_OD(Q  = Q)

# =============================================================================
# d) PATHS
# =============================================================================

# Create path generator
paths_generator = isl.factory.PathsGenerator()

# Generate and Load paths in network
paths_generator.load_k_shortest_paths(network = random_network, k=5, update_incidence_matrices=True)
paths_generator.write_paths(network=random_network, overwrite_input=True)
# paths_generator.read_paths(network=random_network, update_incidence_matrices=True)

# =============================================================================
# j) GENERATION OF SYNTHETIC COUNTS
# =============================================================================

# Generate synthetic traffic counts

counts, counts_withdraw = linkdata_generator.simulate_counts(
    equilibrator = isl.equilibrium.LUE_Equilibrator(
        network = random_network,
        paths_generator = paths_generator,
        utility_function = utility_function,
        uncongested_mode = False,
        max_iters = 100,
        method = 'fw',
        iters_fw = 100,
        # path_size_correction = 2
    ),
    utility_function=utility_function,
    noise_params = {'mu_x': 0, 'sd_x': 0},
    coverage = 0.9
)

random_network.load_traffic_counts(counts=counts)

# =============================================================================
# 3) DESCRIPTIVE STATISTICS
# =============================================================================

# =============================================================================
# a) Networks topology
# =============================================================================

print(isl.descriptive_statistics.summary_table_networks([random_network]))
# Print Latex Table
# networks_df.to_latex(index=False))

# =============================================================================
# c) Links features and counts
# =============================================================================

summary_table_links_df = isl.descriptive_statistics.summary_table_links(links=random_network.links)

with pd.option_context('display.float_format', '{:0.1f}'.format):
    print(summary_table_links_df.to_string())

estimation_reporter.write_table(df = summary_table_links_df, filename = 'summary_table_links.csv', float_format = '%.3f')

# =============================================================================
# 5) BILEVEL OPTIMIZATION
# =============================================================================

# Generate new paths in network
paths_generator.load_k_shortest_paths(network = random_network, k=5)

# Random initilization of initial estimate
# utility_function.random_initializer((-0.1,0.1))

equilibrator_norefined = isl.equilibrium.LUE_Equilibrator(
    network = random_network,
    paths_generator=paths_generator,
    uncongested_mode = False,
    max_iters = 100,
    method = 'fw',
    iters_fw = 100,
    column_generation = {'n_paths': 3,
                         'ods_coverage': 0.07,
                         # 'ods_sampling':'random',
                         'paths_selection': 3
                         },
    path_size_correction = 2
)

outer_optimizer_norefined = isl.estimation.OuterOptimizer(
    method= 'ngd',
    iters= 1,  # 10
    eta= 5e-1,
    # path_size_correction = 1
)


learner_norefined = isl.estimation.Learner(
    equilibrator = equilibrator_norefined,
    outer_optimizer= outer_optimizer_norefined,
    utility_function = utility_function,
    network = random_network,
    name = 'norefined'
)

equilibrator_refined = isl.equilibrium.LUE_Equilibrator(
    network = random_network,
    paths_generator=paths_generator,
    uncongested_mode = True,
    max_iters = 100,
    method = 'fw',
    iters_fw = 100,
    # column_generation = {'n_paths': 2, 'ods_coverage': 1, 'paths_selection': 2},
    path_size_correction = 2
)

outer_optimizer_refined = isl.estimation.OuterOptimizer(
    # method='gauss-newton',
    method='lm',
    # eta=5e-2,
    # lmabda_lm = 1e0,
    iters=10,
    # path_size_correction = 1
)

learner_refined = isl.estimation.Learner(
    network=random_network,
    equilibrator=equilibrator_refined,
    outer_optimizer=outer_optimizer_refined,
    utility_function=utility_function,
    name = 'refined'
)

# =============================================================================
# BENCHMARK PREDICTIONS
# =============================================================================

# Naive prediction using mean counts
mean_counts_prediction_loss, mean_count_benchmark_model, \
    = isl.estimation.mean_count_prediction(counts=np.array(list(counts.values()))[:, np.newaxis])

print('\nObjective function under mean count prediction: ' + '{:,}'.format(round(mean_counts_prediction_loss, 1)))

# Naive prediction using uncongested network
equilikely_prediction_loss, predicted_counts_equilikely = isl.estimation.loss_counts_equilikely_choices(
    network = random_network,
    equilibrator=equilibrator_refined,
    counts=random_network.counts_vector,
    utility_function=utility_function
)

print('Objective function under equilikely route choices: ' + '{:,}'.format(round(equilikely_prediction_loss, 1)))

# =============================================================================
# 3d) ESTIMATION
# =============================================================================

# ii) NO REFINED OPTIMIZATION AND INFERENCE WITH FIRST ORDER OPTIMIZATION METHODS

# bilevel_estimation_norefined = isl.estimation.LUE_Learner(config.theta_0)

print('\nStatistical Inference in no refined stage')

learning_results_norefined, inference_results_norefined, best_iter_norefined = \
    learner_norefined.statistical_inference(h0 = 0, bilevel_iters = 10, alpha = 0.05,
                                            iteration_report = True, constrained_optimization = True)

theta_norefined = learning_results_norefined[best_iter_norefined]['theta']
# theta_norefined = learning_results_norefined[list(learning_results_norefined.keys())[-1]]['theta']

utility_function_full_model = isl.estimation.UtilityFunction(
    features_Y=['tt'],
    features_Z= ['c','k0'],
    initial_values={'tt': 0},
    signs = {'tt':'-', 'c':'-'}
)

# Update utility functions of no refined and refined learners
learner_norefined.utility_function = utility_function_full_model
learner_refined.utility_function = utility_function_full_model

#Initialize value with the estimate obtained from b)
learner_norefined.utility_function.initial_values = theta_norefined

features_Z, features_Y = isl.estimation.feature_selection(utility_function_full_model,
                                                          theta = theta_norefined,
                                                          criterion = 'sign')


# paths_generator.write_paths(network=random_network, overwrite_input=True)

print('\nStatistical Inference in refined stage')

learner_refined.utility_function.initial_values = theta_norefined

learning_results_refined, inference_results_refined, best_iter_refined = \
    learner_refined.statistical_inference(h0=0,
                                          bilevel_iters=10,
                                          alpha=0.05,
                                          iteration_report = True)

# =============================================================================
# 6) REPORTS
# =============================================================================

estimation_reporter.add_items_report(
    theta_norefined=theta_norefined,
    theta_refined=learning_results_refined[best_iter_refined]['theta'],
    best_objective_norefined = learning_results_norefined[best_iter_norefined]['objective'],
    best_objective_refined = learning_results_refined[best_iter_refined]['objective'],
    mean_count=mean_count_benchmark_model,
    mean_counts_prediction_loss = mean_counts_prediction_loss,
    equilikely_prediction_loss = equilikely_prediction_loss
)

# Summary with most relevant options, prediction error, initial parameters, etc
estimation_reporter.write_estimation_report(
    network=random_network,
    learners=[learner_norefined, learner_refined],
    linkdata_generator=linkdata_generator,
    utility_function=utility_function)

# Write tables with results on learning and inference
estimation_reporter.write_learning_tables(
    results_norefined=learning_results_norefined,
    results_refined=learning_results_refined,
    network = random_network,
    utility_function = utility_function,
    simulated_data = True)

estimation_reporter.write_inference_tables(
    results_norefined=inference_results_norefined,
    results_refined=inference_results_refined,
    float_format = '%.3f')

# =============================================================================
# VISUALIZATIONS
# =============================================================================

# Convergence

results_df = isl.descriptive_statistics \
    .get_loss_and_estimates_over_iterations(results_norefined=learning_results_norefined
                                            , results_refined=learning_results_refined)

fig = isl.visualization.Artist().convergence(
    results_norefined_df=results_df[results_df['stage'] == 'norefined'],
    results_refined_df=results_df[results_df['stage'] == 'refined'],
    simulated_data= True,
    filename='convergence_' + random_network.key,
    methods=[outer_optimizer_norefined.method.key, outer_optimizer_refined.method.key],
    theta_true = utility_function.true_values,
    folder = estimation_reporter.dirs['estimation_folder']
)

plt.show()

fig.savefig(estimation_reporter.dirs['estimation_folder'] + '/' + 'bilevel_optimization_convergence.pdf',
            pad_inches=0.1, bbox_inches="tight")
plt.close(fig)

# Distribution of errors across link counts
best_x_norefined = np.array(list(learning_results_norefined[best_iter_refined]['x'].values()))[:, np.newaxis]
best_x_refined = np.array(list(learning_results_refined[best_iter_refined]['x'].values()))[:, np.newaxis]

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(10, 5))

# We can set the number of bins with the `bins` kwarg
axs[0].hist(isl.estimation.error_by_link(observed_counts=random_network.counts_vector, predicted_counts=best_x_norefined))
axs[1].hist(isl.estimation.error_by_link(observed_counts=random_network.counts_vector, predicted_counts=best_x_refined))

for axi in [axs[0], axs[1]]:
    axi.tick_params(axis='x', labelsize=16)
    axi.tick_params(axis='y', labelsize=16)

fig.savefig(estimation_reporter.dirs['estimation_folder'] + '/' + 'distribution_predicted_count_error.pdf',
            pad_inches=0.1, bbox_inches="tight")
plt.close(fig)

sys.exit()