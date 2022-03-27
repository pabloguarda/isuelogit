# =============================================================================
# 1) SETUP
# =============================================================================

# Internal modules
import transportAI as tai

# External modules
import numpy as np
import os
import sys
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 2) NETWORKS FACTORY
# =============================================================================

# network_name = 'Eastern-Massachusetts'
network_name = 'SiouxFalls'
# network_name =  'Berlin-Friedrichshain'
# network_name =  'Berlin-Mitte-Center'
# network_name =  'Barcelona'

# Reporter of estimation results
estimation_reporter = tai.writer.Reporter(foldername=network_name, seed = 2022)

# =============================================================================
# a) READ TNTP DATA
# =============================================================================

# Read input data files
links_df = tai.reader.read_tntp_linkdata(
    folderpath=os.getcwd() + "/input/public/networks/github/",
    subfoldername=network_name)

# Add link key
links_df['link_key'] = [(i, j, '0') for i, j in zip(links_df['init_node'], links_df['term_node'])]

# =============================================================================
# b) CREATION OF NETWORK
# =============================================================================

# Create Network Generator
network_generator = tai.factory.NetworkGenerator()

A = network_generator.generate_adjacency_matrix(links_keys=list(links_df['link_key'].values))

tntp_network = network_generator.build_network(A=A,network_name=network_name)

# =============================================================================
# d) ENDOGENOUS LINK ATTRIBUTES
# =============================================================================

# Set BPR functions among links
bpr_parameters_df = pd.DataFrame({'link_key': tntp_network.links_dict.keys(),
                                  'alpha': links_df.b,
                                  'beta': links_df.power,
                                  'tf': links_df.free_flow_time,
                                  'k': links_df.capacity
                                  })

# bpr_parameters_df['tf'] = preprocessing.scale(bpr_parameters_df['tf'],
#                                               with_mean=True,
#                                               with_std=True,
#                                               axis=0)

tntp_network.set_bpr_functions(bprdata=bpr_parameters_df)

# =============================================================================
# c) EXOGENOUS LINK ATTRIBUTES
# =============================================================================

# Extract data on link features
link_features_df = links_df[['link_key','length', 'speed', 'link_type', 'toll']]

# Create data generator
linkdata_generator = tai.factory.LinkDataGenerator()

# Generate synthetic link attributes
synthetic_features_df = linkdata_generator.simulate_features(links=tntp_network.links,
                                                             features_Z= ['c', 'w', 's'],
                                                             option='continuous',
                                                             range=(0, 1))

# - Number of attributes that will be set to 0, which moderate sparsity: with 20 at least, we observe benefits of regularize
n_sparse_features = 1  # 10 #20 #50
sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

sparse_features_df = linkdata_generator.simulate_features(
    links=tntp_network.links,
    features_Z=sparse_features_labels,
    option='continuous',
    range=(-1, 1))

# Merge dataframes with existing dataframe
link_features_df = link_features_df.merge(synthetic_features_df, left_on='link_key', right_on='link_key')
link_features_df = link_features_df.merge(sparse_features_df, left_on='link_key', right_on='link_key')

# Load features data
tntp_network.load_features_data(linkdata=link_features_df)

# =============================================================================
# e) UTILITY FUNCTION
# =============================================================================

utility_function = tai.estimation.UtilityFunction(features_Y=['tt'],
                                               # features_Z= [],
                                               features_Z=['c', 's'],
                                               # features_Z= ['s', 'c'],
                                               # initial_values={'tt': -0.5, 'c': -4, 's': -2},
                                               # initial_values={'tt': -1, 'c': -6, 's': -2},
                                               # initial_values={'tt': -1.4, 'c': -6.4},
                                               true_values={'tt': -1, 'c': -6, 's': -3}
                                               )

# Add sparse features
utility_function.add_sparse_features(Z=sparse_features_labels)

# =============================================================================
# f) OD
# =============================================================================

# Read od matrix
Q = tai.reader.read_tntp_od(folderpath=os.getcwd() + "/input/public/networks/github/",
                            subfoldername=network_name)

# Load O-D matrix
tntp_network.load_OD(Q= Q)

# =============================================================================
# g) PATHS
# =============================================================================

# Create path generator
paths_generator = tai.factory.PathsGenerator()

# Generate and Load paths in network
paths_generator.load_k_shortest_paths(network=tntp_network, k=3)

# =============================================================================
# j) GENERATION OF SYNTHETIC COUNTS
# =============================================================================

equilibrator = tai.equilibrium.LUE_Equilibrator(network=tntp_network,
                                                utility_function=utility_function,
                                                uncongested_mode=True,
                                                max_iters=100,
                                                method='fw',
                                                iters_fw=100,
                                                search_fw='grid'
                                                # , path_size_correction = 20
                                                )

# Generate synthetic traffic counts

counts, _ = linkdata_generator.simulate_counts(network=tntp_network,
                                               equilibrator=equilibrator,
                                               noise_params={'mu_x': 0, 'sd_x': 0},
                                               coverage=0.75
                                               )
tntp_network.load_traffic_counts(counts=counts)

# =============================================================================
# 3) DESCRIPTIVE STATISTICS
# =============================================================================

# =============================================================================
# a) Networks topology
# =============================================================================

tai.descriptive_statistics.summary_table_networks([tntp_network])

# =============================================================================
# b) LINK COUNTS AND TRAVERSING PATHS
# =============================================================================

tai.descriptive_statistics.adjusted_link_coverage(network=tntp_network, counts=counts)
# =============================================================================
# c) Links features and counts
# =============================================================================

summary_table_links_df = tai.descriptive_statistics.summary_table_links(links=tntp_network.links)

with pd.option_context('display.float_format', '{:0.1f}'.format):
    print(summary_table_links_df.to_string())

estimation_reporter.write_table(df = summary_table_links_df, filename = 'summary_table_links.csv', float_format = '%.3f')


# =============================================================================
# 5) BILEVEL OPTIMIZATION
# =============================================================================

outer_optimizer_norefined = tai.estimation.OuterOptimizer(
    method='ngd',
    iters=1,  # 10
    eta=1e-1,
    # path_size_correction = 1
)

learner_norefined = tai.estimation.Learner(
    equilibrator=equilibrator,
    outer_optimizer=outer_optimizer_norefined,
    utility_function=utility_function,
    network=tntp_network,
    name='norefined'
)

outer_optimizer_refined = tai.estimation.OuterOptimizer(
    # method='gauss-newton',
    method='lm-revised',
    # method='ngd',
    # eta=1e-2,
    iters=1,
    # path_size_correction = 1
)

learner_refined = tai.estimation.Learner(
    network=tntp_network,
    equilibrator=equilibrator,
    outer_optimizer=outer_optimizer_refined,
    utility_function=utility_function,
    name='refined'
)

# =============================================================================
# BENCHMARK PREDICTIONS
# =============================================================================

# Naive prediction using mean counts
mean_counts_prediction_loss, mean_count_benchmark_model, \
    = tai.estimation.mean_count_prediction(counts=np.array(list(counts.values()))[:, np.newaxis])

print('\nObjective function under mean count prediction: ' + '{:,}'.format(round(mean_counts_prediction_loss, 1)))

# Naive prediction using uncongested network
equilikely_prediction_loss, x_eq_equilikely \
    = tai.estimation.loss_counts_uncongested_network(
    network = tntp_network,
    equilibrator=equilibrator,
    counts=tntp_network.counts_vector,
    utility_function=utility_function)

print('Objective function under equilikely route choices: ' + '{:,}'.format(round(equilikely_prediction_loss, 1)))

# =============================================================================
# 3d) ESTIMATION
# =============================================================================

print('\nStatistical Inference with no refined solution')

learning_results_norefined, inference_results_norefined, best_iter_norefined = \
    learner_norefined.statistical_inference(h0=0, bilevel_iters=10, alpha=0.05, iteration_report = True)

theta_norefined = learning_results_norefined[best_iter_norefined]['theta']

print('\nStatistical Inference with refined solution')

learner_refined.utility_function.initial_values = theta_norefined

learning_results_refined, inference_results_refined, best_iter_refined = \
    learner_refined.statistical_inference(h0=0, bilevel_iters=10, alpha=0.05, iteration_report = True)

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
    network=tntp_network,
    learners=[learner_norefined, learner_refined],
    linkdata_generator=linkdata_generator,
    utility_function=utility_function)

# Write tables with results on learning and inference
estimation_reporter.write_learning_tables(
    results_norefined=learning_results_norefined,
    results_refined=learning_results_refined,
    network = tntp_network,
    utility_function = utility_function,
    simulated_data = True)

estimation_reporter.write_inference_tables(
    results_norefined=inference_results_norefined,
    results_refined=inference_results_refined,
    float_format = '%.3f')

# =============================================================================
# 6) VISUALIZATIONS
# =============================================================================

# Convergence

results_df = tai.descriptive_statistics \
    .get_loss_and_estimates_over_iterations(results_norefined=learning_results_norefined
                                            , results_refined=learning_results_refined)

fig = tai.visualization.Artist().convergence(
    results_norefined_df=results_df[results_df['stage'] == 'norefined'],
    results_refined_df=results_df[results_df['stage'] == 'refined'],
    simulated_data= True,
    filename='convergence_' + tntp_network.key,
    methods=[outer_optimizer_norefined.method.key, outer_optimizer_refined.method.key],
    theta_true = utility_function.true_values,
    folder = estimation_reporter.dirs['estimation_folder']
)

fig.savefig(estimation_reporter.dirs['estimation_folder'] + '/' + 'bilevel_optimization_convergence.pdf',
            pad_inches=0.1, bbox_inches="tight")

plt.show()

plt.close(fig)

# Distribution of errors across link counts

best_x_norefined = np.array(list(learning_results_norefined[best_iter_refined]['x'].values()))[:, np.newaxis]
best_x_refined = np.array(list(learning_results_refined[best_iter_refined]['x'].values()))[:, np.newaxis]

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize=(10, 5))

# We can set the number of bins with the `bins` kwarg
axs[0].hist(tai.estimation.error_by_link(observed_counts=tntp_network.counts_vector, predicted_counts=best_x_norefined))
axs[1].hist(tai.estimation.error_by_link(observed_counts=tntp_network.counts_vector, predicted_counts=best_x_refined))

for axi in [axs[0], axs[1]]:
    axi.tick_params(axis='x', labelsize=16)
    axi.tick_params(axis='y', labelsize=16)

fig.savefig(estimation_reporter.dirs['estimation_folder'] + '/' + 'distribution_predicted_count_error.pdf',
            pad_inches=0.1, bbox_inches="tight")

plt.show()

plt.close(fig)

# Heatmap O-D matrix
rows, cols = tntp_network.Q.shape

od_df = pd.DataFrame({'origin': pd.Series([], dtype=int)
                         , 'destination': pd.Series([], dtype=int)
                         , 'trips': pd.Series([], dtype=int)})

counter = 0
for origin in range(0, rows):
    for destination in range(0, cols):
        # od_df.loc[counter] = [(origin+1,destination+1), N['train'][current_network].Q[(origin,destination)]]
        od_df.loc[counter] = [int(origin + 1), int(destination + 1), tntp_network.Q[(origin, destination)]]
        counter += 1

od_df.origin = od_df.origin.astype(int)
od_df.destination = od_df.destination.astype(int)

od_pivot_df = od_df.pivot_table(index='origin', columns='destination', values='trips')

# uniform_data = np.random.rand(10, 12)
fig, ax = plt.subplots()
ax = sns.heatmap(od_pivot_df, linewidth=0.5, cmap="Blues")

plt.show()

fig.savefig(estimation_reporter.dirs['estimation_folder'] + '/' + 'heatmap_OD_matrix.pdf',
            pad_inches=0.1, bbox_inches="tight")
plt.close(fig)

sys.exit()
