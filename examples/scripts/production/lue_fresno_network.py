# =============================================================================
# 1) SETUP
# =============================================================================

# Internal modules
import transportAI as tai

# External modules
import ast
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

estimation_options = {}

# If no change in the prediction is produced over iterations in the no-refined stages, those counts are removed
estimation_options['link_selection'] = True
estimation_options['scaling_Q'] = False
estimation_options['ttest_search_theta'] = False
estimation_options['ttest_search_Q'] = False

# =============================================================================
### NETWORKS FACTORY
# ============================================================================
network_name = 'Fresno'

# Estimation reporter
estimation_reporter = tai.writer.Reporter(foldername=network_name, seed = 2022)

# Reader of geospatial and spatio-temporal data
data_reader = tai.etl.DataReader(network_key=network_name)

# First Tuesday of October, 2019
data_reader.select_period(date='2019-10-01', hour=16)

# First Tuesday of October, 2020
# data_reader.select_period(date = '2020-10-06', hour = 16)

# =============================================================================
### READ FRESNO LINK DATA
# =============================================================================

# Read nodes data
nodes_df = pd.read_csv(tai.dirs['input_folder'] + '/network-data/nodes/'  + 'fresno-nodes-data.csv')

# Read nodes spatiotemporal link data
links_df = pd.read_csv(
    tai.dirs['input_folder'] + 'network-data/links/' + str(data_reader.options['selected_date'])+ '-fresno-link-data.csv',
    converters={"link_key": ast.literal_eval,"pems_id": ast.literal_eval})

# =============================================================================
### Build network
# =============================================================================

# Create Network Generator
network_generator = tai.factory.NetworkGenerator()

A = network_generator.generate_adjacency_matrix(links_keys=list(links_df['link_key'].values))

fresno_network = \
    network_generator.build_fresno_network(A=A, links_df=links_df, nodes_df=nodes_df, network_name= network_name)

# =============================================================================
# f) OD
# =============================================================================

# Reading OD matrix that was written internally
network_generator.read_OD(network=fresno_network, sparse=True)

# Total counts matched in 2020: 2223 and in 2021: 2113, which equates a scale factor of 0.95 in 2021
if data_reader.options['selected_year'] == 2020:
    fresno_network.scale_OD(scale = 0.95)

# =============================================================================
# g) PATHS
# =============================================================================

# Create path generator
paths_generator = tai.factory.PathsGenerator()

# # Generate and Load paths in network
# paths_generator.load_k_shortest_paths(network = fresno_network, k=3)

paths_generator.read_paths(network=fresno_network, update_incidence_matrices=True)

# network_generator.read_incidence_matrices(network = fresno_network,
#                                          matrices = {'sparse_C':True, 'sparse_D':True, 'sparse_M':True})

# =============================================================================
# d) LINK PERFORMANCE FUNCTIONS
# =============================================================================

bpr_parameters_df = pd.DataFrame({'link_key': links_df['link_key'],
                                  'alpha': links_df['alpha'],
                                  'beta': links_df['beta'],
                                  'tf': links_df['tf'],
                                  'k': pd.to_numeric(links_df['k'], errors='coerce', downcast='float')
                                  })
fresno_network.set_bpr_functions(bprdata=bpr_parameters_df)


# =============================================================================
# 3c) FEATURE ENGINEERING
# =============================================================================
fresno_network.load_features_data(links_df, link_key = 'link_key')

# Spatio-temporal data must have read before
tai.etl.feature_engineering_fresno(links=fresno_network.links, network=fresno_network)
# ['low_inc', 'high_inc','no_incidents','no_bus_stops','no_intersections','tt_sd_adj','tt_reliability']

features_list = ['median_inc', 'intersections', 'incidents', 'bus_stops', 'median_age',
                 'tt_avg', 'tt_sd','tt_var', 'tt_cv',
                 'speed_ref_avg', 'speed_avg', 'speed_hist_avg','speed_sd','speed_hist_sd','speed_cv',
                 'tt_sd_adj','tt_reliability']


# Normalization of features to range [0,1]
linkdata = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(fresno_network.Z_data[features_list].values))
linkdata.columns = features_list
linkdata.insert(0, 'link_key', fresno_network.links_keys)

fresno_network.load_features_data(linkdata)

# =============================================================================
# TRAFFIC COUNTS
# =============================================================================

# Read counts from csv
counts_df = pd.read_csv(tai.dirs['input_folder'] + '/network-data/links/' \
                            + str(data_reader.options['selected_date']) + '-fresno-link-counts' + '.csv',
                        converters={'link_key': ast.literal_eval})

counts = dict(zip(counts_df['link_key'].values, counts_df['counts'].values))

# Load counts
fresno_network.load_traffic_counts(counts=counts)

# =============================================================================
# e) UTILITY FUNCTION
# =============================================================================

utility_function = tai.estimation.UtilityFunction(
    features_Y=['tt'],
    # features_Z=['tt_cv'],
    # features_Z= ['tt_cv', 'no_incidents', 'no_intersections', 'no_bus_stops', 'low_inc'],
    # features_Z= ['tt_cv', 'incidents', 'intersections', 'bus_stops', 'median_inc'],
    # initial_values={'tt': -2},
    initial_values={'tt': 0},
)

# Features in utility function
# estimation_options['features'] = ['bus_stops', 'incidents','intersections', 'incidents_year']
# estimation_options['features'] = ['no_incidents', 'no_intersections', 'no_bus_stops']
# estimation_options['features'] = ['median_inc', 'low_inc', 'high_inc', 'bus_stops']
# estimation_options['features'] = ['tt_sd','tt_cv','tt_sd_adj', 'tt_reliability', 'speed_sd', 'road_closures']
# Note: an advantage of tt_cv is that is adimensional, which correct for difference in sizes of inrix and networks segments

# =============================================================================
# 3) DESCRIPTIVE STATISTICS
# =============================================================================

# - Report link coverage

total_counts_observations = np.count_nonzero(~np.isnan(np.array(list(counts.values()))))

total_links = np.array(list(counts.values())).shape[0]

print('\nTotal link counts observations: ' + str(total_counts_observations))
print('Link coverage: ' + "{:.1%}".format(round(total_counts_observations / total_links, 4)))

# - Networks topology
tai.descriptive_statistics.summary_table_networks([fresno_network])

# - Selected feature data on observed links
summary_table_observed_links_df = tai.descriptive_statistics.summary_table_links(
    links=fresno_network.get_observed_links(),
    Z_attrs=['speed_avg', 'tt_sd', 'tt_cv', 'tt_reliability', 'incidents', 'median_inc', 'bus_stops', 'intersections'],
    Z_labels=['speed_avg [mi/hr]', 'tt_sd', 'tt_cv', 'tt_reliability', 'incidents', 'income [1K USD]', 'stops', 'ints']
)

with pd.option_context('display.float_format', '{:0.3f}'.format):
    print(summary_table_observed_links_df.to_string())


# - Feature data of all links
estimation_reporter.write_table(df = fresno_network.Z_data, filename = 'links_data.csv', float_format = '%.4f')

# =============================================================================
# 3) BILEVEL OPTIMIZATION IN CONGESTED NETWORK
# =============================================================================

# HEURISTICS FOR SCALING OF OD MATRIX AND SEARCH OF INITIAL LOGIT ESTIMAT

equilibrator = tai.equilibrium.LUE_Equilibrator(
    network=fresno_network,
    utility_function=utility_function,
    uncongested_mode=True,
    max_iters=100,
    method='fw',
    iters_fw=10,
    paths_generator = paths_generator
)

if estimation_options['ttest_search_theta']:

    # utility_function.parameters.values = utility_function.parameters.initial_values

    ttests = tai.estimation.grid_search_theta_ttest(network=fresno_network,
                                                    equilibrator=equilibrator,
                                                    utility_function=utility_function,
                                                    counts = fresno_network.link_data.counts_vector,
                                                    feature='tt',
                                                    grid= np.arange(-3, -4, -0.5),
                                                    # theta_attr_grid= [1,0,-1,-10,-10000],
                                                    # theta_attr_grid=[-5e-1, -1, -2, -3,-3.2,3.4,3.5,-4,-10],
                                                    )

if estimation_options['ttest_search_Q']:

    # utility_function.parameters.values = dict.fromkeys(utility_function.parameters.initial_values.keys(),-1)
    # utility_function.parameters.values = utility_function.parameters.initial_values

    ttests = tai.estimation.grid_search_Q_ttest(network=fresno_network,
                                                equilibrator=equilibrator,
                                                utility_function=utility_function,
                                                counts = fresno_network.link_data.counts_vector,
                                                scales= np.arange(0.5, 1.5, 0.5),
                                                # theta_attr_grid= [1,0,-1,-10,-10000],
                                                # theta_attr_grid=[-5e-1, -1, -2, -3,-3.2,3.4,3.5,-4,-10],
                                                )
# =============================================================================
# 3d) ESTIMATION
# =============================================================================

equilibrator_norefined = tai.equilibrium.LUE_Equilibrator(
    network=fresno_network,
    paths_generator=paths_generator,
    utility_function=utility_function,
    # , uncongested_mode = True
    max_iters=100,
    method='fw',
    iters_fw=10,
    column_generation={'n_paths': 4,
                       'ods_coverage': 0.1,
                       'paths_selection': None},
    path_size_correction=1
)

outer_optimizer_norefined = tai.estimation.OuterOptimizer(
    method='ngd',
    iters=1,
    eta=5e-1,
)

learner_norefined = tai.estimation.Learner(
    equilibrator=equilibrator_norefined,
    outer_optimizer=outer_optimizer_norefined,
    utility_function=utility_function,
    network=fresno_network,
    name = 'norefined'
)

equilibrator_refined = tai.equilibrium.LUE_Equilibrator(
    network=fresno_network,
    paths_generator=paths_generator,
    utility_function=utility_function,
    # , uncongested_mode = True
    max_iters=100,
    method='fw',
    iters_fw=10,
    path_size_correction=1
)

outer_optimizer_refined = tai.estimation.OuterOptimizer(
    method='lm',
    # method='ngd',
    iters=1,
    # eta=5e-1,
)

learner_refined = tai.estimation.Learner(
    network=fresno_network,
    equilibrator=equilibrator_refined,
    outer_optimizer=outer_optimizer_refined,
    utility_function=utility_function,
    name = 'refined'
)

print('\nStatistical Inference in no refined stage')

learning_results_norefined, inference_results_norefined, best_iter_norefined = \
    learner_norefined.statistical_inference(h0=0, bilevel_iters=10, alpha=0.05, link_report=True)

theta_norefined = learning_results_norefined[best_iter_norefined]['theta']

paths_generator.write_paths(network=fresno_network, overwrite_input=False)
network_generator.write_incidence_matrices(network = fresno_network,
                                           matrices = {'sparse_C':True, 'sparse_D':True, 'sparse_M':True},
                                           overwrite_input = False)

# ii) REFINED OPTIMIZATION AND INFERENCE

print('\nStatistical Inference in refined stage')

if estimation_options['link_selection']:
    new_counts, removed_links_keys = tai.etl.get_informative_links_fresno(
        learning_results=learning_results_norefined,
        network=fresno_network)

    fresno_network.load_traffic_counts(new_counts)

learner_refined.utility_function.parameters.initial_values = theta_norefined

learning_results_refined, inference_results_refined, best_iter_refined = \
    learner_refined.statistical_inference(h0=0, bilevel_iters=2, alpha=0.05)

theta_refined = learning_results_refined[best_iter_refined]['theta']

# =============================================================================
# BENCHMARK PREDICTIONS
# =============================================================================

# Naive prediction using mean counts
mean_counts_prediction_loss, mean_count_benchmark_model \
    = tai.estimation.mean_count_prediction(counts=np.array(list(counts.values()))[:, np.newaxis])

print('\nObjective function under mean count prediction: ' + '{:,}'.format(round(mean_counts_prediction_loss, 1)))

# Naive prediction using uncongested network
equilikely_prediction_loss, predicted_counts_equilikely \
    = tai.estimation.loss_counts_equilikely_choices(
    network = fresno_network,
    equilibrator=equilibrator_refined,
    counts=fresno_network.counts_vector,
    utility_function=utility_function)

print('Objective function under equilikely route choices: ' + '{:,}'.format(round(equilikely_prediction_loss, 1)))

# =============================================================================
# 6) REPORTS
# =============================================================================

estimation_reporter.add_items_report(
    selected_date = data_reader.options['selected_date'],
    selected_hour = data_reader.options['selected_hour'],
    selected_od_periods = data_reader.options['od_periods'],
    theta_norefined=theta_norefined,
    theta_refined= theta_refined,
    best_objective_norefined = round(learning_results_norefined[best_iter_norefined]['objective'],1),
    best_objective_refined = round(learning_results_refined[best_iter_refined]['objective'],1),
    mean_counts=round(mean_count_benchmark_model,1),
    mean_counts_prediction_loss = round(mean_counts_prediction_loss,1),
    equilikely_prediction_loss = round(equilikely_prediction_loss,1)
)

# Summary with most relevant options, prediction error, initial parameters, etc
estimation_reporter.write_estimation_report(
    network=fresno_network,
    learners=[learner_norefined, learner_refined],
    utility_function=utility_function)

# Write tables with results on learning and inference
estimation_reporter.write_learning_tables(
    results_norefined=learning_results_norefined,
    results_refined=learning_results_refined,
    network = fresno_network,
    utility_function = utility_function)

estimation_reporter.write_inference_tables(
    results_norefined=inference_results_norefined,
    results_refined=inference_results_refined,
    float_format = '%.3f')

# =============================================================================
# 6) VISUALIZATIONS
# =============================================================================

# - Convergence

results_df = tai.descriptive_statistics \
    .get_loss_and_estimates_over_iterations(results_norefined=learning_results_norefined
                                            , results_refined=learning_results_refined)

fig = tai.visualization.Artist().convergence(
    results_norefined_df=results_df[results_df['stage'] == 'norefined'],
    results_refined_df=results_df[results_df['stage'] == 'refined'],
    filename='convergence_' + fresno_network.key,
    methods=[outer_optimizer_norefined.method.key, outer_optimizer_refined.method.key],
    folder = estimation_reporter.dirs['estimation_folder'],
    simulated_data = False
)

# - Distribution of errors across link counts

best_predicted_counts_norefined = np.array(list(learning_results_norefined[best_iter_norefined]['x'].values()))[:, np.newaxis]
best_predicted_counts_refined = np.array(list(learning_results_refined[best_iter_refined]['x'].values()))[:, np.newaxis]

fig, axs = plt.subplots(1, 2, sharey='all', tight_layout=True, figsize=(10, 5))

# We can set the number of bins with the `bins` kwarg
axs[0].hist(tai.estimation.error_by_link(observed_counts=np.array(list(counts.values()))[:, np.newaxis],
                                         predicted_counts=best_predicted_counts_norefined))
axs[1].hist(tai.estimation.error_by_link(observed_counts=np.array(list(counts.values()))[:, np.newaxis],
                                         predicted_counts=best_predicted_counts_refined))

for axi in [axs[0], axs[1]]:
    axi.tick_params(axis='x', labelsize=16)
    axi.tick_params(axis='y', labelsize=16)

plt.show()

fig.savefig(estimation_reporter.dirs['estimation_folder'] + '/' + 'distribution_predicted_error_counts.pdf',
            pad_inches=0.1, bbox_inches="tight")

# - Map of congestion

# Include capacity and flow in the link, and their ratio

# list(results_norefined_bilevelopt[estimation_options['bilevel_iters_norefined']]['equilibrium']['x'].keys())[34]
# fresno_network.links[34].key

# Congestion shapefile using the links flows obtained in the best iteration
tai.geographer \
    .write_links_congestion_map_shp(
    links=fresno_network.links,
    flows=best_predicted_counts_refined,
    folderpath=tai.dirs['output_folder'] + 'gis/Fresno/network/congestion', networkname='Fresno',
)

# Congestion shapefile using the links flows obtained with a equilikely assignment
tai.geographer \
    .write_links_congestion_map_shp(links=fresno_network.links, flows=predicted_counts_equilikely
                                    , folderpath=tai.dirs['output_folder'] + 'gis/Fresno/network/congestion'
                                    , networkname='Fresno_equilikely'
                                    )







