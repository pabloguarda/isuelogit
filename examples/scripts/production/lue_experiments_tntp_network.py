# =============================================================================
# 1) SETUP
# =============================================================================

# Internal modules
import transportAI as tai

# External modules
import os
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Global parameters for experiments
_BILEVEL_ITERS = 2
_REPLICATES = 2
_ALPHA = 0.1
_RANGE_INITIAL_VALUES  = (-1,1) # None
_SD_X = 0.03
_ETA_NGD = 1e-1
_SCALE_OD = 1 # 0.1
_N_SPARSE_FEATURES = 6
_SEED = 2022
_SHOW_REPLICATE_PLOT = False

# Experiments
list_experiments  = ['pseudoconvexity', 'convergence','congestion','consistency', 'irrelevant_attributes',
                     'noisy_counts','sensor_coverage','noisy_od','ill_scaled_od']

run_experiment = dict.fromkeys(list_experiments,True)
# run_experiment = dict.fromkeys(list_experiments,False)

run_experiment['pseudoconvexity'] = False
# run_experiment['convergence'] = True
# run_experiment['congestion'] = True
# run_experiment['consistency'] = True
# run_experiment['irrelevant_attributes'] = True
# run_experiment['sensor_coverage'] = True
# run_experiment['noisy_counts'] = True
# run_experiment['noisy_od'] = True
# run_experiment['ill_scaled_od'] = True
# =============================================================================
# 2) NETWORKS FACTORY
# =============================================================================

network_name = 'SiouxFalls'
# network_name = 'Eastern-Massachusetts'

# =============================================================================
# a) READ TNTP DATA
# =============================================================================

# Read input data files
links_df = tai.reader.read_tntp_linkdata(
    folderpath=os.getcwd() + "/input/public/networks/github/",
    subfoldername= network_name)

# Add link key
links_df['link_key'] = [(i, j, '0') for i, j in zip(links_df['init_node'], links_df['term_node'])]

# =============================================================================
# b) CREATION OF NETWORK
# =============================================================================

# Create Network Generator
network_generator = tai.factory.NetworkGenerator()

# Create adjacency matrix
A = network_generator.generate_adjacency_matrix(links_keys = list(links_df['link_key'].values))

# Create network
tntp_network = network_generator.build_network(A = A,network_name= network_name)

# =============================================================================
# c) EXOGENOUS LINK ATTRIBUTES
# =============================================================================

# Extract data on link features
link_features_df = links_df[['link_key', 'speed', 'toll', 'link_type']]

#Load features data
tntp_network.load_features_data(linkdata = link_features_df)

# =============================================================================
# d) ENDOGENOUS LINK ATTRIBUTES
# =============================================================================

# Create BPR functions among links using parameters read from TNTP file
bpr_parameters_df = pd.DataFrame({'link_key': tntp_network.links_dict.keys(),
                                  'alpha': links_df.b,
                                  'beta': links_df.power,
                                  'tf': links_df.free_flow_time,
                                  'k': links_df.capacity
                                  })

bpr_parameters_df['tf'] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(np.array(bpr_parameters_df['tf']).reshape(-1, 1)))

tntp_network.set_bpr_functions(bprdata = bpr_parameters_df)

# =============================================================================
# f) OD
# =============================================================================

# Read od matrix
Q = tai.reader.read_tntp_od(folderpath = os.getcwd() + "/input/public/networks/github/", subfoldername = network_name)

# Load O-D matrix
tntp_network.load_OD(Q  = Q)

# =============================================================================
# g) PATHS
# =============================================================================

# Create path generator
paths_generator = tai.factory.PathsGenerator()

# Generate and Load paths in network
paths_generator.load_k_shortest_paths(network = tntp_network, k=3)

# =============================================================================
# g) EQUILIBRATOR
# =============================================================================

equilibrator = tai.equilibrium.LUE_Equilibrator(
    network = tntp_network,
    max_iters=100,
    method='fw',
    iters_fw=100,
    accuracy=1e-10,
    uncongested_mode = True,
    exogenous_traveltimes=True,
    paths_generator=paths_generator
)

# =============================================================================
# g) Utility function
# =============================================================================

utility_function = tai.estimation.UtilityFunction(features_Y=['tt'],
                                                  features_Z=['c', 's'],
                                                  true_values={'tt': -1, 'c': -6, 's': -3})


# =============================================================================
# 3) DESCRIPTIVE STATISTICS
# =============================================================================

# =============================================================================
# a) SUMMARY OF NETWORK CHARACTERISTICS
# =============================================================================
tai.descriptive_statistics.summary_table_networks([tntp_network])

# =============================================================================
# 4) EXPERIMENTS
# ==============================================================================

# =============================================================================
# a) PSEUDO-CONVEXITY
# ==============================================================================

if run_experiment['pseudoconvexity']:

    pseudoconvexity_experiment = tai.experiments.PseudoconvexityExperiment(
        seed=_SEED,
        name='Pseudo-convexity Experiment',
        utility_function= utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'mu_x': 0, 'sd_x': 0*_SD_X}),
        equilibrator=tai.equilibrium.LUE_Equilibrator(paths_generator = paths_generator, uncongested_mode=True),
        network=tntp_network)

    # Generate new random features ('c,'s') and load them in the network
    tntp_network.load_features_data(
        linkdata=pseudoconvexity_experiment.generate_random_link_features(
        n_sparse_features=0,
        normalization={'mean': False, 'std': False}))

    pseudoconvexity_experiment.run(grid = np.arange(-15, 15+0.1, 0.5),
                                   xticks=np.arange(-15, 15 + 0.1, 5),
                                   features=['tt', 'c'],
                                   features_labels = ['travel time', 'monetary cost'],
                                   colors = ['blue','red']
                                   )

# =============================================================================
# b) CONVERGENCE
# ==============================================================================

if run_experiment['convergence']:

    outer_optimizer_norefined = tai.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=_ETA_NGD
    )

    outer_optimizer_refined = tai.estimation.OuterOptimizer(
        method='lm',
        iters=1,
    )

    convergence_experiment = tai.experiments.ConvergenceExperiment(
        seed=_SEED,
        name='Convergence Experiment',
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        utility_function= utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'mu_x': 0, 'sd_x': 0*_SD_X}),
        equilibrator=equilibrator,
        bilevel_iters=_BILEVEL_ITERS,
        network=tntp_network)

    # Generate new random features ('c,'s') and load them in the network
    tntp_network.load_features_data(linkdata=convergence_experiment.generate_random_link_features(n_sparse_features=0))

    # convergence_experiment.run(range_initial_values = _RANGE_INITIAL_VALUES)
    convergence_experiment.run(range_initial_values = None, iteration_report = True)

# ==============================================================================
# c) CONGESTION
# ==============================================================================

if run_experiment['congestion']:

    outer_optimizer_norefined = tai.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=_ETA_NGD
    )

    outer_optimizer_refined = tai.estimation.OuterOptimizer(
        method='lm',
        iters=1,
    )

    congestion_experiment = tai.experiments.ODExperiment(
        seed=_SEED,
        name='Congestion OD Experiment',
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        utility_function=utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'sd_x': _SD_X}),
        equilibrator=equilibrator,
        bilevel_iters=_BILEVEL_ITERS,
        network=tntp_network)

    # Note: Random features ('c,'s') and 3 sparse attributes are generated and loaded in the network at every replicate

    congestion_experiment.run(
        replicates = _REPLICATES,
        range_initial_values = _RANGE_INITIAL_VALUES,
        alpha=_ALPHA,
        n_sparse_features=_N_SPARSE_FEATURES,
        show_replicate_plot=_SHOW_REPLICATE_PLOT,
        levels=[0.4, 0.8, 1.2],
        # levels = [0.2, 0.3, 0.4],
        # levels=[0.05, 0.15, 0.25],
        type = 'congestion')

# =============================================================================
# d) CONSISTENCY AND INFERENCE
# ==============================================================================

if run_experiment['consistency']:

    tntp_network.scale_OD(scale=_SCALE_OD)

    outer_optimizer_no_refined = tai.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta= _ETA_NGD
    )

    outer_optimizer_refined_1 = tai.estimation.OuterOptimizer(
        method='lm',
        iters=1
    )

    outer_optimizer_refined_2 = tai.estimation.OuterOptimizer(
        method='lm',
        iters=1
    )

    consistency_experiment = tai.experiments.ConsistencyExperiment(
        seed=_SEED,
        name='Consistency Experiment',
        equilibrator=equilibrator,
        outer_optimizers=[outer_optimizer_no_refined,
                          outer_optimizer_refined_1,
                          outer_optimizer_refined_2],
        utility_function=utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'sd_x': _SD_X}),
        network=tntp_network)

    # Note: Random features ('c,'s') are generated and loaded in the network at every replicate
    consistency_experiment.run(
        bilevel_iters = _BILEVEL_ITERS,
        range_initial_values = _RANGE_INITIAL_VALUES,
        replicates = _REPLICATES,
        n_sparse_features= 0*_N_SPARSE_FEATURES,
        show_replicate_plot=_SHOW_REPLICATE_PLOT,
        replicate_report = True,
        alpha = _ALPHA
    )


# ==============================================================================
# e) INCLUSION OF IRRELEVANT ATTRIBUTES
# ==============================================================================

if run_experiment['irrelevant_attributes']:

    tntp_network.scale_OD(scale = _SCALE_OD)

    outer_optimizer_no_refined = tai.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=_ETA_NGD
    )

    outer_optimizer_refined_1 = tai.estimation.OuterOptimizer(
        method='lm',
        iters=1
    )

    outer_optimizer_refined_2 = tai.estimation.OuterOptimizer(
        method='lm',
        iters=1
    )

    irrelevant_attributes_experiment = tai.experiments.ConsistencyExperiment(
        seed=_SEED,
        name='Irrelevant Attributes Experiment',
        equilibrator=equilibrator,
        outer_optimizers=[outer_optimizer_no_refined,
                          outer_optimizer_refined_1,
                          outer_optimizer_refined_2],
        utility_function=utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'sd_x': _SD_X}),
        network=tntp_network)

    # Note: Random features ('c,'s') and 3 sparse attributes are generated and loaded in the network at every replicate

    irrelevant_attributes_experiment.run(
        bilevel_iters=_BILEVEL_ITERS,
        range_initial_values=_RANGE_INITIAL_VALUES,
        replicates=_REPLICATES,
        n_sparse_features=_N_SPARSE_FEATURES,
        show_replicate_plot=_SHOW_REPLICATE_PLOT,
        replicate_report = True,
        alpha=_ALPHA
    )

# ==============================================================================
# 4e) ERROR IN LINK COUNT MEASUREMENTS
# ==============================================================================

if run_experiment['noisy_counts']:

    tntp_network.scale_OD(scale=_SCALE_OD)

    outer_optimizer_norefined = tai.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta= _ETA_NGD
    )

    outer_optimizer_refined = tai.estimation.OuterOptimizer(
        iters=1,
        method='lm',
        # method='ngd',
        # eta=_ETA_NGD
    )

    noisy_counts_experiment = tai.experiments.CountsExperiment(
        seed=_SEED,
        name='Noisy Counts Experiment',
        outer_optimizers= [outer_optimizer_norefined, outer_optimizer_refined],
        utility_function=utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(),
        equilibrator=equilibrator,
        bilevel_iters= _BILEVEL_ITERS,
        network=tntp_network)

    # Note: Random features ('c,'s') and 3 sparse attributes are generated and loaded in the network at every replicate

    noisy_counts_experiment.run(
        replicates = _REPLICATES,
        range_initial_values = _RANGE_INITIAL_VALUES,
        alpha = _ALPHA,
        n_sparse_features=_N_SPARSE_FEATURES,
        show_replicate_plot=_SHOW_REPLICATE_PLOT,
        replicate_report = True,
        # levels = [0,0.05,0.10],
        # levels = [0.05,0.1,0.15],
        levels=[0.05, 0.15, 0.25],
        type = 'noise')

# ==============================================================================
# 4e) SENSOR COVERAGE
# ==============================================================================

if run_experiment['sensor_coverage']:

    tntp_network.scale_OD(scale=_SCALE_OD)

    outer_optimizer_norefined = tai.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta= _ETA_NGD
    )

    outer_optimizer_refined = tai.estimation.OuterOptimizer(
        iters=1,
        method='lm',
        # method='ngd',
        # eta=_ETA_NGD
    )

    sensor_coverage_experiment = tai.experiments.CountsExperiment(
        seed=_SEED,
        name='Sensor Coverage Experiment',
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        # outer_optimizers=[outer_optimizer_norefined],
        utility_function=utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'sd_x': _SD_X}),
        equilibrator=equilibrator,
        bilevel_iters=_BILEVEL_ITERS,
        network=tntp_network)

    # Note: Random features ('c,'s') and 3 sparse attributes are generated and loaded in the network at every replicate
    sensor_coverage_experiment.run(
        alpha = _ALPHA,
        n_sparse_features=_N_SPARSE_FEATURES,
        replicates = _REPLICATES,
        range_initial_values = _RANGE_INITIAL_VALUES,
        show_replicate_plot=_SHOW_REPLICATE_PLOT,
        replicate_report = True,
        # levels=[1, 0.75, 0.5, 0.25],
        # levels=[0.25, 0.5, 0.75, 1.0],
        levels=[0.25, 0.5, 0.75],
        # levels = [0.7,0.8,0.9],
        # levels=[0.1, 0.5, 0.75],
        type = 'coverage')

# ==============================================================================
# 4e) ERROR IN OD MATRIX
# ==============================================================================

if run_experiment['noisy_od']:

    tntp_network.scale_OD(scale=_SCALE_OD)

    outer_optimizer_norefined = tai.estimation.OuterOptimizer(
        iters=1,
        method='ngd',
        eta= _ETA_NGD
    )

    outer_optimizer_refined = tai.estimation.OuterOptimizer(
        iters=1,
        method='lm',
        # method='ngd',
        # eta=_ETA_NGD

    )

    noisy_od_experiment = tai.experiments.ODExperiment(
        seed=_SEED,
        name='Noisy OD Experiment',
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        utility_function=utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'sd_x': _SD_X}),
        equilibrator=equilibrator,
        bilevel_iters=_BILEVEL_ITERS,
        network=tntp_network)

    # Note: Random features ('c,'s') and 3 sparse attributes are generated and loaded in the network at every replicate

    noisy_od_experiment.run(
        replicates = _REPLICATES,
        range_initial_values = _RANGE_INITIAL_VALUES,
        levels = [0.1,0.2,0.3],
        # levels=[0.05, 0.1, 0.15],
        alpha=_ALPHA,
        n_sparse_features=_N_SPARSE_FEATURES,
        show_replicate_plot=_SHOW_REPLICATE_PLOT,
        # levels=[0.05, 0.10, 0.15],
        type = 'noise')

# ==============================================================================
# 4e) SCALE OF  OD MATRIX
# ==============================================================================

if run_experiment['ill_scaled_od']:

    tntp_network.scale_OD(scale = _SCALE_OD)

    outer_optimizer_norefined = tai.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=_ETA_NGD
    )

    outer_optimizer_refined = tai.estimation.OuterOptimizer(
        iters=1,
        method='lm',
        # method='ngd',
        # eta=_ETA_NGD
    )

    ill_scaled_od_experiment = tai.experiments.ODExperiment(
        seed=_SEED,
        name='Ill-scaled OD Experiment',
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        utility_function=utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'sd_x': _SD_X}),
        equilibrator=equilibrator,
        bilevel_iters=_BILEVEL_ITERS,
        network=tntp_network)

    # Note: Random features ('c,'s') and 3 sparse attributes are generated and loaded in the network at every replicate

    ill_scaled_od_experiment.run(
        replicates = _REPLICATES,
        range_initial_values = _RANGE_INITIAL_VALUES,
        alpha = _ALPHA,
        n_sparse_features=_N_SPARSE_FEATURES,
        show_replicate_plot=_SHOW_REPLICATE_PLOT,
        # levels=[0.90, 0.95, 1.05, 1.10],
        # levels = [0.96,0.98,1.02,1.04],
        levels=[0.90, 0.95, 1.05, 1.10],
        type = 'scale')

