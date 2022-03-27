# =============================================================================
# 1) SETUP
# =============================================================================
# Internal modules
import transportAI as tai

# External modules
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Common parameters
_BILEVEL_ITERS = 10
_SD_X = 0
_SEED = 2022

list_experiments  = ['monotonicity','pseudoconvexity','convergence','biased_reference_od']
run_experiment = dict.fromkeys(list_experiments,True)
# run_experiment = dict.fromkeys(list_experiments,False)

# run_experiment['monotonicity'] = True
# run_experiment['pseudoconvexity'] = True
# run_experiment['convergence'] = True
run_experiment['biased_reference_od'] = True

# =============================================================================
# 2) NETWORKS FACTORY
# =============================================================================
small_networks = []

# =============================================================================
# 2.1) CREATION OF SMALL NETWORKS
# =============================================================================

network_generator = tai.factory.NetworkGenerator()

network_names = ['Toy', 'Yang', 'Lo', 'Wang']

# Get dictionaries with adjacency and O-D matrices for a set of custom networks.
As,Qs = network_generator.get_A_Q_custom_networks(network_names)

# Create transportation network using adjacency matrices
for i in network_names:
    small_networks.append(network_generator.build_network(A = As[i], network_name= i))

# =============================================================================
# d) ENDOGENOUS LINK ATTRIBUTES
# =============================================================================

# Create data generator to generate link attributes
linkdata_generator = tai.factory.LinkDataGenerator()

for small_network in small_networks:
    if small_network.key == 'Yang':
        bpr_parameters_df = linkdata_generator.generate_Yang_bpr_parameters()
    elif small_network.key == 'Lo':
        bpr_parameters_df=linkdata_generator.generate_LoChan_bpr_parameters()
    elif small_network.key == 'Wang':
        bpr_parameters_df=linkdata_generator.generate_Wang_bpr_parameters()
    elif small_network.key == 'Toy':
        bpr_parameters_df = linkdata_generator.generate_toy_bpr_parameters()

    # # Normalize travel time to avoid numerical issues
    # bpr_parameters_df['tf'] = pd.DataFrame(
    #     preprocessing.MinMaxScaler().fit_transform(np.array(bpr_parameters_df['tf'])[:,np.newaxis]))

    # Set BPR parameters
    small_network.set_bpr_functions(bpr_parameters_df)

# =============================================================================
# c) ODS
# =============================================================================
for small_network in small_networks:
    small_network.load_OD(Q  = Qs[small_network.key])

# =============================================================================
# d) PATHS
# =============================================================================
paths_generator = tai.factory.PathsGenerator()

for small_network in small_networks:
    # With k>=4 shortest paths, all acyclic path of the networks are included in the path sets
    print('\n'+small_network.key, 'network', '\n')
    paths_generator.load_k_shortest_paths(network = small_network, k=4)

# =============================================================================
# 3) DESCRIPTIVE STATISTICS
# =============================================================================
tai.descriptive_statistics.summary_table_networks(small_networks)

# =============================================================================
# 4) EXPERIMENTS
# =============================================================================

# =============================================================================
# MONOTONOCITY OF TRAFFIC COUNT FUNCTIONS
# ==============================================================================

if run_experiment['monotonicity']:

    equilibrator = tai.equilibrium.LUE_Equilibrator(
        max_iters=100,
        method='fw',
        iters_fw=100,
        accuracy=1e-10,
        uncongested_mode = True
    )

    utility_function = tai.estimation.UtilityFunction(features_Y=['tt'], true_values={'tt': -1})

    monotonicity_experiments = tai.experiments.MonotonicityExperiments(
        seed = _SEED,
        name = 'Monotonicity Experiment',
        utility_function = utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'mu_x': 0, 'sd_x': _SD_X}),
        equilibrator = equilibrator,
        networks = small_networks)

    monotonicity_experiments.run(grid=list(np.arange(-15, 15, 0.1)), feature='tt')

# =============================================================================
# PSEUDOCONVEXITY OF OBJECTIVE FUNCTION
# ==============================================================================

if run_experiment['pseudoconvexity']:

    equilibrator = tai.equilibrium.LUE_Equilibrator(
        max_iters=100,
        method='fw',
        iters_fw=100,
        accuracy=1e-10,
        uncongested_mode = True
    )

    utility_function = tai.estimation.UtilityFunction(features_Y=['tt'], true_values={'tt': -1})

    pseudoconvexity_experiments = tai.experiments.PseudoconvexityExperiments(
        seed = _SEED,
        name = 'Pseudo-convexity Experiment',
        utility_function = utility_function,
        linkdata_generator = tai.factory.LinkDataGenerator(noise_params={'mu_x': 0, 'sd_x': _SD_X}),
        equilibrator = equilibrator,
        networks = small_networks)

    pseudoconvexity_experiments.run(grid = list(np.arange(-15, 15+0.1, 0.5)), feature = 'tt')

# ==============================================================================
# CONVERGENCE EXPERIMENTS
# ==============================================================================

if run_experiment['convergence']:

    equilibrator = tai.equilibrium.LUE_Equilibrator(
        max_iters=100,
        method='fw',
        iters_fw=100,
        accuracy=1e-10,
        uncongested_mode = False,
        paths_generator=paths_generator
    )

    utility_function = tai.estimation.UtilityFunction(
        features_Y=['tt'],
        initial_values={'tt': -14},
        true_values={'tt': -1e-0}
    )

    outer_optimizer_norefined = tai.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=2
    )

    outer_optimizer_refined = tai.estimation.OuterOptimizer(
        # method='gauss-newton',
        method='lm',
        # lambda_lm=0,
        iters=1,
    )

    convergence_experiments = tai.experiments.ConvergenceExperiments(
        seed=_SEED,
        name='Convergence Experiment',
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        utility_function= utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'mu_x': 0, 'sd_x':_SD_X}),
        equilibrator=equilibrator,
        bilevel_iters=_BILEVEL_ITERS,
        networks=small_networks)

    convergence_experiments.run()

# ==============================================================================
# BIAS IN REFERENCE OD MATRIX
# ==============================================================================

if run_experiment['biased_reference_od']:

    equilibrator = tai.equilibrium.LUE_Equilibrator(
        max_iters=100,
        method='fw',
        iters_fw=100,
        accuracy=1e-10,
        uncongested_mode = False,
        paths_generator=paths_generator
    )

    utility_function = tai.estimation.UtilityFunction(features_Y=['tt'],
                                                      initial_values={'tt': -14},
                                                      true_values={'tt': -1.5})

    outer_optimizer_norefined = tai.estimation.OuterOptimizer(
        method='ngd',
        iters=1,
        eta=2e0
    )

    outer_optimizer_refined = tai.estimation.OuterOptimizer(
        # method='gauss-newton',
        method='lm',
        # lambda_lm = 0,
        iters=1
    )

    Yang_network = [network for network in small_networks if network.key == 'Yang'][0]

    # As in Yang paper, traffic counts from 5 links are considered for estimation

    bias_reference_od_experiment = tai.experiments.BiasReferenceODExperiment(
        seed=_SEED,
        name='Bias OD Experiment',
        outer_optimizers=[outer_optimizer_norefined, outer_optimizer_refined],
        utility_function=utility_function,
        linkdata_generator=tai.factory.LinkDataGenerator(noise_params={'mu_x': 0, 'sd_x': _SD_X}),
        equilibrator= equilibrator,
        bilevel_iters = _BILEVEL_ITERS,
        network= Yang_network)

    bias_reference_od_experiment.run(distorted_Q = network_generator.get_A_Q_custom_networks(['Yang2'])[1]['Yang2'])



