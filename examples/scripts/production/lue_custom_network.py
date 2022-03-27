# =============================================================================
# 1) SETUP
# =============================================================================

# Set seed for reproducibility and consistency between experiments
import numpy as np
import random

np.random.seed(2021)
random.seed(2021)

# Internal modules
import transportAI as tai

# External modules
import pandas as pd
import copy

from scipy import stats

estimation_options = {}

estimation_options['ttest_search_theta'] = False
estimation_options['ttest_search_Q'] = False

# Needs to be fixed
estimation_options['scaling_Q'] = False  # True

# Regularization options
estimation_options['prop_validation_sample'] = 0
estimation_options['regularization'] = False

# Fixed effect by link, nodes or OD zone in the simulated experiment
estimation_options['fixed_effects'] = {'Q': False, 'nodes': False, 'links': True, 'coverage': 0.0}
observed_links_fixed_effects = None  # 'custom'
theta_true_fixed_effects = 1e4
theta_0_fixed_effects = 0  # 1e2 #These should be positive to encourage choosing link with observed counts generally, the coding is 1 for the attribute

# Feature selection based on t-test from no refined step and ignoring fixed effects (I should NOT do post-selection inference as it violates basic assumptions)
estimation_options['ttest_selection_norefined'] = False

# We relax the critical value to remove features that are highly "no significant". A simil of regularization
# estimation_options['alpha_selection_norefined'] = 3 #if it is higher than 1, it choose the k minimum values
estimation_options['alpha_selection_norefined'] = 0.05

# Computation of t-test with top percentage of observations in terms of SSE
estimation_options['pct_lowest_sse_norefined'] = 100
estimation_options['pct_lowest_sse_refined'] = 100

# No scaling is performed to compute equilibrium as if normalizing by std, then the solution change significantly.
estimation_options['standardization_regularized'] = {'mean': True, 'sd': True}
estimation_options['standardization_norefined'] = {'mean': True, 'sd': False}
estimation_options['standardization_refined'] = {'mean': True, 'sd': False}
# * It seems scaling helps to speed up convergence

# Size of batch for paths and links used to compute gradient
estimation_options['paths_batch_size'] = 0
estimation_options['links_batch_size'] = 0  # 32
# Note: the improvement in speed is impressive with paths but there is inconsistencies

# Out of sample prediction mode
estimation_options['outofsample_prediction_mode'] = False

# config.set_outofsample_prediction_mode(theta = {'tt': -2, 'wt': -3, 'c': -7}, outofsample_prediction = True, mean_count = 100)

# Out of sample prediction mode
# config.set_outofsample_prediction_mode(theta = {'tt': -1.9891, 'tt_reliability': 1.1245, 'low_inc': -0.3415}
#                                        , outofsample_prediction = True, mean_count = 2218.4848)


# Out of sample prediction mode
# config.set_outofsample_prediction_mode(theta = {'tt': -2, 'wt': -3, 'c': -7}, outofsample_prediction = True, mean_count = 100)


# Random initilization of theta parameter and performed before scaling Q matrix. Do not use boolean, options are 'grid','None', 'random'
estimation_options['theta_search'] = None
# estimation_options['theta_search'] = 'grid'
# estimation_options['theta_search'] = 'random'
# estimation_options['q_random_search'] = True # Include od demand matrix factor variation for random search
# To avoid a wrong scaling factor, at least 20 random draws needs to be performed
# estimation_options['n_draws_random_search'] = 20


# =============================================================================
# 2) NETWORKS FACTORY
# =============================================================================

# =============================================================================
# a) CREATION OF TOY NETWORKS
# =============================================================================

# i) CUSTOM NETWORK CREATION
network_name = 'Yang'
# network_name = 'N3'
# network_name = 'Wang' #(TODO: create test that all path_size factors are 1 because no overlapping)
# network_name = 'LoChan'
# network_name = 'N5' # For path correlation test
# network_name = 'Sheffi'

# Create Network Generator
network_generator = tai.factory.NetworkGenerator()

# Get dictionaries with adjacency and O-D matrices for a set of custom networks.
A, Q = network_generator.get_A_Q_custom_networks([network_name])

# Unpack single values of dictionaries
A, Q = A[network_name], Q[network_name]

# Create transportation network with links and nodes using adjacency matrix
custom_network = network_generator.build_network(A = A,network_name= network_name)
# =============================================================================
# b) EXOGENOUS LINK ATTRIBUTES
# =============================================================================

# Set Link Performance functions and link level attributes

# Create data generator to generate synthetic link attributes
linkdata_generator = tai.factory.LinkDataGenerator(noise_params = {'mu_x': 0, 'sd_x': 0})

# Generate synthetic link attributes
link_features_df = linkdata_generator.simulate_features(
    links = custom_network.links,
    features_Z= ['c','s'],
    # option = 'discrete',
    option = 'continuous',
    range = (0,1))

# - Number of attributes that will be set to 0, which moderate sparsity:
n_sparse_features = 1
sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

sparse_features_df = linkdata_generator.simulate_features(
    links = custom_network.links,
    features_Z= sparse_features_labels,
    option = 'continuous',
    range = (0,1))

#Merge data with existing dataframe
link_features_df = link_features_df.merge(sparse_features_df,
                                          left_on = 'link_key',
                                          right_on = 'link_key')

#Load features data
custom_network.load_features_data(linkdata = link_features_df)

# =============================================================================
# b) ENDOGENOUS LINK ATTRIBUTES
# =============================================================================

# # Set random link performance functions
bpr_parameters_df = linkdata_generator.generate_random_bpr_parameters(links_keys = custom_network.links_keys)
custom_network.set_bpr_functions(bprdata = bpr_parameters_df)

# sheffi_network_bpr_parameters_df = pd.DataFrame({'key': [(0,1,'0'),(0,1,'1')],
#                                  'alpha': [1, 1],
#                                   'beta': [4,4],
#                                   'tf': [1.25,2.5],
#                                   'k': [800,1200]}
#                                  )
#
# custom_network.set_bpr_functions(bprdata = sheffi_network_bpr_parameters_df)

# =============================================================================
# c) BEHAVIORAL PARAMETERS AND UTILITY FUNCTIONS
# =============================================================================

utility_function = tai.estimation.UtilityFunction(features_Y=['tt'],
                                               # features_Z= [],
                                               features_Z= ['c'],
                                               true_values={'tt': -1, 'c': 0},
                                               # initial_values =  {'tt': -1, 'c': -6},
                                               initial_values =  {'tt': 0}
                                               )


# Add sparse features
# utility_function.add_sparse_features(Z = sparse_features_labels)

# =============================================================================
# c) OD
# =============================================================================

# Load O-D matrix
custom_network.load_OD(Q  = Q)

# =============================================================================
# d) PATHS
# =============================================================================

# Create path generator
paths_generator = tai.factory.PathsGenerator()

# Generate and Load paths in network
paths_generator.load_k_shortest_paths(network = custom_network, k=2)

# =============================================================================
# d) INCIDENT MATRICES
# =============================================================================

# Create incidence matrices using the provided OD matrices and the parameters for path and attributes generation
custom_network = network_generator.setup_incidence_matrices(network= custom_network)

# =============================================================================
# i) PATH SPECIFIC UTILITY
# =============================================================================

# CREATE PATH FEATURE TO CONTROL PATH CORRELATION

path_size_factors = tai.paths.compute_path_size_factors(D = custom_network.D,
                                                        paths_od = custom_network.paths_od)



# =============================================================================
# j) GENERATION OF SYNTHETIC COUNTS
# =============================================================================

equilibrator = tai.equilibrium.LUE_Equilibrator(network = custom_network,
                                                uncongested_mode = True,
                                                max_iters = 100,
                                                method = 'fw',
                                                iters_fw = 10,
                                                accuracy = 1e-8,
                                                path_size_correction = 0
                                                )

# custom_network.q = np.array([[500]])
# utility_function.parameters.set_true_values({'tt': -2})

# custom_network.q = np.array([[2000]])
# utility_function.parameters.set_true_values({'tt': -2})

# custom_network.q = np.array([[4000]])
# utility_function.parameters.set_true_values({'tt': -1})


# Generate synthetic traffic counts
counts, counts_withdraw = linkdata_generator.simulate_counts(equilibrator = equilibrator,
                                                             utility_function=utility_function,
                                                             noise_params = {'mu_x': 0, 'sd_x': 0}
                                                             )

custom_network.load_traffic_counts(counts=counts)

# # TODO: transform this in a test (with assert).
# # # Verify Sheffi
# print(counts[(0,1,'0')])
# #
# # counts[(0,1,'1')]
# #
# print(1/(1+np.exp(-1*utility_function.parameters.true_values['tt']*(custom_network.links[0].get_traveltime_from_x(counts[(0,1,'0')])
#              -custom_network.links[1].get_traveltime_from_x(counts[(0,1,'1')])))))
# print(custom_network.q*1/(1+np.exp(-1*utility_function.parameters.true_values['tt']*(custom_network.links[0].get_traveltime_from_x(counts[(0,1,'0')])
#              -custom_network.links[1].get_traveltime_from_x(counts[(0,1,'1')])))))
#
# a = 0

# print(1/(1+np.exp(utility_function.parameters.true_values['tt']*(custom_network.links[0].get_traveltime_from_x(counts[(0,1,'0')])
#                -custom_network.links[1].get_traveltime_from_x(counts[(0,1,'1')])))))
#
# custom_network.links[0].bpr.tf
# custom_network.links[0].bpr.k
#
# custom_network.links[1].bpr.tf
# custom_network.links[1].bpr.k
#
# 1.25*(1+(1845/800)**4)
#
# 2.5*(1+((4000-1845)/1200)**4)


# =============================================================================
# 3) DESCRIPTIVE STATISTICS
# =============================================================================

# =============================================================================
# a) Networks topology
# =============================================================================

tai.descriptive_statistics.summary_table_networks([custom_network])

# =============================================================================
# b) LINK COUNTS AND TRAVERSING PATHS
# =============================================================================

tai.descriptive_statistics.adjusted_link_coverage(network =custom_network, counts= counts)

# =============================================================================
# c) Links features and counts
# =============================================================================
summary_table_links_df \
    = tai.descriptive_statistics.summary_table_links(
    links = custom_network.get_observed_links(),
    Z_attrs = ['w', 's']
)

summary_table_links_df.counts = counts.values()

with pd.option_context('display.float_format', '{:0.1f}'.format):
    print(summary_table_links_df.to_string())



# =============================================================================
# 3d) REGULARIZATION
# =============================================================================

# i) Regularization with first order optimization which is faster and scaling path level values

if estimation_options['regularization']:
    # Evaluate objective function for the grid of values for lamabda

    # # Keep this here for efficiency instead of putting it in gap function
    # Yt[i] = (get_matrix_from_dict_attrs_values({k_y: Yt[i][k_y] for k_y in features_Y}).T @ Dt[i]).T
    # Zt[i] = (get_matrix_from_dict_attrs_values({k_z: Zt[i][k_z] for k_z in features}).T @ Dt[i]).T
    #
    # if scale['mean'] or scale['std']:
    #     # Scaling by attribute
    #     Yt[i] = preprocessing.scale(Yt[i], with_mean=scale['mean'], with_std=scale['std'], axis=0)
    #     Zt[i] = preprocessing.scale(Zt[i], with_mean=scale['mean'], with_std=scale['std'], axis=1)

    # Use lasso thresholding operator

    # Perform regularization before no refined optimization and with a first optimization method as
    #  it is faster. In addition, features should be scaled so, the regularization is done properly. I may use a lower
    # amount of MSA iterations as only an approximated solution is needed

    counts_training = counts

    lasso_standardization = {'mean': True, 'sd': True}

    theta_regularized, objective_regularized, result_eq_regularized, results_regularized \
        = tai.estimation.solve_bilevel_lue(
        # network= tai.factory.clone_network(N['train'][i], label = N['train'][i].label),
        Nt=custom_network,
        k_Y=k_Y, k_Z=estimation_options['features'],
        Zt={1: custom_network.Z_dict},
        q0=custom_network.q,
        # q0 = custom_network.q,
        xct={1: np.array(list(counts_training.values()))},
        theta0=config.theta_0,
        # If change to positive number, a higher number of iterations is required but it works well
        # theta0 = theta_true[i],
        standardization=estimation_options['standardization_regularized'],
        outeropt_params={
            # 'method': 'gauss-newton',
            # 'method': 'lm',_
            # 'method': 'gd',
            'method': 'ngd',
            # 'method': 'adagrad',
            # 'method': 'adam',
            'batch_size': estimation_options['links_batch_size'],
            'paths_batch_size': estimation_options['paths_batch_size'],
            'iters_scaling': int(0e0),
            'iters': estimation_options['iters_regularized'],  # 10
            'eta_scaling': 1e-1,
            'eta': estimation_options['eta_regularized'],  # works well for simulated networks
            # 'eta': 1e-4, # works well for Fresno real network
            'gamma': 0,
            'v_lm': 10, 'lambda_lm': 1,
            'beta_1': 0.8, 'beta_2': 0.8
        },
        inneropt_params={'iters': estimation_options['max_sue_iters_regularized'],
                         'accuracy_eq': estimation_options['accuracy_eq']
            , 'uncongested_mode': estimation_options['uncongested_mode']
                         },
        bilevelopt_params={'iters': estimation_options['bilevel_iters_regularized']},  # {'iters': 10},
        # n_paths_column_generation=estimation_options['n_paths_column_generation']
        # plot_options = {'y': 'objective'}
    )

    # Regularization is made using the soft thresholding operator

    # Grid for lambda

    grid_lambda = [0, 1e-3, 1e-2, 5e-2, 1e-1, 1, 1e2, 1e3]

    tai.estimation.lasso_regularization(network=custom_network, grid_lambda=grid_lambda
                                        , theta_estimate=theta_regularized
                                        , features_Y=k_Y, features_Z=estimation_options['features']
                                        , equilibrator={'iters': estimation_options['max_sue_iters_regularized'],
                                                     'accuracy_eq': estimation_options['accuracy_eq']}
                                        , counts=np.array(list(counts_validation.values()))[:, np.newaxis]
                                        , standardization=lasso_standardization)


# =============================================================================
# 5) BILEVEL OPTIMIZATION
# =============================================================================

# Random initilization of initial estimate
# utility_function.random_initializer((-0.1,0.1))
# utility_function.constant_initializer(-5)

outer_optimizer_norefined = tai.estimation.OuterOptimizer(
    method= 'gd',
    iters= 1,  # 10
    eta= 1e-1
)


learner_norefined = tai.estimation.Learner(equilibrator = equilibrator,
                                           outer_optimizer= outer_optimizer_norefined,
                                           utility_function = utility_function,
                                           network = custom_network
                                           )

outer_optimizer_refined = tai.estimation.OuterOptimizer(
    # method= 'ngd',
    # method='gauss-newton',
    method='lm',
    # lambda_lm = 0,
    # eta=5e-1,
    iters=1,
    # path_size_correction = 1
)

learner_refined = tai.estimation.Learner(network=custom_network,
                                         equilibrator=equilibrator,
                                         outer_optimizer=outer_optimizer_refined,
                                         utility_function=utility_function
                                         )

# =============================================================================
# 3.2) HEURISTICS FOR SCALING OF OD MATRIX AND SEARCH OF INITIAL LOGIT ESTIMATE
# =============================================================================




# Grid and random search are performed under the assumption of an uncongested network to speed up the search

if estimation_options['theta_search'] == 'grid':

    utility_function.parameters.values = utility_function.parameters.initial_values

    best_theta_gs \
        = tai.estimation.grid_search_theta(network= custom_network,
                                           equilibrator=equilibrator,
                                           utility_function=utility_function,
                                           counts=np.array(list(counts.values()))[:, np.newaxis],
                                           feature='tt',
                                           grid=np.arange(-3, -4, -0.5),
                                           # theta_attr_grid= [1,0,-1,-10,-10000],
                                           # theta_attr_grid=[-5e-1, -1, -2, -3,-3.2,3.4,3.5,-4,-10],
                                           )

    # Update the parameter for the initial theta values
    # utility_function.initial_values['tt'] = best_theta_gs


if estimation_options['theta_search'] == 'random':

    utility_function.parameters.values = utility_function.parameters.initial_values

    theta_rs, _ = tai.estimation.random_search_theta(
        network=custom_network,
        utility_function=utility_function,
        counts=np.array(list(counts.values()))[:, np.newaxis],
        n_draws= 20,
        # q_bounds = (0.8,1),
        theta_bounds=(-1, 1),
        equilibrator=equilibrator,
        uncongested_mode = False
    )

    # Update the parameter for the initial theta values
    utility_function.initial_values = theta_rs
    
if estimation_options['ttest_search_theta']:

    # utility_function.parameters.values = utility_function.parameters.initial_values

    ttests = tai.estimation.grid_search_theta_ttest(network=custom_network,
                                                    equilibrator=equilibrator,
                                                    utility_function=utility_function,
                                                    counts = custom_network.link_data.counts_vector,
                                                    feature='tt',
                                                    grid= np.arange(-3, -4, -0.5),
                                                    # theta_attr_grid= [1,0,-1,-10,-10000],
                                                    # theta_attr_grid=[-5e-1, -1, -2, -3,-3.2,3.4,3.5,-4,-10],
                                                    )

if estimation_options['ttest_search_Q']:

    # utility_function.parameters.values = dict.fromkeys(utility_function.parameters.initial_values.keys(),-1)
    # utility_function.parameters.values = utility_function.parameters.initial_values

    ttests = tai.estimation.grid_search_Q_ttest(network=custom_network,
                                                equilibrator=equilibrator,
                                                utility_function=utility_function,
                                                counts = custom_network.link_data.counts_vector,
                                                scales= np.arange(0.5, 1.5, 0.5),
                                                # theta_attr_grid= [1,0,-1,-10,-10000],
                                                # theta_attr_grid=[-5e-1, -1, -2, -3,-3.2,3.4,3.5,-4,-10],
                                                )


if estimation_options['scaling_Q']:

    #TODO: Create Q Scale search function in estimation module to look for the best scale.
    # It should return the loss function for every scale and the best scale. Once finished, use in custom network

    # If the scaling factor is too little, the gradients become 0 apparently.

    # Create grid
    scale_grid_q = [1e-1, 5e-1, 1e0, 2e0, 4e0]

    # Add best scale found by random search into grid
    if estimation_options['theta_search'] == 'random':
        scale_grid_q.append(list(min_q_scale_rs.values())[0])

    # We do not generate new paths but use those that were read already from a I/O
    loss_scaling = tai.estimation.scaling_Q(
        counts=np.array(list(counts.values()))[:, np.newaxis],
        network=custom_network,
        utility_function=utility_function,
        equilibrator=equilibrator,
        # scale_grid = [1e-3,1e-2,1e-1],
        # scale_grid=[10e-1],
        grid=scale_grid_q,
        n_paths=None,  # estimation_options['n_paths_column_generation']
        # , scale_grid = [9e-1,10e-1,11e-1]
        silent_mode=True,
    )

    # Search for best scaling
    min_loss = float('inf')
    min_scale = 1
    for scale, loss in loss_scaling.items():
        if loss < min_loss:
            min_loss = loss
            min_scale = scale

    # Perform scaling that minimizes the loss
    custom_network.Q = min_scale * custom_network.Q
    custom_network.q = tai.networks.denseQ(Q=custom_network.Q,
                                           remove_zeros=custom_network.setup_options['remove_zeros_Q'])

    print('Q matrix was rescaled with a ' + str(round(min_scale, 2)) + ' factor')

    print('minimum loss:', '{:,}'.format(round(min_loss, 1)))

    results_congested_gs = equilibrator.sue_logit_iterative(
        Nt=custom_network,
        theta=theta_0,
        features_Y=utility_function.features_Y,
        features_Z=utility_function.features_Z,
    )

    # ttest_gs, criticalval_gs, pval_gs \
    #     = tai.estimation.ttest_theta(theta_h0=0
    #                                  , theta=theta_0
    #                                  , YZ_x=tai.estimation.get_design_matrix(Y={'tt': results_congested_gs['tt_x']}, Z=custom_network.Z_dict, features_Y=features_Y, features=estimation_options['features'])
    #                                  , counts=np.array(list(counts.values()))[:, np.newaxis]
    #                                  , q=custom_network.q
    #                                  , Ix=custom_network.D
    #                                  , Iq=custom_network.M
    #                                  , C=custom_network.C
    #                                  , pct_lowest_sse=estimation_options['pct_lowest_sse_norefined']
    #                                  , alpha=0.05)

# =============================================================================
# 3d) ESTIMATION
# =============================================================================

# ii) NO REFINED OPTIMIZATION AND INFERENCE WITH FIRST ORDER OPTIMIZATION METHODS

# bilevel_estimation_norefined = tai.estimation.LUE_Learner(config.theta_0)

print('\nStatistical Inference with no refined solution')

learning_results_norefined, inference_results_norefined, best_iter_norefined = \
    learner_norefined.statistical_inference(h0 = 0, bilevel_iters = 10, alpha = 0.05, iteration_report = True)

theta_norefined = learning_results_norefined[best_iter_norefined]['theta']


# TODO: Convert t-test selection in a function of estimation module
if estimation_options['ttest_selection_norefined']:


    if estimation_options['alpha_selection_norefined'] < len(theta_norefined):

        # An alternative to regularization
        ttest_norefined = np.array(parameter_inference_norefined_table['t-test'])
        ttest_norefined_dict = dict(
            zip(k_Y + k_Z_estimation, list(map(float, parameter_inference_norefined_table['t-test']))))

        n = np.count_nonzero(~np.isnan(np.array(list(xc.values()))[:, np.newaxis]))
        p = len(k_Y + k_Z_estimation)

        critical_alpha = estimation_options['alpha_selection_norefined']
        critical_tvalue = stats.t.ppf(1 - critical_alpha / 2, df=n - p)

        if estimation_options['alpha_selection_norefined'] >= 1:
            # It picks the alpha minimum(s) ignoring fixed effects

            ttest_lists = []
            for attr, ttest, idx in zip(ttest_norefined_dict.keys(), ttest_norefined.flatten(), np.arange(p)):
                if attr not in custom_network.k_fixed_effects:
                    ttest_lists.append(ttest)

            critical_tvalue = float(
                -np.sort(-abs(np.sort(ttest_lists)))[estimation_options['alpha_selection_norefined'] - 1])

            print('\nSelecting top ' + str(estimation_options[
                                               'alpha_selection_norefined']) + ' features based on t-values and excluding fixed effects')

        else:
            print('Selecting features based on critical t-value ' + str(critical_tvalue))

            # print('\ncritical_tvalue:', critical_tvalue)

        # Loop over endogenous and exogenous attributes

        for attr, t_test in ttest_norefined_dict.items():

            if attr not in custom_network.k_fixed_effects:

                if abs(t_test) < critical_tvalue - 1e-3:
                    if attr in k_Y:
                        k_Y.remove(attr)

                    if attr in k_Z_estimation:
                        k_Z_estimation.remove(attr)

        print('features_Y:', k_Y)
        print('features:', k_Z_estimation)

# iii) REFINED OPTIMIZATION AND INFERENCE WITH SECOND ORDER OPTIMIZATION METHODS

estimation_options['outofsample_prediction_mode'] = False


def set_outofsample_prediction_mode(theta: dict, mean_count, outofsample_prediction: bool = False):
    estimation_options = {}

    print('\nEnabling out of sample prediction mode')

    estimation_options['outofsample_prediction_mode'] = outofsample_prediction

    if estimation_options['outofsample_prediction_mode']:
        estimation_options['bilevel_iters_norefined'] = 1
        estimation_options['bilevel_iters_refined'] = 1

        estimation_options['ttest_selection_norefined'] = False
        estimation_options['link_selection'] = False

        estimation_results = {}

        # Assumed mean to compute the benchmark mean model (default is None)
        estimation_results['mean_count_benchmark_model'] = mean_count

if estimation_options['outofsample_prediction_mode']:

    theta_refined, objective_refined, result_eq_refined, results_refined = \
        copy.deepcopy(theta_norefined), copy.deepcopy(objective_norefined), copy.deepcopy(
            result_eq_norefined), copy.deepcopy(results_norefined)

    estimation_results['theta_refined'] = copy.deepcopy(theta_refined)
    estimation_results['best_loss_refined'] = copy.deepcopy(objective_refined)

    parameter_inference_refined_table, model_inference_refined_table = \
        copy.deepcopy(parameter_inference_norefined_table), copy.deepcopy(model_inference_norefined_table)


else:

    print('\nStatistical Inference in refined stage')

    theta_norefined = learning_results_norefined[best_iter_norefined]['theta']

    learner_refined.utility_function.initial_values = theta_norefined

    learning_results_refined, inference_results_refined, best_iter_refined = \
        learner_refined.statistical_inference(h0=0, bilevel_iters=10, alpha=0.05, iteration_report = True)

# =============================================================================
# VISUALIZATIONS
# =============================================================================

# Networks graphs

# plot = tai.visualization.Artist(folder_plots = estimation.dirs['estimation_folder'])

# plot.plot_custom_networks(N = {i:custom_networks[i] for i in ['N1','N2','N3','N4']},
#                           show_edge_labels = True, subfoldername = "networks", filename = "custom_networks")

# # Visualization
# plots_options = {}
# plots_options['dim_subplots'] = int(np.ceil(np.sqrt(sim_options['n_random_networks'] + sim_options['n_custom_networks']))), \
#                                 int(np.ceil(np.sqrt(sim_options['n_random_networks'] + sim_options['n_custom_networks'])))

# Plot yang network
# plot.draw_MultiDiNetwork(G=custom_networks['Yang'].G, nodes_pos={0: (0, 1), 1: (1, 1), 2: (2,1)
#                                                     , 3: (0, 0), 4: (1, 0), 5: (2,0)
#                                                     , 6: (0, -1), 7: (1, -1), 8: (2,-1)
#                                                     }, show_edge_labels=False)
#
# plt.title('Yang')
# plt.show()

# plot.draw_MultiDiNetwork(G=custom_networks['Lo&Chan'].G, nodes_pos={0: (0, 1), 1: (1, 1), 2: (2,1)
#                                                     , 3: (0, 0), 4: (1, 0), 5: (2,0)
#                                                     }, show_edge_labels=False)
#
# plt.title('Lo and Chan')
# plt.show()
#
# plot.draw_MultiDiNetwork(G=custom_networks['Wang'].G, nodes_pos={0: (0, 1), 1: (1, 1), 2: (1,0)
#                                                     , 3: (0, 0)
#                                                     }, show_edge_labels=False)
#
# plt.title('Wang')
# plt.show()



# plot.draw_MultiDiNetwork(G = custom_networks['N6'].G, show_edge_labels=False)
# plt.show()

# # Plot real network
# plot.draw_MultiDiNetwork(G = custom_networks[subfoldername].G, node_size=100, font_size=8, edge_curvature= 0.1, show_edge_labels= False)
# plt.show()
#
# Plot all networks together
# plot.plot_all_networks(N = custom_networks, show_edge_labels = False, subfoldername = "networks", filename = "all_networks")



# =============================================================================
# 3.2) FIXED EFFECTS
# =============================================================================

# TODO: Simulation is not accounting for observed link effects because synthetic counts were created before

if observed_links_fixed_effects == 'random' and estimation_options['fixed_effects']['coverage'] > 0:
    custom_network.add_fixed_effects_attributes(estimation_options['fixed_effects'],
                                                             observed_links=observed_links_fixed_effects)

    for k in custom_network.k_fixed_effects:
        theta_true[i][k] = theta_true_fixed_effects  # -float(np.random.uniform(1,2,1))
        config.theta_0[k] = theta_0_fixed_effects
        estimation_options['features'] = [*estimation_options['features'], k]

    # Update dictionary with attributes values at the network level
    custom_network.set_Z_attributes_dict_network(links_dict=custom_network.links_dict)

    if len(custom_network.k_fixed_effects) > 0:
        print('\nFixed effects created within observed links only:', custom_network.k_fixed_effects)

# =============================================================================
# 3b) FIXED EFFECTS
# =============================================================================

# Selection of fixed effects among the group of observed counts only

# TODO: Simulation is not accounting for observed link effects because synthetic counts were created before
if observed_links_fixed_effects == 'random' and estimation_options['fixed_effects']['coverage'] > 0:
    linkdata_generator.add_fixed_effects_attributes(estimation_options['fixed_effects']
                                                , observed_links=observed_links_fixed_effects)

if observed_links_fixed_effects == 'custom':

    # selected_links_keys = [(695, 688, '0'), (1619, 631, '0'), (1192, 355, '0'), (217, 528, '0')]

    selected_links_keys = [(695, 688, '0'), (1192, 355, '0'), (680, 696, '0')]

    custom_network.add_fixed_effects_attributes(estimation_options['fixed_effects'],
                                                observed_links=observed_links_fixed_effects
                                                , links_keys=selected_links_keys)

if observed_links_fixed_effects is not None:

    for k in custom_network.k_fixed_effects:
        theta_true[i][k] = theta_true_fixed_effects  # -float(np.random.uniform(1,2,1))
        config.theta_0[k] = theta_0_fixed_effects
        k_Z = [*estimation_options['features'], k]

    # Update dictionary with attributes values at the network level
    custom_network.set_Z_attributes_dict_network(links_dict=custom_network.links_dict)

    if len(custom_network.k_fixed_effects) > 0:
        print('\nFixed effects created within observed links only:', custom_network.k_fixed_effects)


if estimation_options['prop_validation_sample'] > 0:

    # Get a training and testing sample
    xc, counts_validation = tai.estimation.generate_training_validation_samples(
        xct=counts
        , prop_validation=estimation_options['prop_validation_sample']
    )

else:
    counts_validation = counts



#
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
# q_true = custom_network.q_true
#
# # No refined iterations
# for iter in np.arange(1, len(results_norefined) + 1):
#     q_estimate = np.sum((results_norefined[iter]['q']-q_true)**2)
#     df_bilevel_norefined.loc[iter] = ['norefined'] + [iter] + [q_estimate]
#
# # Refined iterations
# for iter in np.arange(1, len(results_refined) + 1):
#     q_estimate = np.sum((results_refined[iter]['q']-q_true)**2)
#     df_bilevel_refined.loc[iter] = ['refined'] + [iter] + [q_estimate]
#
# # Adjust the iteration numbers
# df_bilevel_refined['iter'] = (df_bilevel_refined['iter'] + df_bilevel_norefined['iter'].max()).astype(int)
#
#
# fig = plot2.q_estimation_convergence(results_norefined_df = df_bilevel_norefined
#                                , results_refined_df = df_bilevel_refined
#                                , methods=[estimation_options['outeropt_method_norefined'],
#                                           estimation_options['outeropt_method_refined']]
#                                , filename = 'q_estimation_convergence.pdf', subfoldername="experiments/inference")
#
# tai.writer.write_figure_to_log_folder(fig=fig
#                                       , filename='q_estimation_convergence.pdf', log_file=config.log_file)
