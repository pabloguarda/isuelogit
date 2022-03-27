import os
import numpy as np

# Internal modules
import isuelogit as isl

# =============================================================================
# 4) LEARNING TRAVELLERS' PREFERENCES FROM LINK-LEVEL DATA
# =============================================================================

# =============================================================================
# 4) c) MULTIDAY DATA
# =============================================================================
#
# N['train']['SiouxFalls'].Z_dict['c']
# theta_true


# The result on identifiability of Z or T depending on the variability of Q or Z is remarkable. It sufficient to have
# variabiability in the exogeneous parameters to determine the effect of travel time as their variability change travel time
# indirectly through a change in equailiburm

n_days = 10#50 # 50
interval_days = 5#10
# n_days_seq =  np.append(1,np.repeat(interval_days , int(n_days/interval_days)))
n_days_seq =  np.repeat(interval_days , int(n_days/interval_days))
# n_days_seq = np.arange(1, n_days + 1, 10)

# remove_zeros_Q = True

# theta0 = copy.copy(theta_true)
# theta0 = {i: 0 for i in [*features_Y, *features]}
# theta0 = {i:1 for i in [*features_Y, *features]}
# theta0 = {i:-1 for i in [*features_Y, *features]}
# theta0 = {i:theta_true[i] for i in [*features_Y, *features]}
# theta0['wt'] = 0
# theta0['c'] = 0
# theta0['tt'] = 0

# end_params = {'theta': True, 'q': False}

# N = {'Custom4': N['train']['Custom4']}

def multiday_estimation_analyses(end_params, N, n_days_seq, remove_zeros_Q, theta_true, R_labels, Z_attrs_classes, bpr_classes, cutoff_paths, n_paths, fixed_effects, q_range, var_Q = 0):

    results_multidays = {'no_disturbance_Q': {}, 'disturbance_Q': {}}

    results_multidays['no_disturbance_Q'] = {'q':{}, 'theta':{}, 'vot':{}, 'time': {}}
    results_multidays['disturbance_Q'] = {'q': {}, 'theta': {}, 'vot': {}, 'time': {}}

    N_multiday = {}

    theta0 = {}
    q0 = {}

    for i in N.keys():

        k_Y = ['tt']
        # features = ['wt','c']
        k_Z = ['wt','c'] #['c'] #
        # features = list(N[i].Z_dict.keys())  #

        # Starting values
        q0[i] = isl.networks.denseQ(Q=N[i].Q, remove_zeros=remove_zeros_Q)
        theta0[i] = {k:theta_true[i][k] for k in [*k_Y,*k_Z]}

        if end_params['q']:
            q0[i] = np.zeros(isl.networks.denseQ(Q=N[i].Q, remove_zeros=remove_zeros_Q).shape)

        if end_params['theta']:
            for j in [*k_Y,*k_Z]:
                theta0[i][j] = 0
            # theta0 = {j:0  for i in N.keys()}
            # theta0 = {i: 0 for i in [*features_Y, *features]}

        results_multidays['no_disturbance_Q']['q'][i] = {}
        results_multidays['no_disturbance_Q']['theta'][i] = {}
        results_multidays['no_disturbance_Q']['vot'][i] = {}
        results_multidays['no_disturbance_Q']['time'][i] = {}
        start_time = {'no_disturbance_Q': 0, 'disturbance_Q': 0}

        N_multiday_old = {i:None}

        acc_days = 0

        for n_day in n_days_seq:

            results_multidays_temp = {'no_disturbance_Q': 0, 'disturbance_Q': 0}

            # No perturbance
            start_time['no_disturbance_Q'] = time.time()

            results_multidays_temp['no_disturbance_Q'], N_multiday_old = isl.estimation.multiday_estimation(N = {i:copy.deepcopy(N[i])}, N_multiday_old = {i:copy.deepcopy(N_multiday_old[i])}
                                                                                                            , end_params = end_params, n_days = n_day
                                                                                                            , k_Y = k_Y, k_Z = k_Z
                                                                                                            , theta0 = theta0[i]
                                                                                                            , q0 = q0
                                                                                                            , randomness_multiday = {'Q': False, 'BPR': False, 'Z': True, 'var_Q': var_Q}
                                                                                                            , remove_zeros_Q = remove_zeros_Q
                                                                                                            , theta_true = theta_true[i]
                                                                                                            , R_labels=R_labels
                                                                                                            , q_range = q_range
                                                                                                            , Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes
                                                                                                            , cutoff_paths=cutoff_paths, n_paths = n_paths
                                                                                                            , fixed_effects= fixed_effects)

            results_multidays['no_disturbance_Q']['time'][i][n_day] = time.time()-start_time['no_disturbance_Q']

            # # Perturbance
            # start_time['disturbance_Q'] = time.time()
            #
            # results_multidays_temp['disturbance_Q'] = isl.logit.multiday_estimation(N={i: copy.deepcopy(N[i])}
            #                                                                    , end_params=end_params, n_days=n_day
            #                                                                    , features_Y=features_Y, features=features
            #                                                                    , theta0=theta0
            #                                                                    , q0=q0
            #                                                                    , randomness_multiday={'Q': False,
            #                                                                                           'BPR': False,
            #                                                                                           'Z': False,
            #                                                                                           'var_Q': var_Q}
            #                                                                    , remove_zeros_Q=remove_zeros_Q
            #                                                                    , theta_true=theta_true
            #                                                                    , R_labels=R_labels
            #                                                                    , q_range=q_range
            #                                                                    , Z_attrs_classes=Z_attrs_classes,
            #                                                                    bpr_classes=bpr_classes
            #                                                                    , cutoff_paths=cutoff_paths).get(i)
            #
            # results_multidays['disturbance_Q']['time'][i][n_day] = time.time() - start_time['disturbance_Q']

            acc_days += n_day

            if end_params['q']:
                # print(i)
                results_multidays['no_disturbance_Q']['q'][i][acc_days] = results_multidays_temp['no_disturbance_Q'][i]['q']

            if end_params['theta']:
                # print(i)
                results_multidays['no_disturbance_Q']['theta'][i][acc_days] = results_multidays_temp['no_disturbance_Q'][i]['theta']

                results_multidays['no_disturbance_Q']['vot'][i][acc_days] = results_multidays_temp['no_disturbance_Q'][i]['vot']


    return results_multidays['no_disturbance_Q']

results_multidays_analyses = {'no_disturbance_Q': {}, 'disturbance_Q': {}}

# results_multidays_analyses['no_disturbance_Q']['end_theta_q'] = multiday_estimation_analyses(end_params = {'theta': True, 'q': True}
#                                                                                              , n_days_seq = n_days_seq, N = N['train'], var_Q = 0, theta_true = theta_true, R_labels=R_labels, q_range= q_range, remove_zeros_Q=remove_zeros_Q, Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes,  fixed_effects= fixed_effects, cutoff_paths=cutoff_paths, n_paths = n_paths)
# results_multidays_analyses['no_disturbance_Q']['end_q'] = multiday_estimation_analyses(end_params = {'theta': False, 'q': True}, n_days_seq = n_days_seq, N = N['train'], var_Q = 0, theta_true = theta_true, R_labels=R_labels, q_range= q_range, remove_zeros_Q=remove_zeros_Q, Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes,  fixed_effects= fixed_effects, cutoff_paths=cutoff_paths, n_paths = n_paths)
results_multidays_analyses['no_disturbance_Q']['end_theta'] = multiday_estimation_analyses(end_params = {'theta': True, 'q': False}, n_days_seq = n_days_seq, N = N['train'], var_Q = 0, theta_true = theta_true, R_labels=R_labels, q_range= q_range, remove_zeros_Q=remove_zeros_Q, Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes,  fixed_effects= fixed_effects, cutoff_paths=cutoff_paths, n_paths = n_paths)

# N['train']['Braess-Example'].Q

# links_temp = N['train']['Braess-Example'].links

type(N['train']['Braess-Example'].Z_dict['length'][(0,3,'0')])
theta_true
list(N['train']['SiouxFalls'].Z_dict['length'].values())
list(N['train']['SiouxFalls'].Z_dict['speed'].values())
list(N['train']['SiouxFalls'].Z_dict['toll'].values())
list(N['train']['SiouxFalls'].Z_dict)

#TODO: there are problems with the order of the Z_dict that produces that the estimate are wrong.
results_multidays_analyses['no_disturbance_Q']['end_theta']['vot']
results_multidays_analyses['no_disturbance_Q']['end_theta']['theta']['Custom4']
results_multidays_analyses['no_disturbance_Q']['end_theta']['theta']['N1']


nx.paths

# results_multidays_analyses['no_disturbance_Q']['end_theta']['vot']
#
# np.linalg.norm(results_multidays_analyses['no_disturbance_Q']['end_q']['q']['SiouxFalls'][1]-{i: isl.network.denseQ(Q= N_i.Q, remove_zeros=remove_zeros_Q) for i,N_i in N['train'].items()}['SiouxFalls'],2)


# #Experimental
# results_multidays_analyses['no_disturbance_Q']['end_theta'] = multiday_estimation_analyses(end_params = {'theta': True, 'q': False}, n_days_seq = np.repeat(20, 4), N = {'N4': N['train']['N4']}, var_Q = 0, theta_true = theta_true, R_labels=R_labels, q_range= q_range, remove_zeros_Q=remove_zeros_Q, Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes,  fixed_effects= fixed_effects, cutoff_paths=cutoff_paths)
#
# results_multidays_analyses['no_disturbance_Q']['end_theta']['theta']['N4'][80]
# theta_true['N4']
# results_multidays_analyses['no_disturbance_Q']['end_theta']['vot']['N4']
# results_multidays_analyses['no_disturbance_Q']['end_q']['q']['N5']
# {i: isl.network.denseQ(Q= N_i.Q, remove_zeros=remove_zeros_Q) for i,N_i in N['train'].items()}['N5']

# N['train']['N5'].A
# results_multidays_analyses['no_disturbance_Q']['end_theta']['theta']['N2'][40]
# results_multidays_analyses['no_disturbance_Q']['end_theta']['vot']['N2'][40]
# theta_true['N2']

# print(results_multidays['train']['N4'][n_days_seq[-1]])
# print(['N4'])
# results_multidays['q']['N1'][n_days]
# {i: isl.network.denseQ(Q=N_i.Q, remove_zeros=remove_zeros_Q) for i, N_i in N['train'].items()}['N5']

# i) No disturbance in Q

# if end_params['theta']:
plot.vot_multidays_consistency(theta_true = theta_true
                               , labels = {'end_theta_q':r"Endogenous $\theta$ and $Q$" , 'end_theta':r"Endogenous $\theta$"}
                               , vot_estimates = {i:results_multidays_analyses['no_disturbance_Q'][i]['vot'] for i in ['end_theta_q','end_theta']}
                               , N_labels = {i:N_i.key for i, N_i in N['train'].items()}
                               , filename = 'vot_multidays_links'
                               , colors = ['b','r']
                               , subfolder = 'link-level'
                               )

# if end_params['q']:
plot.q_multidays_consistency(q_true = {i: isl.networks.denseQ(Q= N_i.Q, remove_zeros=remove_zeros_Q) for i, N_i in N['train'].items()}
                             , labels = {'end_theta_q':r"Endogenous $\theta$ and $Q$" , 'end_q': r"Endogenous $Q$"}
                             , q_estimates = {i:results_multidays_analyses['no_disturbance_Q'][i]['q'] for i in ['end_theta_q','end_q']}
                             , N_labels = {i:N_i.key for i, N_i in N['train'].items()}
                             , filename = 'q_multidays_links'
                             , colors = ['b','g']
                             , subfolder = 'link-level'
                             )

results_multidays_analyses['no_disturbance_Q']['end_theta_q']['q']
results_multidays_analyses['no_disturbance_Q']['end_q']['q']

# plot.computational_time_multidays_consistency(computational_times =  {i:results_multidays_analyses['no_disturbance_Q'][i]['time'] for i in ['end_theta_q','end_q', 'end_theta']}
#                              , N_labels = {i:N_i.label for i, N_i in N['train'].items()}
#                              , filename = 'computational_times_multidays_links'
#                              , colors = ['b','g', 'r']
#                              , network_name = 'link-level'
#                              )

# ii) Applying disturbance q
# var_q = 3
var_q = 'Poisson'

results_multidays_analyses['disturbance_Q']['end_theta_q'] \
    = multiday_estimation_analyses(end_params = {'theta': True, 'q': True}, n_days_seq = n_days_seq, N = N['train'], var_Q = var_q, theta_true = theta_true, R_labels=R_labels, q_range= q_range, remove_zeros_Q=remove_zeros_Q, Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes,  fixed_effects= fixed_effects, cutoff_paths=cutoff_paths, n_paths = n_paths)
results_multidays_analyses['disturbance_Q']['end_theta'] \
    = multiday_estimation_analyses(end_params = {'theta': True, 'q': False}, n_days_seq = n_days_seq, N = N['train'], var_Q = var_q, theta_true = theta_true, R_labels=R_labels, q_range= q_range, remove_zeros_Q=remove_zeros_Q, Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes,  fixed_effects= fixed_effects, cutoff_paths=cutoff_paths, n_paths = n_paths)
results_multidays_analyses['disturbance_Q']['end_q'] \
    = multiday_estimation_analyses(end_params = {'theta': False, 'q': True}, n_days_seq = n_days_seq, N = N['train'], var_Q = var_q, theta_true = theta_true, R_labels=R_labels, q_range= q_range, remove_zeros_Q=remove_zeros_Q, Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes,  fixed_effects= fixed_effects, cutoff_paths=cutoff_paths, n_paths = n_paths)

np.mean(np.array(list(results_multidays_analyses['disturbance_Q']['end_theta']['vot']['N2'].values())))
np.mean(np.array(list(results_multidays_analyses['no_disturbance_Q']['end_theta']['vot']['N4'].values()))[4:])

# - Endogenous Q and theta
plot.vot_multidays_consistency(theta_true = theta_true
                               , labels = {'no_disturbance_Q':r"Endogenous $\theta$ and $Q$ ($\sigma^2_Q = 0$)", 'disturbance_Q':r"Endogenous $\theta$ and $Q$ ($\sigma^2_Q > 0$)" }
                               , vot_estimates = {i:results_multidays_analyses[i]['end_theta_q']['vot'] for i in ['no_disturbance_Q','disturbance_Q']}
                               , N_labels = {i:N_i.key for i, N_i in N['train'].items()}
                               , filename = 'vot_disturbance_q_multidays_links'
                               , colors = ['b','r']
                               , subfolder = 'link-level'
                               )
plot.q_multidays_consistency(q_true = {i: isl.networks.denseQ(Q= N_i.Q, remove_zeros=remove_zeros_Q) for i, N_i in N['train'].items()}
                             , labels = {'no_disturbance_Q':r"Endogenous $\theta$ and $Q$ ($\sigma^2_Q = 0$)", 'disturbance_Q':r"Endogenous $\theta$ and $Q$ ($\sigma^2_Q > 0$)" }
                             , q_estimates = {i:results_multidays_analyses[i]['end_theta_q']['q'] for i in ['no_disturbance_Q', 'disturbance_Q']}
                             , N_labels = {i:N_i.key for i, N_i in N['train'].items()}
                             , filename = 'q_disturbance_q_multidays_links'
                             , colors = ['b','r']
                             , subfolder = 'link-level'
                             )

# - Endogenous theta
plot.vot_multidays_consistency(theta_true = theta_true
                               , labels = {'no_disturbance_Q':r"Endogenous $\theta$ ($\sigma^2_Q = 0$)", 'disturbance_Q':r"Endogenous $\theta$ ($\sigma^2_Q > 0$)" }
                               , vot_estimates = {i:results_multidays_analyses[i]['end_theta']['vot'] for i in ['no_disturbance_Q','disturbance_Q']}
                               , N_labels = {i:N_i.key for i, N_i in N['train'].items()}
                               , filename = 'vot_disturbance_q_endogenous_theta_multidays_links'
                               , colors = ['b','r']
                               , subfolder = 'link-level'
                               )
plot.q_multidays_consistency(q_true = {i: isl.networks.denseQ(Q= N_i.Q, remove_zeros=remove_zeros_Q) for i, N_i in N['train'].items()}
                             , labels = {'no_disturbance_Q':r"Endogenous $Q$ ($\sigma^2_Q = 0$)", 'disturbance_Q':r"Endogenous $Q$ ($\sigma^2_Q > 0$)" }
                             , q_estimates = {i:results_multidays_analyses[i]['end_q']['q'] for i in ['no_disturbance_Q', 'disturbance_Q']}
                             , N_labels = {i:N_i.key for i, N_i in N['train'].items()}
                             , filename = 'q_disturbance_q_endogenous_theta_multidays_links'
                             , colors = ['b','r']
                             , subfolder = 'link-level'
                             )




# results_multidays_analyses['no_disturbance_Q']['end_theta_q']['vot']
# results_multidays_analyses['disturbance_Q']['end_theta_q']['vot']['N5']
# results_multidays_analyses['disturbance_Q']['end_theta_q']['vot']['N5']
# results_multidays_analyses['no_disturbance_Q']['end_theta_q']['vot']['N5']
# results_multidays_analyses['disturbance_Q']['end_theta_q']['vot']['N3']
# results_multidays_analyses['disturbance_Q']['end_theta_q']['theta']['N4']



# plot.computational_time_multidays_consistency(computational_times =  {i:results_multidays_analyses['disturbance_Q'][i]['time'] for i in ['end_theta_q','end_q', 'end_theta']}
#                              , N_labels = {i:N_i.label for i, N_i in N['train'].items()}
#                              , filename = 'computational_times_disturbance_q_multidays_links'
#                              , colors = ['b','g', 'r']
#                              , network_name = 'link-level'
#                              )

# np.random.lognormal(0,np.log(10))


# =============================================================================
# 4A) ESTIMATION PRECISION VERSUS NUMBER OF LINKS
# =============================================================================

# If only link level data is available, the preference parameters can be found by using the link-path satistisfaction
# constraints when replacing by the path flows with the logit probabilities.

# This is done

# Example: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

# len(N['train']['N8'].links)
# n_links = np.arange(5, 30, 5)
n_links = np.arange(10, 40, 5)
n_bootstraps = 1 #Bootstrap of 20 achieves good convergence

optimality_gap = {'train':{}}
vot_estimates_nlinks = {'train':{}}
theta_estimates_nlinks = {'train':{}}

k_Y = ['tt']
# features = list(N['train']['N1'].Z_dict.keys()) # #['wt','c'] #

for N_i in N['train'].keys():

    optimality_gap_N = {}
    vot_estimates_N = {}
    theta_i_estimate_N = {} #Store mean and std of the estimates
    theta_estimates_N = {}

    estimation_done = False
    theta_estimate = {}
    idx_links = {}

    theta_estimates_bootstrap = {}

    k_Y = ['tt']
    # features = ['c','wt'] #
    k_Z = list(N['train'][N_i].Z_dict.keys())

    for i in n_links:

        for j in range(0,n_bootstraps):

            if i <= len(N['train'][N_i].x) or not estimation_done:

                # Select a random sample of links provided the number of links is smaller than the total number of links
                idx_links = range(0, len(N['train'][N_i].x))

                if i < len(N['train'][N_i].x):
                    idx_links = random.sample(range(0, len(N['train'][N_i].x)), i)

                theta_estimate = isl.estimation.solve_link_level_model(end_params={'theta': True, 'q': False},
                                                                       Mt={1: N['train'][N_i].M}, Ct={
                        i: isl.estimation.choice_set_matrix_from_M(N['train'][N_i].M)}, Dt={1: N['train'][N_i].D},
                                                                       k_Y=k_Y, Yt={1: N['train'][N_i].Y_dict}, k_Z=k_Z,
                                                                       Zt={1: N['train'][N_i].Z_dict},
                                                                       xt={1: N['train'][N_i].x},
                                                                       idx_links={1: idx_links},
                                                                       scale={'mean': False, 'std': False},
                                                                       q0=isl.networks.denseQ(Q=N['train'][N_i].Q,
                                                                                              remove_zeros=remove_zeros_Q),
                                                                       theta0=dict.fromkeys([*k_Y, *k_Z], 0),
                                                                       lambda_hp=0)

                # N['train'][N_i].Y_dict['tt'].keys()
                # N['train'][N_i].Z_dict['wt'].keys()

            if i >= len(N['train'][N_i].x):
                estimation_done = True

            theta_estimates_bootstrap[j] = theta_estimate

        theta_estimates_temp_N = {}
        theta_estimates_N = {}

        for k in [*k_Y,*k_Z]:
            theta_estimates_temp_N[k] = [theta_estimates_bootstrap[j]['theta'][k] for j in theta_estimates_bootstrap.keys()]
            theta_estimates_N[k] = {'mean': np.mean(theta_estimates_temp_N[k]),
                                    'sd': np.std(theta_estimates_temp_N[k])}

        theta_i_estimate_N[i] = theta_estimates_N

        vot = np.array(theta_estimates_temp_N['tt']) / np.array(theta_estimates_temp_N['c'])
        vot_estimates_N[i] = {'mean':np.mean(vot), 'sd': np.std(vot)}

        optimality_gap_N[i] = np.mean(np.abs([bootstrap_iter['gap'] for bootstrap_iter in theta_estimates_bootstrap.values()]))

    optimality_gap['train'][N_i] = optimality_gap_N
    vot_estimates_nlinks['train'][N_i] = vot_estimates_N
    theta_estimates_nlinks['train'][N_i] = theta_i_estimate_N

#Value of time
plot.consistency_nonlinear_link_logit_estimation(filename= 'vot_vs_links_training_networks'
                                                 , theta_est= theta_estimates_nlinks['train']
                                                 , vot_est = vot_estimates_nlinks['train']
                                                 , display_parameters = {'vot': True, 'tt': False, 'c': False}
                                                 , theta_true= theta_true
                                                 , N_labels = {i:N['train'][i].key for i in N['train'].keys()}
                                                 , n_bootstraps = n_bootstraps
                                                 , subfolder= "link-level/training"
                                                 )

# #Cost
# plot.consistency_nonlinear_link_logit_estimation(filename= 'c_vs_links_training_networks'
#                                      , theta_est= theta_estimates_nlinks['train']
#                                      , vot_est = vot_estimates_nlinks['train']
#                                      , display_parameters = {'vot': False, 'tt': False, 'c': True}
#                                      , theta_true= theta_true
#                                      , N_labels = {i:N['train'][i].label for i in N['train'].keys()}
#                                      , n_bootstraps = n_bootstraps
#                                      , network_name= "link-level/training"
#                                      )
#
# #Travel time
# plot.consistency_nonlinear_link_logit_estimation(filename= 'traveltime_vs_links_training_networks'
#                                      , theta_est= theta_estimates_nlinks['train']
#                                      , vot_est = vot_estimates_nlinks['train']
#                                      , display_parameters = {'vot': False, 'tt': True, 'c': False}
#                                      , theta_true= theta_true
#                                      , N_labels = {i:N['train'][i].label for i in N['train'].keys()}
#                                      , n_bootstraps = n_bootstraps
#                                      , network_name= "link-level/training"
#                                      )

# =============================================================================
# 4B) REGULARIZATION
# =============================================================================

n_lasso_trials = 6 # 10 #40 # Number of lambda values to be tested for cross validation
lambda_vals = np.append(0,np.logspace(-16, 4, n_lasso_trials))

# Size of sample of links used to fit parameters in training and validation networks
n_links = 10 #Around five, there are failed optimization because the problem become very ill-posed, i.e. extremely overfitted. Use 10


# =============================================================================
# 4B) i) VALIDATION NETWORKS
# =============================================================================

errors_logit = {'train': {}, 'validation': {}}
lambdas_valid = {'train': {}, 'validation': {}}
theta_estimates = {'train': {}, 'validation': {}}
vot_estimates = {'train': {}, 'validation': {}}
x_N = {'train': {}, 'validation': {}}

#Create validation networks

N['validation'] = \
    isl.networks.clone_network(N=N['train'], label='Validation'
                               , randomness={'Q': True, 'BPR': False, 'Z': False}
                               , Z_attrs_classes=None, bpr_classes=None)


valid_network = None

#Compute SUE in clone networks
while valid_network is None:
    try:
        results_sue['validation'] = {i: isl.equilibrium.sue_logit_fisk(q = isl.networks.denseQ(Q = N['validation'][i].Q, remove_zeros = remove_zeros_Q)
                                                                       , M = N['validation'][i].M
                                                                       , D = N['validation'][i].D
                                                                       , links = N['validation'][i].links_dict
                                                                       , paths = N['validation'][i].paths
                                                                       , Z_dict = N['validation'][i].Z_dict
                                                                       , k_Z = []
                                                                       , theta = theta_true
                                                                       , cp_solver = 'ECOS'  #'ECOS': it is faster and crashes sometimes. 'SCS' is slow
                                                                       )
                                     for i in N['validation'].keys()}

    except:
        print('error'+ str(i))
        for i in N['validation'].keys():
            exceptions['SUE']['validation'][i] += 1

        N['validation'] = isl.networks.clone_network(N=N['train'], label='Validation'
                                                     , R_labels=R_labels
                                                     , randomness={'Q': True, 'BPR': False, 'Z': False}
                                                     , Z_attrs_classes=None, bpr_classes=None)

        pass
    else:
        valid_network = True

        # Store travel time, link and path flows in Network objects
        for i in N['validation'].keys():
            N['validation'][i].set_Y_attr_links(y=results_sue['validation'][i]['tt_x'], feature='tt')
            N['validation'][i].x_dict = results_sue['validation'][i]['x']
            N['validation'][i].f_dict = results_sue['validation'][i]['f']

# Estimation using regularization
for i in N['train'].keys():

    # # Select a random sample of links provided the number of links is smaller than the total number of links
    train_idx_links, validation_idx_links = range(0, len(N['validation'][i].x)), range(0, len(N['validation'][i].x))

    if n_links < len(N['train'][i].x) and n_links < len(N['validation'][i].x):
        train_idx_links, validation_idx_links = random.sample(range(0, len(N['train'][i].x)), n_links), random.sample(range(0, len(N['validation'][i].x)), n_links)

    theta_estimates['train'][i] = {}

    vot_estimates['train'][i] = {}
    vot_estimates['validation'][i] = {}

    errors_logit['train'][i] = {}
    errors_logit['validation'][i] = {}

    for lambda_i in lambda_vals:
        theta_estimates['train'][i][lambda_i] = isl.estimation.solve_link_level_model(
            Mt= {i:N['train'][i].M}
            , Ct={i: isl.estimation.choice_set_matrix_from_M(N['train'][i].M)}
            , Dt= {i:N['train'][i].D}
            , q0= {i:isl.networks.denseQ(Q=N['train'][i].Q, remove_zeros=True)}
            , k_Y = ['tt'], k_Z= list(N['train'][i].Z_dict.keys())
            , Yt= {i:N['train'][i].Y_dict}
            , Zt= {i:N['train'][i].Z_dict}
            , xt= {i:N['train'][i].x}  # x_N
            , theta0= dict.fromkeys(theta_true,1)
            , idx_links= {i:train_idx_links}
            , scale = {'mean': False, 'std': False} #{'mean': True, 'std': False}
            , lambda_hp=lambda_i
            # , scale = {'mean': True, 'std': False}
        )['theta']

        if theta_estimates['train'][i][lambda_i]['c'] != 0:
            vot_estimates['train'][i][lambda_i] = theta_estimates['train'][i][lambda_i]['tt']/theta_estimates['train'][i][lambda_i]['c']
        else:
            vot_estimates['train'][i][lambda_i] = np.nan

        errors_logit['train'][i][lambda_i] = isl.estimation.loss_link_level_model(
            theta=np.array(list(theta_estimates['train'][i][lambda_i].values()))
            , lambda_hp=0
            , M= {i:N['train'][i].M}
            , C={i: isl.estimation.choice_set_matrix_from_M(N['train'][i].M)}
            , D= {i:N['train'][i].D}
            , Y= {i:(isl.estimation.get_matrix_from_dict_attrs_values({k_y: N['train'][i].Y_dict[k_y] for k_y in ['tt']}).T @ N['train'][i].D).T}
            , Z= {i:(isl.estimation.get_matrix_from_dict_attrs_values({k_y: N['train'][i].Z_dict[k_y] for k_y in list(N['train'][i].Z_dict.keys())}).T @ N['train'][i].D).T}
            , q= {i:isl.networks.denseQ(Q=N['train'][i].Q, remove_zeros=True)}
            , x= {i:N['train'][i].x}
            , idx_links={i:train_idx_links}
            , norm_o=2, norm_r=1)


        errors_logit['validation'][i][lambda_i] = isl.estimation.loss_link_level_model(
            theta = np.array(list(theta_estimates['train'][i][lambda_i].values()))
            , lambda_hp= 0
            , M = {i:N['validation'][i].M}
            , C= {i: isl.estimation.choice_set_matrix_from_M(N['validation'][i].M)}
            , D = {i:N['validation'][i].D}
            , Y= {i:(isl.estimation.get_matrix_from_dict_attrs_values(
                {k_y: N['validation'][i].Y_dict[k_y] for k_y in ['tt']}).T @ N['validation'][i].D).T}
            , Z= {i:(isl.estimation.get_matrix_from_dict_attrs_values(
                {k_y: N['validation'][i].Z_dict[k_y] for k_y in list(N['validation'][i].Z_dict.keys())}).T @
                     N['validation'][i].D).T}
            , q = {i:isl.networks.denseQ(Q=N['validation'][i].Q, remove_zeros=True)}
            , x = {i:N['validation'][i].x}
            , idx_links = {i:validation_idx_links}
            , norm_o=2, norm_r=1)

    lambdas_valid['train'][i] = lambda_vals
    lambdas_valid['validation'][i] = lambda_vals

theta_estimates['validation'] = theta_estimates['train']


# Regularization path
plot.regularization_path(theta_estimate = theta_estimates['train']
                         , lambdas = lambdas_valid['validation']
                         , errors = errors_logit['validation']
                         , N_labels = {i:N_i.key for i, N_i in N['train'].items()}
                         , filename = 'regularization_path_data_links'
                         , key_attrs = ['wt','c', 'tt']
                         , color = 'b'
                         , subfolder = 'link-level'
                         )

# =============================================================================
# 4ii) TRAINING AND VALIDATION NETWORKS
# =============================================================================

# Training and validation
# # This force to plot the training curve with no regularization
# errors_logit_copy = errors_logit
# for i in N['train'].keys():
#     errors_logit_copy['train'][i][0] = 0 #This trick force that the lower error is with no regularization

plot.regularization_joint_error(errors = errors_logit #errors_logit_copy
                                , lambdas = lambdas_valid
                                , N_labels = {i:N_i.key for i, N_i in N['train'].items()}
                                , filename = 'regularization_error_training_validation_data_links'
                                , colors = ['b','r']
                                , subfolder = 'link-level/both'
                                )

plot.regularization_joint_consistency(errors = errors_logit #errors_logit_copy
                                      , N_labels = {i:N_i.key for i, N_i in N['train'].items()}
                                      , theta_true = theta_true
                                      , theta_estimate = theta_estimates
                                      , filename = 'regularization_consistency_training_validation_networks'
                                      , colors = ['b','r']
                                      , subfolder = 'link-level/both'
                                      )

# VOT estimates versus lambda

# Regularization path
# self = isl.Plot(folder_plots = folder_plots, dim_subplots=dim_subplots)
plot.vot_regularization_path(theta_true = theta_true
                             , errors = errors_logit['validation']
                             , lambdas = lambdas_valid['validation']
                             , vot_estimate = vot_estimates['train']
                             , N_labels = {i:N_i.key for i, N_i in N['validation'].items()}
                             , filename = 'vot_regularization_path_links'
                             , color = 'r'
                             , subfolder = 'link-level'
                             )

# Store values of theta and vot for optimal regularization parameter

validation_thetas = {'noreg': {}, 'reg':{}}
validation_vot_estimates = {'noreg': {}, 'reg':{}}

for network_lbl in N['validation'].keys():

    validation_thetas['noreg'][network_lbl] = theta_estimates['train'][network_lbl][0]
    validation_thetas['reg'][network_lbl] = theta_estimates['train'][network_lbl][list(errors_logit['validation'][i].keys())[np.argmin(np.array(list(errors_logit['validation'][network_lbl].values())))]]

    validation_vot_estimates['noreg'][network_lbl] = validation_thetas['noreg'][network_lbl]['tt']/validation_thetas['noreg'][network_lbl]['c']

    if validation_thetas['reg'][network_lbl]['c'] != 0:
        validation_vot_estimates['reg'][network_lbl] = validation_thetas['reg'][network_lbl]['tt']/validation_thetas['reg'][network_lbl]['c']

theta_wt_validation = [validation_thetas['reg'][i]['wt'] for i in N['validation'].keys()]
theta_tt_validation = [validation_thetas['reg'][i]['tt'] for i in  N['validation'].keys()]
theta_c_validation = [validation_thetas['reg'][i]['c'] for i in  N['validation'].keys()]
non_zero_thetas_validation = [np.count_nonzero(np.abs(np.array(list(validation_thetas['reg'][i].values())))>0.1) for i in  N['validation'].keys()]

print(validation_vot_estimates['noreg'])
print(validation_vot_estimates['reg'])

#Regularization using SUE Loss as error  function
errors_SUE_logit = {'train': {}, 'validation': {}}
for i, N_i in N['validation'].items():
    errors_SUE_logit['train'][i] = {}
    errors_SUE_logit['validation'][i] = {}
    for j in errors_logit['validation'][i].keys():
        errors_SUE_logit['train'][i][j] = isl.estimation.loss_SUE(o = 2, x_obs = N['train'][i].x
                                                                  , q = isl.networks.denseQ(Q = N['train'][i].Q, remove_zeros = remove_zeros_Q)
                                                                  , M = N['train'][i].M, D = N['train'][i].D
                                                                  , links = N['train'][i].links_dict
                                                                  , paths = N['train'][i].paths
                                                                  , Z_dict = N['train'][i].Z_dict
                                                                  , k_Z = []
                                                                  , theta = theta_estimates['train'][i][j]
                                                                  , cp_solver = 'ECOS')
        errors_SUE_logit['validation'][i][j] = isl.estimation.loss_SUE(o = 2, x_obs = N['validation'][i].x
                                                                       , q = isl.networks.denseQ(Q = N['validation'][i].Q, remove_zeros = remove_zeros_Q)
                                                                       , M = N['validation'][i].M, D = N['validation'][i].D
                                                                       , links = N['validation'][i].links_dict
                                                                       , paths = N['validation'][i].paths
                                                                       , Z_dict = N['validation'][i].Z_dict
                                                                       , k_Z = []
                                                                       , theta = theta_estimates['train'][i][j]
                                                                       , cp_solver = 'ECOS')

# Regularization SUE loss
plot.regularization_error(errors = errors_SUE_logit['validation']
                          , lambdas = lambdas_valid['validation']
                          , N_labels =  {i:N_i.key for i, N_i in N['validation'].items()}
                          , filename = 'regularization_SUE_loss_validation_networks'
                          , subfolder = 'path-level/validation'
                          , color = 'r'
                          )

plot.regularization_joint_error(errors = errors_SUE_logit
                                , lambdas = lambdas_valid
                                , N_labels =  {i:N_i.key for i, N_i in N['train'].items()}
                                , filename = 'regularization_error_training_validation_networks'
                                , colors = ['b','r']
                                , subfolder = 'path-level/both'
                                )


plot.regularization_joint_consistency(errors = errors_SUE_logit
                                      , N_labels =  {i:N_i.key for i, N_i in N['validation'].items()}
                                      , theta_true = theta_true
                                      , theta_estimate = theta_estimates
                                      , filename = 'regularization_consistency_training_validation_networks'
                                      , colors = ['b','r']
                                      , subfolder = 'path-level/both'
                                      )


plot.vot_regularization_path(theta_true = theta_true
                             , errors = errors_SUE_logit['validation']
                             , lambdas = lambdas_valid['validation']
                             , vot_estimate = vot_estimates['train']
                             , N_labels = {i:N_i.key for i, N_i in N['validation'].items()}
                             , filename = 'vot_regularization_path_links'
                             , color = 'r'
                             , subfolder = 'link-level'
                             )

errors_SUE_logit['validation']['N5']
errors_logit['validation']['N5']
# =============================================================================
# 4iii) TESTING NETWORKS
# =============================================================================

# Bootstrapping to calculate generalization error (RMSE) in testing networks

test_error = {'noreg': {}, 'reg':{}}

N['test'], results_sue['test'] = {}, {}

n_samples = 10

for i in N['validation'].keys():

    test_error['noreg'][i] = {}
    test_error['reg'][i] = {}

    N['test'][i], results_sue['test'][i] = {}, {}

    for j in range(0,n_samples):

        N['test'][i][j] = isl.networks.clone_network(N={i: N['train'][i]}, label='Test'
                                                     , R_labels=R_labels
                                                     , randomness={'Q': True, 'BPR': False, 'Z': False}
                                                     , q_range=(2, 10), remove_zeros_Q=remove_zeros_Q
                                                     , Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes,
                                                     cutoff_paths=cutoff_paths).get(i)

        valid_network = None

        while valid_network is None:
            try:

                results_sue['test'] = {i: isl.equilibrium.sue_logit_fisk(
                    q=isl.networks.denseQ(Q=N['test'][i][j].Q, remove_zeros=remove_zeros_Q)
                    , M=N['test'][i][j].M
                    , D=N['test'][i][j].D
                    , links=N['test'][i][j].links_dict
                    , paths=N['test'][i][j].paths
                    , Z_dict=N['test'][i][j].Z_dict
                    , k_Z= []
                    , theta=theta_true
                    , cp_solver='ECOS'  # 'ECOS': it is faster and crashes sometimes. 'SCS' is slow
                )
                    for i in N['test'].keys()}


            except:
                N['test'][i][j] = isl.networks.clone_network(N={i: N['train'][i]}, label='Test'
                                                             , R_labels=R_labels
                                                             , randomness={'Q': True, 'BPR': False, 'Z': False}).get(i)
            else:
                valid_network = True

                for i in N['test'].keys():
                    N['test'][i][j].set_Y_attr_links(y=results_sue['test'][i]['tt_x'], feature='tt')
                    N['test'][i][j].x_dict = results_sue['test'][i]['x']
                    N['test'][i][j].f_dict = results_sue['test'][i]['f']


        test_error['noreg'][i][j] = np.round(np.sqrt(isl.estimation.loss_link_level_model(
            theta=np.array(list(validation_thetas['noreg'][i].values()))
            , lambda_hp=0
            , M= {i:N['test'][i][j].M}
            , C={i: isl.estimation.choice_set_matrix_from_M(N['test'][i].M)}
            , D= {i:N['test'][i][j].D}
            , Y= {i:(isl.estimation.get_matrix_from_dict_attrs_values(
                {k_y: N['test'][i][j].Y_dict[k_y] for k_y in ['tt']}).T @ N['test'][i][j].D).T}
            , Z= {i:(isl.estimation.get_matrix_from_dict_attrs_values(
                {k_z: N['test'][i][j].Z_dict[k_z] for k_z in
                 list(N['test'][i][j].Z_dict.keys())}).T @
                     N['test'][i][j].D).T}
            , q= {i:isl.networks.denseQ(Q=N['test'][i][j].Q, remove_zeros=True)}
            , x= {i:N['test'][i][j].x}
            , idx_links= {i:range(0, len(N['test'][i][j].x))}
            , norm_o=2, norm_r=1)), 4)

        test_error['reg'][i][j] \
            = np.round(np.sqrt(
            isl.estimation.loss_link_level_model(
                theta=np.array(list(validation_thetas['reg'][i].values()))
                , M = {i:N['test'][i][j].M}
                , C= {i: isl.estimation.choice_set_matrix_from_M(N['test'][i].M)}
                , D = {i:N['test'][i][j].D}
                , Y= {i:(isl.estimation.get_matrix_from_dict_attrs_values(
                    {k_y: N['test'][i][j].Y_dict[k_y] for k_y in ['tt']}).T @ N['test'][i][j].D).T}
                , Z= {i:(isl.estimation.get_matrix_from_dict_attrs_values(
                    {k_z: N['test'][i][j].Z_dict[k_z] for k_z in
                     list(N['test'][i][j].Z_dict.keys())}).T @
                         N['test'][i][j].D).T}
                , q = {i:isl.networks.denseQ(Q=N['test'][i][j].Q, remove_zeros=True)}
                , x = {i:N['test'][i][j].x}
                , idx_links= {i:range(0, len(N['test'][i][j].x))}
                , lambda_hp=0
                , norm_o=2, norm_r=1))
            , 4)

test_error_plot = {}

# self = isl.Plot(folder_plots = folder_plots, dim_subplots=dim_subplots)

for i in N['test'].keys():
    test_error_plot[i] = {}
    for j in N['test'][i].keys():
        test_error_plot[i]['noreg'] = {'mean':np.mean(np.array(list(test_error['noreg'][i].values()))), 'sd': np.std(np.array(list(test_error['noreg'][i].values())))}
        test_error_plot[i]['reg'] = {'mean': np.mean(np.array(list(test_error['reg'][i].values()))),
                                     'sd': np.std(np.array(list(test_error['reg'][i].values())))}


# print(test_error['noreg']['N6'])
# print(test_error['reg']['N6'])

plot.regularization_error_nonlinear_link_logit_estimation(filename= 'regularization_vs_errors_links_testing_networks'
                                                          , errors = test_error_plot
                                                          , N_labels = {i:'Test '+str(i) for i in N['test'].keys()}
                                                          , n_samples = n_samples
                                                          , subfolder= "link-level/test"
                                                          )

# Only work if all thetas are negative
#results_sue['test-estimated'] = {i: isl.equilibrium.sue_logit(q=isl.denseQ(Q=N_i.Q, remove_zeros=remove_zeros_Q)
#                                                     , M=N_i.M
#                                                     , D=N_i.D
#                                                     , paths=N_i.paths
#                                                     , links=N_i.links_dict
#                                                     , Z_dict=N_i.Z_dict
#                                                     , theta= theta_estimates['validation'][i][np.argmin(np.array(list(errors_logit['validation']['N6'].values())))]
#                                                     , cp_solver='ECOS'
#                                                     )
#                for i, N_i in N['test'].items()}

# TODO: Jacobian matrix for higher precision
# https://www.thedatascientists.com/logistic-regression/





# =============================================================================
# 5) LEARNING TRAVELLERS' PREFERENCES FROM PATH LEVEL DATA
# =============================================================================

# =============================================================================
# 5A) ESTIMATION VIA MLE
# =============================================================================

# scale_features = {'mean': False, 'std': False}
# i = 'N1'
#Likelihood function
likelihood_logit = {}
likelihood_logit['train'] = {i: isl.estimation.likelihood_path_level_logit(f = N['train'][i].f
                                                                           , M = N['train'][i].M
                                                                           , D = N['train'][i].D
                                                                           , k_Z = N['train'][i].Z_dict.keys()  #['c', 'wt'] #
                                                                           , Z = N['train'][i].Z_dict
                                                                           , k_Y = ['tt']
                                                                           , Y = N['train'][i].Y_dict
                                                                           , scale = {'mean': False, 'std': False}  #scale_features
                                                                           )
                             for i in N['train'].keys()}

# Constraints
constraints_theta = {}
constraints_theta['Z'] = {'wt':np.nan, 'c': np.nan}
# constraints_theta['Z'] = {'wt':theta['wt'], 'c': theta['c']}
constraints_theta['Y'] = {'tt': np.nan}

# Maximize likelihood to obtain solutions
i = 'N9'
results_logit = {}
results_logit['train'] = {i: isl.estimation.solve_path_level_logit(cp_ll = likelihood_logit['train'][i]['cp_ll']
                                                                   , cp_theta = likelihood_logit['train'][i]['cp_theta']
                                                                   , constraints_theta = constraints_theta
                                                                   , cp_solver = 'ECOS'  #'SCS'
                                                                   # scaling features makes this method to fail somehow
                                                                   )
                          for i in N['train'].keys()}


# =============================================================================
# - SUMMARY RESULTS
# =============================================================================
#TODO: Create class for tables and results structure (M type)

# A) LOGIT SUE RESULTS

# Add additional indicators based on SUE logit results
for i,results in results_logit['train'].items():
    results_logit['train'][i][0]['theta_true_tt'] = theta_true['tt']
    results_logit['train'][i][0]['theta_true_Z'] = theta_true_Z

    if isinstance(results_logit['train'][i][0]['theta_Y']['tt'],float):
        results_logit['train'][i][0]['diff_theta_tt'] = np.round(results_logit['train'][i][0]['theta_Y']['tt']/ results_logit['train'][i][0]['theta_true_tt']-1, 2)
    else:
        results_logit['train'][i][0]['diff_theta_tt'] = ''


# Print attributes of interest
for i in N['train'].keys():
    print(i + ' :',[k + ': ' + str(results_logit['train'][i][0][k])
                    for k in ['theta_Y','theta_true_tt', 'diff_theta_tt', 'theta_Z', 'theta_true_Z']])

# Tables for latex
theta_true

results_table = {'N':[], 'tt_hat':[],'wt_hat': [], 'c_hat': [], 'vot_hat': [],
                 'tt_gap':[],'wt_gap': [], 'c_gap': [], 'vot_gap': []
                 }

decimals = 2
for i in N['train'].keys():

    try:
        results_table['tt_hat'].append(np.round(results_logit['train'][i][0]['theta_Y']['tt'], decimals))
        results_table['tt_gap'].append(np.round(results_logit['train'][i][0]['theta_Y']['tt'] / theta_true['tt'] - 1, decimals))
    except:
        results_table['tt_hat'].append("-")
        results_table['tt_gap'].append("-")

    try:
        results_table['c_hat'].append(np.round(results_logit['train'][i][0]['theta_Z']['c'], decimals))
        results_table['c_gap'].append(np.round(results_logit['train'][i][0]['theta_Z']['c']/theta_true['c']-1, decimals))
    except:
        results_table['c_hat'].append("-")
        results_table['c_gap'].append("-")

    try:
        results_table['wt_hat'].append(np.round(results_logit['train'][i][0]['theta_Z']['wt'], decimals))
        results_table['wt_gap'].append(np.round(results_logit['train'][i][0]['theta_Z']['wt'] / theta_true['wt'] - 1, decimals))
    except:
        results_table['wt_hat'].append("-")
        results_table['wt_gap'].append("-")

    try:
        results_table['vot_hat'].append(np.round(results_table['tt_hat'][-1]/results_table['c_hat'][-1], decimals))
        results_table['vot_gap'].append(np.round(results_table['vot_hat'][-1]/(theta_true['tt']/theta_true['c'])-1, decimals))
    except:
        results_table['vot_hat'].append("-")
        results_table['vot_gap'].append("-")


# Network results
df = pd.DataFrame()
df['N'] =  np.array(list(N['train'].keys()))
for var in ['tt_hat', 'wt_hat', 'c_hat', 'vot_hat', 'tt_gap', 'wt_gap', 'c_gap', 'vot_gap']:
    df[var] = results_table[var]

# Print Latex Table
print(df.to_latex(index=False))


# =============================================================================
# 5B) REGULARIZATION
# =============================================================================

n_lasso_trials = 10 #6 # 10 #50 # Number of lambda values to be tested for cross validation
lambda_vals = np.append(0,np.logspace(-12, 2, n_lasso_trials))

# =============================================================================
# 5B) i) TRAINING NETWORKS
# =============================================================================
#- Estimation
results_logit['train'] = {i: isl.estimation.solve_path_level_logit(cp_ll = likelihood_logit['train'][i]['cp_ll']
                                                                   , cp_theta = likelihood_logit['train'][i]['cp_theta']
                                                                   , constraints_theta = constraints_theta
                                                                   , r = 1
                                                                   , lambdas = lambda_vals
                                                                   , cp_solver = 'ECOS'
                                                                   )
                          for i in N['train'].keys()}

# - Theta estimates by network and lambda (iter) value
theta_estimates = {'train':{}}
for i in N['train'].keys():
    theta_estimates['train'][i] = {}
    for lambda_val, estimates in results_logit['train'][i].items():
        theta_estimates['train'][i][lambda_val] = {**estimates['theta_Y'],**estimates['theta_Z']}

#  Compute errors for different lambdas
errors_logit = {}
lambdas_valid = {}

errors_logit['train'], lambdas_valid['train'] \
    = isl.estimation.prediction_error_logit_regularization(
    lambda_vals= {i: dict(zip(range(0,len(lambda_vals)),lambda_vals)) for i in N['train'].keys()}
    , theta_estimates = theta_estimates['train']
    , likelihood= likelihood_logit['train']
    , f = {i: N['train'][i].f for i in N['train'].keys()}
    , M = {i:N['train'][i].M for i in N['train'].keys()}
)

# Regularization error
plot.regularization_error(errors = errors_logit['train']
                          , lambdas = lambdas_valid['train']
                          , N_labels =  {i:N['train'][i].key for i in N['train'].keys()}
                          , filename = 'regularization_error_training_networks'
                          , subfolder = 'path-level/training'
                          , color='b'
                          )


# Regularization path
plot.regularization_path(theta_estimate = theta_estimates['train']
                         , lambdas = lambdas_valid['train']
                         , errors = errors_logit['train']
                         , N_labels = {i:N['train'][i].key for i in N['train'].keys()}
                         , filename = 'regularization_path_training_networks'
                         , subfolder = 'path-level/training'
                         , key_attrs = ['wt','c', 'tt']
                         , color = 'b'
                         )

# True versus fitted theta with regularization
plot.regularization_consistency(errors = errors_logit['train']
                                , N_labels =  {i:N_i.key for i, N_i in N['train'].items()}
                                , theta_true = theta_true
                                , theta_estimate = theta_estimates['train']
                                , filename = 'regularization_consistency_training_networks'
                                , subfolder = 'path-level/training'
                                , color = 'b'
                                )

# =============================================================================
# 5B) ii) VALIDATION NETWORKS
# =============================================================================
# Generate copies of the training network including link values but with a different Q matrix

# N['validation']['N1'].links[0].Y_dict
# N['validation']['N1'].Z_dict
N['validation'] = \
    isl.networks.clone_network(N=N['train'], label='Validation'
                               , R_labels=R_labels
                               , randomness={'Q': True, 'BPR': False, 'Z': False}
                               , Z_attrs_classes=None, bpr_classes=None)


valid_network = None

while valid_network is None:
    try:
        results_sue['validation'] = {i: isl.equilibrium.sue_logit_fisk(
            q = isl.networks.denseQ(Q = N['validation'][i].Q, remove_zeros = remove_zeros_Q)
            , M = N['validation'][i].M, D = N['validation'][i].D
            , links = N['validation'][i].links_dict
            , paths = N['validation'][i].paths
            , Z_dict = N['validation'][i].Z_dict
            , k_Z = []
            , theta = theta_true
            , cp_solver = 'ECOS'  #'ECOS': it is faster and crashes sometimes. 'SCS' is slow
        )
            for i in N['validation'].keys()}

    except:
        # print('error'+ str(i))
        # for i in N['validation'].keys():
        # exceptions['SUE']['validation'][i] += 1

        N['validation'] = isl.networks.clone_network(N=N['train'], label='Validation'
                                                     , R_labels=R_labels
                                                     , randomness={'Q': True, 'BPR': False, 'Z': False}
                                                     , Z_attrs_classes=None, bpr_classes=None)

        pass
    else:
        valid_network = True

        # Store travel time, link and path flows in Network objects
        for i in N['validation'].keys():
            N['validation'][i].set_Y_attr_links(y=results_sue['validation'][i]['tt_x'], feature='tt')
            N['validation'][i].x_dict = results_sue['validation'][i]['x']
            N['validation'][i].f_dict = results_sue['validation'][i]['f']


# Get likelihood objects from logit model
likelihood_logit['validation'] = {i: isl.estimation.likelihood_path_level_logit(f=results_sue['validation'][i]['f']
                                                                                , M=N['validation'][i].M
                                                                                , D=N['validation'][i].D
                                                                                , k_Z= list(N['validation'][i].Z_dict.keys())
                                                                                , Z=N['validation'][i].Z_dict
                                                                                , k_Y=['tt']
                                                                                , Y=N['validation'][i].Y_dict
                                                                                , scale=scale_features
                                                                                )
                                  for i in N['validation'].keys()}


# Fit logit with regularization
results_logit['validation'] = {i: isl.estimation.solve_path_level_logit(cp_ll=likelihood_logit['validation'][i]['cp_ll']
                                                                        , cp_theta=likelihood_logit['validation'][i]['cp_theta']
                                                                        , constraints_theta=constraints_theta
                                                                        , cp_solver='ECOS'
                                                                        , lambdas = lambda_vals
                                                                        )
                               for i, N_i in N['validation'].items()}

# - Theta estimates by network and lambda (iter) value
theta_estimates['validation'] = {}
for i in N['validation'].keys():
    theta_estimates['validation'][i] = {}
    for lambda_val, estimates in results_logit['validation'][i].items():
        theta_estimates['validation'][i][lambda_val] = {**estimates['theta_Y'],**estimates['theta_Z']}

# - Compute errors from theta estimates from training dataset.
errors_logit['validation'], lambdas_valid['validation'] \
    = isl.estimation.prediction_error_logit_regularization(theta_estimates = theta_estimates['train']
                                                           , lambda_vals = {i: dict(zip(range(0, len(lambda_vals)), lambda_vals)) for i in N['train'].keys()}
                                                           , likelihood= likelihood_logit['validation']
                                                           , f = {i: N['validation'][i].f for i in N['validation'].keys()}
                                                           , M = {i:N_i.M for i, N_i in N['validation'].items()}
                                                           )

# Regularization error
plot.regularization_error(errors = errors_logit['validation']
                          , lambdas = lambdas_valid['validation']
                          , N_labels =  {i:N_i.key for i, N_i in N['validation'].items()}
                          , filename = 'regularization_error_validation_networks'
                          , subfolder = 'path-level/validation'
                          , color = 'r'
                          )

# Regularization path
plot.regularization_path(theta_estimate = theta_estimates['validation']
                         , lambdas = lambdas_valid['validation']
                         , errors = errors_logit['validation']
                         , N_labels = {i:N_i.key for i, N_i in N['validation'].items()}
                         , filename = 'regularization_path_validation_networks'
                         , subfolder = 'path-level/validation'
                         , key_attrs = ['wt','c', 'tt']
                         , color = 'r'
                         )


# =============================================================================
# 5B) iii) TRAINING AND VALIDATION NETWORKS
# =============================================================================

# True versus fitted theta with regularization
plot.regularization_consistency(errors = errors_logit['validation']
                                , N_labels =  {i:N_i.key for i, N_i in N['validation'].items()}
                                , theta_true = theta_true
                                , theta_estimate = theta_estimates['validation']
                                , filename = 'regularization_consistency_validation_networks'
                                , subfolder = 'path-level/validation'
                                , color = 'r'
                                )

## Training and validation
plot.regularization_joint_error(errors = errors_logit
                                , lambdas = lambdas_valid
                                , N_labels =  {i:N_i.key for i, N_i in N['train'].items()}
                                , filename = 'regularization_error_training_validation_networks'
                                , colors = ['b','r']
                                , subfolder = 'path-level/both'
                                )

plot.regularization_joint_consistency(errors = errors_logit
                                      , N_labels =  {i:N_i.key for i, N_i in N['validation'].items()}
                                      , theta_true = theta_true
                                      , theta_estimate = theta_estimates
                                      , filename = 'regularization_consistency_training_validation_networks'
                                      , colors = ['b','r']
                                      , subfolder = 'path-level/both'
                                      )



#SUE Loss
errors_SUE_logit = {'train': {}, 'validation': {}}
for i, N_i in N['validation'].items():
    errors_SUE_logit['train'][i] = {}
    errors_SUE_logit['validation'][i] = {}
    for j in errors_logit['validation'][i].keys():
        errors_SUE_logit['train'][i][j] = isl.estimation.loss_SUE(o = 2, x_obs = N['train'][i].x
                                                                  , q = isl.networks.denseQ(Q = N['train'][i].Q, remove_zeros = remove_zeros_Q)
                                                                  , M = N['train'][i].M, D = N['train'][i].D
                                                                  , links = N['train'][i].links_dict
                                                                  , paths = N['train'][i].paths
                                                                  , Z_dict = N['train'][i].Z_dict
                                                                  , k_Z = []
                                                                  , theta = theta_estimates['train'][i][j]
                                                                  , cp_solver = 'ECOS')
        errors_SUE_logit['validation'][i][j] = isl.estimation.loss_SUE(o = 2, x_obs = N['validation'][i].x
                                                                       , q = isl.networks.denseQ(Q = N['validation'][i].Q, remove_zeros = remove_zeros_Q)
                                                                       , M = N['validation'][i].M, D = N['validation'][i].D
                                                                       , links = N['validation'][i].links_dict
                                                                       , paths = N['validation'][i].paths
                                                                       , Z_dict = N['validation'][i].Z_dict
                                                                       , k_Z = []
                                                                       , theta = theta_estimates['train'][i][j]
                                                                       , cp_solver = 'ECOS')

# Regularization SUE loss
plot.regularization_error(errors = errors_SUE_logit['validation']
                          , lambdas = lambdas_valid['validation']
                          , N_labels =  {i:N_i.key for i, N_i in N['validation'].items()}
                          , filename = 'regularization_SUE_loss_validation_networks'
                          , subfolder = 'path-level/validation'
                          , color = 'r'
                          )

plot.regularization_joint_error(errors = errors_SUE_logit
                                , lambdas = lambdas_valid
                                , N_labels =  {i:N_i.key for i, N_i in N['train'].items()}
                                , filename = 'regularization_error_training_validation_networks'
                                , colors = ['b','r']
                                , subfolder = 'path-level/both'
                                )


plot.regularization_joint_consistency(errors = errors_SUE_logit
                                      , N_labels =  {i:N_i.key for i, N_i in N['validation'].items()}
                                      , theta_true = theta_true
                                      , theta_estimate = theta_estimates
                                      , filename = 'regularization_consistency_training_validation_networks'
                                      , colors = ['b','r']
                                      , subfolder = 'path-level/both'
                                      )


# =============================================================================
# 10) ADDITIONAL ANALYSES
# =============================================================================

# TODO: Effect of sparsity in theta estimates (travel time parameter get more variance)

#A) Single attribute case:

# i) Estimate of theta versus real theta

delta_theta_tt = 1e-1
theta_t_range = np.arange(theta_true['tt'] - delta_theta_tt, theta_true['tt'] + delta_theta_tt, delta_theta_tt/5)

theta_true_plot = theta_true

# Constraints
constraints_theta = {}
constraints_theta['Z'] = {'wt': 0, 'c': theta_true_plot['c']}
constraints_theta['Y'] = {'tt': np.nan}

k_Z_plot = ['wt','c']

theta_Z = {}
theta_true_t = {}
theta_est_t = {}

for i in N['train'].keys():

    theta_Z[i] = []
    theta_true_t[i] = []
    theta_est_t[i] = []

    theta_i = theta_true_plot

    for theta_ti in theta_t_range:
        theta_i['tt'] = theta_ti
        result = isuelogit.extra.simulation.sue_logit_simulation_recovery(N=N['train'][i]
                                                                            , theta=theta_i
                                                                            , constraints_theta=constraints_theta
                                                                            , k_Z = k_Z_plot
                                                                            , remove_zeros = remove_zeros_Q
                                                                            , scale_features = scale_features
                                                                            )[0]
        theta_Z[i].append(result['theta_Z']['c'])
        theta_true_t[i].append(theta_i['tt'])
        theta_est_t[i].append(result['theta_Y']['tt'])


plot.estimated_vs_true_theta(filename='estimated_vs_true_theta'
                             , theta_est_t = theta_est_t
                             , theta_c = theta_Z
                             , theta_true_t = theta_true_t
                             , constraints_theta=constraints_theta
                             , N_labels =  {i:N_i.key for i, N_i in N['train'].items()}
                             , color = 'b'
                             )

# ii) Variance of optimal flows versus theta (as higher is theta, flow are more dispersed)

theta_true_plot = theta_true

delta_theta_tt = 1e-1
theta_t_range = np.arange(theta_true['tt'] - delta_theta_tt, theta_true['tt'] + delta_theta_tt, delta_theta_tt/5)
k_Z_plot = ['wt','c']

theta_t_plot= {}
x_plot = {}
f_plot = {}

for i in N['train'].keys():

    theta_t_plot[i] = []
    x_plot[i] = []
    f_plot[i] = []

    theta_i = theta_true_plot

    for theta_ti in theta_t_range:

        theta_i['tt'] = theta_ti

        result_sue = isl.equilibrium.sue_logit_fisk(q= isl.networks.denseQ(Q=N['train'][i].Q, remove_zeros=remove_zeros_Q)
                                                    , M=N['train'][i].M
                                                    , D=N['train'][i].D
                                                    , links=N['train'][i].links_dict
                                                    , paths=N['train'][i].paths
                                                    , Z_dict=N['train'][i].Z_dict
                                                    , k_Z= k_Z_plot
                                                    , theta= theta_i
                                                    )

        theta_t_plot[i].append(np.round(theta_ti, 4))
        x_plot[i].append(list(result_sue['x'].values()))
        f_plot[i].append(list(result_sue['f'].values()))

plot.flows_vs_true_theta(filename = 'sd_flows_vs_theta'
                         , x = x_plot
                         , f = f_plot
                         , N_labels = {i:N_i.key for i, N_i in N['train'].items()}
                         )

# iii) Sensitivity respect to beta of BPR function

#B) Two attributes case:

# i) Estimate of theta versus real theta
delta_theta_t = 2e-2
theta_t_range = np.arange(theta_true['tt'] - delta_theta_t, theta_true['tt'] + delta_theta_t, 0.01)


# constraints_Z1 = [theta_Z[0],theta_Z[1]]
constraints_Z1 = [theta_true_Z[0], np.nan]
# constraints_Z1 = [np.nan,np.nan]

for i in N.keys():



    # for i in G_keys:
    plot.estimated_vs_theta_true(filename='Two attribute route choice. Network ' + str(i),
                                 theta_t_range = theta_t_range
                                 , Q = N[i].Q, M = N[i].M, D = N[i].D
                                 , links = N[i].links
                                 , Z = Z[i]
                                 , theta_Z = theta_true_Z, constraints_Z = constraints_Z1)

#c) Three-attribute case:

# i) Estimate of theta versus real theta
delta_theta_t = 1e-1
theta_t_range = np.arange(theta_true['tt'] - delta_theta_t, theta_true['tt'] + delta_theta_t, 0.01)

constraints_Z1 = [np.nan,np.nan]

for i in N.keys():
    plot.estimated_vs_theta_true(filename='Three attribute route choice. Network ' + str(i),
                                 theta_t_range = theta_t_range
                                 , Q = N[i].Q, M = N[i].M, D = N[i].D
                                 , links = N[i].links
                                 , Z = Z[i]
                                 , theta_Z = theta_true_Z, constraints_Z = constraints_Z1)

#Multiattribute decisions (bias in theta is only a single attribute is considered)


################## Chunks to review #########

# isl.writer.write_network_to_dat(root =  root_pablo
#                                 , network_name = "Custom3" , prefix_filename = 'custom3', N = N['train']['N3'])
#
# isl.equilibrium.sue_logit_dial(root = root_pablo, network_name = 'Custom3', prefix_filename = 'custom3', maxIter = 100, accuracy = 0.01, theta = {'tt':1})
# #
# isl.writer.write_network_to_dat(root =  root_pablo
#                                 , network_name = "Custom4" , prefix_filename = 'custom4', N = N['train']['N4'])
#
# isl.equilibrium.sue_logit_dial(root = root_pablo, network_name ='Custom4', prefix_filename ='custom4', maxIter = 100, accuracy = 0.01, theta = {'tt':1})

# isl.writer.write_network_to_dat(root =  root_pablo
#                                 , network_name = "Random5" , prefix_filename = 'random5', N = N['train']['N5'])
#
# isl.equilibrium.sue_logit_dial(root = root_pablo, network_name ='Random5', prefix_filename ='random5', maxIter = 100, accuracy = 0.01, Z_dict = N['train']['N5'].Z_dict, theta = theta_true['N5'], features = ['wt','c'])
#
# isl.writer.write_network_to_dat(root =  root_pablo
#                                 , network_name = "Random6" , prefix_filename = 'random6', N = N['train']['N6'])
#
# results_sue_dial = {}
#
# results_sue_dial['x'],results_sue_dial['tt_x'] = isl.equilibrium.sue_logit_dial(root = root_pablo, network_name ='Random6', prefix_filename ='random6'
#                                , options = {'equilibrium': 'stochastic', 'method': 'MSA', 'maxIter': 100, 'accuracy_eq': 0.01}
#                                , Z_dict = N['train']['N6'].Z_dict, theta = theta_true['N6'], features = ['wt','c'])
#
# N['train']['N6'].x_dict = results_sue_dial['x']
# N['train']['N6'].set_Y_attr_links(y=results_sue_dial['tt_x'], label='tt')

# maxIter = 20
# results_sue_msa = {}

# for i in subfolders_tntp_networks:
#     # To get estimate of the logit parameters is necessary sometimes to increase the number of iterations so higher accuracy is achieved
#     # 200 works great for networks with less than 1000 links but more iterations are needed for larger networks and this increase computing time significantly
#     t0 = time.time()
#     results_sue_msa[i] = isl.equilibrium.sue_logit_msa_k_paths(N = N['train'][i], maxIter = maxIter, accuracy = 0.01, theta = theta_true[i])
#     print('time: ' + str(np.round(time.time()-t0,1)) + '[s]')
#     print('time per iteration: ' + str(np.round((time.time()-t0)/maxIter,1))+ '[s]')
#
#     N['train'][i].x_dict = results_sue_msa[i]['x']# #
#     N['train'][i].set_Y_attr_links(y=results_sue_msa[i]['tt_x'], label='tt')

################ Setup particular network (and get observed link count) ############

# x_current = None

# maxIter = 100 #If there is no enough amount of iterations, the equilibrium solution will have a lot of errors that will affect inference later via SSE.
#
# theta_true[i].keys()
# # i = 'N6'
# i = 'SiouxFalls'
# t0 = time.time()
# theta_test = theta_true[i].copy()
# # theta_test['tt'] = theta_test['tt']*0.01
# results_sue_msa[i] = isl.equilibrium.sue_logit_msa_k_paths(N=N['train'][i], theta=theta_true[i], features_Y=features_Y, features=features,
#                                                            params= {'maxIter': maxIter, 'accuracy_eq': 0.01})
# # x_current = np.array(list(results_sue_msa[i]['x'].values()))
# # x_current = -10*x_current/x_current
# print('time: ' + str(np.round(time.time()-t0,1)) + '[s]')
# print('time per iteration: ' + str(np.round((time.time()-t0)/maxIter,1))+ '[s]')
# print(results_sue_msa[i])
# N['train'][i].x_dict = results_sue_msa[i]['x']# #
# N['train'][i].set_Y_attr_links(y=results_sue_msa[i]['tt_x'], label='tt')
# N['train'][i].x = np.array(list(N['train'][i].x_dict.values()))


# N['train'][i].x.shape

# print(results_sue_msa[i])
# Note that with gaps of around 8.65% we can still achieve perfect accuracy estimating logit parameters.


# plt.plot(df_bilevel['iter'], df_bilevel['error'])
# plt.plot(df_bilevel['iter'], df_bilevel['vot'])
# plt.show()

################ Vectorized OD-theta estimation in Uncongested networks ############

# theta = 100*np.array(list(theta0.values()))[:, np.newaxis]
# theta = np.zeros(len(theta))[:, np.newaxis]
# theta = -100 ** np.ones(len(np.array(list(theta0.values()))))[:, np.newaxis]
# theta = -20*np.ones(len(theta))[:, np.newaxis]

# #Prescaling
#
# theta_myalg, grad_myalg = isl.estimation.single_level_odtheta_estimation(M = {1: N['train'][i].M}
#                     , C={1: isl.estimation.choice_set_matrix_from_M(N['train'][i].M)}
#                     , D = {1: N['train'][i].D}
#                     , q0= isl.network.denseQ(Q=N['train'][i].Q, remove_zeros=remove_zeros_Q)
#                     , features_Y = features_Y, features = config.estimation_options['features']
#                     , Y = {1:N['train'][i].Y_dict}, Z = {1:N['train'][i].Z_dict}
#                     , x = {1:N['train'][i].x} #x_N
#                     , theta0 = dict.fromkeys([*features_Y,*config.estimation_options['features']],0)
#                     # , theta0 = {k: 2*theta_true[i][k] for k in [*features_Y,*config.estimation_options['features']]}
#                     , opt_params = {
#                                     # 'method': 'gauss-newton'
#                                     #    'method': 'gd'
#                                     'method': 'ngd'
#                                     , 'iters_scaling': int(1e1), 'iters': int(0e3)
#                                     , 'eta_scaling': 1, 'eta': 1e-1
#                     #'eta': 1 works well for Sioux falls
#                                     , 'gamma': 1, 'batch_size': 0}
#                                                           )
#
# print(theta_myalg[0] / theta_myalg[2])


# i = 'SiouxFalls'
# i = 'N3'
# isl.estimation.

# features.append('k0')

theta_myalg, grad_myalg, final_objective = isl.estimation.solve_outerlevel_lue(k_Y=k_Y, Yt={1: N['train'][i].Y_dict},
                                                                               k_Z=k_Z, Zt={1: N['train'][i].Z_dict},
                                                                               q0=isl.networks.denseQ(Q=N['train'][i].Q,
                                                                                                                  remove_zeros=N['train'][i].setup_options['remove_zeros_Q']),
                                                                               xct={1: N['train'][i].x},
                                                                               Mt={1: N['train'][i].M},
                                                                               Dt={1: N['train'][i].D},
                                                                               theta0=dict.fromkeys([*k_Y, *k_Z], 0),
                                                                               outeropt_params={
                                                                                               # 'method': 'gauss-newton'
                                                                                               #    'method': 'gd'
                                                                                               'method': 'ngd'
                                                                                               , 'iters_scaling': int(0e2),
                                                                                               'iters': int(3e1)
                                                                                               , 'eta_scaling': 1, 'eta': 2e-1
                                                                                               # 'eta': 1 works well for Sioux falls
                                                                                               , 'gamma': 0, 'batch_size': 0})

print(theta_myalg)
print(theta_myalg['tt'] / theta_myalg['c'])
print(final_objective)

# Gradient descent or gauss newthon fine scale optimization

# isl.estimation.
theta_myalg_adjusted, grad_myalg, final_objective = isl.estimation.solve_outerlevel_lue(k_Y=k_Y,
                                                                                        Yt={1: N['train'][i].Y_dict},
                                                                                        k_Z=k_Z,
                                                                                        Zt={1: N['train'][i].Z_dict},
                                                                                        q0=isl.networks.denseQ(
                                                                                                        Q=N['train'][i].Q,
                                                                                                        remove_zeros=remove_zeros_Q),
                                                                                        xct={1: N['train'][i].x},
                                                                                        Mt={1: N['train'][i].M},
                                                                                        Dt={1: N['train'][i].D},
                                                                                        theta0=theta_myalg,
                                                                                        outeropt_params={
                                                                                                        'method': 'gauss-newton'
                                                                                                        # 'method': 'gd'
                                                                                                        # 'method': 'newton'
                                                                                                        ,
                                                                                                        'iters_scaling': int(0e1),
                                                                                                        'iters': int(1e1)
                                                                                                        , 'eta_scaling': 1e-1,
                                                                                                        'eta': 1e-8
                                                                                                        # 1e-8 works well for Sioux falls
                                                                                                        , 'gamma': 0.1,
                                                                                                        'batch_size': 0})

theta_myalg = theta_myalg_adjusted.copy()
print(theta_myalg['tt'] / theta_myalg['c'])

#T-tests

day = 1

# YZ_x =
theta_h0 =-6
alpha = 0.05

ttest, criticalval, pval = isl.estimation.ttest_theta(theta_h0=0, theta=np.array(list(theta_myalg.values()))[:,np.newaxis],
                                                      YZ_x=isl.estimation.get_design_matrix(Y=N['train'][i].Y_dict,
                                                                                            Z=N['train'][i].Z_dict,
                                                                                            features_Z=k_Z,
                                                                                            features_Y=k_Y),
                                                      xc=N['train'][i].x[:, np.newaxis],
                                                      q=isl.networks.denseQ(Q=N['train'][i].Q,
                                                                            remove_zeros=remove_zeros_Q),
                                                      Ix=N['train'][i].D, Iq=N['train'][i].M,
                                                      C=isl.estimation.choice_set_matrix_from_M(N['train'][i].M),
                                                      alpha=0.05)

print(ttest)
print('pvals :' +  str(pval))

# ttest1, criticalval, pval = isl.estimation.ttest_theta(theta_h0 =1*np.ones(len(theta_myalg))[:,np.newaxis], alpha = 0.05
#                                                       # , theta = np.array(list({k: 1*theta_true[i][k] for k in [*features_Y,*features]}.values()))[:,np.newaxis]
#                                                       , theta = theta_myalg
#                                                       ,YZ_x = isl.estimation.get_design_matrix(Y = N['train'][i].Y_dict, Z = N['train'][i].Z_dict, features_Y = features_Y, features = features)
#                                                       , x_bar = N['train'][i].x[:, np.newaxis]
#                                                       ,q =isl.network.denseQ(Q=N['train'][i].Q,remove_zeros=remove_zeros_Q)
#                                                       , Ix = N['train'][i].D, Iq=N['train'][i].M, C = isl.estimation.choice_set_matrix_from_M(N['train'][i].M) )
#
# print(ttest1)

# pval
# ttest1
# criticalval

confint_theta, width_confint_theta = isl.estimation.confint_theta(
    theta=np.array(list(theta_myalg.values()))[:, np.newaxis],
    YZ_x=isl.estimation.get_design_matrix(Y=N['train'][i].Y_dict, Z=N['train'][i].Z_dict, features_Z=k_Z,
                                          features_Y=k_Y),
    xc=N['train'][i].x[:, np.newaxis], q=isl.networks.denseQ(Q=N['train'][i].Q, remove_zeros=remove_zeros_Q),
    Ix=N['train'][i].D, Iq=N['train'][i].M, C=isl.estimation.choice_set_matrix_from_M(N['train'][i].M), alpha=0.05)

print(confint_theta)
print(width_confint_theta)

# confint_theta.shape

# Post-scaling
# isl.estimation.
# # i = 'N3'
theta_myalg_adjusted, grad_myalg = isl.estimation.solve_outerlevel_lue(k_Y=k_Y,
                                                                       Yt={1: N['train'][i].Y_dict},
                                                                       k_Z=k_Z,
                                                                       Zt={1: N['train'][i].Z_dict},
                                                                       q0=isl.networks.denseQ(
                                                                                       Q=N['train'][i].Q,
                                                                                       remove_zeros=remove_zeros_Q),
                                                                       xct={1: N['train'][i].x},
                                                                       Mt={1: N['train'][i].M},
                                                                       Dt={1: N['train'][i].D},
                                                                       theta0={
                                                                                       k: float(theta_myalg[j])
                                                                                       for
                                                                                       j, k in zip(np.arange(
                                                                                           theta_myalg.shape[0]),
                                                                                           [*k_Y, *k_Z])},
                                                                       outeropt_params={
                                                                                       # 'method': 'gauss-newton'
                                                                                       'method': 'gd'
                                                                                       ,
                                                                                       'iters_scaling': int(1e2),
                                                                                       'iters': int(1e0)
                                                                                       , 'eta_scaling': 1e-2,
                                                                                       'eta': 1e-8
                                                                                       , 'gamma': 1,
                                                                                       'batch_size': 0})
theta_myalg = theta_myalg_adjusted


theta_true_array = np.array([theta_true['N5'][k] for k in [*k_Y, *k_Z]])
print(' theta true: ' + str(theta_true_array))

print(' theta  default python opt: ' + str(np.round(np.array(list(theta_estimate['theta'].values())).T, 2)))

print('theta myalg: ' + str(np.round(np.array(list(theta_myalg.values())), 2)))

print(theta_true_array[0] / theta_true_array[2])
# print(theta_estimate['theta']['tt']/theta_estimate['theta']['c'])
print(theta_myalg['tt'] / theta_myalg['c'])

# # GOF with true theta
# x_bar = N['train'][i].x[:, np.newaxis]
#
# l1 = np.sum((isl.estimation.prediction_x(3 * theta_true_array) - x_bar) ** 2)
#
# prediction_x(0.5 * theta_true_array,YZ_x,Ix,C,Iq)
#
# # my opt algorithm
# l2 = np.sum((prediction_x(theta) - x_bar) ** 2)
#
# l3 = np.sum((prediction_x(np.array(list(theta_estimate['theta'].values()))) - x_bar) ** 2)

# # np.mean(np.abs(x_pred - x_bar))/np.mean(x_bar)
# print(str(l1) + ' true theta loss')
# print(str(l2) + ' myalg')
# print(str(l3) + ' default python opt')

################ Solving for uncongested network with black box scipy minimize outer_optimizer ############

t0 = time.time()
theta_estimate = isl.estimation.solve_link_level_model(end_params={'theta': True, 'q': False}, Mt={1: N['train'][i].M},
                                                       Ct={1: isl.estimation.choice_set_matrix_from_M(N['train'][i].M)},
                                                       Dt={1: N['train'][i].D}, k_Y=k_Y, Yt={1: N['train'][i].Y_dict},
                                                       k_Z=k_Z, Zt={1: N['train'][i].Z_dict}, xt={1: N['train'][i].x},
                                                       idx_links={1: range(0, len(N['train'][i].x))},
                                                       scale={'mean': False, 'std': False},
                                                       q0=isl.networks.denseQ(Q=N['train'][i].Q,
                                                                              remove_zeros=remove_zeros_Q),
                                                       theta0=dict.fromkeys([*k_Y, *k_Z], -1)
                                                       , lambda_hp=0)
print('time: ' + str(np.round(time.time()-t0,1)) + '[s]')
print(theta_estimate['theta']['tt']/theta_estimate['theta']['c'])
print(theta_true[i]['tt']/theta_true[i]['c'])
print(theta_estimate['theta'])


################ Aditional analysis ############

gap_precongestion = theta_estimate['gap']

# Recompute equilibrium to check new gap after accounting for congestion
theta_estimate_congestion = theta_estimate['theta']
theta_estimate_congestion['speed'] = 0
theta_estimate_congestion['length'] = 0
theta_estimate_congestion['toll'] = 0
results_sue_msa_postcongestion = isl.equilibrium.sue_logit_iterative(Nt=N['train'][i], theta=theta_estimate_congestion,
                                                                     k_Y=k_Y, k_Z=k_Z, params={'maxIter': maxIter, 'accuracy_eq': config.estimation_options['accuracy_eq']})


gap_poscongestion = np.sqrt(np.sum((np.array(list(results_sue_msa[i]['x'].values()))-np.array(list(results_sue_msa_postcongestion['x'].values())))**2)/len(np.array(list(results_sue_msa[i]['x']))))

# Gap with initial theta (no estimation)
theta_estimate_initial = dict.fromkeys([*k_Y,*k_Z],0)
theta_estimate_initial['speed'] = 0
theta_estimate_initial['length'] = 0
theta_estimate_initial['toll'] = 0
x_initial = isl.equilibrium.sue_logit_iterative(Nt=N['train'][i], theta=theta_estimate_initial, k_Y=k_Y, k_Z=k_Z,
                                                params={'maxIter': maxIter, 'accuracy_eq': config.estimation_options['accuracy_eq']})['x']

gap_initial = np.sqrt(np.sum((np.array(list(results_sue_msa[i]['x'].values()))-np.array(list(x_initial.values())))**2)/len(np.array(list(results_sue_msa[i]['x']))))



i = 'N5'
results_sue_fiske = isl.equilibrium.sue_logit_fisk(q = isl.networks.denseQ(Q = N['train'][i].Q, remove_zeros = remove_zeros_Q)
                                                   , M = N['train'][i].M
                                                   , D = N['train'][i].D
                                                   , links = N['train'][i].links_dict
                                                   , paths = N['train'][i].paths
                                                   , Z_dict = N['train'][i].Z_dict
                                                   , k_Z = k_Z
                                                   , k_Y = k_Y
                                                   , theta = theta_true[i]
                                                   , cp_solver = 'SCS'  #'ECOS': it is faster and crashes sometimes. 'SCS' is slow
                                                   )

N['train'][i].x_dict = results_sue_fiske['x']
N['train'][i].set_Y_attr_links(y= results_sue_fiske['tt_x'], feature='tt')


print(np.round(list(results_sue_msa['x'].values()),2))
print(np.round(list(results_sue_fiske['x'].values()),2))
print(np.round(list(results_sue_dial['x'].values()),2))

print(np.round(list(results_sue_msa['tt_x'].values()),2))
print(np.round(list(results_sue_fiske['tt_x'].values()),2))
print(np.round(list(results_sue_dial['tt_x'].values()),2))



N['train'][i].Q.shape
N['train']['SiouxFalls'].Q.shape

print(np.round(list(results_sue_msa['tt_x'].values()),2))


#
# np.sum(np.round(list(results_sue_fiske['x'].values()),2))


# Sioux Falls

isl.writer.write_network_to_dat(root =  root_github
                                , subfolder = 'SiouxFalls/' , prefix_filename = 'SiouxFalls', N = N['train']['SiouxFalls'])


x,tt_x = isl.equilibrium.sue_logit_dial(root = root_github, subfolder ='SiouxFalls', prefix_filename ='SiouxFalls'
                                        , options = {'equilibrium': 'stochastic', 'method': 'MSA', 'maxIter': 100, 'accuracy_eq': config.estimation_options['accuracy_eq']} , Z_dict = N['train']['SiouxFalls'].Z_dict, theta = theta_true['SiouxFalls'], k_Z = k_Z)

N['train']['SiouxFalls'].x_dict = x
N['train']['SiouxFalls'].set_Y_attr_links(y=tt_x, feature='tt')

np.sum(list(x.values()))
tt_x.values()

# N['train']['SiouxFalls'].x_dict = dict(zip(list(N['train']['SiouxFalls'].links_dict.keys()),x_iteration))
#
#
# N['train']['SiouxFalls'].set_Y_attr_links(y=dict(zip(list(N['train']['SiouxFalls'].links_dict.keys()),[link.traveltime for link in N_i.links])), label='tt')

theta_estimate = isl.estimation.solve_link_level_model(end_params={'theta': True, 'q': False},
                                                       Mt={1: N['train']['SiouxFalls'].M}, Ct={
        1: isl.estimation.choice_set_matrix_from_M(N['train']['SiouxFalls'].M)}, Dt={1: N['train']['SiouxFalls'].D},
                                                       k_Y=k_Y, Yt={1: N['train']['SiouxFalls'].Y_dict}, k_Z=k_Z,
                                                       Zt={1: N['train']['SiouxFalls'].Z_dict},
                                                       xt={1: N['train']['SiouxFalls'].x},
                                                       idx_links={1: range(0, len(N['train']['SiouxFalls'].x))},
                                                       scale={'mean': False, 'std': False},
                                                       q0=isl.networks.denseQ(Q=N['train']['SiouxFalls'].Q,
                                                                              remove_zeros=remove_zeros_Q),
                                                       theta0=dict.fromkeys([*k_Y, *k_Z], 0), lambda_hp=0)



print(theta_estimate['theta']['tt']/theta_estimate['theta']['c'])

print(theta_true['SiouxFalls']['tt']/theta_true['SiouxFalls']['c'])



#EMA

isl.writer.write_network_to_dat(root =  root_github
                                , subfolder = 'Eastern-Massachusetts' , prefix_filename = 'EMA', N = N['train']['Eastern-Massachusetts'])

x,tt_x = isl.equilibrium.sue_logit_dial(root = root_github, subfolder ='Eastern-Massachusetts', prefix_filename ='EMA', maxIter = 100, accuracy = config.estimation_options['accuracy_eq'], Z_dict = N['train']['Eastern-Massachusetts'].Z_dict, theta = theta_true['Eastern-Massachusetts'], k_Z = k_Z)

N['train']['Eastern-Massachusetts'].x_dict = x
N['train']['Eastern-Massachusetts'].set_Y_attr_links(y=tt_x, feature='tt')

theta_estimate = isl.estimation.solve_link_level_model(end_params={'theta': True, 'q': False},
                                                       Mt={1: N['train']['Eastern-Massachusetts'].M}, Ct={
        i: isl.estimation.choice_set_matrix_from_M(N['train']['Eastern-Massachusetts'].M)},
                                                       Dt={1: N['train']['Eastern-Massachusetts'].D}, k_Y=k_Y,
                                                       Yt={1: N['train']['Eastern-Massachusetts'].Y_dict}, k_Z=k_Z,
                                                       Zt={1: N['train']['Eastern-Massachusetts'].Z_dict},
                                                       xt={1: N['train']['Eastern-Massachusetts'].x}, idx_links={
        1: range(0, len(N['train']['Eastern-Massachusetts'].x))}, scale={'mean': False, 'std': False},
                                                       q0=isl.networks.denseQ(Q=N['train']['Eastern-Massachusetts'].Q,
                                                                              remove_zeros=remove_zeros_Q),
                                                       theta0=dict.fromkeys([*k_Y, *k_Z], 0), lambda_hp=0)

theta_estimate['theta']['tt']/theta_estimate['theta']['c']
theta_true['Eastern-Massachusetts']['tt']/theta_true['Eastern-Massachusetts']['c']

# Austin




# berlin-tiergarten_net.tntp


isl.writer.write_network_to_dat(root =  root_github
                                , subfolder = 'Eastern-Massachusetts' , prefix_filename = 'EMA', N = N['train']['Eastern-Massachusetts'])

x,tt_x = isl.equilibrium.sue_logit_dial(root = root_github, subfolder ='Eastern-Massachusetts', prefix_filename ='EMA', maxIter = 100, accuracy = config.estimation_options['accuracy_eq'], Z_dict = N['train']['Eastern-Massachusetts'].Z_dict, theta = theta_true['Eastern-Massachusetts'], k_Z = k_Z)

N['train']['Eastern-Massachusetts'].x_dict = x
N['train']['Eastern-Massachusetts'].set_Y_attr_links(y=tt_x, feature='tt')

theta_estimate = isl.estimation.solve_link_level_model(end_params={'theta': True, 'q': False},
                                                       Mt={1: N['train']['Eastern-Massachusetts'].M}, Ct={
        i: isl.estimation.choice_set_matrix_from_M(N['train']['Eastern-Massachusetts'].M)},
                                                       Dt={1: N['train']['Eastern-Massachusetts'].D}, k_Y=k_Y,
                                                       Yt={1: N['train']['Eastern-Massachusetts'].Y_dict}, k_Z=k_Z,
                                                       Zt={1: N['train']['Eastern-Massachusetts'].Z_dict},
                                                       xt={1: N['train']['Eastern-Massachusetts'].x}, idx_links={
        1: range(0, len(N['train']['Eastern-Massachusetts'].x))}, scale={'mean': False, 'std': False},
                                                       q0=isl.networks.denseQ(Q=N['train']['Eastern-Massachusetts'].Q,
                                                                              remove_zeros=remove_zeros_Q),
                                                       theta0=dict.fromkeys([*k_Y, *k_Z], 0), lambda_hp=0)



od_filename = [_ for _ in os.listdir(os.path.join(root_github, subfolder_github)) if 'trips' in _ and _.endswith('tntp')]
prefix_filename = od_filename[0].partition('_')[0]

isl.equilibrium.sue_logit_dial(root = root_github, subfolder = subfolder_github, prefix_filename = prefix_filename, maxIter = 100, accuracy = config.estimation_options['accuracy_eq'], theta = {'tt':1})


i = 'N5'
i = 'N6'

N['train']['N6'].Q

N['train']['N6'].Z_dict

while valid_network is None:
    try:
        results_sue['train'] = {i: isl.equilibrium.sue_logit_fisk(q = isl.networks.denseQ(Q = N['train'][i].Q, remove_zeros = remove_zeros_Q)
                                                                  , M = N['train'][i].M
                                                                  , D = N['train'][i].D
                                                                  , links = N['train'][i].links_dict
                                                                  , paths = N['train'][i].paths
                                                                  , Z_dict = N['train'][i].Z_dict
                                                                  , k_Z = k_Z
                                                                  , k_Y = k_Y
                                                                  , theta = theta_true[i]
                                                                  , cp_solver = 'SCS'  #'ECOS': it is faster and crashes sometimes. 'SCS' is slow
                                                                  )
                                for i in N['train'].keys()}

    except:
        print('error'+ str(i)+ '\n  Cloning network and trying again')
        for i in N['train'].keys():
            exceptions['SUE']['train'][i] += 1

        N['train'] = \
            isl.networks.clone_network(N=N['train'][i], label='Train', randomness = {'Q':True, 'BPR':True, 'Z': False, 'var_Q':0}
                                       )

        # isl.network.clone_networks(N=N['train'], label='Train'
        #                            , R_labels=R_labels
        #                            , randomness={'Q': True, 'BPR': True, 'Z': False, 'var_Q': 0}
        #                            , q_range=q_range, remove_zeros_Q=remove_zeros_Q, Z_attrs_classes=Z_attrs_classes,
        #                            bpr_classes=bpr_classes, cutoff_paths=cutoff_paths
        #                            , fixed_effects=fixed_effects, n_paths=n_paths
        #                            )

        pass
    else:
        valid_network = True

        # Store travel time, link and path flows in Network objects
        for i in N['train'].keys():
            N['train'][i].set_Y_attr_links(y=results_sue['train'][i]['tt_x'], feature='tt')
            N['train'][i].x_dict = results_sue['train'][i]['x']
            N['train'][i].f_dict = results_sue['train'][i]['f']


results_sue['train']['Braess-Example'] #ECOS fails because an overflow of the link that has free flow travel time of 2e10.





def main():
    import runpy
    runpy.run_path(os.getcwd() + '/examples/local/production/od-theta-example.py')
