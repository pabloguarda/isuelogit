
import transportAI.equilibrium
import transportAI.networks
import numpy as np

def sue_logit_simulation_recovery(N, theta, constraints_theta, k_Z, remove_zeros, scale_features):

    result_sue = transportAI.equilibrium.sue_logit_fisk(M=N.M
                                                        , D=N.D
                                                        , q = transportAI.networks.denseQ(Q=N.Q, remove_zeros=remove_zeros)
                                                        , links=N.links_dict
                                                        , paths=N.paths
                                                        , Z_dict = N.Z_dict
                                                        , k_Z = k_Z
                                                        , theta=theta
                                                        )

    Y_links = np.hstack(list(result_sue['tt_x'].values()))
    Y_routes = Y_links @ N.D

    # # + Exogenous attributes from link to path level (rows)
    Z_links = transportAI.estimation.get_matrix_from_dict_attrs_values(N.Z_dict)
    Z_routes= (Z_links.T @ N.D).T

    likelihood_logit = transportAI.estimation.likelihood_path_level_logit(f=np.array(list(result_sue['f'].values()))
                                                                          , M=N.M
                                                                          , k_Z= list(N.Z_dict.keys())
                                                                          , Z=Z_routes
                                                                          , k_Y=['tt']
                                                                          , Y= Y_routes
                                                                          , scale=scale_features
                                                                          )

    # Maximize likelihood to obtain solution
    result_logit = transportAI.estimation.solve_path_level_logit(cp_ll=likelihood_logit['cp_ll']
                                                                 , cp_theta=likelihood_logit['cp_theta']
                                                                 , constraints_theta=constraints_theta
                                                                 )

    return result_logit