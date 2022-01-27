from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytypes import Links, Options, Vector
# Links, Matrix, ColumnVector, Links, LogitFeatures, LogitParameters, Paths, Options, Vector
from pastar import neighbors_path, paths_lengths  # These dependencies can be removed

# import torch

from paths import path_generation_nx


import cvxpy as cp
import networkx as nx
from sklearn import preprocessing
import scipy.linalg as la

import numdifftools as nd
import numdifftools.nd_algopy as nda

from autograd import elementwise_grad as egrad
from autograd import jacobian

# import numpy as np
import autograd.numpy as np

from math import e

from scipy import stats

from scipy.optimize import least_squares, fsolve, minimize
import random

import copy


# import transportAI.links
import numeric  # import round_almost_zero_flows
import equilibrium
import networks
import time
from networks import TNetwork

# https://stackoverflow.com/questions/9777783/suppress-scientific-notation-in-numpy-when-creating-array-from-nested-list
from mytypes import ColumnVector, ColumnVector, LogitParameters, Matrix, Proportion
from transportAI import printer
from utils import blockPrinting

import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True, precision=4)
# https://stackoverflow.com/questions/22222818/how-to-printing-numpy-array-with-3-decimal-places

cp_solver = ''


def get_attribute_list_by_choiceset(C, z):
    ''' Return a list with attribute values (vector, i.e. Matrix 1D) for each alternative in the choice set (only with entries different than 0)
    :arg: C: Choice set matrix
    :arg: z: single attribute values
    '''
    z_avail = []

    for i in range(C.shape[0]):
        z_avail.append(z[np.where(C[i, :] == 1)[0]])

    return z_avail


def get_matrix_from_dict_attrs_values(W_dict: dict):
    # Return Matrix Y or Z using Y or Z_dict
    listW = []
    for i in W_dict.keys():
        listW.append([float(x) for x in W_dict[i].values()])

    return np.asarray(listW).T


def get_design_matrix(Y: dict, Z: dict, k_Y: [], k_Z: []):

    if len(k_Z)>0:
        Y_x = get_matrix_from_dict_attrs_values({k_y: Y[k_y] for k_y in k_Y})
        Z_x = get_matrix_from_dict_attrs_values({k_z: Z[k_z] for k_z in k_Z})
        YZ_x = np.column_stack([Y_x, Z_x])

    else:
        Y_x = get_matrix_from_dict_attrs_values({k_y: Y[k_y] for k_y in k_Y})
        YZ_x = np.column_stack([Y_x])

    return YZ_x


def distributed_choice_set_matrix_from_M(M):
    raise NotImplementedError


def choice_set_matrix_from_M(M):
    """Wide to long format
    The new matrix has one rows per alternative
    """

    t0 = time.time()

    assert M.shape[0] > 0, 'Matrix C was not generated because M matrix is empty'

    print('Generating choice set matrix')

    wide_matrix = M.astype(int)

    if wide_matrix.ndim == 1:
        wide_matrix = wide_matrix.reshape(1, wide_matrix.shape[0])

    C = np.repeat(wide_matrix, repeats=np.sum(wide_matrix, axis=1), axis=0)

    print('Matrix C ' + str(C.shape) + ' generated in ' + str(round(time.time() - t0, 1)) + '[s]')

    return C


def v_normalization(v, C):
    '''
    :param v: this has to be an unidimensional array otherwise it returns a matrix
    :param C:
    :return: column vector with normalization
    '''

    # TODO: this function is not compatible with automatic differentiation

    # flattened = False
    if len(v.shape)>1:
        v = v.flatten()
        # flattened = True

    C = C.astype('float')
    C[C == 0] = float('nan') #np.nan
    v_max = np.nanmax(v * C, axis=1)
    # vC = v * C
    # vC[np.isnan(vC)] = float('-inf')


    # Without using npnanmax
    # C = C.astype('float')
    # C[C == 0] = np.nan
    # # v_max = np.nanmax(v * C, axis=1)
    # vC = v * C
    # vC[np.isnan(vC)] = -np.inf
    #
    # v_max = np.amax(vC, axis=1)

    return (v - v_max)[:,np.newaxis]



def widetolong(wide_matrix):
    """Wide to long format
    The new matrix has one rows per route
    """

    if wide_matrix.ndim == 1:
        wide_matrix = wide_matrix.reshape(1, wide_matrix.shape[0])

    long_matrix = np.repeat(wide_matrix, repeats=np.sum(wide_matrix, axis=1), axis=0)

    return long_matrix


def non_zero_matrix(M):
    # # Remove null rows
    M = M[~np.all(M == 0, axis=1)]
    # Remove null columns
    mask = (M == 0).all(0)
    column_indices = np.where(mask)[0]
    M = M[:, ~mask]

    return M


def binaryLogit(x1, x2, theta):
    """ Binary logit model """

    sum_exp = np.exp(theta * x1) + np.exp(theta * x2)
    p1 = np.exp(theta * x1) / sum_exp
    p2 = 1 - p1

    return np.array([p1, p2])  # (P1,P2)


def softmax_probabilities(X, theta, avail=1):
    """ Multinomial logit (or softmax) probabilities
    :arg avail

    """
    # AVAIL

    # Z = X - np.amax(X,axis = 0).reshape(X.shape[0],1)
    Z = X

    return avail * np.exp(Z * theta) / np.sum(np.exp(Z * theta) * avail, axis=1).reshape(Z.shape[0], 1)


def stable_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator

    return softmax


def attribute_matrix_tolist(matrix, remove_zeros=True):
    '''This function assumed that the elements with 0 entries are the
    non-available alternatives
    '''
    list = []
    for row_index in range(matrix.shape[0]):
        row = matrix[row_index, :]
        if remove_zeros:
            list.append(row[row != 0])
    return list


def scaling_routes_attributes_onebyone(Y, Z, M, scale):
    # Scale attributes: TODO: Speed up this method and encapsulate it in its own function

    row = 0
    sum_acc = 0

    wide_M = widetolong(M)
    nozero_M = non_zero_matrix(M)

    Y_c, Z_c = Y, Z

    Y_C = attribute_matrix_tolist(wide_M * Y, remove_zeros=True)
    Z_C = [attribute_matrix_tolist(wide_M * Z_c[:, i], remove_zeros=True) for i in range(Z_c.shape[1])]

    for i in range(len(Y_C)):
        # Scaling of matrix with attribtue values of choice set
        Y_C[i] = preprocessing.scale(Y_C[i], with_mean=scale['mean'], with_std=scale['std'])
        # Scaled fravel time in chosen route obtained from tt_long
        Y_c[i] = Y_C[i][i - sum_acc]

        if (i + 1) == len(Y_C) or nozero_M[row, i + 1] == 0:
            sum_acc += sum(nozero_M[row, :])
            row += 1

    Z_c = {}
    for Z_i in range(len(Z_C)):
        row = 0
        sum_acc = 0
        Z_c[Z_i] = {}
        for route_j in range(len(Z_C[Z_i])):

            Z_C[Z_i][route_j] = preprocessing.scale(Z_C[Z_i][route_j], with_mean=scale['mean'],
                                                    with_std=scale['std'])
            Z_c[route_j][Z_i] = Z_C[Z_i][route_j][route_j - sum_acc]
            # Z_routes[i][j] = Z_routes[i][j]
            if (route_j + 1) == len(Z_C[Z_i]) or nozero_M[row, route_j + 1] == 0:
                sum_acc += sum(nozero_M[row, :])
                row += 1

    return Y_C, Y_c, Z_C, Z_c


def prediction_error_logit_path(f, M, V_c, V_C, cp_theta, theta):
    cp_theta_all = {**cp_theta['Y'], **cp_theta['Z']}

    # Set parameters values to modify utility values and then make predictions
    for k, v in cp_theta_all.items():
        if v is not np.nan:
            cp_theta_all[k].value = theta[k]

    P = []
    pred_F = []
    q_long = f.dot(M.T) @ M  # Demand of the OD pair associated to each route

    for i in range(len(f)):
        # Stable softmax
        max_v = max(V_C[i].value)
        P.append(np.exp(V_c[i].value - max_v) / np.sum(np.exp(V_C[i].value - max_v)))
        # P.append(np.exp(V_c[i].value) / np.sum(np.exp(V_C[i].value)))
        pred_F.append(P[i] * q_long[i])

    error = np.linalg.norm(np.array(pred_F) - f, ord=2)

    return error


def likelihood_path_level_logit(f: np.array, M: np.array, D: np.array
                                , k_Y: list, Y: np.array, k_Z: list,
                                Z: {}, scale={'mean': False, 'std': False}):
    """Logit model fitted from output from SUE

    Arguments
    ----------
    :argument f: vector with path flows
    :argument M: Path/O-D demand incidence matrix Random
    :argument D: Path/link incidence matrix
    :argument Y: Matrix with attributes values of chosen routes or links that are (endogenous) flow dependent  (n_routes X n_attributes)
    :argument k_Y: list of labels of attributes in T that are used to fit the discrete choice model
    :argument Z: Dictionary with attributes values of chosen routes or links that are (exogenous) not flow dependent  (n_routes X n_attributes)
    :argument k_Z: list of labels of attributes in Z that are used to fit the discrete choice model

    Returns
    -------
    ll: likelihood function obtained from cvxpy
    cp_theta: dictionary with lists of cvxpy.Variable. Keys: 'Y' for flow dependent and 'Z' for non flow dependent variables
    V_c_z: List with utility function for chosen alternative
    V_C_z: List with utility functions for alternatives in choice set

    """

    # Matrix of values using subset of variables k_Y and k_Z selected for estimation
    Y = (get_matrix_from_dict_attrs_values({k_y: Y[k_y] for k_y in k_Y}).T @ D).T
    Z = (get_matrix_from_dict_attrs_values({k_z: Z[k_z] for k_z in k_Z}).T @ D).T

    # - Optimization variables
    # cp_theta = {k: cp.Parameter(nonpos=True) for k in Z_dict.keys()}
    cp_theta = {}
    cp_theta['Y'] = {k: cp.Variable(nonpos=True) for k in k_Y}
    cp_theta['Z'] = {k: cp.Variable(nonpos=True) for k in k_Z if k not in k_Y}

    # - Input
    flow_routes = numeric.round_almost_zero_flows(f)

    # Wide M matrix that serves as a choice set matrix (C)
    C = choice_set_matrix_from_M(M)

    # Attributes of chosen alternatives
    Y_c, Z_c = Y, Z

    # TODO: the scaling needs to be done by choice set to give the same solution of logit model.
    if scale['mean'] or scale['std']:
        # Scaling by attribute
        Y_c = preprocessing.scale(Y, with_mean=scale['mean'], with_std=scale['std'], axis=0)
        Z_c = preprocessing.scale(Z, with_mean=scale['mean'], with_std=scale['std'], axis=1)

    # Attributes in alternatives within choice sets
    Y_C = [get_attribute_list_by_choiceset(C=C, z=Y_c[:, i]) for i in range(Y_c.shape[1])]
    Z_C = [get_attribute_list_by_choiceset(C=C, z=Z_c[:, i]) for i in range(Z_c.shape[1])]

    # Loglikelihood function obtained from iterating across OD pairs
    ll = []

    # Utilities for chosen alternative and choice set
    V_c = []
    V_C = []

    for i in range(len(flow_routes)):

        # i = 0
        # List storing the contribution from each choice (expansion) set to the likelihood
        ll_i = []

        # if len(tt_long[i]) > 1:  # If there is a single path, then no information is added for estimation.
        # Z = Z[i]

        V_c_i = [Y_c[i] * cp_theta['Y']['tt'] + Z[i, :] * cp.hstack(list(cp_theta['Z'].values()))]

        V_C_i = []
        for j in range(Y.shape[1]):
            V_C_i.append(Y_C[j][i] * list(cp_theta['Y'].values())[j])

        for j in range(Z.shape[1]):
            V_C_i.append(Z_C[j][i] * list(cp_theta['Z'].values())[j])

        V_c.append(cp.sum(V_c_i))  # Utility of chosen alternative
        V_C.append(cp.sum(V_C_i))  # Utility vector for alternatives in choice set

        ll_i.append((cp.sum(V_c_i) - cp.log_sum_exp(cp.sum(V_C_i))) * flow_routes[i])

        ll.append(cp.sum(ll_i))

    return {'cp_ll': ll, 'cp_theta': cp_theta, 'V_c': V_c, 'V_C': V_C}


def cp_regularizer(beta, p):
    if p == 0:
        return 0
    else:
        return cp.pnorm(beta, p=p) ** p


def solve_path_level_logit(cp_ll, cp_theta, constraints_theta: list, lambdas=np.array([0]), r=1, cp_solver=cp_solver):
    """
    Arguments
    ----------
    :argument cp_ll: likelihood function obtained from cvxpy
    :argument constraints_theta: list that defines which attributes ignore and includes in the maximization of the likelihood
    :argument r: degree of regularizer (p = 1 is lasso regularization, and p = 2 is Ridge regression
    :argument lambdas: array with the range of values for lambda. If no lambda is provided, the array has the element 0.
    :argument cp_theta: list of cvxpy.Variable

    Returns
    -------
    results: TODO: use Results structure (see pylogit and other packages)

    """

    # - Constraints (for T and Z attributes)
    cp_constraints_theta = [cp_theta['Z'][k] == v for k, v in constraints_theta['Z'].items() if v is not np.nan]
    cp_constraints_theta.extend([cp_theta['Y'][k] == v for k, v in constraints_theta['Y'].items() if v is not np.nan])

    # Regularization term
    cp_lambda = cp.Parameter(nonneg=True)

    # TODO: implement softmax trick to avoid overflow and exceptions

    # Objective
    n = 1  # TODO: Define proper normaliztion constant, e.g. len(cp_ll) #Number of choice scenarios to normalize the likelihood
    cp_objective = cp.Maximize(
        cp.sum(cp_ll) / n - cp_lambda * cp.norm(cp.hstack({**cp_theta['Y'], **cp_theta['Z']}.values()), r))
    # cp_objective = cp.Maximize(cp.sum(cp_ll))
    # Problem
    cp_problem = cp.Problem(cp_objective, constraints=cp_constraints_theta)  # Excluding extra attributes

    # Fitting hyperparameter of regularization
    results = {}
    for i, lambda_i in zip(range(len(lambdas)), lambdas):
        cp_lambda.value = lambda_i
        try:
            cp_problem.solve(cp_solver)  # (solver = solver) # solver = 'ECOS', solver = 'SCS'

        except:
            pass  # Ignore invalid entries of lambda when the optimizer fails.
            # theta_Z = {k: '' for k in cp_theta['Z'].keys()} #None
            # theta_Y = {k: '' for k in cp_theta['Y'].keys()} #None

        else:
            theta_Z = {k: v.value for k, v in cp_theta['Z'].items()}
            theta_Y = {k: v.value for k, v in cp_theta['Y'].items()}
            results[i] = {'lambda': cp_lambda.value, 'theta_Y': theta_Y, 'theta_Z': theta_Z}

    return results


def prediction_error_logit_regularization(theta_estimates: {}, lambda_vals: {}, likelihood: {}, f: {}, M: {}):
    errors_logit = {}
    theta_vals = {}
    lambda_valid_vals = {}

    for N_i in lambda_vals.keys():
        errors_logit[N_i] = {}
        theta_vals[N_i] = []
        lambda_valid_vals[N_i] = []
        n_paths = np.sum(M[N_i])

        for iter, val in lambda_vals[N_i].items():
            # theta_values = {**val['theta_Y'],**val['theta_Z']}  # From training

            try:
                raw_error = prediction_error_logit_path(
                    f=f[N_i]
                    , M=M[N_i]
                    , V_c=likelihood[N_i]['V_c']
                    , V_C=likelihood[N_i]['V_C']
                    , cp_theta=likelihood[N_i]['cp_theta']
                    , theta=theta_estimates[N_i][iter])
            except:
                pass
                # errors_logit['train'][i][iter] = ''

            else:
                theta_vals[N_i].append(theta_estimates[N_i])
                lambda_valid_vals[N_i].append(lambda_vals[N_i][iter])
                errors_logit[N_i][iter] = np.round(raw_error, 4) / n_paths

    return errors_logit, lambda_valid_vals


# def X_scaling(x,)

def compute_loss_link_level_model(theta, q, lambda_hp, D, M, C, Y, Z, x, idx_links, norm_o, norm_r):
    v = np.exp(v_normalization(v=np.sum(theta * np.hstack([np.vstack(Y), Z]),
                                        axis=1), C=C))

    V = C.dot(v)  # Denominator logit functions

    p = v / V

    # TODO: May give directly dot product of q and M to speed up as they do not change over iterations
    f = np.multiply(q.dot(M), p)

    # TODO: I may provide directly with the small matrix D
    x_hat = D[idx_links].dot(f)
    x_N = x[idx_links]

    objective_term = np.linalg.norm(x_hat - x_N, ord=norm_o)
    regularization_term = lambda_hp * np.linalg.norm(theta, ord=norm_r)

    loss = objective_term / len(idx_links) + regularization_term

    return loss


def loss_link_level_model(parameters, range_theta, range_q, end_params, q, theta, lambda_hp, D, M, C, Y, Z, x,
                          idx_links, norm_o,
                          norm_r):  # , scale = {'mean': True, 'std': False}
    '''

    :param o: o-norm of the objective function gap
    :param p: p-norm of the regularization term
    '''

    loss = {}

    # parameters = x0

    # theta0 = parameters[range_theta]
    # q0 = parameters[range_q]

    for i in M.keys():

        # n_theta = Y[i].shape[1] + Z[i].shape[1]
        if end_params['theta'] and end_params['q']:
            theta0 = parameters[range_theta]  # parameters[0:n_theta]
            q0 = parameters[range_q]  # parameters[n_theta:]

        elif end_params['theta']:
            theta0 = parameters
            q0 = q  # [i]

        elif end_params['q']:
            q0 = parameters
            theta0 = theta

        loss[i] = compute_loss_link_level_model(theta=theta0, q=q0, lambda_hp=lambda_hp, D=D[i], M=M[i], C=C[i], Y=Y[i],
                                                Z=Z[i]
                                                , x=x[i], idx_links=idx_links[i], norm_o=norm_o, norm_r=norm_r)

    return np.sum(np.array(list(loss.values())))  # objective_term/len(idx_links)


# def loss_link_level_model(theta, lambda_hp, D, M, Y, Z, q, x, idx_links, norm_o = 2, norm_r=1): #, scale = {'mean': True, 'std': False}
#     '''
#
#     :param o: o-norm of the objective function gap
#     :param p: p-norm of the regularization term
#     '''
#
#     objective_term = {}
#     regularization_term = {}
#     loss = {}
#
#     # theta = x0
#
#     for i in M.keys():
#
#         theta_estimate = np.array(theta)
#
#         # theta_estimate = np.ones(np.array(theta).shape)
#
#         v = np.exp(v_normalization(v=np.sum(theta_estimate * np.hstack([np.vstack(Y[i]), Z[i]]),
#             axis=1), M=M[i]))
#
#         V = choice_set_matrix_from_M(M[i]).dot(v)  # Denominator logit functions
#
#         p = v / V
#
#         # np.multiply(q[i].dot(M[i]), p)
#         #
#         # M[i].dot(np.multiply(np.ones(q[i].shape).dot(M[i]),p))
#
#         f = np.multiply(q[i].dot(M[i]),p)
#
#         # M[i].dot(f)
#
#         x_hat = D[i][idx_links[i]].dot(f)
#         x_N = x[i][idx_links[i]]
#
#         # if scale['mean'] or scale['std']:
#         #     # Scaling by attribute
#         #     Y = preprocessing.scale(Y, with_mean=scale['mean'], with_std=scale['std'], axis=0)
#         #     Z = preprocessing.scale(Z, with_mean=scale['mean'], with_std=scale['std'], axis=1)
#
#
#         # if lambda_hp == None:
#         #     return x_hat[idx_links] -x[idx_links]
#         #
#         # else:
#         objective_term[i] = np.linalg.norm(x_hat -x_N, ord = norm_o)
#         regularization_term[i] = lambda_hp*np.linalg.norm(theta, ord=norm_r)
#
#         loss[i] = objective_term[i]/len(idx_links[i]) + regularization_term[i]
#
#     return np.sum(np.array(list(loss.values()))) #objective_term/len(idx_links)

def prediction_x(theta, YZ_x, Ix, C, Iq, q, p_f: ColumnVector = None):
    # Link and path utilities
    # v_x = YZ_x.dot(theta)

    # A = Iq.T.dot(Iq)

    # Path probabilities
    if p_f is None:
        p_f = path_probabilities(theta, YZ_x, Ix, C)

    # f = np.multiply(Iq.T.dot(q), p_f)
    x_pred = Ix.dot(np.multiply(Iq.T.dot(q), p_f))  # Ix.dot(f)

    return x_pred

def generate_fresno_pems_counts(links: Links, data: pd.DataFrame, flow_attribute: str, flow_factor = 1) -> {}:

    """
    Generate masked fresno counts

    :param links:
    :param data:
    :return:
    """

    print('\nMatching PEMS traffic count measurements in network links')

    xct = np.empty(len(links))
    xct[:] = np.nan
    xct_dict = {}

    n_imputations = 0
    n_perfect_matches = 0

    for link, i in zip(links, range(len(links))):
        # if link.pems_station_id == data['station_id'].filter(link.pems_station_id):

        station_rows = data[data['station_id'].isin(link.pems_stations_ids)]

        if flow_attribute == 'flow_total':

            if len(station_rows) > 0:
                xct[i] = np.mean(station_rows[flow_attribute])*flow_factor

        if flow_attribute == 'flow_total_lane':
            lane = link.Z_dict['lane']
            lane_label = flow_attribute + '_' + str(lane)

            if len(station_rows) > 0:

                lane_counts = np.nan

                if np.count_nonzero(~np.isnan(station_rows[lane_label])) > 1:

                    # The complicated cases are those were more than one station gives no nas and the counts are different.
                    lane_counts = np.nanmean(station_rows[lane_label])

                elif np.count_nonzero(~np.isnan(station_rows[lane_label])) == 1:

                    # If two stations are matched but only one gives a no nan link flows, this means that
                    # the list of pems stations ids can be safely reduced

                    link.pems_stations_ids = [link.pems_stations_ids[np.where(~np.isnan(station_rows[lane_label]))[0][0]]]

                    lane_counts = list(station_rows[lane_label])[0]

                else:

                    # Theses cases requires imputation

                    # print(station_rows[lane_label])

                    #TODO: review these cases and maybe just define them as outliers
                    pass




                # Imputation: If no counts are available for the specific lane, we may the average over the non nan values for imputation

                if np.isnan(float(lane_counts)):

                    n_imputations += 1

                    lanes_labels = [flow_attribute + '_' + str(j) for j in np.arange(1,9)]

                    lanes_flows = [np.nansum(station_rows[lanes_labels[j]]) for j in range(len(lanes_labels))]

                    lanes_non_na_count = [np.count_nonzero(~np.isnan(station_rows[lanes_labels[j]])) for j in range(len(lanes_labels))]

                    total_no_nan_lanes = np.sum(lanes_non_na_count)

                    xct[i] = np.nansum(lanes_flows)/total_no_nan_lanes

                else:
                    xct[i] = lane_counts
                    n_perfect_matches += 1

        xct_dict[link.key] = xct[i]

    print(n_perfect_matches, 'links were perfectly matched')
    print(n_imputations, 'links counts were imputed using the average traffic counts among lanes')

    return xct_dict

def adjusted_counts_by_link_capacity(Nt, xct):

    x_bar = np.array(list(xct.values()))[:, np.newaxis]

    link_capacities = np.array([link.bpr.k for link in Nt.links])

    # current_error_by_link = error_by_link(x_bar, x_eq, show_nan=False)

    # matrix_error = np.append(
    #     np.append(np.append(x_eq[idx_nonas], x_bar[idx_nonas], axis=1), current_error_by_link, axis=1), link_capacities,
    #     axis=1)

    x_bar_adj = copy.deepcopy(x_bar)

    counter = 0

    adj_factors = []

    for i in range(x_bar.shape[0]):

        adj_factors.append(np.isnan)

        if not np.isnan(x_bar.flatten()[i]):

            factor = 1

            while x_bar_adj.flatten()[i] > link_capacities[i]:

                factor += 1

                if factor == 2:
                    counter+=1

                x_bar_adj[i] = x_bar[i]/factor


            adj_factors[i] = factor

    print('A total of ' + str(counter) + ' links counts were adjusted by capacity')

    links_keys = [(key[0],key[1]) for key in list(xct.keys())]

    # Link capaciies higher than 10000 are show as inf and the entire feature as a string to reduce space when printing

    link_capacities_print = []

    for i in range(len(link_capacities)):
        if link_capacities[i] > 10000:
            link_capacities_print.append(float('inf'))
        else:
            link_capacities_print.append(link_capacities[i])


    link_adjustment_df = pd.DataFrame({
        'link_key': links_keys
        , 'capacity': link_capacities_print
        , 'old_counts': x_bar.flatten()
        ,'adj_counts': x_bar_adj.flatten()
        ,'adj_factor': adj_factors
    })

    mask = link_adjustment_df.isnull().any(axis=1)

    # print(link_adjustment_df[~mask].to_string())

    with pd.option_context('display.float_format', '{:0.1f}'.format):
        print(link_adjustment_df[~mask].to_string())

    return dict(zip(xct.keys(),x_bar_adj.flatten()))


def masked_observed_counts(xct: ColumnVector, idx: [], complement = False):
    """

    :param xct_hat:
    :param xct:
    :param idx:
    :param complement: if complement is True, then the complement set of idx for xct is set to nan

    :return:
        count vector with all entries in idx equal to nan.

    """

    # idx = [0, 4, 5]
    #
    # xct = np.array([1,3,4,5,6,6,66,9])[:,np.newaxis]

    xct_list = list(xct.flatten())

    if complement is False:
        for id in idx:
            xct_list[id] = np.nan

    else:
        complement_idx = list(set(list(np.arange(len(xct))))-set(idx))

        for id in complement_idx:
            xct_list[id] = np.nan


    return np.array(xct_list)[:,np.newaxis]


def fake_observed_counts(xct_hat: ColumnVector, xct: ColumnVector) -> ColumnVector:
    """

    :param xct_hat: predicted counts
    :param xct: observed counts

    :return:
        count vector with all entries corresponding to the complement of the idx set to be equal to the predicted counts entries. This way, the difference between xct and xct_hat will be zero except for the idx entries.

    """

    # idx = [0, 4, 5]
    #
    # xct = np.array(list(xct.values()))[:, np.newaxis]

    # Replace values in positions with nas using the predicted count vector values

    fake_xct = copy.deepcopy(xct)
    xct_hat_copy = copy.deepcopy(xct_hat.flatten())

    for link_id in range(xct.size):

        if np.isnan(fake_xct[link_id]):
            fake_xct[link_id] = xct_hat_copy[link_id]

    #
    # xct[idx, :] = xct_hat[idx, :]

    return fake_xct

def masked_link_counts_after_path_coverage(Nt, xct: dict, print_nan: bool = False) -> dict:
    """

    Compute t

    :return:
    """

    x_bar = np.array(list(xct.values()))[:, np.newaxis]

    # Number of paths traversing of each link with no nan observations



    # idx_nonas_t0 = np.where(~np.isnan(x_bar))[0]

    idx_no_pathcoverage = np.where(np.sum(Nt.D, axis=1) == 0)[0]

    xct_remasked = masked_observed_counts(xct = x_bar, idx= idx_no_pathcoverage)

    # idx_nonas_t1 = np.where(~np.isnan(x_bar_t1))[0]


    # print('dif in coverage', np.count_nonzero(~np.isnan(x_bar))-np.count_nonzero(~np.isnan( x_bar_remasked )))

    idx_nonas = np.where(~np.isnan(xct_remasked))[0]

    if print_nan:

        # print(dict(zip(list(xct.keys()), np.sum(Nt.D,axis = 1))))

        # print(np.sum(Nt.D, axis=1)[:,np.newaxis])

        pass

    else:

        no_nanas_keys = [list(xct.keys())[i] for i in idx_nonas.tolist()]

        # print(dict(zip(no_nanas_keys, np.sum(Nt.D,axis = 1)[idx_nonas.tolist()])))

        # print(np.sum(Nt.D, axis=1)[idx_nonas.tolist()][:,np.newaxis])

    rows_sums_D_nonans = np.sum(Nt.D, axis=1)[idx_nonas.tolist()]

    print('\nAverage number of paths traversing links with counts:', np.round(np.sum(rows_sums_D_nonans)/len(no_nanas_keys),1))

    print('\nMinimum number of paths traversing links with counts:',
          np.min(rows_sums_D_nonans))


    return dict(zip(xct.keys(),xct_remasked.flatten()))



def generate_link_counts_equilibrium(Nt, theta: np.array, k_Y: [], k_Z: [], uncongested_mode: bool, coverage: Proportion, eq_params: {}, noise_params: {} = None, n_paths: int = None, sparsity_idxs = []) -> (dict,dict):
    """

    :param Nt:
    :param theta:
    :param k_Y:
    :param k_Z:
    :param coverage: percentage of links where data is assumed to be known
    :param eq_params:
    :param noise_params:

    :return:

        dictionary of link ids and link counts

    """

    # N_copy = transportAI.tnetwork.clone_network(N=N, label='Clone', randomness = {'Q': False, 'BPR': False, 'Z': False})

    # for link in N_copy.links:
    #     # print(link.bpr.alpha)
    #     print(link.bpr.beta)

    # return N_copy

    # for link in N.links:
    #     link

    if uncongested_mode:
        # To not alter the link function, first copy the current bpr parameters and then set them to 0
        bpr_alpha = {}
        bpr_beta = {}

        for link, link_num in zip(Nt.links, range(len(Nt.links))):
            bpr_alpha[link_num] = link.bpr.alpha
            bpr_beta[link_num] = link.bpr.beta

            link.bpr.alpha = 0
            link.bpr.beta = 0

    # eq_params = {'iters': 20, 'accuracy_eq': 0.01}
    iterations = eq_params['iters']

    if 'method' in eq_params.keys():
        method = eq_params['method']
    else:
        method = 'msa'

    method_label = method

    if method_label == 'line_search':
        method_label = 'Frank-Wolfe'

    # print("\nGenerating synthetic link counts via " + method + " (" + str(int(iterations)) + ' iterations)\n')

    print("\nGenerating synthetic link counts via " + method_label)

    # Noise or scale difference in Q matrix
    Q_original = Nt.Q.copy()

    if noise_params is not None:

        if noise_params['sd_Q'] != 0:

            # If noise is applied in the q matrix, we generate a copy of the original od demand vector and set a noisy matrix meanwhile to compute equilibrium

            sd_Q = noise_params['sd_Q']

            if noise_params['sd_Q'] != 'Poisson':

                # sd is a parameter computed as proportion of the mean
                sd_Q = np.mean(Nt.Q)*noise_params['sd_Q']

            Q_noisy = Nt.random_disturbance_Q(Nt.Q, sd = sd_Q).copy()

            # Update Q matrix and dense q vector temporarily
            Nt.Q = Q_noisy
            Nt.q = networks.denseQ(Q=Nt.Q, remove_zeros=Nt.setup_options['remove_zeros_Q'])


        # Scaling error in Q matrix

        if noise_params['scale_Q'] != 1:

            assert scale_Q != 0, 'Scale of OD matrix is 0, which is not allowed'

            # Update Q matrix and dense q vector temporarily
            Nt.Q = noise_params['scale_Q']*Nt.Q
            Nt.q = networks.denseQ(Q=Nt.Q, remove_zeros=Nt.setup_options['remove_zeros_Q'])

        if len(sparsity_idxs) != 0:

            # idx = list(set(np.flatnonzero(Nt.Q)))
            # N = int(round(sparsity * Nt.Q.size))
            np.put(Nt.Q, sparsity_idxs, 0.1)
            Nt.q = networks.denseQ(Q=Nt.Q, remove_zeros=Nt.setup_options['remove_zeros_Q'])


        # Store new OD matrix
        Nt.Q_true = copy.deepcopy(Nt.Q)
        Nt.q_true = copy.deepcopy(Nt.q)



    if n_paths is not None:

        # Save previous paths and paths per od lists
        paths, paths_od = Nt.paths, Nt.paths_od
        M,D,C = Nt.M, Nt.D, Nt.C

        # Matrix with link utilities
        Nt.V = Nt.generate_V(A=Nt.A, links=Nt.links, theta=theta)

        # Key to have the minus sign so we look the route that lead to the lowest disutility
        edge_utilities = Nt.generate_edges_weights_dict_from_utility_matrix(V=Nt.V)

        # Generate new paths according to arbitrary size given by n_paths
        Nt.paths, Nt.paths_od = path_generation_nx(A=Nt.A
                                             , ods= Nt.ods
                                             , links=Nt.links_dict
                                             , cutoff=Nt.setup_options['cutoff_paths']
                                             , n_paths= n_paths
                                             , edge_weights=edge_utilities
                                             , silent_mode = True
                                             )

        printer.blockPrint()
        Nt.M = Nt.generate_M(paths_od=Nt.paths_od)
        Nt.D = Nt.generate_D(paths_od=Nt.paths_od, links=Nt.links)
        Nt.C = choice_set_matrix_from_M(Nt.M)
        printer.enablePrint()

    # Store noisy matrix (if noise was added)
    Nt.Q_true = copy.deepcopy(Nt.Q)
    Nt.q_true = copy.deepcopy(Nt.q)

    # eq_params
    # theta['tt'] = 0
    results_sue_msa = equilibrium.sue_logit_iterative(Nt=Nt, theta=theta, k_Y=k_Y, k_Z=k_Z, params=eq_params)

    # Store path flows
    Nt.path_flows = results_sue_msa['f']


    # Exogenous noise in the link count measurements

    # print(results_sue_msa['p_f'])
    true_link_counts = np.array(list(results_sue_msa['x'].values()))[:,np.newaxis]

    assert true_link_counts.shape[1] == 1 and len(true_link_counts.shape) == 2, "vector of true link counts is not a column vector"

    n = true_link_counts.shape[0]

    mean_x = np.nanmean(true_link_counts)

    if noise_params is not None and (noise_params['mu_x'] != 0 or noise_params['sd_x'] != 0 or 'snr_x' in noise_params):

        if 'snr_x' in noise_params and noise_params['snr_x'] is not None:

            # Now we use the signal noise ratio (tibshirani concept, https://arxiv.org/pdf/1707.08692.pdf)

            sd_x = np.sqrt(np.var(true_link_counts)/noise_params['snr_x'])

        else:
            sd_x = noise_params['sd_x'] * mean_x

        link_counts = true_link_counts + np.random.normal(noise_params['mu_x'], sd_x, n)[:, np.newaxis]

        # We truncate link counts so they are positive
        link_counts[link_counts < 0] = 0

    else:
        link_counts = true_link_counts

    assert link_counts.shape[1] == 1 and len(true_link_counts.shape) == 2, "vector of true link counts is not a column vector"


    # Generate a random subset of idxs depending on coverage
    missing_sample_size = n-int(n * coverage)
    idx = list(np.random.choice(np.arange(0, n), missing_sample_size, replace=False))

    # Only a subset of observations is assumed to be known
    train_link_counts = masked_observed_counts(xct=link_counts, idx=idx)

    # Link counts that are not within the covered links for simulation
    withdraw_link_counts = masked_observed_counts(xct=link_counts, idx=list(set(list(np.arange(0, n)))-set(idx)))

    # Revert Q matrix and q dense vector to original form
    if noise_params is not None and (noise_params['sd_Q'] != 0 or noise_params['scale_Q'] != 1):

        Nt.Q_noisy = Nt.Q
        Nt.q_noisy = Nt.q
        Nt.Q = Q_original
        Nt.q = networks.denseQ(Q=Q_original, remove_zeros=Nt.setup_options['remove_zeros_Q'])

    #Revert original paths and incidence matrices
    if n_paths is not None:
        Nt.paths, Nt.paths_od = paths, paths_od
        Nt.M, Nt.D, Nt.C = M,D,C


    if uncongested_mode:
        # Revert original bpr values
        for link, link_num in zip(Nt.links, range(len(Nt.links))):
            link.bpr.alpha = bpr_alpha[link_num]
            link.bpr.beta = bpr_beta[link_num]

    x_training = dict(zip(results_sue_msa['x'].keys(), train_link_counts.flatten()))
    x_withdraw = dict(zip(results_sue_msa['x'].keys(), withdraw_link_counts.flatten()))

    return x_training, x_withdraw

def generate_training_validation_samples(xct: dict, prop_validation , prop_training = None)-> (dict, dict):

    if prop_training is None:
        prop_training = 1-prop_validation

    x_bar = np.array(list(xct.values()))[:, np.newaxis]

    # Generate a random subset of idxs depending on coverage
    idx_nonas = np.where(~np.isnan(x_bar))[0]

    idx_training = list(np.random.choice(idx_nonas, int(np.floor(prop_training*len(idx_nonas))), replace=False))
    idx_validation = list(set(idx_nonas) - set(idx_training))

    # Only a subset of observations is assumed to be known
    train_link_counts = masked_observed_counts(xct=x_bar, idx = idx_training, complement = True)
    validation_link_counts = masked_observed_counts(xct=x_bar, idx=idx_validation, complement = True)

    # Convert to dictionary

    xct_training = dict(zip(xct.keys(), train_link_counts.flatten()))
    xct_validation = dict(zip(xct.keys(), validation_link_counts.flatten()))

    #Sizes deducting nan entries
    adjusted_size_training = np.count_nonzero(~np.isnan(train_link_counts))
    adjusted_size_validation = np.count_nonzero(~np.isnan(validation_link_counts))

    print('\nTraining and validation samples of sizes:', (adjusted_size_training, adjusted_size_validation))

    return xct_training, xct_validation


def generate_link_traveltimes_equilibrium(N, theta: np.array, k_Y: [], k_Z: [], eq_params: {}):
    N_copy = networks.clone_network(N=N, label='Clone', randomness={'Q': False, 'BPR': False, 'Z': False})[
        N.key]

    results_sue_msa = equilibrium.sue_logit_iterative(Nt=N_copy, theta=theta, k_Y=k_Y, k_Z=k_Z, params=eq_params)

    return results_sue_msa['tt_x']


def jacobian_response_function(theta, YZ_x, q, Ix, Iq, C, p_f: ColumnVector = None, paths_batch_size: int = 0, x_bar: ColumnVector = None, normalization = False):
    '''

        :param theta:
        :param YZ_x: data at link level
        :param q: dense vector of OD demand

        :return:
        '''

    # idx_links = np.arange(0, n)

    # YZ_x = YZ_x[idx_links,:]
    # Ix = Ix[idx_links,:]

    # theta = np.array(list(theta0.values()))[:, np.newaxis]

    # Link and path utilities
    # v_x = YZ_x.dot(theta)
    # v_f = Ix.T.dot(v_x)
    # v_f = v_normalization(v_f[:,np.newaxis],C)[:,np.newaxis]
    # np.mean(v_f)

    # Path probabilities (TODO: speed up this operation by avoiding elementwise division)
    # p_f = np.divide(np.exp(v_f), C.dot(np.exp(v_f)))

    paths_idxs = []
    path_reduction = False

    if x_bar is not None:

        # The path reduction will not change results but it is worth to do in cases with low coverage
        if path_reduction:
            # idx_links_nas = np.where(np.isnan(x_bar))[0]
            idx_links_nonas = np.where(~np.isnan(x_bar))[0]

            # Identify indices where paths traverse some link with traffic counts.
            paths_idxs = list(np.where(np.sum(Ix[idx_links_nonas, :], axis=0) == 1)[0])

            print('Path reduction found ' + str(len(paths_idxs)) + ' seemingly irrelevant paths')

        # Subsampling of paths

    if paths_batch_size > 0:
        paths_idxs = list(np.random.choice(paths_idxs, paths_batch_size, replace=False))

    if len(paths_idxs) == 0:

        Ix_sample = Ix
        Iq_sample = Iq
        C_sample = C

    else:
        Ix_sample = Ix[:, paths_idxs]
        Iq_sample = Iq[:, paths_idxs]
        C_sample = C[paths_idxs, paths_idxs]

    # Path probabilities (TODO: I may speed up this operation by avoiding elementwise division)

    if p_f is None:
        p_f = path_probabilities(theta, YZ_x, Ix_sample, C_sample, None, normalization)

    if len(paths_idxs) > 0:
        p_f_sample = p_f[paths_idxs]
    else:
        p_f_sample = p_f

    # TODO: perform the gradient operation for each attribute using autograd
    J = []

    # Jacobian/gradient of response function

    grad_m_terms = {}

    grad_m_terms[0] = Iq_sample.T.dot(q)
    grad_m_terms[1] = C_sample  # computing Iq.T.dot(Iq) is too slow
    grad_m_terms[2] = p_f_sample.dot(p_f_sample.T)

    # This operation is performed for each attribute k
    for k in np.arange(theta.shape[0]):  # np.arange(len([*k_Y,*k_Z])):

        # Attributes vector at link and path level
        Zk_x = YZ_x[:, k][:, np.newaxis]
        Zk_f = Ix_sample.T.dot(Zk_x)

        grad_m_terms[3] = (np.ones(Zk_f.shape).dot(Zk_f.T) - Zk_f.dot(np.ones(Zk_f.shape).T))

        # grad_m = Ix.dot(np.multiply(grad_m_terms[0],
        #                             np.multiply(grad_m_terms[2], grad_m_terms[3]))).dot(
        #     np.ones(Zk_f.shape))
        grad_m = Ix_sample.dot(np.multiply(grad_m_terms[0],np.multiply(grad_m_terms[1], np.multiply(grad_m_terms[2], grad_m_terms[3])))).dot(np.ones(Zk_f.shape))

        # Gradient of objective function

        if k == 0:
            J = grad_m

        if k > 0:
            J = np.column_stack((J, grad_m))


    # if len(paths_idxs) > 0:
    if x_bar is not None:
        return J, p_f_sample, Ix_sample, Iq_sample
    else:
        return J, p_f_sample


def link_utilities(theta, YZ_x):
    # Linkutilities
    v_x = np.dot(YZ_x,theta)

    return v_x


def path_utilities(theta, YZ_x, Ix, C, normalization):

    #link utilities
    v_x = link_utilities(theta, YZ_x)

    #path utilities
    v_f = np.dot(Ix.T,v_x)

    if normalization is True:
    # softmax trick (TODO: this operation is computationally expensive)
        v_f = v_normalization(v_f, C)

    assert v_f.shape[1] == 1, 'vector of link utilities is not a column vector'

    return v_f


def path_utilities_by_attribute(theta, YZ_x, Ix, C,normalization):

    #If the attribute type is absolute, the computation of link utilities can be done efficiently,

    attr_types = None

    counter_idx = 0

    absolute_attr_type_idx = []

    for attr, type in attr_types.items():

        if attr == 'absolute':
            absolute_attr_type_idx.append(counter_idx)

        counter_idx += 1

    # link utilities
    v_x = link_utilities(theta, YZ_x)

    # path utilities
    v_f = Ix.T.dot(v_x)

    if normalization is True:
        # softmax trick (TODO: this operation is computationally expensive)
        v_f = v_normalization(v_f.reshape(v_f.shape[0]), C)[:, np.newaxis]

    return v_f


def path_probabilities(theta, YZ_x, Ix, C, attr_types = None, normalization = True):

    # TODO: the effect of incidents seems to be additive so normalizing by mean will not necessarily help

    epsilon = 1e-12

    if attr_types is None:

        v_f = path_utilities(theta, YZ_x, Ix, C, normalization)

    else:
        v_f = path_utilities_by_attribute(theta, YZ_x, Ix, C, normalization)

    # Path probabilities (TODO: speed up this operation by avoiding elementwise division)
    p_f = np.divide(np.exp(v_f), np.dot(C,np.exp(v_f))+epsilon)

    return p_f


def response_function(Ix, Iq, q, p_f):
    # Response function
    m = np.dot(Ix,np.multiply(Iq.T.dot(q), p_f))

    return m

def error_by_link(x_bar: ColumnVector, x_eq: ColumnVector, show_nan = True):
    """ Difference between observed counts and counts computed at equilibrium.

    """

    # assert x_bar.shape[0] == x_eq.shape[0], ' shape of vectors is different'
    assert x_bar.shape[1] == 1 and x_eq.shape[1] == 1, ' no column vectors'

    # Store the number of elements different than nan
    adjusted_n = np.count_nonzero(~np.isnan(x_bar))
    #
    # x_bar = fake_observed_counts(xct_hat = x_eq, xct= x_bar)

    # list(np.sort(-((x_eq - x_bar) ** 2).flatten())*-1)

    if show_nan:
        return (x_eq - x_bar)

    else:

        idx_nonas = np.where(~np.isnan(x_bar))[0]

        return (x_eq[idx_nonas] - x_bar[idx_nonas])


def loss_function_by_link(x_bar: ColumnVector, x_eq: ColumnVector):
    """ Difference between observed counts and counts computed at equilibrium.

    """

    # assert x_bar.shape[0] == x_eq.shape[0], ' shape of vectors is different'
    assert x_bar.shape[1] == 1 and x_eq.shape[1] == 1, ' no column vectors'

    # Store the number of elements different than nan
    adjusted_n = np.count_nonzero(~np.isnan(x_bar))
    #
    # x_bar = fake_observed_counts(xct_hat = x_eq, xct= x_bar)

    # list(np.sort(-((x_eq - x_bar) ** 2).flatten())*-1)

    return (x_eq - x_bar) ** 2/adjusted_n


def loss_function(x_bar: ColumnVector, x_eq: ColumnVector):
    """ Difference between observed counts and counts computed at equilibrium
        It takes into account that there may not full link coverage.

    """

    # assert x_bar.shape[0] == x_eq.shape[0], ' shape of vectors is different'
    assert x_bar.shape[1] == 1 and x_eq.shape[1] == 1, ' no column vectors'

    # Store the number of elements different than nan
    adjusted_n = np.count_nonzero(~np.isnan(x_bar))

    x_bar = fake_observed_counts(xct_hat = x_eq, xct= x_bar)

    # list(np.sort(-((x_eq - x_bar) ** 2).flatten())*-1)

    return np.sum((x_eq - x_bar) ** 2)/adjusted_n

def lasso_soft_thresholding_operator(lambda_hp, theta_estimate):

    # TODO: theta needs to be refitted each time, the soft-thresholding only
    #  gives us a direction to perform gradient updates. A coordinate descent is used for
    # multiple predictors

    regularized_theta = copy.deepcopy(theta_estimate)

    for attr, theta_val in theta_estimate.items():

        if theta_val > lambda_hp:
            regularized_theta[attr] = theta_val - lambda_hp

        if theta_val < -lambda_hp:
            regularized_theta[attr] = theta_val + lambda_hp


        if abs(theta_val) <= lambda_hp:
            regularized_theta[attr] = 0

    return regularized_theta


def lasso_regularization(Nt, grid_lambda, theta_estimate, k_Y, k_Z: [],eq_params: Options, x_bar: ColumnVector, standardization: dict):

    print('\nPerforming Lasso regularization with lambda grid:', grid_lambda)

    # Write soft-thresholding operator

    # ref: https://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-15.pdf

    regularized_thetas = {}
    losses = {}

    for lambda_hp in grid_lambda:

        regularized_thetas[lambda_hp] = lasso_soft_thresholding_operator(lambda_hp,theta_estimate)

        # Run stochastic user equilibrium
        results_eq = equilibrium.sue_logit_iterative(Nt=Nt, theta=regularized_thetas[lambda_hp]
                                                     , k_Y=k_Y, k_Z=k_Z, params=eq_params
                                                     , silent_mode = True, standardization = standardization)

        x_eq = np.array(list(results_eq['x'].values()))[:, np.newaxis]

        losses[lambda_hp] = loss_function(x_bar = x_bar, x_eq = x_eq)

        # print('theta:', regularized_thetas[lambda_hp] )
        print('\nlambda: ', "{0:.3}".format(float(lambda_hp)))
        print('current theta: ', str({key: round(val, 3) for key, val in regularized_thetas[lambda_hp].items()}))
        print('loss:', losses[lambda_hp])

    return regularized_thetas, losses[lambda_hp]

def monotonocity_traffic_count_functions(Nt, k_Y, k_Z, x_bar, attr_label: str, theta: LogitParameters, theta_attr_grid: np.ndarray, inneropt_params = {'iters': 1, 'accuracy_eq': 0.01}, paths_batch_size = 0):

    """ Analyze the monotonicity of the traffic counts functions and it is analyzed if the range of the function includes the traffic counts measurements (vertical var)"""

    theta_current = {}
    for key in k_Y + k_Z:
        theta_current[key] = theta[key]

    x_eq_vals = []
    x_ids = []
    thetas_list = []
    for iter, theta_attr_val in zip(range(len(theta_attr_grid)), theta_attr_grid):

        printer.printProgressBar(iter, len(theta_attr_grid)-1, prefix='Progress:', suffix='',length=20)

        theta_current[attr_label] = theta_attr_val

        results_eq = equilibrium.sue_logit_iterative(Nt=Nt, theta= theta_current, k_Y=k_Y, k_Z=k_Z, params= inneropt_params, silent_mode = True, n_paths_column_generation= 0)

        # x_eq = np.array(list(results_eq['x'].values()))[:,np.newaxis]
        x_eq = list(results_eq['x'].values())

        thetas_list.extend([theta_attr_val]*len(x_eq))

        # Add 1 to ids so they match the visualization

        x_ids_list = []
        for key in list(results_eq['x'].keys()):
            x_ids_list.append((key[0]+1, key[1]+1, key[2]))

        x_ids.extend(x_ids_list)
        x_eq_vals.extend(x_eq)

    # # Create dictionary of values by link
    # n_links = len(x_eq_vals[0])
    # traffic_count_links_dict = {}
    # for theta_grid in range(len(x_eq_vals)):
    #     for link_id in range(len(x_eq_vals[theta_grid])):
    #         traffic_count_links_dict[link_id] = x_eq_vals[theta_grid][link_id]

    # Create pandas dataframe
    traffic_count_links_df = pd.DataFrame({'link':x_ids, 'theta': thetas_list,'count': x_eq_vals})

    return traffic_count_links_df

def grid_search_optimization(Nt, k_Y, k_Z, x_bar, q0, attr_label: str, theta: LogitParameters, theta_attr_grid: np.ndarray, gradients: bool = False, hessians: bool = False, inneropt_params = {'iters': 1, 'accuracy_eq': 0.01}, paths_batch_size = 0):

    """ Perform grid search optimization by computing equilibrium for a grid of values of the logit parameter associated to one of the attributes"""


    print('\nPerforming grid search for ' + str(attr_label) + '\n')

    loss_function_vals = []
    grad_vals = []
    hessian_vals = []

    q = q0

    theta_current = {}
    for key in k_Y+k_Z:
        theta_current[key] = theta[key]

    printer.enablePrint()

    for iter, theta_attr_val in zip(range(len(theta_attr_grid)), theta_attr_grid):

        printer.printProgressBar(iter, len(theta_attr_grid), prefix='Progress:', suffix='',length=20)

        theta_current[attr_label] = theta_attr_val

        # inneropt_params['iters'] = 100

        results_eq = equilibrium.sue_logit_iterative(Nt=Nt, theta= theta_current, k_Y=k_Y, k_Z=k_Z, q = q0, params= inneropt_params, silent_mode = True, n_paths_column_generation= 0)

        # x_eq = np.array(list(results_eq_initial['x'].values()))
        x_eq = np.array(list(results_eq['x'].values()))[:, np.newaxis]

        p_f = results_eq['p_f']

        loss_function_vals.append(loss_function(x_bar, x_eq))

        print('current loss: ', loss_function_vals[-1])
        # print('current theta: ', str("{0:.0E}".format(theta_attr_val)))
        print('current theta: ', "{0:.3}".format(float(theta_attr_val)))

        if gradients:
            # YZ_x = get_design_matrix(Y={'tt': results_eq['tt_x']}, Z= Nt.Z_dict, k_Y=k_Y, k_Z=k_Z)

            YZ_x = get_design_matrix(Y={'tt': results_eq['tt_x']}, Z= Nt.Z_dict, k_Y=k_Y, k_Z=k_Z)

            theta_current_vector = np.array(list(theta_current.values()))[:, np.newaxis]

            idx_attr = list(theta_current.keys()).index(attr_label)
            grad_vals.append(
                gradient_objective_function(
                    attribute_k =  idx_attr
                    , theta = theta_current_vector, YZ_x = YZ_x, x_bar = x_bar, q = q
                    , Ix = Nt.D, Iq = Nt.M, C = Nt.C, p_f = p_f, paths_batch_size = paths_batch_size)[0]
            )

        else:
            grad_vals.append(0)


        if hessians:

            theta_current_vector = np.array(list(theta_current.values()))[:, np.newaxis]

            # YZ_x = get_design_matrix(Y={'tt': results_eq['tt_x']}, Z=Nt.Z_dict, k_Y=k_Y, k_Z=k_Z)
            YZ_x = get_design_matrix(Y={'tt': results_eq['tt_x']}, Z= Nt.Z_dict, k_Y=k_Y, k_Z=k_Z)

            hessian_vals.append(
                hessian_objective_function(theta=theta_current_vector, x_bar= x_bar, YZ_x = YZ_x, q = q, Ix = Nt.D, Iq = Nt.M, C = Nt.C, p_f=  p_f, approximation = False)[list(theta_current.keys()).index(attr_label)]
            )

        else:
            hessian_vals.append(0)

    return theta_attr_grid, loss_function_vals, grad_vals, hessian_vals

def random_search_optimization(Nt, k_Y, k_Z, x_bar, n_draws: int, theta_bounds: dict, inneropt_params: dict, q_bounds: tuple = None, silent_mode = False, uncongested_mode = True):

    """ Perform grid search optimization by computing equilibrium for a grid of values of the logit parameter associated to one of the attributes"""


    print('\nPerforming random search with ' + str(n_draws) + ' draws\n')

    loss_function_vals = []

    thetas = []
    q_scales = []

    for draw in range(n_draws):

        printer.printProgressBar(draw, n_draws, prefix='Progress:', suffix='',length=20)

        # Get a random theta vector according to the dictionary of bounds tuples
        theta_current = {key:0 for key, val in theta_bounds.items()}

        for attribute, bounds in theta_bounds.items():
            theta_current[attribute] = float(np.random.uniform(*bounds,1))


        if q_bounds is not None:
            q_scale = float(np.random.uniform(*q_bounds,1))

            if silent_mode is True:
                printer.blockPrint()

            loss_dict = scale_Q(x_bar, Nt, k_Y, k_Z, theta_0 = theta_current, scale_grid = [q_scale], n_paths = None, silent_mode = True, uncongested_mode = uncongested_mode, inneropt_params= inneropt_params)

            q_scales.append({'q_scale': q_scale})

            loss_function_vals.append(loss_dict[q_scale])

            if silent_mode is False:
                print('current q scale: ', str("{0:.1E}".format(q_scale)))

        else:

            #Do not generate new paths via column generation to save computation
            results_eq = equilibrium.sue_logit_iterative(Nt=Nt, theta= theta_current, k_Y=k_Y, k_Z=k_Z, params= inneropt_params, silent_mode = True, n_paths_column_generation = 0)

            # x_eq = np.array(list(results_eq_initial['x'].values()))
            x_eq = np.array(list(results_eq['x'].values()))[:, np.newaxis]

            loss_function_vals.append(loss_function(x_bar, x_eq))

        if silent_mode is False:
            # print('current theta: ',str({key: "{0:.1E}".format(val) for key, val in theta_current.items()}))
            print('current theta: ', str({key: round(val, 3) for key, val in theta_current.items()}))

            print('current loss: ', '{:,}'.format(loss_function_vals[-1]), '\n')

        thetas.append(theta_current)

    if silent_mode is True:
        printer.enablePrint()

    return thetas, q_scales, loss_function_vals

@blockPrinting
def loss_predicted_counts_congested_network(x_bar: ColumnVector, Nt: TNetwork, k_Y: [], k_Z: [], theta_0, params, ):

    """ Compute the l2 norm with naive prediction assuming a congested network"""

    # Perform uncongested traffic assignment assuming all preference parameters to be equal to 0
    theta_current = theta_0 #dict.fromkeys(k_Y+k_Z,0)

    results_congested_eq = equilibrium.sue_logit_iterative(Nt=Nt, theta= theta_current, k_Y=k_Y, k_Z=k_Z, params = params)

    # x_eq = np.array(list(results_eq_initial['x'].values()))
    x_eq = np.array(list(results_congested_eq['x'].values()))[:, np.newaxis]

    # Compute l2 norm
    return loss_function(x_bar=x_bar, x_eq=x_eq)


# @blockPrinting
def loss_predicted_counts_uncongested_network(x_bar: ColumnVector, Nt: TNetwork, k_Y: [], k_Z: [], theta_0):

    """ Compute the l2 norm with naive prediction assuming an uncongested network"""

    # To account for the uncongested case, the BPR parameters are set to be equal to zero

    # To not alter the link function, first copy the current bpr parameters and then set them to 0
    bpr_alpha = {}
    bpr_beta = {}

    for link, link_num in zip(Nt.links,range(len(Nt.links))):
        bpr_alpha[link_num]  = link.bpr.alpha
        bpr_beta[link_num] = link.bpr.beta

        link.bpr.alpha = 0
        link.bpr.beta = 0

    # Perform uncongested traffic assignment assuming all preference parameters to be equal to 0
    theta_current = theta_0 #dict.fromkeys(k_Y+k_Z,0)

    results_uncongested_eq = equilibrium.sue_logit_iterative(Nt=Nt, theta= theta_current, k_Y=k_Y, k_Z=k_Z, params = {'iters': 0, 'accuracy_eq': 0.01})

    # x_eq = np.array(list(results_eq_initial['x'].values()))
    x_eq = np.array(list(results_uncongested_eq['x'].values()))[:, np.newaxis]

    # Revert original bpr values
    for link, link_num in zip(Nt.links, range(len(Nt.links))):
        link.bpr.alpha = bpr_alpha[link_num]
        link.bpr.beta = bpr_beta[link_num]

    #Compute l2 norm
    return loss_function(x_bar = x_bar, x_eq = x_eq), x_eq

def scale_Q(x_bar: ColumnVector, Nt: TNetwork, k_Y: [], k_Z: [], theta_0, scale_grid, n_paths, silent_mode = False, uncongested_mode = True, inneropt_params: Options = None):

    """ Compute the l2 norm with naive prediction assuming an uncongested network"""


    print("\nScaling Q matrix with grid: ", str(["{0:.2}".format(val) for val in scale_grid]), '\n')

    # Generate new paths with an arbitrary size

    if n_paths is not None:

        # Save previous paths and paths per od lists
        paths, paths_od = Nt.paths, Nt.paths_od
        M,D,C = Nt.M, Nt.D, Nt.C

        # Matrix with link utilities
        Nt.V = Nt.generate_V(A=Nt.A, links=Nt.links, theta=theta_0)

        # Key to have the minus sign so we look the route that lead to the lowest disutility
        edge_utilities = Nt.generate_edges_weights_dict_from_utility_matrix(V=Nt.V)

        # Generate new paths according to arbitrary size given by n_paths
        Nt.paths, Nt.paths_od = path_generation_nx(A=Nt.A
                                             , ods= Nt.ods
                                             , links=Nt.links_dict
                                             , cutoff=Nt.setup_options['cutoff_paths']
                                             , n_paths= n_paths
                                             , edge_weights=edge_utilities
                                             , silent_mode = True
                                             )

        printer.blockPrint()
        Nt.M = Nt.generate_M(paths_od=Nt.paths_od)
        Nt.D = Nt.generate_D(paths_od=Nt.paths_od, links=Nt.links)
        Nt.C = choice_set_matrix_from_M(Nt.M)
        printer.enablePrint()

    # To account for the uncongested case, the BPR parameters are set to be equal to zero

    # To not alter the link function, first copy the current bpr parameters and then set them to 0

    # Noise or scale difference in Q matrix
    Q_original = Nt.Q.copy()

    loss_scale_dict = {}

    for scale_factor, iter in zip(scale_grid,range(len(scale_grid))):

        printer.printProgressBar(iter, len(scale_grid), prefix='Progress:', suffix='',
                                 length=20)

        assert scale_factor > 0, 'scale factor cannot be 0'

        Q_scaled = scale_factor*Q_original

        # Update Q matrix and dense q vector temporarily
        Nt.Q = Q_scaled
        Nt.q = networks.denseQ(Q=Nt.Q, remove_zeros=Nt.setup_options)

        if uncongested_mode is True:
            loss_after_scale, _ = loss_predicted_counts_uncongested_network(x_bar=x_bar, Nt=Nt, k_Y=k_Y, k_Z=k_Z
                                                                         , theta_0 = theta_0)

        else:
            loss_after_scale = loss_predicted_counts_congested_network(x_bar=x_bar, Nt=Nt, k_Y=k_Y, k_Z=k_Z,
                                                                       theta_0=theta_0, params=inneropt_params)


        loss_scale_dict[scale_factor] = loss_after_scale

        if silent_mode is False:
            print('current scale', scale_factor)
            print('current loss', loss_after_scale, '\n')

    # Revert Q matrix and q dense vector to original form
    Nt.Q = Q_original
    Nt.q = networks.denseQ(Q=Q_original, remove_zeros=Nt.setup_options['remove_zeros_Q'])

    #Revert original paths and incidence matrices
    if n_paths is not None:
        Nt.paths, Nt.paths_od = paths, paths_od
        Nt.M, Nt.D, Nt.C = M,D,C

    if silent_mode:
        printer.enablePrint()


    return loss_scale_dict

def mean_count_l2norm(x_bar, mean_x = None):

    """
    Benchmark prediction with naive model that predicts the mean value count. If a mean value is provided, then the mean of the training sample is not computed
    """

    if mean_x is None:
        mean_x = np.nanmean(x_bar)
        x_benchmark = mean_x*np.ones(x_bar.shape)

    else:
        x_benchmark = mean_x * np.ones(x_bar.shape)

    return loss_function(x_bar = x_bar, x_eq = x_benchmark), mean_x


def objective_function(theta, YZ_x, x_bar, q, Ix, Iq, C, p_f: ColumnVector = None,  normalization = True):
    '''
    SSE: Sum of squared errors


    :param theta:
    :param YZ_x:
    :param x_bar:
    :param q:
    :param Ix:
    :param Iq:
    :param C:
    :return:
    '''

    # Path probabilities (TODO: speed up this operation by avoiding elementwise division)

    if p_f is None:
        p_f = path_probabilities(theta, YZ_x, Ix, C, None, normalization)

    # Response function
    m = response_function(Ix, Iq, q, p_f)

    # Objective function

    # Account for vector positions with NA values

    x_bar_copy = fake_observed_counts(xct_hat = m, xct= x_bar)

    s = (m - x_bar_copy) ** 2

    return s

def objective_function_numeric_jacobian(theta, *args):

    """ Wrapper function to compute the Jacobian numerically.
    """

    # return np.sum(objective_function(np.array(theta)[:, np.newaxis], YZ_x = args[0], x_bar= args[1], q= args[2], Ix= args[3], Iq = args[4], C = args[5]))

    return objective_function(np.array(theta)[:, np.newaxis], YZ_x = args[0], x_bar= args[1], q= args[2], Ix= args[3], Iq = args[4], C = args[5], normalization  = args[6])

def objective_function_numeric_hessian(theta, *args):

    """ Wrapper function to compute the Hessian numerically. This type of computation is unreliable and unstable. The hessian is not guaranteed to be PSD which generates issues with its inversion to obtain the covariance matrix
    """

    # return np.sum(objective_function(np.array(theta)[:, np.newaxis], YZ_x = args[0], x_bar= args[1], q= args[2], Ix= args[3], Iq = args[4], C = args[5]))

    return np.sum(objective_function(np.array(theta)[:, np.newaxis], YZ_x = args[0], x_bar= args[1], q= args[2], Ix= args[3], Iq = args[4], C = args[5], normalization  = args[6]))



def sse(theta, YZ_x, x_bar, q, Ix, Iq, C):
    # YZ_x.shape
    # theta.shape
    # x_bar.shape
    return objective_function(theta, YZ_x, x_bar, q, Ix, Iq, C)


def ttest_theta(theta_h0, theta: {}, YZ_x, xc, q, Ix, Iq, C, pct_lowest_sse = 100, alpha=0.05, silent_mode: bool = False, numeric_hessian = True):
    # TODO: take advantage of kwargs argument to simplify function signature. A network object should be required only

    t0 = time.time()
    print('\nPerforming hypothesis testing (H0: theta = ' + str(theta_h0) + ')')

    theta_array = np.array(list(theta.values()))[:, np.newaxis]

    p = theta_array.shape[0]
    n = np.count_nonzero(~np.isnan(xc))
    # q = q[:, np.newaxis]


    # SSE per observation
    sses = sse(theta_array, YZ_x, xc, q, Ix, Iq, C)
    top_sse = sses

    # pct_lowest_sse = 100

    if pct_lowest_sse < 100:

        idxs_nonan = np.where(~np.isnan(xc))[0]

        n_top_obs = len(sses.flatten()[idxs_nonan])
        top_n = int(n_top_obs*pct_lowest_sse/100)
        top_sse = np.sort(sses.flatten()[idxs_nonan])[0:top_n]

        n = top_n

    epsilon = 1e-7

    sum_sse = np.sum(top_sse) + epsilon

    var_error = sum_sse / (n - p)

    if numeric_hessian is True:
        # objective_function(theta_array, YZ_x, xc, q, Ix, Iq, C)

        print('Hessian is computed numerically')

        H = nda.Hessian(objective_function_numeric_hessian)(list(theta_array.flatten()), YZ_x, xc, q, Ix, Iq, C)

    else:
        print('Hessian is approximated as the Jacobian by its transpose')

        # # Unidimensional inverse function is just the reciprocal but this is multidimensional so inverse is required
        F, pf = jacobian_response_function(theta_array, YZ_x, q, Ix, Iq, C, paths_batch_size=0)

        # #Robust approximation of covariance matrix (almost no difference with previous methods)
        # cov_theta = np.linalg.lstsq(var_error*F.T.dot(F), np.eye(F.shape[1]), rcond=None)[0]
        H = F.T.dot(F)

    cov_theta = np.linalg.pinv(H)

    # Read Nonlinear regression from gallant (1979)
    # T value
    # alpha = 0.98
    critical_tvalue = stats.t.ppf(1 - alpha / 2, df=n - p)
    ttest = (theta_array - theta_h0) / np.sqrt(var_error*np.diag(cov_theta)[:, np.newaxis])
    # width_int = two_tailed_ttest*np.sqrt(cov_theta)
    # pvalue =  (1 - stats.t.cdf(ttest,df=n-p))
    # https://stackoverflow.com/questions/23879049/finding-two-tailed-p-value-from-t-distribution-and-degrees-of-freedom-in-python
    # pvalue = (1 - stats.t.cdf(ttest,df=n-p))#
    pvalue = 2 * stats.t.sf(np.abs(ttest), df=n - p)  # * 2

    # stats.t.sf(np.round(ttest,2), df=n - p)

    # pvalue = 2 * stats.t.sf(1.9, df=n - p)  # * 2
    # return "[" + str(theta - width_int) + "," + str(theta + width_int) + "]"

    if not silent_mode:
        print('Point estimate: ' + str({key: "{0:.0E}".format(val) for key, val in theta.items()}))
        print('T-tests: ' + str(ttest.flatten()))
        print('P-values: ' + str(pvalue.flatten()))

        print('Sample size:', n)
        print(str(round(pct_lowest_sse) ) + '% of the total observations with lowest SSE were used')

    print('Time: ' + str(np.round(time.time() - t0, 1)) + '[s]')

    return ttest, critical_tvalue, pvalue

def confint_theta(theta: {}, YZ_x, xc, q, Ix, Iq, C, alpha=0.05,pct_lowest_sse = 100, silent_mode: bool = False, numeric_hessian = True):

    t0 = time.time()

    print('\nComputing confidence intervals (alpha = ' + str(alpha) + ')')

    # i = 'N6'
    #
    #
    # YZ_x = tai.estimation.get_design_matrix(Y=N['train'][i].Y_dict, Z=N['train'][i].Z_dict, k_Y=k_Y, k_Z=k_Z)
    # theta = np.array(list({k: 1 * theta_true[i][k] for k in [*k_Y, *k_Z]}.values()))[:, np.newaxis]
    # x_bar = N['train'][i].x[:, np.newaxis]
    # q = tai.network.denseQ(Q=N['train'][i].Q, remove_zeros=remove_zeros_Q)
    # Ix = N['train'][i].D
    # Iq = N['train'][i].M
    # C = tai.estimation.choice_set_matrix_from_M(N['train'][i].M)

    theta_array = np.array(list(theta.values()))[:, np.newaxis]

    p = theta_array.shape[0]
    n = np.count_nonzero(~np.isnan(xc))    #xc.shape[0]
    q = q[:, np.newaxis]

    # SSE per observation
    sses = sse(theta_array, YZ_x, xc, q, Ix, Iq, C)
    top_sse = sses

    # pct_lowest_sse = 100

    if pct_lowest_sse < 100:
        idxs_nonan = np.where(~np.isnan(xc))[0]

        n_top_obs = len(sses.flatten()[idxs_nonan])
        top_n = int(n_top_obs * pct_lowest_sse / 100)
        top_sse = np.sort(sses.flatten()[idxs_nonan])[0:top_n]

        n = top_n

    epsilon = 1e-7

    sum_sse = np.sum(top_sse) + epsilon

    var_error = sum_sse / (n - p)

    if numeric_hessian is True:
        # objective_function(theta_array, YZ_x, xc, q, Ix, Iq, C)

        print('Hessian is computed numerically')

        H = nd.Hessian(objective_function_numeric_hessian)(list(theta_array.flatten()), YZ_x, xc, q, Ix, Iq, C)

    else:
        print('Hessian is approximated as the Jacobian by its transpose')

        # # Unidimensional inverse function is just the reciprocal but this is multidimensional so inverse is required
        F, pf = jacobian_response_function(theta_array, YZ_x, q, Ix, Iq, C, paths_batch_size=0)

        # #Robust approximation of covariance matrix (almost no difference with previous methods)
        # cov_theta = np.linalg.lstsq(var_error*F.T.dot(F), np.eye(F.shape[1]), rcond=None)[0]
        H = F.T.dot(F)

    cov_theta = np.linalg.pinv(H)

    # T value (two-tailed)
    critical_tvalue = stats.t.ppf(1 - alpha / 2, df=n - p)

    width_confint = critical_tvalue * np.sqrt(np.diag(cov_theta)[:, np.newaxis])

    confint_list = ["[" + str(round(float(i - j), 4)) + ", " + str(round(float(i + j), 4)) + "]" for i, j in
                   zip(theta_array, width_confint)]

    # confint_list = ["[" + "{0:.1E}".format(i - j) + ", " + "{0:.1E}".format(i + j) + "]" for i, j in
    #                zip(theta_np, width_confint)]

    if not silent_mode:

        print('Point estimate: ' + str({key: "{0:.0E}".format(val) for key, val in theta.items()}))

        # print('Confidence intervals: ' + str({key: confint_i for key, confint_i in zip(theta.keys(), confint_list)}))

        print('Confidence intervals :' + str(confint_list))
        print('Time: ' + str(np.round(time.time() - t0, 1)) + '[s]')

    return confint_list, width_confint

def ftest(theta_m1: dict, theta_m2: dict, YZ_x, xc, q, Ix, Iq, C, alpha=0.05, pct_lowest_sse = 100, silent_mode: bool = False):

    print('\nComputing F-test')

    t0 = time.time()

    # model 1 is 'nested' within model 2

    n = np.count_nonzero(~np.isnan(xc))

    theta_m1_array = np.array(list(theta_m1.values()))[:, np.newaxis]
    theta_m2_array = np.array(list(theta_m2.values()))[:, np.newaxis]

    # SSE for each observation of the first model
    sses_1 = sse(theta_m1_array, YZ_x, xc, q, Ix, Iq, C)
    top_sse_1 = sses_1

    # SSE  for each observation of the second model (full model)
    sses_2 = sse(theta_m2_array, YZ_x, xc, q, Ix, Iq, C)
    top_sse_2 = sses_2

    if pct_lowest_sse < 100:
        idxs_nonan = np.where(~np.isnan(xc))[0]

        n_top_obs = len(sses_2.flatten()[idxs_nonan])
        top_n = int(n_top_obs * pct_lowest_sse / 100)
        n = top_n

        # Subset of SSE models 1 and 2
        top_sse_1 = np.sort(sses_1.flatten()[idxs_nonan])[0:top_n]
        top_sse_2 = np.sort(sses_2.flatten()[idxs_nonan])[0:top_n]

    # # SSE per observation
    # sses = sse(theta_array, YZ_x, xc, q, Ix, Iq, C)
    # top_sse = sses


    #Source: https://en.wikipedia.org/wiki/F-test

    p_1 = 0#theta_m1_array.shape[0]
    p_2 = theta_m2_array.shape[0]

    numerator_ftest = (np.sum(top_sse_1)-np.sum(top_sse_2))/(p_2-p_1)
    denominator_ftest = np.sum(top_sse_2)/(n-p_2)

    # source: https://stackoverflow.com/questions/39813470/f-test-with-python-finding-the-critical-value

    critical_fvalue = stats.f.ppf(1-alpha, dfn = p_2-p_1,dfd = n-p_2)

    ftest_value = numerator_ftest/denominator_ftest

    pvalue = stats.f.sf(np.abs(ftest_value), dfn = p_2-p_1,dfd = n-p_2)  # * 2

    if not silent_mode:
        # print('Point estimate: ' + str({key: "{0:.0E}".format(val) for key, val in theta.items()}))
        print('F-test: ' + str(round(ftest_value,4)))
        print('P-value: ' + str(round(pvalue,4)))
        print('Critical f-value: ' + str(round(critical_fvalue, 4)))
        print('Sample size:', n)
        print(str(round(pct_lowest_sse)) + '% of the total observations with lowest SSE were used')

        print('Time: ' + str(np.round(time.time() - t0, 1)) + '[s]')

    return ftest_value, critical_fvalue, pvalue

def hypothesis_tests(theta_h0, theta: {}, YZ_x, xc, q, Ix, Iq, C, pct_lowest_sse = 100, alpha=0.05, x_eq = None, normalization = False, numeric_hessian = False):
    print('\nPerforming hypothesis testing (H0: theta = ' + str(theta_h0) + ')')

    t0 = time.time()

    theta_array = np.array(list(theta.values()))[:, np.newaxis]

    p = theta_array.shape[0]
    n = np.count_nonzero(~np.isnan(xc))

    assert q.shape[1] == 1, 'q is not a column vector'
    # q = q[:, np.newaxis]

    # SSE per observation
    if x_eq is not None:
        # sses = (x_eq-xc)**2
        sses = sse(theta_array, YZ_x, xc, q, Ix, Iq, C)

    else:

        # TODO: Review why the sses calculated from the travel times at equilibrium and the best theta does not return a link count (response function) consistent with the link counts obtained at equilibrium (x_eq). There may be an error with the column of travel times in YZ_x. In fact, for the uncongested case, there are no difference between calculation. If x_eq is not used, the inference is worsen for the congested case.

        #TODO: The intuition for why this happens is because with the best theta obtained from the outer optimization we run the inner optimization to obtain the new equilibrium solution. The output of applying the best theta in the response function will not be necessarily at equilibrium in the congested case, and that is why the value of the objective function change. This would not occur if theta were a local minima as it usually happens in the uncongested case.

        #TODO: All in all, I would use not the output of the equilibrium solution since otherwise it generates more false negatives whereas it decrease the amounf of false positives. When selecting the solution of the outer optimization, we are assuming that the travel times are at equilibria and that the system will be at equilibria. This is fulfilled in the uncongested problem but not necessarily in a congested network.

        # p_f = path_probabilities(theta_array, YZ_x, Ix, C)
        # m =  response_function(Ix, Iq, q, p_f)

        sses = sse(theta_array, YZ_x, xc, q, Ix, Iq, C)

    top_sse = sses

    if pct_lowest_sse < 100:
        idxs_nonan = np.where(~np.isnan(xc))[0]

        n_top_obs = len(sses.flatten()[idxs_nonan])
        top_n = int(n_top_obs * pct_lowest_sse / 100)
        top_sse = np.sort(sses.flatten()[idxs_nonan])[0:top_n]

        n = top_n


    # We add some small constant to avoid problems in inference when the errors are close to 0 (1/\sigma^2 tends to infinity)

    epsilon = 1e-10

    sum_sse = np.sum(top_sse) # + epsilon

    var_error = sum_sse / (n - p)

    if numeric_hessian is True:
        # objective_function(theta_array, YZ_x, xc, q, Ix, Iq, C)


        # With normalization equals False, there is a 20X speed up in the numeric computation of the Hessian but inference is worsen
        print('Hessian is being computed numerically')
        H = nd.Hessian(objective_function_numeric_hessian)(list(theta_array.flatten()), YZ_x, xc,q, Ix, Iq, C, normalization)


        # # Automatic differentiation must be used with normalization = False because no gradient for np.nanmax is registered in autograd
        # It generates NA after inversion of Hessian in many instances. Also, it does not work with only a subset of the traffic counts is available

        # print('Hessian is being computed with automatic differentiation')
        # df = egrad(objective_function_numeric_hessian)
        # H = jacobian(egrad(df))
        # H = H(theta_array.flatten(), YZ_x, xc,q, Ix, Iq, C, normalization)

        # Replaced nan values in Hessian
        if np.any(np.isnan(H)):
            print('Cells of the Hessian matrix have NA values')
            H[np.isnan(H)] = epsilon



    else:
        print('Hessian is being approximated as its Jacobian J by the transpose of J')

        # # Unidimensional inverse function is just the reciprocal but this is multidimensional so inverse is required
        F, pf = jacobian_response_function(theta_array, YZ_x, q, Ix, Iq, C, paths_batch_size=0, x_bar = None, normalization = normalization)

        # Jacobian with automatic differentiation (nan max is not compatible)
        # F = jacobian(objective_function_numeric_jacobian)(theta_array.flatten(), YZ_x, xc,q, Ix, Iq, C, normalization)

        # Jacobian wth numeric computation
        # F = nd.Jacobian(objective_function_numeric_jacobian)(list(theta_array.flatten()), YZ_x, xc,q, Ix, Iq, C, normalization)

        H = F.T.dot(F)

        # Replaced nan values in Hessian
        if np.any(np.isnan(H)):
            print('Cells of the Hessian matrix have NA values')
            H[np.isnan(H)] = epsilon

    # #Robust approximation of covariance matrix (almost no difference with previous method)
    # cov_theta = np.linalg.lstsq(H, np.eye(F.shape[1]), rcond=None)[0]

    cov_theta = np.linalg.pinv(H)

    # i) T-tests

    critical_tvalue = stats.t.ppf(1 - alpha / 2, df=n - p)
    # ttest = (theta_array - theta_h0) / np.sqrt(np.diag(cov_theta)[:, np.newaxis])

    # Summing a small epsilon avoid cases where the terms in the diagonal of the covariance matrix are close to 0

    # Constrained terms in the covariance matrix to be positive or equal to epsilon
    ttest = (theta_array - theta_h0) / (np.sqrt(var_error * np.maximum(np.diag(cov_theta)[:, np.newaxis],epsilon)))

    pvalues = 2 * stats.t.sf(np.abs(ttest), df=n - p)  # * 2

    # ii) Confidence intervals
    width_confint = critical_tvalue * np.sqrt(var_error * np.maximum(np.diag(cov_theta)[:, np.newaxis],epsilon))

    confint_list = ["[" + str(round(float(i - j), 3)) + ", " + str(round(float(i + j), 3)) + "]" for i, j in zip(theta_array, width_confint)]

    # (iii) F-test

    theta_m1 = dict(zip(theta.keys(),theta_h0*np.ones(len(theta))))
    theta_m2 = theta

    ftest_value, criticalval_f, pval_f = ftest(theta_m1, theta_m2, YZ_x, xc, q, Ix, Iq, C, pct_lowest_sse = pct_lowest_sse, alpha = alpha, silent_mode = True)

    # summary_inference_parameters = pd.DataFrame(
    #     {'parameter': theta.keys(), 't-test': theta.values(), 'critical-t-value': list(criticalval_t * np.ones(len(theta))), 'p-value': pvals.flatten(), 'CI': confint_list, 'null_f_test': theta_m1.values()})

    summary_inference_parameters = pd.DataFrame(
        {'parameter': theta.keys(), 'est': theta_m2.values(),  'CI': confint_list, 'width_CI': width_confint.flatten(), 't-test': ttest.flatten(), 'p-value': pvalues.flatten()})

    summary_inference_model = pd.DataFrame({'f_test': np.array([ftest_value]),'critical-f-value': np.array([criticalval_f]) , 'p-value': np.array([pval_f]), 'sample_size': np.array([n])})


    print(str(round(pct_lowest_sse)) + '% of the total observations with lowest SSE were used')

    print('Time: ' + str(np.round(time.time() - t0, 1)) + '[s]')


    # return a pandas dataframe with summary of inference

    return summary_inference_parameters, summary_inference_model


def gradient_objective_function(theta: ColumnVector, YZ_x: Matrix, x_bar: ColumnVector, q: ColumnVector, Ix: Matrix,Iq: Matrix, C: Matrix, p_f: ColumnVector = None, attribute_k: int = None, standardization: dict = None, paths_batch_size: int = 0) -> ColumnVector:

    paths_idxs = []

    path_reduction = False

    # The path reduction will change results by eliminating those not traversing observed paths will change results, because in the logit model the alternatives probability are dependent of the utilities of the other in the same path set.

    if path_reduction:

        # idx_links_nas = np.where(np.isnan(x_bar))[0]
        idx_links_nonas = np.where(~np.isnan(x_bar))[0]

        # Identify indices where paths traverse some link with traffic counts.
        paths_idxs = list(np.where(np.sum(Ix[idx_links_nonas, :],axis = 0) == 1)[0])

        print('Path reduction found ' + str(len(paths_idxs)) + ' seemingly irrelevant paths')

    # Subsampling of paths

    if paths_batch_size > 0:
        paths_idxs = list(np.random.choice(paths_idxs,paths_batch_size, replace=False))

    if len(paths_idxs) == 0:

        Ix_sample = Ix
        Iq_sample = Iq
        C_sample = C

    else:
        Ix_sample = Ix[:, paths_idxs]
        Iq_sample = Iq[:, paths_idxs]
        C_sample = C[paths_idxs, paths_idxs]


    # Path probabilities (TODO: I may speed up this operation by avoiding elementwise division)

    if p_f is None:
        p_f = path_probabilities(theta, YZ_x, Ix_sample, C_sample)

    if len(paths_idxs) > 0:
        p_f_sample = p_f[paths_idxs]
    else:
        p_f_sample = p_f

    # Paths and links flows
    # f = np.multiply(Iq.T.dot(q), p_f)
    # x = Ix.dot(f)

    # TODO: perform the gradient operation for each attribute using a tensor
    gradient_l2norm = []

    # Jacobian/gradient of response function

    grad_m_terms = {}

    grad_m_terms[0] = Iq_sample.T.dot(q)

    # This is the availability matrix and it is very expensive to compute when using matrix operation Iq.T.dot(Iq) but not when calling function choice_set_matrix_from_M
    grad_m_terms[1] = C_sample  # computing Iq.T.dot(Iq) is too slow
    grad_m_terms[2] = p_f_sample.dot(p_f_sample.T)

    # This operation is performed for each attribute k. Then, the compl

    # Objective function
    m = response_function(Ix_sample, Iq_sample, q, p_f_sample)

    # Store the number of elements different than nan
    adjusted_n = np.count_nonzero(~np.isnan(x_bar))

    # To account for missing link counts
    x_bar = fake_observed_counts(xct_hat=m, xct=x_bar)

    for k in np.arange(theta.shape[0]):  # np.arange(len([*k_Y,*k_Z])):

        if attribute_k is not None:
            k = attribute_k

        # Attributes vector at link and path levels
        Zk_x = YZ_x[:, k][:, np.newaxis]
        Zk_f = Ix_sample.T.dot(Zk_x)

        if standardization is not None:
            Zk_f = preprocessing.scale(Zk_f, with_mean=standardization['mean'], with_std=standardization['sd'], axis=0)

        # TODO: This operation is very expensive and may be simplified
        grad_m_terms[3] = (np.ones(Zk_f.shape).dot(Zk_f.T) - Zk_f.dot(np.ones(Zk_f.shape).T))


        # TODO: This operation is even more expensive and may be simplified
        grad_m = Ix_sample.dot(np.multiply(grad_m_terms[0],
                                    np.multiply(grad_m_terms[1], np.multiply(grad_m_terms[2], grad_m_terms[3])))).dot(np.ones(Zk_f.shape))

        gradient_l2norm_k = float(2 * grad_m.T.dot(x_bar - m)) #/ adjusted_n

        gradient_l2norm.append(float(gradient_l2norm_k))

        if attribute_k is not None:
            break

    return np.array(gradient_l2norm)[:, np.newaxis]


def numeric_gradient_objective_function(theta: ColumnVector, YZ_x, x_bar, q, Ix, Iq, C, epsilon=1e-7):
    grads_theta = []

    for i in np.arange(theta.shape[0]):
        epsilon_v = np.zeros(len(theta))[:, np.newaxis]
        epsilon_v[i] = epsilon

        H = np.mean(objective_function(theta=theta - epsilon_v, YZ_x=YZ_x, x_bar=x_bar, q=q, Ix=Ix, Iq=Iq, C=C))
        F = np.mean(objective_function(theta=theta + epsilon_v, YZ_x=YZ_x, x_bar=x_bar, q=q, Ix=Ix, Iq=Iq, C=C))

        grads_theta.append((F - H) / (2 * epsilon))

    return np.array(grads_theta)[:, np.newaxis]


def gradient_objective_function_check(theta: ColumnVector, YZ_x, q, Ix, Iq, C, x_bar,paths_batch_size):
    # Source: https://towardsdatascience.com/debugging-your-neural-nets-and-checking-your-gradients-f4d7f55da167
    # theta = np.array(list(theta0.values()))[:,np.newaxis]

    return np.linalg.norm(
        numeric_gradient_objective_function(theta, YZ_x, x_bar, q, Ix, Iq, C) - gradient_objective_function(theta, YZ_x, x_bar, q, Ix, Iq, C, paths_batch_size = paths_batch_size))


def hessian_objective_function(theta: ColumnVector, x_bar, YZ_x, q, Ix, Iq, C, p_f: ColumnVector = None, approximation = False, paths_batch_size: int = 0):


    #TODO: Enable estimation of Hessian using batch size as it is done for gradient computation

    # Hesisan was debugged. Something mandatory is that at the optima, the hessian is positive as it is a minimizer

    # http://math.gmu.edu/~igriva/book/Appendix%20D.pdf
    # https: // www.eecs189.org / static / notes / n12.pdf

    if p_f is None:
        p_f = path_probabilities(theta, YZ_x, Ix, C)

    hessian_l2norm = []

    if approximation:
        # epsilon = 1e-3
        # # hessian_l2norm_k = float(jac_m_k.T.dot(jac_m_k))
        # f_plus, p_f_sample   = jacobian_response_function(theta+epsilon, YZ_x, q, Ix, Iq, C, paths_batch_size=0)
        # f_minus, p_f_sample = jacobian_response_function(theta-epsilon, YZ_x, q, Ix, Iq, C, paths_batch_size=0)
        #
        # hessian_l2norm = np.mean((f_plus - f_minus) / (2 * epsilon), axis=0)

        print('Hessian is being computed numerically')
        theta_array = theta
        H = np.diag(nd.Hessian(objective_function_numeric_hessian)(list(theta_array.flatten()), YZ_x, x_bar,q, Ix, Iq, C,  False))

    else:

        # This operation is performed for each attribute k
        for k in np.arange(theta.shape[0]):  # np.arange(len([*k_Y,*k_Z])):
            # k = 0
            grad_p_f_terms = {}

            # Attributes vector at link and path level
            Zk_x = YZ_x[:, k][:, np.newaxis]
            Zk_f = Ix.T.dot(Zk_x)

            # Gradient for path probabilities

            grad_p_f_terms[0] = C  # grad_m_terms[1]
            grad_p_f_terms[1] = p_f.dot(p_f.T)  # grad_m_terms[2]
            grad_p_f_terms[2] = (np.ones(Zk_f.shape).dot(Zk_f.T) - Zk_f.dot(np.ones(Zk_f.shape).T))  # grad_m_terms[3]
            grad_p_f = np.multiply(grad_p_f_terms[0], np.multiply(grad_p_f_terms[1], grad_p_f_terms[2])).dot(np.ones(Zk_f.shape))

            # Gradient of objective function
            jac_m_k_terms = {}
            jac_m_k_terms[0] = Iq.T.dot(q)
            jac_m_k_terms[1] = C  # computing Iq.T.dot(Iq) is too slow
            jac_m_k_terms[2] = grad_p_f_terms[1]
            jac_m_k_terms[3] = (np.ones(Zk_f.shape).dot(Zk_f.T) - Zk_f.dot(np.ones(Zk_f.shape).T))

            jac_m_k = Ix.dot(np.multiply(jac_m_k_terms[0], np.multiply(jac_m_k_terms[1], np.multiply(jac_m_k_terms[2],jac_m_k_terms[3])))).dot(np.ones(Zk_f.shape))

            # jac_m = Ix.dot(np.multiply(jac_m_k_terms[0], np.multiply(jac_m_k_terms[1], np.multiply(jac_m_k_terms[2],jac_m_k_terms[3]))))

            hessian_m_terms = {}
            hessian_m_terms[0] = jac_m_k_terms[0]  # jac_m_k_terms[0]
            hessian_m_terms[1] = jac_m_k_terms[1]  # jac_m_k_terms[1]
            hessian_m_terms[2] = grad_p_f.dot(p_f.T) + p_f.dot(grad_p_f.T) #This is the key term
            hessian_m_terms[3] = jac_m_k_terms[3]

            hessian_m_k = Ix.dot(np.multiply(hessian_m_terms[0], np.multiply(hessian_m_terms[1], np.multiply(hessian_m_terms[2],hessian_m_terms[3]))).dot(np.ones(Zk_f.shape)))

            # hessian_m_k = Ix.dot(np.multiply(hessian_m_terms[0], np.multiply(hessian_m_terms[1], np.multiply(hessian_m_terms[2],hessian_m_terms[3]))))

            m = response_function(Ix, Iq, q, p_f)

            # Hessian of objective function

            # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts
            # hessian_l2norm_k = float(2 * jac_m_k.T.dot(jac_m_k) + hessian_m_k.T.dot(x_bar - m))

            # To account for missing link counts
            x_bar = fake_observed_counts(xct_hat = m, xct = x_bar)




            hessian_l2norm_k = -2*float(hessian_m_k.T.dot(x_bar-m)-jac_m_k.T.dot(jac_m_k))


            # Store hessian for particular attribute k
            hessian_l2norm.append(hessian_l2norm_k)

            H = hessian_l2norm



    return np.array(H)[:, np.newaxis] / x_bar.shape[0]


def hessian_check(theta):
    # Autograd https://rlhick.people.wm.edu/posts/mle-autograd.html
    # http: // www.cs.toronto.edu / ~rgrosse / courses / csc321_2017 / tutorials / tut4.pdf

    # Source: https://towardsdatascience.com/debugging-your-neural-nets-and-checking-your-gradients-f4d7f55da167
    # theta = np.array(list(theta0.values()))

    # Numeric diff
    # https: // v8doc.sas.com / sashtml / ormp / chap5 / sect28.htm

    # grads_theta = []
    #
    # epsilon = 1e-7
    #
    # for i in np.arange(theta.shape[0]):
    #     epsilon_v = np.zeros(len(theta))[:, np.newaxis]
    #     epsilon_v[i] = epsilon
    #
    #     fun = lambda theta_x: gradient_objective_function(theta=theta_x, Ix=Ix, YZ_x=YZ_x)
    #
    #     grad_fun = nd.Derivative(fun)
    #     grad_fun_theta = grad_fun(theta)
    #     grad_fun_theta/np.linalg.norm(grad_fun_theta)
    #
    #     hessian_fun = nd.Derivative(fun,2)
    #     hessian_fun_theta = hessian_fun(theta)
    #     hessian_fun_theta / np.linalg.norm(hessian_fun_theta)
    #
    #     b = hessian_l2norm(theta, YZ_x, Ix)
    #     b/ np.linalg.norm(b)
    #
    #     J, pf =jacobian_response_function(theta, YZ_x, Ix)
    #     c = J.T.dot(J)
    #
    #     np.diag(c)
    #
    #     np.diag(c)/np.linalg.norm(np.diag(c))
    #
    #     b = numeric_gradient_l2norm(theta)
    #
    #     numeric_gradient_l2norm(theta)/np.linalg.norm(b)
    #
    #     epsilon_v = np.zeros(len(theta))[:, np.newaxis]
    #     epsilon_v[i] = epsilon
    #
    #     H = np.mean(objective_function(theta=theta - epsilon_v, Ix=Ix, YZ_x=YZ_x))
    #     F = np.mean(objective_function(theta=theta + epsilon_v, Ix=Ix, YZ_x=YZ_x))
    #
    #     grads_theta.append((F - H) / (2 * epsilon))
    #
    #
    #     gradient_check(theta)
    #
    #
    #     H = np.mean(gradient_objective_function(theta=theta - epsilon_v, Ix=Ix, YZ_x=YZ_x))
    #     F = np.mean(gradient_objective_function(theta=theta + epsilon_v, Ix=Ix, YZ_x=YZ_x))
    #
    #     grads_theta.append((F - H) / (2 * epsilon))
    # #
    # # print(np.array(grads_theta))
    # #
    # # print(gradient_objective_function(theta, YZ_x, Ix))
    #
    # gap = np.linalg.norm(np.array(grads_theta)[:, np.newaxis] - hessian_l2norm(theta, YZ_x, Ix))
    #
    #
    # gap = np.linalg.norm(np.array(H)[:,np.newaxis]-hessian_l2norm(theta, YZ_x, Ix))

    raise NotImplementedError


class Estimation:
    def __init__(self, theta_0):

        self.theta_0 = theta_0


        # Past gradient for momentum
        self.grad_old = None

        # Accumulator of gradients for adagrad
        self.acc_grads = np.zeros(len(self.theta_0))[:, np.newaxis]

        # First (m) and second moments (v) accumulators for adam
        self.acc_m = np.zeros(len(self.theta_0))[:, np.newaxis]
        self.acc_v = np.zeros(len(self.theta_0))[:, np.newaxis]

    def reset_gradients(self):
        # Accumulator of gradients for adagrad
        self.acc_grads = np.zeros(len(self.theta_0))[:, np.newaxis]

        # First (m) and second moments (v) accumulators for adam
        self.acc_m = np.zeros(len(self.theta_0))[:, np.newaxis]
        self.acc_v = np.zeros(len(self.theta_0))[:, np.newaxis]


    # @blockPrinting
    def odtheta_estimation_outer_problem(self, Nt: TNetwork, k_Y: [], Yt: {}, k_Z: [], Zt: {}, q0: ColumnVector, xct: {}, theta0: LogitParameters, outeropt_params: {}, p_f_0: ColumnVector = None, q_bar: ColumnVector = None, standardization: dict = None):
        """ Address uncongested case first only and with data from a given day only.
        TODO: congested case, addressing multiday data, stochastic gradient descent

        Arguments
        ----------
        :argument f: vector with path flows
        :argument  opt_params={'method': None, 'iters_scaling': 1, 'iters_gd': 0, 'gamma': 0, 'eta_scaling': 1,'eta': 0, 'batch_size': int(0)}
        :argument

        """

        # batch_size = 0

        # Adam hyperparameters beta_1, beta_2 (when set to 0, we get adagrad)
        # beta_1 = 0.5#0.9
        # beta_2 = 0.5#0.999

        method, iters_scaling, iters, eta_scaling, eta, gamma, batch_size, paths_batch_size,v_lm, lambda_lm, beta_1, beta_2 = [outeropt_params[i] for i in ['method', 'iters_scaling', 'iters','eta_scaling', 'eta', 'gamma', 'batch_size', 'paths_batch_size', 'v_lm', 'lambda_lm', 'beta_1', 'beta_2']]

        if method == 'gd' or method == 'ngd':
            print('\nLearning logit params via ' + method + ' ('+ str(int(iters)) + ' iters, eta = ' + "{0:.1E}".format(eta) + ')\n')

        if method == 'nsgd':
            print('\nLearning logit params via ' + method + ' ('+ str(int(iters)) + ' iters, eta = ' + "{0:.1E}".format(eta) + ')\n')

        if method == 'adagrad' or method == 'adam':
            print('\nLearning logit params via ' + method + ' ('+ str(int(iters)) + ' iters, eta = ' + "{0:.1E}".format(eta) + ')\n')

        if method in ['gauss-newton','lm','lm-revised']:
            print('\nLearning logit params via ' + method + ' (' + str(int(iters)) + ' iters )\n')

            print('Damping factor: ' + "{0:.1E}".format(lambda_lm))


        if batch_size > 0:
            print('batch size for observed link counts = ' + str(batch_size))
        if paths_batch_size > 0:
            print('batch size for paths = ' + str(paths_batch_size))


        # TODO: the analysis with multiday data remains to be implemented
        day = 1

        n = np.array(list(Yt[day]['tt'].values())).shape[0]

        # days = Mt.keys()
        # n_days = len(list(days))

        # Starting values for optimization (q0, theta0)
        # theta0 = dict.fromkeys([*k_Y, *k_Z], -1)

        # Matrix of  endogenous (|Y|x|K_Y|) and exogenous link attributes (|Z|x|K_Z|)
        YZ_x = get_design_matrix(Y=Yt[day], Z=Zt[day], k_Y=k_Y, k_Z=k_Z)

        # Incidences matrices
        Ix = Nt.D
        Iq = Nt.M

        # OD
        # q = q0[:, np.newaxis]
        q = q0 #Nt.q

        # Measurements of dependent variable
        x_bar = xct[day][:, np.newaxis]

        total_no_nans = np.count_nonzero(~np.isnan(x_bar))

        assert batch_size < total_no_nans, 'Batch size larger than size of observed counts vector'


        # This is the availability matrix or choice set matrix. Note that it is very expensive to compute when using matrix operation Iq.T.dot(Iq) so a more programatic method was preferred
        C = Nt.C

        # Gradient and hessian checks
        # a = hessian_l2norm(theta, YZ_x, Ix, idx_links = np.arange(0, n))
        # print('hessian_diff : ' + str(hessian_check(theta=np.array(list(theta0.values()))[:, np.newaxis])))
        # assert gradient_check(theta = np.array(list(theta0.values()))[:,np.newaxis])<1e-6, 'unreliable gradients'
        # print('gradient_diff : ' + str(gradient_objective_function_check(theta =np.array(list(theta0.values()))[:, np.newaxis], YZ_x = YZ_x, q = q, Ix = Ix, Iq = Iq, C = C, x_bar = x_bar)))
        # gradient_check(theta = 10*np.ones(len(np.array(list(theta0.values()))))[:,np.newaxis])

        theta = np.array(list(theta0.values()))[:, np.newaxis]

        # Path probabilities
        if p_f_0 is None:
            p_f_0 = path_probabilities(theta, YZ_x, Ix, C)

        # List to store infromation over iterations
        thetas = [theta]
        grads = []
        times = [0]
        acc_t = 0

        t0 = time.time()

        # epsilon = 1e-4

        # batch_size = 0#int(n*1)
        # idx = np.arange(0, n)

        # i) Scaling method will set all parameters to be equal in order to achieve quickly a convex region of the problem

        for iter in range(0, iters_scaling):

            if iter == 0:
                theta = np.array([1])[:, np.newaxis]
            # thetas[0] * float(theta)

            grad_old = gradient_objective_function(theta, YZ_x.dot(thetas[0].dot(float(theta))), x_bar, q, Ix, Iq, C, paths_batch_size = paths_batch_size)

            # For scaling this makes more sense using a sign function as it operates as a grid search and reduce numerical errors

            theta = theta - np.sign(grad_old) * eta_scaling

            # Record theta with the new scaling
            thetas.append(thetas[0] * theta)
            # obj_fun = objective_function(theta=thetas[0] * theta, Ix=Ix, YZ_x=YZ_x)

            # For multidimensional case
            # theta = theta - (grad_old)/np.linalg.norm(grad_old+epsilon) * eta_ngd  #epsilon to avoid problem when gradient is 0

            if iter == iters_scaling - 1:
                initial_objective = np.mean(
                    objective_function(theta=thetas[0], YZ_x=YZ_x, x_bar=x_bar, q=q, Ix=Ix, Iq=Iq, C=C))
                print('initial objective: ' + str(initial_objective))
                final_objective = np.mean(
                    objective_function(theta=thetas[-1], YZ_x=YZ_x, x_bar=x_bar, q=q, Ix=Ix, Iq=Iq, C=C))
                print('objective in iter ' + str(iter + 1) + ': ' + str(final_objective))
                print('scaling factor: ' + str(theta))
                print('theta after scaling: ' + str(thetas[-1].T))
                print('objective improvement due scaling: ' + str(1 - final_objective / initial_objective))

        ## ii) Gradient descent or newton-gauss for fine scale optimization

        theta = thetas[-1]

        theta_update_new = 0

        p_f = p_f_0.copy()

        for iter in range(0, iters):

            printer.printProgressBar(iter, iters, prefix='Progress:', suffix='',
                                     length=20)

            # It is very expensive to compute p_f so adequating the code is important
            if iter > 0:
                p_f = path_probabilities(theta, YZ_x, Ix, C)

            # Stochastic gradient/hessian
            if batch_size > 0:
                # Generate a random subset of idxs depending on batch_size
                missing_sample_size = total_no_nans - batch_size

                # Sample over the cells of the x_bar vector with no nan

                idx_nonas = np.where(~np.isnan(x_bar))[0]

                # Set nan in the sample outside of batch that has no nan
                idx_nobatch = list(np.random.choice(idx_nonas, missing_sample_size, replace=False))

                x_bar_masked = masked_observed_counts(xct=x_bar, idx=idx_nobatch)

            # i) First order optimization methods

            if method in ['gd', 'ngd', 'sgd', 'nsgd', 'adagrad', 'adam']:

                x_bar_masked = copy.deepcopy(x_bar)

                # Sample a batch in the gradient vector

                grad_current = gradient_objective_function(theta, YZ_x, x_bar_masked, q, Ix, Iq, C, p_f, standardization=standardization, paths_batch_size = paths_batch_size)#/total_no_nans



                # TODO: review momentumformula (https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d)
                # grad_old = (1 - gamma) * grad_old + gamma * grad_new
                if self.grad_old is None:
                    # So we avoid that first graident is 0
                    grad_adj =  grad_current

                else:
                    grad_adj = gamma * self.grad_old + (1-gamma) * grad_current

                self.grad_old = grad_adj

                # TODO implement nesterov acelerated gradient descent
                # https: // ruder.io / optimizing - gradient - descent /

                if method == 'gd':
                    #theta_update_new = gamma * (theta_update_old) + eta * grad_new

                    # Gradient update (with momentum)
                    theta = theta - grad_adj*eta

                    #theta_update_old = theta_update_new

                if method == 'ngd':
                    epsilon = 1e-7

                    if len(theta) == 1:
                        theta = theta - np.sign(grad_adj)* eta
                    else:
                        theta = theta - (grad_adj) / np.linalg.norm(grad_adj + epsilon) * eta  # epsilon to avoid problem when gradient is 0

                    # print('gradient_diff : ' + str(gradient_check(theta=theta)))


                if method == 'adagrad':

                    # TODO: Fix adagrad using diagonal matrix for accumulated gradients and use acumulated gradients from past bilevel iterations. The same applies for Adam and momentum updates

                    epsilon = 1e-7

                    # self.acc_grads = 0

                    self.acc_grads += grad_adj**2

                    theta = theta - eta/(np.sqrt(self.acc_grads) +epsilon)*grad_adj

                if method == 'adam':

                    epsilon = 1e-7

                    # Compute and update first moments (m) and second moments (v)
                    self.acc_m = beta_1*self.acc_m + (1-beta_1)*grad_adj
                    self.acc_v = beta_2*self.acc_v + (1-beta_2)*grad_adj**2

                    # Adjusted first and second moments
                    adj_m = self.acc_m/(1-beta_1)
                    adj_v = self.acc_v/(1-beta_2)

                    # Parameter update
                    theta = theta - eta* (adj_m / (np.sqrt(adj_v) + epsilon))

                grads.append(grad_adj)

            # ii) Second order optimization methods

            # Gauss-newthon exploit when it is not in a convex region

            if method == 'newton':
                # TODO: this can be speed up by avoid recomputing some terms two times for both the gradient and Hessian

                G = gradient_objective_function(theta, YZ_x, x_bar, q, Ix, Iq, C, p_f, paths_batch_size = paths_batch_size)

                H = hessian_objective_function(theta, x_bar, YZ_x, q, Ix, Iq, C, p_f, paths_batch_size = paths_batch_size)

                theta = theta - np.linalg.pinv(H).dot(G)

                # # Stable version
                # theta = theta + np.linalg.lstsq(H, -G, rcond=None)[0]

            if method == 'gauss-newton':

                # The computation time is roughly the same with NGD as this method does not compute the Hessian but approximate it


                J, p_f_sample, Ix_sample, Iq_sample = jacobian_response_function(theta, YZ_x, q, Ix, Iq, C, p_f, paths_batch_size, x_bar)

                pred_x = response_function(Ix_sample, Iq_sample, q, p_f_sample)

                delta_y = fake_observed_counts(pred_x,x_bar)-pred_x

                # # Package solution from scipy
                # theta = theta + la.lstsq(J, -delta_y )[0]

                # # Update
                # theta = theta +  np.linalg.inv(J.T.dot(J)).dot(J.T).dot(delta_y)
                # theta = theta + np.linalg.pinv(J.T.dot(J)).dot(J.T).dot(delta_y)

                # #Stable solution but works only if matrix is full rank
                # theta = theta +  np.linalg.solve(J.T.dot(J), J.dot(delta_y))

                # Lstsq is used to obtain an approximate solution for the system
                # https://nmayorov.wordpress.com/2015/06/18/basic-asdlgorithms-for-nonlinear-least-squares/
                theta = theta +  np.linalg.lstsq(J.T.dot(J), J.T.dot(delta_y))[0]

                # Understand why we ignore the J^T term when doing this, maybe this means just to multiply for a pseudo inverse. This is certainly more numerically stable
                # theta = theta + np.linalg.lstsq(J, -delta_y, rcond=None)[0]

                # print('here')

            if method == 'lm' or method == 'lm-revised':

                #Source: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm


                J, p_f_sample, Ix_sample, Iq_sample = jacobian_response_function(theta, YZ_x, q, Ix, Iq, C, p_f, paths_batch_size, x_bar)

                pred_x = response_function(Ix_sample, Iq_sample, q, p_f_sample)

                delta_y = fake_observed_counts(pred_x, x_bar) - pred_x

                # lambda_lm = 0.1#0.2

                J_T_J = J.T.dot(J)

                if method == 'lm-revised':

                    theta = theta + np.linalg.lstsq(J_T_J + lambda_lm*np.diag(J_T_J), -J.T.dot(delta_y), rcond=None)[0]

                if method == 'lm':
                    theta = theta + np.linalg.lstsq(J_T_J + lambda_lm*np.eye(J_T_J.shape[0]), -J.T.dot(delta_y), rcond=None)[0]

                #Choice of damping factor

                # Marquardt recommended starting with a value \lambda _{0} and a factor \nu > 1. Initially setting \lambda =\lambda _{0} and computing the residual sum of squares  after one step from the starting point with the damping factor of \lambda =\lambda _{0} and secondly with \lambda _{0}/\nu . If both of these are worse than the initial point, then the damping is increased by successive multiplication by \nu  until a better point is found with a new damping factor of {\lambda _{0}\nu^{k} for some k.


            delta_t = time.time() - t0
            acc_t += delta_t

            thetas.append(theta)
            times.append(acc_t)

        q = q0

        if 'od_estimation' in outeropt_params and outeropt_params['od_estimation']:

            # Optimization of OD matrix (using plain gradient descent meantime)
            jacobian_q_x = Ix.dot(p_f.dot(np.ones([p_f.size,1]).T)).dot(Iq.T)

            # Prediction of link flow
            pred_x = response_function(Ix, Iq, q, p_f)

            # Hyperparam
            # lambda_q = 1e2 # This works very well for Yang network
            lambda_q = 1e2 # This works very well for Sioux network

            # This is a sort of RELU in neural networks terms
            grad_new_q = 1/x_bar.size*(lambda_q*2*(q0-q_bar)+2*jacobian_q_x.T.dot(pred_x-x_bar))

            #Projected Gradient descent update
            # https://www.cs.ubc.ca/~schmidtm/Courses/5XX-S20/S5.pdf
            # eta_q = 1e-2 # This works very well for Yang network and GD
            #
            eta_q = 1e-4 # This works very well for Sioux network and GD

            q =  np.maximum(np.zeros([q0.size,1]), q0 - grad_new_q * eta_q)

            # # Projected adagrad
            #
            # # eta_q = 1e-7  # This works very well for Yang with adagrad
            # eta_q = 1e-11 # This works very well for Sioux with adagrad
            # #
            # acc_grads_q += grad_new_q ** 2
            # #
            # epsilon = 1e-8
            # Gt = np.diag(acc_grads_q.flatten())
            # q = np.maximum(np.zeros([q0.size,1]), q0 - (eta_q / np.sqrt(Gt+ epsilon)).dot(grad_new_q))



            print(repr(np.round(q.T,1)))

        # # Values of objective function
        # obj_fun = objective_function(theta=theta, Ix=Ix, YZ_x=YZ_x)

        # if iter == 0:
        #     print('initial objective function:' + str(np.sum(obj_fun)))
        #     print('initial objective function:' + str(np.sum(obj_fun)))
        # print(theta)

        # print('Method used is :' + method)
        # print('initial theta: ' + str(thetas[0].T))
        # print('theta in iter ' + str(iter + 1) + ': ' + str(theta.T))

        # initial_objective = np.mean(
        #     objective_function(theta=thetas[0], YZ_x=YZ_x, x_bar=x_bar, q=q, Ix=Ix, Iq=Iq, C=C, p_f=p_f_0))

        # Do not provide p_f in this part, because it does not correspond to the p_f in the last iteration but the one before the last
        # Thus, if there is only one iteration p_f and p_f_0 are equal, and the value of the objective function is wrongly assumed to be the same
        final_objective = np.mean(
            objective_function(theta=thetas[-1], YZ_x=YZ_x, x_bar=x_bar, q=q0, Ix=Ix, Iq=Iq, C=C))


        # print('objective in iter ' + str(iter + 1) + ': ' + str(final_objective))
        # print('objective improvement : ' + str(1 - final_objective / initial_objective))

        # print('theta: ' + str(np.array(list(theta_current.values()))))

        # grad_old = grads[-1]

        theta = thetas[-1]

        # Conver theta into a dictionary to return it then
        theta_dict = dict(zip([*k_Y, *k_Z], theta.flatten()))

        # print('theta: ' + str({key: round(val, 4) for key, val in theta_dict.items()}))

        print('theta: ' + str({key: "{0:.1E}".format(val) for key, val in theta_dict.items()}))
        # print('theta: ' + str({key: round(val,3) for key, val in theta_dict.items()}))

        if 'c' in theta_dict.keys() and theta_dict['c'] != 0:
            print('Current ratio theta: ' + str(round(theta_dict['tt'] / theta_dict['c'], 4)))

        print('time: ' + str(np.round(time.time() - t0, 1)) + '[s]')

        return q, theta_dict, grads, final_objective
        # return theta_dict, None, final_objective


    def odtheta_estimation_bilevel(self, Nt: TNetwork, Zt: {}, k_Y: list, k_Z: list, xct: {}, theta0: {}, q0: np.ndarray, outeropt_params: {}, inneropt_params: {}, bilevelopt_params: {}, q_bar: ColumnVector = None, standardization: dict = None, n_paths_column_generation: int = 0, silent_mode = True):

        """ Congested case where Yt do not necessarily inform on the optimal travel times

        Arguments
        ----------
        :argument f: vector with path flows
        :argument M: Path/O-D demand incidence
        :argument D: Path/link incidence matrix
        :argument xct: link counts
        :argument Ni: Network object (TODO: name will be later updated to N)
        """

        print('\nBilevel optimization for ' + str(Nt.key) + ' network \n')

        # TODO: Schedule the number of iterations so they are higher when the bilevel optimization is finishing

        # Mt = {1: N['train'][i].M}
        # Ct = {1: tai.estimation.choice_set_matrix_from_M(N['train'][i].M)}
        # Dt = {1: N['train'][i].D}
        # k_Y = k_Y; k_Z = k_Z
        # Yt = {1: N['train'][i].Y_dict}
        # Zt = {1: N['train'][i].Z_dict}
        # q0 = tai.network.denseQ(Q=N['train'][i].Q,remove_zeros=remove_zeros_Q)
        # theta0 = dict.fromkeys([*k_Y, *k_Z], 0)
        #
        # results_sue_msa = tai.equilibrium.sue_logit_msa_k_paths(N=N['train'][i], maxIter=maxIter, accuracy=0.01,
        #                                                            theta=theta_true[i])
        #
        # N['train'][i].set_Y_attr_links(y=results_sue_msa['tt_x'], label='tt')
        # # N['train'][i].Y_dict
        # xct = np.array(list(results_sue_msa['x'].values()))

        # xct = {1: N['train'][i].x}

        # TODO: the analysis with multiday data remains to be implemented
        day = 1

        # Update od
        q_current = copy.deepcopy(q0)


        # print('there')

        # N_copy = transportAI.networks.clone_network(N=Nt, label='Clone', randomness = {'Q': False, 'BPR': False, 'Z': False})#[N.label]
        # N_copy = Nt

        # Initialization

        # TODO: this assume that k_Z has at least one attribute but it may be not the case
        theta_current = {i: theta0[i] for i in [*k_Y, *k_Z]}



        # print(theta_current)
        # theta_current = theta_true[i].copy()
        # theta_new = theta_current.copy()


        x_bar = xct[day][:, np.newaxis]

        print('Iteration : ' + str(1) + '/' + str(int(bilevelopt_params['iters'])))


        results = {}
        results_eq = {}

        # Initial objective function

        initial_inneropt_params = copy.deepcopy(inneropt_params)
        initial_inneropt_params['k_path_set_selection'] = 0

        results_eq[1] = equilibrium.sue_logit_iterative(Nt=Nt, theta= theta_current, q = q_current, k_Y=k_Y, k_Z=k_Z, params=initial_inneropt_params, n_paths_column_generation=0, standardization = standardization)

        # x_eq = np.array(list(results_eq_initial['x'].values()))
        x_eq = np.array(list(results_eq[1]['x'].values()))[:,np.newaxis]

        # Update values in network
        Nt.x_dict = results_eq[1]['x']  # #
        Nt.x = np.array(list(Nt.x_dict.values()))[:,np.newaxis]
        Nt.set_Y_attr_links(y=results_eq[1]['tt_x'], label='tt')

        initial_objective = loss_function(x_bar=x_bar, x_eq=x_eq)

        results[1] = {'theta': theta_current, 'q': q_current, 'objective': initial_objective,'equilibrium': results_eq[1]}

        # Ix = Nt.D
        # Iq = Nt.M
        # YZ_x = get_design_matrix(Y={'tt': results_eq_initial['tt_x']}, Z=Zt[day], k_Y=k_Y, k_Z=k_Z)

        # final_objective = np.mean(
        #     objective_function(theta=np.array(list(theta_current.values())), YZ_x=YZ_x, x_bar=x_bar, q=q0[:, np.newaxis],
        #                        Ix=Ix, Iq=Iq, C=Nt.C, p_f=results_eq_initial['p_f']))


        # print('\nInitial theta: ' +str({key: round(val, 3) for key, val in theta_current.items()}))
        print('Initial theta: ' + str({key: "{0:.1E}".format(val) for key, val in theta_current.items()}))
        # print('Initial q: ', repr(np.round(q_current.T,1)))
        print('Initial objective: ' + '{:,}'.format(round(initial_objective)))

        if 'c' in theta_current.keys() and theta_current['c'] != 0:
            print('Initial ratio theta: ' + str(round(theta_current['tt'] / theta_current['c'], 4)))

        # if iter < 1:

        idx_nonas = list(np.where(~np.isnan(x_bar))[0])

        link_keys = [(link.key[0],link.key[1]) for link in Nt.links if not np.isnan(link.observed_count)]

        # [link.observed_count for link in Nt.links]

        link_capacities = np.array([link.bpr.k for link in Nt.links if not np.isnan(link.observed_count)])

        link_capacities_print = []
        for i in range(len(link_capacities)):
            if link_capacities[i] > 10000:
                link_capacities_print.append(float('inf'))
            else:
                link_capacities_print.append(link_capacities[i])

        initial_error_by_link = error_by_link(x_bar, x_eq, show_nan=False)

        # Travel time are originally in minutes
        link_travel_times = \
            np.array([str(np.round(link.get_traveltime_from_x(Nt.x_dict[link.key]), 2)) for link in Nt.links if not np.isnan(link.observed_count)])

        # This may raise a warning for OD connectors where the length is equal to 0

        if 'length' in Nt.links[0].Z_dict:
            link_speed_av = \
                np.array([np.round(60 * link.Z_dict['length'] / link.get_traveltime_from_x(Nt.x_dict[link.key]), 1) for link in
                          Nt.links if not np.isnan(link.observed_count)])

            # link_speed_av = \
            #     np.array([np.round(60 * link.Z_dict['length'] / link.get_traveltime_from_x(Nt.x_dict[link.key]), 1) for link in
            #               Nt.links])[
            #     idx_nonas]

        else:
            link_speed_av = np.array([0]*len(link_travel_times.flatten()))

        summary_table = pd.DataFrame(
            {'link_key': link_keys
                , 'capacity': link_capacities_print
                , 'tt[min]': link_travel_times.flatten()
                , 'speed[mi/hr]': link_speed_av.flatten()
                , 'count': x_bar[idx_nonas].flatten()
                , 'predicted': x_eq[idx_nonas].flatten()
                , 'error': initial_error_by_link.flatten()})

        if silent_mode is False:

            with pd.option_context('display.float_format', '{:0.1f}'.format):
                print('\n' + summary_table.to_string())

        # initial_error_by_link = error_by_link(x_bar, x_eq, show_nan=False)
        # initial_error_by_link = error_by_link(x_bar, x_eq, show_nan=False)
        # print('Initial error by link', initial_error_by_link)

        # print(results_eq_initial['tt_x'])



        # p_f_0 = results_eq[0]['p_f']
        # objective_vals = {}

        best_iter = 1
        best_objective = initial_objective #float("inf")
        best_q = copy.deepcopy(q_current)
        best_theta = copy.deepcopy(theta_current)

        objective_values = [initial_objective]

        errors_by_link = [initial_error_by_link]


        outeropt_params['lambda_lm'] = outeropt_params['lambda_lm']/outeropt_params['v_lm']

        t0_global = time.time()

        for iter in np.arange(2,bilevelopt_params['iters']+1, 1):

            print('\nIteration : ' + str(iter) + '/' + str(int(bilevelopt_params['iters'])) )

            t0 = time.time()

            # if iter > 0:
            p_f_0 = results_eq[iter-1]['p_f']

            # Outer problem (* it takes more time than inner problem)
            q_new, theta_new, grad_new, objective_value \
                = self.odtheta_estimation_outer_problem(Nt=Nt, k_Y=k_Y, k_Z=k_Z,
                                                   Yt={1: Nt.Y_dict}, Zt=Zt, q0=q_current,
                                                   xct=xct,
                                                   theta0=theta_current, outeropt_params=outeropt_params,
                                                   p_f_0= p_f_0
                                                   , standardization = standardization
                                                   , q_bar = q_bar
                                                   )

            q_current = q_new.copy()
            theta_current = theta_new.copy()

            # Inner problem
            results_eq[iter] = equilibrium.sue_logit_iterative(Nt=Nt, theta=theta_current, q = q_current, k_Y=k_Y, k_Z=k_Z, params=inneropt_params, n_paths_column_generation= n_paths_column_generation, standardization=standardization)

            # Compute new value of link flows and objective function at equilibrium
            x_eq = np.array(list(results_eq[iter]['x'].values()))[:,np.newaxis]
            objective_value = loss_function(x_bar=x_bar, x_eq=x_eq)

            # Update values in network
            Nt.x_dict = results_eq[iter]['x']  # #
            Nt.x = np.array(list(Nt.x_dict.values()))[:,np.newaxis]
            Nt.set_Y_attr_links(y=results_eq[iter]['tt_x'], label='tt')

            if objective_value < best_objective:
                best_objective = objective_value
                best_x_eq = copy.deepcopy(x_eq)
                best_theta = copy.deepcopy(theta_current)
                best_q = copy.deepcopy(q_current)
                best_iter = iter

            print('\nCurrent objective_value: ' +'{:,}'.format(round(objective_value)))
            print('Current objective improvement: ' + "{:.2%}". format(np.round(1 - best_objective / initial_objective, 4)))
            # print('Time current iteration: ' + str(np.round(time.time() - t0, 1)) + ' [s]')



            results[iter] = {'theta': theta_current, 'q':q_current,'objective': objective_value, 'equilibrium': results_eq[iter]}

            objective_values.append(objective_value)

            if len(objective_values)>=2:

                if objective_values[-1] < objective_values[-2]:

                    # Choice of damping factor for lm
                    outeropt_params['lambda_lm'] = outeropt_params['lambda_lm'] * outeropt_params['v_lm']

                else:

                    # Choice of damping factor for lm
                    outeropt_params['lambda_lm'] = outeropt_params['lambda_lm'] / outeropt_params['v_lm']

                print('Marginal objective improvement: ' + "{:.2%}".format(
                    np.round(1 - objective_values[-1] / objective_values[-2], 4)))
                print('Marginal objective improvement value: ' + '{:,}'.format(
                    np.round(objective_values[-2] - objective_values[-1], 1)))

            idx_nonas = list(np.where(~np.isnan(x_bar))[0])

            link_keys = [(link.key[0], link.key[1]) for link in Nt.links if not np.isnan(link.observed_count)]

            link_capacities = np.array([link.bpr.k for link in Nt.links])[idx_nonas, np.newaxis].flatten()

            link_capacities_print = []
            for i in range(len(link_capacities)):
                if link_capacities[i] > 10000:
                    link_capacities_print.append(float('inf'))
                else:
                    link_capacities_print.append(link_capacities[i])

            # Travel time are originally in minutes
            link_travel_times = \
            np.array([str(np.round(link.get_traveltime_from_x(Nt.x_dict[link.key]), 2)) for link in Nt.links])[
                idx_nonas, np.newaxis]


            # This may raise a warning for OD connectors where the length is equal to 0

            if 'length' in Nt.links[0].Z_dict:
                # link = Nt.links[0]
                link_speed_av = \
                    np.array(
                        [np.round(60 * link.Z_dict['length'] / link.get_traveltime_from_x(Nt.x_dict[link.key]), 1) for link
                         in Nt.links if not np.isnan(link.observed_count)])
            else:
                link_speed_av = np.array([0] * len(link_travel_times.flatten()))


            current_error_by_link = error_by_link(x_bar, x_eq, show_nan=False)

            d_error = current_error_by_link-errors_by_link[-1]

            d_error_print = ["{0:.1E}".format(d_error_i[0]) for d_error_i in list(d_error)]

            # if iter < 1:
            #
            #     summary_table = pd.DataFrame(
            #         {'link_key': link_keys, 'capacity': link_capacities.flatten(), 'predicted': x_eq[idx_nonas].flatten(),
            #          'obs. count': x_bar[idx_nonas].flatten()
            #             , 'error': current_error_by_link.flatten()})
            #
            #     with pd.option_context('display.float_format', '{:0.1f}'.format):
            #         print('\n' + summary_table.to_string())
            #
            # else:


            summary_table = pd.DataFrame(
            {'link_key': link_keys
                , 'capacity': link_capacities_print
                , 'tt[min]': link_travel_times.flatten()
                , 'speed[mi/hr]': link_speed_av.flatten()
                , 'count': x_bar[idx_nonas].flatten()
                , 'predicted': x_eq[idx_nonas].flatten()
                , 'prev_error': errors_by_link[-1].flatten()
                ,  'error': current_error_by_link.flatten()
                , 'd_error': d_error_print
                })


            if silent_mode is False:
                with pd.option_context('display.float_format', '{:0.1f}'.format):
                    print('\n' + summary_table.to_string())

            errors_by_link.append(current_error_by_link)

            # print('\nKey, Prediction, observed count, error and capacities by link\n', s)

            # print('\nImprovement by link', abs(initial_error_by_link)-abs(error_by_link(x_bar, x_eq, show_nan=False))/abs(initial_error_by_link))


            # if plot_options['y'] != '':
            #     plot_bilevel_optimization(results, list(theta_current.keys()), bilevelopt_params['iters'], plot_options)

            # Select best theta based on minimum objective value



        print('\nSummary results of bilevel optimization')

        best_result_eq = results[best_iter]['equilibrium']

        # best_theta = dict(zip(theta_current.keys(), list(results[best_iter]['theta'].values())))
        # best_theta = theta_current

        print('best iter: ' + str(best_iter))
        # print('best theta: ' + str({key:round(val, 3) for key, val in best_theta.items()}))
        print('best theta: ' + str({key: "{0:.1E}".format(val) for key, val in best_theta.items()}))

        # print('best q: ' + repr(np.round(best_q.T,1)))

        print('best objective_value: ' + '{:,}'.format(round(best_objective)))

        if 'c' in theta_current.keys() and best_theta['c'] != 0:
            print('best ratio theta: ' + str(round(best_theta['tt'] / best_theta['c'], 4)))


        print('Final best objective improvement: ' + "{:.2%}". format(np.round(1 - best_objective / initial_objective, 4)))
        print('Final best objective improvement value: ' + '{:,}'.format(
            np.round(initial_objective - best_objective, 1)))

        print('Total time: ' + str(np.round(time.time() - t0_global, 1)) + ' [s]')
        # best_theta_dict = dict(zip([*k_Y, *k_Z], best_theta.flatten()))

        # print('Loss by link', error_by_link(x_bar, x_eq,show_nan = False))

        return best_q, best_theta, best_objective, best_result_eq, results


    # def plot_bilevel_optimization(results: {}, theta_keys, total_iters, plot_options):
    #
    #     iters = len(list(results.keys()))
    #
    #     columns_df = ['iter'] + ['theta_' + str(i) for i in list(theta_keys)] + ['error']
    #     # df_results = pd.DataFrame(columns=columns_df)
    #     df_results = pd.DataFrame(columns=columns_df)
    #
    #     # print(columns_df)
    #
    #     # Create pandas dataframe using each row of the dictionary returned by the bilevel methodç
    #
    #     # Create pandas dataframe with no refined solution
    #     for iter in np.arange(iters):
    #         df_results.loc[iter] = [iter] + list(results[iter]['theta']) + [results[iter]['objective']]
    #
    #     # Create additional variables
    #     df_results['vot'] = df_results['theta_tt'] / df_results['theta_c']
    #
    #
    #     if plot_options['y']== 'theta':
    #         plt.plot(df_results['iter'], df_results['vot'])
    #         plt.xticks(np.arange(0, total_iters, 1.0))
    #         plt.yticks(np.arange(0, 1, 0.2))
    #         plt.axhline(0.1667, linestyle='dashed')
    #
    #     if plot_options['y'] == 'objective':
    #         plt.plot(df_results['iter'], df_results['error'])
    #         plt.xticks(np.arange(0, total_iters, 1.0))
    #         plt.ylim(0, df_results['error'].iloc[0])
    #
    #         # plt.axhline(0.1667, linestyle='dashed')
    #
    #
    #
    #     # print(np.arange(0, total_iters + 1, 1.0))
    #     # ax[(0,0)].plot(df_results['iter'], df_results['vot'])
    #
    #     # plt.plot(df_results['iter'], df_results['error'])
    #
    #
    #     # plt.set
    #     # ax[(0,0)].set_xticks(df_results['iter'])
    #
    #     plt.show()

def solve_link_level_model(end_params: {}, Mt: {}, Ct: {}, Dt: {}, k_Y: list, Yt: {}, k_Z: list, Zt: {}, xt: {},
                           idx_links: {}, scale: {}, q0: np.array, theta0: {}
                           , lambda_hp: float, constraints_q0=[], norm_o=2, norm_r=1):
    '''

        Parameters are dictionaries where each key contains the value of a matrix or dictionary associated a specific instance of a network (e.g. day)

        :param theta0: avoid setting it to zero because it gives advantage to not regularized model as it starts from the base that the sparse parameters are zero.
        :param Mt:
        :param Dt:
        :param k_Y:
        :param Yt:
        :param k_Z:
        :param Zt:
        :param q:
        :param xt:
        :param idx_links:
        :param scale:
        :param lambda_hp:

        :return:
        :argument

        '''

    # if not isinstance(M, dict):
    #     D, M, Y, Z, q, x, idx_links = {1: D}, {1: M}, {1: Y}, {1: Z}, {1:q}, {1:x}, {1:idx_links}
    # i = 'N6'

    days = Mt.keys()
    n_days = len(list(days))

    for i in days:

        # Keep this here for efficiency instead of putting it in gap function
        Yt[i] = (get_matrix_from_dict_attrs_values({k_y: Yt[i][k_y] for k_y in k_Y}).T @ Dt[i]).T
        Zt[i] = (get_matrix_from_dict_attrs_values({k_z: Zt[i][k_z] for k_z in k_Z}).T @ Dt[i]).T

        if scale['mean'] or scale['std']:
            # Scaling by attribute
            Yt[i] = preprocessing.scale(Yt[i], with_mean=scale['mean'], with_std=scale['std'], axis=0)
            Zt[i] = preprocessing.scale(Zt[i], with_mean=scale['mean'], with_std=scale['std'], axis=1)

    # Starting values (q0, theta0)
    q = q0  # np.zeros(M[0].shape[0])
    theta = np.array(
        list(list(theta0.values())))  # np.array(list(list(theta0.values())))  # dict.fromkeys(theta_true, 1)

    # TODO: provide directly Subset of matrices with idx_links

    if end_params['theta'] and end_params['q']:

        # q = q0  # np.zeros(M[0].shape[0])

        # Constraints:

        range_theta = range(0, len(theta))
        range_q = range(len(theta), len(theta) + len(q))

        # Estimation

        # TODO: use lmfit or other non-linear least square optimizer to obtain t-statistic and more robust and faster performance
        # strt using method "Levenberg-Marquardt". See https://lmfit.github.io/lmfit-py/fitting.html#choosing-different-fitting-methods
        estimation_results = minimize(loss_link_level_model
                                      , x0=np.hstack([theta, q])
                                      , args=(
                range_theta, range_q, end_params, q, theta, lambda_hp, Dt, Mt, Ct, Yt, Zt, xt, idx_links, norm_o,
                norm_r)
                                      , constraints=constraints_q0
                                      # , method='BFGS', options={'gtol': 1e-6, 'disp': True}
                                      )

        return {'theta': dict(zip(theta0.keys(), np.round(estimation_results['x'][range_theta], 4))),
                'q': np.round(estimation_results['x'][range_q], 1),
                'gap': np.round(estimation_results['fun'] / n_days, 4)}


    elif end_params['theta']:

        range_theta = range(0, len(theta))
        range_q = range(len(theta), len(theta))

        estimation_results = minimize(loss_link_level_model
                                      , x0=theta
                                      , args=(
                range_theta, range_q, end_params, q, theta, lambda_hp, Dt, Mt, Ct, Yt, Zt, xt, idx_links, norm_o,
                norm_r)
                                      # , constraints=constraints_q0
                                      # , method='BFGS', options={'gtol': 1e-6, 'disp': True}
                                      )

        return {'theta': dict(zip(theta0.keys(), np.round(estimation_results['x'], 4))),
                'gap': np.round(estimation_results['fun'] / n_days, 4)}

    elif end_params['q']:

        range_theta = range(0, 0)
        range_q = range(0, len(q))

        estimation_results = minimize(loss_link_level_model
                                      , x0=q
                                      , args=(
                range_theta, range_q, end_params, q, theta, lambda_hp, Dt, Mt, Ct, Yt, Zt, xt, idx_links, norm_o,
                norm_r)
                                      , constraints=constraints_q0
                                      # , method='BFGS', options={'gtol': 1e-6, 'disp': True}

                                      )

        return {'q': np.round(estimation_results['x'], 1),
                'gap': np.round(estimation_results['fun'] / n_days, 4)}


def loss_SUE(o, x_obs, D, M, q, links: {}, paths: [], theta: np.array, Z_dict: {}, cp_solver='ECOS', k_Z=[]):
    # if theta['tt'] < 0:
    #     theta['tt'] = -theta['tt']

    # TODO: use an iterative method that does not require a non-positive theta
    for k in theta.keys():
        if theta[k] > 0:
            theta[k] = 0
            # theta[k] = -theta[k]
            # Z_dict[k] = dict(zip(Z_dict[k].keys(), list(-1*np.array(list(Z_dict[k].values())))))
    try:
        results_sue = equilibrium.sue_logit_fisk(q=q
                                                 , M=M
                                                 , D=D
                                                 , links=links
                                                 , paths=paths
                                                 , Z_dict=Z_dict
                                                 , k_Z=[]
                                                 , theta=theta
                                                 , cp_solver='ECOS'
                                                 # 'ECOS': it is faster and crashes sometimes. 'SCS' is slow
                                                 )

    except:
        return np.nan
    else:
        return np.linalg.norm(np.array(list(results_sue['x'].values())) - x_obs, o)


def multiday_estimation(N, end_params, theta0, q0, theta_true, remove_zeros_Q, n_days, randomness_multiday, k_Y, k_Z,
                        R_labels, Z_attrs_classes, bpr_classes, cutoff_paths, od_paths, fixed_effects, q_range,
                        N_multiday_old=None):
    N_multiday = {}
    N_multiday_new = {}
    results_multiday = {}
    # network_label = 'N1'
    for network_label in N.keys():

        results_multiday[network_label] = {}

        N_multiday_new[network_label] = networks.multiday_network(N=N[network_label], n_days=n_days, label=network_label
                                                                  , R_labels=R_labels
                                                                  , randomness=randomness_multiday
                                                                  , q_range=q_range, remove_zeros_Q=remove_zeros_Q
                                                                  , Z_attrs_classes=Z_attrs_classes,
                                                                  bpr_classes=bpr_classes
                                                                  , cutoff_paths=cutoff_paths, od_paths=od_paths
                                                                  , fixed_effects=fixed_effects)

        results_sue_multiday = {}

        valid_network = None

        # N_i = N_multiday_new[network_label][0]
        while valid_network is None:
            try:
                results_sue_multiday[network_label] = {i: equilibrium.sue_logit_fisk(
                    q=networks.denseQ(Q=N_i.Q, remove_zeros=remove_zeros_Q)
                    , M=N_i.M, D=N_i.D
                    , links=N_i.links_dict
                    , paths=N_i.paths
                    , Z_dict=N_i.Z_dict
                    , k_Z=[]
                    , theta=theta_true
                    , cp_solver='ECOS'  # 'ECOS': it is faster and crashes sometimes. 'SCS' is slow
                )
                    for i, N_i in N_multiday_new[network_label].items()}

            except:
                print('error' + str(network_label))
                # for i in N['validation'].keys():
                # exceptions['SUE']['validation'][i] += 1

                N_multiday_new[network_label] = networks.multiday_network(N=N[network_label], n_days=n_days,
                                                                          label=network_label
                                                                          , R_labels=R_labels
                                                                          , randomness=randomness_multiday
                                                                          , q_range=q_range,
                                                                          remove_zeros_Q=remove_zeros_Q
                                                                          , Z_attrs_classes=Z_attrs_classes,
                                                                          bpr_classes=bpr_classes
                                                                          , cutoff_paths=cutoff_paths,
                                                                          od_paths=od_paths,
                                                                          fixed_effects=fixed_effects)

                pass
            else:
                valid_network = True
                # Store travel time, link and path flows in Network objects
                for i, N_i in N_multiday_new[network_label].items():
                    N_i.set_Y_attr_links(y=results_sue_multiday[network_label][i]['tt_x'], label='tt')
                    N_i.x_dict = results_sue_multiday[network_label][i]['x']
                    N_i.f_dict = results_sue_multiday[network_label][i]['f']

        if N_multiday_old[network_label] is not None:
            N_multiday_list = [*list(N_multiday_old[network_label].values()),
                               *list(N_multiday_new[network_label].values())]
            N_multiday[network_label] = dict(zip(np.arange(0, len(N_multiday_list)), N_multiday_list, ))

        else:
            N_multiday = N_multiday_new
        theta_estimates_multiday = {}
        results_logit_sue_links = {}

        range_theta = range(0, len(theta0))
        range_params_q = range(len(theta0), len(theta0) + len(q0[network_label]))

        if end_params['q'] and not end_params['theta']:
            range_theta = range(0, 0)
            range_params_q = range(0, len(q0[network_label]))

        if end_params['theta'] and not end_params['q']:
            range_params_theta = range(0, len(theta0))
            range_params_q = range(0, 0)

        # Constraints
        constraints_q0 = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[range_params_q]) - np.sum(
                networks.denseQ(Q=N_multiday[network_label][0].Q, remove_zeros=remove_zeros_Q))}
            , {'type': 'ineq', 'fun': lambda x: x[range_params_q]}
        ]
        # constraints_q0  = [{'type':'ineq', 'fun': lambda x: x[range_q]}]
        # constraints_q0=[]

        results_logit_sue_links[network_label] = tai.estimation.solve_link_level_model(end_params=end_params,
                                                                                   Mt={i: N_i.M for i, N_i
                                                                                       in N_multiday[
                                                                                           network_label].items()},
                                                                                   Ct={
                                                                                       i: N_i.choice_set_matrix_from_M(
                                                                                           N_i.M) for i, N_i
                                                                                       in N_multiday[
                                                                                           network_label].items()},
                                                                                   Dt={i: N_i.D for i, N_i
                                                                                       in N_multiday[
                                                                                           network_label].items()},
                                                                                   k_Y=k_Y,
                                                                                   Yt={i: N_i.Y_dict for
                                                                                       i, N_i in N_multiday[
                                                                                           network_label].items()},
                                                                                   k_Z=k_Z,
                                                                                   Zt={i: N_i.Z_dict for
                                                                                       i, N_i in N_multiday[
                                                                                           network_label].items()},
                                                                                   xt={i: N_i.x for i, N_i
                                                                                       in N_multiday[
                                                                                           network_label].items()},
                                                                                   idx_links={
                                                                                       i: range(len(N_i.x))
                                                                                       for i, N_i in
                                                                                       N_multiday[
                                                                                           network_label].items()},
                                                                                   scale={'mean': False,
                                                                                          'std': False},
                                                                                   q0=q0[network_label],
                                                                                   theta0=theta0,
                                                                                   lambda_hp=0,
                                                                                   constraints_q0=constraints_q0,
                                                                                   norm_o=2, norm_r=1)
        if end_params['q']:
            results_multiday[network_label]['q'] = np.round(results_logit_sue_links[network_label]['q'], 1)

        if end_params['theta']:
            results_multiday[network_label]['theta'] = results_logit_sue_links[network_label]['theta']

            if results_logit_sue_links[network_label]['theta']['c'] == 0:
                results_multiday[network_label]['vot'] = np.nan
            else:
                results_multiday[network_label]['vot'] = results_logit_sue_links[network_label]['theta']['tt'] / \
                                                         results_logit_sue_links[network_label]['theta']['c']

        # print(np.round(results_logit_sue_links[network_label]['q'],1))
        # print(tai.network.denseQ(N_multiday[network_label][0].Q, remove_zeros=remove_zeros_Q))

        # print(results_logit_sue_links[network_label]['theta'])
        # print(theta_true)

        # print(results_logit_sue_links[network_label]['theta']['tt']/results_logit_sue_links[network_label]['theta']['c'])
        # print(theta_true['tt']/theta_true['c'])

    return results_multiday, N_multiday


def generate_choice_set_matrix_from_observed_path(G, observed_path):
    ''' Receive a list with the nodes in an observed path
        Return a matrix that encodes the choice sets at each node of the observed path
    '''

    nNodes = len(np.unique(list(G.nodes)))
    expanded_nodes = observed_path[:-1]  # All nodes except target node
    # next_nodes = dict(zip(optimal_path[:-1], optimal_path[1:]))
    connected_nodes = neighbors_path(G=G, path=observed_path)

    choice_set_matrix = np.zeros([nNodes, nNodes])

    for expanded_node in expanded_nodes:
        choice_set_matrix[expanded_node, connected_nodes[expanded_node]] = 1
        # nRow += 1

    return choice_set_matrix
    # return avail_matrix


def compute_edge_utility(G, theta: dict):
    attributes = list(theta.keys())

    utility = np.zeros(len(G.edges))

    for attribute in attributes:
        utility += theta[attribute] * np.array(list(nx.get_edge_attributes(G, attribute).values()))

    return dict(zip(G.edges, utility))


def compute_goal_dependent_heuristic_utility(G, observed_paths: dict, H_G: dict, theta_H: dict, K_H: dict):
    ''':argument '''

    utility_G = {}  # Node to node utility matrix which is dependent on the goal in the observed path

    for i, path in observed_paths.items():
        utility_G[i] = np.zeros([len(G.nodes()), len(G.nodes())])
        for k_H in K_H:
            utility_G[i] += theta_H[k_H] * np.array(H_G[k_H][i])

    return utility_G


def recursive_logit_estimation(C, X, K_X, y, H, K_H, g):
    '''

    :argument C: dictionary with matrices encoding the choice sets (set) associated to each (expanded) node in observed path i
    :argument X: dictionary with edge-to-edge matrices with network attributes values
    :argument K_X: subset of attributes from X chosen to fit discrete choice model
    :argument y: dictionary with list of chosen edges in path i
    :argument H: Nested dictionary (one per goal node) with edge-to-edge matrices of attribute values which are goal dependent (heuristic costs that varies with the goal)
    :argument K_H: dictionary with subset of attributes from H chosen to fit discrete choice model
    :argument g: dictionary with goal (destination) in path i
    '''

    # List with all attributes
    K = K_X + K_H

    # Dictionary with all attribute values together
    XH = X
    XH.update(H)

    # Estimated parameters to be optimized (learned)
    cp_theta = {i: cp.Variable(1) for i in K}

    # Number of paths
    n_paths = len(C.items())

    # Dictionary with list for nodes connected (alternatives) with each observed (expanded) node in each observed path i (key)
    nodes_alternatives = {i: [y_j[0] for y_j in y_i] for i, y_i in zip(range(len(y)), y.values())}

    # Dictionary with list of nodes expanded (chosen) in each observed path i (key)
    nodes_chosen = {i: [y_j[1] for y_j in y_i] for i, y_i in zip(range(len(y)), y.values())}

    # Nested dictionary of the attribute's (bottom level) values of the nodes(alternatives)
    # connected to each (expanded) node in the each observed path (top level)
    X_c = {}
    for i, c_matrix_path in C.items():
        X_c[i] = {attribute: get_list_attribute_vectors_choice_sets(c_matrix_path, X[attribute]) for attribute in K_X}

    # Nested dictionary for heuristic attributes which are goal dependent
    H_c = {}
    for i, c_matrix_path in C.items():
        H_c[i] = {attribute: get_list_attribute_vectors_choice_sets(c_matrix_path, H[attribute][i]) for attribute in
                  K_H}

    # Loglikelihood function obtained from iterating across choice sets
    Z = []

    for i in range(n_paths):

        # List storing the contribution from each choice (expansion) set to the likelihood
        Z_i = []

        for j, k in zip(nodes_alternatives[i], nodes_chosen[i]):
            Z_chosen_attr = []
            Z_logsum_attr = []

            for attribute in K_X:
                Z_chosen_attr.append(X[attribute][j, k] * cp_theta[attribute])
                Z_logsum_attr.append(X_c[i][attribute][j] * cp_theta[attribute])

            for attribute in K_H:
                Z_chosen_attr.append(H[attribute][i][j, k] * cp_theta[attribute])
                Z_logsum_attr.append(H_c[i][attribute][j] * cp_theta[attribute])

            Z_i.append(cp.sum(Z_chosen_attr) - cp.log_sum_exp(cp.sum(Z_logsum_attr)))

        Z.append(cp.sum(Z_i))

    cp_objective_logit = cp.Maximize(cp.sum(Z))

    cp_problem_logit = cp.Problem(cp_objective_logit, constraints=[])  # Excluding heuristic constraints

    cp_problem_logit.solve()

    return {key: val.value for key, val in cp_theta.items()}


def logit_path_predictions(G, observed_paths: dict, theta_logit: dict):
    G_copy = G.copy()

    # Edge attributes component in utility
    edge_utilities = compute_edge_utility(G, theta=theta_logit)
    edge_weights = {edge: -u for edge, u in edge_utilities.items()}

    # All utilities are positive so a-star can run properly.
    min_edge_weight = min(list(edge_weights.values()))
    if min_edge_weight < 0:
        edge_weights = {i: w + abs(min_edge_weight) for i, w in edge_weights.items()}

    nx.set_edge_attributes(G_copy, values=edge_weights, name='weights_prediction')

    # nx.get_edge_attributes(G_copy, 'utility_prediction')

    predicted_paths = {}

    for key, observed_path in observed_paths.items():
        predicted_paths[key] = nx.astar_path(G=G_copy, source=observed_path[0], target=observed_path[-1],
                                             weight='weights_prediction')

    predicted_paths_length = paths_lengths(G_copy, predicted_paths,
                                           'utility')  # Utility acts as a proxy of the path length (negative)

    return {'predicted_path': predicted_paths, 'length': predicted_paths_length}


def pastar_path_predictions(G, observed_paths: dict, H: dict, theta_X: dict, theta_H: dict, endogenous_heuristic=False):
    predicted_paths = {}
    predicted_paths_length = {}
    U_H = 0

    # Keep track of numbers of iterations made by astar for each path
    n_iterations = {}

    if len(theta_H) == 0:
        predictions = logit_path_predictions(G=G, observed_paths=observed_paths
                                             , theta_logit=theta_X)
        predicted_paths = predictions['predicted_path']
        predicted_paths_length = predictions['length']

    else:
        # Edge attributes component in utility
        edge_utilities = compute_edge_utility(G, theta=theta_X)
        edge_weights = {edge: -u for edge, u in edge_utilities.items()}

        # All utilities are positive so a-star can run properly.
        min_edge_weight = min(list(edge_weights.values()))
        if min_edge_weight < 0:
            edge_weights = {i: w + abs(min_edge_weight) for i, w in edge_weights.items()}

        # Heuristic components in utility
        K_H = list(H.keys())

        U_H = compute_goal_dependent_heuristic_utility(G, observed_paths=observed_paths, H_G=H, theta_H=theta_H,
                                                       K_H=K_H)

        for i, observed_path in observed_paths.items():

            G_copy = G.copy()
            edge_heuristic_weights = edge_weights.copy()
            n_iterations[i] = 0

            heuristic_weights_path = -U_H[i]
            min_heuristic = np.min(heuristic_weights_path)

            if min_heuristic < 0:
                heuristic_weights_path = heuristic_weights_path + abs(min_heuristic)

            # TODO: The loop below introduces correlation between alternatives which violates key assumption in Multinomial Logit Model
            # Correlation arises from the fact that multiple edges may have the same or similar heuristic cost.
            # As expected, there is a significant increse in accuracy

            if endogenous_heuristic is True:
                for edge in edge_heuristic_weights.keys():
                    edge_heuristic_weights[edge] += heuristic_weights_path[edge]

            nx.set_edge_attributes(G_copy, values=edge_heuristic_weights, name='weights_prediction')

            def astar_heuristic(a, b):
                # a is the neighboor and the heuristc_weights matrix have all rows equal and the column (:,a) gives distance o goal
                # print(heuristic_weights_path[(0,a)])
                n_iterations[i] += 1
                return heuristic_weights_path[(0, a)]

            predicted_paths[i] = nx.astar_path(G=G_copy, source=observed_path[0], target=observed_path[-1],
                                               weight='weights_prediction', heuristic=astar_heuristic)

        predicted_paths_length = paths_lengths(G, predicted_paths,
                                               'utility')  # Utility acts as a proxy of the path length (negative)

    return {'predicted_path': predicted_paths, 'length': predicted_paths_length, 'n_iterations_astar': n_iterations}


def accuracy_pastar_predictions(G, predicted_paths: dict, observed_paths: dict):
    # paths_lengths(G, paths = predicted_paths, attribute = 'utility')

    edge_acc = {}
    for key, observed_path in observed_paths.items():
        edge_acc[key] = sum(el in observed_path for el in predicted_paths[key]) / len(observed_path)

    path_acc = dict(zip(edge_acc.keys(), np.where(np.array(list(edge_acc.values())) < 1, 0, 1)))

    x_correct_edges = np.round(np.mean(np.array(list(edge_acc.values()))), 4)
    x_correct_paths = np.round(np.mean(np.array(list(path_acc.values()))), 4)

    # utility_diff = abs(sum(paths_lengths(G, paths= predicted_paths, attribute='utility').values())
    #                   -abs(sum(paths_lengths(G, paths= observed_paths, attribute='utility').values()))
    #                   )

    return {'acc_edges': x_correct_edges, 'acc_paths': x_correct_paths}
