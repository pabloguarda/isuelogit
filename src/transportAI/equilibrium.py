from __future__ import annotations

from typing import TYPE_CHECKING

from typing import Dict

import copy

if TYPE_CHECKING:
    from mytypes import Links, Matrix, ColumnVector, Links, LogitFeatures, LogitParameters, Paths, Options, Vector

import cvxpy as cp

import transportAI.printer as printer
# from transportAI import printer

from utils import blockPrinting
import estimation

from sklearn import preprocessing

from itertools import combinations

from paths import path_generation_nx

cp_solver = ''

# import transportAI.links
# import transportAI.networks
# import transportAI.estimation

from transportAI.networks import TNetwork

import math
import time
import heapq
import numpy as np
import os
from scipy import optimize
from scipy.stats import entropy



# TODO: Implement frankwolfe and MSE using as reference ta.py and comparing result with observed flows in files.





def sue_objective_function_fisk(Nt: TNetwork, x_dict: dict, f: Vector, theta: dict, k_Z: [], k_Y: [] = ['tt']):

    links_dict = Nt.links_dict
    x_vector = np.array(list(x_dict.values()))
    theta_Z = {attr:theta[attr] for attr in k_Z}

    # Objective function

    # Component for endogeonous attributes dependent on link flow
    bpr_integrals = [link.bpr.bpr_integral_x(x=x_dict[i]) for i, link in links_dict.items()]

    tt_utility_integral = theta[k_Y[0]] * np.sum(np.sum(bpr_integrals))

    # Component for exogenous attributes (independent on link flow)
    Z_utility_integral =0

    for attr in k_Z:
        Zx_vector = np.array(list(Nt.Z_dict[attr].values()))[:,np.newaxis]
        Z_utility_integral += theta_Z[attr]*Zx_vector.T.dot(x_vector)

    # Objective function in multiattribute problem
    utility_integral = tt_utility_integral + Z_utility_integral

    # entropy = cp.sum(cp.entr(cp.hstack(list(cp_f.values()))))
    entropy_function = np.sum(entropy(f))

    objective_function = utility_integral + entropy_function


    return float(objective_function)




def sue_logit_fisk(D: Matrix, M: Matrix, q: ColumnVector, links: Links, paths: Paths, theta: LogitParameters, Z_dict: {}, k_Z: LogitFeatures, k_Y: LogitFeatures = ['tt'], cp_solver='ECOS', feastol=1e-24) -> Dict[str, Dict]:
    """Computation of Stochastic User Equilibrium with Logit Assignment
    :arg q: Long vector of demand obtained from demand matrix between OD pairs
    :arg M: OD pair - link incidence matrix
    :arg D: path-link incidence matrix
    :arg links: dictionary containing the links in the network
    :arg Z: matrix with exogenous attributes values at each link that does not depend on the link flows
    :arg theta: vector with parameters measuring individual preferences for different route attributes
    :arg k_Z: subset of attributes from X chosen to perform assignment
    """

    # i = 'SiouxFalls'

    # q = tai.network.denseQ(Q=N['train'][i].Q, remove_zeros=remove_zeros_Q); M = N['train'][i].M; D = N['train'][i].D; links = N['train'][i].links_dict; paths = N['train'][i].paths; Z_dict = N['train'][i].Z_dict; k_Z = []; theta = theta_true[i]; cp_solver = 'ECOS'; feastol = 1e-24; k_Y = ['tt']

    # Subset of attributes
    if len(k_Z) == 0:
        k_Z = [k for k in theta.keys() if k not in k_Y]

    # #Solving issue with negative sign of parameters
    # for k in k_Z:
    #     if theta[k] > 0:
    #         theta[k] = -theta[k]
    #         Z_dict[k] = dict(zip(Z_dict[k].keys(), list(-1 * np.array(list(Z_dict[k].values())))))

    # Get Z matrix from Z dict
    Z_subdict = {k: Z_dict.get(k) for k in k_Z}
    Z = estimation.get_matrix_from_dict_attrs_values(W_dict=Z_subdict)

    # Decision variables
    # cp_f = {i:cp.Variable(nonneg=True) for i in range(D.shape[1])} # cp.Variable([nRoutes]) #TODO: Transform this in dictionary as with x
    # cp_x ={i:cp.Variable() for i, l_i in links.items()}  # cp.Variable([nLinks])

    cp_f = cp.Variable(D.shape[1], nonneg=True)
    # cp_f.value = np.ones(D.shape[1])*10000

    # cp_f.value[0] = q[0]
    # cp_f.value[2] = q[1]
    # cp_f.value[4] = q[2]
    # cp_f.value[0] = q[0:1]
    # np.sum(M[0:3,:]*cp_f.value,axis=1)
    # q[0:3]

    # # Set equality for link flows instead of additional constraints
    # cp_x = dict(zip(list(links.keys()), list(D * cp.hstack(list(cp_f.values())))))
    cp_x = D * cp_f

    # q

    # np.sum(cp_f.value)

    # np.sum(M[0:3, :],axis = 1)

    # Constraints (* is equivalent to numpy dot product)
    cp_constraints = []
    # cp_constraints += [M*cp.hstack(list(cp_f.values())) == q]
    cp_constraints += [M * cp.vstack(cp_f) == np.vstack(q)]
    # cp_constraints += [M * cp.vstack(cp_f) >= np.vstack(q)]
    # np.sum(M,axis = 0)

    # cp_constraints += [cp.sum(M[0:3,:] * cp.hstack(cp_f), axis=1) == q[0:3]]
    # cp_constraints += [M[0:3, :] * cp_f == q[0:3]]
    # cp_constraints += [M[0:1, :] * cp_f == q[0:1]]
    # cp_constraints += [D*cp.hstack(list(cp_f.values())) == cp.hstack(list(cp_x.values()))] #This constraint might be replaced directly into other constraints and objective function
    #
    # warm_start = np.zeros(D.shape[1])
    # warm_start[0] =
    # A = M[0:3, :]

    # type(q[0])

    # Check constraints
    # cp_constraints[0].violation()

    # [0].value

    # Parameters for cvxpy
    cp_theta = {}

    # Parameters that are dependent on link flow (only 'tt' so far)
    cp_theta['Y'] = {'tt': cp.Parameter(nonpos=True)}  # cp.Parameter(nonpos=True)
    # cp_theta['Y'] = {'tt': cp.Variable(pos=False)}  # cp.Parameter(nonpos=True)

    # Parameters for attributes not dependent on link flow (all except for 'tt't)
    cp_theta['Z'] = {k: cp.Parameter(nonpos=True) for k in k_Z if
                     k != 'tt'}  # nonpos=True is required to find unique equilibrium

    # Objective function

    # Component for endogeonous attributes dependent on link flow
    # bpr_integrals = [link.bpr.bpr_integral_x(x=cp_x[link.label]) for i,link in links.items()]
    bpr_integrals = [link.bpr.bpr_integral_x(x=cp_x[i]) for i, link in zip(range(0, len(list(links))), links.values())]
    tt_utility_integral = cp_theta['Y']['tt'] * cp.sum(cp.sum(bpr_integrals))

    # Component for attributes (independent on link flow)
    # Z_utility_integral = cp.sum(cp.multiply(Z*cp.hstack(list(cp_theta['Z'].values())),cp.hstack(list(cp_x.values()))))
    Z_utility_integral = cp.sum(
        cp.multiply(Z * cp.hstack(list(cp_theta['Z'].values())), cp.hstack(cp_x)))

    # Objective function for multiattribute problem
    utility_integral = tt_utility_integral + Z_utility_integral

    # entropy = cp.sum(cp.entr(cp.hstack(list(cp_f.values()))))
    entropy = cp.sum(cp.entr(cp_f))

    simple_obj = cp.Parameter()
    cp_objective = cp.Maximize(utility_integral + entropy)
    # cp_objective = cp.Maximize(entropy)

    # Problem
    # cp_problem = cp.Problem(cp_objective)
    cp_problem = cp.Problem(cp_objective, cp_constraints)

    # cp_problem.is_dcp()

    # Assign parameters values in objective function
    cp_theta['Y']['tt'].value = theta['tt']
    for k in k_Z:
        cp_theta['Z'][k].value = theta[k]

    # feastol = 100
    # Solve

    objective_value = None

    try:
        # objective_value = cp_problem.solve(solver=cp.ECOS, feastol=feastol)
        # cp_theta['Y']['tt'] = 0 #no congestion
        objective_value = cp_problem.solve(solver=cp.ECOS)

    except:  # SCS failed less often because of numerical problems
        objective_value = cp_problem.solve(solver=cp.SCS)
        # cp_theta['Y']['tt'] = -10
        # objective_value = cp_problem.solve(solver=cp.SCS, verbose = True, warm_start = True)

    # if cp_solver == 'ECOS':
    #     # objective_value = cp_problem.solve(solver = cp.ECOS, feastol = feastol)
    #     objective_value = cp_problem.solve(solver=cp.ECOS, feastol=feastol, verbose = True)
    #     # objective_value = cp_problem.solve(verbose=True)
    #
    # else:
    #     objective_value = cp_problem.solve(solver=cp_solver, feastol=feastol,
    #                                        verbose=True)  # #(solver = solver) #'ECOS' solver crashes with some problems

    # Results

    tt = {}
    # - Travel time by link
    # tt['x'] = {i:link.bpr.bpr_function_x(x=cp_x[link.label].value) for i,link in links.items()}
    tt['x'] = {j: link.bpr.bpr_function_x(x=cp_x.value[i]) for i, j, link in
               zip(range(0, len(list(links))), links.keys(), links.values())}

    # Link flows
    # x = {k:v.value for k,v in cp_x.values()}
    x = {k: v for k, v in zip(links.keys(), cp_x.value)}

    # cp_x.value

    # np.sum(np.array(list(x.values())))
    # np.sum(M* np.hstack(cp_f.value),axis = 0) == q[0]
    # q[-1]

    # [cons.violation() for cons in cp_constraints]

    # cp_constraints[0].violation()

    # Path flows
    f = {k: v for k, v in zip(range(len(cp_f.value)), cp_f.value)}
    # f = {k: v.value for k, v in cp_f.items()}

    # np.sum(np.array(list(cp_f.value)))
    #
    # np.sum(cp_f.value)
    # np.sum(q)

    # np.sum(M[0, :] * np.hstack(np.array(list(f.values())))) == q[0]

    # np.sum(list(x.values()))

    # # Todo: Flow by route. Require class path
    # f = {k:v.value for k,v in cp_x.items()}

    # results = dict({'f': cp_f.value, 'x': x, 'tt': tt})

    return {'x': x, 'f': f, 'tt_x': tt['x']}

def sue_logit_OD_estimation(D, M, q_obs, tt_obs, links: Links, paths: Paths, theta: LogitParameters, Z_dict: {}, x_obs: np.array, k_Z=[], cp_solver='ECOS') -> Dict[str, Dict]:
    """Computation of Stochastic User Equilibrium with Logit Assignment
    :arg q: Long vector of demand obtained from demand matrix between OD pairs
    :arg M: OD pair - link incidence matrix
    :arg D: path-link incidence matrix
    :arg links: dictionary containing the links in the network
    :arg Z: matrix with exogenous attributes values at each link that does not depend on the link flows
    :arg theta: vector with parameters measuring individual preferences for different route attributes
    :arg k_Z: subset of attributes from X chosen to perform assignment
    """
    # i = 'N6'
    # q_obs = tai.network.denseQ(Q = N['train'][i].Q, remove_zeros = remove_zeros_Q)
    # x_obs = np.array(list(results_sue['train'][i]['x'].values()))
    # tt_obs = np.array(list(results_sue['train'][i]['tt_x'].values()))
    # np.sum(x_obs)
    # np.sum(q_obs)
    # D, M, links, paths, Z_dict, theta, cp_solver, K_Z = N['train'][i].D,N['train'][i].M, N['train'][i].links_dict,N['train'][i].paths, N['train'][i].Z_dict,theta_true,'ECOS', []

    # Subset of attributes
    if len(k_Z) == 0:
        k_Z = [k for k in theta.keys() if k != 'tt']

    # Get Z matrix from Z dict
    Z_subdict = {k: Z_dict.get(k) for k in k_Z}
    Z = estimation.get_matrix_from_dict_attrs_values(W_dict=Z_subdict)

    # Decision variables
    cp_x = {i: cp.Variable() for i, l_i in links.items()}  # cp.Variable([nLinks])
    cp_f = {i: cp.Variable() for i in
            range(D.shape[1])}  # cp.Variable([nRoutes]) #TODO: Transform this in dictionary as with x
    # q  values
    cp_q = {i: cp.Variable(pos=True) for i in range(q_obs.shape[0])}  # cp.Variable(

    # Constraints
    cp_constraints = [M * cp.hstack(list(cp_f.values())) == cp.hstack(list(cp_q.values()))]
    # cp_constraints = [M * cp.hstack(list(cp_f.values())) == q]
    cp_constraints += [D * cp.hstack(list(cp_f.values())) == cp.hstack(list(cp_x.values()))]
    cp_constraints += [cp.sum(cp.hstack(list(cp_q.values()))) == np.sum(q_obs)]

    # Parameters for cvxpy
    cp_theta = {}
    cp_scale = cp.Variable(nonpos=False)
    # Parameters that are dependent on link flow (only 'tt' so far)
    cp_theta['Y'] = {'tt': cp.Parameter(nonpos=True)}  # cp.Parameter(nonpos=True)
    # cp_theta['Y'] = {'tt': cp.Variable(pos=False)}  # cp.Parameter(nonpos=True)

    # Parameters for attributes not dependent on link flow (all except for 'tt't)
    cp_theta['Z'] = {k: cp.Parameter(nonpos=True) for k in k_Z if
                     k != 'tt'}  # nonpos=True is required to find unique equilibrium
    # cp_theta['Z']['c'] = cp.Variable(nonpos=True)

    # Objective function

    # Component for endogeonous attributes dependent on link flow
    bpr_integrals = [link.bpr.bpr_integral_x(x=cp_x[link.key]) for i, link in links.items()]
    tt_utility_integral = cp_theta['Y']['tt'] * cp.sum(cp.sum(bpr_integrals))

    # Component for attributes (independent on link flow)
    Z_utility_integral = cp.sum(
        cp.multiply(Z * cp.hstack(list(cp_theta['Z'].values())), cp.hstack(list(cp_x.values()))))

    # x_obs = np.array(list(results_sue['train'][i]['x'].values()))
    # q_obs = q
    dq = 1 / 2 * cp.sum((cp.hstack(list(cp_q.values())) - q_obs) ** 2)
    cp_tt = [link.bpr.bpr_function_x(cp_x[link.key]) for i, link in links.items()]
    dt = cp.sum(cp.hstack(np.array(cp_tt) - tt_obs))  # -cp.sum(cp.multiply(np.array(cp_tt),np.array(list(q.values()))))
    dx = 1 / 2 * cp.sum((cp.hstack(list(cp_x.values())) - x_obs) ** 2)

    OD_term = 100 * dq + dx + 0 * dt

    # Objective function for multiattribute problem
    utility_integral = tt_utility_integral + Z_utility_integral

    entropy = cp.sum(cp.entr(cp.hstack(list(cp_f.values()))))
    cp_objective = cp.Maximize(utility_integral + entropy - OD_term)

    # Problem
    # cp_problem = cp.Problem(cp_objective)
    cp_problem = cp.Problem(cp_objective, cp_constraints)

    # cp_problem.is_dcp()

    # Assign parameters values in objective function
    cp_theta['Y']['tt'].value = theta['tt']
    for k in k_Z:
        cp_theta['Z'][k].value = theta[k]

    cp_theta['Z']['c'].value

    # Solve
    objective_value = cp_problem.solve(solver=cp_solver,
                                       verbose=True)  # (solver = solver) #'ECOS' solver crashes with some problems

    # Results

    tt = {}
    # - Travel time by link
    tt['x'] = {i: link.bpr.bpr_function_x(x=cp_x[link.key].value) for i, link in links.items()}

    # Link flows
    x = {k: v.value for k, v in cp_x.items()}

    # Path flows
    f = {k: v.value for k, v in cp_f.items()}

    # OD terms
    q = {k: v.value for k, v in cp_q.items()}

    # print(np.sum(q_obs))
    # print(np.sum(np.array(list(q.values()))))
    # print(np.sum(np.array(list(f.values()))))
    # print(np.sum(np.array(list(x.values()))))
    # print(np.sum(x_obs))
    # tt['x']
    # tt_obs
    # tt['x']
    # np.round(np.array(list(q.values())),1)
    # q_obs

    # # Todo: Flow by route. Require class path
    # f = {k:v.value for k,v in cp_x.items()}

    # results = dict({'f': cp_f.value, 'x': x, 'tt': tt})

    return {'x': x, 'f': f, 'tt_x': tt['x']}

def traffic_assignment_path_space(Nt, q, vf: ColumnVector):

    """

    :param vf: Vector of path utilities
    :param q: assume that q is a row vector but this should change
    """

    # Network matrices
    C = Nt.C
    # q = Nt.q

    assert q.shape[1] == 1, 'od vector is not a column vector'

    # TODO: store this matrix in the network object eventually to save computation

    # if len(q.shape) > 1 and q.shape[0] > 1:
    #     q = q.reshape((q.T.shape))

    # qM = q.dot(Nt.M)


    vf = estimation.v_normalization(v=vf, C=C)
    exp_vf = np.exp(vf)
    # v = np.exp(np.sum(V_Z, axis=1) + V_Y)

    # Denominator logit functions
    sum_exp_vf = C.dot(exp_vf)

    p_f = exp_vf / sum_exp_vf

    f = np.multiply(Nt.M.T.dot(q), p_f)

    return f,p_f

def traffic_assignment(Nt, q, vf: Vector):

    """ vf is assumed to be a column vector"""

    assert vf.shape[1] == 1, 'vector of path flows is not a column vector'

    f,p_f = traffic_assignment_path_space(Nt, q, vf)

    # if len(f.shape) > 1 and f.shape[1] > 1:
    #     f = f.reshape(f.T.shape)

    x = Nt.D.dot(f)

    return x,f,p_f

# @blockPrinting
def sue_logit_iterative(Nt: TNetwork, theta: {}, k_Y: LogitFeatures, k_Z: LogitFeatures, params: Options, x_current=None, q: ColumnVector = None, n_paths_column_generation: int = 0, k_path_set_selection = 0, silent_mode = False, standardization: {} = None):

    t0 = time.time()
    maxIter, accuracy = params['iters'], params['accuracy_eq']

    if 'method' in params.keys():
        method = params['method']
    else:
        method = 'msa'

    if 'k_path_set_selection' in params.keys() and params['k_path_set_selection'] > 0:
        k_path_set_selection = params['k_path_set_selection']
        dissimilarity_weight = params['dissimilarity_weight']
    else:
        k_path_set_selection = 0

    if q is None:
        q = Nt.q



    if not silent_mode:

        method_label = method

        if method_label == 'line_search':
            method_label = 'Frank-Wolfe'

        print("\nSUE via " + method_label +  " (max iters: " + str(int(maxIter)) + ')', '\n')

    # print(N.links[0].bpr.bpr_function_x(0))

    # Initialize link travel times with free flow travel time
    for link in Nt.links:
        link.set_traveltime_from_x(x=0)
        # print(link.traveltime)

    # Path travel times
    # for path in Nt.paths:
    #     print(path.traveltime)

    # MSA
    # x_current = None
    x_weighted = None
    gap = float("inf")
    gap_x = []
    lambdas_ls = []
    it = 0
    end_algorithm = False

    path_set_selection_done = False

    fisk_objective_functions = []

    # n_paths = Nt.setup_options['n_initial_paths']

    # TODO: Not sure if I should do column generation at every MSA iteration, which is costly
    if n_paths_column_generation > 0:

        # printer.blockPrint()
        sue_column_generation(Nt, theta=theta, n_paths=n_paths_column_generation)
        # printer.enablePrint()
        # print('printing')
        # n_paths += 1

    # Path utilities associated to exogenous attributes (do not change across iterations if path set is fixed). This operation is expensive when there are many paths and its complexity depend on the number of exogenous attributes

    if len(k_Z) > 0:
        listZ = []
        for path in Nt.paths:
            # print(path.Z_dict)

            listZ_path = []
            for key in k_Z:
                if key in path.Z_dict.keys():
                    listZ_path.append(float(path.Z_dict[key]) * theta[key])

            listZ.append(listZ_path)

        # Total utility of each path
        vf_Z = np.sum(np.asarray(listZ), axis=1)[:, np.newaxis]

        # if standardization is not None:
        #     vf_Z = preprocessing.scale(vf_Z, with_mean=standardization['mean'], with_std=standardization['sd'], axis=0)

    else:
        vf_Z = 0

    while end_algorithm is False:

        if not silent_mode and maxIter > 0:
            printer.printProgressBar(it, maxIter, prefix='Progress:', suffix='', length=20)

        if it >= 1 or x_current is None:

            # Path utilities associated to endogenous attributes (travel time) is the only that changes over iterations

            vf_Y = np.array([path.traveltime * theta['tt'] for path in Nt.paths])[:,np.newaxis]

            # if standardization is not None:
            #     vf_Y = preprocessing.scale(vf_Y, with_mean=standardization['mean'], with_std=standardization['sd'], axis=0)

            # Traffic assignment
            x,f,p_f = traffic_assignment(Nt, q = q, vf = vf_Y + vf_Z)

            x_current = copy.deepcopy(x)
            f_current = copy.deepcopy(f)

        if it == 0:
            x_weighted = copy.deepcopy(x_current)
            f_weighted = copy.deepcopy(f_current)

        if it >= 1:

            if method == 'line_search':

                if it>1:
                    iters_ls = params['iters_ls']

                    x_weighted_dict = dict(zip(list(Nt.links_dict.keys()), x_weighted))
                    x_current_dict = dict(zip(list(Nt.links_dict.keys()), x_current))

                    lambda_ls, xmin_ls, fmin_ls, objectivemin_ls = sue_line_search(Nt = Nt, theta = theta, k_Z = k_Z
                        , iters = iters_ls, method = 'grid_search'
                        , x1_dict = x_weighted_dict, x2_dict = x_current_dict
                        ,f1 = f_weighted, f2 = f_current)


                    f_weighted = fmin_ls
                    x_weighted = xmin_ls

                # x_dict = dict(zip(list(Nt.links_dict.keys()), x_weighted))

                    lambdas_ls.append(lambda_ls)

                # print('fisk objective', sue_objective_function_fisk(Nt=Nt, x_dict=x_dict , f=f_weighted, theta=theta, k_Z=k_Z))

            if method == 'msa':
                x_weighted = (1 / (it + 1)) * x_current + (1 - 1 / (it + 1)) * x_weighted
                f_weighted = f_current

            #evaluate sue objective function
            x_dict = dict(zip(list(Nt.links_dict.keys()), list(x_weighted.flatten())))

            fisk_objective_function = sue_objective_function_fisk(Nt=Nt, x_dict=x_dict, f=f_weighted, theta=theta, k_Z=k_Z)

            fisk_objective_functions.append(fisk_objective_function)

            if len(fisk_objective_functions)>2:
                max_fisk_objective_functions = np.max(fisk_objective_functions[:-1])
                #TODO: change definition for equilibrium gap (boyles)

                # change = (x_weighted - x_current)
                # gap = round(np.linalg.norm(
                #     np.divide(change, x_weighted, out=np.zeros_like(change), where=x_weighted != 0)), 2)

                change = (fisk_objective_function - max_fisk_objective_functions)

                gap = np.linalg.norm(
                    np.divide(change, max_fisk_objective_functions, out=np.zeros_like(change), where=fisk_objective_function != 0))

                gap_x.append(gap)
            # gap_x.append(np.sum(np.abs(x_msa - x_current) / len(x_current)))

        # Create dictionary with path probabilities
        pf_dict = {str(path.get_nodes_labels()): p_f[i] for i, path in zip(np.arange(len(p_f)), Nt.paths)}

        if k_path_set_selection > 0 and path_set_selection_done is False:



            # print('\nPerforming path selection:', 'dissimilarity_weight=' + str(dissimilarity_weight), 'k=' + str(k_path_set_selection))

            total_paths = 0

            # Combinatorial problem which works well for small path set
            for od, paths in Nt.paths_od.items():

                if len(paths) > k_path_set_selection:
                    Nt.paths_od[od], best_score = path_set_selection(paths = paths,pf_dict = pf_dict, k = k_path_set_selection, dissimilarity_weight=dissimilarity_weight)

                total_paths += len(Nt.paths_od[od])

            path_set_selection_done = True

            print('\nPath selection with k=' + str(k_path_set_selection), '(total paths: ' + str(total_paths) +')' )

        # Update travel times in links (and thus in paths)
        for link, j in zip(Nt.links, range(len(x_weighted))):
            link.set_traveltime_from_x(x=x_weighted[j])

        it += 1



        # print('fisk objective', sue_objective_function_fisk(Nt = Nt, x_dict = x_dict,f = f,theta = theta,k_Z = k_Z))


            # n_paths += 1

            # print(maxIter)

        if it > maxIter:
            end_algorithm = True

        elif gap < accuracy and it > 2:
            end_algorithm = True

    if gap > accuracy:
        if not silent_mode:
            print("Assignment did not converge with the desired gap")
        # print("Traffic assignment did not converge with the desired gap and max iterations are reached")

    # if end_algorithm:
    #     printer.printProgressBar(maxIter, maxIter, prefix='Progress:', suffix='',
    #                              length=20)

    # print("total iterations: ", str(it) + )
    if not silent_mode and maxIter > 0:
        # print('current theta: ',str({key: "{0:.1E}".format(val) for key, val in theta_current.items()}))

        print('gaps:', ["{0:.0E}".format(val) for val in gap_x])
        # print('gaps:', np.round(gap_x,2))

        if method == 'line_search':
            print('lambdas:', ["{0:.2E}".format(val) for val in lambdas_ls])

        print('initial sue fisk objective: ' + '{:,}'.format(
            round(fisk_objective_functions[0],2)))
        print('final sue fisk objective: ' + '{:,}'.format(
            round(fisk_objective_functions[-1],2)))

        print('iters: ' + str(it-1))

        print('Time: ' + str(round(time.time() - t0, 1)) + ' [s]' + '. Final gap: ' + "{0:.0E}".format(gap) + '. Acc. bound: ' + "{0:.0E}".format(accuracy)  )


    links_keys = list(Nt.links_dict.keys())

    x_final = dict(zip(links_keys, list(x_weighted.flatten())))

    if 'uncongested_mode' in params.keys() and params['uncongested_mode']:
        for link in Nt.links:
            link.set_traveltime_from_x(x=0)


    # print('iteration :' + str(it))

    # # Store link flows of current iteration
    # if i < maxIter-1:
    #     x_previous = x_current

    # print(np.round(x_iteration,2))
    # print(np.round(np.array([link.traveltime for link in N.links]),1))



    return {'x': x_final
        , 'tt_x': dict(zip(links_keys, [link.traveltime for link in Nt.links]))
        , 'gap_x': gap_x  # np.sum(np.abs(x_msa - x_current)/len(x_current))
        , 'p_f': p_f
            }

def sue_line_search(iters, method, Nt, x1_dict:dict, x2_dict:dict, f1: Vector, f2: Vector
                    , theta: dict, k_Z: [], k_Y: [] = ['tt']):


    # Under the assumption the best lambda result from solving a convex problem, we can use the bisection method


    # x1_vector = np.array(list(x1_dict.values()))
    # x2_vector = np.array(list(x2_dict.values()))

    # Bisection:

    if method == 'bisection':

        raise NotImplementedError

        objective_1 = sue_objective_function_fisk(Nt=Nt, x_dict=x2_dict, f=f1, theta=theta, k_Z=k_Z)
        objective_2 = sue_objective_function_fisk(Nt=Nt, x_dict=x2_dict, f=f2, theta=theta, k_Z=k_Z)

        left_lambda = 0
        right_lambda = 1

        for iter in iterations:

            mid_lambda = 0.5*(left_lambda + right_lambda)
            mid_f = mid_lambda * f1 + (1 - mid_lambda) * f2



            left_f = left_lambda * f1 + (1 - lambda_ls) * f2
            left_x = Nt.D.dot(left_f)
            xnew_dict = dict(zip(list(x1_dict.keys()), xnew))

            left_objective = sue_objective_function_fisk(Nt=Nt, x_dict=x1_dict, f=f1, theta=theta, k_Z=k_Z)






    if method == 'grid_search':
        objective_opt = float('-inf')
        lambda_opt = None
        xopt = None
        fopt = None

        grid_lambda = np.linspace(0,1, iters)

        for lambda_ls in grid_lambda:
            fnew = lambda_ls * f1 + (1 - lambda_ls) * f2
            # TODO: It may make sense to just do the convex combination in link space
            xnew = Nt.D.dot(fnew)
            xnew_dict = dict(zip(list(x1_dict.keys()), xnew))

            objective_new = sue_objective_function_fisk(Nt=Nt, x_dict=xnew_dict, f=fnew, theta=theta, k_Z=k_Z)

            if objective_new >= objective_opt:
                objective_opt = objective_new
                xopt = xnew
                fopt = fnew
                lambda_opt = lambda_ls

    return lambda_opt, xopt, fopt,objective_opt

def path_set_selection(paths, pf_dict, k, dissimilarity_weight):

    # https://www.geeksforgeeks.org/python-percentage-similarity-of-lists/
    # https://stackoverflow.com/questions/41680388/how-do-i-iterate-through-combinations-of-a-list

    best_score = -float('inf')

    for path_set in combinations(paths, k): # 2 for pairs, 3 for triplets, etc

        total_probability = 0
        total_similarity = 0

        for path in path_set:
            total_probability = pf_dict[str(path.get_nodes_labels())]

        for paths_pair in combinations(paths, 2):

            path1_sequence = paths_pair[0].get_nodes_labels()
            path2_sequence = paths_pair[1].get_nodes_labels()

            similarity = len(set(path1_sequence) & set(path2_sequence)) / float(len(set(path1_sequence) | set(path2_sequence)))
            total_similarity += similarity

        average_dissimilarity = 1-total_similarity/len(path_set)
        average_probability = total_probability/len(path_set)

        score = dissimilarity_weight*average_dissimilarity + (1-dissimilarity_weight)*average_probability

        if score >= best_score:
            best_score = score
            best_path_set = path_set
            best_average_probability = average_probability
            best_average_dissimilarity = average_dissimilarity



    return list(best_path_set), best_score


def sue_column_generation(Nt, theta, n_paths, silent_mode = False, path_selection= True ) -> None:

    ods_coverage = Nt.setup_options['ods_coverage_column_generation']

    print('Column generation:', str(n_paths) + ' paths per od, '+ "{:.1%}". format(ods_coverage) + ' od coverage')

    t0 = time.time()

    cutoff_paths = Nt.setup_options['cutoff_paths']

    # 0.Initialization

    # - Perform traffic assignment by computing shortest paths using the current estimate of the logit parameters and free flow travel times (1 iteration of MSA)

    # 1. Restricted master problem phase

    # Loop:

    # i) Perform traffic assignment again with new travel times

    # ii) Line search to find the minimum objective function for SUE

    #TODO: Function to compute SUE in multiattribute setting

    # 2. Column generation phase

    # i) Augment the path set used in 1, by for instance, adding the next shortest path

    # * I may do the augmentation only once for efficiency but define a factor to control for this. The augmentation may be based on the shortest path

    # * In the algorithm proposed by Damberg et al. (1996)  it is, however, possible to avoid generating flows on overlapping routes by deleting (or suitably modifying) any route generated that overlaps with a previously generated one more than a maximal allowed measure of overlapping; depending on the overlap measure, this may be easily performed by augmenting the route generation phase with a suitable check.

    #Path generation

    # theta['tt'] = 1

    # Matrix with link utilities
    Nt.V = Nt.generate_V(A = Nt.A, links = Nt.links, theta = theta)

    # print(Nt.V)

    # edge_utilities = Nt.generate_edges_weights_dict_from_utility_matrix(V = Nt.V)

    # edge_utilities = Nt.generate_edges_weights_dict_from_utility_matrix(V=np.zeros(Nt.V.shape))

    # Key to have the minus sign so we look the route that lead to the lowest disutility
    edge_utilities = Nt.generate_edges_weights_dict_from_utility_matrix(V=Nt.V)


    # Sample part of the ods according to the coverage set for column generation


    if ods_coverage > 0 and ods_coverage <= 1:
        n_ods_sample = int(np.floor(ods_coverage*len(Nt.ods)))
        ods_sample = [Nt.ods[idx] for idx in np.random.choice(np.arange(len(Nt.ods)), n_ods_sample, replace=False)]

    else:
        ods_sample = Nt.ods

    paths, paths_od = path_generation_nx(A=Nt.A
                                               , ods= ods_sample
                                               , links=Nt.links_dict
                                               , cutoff=cutoff_paths
                                               , n_paths=n_paths
                                               , edge_weights = edge_utilities
                                               , silent_mode = True
                                               )

    printer.blockPrint()


    # See if new paths were found so they are added into the existing path set
    paths_added = 0
    n_ods_added = 0

    for od, paths in paths_od.items():

        some_path_added_od = False

        existing_paths_keys = [path.get_nodes_labels() for path in Nt.paths_od[od]]
        for path in paths_od[od]:
            if path.get_nodes_labels() not in existing_paths_keys:
                Nt.paths_od[od].append(path)
                # Nt.paths.append(path)
                paths_added += 1
                some_path_added_od = True

        if some_path_added_od:
            n_ods_added += 1

    Nt.paths = Nt.get_paths_from_paths_od(Nt.paths_od)


    Nt.M = Nt.generate_M(paths_od=Nt.paths_od)
    Nt.D = Nt.generate_D(paths_od=Nt.paths_od, links=Nt.links)
    Nt.C = estimation.choice_set_matrix_from_M(Nt.M)

    printer.enablePrint()

    # print("Total number of links among paths: ", np.sum(Nt.D))
    print(str(paths_added) + ' paths added/replaced among ' + str(n_ods_added) + ' ods (total paths: ' + str(len(Nt.paths)) + ')')
    # print('- Computation time: ' + str(np.round(time.time() - t0, 1)) + ' [s]')


def sue_logit_dial(root, subfolder, prefix_filename, options, Z_dict, k_Z, theta={'tt': 1}):
    # -*- coding: utf-8 -*-
    """
    Created on Sun May 28 21:09:46 2017

    @author: Pramesh Kumar

    Source: https://github.com/prameshk/Traffic-Assignment
    """

    # inputLocation = "Sioux Falls network/"

    inputLocation = root + subfolder

    od_filename = [_ for _ in os.listdir(os.path.join(root, subfolder)) if 'trips' in _ and _.endswith('tntp')]

    # prefix_filenames = od_filename[0].partition('_')[0]

    def readDemand():
        inFile = open(inputLocation + '/' + prefix_filename + "_demand.dat")
        tmpIn = inFile.readline().strip().split("\t")
        for x in inFile:
            tmpIn = x.strip().split("\t")
            tripSet[tmpIn[0], tmpIn[1]] = Demand(tmpIn)
            if tmpIn[0] not in zoneSet:
                zoneSet[tmpIn[0]] = Zone([tmpIn[0]])
            if tmpIn[1] not in zoneSet:
                zoneSet[tmpIn[1]] = Zone([tmpIn[1]])
            if tmpIn[1] not in zoneSet[tmpIn[0]].destList:
                zoneSet[tmpIn[0]].destList.append(tmpIn[1])

        inFile.close()
        print(len(tripSet), "OD pairs")
        print(len(zoneSet), "zones")

    def readNetwork():
        inFile = open(inputLocation + '/' + prefix_filename + "_network.dat")
        tmpIn = inFile.readline().strip().split("\t")
        for x in inFile:
            tmpIn = x.strip().split("\t")
            linkSet[tmpIn[0], tmpIn[1]] = Link(tmpIn)
            if tmpIn[0] not in nodeSet:
                nodeSet[tmpIn[0]] = Node(tmpIn[0])
            if tmpIn[1] not in nodeSet:
                nodeSet[tmpIn[1]] = Node(tmpIn[1])
            if tmpIn[1] not in nodeSet[tmpIn[0]].outLinks:
                nodeSet[tmpIn[0]].outLinks.append(tmpIn[1])
            if tmpIn[0] not in nodeSet[tmpIn[1]].inLinks:
                nodeSet[tmpIn[1]].inLinks.append(tmpIn[0])

        inFile.close()
        print(len(nodeSet), "nodes")
        print(len(linkSet), "links")

    def read_Z_links():

        for k_z in Z_dict.keys():
            for link in linkSet:
                linkSet[link].Z_dict[k_z] = Z_dict[k_z][
                    (int(linkSet[link].tailNode) - 1, int(linkSet[link].headNode) - 1, '0')]

    class Zone:
        def __init__(self, _tmpIn):
            self.zoneId = _tmpIn[0]
            self.lat = 0
            self.lon = 0
            self.destList = []

    class Node:
        '''
        This class has attributes associated with any node
        '''

        def __init__(self, _tmpIn):
            self.Id = _tmpIn[0]
            self.lat = 0
            self.lon = 0
            self.outLinks = []
            self.inLinks = []
            self.label = float("inf")
            self.pred = ""
            self.inDegree = 0
            self.outDegree = 0
            self.order = 0  # Topological order
            self.wi = 0.0  # Weight of the node in Dial's algorithm
            self.xi = 0.0  # Toal flow crossing through this node in Dial's algorithm

    class Link:
        '''
        This class has attributes associated with any link
        '''

        def __init__(self, _tmpIn):
            self.tailNode = _tmpIn[0]
            self.headNode = _tmpIn[1]
            self.capacity = float(_tmpIn[2])  # veh per hour
            self.length = float(_tmpIn[3])  # Length
            self.fft = float(_tmpIn[4])  # Free flow travel time (min)
            self.beta = float(_tmpIn[6])
            self.alpha = float(_tmpIn[5])
            self.speedLimit = float(_tmpIn[7])
            # self.toll = float(_tmpIn[9])
            # self.linkType = float(_tmpIn[10])
            self.flow = 0.0
            self.cost = float(_tmpIn[
                                  4])  # float(_tmpIn[4])*(1 + float(_tmpIn[5])*math.pow((float(_tmpIn[7])/float(_tmpIn[2])), float(_tmpIn[6])))
            self.traveltime = 0.0
            self.logLike = 0.0
            self.reasonable = True  # This is for Dial's stochastic loading
            self.wij = 0.0  # Weight in the Dial's algorithm
            self.xij = 0.0  # Total flow on the link for Dial's algorithm

            # Extra link attributes
            self.Z_dict = {}

    class Demand:
        def __init__(self, _tmpIn):
            self.fromZone = _tmpIn[0]
            self.toNode = _tmpIn[1]
            self.demand = float(_tmpIn[2])

    ###########################################################################################################################

    readStart = time.time()

    tripSet = {}
    zoneSet = {}
    linkSet = {}
    nodeSet = {}

    readDemand()
    readNetwork()
    read_Z_links()

    originZones = set([k[0] for k in tripSet])
    print("Reading the network data took", round(time.time() - readStart, 2), "secs")

    #############################################################################################################################
    #############################################################################################################################

    def DijkstraHeap(origin):
        '''
        Calcualtes shortest path from an origin to all other destinations.
        The labels and preds are stored in node instances.
        '''
        for n in nodeSet:
            nodeSet[n].key = float("inf")
            nodeSet[n].pred = ""
        nodeSet[origin].key = 0.0
        nodeSet[origin].pred = "NA"
        SE = [(0, origin)]
        while SE:
            currentNode = heapq.heappop(SE)[1]
            currentLabel = nodeSet[currentNode].key
            for toNode in nodeSet[currentNode].outLinks:
                link = (currentNode, toNode)
                newNode = toNode
                newPred = currentNode
                existingLabel = nodeSet[newNode].key
                newLabel = currentLabel + linkSet[link].cost
                if newLabel < existingLabel:
                    heapq.heappush(SE, (newLabel, newNode))
                    nodeSet[newNode].key = newLabel
                    nodeSet[newNode].pred = newPred

    def updateCost():

        # TODO: Replace cost by generalized cost function

        for l in linkSet:
            linkSet[l].cost = -linkSet[l].traveltime * theta['tt']
            for k_z in k_Z:
                linkSet[l].cost += -linkSet[l].Z_dict[k_z] * theta[k_z]

    def updateTravelTime():
        '''
        This method updates the travel time on the links with the current flow
        '''
        for l in linkSet:
            linkSet[l].traveltime = linkSet[l].fft * (
                    1 + linkSet[l].alpha * math.pow((linkSet[l].flow * 1.0 / linkSet[l].capacity), linkSet[l].beta))

        updateCost()

    from scipy.optimize import fsolve

    def findAlpha(x_bar):
        '''
        This uses unconstrained optimization to calculate the optimal step size required
        for Frank-Wolfe Algorithm

        ******************* Need to be revised: Currently not working.**********************************************
        '''

        # alpha = 0.0

        def df(alpha):
            sum_derivative = 0  ## this line is the derivative of the objective function.
            for l in linkSet:
                tmpFlow = (linkSet[l].flow + alpha * (x_bar[l] - linkSet[l].flow))
                # print("tmpFlow", tmpFlow)
                tmpCost = linkSet[l].fft * (
                        1 + linkSet[l].alpha * math.pow((tmpFlow * 1.0 / linkSet[l].capacity), linkSet[l].beta))
                sum_derivative = sum_derivative + (x_bar[l] - linkSet[l].flow) * tmpCost
            return sum_derivative

        sol = optimize.root(df, np.array([0.1]))
        sol2 = fsolve(df, np.array([0.1]))
        # print(sol.x[0], sol2[0])
        return max(0.1, min(1, sol2[0]))
        '''
        def int(alpha):
            tmpSum = 0
            for l in linkSet:
                tmpFlow = (linkSet[l].flow + alpha*(x_bar[l] - linkSet[l].flow))
                tmpSum = tmpSum + linkSet[l].fft*(tmpFlow + linkSet[l].alpha * (math.pow(tmpFlow, 5) / math.pow(linkSet[l].capacity, 4)))
            return tmpSum

        bounds = ((0, 1),)
        init = np.array([0.7])
        sol = optimize.minimize(int, x0=init, method='SLSQP', bounds = bounds)

        print(sol.x, sol.success)
        if sol.success == True:
            return sol.x[0]#max(0, min(1, sol[0]))
        else:
            return 0.2
        '''

    def tracePreds(dest):
        '''
        This method traverses predecessor nodes in order to create a shortest path
        '''
        prevNode = nodeSet[dest].pred
        spLinks = []
        while nodeSet[dest].pred != "NA":
            spLinks.append((prevNode, dest))
            dest = prevNode
            prevNode = nodeSet[dest].pred
        return spLinks

    def loadAON():
        '''
        This method produces auxiliary flows for all or nothing loading.
        '''
        x_bar = {l: 0.0 for l in linkSet}
        SPTT = 0.0
        for r in originZones:
            DijkstraHeap(r)
            for s in zoneSet[r].destList:
                try:
                    dem = tripSet[r, s].demand
                except KeyError:
                    dem = 0.0
                SPTT = SPTT + nodeSet[s].key * dem
                if r != s:
                    for spLink in tracePreds(s):
                        x_bar[spLink] = x_bar[spLink] + dem
        return SPTT, x_bar

    def findReasonableLinks():
        '''Reasonable criterion is defined as routes that are farther from the origin.
        Label attribute of nodes corresponde to the shortest path from origin to node '''
        for l in linkSet:
            if nodeSet[l[1]].key > nodeSet[l[0]].key:
                linkSet[l].reasonable = True
            else:
                linkSet[l].reasonable = False

    def computeLogLikelihood():
        '''2
        This method computes link likelihood for the Dial's algorithm
        '''
        for l in linkSet:
            if linkSet[l].reasonable == True:  # If reasonable link
                linkSet[l].logLike = math.exp(nodeSet[l[1]].key - nodeSet[l[0]].key - linkSet[
                    l].cost)  # Label has shortest path value to node

    def topologicalOrdering():
        '''
        * Assigns topological order to the nodes based on the inDegree of the node
        * Note that it only considers reasonable links, otherwise graph will be acyclic
        '''
        for e in linkSet:
            if linkSet[e].reasonable == True:
                nodeSet[e[1]].inDegree = nodeSet[e[1]].inDegree + 1
        order = 0
        SEL = [k for k in nodeSet if nodeSet[k].inDegree == 0]
        while SEL:
            i = SEL.pop(0)
            order = order + 1
            nodeSet[i].order = order
            for j in nodeSet[i].outLinks:
                if linkSet[i, j].reasonable == True:
                    nodeSet[j].inDegree = nodeSet[j].inDegree - 1
                    if nodeSet[j].inDegree == 0:
                        SEL.append(j)
        if order < len(nodeSet):
            print("the network has cycle(s)")

    def resetDialAttributes():
        for n in nodeSet:
            nodeSet[n].inDegree = 0
            nodeSet[n].outDegree = 0
            nodeSet[n].order = 0
            nodeSet[n].wi = 0.0
            nodeSet[n].xi = 0.0
        for l in linkSet:
            linkSet[l].logLike = 0.0
            linkSet[l].reasonable = True
            linkSet[l].wij = 0.0
            linkSet[l].xij = 0.0

    def DialLoad():
        '''
        This method runs the Dial's algorithm and prepare a stochastic loading.
        '''
        resetDialAttributes()
        x_bar = {l: 0.0 for l in linkSet}
        for r in originZones:
            DijkstraHeap(r)
            findReasonableLinks()
            topologicalOrdering()
            computeLogLikelihood()

            '''
            Assigning weights to nodes and links
            '''
            order = 1
            while (order <= len(nodeSet)):
                i = [k for k in nodeSet if nodeSet[k].order == order][0]  # Node with order no equal to current order
                if order == 1:
                    nodeSet[i].wi = 1.0
                else:
                    nodeSet[i].wi = sum(
                        [linkSet[k, i].wij for k in nodeSet[i].inLinks if linkSet[k, i].reasonable == True])
                for j in nodeSet[i].outLinks:
                    if linkSet[i, j].reasonable == True:
                        linkSet[i, j].wij = nodeSet[i].wi * linkSet[i, j].logLike
                order = order + 1
            '''
            Assigning load to nodes and links
            '''
            order = len(nodeSet)  # The loading works in reverse direction
            while (order >= 1):
                j = [k for k in nodeSet if nodeSet[k].order == order][0]  # Node with order no equal to current order
                try:
                    dem = tripSet[r, j].demand
                except KeyError:
                    dem = 0.0

                nodeSet[j].xj = dem + sum(
                    [linkSet[j, k].xij for k in nodeSet[j].outLinks if linkSet[j, k].reasonable == True])
                for i in nodeSet[j].inLinks:
                    if linkSet[i, j].reasonable == True:
                        # if dem >0:
                        #     # If a node is not accesible but there is no demand, then this does not have to fail (wi = 0 in denominator
                        linkSet[i, j].xij = nodeSet[j].xj * (linkSet[i, j].wij / nodeSet[j].wi)

                order = order - 1
            for l in linkSet:
                if linkSet[l].reasonable == True:
                    x_bar[l] = x_bar[l] + linkSet[l].xij

        return x_bar

    def assignment(loading, algorithm, accuracy=0.01, maxIter=100):
        '''
        * Performs traffic assignment
        * Type is either deterministic or stochastic
        * Algorithm can be MSA or FW
        * Accuracy to be given for convergence
        * maxIter to stop if not converged
        '''
        it = 1
        gap = float("inf")
        x_bar = {l: 0.0 for l in linkSet}
        startP = time.time()
        while gap > accuracy:
            if algorithm == "MSA" or it < 2:
                alpha = (1.0 / it)
            elif algorithm == "FW":
                alpha = findAlpha(x_bar)
                # print("alpha", alpha)
            else:
                print("Terminating the program.....")
                print("The solution algorithm ", algorithm, " does not exist!")
            prevLinkFlow = np.array([linkSet[l].flow for l in linkSet])
            for l in linkSet:
                linkSet[l].flow = alpha * x_bar[l] + (1 - alpha) * linkSet[l].flow
            updateTravelTime()
            if loading == "deterministic":
                SPTT, x_bar = loadAON()
                # print([linkSet[a].flow * linkSet[a].cost for a in linkSet])
                TSTT = round(sum([linkSet[a].flow * linkSet[a].cost for a in linkSet]), 3)
                SPTT = round(SPTT, 3)
                gap = round(abs((TSTT / SPTT) - 1), 5)
                # print(TSTT, SPTT, gap)
                if it == 1:
                    gap = gap + float("inf")
            elif loading == "stochastic":
                x_bar = DialLoad()
                currentLinkFlow = np.array([linkSet[l].flow for l in linkSet])
                change = (prevLinkFlow - currentLinkFlow)
                if it < 3:
                    gap = gap + float("inf")
                else:
                    gap = round(np.linalg.norm(
                        np.divide(change, prevLinkFlow, out=np.zeros_like(change), where=prevLinkFlow != 0)), 2)

            else:
                print("Terminating the program.....")
                print("The loading ", loading, " is unknown")

            it = it + 1
            if it > maxIter:
                print("The assignment did not converge with the desired gap and max iterations are reached")
                print("current gap ", gap)
                break
        print("Assignment took", time.time() - startP, " seconds")
        print("assignment converged in ", it, " iterations")

    def writeUEresults():
        outFile = open(inputLocation + '/' + prefix_filename + "_SUE_results.dat", "w")  # IVT, WT, WK, TR
        tmpOut = "tailNode\theadNode\tcapacity\tlength\tfft\tUE_travelTime\tUE_flow"
        outFile.write(tmpOut + "\n")
        for i in linkSet:
            tmpOut = str(linkSet[i].tailNode) + "\t" + str(linkSet[i].headNode) + "\t" + str(
                linkSet[i].capacity) + "\t" + str(linkSet[i].length) + "\t" + str(linkSet[i].fft) + "\t" + str(
                linkSet[i].cost) + "\t" + str(linkSet[i].flow)
            outFile.write(tmpOut + "\n")
        outFile.close()

    ###########################################################################################################################

    # # assignment("stochastic", "FW", accuracy = 0.001, maxIter=1000)
    # assignment("deterministic", "FW", accuracy=0.001, maxIter=1000)
    # writeUEresults()

    assignment(options['equilibrium'], options['method'], accuracy=options['accuracy_eq'], maxIter=options['maxIter'])
    # assignment("stochastic", "MSA", accuracy=accuracy, maxIter=maxIter)
    writeUEresults()

    x = {(int(i[0]) - 1, int(i[1]) - 1, '0'): link.flow for i, link in linkSet.items()}
    tt_x = {(int(i[0]) - 1, int(i[1]) - 1, '0'): link.traveltime for i, link in linkSet.items()}

    return x, tt_x
