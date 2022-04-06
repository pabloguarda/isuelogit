from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytypes import Links, Matrix, ColumnVector, Features, Paths, Options, Option, Vector, Optional, List

from printer import block_output

from itertools import combinations

from paths import compute_path_size_factors
from networks import TNetwork
from estimation import UtilityFunction
from utils import v_normalization, almost_zero, Options

import math
import time
import heapq
import numpy as np
import os
from scipy import optimize
from abc import ABC, abstractmethod
import copy


class Equilibrator(ABC):

    def __init__(self,
                 network: TNetwork = None,
                 utility_function: UtilityFunction = None,
                 paths_generator=None,
                 **kwargs
                 ):
        self.network = network
        self.utility_function = utility_function
        self.paths_generator = paths_generator

        # Check that all bpr functions of the links have been defined

        # Check that an OD matrix is available
        # assert network.Q is not None, 'The network has no O-D matrix'

        # Check that logit parameters have been provided

        # Dictionary to store options

        self.set_default_options()

        self.options = self.options.get_updated_options(new_options=kwargs)

    @abstractmethod
    def set_default_options(self):
        raise NotImplementedError

    def get_updated_options(self, **kwargs):
        return copy.deepcopy(self.options.get_updated_options(new_options=kwargs))

    def update_options(self, **kwargs):
        self.options = self.get_updated_options(**kwargs)

    def update_network(self, network):
        self.network = network

    def update_utility_parameters(self, value):
        self.utility_function = value


class LUE_Equilibrator(Equilibrator):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Check that a matrix M and D has been provided

        pass

    def set_default_options(self):

        '''
        Congested or uncongested mode
        # * In uncongested mode only 1 MSA iteration is made. More iterations are useful for the congested case
        # In addition, the parameters of the bpr functions are set equal to 0 so the link travel time is just
        free flow travel time

        Returns:

        '''

        self.options = Options()

        # Uncongested mode
        self.options['uncongested_mode'] = False

        # Exogenous travel times (i.e. true travel times obtained after simulating counting data)
        self.options['exogenous_traveltimes'] = False

        # Maximum number of iterations for equilibrium algorithms
        self.options['max_iters'] = 9

        # accuracy for relative gap used for equilibrium algorithm
        self.options['accuracy'] = 1e-4

        # Method to compute equilibrium ('fw' or 'msa')
        self.options['method'] = 'fw'

        # Options  for frank wolfe

        # - Granularity/iterations for line search in Frank-Wolfe(fw) algorithm
        self.options['iters_fw'] = 11

        # - Type of line search for fw ('grid' or bisection)
        self.options['search_fw'] = 'grid'

        # Options for column generation
        self.options['column_generation'] = {}

        # Number of paths generated per of pair during column generation (If 0, no column generation is performed)
        self.options['column_generation']['n_paths'] = None

        # Number of paths selected for column generation
        self.options['column_generation']['paths_selection'] = None

        # Dissimilarity weight
        self.options['column_generation']['dissimilarity_weight'] = 0

        # Coverage of OD pairs to sample new paths
        self.options['column_generation']['ods_coverage'] = 1

        # Select ods at 'random', 'demand', 'demand_sequential'
        self.options['column_generation']['ods_sampling'] = 'demand'

        # Record the number of times that the ods sampling has been performed
        self.options['column_generation']['n_ods_sampling'] = 0

        # Correction using path size logit
        self.options['path_size_correction'] = 0

    # TODO: Implement frankwolfe and MSE using as reference ta.py and comparing result with observed predicted_counts in files.

    def derivative_sue_objective_function_fisk(self,
                                               f1: np.array,
                                               f2: np.array,
                                               lambda_bs: float,
                                               theta: dict,
                                               k_Z: List = [],
                                               k_Y: List = ['tt'],
                                               network=None
                                               ):

        ''' Numerical issues generates innacuracy to compute the derivative and thus, to perofrm the binary search later. '''

        if network is None:
            network = self.network

        # Path flow
        # f = lambda_bs * f2 + (1 - lambda_bs) * f1
        f = f1 + lambda_bs * (f2 - f1)

        # x1 = network.D.dot(f1)
        # x2 = network.D.dot(f2)

        delta_x2x1 = network.D.dot(f2 - f1)

        # x_weighted = x1 + lambda_bs*(x2-x1)
        # np.allclose(x1 + lambda_bs * (x2 - x1), network.D.dot(f))
        x_weighted = network.D.dot(f)

        # Objective function

        # Component for exogenous attributes (independent on link flow)
        Z_dlambda = 0

        if k_Z:
            for attr in k_Z:
                Zx_vector = np.array(list(network.Z_data[attr]))[:, np.newaxis]
                Z_dlambda += theta[attr] * Zx_vector

            Z_dlambda = float(Z_dlambda.T.dot(delta_x2x1))

        # Component for endogeonous attributes dependent on link flow
        traveltimes = [link.bpr.bpr_function_x(x=float(x)) for link, x in
                       zip(network.links, x_weighted.flatten().tolist())]

        traveltimes_dlambda = theta[k_Y[0]] * np.array(traveltimes)[:, np.newaxis].T.dot(delta_x2x1)

        # Entropy term

        epsilon = 1e-12
        # f1 + mid_lambda * (f2 - f1)
        # entropy_dlambda = np.sum((f2-f1)*(np.log(f)+1))
        entropy_dlambda = np.sum(almost_zero(f2 - f1, tol=epsilon) * (np.log(f + epsilon) + 1))

        # np.sum(almost_zero(f2-f1))

        # almost

        # derivative_objective_function = Z_dlambda - entropy_dlambda

        derivative_objective_function = float(traveltimes_dlambda) + Z_dlambda - entropy_dlambda

        return derivative_objective_function

    def entropy_path_flows_sue(self, f):

        ''' It corrects for numerical issues by masking terms before the entropy computation'''

        epsilon = 1e-12

        return np.sum(almost_zero(f, tol=epsilon) * (np.log(f + epsilon)))

        # return np.log(f) * f

    def sue_objective_function_fisk(self,
                                    f: Vector,
                                    theta: dict,
                                    k_Z: [],
                                    k_Y: [] = ['tt'],
                                    network=None
                                    ):

        if network is None:
            network = self.network

        # links_dict = network.links_dict
        # x_vector = np.array(list(x_dict.values()))
        x_vector = network.D.dot(f)

        if not np.all(f >= 0):
            print('some elements in the path flow vector are negative')

        # Objective function

        # Component for endogeonous attributes dependent on link flow
        bpr_integrals = [float(link.bpr.bpr_integral_x(x=x)) for link, x in
                         zip(network.links, x_vector.flatten().tolist())]
        # bpr_integrals = [float(link.bpr.bpr_integral_x(x=x_dict[i])) for i, link in links_dict.items()]

        tt_utility_integral = float(theta[k_Y[0]]) * np.sum(np.sum(bpr_integrals))

        # Component for exogenous attributes (independent on link flow)
        Z_utility_integral = 0

        if k_Z:
            for attr in k_Z:
                Zx_vector = np.array(list(network.Z_data[attr]))[:, np.newaxis]
                Z_utility_integral += float(theta[attr]) * Zx_vector.T.dot(x_vector)

        # Objective function in multiattribute problem
        utility_integral = tt_utility_integral + float(Z_utility_integral)
        # utility_integral = float(Z_utility_integral)

        # entropy = cp.sum(cp.entr(cp.hstack(list(cp_f.values()))))

        entropy_function = self.entropy_path_flows_sue(f)

        # if not np.all(f > 0):
        #     print('some elements in the path flow vector are 0')
        #     f = no_zeros(f)
        # entropy_function = np.sum(np.log(f)*f)

        # objective_function = utility_integral #- entropy_function
        objective_function = utility_integral - entropy_function

        return float(objective_function)

    def traffic_assignment_path_space(self,
                                      q,
                                      vf: ColumnVector,
                                      network=None):

        """

        :param vf: Vector of path utilities
        :param q: assume that q is a row vector but this should change
        """

        if network is None:
            network = self.network

        # Network matrices
        C = network.C
        # q = network.q

        assert q.shape[1] == 1, 'od vector is not a column vector'

        # TODO: store this matrix in the network object eventually to save computation

        # if len(q.shape) > 1 and q.shape[0] > 1:
        #     q = q.reshape((q.T.shape))

        # qM = q.dot(network.M)

        vf = v_normalization(v=vf, C=C)
        exp_vf = np.exp(vf)
        # v = np.exp(np.sum(V_Z, axis=1) + V_Y)

        # Denominator logit functions
        sum_exp_vf = C.dot(exp_vf)

        p_f = exp_vf / sum_exp_vf

        f = np.multiply(network.M.T.dot(q), p_f)

        return f, p_f

    def traffic_assignment(self,
                           q,
                           vf: Vector,
                           network=None
                           ):

        """ vf is assumed to be a column vector"""

        if network is None:
            network = self.network

        assert vf.shape[1] == 1, 'vector of path predicted_counts is not a column vector'

        f, p_f = self.traffic_assignment_path_space(network=network,
                                                    q=q,
                                                    vf=vf)

        x = network.D.dot(f)

        return x, f, p_f

    # @blockPrinting
    def path_based_suelogit_equilibrium(self,
                                        features_Z=None,
                                        theta=None,
                                        q: ColumnVector = None,
                                        silent_mode=False,
                                        network=None,
                                        **kwargs):

        t0 = time.time()

        exogenous_traveltimes = kwargs.pop('exogenous_traveltimes', self.options['exogenous_traveltimes'])

        options = self.get_updated_options(**kwargs)

        if theta is None:
            theta = self.utility_function.true_values

        if network is None:
            network = self.network

        if q is None:
            q = network.q

        if features_Z is None:
            features_Z = self.utility_function.features_Z

        max_iters = options['max_iters']

        if options['uncongested_mode'] or exogenous_traveltimes:
            max_iters = 0

        if not silent_mode:
            print("\nSUE via " + options['method'] + " (max iters: " + str(max_iters) + ')', end='\n')

        gap = float("inf")
        gap_x = []
        lambdas_ls = []
        it = 0
        end_algorithm = False

        path_set_selection_done = False
        column_generation_done = False

        fisk_objective_functions = []

        # if standardization is not None:
        #     vf_Z = preprocessing.scale(vf_Z, with_mean=standardization['mean'], with_std=standardization['sd'], axis=0)

        while end_algorithm is False:

            # Step 0
            if it == 0:

                if exogenous_traveltimes:
                    for link in network.links:
                        link.traveltime = link.true_traveltime

                else:
                    # Initialize link travel times with free flow travel time
                    for link in network.links:
                        link.set_traveltime_from_x(x=0)

                # Generating new paths at every iteration is costly
                if options['column_generation']['n_paths'] is not None and column_generation_done is False:

                    if options['column_generation'].get('ods_sampling', None) == 'sequential':
                        ods_coverage = options['column_generation'].get('ods_coverage', 1)
                        options['column_generation']['ods_coverage'] = ods_coverage / kwargs.get('bilevel_iters', 1)

                    self.sue_column_generation(theta=theta,
                                               n_paths=options['column_generation']['n_paths'],
                                               ods_coverage=options['column_generation']['ods_coverage'],
                                               ods_sampling=options['column_generation']['ods_sampling'],
                                               network=network
                                               )
                    column_generation_done = True

                if options['path_size_correction'] > 0:

                    path_specific_utilities = []

                    path_size_factors = compute_path_size_factors(D=network.D, paths_od=network.paths_od)

                    for idx, path in zip(np.arange(0, len(network.paths)), network.paths):
                        path.specific_utility = float(options['path_size_correction'] * np.log(path_size_factors[idx]))
                        path_specific_utilities.append(path.specific_utility)

                    print('Path size correction with factor', options['path_size_correction'])

                    # network.Z_data[features_Z]

                    # [path.utility_summary(theta = theta) for path in network.paths_od[((1618, 1773))]]
                    # (1618, 1775), (1618, 1776), (1618, 1777)

                # Traffic assignment
                x, f, p_f = self.traffic_assignment(
                    network=network,
                    q=q,
                    vf=network.get_paths_utility(theta=theta, features_Z=features_Z))

                initial_fisk_objective_function = self.sue_objective_function_fisk(
                    network=network,
                    f=f,
                    theta=theta,
                    k_Z=features_Z)

            # if not silent_mode and max_iters > 0:
            #     printProgressBar(it, max_iters, prefix='Progress:', suffix='', length=20)

            if it >= 1:

                # Traffic assignment to get auxiliary link flow y
                y, f_y, p_f = self.traffic_assignment(
                    network=network,
                    q=q,
                    vf=network.get_paths_utility(theta=theta, features_Z=features_Z))

                if options['method'] == 'msa':
                    alpha_n = 1 / (it + 1)

                    # x = x + alpha_n * (y-x)
                    f = f + alpha_n * (f_y - f)

                if options['method'] == 'fw':
                    lambda_ls, xmin_ls, fmin_ls, objectivemin_ls \
                        = self.sue_line_search(theta=theta,
                                               features_Z=features_Z,
                                               iters=options['iters_fw'],
                                               search_type=options['search_fw'],
                                               network=network,
                                               f1=f,
                                               f2=f_y)

                    x = xmin_ls
                    f = fmin_ls

                    lambdas_ls.append(lambda_ls)

                    # print('fisk objective', sue_objective_function_fisk(network=network, x_dict=x_dict , f=f_weighted, theta=theta, features=features))

                # evaluate sue objective function
                fisk_objective_function = self.sue_objective_function_fisk(network=network,
                                                                           f=f,
                                                                           theta=theta,
                                                                           k_Z=features_Z)

                fisk_objective_functions.append(fisk_objective_function)

                if len(fisk_objective_functions) >= 2:
                    # if it >= 2:
                    # max_fisk_objective_functions = np.max(fisk_objective_functions[:-1])
                    # #TODO: Definition for equilibrium gap does not apply for SUE-logit but may adapt it
                    # change = (fisk_objective_function - max_fisk_objective_functions)
                    # gap = np.linalg.norm(
                    #     np.divide(change, max_fisk_objective_functions, out=np.zeros_like(change), where=fisk_objective_function != 0))

                    change = (fisk_objective_functions[-1] - fisk_objective_functions[-2])

                    gap = np.linalg.norm(
                        np.divide(change, fisk_objective_functions[-1], out=np.zeros_like(change),
                                  where=fisk_objective_functions[-1] != 0))

                    gap_x.append(gap)

            if options['column_generation']['paths_selection'] is not None and path_set_selection_done is False:

                # Create dictionary with path probabilities
                pf_dict = {str(path.get_nodes_keys()): p_f[i] for i, path in zip(np.arange(len(p_f)), network.paths)}

                print('\nPath selection:', 'probability_weight: '
                      + str(round(1 - options['column_generation']['dissimilarity_weight'], 1)) +
                      ', maximum number of paths per od: ' + str(options['column_generation']['paths_selection']))

                total_paths = 0
                total_ods = 0
                total_paths_removed = 0

                # Combinatorial problem which works well for small path set
                for od, paths in network.paths_od.items():

                    total_paths_od = len(paths)

                    if total_paths_od > options['column_generation']['paths_selection']:
                        network.paths_od[od], best_score \
                            = self.path_set_selection(
                            paths=paths,
                            pf_dict=pf_dict,
                            k=options['column_generation']['paths_selection'],
                            dissimilarity_weight=options['column_generation']['dissimilarity_weight']
                        )

                        total_paths_removed += total_paths_od - options['column_generation']['paths_selection']

                        total_ods += 1

                    total_paths += len(network.paths_od[od])

                with block_output(show_stdout=False, show_stderr=False):
                    network.load_paths(paths_od=network.paths_od)

                path_set_selection_done = True

                print(str(total_paths_removed) + ' paths removed among ' + str(
                    total_ods) + ' ods (New total paths: ' + str(len(network.paths)) + ')')

                # print('Path selection with k=' + str(options['column_generation']['paths_selection']),
                #       '(New total paths: ' + str(total_paths) +')' )

                # New paths change trajectory of equilibria so the process is restarted
                it = -1

            if not options['uncongested_mode'] and not exogenous_traveltimes:
                for link, j in zip(network.links, range(len(x))):
                    link.set_traveltime_from_x(x=float(x[j]))

            it += 1

            if it > max_iters:
                end_algorithm = True

            elif gap < options['accuracy'] and it > 1:
                end_algorithm = True

        if gap > options['accuracy'] and not options['uncongested_mode'] and not exogenous_traveltimes:
            if not silent_mode:
                print("Assignment did not converge with the desired gap")
                # print("Traffic assignment did not converge with the desired gap and max iterations are reached")

        if not silent_mode and max_iters > 0:

            print('\nEquilibrium gaps:', ["{0:.0E}".format(val) for val in gap_x])

            if options['method'] == 'line_search':
                print('lambdas:', ["{0:.2E}".format(val) for val in lambdas_ls])

            final_fisk_objective_function = fisk_objective_functions[-1]

            print('Initial Fisk Objective: ' + '{:,}'.format(
                round(initial_fisk_objective_function, 2)))
            print('Final Fisk Objective: ' + '{:,}'.format(
                round(final_fisk_objective_function, 2)))

            if initial_fisk_objective_function != 0:
                print('Improvement Fisk Objective: ' + "{:.2%}".format(
                    np.round((final_fisk_objective_function - initial_fisk_objective_function) / abs(
                        initial_fisk_objective_function), 4)))
            print('Final gap: ' + "{0:.0E}".format(gap) + '. Acc. bound: ' + "{0:.0E}".format(
                options['accuracy']) + '. Time: ' + str(round(time.time() - t0, 1)) + ' [s]')
            # print('Iters:',str(it - 1))

        links_keys = list(network.links_dict.keys())

        x_final = dict(zip(links_keys, list(x.flatten())))

        tt_final = dict(zip(links_keys, [link.traveltime for link in network.links]))

        for link in network.links:
            link.set_traveltime_from_x(x=0)

        return {'x': x_final
            , 'tt_x': tt_final
            , 'gap_x': gap_x
            , 'p_f': p_f
            , 'f': f
                }

    def sue_line_search(self,
                        iters,
                        search_type,
                        f1: Vector,
                        f2: Vector,
                        theta: dict,
                        features_Z: [],
                        network=None,
                        ):

        # Under the assumption the best lambda result from solving a convex problem, we can use the bisection method

        # def sue_objective_function_fisk_numeric_grad(lambda_bs):
        #
        #     fnew = f1 + lambda_bs * (f2-f1)
        #     # fnew = f1*(1-lambda_bs) + lambda_bs * f2
        #
        #     if not np.all(fnew >= 0):
        #         here = 0
        #
        #     xnew = network.D.dot(fnew)
        #     xnew_dict = dict(zip(list(x1_dict.keys()), xnew))
        #
        #     objective_new = self.sue_objective_function_fisk(theta=theta,
        #                                                      x_dict=xnew_dict,
        #                                                      f=xnew,
        #                                                      k_Z=k_Z)
        #
        #     return objective_new

        def plot_fisk_objective_derivatives_lambdas():

            import matplotlib.pyplot as plt

            values = []
            derivatives = []

            lambdas_bs = np.arange(0, 1, 0.01)

            for lambda_bs in np.arange(0, 1, 0.01):
                fopt = f1 + lambda_bs * (f2 - f1)

                values.append(self.sue_objective_function_fisk(f=fopt,
                                                               theta=theta,
                                                               network=self.network,
                                                               k_Z=features_Z))

                derivatives.append(self.derivative_sue_objective_function_fisk(
                    f1=f1,
                    f2=f2,
                    theta=theta,
                    lambda_bs=lambda_bs,
                    network=network,
                    k_Z=features_Z
                ))

            plt.plot(lambdas_bs, values)
            plt.show()

            plt.plot(lambdas_bs, derivatives)
            plt.show()

            return values, derivatives

        if network is None:
            network = self.network

        # values, derivatives = plot_fisk_objective_derivatives_lambdas()

        if search_type == 'bisection':

            left_lambda = 0
            right_lambda = 1

            for iter in range(iters):

                mid_lambda = 0.5 * (left_lambda + right_lambda)

                # derivative = nd.Gradient(sue_objective_function_fisk_numeric_grad)(
                #     mid_lambda)
                # print(derivative)

                derivative = self.derivative_sue_objective_function_fisk(
                    f1=f1,
                    f2=f2,
                    theta=theta,
                    lambda_bs=mid_lambda,
                    network=network,
                    k_Z=features_Z
                )
                # print(derivative)

                if derivative < 0:
                    left_lambda = left_lambda
                    right_lambda = 0.5 * (left_lambda + right_lambda)

                else:
                    left_lambda = 0.5 * (left_lambda + right_lambda)

            lambda_opt = mid_lambda

            fopt = f1 + mid_lambda * (f2 - f1)

            xopt = network.D.dot(fopt)

            objective_opt = self.sue_objective_function_fisk(f=fopt,
                                                             theta=theta,
                                                             network=network,
                                                             k_Z=features_Z)

        if search_type == 'grid':
            objective_opt = float('-inf')
            lambda_opt = None
            xopt = None
            fopt = None

            grid_lambda = np.linspace(0, 1, iters)
            objectives = []

            for lambda_ls in grid_lambda:

                # From Damberg (1996)
                fnew = f1 + lambda_ls * (f2 - f1)
                xnew = network.D.dot(fnew)

                objective_new = self.sue_objective_function_fisk(f=fnew,
                                                                 theta=theta,
                                                                 k_Z=features_Z,
                                                                 network=network
                                                                 )

                objectives.append(objective_new)

                if objective_new > objective_opt:
                    objective_opt = objective_new
                    xopt = xnew
                    fopt = fnew
                    lambda_opt = lambda_ls

                else:
                    # Assume that the objective function is always concave to save iterations
                    break

        return lambda_opt, xopt, fopt, objective_opt

    def path_set_selection(self,
                           paths,
                           pf_dict,
                           k,
                           dissimilarity_weight):

        # https://www.geeksforgeeks.org/python-percentage-similarity-of-lists/
        # https://stackoverflow.com/questions/41680388/how-do-i-iterate-through-combinations-of-a-list

        best_score = -float('inf')

        for path_set in combinations(paths, k):  # 2 for pairs, 3 for triplets, etc

            total_probability = 0
            total_similarity = 0

            for path in path_set:
                total_probability = pf_dict[str(path.get_nodes_keys())]

            for paths_pair in combinations(paths, 2):
                path1_sequence = paths_pair[0].get_nodes_keys()
                path2_sequence = paths_pair[1].get_nodes_keys()

                similarity = len(set(path1_sequence) & set(path2_sequence)) / float(
                    len(set(path1_sequence) | set(path2_sequence)))
                total_similarity += similarity

            average_dissimilarity = 1 - total_similarity / len(path_set)
            average_probability = total_probability / len(path_set)

            score = dissimilarity_weight * average_dissimilarity + (1 - dissimilarity_weight) * average_probability

            if score >= best_score:
                best_score = score
                best_path_set = path_set
                best_average_probability = average_probability
                best_average_dissimilarity = average_dissimilarity

        return list(best_path_set), best_score

    def sue_column_generation(self,
                              network,
                              theta,
                              n_paths,
                              ods_coverage=None,
                              ods_sampling=None,
                              silent_mode=False) -> None:

        '''

        Algorithm is the followig:
        0.Initialization

        # - Perform traffic assignment by computing shortest paths using the current estimate of the logit parameters and free flow travel times (1 iteration of MSA)

        # 1. Restricted master problem phase

        # Loop:

        # i) Perform traffic assignment again with new travel times

        # ii) Line search to find the minimum objective function for SUE

        # 2. Column generation phase

        # i) Augment the path set used in 1, by for instance, adding the next shortest path

        # * I may do the augmentation only once for efficiency but define a factor to control for this. The augmentation may be based on the shortest path

        # * In the algorithm proposed by Damberg et al. (1996)  it is, however, possible to avoid generating predicted_counts on overlapping routes by deleting (or suitably modifying) any route generated that overlaps with a previously generated one more than a maximal allowed measure of overlapping; depending on the overlap measure, this may be easily performed by augmenting the route generation phase with a suitable check.

        Args:
            network:
            theta:
            n_paths:
            ods_coverage:
            ods_sampling:
            silent_mode:

        Returns:

        '''

        t0 = time.time()

        if ods_coverage is None:
            ods_coverage = self.options['column_generation']['ods_coverage']

        if ods_sampling is None:
            ods_sampling = self.options['column_generation']['ods_sampling']

        print('\nColumn generation:', str(n_paths) + ' paths per od, ' + "{:.1%}".format(
            ods_coverage) + ' od coverage, ' + ods_sampling + ' sampling')

        ods_sample = None

        # Sample part of the ods according to the coverage defined for column generation
        if ods_coverage > 0 and ods_coverage <= 1:

            if ods_sampling == 'random':
                ods_sample = network.OD.random_ods(ods_coverage)

            if ods_sampling == 'demand':
                ods_sample = network.OD.sample_ods_by_demand(proportion=ods_coverage)

            if ods_sampling == 'sequential':
                ods_sample = network.OD.sample_ods_by_demand_sequentially(
                    proportion=ods_coverage,
                    k=self.options['column_generation']['n_ods_sampling'])
                self.options['column_generation']['n_ods_sampling'] += 1

        if ods_sample is None:
            ods_sample = network.ods

        with block_output(show_stdout=False, show_stderr=False):
            paths, paths_od = self.paths_generator.k_shortest_paths(network=network,
                                                                    theta=theta,
                                                                    k=n_paths,
                                                                    ods=ods_sample,
                                                                    paths_per_od=n_paths,
                                                                    silent_mode=True)

        # See if new paths were found so they are added into the existing path set
        paths_added = 0
        n_ods_added = 0

        for od, paths in paths_od.items():

            some_path_added_od = False

            existing_paths_keys = [path.get_nodes_keys() for path in network.paths_od[od]]
            for path in paths_od[od]:
                if path.get_nodes_keys() not in existing_paths_keys:
                    network.paths_od[od].append(path)
                    # network.paths.append(path)
                    paths_added += 1
                    some_path_added_od = True

            if some_path_added_od:
                n_ods_added += 1

        # network.load_paths(paths_od = network.paths_od)
        with block_output(show_stdout=False, show_stderr=False):
            network.update_incidence_matrices()

        # enablePrint()

        # print("Total number of links among paths: ", np.sum(network.D))
        print(str(paths_added) + ' paths added/replaced among ' + str(n_ods_added) + ' ods (New total paths: ' + str(
            len(network.paths)) + ')')
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
        This method produces auxiliary predicted_counts for all or nothing loading.
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
