from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytypes import Dict, Matrix, Optional, List, Proportion, Links, Features, Feature, UtilityFunction, ColumnVector

from networks import TNetwork, MultiDiTNetwork, DiTNetwork, OD

from utils import Options

from links import Link, BPR
from nodes import Node
from geographer import LinkPosition, NodePosition
from paths import k_path_generation_nx, k_simple_paths_nx, Path, get_paths_od_from_paths
from descriptive_statistics import  nrmse, rmse

from abc import ABC, abstractmethod

from equilibrium import LUE_Equilibrator
from etl import masked_observed_counts

from reader import read_tntp_network, read_colombus_network, read_sacramento_network, read_internal_network_files, read_internal_paths
from writer import write_internal_network_files,write_internal_paths, write_tntp_github_to_dat, write_internal_network_files

import printer

from sklearn import preprocessing
import pandas as pd
import numpy as np
import random
import copy
import time

# @staticmethod
def generate_Q(network: TNetwork,
               min_q: float,
               max_q: float,
               cutoff: int,
               n_paths: int,
               sparsity: Proportion = 0):

    print('Generating matrix Q')

    t0 = time.time()

    # G = nx.DiGraph(network.A)

    Q_mask = np.random.randint(min_q, max_q, network.A.shape)
    Q = np.zeros(Q_mask.shape)

    # Q = np.random.randint(2, 10, (100,100))

    # Set terms to 0 if there is no a path in between nodes on the graph produced by A

    nonzero_entries_Q_mask = list(Q_mask.nonzero())
    random.shuffle(nonzero_entries_Q_mask)

    total_entries_Q_mask = Q.shape[0] ** 2
    expected_non_zero_Q_entries = int(total_entries_Q_mask * (1 - sparsity))

    print('The expected number of matrix entries to fill out is ' + str(
        expected_non_zero_Q_entries) + '. Sparsity: ' + "{:.0%}".format(sparsity))

    # total = len(nonzero_Q_entries)
    counter = 0

    for (i, j) in zip(*tuple(nonzero_entries_Q_mask)):

        # Q = isl.config.sim_options['custom_networks']['A']['N2']
        # print((i,j))

        printer.printProgressBar(counter, expected_non_zero_Q_entries, prefix='Progress:', suffix='', length=20)

        # Very ineficient as it requires to enumerate all paths
        # if len(list(nx.all_simple_paths(G, source=i, target=j, cutoff=cutoff))) == 0:

        k_paths_od = k_simple_paths_nx(k=n_paths, source=i, target=j, cutoff=cutoff, G=network.G,
                                       links=network.links_dict)

        if len(list(k_paths_od)) == n_paths:

            Q[(i, j)] = Q_mask[(i, j)]

            counter += 1

            if counter > expected_non_zero_Q_entries:
                break
        else:
            # print('No paths with less than ' + str(cutoff) + ' links were found in o-d pair ' + str((i,j)))
            pass

    non_zero_entries_final_Q = np.count_nonzero(Q != 0)  # / float(Q.size)

    sparsity_final_Q = 1 - non_zero_entries_final_Q / float(Q.size)

    #  Fill the Q matrix with zeros according to the degree of sparsity. As higher is the sparsity, faster is the generation of the Q matrix because there is less computation of shortest paths below

    # example: https://stackoverflow.com/questions/40058912/randomly-controlling-the-percentage-of-non-zero-values-in-a-matrix-using-python

    # idx = np.flatnonzero(Q)
    # N = np.count_nonzero(Q != 0) - int(round((1-sparsity) * Q.size))
    # np.put(Q, np.random.choice(idx, size=N, replace=False), 0)

    assert Q.shape[0] > 0, 'Matrix Q could not be generated'

    print(
        str(non_zero_entries_final_Q) + ' entries were filled out. Sparsity: ' + '{:.0%}'.format(sparsity_final_Q))

    print('Matrix Q ' + str(Q.shape) + ' generated in ' + str(round(time.time() - t0, 1)) + '[s]')

    return Q

# @staticmethod
def random_disturbance_Q(Q, sd=0):
    '''Add a random disturbance but only for od pairs with trips'''

    Q_original = Q.copy()

    non_zeros_entries = 0
    # print(var)
    if sd == 'Poisson':
        for (i, j) in zip(*Q.nonzero()):
            Q[(i, j)] = np.random.poisson(lam=Q[(i, j)])
            non_zeros_entries += 1
            if Q[(i, j)] == 0:
                Q[(i, j)] += 1e-7  # To avoid bugs when zeros are removed from Q matrix for other methods

    elif sd > 0:

        # # Lognormal
        # for (i, j) in zip(*Q.nonzero()):
        #     non_zeros_entries += 1
        #     Q[(i, j)] += np.random.lognormal(mean = 0, sigma = np.log(np.sqrt(var)))
        #
        # Truncated normal
        for (i, j) in zip(*Q.nonzero()):
            non_zeros_entries += 1
            Q[(i, j)] += np.random.normal(loc=0, scale=sd)

        # We truncate entries so they are positive by small number to avoid bugs when zeros are removed from Q matrix for other methods
        Q[Q < 0] = 1e-7
    # Compute difference between cell values in original and new demand matrix that were non zeros
    print('Mean of nonzero entries in the original demand matrix: ',
          "{0:.1f}".format(Q_original[np.nonzero(Q)].mean()))
    print('Mean absolute difference between the nonzero entries of the noisy and original:',
          "{0:.1f}".format(np.sum(np.abs(Q_original - Q)) / non_zeros_entries))
    print('Approximated proportion change:',
          "{0:.1%}".format(
              np.sum(np.abs(Q_original - Q)) / (non_zeros_entries * Q_original[np.nonzero(Q_original)].mean())))

    return Q


class Generator(ABC):

    def __init__(self, **kwargs):

        self.set_default_options()
        self.update_options(**kwargs)
        # self.options = self.get_updated_options(new_options=kwargs)

    @abstractmethod
    def set_default_options(self):
        raise NotImplementedError

    def get_updated_options(self, **kwargs):
        return self.options.get_updated_options(new_options=kwargs)

    def update_options(self, **kwargs):
        self.options = self.get_updated_options(**kwargs)

    # def set_options(self, kwargs):
    #     for key, value in kwargs.items():
    #         self.options[key] = value

class LinkDataGenerator(Generator):

    def __init__(self, **kwargs):

        # self.network = network

        super().__init__(**kwargs)

        # # Attribute levels in matrix Z which are non-zero
        # self.sim_options['Z_attrs_classes'] = {}
        # #  - Waiting time (minutes)
        # self.sim_options['Z_attrs_classes']['s'] = dict({'1': 3, '2': 2, '3': 1, '4': 1, '5': 5, '6': 3})
        # #  - Cost in USD (e.g. toll)
        # self.sim_options['Z_attrs_classes']['c'] = dict({'1': 1, '2': 2, '3': 1.5, '4': 1, '5': 3})

        # self.features = features
        # self.features_Y = features_Y

    def set_default_options(self):
        '''

        # 'scale_Q':  Scaling of OD matrix (1 is the default)
        # sd_Q: Variance of lognormal distribution or Poisson in noise in OD matrix (values: Union['Poisson',float])
        # Notes: with poisson the noise is litte and there is almost no impact on convergence.
        # No need to set parameters for distribution as they are determined by mean of non-zero entries of OD matrix

        Returns:

        '''

        self.options = Options()

        # # Boolean for whether the network is assumed uncongested
        # self.options['uncongested_mode'] = False

        # Proportion of links in the network with traffic count data
        self.options['coverage'] = 1

        # Dictionary with options about the noise introduced in the traffic counts (e.g. )
        self.options['noise_params'] = {'mu_x': 0, 'sd_x': 0, 'snr_x': 0, 'sd_Q': 0, 'scale_Q': 1, 'congestion_Q': 1}




        # TODO: Confirm what this is used for
        self.options['sparsity_idx'] = 0

        # Number of paths per O-D for path generation. If None, new paths are not generated
        self.options['n_paths'] = None

        # # Copy options from equilibrator
        # self.options['equilibrator'] = self.equilibrator.options.copy()

        # Define multiple BPR instances/classes that will affect the travel time solution at equilibrium
        self.bpr_classes = {}
        self.bpr_classes['1'] = {'alpha': 0.15, 'beta': 4, 'tf': 2e-1, 'k': 1800}
        self.bpr_classes['2'] = {'alpha': 0.15, 'beta': 4, 'tf': 4e-1, 'k': 1800}
        self.bpr_classes['3'] = {'alpha': 0.15, 'beta': 4, 'tf': 6e-1, 'k': 1800}
        self.bpr_classes['4'] = {'alpha': 0.15, 'beta': 4, 'tf': 8e-1, 'k': 1800}
        self.bpr_classes['5'] = {'alpha': 0.15, 'beta': 4, 'tf': 10e-1, 'k': 1800}
        # self.bpr_classes['1']  = {'alpha': 0.15, 'beta': 4, 'tf': 1e0, 'k': 1800}
        # self.bpr_classes['2']  = {'alpha': 0.15, 'beta': 4, 'tf': 2e0, 'k': 1800}
        # self.bpr_classes['3']  = {'alpha': 0.15, 'beta': 4, 'tf': 3e0, 'k': 1800}
        # self.bpr_classes['4']  = {'alpha': 0.15, 'beta': 4, 'tf': 4e0, 'k': 1800}
        # self.bpr_classes['5']  = {'alpha': 0.15, 'beta': 4, 'tf': 5e0, 'k': 1800}

    def generate_random_bpr_parameters(self,
                                       links_keys: List[str]) -> pd.DataFrame:

        # i) Assign randomly BPR function to the links in each network -> travel time:

        alphas, betas, tfs, ks = [], [], [], []

        for key in links_keys:
            bpr_class = random.choice(list(self.bpr_classes.values()))

            alphas.append(bpr_class['alpha'])
            betas.append(bpr_class['beta'])
            tfs.append(bpr_class['tf'])
            ks.append(bpr_class['k'])

        df = pd.DataFrame({'link_key': links_keys,
                           'alpha': alphas,
                           'beta': betas,
                           'tf': tfs,
                           'k': ks})

        return df

    def generate_toy_bpr_parameters(self) -> pd.DataFrame:

        '''

        Toy network in Guarda & Qian 2022

        Returns:

        '''

        links_tuples = {1:(1, 3,'0'), 2:(2, 3,'0'), 3:(3, 4,'0'), 4:(3, 4,'1')}

        # Internally, nodes keys start from 0 and there is a '0' added in a tuple with three elements, indicating
        # one of many parallels links
        links_keys = []

        for id, link_key in links_tuples.items():
            links_keys.append((link_key[0] - 1, link_key[1] - 1, link_key[2]))

        bpr_parameters_df = pd.DataFrame({'link_key': links_keys,
                                          'tf': [1e-0, 2e-0, 3e-0, 4e-0],
                                          'k': [1800,1800,1800,1800],
                                          'alpha': 0.15,
                                          'beta': 4})

        bpr_parameters_df['tf'] = bpr_parameters_df['tf'] * 1.0

        return bpr_parameters_df

    def generate_Yang_bpr_parameters(self) -> pd.DataFrame:

        '''

        Based on Table 4 and Figure 1 from
        Yang, H., Meng, Q. and Bell, M.G.H., 2001. Simultaneous estimation of the origin-destination
        matrices and travel-cost coefficient for congested networks in a stochastic user equilibrium.
        Transportation Science 35, 107–123.

        Returns:

        '''

        links_tuples = {1:(1, 2), 2:(1, 4), 3:(1, 5), 4:(2, 3), 5:(2, 5), 6:(3, 6),
                        7:(4, 5), 8:(4, 7), 9:(5, 6), 10:(5, 8), 11:(5, 9), 12:(6, 9),
                        13: (7, 8), 14:(8, 9)}
        
        # Internally, nodes keys start from 0 and there is a '0' added in a tuple with three elements, indicating
        # one of many parallels links
        links_keys = []
        
        for id, link_key in links_tuples.items():
            links_keys.append((link_key[0]-1,link_key[1]-1,'0'))

        bpr_parameters_df = pd.DataFrame({'link_key': links_keys,
                      'tf': [2.0,1.5,3.0,1.0,1.0,2.0,2.0,1.0,1.5,1.0,2.0,1.0,1.0,1.0],
                      'k': [280,290,280,280,600,300,500,400,500,700,250,300,350,220],
                      'alpha':0.15,
                      'beta': 4})

        bpr_parameters_df['tf'] = bpr_parameters_df['tf'] * 1.5

        return bpr_parameters_df

    def generate_LoChan_bpr_parameters(self) -> pd.DataFrame:

        '''

        Figure 1 from Lo, H.P. and Chan, C.P., 2003. Simultaneous estimation of an origin-destination matrix and
        link choice proportions using traffic counts. Transportation Research Part A: Policy and Practice 37, 771–788.

        Returns:

        '''

        # Lo and Chan (2003)
        links_tuples = {1:(1, 2), 2:(2, 1), 3:(2, 3), 4:(3, 2), 5:(3, 6), 6:(6, 3) , 7:(6, 5), 8:(5, 6), 9:(5, 4),
                        10:(4, 5), 11: (1, 4), 12: (4, 1), 13: (2, 5), 14:(5, 2)}

        # Internally, nodes keys start from 0 and there is a '0' added in a tuple with three elements, indicating
        # one of many parallels links
        links_keys = []

        for id, link_key in links_tuples.items():
            links_keys.append((link_key[0] - 1, link_key[1] - 1, '0'))


        bpr_parameters_df = pd.DataFrame({'link_key': links_keys,
                                          'tf': 7.6,
                                          'k': 600,
                                          'alpha': 0.15,
                                          'beta': 4})

        bpr_parameters_df['tf'] = bpr_parameters_df['tf'] * 2e-1

        return bpr_parameters_df


    def generate_Wang_bpr_parameters(self) -> pd.DataFrame:

        '''
        Based on Table 6 from

        Wang, Yong, Ma, X., Liu, Y., Gong, K., Henricakson, K.C., Xu, M. and Wang, Yinhai, 2016.
        A two-stage algorithm for origin-destination matrices estimation considering dynamic dispersion
        parameter for route choice. PLoS ONE 11, 1–24.

        Returns:

        '''

        links_tuples = {1: (1, 2), 2: (2, 1), 3: (3, 4), 4: (4, 3), 5: (4, 1), 6: (1, 4),
                        7: (2, 3), 8: (3, 2)}

        # Internally, nodes keys start from 0 and there is a '0' added in a tuple with three elements, indicating
        # one of many parallels links
        links_keys = []

        for id, link_key in links_tuples.items():
            links_keys.append((link_key[0] - 1, link_key[1] - 1, '0'))

        bpr_parameters_df = pd.DataFrame({'link_key': links_keys,
                                          'tf': [0.1162, 0.1162, 0.0667, 0.0667, 0.1016, 0.1016, 0.1332, 0.1332],
                                          'k': [4149, 4149, 8685, 8685, 9683, 9683, 7961, 7961],
                                          'alpha': [0.1450, 0.1450, 0.1035, 0.1035,0.0988,0.0988,0.1242,0.1242],
                                          'beta': [3.5,3.5,2.7,2.7,2.7,2.7,3.5,3.5]})

        bpr_parameters_df['tf'] = bpr_parameters_df['tf']*1.5

        return bpr_parameters_df


    def set_bpr_functions(self, links_bprs: pd.DataFrame, network) -> None:

        # def set_random_link_BPR_network(self, bpr_classes: {}):
        '''
            :argument bpr_classes: different set of values for bpr functions
        '''

        for link_key, bpr in links_bprs.items():
            network.links_dict[link_key].bpr = links_bprs[link_key]


    def set_random_bpr_functions(self, network) -> None:

        links_bprs_dict = self.generate_random_bpr_parameters(network.links)

        self.set_bpr_functions(links_bprs=links_bprs_dict,
                               network=network)

    def simulate_features(self,
                          links,
                          features_Z: List[Feature],
                          range: tuple[float, float],
                          link_key: str = 'link_key',
                          option: str = None,
                          normalization = None,
                          ) -> pd.DataFrame:

        '''Generate random values for an arbitrary number of exogenous features
        Options define the type of randomness

        '''

        if normalization is None:
            normalization = {'mean': False, 'std': False}

        links_df = pd.DataFrame({link_key: [link.key for link in links]})

        n_links = len(links_df)

        for feature in features_Z:
            if option == 'discrete':
                links_df[feature] = np.random.randint(range[0], range[1]+1, n_links)
            if option == 'continuous':
                links_df[feature] = np.random.uniform(range[0], range[1], n_links)

            links_df[feature] = preprocessing.scale(links_df[feature],
                                with_mean=normalization['mean'],
                                with_std=normalization['std'],
                                axis=0)

        return links_df

    def add_error_counts(self,
                         original_counts: ColumnVector,
                         mu_x=0,
                         sd_x=0,
                         snr_x=0):

        mean_x = np.nanmean(original_counts)

        n = original_counts.shape[0]

        link_counts = original_counts.copy()

        if mu_x != 0 or sd_x != 0 or snr_x != 0:

            if snr_x != 0:
                # Signal noise ratio (tibshirani concept, https://arxiv.org/pdf/1707.08692.pdf)
                sd_x = np.sqrt(np.var(original_counts) / snr_x)

            else:
                sd_x = sd_x * mean_x

            link_counts = original_counts + np.random.normal(mu_x, sd_x, n)[:, np.newaxis]

            # We truncate link counts so they are positive
            link_counts[link_counts < 0] = 0

        return link_counts

    def mask_counts_by_coverage(self,
                                original_counts: ColumnVector,
                                coverage):

        link_counts = original_counts.copy()

        # Generate a random subset of idxs depending on coverage
        n = original_counts.shape[0]
        missing_sample_size = n - int(n * coverage)
        idx = list(np.random.choice(np.arange(0, n), missing_sample_size, replace=False))

        # Only a subset of observations is assumed to be known
        observed_link_counts = masked_observed_counts(counts=link_counts, idx=idx)

        # Set of link counts associated to observations that are not within the set of observed links
        withdrawn_link_counts = masked_observed_counts(counts=link_counts,
                                                       idx=list(set(list(np.arange(0, n))) - set(idx)))

        return observed_link_counts, withdrawn_link_counts

    def simulate_counts(self,
                        equilibrator: LUE_Equilibrator,
                        network=None,
                        utility_function: Optional[UtilityFunction] = None,
                        **kwargs) -> (Dict, Dict):
        """

        :param coverage: proportion of links where data is assumed to be known
        :param equilibrium_args:
        :param noise_params:


        :return:

            dictionary of link ids and link counts

        """

        assert isinstance(equilibrator, LUE_Equilibrator), 'equilibrator is not of type LUE_Equilibrator'

        if utility_function is None:
            utility_function = equilibrator.utility_function

        if network is None:
            network = equilibrator.network

        # Check that utility function parameters have no none true values, otherwise it sets them to 0
        for feature, value in utility_function.true_values.items():
            if value is None:
                utility_function.true_values[feature] = 0

        # Options
        options = self.options = self.get_updated_options(**kwargs)

        sd_Q = options['noise_params']['sd_Q']
        scale_Q = options['noise_params']['scale_Q']
        congestion_Q = options['noise_params']['congestion_Q']
        sd_x = options['noise_params']['sd_x']
        snr_x = options['noise_params']['snr_x']
        mu_x = options['noise_params']['mu_x']
        coverage = options['coverage']
        path_size_correction = options.get('path_size_correction', equilibrator.options['path_size_correction'])

        # equilibrator.options['uncongested_mode'] = True

        if equilibrator.options['uncongested_mode']:

            # To not alter the link function, first copy the current bpr parameters and then set them to 0
            bpr_alpha = {}
            bpr_beta = {}

            for link, link_num in zip(network.links, range(len(network.links))):
                bpr_alpha[link_num] = link.bpr.alpha
                bpr_beta[link_num] = link.bpr.beta

                link.bpr.alpha = 0
                link.bpr.beta = 0

        method = equilibrator.options['method']

        if method == 'fw':
            method_label = 'Frank-Wolfe'
        if method == 'msa':
            method_label = 'MSA'

        # iterations = equilibrator.options['max_iters']
        # print("\nGenerating synthetic link counts via " + method_label + " ("+ 'max iterations:'  + str(int(iterations)) + ')' + '\n')

        print("\nGenerating synthetic link counts via " + method_label)

        assert sd_Q >= 0, 'Standard deviation of OD matrix cannot be <= 0'

        # Store original OD matrix
        Q_original = copy.deepcopy(network.Q_true)

        # Manipulatin of noise or scale in Q matrix
        if sd_Q > 0:

            # assert sd_Q > 0, 'Standard deviation of OD matrix cannot be <= 0'

            # If noise is applied in the q matrix, we generate a copy of the original od demand vector and set a noisy matrix meanwhile to compute equilibrium

            if sd_Q != 'Poisson':
                # sd is a parameter computed as proportion of the mean
                sd_Q = np.mean(network.Q) * sd_Q

            Q_noisy = random_disturbance_Q(network.Q, sd=sd_Q).copy()

            # Update Q matrix and dense q vector temporarily
            network.load_OD(Q = Q_noisy)

        # Scaling error in Q matrix
        if scale_Q != 1:
            assert scale_Q >0, 'Scale of OD matrix cannot be <= 0'

            # Update Q matrix and dense q vector temporarily
            network.scale_OD(scale = 1/scale_Q)

        if congestion_Q != 1:
            assert congestion_Q > 0, 'Congestion factor of OD matrix cannot <= 0'
            network.scale_OD(scale = congestion_Q )

        # if len(options['sparsity_idxs']) != 0:
        #     # idx = list(set(np.flatnonzero(network.Q)))
        #     # N = int(round(sparsity * network.Q.size))
        #     np.put(network.Q, options['sparsity_idxs'], 0.1)
        #     network.q = PathGenerator.denseQ(Q=network.Q
        #                                      ,remove_zeros=network.setup_options['remove_zeros_Q'])
        # # Store new OD matrix
        # network.Q_true = copy.deepcopy(network.Q)
        # network.q_true = copy.deepcopy(network.q)

        if options['n_paths'] is not None:
            # Save previous paths and paths per od lists
            original_paths = copy.deepcopy(network.paths)
            original_paths_od = copy.deepcopy(network.paths_od)
            M, D, C = network.M.copy(), network.D.copy(), network.C.copy()

            paths, paths_od = equilibrator.paths_generator.k_shortest_paths(
                k = options['n_paths'],
                utility_function = utility_function.parameters.true_values)

            # printer.blockPrint()
            network.load_paths(paths_od = paths_od)
            # printer.enablePrint()

        # Initialize link travel times with free flow travel time
        for link in network.links:
            link.set_traveltime_from_x(x=0)

        # To generate counts, we ignore the constraint that traveltimes are exogenous.

        # eq_params
        # theta['tt'] = 0
        results_equilibrium = equilibrator.path_based_suelogit_equilibrium(
            theta= utility_function.true_values,
            features_Z=utility_function.features_Z,
            network=network,
            exogenous_traveltimes = False,
            # column_generation = {'n_paths': None},
            path_size_correction = path_size_correction
        )

        # Store path predicted_counts
        network.path_flows = results_equilibrium['f']

        # Exogenous noise in the link count measurements
        true_link_counts = np.array(list(results_equilibrium['x'].values()))[:, np.newaxis]

        assert true_link_counts.shape[1] == 1 and len(
            true_link_counts.shape) == 2, "vector of true link counts is not a column vector"

        # Account for random error
        link_counts = self.add_error_counts(true_link_counts, mu_x = mu_x, snr_x = snr_x, sd_x = sd_x)

        # Account for link coverage
        observed_link_counts, withdrawn_link_counts = self.mask_counts_by_coverage(original_counts= link_counts,
                                                                                   coverage = coverage)
        # Revert Q matrix and q dense vector to original form
        if scale_Q != 1:
            network.scale_OD(scale = 1)

        if sd_Q > 0:
            network.load_OD(Q = Q_original)

        # Do not do nothing with congestion factor on Q because it is still assumed that the true OD matrix is known

        # Revert original paths and incidence matrices
        if options['n_paths'] is not None:
            network.paths, network.paths_od = original_paths, original_paths_od
            network.M, network.D, network.C = M, D, C

        if equilibrator.options['uncongested_mode']:
            # Revert original bpr values
            for link, link_num in zip(network.links, range(len(network.links))):
                link.bpr.alpha = bpr_alpha[link_num]
                link.bpr.beta = bpr_beta[link_num]

        network.load_traveltimes(results_equilibrium['tt_x'])

        for link in network.links:

            #Store true travel time
            link.true_traveltime = link.traveltime

            # Set link travel times equal to free flow
            link.set_traveltime_from_x(x=0)

        x_observed = dict(zip(results_equilibrium['x'].keys(), observed_link_counts.flatten()))
        x_withdrawn = dict(zip(results_equilibrium['x'].keys(), withdrawn_link_counts.flatten()))

        # Report ratio of total counts versus capacity
        total_counts = 0
        total_capacity = 0
        total_links_over_capacity = 0

        for key, counts  in x_observed.items():

            if not np.isnan(counts):

                total_counts += counts
                link_capacity = network.links_dict[key].bpr.k
                total_capacity += link_capacity

                if counts > link_capacity:
                    total_links_over_capacity += 1

        # Add this number in experiment and estimation report
        self.options['ratio_counts_capacity'] = "{:.1%}".format(total_counts/total_capacity)
        self.options['x_links_over_capacity'] = "{:.1%}".format(total_links_over_capacity / len(network.links))
        self.options['nrmse_val'] = np.round(nrmse(actual = link_counts, predicted = true_link_counts),3)

        print('Ratio of counts versus capacity:', self.options['ratio_counts_capacity'])
        print('Proportion of links over capacity:', self.options['x_links_over_capacity'])
        print('Normalized RMSE:', self.options['nrmse_val'])

        return x_observed, x_withdrawn

    def generate_fixed_effects(self) -> Dict[Link, Dict[Feature, float]]:
        raise NotImplementedError


    def add_fixed_effects_attributes(self,
                                     network,
                                     fixed_effects,
                                     observed_links: str = None,
                                     links_keys: [tuple] = None):

        '''
            :argument fixed_effects: by q (direction matters) or nodes (half because it does not matter the direction of the links)

            notes: fixed effect at the od or nodes pair level are not identifiable if using data from one time period only
        '''

        coverage = fixed_effects['coverage']

        # i) Links

        if fixed_effects['links']:

            # Store list of fixed effects
            k_fixed_effects = []

            if observed_links == 'random':

                observed_link_idxs = []

                for link, i in zip(network.links, np.arange(len(network.links))):

                    if not np.isnan(link.observed_count):
                        # print(link.key)
                        observed_link_idxs.append(i)

                n_coverage = int(np.floor(len(observed_link_idxs) * coverage))

                idxs = np.random.choice(np.arange(len(observed_link_idxs)), size=n_coverage, replace=False)

                selected_links = [network.links[observed_link_idxs[idx]] for idx in idxs]

            elif observed_links == 'custom':

                selected_links = [network.links_dict[link_key] for link_key in links_keys]

            else:

                n_coverage = int(np.floor(len(network.links) * coverage))

                idxs = np.random.choice(np.arange(len(network.links)), size=n_coverage, replace=False)

                selected_links = [network.links[idx] for idx in idxs]

            for selected_link in selected_links:

                attr_lbl = 'l' + str(selected_link.key[0]) + '-' + str(selected_link.key[1])

                for link_key, link in network.links_dict.items():
                    if link_key[0] == link.key[0] and link_key[1] == selected_link.key[1]:
                        link.Z_dict[attr_lbl] = 1
                    else:
                        link.Z_dict[attr_lbl] = 0

                k_fixed_effects.append(attr_lbl)

        # ii) OD matrix

        if fixed_effects['Q'] or fixed_effects['nodes']:

            n_coverage = int(np.floor(len(network.A.nonzero()[0]) * coverage))

            idxs = np.random.choice(np.arange(len(network.A.nonzero()[0])), size=n_coverage, replace=False)

            selected_idxs = [(network.A.nonzero()[0][idx], network.A.nonzero()[1][idx]) for idx in idxs]

            # Store list of fixed effects
            k_fixed_effects = []

            if fixed_effects['Q']:

                for (i, j) in selected_idxs:
                    attr_lbl = 'q' + str(i + 1) + '-' + 'q' + str(j + 1)

                    for link_i, link in network.links_dict.items():
                        if link_i[0] == i and link_i[1] == j:
                            link.Z_dict[attr_lbl] = 1
                        else:
                            link.Z_dict[attr_lbl] = 0

                    k_fixed_effects.append(attr_lbl)

            # iii) Pair of nodes

            if fixed_effects['nodes']:
                for (i, j) in selected_idxs:

                    attr_lbl = ''
                    if i > j:
                        attr_lbl = 'n' + str(i + 1) + ',' + 'n' + str(j + 1)
                    if i < j:
                        attr_lbl = 'n' + str(j + 1) + ',' + 'n' + str(i + 1)

                    for link_i, link in network.links_dict.items():
                        if link_i[0] == i and link_i[1] == j or link_i[0] == j and link_i[1] == i:
                            link.Z_dict[attr_lbl] = 1
                        else:
                            link.Z_dict[attr_lbl] = 0

                    k_fixed_effects.append(attr_lbl)

        return k_fixed_effects



class PathsGenerator(Generator):

    def __init__(self,
                 network = None,
                 utility_function = None,
                 Q = None,
                 **kwargs):

        super().__init__(**kwargs)

        self.network = network
        self.utility_function = utility_function

        self.Q = Q

        if network is not None:
            self.Q = self.network.Q

        # assert self.Q is not None, 'Network has not an O-D matrix'

        pass

    def set_default_options(self):

        ''':cvar

        options: cutoff paths, n_paths, etc.

        '''

        self.options = Options()

        # Maximum numebr of paths per od for initial paths
        self.options['n_paths'] = 1

        # Maximum number of links for path generation
        self.options['cutoff'] = 50

        # If no path is found with the cutoff, the latter is increased by the following factor
        self.options['cutoff_increase_factor'] = 10

        # Maximum number of times that the cutoff is increased by the cutoff factor.
        self.options['max_attempts'] = 10

    def k_shortest_paths(self,
                         k: int = None,
                         network = None,
                         theta = None,
                         ods = None,
                         **kwargs):
        '''
        It assumes that the network has an OD matrix

        Args:
            k: number of paths_per_od
            **kwargs:

        Returns:

        '''

        # print('Generating paths')

        kwargs = self.options.get_updated_options(new_options = kwargs)

        if network is None:
            network = self.network

        if ods is None:
            ods = network.ods

        # if theta is None:
        #     theta = self.utility_function.values()

        Q = self.Q = kwargs.get('Q', network.Q)

        if k is not None:
            kwargs['n_paths'] = k


        assert network is not None, 'network is None'

        assert Q is not None, 'Network has not an O-D matrix'

        # assert utility_function is not None, 'utility_function is None'

        if theta is not None:

            # Matrix with link utilities
            network.V = network.generate_V(A=network.A,
                                           links=network.links,
                                           theta= theta)

            # Key to have the minus sign so we look the route that lead to the lowest disutility
            edge_utilities = network.generate_edges_weights(V=network.V)

            # Generate new paths according to arbitrary size given by n_paths
            paths, paths_od = k_path_generation_nx(A= network.A,
                                                   ods= ods,
                                                   links= network.links_dict,
                                                   edge_weights = edge_utilities,
                                                   **kwargs
                                                   )

        else:
            # This uses the number of links for shortest paths only
            paths, paths_od = k_path_generation_nx(A=network.A,
                                                   ods=ods,
                                                   links=network.links_dict,
                                                   **kwargs)

        return paths, paths_od


    def load_k_shortest_paths(self,
                              network,
                              **kwargs):

        paths, paths_od = self.k_shortest_paths(network = network, **kwargs)

        network.load_paths(paths = paths, paths_od = paths_od,
                           update_incidence_matrices = kwargs.get('update_incidence_matrices', True))


    @staticmethod
    def get_paths_from_paths_od(paths_od):

        paths_list = []

        for od,paths in paths_od.items():
            # This solves the problem to append paths when there is only one path per OD
            paths_list.extend(list(paths))

        return paths_list

    @staticmethod
    def get_paths_od_from_paths(paths):

        return get_paths_od_from_paths(paths)

    def read_paths(self, network, **kwargs):

        # options = self.get_updated_options(**{'reading':{'paths':True}})
        #
        # if options['reading']['paths']:
        #     # print('reading paths')

        paths = read_internal_paths(network=network,
                                    filename = kwargs.get('filename', None),
                                    folderpath = kwargs.get('folderpath', None))
        network.load_paths(paths=paths, update_incidence_matrices=kwargs.get('update_incidence_matrices', False))

    def write_paths(self, network, **kwargs):

        # options = self.get_updated_options(**{'writing':{'paths':True}})
        #
        # write_internal_network_files(network=network, options=options, **kwargs)
        write_internal_paths(network.paths, network.key, **kwargs)

        if kwargs.get('overwrite_input', False):
            kwargs['overwrite_input'] = False
            write_internal_paths(network.paths, network.key, **kwargs)
            # write_internal_network_files(network=network, options=options, **kwargs)


class ODGenerator(Generator):

    def __init__(self,
                 network = None,
                 path_generator = None,
                 **kwargs):

        super().__init__(**kwargs)

        self.network = network
        self.path_generator = path_generator



    def set_default_options(self):

        self.options = Options()

        # self.options['min_Q'] = 0
        # self.options['max_Q'] = 0



    # @staticmethod
    def generate_Q(self,
                   network: TNetwork,
                   min_q: float = 0,
                   max_q: float = 0,
                   cutoff: float = 1e5,
                   n_paths =1,
                   sparsity_Q: Proportion = 0.0):

        '''

        Args:
            network:
            min_q:  Range of values on a single cell of the OD matrix of the random networks
            max_q: Range of values on a single cell of the OD matrix of the random networks
            cutoff:
            n_paths:
            sparsity_Q:

        Returns:

        '''

        print('Generating matrix Q')

        t0 = time.time()

        # G = nx.DiGraph(network.A)

        Q_mask = np.random.randint(min_q,
                                   max_q,
                                   network.A.shape)
        Q = np.zeros(Q_mask.shape)

        # Set terms to 0 if there is no a path in between nodes on the graph produced by A

        nonzero_entries_Q_mask = list(Q_mask.nonzero())
        random.shuffle(nonzero_entries_Q_mask)

        total_entries_Q_mask = Q.shape[0] ** 2
        expected_non_zero_Q_entries = int(total_entries_Q_mask * (1 - sparsity_Q))

        print('The expected number of matrix entries to fill out is ' + str(
            expected_non_zero_Q_entries) + '. Sparsity: ' + "{:.0%}".format(sparsity_Q))

        # total = len(nonzero_Q_entries)
        counter = 0

        for (i, j) in zip(*tuple(nonzero_entries_Q_mask)):

            # Q = isl.config.sim_options['custom_networks']['A']['N2']
            # print((i,j))

            printer.printProgressBar(counter, expected_non_zero_Q_entries, prefix='Progress:', suffix='', length=20)

            # Very ineficient as it requires to enumerate all paths
            # if len(list(nx.all_simple_paths(G, source=i, target=j, cutoff=cutoff))) == 0:

            k_paths_od = k_simple_paths_nx(k=n_paths,
                                           source=i,
                                           target=j,
                                           cutoff=cutoff,
                                           G=network.G,
                                           links=network.links_dict)

            if len(list(k_paths_od)) == n_paths:

                Q[(i, j)] = Q_mask[(i, j)]

                counter += 1

                if counter > expected_non_zero_Q_entries:
                    break
            else:
                # print('No paths with less than ' + str(cutoff) + ' links were found in o-d pair ' + str((i,j)))
                pass

        non_zero_entries_final_Q = np.count_nonzero(Q != 0)  # / float(Q.size)

        sparsity_final_Q = 1 - non_zero_entries_final_Q / float(Q.size)

        #  Fill the Q matrix with zeros according to the degree of sparsity. As higher is the sparsity, faster is the generation of the Q matrix because there is less computation of shortest paths below

        # example: https://stackoverflow.com/questions/40058912/randomly-controlling-the-percentage-of-non-zero-values-in-a-matrix-using-python

        # idx = np.flatnonzero(Q)
        # N = np.count_nonzero(Q != 0) - int(round((1-sparsity) * Q.size))
        # np.put(Q, np.random.choice(idx, size=N, replace=False), 0)

        assert Q.shape[0] > 0, 'Matrix Q could not be generated'

        print(
            str(non_zero_entries_final_Q) + ' entries were filled out. Sparsity: ' + '{:.0%}'.format(sparsity_final_Q))

        print('Matrix Q ' + str(Q.shape) + ' generated in ' + str(round(time.time() - t0, 1)) + '[s]')

        return Q

    # @staticmethod
    def random_disturbance_Q(self,
                             *args,
                             new_ods=False,
                             **kwargs):
        '''Add a random disturbance to OD matrix but only for od pairs with trips to avoid generating new paths.
        If new_ods is true, then path generation would need to be done again
        '''

        Q = random_disturbance_Q(*args,**kwargs)

        return Q


class NetworkGenerator(Generator):

    def __init__(self,
                 **kwargs):

        super().__init__(**kwargs)
        self.links = None

    def get_updated_options(self, **kwargs):
        return copy.deepcopy(self.options.get_updated_options(new_options=kwargs))

    def set_default_options(self):

        self.options = Options()

        # range for number of nodes used to create random network
        self.options['nodes_range'] = np.arange(1, 6)

        self.options['fixed_effects'] = False

        self.options['generation'] = {'C': True, 'D': True, 'M': True}

        self.options['reading'] = {'paths': False, 'C': False, 'D': False, 'M': False, 'Q': False,
                                  'sparse_C': False, 'sparse_D': False, 'sparse_M': False, 'sparse_Q': False}

        self.options['writing'] = {'paths': False, 'C': False, 'D': False, 'M': False, 'Q': False,
                                  'sparse_C': False, 'sparse_D': False, 'sparse_M': False, 'sparse_Q': False}


        # min_q, max_q = setup_options['q_range']
        # q_sparsity = setup_options['q_sparsity']
        # n_paths = setup_options['n_initial_paths']

    def build_network(self,
                      network_name: str,
                      A: Matrix,
                      links: Optional[List[Links]] = None,
                      **kwargs) -> TNetwork:
        """ Construct a transportation network

        Args:
            label (Dict):
                name for the created network
            factory_options (Dict):
                tbc
            A (Dict[str, Matrix]):
                the adjacency matrix
            links (Optional[Links]): list of Link objects

        Returns:
            TNetwork: the transportation network object

        Notes:
            If links are none, they are created based on the adjacency matrix

            returns a Network object with an adjacency matrix and with a list of links. If links were not provided, they are created
         """

        # # Options
        # options = self.get_updated_options(**kwargs)

        N = None

        print('\n' + 'Creating ' + str(network_name) + ' network\n')

        multinetwork = (A > 1).any()
        dinetwork = (A <= 1).any()

        if multinetwork:
            N = MultiDiTNetwork(A=A, links=links)
        elif dinetwork:
            N = DiTNetwork(A=A, links=links)

        N.key = network_name

        print('Nodes: ' + str(N.get_n_nodes()) + ', Links: ' + str(N.get_n_links()))

        return N

        # - Validation (TODO: Create tests with these matrices)

        # q1 = np.array([10])  # 1-2
        # N['N1'].M = np.array([1, 1, 1])[np.newaxis, :]
        # N['N1'].D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # q2 = np.array([10, 20, 30])  # 1-4, 2-4, 3-4
        # N['N2'].M = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]])
        # N['N2'].D = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])

        # q3 = np.array([10, 20])  # 1-3, 2-3
        # N['N3'].M = np.array([[1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1]])
        # N['N3'].D = np.array([[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0],
        #                [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]]
        #               )

    def build_random_network(self,
                             network_name: str,
                             **kwargs):

        """ Create a dictionary of network objects with a random adjacency matrices and their corresponding set of links """

        # for key, value in kwargs.items():
        #     options[key] = value

        options = self.get_updated_options(**kwargs)

        nodes_range = options['nodes_range']

        A = TNetwork.randomDiNetwork(n_nodes=random.randint(nodes_range[0], nodes_range[-1])).A

        return self.build_network(network_name= network_name, A=A)

    def build_tntp_network(self, subfoldername, folderpath, options, config, **kwargs):

        # subfoldersnames = config.tntp_networks
        #
        # assert subfoldername in subfoldersnames, 'Invalid network name'

        # for key, value in kwargs.items():
        #     options[key] = value

        options = self.get_updated_options(**kwargs)

        # print(options['writing'])

        # Write dat files
        write_tntp_github_to_dat(folderpath, subfoldername)

        # Read input dat file
        A_real, links_attrs = read_tntp_network(folderpath=folderpath, subfoldername=subfoldername)

        # Create network object
        Nt_label = subfoldername

        Nt = self.build_network(factory_options=options, A=A_real, network_name=Nt_label)

        links = Nt.links_dict
        # Set BPR functions and attributes values associated each link
        for index, row in links_attrs.iterrows():
            # TODO: Review if the parameters from the data have a consistent meaning with the way the BPR function is written in my code
            link_label = (int(row['init_node']), int(row['term_node']), '0')

            # BPR functions
            links[link_label].performance_function = BPR(alpha=row['b'], beta=row['power'],
                                                         tf=row['free_flow_time'], k=row['capacity'])

            # Attributes
            links[link_label].Z_dict['speed'] = pd.to_numeric(row['speed'], errors='coerce',
                                                              downcast='float')  # It ssemes to be the speed limit
            links[link_label].Z_dict['toll'] = pd.to_numeric(row['toll'], errors='coerce', downcast='float')
            links[link_label].Z_dict['length'] = pd.to_numeric(row['length'], errors='coerce', downcast='float')

            # Replace Nan values with nas
            for i in ['speed', 'toll', 'length']:
                if np.isnan(links[link_label].Z_dict[i]):
                    links[link_label].Z_dict[i] = float('nan')  # float(0)
            # Identical/Similar to free flow travel time

        # Write paths if required
        # print('here')
        # print(len(network.paths_od.keys()))

        # print('here 2')
        # print(len(network.paths_od.keys()))

        # # Generate D and M matrix
        # network.M = network.generate_M(paths_od=network.paths_od, paths=network.paths)
        # network.D = network.generate_D(paths_od=network.paths_od, links=network.links, paths=network.paths)
        #
        # # Generate abd store choice set matrix
        # network.C = estimation.choice_set_matrix_from_M(network.M)

        # reader.network_reader(network, options)
        # writer.network_writer(network, options)

        return Nt

    def build_colombus_network(self, folder, label, options, **kwargs):

        # # Write dat files
        # writer.write_tntp_github_to_dat(folder, network_name)

        # print(read_paths)

        # folder = isl.config.paths['Colombus_network']

        for key, value in kwargs.items():
            options[key] = value

        # Read files
        A, links_df, nodes_df = read_colombus_network(
            folderpath=folder)

        # Create link objects and set BPR functions and attributes values associated each link
        links = {}

        for index, row in links_df.iterrows():
            # TODO: Review if the parameters from the data have a consistent meaning with the way the BPR function is written in my code

            link_key = (int(row['init_node_key']), int(row['term_node_key']), '0')

            # Adding gis information via nodes object store in each link
            init_node_row = nodes_df[nodes_df['key'] == link_key[0]]
            term_node_row = nodes_df[nodes_df['key'] == link_key[1]]

            # TODO: double check with Bin about the matching of the network elements and the shape file information. He told me that the ids did not perfectly match and that I should look at the jupyter notebook

            # x_cord_origin, y_cord_origin = tuple(list(init_node_row[['x', 'y']].values[0]))
            # x_cord_term, y_cord_term = tuple(list(term_node_row[['x', 'y']].values[0]))
            #
            # node_init = Node(key=link_key[0], position=NodePosition(x_cord_origin, y_cord_origin, crs='xy'))
            # node_term = Node(key=link_key[1], position=NodePosition(x_cord_term, y_cord_term, crs='xy'))

            node_init = Node(key=link_key[0])
            node_term = Node(key=link_key[1])

            links[link_key] = Link(key=link_key, init_node=node_init, term_node=node_term)

            # Store original ids from nodes and links
            links[link_key].init_node.id = str(init_node_row['id'].values[0])
            links[link_key].term_node.id = str(term_node_row['id'].values[0])
            # note that some ids include a large tab before the number comes up ('   1), I may remove those spaces
            links[link_key].id = row['id']

            # Attributes
            links[link_key].Z_dict['capacity_car'] = pd.to_numeric(row['capacity_car'], errors='coerce',
                                                                   downcast='float')
            links[link_key].Z_dict['capacity_truck'] = pd.to_numeric(row['capacity_truck'], errors='coerce',
                                                                     downcast='float')
            links[link_key].Z_dict['ff_speed_car'] = pd.to_numeric(row['ff_speed_car'], errors='coerce',
                                                                   downcast='float')
            links[link_key].Z_dict['ff_speed_truck'] = pd.to_numeric(row['ff_speed_truck'], errors='coerce',
                                                                     downcast='float')
            links[link_key].Z_dict['rhoj_car'] = pd.to_numeric(row['rhoj_car'], errors='coerce', downcast='float')
            links[link_key].Z_dict['rhoj_truck'] = pd.to_numeric(row['rhoj_truck'], errors='coerce', downcast='float')
            links[link_key].Z_dict['length'] = pd.to_numeric(row['length'], errors='coerce', downcast='float')
            links[link_key].Z_dict['lane'] = pd.to_numeric(row['lane'], errors='coerce', downcast='integer')
            links[link_key].Z_dict['conversion_factor'] = pd.to_numeric(row['conversion_factor'], errors='coerce',
                                                                        downcast='float')

            # We assume that the free flow speed and capacity are those associated to cars and not trucks
            links[link_key].Z_dict['capacity'] = links[link_key].Z_dict['capacity_car']
            links[link_key].Z_dict['ff_speed'] = links[link_key].Z_dict['ff_speed_car']

            # BPR functions
            # Parameters of BPR function are assumed to be (alpha, beta) = (0.15, 4). Source: https://en.wikipedia.org/wiki/Route_assignment

            links[link_key].performance_function \
                = BPR(alpha=0.15, beta=4
                      , tf=links[link_key].Z_dict['ff_speed']
                      , k=links[link_key].Z_dict['capacity'])

        # Create network object
        Nt = self.build_network(factory_options=options, A={label: A}
                           , labels={label: label}, links=list(links.values()))[label]

        # TODO: Optimize the generation of the path link incidence matrix as it is taking too long for Ohio. Sparse matrix and a better index method to index links may help.
        # network.D = network.generate_D(paths_od=network.paths_od, links = network.links, paths = network.paths)

        return Nt

    def build_fresno_network(self,
                             A,
                             links_df,
                             nodes_df,
                             network_name,
                             **kwargs):

        # Create network object
        # TODO: Run shortest path with Igraph and Graphx for dinetworks. This can speed up this execution. Alternatively, the paths could be read and generated from path file

        network = self.build_network(A=A
                                     , network_name=network_name,
                                     **kwargs)

        # Create link objects and set BPR functions and attributes values associated each link
        network.links_dict = {}
        network.nodes_dict = {}

        for index, row in links_df.iterrows():
            # TODO: Review if the parameters from the data have a consistent meaning with the way the BPR function is written in my code

            # print(row)
            # link_key = (int(row['init_node_key']), int(row['term_node_key']), '0')
            link_key = row['link_key']

            # Adding gis information via nodes object store in each link
            init_node_row = nodes_df[nodes_df['key'] == link_key[0]]
            term_node_row = nodes_df[nodes_df['key'] == link_key[1]]

            x_cord_origin, y_cord_origin = tuple(list(init_node_row[['x', 'y']].values[0]))
            x_cord_term, y_cord_term = tuple(list(term_node_row[['x', 'y']].values[0]))

            if link_key[0] not in network.nodes_dict.keys():
                network.nodes_dict[link_key[0]] = Node(key=link_key[0],
                                                    position=NodePosition(x_cord_origin, y_cord_origin, crs='xy'))

            if link_key[1] not in network.nodes_dict.keys():
                network.nodes_dict[link_key[1]] = Node(key=link_key[1],
                                                    position=NodePosition(x_cord_term, y_cord_term, crs='xy'))

            node_init = network.nodes_dict[link_key[0]]
            node_term = network.nodes_dict[link_key[1]]

            # links[link_key] = Link(key= link_key, init_node = node_init, term_node = node_term)
            network.links_dict[link_key] = Link(key=link_key, init_node=node_init, term_node=node_term)

            # Store original ids from nodes and links
            network.links_dict[link_key].init_node.id = str(init_node_row['id'].values[0])
            network.links_dict[link_key].term_node.id = str(term_node_row['id'].values[0])
            # note that some ids include a large tab before the number comes up ('   1), I may remove those spaces
            network.links_dict[link_key].id = row['id']

        # TODO: Optimize the generation of the path link incidence matrix as it is taking too long for Ohio. Sparse matrix and a better index method to index links may help.
        # network.D = network.generate_D(paths_od=network.paths_od, links = network.links, paths = network.paths)

        # Update LinkData object


        return network

    def build_sacramento_network(self, folder, label, options, **kwargs):

        # TODO: Take advantage of nodes information

        for key, value in kwargs.items():
            options[key] = value

        # Read files
        A, links_df, nodes_df = read_sacramento_network(
            folderpath=folder)

        # Create link objects and set BPR functions and attributes values associated each link
        links = {}

        for index, row in links_df.iterrows():
            # TODO: Review if the parameters from the data have a consistent meaning with the way the BPR function is written in my code

            # print(row)
            link_key = (int(row['init_node_key']), int(row['term_node_key']), '0')

            # Adding gis information via nodes object store in each link
            init_node_row = nodes_df[nodes_df['key'] == link_key[0]]
            term_node_row = nodes_df[nodes_df['key'] == link_key[1]]

            x_cord_origin, y_cord_origin = tuple(list(init_node_row[['x', 'y']].values[0]))
            x_cord_term, y_cord_term = tuple(list(term_node_row[['x', 'y']].values[0]))

            node_init = Node(key=link_key[0], position=NodePosition(x_cord_origin, y_cord_origin, crs='xy'))
            node_term = Node(key=link_key[1], position=NodePosition(x_cord_term, y_cord_term, crs='xy'))

            # links[link_key] = Link(key= link_key, init_node = node_init, term_node = node_term)

            links[link_key] = Link(key=link_key, init_node=node_init, term_node=node_term)

            # Store original ids from nodes and links
            links[link_key].init_node.id = str(init_node_row['id'].values[0])
            links[link_key].term_node.id = str(term_node_row['id'].values[0])
            # note that some ids include a large tab before the number comes up ('   1), I may remove those spaces
            links[link_key].id = row['id']

            # Attributes
            links[link_key].Z_dict['capacity'] = pd.to_numeric(row['capacity'], errors='coerce', downcast='float')
            links[link_key].Z_dict['ff_speed'] = pd.to_numeric(row['ff_speed'], errors='coerce', downcast='float')
            links[link_key].Z_dict['rhoj'] = pd.to_numeric(row['rhoj'], errors='coerce', downcast='float')
            links[link_key].Z_dict['length'] = pd.to_numeric(row['length'], errors='coerce', downcast='float')
            links[link_key].Z_dict['lane'] = pd.to_numeric(row['lane'], errors='coerce', downcast='integer')

            # BPR functions
            # Parameters of BPR function are assumed to be (alpha, beta) = (0.15, 4). Source: https://en.wikipedia.org/wiki/Route_assignment
            links[link_key].performance_function \
                = BPR(alpha=0.15, beta=4
                      , tf=links[link_key].Z_dict['ff_speed']
                      , k=links[link_key].Z_dict['capacity'])

        # Create network object
        # TODO: Run shortest path with Igraph and Graphx for dinetworks. This can speed up this execution. Alternatively, the paths could be read and generated from path file

        Nt = self.build_network(factory_options=options, A={label: A}
                           , labels={label: label}, links=list(links.values()))[label]

        # TODO: Optimize the generation of the path link incidence matrix as it is taking too long for Ohio. Sparse matrix and a better index method to index links may help.
        # network.D = network.generate_D(paths_od=network.paths_od, links = network.links, paths = network.paths)

        return Nt


    def generate_C(self, **kwargs):
        """Wide to long format
        Choice_set_matrix_from_M
        The new matrix has one rows per alternative
        """
        return TNetwork.generate_C(**kwargs)

    def generate_D(self, **kwargs):
        """Matrix D: Path-link incidence matrix"""

        return TNetwork.generate_D(**kwargs)

    def generate_M(**kwargs):
        """Matrix M: Path-OD pair incidence matrix"""

        return TNetwork.generate_M(**kwargs)

    def generate_V(self, **kwargs):

        """ Matrix with link utilities with the same shape than the adjacency matrix """

        return TNetwork.generate_V(**kwargs)

    def update_incidence_matrices(self, network, paths_od = None):

        if paths_od is None:
            paths_od = network.paths_od

        print('Updating incidence matrices')

        # printer.blockPrint()

        network.D = self.generate_D(paths_od=paths_od, links=network.links)
        network.M = self.generate_M(paths_od=paths_od)
        network.C = self.generate_C(M = network.M)

        # printer.enablePrint()

    def setup_incidence_matrices(self,
                                 network: TNetwork,
                                 M: Matrix = None,
                                 D: Matrix = None,
                                 **kwargs):
        '''
        Create network with random attributes and Q matrix if required. The mandatory input if a Tnetwork object

        # - Set values of attributes in matrix Z, including those sparse (n_R) or not.

        # Incidence matrices are created

        This function allows to deal with reading data from files which are generated by this package

        :argument randomness: dictionary of booleans with keys ['Q', 'BPR', 'Z']
        :argument network: dictionary of networks

        #TODO: I may combine this method with the custom_factory to avoid dupplication

        TODO: network object should store all the input in a dictionary called 'setup_options'

        '''

        options = self.get_updated_options(**kwargs)

        print('\n' + 'Setting up network matrices in ' + str(network.key) + '\n')

        # min_q, max_q = setup_options['q_range']
        # q_sparsity = setup_options['q_sparsity']
        # n_paths = setup_options['n_initial_paths']

        # Store setup options
        network.setup_options = options

        # Call reader
        read_internal_network_files(network, options)

        # if setup_options['generation']['Q'] and Q is None:

        # TODO: provide paths per od to generate Q, otherwise is very expensive to generate this matrix. If paths are not provided, then path generation should be performed.

        # network.Q = network.generate_Q(network=network, min_q=min_q, max_q=max_q, cutoff=cutoff_paths, n_paths = n_paths, sparsity = q_sparsity)

        # if Q is not None:
        #     network.Q = Q

        # assert network.Q.shape[0] > 0, 'Invalid matrix Q was provided or it could not be generated'
        # print('Matrix Q was successfuly generated')

        # # Dense od vector
        # network.q = denseQ(network.Q, remove_zeros=remove_zeros_Q)
        #
        # # Store od pairs
        # network.ods = network.ods_fromQ(Q=network.Q, remove_zeros=remove_zeros_Q)

        # print(str(len(network.ods)) + ' o-d pairs')

        # TODO: Enable path reading and writing

        # Write paths if required
        # if setup_options['writing']['paths']:
        #     writer.write_paths(network.paths, network.label)
        #
        #     # Links in the network should be updated so paths and links are associated as wished.
        # if setup_options['generation']['paths'] and paths is None:
        # print('No paths provided... Generating paths ...')

        # network.paths, network.paths_od = paths_generator.generate_paths(network = network, Q = Q)

        # Network matrices

        # - Path-OD incidence matrix D

        if options['generation']['M'] and M is None:
            network.M = network.generate_M(paths_od=network.paths_od)

        if M is not None:
            network.M = M

        # assert network.M.shape[0] > 0, 'Invalid matrix M was provided or it could not be generated'

        # - Path-link incidence matrix D
        if options['generation']['D'] and D is None:
            network.D = network.generate_D(paths_od=network.paths_od, links=network.links)

        if D is not None:
            network.D = D

        # assert network.D.shape[0] > 0, 'Invalid matrix D was provided or it could not be generated'

        # Choice set matrix
        if options['generation']['C']:
            network.C = self.generate_C(M = network.M)

        # assert network.C.shape[0] > 0, 'Choice set matrix could not be generated'

        # Call writer
        write_internal_network_files(network = network, options = options)

        return network  # results_sue, exceptions, Y_links, Y_routes, Z_links, Z_routes

    def setup_tntp_network_matrices(self,
                                    network: TNetwork,
                                    setup_options: Options,
                                    **kwargs):

        """ This function allows to deal with reading data from tntp files which are not generated by this package """

        print('\n' + 'Setting up ' + str(network.key) + ' network \n')

        for key, value in kwargs.items():
            # print(key)
            setup_options[key] = value

        # Q = None
        #
        # if setup_options['reading']['Q'] and not setup_options['reading']['sparse_Q']:
        #     # print('here reading ')
        #
        #     Q = reader.read_tntp_od(folderpath=setup_options['folder'], network_name=setup_options['network_name'])
        #
        #     # Do not need to read again using internal reader
        #     setup_options['reading']['Q'] = False

        # The matrix Q was read and store in network object using tntp_factory method
        network = self.setup_incidence_matrices(network, message=False, setup_options=setup_options)

        return network

    def setup_colombus_network(self, Nt: TNetwork, setup_options: Options, **kwargs):
        """ This function allows to deal with reading data from colombus files which are not generated by this package """

        for key, value in kwargs.items():
            # print(key)
            setup_options[key] = value

        print('\n' + 'Setting up network ' + str(Nt.key) + '\n')

        Q = None

        # TODO: reading should be done later only from internal files. After the reading of external files is done, a translation method should write them in the internal format

        # Create paths using dictionary of existing links and information read from txt
        if setup_options['reading']['paths']:
            reader.read_colombus_paths(network=Nt, filepath=setup_options['folder'] + '/ODE_outputs/path_table')

            # Do not need to read again using internal reader
            setup_options['reading']['paths'] = False

        if setup_options['reading']['Q'] and not setup_options['reading']['sparse_Q']:
            Q = reader.read_colombus_od(A=Nt.A, nodes=Nt.nodes, folderpath=setup_options['folder'])

            # Do not need to read again using internal reader
            setup_options['reading']['Q'] = False

        # The matrix Q was read and store in network object using colombus_factory method
        Nt = self.setup_incidence_matrices(Nt, Q=Q, message=False, setup_options=setup_options)

        return Nt

    def setup_sacramento_network(self, network: TNetwork, setup_options: Options, **kwargs):
        """ This function allows to deal with reading data from colombus files which are not generated by this package """

        for key, value in kwargs.items():
            # print(key)
            setup_options[key] = value

        print('\n' + 'Setting up network ' + str(network.key) + '\n')

        Q = None

        # TODO: reading should be done later only from internal files. After the reading of external files is done, a translation method should write them in the internal format

        if setup_options['reading']['Q'] and not setup_options['reading']['sparse_Q']:
            Q = read_sacramento_od(A=network.A, folder=setup_options['folder'], nodes=network.nodes)

            # Do not need to read again using internal reader
            setup_options['reading']['Q'] = False

        setup_options['folder'] = setup_options['folder'] + '/sac'

        # The matrix Q was read and store in network object using colombus_factory method
        network = self.setup_incidence_matrices(network, Q=Q, message=False, setup_options=setup_options)

        return network

    def clone_network(self, network: TNetwork, **kwargs) -> TNetwork:
        '''

        :param network: Single network
        :param kwargs:
        :return:
        '''

        # TODO: Fix clone method under new modifications of setup and factory methods

        clone_options = network.setup_options

        for key, value in kwargs.items():
            clone_options[key] = value
            # print("%s == %s" % (key, value))

        # Default for randomness
        # clone_options['randomness'] = {'Q': False, 'BPR': False, 'Z': False}

        # Q = copy.deepcopy(N.Q)
        # A = copy.deepcopy(N.A)
        # N.links[0].bpr.bpr_function_x(0)
        # print(N.links[0].bpr.bpr_function_x(0))

        N_copy = self.build_network(A={network.key: network.A}, Q={network.key: network.Q}
                                    , M={network.key: network.M}, D={network.key: network.D}
                                    , links=network.links, paths_od={network.key: network.paths_od}
                                    , labels={network.key: clone_options['label']}
                                    , factory_options=clone_options)

        # print(N_copy[N.label].links[0].bpr)

        N_copy = self.setup_incidence_matrices(network=N_copy, **clone_options)

        return N_copy[network.key]

    def generate_adjacency_matrix(self, links_keys: List[tuple]):

        init_nodes = []
        term_nodes = []

        for init_node, term_node, _ in links_keys:
            init_nodes.append(init_node)
            term_nodes.append(term_node)

        # Create adjacency matrix
        # dimension_A = len(links_attrs['init_node'].append(links_attrs['term_node']).unique())
        # dimension_A = links_attrs['init_node'].append(links_attrs['term_node']).unique().max() + 1
        dimension_A = max(init_nodes+term_nodes) + 1

        A = np.zeros([dimension_A, dimension_A])

        for init_node, term_node in zip(init_nodes,term_nodes):
            A[(int(init_node), int(term_node))] = 1

        return A

    def read_incidence_matrices(self, network, matrices:Dict[str, bool] = None):

        new_options = {'reading':{}}

        if matrices is None:
            new_options['reading'] = dict.fromkeys(['C','D','M'],True)
        else:
            new_options['reading'] = dict.fromkeys(['C', 'D', 'M', 'sparse_C', 'sparse_D', 'sparse_M'], False)

        for k,v in matrices.items():
            if k in new_options['reading'].keys():
                new_options['reading'][k] = v

        options = self.get_updated_options(**new_options)

        return read_internal_network_files(network = network, options = options)

    def read_OD(self,network, sparse: bool = False):

        if sparse:
            options = self.get_updated_options(**{'reading': {'sparse_Q': True}})
        else:
            options = self.get_updated_options(**{'reading': {'Q': True}})

        return read_internal_network_files(network=network, options=options)

    def write_incidence_matrices(self, network, matrices:Dict[str, bool] = {}, **kwargs):

        new_options = {'writing':{}}
        # new_options['writing'] = dict.fromkeys(['C','D','M'],True)

        if matrices is None:
            new_options['writing'] = dict.fromkeys(['C', 'D', 'M'], True)
        else:
            new_options['writing'] = dict.fromkeys(['C', 'D', 'M', 'sparse_C', 'sparse_D', 'sparse_M'], False)

        for k,v in matrices.items():
            if k in new_options['writing'].keys():
                new_options['writing'][k] = v

        options = self.get_updated_options(**new_options)

        write_internal_network_files(network = network, options = options, **kwargs)

        if kwargs.get('overwrite_input',False):
            kwargs['overwrite_input'] = False
            write_internal_network_files(network=network, options=options, **kwargs)



    def write_OD_matrix(self, network, sparse: bool = None, **kwargs):

        if sparse is None:
            options = self.get_updated_options(**{'writing': {'Q': True, 'sparse_Q': True}})

        else:
            if sparse:
                options = self.get_updated_options(**{'writing': {'sparse_Q': True}})
            else:
                options = self.get_updated_options(**{'writing': {'Q': True}})

        write_internal_network_files(network=network, options=options, **kwargs)

        if kwargs.get('overwrite_input',False):
            kwargs['overwrite_input'] = False
            write_internal_network_files(network=network, options=options, **kwargs)

    @staticmethod
    def generate_A_Q_custom_networks():

        A, Q, links_tuples = {}, {}, {}

        A['N1'] = np.array([[0, 3], [0, 0]])
        Q['N1'] = np.array([[0, 100], [0, 0]])

        A['N2'] = np.array([[0, 3, 0], [0, 0, 2], [0, 0, 0]])
        Q['N2'] = np.array([[0, 0, 100], [0, 0, 200], [0, 0, 0]])

        A['N4'] = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]])
        Q['N4'] = np.array([[0, 0, 100, 200], [0, 0, 300, 400], [0, 0, 0, 500], [0, 0, 0, 0]])

        # Path correction factors (Bovy, P.H.L., Bekhor, S. and Prato, C.G., 2008.
        # The factor of revisited path size: Alternative derivation. Transportation Research Record 132–140)
        A['N5'] = np.array([[0, 2, 1], [0, 0, 1], [0, 0, 0]])
        Q['N5'] = np.array([[0, 0, 100], [0, 0, 0], [0, 0, 0]])

        A['Toy'] = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 2], [0, 0, 0, 0]])
        Q['Toy'] = np.array([[0, 0, 0, 500], [0, 0, 0, 500], [0, 0, 0, 500], [0, 0, 0, 0]])

        #Sheffi (pp 329)
        A['Sheffi'] = np.array([[0, 2], [0, 0]])
        Q['Sheffi'] = np.array([[0, 4000], [0, 0]])

        # Yang network with no error in OD matrix

        links_tuples['Yang'] = [(1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (3, 6), (4, 5), (4, 7), (5, 6), \
                               (5, 8), (5, 9), (6, 9), (7, 8), (8, 9)]

        A['Yang'] = np.zeros([9, 9], dtype = int)

        for link_tuple in links_tuples['Yang']:
            A['Yang'][(link_tuple[0] - 1, link_tuple[1] - 1)] = 1

        A_Yang = np.array([
            [0, 1, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0]
            , [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0, 0]
            , [0, 0, 0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]
            , [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]
            , [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        assert np.allclose(A['Yang'], A_Yang)

        # - True demand martrix
        demand_dict = {(1, 6): 120, (1, 8): 150, (1, 9): 100, (2, 6): 130, (2, 8): 200, (2, 9): 90, (4, 6): 80, (4, 8): 180,
                       (4, 9): 110}

        Q['Yang'] = np.zeros([9, 9])

        for od, demand in demand_dict.items():
            Q['Yang'][(od[0] - 1, od[1] - 1)] = demand

        # Yang network with error in OD matrix
        A['Yang2'] = copy.deepcopy(A['Yang'])
        Q['Yang2'] = np.zeros([9, 9])

        # - Reference demand matrix (distorted)
        demand_dict = {(1, 6): 100, (1, 8): 130, (1, 9): 120, (2, 6): 120, (2, 8): 170, (2, 9): 140, (4, 6): 110,
                       (4, 8): 170, (4, 9): 105}

        for od, demand in demand_dict.items():
            Q['Yang2'][(od[0] - 1, od[1] - 1)] = demand

        # Lo and Chan (2003)
        links_tuples = [(1, 2), (2, 1), (2, 3), (3, 2), (3, 6), (6, 3), (6, 5), (5, 6), (5, 4),
                        (4, 5), (1, 4), (4, 1), (2, 5), (5, 2)]

        A['Lo'] = np.zeros([6, 6])

        for link_tuple in links_tuples:
            A['Lo'][(link_tuple[0] - 1, link_tuple[1] - 1)] = 1

        # - True demand matrix
        demand_dict = {(1, 3): 500, (1, 4): 250, (1, 6): 250, (3, 1): 500, (3, 4): 250, (3, 6): 250, (4, 1): 250,
                       (4, 3): 500, (4, 6): 250, (6, 1): 500, (6, 3): 250, (6, 4): 250}

        Q['Lo'] = np.zeros([6, 6])

        for od, demand in demand_dict.items():
            Q['Lo'][(od[0] - 1, od[1] - 1)] = demand

        # Wang et al. (2016), Table 6

        links_tuples = [(1, 2), (1, 4), (2, 1), (2, 3), (3, 2), (3, 4), (4, 1), (4, 3)]

        A['Wang'] = np.zeros([4, 4])

        for link_tuple in links_tuples:
            A['Wang'][(link_tuple[0] - 1, link_tuple[1] - 1)] = 1

        # - True demand matrix
        demand_dict = {(1, 2): 3067, (1, 3): 2489, (1, 4): 4814, (2, 1): 2389, (2, 3): 3774, (2, 4): 1946, (3, 1): 3477,
                       (3, 2): 5772, (3, 4): 4604, (4, 1): 4497, (4, 2): 2773, (4, 3): 4284}

        Q['Wang'] = np.zeros([4, 4])
        for od, demand in demand_dict.items():
            Q['Wang'][(od[0] - 1, od[1] - 1)] = demand

        # Q['Wang'] = 0.5*Q['Wang']

        return A,Q

    @staticmethod
    def get_A_Q_custom_networks(networks_names: List[str]):

        A,Q = NetworkGenerator.generate_A_Q_custom_networks()

        valid_network_names = list(A.keys())

        # if len(networks_names) == 0:
        #     return A, Q

        A_new = {}
        Q_new = {}

        for i in networks_names:

            if i not in valid_network_names:
                print('\n' + i + ' is an incorrect name. \n Please select among the following networks names:', valid_network_names)

            # elif len(networks_names) == 1:
            #     return A[i], Q[i]

            else:
                A_new[i] = A[i]
                Q_new[i] = Q[i]

        return A_new, Q_new



