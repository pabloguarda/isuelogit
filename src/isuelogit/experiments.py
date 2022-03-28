from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytypes import Links, Options, Optional, Proportion, Matrix, Features, List, Feature

import random
import os
import numpy as np
import pandas as pd
import time
from sortedcontainers import SortedSet
import matplotlib.pyplot as plt

import visualization
from printer import block_output, printProgressBar,printIterationBar
from descriptive_statistics import nrmse, rmse, get_loss_and_estimates_over_iterations
from estimation import monotonocity_traffic_count_functions, grid_search_optimization, \
    Learner, OuterOptimizer, compute_vot
from etl import masked_observed_counts
from writer import Reporter
from equilibrium import LUE_Equilibrator
from networks import denseQ
import config


class NetworkExperiment(Reporter):

    '''
    A basic feature of a NetworkExperiment instance is to have the capability of simulate counts. This requires
    to have an equilibrator and a data generator, which are arguments of the parent class Estimation.
    '''

    def __init__(self,
                 **kwargs
                 ):

        self.experiment_replicate = None

        self.experiment_started = False

        super().__init__(**kwargs)

        self.make_dirs(folderpath= kwargs.get('folderpath'))

        self.artist = visualization.Artist(folder_plots=self.dirs['experiment_folder'])

    def setup_experiment(self,
                         replicates = None,
                         bilevel_iters = None,
                         alpha=None,
                         range_initial_values: tuple = None,
                         **kwargs):

        print('\n'+self.options['name'], end = '\n')

        # self.config.set_experiments_log_files(networkname=self.config.sim_options['current_network'].lower())

        self.options['range_initial_values'] = range_initial_values

        if alpha is not None:
            self.options['alpha'] = alpha

        if bilevel_iters is not None:
            self.options['bilevel_iters'] = bilevel_iters

        if replicates is not None:
            self.options['replicates'] = replicates

    def assign_stats_columns(self, results, theta_true, h0 = 0, alpha = 0.05):
        '''
         # Add true values of the parameters, including vot, to dataframe as well as false positives and negatives

        Args:
            results:
            theta_true:
            h0:
            alpha:

        Returns:

        '''

        results_df = results.copy()

        if 'c' in theta_true.keys():
            theta_true['vot'] = float(theta_true['tt']) / float(theta_true['c'])

        results_df = pd.merge(results_df,
                                     pd.DataFrame({'parameter': theta_true.keys(), 'theta_true': theta_true.values()}), how = 'left', sort = False)

        results_df['bias'] = results_df['est'] - results_df['theta_true']

        results_df['fn'] = 0
        results_df['fp'] = 0
        results_df['f_type'] = ''

        for i in results_df.index:

            if results_df.at[i, 'theta_true'] == h0:

                results_df.at[i, 'f_type'] = 'fp'

                if results_df.at[i, 'p-value'] < alpha:
                    results_df.at[i, 'fp'] = 1

            else:

                results_df.at[i, 'f_type'] = 'fn'

                if results_df.at[i, 'p-value'] > alpha:
                    results_df.at[i, 'fn'] = 1

        return results_df

    def make_dirs(self, folderpath = None):
        '''

        Store results into log file and store a folder with a summary of the estimation
        Create a network_name to store the estimates of the current experiment

        Args:
            folderpath:

        Returns:

        '''

        # if folderpath is None:
        #     folderpath = self.network.key

        if not self.experiment_started:

            # Create a subfolder based on starting date
            self.dirs['experiment_folder'] = folderpath + '/' + self.options['date']

            if not os.path.exists(self.dirs['experiment_folder']):
                os.makedirs(self.dirs['experiment_folder'])

            # Create a subfolder based on starting date and time of the simulation
            self.dirs['experiment_folder'] += '/' + self.options['time']

            os.makedirs(self.dirs['experiment_folder'])

            self.experiment_started = True
            self.experiment_replicate = 0

    def generate_random_link_features(self,
                                      n_sparse_features=0,
                                      normalization=None):

        if normalization is None:
            normalization = {'mean': False, 'std': False}

        synthetic_feature_c_df = self.linkdata_generator.simulate_features(links=self.network.links,
                                                                           features_Z={'c'},
                                                                           option='continuous',
                                                                           range=(0, 1),
                                                                           normalization=normalization
                                                                           )

        synthetic_feature_s_df = self.linkdata_generator.simulate_features(links=self.network.links,
                                                                           features_Z={'s'},
                                                                           option='discrete',
                                                                           range=(0, 1),
                                                                           normalization=normalization
                                                                           )

        # Merge dataframes with existing dataframe
        link_features_df = synthetic_feature_c_df.merge(synthetic_feature_s_df, left_on='link_key', right_on='link_key')

        if n_sparse_features > 0:
            sparse_features_labels = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

            sparse_features_df = self.linkdata_generator.simulate_features(
                links=self.network.links,
                features_Z=sparse_features_labels,
                option='continuous',
                range=(0, 1),
                normalization=normalization
            )

            link_features_df = link_features_df.merge(sparse_features_df, left_on='link_key', right_on='link_key')

        return link_features_df

    def write_table(self,
                    df,
                    folder = None,
                    filename=None,
                    float_format='%.3f',
                    **kwargs):

        if filename is None:
            filename = 'experiment_report.csv'

        if folder is None:
            folder = self.dirs['experiment_folder']

        df.to_csv(folder + '/' + filename,
                  sep=',', encoding='utf-8', index=False, float_format=float_format, **kwargs)

    def create_replicate_folder(self, replicate):

        self.dirs['replicate_folder'] = self.dirs['experiment_folder'] + '/' + str(replicate)

        if not os.path.exists(self.dirs['replicate_folder']):
            os.makedirs(self.dirs['replicate_folder'])

    def write_replicate_table(self,
                              df,
                              filename,
                              replicate,
                              **kwargs):

        self.create_replicate_folder(replicate)

        filename = filename + '.csv'

        self.write_table(df = df,
                         filename = filename,
                         folder = self.dirs['replicate_folder'],
                         **kwargs)


    def write_experiment_report(self,
                                folder = None,
                                filename = None):
        # self.write_report(filepath = self.dirs['experiment_folder'] + '/' + 'experiment_options.csv')

        if filename is None:
            filename = 'experiment_report.csv'
        if folder is None:
            folder = self.dirs['experiment_folder']

        filepath = folder + '/' + filename

        # Network information
        if self.network is not None:
            self.options['network'] = self.network.key
            self.options['links'] = len(self.network.links)
            self.options['paths'] = len(self.network.paths)
            self.options['ods'] = len(self.network.ods)
            self.options['scale_OD'] = self.network.OD.scale

        self.options['features'] = self.utility_function.features
        # self.options['initial parameters'] = utility_function.initial_values
        self.options['true parameters'] = self.utility_function.true_values

        linkdata_generator = self.linkdata_generator
        self.options['data_generator'] = self.linkdata_generator.options

        for learner in self.learners:

            self.options[learner.name + '_learner'] \
                = {k: v for k, v in learner.options.items() if
                   k in ['bilevel_iters']}

            self.options[learner.name + '_optimizer'] \
                = {k: v for k, v in learner.outer_optimizer.options.items() if
                   k in ['method', 'eta', 'iters']}

            self.options[learner.name + '_equilibrator'] = {k: v for k, v in learner.equilibrator.options.items()}

        df = pd.DataFrame({'option': self.options.keys(), 'value': self.options.values()})

        df.to_csv(filepath,
                  sep=',',
                  encoding='utf-8',
                  index=False)

    def write_experiment_report_1(self, filename, decimals):
        report_dict = {}


        if config.experiment_options['experiment_mode'] == 'noise_experiment':
            report_dict['theta_norefined'] = str(
                {key: round(val, decimals) for key, val in config.experiment_results['theta_norefined'].items()})

            report_dict['theta_refined'] = str(
                {key: round(val, decimals) for key, val in config.experiment_results['theta_refined'].items()})

            # if 'theta_refined_combined' in config.estimation_results.keys():
            report_dict['theta_refined_combined'] = str(
                {key: round(val, decimals) for key, val in config.experimentresults['theta_refined_combined'].items()})

            report_dict['norefined_prediction_loss'] = '{:,}'.format(
                round(float(config.experiment_results['best_loss_norefined']), decimals))

            report_dict['refined_prediction_loss'] = '{:,}'.format(
                round(float(config.experiment_results['best_loss_refined']), decimals))

            # if 'best_loss_refined_combined' in config.experiment_results.keys():

            report_dict['refined_combined_prediction_loss'] = '{:,}'.format(
                round(float(config.estimation_results['best_loss_refined_combined']), decimals))

        if config.experiment_options['experiment_mode'] == 'optimization benchmark':
            # report_dict['theta_estimates'] = str(
            #     {key: round(val, decimals) for key, val in config.experiment_results['theta_estimate'].items()})

            # report_dict['theta_estimates'] = config.experiment_results['theta_estimates']

            for method, estimates in config.experiment_results['theta_estimates'].items():
                report_dict['theta_estimates_' + method] = str(
                    {key: round(val, decimals) for key, val in estimates.items()})

            # Add VOT estimates
            # report_dict['vot_estimates']
            vot_estimates = {}
            for method, estimates in config.experiment_results['theta_estimates'].items():
                vot_estimates[method] = str(round(estimates['tt'] / estimates['c'], decimals))
                # report_dict['vot_estimate' + method] = str(round(estimates['tt'] / estimates['c'], decimals))

            report_dict['vot_estimates'] = vot_estimates

            # for method, losses in  config.experiment_results['losses'].items():
            #     report_dict['losses_' + method] = str({key: round(val, decimals) for key, val in losses.items()})

            report_dict['losses'] = {key: '{:,}'.format(round(val, decimals)) for key, val in
                                     config.experiment_results['losses'].items()}

        report_df = pd.DataFrame({'item': report_dict.keys(), 'value': report_dict.values()})

        self.write_replicate_table(df=report_df, filename=filename)


class NetworksExperiment(NetworkExperiment):


    '''
    A basic characteristic of a NetworkExperiment is to have the capability of simulate counts, which requires
    to have an equilibrator and a data generator.
    '''

    def __init__(self,
                 networks,
                 **kwargs
                 ):

        # kwargs['folderpath'] = 'small-networks'
        # kwargs['network'] = kwargs['networks'][0]

        super().__init__(**kwargs)

        self.networks = networks


class ConvergenceExperiment(NetworkExperiment):

    def __init__(self,
                 outer_optimizers: List[OuterOptimizer],
                 bilevel_iters,
                 equilibrator,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # It is assumed that the same equilibrator is used for both learners
        self.equilibrator = equilibrator

        self.outer_optimizer_norefined = outer_optimizers[0]

        bilevel_iters_norefined = bilevel_iters

        if len(outer_optimizers) == 1:
            bilevel_iters_norefined = 1

        self.learner_norefined = Learner(
            equilibrator=self.equilibrator,
            outer_optimizer=self.outer_optimizer_norefined,
            utility_function=self.utility_function,
            network=self.network,
            bilevel_iters=bilevel_iters_norefined,
            name='norefined',
        )

        self.outer_optimizer_refined = outer_optimizers[0]

        if len(outer_optimizers) == 2:
            self.outer_optimizer_refined = outer_optimizers[1]

        self.learner_refined = Learner(
            equilibrator=self.equilibrator,
            outer_optimizer=self.outer_optimizer_refined,
            utility_function=self.utility_function,
            network=self.network,
            bilevel_iters=bilevel_iters,
            name='refined',
        )

        self.learners = [self.learner_norefined, self.learner_refined]

        self.options['bilevel_iters'] = bilevel_iters

    def run(self,
            range_initial_values: tuple = None,
            **kwargs):


        # Update options
        self.setup_experiment(**kwargs)

        self.write_experiment_report(filename='experiment_report.csv')

        results_df = {}

        scenarios = {'uncongested': True, 'congested': False}
        # scenarios = {'congested': False, 'uncongested': True}

        # Initilization of initial estimate
        if range_initial_values is not None:
            self.utility_function.random_initializer(range_initial_values)
        else:
            self.utility_function.zero_initializer()

        initial_values = copy.deepcopy(self.utility_function.initial_values)

        for scenario, uncongested_mode in scenarios.items():

            print('\nScenario:', scenario)

            self.equilibrator.update_options(uncongested_mode=uncongested_mode)

            # Generate synthetic traffic counts
            counts, _ = self.linkdata_generator.simulate_counts(
                network=self.network,
                equilibrator=self.equilibrator,
                utility_function=self.utility_function)

            self.network.load_traffic_counts(counts=counts)

            print('\nStatistical Inference in No Refined Stage')

            self.utility_function.initial_values = copy.deepcopy(initial_values)

            learning_results_norefined, inference_results_norefined, best_iter_norefined = \
                self.learner_norefined.statistical_inference(**kwargs)

            theta_norefined = learning_results_norefined[best_iter_norefined]['theta']
            # objective_norefined = learning_results_norefined[best_iter_norefined]['objective']
            # self.config.estimation_results['theta_norefined'] = theta_norefined
            # self.config.estimation_results['best_loss_norefined'] = objective_norefined

            print('\nStatistical Inference in Refined Stage')

            self.utility_function.initial_values = theta_norefined

            learning_results_refined, inference_results_refined, best_iter_refined = \
                self.learner_refined.statistical_inference(**kwargs)

            self.write_inference_tables(results_norefined = inference_results_norefined,
                                        results_refined = inference_results_refined,
                                        folder=self.dirs['experiment_folder'],
                                        filename =  scenario)

            results_df[scenario] = self.write_convergence_table(results_norefined=learning_results_norefined,
                                         results_refined=learning_results_refined,
                                         folder=self.dirs['experiment_folder'],
                                         filename='convergence_' + scenario + '.csv', float_format='%.3f')


        self.artist.convergence_network_experiment(
            results_df=results_df,
            filename='loss-vs-vot-over-iterations_' + scenario,
            folder = self.dirs['experiment_folder'],
            methods=[self.outer_optimizer_norefined.method.key, self.outer_optimizer_refined.method.key],
            theta_true=self.utility_function.true_values,
            colors=['blue', 'red'],
            labels=['Uncongested', 'Congested']
        )


class PseudoconvexityExperiment(NetworkExperiment):

    def __init__(self,
                 equilibrator,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.equilibrator = equilibrator

    def run(self,
            grid,
            features,
            colors,
            xticks=None,
            features_labels=None
            ):

        # TODO: Redesign this code such that it is common with the pseudoconvexity experiment for small networks. Maybe I will need to have a Multi and Single Network Experiment

        self.make_dirs(folderpath=self.network.key.lower())

        if features_labels is None:
            features_labels = features

        # Generate synthetic traffic counts
        counts, _ = self.linkdata_generator.simulate_counts(
            network=self.network,
            equilibrator=self.equilibrator,
            utility_function=self.utility_function)

        self.network.load_traffic_counts(counts=counts)

        pseudoconvexity_experiment_df = pd.DataFrame()

        # colors = ['black', 'red', 'blue']
        # attributes = ['tt', 'c','s']
        # features = ['tt', 'c']

        self.utility_function.values = self.utility_function.true_values

        for feature, color in zip(features, colors):
            # isl.printer.blockPrint()
            theta_attr_grid, f_vals, grad_f_vals, hessian_f_vals \
                = grid_search_optimization(network=self.network,
                                           equilibrator=self.equilibrator,
                                           counts=self.network.counts_vector,
                                           q=self.network.q,
                                           theta_attr_grid=grid,
                                           utility_function=self.utility_function,
                                           feature=feature,
                                           gradients=True,
                                           hessians=True
                                           )

            # Create pandas dataframe
            pseudoconvexity_experiment_attr_df = \
                pd.DataFrame({'attr': feature,
                              'theta_attr_grid': theta_attr_grid,
                              'f_vals': np.array(f_vals).flatten(),
                              'grad_f_vals': np.array(grad_f_vals).flatten(),
                              'hessian_f_vals': np.array(hessian_f_vals).flatten()})

            pseudoconvexity_experiment_df = pseudoconvexity_experiment_df.append(pseudoconvexity_experiment_attr_df)

            # Write csv file
            self.write_table(df=pseudoconvexity_experiment_attr_df,
                             filename='pseudoconvexity_' + feature + '.csv',
                             float_format='%.1f')

            # Plot
            self.artist.pseudoconvexity_loss_function(
                filename='pseudoconvexity_lossfunction_' + feature,
                color=color,
                f_vals=f_vals,
                grad_f_vals=grad_f_vals,
                hessian_f_vals=hessian_f_vals,
                x_range=theta_attr_grid,  # np.arange(-3,3, 0.5)
                theta_true=self.utility_function.true_values[feature],
                xticks=xticks)

            # plt.show()

        # Write csv file
        self.write_table(df=pseudoconvexity_experiment_df,
                         filename='pseudoconvexity.csv',
                         float_format='%.1f')

        # Combined pseudo-convexity plot

        self.artist.coordinatewise_pseudoconvexity_loss_function(
            filename='coordinatewise_pseudoconvexity_lossfunction_' + self.network.key
            , results_df=pseudoconvexity_experiment_df
            , x_range=theta_attr_grid  # np.arange(-3,3, 0.5)
            , theta_true=self.utility_function.true_values
            , colors=colors
            , labels=features_labels
            , xticks=xticks
            # , colors = ['black','red', 'blue']
            # , labels = ['travel time', 'cost', 'intersections']

        )

        self.write_experiment_report()

    def write_experiment_report(self,
                                filename=None):

        if filename is None:
            filename = 'experiment_report.csv'

        filepath = self.dirs['experiment_folder'] + '/' + filename

        self.options['equilibrator'] = self.equilibrator.options

        df = pd.DataFrame({'option': self.options.keys(), 'value': self.options.values()})

        df.to_csv(filepath,
                  sep=',',
                  encoding='utf-8',
                  index=False)


class ConsistencyExperiment(NetworkExperiment):

    def __init__(self,
                 outer_optimizers: List[OuterOptimizer],
                 equilibrator,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # It is assumed that the same equilibrator is used for both learners
        self.equilibrator = equilibrator

        # Learner in no refined stage

        self.learner_norefined_1 = Learner(
            equilibrator=self.equilibrator,
            outer_optimizer=outer_optimizers[0],
            utility_function=self.utility_function,
            network=self.network,
            name='norefined1'
        )

        # First class of learner in refined stage

        self.learner_norefined_2 = Learner(
            equilibrator=self.equilibrator,
            outer_optimizer=outer_optimizers[1],
            utility_function=self.utility_function,
            network=self.network,
            name='norefined2'
        )

        # Second class of learner in refined stage
        self.learner_refined = Learner(
            equilibrator=self.equilibrator,
            outer_optimizer=outer_optimizers[2],
            utility_function=self.utility_function,
            network=self.network,
            name='refined'
        )

        self.learners = [self.learner_norefined_1, self.learner_norefined_2, self.learner_refined]

        # self.set_experiments_log_folders(networkname=self.network.key)

    def run(self,
            show_replicate_plot=False,
            replicate_report=False,
            n_sparse_features = 0,
            **kwargs
            ):

        self.setup_experiment(**kwargs)

        self.utility_function.add_sparse_features(Z=['k' + str(i) for i in np.arange(0, n_sparse_features)])

        self.write_experiment_report()

        results_experiment = pd.DataFrame({})

        alpha =self.options.get('alpha')
        range_initial_values = self.options.get('range_initial_values')
        bilevel_iters = self.options.get('bilevel_iters')
        replicates = self.options.get('replicates')

        for replicate in range(1, replicates + 1):

            if replicate_report or show_replicate_plot:
                print('\nReplicate', replicate)
            elif not replicate_report:
                printIterationBar(replicate, replicates, prefix='Replicates:', length=20)

            self.create_replicate_folder(replicate=replicate)

            # Generate new random features and load them in the network
            self.network.load_features_data(linkdata=self.generate_random_link_features(
                n_sparse_features=n_sparse_features,
                normalization={'mean': False, 'std': False}))

            with block_output(show_stdout=replicate_report, show_stderr=replicate_report):

                # Generate synthetic traffic counts
                counts, _ = self.linkdata_generator.simulate_counts(
                    network=self.network,
                    equilibrator=self.equilibrator,
                    utility_function=self.utility_function)

                self.network.load_traffic_counts(counts=counts)

                sd_x = self.linkdata_generator.options['noise_params']['sd_x']

                method_norefined_1 = self.learner_norefined_1.outer_optimizer.method.key
                method_norefined_2 = self.learner_norefined_2.outer_optimizer.method.key
                method_refined = self.learner_refined.outer_optimizer.method.key

                t0 = time.time()

                # Initilization of initial estimate
                if range_initial_values is not None:
                    self.utility_function.random_initializer(range_initial_values)
                else:
                    self.utility_function.zero_initializer()

                initial_values = copy.deepcopy(self.utility_function.initial_values)

                print('\nStatistical Inference in No Refined Stage using', method_norefined_1)

                self.utility_function.initial_values = copy.deepcopy(initial_values)

                learning_results_norefined_1, inference_results_norefined_1, best_iter_norefined_1 = \
                    self.learner_norefined_1.statistical_inference(bilevel_iters=bilevel_iters,
                                                                   alpha=alpha)

                theta_norefined_1 = learning_results_norefined_1[best_iter_norefined_1]['theta']

                predicted_counts_norefined_1 = np.array(
                    list(learning_results_norefined_1[best_iter_norefined_1]['x'].values()))

                time_norefined_1 = time.time() - t0

                t0 = time.time()

                self.utility_function.initial_values =copy.deepcopy(initial_values)

                print('\nStatistical Inference in No Refined Stage using', method_norefined_2)

                learning_results_norefined_2, inference_results_norefined_2, best_iter_norefined_2 = \
                    self.learner_norefined_2.statistical_inference(bilevel_iters=bilevel_iters,
                                                                   alpha=alpha)

                theta_norefined_2 = learning_results_norefined_2[best_iter_norefined_2]['theta']

                predicted_counts_norefined_2 = np.array(
                    list(learning_results_norefined_2[best_iter_norefined_2]['x'].values()))

                time_norefined_2 = time.time() - t0

                t0 = time.time()

                # Combined methods

                print('\nStatistical Inference in Refined Stage using', method_refined)

                self.learner_refined.utility_function.initial_values = theta_norefined_1

                learning_results_refined, inference_results_refined, best_iter_refined = \
                    self.learner_refined.statistical_inference(bilevel_iters=bilevel_iters, alpha=alpha)

                theta_refined = learning_results_refined[best_iter_refined]['theta']

                predicted_counts_refined = np.array(
                    list(learning_results_refined[best_iter_refined]['x'].values()))

                time_refined = time.time() - t0 + time_norefined_1

                # Prepare pandas dataframe
                inference_results_norefined_1 = \
                    pd.concat([inference_results_norefined_1['parameters'],
                               pd.DataFrame({'parameter': ['vot'], 'est': compute_vot(theta_norefined_1)})])

                inference_results_norefined_1 = inference_results_norefined_1.assign(
                    time=time_norefined_1, method=method_norefined_1, rep=replicate,
                    objective=learning_results_norefined_1[best_iter_norefined_1]['objective'],
                    nrmse=nrmse(actual=np.array(list(counts.values())), predicted=predicted_counts_norefined_1))

                inference_results_norefined_2 = \
                    pd.concat([inference_results_norefined_2['parameters'],
                               pd.DataFrame({'parameter': ['vot'], 'est': compute_vot(theta_norefined_2)})])

                inference_results_norefined_2 = inference_results_norefined_2.assign(
                    time=time_norefined_2, method=method_norefined_2, rep=replicate,
                    objective=learning_results_norefined_2[best_iter_norefined_2]['objective'],
                    nrmse=nrmse(actual=np.array(list(counts.values())), predicted=predicted_counts_norefined_2))

                inference_results_refined = \
                    pd.concat([inference_results_refined['parameters'],
                               pd.DataFrame({'parameter': ['vot'], 'est': compute_vot(theta_refined)})])

                inference_results_refined = inference_results_refined.assign(
                    time=time_refined, method=method_norefined_1 + '+' + method_refined, rep=replicate,
                    objective=learning_results_refined[best_iter_refined]['objective'],
                    nrmse=nrmse(actual=np.array(list(counts.values())), predicted=predicted_counts_refined))

                results_replicate = pd.concat([inference_results_norefined_1,
                                               inference_results_norefined_2,
                                               inference_results_refined])

                results_replicate = self.assign_stats_columns(results=results_replicate, alpha=alpha,
                                                              theta_true=self.utility_function.true_values)

                results_replicate = results_replicate.rename(
                    columns={"CI": "width_confint", "parameter": "attr", 'p-value': 'pvalue', 'est': 'theta'})

                results_experiment = pd.concat([results_experiment, results_replicate])

                # self.write_csv_to_experiment_log_folder(df=inference_results_refined['model'],
                #                                         filename='model_inference_table',
                #                                         log_file=self.dirs)

                self.write_replicate_table(df=results_replicate, filename='inference', replicate=replicate)

                self.write_replicate_table(df=results_experiment, filename='cumulative_inference', replicate=replicate)

                # Visualization
                fig = self.artist.consistency_experiment(
                    results_experiment=results_experiment,
                    alpha = alpha,
                    sd_x=sd_x,
                    range_initial_values=range_initial_values,
                    folder=self.dirs['replicate_folder'])

                if show_replicate_plot:
                    plt.show()
                else:
                    plt.close(fig)

        plt.show()

        self.artist.consistency_experiment(
            results_experiment=results_experiment,
            range_initial_values=range_initial_values,
            alpha = alpha,
            sd_x=sd_x,
            folder=self.dirs['experiment_folder'])


class CountsExperiment(ConvergenceExperiment):
    ''' Experiment that manipulates sensor coverage and noise in traffic counts'''

    def __init__(self,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

    def run(self,
            levels: [],
            type,
            n_sparse_features = 0,
            show_replicate_plot=False,
            replicate_report = False,
            **kwargs):

        # Update options
        self.setup_experiment(**kwargs)

        self.utility_function.add_sparse_features(Z=['k' + str(i) for i in np.arange(0, n_sparse_features)])

        self.write_experiment_report()

        alpha = self.options.get('alpha')
        range_initial_values = self.options.get('range_initial_values')
        # bilevel_iters = self.options['bilevel_iters']
        replicates = self.options['replicates']

        self.options['levels'] = levels
        self.options['type'] = type

        results_experiment = pd.DataFrame({})

        failed_replicates = 0

        # if type == 'noise':
        #     # Add level with no error to check convergence but it does not visualize it
        #     levels.insert(0, 0)

        for replicate in range(1, replicates + 1):

            if replicate_report or show_replicate_plot:
                print('\nReplicate', replicate)
            elif not replicate_report:
                printIterationBar(replicate, replicates, prefix='Replicates:', length=20)

            self.create_replicate_folder(replicate=replicate)

            # Initilization of initial estimate
            if range_initial_values is not None:
                self.utility_function.random_initializer(range_initial_values)
            else:
                self.utility_function.zero_initializer()

            initial_values = copy.deepcopy(self.utility_function.initial_values)

            results_replicate = pd.DataFrame({})

            # Generate new random features and load them in the network
            self.network.load_features_data(linkdata=self.generate_random_link_features(
                n_sparse_features=n_sparse_features,
                normalization={'mean': False, 'std': False}))

            with block_output(show_stdout=replicate_report, show_stderr=replicate_report):

                # Generate synthetic traffic counts
                counts_replicate, _ = self.linkdata_generator.simulate_counts(
                    network=self.network,
                    equilibrator=self.equilibrator,
                    utility_function=self.utility_function)

                counts_replicate_vector = np.array(list(counts_replicate.values()))[:,np.newaxis]

                for level in levels:

                    if type == 'coverage':

                        sd_x = self.linkdata_generator.options['noise_params']['sd_x']

                        counts,_ = self.linkdata_generator.mask_counts_by_coverage(counts_replicate_vector, coverage=level)

                    if type == 'noise':

                        counts = self.linkdata_generator.add_error_counts(counts_replicate_vector, sd_x = level)

                        sd_x = level

                    self.network.load_traffic_counts(counts=dict(zip(counts_replicate.keys(),counts.flatten())))

                    self.utility_function.initial_values = initial_values

                    print('\nStatistical Inference in No Refined Stage')

                    # self.utility_function.zero_initializer()

                    learning_results_norefined, inference_results_norefined, best_iter_norefined = \
                        self.learner_norefined.statistical_inference(alpha=alpha)

                    theta_norefined = learning_results_norefined[best_iter_norefined]['theta']

                    print('\nStatistical Inference in Refined Stage')

                    self.utility_function.initial_values = theta_norefined

                    learning_results_refined, inference_results_refined, best_iter_refined = \
                        self.learner_refined.statistical_inference(alpha=alpha)

                    theta_refined = learning_results_refined[best_iter_refined]['theta']

                    self.write_inference_tables(results_norefined=inference_results_norefined,
                                                results_refined=inference_results_refined,
                                                folder = self.dirs['replicate_folder'],
                                                filename= 'convergence_level_' + str(level) + '.csv')

                    self.write_convergence_table(results_norefined=learning_results_norefined,
                                                results_refined=learning_results_refined,
                                                folder=self.dirs['replicate_folder'],
                                                filename= 'convergence_' + str(level) + '.csv')

                    # Generate pandas dataframe to consolite replicate results
                    results = pd.concat([inference_results_refined['parameters'],
                                         pd.DataFrame({'parameter': ['vot'], 'est': compute_vot(theta_refined)})])

                    results = results.assign(
                        rep=replicate,
                        level=level,
                        objective=learning_results_refined[best_iter_refined]['objective'],
                        nrmse=nrmse(actual=np.array(list(counts_replicate.values())),
                                    predicted=np.array(list(learning_results_refined[best_iter_refined]['x'].values())))
                    )

                    results_replicate = pd.concat([results_replicate, results])

                    # converged = True
                    # if level == 0:
                    #
                    #     current_results = results_experiment[results_experiment.rep == replicate]
                    #
                    #     objective_function = learning_results_refined[best_iter_refined]['objective']
                    #
                    #     # if objective_function > 1e3 or np.sum(current_results.fp + current_results.fn) > 0:
                    #
                    #     if np.sum(current_results.fp + current_results.fn) > 0:
                    #         failed_replicates += 1
                    #
                    #         converged = False
                    #         # Remove rows from this replicate results for level 0
                    #         results_experiment = results_experiment[results_experiment.rep != replicate]
                    #
                    #     else:
                    #         results_experiment = results_experiment[results_experiment.level != 0]
                    #
                    # if not converged:
                    #     break

                # Add true values of the parameters, including vot, to dataframe as well as false positives and negatives
                results_replicate = self.assign_stats_columns(results=results_replicate, alpha=alpha,
                                                              theta_true=self.utility_function.true_values)

                results_replicate = results_replicate.rename(
                    columns={"CI": "width_confint", "parameter": "attr", 'p-value': 'pvalue', 'est': 'theta'})

                results_experiment = pd.concat([results_experiment, results_replicate])

                # self.write_csv_to_experiment_log_folder(df=inference_results_refined['model'],
                #                                         filename='model_inference_table',
                #                                         log_file=self.dirs)

                self.write_replicate_table(df=results_replicate, filename='inference', replicate=replicate)

                self.write_replicate_table(df=results_experiment, filename='cumulative_inference', replicate=replicate)

                fig = self.artist.levels_experiment(
                    results_experiment=results_experiment,
                    alpha = alpha,
                    sd_x = sd_x,
                    folder=self.dirs['replicate_folder'],
                    range_initial_values=range_initial_values)

                if show_replicate_plot:
                    plt.show()
                else:
                    plt.close(fig)

        self.artist.levels_experiment(
            results_experiment=results_experiment,
            alpha=alpha,
            sd_x=sd_x,
            folder=self.dirs['experiment_folder'],
            range_initial_values=range_initial_values)

        plt.show()


class ODExperiment(ConvergenceExperiment):
    ''' Experiment that manipulates OD matrix'''

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def run(self,
            levels: List,
            type,
            n_sparse_features = 0,
            show_replicate_plot=False,
            replicate_report=False,
            **kwargs):

        # Update options
        self.setup_experiment(**kwargs)

        self.utility_function.add_sparse_features(Z=['k' + str(i) for i in np.arange(0, n_sparse_features)])

        self.write_experiment_report()

        alpha = self.options.get('alpha')
        range_initial_values = self.options.get('range_initial_values')
        replicates = self.options['replicates']

        self.options['levels'] = levels
        self.options['type'] = type

        results_experiment = pd.DataFrame({})

        generator_options = {}

        failed_replicates = 0

        # Add level with no error to check convergence but it does not visualize it

        # if type == 'noise':
        #     levels.insert(0,0)

        # Store original OD matrix
        # Noise or scale difference in Q matrix
        # Q_original = copy.deepcopy(self.network.Q_true)

        for replicate in range(1, replicates + 1):

            if replicate_report or show_replicate_plot:
                print('\nReplicate', replicate)
            elif not replicate_report:
                printIterationBar(replicate, replicates, prefix='Replicates:', length=20)

            self.create_replicate_folder(replicate=replicate)

            # Initilization of initial estimate
            if range_initial_values is not None:
                self.utility_function.random_initializer(range_initial_values)
            else:
                self.utility_function.zero_initializer()

            initial_values = copy.deepcopy(self.utility_function.initial_values)

            # Generate new random features and load them in the network
            self.network.load_features_data(linkdata=self.generate_random_link_features(n_sparse_features=n_sparse_features))

            results_replicate = pd.DataFrame({})

            with block_output(show_stdout=replicate_report, show_stderr=replicate_report):
                for level in levels:

                    if type == 'scale':
                        generator_options['noise_params'] = {'scale_Q': level}

                    if type == 'noise':
                        generator_options['noise_params'] = {'sd_Q': level}

                    if type == 'congestion':
                        generator_options['noise_params'] = {'congestion_Q': level}

                        # # Update Q matrix with original
                        # self.network.load_OD(Q_original)

                    # Generate synthetic traffic counts
                    counts, _ = self.linkdata_generator.simulate_counts(
                        network=self.network,
                        equilibrator=self.equilibrator,
                        utility_function=self.utility_function,
                        **generator_options
                    )

                    self.network.load_traffic_counts(counts=counts)

                    self.utility_function.initial_values = initial_values

                    print('\nStatistical Inference in No Refined Stage')

                    learning_results_norefined, inference_results_norefined, best_iter_norefined = \
                        self.learner_norefined.statistical_inference()

                    theta_norefined = learning_results_norefined[best_iter_norefined]['theta']

                    # parameter_inference_norefined_table = inference_results_norefined['parameters']
                    # model_inference_norefined_table = inference_results_norefined['model']

                    print('\nStatistical Inference in Refined Stage')

                    self.utility_function.initial_values = theta_norefined

                    learning_results_refined, inference_results_refined, best_iter_refined = \
                        self.learner_refined.statistical_inference()

                    theta_refined = learning_results_refined[best_iter_refined]['theta']

                    self.write_inference_tables(results_norefined=inference_results_norefined,
                                                results_refined=inference_results_refined,
                                                folder = self.dirs['replicate_folder'],
                                                filename= 'convergence_level_' + str(level) + '.csv')

                    self.write_convergence_table(results_norefined=learning_results_norefined,
                                                results_refined=learning_results_refined,
                                                folder=self.dirs['replicate_folder'],
                                                filename= 'convergence_' + str(level) + '.csv')

                    # Generate pandas dataframe
                    results = pd.concat([inference_results_refined['parameters'],
                                         pd.DataFrame({'parameter': ['vot'], 'est': compute_vot(theta_refined)})])

                    results = results.assign(
                        rep=replicate,
                        level=level,
                        objective=learning_results_refined[best_iter_refined]['objective'],
                        nrmse=nrmse(actual=np.array(list(counts.values())),
                                    predicted= np.array(list(learning_results_refined[best_iter_refined]['x'].values())))
                    )

                    results_replicate = pd.concat([results_replicate, results])

                    # converged = True

                    # if level == 0:
                    #
                    #     current_results = results[results.rep == replicate]
                    #
                    #     objective_function = learning_results_refined[best_iter_refined]['objective']
                    #
                    #     # if objective_function > 1e3 or np.sum(current_results.fp + current_results.fn) > 0:
                    #
                    #     if np.sum(current_results.fp + current_results.fn) > 0:
                    #         failed_replicates += 1
                    #
                    #         converged = False
                    #
                    #     # Remove this replicate results
                    #     results_replicate = results_replicate[results.rep != replicate]

                # Add true values of the parameters, including vot, to dataframe as well as false positives and negatives
                results_replicate = self.assign_stats_columns(results=results_replicate, alpha=alpha,
                                                              theta_true=self.utility_function.true_values)

                results_replicate = results_replicate.rename(
                    columns={"CI": "width_confint", "parameter": "attr", 'p-value': 'pvalue', 'est': 'theta'})

                results_experiment = pd.concat([results_experiment, results_replicate])

                self.write_replicate_table(df=results_replicate, filename='inference', replicate=replicate)

                self.write_replicate_table(df=results_experiment, filename='cumulative_inference', replicate=replicate)

                fig = self.artist.levels_experiment(
                    results_experiment=results_experiment,
                    sd_x = self.linkdata_generator.options['noise_params']['sd_x'],
                    alpha=alpha,
                    folder=self.dirs['replicate_folder'],
                    range_initial_values=range_initial_values)

                if show_replicate_plot:
                    plt.show()
                else:
                    plt.close(fig)

        self.artist.levels_experiment(
            results_experiment=results_experiment,
            sd_x=self.linkdata_generator.options['noise_params']['sd_x'],
            alpha=alpha,
            folder=self.dirs['experiment_folder'],
            range_initial_values=range_initial_values)

        plt.show()


class MonotonicityExperiments(NetworksExperiment):

    def __init__(self,
                 equilibrator,
                 **kwargs):

        self.equilibrator = equilibrator

        super().__init__(**kwargs)

        # self.linkdata_generator = linkdata_generator

    def run(self,
            grid: List[float],
            feature: Feature,
            n_links=4
            ):

        self.write_experiment_report()

        for network in self.networks:

            # Generate synthetic traffic counts
            counts, _ = self.linkdata_generator.simulate_counts(
                network=network,
                equilibrator=self.equilibrator,
                utility_function=self.utility_function)

            network.load_traffic_counts(counts=counts)

            traffic_count_links_df = monotonocity_traffic_count_functions(
                theta_attr_grid=grid,
                feature=feature,
                network=network,
                equilibrator=self.equilibrator,
                utility_function=self.utility_function
            )

            # Select a random number of links
            unique_links = list(SortedSet(list(traffic_count_links_df.link)))
            idx_links = unique_links

            if len(unique_links) > n_links:
                idx_n_links = sorted(random.sample(list(np.arange(0, len(unique_links))), n_links))
                idx_links = [unique_links[i] for i in idx_n_links]

            traffic_count_links_subset = traffic_count_links_df[traffic_count_links_df['link'].isin(idx_links)]

            self.artist.monotonocity_traffic_count_functions(
                filename='monotonocity_' + network.key,
                folder=self.dirs['experiment_folder'],
                traffic_count_links_df=traffic_count_links_subset)

            plt.show()

    def write_experiment_report(self,
                                filename=None):

        if filename is None:
            filename = 'experiment_report.csv'

        filepath = self.dirs['experiment_folder'] + '/' + filename

        self.options['equilibrator'] = self.equilibrator.options

        df = pd.DataFrame({'option': self.options.keys(), 'value': self.options.values()})

        df.to_csv(filepath,
                  sep=',',
                  encoding='utf-8',
                  index=False)


class PseudoconvexityExperiments(MonotonicityExperiments):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self,
            grid,
            feature=['tt']):

        self.setup_experiment()

        self.write_experiment_report()

        f_vals = {}

        for network in self.networks:

            print(network.key, 'network')

            with block_output(show_stdout=False, show_stderr=False):
                # Generate synthetic traffic counts
                counts, _ = self.linkdata_generator.simulate_counts(
                    network=network,
                    equilibrator=self.equilibrator,
                    utility_function=self.utility_function)

            network.load_traffic_counts(counts=counts)

            theta_attr_grid, f_vals[network], grad_f_vals, hessian_f_vals \
                = grid_search_optimization(network=network,
                                           equilibrator=self.equilibrator,
                                           counts=network.counts_vector,
                                           q=network.q,
                                           theta_attr_grid=grid,
                                           utility_function=self.utility_function,
                                           feature=feature,
                                           gradients=True,
                                           hessians=True)

            # Create pandas dataframe
            pseudoconvexity_experiment_df = pd.DataFrame(
                {'theta_attr_grid': theta_attr_grid,
                 'f_vals': np.array(f_vals[network]).flatten(),
                 'grad_f_vals': np.array(grad_f_vals).flatten(),
                 'hessian_f_vals': np.array(hessian_f_vals).flatten()})

            # Write csv file
            self.write_table(df=pseudoconvexity_experiment_df,
                             filename='pseudoconvexity_' + network.key+ '.csv',
                             float_format='%.1f')

            self.artist.pseudoconvexity_loss_function_small_networks(
                filename='pseudo_convexity_loss_function_' + network.key,
                f_vals=f_vals[network],
                grad_f_vals=grad_f_vals,
                hessian_f_vals=hessian_f_vals,
                x_range=theta_attr_grid,  # np.arange(-3,3, 0.5)
                theta_true=self.utility_function.true_values['tt'])

            plt.show()

        print('Plot of objective function in the four networks')

        self.artist.pseudoconvexity_loss_function_small_networks_lite(
            filename='pseudo_convexity_loss_function_small_networks'
            , f_vals=f_vals
            , x_range=theta_attr_grid  # np.arange(-3,3, 0.5)
            , colors=['blue', 'red', 'black', 'green']
            , labels=['Toy', 'Wang', 'Lo', 'Yang']
            , theta_true=self.utility_function.true_values['tt'])

        plt.show()


class ConvergenceExperiments(ConvergenceExperiment):

    '''

    Experiment in small networks that computes t-tests shifting the order in which the optimization methods are applied

    # - Conduct hypothesis testing by starting with NGD or Gauss-newton and viceversa.
    # - Show the convergence of all networks in the same plot

    '''

    def __init__(self,
                 networks,
                 **kwargs):

        # kwargs['folderpath'] = 'small-networks'

        super().__init__(**kwargs)

        self.networks = networks

        with block_output(show_stdout=False, show_stderr=False):
            for network in self.networks:
                    # Generate synthetic traffic counts
                    counts, _ = self.linkdata_generator.simulate_counts(
                        network= network,
                        equilibrator=self.equilibrator,
                        utility_function=self.utility_function)

                    network.load_traffic_counts(counts=counts)

    def run(self,
            range_initial_values: tuple = None,
            replicate_report=False,
            **kwargs):

        # Update options
        self.setup_experiment(**kwargs)

        # Summary report
        self.write_experiment_report()

        methods1 = [self.learner_norefined.outer_optimizer.method.key,
                    self.learner_refined.outer_optimizer.method.key]
        methods2 = list(reversed(methods1))

        results = {}

        shifted_methods = [False, True]
        methods_order = [methods1,methods2]

        # Initilization of initial estimate
        if range_initial_values is not None:
            self.utility_function.random_initializer(range_initial_values)
        # else:
        #     self.utility_function.zero_initializer()

        initial_values = copy.deepcopy(self.utility_function.initial_values)

        for shifted, methods in zip(shifted_methods, methods_order):
            print('\nMethods:',methods)
            # self.equilibrator.update_options(uncongested_mode=uncongested_mode)

            if shifted:
                learner_norefined = self.learner_refined
                learner_refined = self.learner_norefined

            else:
                learner_norefined = self.learner_norefined
                learner_refined = self.learner_refined

            for network in self.networks:
                learner_norefined.load_network(network)
                learner_refined.load_network(network)

                with block_output(show_stdout=replicate_report, show_stderr=replicate_report):

                    print('\nStatistical Inference in No Refined Stage')

                    self.utility_function.initial_values = copy.deepcopy(initial_values)

                    learning_results_norefined, inference_results_norefined, best_iter_norefined = \
                        learner_norefined.statistical_inference()

                    theta_norefined = learning_results_norefined[best_iter_norefined]['theta']

                    print('\nStatistical Inference in Refined Stage')

                    self.utility_function.initial_values = theta_norefined

                    learning_results_refined, inference_results_refined, best_iter_refined = \
                        learner_refined.statistical_inference()

                    methods_label = methods[0] + '_' + methods[1]

                    self.write_inference_tables(results_norefined=inference_results_norefined,
                                                results_refined=inference_results_refined,
                                                filename= methods_label + '_' + network.key + '.csv')

                    # Losses over iterations
                    results[network] = get_loss_and_estimates_over_iterations(
                        results_norefined=learning_results_norefined,
                        results_refined=learning_results_refined)

            self.artist.convergence_networks_experiment(
                results=results,
                # filename='loss-vs-vot-over-iterations_' + str(methods),
                filename='convergence_' + methods_label,
                methods=[methods[0], methods[1]],
                colors=['blue', 'red', 'black', 'green'],
                labels=['Toy', 'Wang', 'Lo', 'Yang'],
                theta_true = self.utility_function.true_values
            )

            plt.show()


class BiasReferenceODExperiment(ConvergenceExperiment):

    def __init__(self,
                 *args,
                 **kwargs):

        # kwargs['folderpath'] = 'small-networks'

        super().__init__(*args, **kwargs)

    def run(self,
            distorted_Q: Matrix):

        # Summary report
        self.write_experiment_report()

        with block_output(show_stdout=False, show_stderr=False):
            # Generate synthetic traffic counts
            counts, _ = self.linkdata_generator.simulate_counts(
                network=self.network,
                equilibrator=self.equilibrator,
                utility_function=self.utility_function)

        # Table 4, Yang and Bell (2000)
        missing_idxs = [0,1,2,3,4,6,7,11,13]

        counts = dict(zip(counts.keys(), masked_observed_counts(counts=np.array(list(counts.values())), idx=missing_idxs).flatten()))

        self.network.load_traffic_counts(counts=counts)

        q0s = {'true_od': self.network.q,
               'distorted_od': denseQ(distorted_Q)}

        results_df = {}

        initial_values = copy.deepcopy(self.utility_function.initial_values)

        for scenario, q0 in q0s.items():

            print('\nScenario:',scenario)

            self.utility_function.initial_values = initial_values

            print('\nStatistical Inference in No Refined Stage')

            self.network.OD.update_Q_from_q(q = q0, Q = self.network.Q)

            # with block_output(show_stdout=False, show_stderr=False):
            learning_results_norefined, inference_results_norefined, best_iter_norefined = \
                self.learner_norefined.statistical_inference()

            theta_norefined = learning_results_norefined[best_iter_norefined]['theta']

            print('\nStatistical Inference in Refined Stage')

            self.utility_function.initial_values = theta_norefined

            # with block_output(show_stdout=False, show_stderr=False):
            learning_results_refined, inference_results_refined, best_iter_refined = \
                self.learner_refined.statistical_inference()

            self.write_inference_tables(results_norefined = inference_results_norefined,
                                       results_refined = inference_results_refined,
                                       filename = scenario)

            results_df[scenario] = self.write_convergence_table(results_norefined=learning_results_norefined,
                                         results_refined=learning_results_refined,
                                         filename='convergence_' + scenario + '.csv', float_format='%.3f')

        # Joint bilevel optimization convergence plot
        self.artist.convergence_experiment_yang(
            results_df =results_df,
            filename='convergence_' + self.network.key,
            methods= [self.outer_optimizer_norefined.method.key, self.outer_optimizer_refined.method.key],
            theta_true= self.utility_function.true_values,
            colors=['blue', 'red'],
            labels=['True O-D', 'Distorted O-D']
        )

        plt.show()


class IrrelevantAttributesExperiment(NetworkExperiment):

    def __init__(self,
                 outer_optimizers: List[OuterOptimizer],
                 equilibrator: LUE_Equilibrator,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # It is assumed that the same equilibrator is used for both learners
        self.equilibrator = equilibrator

        # Learner in no refined stage

        self.learner_norefined = Learner(
            equilibrator=self.equilibrator,
            outer_optimizer=outer_optimizers[0],
            utility_function=copy.deepcopy(self.utility_function),
            network=self.network,
            name='norefined')

        # First class of learner in refined stage

        self.learner_refined_1 = Learner(
            equilibrator=self.equilibrator,
            outer_optimizer=outer_optimizers[1],
            utility_function=copy.deepcopy(self.utility_function),
            network=self.network,
            name='refined1')

        # Second class of learner in refined stage
        self.learner_refined_2 = Learner(
            equilibrator=self.equilibrator,
            outer_optimizer=outer_optimizers[2],
            utility_function=copy.deepcopy(self.utility_function),
            network=self.network,
            name='refined2')

        self.learners = [self.learner_norefined, self.learner_refined_1, self.learner_refined_2]

    def run(self,
            iteration_plot=False,
            **kwargs):

        self.setup_experiment(**kwargs)

        self.write_experiment_report()

        results_experiment = pd.DataFrame({})

        alpha = self.options['alpha']
        range_initial_values = self.options['range_initial_values']
        bilevel_iters = self.options['bilevel_iters']
        replicates = self.options['replicates']

        # Second order optimization fail when the starting point for optimization is far from true, e.g. +1

        for replicate in range(1, replicates + 1):

            printIterationBar(replicate, replicates, prefix='\nProgress:', suffix='', length=20)

            # Generate synthetic traffic counts
            counts, _ = self.linkdata_generator.simulate_counts(
                network=self.network,
                equilibrator=self.equilibrator,
                utility_function=self.utility_function)

            self.network.load_traffic_counts(counts=counts)

            print('\nreplicate: ' + str(replicate + 1))

            # Initilization of initial estimate
            if range_initial_values is not None:
                self.utility_function.random_initializer(range_initial_values)
            else:
                self.utility_function.zero_initializer()

            method_norefined = self.learner_norefined.outer_optimizer.options['method']
            method_refined_1 = self.learner_refined_1.outer_optimizer.options['method']
            method_refined_2 = self.learner_refined_2.outer_optimizer.options['method']

            t0 = time.time()

            print('\nStatistical Inference in No Refined Stage using', method_norefined)

            learning_results_norefined, inference_results_norefined, best_iter_norefined = \
                self.learner_norefined.statistical_inference(bilevel_iters=bilevel_iters,
                                                             alpha=alpha)

            theta_norefined = copy.deepcopy(learning_results_norefined[best_iter_norefined]['theta'])

            time_norefined = time.time() - t0

            t0 = time.time()

            print('\nStatistical Inference in Refined Stage using', method_refined_1)

            self.learner_refined_1.utility_function.initial_values = theta_norefined

            learning_results_refined_1, inference_results_refined_1, best_iter_refined_1 = \
                self.learner_refined_1.statistical_inference(bilevel_iters=bilevel_iters,
                                                             alpha=alpha)

            theta_refined_1 = learning_results_refined_1[best_iter_refined_1]['theta']

            time_refined_1 = time.time() - t0 + time_norefined

            t0 = time.time()

            # Combined methods

            print('\nStatistical Inference in Refined Stage using', method_refined_2)

            self.learner_refined_2.utility_function.initial_values = theta_norefined

            learning_results_refined_2, inference_results_refined_2, best_iter_refined_2 = \
                self.learner_refined_2.statistical_inference(bilevel_iters=bilevel_iters)

            theta_refined_2 = learning_results_refined_2[best_iter_refined_2]['theta']

            time_refined_2 = time.time() - t0 + time_norefined

            # Prepare pandas dataframe
            inference_results_norefined = \
                pd.concat([inference_results_norefined['parameters'],
                           pd.DataFrame({'parameter': ['vot'], 'est': compute_vot(theta_norefined)})])

            inference_results_norefined = inference_results_norefined.assign(
                time=time_norefined, method=method_norefined, rep=replicate)

            inference_results_refined_1 = \
                pd.concat([inference_results_refined_1['parameters'],
                           pd.DataFrame({'parameter': ['vot'], 'est': compute_vot(theta_refined_1)})])

            inference_results_refined_1 = inference_results_refined_1.assign(
                time=time_refined_1, method=method_norefined + '+' + method_refined_1, rep=replicate)

            inference_results_refined_2 = \
                pd.concat([inference_results_refined_2['parameters'],
                           pd.DataFrame({'parameter': ['vot'], 'est': compute_vot(theta_refined_2)})])

            inference_results_refined_2 = inference_results_refined_2.assign(
                time=time_refined_2, method=method_norefined + '+' + method_refined_2, rep=replicate)

            results_replicate = pd.concat([inference_results_norefined,
                                           inference_results_refined_1,
                                           inference_results_refined_2])

            results_replicate = self.assign_stats_columns(results=results_replicate, alpha=alpha,
                                                          theta_true=self.utility_function.true_values)

            results_replicate = results_replicate.rename(
                columns={"CI": "width_confint", "parameter": "attr", 'p-value': 'pvalue', 'est': 'theta'})

            results_experiment = pd.concat([results_experiment, results_replicate])

            self.write_replicate_table(df=results_replicate, filename='inference', replicate=replicate)

            self.write_replicate_table(df=results_experiment, filename='cumulative_inference', replicate=replicate)

            # Visualization

            if method_refined_1 == 'gauss-newton':
                method_refined_1 = 'gn'

            if method_refined_2 == 'gauss-newton':
                method_refined_2 = 'gn'

            self.options['methods'] = [method_norefined, method_norefined + '+' + method_refined_1,
                                       method_norefined + '+' + method_refined_2]

            # Visualization
            self.artist.consistency_experiment(
                results_experiment=results_experiment,
                range_initial_values=range_initial_values,
                alpha = alpha,
                folder=self.dirs['replicate_folder'])

            if iteration_plot:
                plt.show()

        self.artist.consistency_experiment(
            results_experiment=results_experiment,
            range_initial_values=range_initial_values,
            alpha = alpha,
            folder=self.dirs['experiment_folder'])

        plt.show()

