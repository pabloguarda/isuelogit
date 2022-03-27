"""
Writer create reports of the networks

"""

import os
import numpy as np
import pandas as pd
import csv
import time
import random
import scipy.sparse
import copy
from abc import ABC, abstractmethod

# from scipy import sparse, io
# import tables
# import h5py
# import omx

import printer
import config
from paths import Path
from visualization import Artist
from descriptive_statistics import get_gap_estimates_over_iterations, get_loss_and_estimates_over_iterations,get_predicted_traveltimes_over_iterations_df, get_gap_predicted_link_counts_over_iterations_df, get_predicted_link_counts_over_iterations_df
from utils import Options

class Reporter(ABC):

    def __init__(self,
                 foldername=None,
                 seed: int = None,
                 network=None,
                 utility_function = None,
                 linkdata_generator=None,
                 name: str = None,
                 learners = None,
                 **kwargs
                 ):


        self.dirs = {}
        self.artist = Artist()

        self.name = name
        self.network = network
        self.linkdata_generator = linkdata_generator
        self.utility_function = utility_function
        self.learners = learners

        # Seed for reproducibility and consistency in estimation results
        self.seed = seed

        if self.seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.options = Options()
        self.options = self.options.get_updated_options(new_options=kwargs)

        # Store options
        self.options['name'] = self.name
        datetime = time.strftime("%y%m%d %H%M%S")
        self.options['date'] = datetime.split(' ')[0]
        self.options['time'] = datetime.split(' ')[1]
        self.options['seed'] = self.seed

        # Create report folders
        self.make_dirs(foldername)

    # @abstractmethod
    # def set_default_options(self):
    #     raise NotImplementedError

    def get_updated_options(self, **kwargs):
        return copy.deepcopy(self.options.get_updated_options(new_options=kwargs))

    def update_options(self, **kwargs):
        self.options = self.get_updated_options(**kwargs)

    def add_items_report(self, **kwargs):
        self.update_options(**kwargs)
        # self.options = self.options.get_updated_options()

    def make_dirs(self, foldername = None):
        '''

        Store results into log file and store a folder with a summary of the estimation
        Create a subfoldername to store the estimates of the current experiment

        Args:
            foldername:

        Returns:

        '''

        if foldername is None:
            foldername = self.network.key

        # Create a subfolder based on starting date
        self.dirs['estimation_folder'] = \
            config.dirs['output_folder'] + 'estimations/' + foldername + '/' + self.options['date']

        if not os.path.exists(self.dirs['estimation_folder']):
            os.makedirs(self.dirs['estimation_folder'])

        # Create a subfolder based on starting date and time of the simulation
        self.dirs['estimation_folder'] += '/' + self.options['time']

        os.makedirs(self.dirs['estimation_folder'])

    def write_estimation_report(self,
                                network = None,
                                learners = None,
                                utility_function = None,
                                linkdata_generator = None,
                                filename = None,
                                **kwargs):

        if network is None:
            network = self.network

        if network is not None:
            # Network information
            self.options['network'] = network.key
            self.options['links'] = len(network.links)
            self.options['paths'] = len(network.paths)
            self.options['ods'] = len(network.ods)
            self.options['scale_OD'] = network.OD.scale

        if utility_function is None:
            utility_function = self.utility_function

        if utility_function is not None:
            self.options['features'] = utility_function.features
            self.options['initial parameters'] = utility_function.initial_values
            self.options['true parameters'] = utility_function.true_values

        if linkdata_generator is None:
            linkdata_generator = self.linkdata_generator

        if linkdata_generator is not None:
            self.options['data_generator'] = linkdata_generator.options

        if learners is None:
            learners = self.learners

        if learners is not None:
            for learner in learners:

                self.options[learner.name + '_learner'] \
                    = {k: v for k, v in learner.options.items() if
                       k in ['bilevel_iters']}

                self.options[learner.name + '_optimizer'] \
                    = {k: v for k, v in learner.outer_optimizer.options.items() if
                       k in ['method', 'eta', 'iters']}

                self.options[learner.name + '_equilibrator'] = {k: v for k, v in learner.equilibrator.options.items()}

        if filename is None:
            filename = 'estimation_report.csv'

        df = pd.DataFrame({'option': self.options.keys(), 'value': self.options.values()})

        self.write_table(df = df, folder = self.dirs['estimation_folder'], filename =  filename, **kwargs)

    def write_table(self,
                    df,
                    filename = None,
                    folder = None,
                    **kwargs):

        if folder is None:
            folder = self.dirs['estimation_folder']

        df.to_csv(folder + '/' + filename,
                  sep=',',
                  encoding='utf-8',
                  index=False,
                  **kwargs
                  )

    def write_learning_tables(self,
                              results_norefined,
                              results_refined,
                              network,
                              utility_function,
                              simulated_data = False
                              ):

       # Analysis of predicted counts over iterations
        predicted_link_counts_over_iterations_df = \
            get_predicted_link_counts_over_iterations_df(
                results_norefined=results_norefined,
                results_refined=results_refined,
                network=network
            )

        gap_predicted_link_counts_over_iterations_df = \
            get_gap_predicted_link_counts_over_iterations_df(
                results_norefined=results_norefined,
                results_refined=results_refined,
                network=network
            )

       # Analysis of travel times over iterations
        predicted_link_traveltime_over_iterations_df = \
            get_predicted_traveltimes_over_iterations_df(
                results_norefined=results_norefined,
                results_refined=results_refined,
                network=network
            )

        if simulated_data:

            gap_estimates_over_iterations_df = \
                get_gap_estimates_over_iterations(
                    results_norefined=results_norefined
                    , results_refined=results_refined
                    , theta_true= utility_function.true_values
                )

            self.write_table(df=gap_estimates_over_iterations_df,
                             filename='gap_estimates_over_iterations_df.csv',
                             float_format = '%.3f')

        # print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
        self.write_table(df=predicted_link_counts_over_iterations_df,
                         filename='predicted_link_counts_over_iterations_df.csv',
                         float_format='%.1f')

        self.write_table(df=gap_predicted_link_counts_over_iterations_df,
                         filename='gap_predicted_link_counts_over_iterations_df.csv',
                         float_format='%.1f')


        self.write_table(df=predicted_link_traveltime_over_iterations_df,
                         filename='predicted_link_traveltimes_over_iterations_df.csv',
                         float_format='%.2f')

        # Convergence of objective function and parameters
        self.write_convergence_table(results_norefined=results_norefined,
                                     results_refined=results_refined,
                                     float_format='%.3f')

    def write_inference_tables(self,
                               results_norefined,
                               results_refined,
                               folder = None,
                               filename = None,
                               **kwargs):

        parameter_inference_norefined_table = results_norefined['parameters'].copy()
        model_inference_norefined_table = results_norefined['model'].copy()

        parameter_inference_refined_table = results_refined['parameters'].copy()
        model_inference_refined_table = results_refined['model'].copy()

        # T-tests, confidence intervals and parameter estimates
        parameter_inference_norefined_table.insert(0, 'stage', 'norefined')
        parameter_inference_refined_table.insert(0, 'stage', 'refined')
        parameters_inference_table = parameter_inference_norefined_table.append(parameter_inference_refined_table)

        # F-test and model summary statistics
        model_inference_norefined_table.insert(0, 'stage', 'norefined')
        model_inference_refined_table.insert(0, 'stage', 'refined')
        model_inference_table = model_inference_norefined_table.append(model_inference_refined_table)

        filename_parameters = 'parameters_inference_table'
        filename_model = 'model_inference_table'

        if filename is not None:
            filename_parameters += '_' + filename
            filename_model += '_' + filename

        self.write_table(df=parameters_inference_table, filename=filename_parameters+ '.csv', folder = folder,**kwargs)
        self.write_table(df=model_inference_table, filename=filename_model + '.csv', folder = folder,**kwargs)

        # return parameters_inference_table, model_inference_table

    def write_convergence_table(self,
                                results_norefined,
                                results_refined,
                                folder=None,
                                filename=None,
                                **kwargs):

        results_df = get_loss_and_estimates_over_iterations(
            results_norefined=results_norefined, results_refined=results_refined)

        if filename is None:
            filename = 'convergence.csv'

        self.write_table(df=results_df, filename=filename, folder=folder, **kwargs)

        return results_df

def write_tntp_github_to_dat(root, subfolder):
    """
    This method generate dat files that are used by the method  transportAI.equilibrium.sue_logit_dial
    """

    inputLocation = root + subfolder

    od_filename = [_ for _ in os.listdir(os.path.join(root, subfolder)) if 'trips' in _ and _.endswith('tntp')]

    prefix_filenames = od_filename[0].partition('_')[0]

    od = {}
    origin_no = 0  # our starting Origin number in case the file doesn't begin with one
    with open(inputLocation + '/' + od_filename[0], "r") as f:

        new_origin = False  # this boolean was added by PG to correct error in code

        for line in f:
            line = line.rstrip()  # we're not interested in the newline at the
            # end
            if not line:  # empty line, skip
                new_origin = False
                continue

            elif line.startswith("Origin"):
                origin_no = int(line[7:].strip())  # grab the integer following Origin
                new_origin = True
            elif new_origin:
                elements = line.split(";")  # get our elements by splitting by semi-colon
                for element in elements:  # loop through each of them:
                    if not element:  # we're not interested in the last element
                        continue
                    element_no, element_value = element.split(":")  # get our pair
                    # beware, these two are now most likely padded strings!
                    # that's why we'll strip them from whitespace and convert to integer/float
                    if (origin_no != int(element_no.strip())):
                        od[(origin_no, int(element_no.strip()))] = float(element_value.strip())

    network = {}
    metadata = True
    counter = 0

    with open(inputLocation + '/' + prefix_filenames + "_net.tntp", "r") as f:
        # next(f)

        for line in f:
            if counter <= 8:  # empty line, skip
                counter += 1
                continue

            else:
                line = line.rstrip()
                line = line.split(";")[0].split('\t')
                # print([line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8]])
                # init_node	term_node	capacity	length	free_flow_time	b	power	speed	toll	link_type
                network[line[1], line[2]] = [line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8]]

    def write_OD_flows():
        outFile = open(inputLocation + '/' + prefix_filenames + "_demand.dat", "w")
        tmpOut = "origin\tdest\tdemand"
        outFile.write(tmpOut + "\n")
        for d in od:
            tmpOut = str(d[0]) + "\t" + str(d[1]) + "\t" + str(od[d])
            outFile.write(tmpOut + "\n")
        outFile.close()

    write_OD_flows()

    def write_Network():
        outFile = open(inputLocation + '/' + prefix_filenames + "_network.dat", "w")
        tmpOut = "origin\tdest\tcapacity\tlength\tfft\talpha\tbeta\tspeedLimit"
        outFile.write(tmpOut + "\n")
        for link in network:
            tmpOut = '\t'.join(network[link])
            outFile.write(tmpOut + "\n")
        outFile.close()

    write_Network()


def write_network_to_dat(root, subfolder, prefix_filename, N):
    """
    This method generate dat files that are used by the method  transportAI.equilibrium.sue_logit_dial
    """

    inputLocation = root + subfolder

    # prefix_filenames = od_filename[0].partition('_')[0]

    od = {}

    # for i,j in np.ndenumerate(N.Q):
    #     print(i,j)

    def write_OD_flows():
        outFile = open(inputLocation + '/' + prefix_filename + "_demand.dat", "w")
        tmpOut = "origin\tdest\tdemand"
        outFile.write(tmpOut + "\n")
        for od, q in np.ndenumerate(N.Q):
            tmpOut = str(od[0] + 1) + "\t" + str(od[1] + 1) + "\t" + str(q)
            outFile.write(tmpOut + "\n")
        outFile.close()

    write_OD_flows()

    def write_Network():
        outFile = open(inputLocation + '/' + prefix_filename + "_network.dat", "w")
        tmpOut = "origin\tdest\tcapacity\tlength\tfft\talpha\tbeta\tspeedLimit"
        outFile.write(tmpOut + "\n")

        network = {}

        for link in N.links:
            network[link.key[0] + 1, link.key[1] + 1] = [str(link.key[0] + 1), str(link.key[1] + 1),
                                                         str(link.bpr.k), str(link.bpr.tf), str(link.bpr.tf),
                                                         str(link.bpr.alpha), str(link.bpr.beta), str(0)]

        for link in network.keys():
            tmpOut = '\t'.join(network[link])
            outFile.write(tmpOut + "\n")
        outFile.close()

    write_Network()


def write_colombus_ohio_to_tntp_format():
    raise NotImplementedError


def write_internal_paths(paths: [Path],
                         network_key: str,
                         overwrite_input = False,
                         filename=None):
    """ If two identical paths are written, it means that it is a multidinetwork """

    t0 = time.time()

    if overwrite_input:
        folder = config.dirs['read_network_data']
    else:
        folder = config.dirs['write_network_data']

    root_dir = folder  + 'paths/' + 'paths-' + network_key

    if filename is not None:
        root_dir = folder  + 'paths/' + filename

    lines = []

    total_paths = len(paths)

    for path, counter in zip(paths, range(total_paths)):
        printer.printProgressBar(counter, total_paths-1, prefix='Progress (paths):', suffix='', length=20)

        line = []
        for node in path.nodes:
            line.append(node.key)
        # print(line)

        lines.append(line)

    with open(root_dir + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(lines)

    print(str(total_paths) + ' paths were written in ' + str(np.round(time.time() - t0, 1)) + '[s]')


def write_internal_C(C: np.ndarray,
                     network_key: str,
                     overwrite_input = False,
                     sparse_format=False):

    format_label = 'sparse' if sparse_format else 'dense'

    if overwrite_input:
        folder = config.dirs['read_network_data']
    else:
        folder = config.dirs['write_network_data']

    # print('Writing C in ' + format_label + ' format')

    t0 = time.time()

    filefolder = folder + 'C/'

    # if filename is not None:
    #     root_dir = config.paths['write_network_data'] + 'C/' + filename

    if sparse_format:
        filepath = filefolder + 'C-sparse-' + network_key + '.npz'
        scipy.sparse.save_npz(filepath, scipy.sparse.csc_matrix(C))

    else:

        filepath = filefolder + 'C-' + network_key + '.csv'

        lines = []
        total_rows = C.shape[0]

        for row, counter in zip(C, range(total_rows)):
            printer.printProgressBar(counter, total_rows, prefix='Progress (C):', suffix='', length=20)

            lines.append(row)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(lines)


    print('Matrix C ' + str(C.shape) + ' written in ' + str(
        round(time.time() - t0, 1)) + '[s]' + ' with ' + format_label + ' format')


def write_internal_D(D: np.ndarray,
                     network_key: str,
                     overwrite_input = False,
                     sparse_format=False):

    format_label = 'sparse' if sparse_format else 'dense'

    if overwrite_input:
        folder = config.dirs['read_network_data']
    else:
        folder = config.dirs['write_network_data']

    # print('Writing D in ' + format_label + ' format')

    t0 = time.time()

    filefolder = folder + 'D/'

    # if filename is not None:
    #     root_dir = config.paths['write_network_data'] + 'D/' + filename

    if sparse_format:
        filepath = filefolder + 'D-sparse-' + network_key + '.npz'
        scipy.sparse.save_npz(filepath, scipy.sparse.csc_matrix(D))

    else:

        filepath = filefolder + 'D-' + network_key + '.csv'

        lines = []
        total_rows = D.shape[0]

        for row, counter in zip(D, range(total_rows)):
            printer.printProgressBar(counter, total_rows, prefix='Progress (D):', suffix='', length=20)

            lines.append(row)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(lines)


    print('Matrix D ' + str(D.shape) + ' written in ' + str(
        round(time.time() - t0, 1)) + '[s]' + ' with ' + format_label + ' format')

def write_internal_M(M: np.ndarray,
                     network_key: str,
                     overwrite_input = False,
                     sparse_format=False):

    format_label = 'sparse' if sparse_format else 'dense'

    if overwrite_input:
        folder = config.dirs['read_network_data']
    else:
        folder = config.dirs['write_network_data']

    t0 = time.time()

    filefolder = folder + 'M/'

    # if filename is not None:
    #     root_dir = config.paths['write_network_data'] + 'M/' + filename

    if sparse_format:
        filepath = filefolder + 'M-sparse-' + network_key + '.npz'

        M_sparse = scipy.sparse.csc_matrix(M)
        # print('Can generate sparse matrix')
        scipy.sparse.save_npz(filepath, scipy.sparse.csc_matrix(M_sparse))

        # print('It could read sparse matrix')

    else:

        filepath = filefolder + 'M-' + network_key + '.csv'

        lines = []
        total_rows = M.shape[0]

        for row, counter in zip(M, range(total_rows)):
            printer.printProgressBar(counter, total_rows, prefix='Progress (M):', suffix='', length=20)

            lines.append(row)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(lines)

    print('Matrix M ' + str(M.shape) + ' written in ' + str(
        round(time.time() - t0, 1)) + '[s]' + ' with ' + format_label + ' format')


def write_internal_Q(Q: np.ndarray,
                     network_key: str,
                     overwrite_input = False,
                     sparse_format=False):

    format_label = 'sparse' if sparse_format else 'dense'

    if overwrite_input:
        folder = config.dirs['read_network_data']
    else:
        folder = config.dirs['write_network_data']

    t0 = time.time()

    # print('Writing Q in ' + format_label + ' format')

    t0 = time.time()

    filefolder = folder + 'Q/'

    if sparse_format:
        filepath = filefolder + 'Q-sparse-' + network_key + '.npz'
        scipy.sparse.save_npz(filepath, scipy.sparse.csc_matrix(Q))

    else:

        filepath = filefolder + 'Q-' + network_key + '.csv'

        lines = []
        total_rows = Q.shape[0]

        for row, counter in zip(Q, range(total_rows)):
            printer.printProgressBar(counter, total_rows, prefix='Progress (Q):', suffix='', length=20)

            lines.append(row)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(lines)

    print('Matrix Q ' + str(Q.shape) + ' written in ' + str(
        round(time.time() - t0, 1)) + '[s]' + ' with ' + format_label + ' format')


def write_internal_network_files(network, options, **kwargs):
    """ Wrapper function for writing"""

    if options['writing']['C'] or options['writing']['sparse_C']:
        sparse_format = False

        if options['writing']['sparse_C']:
            sparse_format = True

        write_internal_C(network.C, network.key, sparse_format=sparse_format, **kwargs)

    if options['writing']['D'] or options['writing']['sparse_D']:

        sparse_format = False

        if options['writing']['sparse_D']:
            sparse_format = True

        write_internal_D(network.D, network.key, sparse_format=sparse_format, **kwargs)

    if options['writing']['M'] or options['writing']['sparse_M']:

        sparse_format = False

        if options['writing']['sparse_M']:
            sparse_format = True

        write_internal_M(network.M, network.key, sparse_format=sparse_format, **kwargs)

    if options['writing']['Q'] or options['writing']['sparse_Q']:

        sparse_format = False

        if options['writing']['sparse_Q']:
            sparse_format = True

        write_internal_Q(Q=network.Q, network_key=network.key, sparse_format=sparse_format, **kwargs)

def write_csv_to_log_folder(log_file, df, filename, float_format='%.3f'):

    # if 'folderpath' in log_file.keys():

    # Locate subfoldername within folderpath with the network name
    folderpath = log_file['folderpath']

    # Export tables in txt format
    df.to_csv(folderpath + '/' + filename + '.csv', sep=',', encoding='utf-8',index=False, float_format = float_format)

    # ...

def write_csv_to_experiment_log_folder(log_file, df, filename, float_format='%.3f'):

    # assert set(['experimentpath','replicatepath']).issubset(log_file.keys())

        # Locate subfoldername within folderpath with the network name
        # folderpath = log_file['folderpath']

        if 'replicatepath' in log_file.keys():

            # Export tables in csv format
            df.to_csv(log_file['replicatepath'] + '/' + filename + '.csv', sep=',', encoding='utf-8', index=False,float_format=float_format)

        elif 'experimentpath' in log_file.keys():
            df.to_csv(log_file['folderpath'] + '/' + filename + '.csv', sep=',', encoding='utf-8', index=False,
                      float_format=float_format)

    # ...

def write_figure_to_log_folder(log_file, filename, fig, dpi=50):

    folderpath = log_file['folderpath']
    fig.savefig(folderpath + '/' + filename, pad_inches=0.1, bbox_inches="tight", dpi= dpi)  #

def write_figure_experiment_to_log_folder(log_file, filename, fig, dpi=50):

    subfolderpath = log_file['subfolderpath']
    fig.savefig(subfolderpath + '/' + filename, pad_inches=0.1, bbox_inches="tight", dpi= dpi)  #


def write_estimation_report_1(filename,
                            config,
                            decimals,
                            float_format=None):

    report_dict = {}

    if 'selected_date' in config.estimation_options.keys():
        report_dict['selected_date'] = config.estimation_options['selected_date']

    if 'selected_hour' in config.estimation_options.keys():
        report_dict['selected_hour'] = config.estimation_options['selected_hour']

    report_dict['uncongested_mode'] = config.sim_options['uncongested_mode']

    report_dict['outofsample_prediction_mode'] = config.estimation_options['outofsample_prediction_mode']

    report_dict['known_pathset_mode'] = config.estimation_options['known_pathset_mode']

    if report_dict['outofsample_prediction_mode']:
        report_dict['logit_vector'] = config.theta_0

    else:

        report_dict['simulated_counts'] = config.sim_options['simulated_counts']

        if config.sim_options['simulated_counts']:

            report_dict['max_link_coverage'] = config.sim_options['max_link_coverage']

            report_dict['noise_params'] = config.sim_options['noise_params']


        else:
            report_dict['data_processing'] = config.gis_options['data_processing']
            report_dict['inrix_matching'] = config.gis_options['inrix_matching']
            report_dict['matching_stats'] = config.gis_results['matching_stats']

        report_dict['ttest_selection_norefined'] = config.estimation_options['ttest_selection_norefined']

        report_dict['alpha_selection_norefined'] = config.estimation_options['alpha_selection_norefined']

        report_dict['n_paths_column_generation'] = config.estimation_options['n_paths_column_generation']

        report_dict['ods_coverage_column_generation'] = config.estimation_options['ods_coverage_column_generation']

        # report_dict['k_path_set_selection']  = config.estimation_options['k_path_set_selection']
        #
        # report_dict['dissimilarity_weight'] = config.estimation_options['dissimilarity_weight']

        report_dict['bilevel_iters_norefined'] = config.estimation_options['bilevel_iters_norefined']

        report_dict['bilevel_iters_refined'] = config.estimation_options['bilevel_iters_refined']

        report_dict['outeropt_method_norefined'] = config.estimation_options['outeropt_method_norefined']

        report_dict['outeropt_method_refined'] = config.estimation_options['outeropt_method_refined']

        report_dict['eta_norefined'] = config.estimation_options['eta_norefined']

        report_dict['eta_refined'] = config.estimation_options['eta_refined']

        report_dict['true_logit_vector'] = config.theta_true

        report_dict['initial_logit_vector'] = config.theta_0

        report_dict['theta_norefined'] = str(
            {key: round(val, decimals) for key, val in config.estimation_results['theta_norefined'].items()})

        report_dict['theta_refined'] = str(
            {key: round(val, decimals) for key, val in config.estimation_results['theta_refined'].items()})

    report_dict['mean_count_benchmark_model'] = '{:,}'.format(
        round(float(config.estimation_results['mean_count_benchmark_model']), decimals))

    report_dict['mean_link_counts_prediction_loss'] = '{:,}'.format(
        round(float(config.estimation_results['mean_counts_prediction_loss']), decimals))

    report_dict['equilikely_path_choice_prediction_loss'] = '{:,}'.format(
        round(float(config.estimation_results['equilikely_prediction_loss']), decimals))

    report_dict['norefined_prediction_loss'] = '{:,}'.format(
        round(float(config.estimation_results['best_loss_norefined']), decimals))

    report_dict['refined_prediction_loss'] = '{:,}'.format(
        round(float(config.estimation_results['best_loss_refined']), decimals))

    report_dict['norefined_prediction_improvement'] = "{:.1%}".format(
        1 - config.estimation_results['best_loss_norefined'] / config.estimation_results[
            'equilikely_prediction_loss'])

    report_dict['refined_prediction_improvement'] = "{:.1%}".format(
        1 - config.estimation_results['best_loss_refined'] / config.estimation_results[
            'equilikely_prediction_loss'])

    report_df = pd.DataFrame({'item': report_dict.keys(), 'value': report_dict.values()})

    write_csv_to_log_folder(log_file=config.log_file, df=report_df, filename=filename, float_format=float_format)



# def write_big_matrix(matrix):
#     # https: // stackoverflow.com / questions / 8843062 / python - how - to - store - a - numpy - multidimensional - array - in -pytables
#     # https://www.pytables.org/usersguide/tutorials.html
#
#     # TODO: read about OMX https://github.com/osPlanning/omx-python/blob/master/example/python-omx-sample.py. I should write the matrix by origin node
#
#     from scipy.sparse import csr_matrix, rand
#     import tables
#     a = rand(2000, 2000, format='csr')  # imagine that many values are stored in this matrix and that sparsity is low
#
#
#     # Pytables
#
#     # https: // stackoverflow.com / questions / 8843062 / python - how - to - store - a - numpy - multidimensional - array - in -pytables / 8843489  # 8843489
#
#
#     # f = tables.open_file('test.hdf', 'w')
#     # atom = tables.Atom.from_dtype(x.dtype)
#     # filters = tables.Filters(complib='blosc', complevel=5)
#     # ds = f.create_carray(f.root, 'somename', atom, x.shape, filters=filters)
#     # ds[:] = x
#     # f.close()
#
#
#     #h5py
#
#     # https: // stackoverflow.com / questions / 20928136 / input - and -output - numpy - arrays - to - h5py / 20938742  # 20938742
#     # import h5py
#     # x = N['train']['Colombus'].M
#     # h5f = h5py.File('io/network-data/M/M_Colombus.h5', 'w')
#     # h5f.create_dataset('test', data=x)
#
#
#
#
#     h5f.close()
#
#     # load data
#     h5f = h5py.File('data.h5', 'r')
#     b = h5f['dataset_1'][:]
#     h5f.close()
#
#
#     # sparse_M = csr_matrix(N['train']['Colombus'].M)
#
#
#
#     # network.M
#     b = a.T
#     l, m, n = a.shape[0], a.shape[1], b.shape[1]
#
#     f = tb.open_file('dot.h5', 'w')
#     filters = tb.Filters(complevel=5, complib='blosc')
#     out = f.create_carray(f.root, 'data', tb.Float32Atom(), shape=(l, n), filters=filters)
#
#     bl = 1000  # this is the number of rows we calculate each loop
#     # this may not the most efficient value
#     # look into buffersize usage in PyTables and adopt the buffersite of the
#     # carray accordingly to improve specifically fetching performance
#
#     b = b.tocsc()  # we slice b on columns, csc improves performance
#
#     # this can also be changed to slice on rows instead of columns
#     for i in range(0, l, bl):
#         out[:, i:min(i + bl, l)] = (a.dot(b[:, i:min(i + bl, l)])).toarray()
#
#     f.close()
#
#
#     raise NotImplementedError