"""
Writer create reports of the networks

"""

import os
import numpy as np
import pandas as pd
import config
from paths import Path
import csv
import scipy.sparse
# from scipy import sparse, io
import time
import tables
import h5py
# import omx
import printer

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


def write_internal_paths(paths: [Path], network_id: str, filename=None):
    """ If two identical paths are written, it means that it is a multidinetwork """

    t0 = time.time()

    # TODO: Eliminate this dependency from config module
    root_dir = config.Config('default').paths['network_data'] + 'paths/' + 'paths-' + network_id

    if filename is not None:
        root_dir = config.Config('default').paths['network_data'] + 'paths/' + filename

    lines = []

    total_paths = len(paths)

    for path, counter in zip(paths, range(total_paths)):
        printer.printProgressBar(counter, total_paths, prefix='Progress (paths):', suffix='', length=20)

        line = []
        for node in path.nodes:
            line.append(node.key)
        # print(line)

        lines.append(line)

    with open(root_dir + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(lines)

    print(str(total_paths) + ' paths were written in ' + str(np.round(time.time() - t0, 1)) + '[s]')


def write_internal_C(C: np.ndarray, network_id: str, sparse_format=False):

    format_label = 'sparse' if sparse_format else 'dense'

    print('Writing C in ' + format_label + ' format')

    t0 = time.time()

    # TODO: Eliminate this dependency from config module
    filefolder = config.Config('default').paths['network_data'] + 'C/'

    # if filename is not None:
    #     root_dir = config.paths['network_data'] + 'C/' + filename

    if sparse_format:
        filepath = filefolder + 'C-sparse-' + network_id + '.npz'
        scipy.sparse.save_npz(filepath, scipy.sparse.csc_matrix(C))

    else:

        filepath = filefolder + 'C-' + network_id + '.csv'

        lines = []
        total_rows = C.shape[0]

        for row, counter in zip(C, range(total_rows)):
            printer.printProgressBar(counter, total_rows, prefix='Progress (C):', suffix='', length=20)

            lines.append(row)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(lines)


    print('Matrix C ' + str(C.shape) + ' written in ' + str(
        round(time.time() - t0, 1)) + '[s]')


def write_internal_D(D: np.ndarray, network_id: str, sparse_format=False):

    format_label = 'sparse' if sparse_format else 'dense'

    print('Writing D in ' + format_label + ' format')

    t0 = time.time()

    # TODO: Eliminate this dependency from config module
    filefolder = config.Config('default').paths['network_data'] + 'D/'

    # if filename is not None:
    #     root_dir = config.paths['network_data'] + 'D/' + filename

    if sparse_format:
        filepath = filefolder + 'D-sparse-' + network_id + '.npz'
        scipy.sparse.save_npz(filepath, scipy.sparse.csc_matrix(D))

    else:

        filepath = filefolder + 'D-' + network_id + '.csv'

        lines = []
        total_rows = D.shape[0]

        for row, counter in zip(D, range(total_rows)):
            printer.printProgressBar(counter, total_rows, prefix='Progress (D):', suffix='', length=20)

            lines.append(row)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(lines)


    print('Matrix D ' + str(D.shape) + ' written in ' + str(
        round(time.time() - t0, 1)) + '[s]')


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
#     # Nt.M
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

def write_internal_M(M: np.ndarray, network_id: str, sparse_format=False):

    format_label = 'sparse' if sparse_format else 'dense'

    print('Writing M in ' + format_label + ' format')

    t0 = time.time()

    # TODO: Eliminate this dependency from config module
    filefolder = config.Config('default').paths['network_data'] + 'M/'

    # if filename is not None:
    #     root_dir = config.paths['network_data'] + 'M/' + filename

    if sparse_format:
        filepath = filefolder + 'M-sparse-' + network_id + '.npz'

        M_sparse = scipy.sparse.csc_matrix(M)
        # print('Can generate sparse matrix')
        scipy.sparse.save_npz(filepath, scipy.sparse.csc_matrix(M_sparse))

        # print('It could read sparse matrix')

    else:

        filepath = filefolder + 'M-' + network_id + '.csv'

        lines = []
        total_rows = M.shape[0]

        for row, counter in zip(M, range(total_rows)):
            printer.printProgressBar(counter, total_rows, prefix='Progress (M):', suffix='', length=20)

            lines.append(row)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(lines)

    print('Matrix M ' + str(M.shape) + ' written in ' + str(
        round(time.time() - t0, 1)) + '[s]')


def write_internal_Q(Q: np.ndarray, network_id: str, sparse_format=False):
    # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.sparse.save_npz.html  # scipy.sparse.save_npz

    format_label = 'sparse' if sparse_format else 'dense'

    print('Writing Q in ' + format_label + ' format')

    t0 = time.time()

    # TODO: Eliminate this dependency from config module
    filefolder = config.Config('default').paths['network_data'] + 'Q/'

    if sparse_format:
        filepath = filefolder + 'Q-sparse-' + network_id + '.npz'
        scipy.sparse.save_npz(filepath, scipy.sparse.csc_matrix(Q))

    else:

        filepath = filefolder + 'Q-' + network_id + '.csv'

        lines = []
        total_rows = Q.shape[0]

        for row, counter in zip(Q, range(total_rows)):
            printer.printProgressBar(counter, total_rows, prefix='Progress (Q):', suffix='', length=20)

            lines.append(row)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(lines)

    print('Matrix Q ' + str(Q.shape) + ' written in ' + str(
        round(time.time() - t0, 1)) + '[s]')


def write_internal_network_files(Nt, options):
    """ Wrapper function for writing"""

    if options['writing']['paths']:
        write_internal_paths(Nt.paths, Nt.key)

    if options['writing']['C']:
        sparse_format = False

        if options['writing']['sparse_C']:
            sparse_format = True

        write_internal_C(Nt.C, Nt.key, sparse_format=sparse_format)

    if options['writing']['D']:

        sparse_format = False

        if options['writing']['sparse_D']:
            sparse_format = True

        write_internal_D(Nt.D, Nt.key, sparse_format=sparse_format)

    if options['writing']['M']:

        sparse_format = False

        if options['writing']['sparse_M']:
            sparse_format = True

        write_internal_M(Nt.M, Nt.key, sparse_format=sparse_format)

    if options['writing']['Q']:

        sparse_format = False

        if options['writing']['sparse_Q']:
            sparse_format = True

        write_internal_Q(Q=Nt.Q, network_id=Nt.key, sparse_format=sparse_format)

def write_csv_to_log_folder(log_file, df, filename, float_format='%.3f'):

    if 'folderpath' in log_file.keys():

        # Locate subfolder within folderpath with the network name
        folderpath = log_file['folderpath']

        # Export tables in txt format
        df.to_csv(folderpath + '/' + filename + '.csv', sep=',', encoding='utf-8',index=False, float_format = float_format)

    # ...

def write_csv_to_experiment_log_folder(log_file, df, filename, float_format='%.3f'):

    if 'folderpath' in log_file.keys():

        # Locate subfolder within folderpath with the network name
        # folderpath = log_file['folderpath']

        if 'subfolderpath' in log_file:
            subfolderpath = log_file['subfolderpath']

            # Export tables in txt format
            df.to_csv(subfolderpath + '/' + filename + '.csv', sep=',', encoding='utf-8', index=False,
                      float_format=float_format)

        else:

            # Export tables in txt format
            df.to_csv(log_file['folderpath'] + '/' + filename + '.csv', sep=',', encoding='utf-8', index=False,
                      float_format=float_format)




    # ...


def write_figure_to_log_folder(log_file, filename, fig, dpi=50):

    folderpath = log_file['folderpath']
    fig.savefig(folderpath + '/' + filename, pad_inches=0.1, bbox_inches="tight", dpi= dpi)  #

def write_figure_experiment_to_log_folder(log_file, filename, fig, dpi=50):

    subfolderpath = log_file['subfolderpath']
    fig.savefig(subfolderpath + '/' + filename, pad_inches=0.1, bbox_inches="tight", dpi= dpi)  #

def write_estimation_report(filename, config, decimals, float_format = None):

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

        report_dict['alpha_selection_norefined'] =  config.estimation_options['alpha_selection_norefined']

        report_dict['n_paths_column_generation'] = config.estimation_options['n_paths_column_generation']

        report_dict['ods_coverage_column_generation']  = config.estimation_options['ods_coverage_column_generation']

        # report_dict['k_path_set_selection']  = config.estimation_options['k_path_set_selection']
        #
        # report_dict['dissimilarity_weight'] = config.estimation_options['dissimilarity_weight']

        report_dict['bilevel_iters_norefined'] = config.estimation_options['bilevel_iters_norefined']

        report_dict['bilevel_iters_refined'] = config.estimation_options['bilevel_iters_refined']

        report_dict['outeropt_method_norefined'] =  config.estimation_options['outeropt_method_norefined']

        report_dict['outeropt_method_refined'] = config.estimation_options['outeropt_method_refined']

        report_dict['eta_norefined'] = config.estimation_options['eta_norefined']

        report_dict['eta_refined'] = config.estimation_options['eta_refined']

        report_dict['true_logit_vector'] = config.theta_true

        report_dict['initial_logit_vector'] = config.theta_0

        report_dict['theta_norefined'] = str({key:round(val, decimals) for key, val in config.estimation_results['theta_norefined'].items()})

        report_dict['theta_refined'] = str({key:round(val, decimals) for key, val in config.estimation_results['theta_refined'].items()})


    report_dict['mean_count_benchmark_model'] = '{:,}'.format(round(float(config.estimation_results['mean_count_benchmark_model']), decimals))

    report_dict['mean_link_counts_prediction_loss'] = '{:,}'.format(round(float(config.estimation_results['mean_counts_prediction_loss']), decimals))

    report_dict['equilikely_path_choice_prediction_loss'] = '{:,}'.format(round(float(config.estimation_results['equilikely_prediction_loss']), decimals))

    report_dict['norefined_prediction_loss'] = '{:,}'.format(round(float(config.estimation_results['best_loss_norefined']), decimals))

    report_dict['refined_prediction_loss'] = '{:,}'.format(round(float(config.estimation_results['best_loss_refined']), decimals))

    report_dict['norefined_prediction_improvement'] = "{:.1%}".format(
        1 - config.estimation_results['best_loss_norefined'] / config.estimation_results['equilikely_prediction_loss'])

    report_dict['refined_prediction_improvement'] = "{:.1%}".format(1-config.estimation_results['best_loss_refined']/config.estimation_results['equilikely_prediction_loss'])

    report_df = pd.DataFrame({'item': report_dict.keys(), 'value': report_dict.values()})

    write_csv_to_log_folder(log_file = config.log_file, df = report_df, filename = filename, float_format = float_format)

def write_experiment_report(filename, decimals, config):
    report_dict = {}

    report_dict['experiment_mode'] = config.experiment_options['experiment_mode']
    report_dict['uncongested_mode'] = config.experiment_options['uncongested_mode']

    report_dict['initial_logit_vector'] = config.experiment_options['theta_0']
    report_dict['true_logit_vector'] = config.experiment_options['theta_true']

    report_dict['noise_params'] = config.experiment_options['noise_params']

    if 'theta_0_range' in config.experiment_options:
        report_dict['theta_0_range'] = config.experiment_options['theta_0_range']
    else:
        report_dict['theta_0_range'] = report_dict['initial_logit_vector']


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
            report_dict['theta_estimates_' + method] = str({key: round(val, decimals) for key, val in estimates.items()})

        # Add VOT estimates
        # report_dict['vot_estimates']
        vot_estimates = {}
        for method, estimates in config.experiment_results['theta_estimates'].items():
            vot_estimates[method] = str(round(estimates['tt'] / estimates['c'], decimals))
            # report_dict['vot_estimate' + method] = str(round(estimates['tt'] / estimates['c'], decimals))

        report_dict['vot_estimates'] = vot_estimates

        # for method, losses in  config.experiment_results['losses'].items():
        #     report_dict['losses_' + method] = str({key: round(val, decimals) for key, val in losses.items()})

        report_dict['losses'] = {key: '{:,}'.format(round(val,decimals)) for key, val in config.experiment_results['losses'].items()}

    report_df = pd.DataFrame({'item': report_dict.keys(), 'value': report_dict.values()})


    write_csv_to_experiment_log_folder(log_file = config.log_file, df = report_df, filename = filename)


def write_experiment_options_report(filename, config):
    options_df = pd.DataFrame({'group': [], 'option': [], 'value': []})

    for key, value in config.experiment_options.items():
        options_df = options_df.append({'group': 'experiment_options', 'option': key, 'value': value},ignore_index=True)

    write_csv_to_log_folder(df=options_df,
                                       filename= filename
                                       , log_file=config.log_file
                                       , float_format='%.1f'
                                       )



