# =============================================================================
# 1) SETUP
# =============================================================================

import os
#=============================================================================
# 1.1) MODULES
#==============================================================================

# Internal modules
import transportAI as tai

# import transportAI.modeller

# External modules
import numpy as np
import sys
import pandas as pd
import os
import copy
import time
import datetime

from scipy import stats

import matplotlib

# from matplotlib import rc
# matplotlib.rcParams['text.usetex'] = True
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)

import matplotlib.pyplot as plt
import seaborn as sns

# Memory usage
import tracemalloc
# https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python

# Configuration object
config = tai.config.Config(network_key = 'Fresno')

# Data analyst
data_analyst = tai.analyst.Analyst()


# Period selected for data analysis

config.estimation_options['selected_hour'] = 16
config.estimation_options['selected_date'] = '2019-10-01' # -> First Tuesday of October, 2019
# config.estimation_options['selected_date'] = '2020-10-06' # -> First Tuesday of October, 2020
config.estimation_options['selected_date_datetime'] =  datetime.date.fromisoformat(config.estimation_options['selected_date'])

# Get year, day of month, month and day of week using datetime package functionalities
config.estimation_options['selected_year'] = config.estimation_options['selected_date_datetime'].year
config.estimation_options['selected_day_month'] = config.estimation_options['selected_date_datetime'].day
config.estimation_options['selected_month'] = config.estimation_options['selected_date_datetime'].month
config.estimation_options['selected_day_week'] = int(config.estimation_options['selected_date_datetime'].strftime('%w'))+1
# Note: for consistency with spark, the weekday number from datetime was changed a bit
# https://stackoverflow.com/questions/9847213/how-do-i-get-the-day-of-week-given-a-date


print('\nSelected date is ' + config.estimation_options['selected_date'] + ', ' + config.estimation_options['selected_date_datetime'].strftime('%A') + ' at ' + str(config.estimation_options['selected_hour']) + ':00' )


#Examples:

# October 1, 2020 is Thursday (day_week = 5).
# config.estimation_options['selected_year'] = 2020
# config.estimation_options['selected_date'] = '2020-10-01'
# print('\nSelected period is October 1, 2020, Thursday at ' + str(config.estimation_options['selected_hour']) + ':00' )

# October 1, 2019 is Tuesday  (day_week = 3).
# config.estimation_options['selected_year'] = 2019
# config.estimation_options['selected_date'] = '2019-10-01'
# print('\nSelected period is October 1, 2019, Tuesday at ' + str(config.estimation_options['selected_hour']) + ':00' )

# OD Demand

# - Periods (6 periods of 15 minutes each)
# config.estimation_options['od_periods'] = [1,2,3]
config.estimation_options['od_periods'] = [1,2,3,4]
# config.estimation_options['od_periods'] = [1]

# Duration of pems counts data retrieval is in minutes
config.estimation_options['selected_period_pems_counts'] = \
    {'hour': config.estimation_options['selected_hour'], 'duration': int(len(config.estimation_options['od_periods'])*15)}

config.estimation_options['selected_period_inrix'] = {}
config.estimation_options['selected_period_inrix'] = \
    {'year': [config.estimation_options['selected_year']], 'month': [config.estimation_options['selected_month']], 'day_month': [config.estimation_options['selected_day_month']], 'hour': [config.estimation_options['selected_hour']-1, config.estimation_options['selected_hour']]}
# config.estimation_options['selected_period_inrix'] = \
#     {'year': [config.estimation_options['selected_year']], 'month': [config.estimation_options['selected_month']]}
# config.estimation_options['selected_period_inrix'] = \
#     {'year': [config.estimation_options['selected_year']], 'month': [config.estimation_options['selected_month']], 'day_week': config.estimation_options['selected_day_week'], 'hour': [config.estimation_options['selected_hour']]}

config.extra_options['write_inrix_daily_data'] = False
config.extra_options['read_inrix_daily_data'] = True

config.estimation_options['selected_period_incidents'] = {}
config.estimation_options['selected_period_incidents'] = {'year': [config.estimation_options['selected_year']], 'month': [9,10]}

# Sean suggests to use a longer time window to avoid too many zero incidents
# config.estimation_options['selected_period_incidents'] =  {'year': [config.estimation_options['selected_year']], 'month': [7,8,9,10]}

# Reading options
config.gis_options['data_processing'] \
    = {'inrix_segments': False, 'inrix_data': False, 'census': False, 'incidents': False, 'bus_stops': False, 'streets_intersections': False}

# config.gis_options['data_processing'] \
#     = {'inrix_segments': True, 'inrix_data': True, 'census': True, 'incidents': True, 'bus_stops': True, 'streets_intersections': True}

# config.gis_options['data_processing'] \
#     = {'inrix_segments': False, 'inrix_data': False, 'census': False, 'incidents': False, 'bus_stops': False, 'streets_intersections':False}

# GIS options

# - Matching GIS layers with inrix instead of the network links.
config.gis_options['inrix_matching']= {'census': False, 'incidents': True, 'bus_stops': True, 'streets_intersections': True}

# config.gis_options['inrix_matching']= {'census': True, 'incidents': True, 'bus_stops': True, 'streets_intersections': True}

# - Buffer (in feets, which is the unit of the CA crs)
config.gis_options['buffer_size'] = {'inrix': 200, 'bus_stops': 50, 'incidents': 50, 'streets_intersections': 50}

# Travel time units
# config.estimation_options['tt_units'] = 'seconds'
config.estimation_options['tt_units'] = 'minutes'
config.estimation_options['update_ff_tt_inrix'] = True

#BPR Function parameters
config.estimation_options['bpr_parameters'] = {'alpha': 0.15, 'beta': 4} #Standard parameters are 0.15 and 4

# Features

# Features in utility function
k_Y = ['tt']
config.estimation_options['k_Z'] = []
# config.estimation_options['k_Z'] = ['tt_sd_adj', 'incidents', 'median_inc', 'intersections', 'bus_stops', 'high_inc']
# config.estimation_options['k_Z'] = ['tt_reliability']
# config.estimation_options['k_Z'] = ['tt_sd_adj', 'no_incidents', 'no_intersections', 'no_bus_stops', 'low_inc']
# config.estimation_options['k_Z'] = ['tt_reliability', 'no_incidents', 'no_intersections', 'no_bus_stops', 'low_inc']
# config.estimation_options['k_Z'] = ['tt_sd_adj','bus_stops']
# config.estimation_options['k_Z'] = ['high_inc']
# config.estimation_options['k_Z'] = ['tt_sd','tt_cv']
# config.estimation_options['k_Z'] = ['tt_sd', 'incidents', 'high_inc']
# config.estimation_options['k_Z'] = ['tt_sd_adj', 'intersections', 'incidents', 'high_inc', 'bus_stops']
# config.estimation_options['k_Z'] = ['high_inc', 'bus_stops', 'speed_sd', 'intersections', 'road_closures']
# config.estimation_options['k_Z'] = ['incidents_year', 'median_inc', 'speed_sd']
# config.estimation_options['k_Z'] = ['speed_sd']
# config.estimation_options['k_Z'] = ['median_inc'] #['n2,n1']
# config.estimation_options['k_Z'] = config.estimation_options['k_Z']

# If any of the two following is set to be equal to none, then  k_Z is used and thus, it includes all features
k_Z_simulation = None #k_Z #
k_Z_estimation = None #k_Y

# Set initial theta
config.theta_0 = dict.fromkeys(k_Y + config.estimation_options['k_Z'], 0)


#To run the algorithm with real data, it suffices to comment out config.set_simulated_counts

# Std of 0.2 is tolerable in terms of consistency of statistical inference
# config.set_simulated_counts(coverage = 0.1, sd_x = 0.4, sd_Q = 0.2)
# config.set_simulated_counts(coverage = 1, sd_x = 0, sd_Q = 0)
# config.set_simulated_counts(coverage = 0.1, sd_x = 0.4, sd_Q = 0.2)

#config.set_simulated_counts(False)

config.sim_options['prop_validation_sample'] = 0
config.sim_options['regularization'] = False

# Number of paths in the initial path set
config.estimation_options['n_initial_paths'] = 2

#Synthetic counts
config.sim_options['n_paths_synthetic_counts'] = 2#None #3  # If none, the initial path set is used to generate counts
config.sim_options['max_sue_iters'] = 30 #itereates to generate synthetic counts

# accuracy for relative gap
config.estimation_options['accuracy_eq'] = 1e-4


# Column generation

config.estimation_options['k_path_set_selection'] = 2
config.estimation_options['dissimilarity_weight'] = 0.5

# Path per od pair for column generation (must be an integer, which if zero, then no column generation is performed)
config.estimation_options['n_paths_column_generation'] = 2 #2
# Coverage of OD pairs to sample new paths
config.estimation_options['ods_coverage_column_generation'] = 0.1 #0.1

# No scaling is performed to compute equilibrium as if normalizing by std, then the solution change significantly.
config.estimation_options['standardization_regularized'] = {'mean': True, 'sd': True}
config.estimation_options['standardization_norefined'] = {'mean': True, 'sd': True}
config.estimation_options['standardization_refined'] = {'mean': False, 'sd': False}

# Fixed effect by link, nodes or OD zone in the simulated experiment
config.sim_options['fixed_effects'] = {'Q': False, 'nodes': False, 'links': True, 'coverage': 0.0}
observed_links_fixed_effects = None #'custom'
theta_true_fixed_effects = 1e4
theta_0_fixed_effects = 0 #1e2 #These should be positive to encourage choosing link with observed counts generally, the coding is 1 for the attribute

# Feature selection based on t-test from no refined step
config.estimation_options['ttest_selection_norefined'] = False #True

# We relax the critical value to remove features that are highly "no significant". A simil of regularization
# config.estimation_options['alpha_selection_norefined'] = 3 #if it is higher than 1, it choose the k minimum values
config.estimation_options['alpha_selection_norefined'] = 0.05

# If no change in the prediction is produced over iterations in the no-refined stages, those counts are removed
config.estimation_options['link_selection'] = False

# Computation of t-test with top percentage of observations in terms of SSE
config.estimation_options['pct_lowest_sse_norefined'] = 100
config.estimation_options['pct_lowest_sse_refined'] = 100

# * It seems scaling helps to speed up convergence

# Optimization methods used in no refined and refined stages
config.estimation_options['outeropt_method_norefined'] = 'ngd' #adam works better with real world data
config.estimation_options['outeropt_method_refined'] = 'ngd'

# Size of batch for pathsand links used to compute gradient
config.estimation_options['paths_batch_size'] = 0
config.estimation_options['links_batch_size'] = 32
# Note: the improvement in speed is impressive with paths but there is inconsistencies

# Learning rate for first order optimization
config.estimation_options['eta_norefined'] = 2e-2
config.estimation_options['eta_refined'] = 1e-2 #5e-2 works well because the decrease is steady although slower

# Bilevel iters
config.estimation_options['bilevel_iters_norefined'] = 10  # 10
config.estimation_options['bilevel_iters_refined'] = 10  # 5

# TODO: implement fastest shortest path algorithm so the coverage of ods can be let to be higher
# Uncongested mode
# config.set_uncongested_mode(True)

# Under this mode, the true path is used as the path set to learn the logit parameters
# config.set_known_pathset_mode(True)

# Out of sample prediction mode
# config.set_outofsample_prediction_mode(theta = {'tt': -1.9891, 'tt_reliability': 1.1245, 'low_inc': -0.3415}
#                                        , outofsample_prediction = True, mean_count = 2218.4848)


#TODO Regularization

# # If grid search found a theta for travel time with inconsistent sign, then the optimization will be consistent with that

# config.estimation_options['theta_search'] = 'random' # Do not use boolean, options are 'grid','None', 'random'
# config.estimation_options['q_random_search'] = True # Include od demand matrix factor variation for random search
# config.estimation_options['n_draws_random_search'] = 20 # To avoid a wrong scaling factor many random draws needs to be performed
config.estimation_options['scaling_Q'] = True #True

# # Initial theta for optimization
# config.theta_0['tt'] = 0.5

# - Number of attributes that will be set to 0, which moderate sparsity: with 20 at least, we observe benefits of regularize
config.sim_options['n_R'] = 0 # 2 #5 #10 #20 #50

#Labels of sparse attributes
config.sim_options['R_labels'] = ['k' + str(i) for i in np.arange(0, config.sim_options['n_R'])]

# Key internal objects for analysis and visualization
artist = tai.visualization.Artist(folder_plots = config.plots_options['folder_plots'], dim_subplots=config.plots_options['dim_subplots'])


# =============================================================================
# 1c) LOG-FILE
# =============================================================================

# Record starting date and time of the simulation
# https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python

config.set_log_file(networkname = config.sim_options['current_network'].lower())


# =============================================================================
# 2) NETWORKS FACTORY
# =============================================================================
# Dictionary with training and testing networks
N = {}
N['train'] = {}

# Label of networks selected (to avoid commeting out everytime)
current_network = config.sim_options['current_network']

# =============================================================================
# a) CREATION OF REAL LARGE SCALE NETWORKS
# =============================================================================

# FROM PRIVATE DATA

# - Fresno
if config.sim_options['current_network'] == 'Fresno':

    N_Fresno = tai.modeller.fresno_network_factory(
        folder = config.paths['Fresno_network']
        , options ={**config.sim_options,**config.estimation_options}
        , label ='Fresno'
    )
    N['train'].update({N_Fresno.key: N_Fresno})

# -Sacramento
if config.sim_options['current_network'] == 'Sacramento':
    N_Sacramento = tai.modeller \
        .sacramento_network_factory(
        folder = config.paths['Sacramento_network']
        , options = {**config.sim_options,**config.estimation_options}
        , label ='Sacramento'
    )
    # N['train'].update({N_Sacramento.key: N_Sacramento})
    #
    # tai.geographer \
    #     .write_line_segment_shp_from_links \
    #     (
    #         links = N_Sacramento.links
    #         , folderpath = config.paths['gis_data']
    #         , subfoldername ='Sacramento'
    #     )
    #

# - Colombus, Ohio
if config.sim_options['current_network'] == 'Colombus':

    N_Colombus = tai.modeller.colombus_network_factory(
        folder = config.paths['Colombus_network']
        , options = {**config.sim_options,**config.estimation_options}
        , label ='Colombus'
    )
    N['train'].update({N_Colombus.key: N_Colombus})
# N['train']['Colombus, Ohio'].paths_od[(1000471,2000473)]

# iv) Los Angeles

# =============================================================================
# b) SETUP INCIDENCE MATRICES AND LINK ATTRIBUTES IN NETWORKS
# =============================================================================

# - Set values of attributes in matrix Z, including those sparse (n_R) or not. Q matrix is not generated again.

# Fresno
if config.sim_options['current_network'] == 'Fresno':
    N['train']['Fresno']= tai.modeller.setup_fresno_network(

        Nt=N['train']['Fresno']
        , setup_options= {**config.sim_options, **config.estimation_options}
        , folder=config.paths['Fresno_Sac_networks']
        , subfolder=N['train']['Fresno'].key

        # (i) First time to write network matrices consistently

        # , reading=dict(config.sim_options['reading'],
        #                **{'paths': False
        #                    , 'C': False, 'M': False, 'D': False, 'Q': True
        #                    , 'sparse_C': True, 'sparse_D': True, 'sparse_M': True, 'sparse_Q': False
        #                   })
        #
        # , generation=dict(config.sim_options['generation'],
        #                   **{'paths': True, 'bpr': False, 'Z': True
        #                       , 'C': True, 'D': True, 'M': True, 'Q': False})
        #
        # , writing=dict(config.sim_options['writing'],
        #                **{'paths': True
        #                    , 'C': True, 'D': True, 'M': True, 'Q': True
        #                    , 'sparse_C': True, 'sparse_D': True, 'sparse_M': True, 'sparse_Q': True
        #                   })

        # ii) After all network elements have been properly written for first time

        , reading=dict(config.sim_options['reading'],
                       **{'paths': True
                           , 'C': True, 'M': True, 'D': True, 'Q': True
                           , 'sparse_C': True, 'sparse_D': True, 'sparse_M': True, 'sparse_Q': False
                          })
        , generation=dict(config.sim_options['generation'],
                          **{'paths': False, 'bpr': False, 'Z': True
                              , 'C': False,'D': False, 'M':False, 'Q': False})

    )

# Sacramento
if config.sim_options['current_network'] == 'Sacramento':

    N['train']['Sacramento'] \
        = tai.modeller \
        .setup_sacramento_network(
        Nt=N['train']['Sacramento']
        , setup_options= {**config.sim_options, **config.estimation_options}
        , reading=dict(config.sim_options['reading'],
                       **{'paths': True
                           , 'C': True, 'M': True, 'D': True, 'Q': True
                           , 'sparse_C': True, 'sparse_D': True, 'sparse_M': True, 'sparse_Q': False
                          })
        , generation=dict(config.sim_options['generation'],
                          **{'paths': False, 'bpr': False, 'Z': True
                              , 'C': False,'D': False, 'M':False, 'Q': False})

        # , writing=dict(config.sim_options['writing'],
        #                **{'paths': True
        #                    , 'C': True, 'D': True, 'M': True, 'Q': True
        #                    , 'sparse_C': True, 'sparse_D': True, 'sparse_M': True, 'sparse_Q': True
        #                   })
        , folder= config.paths['Fresno_Sac_networks']
        , subfolder=N['train']['Sacramento'].key
    )


# + Colombus, Ohio
if config.sim_options['current_network'] == 'Colombus':

    # TODO: there is memory overflow to handle the network matrices and they cannot be stored,except for Q.

    N['train']['Colombus'] = tai.modeller\
        .setup_colombus_network\
        (
            Nt=N['train']['Colombus']
            , setup_options= {**config.sim_options, **config.estimation_options}
            , reading=dict(config.sim_options['reading'],
                           **{'paths': True
                               , 'C': False, 'M': False, 'D': False, 'Q': True
                               , 'sparse_C': True, 'sparse_D': True, 'sparse_M': True, 'sparse_Q': False
                              })
            , generation=dict(config.sim_options['generation'],
                              **{'paths': False, 'bpr': False, 'Z': True
                                  , 'C': True,'D': False, 'M':True, 'Q': False})
            , writing=dict(config.sim_options['writing'],
                           **{'paths': True
                               , 'C': False, 'D': False, 'M': True, 'Q': True
                               , 'sparse_C': True, 'sparse_D': True
                               , 'sparse_M': True, 'sparse_Q': False
                              })
            , folder= config.paths['Colombus_network']
        )

# =============================================================================
# c) BEHAVIORAL PARAMETERS AND UTILITY FUNCTIONS
# =============================================================================

# Initialize theta vector with travellers preferences for every attribute
# theta_true_Z = dict(zip(Z_lb, np.zeros(len(Z_lb))))
for k in config.sim_options['R_labels']:
    if k not in config.theta_true['Z'].keys():
        config.theta_true['Z'][k] = 0 #-0.001

        # Update theta_0 vector
        config.theta_0[k] = 0

theta_true = {}
for i in N['train'].keys():
    theta_true[i] = {**config.theta_true['Y'], **config.theta_true['Z']}

# TODO: Normalize preference vector to 1 for analysis purposes:
# for i in N['train'].keys():
#     theta_true[i] =     # {**theta_true_Y, **theta_true_Z}

# theta_Z = {k: v for k,v in theta.items() if k != 'tt'}

# Labels in sparse vector are added in list of labels of exogenous attributes
config.estimation_options['k_Z'] = [*config.estimation_options['k_Z'], *config.sim_options['R_labels']]

# # Store path utilities associated to exogenous attributes
# for i in N['train'].keys():
#     N['train'][i].set_V_Z(paths = N['train'][i].paths, theta = theta_true)

# Fixed effects parameters (dependent on the network)

if observed_links_fixed_effects is not None and config.sim_options['fixed_effects']['coverage']>0:
    for i in N['train'].keys():
        Z_lb = list(N['train'][i].Z_dict.keys())  # Extract column names in the network
        for k in N['train'][i].k_fixed_effects:
            theta_true[i][k] = theta_true_fixed_effects  # -float(np.random.uniform(1,2,1))
            config.theta_0[k] = theta_0_fixed_effects
            config.estimation_options['k_Z'] = [*config.estimation_options['k_Z'], k]

    if len(N['train'][current_network].k_fixed_effects) > 0:
        print('\nFixed effects created:', N['train'][current_network].k_fixed_effects)


# =============================================================================
# 1.8) Precessing of GIS information and creation of additional link level attributes
# =============================================================================

if config.sim_options['current_network'] == 'Fresno' and config.sim_options['simulated_counts'] is False:

    print('\nMatching geospatial datasets using links of type "LWRLK" ')

    link_types_pd= pd.DataFrame({'link_types': [link.link_type for link in N['train']['Fresno'].links]})

    with pd.option_context('display.float_format', "{:.1%}".format):
        print(pd.concat([link_types_pd.value_counts(normalize=True),link_types_pd.value_counts()]
                        , axis=1,keys=('perc','count')).to_string())

    #TODO: enable warm start for some operations as some ofthe processing with spark or geopandas can be done offline
    # (e.g census data and inrix line to line matching is the same regardless the period of analysis)

    # i) Network data

    # Rescaling coordinates to ease the matching of the x,y coordinates to real coordinates in Qqis
    tai.geographer.adjust_fresno_nodes_coordinates(nodes = N['train'][current_network].nodes, rescale_factor = 1)

    # Set link orientation according to the real coordinates
    tai.geographer.set_cardinal_direction_links(links=N['train'][current_network].get_regular_links())

    # a) Write line and points shapefiles for further processing (need to be done only once)

    tai.geographer \
        .write_node_points_shp(nodes = N['train'][current_network].nodes
                               , folderpath = config.paths['output_folder'] + 'gis/Fresno/network/nodes'
                               , networkname ='Fresno'
                               , config = config
                               )

    tai.geographer \
        .write_line_segments_shp(links = N['train'][current_network].links
                                 , folderpath = config.paths['output_folder'] + 'gis/Fresno/network/links'
                                 , networkname ='Fresno'
                                 , config = config
                                 )

    # b) Read shapefile of layer edited in Qgis where final polish was made (additional spatial adjustments)

    # network_filepath =  "/Users/pablo/google-drive/university/cmu/2-research/datasets/private/od-fresno-sac/SR41/shapefile/fresno-gis-nad83.shp"

    # network_filepath = config.paths['input_folder'] + "private/Fresno/network/qgis/raw/fresno-qgis-nad83-adj.shp"

    network_filepath = config.paths['input_folder'] + "private/Fresno/network/qgis/adjusted/Fresno_links_adj.shp"

    Fresno_network_gdf = tai.geographer.read_qgis_shp_fresno(filepath=network_filepath, config = config)

    # TODO: update node/link coordinates consisently


    # ii) PEMS stations

    manual_matching = True

    if manual_matching is False:

        # Read data for PeMS stations for fresno and return a geodataframe. Then, match pems stations with network links
        path_pems_stations = 'input/public/pems/stations/raw/D06/' + 'd06_text_meta_2020_09_11.txt'

        # Original
        pems_stations_gdf = tai.geographer.read_pems_stations_fresno(filepath=path_pems_stations,
                                                                     adjusted_gis_stations=True
                                                                     , config = config
                                                                     )

        tai.geographer.match_network_and_stations_fresno(stations_gdf=pems_stations_gdf
                                               , network_gdf=Fresno_network_gdf
                                               , links=N['train']['Fresno'].get_regular_links()
                                               , folderpath = config.paths['output_folder'] + 'gis/Fresno/pems-stations'
                                               , adjusted_gis_stations=True
                                               , config = config
                                               )

    else:

        # Assignemnt of station ids according to manual match made in qgis and recorded in the columns pems_id1, pems_id2, pems_id3
        # which are three candidate stations.

        # Path of shapefile with adjustment made in qgis
        path_pems_stations = config.paths['input_folder'] + '/public/pems/stations/gis/adjusted/fresno_stations_adj.shp'

        tai.geographer.manual_match_network_and_stations_fresno(network_gdf=Fresno_network_gdf
                                               , links=N['train']['Fresno'].get_regular_links()
                                               )


    # iii) INRIX data

    # Read and match inrix data

    if config.gis_options['data_processing']['inrix_segments']:

        path_inrix_shp = config.paths['input_folder'] + 'private/Fresno/inrix/shapefiles/USA_CA_RestOfState_shapefile/USA_CA_RestOfState.shp'

        inrix_gdf = tai.geographer.read_inrix_shp(filepath=path_inrix_shp, county='Fresno')

        # Do this only once
        # tai.geographer.export_inrix_shp(inrix_gdf, folderpath = config.paths['folder_gis_data']+'Fresno/inrix', config = config)

        # inrix_gdf.plot(figsize=(5, 5), edgecolor="purple", facecolor="None")
        # plt.show()

        links_buffer_inrix_gdf = tai.geographer.match_network_links_and_inrix_segments_fresno(inrix_gdf= inrix_gdf
                                                          , network_gdf=Fresno_network_gdf
                                                          , links=N['train']['Fresno'].get_regular_links()
                                                           , buffer_size= config.gis_options['buffer_size']['inrix']
                                                           , centroids = True
                                                           , config = config
                                                          )

        # Export buffer created to merge data from inrix segments with network links
        tai.geographer.export_buffer_shp(gdf = links_buffer_inrix_gdf
                                         , folderpath = config.paths['output_folder'] + 'gis/Fresno/inrix/'
                                         , filename ='links_buffer_inrix'
                                         , config = config
                                         )

        if config.gis_options['data_processing']['inrix_data']:


            # Paths were the original data is stored
            path_inrix_data_part1 = config.paths[
                                        'input_folder'] + 'private/Fresno/inrix/speed/Fresno_CA_2019-10-01_to_2019-11-01_15_min_part_1/data.csv'

            path_inrix_data_part2 = config.paths[
                                        'input_folder'] + 'private/Fresno/inrix/speed/Fresno_CA_2019-10-01_to_2019-11-01_15_min_part_2/data.csv'

            filepaths = [path_inrix_data_part1, path_inrix_data_part2]

            if config.extra_options['write_inrix_daily_data']:

                data_analyst.write_partition_inrix_data(filepaths
                                                        , output_folderpath=config.paths['input_folder']
                                                                            + '/private/Fresno/inrix/speed/by-day/')

            if  config.extra_options['read_inrix_daily_data']:
                filepaths = config.paths['input_folder'] + '/private/Fresno/inrix/speed/by-day/'+ config.estimation_options['selected_date'] + '.csv'
                inrix_data_df = data_analyst.generate_inrix_data_by_segment(filepaths=filepaths
                                                                            , selected_period=config.estimation_options['selected_period_inrix']
                                                                            )

            else:
                # Generate a pandas dataframe with the average and standard deviaetion of the speed among INRIX link segments
                inrix_data_df = data_analyst.generate_inrix_data_by_segment(filepaths=filepaths
                                                                            , selected_period=config.estimation_options['selected_period_inrix']
                                                                            )

            # Merge speed data based on the inrix_id among links
            data_analyst.merge_inrix_data(links = N['train']['Fresno'].get_regular_links(), speed_df = inrix_data_df
                                          , options ={**config.sim_options,**config.estimation_options}
                                          , config=config
                                          )

        # inrix_speed_sdf.head()


    # iv) Census data

    # Read census data and match it with network links

    # Write shapefile with relevant block data from TIGER files (need to be done only once)
    # tai.geographer \
    #     .write_census_blocks_data_fresno(countyname= 'Fresno', config = config
    #                                      , filepath= '/Users/pablo/google-drive/data-science/github/transportAI/input/public/census/ACS_2018_5YR_BG_06_CALIFORNIA.gdb')

    if config.gis_options['data_processing']['census']:

        census_tract_path = config.paths['output_folder'] + 'gis/Fresno/census/Fresno_census_shp/Fresno_census_shp.shp'
        census_tracts_data_gdf = tai.geographer.read_census_tracts_shp_fresno(filepath = census_tract_path)

        if config.gis_options['inrix_matching']['census'] and config.gis_options['inrix_matching']:
            tai.geographer.match_network_links_and_census_tracts_fresno(census_tracts_gdf=census_tracts_data_gdf
                                                              , network_gdf=inrix_gdf
                                                              , links=N['train']['Fresno'].get_regular_links()
                                                              , attrs=['median_inc', 'median_age']
                                                              , inrix_matching = True
                                                              , config=config
                                                              )

        else:
           tai.geographer.match_network_links_and_census_tracts_fresno(census_tracts_gdf=census_tracts_data_gdf
                                                              , network_gdf=Fresno_network_gdf
                                                              , links=N['train']['Fresno'].get_regular_links()
                                                              , attrs = ['median_inc', 'median_age']
                                                              , inrix_matching= False
                                                              , config=config
                                                              )



    if config.gis_options['data_processing']['incidents']:

        # v) Traffic incident data

        # Read and match traffic incident data
        incidents_path = config.paths['input_folder'] + "public/traffic-incidents/US_Accidents_Dec20.csv"

        traffic_incidents_Fresno_df = data_analyst \
            .read_traffic_incidents(filepath=incidents_path
                                    , selected_period= config.estimation_options['selected_period_incidents']
                                    )

        # # Export shapefile with traffic incidents locations
        # tai.geographer.export_fresno_incidents_shp(incidents_df = traffic_incidents_Fresno, folderpath = config.paths['folder_gis_data']+'Fresno/incidents')

        if config.gis_options['inrix_matching']['incidents'] and config.gis_options['inrix_matching']:
            links_buffer_incidents_gdf \
                = tai.geographer.match_network_links_and_fresno_incidents(incidents_df=traffic_incidents_Fresno_df
                                                                          , network_gdf=inrix_gdf,
                                                                          links=N['train']['Fresno'].get_regular_links()
                                                                          ,buffer_size=config.gis_options['buffer_size'][
                                                                              'incidents']
                                                                          , inrix_matching=True
                                                                          , config=config
                                                                          )

        else:

            # Count the number of incidents within a buffer of each link to have an indicator about safety

            # It allows to select month as well, so the incident information matches date of traffic counts
            links_buffer_incidents_gdf \
                = tai.geographer.match_network_links_and_fresno_incidents(incidents_df = traffic_incidents_Fresno_df
                                                                          , network_gdf = Fresno_network_gdf
                                                                          , links=N['train']['Fresno'].get_regular_links()
                                                                          , buffer_size= config.gis_options['buffer_size']['incidents']
                                                                          , inrix_matching=False
                                                                          , config=config
                                                                          )

            # Export buffer created to merge data from incidents with network links
            tai.geographer.export_buffer_shp(gdf = links_buffer_incidents_gdf
                                             , folderpath = config.paths['output_folder'] + 'gis/Fresno/incidents/'
                                             , filename ='links_buffer_incidents'
                                             , config = config
                                             )

    # vi) Bus stop information (originally with txt file, similar to pems stations)

    if config.gis_options['data_processing']['bus_stops']:

        # Path of bus stop txt file
        input_path_bus_stops_fresno = config.paths['input_folder'] + "public/transit/adjusted/stops.txt"

        # Read txt file and return pandas dataframe
        bus_stops_fresno_df = data_analyst.read_bus_stops_txt(filepath = input_path_bus_stops_fresno)

        # Return geodataframe and export a shapefile if required (that may be read in qgis)
        bus_stops_fresno_gdf = tai.geographer.generate_fresno_bus_stops_gpd(bus_stops_df = bus_stops_fresno_df, config = config)

        # # Export bus stop shapefile
        # tai.geographer.export_bus_stops_shp(bus_stops_gdf = bus_stops_fresno_gdf,
        #                                     folderpath = config.paths['output_folder'] + 'gis/Fresno/bus-stops'
        #                                     ,config = config
        #                                     )

        if config.gis_options['inrix_matching']['bus_stops'] and config.gis_options['inrix_matching']:

            links_buffer_bus_stops_gdf \
                = tai.geographer.match_network_links_and_fresno_bus_stops(bus_stops_gdf=bus_stops_fresno_gdf
                                                                          , network_gdf = inrix_gdf
                                                                          , links=N['train']['Fresno'].get_regular_links()
                                                                          , buffer_size=config.gis_options['buffer_size'][
                                                                              'bus_stops']
                                                                          , inrix_matching=True
                                                                          , config=config
                                                                          )

        else:

            # Match bus stops with network links

            # - Count the number of bus stops within a buffer around each link
            links_buffer_bus_stops_gdf\
                = tai.geographer.match_network_links_and_fresno_bus_stops(bus_stops_gdf = bus_stops_fresno_gdf
                                                                      , network_gdf = Fresno_network_gdf
                                                                          , links=N['train']['Fresno'].get_regular_links()
                                                                          , buffer_size = config.gis_options['buffer_size']['bus_stops']
                                                                          , inrix_matching=False
                                                                          , config = config
                                                                      )

            tai.geographer.export_buffer_shp(gdf=links_buffer_bus_stops_gdf
                                             , folderpath= config.paths['output_folder'] + 'gis/Fresno/bus-stops/'
                                             , filename='links_buffer_bus_stops'
                                             , config = config
                                             )

    # vii) Streets intersections

    if config.gis_options['data_processing']['streets_intersections']:

        input_path_intersections_fresno =  config.paths['input_folder'] + "public/traffic/streets-intersections/adjusted/intrsect.shp"

        # - Return a geodataframe to perform gis matching later
        streets_intersections_fresno_gdf = tai.geographer.generate_fresno_streets_intersections_gpd(filepath = input_path_intersections_fresno)

        if config.gis_options['inrix_matching']['streets_intersections'] and config.gis_options['inrix_matching']:

            links_buffer_streets_intersections_gdf \
                = tai.geographer.match_network_links_and_fresno_streets_intersections(
                streets_intersections_gdf=streets_intersections_fresno_gdf
                , network_gdf=inrix_gdf
                , links=N['train']['Fresno'].get_regular_links()
                , buffer_size=config.gis_options['buffer_size']['streets_intersections']
                , inrix_matching=True
                , config = config
                )


        else:

            # - Count the number of intersections within a buffer around each link
            links_buffer_streets_intersections_gdf \
                = tai.geographer.match_network_links_and_fresno_streets_intersections(streets_intersections_gdf = streets_intersections_fresno_gdf
                                                                                      , network_gdf = Fresno_network_gdf
                                                                                      , links=N['train']['Fresno'].get_regular_links()
                                                                                      , buffer_size= config.gis_options['buffer_size']['streets_intersections']
                                                                                      , inrix_matching=False
                                                                                      , config = config
                                                                                      )

            tai.geographer.export_buffer_shp(gdf= links_buffer_streets_intersections_gdf
                                             , folderpath=config.paths['output_folder'] + 'gis/Fresno/streets-intersections'
                                             , filename='links_buffer_streets_intersections'
                                             , config = config
                                             )

    # Make sure that link of connector type have attributes equal to 0
    attr_links = list(N['train']['Fresno'].get_regular_links()[0].Z_dict.keys())

    print('Attributes values of link with type diffeent than "LWRLK" are set to 0')

    for link in N['train']['Fresno'].get_non_regular_links():
        for key in attr_links:
            link.Z_dict[key] = 0
            # print(link.link_type)

    # Update link atribute matrix in network
    N['train']['Fresno'].set_Z_attributes_dict_network(links_dict = N['train']['Fresno'].links_dict)


# =============================================================================
# 1.7) SUMMARY OF NETWORKS CHARACTERISTICS
# =============================================================================
networks_table = {'nodes':[],'links':[], 'paths': [], 'ods': []}

# Network description
for i in N['train'].keys():
    networks_table['ods'].append(len(tai.networks.denseQ(N['train'][i].Q, remove_zeros=True)))
    networks_table['nodes'].append(N['train'][i].A.shape[0])
    networks_table['links'].append(len(N['train'][i].links))
    networks_table['paths'].append(len(N['train'][i].paths))

networks_df = pd.DataFrame()
networks_df['N'] = np.array(list(N['train'].keys()))

for var in ['nodes','links', 'ods','paths']:
    networks_df[var] = networks_table[var]

# Print Latex Table
print('Networks summary')
print(networks_df)

# # Print Latex Table
# print(networks_df.to_latex(index=False))





# =============================================================================
# 1.8) NETWORK EQUILIBRIUM
# =============================================================================
results_sue = {}
valid_network = None

# Exceptions
exceptions = {}

# TODO: Understand why SUE Logit fail sometimes
exceptions['SUE'] = {'train': {}}
exceptions_train = {} #To record the number of exceptions

for i in N['train'].keys():
    exceptions['SUE']['train'][i] = 0

# =============================================================================
# 2) DATA READING / GENERATION
# =============================================================================

# =============================================================================
# 2.1) UNCONGESTED/CONGESTED MODE (MUST BE BEFORE SYNTHETIC COUNTS)
# =============================================================================

if config.sim_options['uncongested_mode'] is True:

    print('Algorithm is performed assuming an uncongested network')

    # TODO: implement good copy and clone methods so the code does not need to be run from scratch everytime the bpr functions are set to zero
    # To account for the uncongested case, the BPR parameters are set to be equal to zero
    for link in N['train'][current_network].links:
        link.bpr.alpha = 0
        link.bpr.beta = 0


    # for link in N['train'][i].links:
    #     print(link.bpr.alpha)
    #     # print(link.bpr.beta)

else:

    print('Algorithm is performed assuming a congested network')

    # Given that it is very expensive to compute path probabitilies and the resuls are good already, it seems fine to perform only one iteration for outer problem
    # iters_est = config.sim_options['iters']  #5





# =============================================================================
# 2.1) SYNTHETIC COUNTS
# =============================================================================

if config.sim_options['simulated_counts'] is True:

    if k_Z_simulation is None:
        k_Z_simulation = config.estimation_options['k_Z']

    # Generate synthetic traffic counts
    xc_simulated, xc_withdraw = tai.estimation.generate_link_counts_equilibrium(
        Nt=N['train'][current_network]  # tai.modeller.clone_network(N['train'][i], label = 'clone')
        , theta = theta_true[current_network]
        , k_Y = k_Y, k_Z = k_Z_simulation
        , eq_params = {'iters': config.sim_options['max_sue_iters'], 'accuracy_eq': config.estimation_options['accuracy_eq'], 'method': 'line_search', 'iters_ls': 20}
        , coverage = config.sim_options['link_coverage']
        , noise_params = config.sim_options['noise_params']
        , n_paths = config.sim_options['n_paths_synthetic_counts']
    )

    xc = xc_simulated

    N['train'][current_network].reset_link_counts()
    N['train'][current_network].store_link_counts(xct=xc_simulated)

    x_bar = np.array(list(xc_simulated.values()))[:, np.newaxis]

    print('Synthetic observed links counts:')

    dict_observed_link_counts = {link_key: np.round(count,1) for link_key, count in xc_simulated.items() if not np.isnan(count)}

    with pd.option_context('display.float_format', '{:0.1f}'.format):
        print(pd.DataFrame(
            {'link_key': dict_observed_link_counts.keys(), 'counts': dict_observed_link_counts.values()}).to_string())

    #Update matrix Q in network

# =============================================================================
# 2.2) REAL DATA
# =============================================================================


# Read data to later
elif config.sim_options['current_network'] == 'Fresno':

    # ii) Read data from PEMS count and perform matching GIS operations to combine station shapefiles

    path_pems_counts = ''

    # if config.estimation_options['selected_period_pems_counts']['year'] == 2020:

    date_pathname = config.estimation_options['selected_date'].replace('-','_')
    # October 1, 2020 is Thursday
    path_pems_counts = config.paths['input_folder'] + 'public/pems/counts/data/' + \
                       'd06_text_station_5min_' + date_pathname + '.txt.gz'

    # if config.estimation_options['selected_period_pems_counts']['year'] == 2019:
    #
    #     # October 1, 2019 is Tuesday
    #     path_pems_counts = config.paths['input_folder'] + 'public/pems/counts/raw/d06/' + \
    #                        'd06_text_station_5min_2019_10_01.txt.gz'

    # Read and match count data from a given period

    # Duration is set at 2 because the simulation time for the OD matrix was set at that value
    count_interval_df \
        = data_analyst.read_pems_counts_by_period(filepath=path_pems_counts
                                                  , selected_period = config.estimation_options['selected_period_pems_counts'])

    # Generate a masked vector that fill out count values with no observations with nan

    #TODO: normalized flow total by the number of lanes with data modyfing the read_pems_count method. The flow factor should
    # not required for the algorithm to work. Need to confirm if links in the Fresno network correspond to an aggregate of lanes
    xc = tai.estimation.generate_fresno_pems_counts(links = N['train'][current_network].links
                                                    , data = count_interval_df
                                                    # , flow_attribute='flow_total'
                                                    , flow_attribute='flow_total_lane'
                                                    , flow_factor = 1  # 0.1
                                                    )
                                                      # , flow_attribute = 'flow_total_lane_1')

    N['train'][current_network].reset_link_counts()
    N['train'][current_network].store_link_counts(xct=xc)

    # print('\nSummary of links with observed links counts:\n')
    # summary_table_links_df = tai.descriptive_statistics.summary_table_links(links = N['train'][current_network].get_observed_links())
    #
    # with pd.option_context('display.float_format', '{:0.1f}'.format):
    #     print(summary_table_links_df.to_string())

    # xc = xc_fresno

    # If the algorithm is working well, the cost parameter should be found to be equal to 0 as it the corresponding attribute
    # is just a random perturbation
    # config.estimation_options['k_Z'] = ['k0']
    # config.estimation_options['k_Z'] = []
    # config.estimation_options['k_Z'] = ['k0', 'median_inc','median_age']
    # config.estimation_options['k_Z'] = ['median_inc']+config.sim_options['R_labels']
    # config.estimation_options['k_Z'] = config.sim_options['R_labels']



    # TODO: if there is no initial value for a parameter, the default should be to set that key of theta_0 equal to 0
    # config.theta_0['median_inc'] = 0
    # config.theta_0['median_age'] = 0

    # As more attributes, there are more expensive is to copmute the gradient for ngd
    # The census variables may need to be normalized by the number of links, as otherwise, there may be a bias toward a negative sign (i.e. correlated with number of links in the path which negatively correlated with utility)

    # config.estimation_options['k_Z'] = ['median_inc','median_age', 'k0']
    # config.estimation_options['k_Z'] = ['lane', 'k0']
    # config.estimation_options['k_Z'] = ['c', 'length']
    # config.estimation_options['k_Z'] = ['c', 'lane']

    # TODO: the algorithm fails to estimate if there are links with missing information.
    #  A kernel density for imputation may be appropiate using as input the speed/travel time variability data from Inrix
    # config.estimation_options['k_Z'] = ['c', 'speed_avg', 'speed_sd']

    # theta_0 = dict.fromkeys(k_Y+config.estimation_options['k_Z'],-1)

    # Add speed data (TODO: the problem here is that speed data is only available for the counter stations. Thus, a fix to is to match inrix speed data into the network links with a line to line matching operation ). However, there will be still many links with missing data.


    # for link in links:
    #     # if link.pems_station_id == data['station_id'].filter(link.pems_station_id):
    #
    #     if len(speed_interval_pems_df[speed_interval_pems_df['station_id'] == link.pems_station_id]) > 0:
    #         pd_row = speed_interval_pems_df[speed_interval_pems_df['station_id'] == link.pems_station_id]
    #
    #         link.Z_dict['speed_avg'] = pd_row['speed_avg']
    #         link.Z_dict['speed_sd'] = pd_row['speed_sd']

    # # Update attribute in the network object
    # N['train'][current_network].copy_Z_attributes_dict_links(links_dict=N['train'][current_network].links_dict)
    # N['train'][current_network].set_Z_attributes_dict_network(links_dict = N['train'][current_network].links_dict)


# =============================================================================
# 3b) TRAINING/VALIDATION SPLIT
# =============================================================================

if config.sim_options['prop_validation_sample'] > 0:

    # Get a training and testing sample
    xc, xc_validation = tai.estimation.generate_training_validation_samples(
        xct = xc, prop_validation = config.sim_options['prop_validation_sample']
    )

else:
    xc = xc
    xc_validation = xc


# =============================================================================
# 3) DATA REFINING
# =============================================================================

# TODO: descriptive statistics about link flows

# Report link count information

total_counts_observations = np.count_nonzero(~np.isnan(np.array(list(xc.values()))))

if config.sim_options['prop_validation_sample'] > 0:
    total_counts_observations +=  np.count_nonzero(~np.isnan(np.array(list(xc_validation.values()))))

total_links = np.array(list(xc.values())).shape[0]

print('\nTotal link counts observations: ' + str(total_counts_observations))
print('Link coverage: ' + "{:.1%}". format(round(total_counts_observations/total_links,4)))

if config.sim_options['simulated_counts'] is True:
 print('Std set to simulate link observations: ' + str(config.sim_options['noise_params']['sd_x']))

# =============================================================================
# 3a) LINK COUNTS AND TRAVERSING PATHS
# =============================================================================

#Initial coverage
x_bar = np.array(list(xc.values()))[:, np.newaxis]

# If no path are traversing some link observations, they are set to nan values
xc = tai.estimation.masked_link_counts_after_path_coverage(N['train'][current_network], xct = xc)

x_bar_remasked = np.array(list(xc.values()))[:, np.newaxis]

true_coverage = np.count_nonzero(~np.isnan(x_bar_remasked))/x_bar.shape[0]
# After accounting for path coverage (not all links may be traversed)

total_true_counts_observations = np.count_nonzero(~np.isnan(np.array(list(xc.values()))))

print('\nAdjusted link observations: ' + str(total_true_counts_observations))
print('Adjusted link coverage:', "{:.1%}". format(true_coverage))

# =============================================================================
# 3c) DATA CURATION
# =============================================================================

# i) Capacity adjustment

# If the link counts are higher than the capacity of the lane, we applied an integer deflating factor until
# the capacity of the link is not surpasses. This tries to correct for the fact that the link counts may be recording
# more than 1 lane.

 #The numebr of lane information in Sean file to match the corresponding lane from PEMS. This should reduce the adjustments by capacity

# xc = tai.estimation.adjusted_counts_by_link_capacity(Nt = N['train'][current_network], xct = xc)

if config.sim_options['prop_validation_sample'] > 0:
    xc_validation = tai.estimation.adjusted_counts_by_link_capacity(Nt = N['train'][current_network], xct = xc_validation)

else:
    xc_validation = xc

# ii) Outliers

# - Remove traffic stations where a huge errors is observed and which indeed are considered outliers

removed_links_keys = []

outliers_links_keys = []

# outliers_links_keys = [ (136, 385,'0'), (179, 183,'0'), (620, 270,'0'), (203, 415,'0') , (217, 528,'0'), (236, 260,'0'), (239, 242,'0'), (243, 261,'0'), (276, 277,'0'), (277, 278,'0'), (282, 281,'0'), (283, 197,'0'), (284, 285,'0'),(285, 286,'0'), (385, 203,'0'), (587, 583,'0'), (676, 174,'0'),  (677, 149,'0')]

# Flow is too low maybe due to problems in od matrix
#, (1590, 1765, '0'), (1039, 125, '0')

for key in outliers_links_keys:
    removed_links_keys.append(key)

od_connectors_keys = [(1542, 1717, '0'), (92, 1610, '0'),(114, 1571, '0'),(1400, 1657, '0'), (1244, 1632, '0'), (1459, 996, '0')
                         , (1444, 781, '0'), (42, 21, '0'), (1610, 1785, '0'), (1590, 1765, '0'), (1580, 1755, '0'), (1571, 1746, '0')]

for key in od_connectors_keys:
    removed_links_keys.append(key)


# List link with unlimited capacity as those are the ones with the higher errors, but this is probably because I did not perform adjustments

removed_counter = 0
for link_key, count in xc.items():
    if link_key in removed_links_keys:

        if not np.isnan(xc[link_key]) or not np.isnan(xc_validation[link_key]):
            removed_counter+= 1

            link.observed_count = np.nan
            xc[link_key] = np.nan
            xc_validation[link_key] = np.nan


print("\n" + str(removed_counter) + " traffic counts observations were removed because belonging to OD connectors or assumed to be outliers")

# TODO: print rows with outliers
new_total_counts_observations = np.count_nonzero(~np.isnan(np.array(list(xc.values()))))

print('New total of link observations: ' + str(new_total_counts_observations))

# Store synthetic counts into link objects of network object
N['train'][current_network].reset_link_counts()
N['train'][current_network].store_link_counts(xct=xc_validation)
N['train'][current_network].store_link_counts(xct=xc)

# print('\nNew summary of links with observed links counts:\n')
#
# new_summary_table_links_df = tai.descriptive_statistics.summary_table_links(links=N['train'][current_network].get_observed_links())
#
# with pd.option_context('display.float_format', '{:0.1f}'.format):
#     print(new_summary_table_links_df.to_string())

# - Generate fixed effect for links with large error

# Note: The assumption is that those links are



# =============================================================================
# 3b) FIXED EFFECTS
# =============================================================================

# Selection of fixed effects among the group of observed counts only

#TODO: Simulation is not accounting for observed link effects because synthetic counts were created before
if observed_links_fixed_effects == 'random' and config.sim_options['fixed_effects']['coverage'] > 0:

    # Store synthetic counts into link objects of network object
    N['train'][current_network].reset_link_counts()
    N['train'][current_network].store_link_counts(xct=xc_validation)
    N['train'][current_network].store_link_counts(xct=xc)


    N['train'][current_network].set_fixed_effects_attributes(config.sim_options['fixed_effects']
                                                             , observed_links = observed_links_fixed_effects)

if observed_links_fixed_effects == 'custom':

    # selected_links_keys = [(695, 688, '0'), (1619, 631, '0'), (1192, 355, '0'), (217, 528, '0')]

    selected_links_keys = [(695, 688, '0'), (1192, 355, '0'), (680, 696, '0')]

    N['train'][current_network].set_fixed_effects_attributes(config.sim_options['fixed_effects'],
                                                             observed_links=observed_links_fixed_effects
                                                             , links_keys=selected_links_keys)

if observed_links_fixed_effects is not None:

    for k in N['train'][current_network].k_fixed_effects:
        theta_true[i][k] = theta_true_fixed_effects # -float(np.random.uniform(1,2,1))
        config.theta_0[k] = theta_0_fixed_effects
        k_Z = [*config.estimation_options['k_Z'], k]

    # Update dictionary with attributes values at the network level
    N['train'][current_network].set_Z_attributes_dict_network(links_dict=N['train'][current_network].links_dict)

    if len(N['train'][current_network].k_fixed_effects) > 0:
        print('\nFixed effects created within observed links only:', N['train'][current_network].k_fixed_effects)


# =============================================================================
# 3c) CREATION OF NEW FEATURES
# =============================================================================

existing_Z_attrs = N['train'][current_network].get_regular_links()[0].Z_dict


# i) High and low income dummies

# - Percentile used to segmentation income level from CENSUS blocks income data (high income are those links with income higher than pct)
config.estimation_options['pct_income'] = 30

# - Get percentile income distribution first
if 'median_inc' in existing_Z_attrs:
    links_income_list = [link.Z_dict['median_inc'] for link in N['train'][current_network].get_regular_links()]

    for link in N['train'][current_network].get_regular_links():

        # Create dummy variable for high income areas
        link_pct_income = np.percentile(np.array(links_income_list),config.estimation_options['pct_income'])

        if 'median_inc' in link.Z_dict:
            link.Z_dict['low_inc'] = 1
            link.Z_dict['high_inc'] = 0

            if link.Z_dict['median_inc'] > link_pct_income:
                link.Z_dict['low_inc'] = 0
                link.Z_dict['high_inc'] = 1

# (ii) No incidents
if 'incidents' in existing_Z_attrs:
    for link in N['train'][current_network].get_regular_links():
        link.Z_dict['no_incidents'] = 1

        if link.Z_dict['incidents'] > 0:
            link.Z_dict['no_incidents'] = 0

# (iii) No bus stops

if 'bus_stops' in existing_Z_attrs:
    for link in N['train'][current_network].get_regular_links():
        link.Z_dict['no_bus_stops'] = 1

        if link.Z_dict['bus_stops'] > 0:
            link.Z_dict['no_bus_stops'] = 0

# (iv) No street intersections

if 'intersections' in existing_Z_attrs:

    for link in N['train'][current_network].get_regular_links():
        link.Z_dict['no_intersections'] = 1

        if link.Z_dict['intersections'] > 0:
            link.Z_dict['no_intersections'] = 0

# (v) Travel time variability

# - Adjusted standard deviation of travel time

if 'tt_cv' in existing_Z_attrs:

    for link in N['train'][current_network].get_regular_links():
        link.Z_dict['tt_sd_adj'] = link.Z_dict['ff_traveltime']*link.Z_dict['tt_cv']


# - Measure of reliability as PEMS which is relationship between true and free flow travel times
if 'speed_avg' in existing_Z_attrs and 'speed_ref_avg' in existing_Z_attrs:
    
    for link in N['train'][current_network].get_regular_links():
        # link.Z_dict['tt_reliability'] = min(1,link.Z_dict['speed_avg']/link.Z_dict['speed_ref_avg'])
        link.Z_dict['tt_reliability'] = link.Z_dict['speed_avg'] / link.Z_dict['speed_ref_avg']


# Make sure that link of connector type have attributes equal to 0
attr_links = list(N['train']['Fresno'].get_regular_links()[0].Z_dict.keys())

print('Attributes values of link with type diffeent than "LWRLK" are set to 0')

for link in N['train']['Fresno'].get_non_regular_links():
    for key in attr_links:
        link.Z_dict[key] = 0
        # print(link.link_type)

# Update link atribute matrix in network
N['train']['Fresno'].set_Z_attributes_dict_network(links_dict=N['train']['Fresno'].links_dict)


# =============================================================================
# 3d) DATA IMPUTATION
# =============================================================================

# TODO: Imputation method for regular link with missing attributes: assuming that the missing value is always zero is not the best idea


# =============================================================================
# 3d) DESCRIPTIVE STATISTICS
# =============================================================================

# . Link characteristics
summary_table_links_df = tai.descriptive_statistics.summary_table_links(links = N['train'][current_network].get_observed_links()
                                                                        , Z_attrs = ['speed_avg', 'tt_sd', 'tt_cv','tt_reliability', 'incidents', 'median_inc', 'bus_stops', 'intersections']
                                                                        , Z_labels = ['speed_avg [mi/hr]', 'tt_sd', 'tt_cv','tt_reliability', 'incidents', 'income [1K USD]','stops', 'ints']
                                                                        )

with pd.option_context('display.float_format', '{:0.3f}'.format):
    print(summary_table_links_df.to_string())

# Write log file
tai.writer.write_csv_to_log_folder(df = summary_table_links_df, filename = 'summary_table_links_df'
                          , log_file = config.log_file)


selected_links_ids_pems_statistics = [link.pems_stations_ids[0] for link in N['train'][current_network].get_observed_links() if len(link.pems_stations_ids) == 1]

selected_links_ids_pems_statistics = list(np.random.choice(selected_links_ids_pems_statistics, 4, replace=False))


distribution_pems_counts_figure = tai.descriptive_statistics.distribution_pems_counts(filepath = path_pems_counts
                                                                                      , selected_period = {'year': config.estimation_options['selected_year'], 'month': config.estimation_options['selected_month'], 'day_month': config.estimation_options['selected_day_month'], 'hour': 6, 'duration': 900}
                                                                                      , selected_links = selected_links_ids_pems_statistics
                                                                                      )

tai.writer.write_figure_to_log_folder(fig = distribution_pems_counts_figure
                                      , filename = 'distribution_pems_counts.pdf', log_file = config.log_file)


# Continous and categorical features
continuous_features = ['counts', 'capacity [veh]', 'speed_ff[mi/hr', 'tt_ff [min]', 'income [1K USD]', 'speed_avg [mi/hr]', 'tt_sd', 'tt_cv', 'tt_reliability', 'incidents']
categorical_features = ['high_inc','stops', 'ints']

existing_continous_features = set(summary_table_links_df.keys()).intersection(set(continuous_features))

summary_table_links_scatter_df = summary_table_links_df[existing_continous_features]

scatter_fig1, scatter_fig2 = tai.descriptive_statistics.scatter_plots_features_vs_counts(links_df = summary_table_links_scatter_df)

tai.writer.write_figure_to_log_folder(fig = scatter_fig1
                                      , filename = 'scatter_plot1.pdf', log_file = config.log_file)


tai.writer.write_figure_to_log_folder(fig = scatter_fig2
                                      , filename = 'scatter_plot2.pdf', log_file = config.log_file)

#TODO: visualize box plot for the non continuous features
# data_analyst.read_pems_counts_by_period(filepath=path_pems_counts
#                                                   , selected_period = config.estimation_options['selected_period_pems_counts'])


# =============================================================================
# 3) BILEVEL OPTIMIZATION IN CONGESTED NETWORK
# =============================================================================

# =============================================================================
# 3.2) HEURISTICS FOR SCALING OF OD MATRIX AND SEARCH OF INITIAL LOGIT ESTIMATE
# =============================================================================

if config.estimation_options['theta_search'] is not None:

    # # Generate synthetic traffic counts
    # xc_simulated = tai.estimation.generate_link_counts_equilibrium(
    #     Nt=N['train'][current_network]  # tai.modeller.clone_network(N['train'][i], label = 'clone')
    #     , theta=theta_true[current_network]
    #     , k_Y=k_Y, k_Z=config.estimation_options['k_Z']
    #     , eq_params={'iters': config.sim_options['max_sue_iters'], 'accuracy_eq': config.estimation_options['accuracy_eq']}
    #     , coverage=config.sim_options['link_coverage']
    #     , noise_params=config.sim_options['noise_params']
    # )

    # Grid and random search are performed under the assumption of an uncongested network to speed up the search

    if config.estimation_options['theta_search'] == 'grid':
        theta_attr_grid, f_vals, grad_f_vals, hessian_f_vals \
            = tai.estimation.grid_search_optimization(Nt=N['train'][current_network]
                                                      , k_Y=k_Y, k_Z=config.estimation_options['k_Z']
                                                      , x_bar=np.array(list(xc.values()))[:, np.newaxis]
                                                      # , theta_attr_grid= np.arange(-10, 10, 0.5)
                                                      # , theta_attr_grid= [1,0,-1,-10,-10000]
                                                      , theta_attr_grid=[1, 0, -1e-3, -1e-1, -5e-1, -1, -2]
                                                      , attr_label='tt'
                                                      , theta=config.theta_0
                                                      , gradients=False, hessians=False
                                                      , inneropt_params={
                'iters': 0*config.estimation_options['max_sue_iters_refined'], 'accuracy_eq': config.estimation_options['accuracy_eq']}
                                                      )

    # print('grid for theta_t', theta_attr_grid)
    # print('losses ', f_vals)
    # print('gradients ', grad_f_vals)

        # Plot
        plot1 = tai.Artist(folder_plots = config.plots_options['folder_plots'], dim_subplots=(2, 2))

        plot1.pseudoconvexity_loss_function(filename ='quasiconvexity_l2norm_' + config.sim_options['current_network']
                                            , subfolder = "experiments/quasiconvexity"
                                            , f_vals = f_vals, grad_f_vals = grad_f_vals, hessian_f_vals = hessian_f_vals
                                            , x_range = theta_attr_grid  #np.arange(-3,3, 0.5)
                                            , theta_true = theta_true[current_network]['tt'])


        # Initial point to perform the scaling using the best value for grid search of the travel time parameter
        min_loss = float('inf')
        min_theta_gs = 0

        for theta_gs, loss in zip(theta_attr_grid,f_vals):
            if loss < min_loss:
                min_loss = loss
                min_theta_gs = theta_gs

        print('best theta is ', str({key: round(val, 3) for key, val in min_theta_gs.items()}))
        # print('best theta: ', str("{0:.0E}".format(min_theta_gs)))

        config.theta_0['tt'] = min_theta_gs

    if config.estimation_options['theta_search'] == 'random':

        uncongested_network_objective_function = tai.estimation.loss_predicted_counts_uncongested_network \
            (x_bar=np.array(list(xc.values()))[:, np.newaxis], Nt=N['train'][current_network]
             , k_Y=k_Y, k_Z=config.estimation_options['k_Z'], theta_0=config.theta_0)

        print('Objective function under equilikely route choices: ' + '{:,}'.format(
            round(uncongested_network_objective_function)))

        # Bound for values of the theta vector entries
        config.bounds_theta_0 = {key: (-0.2, 0.2) for key, val in config.theta_0.items()}

        # No random search is performed along the scale of the od matrix is the q_random_search option is false

        if config.estimation_options['q_random_search']:
            config.bounds_q = (0, 2)

        else:
            config.bounds_q = (1, 1)

        thetas_rs, q_scales_rs, f_vals \
            = tai.estimation.random_search_optimization(Nt=N['train'][current_network]
                                                      , k_Y=k_Y, k_Z=config.estimation_options['k_Z']
                                                      , x_bar=np.array(list(xc.values()))[:, np.newaxis]
                                                      , n_draws = config.estimation_options['n_draws_random_search']
                                                      , theta_bounds = config.bounds_theta_0
                                                      , q_bounds= config.bounds_q
                                                      , inneropt_params={
                                                        'iters': config.estimation_options['max_sue_iters_refined'], 'accuracy_eq': config.estimation_options['accuracy_eq']}
                                                      , silent_mode = True
                                                      )

        min_loss = float('inf')
        min_theta_rs = 0
        min_q_scale_rs = 0

        for theta_rs, q_scale_rs, loss in zip(thetas_rs, q_scales_rs, f_vals):
            if loss < min_loss:
                min_loss = loss
                min_theta_rs = theta_rs
                min_q_scale_rs = q_scale_rs

        # print('best theta is: ', str({key: "{0:.1E}".format(val) for key, val in min_theta_rs.items()}))

        print('best theta:', str({key: round(val, 3) for key, val in min_theta_rs.items()}))

        # print('best q scale is: ', str({key: "{0:.1E}".format(val) for key, val in min_q_scale_rs.items()}))
        print('best q scale:', str({key: round(val, 3) for key, val in min_q_scale_rs.items()}))

        print('minimum loss:', '{:,}'.format(min_loss))
        # print('initial loss:', '{:,}'.format(min_loss))

        # ttest_rs, criticalval_rs, pval_rs \
        #     = tai.estimation.ttest_theta(theta_h0=0
        #                                  , theta=config.theta_0
        #                                  , YZ_x=tai.estimation.get_design_matrix(
        #         Y={'tt': results_eq_gs['tt_x']}
        #         , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=config.estimation_options['k_Z'])
        #                                  , xc=np.array(list(xc.values()))[:, np.newaxis]
        #                                  , q=tai.networks.denseQ(Q=N['train'][current_network].Q
        #                                                          ,
        #                                                          remove_zeros=N['train'][current_network].setup_options[
        #                                                              'remove_zeros_Q'])
        #                                  , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
        #                                  , C=N['train'][current_network].C
        #                                  , pct_lowest_sse=config.estimation_options['pct_lowest_sse_norefined']
        #                                  , alpha=0.05)



        #Update the parameter for the initial theta values
        for attr in min_theta_rs.keys():
            config.theta_0[attr] = min_theta_rs[attr]


if config.estimation_options['scaling_Q']:

    # If the scaling factor is too little, the gradients become 0 apparently.


    # Create grid
    scale_grid_q = [1e-1, 5e-1, 1e0, 2e0, 4e0]

    # Add best scale found by random search into grid
    if config.estimation_options['theta_search'] == 'random':
        scale_grid_q.append(list(min_q_scale_rs.values())[0])

    # We do not generate new paths but use those that were read already from a I/O
    loss_scaling = tai.estimation.scale_Q(x_bar = np.array(list(xc.values()))[:, np.newaxis]
                                          , Nt = N['train'][current_network], k_Y = k_Y, k_Z = config.estimation_options['k_Z']
                                          , theta_0 = config.theta_0
                                          # , scale_grid = [1e-3,1e-2,1e-1]
                                          # , scale_grid=[10e-1]
                                          , scale_grid = scale_grid_q
                                          , n_paths = None #config.estimation_options['n_paths_column_generation']
                                          # , scale_grid = [9e-1,10e-1,11e-1]
                                          , silent_mode= True
                                          )

    # Search for best scaling

    min_loss = float('inf')
    min_scale = 1
    for scale,loss in loss_scaling.items():
        if loss < min_loss:
            min_loss = loss
            min_scale = scale

    # Perform scaling that minimizes the loss
    N['train'][current_network].Q = min_scale*N['train'][current_network].Q
    N['train'][current_network].q = tai.networks.denseQ(Q=N['train'][current_network].Q, remove_zeros= N['train'][current_network].setup_options['remove_zeros_Q'])

    print('Q matrix was rescaled with a ' + str(round(min_scale,2)) + ' factor')

    print('minimum loss:', '{:,}'.format(round(min_loss,1)))


    results_congested_gs = tai.equilibrium.sue_logit_iterative(
        Nt=N['train'][current_network], theta=config.theta_0, k_Y=k_Y
        , k_Z=config.estimation_options['k_Z']
        , params = {'iters': config.estimation_options['max_sue_iters_norefined'], 'accuracy_eq': config.estimation_options['accuracy_eq']
            , 'method': 'line_search', 'iters_ls': 10, 'uncongested_mode': config.sim_options['uncongested_mode']
                    }
    )

    # ttest_gs, criticalval_gs, pval_gs \
    #     = tai.estimation.ttest_theta(theta_h0=0
    #                                  , theta=config.theta_0
    #                                  , YZ_x=tai.estimation.get_design_matrix(Y={'tt': results_congested_gs['tt_x']}, Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=config.estimation_options['k_Z'])
    #                                  , xc=np.array(list(xc.values()))[:, np.newaxis]
    #                                  , q=N['train'][current_network].q
    #                                  , Ix=N['train'][current_network].D
    #                                  , Iq=N['train'][current_network].M
    #                                  , C=N['train'][current_network].C
    #                                  , pct_lowest_sse=config.estimation_options['pct_lowest_sse_norefined']
    #                                  , alpha=0.05)


# =============================================================================
# 3.2) BENCHMARK PREDICTIONS
# =============================================================================

x_bar = np.array(list(xc.values()))[:, np.newaxis]

# Naive prediction using mean counts
config.estimation_results['mean_counts_prediction_loss'], config.estimation_results['mean_count_benchmark_model'] = tai.estimation.mean_count_l2norm(x_bar =  np.array(list(xc.values()))[:, np.newaxis], mean_x = config.estimation_results['mean_count_benchmark_model'])

print('\nObjective function under mean count prediction: ' + '{:,}'.format(round(config.estimation_results['mean_counts_prediction_loss'],1)))

# Naive prediction using uncongested network
config.estimation_results['equilikely_prediction_loss'] \
    = tai.estimation.loss_predicted_counts_uncongested_network(
    x_bar = np.array(list(xc.values()))[:, np.newaxis], Nt = N['train'][current_network]
    , k_Y = k_Y, k_Z = config.estimation_options['k_Z'], theta_0 = dict.fromkeys(config.theta_0, 0))

print('Objective function under equilikely route choices: ' + '{:,}'.format(round(config.estimation_results['equilikely_prediction_loss'],1)))



# =============================================================================
# 3d) ESTIMATION
# =============================================================================

# i) NO REFINED OPTIMIZATION AND INFERENCE

# Features includes in utility function for estimation
if k_Z_estimation is None:
    k_Z_estimation = config.estimation_options['k_Z']

q_norefined_bilevelopt, theta_norefined_bilevelopt, objective_norefined_bilevelopt,result_eq_norefined_bilevelopt, results_norefined_bilevelopt \
    = tai.estimation.odtheta_estimation_bilevel(
    # Nt= tai.modeller.clone_network(N['train'][i], label = N['train'][i].label),
    Nt= N['train'][current_network],
    k_Y=k_Y, k_Z=k_Z_estimation,
    Zt={1: N['train'][current_network].Z_dict},
    q0 = N['train'][current_network].q,
    xct={1: np.array(list(xc.values()))},
    theta0=  config.theta_0, # If change to positive number, a higher number of iterations is required but it works well
    # theta0 = theta_true[i],
    standardization = config.estimation_options['standardization_norefined'],
    outeropt_params={
        # 'method': 'adagrad',
        'method': config.estimation_options['outeropt_method_norefined'],
        # 'method': 'adam',
        'iters_scaling': int(0e0),
        'iters': config.estimation_options['iters_norefined'], #10
        'batch_size': config.estimation_options['links_batch_size'],
        'paths_batch_size': config.estimation_options['paths_batch_size'],
        'eta_scaling': 1e-2,
        'eta': config.estimation_options['eta_norefined'], # works well for simulated networks
        # 'eta': 1e-4, # works well for Fresno real network
        'gamma': 0,
        'v_lm': 1, 'lambda_lm': 0,
        'beta_1': 0.9, 'beta_2': 0.99
    },
    inneropt_params = {'iters': config.estimation_options['max_sue_iters_norefined'], 'accuracy_eq': config.estimation_options['accuracy_eq']
        , 'method': 'line_search', 'iters_ls': 20
        , 'k_path_set_selection': config.estimation_options['k_path_set_selection']
        ,'dissimilarity_weight' : config.estimation_options['dissimilarity_weight']
        , 'uncongested_mode': config.sim_options['uncongested_mode']
                       },
    bilevelopt_params = {'iters': config.estimation_options['bilevel_iters_norefined']}  # {'iters': 10},
    # plot_options = {'y': 'objective'}
    , n_paths_column_generation= config.estimation_options['n_paths_column_generation']
    , silent_mode = True
)

config.estimation_results['theta_norefined'] = theta_norefined_bilevelopt
config.estimation_results['best_loss_norefined'] = objective_norefined_bilevelopt


# Statistical inference
print('\nInference with no refined solution')
# print('\ntheta no refined: ' + str(np.array(list(theta_norefined_bilevelopt.values()))[:,np.newaxis].T))

# ttest_norefined, criticalval_norefined, pval_norefined \
#     = tai.estimation.ttest_theta(theta_h0=0
#                                  , theta=theta_norefined_bilevelopt
#                                  ,YZ_x=tai.estimation.get_design_matrix(Y={'tt': result_eq_norefined_bilevelopt['tt_x']}
#                                                                         , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
#                                  , xc= np.array(list(xc.values()))[:, np.newaxis]
#                                  , q=tai.networks.denseQ(Q=N['train'][current_network].Q
#                                                          , remove_zeros=N['train'][current_network].setup_options['remove_zeros_Q'])
#                                  , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
#                                  , C=N['train'][current_network].C
#                                  , pct_lowest_sse = config.estimation_options['pct_lowest_sse_norefined']
#                                  , alpha = 0.05)

parameter_inference_norefined_table, model_inference_norefined_table \
    = tai.estimation.hypothesis_tests(theta_h0 = 0
                                      , theta = theta_norefined_bilevelopt
                                      , YZ_x=tai.estimation.get_design_matrix(Y={'tt': result_eq_norefined_bilevelopt['tt_x']}
                                                                              , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
                                      , xc=np.array(list(xc.values()))[:, np.newaxis]
                                      , q=tai.networks.denseQ(Q=N['train'][current_network].Q
                                                              , remove_zeros=N['train'][current_network].setup_options['remove_zeros_Q'])
                                      , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
                                      , C=N['train'][current_network].C
                                      , pct_lowest_sse=config.estimation_options['pct_lowest_sse_norefined']
                                      , alpha=0.05)


with pd.option_context('display.float_format', '{:0.3f}'.format):

    print('\nSummary of logit parameters: \n', parameter_inference_norefined_table.to_string(index = False))
    # tai.writer.write_csv_to_log_folder(df=parameter_inference_norefined_table, filename='parameter_inference_norefined_table'
    #                                    , log_file=config.log_file)


    print('\nSummary of model: \n', model_inference_norefined_table.to_string(index = False))
    # tai.writer.write_csv_to_log_folder(df=model_inference_norefined_table,
    #                                    filename='model_inference_norefined_table'
    #                                    , log_file=config.log_file)


# confint_theta_norefined, width_confint_theta_norefined = tai.estimation.confint_theta(
#     theta=theta_norefined_bilevelopt
#     , YZ_x=tai.estimation.get_design_matrix(
#         Y={'tt': result_eq_norefined_bilevelopt['tt_x']}
#         , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=config.estimation_options['k_Z'])
#     , xc= np.array(list(xc.values()))[:, np.newaxis]
#     , q=tai.networks.denseQ(Q=N['train'][current_network].Q, remove_zeros=N['train'][current_network].setup_options['remove_zeros_Q']), Ix=N['train'][current_network].D, Iq=N['train'][current_network].M,
#     C=N['train'][current_network].C, alpha=0.05)

# ttest_norefined, criticalval_norefined, pval_norefined \
#     = tai.estimation.ttest_theta(theta_h0=0
#                                       , theta=theta_norefined_bilevelopt
#                                       , YZ_x=tai.estimation.get_design_matrix(Y={'tt': result_eq_norefined_bilevelopt['tt_x']}
#                                                                         , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
#                                       , xc= np.array(list(xc.values()))[:, np.newaxis]
#                                       , q=tai.networks.denseQ(Q=N['train'][current_network].Q
#                                                          , remove_zeros=N['train'][current_network].setup_options['remove_zeros_Q'])
#                                       , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
#                                       , C=N['train'][current_network].C
#                                       , pct_lowest_sse = config.estimation_options['pct_lowest_sse_norefined']
#                                       , alpha = 0.05)
#
#
# ftest, critical_fvalue, pvalue = tai.estimation.ftest(theta_m1
#                                                       = dict(zip(theta_norefined_bilevelopt.keys(),np.zeros(len(theta_norefined_bilevelopt))))
#                                                       , theta_m2 = theta_norefined_bilevelopt
#                                                       , YZ_x=tai.estimation.get_design_matrix(Y={'tt': result_eq_norefined_bilevelopt['tt_x']}
#                                                                                               , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
#                                                       , xc=np.array(list(xc.values()))[:, np.newaxis]
#                                                       , q=tai.networks.denseQ(Q=N['train'][current_network].Q
#                                                                               , remove_zeros=N['train'][current_network].setup_options['remove_zeros_Q'])
#                                                       , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
#                                                       , C=N['train'][current_network].C
#                                                       , pct_lowest_sse=config.estimation_options['pct_lowest_sse_norefined']
#                                                       , alpha=0.05)

if config.estimation_options['ttest_selection_norefined'] :

    if config.estimation_options['alpha_selection_norefined'] < len(theta_norefined_bilevelopt):

        # An alternative to regularization
        ttest_norefined = np.array(parameter_inference_norefined_table['t-test'])
        ttest_norefined_dict = dict(
            zip(k_Y + k_Z_estimation, list(map(float, parameter_inference_norefined_table['t-test']))))

        n = np.count_nonzero(~np.isnan(np.array(list(xc.values()))[:, np.newaxis]))
        p = len(k_Y + k_Z_estimation)

        critical_alpha = config.estimation_options['alpha_selection_norefined']
        critical_tvalue = stats.t.ppf(1 - critical_alpha / 2, df=n - p)

        if config.estimation_options['alpha_selection_norefined'] >=1:
            # It picks the alpha minimum(s) ignoring fixed effects

            ttest_lists = []
            for attr, ttest, idx in zip(ttest_norefined_dict.keys(),ttest_norefined.flatten(), np.arange(p)):
                if attr not in N['train'][current_network].k_fixed_effects:
                    ttest_lists.append(ttest)

            critical_tvalue = float(-np.sort(-abs(np.sort(ttest_lists)))[config.estimation_options['alpha_selection_norefined']-1])

            print('Selecting top ' + str(config.estimation_options['alpha_selection_norefined']) + ' features based on t-values and excluding fixed effects' )

        else:
            print('Selecting features based on critical t-value ' + str(critical_tvalue))

        # print('\ncritical_tvalue:', critical_tvalue)

        # Loop over endogenous and exogenous attributes

        for attr, t_test in ttest_norefined_dict.items():

            if attr not in N['train'][current_network].k_fixed_effects:

                if abs(t_test) < critical_tvalue-1e-3:
                    if attr in k_Y:
                        k_Y.remove(attr)

                    if attr in k_Z_estimation:
                        k_Z_estimation.remove(attr)

        print('k_Y:', k_Y)
        print('k_Z:', k_Z_estimation)

if config.estimation_options['link_selection']:

    # Create a dictionary with the predicted counts for every link over iterations
    predicted_counts_link_dict = {iter:results_norefined_bilevelopt[iter]['equilibrium']['x'] for iter in results_norefined_bilevelopt.keys()}

    #Create a dictionary with link keys and predicted counts over iterations as values
    predicted_link_counts_dict = {}
    counter = 0

    for iteration, link_counts_dict in predicted_counts_link_dict.items():

        for link_key, count in link_counts_dict.items():

            if counter == 0:
                predicted_link_counts_dict[link_key] = [count]
            else:
                predicted_link_counts_dict[link_key].append(count)

        counter += 1

    removed_links_counter= 0
    removed_links_list = []
    # Compute variance in predictions over iterations of links with observed counts
    for link  in N['train'][current_network].get_observed_links():
        counts_variance = np.var(predicted_link_counts_dict[link.key])

        # If variance is close to zero, the model is not able to predict well for those links
        if np.isclose(counts_variance,0,1e-3):
            removed_links_list.append(link.key)

            link.observed_count = np.nan

            xc[link.key] = np.nan
            xc_validation[link.key] = np.nan
            removed_links_counter += 1

    print("\n" + str(removed_links_counter), " traffic counts observations were removed due to little or null variance in predicted counts over iterations of non refined optimization")
    print('removed links:', removed_links_list)

# print(confint_theta)
# print(width_confint_theta)

# for iter in np.arange(len(list(results_bilevelopt.values()))):
#     print('\n iter: ' + str(iter) )
#     print('theta : ' + str(np.round(results_bilevelopt[iter]['theta'],2)))
#     print('objective: ' + str(np.round(results_bilevelopt[iter]['objective'],2)))

# ii) REFINED OPTIMIZATION AND INFERENCE

# k_Z_estimation = []

if config.estimation_options['outofsample_prediction_mode']:

    theta_refined_bilevelopt, objective_refined_bilevelopt, result_eq_refined_bilevelopt, results_refined_bilevelopt = \
        copy.deepcopy(theta_norefined_bilevelopt), copy.deepcopy(objective_norefined_bilevelopt), copy.deepcopy(result_eq_norefined_bilevelopt), copy.deepcopy(results_norefined_bilevelopt)

    config.estimation_results['theta_refined'] = copy.deepcopy(theta_refined_bilevelopt)
    config.estimation_results['best_loss_refined'] = copy.deepcopy(objective_refined_bilevelopt)

    parameter_inference_refined_table, model_inference_refined_table = \
        copy.deepcopy(parameter_inference_norefined_table), copy.deepcopy(model_inference_norefined_table)

else:

    # Fine scale solution (the initial objective can be different because we know let's more iterations to be performed to achieve equilibrium)
    q_refined_bilevel_opt, theta_refined_bilevelopt, objective_refined_bilevelopt,result_eq_refined_bilevelopt, results_refined_bilevelopt \
        = tai.estimation.odtheta_estimation_bilevel(Nt= N['train'][current_network],
                                                    k_Y=k_Y, k_Z=k_Z_estimation,
                                                    Zt={1: N['train'][current_network].Z_dict},
                                                    q0 = N['train'][current_network].q,
                                                    xct={1: np.array(list(xc_validation.values()))},
                                                    theta0= theta_norefined_bilevelopt,
                                                    # theta0= dict.fromkeys(k_Y+config.estimation_options['k_Z'],0),
                                                    outeropt_params={
                                                        # 'method': 'gauss-newton'
                                                          'method': config.estimation_options['outeropt_method_refined']
                                                        # 'method': 'lm-revised'
                                                        # 'method': 'gd'
                                                        # 'method': 'ngd'
                                                        ,'iters_scaling': int(0e0)
                                                        ,'iters': config.estimation_options['iters_refined'] #int(2e1)
                                                        , 'batch_size': 0*config.estimation_options['links_batch_size']
                                                        , 'paths_batch_size': config.estimation_options['paths_batch_size']
                                                        , 'eta_scaling': 1e-2
                                                        , 'eta': config.estimation_options['eta_refined'] #1e-6
                                                        , 'gamma': 0
                                                        , 'v_lm': 1e3, 'lambda_lm': 1e1
                                                        , 'beta_1': 0.9, 'beta_2': 0.99
                                                        },
                                                    inneropt_params = {'iters': config.estimation_options['max_sue_iters_refined'], 'accuracy_eq': config.estimation_options['accuracy_eq']
                                                        , 'method': 'line_search', 'iters_ls': 20
                                                        , 'uncongested_mode': config.sim_options['uncongested_mode']
                                                        },  #{'iters': 100, 'accuracy_eq': config.estimation_options['accuracy_eq']},
                                                    bilevelopt_params = {'iters': config.estimation_options['bilevel_iters_refined']}  #{'iters': 10}
                                                    # , plot_options = {'y': 'objective'}
                                                    , n_paths_column_generation=config.estimation_options['n_paths_column_generation']
                                                    , silent_mode = True
                                                    )

    config.estimation_results['theta_refined'] = theta_refined_bilevelopt
    config.estimation_results['best_loss_refined'] = objective_refined_bilevelopt

    # Statistical inference
    print('\nInference with refined solution')

    # print('\ntheta refined: ' + str(np.array(list(theta_refined_bilevelopt.values()))[:,np.newaxis].T))
    # Note: this may sound countertuituive but the fact that the refined solution makes the error to be close to 0, it makes the confidence intervals to be very narrow
    # and because of that, we tend to increase the type II error. This does not happen with the no refined solution where there is some amount of error.
    # ttest_refined, criticalval_refined, pval_refined \
    #     = tai.estimation.ttest_theta(theta_h0=0
    #                                  , theta=theta_refined_bilevelopt,
    #                                  YZ_x=tai.estimation.get_design_matrix(
    #                                      Y={'tt': result_eq_refined_bilevelopt['tt_x']}
    #                                      , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
    #                                  , xc= np.array(list(xc_validation.values()))[:, np.newaxis]
    #                                  , q=tai.networks.denseQ(Q=N['train'][current_network].Q, remove_zeros=N['train'][current_network].setup_options['remove_zeros_Q']),
    #                                  Ix=N['train'][current_network].D, Iq=N['train'][current_network].M,
    #                                  C=N['train'][current_network].C
    #                                  , pct_lowest_sse = config.estimation_options['pct_lowest_sse_refined']
    #                                  , alpha = 0.05)


    # confint_theta_refined, width_confint_theta_refined = tai.estimation.confint_theta(
    #     theta=theta_refined_bilevelopt
    #     , YZ_x=tai.estimation.get_design_matrix(Y={'tt': result_eq_refined_bilevelopt['tt_x']}, Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
    #     , xc= np.array(list(xc_validation.values()))[:, np.newaxis]
    #     , q=tai.networks.denseQ(Q=N['train'][current_network].Q, remove_zeros=N['train'][current_network].setup_options['remove_zeros_Q'])
    #     , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
    #     , C=N['train'][current_network].C, alpha=0.05)

    # ftest_refined, critical_fvalue_refined, pvalue_refined = tai.estimation.ftest(theta_m1 = dict(zip(theta_refined_bilevelopt.keys(),np.zeros(len(theta_refined_bilevelopt))))
    #                                                       , theta_m2 = theta_refined_bilevelopt
    #                                                       , YZ_x=tai.estimation.get_design_matrix(Y={'tt': result_eq_refined_bilevelopt['tt_x']}
    #                                                                                               , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
    #                                                       , xc=np.array(list(xc.values()))[:, np.newaxis]
    #                                                       , q=tai.networks.denseQ(Q=N['train'][current_network].Q
    #                                                                               , remove_zeros=N['train'][current_network].setup_options['remove_zeros_Q'])
    #                                                       , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
    #                                                       , C=N['train'][current_network].C
    #                                                       , pct_lowest_sse=config.estimation_options['pct_lowest_sse_refined']
    #                                                       , alpha=0.05)


    parameter_inference_refined_table, model_inference_refined_table \
        = tai.estimation.hypothesis_tests(theta_h0 = 0
                                          , theta = theta_refined_bilevelopt
                                          , YZ_x=tai.estimation.get_design_matrix(Y={'tt': result_eq_refined_bilevelopt['tt_x']}
                                                                                  , Z=N['train'][current_network].Z_dict, k_Y=k_Y, k_Z=k_Z_estimation)
                                          , xc=np.array(list(xc.values()))[:, np.newaxis]
                                          , q=tai.networks.denseQ(Q=N['train'][current_network].Q
                                                                  , remove_zeros=N['train'][current_network].setup_options['remove_zeros_Q'])
                                          , Ix=N['train'][current_network].D, Iq=N['train'][current_network].M
                                          , C=N['train'][current_network].C
                                          , pct_lowest_sse=config.estimation_options['pct_lowest_sse_refined']
                                          , alpha=0.05)

    with pd.option_context('display.float_format', '{:0.3f}'.format):

        print('\nSummary of logit parameters: \n', parameter_inference_refined_table.to_string(index = False))
        # tai.writer.write_csv_to_log_folder(df=parameter_inference_refined_table, filename='parameter_inference_refined_table'
        #                                    , log_file=config.log_file)

        print('\nSummary of model: \n', model_inference_refined_table.to_string(index = False))
        # tai.writer.write_csv_to_log_folder(df=model_inference_refined_table, filename='model_inference_refined_table'
        #                                    , log_file=config.log_file)



# VISUALIZATION

# Distribution of errors across link counts

best_x_eq_norefined = np.array(list(results_norefined_bilevelopt[config.estimation_options['bilevel_iters_norefined']]['equilibrium']['x'].values()))[:,np.newaxis]

best_x_eq_refined = np.array(list(results_refined_bilevelopt[config.estimation_options['bilevel_iters_refined']]['equilibrium']['x'].values()))[:,np.newaxis]

# print('Loss by link', tai.estimation.loss_function_by_link(x_bar = np.array(list(xc.values()))[:,np.newaxis], x_eq = best_x_eq_norefined))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True,figsize=(10,5))

# We can set the number of bins with the `bins` kwarg
axs[0].hist(tai.estimation.error_by_link(x_bar = np.array(list(xc.values()))[:, np.newaxis], x_eq = best_x_eq_norefined))
axs[1].hist(tai.estimation.error_by_link(x_bar = np.array(list(xc.values()))[:, np.newaxis], x_eq = best_x_eq_refined))

for axi in [axs[0],axs[1]]:
    axi.tick_params(axis = 'x', labelsize=16)
    axi.tick_params(axis = 'y', labelsize=16)

plt.show()

tai.writer.write_figure_to_log_folder(fig = fig
                                      , filename = 'distribution_predicted_count_error.pdf', log_file = config.log_file)


# - Generate pandas dataframe prior plotting
results_norefined_refined_df = tai.descriptive_statistics \
    .get_loss_and_estimates_over_iterations(results_norefined = results_norefined_bilevelopt
                                            , results_refined = results_refined_bilevelopt)

# Plot
plot1 = tai.Artist(folder_plots = config.plots_options['folder_plots'], dim_subplots=(2, 2))

fig = plot1.bilevel_optimization_convergence(
    results_norefined_df = results_norefined_refined_df[results_norefined_refined_df['stage'] == 'norefined']
    , results_refined_df = results_norefined_refined_df[results_norefined_refined_df['stage'] == 'refined']
    , simulated_data = config.sim_options['simulated_counts']
    , filename='loss-vs-vot-over-iterations_' + config.sim_options['current_network']
    , subfolder="experiments/inference"
    , methods = [config.estimation_options['outeropt_method_norefined'],config.estimation_options['outeropt_method_refined']]
)

plt.show()

tai.writer.write_figure_to_log_folder(fig = fig
                                      , filename = 'bilevel_optimization_convergence.pdf', log_file = config.log_file)



# =============================================================================
# 6) LOG FILE
# =============================================================================

# =============================================================================
# 6a) Summary with most relevant options, prediction error, initial parameters, etc
# =============================================================================

tai.writer.write_estimation_report(filename='summary_report'
                                   , config = config
                                   , decimals = 4
                                   # , float_format='%.3f'
                                   )

# =============================================================================
# 6b) General options (sim_options, estimation_options and matching statistics)
# =============================================================================

# Update vector with exogenous covariates
config.estimation_options['k_Z'] = config.estimation_options['k_Z']

# general_dict = {'type': 'sim_option', 'key': 'selected_year', 'value': 2019}
options_df = pd.DataFrame({'group': [], 'option': [], 'value': []})

for key, value in config.sim_options.items():
    options_df = options_df.append(pd.DataFrame({'group': ['sim_options'], 'option': [key], 'value': [value]}), ignore_index = True)

for key, value in config.estimation_options.items():
    options_df = options_df.append({'group': 'estimation_options', 'option': key, 'value': value}, ignore_index = True)

for key, value in config.gis_options.items():
    options_df = options_df.append({'group': 'gis_options', 'option': key, 'value': value}, ignore_index = True)

# for key, value in config.gis_results.items():
#     options_df = options_df.append({'group': 'gis_results', 'option': key, 'value': value}, ignore_index = True)


tai.writer.write_csv_to_log_folder(df= options_df,
                                   filename='global_options'
                                   , log_file=config.log_file
                                   , float_format='%.1f'
                                   )

# =============================================================================
# 6b) Analysis of predicted counts over iterations
# =============================================================================

# Log file
predicted_link_counts_over_iterations_df \
    = tai.descriptive_statistics.get_predicted_link_counts_over_iterations_df(
    results_norefined = results_norefined_bilevelopt
    , results_refined = results_refined_bilevelopt
    ,  Nt = N['train'][current_network])

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
tai.writer.write_csv_to_log_folder(df=predicted_link_counts_over_iterations_df ,
                                   filename='predicted_link_counts_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.1f'
                                   )


gap_predicted_link_counts_over_iterations_df \
    = tai.descriptive_statistics.get_gap_predicted_link_counts_over_iterations_df(
    results_norefined = results_norefined_bilevelopt
    , results_refined = results_refined_bilevelopt
    ,  Nt = N['train'][current_network])

tai.writer.write_csv_to_log_folder(df=gap_predicted_link_counts_over_iterations_df ,
                                   filename='gap_predicted_link_counts_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.1f'
                                   )


# Travel times
predicted_link_traveltime_over_iterations_df \
    = tai.descriptive_statistics.get_predicted_traveltimes_over_iterations_df(
    results_norefined = results_norefined_bilevelopt
    , results_refined = results_refined_bilevelopt
    ,  Nt = N['train'][current_network])

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
tai.writer.write_csv_to_log_folder(df=predicted_link_traveltime_over_iterations_df ,
                                   filename='predicted_link_traveltimes_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.2f'
                                   )



# =============================================================================
# 6c) Analysis of parameter estimates and loss over iterations
# =============================================================================

# Log file
loss_and_estimates_over_iterations_df \
    = tai.descriptive_statistics.get_loss_and_estimates_over_iterations(
    results_norefined = results_norefined_bilevelopt
    , results_refined = results_refined_bilevelopt)

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
tai.writer.write_csv_to_log_folder(df=loss_and_estimates_over_iterations_df ,
                                   filename='estimates_and_losses_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.3f'
                                   )


gap_estimates_over_iterations_df \
    = tai.descriptive_statistics.get_gap_estimates_over_iterations(
    results_norefined=results_norefined_bilevelopt
    , results_refined=results_refined_bilevelopt
    ,  theta_true = theta_true[current_network])

tai.writer.write_csv_to_log_folder(df=gap_estimates_over_iterations_df ,
                                   filename='gap_estimates_over_iterations_df'
                                   , log_file=config.log_file
                                   , float_format='%.3f'
                                   )





# =============================================================================
# 6c) Best parameter estimates and inference at the end of norefined and refined stages
# =============================================================================

# T-tests, confidence intervals and parameter estimates
parameter_inference_norefined_table.insert(0,'stage','norefined')
parameter_inference_refined_table.insert(0,'stage','refined')
parameter_inference_table = parameter_inference_norefined_table.append(parameter_inference_refined_table)

# print('\nSummary of logit parameters: \n', parameter_inference_norefined_table.to_string(index=False))
tai.writer.write_csv_to_log_folder(df=parameter_inference_table, filename='parameter_inference_table'
                                   , log_file=config.log_file)


# F-test and model summary statistics
model_inference_norefined_table.insert(0,'stage','norefined')
model_inference_refined_table.insert(0,'stage','refined')

model_inference_table = model_inference_norefined_table.append(model_inference_refined_table)

# print('\nSummary of model: \n', model_inference_norefined_table.to_string(index=False))
tai.writer.write_csv_to_log_folder(df=model_inference_table,
                                   filename='model_inference_table'
                                   , log_file=config.log_file)









sys.exit()

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
# theta_myalg, grad_myalg = tai.estimation.single_level_odtheta_estimation(M = {1: N['train'][i].M}
#                     , C={1: tai.estimation.choice_set_matrix_from_M(N['train'][i].M)}
#                     , D = {1: N['train'][i].D}
#                     , q0= tai.network.denseQ(Q=N['train'][i].Q, remove_zeros=remove_zeros_Q)
#                     , k_Y = k_Y, k_Z = config.estimation_options['k_Z']
#                     , Y = {1:N['train'][i].Y_dict}, Z = {1:N['train'][i].Z_dict}
#                     , x = {1:N['train'][i].x} #x_N
#                     , theta0 = dict.fromkeys([*k_Y,*config.estimation_options['k_Z']],0)
#                     # , theta0 = {k: 2*theta_true[i][k] for k in [*k_Y,*config.estimation_options['k_Z']]}
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
# tai.estimation.

# k_Z.append('k0')

theta_myalg, grad_myalg, final_objective = tai.estimation.odtheta_estimation_outer_problem(k_Y=k_Y, Yt={1: N['train'][i].Y_dict},
                                                                                           k_Z=k_Z, Zt={1: N['train'][i].Z_dict},
                                                                                           q0=tai.networks.denseQ(Q=N['train'][i].Q,
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

# tai.estimation.
theta_myalg_adjusted, grad_myalg, final_objective = tai.estimation.odtheta_estimation_outer_problem(k_Y=k_Y,
                                                                                                    Yt={1: N['train'][i].Y_dict},
                                                                                                    k_Z=k_Z,
                                                                                                    Zt={1: N['train'][i].Z_dict},
                                                                                                    q0=tai.networks.denseQ(
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

ttest, criticalval, pval = tai.estimation.ttest_theta(theta_h0=0, theta=np.array(list(theta_myalg.values()))[:,np.newaxis],
                                                      YZ_x=tai.estimation.get_design_matrix(Y=N['train'][i].Y_dict,
                                                                                            Z=N['train'][i].Z_dict,
                                                                                            k_Y=k_Y, k_Z=k_Z),
                                                      xc=N['train'][i].x[:, np.newaxis],
                                                      q=tai.networks.denseQ(Q=N['train'][i].Q,
                                                                            remove_zeros=remove_zeros_Q),
                                                      Ix=N['train'][i].D, Iq=N['train'][i].M,
                                                      C=tai.estimation.choice_set_matrix_from_M(N['train'][i].M),
                                                      alpha=0.05)

print(ttest)
print('pvals :' +  str(pval))

# ttest1, criticalval, pval = tai.estimation.ttest_theta(theta_h0 =1*np.ones(len(theta_myalg))[:,np.newaxis], alpha = 0.05
#                                                       # , theta = np.array(list({k: 1*theta_true[i][k] for k in [*k_Y,*k_Z]}.values()))[:,np.newaxis]
#                                                       , theta = theta_myalg
#                                                       ,YZ_x = tai.estimation.get_design_matrix(Y = N['train'][i].Y_dict, Z = N['train'][i].Z_dict, k_Y = k_Y, k_Z = k_Z)
#                                                       , x_bar = N['train'][i].x[:, np.newaxis]
#                                                       ,q =tai.network.denseQ(Q=N['train'][i].Q,remove_zeros=remove_zeros_Q)
#                                                       , Ix = N['train'][i].D, Iq=N['train'][i].M, C = tai.estimation.choice_set_matrix_from_M(N['train'][i].M) )
#
# print(ttest1)

# pval
# ttest1
# criticalval

confint_theta, width_confint_theta = tai.estimation.confint_theta(
    theta=np.array(list(theta_myalg.values()))[:, np.newaxis],
    YZ_x=tai.estimation.get_design_matrix(Y=N['train'][i].Y_dict, Z=N['train'][i].Z_dict, k_Y=k_Y, k_Z=k_Z),
    xc=N['train'][i].x[:, np.newaxis], q=tai.networks.denseQ(Q=N['train'][i].Q, remove_zeros=remove_zeros_Q),
    Ix=N['train'][i].D, Iq=N['train'][i].M, C=tai.estimation.choice_set_matrix_from_M(N['train'][i].M), alpha=0.05)

print(confint_theta)
print(width_confint_theta)

# confint_theta.shape

# Post-scaling
# tai.estimation.
# # i = 'N3'
theta_myalg_adjusted, grad_myalg = tai.estimation.odtheta_estimation_outer_problem(k_Y=k_Y,
                                                                                   Yt={1: N['train'][i].Y_dict},
                                                                                   k_Z=k_Z,
                                                                                   Zt={1: N['train'][i].Z_dict},
                                                                                   q0=tai.networks.denseQ(
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
# l1 = np.sum((tai.estimation.prediction_x(3 * theta_true_array) - x_bar) ** 2)
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

################ Solving for uncongested network with black box scipy minimize optimizer ############

t0 = time.time()
theta_estimate = tai.estimation.solve_link_level_model(end_params={'theta': True, 'q': False}, Mt={1: N['train'][i].M},
                                                       Ct={1: tai.estimation.choice_set_matrix_from_M(N['train'][i].M)},
                                                       Dt={1: N['train'][i].D}, k_Y=k_Y, Yt={1: N['train'][i].Y_dict},
                                                       k_Z=k_Z, Zt={1: N['train'][i].Z_dict}, xt={1: N['train'][i].x},
                                                       idx_links={1: range(0, len(N['train'][i].x))},
                                                       scale={'mean': False, 'std': False},
                                                       q0=tai.networks.denseQ(Q=N['train'][i].Q,
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
results_sue_msa_postcongestion = tai.equilibrium.sue_logit_iterative(Nt=N['train'][i], theta=theta_estimate_congestion,
                                                                     k_Y=k_Y, k_Z=k_Z, params={'maxIter': maxIter, 'accuracy_eq': config.estimation_options['accuracy_eq']})


gap_poscongestion = np.sqrt(np.sum((np.array(list(results_sue_msa[i]['x'].values()))-np.array(list(results_sue_msa_postcongestion['x'].values())))**2)/len(np.array(list(results_sue_msa[i]['x']))))

# Gap with initial theta (no estimation)
theta_estimate_initial = dict.fromkeys([*k_Y,*k_Z],0)
theta_estimate_initial['speed'] = 0
theta_estimate_initial['length'] = 0
theta_estimate_initial['toll'] = 0
x_initial = tai.equilibrium.sue_logit_iterative(Nt=N['train'][i], theta=theta_estimate_initial, k_Y=k_Y, k_Z=k_Z,
                                                params={'maxIter': maxIter, 'accuracy_eq': config.estimation_options['accuracy_eq']})['x']

gap_initial = np.sqrt(np.sum((np.array(list(results_sue_msa[i]['x'].values()))-np.array(list(x_initial.values())))**2)/len(np.array(list(results_sue_msa[i]['x']))))



i = 'N5'
results_sue_fiske = tai.equilibrium.sue_logit_fisk(q = tai.networks.denseQ(Q = N['train'][i].Q, remove_zeros = remove_zeros_Q)
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
N['train'][i].set_Y_attr_links(y= results_sue_fiske['tt_x'], label='tt')


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

tai.writer.write_network_to_dat(root =  root_github
                                , subfolder = 'SiouxFalls/' , prefix_filename = 'SiouxFalls', N = N['train']['SiouxFalls'])


x,tt_x = tai.equilibrium.sue_logit_dial(root = root_github, subfolder ='SiouxFalls', prefix_filename ='SiouxFalls'
                                        , options = {'equilibrium': 'stochastic', 'method': 'MSA', 'maxIter': 100, 'accuracy_eq': config.estimation_options['accuracy_eq']} , Z_dict = N['train']['SiouxFalls'].Z_dict, theta = theta_true['SiouxFalls'], k_Z = k_Z)

N['train']['SiouxFalls'].x_dict = x
N['train']['SiouxFalls'].set_Y_attr_links(y=tt_x, label='tt')

np.sum(list(x.values()))
tt_x.values()

# N['train']['SiouxFalls'].x_dict = dict(zip(list(N['train']['SiouxFalls'].links_dict.keys()),x_iteration))
#
#
# N['train']['SiouxFalls'].set_Y_attr_links(y=dict(zip(list(N['train']['SiouxFalls'].links_dict.keys()),[link.traveltime for link in N_i.links])), label='tt')

theta_estimate = tai.estimation.solve_link_level_model(end_params={'theta': True, 'q': False},
                                                       Mt={1: N['train']['SiouxFalls'].M}, Ct={
        1: tai.estimation.choice_set_matrix_from_M(N['train']['SiouxFalls'].M)}, Dt={1: N['train']['SiouxFalls'].D},
                                                       k_Y=k_Y, Yt={1: N['train']['SiouxFalls'].Y_dict}, k_Z=k_Z,
                                                       Zt={1: N['train']['SiouxFalls'].Z_dict},
                                                       xt={1: N['train']['SiouxFalls'].x},
                                                       idx_links={1: range(0, len(N['train']['SiouxFalls'].x))},
                                                       scale={'mean': False, 'std': False},
                                                       q0=tai.networks.denseQ(Q=N['train']['SiouxFalls'].Q,
                                                                              remove_zeros=remove_zeros_Q),
                                                       theta0=dict.fromkeys([*k_Y, *k_Z], 0), lambda_hp=0)



print(theta_estimate['theta']['tt']/theta_estimate['theta']['c'])

print(theta_true['SiouxFalls']['tt']/theta_true['SiouxFalls']['c'])



#EMA

tai.writer.write_network_to_dat(root =  root_github
                                , subfolder = 'Eastern-Massachusetts' , prefix_filename = 'EMA', N = N['train']['Eastern-Massachusetts'])

x,tt_x = tai.equilibrium.sue_logit_dial(root = root_github, subfolder ='Eastern-Massachusetts', prefix_filename ='EMA', maxIter = 100, accuracy = config.estimation_options['accuracy_eq'], Z_dict = N['train']['Eastern-Massachusetts'].Z_dict, theta = theta_true['Eastern-Massachusetts'], k_Z = k_Z)

N['train']['Eastern-Massachusetts'].x_dict = x
N['train']['Eastern-Massachusetts'].set_Y_attr_links(y=tt_x, label='tt')

theta_estimate = tai.estimation.solve_link_level_model(end_params={'theta': True, 'q': False},
                                                       Mt={1: N['train']['Eastern-Massachusetts'].M}, Ct={
        i: tai.estimation.choice_set_matrix_from_M(N['train']['Eastern-Massachusetts'].M)},
                                                       Dt={1: N['train']['Eastern-Massachusetts'].D}, k_Y=k_Y,
                                                       Yt={1: N['train']['Eastern-Massachusetts'].Y_dict}, k_Z=k_Z,
                                                       Zt={1: N['train']['Eastern-Massachusetts'].Z_dict},
                                                       xt={1: N['train']['Eastern-Massachusetts'].x}, idx_links={
        1: range(0, len(N['train']['Eastern-Massachusetts'].x))}, scale={'mean': False, 'std': False},
                                                       q0=tai.networks.denseQ(Q=N['train']['Eastern-Massachusetts'].Q,
                                                                              remove_zeros=remove_zeros_Q),
                                                       theta0=dict.fromkeys([*k_Y, *k_Z], 0), lambda_hp=0)

theta_estimate['theta']['tt']/theta_estimate['theta']['c']
theta_true['Eastern-Massachusetts']['tt']/theta_true['Eastern-Massachusetts']['c']

# Austin




# berlin-tiergarten_net.tntp


tai.writer.write_network_to_dat(root =  root_github
                                , subfolder = 'Eastern-Massachusetts' , prefix_filename = 'EMA', N = N['train']['Eastern-Massachusetts'])

x,tt_x = tai.equilibrium.sue_logit_dial(root = root_github, subfolder ='Eastern-Massachusetts', prefix_filename ='EMA', maxIter = 100, accuracy = config.estimation_options['accuracy_eq'], Z_dict = N['train']['Eastern-Massachusetts'].Z_dict, theta = theta_true['Eastern-Massachusetts'], k_Z = k_Z)

N['train']['Eastern-Massachusetts'].x_dict = x
N['train']['Eastern-Massachusetts'].set_Y_attr_links(y=tt_x, label='tt')

theta_estimate = tai.estimation.solve_link_level_model(end_params={'theta': True, 'q': False},
                                                       Mt={1: N['train']['Eastern-Massachusetts'].M}, Ct={
        i: tai.estimation.choice_set_matrix_from_M(N['train']['Eastern-Massachusetts'].M)},
                                                       Dt={1: N['train']['Eastern-Massachusetts'].D}, k_Y=k_Y,
                                                       Yt={1: N['train']['Eastern-Massachusetts'].Y_dict}, k_Z=k_Z,
                                                       Zt={1: N['train']['Eastern-Massachusetts'].Z_dict},
                                                       xt={1: N['train']['Eastern-Massachusetts'].x}, idx_links={
        1: range(0, len(N['train']['Eastern-Massachusetts'].x))}, scale={'mean': False, 'std': False},
                                                       q0=tai.networks.denseQ(Q=N['train']['Eastern-Massachusetts'].Q,
                                                                              remove_zeros=remove_zeros_Q),
                                                       theta0=dict.fromkeys([*k_Y, *k_Z], 0), lambda_hp=0)



od_filename = [_ for _ in os.listdir(os.path.join(root_github, subfolder_github)) if 'trips' in _ and _.endswith('tntp')]
prefix_filename = od_filename[0].partition('_')[0]

tai.equilibrium.sue_logit_dial(root = root_github, subfolder = subfolder_github, prefix_filename = prefix_filename, maxIter = 100, accuracy = config.estimation_options['accuracy_eq'], theta = {'tt':1})


i = 'N5'
i = 'N6'

N['train']['N6'].Q

N['train']['N6'].Z_dict

while valid_network is None:
    try:
        results_sue['train'] = {i: tai.equilibrium.sue_logit_fisk(q = tai.networks.denseQ(Q = N['train'][i].Q, remove_zeros = remove_zeros_Q)
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
            tai.networks.clone_network(N=N['train'][i], label='Train', randomness = {'Q':True, 'BPR':True, 'Z': False, 'var_Q':0}
                                       )

        # tai.network.clone_networks(N=N['train'], label='Train'
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
            N['train'][i].set_Y_attr_links(y=results_sue['train'][i]['tt_x'], label='tt')
            N['train'][i].x_dict = results_sue['train'][i]['x']
            N['train'][i].f_dict = results_sue['train'][i]['f']


results_sue['train']['Braess-Example'] #ECOS fails because an overflow of the link that has free flow travel time of 2e10.




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
# theta0 = {i: 0 for i in [*k_Y, *k_Z]}
# theta0 = {i:1 for i in [*k_Y, *k_Z]}
# theta0 = {i:-1 for i in [*k_Y, *k_Z]}
# theta0 = {i:theta_true[i] for i in [*k_Y, *k_Z]}
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
        # k_Z = ['wt','c']
        k_Z = ['wt','c'] #['c'] #
        # k_Z = list(N[i].Z_dict.keys())  #

        # Starting values
        q0[i] = tai.networks.denseQ(Q=N[i].Q, remove_zeros=remove_zeros_Q)
        theta0[i] = {k:theta_true[i][k] for k in [*k_Y,*k_Z]}

        if end_params['q']:
            q0[i] = np.zeros(tai.networks.denseQ(Q=N[i].Q, remove_zeros=remove_zeros_Q).shape)

        if end_params['theta']:
            for j in [*k_Y,*k_Z]:
                theta0[i][j] = 0
            # theta0 = {j:0  for i in N.keys()}
            # theta0 = {i: 0 for i in [*k_Y, *k_Z]}

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

            results_multidays_temp['no_disturbance_Q'], N_multiday_old = tai.estimation.multiday_estimation(N = {i:copy.deepcopy(N[i])}, N_multiday_old = {i:copy.deepcopy(N_multiday_old[i])}
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
            # results_multidays_temp['disturbance_Q'] = tai.logit.multiday_estimation(N={i: copy.deepcopy(N[i])}
            #                                                                    , end_params=end_params, n_days=n_day
            #                                                                    , k_Y=k_Y, k_Z=k_Z
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
# np.linalg.norm(results_multidays_analyses['no_disturbance_Q']['end_q']['q']['SiouxFalls'][1]-{i: tai.network.denseQ(Q= N_i.Q, remove_zeros=remove_zeros_Q) for i,N_i in N['train'].items()}['SiouxFalls'],2)


# #Experimental
# results_multidays_analyses['no_disturbance_Q']['end_theta'] = multiday_estimation_analyses(end_params = {'theta': True, 'q': False}, n_days_seq = np.repeat(20, 4), N = {'N4': N['train']['N4']}, var_Q = 0, theta_true = theta_true, R_labels=R_labels, q_range= q_range, remove_zeros_Q=remove_zeros_Q, Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes,  fixed_effects= fixed_effects, cutoff_paths=cutoff_paths)
#
# results_multidays_analyses['no_disturbance_Q']['end_theta']['theta']['N4'][80]
# theta_true['N4']
# results_multidays_analyses['no_disturbance_Q']['end_theta']['vot']['N4']
# results_multidays_analyses['no_disturbance_Q']['end_q']['q']['N5']
# {i: tai.network.denseQ(Q= N_i.Q, remove_zeros=remove_zeros_Q) for i,N_i in N['train'].items()}['N5']

# N['train']['N5'].A
# results_multidays_analyses['no_disturbance_Q']['end_theta']['theta']['N2'][40]
# results_multidays_analyses['no_disturbance_Q']['end_theta']['vot']['N2'][40]
# theta_true['N2']

# print(results_multidays['train']['N4'][n_days_seq[-1]])
# print(['N4'])
# results_multidays['q']['N1'][n_days]
# {i: tai.network.denseQ(Q=N_i.Q, remove_zeros=remove_zeros_Q) for i, N_i in N['train'].items()}['N5']

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
plot.q_multidays_consistency(q_true = {i: tai.networks.denseQ(Q= N_i.Q, remove_zeros=remove_zeros_Q) for i, N_i in N['train'].items()}
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
#                              , subfolder = 'link-level'
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
plot.q_multidays_consistency(q_true = {i: tai.networks.denseQ(Q= N_i.Q, remove_zeros=remove_zeros_Q) for i, N_i in N['train'].items()}
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
plot.q_multidays_consistency(q_true = {i: tai.networks.denseQ(Q= N_i.Q, remove_zeros=remove_zeros_Q) for i, N_i in N['train'].items()}
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
#                              , subfolder = 'link-level'
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
# k_Z = list(N['train']['N1'].Z_dict.keys()) # #['wt','c'] #

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
    # k_Z = ['c','wt'] #
    k_Z = list(N['train'][N_i].Z_dict.keys())

    for i in n_links:

        for j in range(0,n_bootstraps):

            if i <= len(N['train'][N_i].x) or not estimation_done:

                # Select a random sample of links provided the number of links is smaller than the total number of links
                idx_links = range(0, len(N['train'][N_i].x))

                if i < len(N['train'][N_i].x):
                    idx_links = random.sample(range(0, len(N['train'][N_i].x)), i)

                theta_estimate = tai.estimation.solve_link_level_model(end_params={'theta': True, 'q': False},
                                                                       Mt={1: N['train'][N_i].M}, Ct={
                        i: tai.estimation.choice_set_matrix_from_M(N['train'][N_i].M)}, Dt={1: N['train'][N_i].D},
                                                                       k_Y=k_Y, Yt={1: N['train'][N_i].Y_dict}, k_Z=k_Z,
                                                                       Zt={1: N['train'][N_i].Z_dict},
                                                                       xt={1: N['train'][N_i].x},
                                                                       idx_links={1: idx_links},
                                                                       scale={'mean': False, 'std': False},
                                                                       q0=tai.networks.denseQ(Q=N['train'][N_i].Q,
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
#                                      , subfolder= "link-level/training"
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
#                                      , subfolder= "link-level/training"
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
    tai.networks.clone_network(N=N['train'], label='Validation'
                               , randomness={'Q': True, 'BPR': False, 'Z': False}
                               , Z_attrs_classes=None, bpr_classes=None)


valid_network = None

#Compute SUE in clone networks
while valid_network is None:
    try:
        results_sue['validation'] = {i: tai.equilibrium.sue_logit_fisk(q = tai.networks.denseQ(Q = N['validation'][i].Q, remove_zeros = remove_zeros_Q)
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

        N['validation'] = tai.networks.clone_network(N=N['train'], label='Validation'
                                                     , R_labels=R_labels
                                                     , randomness={'Q': True, 'BPR': False, 'Z': False}
                                                     , Z_attrs_classes=None, bpr_classes=None)

        pass
    else:
        valid_network = True

        # Store travel time, link and path flows in Network objects
        for i in N['validation'].keys():
            N['validation'][i].set_Y_attr_links(y=results_sue['validation'][i]['tt_x'], label='tt')
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
        theta_estimates['train'][i][lambda_i] = tai.estimation.solve_link_level_model(
            Mt= {i:N['train'][i].M}
            , Ct={i: tai.estimation.choice_set_matrix_from_M(N['train'][i].M)}
            , Dt= {i:N['train'][i].D}
            , q0= {i:tai.networks.denseQ(Q=N['train'][i].Q, remove_zeros=True)}
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

        errors_logit['train'][i][lambda_i] = tai.estimation.loss_link_level_model(
            theta=np.array(list(theta_estimates['train'][i][lambda_i].values()))
            , lambda_hp=0
            , M= {i:N['train'][i].M}
            , C={i: tai.estimation.choice_set_matrix_from_M(N['train'][i].M)}
            , D= {i:N['train'][i].D}
            , Y= {i:(tai.estimation.get_matrix_from_dict_attrs_values({k_y: N['train'][i].Y_dict[k_y] for k_y in ['tt']}).T @ N['train'][i].D).T}
            , Z= {i:(tai.estimation.get_matrix_from_dict_attrs_values({k_y: N['train'][i].Z_dict[k_y] for k_y in list(N['train'][i].Z_dict.keys())}).T @ N['train'][i].D).T}
            , q= {i:tai.networks.denseQ(Q=N['train'][i].Q, remove_zeros=True)}
            , x= {i:N['train'][i].x}
            , idx_links={i:train_idx_links}
            , norm_o=2, norm_r=1)


        errors_logit['validation'][i][lambda_i] = tai.estimation.loss_link_level_model(
            theta = np.array(list(theta_estimates['train'][i][lambda_i].values()))
            , lambda_hp= 0
            , M = {i:N['validation'][i].M}
            , C= {i: tai.estimation.choice_set_matrix_from_M(N['validation'][i].M)}
            , D = {i:N['validation'][i].D}
            , Y= {i:(tai.estimation.get_matrix_from_dict_attrs_values(
                {k_y: N['validation'][i].Y_dict[k_y] for k_y in ['tt']}).T @ N['validation'][i].D).T}
            , Z= {i:(tai.estimation.get_matrix_from_dict_attrs_values(
                {k_y: N['validation'][i].Z_dict[k_y] for k_y in list(N['validation'][i].Z_dict.keys())}).T @
                     N['validation'][i].D).T}
            , q = {i:tai.networks.denseQ(Q=N['validation'][i].Q, remove_zeros=True)}
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
# self = tai.Plot(folder_plots = folder_plots, dim_subplots=dim_subplots)
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
        errors_SUE_logit['train'][i][j] = tai.estimation.loss_SUE(o = 2, x_obs = N['train'][i].x
                                                                  , q = tai.networks.denseQ(Q = N['train'][i].Q, remove_zeros = remove_zeros_Q)
                                                                  , M = N['train'][i].M, D = N['train'][i].D
                                                                  , links = N['train'][i].links_dict
                                                                  , paths = N['train'][i].paths
                                                                  , Z_dict = N['train'][i].Z_dict
                                                                  , k_Z = []
                                                                  , theta = theta_estimates['train'][i][j]
                                                                  , cp_solver = 'ECOS')
        errors_SUE_logit['validation'][i][j] = tai.estimation.loss_SUE(o = 2, x_obs = N['validation'][i].x
                                                                       , q = tai.networks.denseQ(Q = N['validation'][i].Q, remove_zeros = remove_zeros_Q)
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

        N['test'][i][j] = tai.networks.clone_network(N={i: N['train'][i]}, label='Test'
                                                     , R_labels=R_labels
                                                     , randomness={'Q': True, 'BPR': False, 'Z': False}
                                                     , q_range=(2, 10), remove_zeros_Q=remove_zeros_Q
                                                     , Z_attrs_classes=Z_attrs_classes, bpr_classes=bpr_classes,
                                                     cutoff_paths=cutoff_paths).get(i)

        valid_network = None

        while valid_network is None:
            try:

                results_sue['test'] = {i: tai.equilibrium.sue_logit_fisk(
                    q=tai.networks.denseQ(Q=N['test'][i][j].Q, remove_zeros=remove_zeros_Q)
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
                N['test'][i][j] = tai.networks.clone_network(N={i: N['train'][i]}, label='Test'
                                                             , R_labels=R_labels
                                                             , randomness={'Q': True, 'BPR': False, 'Z': False}).get(i)
            else:
                valid_network = True

                for i in N['test'].keys():
                    N['test'][i][j].set_Y_attr_links(y=results_sue['test'][i]['tt_x'], label='tt')
                    N['test'][i][j].x_dict = results_sue['test'][i]['x']
                    N['test'][i][j].f_dict = results_sue['test'][i]['f']


        test_error['noreg'][i][j] = np.round(np.sqrt(tai.estimation.loss_link_level_model(
            theta=np.array(list(validation_thetas['noreg'][i].values()))
            , lambda_hp=0
            , M= {i:N['test'][i][j].M}
            , C={i: tai.estimation.choice_set_matrix_from_M(N['test'][i].M)}
            , D= {i:N['test'][i][j].D}
            , Y= {i:(tai.estimation.get_matrix_from_dict_attrs_values(
                {k_y: N['test'][i][j].Y_dict[k_y] for k_y in ['tt']}).T @ N['test'][i][j].D).T}
            , Z= {i:(tai.estimation.get_matrix_from_dict_attrs_values(
                {k_z: N['test'][i][j].Z_dict[k_z] for k_z in
                 list(N['test'][i][j].Z_dict.keys())}).T @
                     N['test'][i][j].D).T}
            , q= {i:tai.networks.denseQ(Q=N['test'][i][j].Q, remove_zeros=True)}
            , x= {i:N['test'][i][j].x}
            , idx_links= {i:range(0, len(N['test'][i][j].x))}
            , norm_o=2, norm_r=1)), 4)

        test_error['reg'][i][j] \
            = np.round(np.sqrt(
            tai.estimation.loss_link_level_model(
                theta=np.array(list(validation_thetas['reg'][i].values()))
                , M = {i:N['test'][i][j].M}
                , C= {i: tai.estimation.choice_set_matrix_from_M(N['test'][i].M)}
                , D = {i:N['test'][i][j].D}
                , Y= {i:(tai.estimation.get_matrix_from_dict_attrs_values(
                    {k_y: N['test'][i][j].Y_dict[k_y] for k_y in ['tt']}).T @ N['test'][i][j].D).T}
                , Z= {i:(tai.estimation.get_matrix_from_dict_attrs_values(
                    {k_z: N['test'][i][j].Z_dict[k_z] for k_z in
                     list(N['test'][i][j].Z_dict.keys())}).T @
                         N['test'][i][j].D).T}
                , q = {i:tai.networks.denseQ(Q=N['test'][i][j].Q, remove_zeros=True)}
                , x = {i:N['test'][i][j].x}
                , idx_links= {i:range(0, len(N['test'][i][j].x))}
                , lambda_hp=0
                , norm_o=2, norm_r=1))
            , 4)

test_error_plot = {}

# self = tai.Plot(folder_plots = folder_plots, dim_subplots=dim_subplots)

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
#results_sue['test-estimated'] = {i: tai.equilibrium.sue_logit(q=tai.denseQ(Q=N_i.Q, remove_zeros=remove_zeros_Q)
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
likelihood_logit['train'] = {i: tai.estimation.likelihood_path_level_logit(f = N['train'][i].f
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
results_logit['train'] = {i: tai.estimation.solve_path_level_logit(cp_ll = likelihood_logit['train'][i]['cp_ll']
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
results_logit['train'] = {i: tai.estimation.solve_path_level_logit(cp_ll = likelihood_logit['train'][i]['cp_ll']
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
    = tai.estimation.prediction_error_logit_regularization(
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
    tai.networks.clone_network(N=N['train'], label='Validation'
                               , R_labels=R_labels
                               , randomness={'Q': True, 'BPR': False, 'Z': False}
                               , Z_attrs_classes=None, bpr_classes=None)


valid_network = None

while valid_network is None:
    try:
        results_sue['validation'] = {i: tai.equilibrium.sue_logit_fisk(
            q = tai.networks.denseQ(Q = N['validation'][i].Q, remove_zeros = remove_zeros_Q)
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

        N['validation'] = tai.networks.clone_network(N=N['train'], label='Validation'
                                                     , R_labels=R_labels
                                                     , randomness={'Q': True, 'BPR': False, 'Z': False}
                                                     , Z_attrs_classes=None, bpr_classes=None)

        pass
    else:
        valid_network = True

        # Store travel time, link and path flows in Network objects
        for i in N['validation'].keys():
            N['validation'][i].set_Y_attr_links(y=results_sue['validation'][i]['tt_x'], label='tt')
            N['validation'][i].x_dict = results_sue['validation'][i]['x']
            N['validation'][i].f_dict = results_sue['validation'][i]['f']


# Get likelihood objects from logit model
likelihood_logit['validation'] = {i: tai.estimation.likelihood_path_level_logit(f=results_sue['validation'][i]['f']
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
results_logit['validation'] = {i: tai.estimation.solve_path_level_logit(cp_ll=likelihood_logit['validation'][i]['cp_ll']
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
    = tai.estimation.prediction_error_logit_regularization(theta_estimates = theta_estimates['train']
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
        errors_SUE_logit['train'][i][j] = tai.estimation.loss_SUE(o = 2, x_obs = N['train'][i].x
                                                                  , q = tai.networks.denseQ(Q = N['train'][i].Q, remove_zeros = remove_zeros_Q)
                                                                  , M = N['train'][i].M, D = N['train'][i].D
                                                                  , links = N['train'][i].links_dict
                                                                  , paths = N['train'][i].paths
                                                                  , Z_dict = N['train'][i].Z_dict
                                                                  , k_Z = []
                                                                  , theta = theta_estimates['train'][i][j]
                                                                  , cp_solver = 'ECOS')
        errors_SUE_logit['validation'][i][j] = tai.estimation.loss_SUE(o = 2, x_obs = N['validation'][i].x
                                                                       , q = tai.networks.denseQ(Q = N['validation'][i].Q, remove_zeros = remove_zeros_Q)
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
        result = tai.simulation.sue_logit_simulation_recovery(N=N['train'][i]
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

        result_sue = tai.equilibrium.sue_logit_fisk(q= tai.networks.denseQ(Q=N['train'][i].Q, remove_zeros=remove_zeros_Q)
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

# tai.writer.write_network_to_dat(root =  root_pablo
#                                 , subfolder = "Custom3" , prefix_filename = 'custom3', N = N['train']['N3'])
#
# tai.equilibrium.sue_logit_dial(root = root_pablo, subfolder = 'Custom3', prefix_filename = 'custom3', maxIter = 100, accuracy = 0.01, theta = {'tt':1})
# #
# tai.writer.write_network_to_dat(root =  root_pablo
#                                 , subfolder = "Custom4" , prefix_filename = 'custom4', N = N['train']['N4'])
#
# tai.equilibrium.sue_logit_dial(root = root_pablo, subfolder ='Custom4', prefix_filename ='custom4', maxIter = 100, accuracy = 0.01, theta = {'tt':1})

# tai.writer.write_network_to_dat(root =  root_pablo
#                                 , subfolder = "Random5" , prefix_filename = 'random5', N = N['train']['N5'])
#
# tai.equilibrium.sue_logit_dial(root = root_pablo, subfolder ='Random5', prefix_filename ='random5', maxIter = 100, accuracy = 0.01, Z_dict = N['train']['N5'].Z_dict, theta = theta_true['N5'], k_Z = ['wt','c'])
#
# tai.writer.write_network_to_dat(root =  root_pablo
#                                 , subfolder = "Random6" , prefix_filename = 'random6', N = N['train']['N6'])
#
# results_sue_dial = {}
#
# results_sue_dial['x'],results_sue_dial['tt_x'] = tai.equilibrium.sue_logit_dial(root = root_pablo, subfolder ='Random6', prefix_filename ='random6'
#                                , options = {'equilibrium': 'stochastic', 'method': 'MSA', 'maxIter': 100, 'accuracy_eq': 0.01}
#                                , Z_dict = N['train']['N6'].Z_dict, theta = theta_true['N6'], k_Z = ['wt','c'])
#
# N['train']['N6'].x_dict = results_sue_dial['x']
# N['train']['N6'].set_Y_attr_links(y=results_sue_dial['tt_x'], label='tt')

# maxIter = 20
# results_sue_msa = {}

# for i in subfolders_tntp_networks:
#     # To get estimate of the logit parameters is necessary sometimes to increase the number of iterations so higher accuracy is achieved
#     # 200 works great for networks with less than 1000 links but more iterations are needed for larger networks and this increase computing time significantly
#     t0 = time.time()
#     results_sue_msa[i] = tai.equilibrium.sue_logit_msa_k_paths(N = N['train'][i], maxIter = maxIter, accuracy = 0.01, theta = theta_true[i])
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
# results_sue_msa[i] = tai.equilibrium.sue_logit_msa_k_paths(N=N['train'][i], theta=theta_true[i], k_Y=k_Y, k_Z=k_Z,
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





def main():
    import runpy
    runpy.run_path(os.getcwd() + '/examples/local/production/od-theta-example.py')


