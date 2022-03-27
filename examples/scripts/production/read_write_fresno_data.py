# =============================================================================
# 1) SETUP
# =============================================================================

# External modules
import ast
import sys
import pandas as pd

# Internal modules
import transportAI as tai

# =============================================================================
# 2) NETWORK FACTORY
# ============================================================================
network_name = 'Fresno'

# =============================================================================
# a) READ FRESNO LINK DATA
# =============================================================================

# Reader of geospatial and spatio-temporal data
data_reader = tai.etl.DataReader(network_key=network_name,setup_spark=True)

# Read files
links_df, nodes_df = tai.reader.read_fresno_network(folderpath=tai.dirs['Fresno_network'])

nodes_df.to_csv(tai.dirs['output_folder'] + '/network-data/nodes/'  + 'fresno-nodes-data.csv',
                sep=',', encoding='utf-8', index=False, float_format='%.3f')

# Add link key in dataframe
links_df['link_key'] = [(int(i), int(j), '0') for i, j in zip(links_df['init_node_key'], links_df['term_node_key'])]

# =============================================================================
# a) BUILD NETWORK
# =============================================================================

# Create Network Generator
network_generator = tai.factory.NetworkGenerator()

A = network_generator.generate_adjacency_matrix(links_keys=list(links_df.link_key.values))

fresno_network = \
    network_generator.build_fresno_network(A=A, links_df=links_df, nodes_df=nodes_df, network_name= network_name)

# =============================================================================
# f) OD
# =============================================================================

# - Periods (6 periods of 15 minutes each)
data_reader.options['od_periods'] = [1, 2, 3, 4]

# Read OD from raw data
Q = tai.reader.read_fresno_dynamic_od(network=fresno_network,
                                  filepath=tai.dirs['Fresno_network'] + '/SR41.dmd',
                                  periods=data_reader.options['od_periods'])

network_generator.write_OD_matrix(network = fresno_network, sparse = True, overwrite_input=False)

# =============================================================================
# g) PATHS
# =============================================================================

# Create path generator
paths_generator = tai.factory.PathsGenerator()

# Generate and Load paths in network
paths_generator.load_k_shortest_paths(network = fresno_network, k=4)
#
# Write paths and incident matrices
paths_generator.write_paths(network=fresno_network, overwrite_input=False)

network_generator.write_incidence_matrices(network = fresno_network,
                                           matrices = {'sparse_C':True, 'sparse_D':True, 'sparse_M':True},
                                           overwrite_input = False)

paths_generator.read_paths(network=fresno_network, update_incidence_matrices=True)

# =============================================================================
# c) LINK FEATURES FROM NETWORK FILE
# =============================================================================

# Extract data on link features from network file
link_features_df = links_df[['link_key', 'id', 'link_type', 'rhoj', 'lane', 'ff_speed', 'length']]

# Attributes
link_features_df['link_type'] = link_features_df['link_type'].apply(lambda x: x.strip())
link_features_df['rhoj'] = pd.to_numeric(link_features_df['rhoj'], errors='coerce', downcast='float')
link_features_df['lane'] = pd.to_numeric(link_features_df['lane'], errors='coerce', downcast='integer')
link_features_df['length'] = pd.to_numeric(link_features_df['length'], errors='coerce', downcast='float')

# Load features data
fresno_network.load_features_data(linkdata=link_features_df, link_key = 'link_key')

# =============================================================================
# d) LINK PERFORMANCE FUNCTIONS
# =============================================================================

options = {'tt_units': 'minutes'}

# Create two new features
if options['tt_units'] == 'minutes':
    # Weighting by 60 will leave travel time with minutes units, because speeds are originally in per hour units
    tt_factor = 60

if options['tt_units'] == 'seconds':
    tt_factor = 60 * 60

links_df['ff_speed'] = pd.to_numeric(links_df['ff_speed'], errors='coerce', downcast='float')
links_df['ff_traveltime'] = tt_factor * links_df['length'] / links_df['ff_speed']

bpr_parameters_df = pd.DataFrame({'link_key': links_df['link_key'],
                                  'alpha': 0.15,
                                  'beta': 4,
                                  'tf': links_df['ff_traveltime'],
                                  'k': pd.to_numeric(links_df['capacity'], errors='coerce', downcast='float')
                                  })

fresno_network.set_bpr_functions(bprdata=bpr_parameters_df, link_key = 'link_key')

# =============================================================================
# d) SPATIO-TEMPORAL LINK FEATURES AND TRAFFIC COUNTS
# =============================================================================

dates = ['2019-10-01','2020-10-06']

options['update_ff_tt_inrix'] = True

for date in dates:

    # First Tuesday of October, 2019 (2019-10-01)
    data_reader.select_period(date=date, hour=16)

    # First Tuesday of October, 2020 (2020-10-06)
    data_reader.select_period(date=date, hour=16)

    # =============================================================================
    # SPATIO-TEMPORAL LINK FEATURES
    # =============================================================================

    filepath = tai.dirs['output_folder'] + '/network-data/links/' + str(data_reader.options['selected_date']) \
               + '-fresno-spatiotemporal-link-data.csv'

    spatiotemporal_features_df, spatiotemporal_features_list = data_reader.read_spatiotemporal_data_fresno(
            lwrlk_only=False,
            network=fresno_network,
            selected_period_incidents={'year': [data_reader.options['selected_year']],
                                       'month': [7, 8, 9, 10]},
            data_processing={'inrix_segments': True, 'inrix_data': True, 'census': True, 'incidents': True,
                             'bus_stops': True, 'streets_intersections': True},
            # data_processing={'inrix_segments': False, 'inrix_data': False, 'census': False, 'incidents': False,
            #                  'bus_stops': False, 'streets_intersections': False},
            inrix_matching={'census': False, 'incidents': True, 'bus_stops': True, 'streets_intersections': True},
            buffer_size={'inrix': 200, 'bus_stops': 50, 'incidents': 50, 'streets_intersections': 50},
            tt_units='minutes'
        )

    spatiotemporal_features_df.to_csv(filepath, sep=',', encoding='utf-8', index=False, float_format='%.3f')

    # Test Reader
    spatiotemporal_features_df = pd.read_csv(filepath)

    fresno_network.load_features_data(spatiotemporal_features_df)

    # =============================================================================
    # d) FREE FLOW TRAVEL TIME OF LINK PERFORMANCE FUNCTIONS
    # =============================================================================

    # Create two new features
    if options['tt_units'] == 'minutes':
        # Weighting by 60 will leave travel time with minutes units, because speeds are originally in per hour units
        tt_factor = 60

    if options['tt_units'] == 'seconds':
        tt_factor = 60 * 60

    if options['update_ff_tt_inrix']:
        for link in fresno_network.links:
            if link.link_type == 'LWRLK' and link.Z_dict['speed_ref_avg']!=0:
                # Multiplied by 60 so speeds are in minutes
                link.bpr.tf = tt_factor * link.Z_dict['length'] / link.Z_dict['speed_max']
                # link.bpr.tf = tt_factor * link.Z_dict['length'] / link.Z_dict['speed_ref_avg']
                # else:
                #     link.bpr.tf = links_df[links_df['link_key'].astype(str) == str(link.key)]['ff_traveltime']

        fresno_network.set_bpr_functions(bprdata=bpr_parameters_df, link_key = 'link_key')

    # =============================================================================
    # 3c) DATA CURATION
    # =============================================================================

    # a) Imputation to correct for outliers and observations with zero values because no GIS matching
    features_list = ['median_inc', 'intersections', 'incidents', 'bus_stops', 'median_age',
                     'tt_avg', 'tt_sd','tt_var', 'tt_cv',
                     'speed_ref_avg', 'speed_avg','speed_sd','speed_cv']

    for feature in features_list:
        fresno_network.link_data.feature_imputation(feature =feature, pcts = (2, 98))

    # b) Feature values in "connectors" links
    for key in features_list:
        for link in fresno_network.get_non_regular_links():
            link.Z_dict[key] = 0
    print('Features values of link with types different than "LWRLK" were set to 0')

    # a) Capacity adjustment

    # counts = tai.etl.adjust_counts_by_link_capacity(network = fresno_network, counts = counts)

    # b) Outliers

    # tai.etl.remove_outliers_fresno(fresno_network)

    # =============================================================================
    # 2.2) TRAFFIC COUNTS
    # =============================================================================

    # ii) Read data from PEMS count and perform matching GIS operations to combine station shapefiles

    date_pathname = data_reader.options['selected_date'].replace('-', '_')

    path_pems_counts = tai.dirs['input_folder'] + 'public/pems/counts/data/' + \
                       'd06_text_station_5min_' + date_pathname + '.txt.gz'

    # Load pems station ids in links
    tai.etl.load_pems_stations_ids(network=fresno_network)

    # Read and match count data from a given period

    # Duration is set at 2 because the simulation time for the OD matrix was set at that value
    count_interval_df \
        = data_reader.read_pems_counts_by_period(
        filepath=path_pems_counts,
        selected_period={'hour': data_reader.options['selected_hour'],
                         'duration': int(len(data_reader.options['od_periods']) * 15)})

    # Generate a masked vector that fill out count values with no observations with nan
    counts = tai.etl.generate_fresno_pems_counts(links=fresno_network.links
                                                 , data=count_interval_df
                                                 # , flow_attribute='flow_total'
                                                 # , flow_attribute = 'flow_total_lane_1')
                                                 , flow_attribute='flow_total_lane'
                                                 , flow_factor=1  # 0.1
                                                 )
    # Write counts in csv

    filepath = tai.dirs['output_folder'] + 'network-data/links/' + str(data_reader.options['selected_date']) \
               + '-fresno-link-counts.csv'

    counts_df = pd.DataFrame({'link_key': counts.keys(),
                              'counts': counts.values(),
                              'pems_ids': [link.pems_stations_ids for link in fresno_network.links]})
    counts_df.to_csv(filepath, sep=',', encoding='utf-8', index=False, float_format='%.3f')

    # Read counts from csv
    counts_df = pd.read_csv(filepath, converters={"link_key": ast.literal_eval})

    counts = dict(zip(counts_df['link_key'].values, counts_df['counts'].values))

    # Load counts
    fresno_network.load_traffic_counts(counts=counts)

    # =============================================================================
    # c) WRITE LINK FEATURES AND COUNTS
    # =============================================================================
    summary_table_links_df = tai.descriptive_statistics.summary_table_links(links=fresno_network.links)

    summary_table_links_df.to_csv(tai.dirs['output_folder'] + 'network-data/links/'
                     + str(data_reader.options['selected_date'])+ '-fresno-link-data.csv',
                     sep=',', encoding='utf-8', index=False, float_format='%.3f')

# =============================================================================
# 3d) DESCRIPTIVE STATISTICS
# =============================================================================

sys.exit()

#TODO: No working

# Scatter plot of Continous and categorical features
continuous_features = ['counts', 'capacity [veh]', 'speed_ff[mi/hr', 'tt_ff [min]', 'income [1K USD]',
                       'speed_avg [mi/hr]', 'tt_sd', 'tt_cv', 'tt_reliability', 'incidents']
categorical_features = ['high_inc', 'stops', 'ints']

existing_continous_features = set(summary_table_links_df.keys()).intersection(set(continuous_features))

summary_table_links_scatter_df = summary_table_links_df[existing_continous_features]

scatter_fig1, scatter_fig2 = \
    tai.descriptive_statistics.scatter_plots_features_vs_counts(links_df=summary_table_links_scatter_df)

scatter_fig1.savefig(reporter.dirs['estimation_folder'] + '/' + 'scatter_plot1.pdf',
            pad_inches=0.1, bbox_inches="tight")

scatter_fig1.savefig(reporter.dirs['estimation_folder'] + '/' + 'scatter_plot2.pdf',
            pad_inches=0.1, bbox_inches="tight")

# TODO: visualize box plot for the non continuous features
# data_reader.read_pems_counts_by_period(filepath=path_pems_counts
#                                                   , selected_period = estimation_options['selected_period_pems_counts'])



# TODO: PEMS station ids are not being read from csv file but are added when the files are read on execution

read_filepath_pems_counts = tai.dirs['input_folder'] + '/network-data/links/' \
                            + str(data_reader.options['selected_date']) + '-fresno-link-counts' + '.csv'

selected_links_ids_pems_statistics = [link.pems_stations_ids[0] for link in fresno_network.get_observed_links() if
                                      len(link.pems_stations_ids) == 1]

selected_links_ids_pems_statistics = list(np.random.choice(selected_links_ids_pems_statistics, 4, replace=False))

distribution_pems_counts_figure = tai.descriptive_statistics.distribution_pems_counts(
    filepath=read_filepath_pems_counts,
    data_reader=data_reader,
    selected_period={'year': data_reader.options['selected_year'],
                     'month': data_reader.options['selected_month'],
                     'day_month': data_reader.options['selected_day_month'],
                     'hour': 6, 'duration': 900},
    selected_links=selected_links_ids_pems_statistics
)

# plt.show()

tai.writer.write_figure_to_log_folder(fig=distribution_pems_counts_figure, filename='distribution_pems_counts.pdf')





