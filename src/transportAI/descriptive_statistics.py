""" Descriptive statistics in terms of tables or plots"""

import pandas as pd


import numpy as np

import analyst
import writer
import matplotlib.pyplot as plt
import matplotlib
# plt.rcParams['figure.dpi'] = 30
# plt.rcParams['savefig.dpi'] = 30

import seaborn as sns
# sns.set(rc={"figure.dpi":30, 'savefig.dpi':30})



import copy
import datetime
# matplotlib.rcParams['text.usetex'] = False


def distribution_pems_counts(filepath, selected_period, selected_links = None, col_wrap = 2):
    """ Analyze counts from a selected group of links. We may compare counts in 2019 and 2020 and over the same day of the week during each month to analyze the validity of Gaussian distribution and see differences between years """


    # Selection of an arbitrary of links to show the distribution of traffic counts via selected_links

    # TODO: Add vertical bar at the point with maximum traffic flow

    # Avoid clash with underscore in feature names, e.g. flow_total
    matplotlib.rcParams['text.usetex'] = False

    data_analyst = analyst.Analyst()

    pems_count_sdf = data_analyst.read_pems_counts_data(filepath, selected_period)
    pems_count_df = pems_count_sdf.toPandas()

    # selected_period = {'year': 2019, 'hour': 6, 'duration': 900}

    # pems_count_df = pems_count_df[pems_count_df.station_id == 601346]

    pems_count_subset_df = copy.deepcopy(pems_count_df[pems_count_df.station_id.isin(selected_links)])

    # [datetime.datetime.time(d) for d in pd.to_datetime(pems_count_subset_df['ts'])]

    pems_count_subset_df['ts'] = pd.to_datetime(pems_count_subset_df['ts'])
    # pems_count_subset_df['hour'] = pems_count_subset_df['ts'].dt.time

    pems_count_subset_df.index = pems_count_subset_df['ts']

    # pems_count_df.plot(figsize=(8, 4))
    # pems_count_df.plot(subplots=True, figsize=(15, 6))
    # pems_count_df.plot(y=["R", "F10.7"], figsize=(15, 4))

    # ax = pems_count_df[['flow_total']].resample("30min").median().plot(figsize=(8, 4))
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # fig, ax = plt.subplots(figsize=(8, 4))

    pems_count_resample_df = pems_count_subset_df.groupby('station_id').resample("30min")['flow_total'].sum()
    # # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    pems_count_resample_df = pems_count_resample_df.reset_index()
    pems_count_resample_df['time'] = pems_count_resample_df['ts'].dt.time.astype(str)

    # sns.set(font_scale=1.5, rc={'text.usetex': False})

    #
    fg = sns.FacetGrid(pems_count_resample_df, col='station_id', col_wrap=col_wrap)
    fg.map(sns.lineplot, 'time', 'flow_total')

    # for ax in fg.axes.flatten():
    #     ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    fg.set_xticklabels(rotation=90)

    fg.set(xticks=pems_count_resample_df.time[0::2])

    # sns.lineplot(x = 'ts', y = 'flow_total',data = pems_count_resample_df)

    # xtix = ax.get_xticks()
    # ax.set_xticks(xtix[::freq])

    # ax = pems_count_df.plot(x='ts', y=['flow_total'], style='.')

    fg.fig.tight_layout()

    # plt.show()  #

    # plt.savefig('test_figure')

    return fg


def get_link_Z_attributes_df(links, attrs_list, Z_labels):

    df_dict = {}

    # df_dict['key'] = [link.key for link in links]

    missing_attrs = []

    if Z_labels is None:
        Z_labels = attrs_list

    for attr,z_label in zip(attrs_list,Z_labels):

        if attr in links[0].Z_dict:
            df_dict[z_label] = [link.Z_dict[attr] for link in links]

        else:
            missing_attrs.append(attr)

    links_df = pd.DataFrame(df_dict)

    if len(missing_attrs) > 0:
        print('Missing attributes:', missing_attrs)

    return links_df


def get_loss_and_estimates_over_iterations(results_norefined: pd.DataFrame, results_refined: pd.DataFrame):

    results_norefined_bilevelopt = results_norefined
    results_refined_bilevelopt = results_refined

    refined_bilevel_iters = len(list(results_refined_bilevelopt.keys()))
    norefined_bilevel_iters = len(list(results_norefined_bilevelopt.keys()))

    theta_keys = results_norefined_bilevelopt[1]['theta'].keys()

    for key in  results_refined_bilevelopt[1]['theta'].keys():
        if key not in theta_keys:
            theta_keys.append(key)

    columns_df = ['stage'] + ['iter'] + ['theta_' + str(i) for i in theta_keys] + ['objective']

    df_bilevel_norefined = pd.DataFrame(columns=columns_df)
    df_bilevel_refined = pd.DataFrame(columns=columns_df)

    # Create pandas dataframe using each row of the dictionary returned by the bilevel methodç

    # Create pandas dataframe with no refined solution
    for iter in np.arange(1, norefined_bilevel_iters + 1):

        estimates = []
        for attr in theta_keys:
            if attr in results_norefined_bilevelopt[1]['theta'].keys():
                estimates.append(float(results_norefined_bilevelopt[iter]['theta'][attr]))
            else:
                estimates.append(float(np.nan))

        df_bilevel_norefined.loc[iter] = ['norefined'] + [iter] + estimates + [
            results_norefined_bilevelopt[iter]['objective']]

    if 'c' in list(results_norefined_bilevelopt[1]['theta'].keys()):
        # Create additional variables
        df_bilevel_norefined['vot'] = df_bilevel_norefined['theta_tt'].div( df_bilevel_norefined['theta_c'].where(df_bilevel_norefined['theta_c'] != 0, np.nan))

    elif 'tt_sd' in list(results_norefined_bilevelopt[1]['theta'].keys()):
        # Create additional variables
        df_bilevel_norefined['vot'] = df_bilevel_norefined['theta_tt'] / df_bilevel_norefined['theta_tt_sd']

    # elif 'tt_sd_adj'in list(results_norefined_bilevelopt[1]['theta'].keys()):
    #     df_bilevel_norefined['vot'] = df_bilevel_norefined['theta_tt'] / df_bilevel_norefined['theta_tt_sd_adj']
        
    else:
        df_bilevel_norefined['vot'] = float('nan')

    # Create pandas dataframe with refined solution
    for iter in np.arange(1, refined_bilevel_iters + 1):

        estimates = []
        for attr in theta_keys:
            if attr in results_refined_bilevelopt[1]['theta'].keys():
                estimates.append(float(results_refined_bilevelopt[iter]['theta'][attr]))
            else:
                estimates.append(float(np.nan))

        df_bilevel_refined.loc[iter] = ['refined'] + [iter] + estimates + [
            results_refined_bilevelopt[iter]['objective']]

    if 'c' in list(results_refined_bilevelopt[iter]['theta'].keys()):
        # Create additional variables
        df_bilevel_refined['vot'] = df_bilevel_refined['theta_tt'].div( df_bilevel_refined['theta_c'].where(df_bilevel_refined['theta_c'] != 0, np.nan))


    elif 'tt_sd' in list(results_refined_bilevelopt[1]['theta'].keys()):

        # Create additional variables

        df_bilevel_refined['vot'] = df_bilevel_refined['theta_tt'] / df_bilevel_refined['theta_tt_sd']


    elif 'tt_sd_adj' in list(results_refined_bilevelopt[1]['theta'].keys()):

        df_bilevel_refined['vot'] = df_bilevel_refined['theta_tt'] / df_bilevel_refined['theta_tt_sd_adj']


    else:

        df_bilevel_refined['vot'] = float('nan')

    # Adjust the iteration numbers
    df_bilevel_refined['iter'] = (df_bilevel_refined['iter'] + df_bilevel_norefined['iter'].max()).astype(int)

    # Append dataframes
    bilevel_estimation_df = df_bilevel_norefined.append(df_bilevel_refined)

    return bilevel_estimation_df


def get_gap_estimates_over_iterations(results_norefined: pd.DataFrame, results_refined: pd.DataFrame, theta_true: dict):

    gap_estimates_over_iterations_df = get_loss_and_estimates_over_iterations(results_norefined,results_refined)

    # Drop objective value column as it is unnecessary
    gap_estimates_over_iterations_df = gap_estimates_over_iterations_df.drop(['objective'], axis = 1)



    for attr, theta_true_i in theta_true.items():

        # attr_key = attr[attr.index('_'):]
        attr_key = 'theta_' + attr

        if attr_key in gap_estimates_over_iterations_df.keys():

            gap_estimates_over_iterations_df[attr_key] -= theta_true_i

            # rename column
            gap_estimates_over_iterations_df = gap_estimates_over_iterations_df.rename(columns = {attr_key: 'gap_'+attr_key})

    #VOT
    if 'vot' in gap_estimates_over_iterations_df.keys() and not gap_estimates_over_iterations_df['vot'].isnull().values.any():

        gap_estimates_over_iterations_df = gap_estimates_over_iterations_df.rename(
            columns={'vot': 'gap_vot'})

        gap_estimates_over_iterations_df['gap_vot'] -= theta_true['tt']/theta_true['c']



    return gap_estimates_over_iterations_df


def get_predicted_link_counts_over_iterations_df(results_norefined: dict, results_refined: dict,  Nt):
    # Create pandas dataframe with link_keys as rows and add observed link column

    link_keys = results_norefined[1]['equilibrium']['x'].keys()

    predicted_counts_link_df = pd.DataFrame(
        {'link_key': link_keys,
         'observed': 0,
         'true_count': np.nan
         }
    )

    observed_links_keys = [link.key for link in Nt.get_observed_links()]

    for link in Nt.links:
        if link.key in observed_links_keys:
            predicted_counts_link_df.loc[predicted_counts_link_df['link_key'] == link.key,['observed']] = 1
            predicted_counts_link_df.loc[predicted_counts_link_df['link_key'] == link.key, ['true_count']] = link.observed_count

    # Create a dictionary with the predicted counts for every link over iterations
    predicted_counts_norefined_link_dict = {iter: results_norefined[iter]['equilibrium']['x'] for iter in results_norefined.keys()}

    # Create a dictionary with link keys and predicted counts over iterations as values
    predicted_link_counts_dict = {}
    counter = 0

    #Fill out with predicted count of no refined stage
    for iteration, link_counts_dict in predicted_counts_norefined_link_dict.items():

        for link_key, count in link_counts_dict.items():

            if counter == 0:
                predicted_link_counts_dict[link_key] = [count]
            else:
                predicted_link_counts_dict[link_key].append(count)

        counter += 1

        # Now create one column for the predicted count in each iteration
        predicted_counts_link_df['iter_' + str(counter)] = list(predicted_counts_norefined_link_dict[iteration].values())

    # Fill out with predicted count of refined stage
    predicted_counts_refined_link_dict = {iter: results_refined[iter]['equilibrium']['x'] for iter in results_refined.keys()}

    for iteration, link_counts_dict in predicted_counts_refined_link_dict.items():

        for link_key, count in link_counts_dict.items():

            if counter == 0:
                predicted_link_counts_dict[link_key] = [count]
            else:
                predicted_link_counts_dict[link_key].append(count)

        counter += 1

        # Now create one column for the predicted count in each iteration
        predicted_counts_link_df['iter_' + str(counter)] = list(predicted_counts_refined_link_dict[iteration].values())

    return predicted_counts_link_df


def get_gap_predicted_link_counts_over_iterations_df(results_norefined: dict, results_refined: dict, Nt):

    predicted_link_counts_over_iterations_df = get_predicted_link_counts_over_iterations_df(results_norefined, results_refined, Nt)

    for key in predicted_link_counts_over_iterations_df.keys():

        if 'iter' in key:
            predicted_link_counts_over_iterations_df[key] -= predicted_link_counts_over_iterations_df['true_count']


    return predicted_link_counts_over_iterations_df


def get_predicted_traveltimes_over_iterations_df(results_norefined: dict, results_refined: dict,  Nt):
    # Create pandas dataframe with link_keys as rows and add observed link column

    link_keys = results_norefined[1]['equilibrium']['tt_x'].keys()

    predicted_traveltime_link_df = pd.DataFrame(
        {'link_key': link_keys,
         'observed': 0
         }
    )

    observed_links_keys = [link.key for link in Nt.get_observed_links()]

    for link in Nt.links:
        if link.key in observed_links_keys:
            predicted_traveltime_link_df.loc[predicted_traveltime_link_df['link_key'] == link.key,['observed']] = 1

    # Create a dictionary with the predicted traveltime for every link over iterations
    predicted_traveltime_norefined_link_dict = {iter: results_norefined[iter]['equilibrium']['tt_x'] for iter in results_norefined.keys()}

    # Create a dictionary with link keys and predicted traveltime over iterations as values
    predicted_link_traveltime_dict = {}
    counter = 0

    #Fill out with predicted travel times of no refined stage
    for iteration, link_traveltime_dict in predicted_traveltime_norefined_link_dict.items():

        for link_key, traveltime in link_traveltime_dict.items():

            if counter == 0:
                predicted_link_traveltime_dict[link_key] = [traveltime]
            else:
                predicted_link_traveltime_dict[link_key].append(traveltime)

        counter += 1

        # Now create one column for the predicted count in each iteration
        predicted_traveltime_link_df['iter_' + str(counter)] = list(predicted_traveltime_norefined_link_dict[iteration].values())

    # Fill out with predicted count of refined stage
    predicted_traveltime_refined_link_dict = {iter: results_refined[iter]['equilibrium']['tt_x'] for iter in results_refined.keys()}

    for iteration, link_traveltime_dict in predicted_traveltime_refined_link_dict.items():

        for link_key, traveltime in link_traveltime_dict.items():

            if counter == 0:
                predicted_link_traveltime_dict[link_key] = [traveltime]
            else:
                predicted_link_traveltime_dict[link_key].append(traveltime)

        counter += 1

        # Now create one column for the predicted count in each iteration
        predicted_traveltime_link_df['iter_' + str(counter)] = list(predicted_traveltime_refined_link_dict[iteration].values())

    return predicted_traveltime_link_df


def summary_table_links(links: [], Z_attrs = None, Z_labels = None) -> pd.DataFrame:
    
    """ It is expected to receive list of links with observed counts but it can receive an arbitrary list of links as well """

    # Allow to receive arbitrary link labels and generate a table showing those labels as column in the panda dataframe
    
    link_pems_station_id = [link.pems_stations_ids for link in links]

    # Traffic counts
    # dict_link_counts = {(link.key[0], link.key[1]): np.round(link.observed_count, 1) for link in links}

    dict_link_counts = {link.key: np.round(link.observed_count, 1) for link in links}

    link_capacities = np.array(
        [int(link.bpr.k) for link in links ])

    link_capacities_print = []
    for i in range(len(link_capacities)):
        if link_capacities[i] > 10000:
            link_capacities_print.append(float('inf'))
        else:
            link_capacities_print.append(link_capacities[i])

    link_speed_ff =  np.empty(len(dict_link_counts))

    if 'ff_speed' in links[0].Z_dict:
        link_speed_ff = np.array(
            [link.Z_dict['ff_speed'] for link in links ])

    else:
        link_speed_ff[:] = np.nan

    link_speed_ff_print = []
    for i in range(len(link_speed_ff)):
        if link_speed_ff[i] > 1000:  # 99999
            link_speed_ff_print.append(float('inf'))
        else:
            link_speed_ff_print.append(link_speed_ff[i])

    # Travel time are originally in minutes
    # link_traveltime_ff = np.array(
    #     [str(np.round(link.Z_dict['ff_traveltime'], 2)) for link in links if
    #      not np.isnan(link.observed_count)])
    link_traveltime_ff = np.array(
        [link.bpr.tf for link in links ])

    # Inrix id
    link_inrix_id = np.array(
        [link.inrix_id for link in links ])

    # link_incidents = np.empty(len(dict_link_counts))
    # link_incidents[:] = np.nan

    # link_streets_intersections = np.empty(len(dict_link_counts))
    # link_streets_intersections[:] = np.nan
    #
    # if 'intersections' in links[0].Z_dict:
    #     link_streets_intersections = np.array(
    #         [int(link.Z_dict['intersections']) for link in links ])

    # Link internal attributes
    summary_links_df = pd.DataFrame({'link_key': dict_link_counts.keys()
                                                 , 'inrix_id': link_inrix_id
                                                 , 'pems_id': link_pems_station_id
                                                 , 'counts': dict_link_counts.values()
                                                 , 'capacity [veh]': link_capacities_print
                                                 , 'speed_ff[mi/hr]': link_speed_ff_print
                                                 , 'tt_ff [min]': link_traveltime_ff
                                              })

    # Exogenous link attributes

    summary_links_Z_attrs_df = get_link_Z_attributes_df(links, Z_attrs, Z_labels )

    # summary_links_Z_attrs_df.columns = Z_labels

    summary_links_df  = pd.concat([summary_links_df,summary_links_Z_attrs_df], join='outer', axis=1)

    return summary_links_df

def scatter_plots_features_vs_counts(links_df):

    """ Scatter plot between traffic counts and travel time/speed reliability and average. Repeat the same but for the remaining covariates """

    # plt.figure()

    fig1 = plt.figure()

    # https://seaborn.pydata.org/tutorial/axis_grids.html

    g1 = sns.PairGrid(links_df)
    g1.map_diag(sns.histplot)
    # g.map(sns.scatterplot)
    g1.map_offdiag(sns.regplot)
    # fig1.show()
    # g1.savefig('output1.png')

    # plt.show()

    # Save figure




    # Plot differentiating relationship based on capacity

    fig2 = plt.figure()

    # https://seaborn.pydata.org/tutorial/axis_grids.html

    g2 = sns.PairGrid(links_df, hue="capacity [veh]")
    g2.map_diag(sns.histplot)
    # g.map(sns.scatterplot)
    g2.map_offdiag(sns.regplot)
    g2.add_legend()
    # g2.savefig('output2.png')

    # g.fig.show()
    # plt.show()

    # Bar plots to depict relationships between ordinal predictions

    # g = sns.FacetGrid(tips, col="day", height=4, aspect=.5)
    # g.map(sns.barplot, "sex", "total_bill", order=["Male", "Female"])





    # matplotlib.rcParams['text.usetex'] = True

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # sns.scatterplot(data=summary_links_df, x="stops", y="counts")
    # sns.scatterplot(data=summary_links_df, x='tt_var', y="counts")
    # sns.scatterplot(data=summary_links_df, x='tt_sd', y="counts")
    # sns.scatterplot(data=summary_links_df, x='speed_avg [mi/hr]', y="counts")
    # sns.scatterplot(data=summary_links_df, x='tt_ff [min]', y="counts")
    # sns.scatterplot(data=summary_links_df, x='tt_sd_adj', y="counts")
    # sns.scatterplot(data=summary_links_df, x='incidents', y="counts")
    # sns.scatterplot(data=summary_links_df, x='ints', y="counts")
    # sns.scatterplot(data=summary_links_df, x='income [1K USD]', y="counts")

    # plt.show()
    
    return g1, g2


def regression_counts_features_link_level(links):

    # TODO: implement this but after implementing a method that returns a pandas dataframe for a selected number of links

    pass

def predicted_link_counts_over_iterations():


    pass

def estimators_over_iterations():


    pass


def pre_post_covid_comparison():
    """ Compare traffic flows between October 2019 and October 2020 which is basically pre and pos covid outbreak (March 2020). This is 15 minutes counts over the October 1st of each year"""


    pass


def analysis_inrix_free_flow_speed_imputation():

    """ Compare the free flow speed from inrix and the ones encoded in the network  """


def congestion_plot_map():
    """ Shows a congestion plot using a red color palette in links with higher counts (similar to PEMS map) """

