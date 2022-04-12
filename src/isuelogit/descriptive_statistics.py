""" Descriptive statistics in terms of tables or plots"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytypes import List, Dict
    from networks import TNetwork

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.transforms import BlendedGenericTransform

import seaborn as sns

import copy

from etl import masked_link_counts_after_path_coverage

from scipy.stats import pearsonr


def distribution_pems_counts(filepath,
                             selected_period,
                             data_reader,
                             selected_links = None,
                             col_wrap = 2):
    """ Analyze counts from a selected group of links. We may compare counts in 2019 and 2020 and over the same day
    of the week during each month to analyze the validity of Gaussian distribution and see differences between years """

    # Selection of an arbitrary of links to show the distribution of traffic counts via selected_links

    # TODO: Add vertical bar at the point with maximum traffic flow

    # Avoid clash with underscore in feature names, e.g. flow_total
    matplotlib.rcParams['text.usetex'] = False

    pems_count_sdf = data_reader.read_pems_counts_data(filepath, selected_period)
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


def get_link_Z_attributes_df(links,
                             attrs_list: List = None,
                             Z_labels: List = None) -> pd.Dataframe:

    df_dict = {}

    # df_dict['key'] = [link.key for link in links]

    missing_attrs = []

    if attrs_list is None:
        attrs_list = list(links[0].Z_dict.keys())

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


def get_loss_and_estimates_over_iterations(results_norefined: pd.DataFrame,
                                           results_refined: pd.DataFrame):

    results_norefined_bilevelopt = results_norefined
    results_refined_bilevelopt = results_refined

    results_norefined_bilevelopt[1].keys()

    refined_bilevel_iters = len(list(results_refined_bilevelopt.keys()))
    norefined_bilevel_iters = len(list(results_norefined_bilevelopt.keys()))

    theta_keys = results_norefined_bilevelopt[1]['theta'].keys()

    for key in  results_refined_bilevelopt[1]['theta'].keys():
        if key not in theta_keys:
            theta_keys.append(key)

    columns_df = ['stage'] + ['iter'] + ['theta_' + str(i) for i in theta_keys] + ['objective'] \
                 + ['n_paths'] + ['n_paths_added'] + ['n_paths_effectively_added']

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

        df_bilevel_norefined.loc[iter] = ['norefined'] + [iter] + estimates \
                                       + [results_norefined_bilevelopt[iter]['objective']] \
                                       + [len(results_norefined_bilevelopt[iter]['f'])] \
                                       + [results_norefined_bilevelopt[iter]['n_paths_added']] \
                                       + [results_norefined_bilevelopt[iter]['n_paths_effectively_added']]

    if 'c' in list(results_norefined_bilevelopt[1]['theta'].keys()):
        # Create additional variables
        df_bilevel_norefined['vot'] = df_bilevel_norefined['theta_tt'].\
            div( df_bilevel_norefined['theta_c'].where(df_bilevel_norefined['theta_c'] != 0, np.nan))

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

        df_bilevel_refined.loc[iter] = ['refined'] + [iter] + estimates \
                                       + [results_refined_bilevelopt[iter]['objective']] \
                                       + [len(results_refined_bilevelopt[iter]['f'])] \
                                       + [results_refined_bilevelopt[iter]['n_paths_added']] \
                                       + [results_refined_bilevelopt[iter]['n_paths_effectively_added']]
        

    if 'c' in list(results_refined_bilevelopt[iter]['theta'].keys()):
        # Create additional variables
        df_bilevel_refined['vot'] = df_bilevel_refined['theta_tt']\
            .div( df_bilevel_refined['theta_c'].where(df_bilevel_refined['theta_c'] != 0, np.nan))


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
    bilevel_estimation_df = pd.concat([df_bilevel_norefined,df_bilevel_refined])

    return bilevel_estimation_df


def get_gap_estimates_over_iterations(results_norefined: pd.DataFrame,
                                      results_refined: pd.DataFrame,
                                      theta_true: dict):

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


def get_predicted_link_counts_over_iterations_df(results_norefined: dict,
                                                 results_refined: dict,
                                                 network):
    # Create pandas dataframe with link_keys as rows and add observed link column

    link_keys = results_norefined[1]['x'].keys()

    predicted_counts_link_df = pd.DataFrame(
        {'link_key': link_keys,
         'observed': 0,
         'true_count': np.nan
         }
    )

    observed_links_keys = [link.key for link in network.get_observed_links()]

    for link in network.links:
        if link.key in observed_links_keys:
            predicted_counts_link_df.loc[predicted_counts_link_df['link_key'] == link.key,['observed']] = 1
            predicted_counts_link_df.loc[predicted_counts_link_df['link_key'] == link.key, ['true_count']] = link.count

    # Create a dictionary with the predicted counts for every link over iterations
    predicted_counts_norefined_link_dict = {iter: results_norefined[iter]['x'] for iter in results_norefined.keys()}

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
    predicted_counts_refined_link_dict = {iter: results_refined[iter]['x'] for iter in results_refined.keys()}

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


def get_gap_predicted_link_counts_over_iterations_df(results_norefined: dict,
                                                     results_refined: dict,
                                                     network):

    predicted_link_counts_over_iterations_df = get_predicted_link_counts_over_iterations_df(results_norefined, results_refined, network)

    for key in predicted_link_counts_over_iterations_df.keys():

        if 'iter' in key:
            predicted_link_counts_over_iterations_df[key] -= predicted_link_counts_over_iterations_df['true_count']

    return predicted_link_counts_over_iterations_df


def get_predicted_traveltimes_over_iterations_df(results_norefined: dict,
                                                 results_refined: dict,
                                                 network):
    # Create pandas dataframe with link_keys as rows and add observed link column

    link_keys = results_norefined[1]['tt_x'].keys()

    predicted_traveltime_link_df = pd.DataFrame(
        {'link_key': link_keys,
         'observed': 0
         }
    )

    observed_links_keys = [link.key for link in network.get_observed_links()]

    for link in network.links:
        if link.key in observed_links_keys:
            predicted_traveltime_link_df.loc[predicted_traveltime_link_df['link_key'] == link.key,['observed']] = 1

    # Create a dictionary with the predicted traveltime for every link over iterations
    predicted_traveltime_norefined_link_dict = {iter: results_norefined[iter]['tt_x'] for iter in results_norefined.keys()}

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
    predicted_traveltime_refined_link_dict = {iter: results_refined[iter]['tt_x'] for iter in results_refined.keys()}

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


def summary_table_links(links: List = None,
                        Z_attrs = None,
                        Z_labels = None) -> pd.DataFrame:
    
    """ It is expected to receive list of links with observed counts but it can receive an arbitrary list of links as well

    # Allow to receive arbitrary link labels and generate a table showing those labels as column in the panda dataframe

    """

    # Traffic counts
    # dict_link_counts = {(link.key[0], link.key[1]): np.round(link.count, 1) for link in links}

    dict_link_counts = {link.key: np.round(link.count, 1) for link in links}

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
        link_speed_ff = np.array([link.Z_dict['ff_speed'] for link in links ])

    else:
        link_speed_ff[:] = np.nan

    # link_speed_ff_print = []
    # for i in range(len(link_speed_ff)):
    #     if link_speed_ff[i] > 1000:  # 99999
    #         link_speed_ff_print.append(float('inf'))
    #     else:
    #         link_speed_ff_print.append(link_speed_ff[i])
    link_speed_ff_print = link_speed_ff

    # Travel time are originally in minutes
    # link_traveltime_ff = np.array(
    #     [str(np.round(link.Z_dict['ff_traveltime'], 2)) for link in links if
    #      not np.isnan(link.count)])
    link_traveltime_ff = np.array([link.bpr.tf for link in links ])

    # link_incidents = np.empty(len(dict_link_counts))
    # link_incidents[:] = np.nan

    # link_streets_intersections = np.empty(len(dict_link_counts))
    # link_streets_intersections[:] = np.nan
    #
    # if 'intersections' in links[0].Z_dict:
    #     link_streets_intersections = np.array(
    #         [int(link.Z_dict['intersections']) for link in links ])

    #TODO: only show columns where there is data. For that first create a dictionary and then a dataframe with selected
    # keys and values

    # Link internal attributes
    summary_links_df = pd.DataFrame({'link_key': dict_link_counts.keys(),
                                     'observed': [int(~np.isnan(link.count)) for link in links],
                                     'counts': dict_link_counts.values(),
                                     'capacity [veh]': link_capacities_print,
                                     'tt_ff [min]': link_traveltime_ff,
                                     'speed_ff[mi/hr]': link_speed_ff_print,
                                     })

    summary_links_df['inrix_id'] =[link.inrix_id for link in links ]
    summary_links_df['inrix_id'] = summary_links_df['inrix_id'].astype('Int64')
    summary_links_df['pems_ids'] = [link.pems_stations_ids for link in links]

    # Link attributes
    summary_links_Z_attrs_df = get_link_Z_attributes_df(links, Z_attrs, Z_labels)

    # summary_links_Z_attrs_df.columns = Z_labels

    summary_links_df  = pd.concat([summary_links_df,summary_links_Z_attrs_df], join='outer', axis=1)

    return summary_links_df

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.7, .9), xycoords=ax.transAxes)

def corrfunc_hue(x, y, **kws):
    # https://stackoverflow.com/questions/43251021/show-two-correlation-coefficients-on-pairgrid-plot-with-hue-categorical-variabl

    nas = np.logical_or(np.isnan(x.values), np.isnan(y.values))

    r, _ = pearsonr(x[~nas], y[~nas])
    ax = plt.gca()
    # count how many annotations are already present
    n = len([c for c in ax.get_children() if
                  isinstance(c, matplotlib.text.Annotation)])
    # pos = (.1, .9 - .3*n)
    # or make positions for every label by hand
    pos = (.7, .9) if kws['label'] == '2019-10-01' else (.7,.8)
    color = sns.color_palette()[0] if kws['label'] == '2019-10-01' else sns.color_palette()[1]

    ax.annotate(f'ρ = {r:.2f}', xy=pos, xycoords=ax.transAxes, color = color)

    # ax.annotate("{}: r = {:.2f}".format(kws['label'],r),
    #             xy=pos, xycoords=ax.transAxes)

def scatter_plots_features(links_df,
                           features: Dict[str, str],
                           folder: str,
                           filename: str,
                           hue = None,
                           normalized = True):

    """ Scatter plot between traffic counts and travel time/speed reliability and average. Repeat the same but for the remaining covariates """

    # plt.figure()

    df = links_df.copy()

    df.rename(columns = features, inplace = True)

    # existing_continous_features = set(links_df.keys()).intersection(set(features))
    existing_continous_features = [label for feature, label in features.items() if feature in links_df.keys()]

    if hue is not None:
        df = df[existing_continous_features + [hue]]
    else:
        df = df[existing_continous_features]

    #Randomly sample points to avoid having a heavy figure
    df = df.sample(frac=0.1, replace=False, random_state=1)

    # fig = plt.figure()

    # https://seaborn.pydata.org/tutorial/axis_grids.html
    if hue is not None:
        g = sns.PairGrid(df, corner=True, hue = hue)
        g.map_lower(corrfunc_hue)
    else:
        g = sns.PairGrid(df, corner=True)
        g.map_lower(corrfunc)

    g.map_diag(sns.histplot)

    # g.map(sns.scatterplot)
    g.map_offdiag(sns.regplot)
    # fig1.show()
    # g1.savefig('output1.png')

    g.fig.set_size_inches(14, 12)

    if normalized:
        range_ticks = [0]+list(np.round(np.arange(0.2,1,0.2),1)) + [1]
        g.set(xlim=[-0.03, 1.03], ylim=[-0.03, 1.03], xticks = range_ticks, yticks = range_ticks)
        f = lambda x, pos: str(x).rstrip('0').rstrip('.')

    for ax in plt.gcf().axes:
        if normalized:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)

    if hue is not None:
        handles = g._legend_data.values()
        labels = g._legend_data.keys()

        g.add_legend(fontsize=14, handles=handles, labels=labels, loc='upper center',
                     title = 'Date', bbox_to_anchor=(.82, .6), frameon=False)

        g. legend.get_title().set_fontsize(14)

        # g.fig.subplots_adjust(top=0.92, bottom=0.08)

    # sns.move_legend(g, "center right")

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

    # # TODO: visualize box plot for the non continuous features
    # categorical_features = ['high_inc', 'stops', 'ints']

    g.savefig(folder + '/' + filename, pad_inches=0.1, bbox_inches="tight", dpi=200)
    
    return g


def scatter_plots_features_vs_counts_by_capacity(links_df):
    """ Scatter plot between traffic counts and travel time/speed reliability and average. Repeat the same but for the remaining covariates """

    # Plot differentiating relationship based on capacity

    fig = plt.figure()

    # https://seaborn.pydata.org/tutorial/axis_grids.html

    g = sns.PairGrid(links_df, hue="capacity [veh]")
    g.map_diag(sns.histplot)
    # g.map(sns.scatterplot)
    g.map_offdiag(sns.regplot)
    g.add_legend()

    # g2.savefig('output2.png')

    # g.fig.show()
    # plt.show()

    # Bar plots to depict relationships between ordinal predictions

    # g = sns.FacetGrid(tips, col="day", height=4, aspect=.5)
    # g.map(sns.barplot, "sex", "total_bill", order=["Male", "Female"])

    return g

def summary_table_networks(networks: List[TNetwork]) -> None:

    networks_table = {'nodes': [], 'links': [], 'paths': [], 'ods': []}
    networks_df = pd.DataFrame()

    # Network description
    for network in networks:
        networks_table['ods'].append(len(network.OD.denseQ(network.Q, remove_zeros=True)))
        networks_table['nodes'].append(network.A.shape[0])
        networks_table['links'].append(len(network.links))
        networks_table['paths'].append(len(network.paths))

    networks_df['network'] = np.array([network.key for network in networks])

    for var in ['nodes', 'links', 'ods', 'paths']:
        networks_df[var] = networks_table[var]

    return networks_df



def adjusted_link_coverage(network, counts) -> None:

    # Initial coverage
    x_bar = np.array(list(counts.values()))[:, np.newaxis]

    # If no path are traversing some link observations, they are set to nan values
    counts = masked_link_counts_after_path_coverage(network, xct=counts)

    x_bar_remasked = np.array(list(counts.values()))[:, np.newaxis]

    true_coverage = np.count_nonzero(~np.isnan(x_bar_remasked)) / x_bar.shape[0]

    # After accounting for path coverage (not all links may be traversed)

    total_true_counts_observations = np.count_nonzero(~np.isnan(np.array(list(counts.values()))))

    print('Adjusted link coverage:', "{:.1%}".format(true_coverage))
    print('Adjusted total link observations: ' + str(total_true_counts_observations))

    # print('dif in coverage', np.count_nonzero(~np.isnan(x_bar))-np.count_nonzero(~np.isnan( x_bar_remasked)))


def summary_links_report(network):

    x_bar = network.link_data.observed_counts_vector
    x_eq = network.link_data.predicted_counts_vector

    idx_nonas = list(np.where(~np.isnan(x_bar))[0])

    link_keys = [(link.key[0], link.key[1]) for link in network.links if not np.isnan(link.count)]

    # [link.count for link in network.links]

    link_capacities = np.array([link.bpr.k for link in network.links if not np.isnan(link.count)])

    link_capacities_print = []
    for i in range(len(link_capacities)):
        if link_capacities[i] > 10000:
            link_capacities_print.append(float('inf'))
        else:
            link_capacities_print.append(link_capacities[i])

    # Travel time are originally in minutes
    link_travel_times = \
        np.array(
            [str(np.round(link.get_traveltime_from_x(network.link_data.x[link.key]), 2)) for link in network.links if
             not np.isnan(link.count)])

    # This may raise a warning for OD connectors where the length is equal to 0

    if 'length' in network.links[0].Z_dict:
        link_speed_av = \
            np.array(
                [np.round(60 * link.Z_dict['length'] / link.get_traveltime_from_x(network.link_data.x[link.key]), 1)
                 for link
                 in
                 network.links if not np.isnan(link.count)])

        # link_speed_av = \
        #     np.array([np.round(60 * link.Z_dict['length'] / link.get_traveltime_from_x(network.x_dict[link.key]), 1) for link in
        #               network.links])[
        #     idx_nonas]

    else:
        link_speed_av = np.array([0] * len(link_travel_times.flatten()))

    summary_table = pd.DataFrame(
        {'link_key': link_keys
            , 'capacity': link_capacities_print
            , 'tt[min]': link_travel_times.flatten()
            , 'speed[mi/hr]': link_speed_av.flatten()
            , 'count': x_bar[idx_nonas].flatten()
            , 'predicted': x_eq[idx_nonas].flatten()})


    return summary_table


def regression_counts_features_link_level(links):

    # TODO: implement this but after implementing a method that returns a pandas dataframe for a selected number of links

    pass

def predicted_link_counts_over_iterations():


    pass

def estimators_over_iterations():


    pass


def pre_post_covid_comparison():
    """ Compare traffic predicted_counts between October 2019 and October 2020 which is basically pre and pos covid outbreak (March 2020). This is 15 minutes counts over the October 1st of each year"""


    pass


def analysis_inrix_free_flow_speed_imputation():

    """ Compare the free flow speed from inrix and the ones encoded in the network  """


def congestion_plot_map():
    """ Shows a congestion plot using a red color palette in links with higher counts (similar to PEMS map) """

import numpy as np

EPSILON = 1e-10


# def _error(actual: np.ndarray, predicted: np.ndarray):
#     """ Simple error """
#     return actual - predicted

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error considering nan in array entries"""

    # Remove elements associated to link with no traffic counts
    idxs = np.where(np.isnan(actual))[0]
    actual = copy.deepcopy(np.delete(actual,idxs,axis = 0))
    predicted = copy.deepcopy(np.delete(predicted, idxs,axis = 0))

    return actual - predicted

def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) /\
               (_error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON)

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


def _bounded_relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Bounded Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        abs_err = np.abs(_error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(_error(actual[seasonality:], _naive_forecasting(actual, seasonality)))
    else:
        abs_err = np.abs(_error(actual, predicted))
        abs_err_bench = np.abs(_error(actual, benchmark))

    return abs_err / (abs_err + abs_err_bench + EPSILON)


def _geometric_mean(a, axis=0, dtype=None):
    """ Geometric mean """
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))

# def rmse_nan(actual: np.ndarray, predicted: np.ndarray):
#     """ Root Mean Squared Error with nan """
#     return np.sqrt(mse(actual, predicted))

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error"""

    assert actual.shape == predicted.shape

    return rmse(actual, predicted)/np.nanmean(actual)

# def nrmse(actual: np.ndarray, predicted: np.ndarray):
#     """ Normalized Root Mean Squared Error """
#     return rmse(actual, predicted) / (actual.max() - actual.min())


def me(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Error """
    return np.mean(_error(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


mad = mae  # Mean Absolute Deviation (it is the same as MAE)


def gmae(actual: np.ndarray, predicted: np.ndarray):
    """ Geometric Mean Absolute Error """
    return _geometric_mean(np.abs(_error(actual, predicted)))


def mdae(actual: np.ndarray, predicted: np.ndarray):
    """ Median Absolute Error """
    return np.median(np.abs(_error(actual, predicted)))


def mpe(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Percentage Error """
    return np.mean(_percentage_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.median(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def smdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.median(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def maape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(actual, predicted) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))


def std_ae(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Error """
    __mae = mae(actual, predicted)
    return np.sqrt(np.sum(np.square(_error(actual, predicted) - __mae))/(len(actual) - 1))


def std_ape(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Percentage Error """
    __mape = mape(actual, predicted)
    return np.sqrt(np.sum(np.square(_percentage_error(actual, predicted) - __mape))/(len(actual) - 1))


def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Mean Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Median Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def rmsse(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """ Root Mean Squared Scaled Error """
    q = np.abs(_error(actual, predicted)) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(np.mean(np.square(q)))


def inrse(actual: np.ndarray, predicted: np.ndarray):
    """ Integral Normalized Root Squared Error """
    return np.sqrt(np.sum(np.square(_error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual))))


def rrse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))


def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Error """
    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Absolute Error """
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Median Relative Absolute Error """
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def gmrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Geometric Mean Relative Absolute Error """
    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Bounded Relative Absolute Error """
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Unscaled Mean Bounded Relative Absolute Error """
    __mbrae = mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae)


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))


METRICS = {
    'mse': mse,
    'rmse': rmse,
    'nrmse': nrmse,
    'me': me,
    'mae': mae,
    'mad': mad,
    'gmae': gmae,
    'mdae': mdae,
    'mpe': mpe,
    'mape': mape,
    'mdape': mdape,
    'smape': smape,
    'smdape': smdape,
    'maape': maape,
    'mase': mase,
    'std_ae': std_ae,
    'std_ape': std_ape,
    'rmspe': rmspe,
    'rmdspe': rmdspe,
    'rmsse': rmsse,
    'inrse': inrse,
    'rrse': rrse,
    'mre': mre,
    'rae': rae,
    'mrae': mrae,
    'mdrae': mdrae,
    'gmrae': gmrae,
    'mbrae': mbrae,
    'umbrae': umbrae,
    'mda': mda,
}


def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics=('mae', 'mse', 'smape', 'umbrae')):
    results = {}
    for name in metrics:
        try:
            results[name] = METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            print('Unable to compute metric {0}: {1}'.format(name, err))
    return results


def evaluate_all(actual: np.ndarray, predicted: np.ndarray):
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))