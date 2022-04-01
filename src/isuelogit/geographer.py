""" Module for gis related operations and classes """

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytypes import Positions, Links, Nodes

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
import addfips
import webbrowser
import urllib
import os
import math

import shapely.geometry
from shapely.geometry import Point, LineString

import config


class NodePosition(Point):

    def __init__(self, x, y, crs):
        '''
        :argument crs: internal crs system 
        '''

        # crs_options = ['xy','lat-lon']

        super().__init__(x, y)

        # self.x = x
        # self.y = y
        # self.xy = (x,y)
        self.crs = crs

    # @property
    # def x(self):
    #     return self._x
    #
    # @x.setter
    # def x(self, value):
    #     self._x = value
    #
    # @property
    # def y(self):
    #     return self._y
    #
    # @y.setter
    # def y(self, value):
    #     self._y = value

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        self._crs = value

    # @property
    def get_xy(self):
        return tuple(np.array(self.xy).flatten())

    # @xy.setter
    # def xy(self, value):
    #     self._xy = value


class LinkPosition(LineString):

    def __init__(self, x_1, y_1, x_2, y_2, crs):
        '''
        :argument crs: internal crs system
        '''

        # crs_options = ['xy','lat-lon']
        super().__init__([[x_1, y_1], [x_2, y_2]])

        # self.x = x
        # self.y = y
        # self.xy = (x,y)
        self.crs = crs

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        self._crs = value

    # @property
    def get_xy(self):
        xy = self.xy
        return tuple([tuple(np.array(xy[0]).flatten()), tuple(np.array(xy[1]).flatten())])


def mirror_x_coordinates(nodes: Nodes):
    positions = [node.position.get_xy() for node in nodes]

    max_x_position = max(list(map(lambda x: x[0], positions)))
    new_positions = list(map(lambda x: (x[0] + 2 * (max_x_position - x[0]), x[1]), positions))

    for node, i in zip(nodes, range(len(nodes))):
        node.position = NodePosition(*new_positions[i], crs='xy')

    return nodes


def mirror_y_coordinates(nodes: Nodes):
    positions = [node.position.get_xy() for node in nodes]

    max_y_position = max(list(map(lambda x: x[1], positions)))
    new_positions = list(map(lambda x: (x[0], x[1] + 2 * (max_y_position - x[1])), positions))

    for node, i in zip(nodes, range(len(nodes))):
        node.position = NodePosition(*new_positions[i], crs='xy')

    return nodes


def rotate_coordinates(nodes: Nodes,
                       degrees,
                       origin,
                       bbox_ranges):
    # https://calcworkshop.com/transformations/rotation-rules/

    nodes_to_rotate = []
    new_positions = []
    if bbox_ranges is not None:

        for node in nodes:

            position = node.position.get_xy()
            # new_positions.append(node.position.get_xy())

            if position[0] > bbox_ranges[0][0] and position[0] <  bbox_ranges[0][1]:

                if position[1] > bbox_ranges[1][0] and position[1] < bbox_ranges[1][1]:

                    nodes_to_rotate.append(node)



    points = [Point(*node.position.get_xy()) for node in nodes_to_rotate]

    gdf = gpd.GeoSeries(points).rotate(degrees, origin=origin)

    for point in list(gdf.values):
        new_positions.append((list(point.coords.xy[0])[0],list(point.coords.xy[1])[0]))

    for node, i in zip(nodes_to_rotate, range(len(nodes_to_rotate))):
        node.position = NodePosition(*new_positions[i], crs=node.position.crs)

    return nodes


def rotate_coordinates_90degrees(nodes: Nodes):
    # https://calcworkshop.com/transformations/rotation-rules/

    positions = [node.position.get_xy() for node in nodes]
    new_positions = list(map(lambda x: (-x[1], x[0]), positions))

    for node, i in zip(nodes, range(len(nodes))):
        node.position = NodePosition(*new_positions[i], crs=node.position.crs)

    return nodes


def rotate_coordinates_180degrees(nodes: Nodes):
    # https://calcworkshop.com/transformations/rotation-rules/

    positions = [node.position.get_xy() for node in nodes]
    new_positions = list(map(lambda x: (-x[0], -x[1]), positions))

    for node, i in zip(nodes, range(len(nodes))):
        node.position = NodePosition(*new_positions[i], crs=node.position.crs)

    return nodes


def rescale_coordinates(nodes: Nodes,
                        factor: float):
    # https://calcworkshop.com/transformations/rotation-rules/

    positions = [node.position.get_xy() for node in nodes]
    new_positions = list(map(lambda x: (x[0] * factor, x[1] * factor), positions))

    for node, i in zip(nodes, range(len(nodes))):
        node.position = NodePosition(*new_positions[i], crs=node.position.crs)

    return nodes

def adjust_by_base_coordinates(nodes: Nodes,
                               base_node,
                               base_coordinates,
                               bbox_ranges: tuple  = None):
    # https://calcworkshop.com/transformations/rotation-rules/

    delta_coordinates = tuple((base_coordinates[i]-base_node.position.get_xy()[i] for i in [0,1]))

    new_positions = []

    positions = [node.position.get_xy() for node in nodes]

    if bbox_ranges is not None:

        counter = 0

        for position in positions:
            new_positions.append(position)

            if position[0] > bbox_ranges[0][0] and position[0] <  bbox_ranges[0][1]:
                if position[1] > bbox_ranges[1][0] and position[1] < bbox_ranges[1][1]:
                    new_positions[counter] = (position[0] + delta_coordinates[0], position[1] + delta_coordinates[1])

            counter += 1

    else:
        new_positions = list(map(lambda x: (x[0] + delta_coordinates[0], x[1] + delta_coordinates[1]), positions))

    for node, i in zip(nodes, range(len(nodes))):
        node.position = NodePosition(*new_positions[i], crs=node.position.crs)

    return nodes


# def adjust_by_base_coordinates(nodes: Nodes, base_node, base_coordinates):
#     # https://calcworkshop.com/transformations/rotation-rules/
#
#     delta_coordinates = tuple((base_coordinates[i]-base_node.position.get_xy()[i] for i in [0,1]))
#
#     positions = [node.position.get_xy() for node in nodes]
#     new_positions = list(map(lambda x: (x[0] + delta_coordinates[0], x[1] + delta_coordinates[1]), positions))
#
#     for node, i in zip(nodes, range(len(nodes))):
#         node.position = NodePosition(*new_positions[i], crs=node.position.crs)
#
#     return nodes

def adjust_fresno_nodes_coordinates(nodes: Nodes,
                                    rescale_factor = None) -> None:
    # Mirroring x coordinates to match the visualization of the network shown in papers
    nodes = mirror_x_coordinates(nodes=nodes)

    # Rotate coordinates in 180 degrees
    nodes = rotate_coordinates_180degrees(nodes=nodes)

    # Rescaling coordinates by the magic factor
    # nodes = rescale_coordinates(nodes=nodes, factor=0.387)
    if rescale_factor is None:
        nodes = rescale_coordinates(nodes=nodes, factor=0.387)
    else:
        nodes = rescale_coordinates(nodes=nodes, factor=rescale_factor)

    # # Assign coordinate to node with the lowest y coordinate
    # base_coordinate = (-119.785332, 36.678848)

    # node_base_id = '91'
    # base_coordinates = (6332273.829, 2138737.471)

    # node_base_id = '86'
    # base_coordinates = (6332100.01,2130092.56)

    # This point is in the top right of the network map
    node_base_id1 = '730'
    base_coordinates_1 = (6332655.000030, 2197312.999844)
    xrange_bbox_1 = (float('-inf'), float('inf'))
    yrange_bbox_1 = (2175760.20, float('inf'))

    base_node1 = None
    for node in nodes:
        if node.id == node_base_id1:
            base_node1 = node

    # First adjustment
    nodes = adjust_by_base_coordinates(base_node= base_node1
                                       , base_coordinates = base_coordinates_1,nodes=nodes)



    # The readjustment is done separately for the links located in the top and bottom part of the network graph.

    # TODO: Adjustment for bottom part which is not done temprarilly


    node_base_id2 = '283'
    base_coordinates_2 = (6331218.3,2132085.5)
    xrange_bbox_2 = (float('-inf'), float('inf'))
    yrange_bbox_2 = (float('-inf'),2175760.20)
    # xrange_bbox_2 = (float('-inf'), float('inf'))
    # yrange_bbox_2 = (float('-inf'), float('inf'))

    base_node2 = None

    for node in nodes:
        if node.id == node_base_id2:
            base_node2 = node

    # Rotate points in the bottom by 2.4 degrees
    nodes = rotate_coordinates(nodes = nodes, origin=base_node2.position.get_xy(), degrees=-3,
                               bbox_ranges=(xrange_bbox_2, yrange_bbox_2))

    # nodes = adjust_by_base_coordinates(base_node= base_node1
    #                                    , base_coordinates = base_coordinates_1
    #                                    , bbox_ranges = (xrange_bbox_1,yrange_bbox_1),nodes=nodes)

    nodes = adjust_by_base_coordinates(base_node=base_node2
                                       , base_coordinates=base_coordinates_2
                                       , bbox_ranges=(xrange_bbox_2, yrange_bbox_2), nodes=nodes)

    # Final adjustment for three blocks in the middle that gets desaligned

    node_base_id3 = '515' #(link id: 1189  )
    base_coordinates_3 = (6332868.40,2173134.60)
    xrange_bbox_3 = (float('-inf'), 6333356.2)
    yrange_bbox_3 = (2162566.1,2175740.5)
    base_node3 = None

    for node in nodes:
        if node.id == node_base_id3:
            base_node3 = node

    # # Rotate points in the bottom by -2.4 degrees
    nodes = rotate_coordinates(nodes = nodes, origin=base_node3.position.get_xy(), degrees=2,
                               bbox_ranges=(xrange_bbox_3, yrange_bbox_3))

    nodes = adjust_by_base_coordinates(base_node=base_node3
                                       , base_coordinates=base_coordinates_3
                                       , bbox_ranges=(xrange_bbox_3, yrange_bbox_3), nodes=nodes)


    # return nodes

# def adjust_by_base_coordinates(nodes: Nodes, base_node, base_coordinates):
#     # https://calcworkshop.com/transformations/rotation-rules/
#
#     delta_coordinates = tuple((base_coordinates[i]-base_node.position.get_xy()[i] for i in [0,1]))
#
#     positions = [node.position.get_xy() for node in nodes]
#     new_positions = list(map(lambda x: (x[0] + delta_coordinates[0], x[1] + delta_coordinates[1]), positions))
#
#     for node, i in zip(nodes, range(len(nodes))):
#         node.position = NodePosition(*new_positions[i], crs=node.position.crs)
#
#     return nodes


def set_cardinal_direction_links(links):

    for link in links:

        directions = [None,None]

        init_position = link.init_node.position.get_xy()
        term_position =  link.term_node.position.get_xy()

        if term_position[1] > init_position[1]:
            directions[0] = 'N'
        else:
            directions[0] = 'S'

        if term_position[0] > init_position[0]:
            directions[1] = 'E'

        else:
            directions[1] = 'W'

        link.direction = (directions[0],directions[1])
        link.crs_distance = ((term_position[0]-init_position[0])**2 + (term_position[1]-init_position[1])**2)**0.5

        link.direction_confidence = (abs(term_position[1]-init_position[1]), abs(term_position[0]-init_position[0]))

        denominator = link.direction_confidence[0]+link.direction_confidence[1]

        # An OD connector may have the same origin and destination coordinate
        if denominator == 0:
            link.direction_confidence = (0,0)
        else:
            #Normalize by the sum of the confidence for each coordinate in the tuple
            link.direction_confidence = (link.direction_confidence[0]/denominator, link.direction_confidence[1]/denominator)



def write_node_points_shp(nodes,
                          folderpath: str,
                          networkname: str) -> None:
    """ Save line segments using x,y coordinates from nodes """

    # links = N['train']['Fresno'].links

    # Create dataframe
    # links
    # df = pd.DataFrame(columns 0 )
    # link.position

    df = pd.DataFrame()

    df['geometry'] = [Point(node.position) for node in nodes]
    df['id'] = [str(node.id) for node in nodes]
    df['key'] = [str(node.key) for node in nodes]
    df['node_xy'] = [str(node.position.get_xy()) for node in nodes]
    gdf = gpd.GeoDataFrame(df, geometry=df.geometry)

    # Set coordinate system
    gdf = gdf.set_crs(config.gis_options['crs_ca'])

    # Save shapefile
    gdf.to_file(driver='ESRI Shapefile', filename=folderpath + '/' + networkname + "_nodes.shp")

    print('\nShapefile with nodes of ' + networkname + ' was written')

def write_line_segments_shp(links,
                            folderpath: str,
                            networkname: str) -> None:
    """ Save line segments using x,y coordinates from nodes """

    # Links positions comes from x,y coordinates of nodes, which are expected to be different from the raw file to
    # better match the shape of the network in a real map

    df = pd.DataFrame()

    df['geometry'] = [LineString(link.position) for link in links]
    df['id'] = [str(link.id) for link in links]
    df['key'] = [str(link.key) for link in links]
    df['init_id'] = [str(link.init_node.id) for link in links]
    df['term_id'] = [str(link.term_node.id) for link in links]
    df['init_xy'] = [str(np.round(link.init_node.position.get_xy())) for link in links]
    df['term_xy'] = [str(np.round(link.term_node.position.get_xy())) for link in links]
    df['direction'] = [str(link.direction) for link in links]
    df['direction_confidence'] = [str((round(link.direction_confidence[0],1),round(link.direction_confidence[1],1))) for link in links]
    df['lane'] = [str(link.Z_dict['lane']) for link in links]
    df['link_type'] = [str(link.link_type) for link in links]
    df['pems_id1'] = ''
    df['pems_id2'] = ''
    df['pems_id3'] = ''
    df['pems_id4'] = ''

    gdf = gpd.GeoDataFrame(df, geometry=df.geometry)

    # gdf.keys()

    # crs = {'init': 'epsg:27700'}

    # gdf.plot()

    # plt.show()

    gdf.set_crs(config.gis_options['crs_ca'])

    # Save shapefile
    gdf.to_file(driver='ESRI Shapefile', filename=folderpath + '/' + networkname + "_links.shp")

    print('\nShapefile with links of ' + networkname + ' was written')


def write_links_congestion_map_shp(links,
                                   predicted_counts,
                                   folderpath: str,
                                   networkname: str) -> None:
    """ Save line segments using x,y coordinates from nodes """

    # Links positions comes from x,y coordinates of nodes, which are expected to be different from the raw file to
    # better match the shape of the network in a real map

    df = pd.DataFrame()

    df['geometry'] = [LineString(link.position) for link in links]
    df['id'] = [str(link.id) for link in links]
    df['key'] = [str(link.key) for link in links]
    df['init_id'] = [str(link.init_node.id) for link in links]
    df['term_id'] = [str(link.term_node.id) for link in links]
    df['init_xy'] = [str(np.round(link.init_node.position.get_xy())) for link in links]
    df['term_xy'] = [str(np.round(link.term_node.position.get_xy())) for link in links]
    df['direction'] = [str(link.direction) for link in links]
    df['direction_confidence'] = [str((round(link.direction_confidence[0],1),round(link.direction_confidence[1],1))) for link in links]
    df['lane'] = [str(link.Z_dict['lane']) for link in links]
    df['link_type'] = [str(link.link_type) for link in links]
    df['capacity'] = [str(link.bpr.k) for link in links]
    df['predicted_counts'] = list(predicted_counts.flatten())

    df['cong_idx'] = np.minimum(df['predicted_counts']/df['capacity'].astype('float'),1).round(4)

    gdf = gpd.GeoDataFrame(df, geometry=df.geometry)

    # gdf.keys()

    # crs = {'init': 'epsg:27700'}

    # gdf.plot()

    # plt.show()

    gdf.set_crs(config.gis_options['crs_ca'])

    # Save shapefile
    gdf.to_file(driver='ESRI Shapefile', filename=folderpath + '/' + networkname + "_congestion_map.shp")

    print('\nShapefile with links congestion of ' + networkname + ' was written')


def write_census_blocks_data_fresno(countyname: str,
                                    filepath: str):

    # note that field names with more than 10 characters will be shorten down

    # =============================================================================
    # CENSUS DATA
    # =============================================================================

    # Census data will allow to depict nodes/street characteristics

    # https://pypi.org/project/CensusData/
    # https://jtleider.github.io/censusdata/

    # # i) DIVISION LEVEL
    #
    # # Reading census divisions
    #
    # census_divisions_shp_path = 'data/public/census/tl_2020_us_cd116/' + \
    #                             'tl_2020_us_cd116.shp'
    #
    # # This convers the entire US
    # census_divisions_gdf = gpd.read_file(census_divisions_shp_path)
    #
    # census_divisions_gdf.crs
    #
    # # ii) TRACT LEVEL
    # census_tracts_shp_path = 'data/public/census/tl_2020_06_tract/' + \
    #                          'tl_2020_06_tract.shp'
    #
    # census_tract_gdf = gpd.read_file(census_tracts_shp_path)

    # census_tract_gdf.plot()
    # plt.show()

    # iii) BLOCK GROUP LEVEL

    # source: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-data.html
    # metadata: https://www2.census.gov/geo/tiger/TIGER_DP/2018ACS/Metadata/BG_METADATA_2018.txt

    # Read census block data with sociodemographic information

    census_blocks_gdf = gpd.read_file(filepath, driver="FileGDB", layer="ACS_2018_5YR_BG_06_CALIFORNIA")

    # These layers takes time to read
    block_age_sex_gdf = gpd.read_file(filepath, driver="FileGDB", layer='X01_AGE_AND_SEX')
    block_income_gdf = gpd.read_file(filepath, driver="FileGDB", layer='X19_INCOME')

    # # This layer has no info for CA apparently
    # block_commuting_gdf = gpd.read_file(gdb_path, driver="FileGDB", layer = 'X08_COMMUTING')
    #
    # # These are empty
    # block_employment_status_gdf = gpd.read_file(gdb_path, driver="FileGDB", layer = 'X23_EMPLOYMENT_STATUS')
    # block_poverty_gdf = gpd.read_file(gdb_path, driver="FileGDB", layer = 'X17_POVERTY')

    # Filter blocks corresponding to Fresno only. Use FIPS information to filter based on GEOID.
    # Otherwise, it takes too long to read a file
    # https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html

    # county_name = 'Fresno'

    fips_county = addfips.AddFIPS().get_county_fips(countyname, state='California')  # 06019

    census_blocks_geometry_county_gdf = census_blocks_gdf[
        census_blocks_gdf['GEOID'].apply(lambda x: x[0:5]) == fips_county]
    block_income_county_gdf = block_income_gdf[
        block_income_gdf['GEOID'].apply(lambda x: x.split('US')[1][0:5]) == fips_county]
    block_age_sex_county_gdf = block_age_sex_gdf[
        block_age_sex_gdf['GEOID'].apply(lambda x: x.split('US')[1][0:5]) == fips_county]

    # print(list(block_income_county_gdf.keys()))
    # print(list(block_age_sex_county_gdf.keys()))
    #
    # # MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS
    # list(block_income_county_gdf['B19013e1'])
    #
    # # Total Population -- (Estimate)
    # list(block_age_sex_county_gdf['B01001e1'])
    #
    # # Male: Total Population -- (Estimate)
    # list(block_age_sex_county_gdf['B01001e2'])
    #
    # # SEX BY AGE: Female: Total Population -- (Estimate)
    # list(block_age_sex_county_gdf['B01001e26'])
    #
    # # MEDIAN AGE BY SEX: Total: Male: Total Population -- (Estimate)
    # list(block_age_sex_county_gdf['B01002e2'])
    #
    # # MEDIAN AGE BY SEX: Total: Female: Total Population -- (Estimate)
    # list(block_age_sex_county_gdf['B01002e3'])
    #
    # # B01002e1	MEDIAN AGE BY SEX: Total: Total Population -- (Estimate)
    # list(block_age_sex_county_gdf['B01002e1'])

    # Merge data into a single geodataframe

    census_blocks_county_gdf = census_blocks_geometry_county_gdf[['GEOID', 'geometry']]

    block_income_county_gdf = block_income_county_gdf[['GEOID', 'B19013e1']]
    block_income_county_gdf = block_income_county_gdf.rename(columns={'B19013e1': 'median_inc'})
    block_income_county_gdf['GEOID'] = block_income_county_gdf['GEOID'].apply(lambda x: x[7:])

    block_age_sex_county_gdf = block_age_sex_county_gdf[['GEOID', 'B01002e1']]
    block_age_sex_county_gdf = block_age_sex_county_gdf.rename(columns={'B01002e1': 'median_age'})
    block_age_sex_county_gdf['GEOID'] = block_age_sex_county_gdf['GEOID'].apply(lambda x: x[7:])

    # gpd.merge(census_blocks_county_gdf,block_income_county_gdf)

    census_blocks_county_gdf \
        = census_blocks_county_gdf.merge(block_income_county_gdf, left_on='GEOID', right_on='GEOID').\
        merge(block_age_sex_county_gdf, left_on='GEOID', right_on='GEOID')

    # Write geodatabase for Fresno only
    filename = config.paths['folder_gis_data'] + countyname + '/census/' + countyname + '_census_shp'
    census_blocks_county_gdf.to_file(driver='ESRI Shapefile',filename=filename)


def read_census_tracts_shp_fresno(filepath: str) -> gpd.GeoDataFrame:

    # Read geopandas dataframe
    census_tracts_gdf = gpd.read_file(filepath, header=0)

    # census_tracts_df = census_tracts_df.to_crs(epsg=config.gis_options['crs_ca'])

    return census_tracts_gdf


def match_network_links_and_census_tracts_fresno(network_gdf: gpd.GeoDataFrame,
                                                 census_tracts_gdf: gpd.GeoDataFrame,
                                                 links: Links, attrs: [],
                                                 inrix_matching: bool = False) -> None:

    """

    :param pct: percentile to define the high income feature

    """

    print('\nReading census data at the block level')

    census_tracts_gdf = census_tracts_gdf.to_crs(epsg=config.gis_options['crs_ca'])
    network_gdf = network_gdf.to_crs(epsg=config.gis_options['crs_ca'])

    network_census_data_gdf = gpd.sjoin(network_gdf, census_tracts_gdf, how="left", predicate='intersects').drop(
        ['index_right'], axis=1)

    # Add census data to link objects
    n_matched_links = 0
    n_imputed_links = 0

    for link in links:

        matched_link = False
        imputed_link = False

        if inrix_matching:
            internal_link_id = link.inrix_id

            link_rows = network_census_data_gdf.loc[network_census_data_gdf['XDSegID'] == str(internal_link_id)]

        else:
            internal_link_id = link.id
            link_rows = network_census_data_gdf.loc[network_census_data_gdf['id'] == str(internal_link_id)]

        for index, link_row in link_rows.iterrows():

            for attr in attrs:

                link.Z_dict[attr] = link_row[attr]
                matched_link = True

                if attr == 'median_inc':
                    # Adjust units of income data from USD to 1000 USD (K)
                    link.Z_dict['median_inc'] = link.Z_dict['median_inc'] / 1000

        # Impute missing values
        for attr in attrs:
            # if matched_link is False:

            if attr not in link.Z_dict or np.isnan(link.Z_dict[attr]):
                link.Z_dict[attr] = np.mean(network_census_data_gdf[attr])

                if attr == 'median_inc':
                    # Adjust units of income data from USD to 1000 USD (K)
                    link.Z_dict['median_inc'] = link.Z_dict['median_inc'] / 1000

                imputed_link = True
                matched_link = False



        if matched_link is True:
            n_matched_links += 1

        if imputed_link is True:
            n_imputed_links += 1

    # assert len(network_census_data_gdf) == n_matched_links, 'errors in matching'

    print(str(n_matched_links) + ' links were matched (' + "{:.1%}". format(n_matched_links/len(links)) + ' of links)')
    print(str(n_imputed_links) + ' links were imputed (' + "{:.1%}". format(n_imputed_links/len(links)) + ' of links)')

def match_network_links_and_inrix_segments_fresno(network_gdf: gpd.GeoDataFrame,
                                                  inrix_gdf: gpd.GeoDataFrame,
                                                  links: Links,
                                                  buffer_size: float = 0,
                                                  centroids: bool = False):

    """
    :param buffer_size: in feets if crs from CA is used

    """

    print('Matching INRIX segments (N=' + str(len(inrix_gdf)) + ') with network links')


    crs_ca = config.gis_options['crs_ca']

    # i) Buffer around INRIX segments

    # Reproject coordinates into CA system
    inrix_gdf = inrix_gdf.to_crs(epsg=crs_ca)
    network_buffer_gdf = network_gdf.to_crs(epsg=crs_ca)

    # # Generate buffer around traffic incidents points
    # inrix_buffer_gdf['geometry'] = inrix_buffer_gdf.geometry.buffer(10)
    #
    # # Spatial join of traffic incidents and links geometries
    # incidents_network_gdf = gpd.sjoin(network_gdf, inrix_buffer_gdf, how='inner', predicate='intersects')
    #
    # # Verify that the spatial join is one to one

    # ii) Strategy using centroids of network link segments
    link_centroids_gdf = network_buffer_gdf.to_crs(crs=crs_ca)

    if centroids is True:
        link_centroids_gdf['geometry'] = link_centroids_gdf.centroid

    # inrix_centroids_gdf.plot(facecolor="None", edgecolor="blue",markersize=0.2)
    # plt.show()

    # Generate buffers
    link_centroids_buffer_gdf = link_centroids_gdf.to_crs(epsg=crs_ca)
    link_centroids_buffer_gdf['geometry'] = link_centroids_buffer_gdf.geometry.buffer(buffer_size)

    # link_centroids_buffer_gdf.plot(edgecolor="blue")
    # plt.show()

    # Join inrix street centroids with buffer to osmn data
    inrix_network_gdf = gpd.sjoin(link_centroids_buffer_gdf, inrix_gdf,
                                  how="left", predicate='intersects').drop(['index_right'], axis=1)

    # if selected_years is not None:
    #     incidents_network_gdf = incidents_network_gdf.loc[incidents_network_gdf['year'].isin(selected_years)]
    #
    # n_years = len(incidents_network_gdf ['year'].unique())

    # TODO: see how to account for dups. Now, it picks the last row in the matching which is not necessarily the best.

    counter = 0
    sum_confidence = 0

    for link in links:

        match = False

        link_key = str(link.key)

        best_confidence = 0

        inrix_segments = inrix_network_gdf.loc[inrix_network_gdf['key'] == link_key]

        for index, inrix_segment in inrix_segments.iterrows():
            inrix_id = pd.to_numeric(inrix_segment['XDSegID'], errors='coerce', downcast = 'integer')

            if np.isnan(inrix_id):
                link.inrix_id = None

            else:
                # Use the direction (N,S,W,E) as an additional criteria to perform the matching

                for i in range(len(link.direction)):

                    link_confidence = link.direction_confidence[i]

                    # Choose the direction with the stronger confidence if two are matched

                    if inrix_segment['Bearing'] == link.direction[i] and link_confidence > best_confidence:

                        link.inrix_id = inrix_id
                        best_confidence = link_confidence

                        match = True

        # print({'link_key': link_key, 'inrix_id': link.inrix_id})

        if match is True:
            sum_confidence += best_confidence
            counter += 1

    # for link in links:
    #     print(link.key)
    #     print(link.inrix_id)

    # config.gis_results['inrix_matching'] \
    #     = {'perc_matching': "{:.1%}".format(counter / len(links)), 'conf_matching': "{:.1%}".format(sum_confidence/counter) }

    # Compute percentage matching
    print(str(counter) + ' network links were matched (' + "{:.1%}".format(counter / len(links)) + ' of links) with a '
          + "{:.1%}".format(sum_confidence/counter)+ ' confidence')



    return link_centroids_buffer_gdf


def read_inrix_shp(filepath: str,
                   county: str) -> gpd.GeoDataFrame:

    print('\nReading inrix shapefile of Fresno')

    # Reading shapefiles depicting sets of parts in CA takes time
    california_gdf = gpd.read_file(filepath)

    # Pick a certain county in California
    inrix_gdf = california_gdf[california_gdf.County == county]

    return inrix_gdf

def export_bus_stops_shp(bus_stops_gdf: gpd.GeoDataFrame,
                         folderpath):

    bus_stops_gdf = bus_stops_gdf.to_crs(epsg=config.gis_options['crs_ca'])

    # incidents_gdf.set_crs(config.Config('default').gis_options['crs_ca'])

    bus_stops_gdf.to_file(driver='ESRI Shapefile', filename=folderpath + '/' + 'Fresno' + "_bus_stops.shp")

def export_inrix_shp(inrix_gdf: gpd.GeoDataFrame,
                     folderpath):

    inrix_gdf= inrix_gdf.to_crs(epsg=config.gis_options['crs_ca'])

    # incidents_gdf.set_crs(config.Config('default').gis_options['crs_ca'])

    inrix_gdf.to_file(driver='ESRI Shapefile', filename=folderpath + '/' + 'Fresno' + "_inrix.shp")

def export_buffer_shp(gdf: gpd.GeoDataFrame,
                      folderpath,
                      filename):

    gdf= gdf.to_crs(epsg=config.gis_options['crs_ca'])

    # incidents_gdf.set_crs(config.gis_options['crs_ca'])

    gdf.to_file(driver='ESRI Shapefile', filename=folderpath + '/' + filename)



def read_qgis_shp_fresno(filepath: str) -> gpd.GeoDataFrame:

    print('\nReading network shapefile generated from x,y coordinates and qgis')

    fresno_network_gdf = gpd.read_file(filepath)

    fresno_network_gdf = fresno_network_gdf.set_crs(config.gis_options['crs_ca'])

    # fresno_network_gdf.plot()
    #
    # plt.show()

    ### Read data from PeMS station

    return fresno_network_gdf


def read_pems_stations_fresno(filepath: str = None,
                              adjusted_gis_stations: bool = False):


    if adjusted_gis_stations is False:
        # County fields has the FIPS number but without the state number for california (06).
        # Ref: https://www2.census.gov/geo/docs/reference/codes/files/st06_ca_cou.txt


        fips_fresno = int(addfips.AddFIPS().get_county_fips('Fresno', state='California')[-3:])

        stations_df = pd.read_csv(filepath, header=0, delimiter='\t')

        stations_df = stations_df[stations_df.County == fips_fresno]

        # Create geopandas dataframe
        stations_gdf = gpd.GeoDataFrame(stations_df,
                                        geometry=gpd.points_from_xy(stations_df.Longitude, stations_df.Latitude),
                                        crs=config.gis_options['crs_mercator'])

        stations_gdf = stations_gdf.to_crs(epsg=config.gis_options['crs_ca'])

    else:
        stations_gdf = gpd.read_file(filepath)

    return stations_gdf

    # # Plot
    # stations_gdf.plot(figsize=(5, 5), edgecolor="purple", facecolor="None")

def manual_match_network_and_stations_fresno(network_gdf: gpd.GeoDataFrame,
                                             links: Links) -> None:

    # REplace nan by NOne

    network_gdf = network_gdf.where(pd.notnull(network_gdf), None)

    for link in links:
        link_info = network_gdf.loc[pd.to_numeric(network_gdf['id']) == link.id]

        # bus_stops_network_gdf.loc[bus_stops_network_gdf['key'] == str(link.key)]

        # print(int(link.id))

        if len(link_info)>0:
            pems_id1 = str(list(link_info['pems_id1'])[0])
            pems_id2 = str(list(link_info['pems_id2'])[0])
            pems_id3 = str(list(link_info['pems_id3'])[0])

            # if math.isnan(pems_id1):
            #     pems_id1 = None
            #
            # if np.isnan(pems_id2):
            #     pems_id2 = None
            #
            # if np.isnan(pems_id3):
            #     pems_id3 = None


            # print(pems_id1)
            if pems_id1.isdigit():
                link.pems_stations_ids.append(int(pems_id1))

            if pems_id2.isdigit():
                link.pems_stations_ids.append(int(pems_id2))

            if pems_id3.isdigit():
                link.pems_stations_ids.append(int(pems_id3))
                # print(link.pems_stations_ids)

def match_network_and_stations_fresno(network_gdf: gpd.GeoDataFrame,
                                      stations_gdf: gpd.GeoDataFrame,
                                      links: Links,
                                      folderpath,
                                      adjusted_gis_stations: bool = False):
    # Generate a bounding box to cover only the area covered in the network file
    fresno_network_bbox = shapely.geometry.box(*network_gdf.total_bounds)

    fresno_network_bbox_gdf = gpd.GeoDataFrame(gpd.GeoSeries(fresno_network_bbox),
                                               columns=['geometry'],crs=config.gis_options['crs_ca'])



    if adjusted_gis_stations is False:
        stations_gdf = gpd.sjoin(stations_gdf, fresno_network_bbox_gdf, how="inner",
                                 predicate='intersects').drop(['index_right'],axis=1)

        #Write shapefile with PeMS stations in fresno
        # Save shapefile
        stations_gdf.to_file(driver='ESRI Shapefile', filename= folderpath + '/' + 'fresno' + "_stations.shp")

        print('Shapefile of Fresno PeMS stations written')

    else:
        print('\nReading adjusted shapefile of Fresno PeMS stations')

        stations_gdf = gpd.sjoin(stations_gdf, fresno_network_bbox_gdf, how="inner", predicate='intersects').drop(
            ['index_right'],
            axis=1)

    # Matching PeMS stations to closest link in Fresno network
    closest_network_segment_to_stations = dict(
        zip(stations_gdf['ID'], np.repeat(float('inf'), len(stations_gdf.geometry))))

    # closest_station_to_network_link = dict(zip(network_gdf.index, np.repeat(float('inf'), len(network_gdf.index))))

    # Store minimum distance from segment to a traffic station
    # closest_distance_station_to_network_segment = dict(zip(network_gdf.index, np.repeat(float('inf'), len(network_gdf.index))))

    closest_distance_station_to_network_link = {}
    for link_id in network_gdf['id']:
        closest_distance_station_to_network_link[link_id] = {'station': '', 'min_dist': float('inf')}

    for i, station_id in zip(stations_gdf.geometry, closest_network_segment_to_stations.keys()):
        min_dist = float('inf')

        for l, link_id in zip(network_gdf.geometry, network_gdf['id']):
            dist = i.hausdorff_distance(l)
            # dist = i.distance(j)

            # The closest street segment could have been already matched to a traffic station, and
            # we want to keep the traffic station that is closer to the segment

            if dist < min_dist:
                min_dist = dist
                closest_distance_station_to_network_link[link_id]['station'] = station_id
                closest_distance_station_to_network_link[link_id]['min_dist'] = min_dist

    # Multiple links may be matched to the same station. The link that has the lowest distance to the station is chosen

    unique_closest_network_segment_to_station = dict(
        zip(stations_gdf['ID'], np.repeat(float('inf'), len(stations_gdf.geometry))))

    for station in unique_closest_network_segment_to_station.keys():

        min_dist = float('inf')

        for network_link in closest_distance_station_to_network_link.keys():

            if closest_distance_station_to_network_link[network_link]['station'] == station:
                dist = closest_distance_station_to_network_link[network_link]['min_dist']

                if dist < min_dist:
                    min_dist = dist
                    unique_closest_network_segment_to_station[station] = network_link

    # Remove keys with values infinity
    unique_closest_network_segment_to_station_copy = unique_closest_network_segment_to_station.copy()

    for station in unique_closest_network_segment_to_station_copy.keys():
        if unique_closest_network_segment_to_station_copy[station] == float('inf'):
            del unique_closest_network_segment_to_station[station]

    assert len(set(unique_closest_network_segment_to_station.values())) \
           == len(unique_closest_network_segment_to_station.values()), \
        'there are non unique keys in the matching to stations to links'

    # stations_matched = list(map(lambda x: x['station'], list(closest_distance_station_to_network_link.values())))
    #
    # # Remove cases where no stations were matched
    # [station for station in stations_matched if station != '']

    # a = list(map(lambda x: [y for y in x['station'] if y != ''], list(closest_distance_station_to_network_link.values())))

    # Change coordinates of stations according to the centroid of the network segment that was matched
    # stations_gdf

    # Create a geodataframe for the network segments that were matched
    # matched_network_segments_idx = list(set(list(closest_network_segment_to_stations.values())))
    # matched_network_segments_idx = list(closest_network_segment_to_stations.values())
    # fresno_network_segments_with_stations_gdf = network_gdf.iloc[matched_network_segments_idx]

    # network_links_ids = list(unique_closest_network_segment_to_station.values())

    # fresno_network_segments_with_stations_gdf = network_gdf[network_gdf['id'].isin(network_links_ids)]

    # Plot

    # fig, ax = plt.subplots(figsize=(10, 6))
    #
    # stations_gdf.plot(ax = ax, facecolor="None", edgecolor = 'blue')
    # network_gdf.plot(ax = ax, edgecolor = 'red')
    # fresno_network_segments_with_stations_gdf.plot(ax = ax, edgecolor = 'green')
    #
    # plt.show()

    # read_fresno_qgis_shp().plot()
    # plt.show()
    # #
    # read_PeMS_stations_fresno().plot(color = 'red')
    # plt.show()

    # Set the station id attribute in each link of the network

    # links_ids = unique_closest_network_segment_to_station.keys()
    n_matched_links = 0

    for station_id, link_id in unique_closest_network_segment_to_station.items():
        for link in links:
            if int(link.id) == int(link_id):
                link.pems_stations_ids.append(station_id)
                n_matched_links += 1

    assert len(unique_closest_network_segment_to_station.values()) == n_matched_links, 'some links were not matched'

    print(str(n_matched_links) + ' links were matched (' + "{:.1%}".format(n_matched_links / len(links)) + ' of links)')

def match_network_links_and_fresno_incidents(network_gdf: gpd.GeoDataFrame,
                                             incidents_df: pd.DataFrame,
                                             links: Links,
                                             inrix_matching: bool = False,
                                             buffer_size: float = 0):

    """:param buffer_size: in feets if crs from CA is used"""

    print('Matching incidents (N=' + str(len(incidents_df)) + ') with network links')

    # Create geodataframe
    incidents_gdf = gpd.GeoDataFrame(incidents_df,
                                     geometry=gpd.points_from_xy(incidents_df.Start_Lng, incidents_df.Start_Lat),
                                     crs=config.gis_options['crs_mercator'])

    #Reproject coordinates into CA system
    incidents_gdf = incidents_gdf.to_crs(epsg=config.gis_options['crs_ca'])
    network_gdf = network_gdf.to_crs(epsg=config.gis_options['crs_ca'])

    # Generate buffer around network links
    links_buffer_gdf = network_gdf
    links_buffer_gdf['geometry'] = network_gdf.geometry.buffer(buffer_size)

    # Spatial join of traffic incidents and links geometries
    incidents_network_gdf = gpd.sjoin(links_buffer_gdf, incidents_gdf, how='inner', predicate='intersects')

    # if selected_years is not None:
    #     incidents_network_gdf = incidents_network_gdf.loc[incidents_network_gdf['year'].isin(selected_years)]

    n_years = len(incidents_network_gdf['year'].unique())

    n_matched_links = 0
    n_matched_incidents = 0

    # Add list of incidents in each link

    #Implicitly, the deafult number of incidents is zero as it where the length of the incident list is 0
    for link in links:

        link.incidents_list = []


        if inrix_matching:
            incidents = incidents_network_gdf.loc[incidents_network_gdf['XDSegID'] == str(link.inrix_id)]
        else:
            incidents = incidents_network_gdf.loc[incidents_network_gdf['key'] == str(link.key)]

        for index, incident in incidents.iterrows():
            # print(incident['hour'])
            incident_dict = dict(incident[['Severity', 'year','month','day_month', 'hour','minute','Start_Lat', 'Start_Lng']])

            new_labels = ['severity', 'year','month','day_month', 'hour','minute','lat', 'lon']

            incident_dict = dict(zip(new_labels,  incident_dict.values()))

            link.incidents_list.append(incident_dict)

        link.Z_dict['incidents'] = len(link.incidents_list) #np.round(len(link.incidents_list)/n_years,4)

        if len(link.incidents_list) > 0:
            n_matched_links+= 1
            n_matched_incidents += len(link.incidents_list)

    # gis_results['matching_stats']['incidents'] = {'perc_matching': "{:.1%}".format(n_matched_links / len(links))}

    print(str(n_matched_incidents)+ ' incidents were matched to ' + str(n_matched_links) + ' links  ('
          + "{:.1%}".format(n_matched_links / len(links)) + ' of links)'  )




    # for link in links:
    #     # print(link.Z_dict['incidents_year'])
    #     print(link.incidents_list)

    # Count the number of incidents per link

    # Add attribute to each link in the list


    # #To show in a map
    # incidents_buffer_gdf = incidents_buffer_gdf.to_crs(config.gis_options['crs_mercator'])

    return links_buffer_gdf

def generate_fresno_streets_intersections_gpd(filepath:str):

    print('\nReading shapefiles with street intersections in Fresno')

    streets_intersections_gpd = gpd.read_file(filepath)

    return streets_intersections_gpd


def match_network_links_and_fresno_streets_intersections(network_gdf: gpd.GeoDataFrame,
                                                         streets_intersections_gdf: gpd.GeoDataFrame,
                                                         links: Links,
                                                         inrix_matching: bool = False,
                                                         buffer_size: float = 0):

    """:param buffer_size: in feets if crs from CA is used"""

    print('\nMatching street intersections (N=' + str(len(streets_intersections_gdf)) + ') with network links')

    #Reproject coordinates into CA system
    incidents_gdf = streets_intersections_gdf.to_crs(epsg=config.gis_options['crs_ca'])
    network_gdf = network_gdf.to_crs(epsg=config.gis_options['crs_ca'])

    # Generate buffer around network links
    links_buffer_gdf = network_gdf
    links_buffer_gdf['geometry'] = network_gdf.geometry.buffer(buffer_size)

    # Spatial join of traffic incidents and links geometries
    streets_intersections_network_gdf = gpd.sjoin(links_buffer_gdf, incidents_gdf, how='inner', predicate='intersects')

    n_matched_links = 0
    n_matched_streets_intersections = 0

    # Add list of incidents in each link
    for link in links:

        link.incidents_list = []

        if inrix_matching:
            streets_intersections = streets_intersections_network_gdf.loc[
                streets_intersections_network_gdf['XDSegID'] == str(link.inrix_id)]

        else:
            streets_intersections = streets_intersections_network_gdf.loc[streets_intersections_network_gdf['key'] == str(link.key)]

        for index, streets_intersection in streets_intersections.iterrows():

            streets_intersection_dict = dict(
                streets_intersection[
                    ['STRNAME1', 'STRNAME2']])

            streets_intersection_dict = dict(
                zip(streets_intersection_dict.keys(), streets_intersection_dict.values()))


            link.streets_intersections_list.append(streets_intersection_dict)

        link.Z_dict['intersections'] = len(link.streets_intersections_list)



        if len(link.streets_intersections_list) > 0:
            n_matched_links+= 1
            n_matched_streets_intersections += len(link.streets_intersections_list)

    # config.gis_results['matching_stats']['streets_intersections'] = {'perc_matching': "{:.1%}".format(n_matched_links / len(links))}

    print(str(n_matched_streets_intersections)+ ' street intersecions were matched to ' + str(n_matched_links) + ' links (' + "{:.1%}".format(n_matched_links / len(links)) + ' of links)'  )



    # #To show in a map
    # incidents_buffer_gdf = incidents_buffer_gdf.to_crs(config.gis_options['crs_mercator'])

    return links_buffer_gdf


def match_network_links_and_fresno_bus_stops(network_gdf: gpd.GeoDataFrame,
                                             bus_stops_gdf: gpd.GeoDataFrame,
                                             links: Links,
                                             inrix_matching: bool = False,
                                             buffer_size: float = 0):

    """:param buffer_size: in feets if crs from CA is used"""

    print('\nMatching bus stops (N=' + str(len(bus_stops_gdf)) + ') with network links')

    #Reproject coordinates into CA system
    incidents_gdf = bus_stops_gdf.to_crs(epsg=config.gis_options['crs_ca'])
    network_gdf = network_gdf.to_crs(epsg=config.gis_options['crs_ca'])

    # Generate buffer around network links
    links_buffer_gdf = network_gdf
    links_buffer_gdf['geometry'] = network_gdf.geometry.buffer(buffer_size)

    # Spatial join of traffic incidents and links geometries
    bus_stops_network_gdf = gpd.sjoin(links_buffer_gdf, incidents_gdf, how='inner', predicate='intersects')

    n_matched_links = 0
    n_matched_bus_stops = 0

    # Add list of incidents in each link
    for link in links:

        link.bus_stops_list = []
        link.Z_dict['bus_stops'] = 0


        if inrix_matching:
            bus_stops = bus_stops_network_gdf.loc[bus_stops_network_gdf['XDSegID'] == str(link.inrix_id)]
        else:
            bus_stops = bus_stops_network_gdf.loc[bus_stops_network_gdf['key'] == str(link.key)]

        for index, bus_stop in bus_stops.iterrows():

            bus_stop_dict = dict(
                bus_stop[
                    ['stop_id', 'stop_code', 'stop_name', 'stop_desc', 'stop_lat', 'stop_lon', 'zone_id', 'stop_url',
                     'location_type', 'parent_station', 'stop_timezone', 'wheelchair_boarding']])

            bus_stop_dict = dict(zip(bus_stop_dict.keys(), bus_stop_dict.values()))

            link.bus_stops_list.append(bus_stop_dict)

        link.Z_dict['bus_stops'] = len(link.bus_stops_list)

        if len(link.bus_stops_list) > 0:
            n_matched_links+= 1
            n_matched_bus_stops += len(link.bus_stops_list)

    # config.gis_results['matching_stats']['bus_stops'] = {'perc_matching': "{:.1%}".format(n_matched_links / len(links))}

    print(str(n_matched_bus_stops)+ ' bus stops were matched to ' + str(n_matched_links) + ' links (' + "{:.1%}".format(n_matched_links / len(links)) + ' of links)'  )

    # #To show in a map
    # incidents_buffer_gdf = incidents_buffer_gdf.to_crs(config.gis_options['crs_mercator'])

    return links_buffer_gdf

def generate_fresno_bus_stops_gpd(bus_stops_df: pd.Dataframe,
                                  export_filepath: str = None):

    bus_stops_gdf = gpd.GeoDataFrame(bus_stops_df,
                                     geometry=gpd.points_from_xy(bus_stops_df.stop_lon, bus_stops_df.stop_lat),
                                     crs=config.gis_options['crs_mercator'])

    bus_stops_gdf = bus_stops_gdf.to_crs(epsg=config.gis_options['crs_ca'])

    # incidents_gdf.set_crs(config.gis_options['crs_ca'])

    if export_filepath is not None:

        bus_stops_gdf.to_file(driver='ESRI Shapefile', filename=export_filepath)

    return bus_stops_gdf


def export_fresno_incidents_shp(incidents_df: pd.Dataframe, folderpath):

    incidents_gdf = gpd.GeoDataFrame(incidents_df,
                                     geometry=gpd.points_from_xy(incidents_df.Start_Lng, incidents_df.Start_Lat),
                                     crs=config.gis_options['crs_mercator'])

    incidents_gdf = incidents_gdf.to_crs(epsg=config.gis_options['crs_ca'])

    # incidents_gdf.set_crs(config.gis_options['crs_ca'])

    incidents_gdf.to_file(driver='ESRI Shapefile', filename=folderpath + '/' + 'Fresno' + "_incidents.shp")


def show_folium_map(map, filename, path='/plots/maps'):
    chrome_path = 'open -a /Applications/Google\ Chrome.app %s'

    # print(os.getcwd() + path + filename)
    filename_path = os.getcwd() + path + filename + '.html'
    map.save(filename_path)
    webbrowser.get(chrome_path).open(filename_path)  # open in new tab
