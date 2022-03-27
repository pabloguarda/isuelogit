import isuelogit as isl
import osmnx as ox
import geopandas as gpd
import networkx as nx
import folium

from pyspark.sql import functions as F

import matplotlib.pyplot as plt

import shapely.geometry


path_data_folder = "/Users/pablo/google-drive/university/cmu/2-research/datasets/"

#=============================================================================
# SPARK (environment setup)
#=============================================================================

data_analyst = isl.analyst.DataReader()

# sc, sqlContext = data_analyst.setup_spark_context()

#===================================================================
# SHAPE FILE (INRIX)
#===================================================================

# path_inrix_shp = path_data_folder + 'private/inrix/fresno/shapefiles/USA_CA_BayArea_shp/USA_CA_BayArea.shp'
#
path_inrix_shp = path_data_folder + 'private/inrix/fresno/shapefiles/USA_CA_RestOfState_shapefile/USA_CA_RestOfState.shp'

inrix_gdf = isl.geographer.read_inrix_shp(filepath = path_inrix_shp, county ='Fresno')

inrix_gdf.plot(figsize=(5, 5), edgecolor="purple", facecolor="None")

plt.show()

inrix_gdf.keys()

#TODO: add data from the average speed of one day and show color depending of the level of speed in each segment (choropleth map)


#===================================================================
# SHAPE FILE (OSMNX)
#===================================================================

# TODO: Obtain positions from street intersections in OSMNX and map them into INRIX shapefile.
#
#  TODO: The alternative approach is to add census information in traffic counters in CA, as well as amenities information

# TODO: let's pick the osmnx street segments, create a buffer around them and intersect with inrix segments
#

# With query 'Fresno, USA', only data from Fresno city is used. It takes time to download data for the whole county.
osm_graph = ox.graph_from_place('Fresno, USA', network_type='drive', custom_filter= '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary_link"]')
# osm_graph = ox.graph_from_place('Fresno, USA', network_type='drive')
ox.plot_graph(osm_graph, bgcolor="white", edge_color='blue')

# Consolidate intersections
Gc = ox.consolidate_intersections(ox.project_graph(osm_graph), tolerance=10, rebuild_graph=True, dead_ends=False)
ox.plot_graph(Gc, bgcolor="white", edge_color='blue')

Gc = ox.utils_graph.get_largest_component(osm_graph, strongly=True)
ox.plot_graph(Gc, bgcolor="white", edge_color='blue')
# isl.geographer.show_folium_map(ox.plot_graph_folium(Gc), 'test')


m = folium.Map(tiles='openstreetmap', zoom_start = 10)
folium.GeoJson(inrix_gdf).add_to(m)
# isl.geographer.show_folium_map(m, 'test')

# ox.plot_graph(ox.graph_from_place('Carnegie Mellon, Pittsburgh, USA', network_type='drive',buffer_dist=1000))

# ox.plot_graph(ox.graph_from_place('Pittsburgh, USA', buffer_dist=1000), bgcolor="white", edge_color='blue', save=True, filepath = 'plots/PittsburghOSMNX.png')
#
#
# G = ox.graph_from_place('Concepcion, Chile', network_type='drive')
# m = ox.plot_graph_folium(G)

osm_nodes_gdf, osm_gdf = ox.graph_to_gdfs(osm_graph) # This method returns nodes, edges.

# fresno_osm_graph = ox.graph_from_place('Fresno, USA', network_type='drive', custom_filter= '["highway"~"motorway|motorway_link|primary"]')
# fresno_osm_graph = ox.graph_from_place('Fresno, USA', network_type='drive')

#TODO: add data from the average speed of one day and show color depending of the level of speed in each segment (choropleth map)

#===================================================================
# SHAPE FILE (PEMS stations)
#===================================================================

# This is not a shapefile but a sort of dataset of the stations
path_pems_stations = 'data/public/pems/stations/raw/D06/'\
                               + 'd06_text_meta_2020_09_11.txt'

# stations_df = pd.read_csv(path_pems_stations, header =0, delimiter = '\t')
#
# # County fields has the FIPS number but without the state number for california (06). Ref: https://www2.census.gov/geo/docs/reference/codes/files/st06_ca_cou.txt
#
# fips_fresno = int(addfips.AddFIPS().get_county_fips('Fresno', state = 'California')[-3:])
#
# stations_df = stations_df[stations_df.County == fips_fresno]
#
# #Create geopandas dataframe
# stations_gdf = gpd.GeoDataFrame(stations_df, geometry = gpd.points_from_xy(stations_df.Longitude,stations_df.Latitude), crs = inrix_gdf.crs)

stations_gdf = isl.geographer.read_pems_stations_fresno(path_pems_stations)

#Plot
stations_gdf.plot(figsize=(5,5), edgecolor="purple", facecolor="None")

plt.show()

#=============================================================================
# PEMS traffic count
#=============================================================================

#  Download PEMS traffic count
# data_analyst.download_pems_data()

# Local data folder within project directory
path_pems_counts = 'data/public/pems/counts/raw/d06/' + \
                   'd06_text_station_5min_2020_10_01.txt.gz'

count_interval_df = data_analyst.read_pems_counts_by_period(filepath = path_pems_counts)

#=============================================================================
# SPEED (INRIX)
#=============================================================================

# Weiran paper about Inrix data
# https://arxiv.org/pdf/2005.13522.pdf
# Another cool reference working with Inrix, gps data
# https://www.sciencedirect.com/science/article/pii/S2352146520303355/pdf?md5=9031ace2965fc8f78a2173cffd873ef3&pid=1-s2.0-S2352146520303355-main.pdf
#

#TODO: there are two folders with files for the same period. I am not sure if they are complementary  of if one have data more recent than the other. I just picked one file at random

path_inrix_speed = path_data_folder + 'private/inrix/fresno/speed/Fresno_CA_2019-10-01_to_2019-11-01_15_min_part_1/data.csv'

speed_sdf = data_analyst.generate_inrix_data_by_segment(filepath= path_inrix_speed)

speed_sdf.head()

# https://medium.com/expedia-group-tech/deep-dive-into-apache-spark-datetime-functions-b66de737950a

# speed_sdf.select('dt').withColumn('month', F.dayofmonth(speed_sdf['dt'])).distinct().collect()

# #Create year,month and day of the month fields
# speed_sdf = speed_sdf.withColumn('year', F.year(speed_sdf['dt']))
# speed_sdf = speed_sdf.withColumn('month', F.month(speed_sdf['dt']))
# speed_sdf = speed_sdf.withColumn('day_of_month', F.dayofmonth(speed_sdf['dt']))

# speed_sdf.head()

# Number of observations per year
# speed_sdf.groupby('year').count().show()

# Difference in average speed between 2019 and 2020
# speed_sdf.groupby('year').agg(F.mean('speed')).show()

speed_day_sdf = speed_sdf.filter(speed_sdf['dt'] == F.lit('2020-10-01')).cache()

#Compute averages over 5 minutes interval of a given day
speed_interval_sdf = speed_sdf.groupby('segment_id').agg(F.mean('speed').alias('speed_avg'),F.stddev('speed').alias('speed_sd')).cache()

speed_interval_df = speed_interval_sdf.toPandas()

speed_interval_df.count()

#=============================================================================
# MASKING
#=============================================================================

#Coordinate systems

# 'California State Plane Zone 4' system is epsg: 2228. It is in feets. 1ft is 0.3 mt
crs_ca = 2228
crs_mercator = 4326

# Data from fresno city will be used only. The margins are obtained from Fresno city shp in osm

# https://gis.stackexchange.com/questions/266730/filter-by-bounding-box-in-geopandas/266833
# osm_bb_gdf = gpd.GeoDataFrame(gpd.GeoSeries(osm_gdf.envelope),columns = ['geometry'])
bb = shapely.geometry.box(*osm_gdf.total_bounds)
osm_bb_gdf = gpd.GeoDataFrame(gpd.GeoSeries(bb),columns = ['geometry'], crs = crs_mercator)

# osm_bb_gdf.plot()
# plt.show()

# osm_gdf.plot()
# plt.show()

# Masking
inrix_gdf = gpd.sjoin(inrix_gdf, osm_bb_gdf, how = "inner", op = 'intersects')
# pems_gdf = gpd.sjoin(pems_gdf, osm_bb_gdf, how = "inner", op = 'intersects')

# # Need to remove this columns to perform further spatial joins
inrix_gdf = inrix_gdf.drop(['index_right'], axis = 1)
# pems_gdf = pems_gdf.drop(['index_right'], axis = 1)

# pems_test_gdf.plot()
# inrix_test_gdf.plot()
# plt.show()

#=============================================================================
# NETWORK REDUCTION
#=============================================================================

# i) Buffer around INRIX segments

inrix_buffer_gdf = inrix_gdf.to_crs(crs_ca)
inrix_buffer_gdf['geometry'] = inrix_buffer_gdf.geometry.buffer(10)
inrix_buffer_gdf = inrix_buffer_gdf.to_crs(crs_mercator)

# inrix_gdf.plot()
# plt.show()

location_center = list(inrix_buffer_gdf.unary_union.centroid.coords)[0]

m = folium.Map(tiles='openstreetmap', zoom_start = 10, location = list(reversed(location_center)))
folium.GeoJson(inrix_buffer_gdf).add_to(m)
# isl.geographer.show_folium_map(m,'test')

# ii) Spatial join between inrix and osm
#
inrix_osm_gdf = gpd.sjoin(osm_gdf, inrix_buffer_gdf, how = 'inner', op = 'intersects')

location_center2 = list(inrix_osm_gdf.unary_union.centroid.coords)[0]

m2 = folium.Map(tiles='openstreetmap', zoom_start = 11, location = list(reversed(location_center2)))
folium.GeoJson(inrix_osm_gdf).add_to(m2)
# isl.geographer.show_folium_map(m2,'test')

# inrix_osm_gdf.plot()
# plt.show()

# osm_gdf.plot()
# plt.show()

location_center3 = list(osm_gdf.unary_union.centroid.coords)[0]
m3 = folium.Map(tiles='openstreetmap', zoom_start = 11, location = list(reversed(location_center3)))
folium.GeoJson(osm_gdf).add_to(m3)
# isl.geographer.show_folium_map(m3,'test')


# iii) Remove dupplicates
# https://github.com/geopandas/geopandas/issues/1265

# - start with naive way of just picking the first intersection (TODO: maybe use overlay function and generating buffer around osmnx street segments as well?? https://gis.stackexchange.com/questions/375844/removing-one-to-many-links-when-performing-spatial-join-in-geopandas)
inrix_osm_gdf = inrix_osm_gdf.groupby(inrix_osm_gdf.index).first()

inrix_osm_gdf.plot()
plt.show()

# The highway attribute needs to be flatten as some entries have two different types of highway because it is a two way link

def flatten(L):
    for item in L:
        if isinstance(item, list):
            for subitem in item:
                yield subitem
        else:
            yield item

# https://wiki.openstreetmap.org/wiki/Key:highway
highway_types = set(list(flatten(list(inrix_osm_gdf["highway"].values))))
main_highway_types = ['motorway', 'trunk', 'primary']

# Get list of link that correspond to any of these highway types
main_highways_links = []
for highway_type,link in zip(inrix_osm_gdf["highway"], inrix_osm_gdf.index):
    if highway_type in main_highway_types:
        main_highways_links.append(link)

# Note that most of links have no speed information
list(inrix_osm_gdf["maxspeed"].values)[96]

list(inrix_osm_gdf["oneway"].values)[97]

inrix_osm_gdf.values

'primary', 'primary_link',



inrix_osm_gdf["highway"].values

# - Get largest strongly connected component of the network
# https://stackoverflow.com/questions/62532373/plot-only-the-strongly-connected-components-of-a-osmnx-network

Gc = ox.utils_graph.get_largest_component(osm_graph, strongly=True)
print(len(Gc)) #9503
fig, ax = ox.plot_graph(Gc, node_size=0)

#=============================================================================
# BASE SHAPEFILE
#=============================================================================


# Inrix shapefile is used for the base of the analysis
# fresno_gdf = inrix_gdf
fresno_gdf = osm_gdf

#=============================================================================
# CENSUS DATA
#=============================================================================

# Census data will allow to depict nodes/street characteristics

# https://pypi.org/project/CensusData/
#https://jtleider.github.io/censusdata/

# Reading census divisions

census_divisions_shp_path = 'data/public/census/tl_2020_us_cd116/' + \
                 'tl_2020_us_cd116.shp'

# This convers the entire US
census_divisions_gdf = gpd.read_file(census_divisions_shp_path)

census_divisions_gdf.crs

# Census tract
census_tracts_shp_path = 'data/public/census/tl_2020_06_tract/' + \
                 'tl_2020_06_tract.shp'

census_tract_gdf = gpd.read_file(census_tracts_shp_path)

# census_tract_gdf.plot()
# plt.show()

#=============================================================================
# MERGES
#=============================================================================

# - Add counts to stations gpd
pems_gdf = stations_gdf.merge(count_interval_df, left_on = 'ID', right_on = 'station_id')
# - Add speed to INRIX gpd

#Remove anyoing warning
# pd.options.mode.chained_assignment = None
inrix_gdf.XDSegID = inrix_gdf.XDSegID.astype(int)

inrix_gdf = inrix_gdf.merge(speed_interval_df, left_on = 'XDSegID', right_on = 'segment_id')

#=============================================================================
# SPATIAL JOINS
#=============================================================================

# Perform a spatial join to add traffic counts to inrix dataset.
pems_gdf.keys()

# Create a buffer around station and then intersect with street segment. It is important to project to a more fine resolution projection system so the buffer calculation is more precise
# pems_gdf= gpd.GeoDataFrame(pems_gdf,geometry = pems_gdf['geometry'].buffer(0.1))
pems_buffer_gdf = pems_gdf.to_crs(epsg = crs_ca)

# pems_buffer_gdf.crs
# gpd.crs
pems_buffer_gdf['geometry'] = pems_buffer_gdf.geometry.buffer(30)

#Project back
pems_buffer_gdf = pems_buffer_gdf.to_crs(epsg = crs_mercator)

# pems_gdf.crs

# pems_gdf_buffer.plot()
# fig, ax = plt.subplots(1, figsize=(10, 6))
pems_buffer_gdf.plot(facecolor="None", edgecolor="blue")
# pems_gdf.plot(edgecolor="blue", marker='.')
# plt.show()


# Join station data in inrix street segments
fresno_gdf = gpd.sjoin(fresno_gdf, pems_buffer_gdf, how = "left", op = 'intersects')
fresno_gdf = fresno_gdf.drop(['index_right'], axis = 1)

# fresno_gdf.plot()
# plt.show()

# print(fresno_gdf.station_id)
#
# test = pd.DataFrame(fresno_gdf)
#
# test1 = test[test.station_id.notnull()]
#
# print(test1.count)

# Join information of census divisions to use information from nhts

# test = gpd.sjoin(fresno_gdf, census_divisions_gdf.to_crs(crs_mercator), how = "left", op = 'intersects')

# test = gpd.sjoin(census_divisions_gdf.to_crs(crs_mercator),fresno_gdf,  how = "inner", op = 'intersects')

test = gpd.sjoin(census_tract_gdf.to_crs(crs_mercator),fresno_gdf,  how = "inner", op = 'intersects')

# Map pems stations to osm
fresno_pems_gdf = gpd.sjoin(osm_gdf, pems_buffer_gdf, how = "left", op = 'intersects')

#Go back to osmnx


#=============================================================================
# MATCHING INRIX TO OSM
#=============================================================================

# Representative point may be better as it is much cheaper to compute

# https://gis.stackexchange.com/questions/373547/creating-centroid-inside-of-polygon

test_x = osm_gdf.centroid.map(lambda p: p.x).keys()


# Transform inrix gpd into points corresponding to segments centroids:


# osm_bb_gdf = gpd.GeoDataFrame(gpd.GeoSeries(bb),columns = ['geometry'], crs = crs_mercator)

inrix_centroids_gdf = inrix_gdf.to_crs(crs = crs_ca)

inrix_centroids_gdf['geometry'] = inrix_centroids_gdf.centroid

# inrix_centroids_gdf = gpd.GeoDataFrame(inrix_centroids_gdf,columns = ['geometry'])

inrix_centroids_gdf = inrix_centroids_gdf.to_crs(crs_mercator)

inrix_centroids_gdf.plot()
# plt.show()

# Perform spatial join with osm data using a buffer
inrix_centroids_buffer_gdf = inrix_centroids_gdf.to_crs(epsg = crs_ca)

# pems_buffer_gdf.crs
# gpd.crs
inrix_centroids_buffer_gdf['geometry'] = inrix_centroids_buffer_gdf.geometry.buffer(30)

#Project back
inrix_centroids_buffer_gdf =inrix_centroids_buffer_gdf.to_crs(epsg = crs_mercator)

# Join inrix street centroids with buffer to osmn data

fresno_gdf = gpd.sjoin(fresno_gdf, inrix_centroids_buffer_gdf,  how = "left", op = 'intersects').drop(['index_right'], axis = 1)

fresno_gdf.keys()


# fresno_gdf.plot()
# plt.show()


#=============================================================================
# GDF to GRAPH (with OSMNX support)
#=============================================================================

# Transform gpd to osmnx graph

#graph attrs is helpful to integrate the attribtues values from the original osmnx object
fresno_graph = ox.graph_from_gdfs(gdf_edges=fresno_gdf, gdf_nodes=osm_nodes_gdf, graph_attrs=osm_graph.graph)


#Adding information about flow average into the graph generated by osmnx
attrs_values = fresno_gdf.values[:,list(fresno_gdf.keys()).index('flow_avg')]
nx.set_edge_attributes(fresno_graph, dict(zip(fresno_gdf.index, list(attrs_values))), 'flow_avg')
# fresno_graph.edges(data = True)

# Speed
attrs_values = fresno_gdf.values[:,list(fresno_gdf.keys()).index('speed_avg')]
nx.set_edge_attributes(fresno_graph, dict(zip(fresno_gdf.index, list(attrs_values))), 'speed_avg')

# print(fresno_graph.edges(data= True))

# Check the number of street segments with no records of speed
# x = fresno_gdf.values[:,list(fresno_gdf.keys()).index('speed_avg')]
# # Remove NA counts
# b = [z for z in x if not math.isnan(z)]
# # len(b)


# fresno_gdf.plot()
# plt.show()

fresno_graph = ox.graph_from_gdfs(gdf_edges=ox.graph_to_gdfs(osm_graph)[1], gdf_nodes=osm_nodes_gdf, graph_attrs=osm_graph.graph)
ox.plot_graph(fresno_graph, bgcolor="white", edge_color='black')
# plt.show()

# isl.geographer.show_folium_map(ox.plot_graph_folium(fresno_graph), filename = 'test1')


#=============================================================================
# PATH VISUALIZATION
#=============================================================================

origin_point = ox.graph_from_address('California State University', return_coords = True)[1]
destination_point = ox.graph_from_address('Airport, Fresno city', return_coords = True)[1]

orig = ox.get_nearest_node(fresno_graph, origin_point)
dest = ox.get_nearest_node(fresno_graph, destination_point)

# route = nx.shortest_path(fresno_graph, orig, dest, 'travel_time')
# route = nx.shortest_path(fresno_graph, dest, orig, 'speed_avg')
route = nx.shortest_path(fresno_graph, dest, orig, 'speed')

fresno_graph.edges(data = True)

ox.plot_graph_route(fresno_graph,route, bgcolor="white", edge_color='blue')


route_map = ox.plot_route_folium(fresno_graph, route)
#
# m = folium.Map()
# folium.Choropleth(test, data=test['flow_avg'],
#              columns=['flow_avg'], fill_color='YlOrBr').add_to(m)

# isl.show_folium_map(route_map, filename = 'test')



inrix_gdf.keys()

inrix_gdf['RoadNumber'].unique()


#=============================================================================
# Visualization plotting the different layers (inrix segments, station points)
#=============================================================================

# Map with stations
fig, ax = plt.subplots(1, figsize=(10, 6))
# test.plot(ax = ax, edgecolor="black", facecolor="None")
# fresno_gdf.plot(ax = ax, edgecolor="purple", facecolor="None")
pems_gdf.plot(ax = ax, edgecolor="green", facecolor="None")
# plt.show()

# Map with stations and osm segments
fig, ax = plt.subplots(1, figsize=(10, 6))
# fresno_pems_gdf.plot(ax = ax, edgecolor="black", facecolor="None")
# fresno_pems_gdf.plot(ax = ax, edgecolor="purple", facecolor="None")
fresno_gdf.plot(ax = ax, column = 'flow_avg')
# pems_gdf.plot(ax = ax, edgecolor="blue", facecolor="None")
# plt.show()

# Map with average counts mapped to INRIX segments

# fresno_gdf.plot()
fresno_gdf.plot(column = 'flow_avg')
# plt.show()


# Map with average speeds obtained from Inrix

fig, ax = plt.subplots(1, figsize=(10, 6))

# test_shp.plot(column = 'speed_avg', ax = ax)

# pems_gdf.plot(ax = ax, edgecolor="blue", facecolor="None")
# test.plot(ax = ax, edgecolor="black", facecolor="None")
# pems_gdf.plot(ax = ax, edgecolor="blue", facecolor="None")

# inrix_gdf.plot(ax = ax, edgecolor="gray", facecolor="None")

fresno_gdf.plot(ax = ax)

fresno_gdf.plot(column = 'flow_avg', ax = ax
              , cmap = 'Reds'
              , scheme='quantiles'
              , k=10)

# fresno_shp.plot(column = 'segment_id', ax = ax)
# fresno_shp.plot(column = 'XDSegID', ax = ax)

# fresno_shp.plot(figsize=(5,5), edgecolor="purple", facecolor="None")

# plt.show()

# def show_folium_map(map, filename, path = '/plots/maps'):
#
#     chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
#
#     # print(os.getcwd() + path + filename)
#     filename_path = os.getcwd() + path + filename + '.html'
#     map.save(filename_path)
#     webbrowser.get(chrome_path).open(filename_path )  # open in new tab


m = folium.Map()
folium.Choropleth(test, data=test['flow_avg'],
             columns=['flow_avg'], fill_color='YlOrBr').add_to(m)

isl.geographer.show_folium_map(m, filename = 'test')


#=============================================================================
# EXTRA ANALYSIS GIS
#=============================================================================

# TRANSFORMING POINTS TO SEGMENTS
# http://ryan-m-cooper.com/blog/gps-points-to-line-segments.html

# Nearest (spatial) join as a new feature to geopandas?
# https://github.com/geopandas/geopandas/issues/1096

# Look at GIS F2E tool crated at the university of chigago to transform shapefiles into node-edge structure:

# https://github.com/csunlab/GISF2E/tree/master/Python/v1.20
#https://www.nature.com/articles/sdata201646

# However, G. Boeing critices their approach here https://arxiv.org/pdf/1611.01890.pdf saying that
# - there is arbitrary break points between OpenStreetMap IDs or line digitization
# - , it discards to and from nodes, thus making it unclear in which direction the one-way goes

#=============================================================================


#=============================================================================
#  DATA IMPUTATION (what about KNN?)
#=============================================================================

# KERNEL DENSITY?
# https://residentmario.github.io/geoplot/plot_references/plot_reference.html#kdeplot

# The most simple thing to do would be to impute the maximum speed in the street segment
# MATCHING
#=============================================================================

# Note that line to line matching is missing in osmnx

# Resource: https://geoffboeing.com/2016/10/r-tree-spatial-index-python/
# https://gis.stackexchange.com/questions/204361/find-a-linestring-closest-to-a-given-linestring

#=============================================================================
# MATCHING INRIX WITH OMS
#=============================================================================


# I would like to propose an algorithm to match Inrix speed data to osm:
#
# - Pick a route in osm. 
# - Generate a gps trace where each waypoint is the centroid of an osm street segment
# - Find the nearest line segments to those points in the Inrix shapefile
# - Return a dictionary with the inrix line segments ids and their corresponding speeds
#
# I think this algorithm may be very similar to the one needed when working with gps data later on. The difference is that we will have gps traces and we will need to call our method to decompose the gps traces into segments
#
#
# More detailed:
#
# # TODO: query travel times in inrix shapefile but working from oms. The procedure is (i) a path between an od is generated using oms and I transformed it in a collection of segments. (ii) For each osm segment in the path, the centroid is computed. (iii) We build a buffer around each centroid and intersect it with the inrix segments. (iv) After matching, the speed associated to the centroid point of the osm segment is equated to the speed in the entire segment of the path generated with osm in (i). (v) a hash table that matches inrix id segments and osmn ids is created so further route queries do not require to perform this expensive GIS operation. vi) We repeat this for every segment of the paths within the path set of every OD pair.



#=============================================================================
# National Household Travel Survey
#=============================================================================

#Source: https://nhts.ornl.gov/
# Documentation with FAQS: https://nhts.ornl.gov/faq
#Use Respective Census Division (CENSUS_D) as key. The problem is that the areas are too large