# =============================================================================
# Network modelling with transportAI
# =============================================================================

import os
import pandas as pd
import numpy as np

#import pylogit as pl

#Garbage collection
import gc
gc.enable()
gc.collect()
os.getcwd()

# =============================================================================
# Data Reading
# =============================================================================

## Select a random sample of trips
# df = df_raw.sample(n = 10000)
subdirectory = "examples/data"
subfolder = "/beijing-subway/"
folder_path = os.getcwd() + "/" + subdirectory + subfolder

# raw_df = tt.reader.read_beijing_data(folder_path=folder_path, filenames = ['7.csv'])
raw_df = transportAI.extra.beijing.read_beijing_data(folder_path=folder_path)

# =============================================================================
# Codebook
# =============================================================================
df_dict = transportAI.extra.beijing.get_dictionary_beijing_data()

# =============================================================================
# Subset of data
# =============================================================================

import random
#raw_df = reader.read_beijing_data()
# df = raw_df.iloc[:1000]

df = raw_df.loc[(raw_df.oline.isin([1,2,3,4,5])) & (raw_df.dline.isin([1,2,3,4,5]))]

# # b = df.head()
# # print(df)

df = df.iloc[random.sample(range(1, len(df)), 10000)]

# print(df.head())

# =============================================================================
# Translation from chinese to english
# =============================================================================
# from googletrans import Translator
# translator = Translator()
#
# # Unique station names
# station_names = raw_df['ostationname'].unique()
#
# a = [translator.translate(word, src = 'zh-cn') for word in station_names]
#
# proxy = {
#         'http': 'http://username:password@1.1.1.1:1234',
#         'https': 'http://username:password@1.1.1.1:1234',
# }
#
# translator1 = Translator(proxies = proxy)
# result = translator1.translate(raw_df.iloc[4]['ostationname'], src = 'zh-cn', dest='en')
#
# a[0].pronunciation
# a[0].text
#
# for word in a:
#     print(word.text)
#
# result
#
# df = df.merge(a,)

# =============================================================================
# Gis data from Beijing
# =============================================================================

import transportAI as tt

# r = tt.beijing.request_gis_data(url = "http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json")
r = transportAI.extra.beijing.request_gis_data(url ="") # Go and read json local file

lines_info ,stations_info = transportAI.extra.beijing.get_lines_stations_info(r.text)
# print(lines_info,'\n',stations_info)
# 建立邻接表
neighbor_info = transportAI.extra.beijing.get_neighbor_info(lines_info)
print(neighbor_info)

# # Very slow
# paths = tt.beijing.get_path_DFS_ALL(lines_info, neighbor_info, '回龙观', '西二旗')

# Unique station names
station_names = df['ostationname'].unique()

ostation = station_names[0]
dstation = station_names[1]

transportAI.extra.beijing.get_distance(stations_info, ostation, dstation)
final_route = transportAI.extra.beijing.get_path_BFS(neighbor_info, lines_info, ostation, dstation)

# print('共有{}种路径。'.format(len(paths)))
# for item in paths:
#     print("此路径总计{}站:".format(len(item)-1))
#     print('-'.join(item))

# 第二种算法：没有启发函数的简单宽度优先

paths = transportAI.extra.beijing.get_path_Astar(lines_info, neighbor_info, stations_info, '回龙观', '西二旗')
if paths:
    print("路径总计{}d站。".format(len(paths) - 1))
    print("-".join(paths))

# tt.beijing.plot_network_and_labels(stations_info = stations_info, neighbor_info = neighbor_info)


# =============================================================================
# Rewrite file
# =============================================================================


# =============================================================================
# Variables creation
# =============================================================================

## Position of the stations (latitude and longitude)
stations_df = pd.DataFrame()
stations_df['name'] = stations_info.keys()
stations_df['lat'] = [x for (x,y) in list(stations_info.values())]
stations_df['lon'] = [y for (x,y) in list(stations_info.values())]

df = pd.merge(df,stations_df, how = 'left', left_on = 'ostationname', right_on = 'name').rename(columns = {'lat': 'ostation_lat', 'lon':'ostation_lon'}).drop('name', axis = 1)
df = pd.merge(df,stations_df, how = 'left', left_on = 'dstationname', right_on = 'name').rename(columns = {'lat': 'dstation_lat', 'lon':'dstation_lon'}).drop('name', axis = 1)

# df['aerial_distance'] = 1000*((df['ostation_x']-df['dstation_x'])**2 + (df['ostation_y']-df['dstation_y'])**2)**(0.5)
# x1,y1, x2, y2 = df['ostation_lat'].iloc[0],df['ostation_lon'].iloc[0],df['dstation_lat'].iloc[0],df['dstation_lon'].iloc[0]
# tt.beijing.get_distance_metres(x1,y1, x2, y2)

df['aerial_distance'] = [transportAI.extra.beijing.get_distance_metres(*row) for row in df[['ostation_lat', 'ostation_lon', 'dstation_lat', 'dstation_lon']].values]

#Sometimes the aerial distance is greater than distance
# np.mean(df['aerial_distance'])
# np.mean(df['distance'])

# Review open street map and osmnx package from Boeing. There should be data from the beijing subway there

# =============================================================================
# Create OD Matrix
# =============================================================================

# travellers = tt.create_agents(beijing_df=df, type = 'travellers')
# od = tt.create_od(trips = travellers, from_ = 'agents')

od = tt.create_od(trips = df, from_ = 'system')

print(od)

n_origins = df['ostation'].unique()
n_destinations = df['dstation'].unique()

print(len(n_origins), len(n_destinations))

od = pd.pivot_table(df[['ostation','dstation']] ,index=['ostation'], columns = ['dstation'], aggfunc = [len])
od = od.replace(np.nan,0)

np.sum(od.to_numpy()) #Checking the total number of trips is consistent with the original dataset


# =============================================================================
# Create OD Matrix of travel times
# =============================================================================

od_tt = pd.pivot_table(df[['ostation','dstation','tt']] ,index=['ostation'], columns = ['dstation'], values = ['tt'], aggfunc = [np.mean])
od_tt = od_tt.replace(np.nan,0)

od_tt_ij = df[(df['ostation'] == 1) & (df['dstation'] == 20)]['tt']

# df[(df['ostation'] == 1) & (df['dstation'] == 20)]['tt'].hist()
od_tt_ij.hist()
np.mean(od_tt_ij)

od_tt.to_numpy()[0,19]

od_var_tt = pd.pivot_table(df[['ostation','dstation','tt']] ,index=['ostation'], columns = ['dstation'], values = ['tt'], aggfunc = [np.var])
od_var_tt = od_var_tt.replace(np.nan,0)

np.round(np.sqrt(od_var_tt.to_numpy())/np.mean(od_tt_ij),0)

od_var_tt.shape

type(od_var_tt)

df.columns
df.head()

len(df.nid.unique())
df.shape

df['oline'].unique()

df['dline'].unique()
df['dline'].unique()
len(df['dline'].unique())

df['dstation'].unique()
df['ostation'].unique()
len(df['ostation'].unique())
len(df['dstation'].unique())

len(df['distance'].unique())

58*58

# Consistent with wkipedia information
print('line 1', ':', len(df.loc[(df['oline'] == 1)]['ostation'].unique())) # 23
print('line 2', ':',len(df.loc[(df['oline'] == 2)]['ostation'].unique())) # 18
print('line 3', ':',len(df.loc[(df['oline'] == 3)]['ostation'].unique())) # 0
print('line 4', ':',len(df.loc[(df['oline'] == 4)]['ostation'].unique())) # 23
print('line 5', ':',len(df.loc[(df['oline'] == 5)]['ostation'].unique()))

# Not consistent with wkipedia information
print('line 6', ':',len(df.loc[(df['oline'] == 6)]['ostation'].unique()))
print(len(df.loc[(df['oline'] == 7)]['ostation'].unique()))
print(len(df.loc[(df['oline'] == 8)]['ostation'].unique()))
print(len(df.loc[(df['oline'] == 9)]['ostation'].unique()))


df.loc[(df['oline'] == 1.0)]['ostation'].unique()
df.loc[(df['oline'] == 2.0)]['ostationname'].unique()
temp = df.loc[(df['oline'] == 2.0)]['ostation']
df.loc[(df['oline'] == 4.0)]['ostation'].unique()
df.loc[(df['oline'] == 5.0)]['ostation'].unique()


df_sub =(df.loc[(df['ostation'] == 1.0) & (df['dstation'] == 3.0)])

df_sub =(df.loc[(df['ostation'] == 9.0) & (df['dstation'] == 13.0)])

df_sub.traveltime * df_sub.speed

pd.value_counts((df.loc[(df['ostation'] == 9.0) & (df['dstation'] == 13.0)]).dindex)

df.loc[(df.ostation == 9.0) | (df.dstation == 3.0)]

# =============================================================================
# Summary statistics with dfply
# =============================================================================

# https://github.com/kieferk/dfply

from dfply import *

df >> head(10)

df.head(10)

df >> select(starts_with('o')) >> head(20)

# =============================================================================
# Summary statistics with Pyspark
# =============================================================================

# Working with dataframes in Spark

#https://github.com/tirthajyoti/Spark-with-Python/blob/master/Dataframe_basics.ipynb

from pyspark.sql import SparkSession

# Create a SparkSession app object
spark = SparkSession.builder.appName('Basics').getOrCreate()

#Read csv file
df_spark = spark.read.csv(folder_path,header=False)

# Add columns labels
cols_path = folder_path + "fields.csv"
headers = list(pd.read_csv(cols_path).columns)
df_spark = df_spark.toDF(*headers)

df_spark.show()

df_spark.printSchema()

df_spark.columns

df_spark.describe()

df_spark.summary().show()


# =============================================================================
# Create network x graph with beijing metro network
# =============================================================================


# =============================================================================
# Learning link attributes from passenger trajectories
# =============================================================================


