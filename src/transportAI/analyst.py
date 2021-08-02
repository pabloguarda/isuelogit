""" In charge of doing data processing, merging, data cleaning, etc. It reads data but it requires more sophicated procedures to do it (e.g. setup spark context), which explains why those functions are not includes in the reader module """

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytypes import Positions, Links, Nodes

from utils import blockPrinting

import requests
import json
import shutil
import os
import warnings
import pandas as pd
import numpy as np

import printer

import datetime

import sys

# Loading spark sub libraries


import findspark
import pyspark

from pyspark.sql.types import StructType, StructField, DateType, IntegerType, TimestampType, DoubleType, \
    BooleanType, StringType

from pyspark.sql import SQLContext
from pyspark.sql import SparkSession


# from pyspark.sql.functions import *
from pyspark.sql import functions as F

#=============================================================================
# SPARK (environment setup)
#=============================================================================

class Analyst:
    
    def __init__(self):

        # Setup spark context
        self._sc, self._sqlContext = self.setup_spark_context()

        pass

        # Set number of cores
        # https://stackoverflow.com/questions/21791723/why-is-spark-not-using-all-cores-on-local-machine

        # sc, sqlContext = transportAI.utils.setup_spark_context(n_cores = 1, ram_memory = 1)
        # sc = transportAI.utils.setup_spark_context()

        # To clear cache
        # self.clear_sql_cache()

    @property
    def sc(self):
        return self._sc

    @sc.setter
    def sc(self, value):
        self._sc = value
        
    @property
    def sqlContext(self):
        return self._sqlContext

    @sqlContext.setter
    def sqlContext(self, value):
        self._sqlContext = value


    def clear_sql_cache(self) -> None:
        # To clear cache
        self.sqlContext.clearCache()
    
    @blockPrinting
    def setup_spark_context(self, n_cores = None, ram_memory= None):

        """ Make sure that the Apache spark version match the pyspark version"""

        # Notebook with clear explanations on all steps required to setup pyspark directly from a jupyter notebook run from google colab
        # https://colab.research.google.com/drive/1cIjUEHjbBmvlELFVpKdx_VcSmdjDE46x?authuser=1


        # To setup spark environment variables:
        # https://kevinvecmanis.io/python/pyspark/install/2019/05/31/Installing-Apache-Spark.html
        # https://stackoverflow.com/questions/34685905/how-to-link-pycharm-with-pyspark
        # https://medium.com/@anoop.vasant.kumar/simple-steps-to-setup-pyspark-on-macos-with-python3-87b03ad51bb2
    
        # Path for spark source folder. Download spark from website directly
        os.environ['SPARK_HOME']="/Users/pablo/google-drive/university/cmu/1-courses/2021-1/10605-ml-with-ld/software/spark-3.0.1-bin-hadoop2.7"
    
        # Path for JAVA installation
        os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk-13.0.1.jdk/Contents/Home"

        # Need to Explicitly point to python3 if you are using Python 3.x
        os.environ['PYSPARK_PYTHON']="/Applications/anaconda3/envs/transportAI/bin/python"

        # sys.path.append(os.environ['SPARK_HOME'] + '/python')
        # sys.path.append(os.environ['PYSPARK_PYTHON'] + '/python/lib/py4j-0.9-src.zip"')
    
        #TODO: look a way to ignore pyspark warnings at the beginning
    
        def setup_spark_context_with_warnings():

            findspark.init()

            # https://stackoverflow.com/questions/48607642/disable-info-messages-in-spark-for-an-specific-application


            # conf = SparkConf().setMaster('local').setAppName('example_spark')
            # sc = SparkContext(conf=conf)


    
    
            # TODO: custom configurations are not working so far in execution. I may try to read documentation: https://spark.apache.org/docs/latest/configuration.html
            #  https://stackoverflow.com/questions/26562033/how-to-set-apache-spark-executor-memory

            # from pyspark import SparkConf

            # conf = SparkConf()
            # conf.set('spark.logConf', 'true')
    
            # # Setting ram memory and number of cores used by executors
            # if ram_memory is not None:
            #     conf.set("spark.executor.memory", str(ram_memory) + "g")
            #
            # if n_cores is not None:
            #     conf.set("spark.executor.cores", str(n_cores))
    
            # spark = SparkSession.builder \
            #     .config(conf=conf) \
            #     .appName("hw") \
            #     .getOrCreate()
    
            # spark.sparkContext
    
            # sc = pyspark.SparkContext(appName="hw")#.setLogLevel("OFF")#.setLogLevel("WARN")
    
            sc = (
                SparkSession
                    .builder
                    .appName("hw")
                    .getOrCreate()
            )



            # sqlContext = sc.sqlContext
    
            sqlContext = SQLContext(sc)
    
            # Conf details
    
            # ram_memory = 1
    
            # sc.conf.get('spark.executor.cores')
            # sc.conf.get('spark.executor.memory')
    
    
    
            #
            return sc, sqlContext
    
            # return None
    
        #Not working to present warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return setup_spark_context_with_warnings()

    def download_pems_data(self, path_download, years):

        # Download pems data (Note that 2001 data have 0 for all rows so that info is useless)

        # TODO: add the option to download data from specific date, months and years, and district.
        # Meanwhile I manually download data from Oct 2020 and 2019 but script is working

        # transportAI.utils.download_pems_data(path_download = os.getcwd() + '/data/public/pems/counts/raw'
        #                    , years = [str(e) for e in range(2019, 2021)]
        #                    )

        # TODO: Use a separate process to download the data.

        # TODO: write script to download station metadata. An alternative is provide the type of data in the current download method. Meanwhile I downloaded manually the earliest file for 2019 (d04_text_meta_2021_01_29). I should download data from all stations in california, as it is not clear what it is the mapping between counties and districts. Districts is an internal defition from Caltrans.

        # District and counties list https://en.wikipedia.org/wiki/California_Department_of_Transportation
        # Fresno is district 6.


        url = "http://pems.dot.ca.gov"
        payload = {'username': 'pabloguarda@cmu.edu', 'password': 'PaulCmu1087*'}
        target_url = 'http://pems.dot.ca.gov/?srq=clearinghouse&district_id=4&geotag=null&yy={year}&type=station_5min&returnformat=text'

        # The current link structure is a bit different:
        # http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=4&submit=Submit

        data_page_list = list()
        with requests.Session() as s:
            con = s.post(url, data=payload, verify=False)
            for year in years:
                tmp_page = s.get(target_url.format(year=year))
                data_page_list.append(tmp_page)

        def get_download_url_dict(url_dict, j):
            data = j['data']
            for month in data.keys():
                tmp_list = data[month]
                for e in tmp_list:
                    url_dict[e['file_name']] = e['url']

        url_dict = dict()
        for data_page in data_page_list:
            page_text = data_page.text
            j = json.loads(page_text)
            try:
                get_download_url_dict(url_dict, j)
            except:
                continue

        save_path = path_download
        sorted_file = sorted(list(url_dict.keys()))

        assert os.path.exists(save_path), 'path of folder to download data does not exist'

        with requests.Session() as s:
            con = s.post(url, data=payload, verify=False)
            for name in sorted_file:
                try:
                    tail_url = url_dict[name]
                    print("Now processing:", name, tail_url)
                    r = s.get(url + tail_url, stream=True)
                    if r.status_code == 200:
                        with open(os.path.join(save_path, name), 'wb') as f:
                            r.raw.decode_content = True
                            shutil.copyfileobj(r.raw, f)
                except:
                    print("FAIL:", name, tail_url)



    def read_pems_counts_data(self, filepath: 'str', selected_period: {}) -> (pd.DataFrame,pd.DataFrame):


        # Read pems data with pyspark and select for a certain time window.

        # Raw data has no headers but they can be learned from the data dictionary

        # When the street station is counting at a segment with less than X lanes, then an empty value is recorded for the remaining lengths. Not sure which is the maximum number of lines in the dataset.

        # Read with pandas the first line of the dataset to have a sense of how many entries there are

        if 'hour' in selected_period.keys() and 'duration' in selected_period.keys():

            print('Reading pems counts starting at ' + str(selected_period['hour']) +':00' + ' and during ' + str(selected_period['duration']) + ' minutes')

        else:
            print('Reading pems counts')

        counts_head_df = pd.read_csv(filepath, nrows=10, header=None)

        # view(counts_head_df)
        # speed_df.iloc[0]

        # There are 12 lane independent attributes: len(lane_independent_columns)
        lane_independent_columns = [ \
            StructField('ts', StringType(), True), \
            StructField('station_id', IntegerType(), True), \
            StructField('district_id', IntegerType(), True), \
            StructField('freeway_id', IntegerType(), True), \
            StructField('travel_direction', StringType(), True), \
            StructField('lane_type', StringType(), True), \
            StructField('observed_lanes_pct', DoubleType(), True), \
            StructField('station_length', DoubleType(), True), \
            StructField('samples', IntegerType(), True), \
            StructField('flow_total', DoubleType(), True), \
            StructField('occupancy_avg', DoubleType(), True), \
            StructField('speed_avg', DoubleType(), True), \
            ]

        # The dataset have 52 columns: len(counts_df.columns)

        # There are 5 lane dependent attributes according to the pems dictionary, and thus, each of the attributes is computed under assumption of a maximum of 8 lanes (12 + 5 x 8 = 52)

        lane_dependent_attributes = ['samples_lane', 'flow_total_lane', 'occupancy_avg_lane', 'speed_avg_lane','observed_lane']

        # This was being done badly as the lane dependent attributes are included for one lane at a time and not for all lanes and each attribute one time.
        # lane_dependent_labels = [str(i) + '_' + str(j) for i in lane_dependent_attributes for j in np.arange(1, 9)]

        lane_dependent_labels = [str(j) + '_' + str(i) for i in np.arange(1, 9)  for j in lane_dependent_attributes]

        lane_dependent_columns = [StructField(key, DoubleType(), True) for key in lane_dependent_labels]

        custom_schema = StructType(lane_independent_columns + lane_dependent_columns)

        counts_sdf = self.sqlContext.read.csv(filepath, header=False, schema=custom_schema)

        # counts_sdf.show(2)

        # counts_sdf selected_time

        counts_sdf = counts_sdf.withColumn("ts_casted", F.to_timestamp(F.col("ts"),"MM/dd/yyyy HH:mm:ss" ))

        # https://stackoverflow.com/questions/30949202/spark-dataframe-timestamptype-how-to-get-year-month-day-values-from-field
        counts_sdf  = counts_sdf.withColumn('year', F.year(counts_sdf.ts_casted))\
                                  .withColumn('month', F.month(counts_sdf.ts_casted))\
                                  .withColumn('day_month',F.dayofmonth(counts_sdf.ts_casted))\
                                  .withColumn('hour', F.hour(counts_sdf.ts_casted))\
                                  .withColumn('minute', F.minute(counts_sdf.ts_casted))

        # counts_sdf.printSchema()

        # Compute average flow per station for selected time

        # Filter observatiosn based on selected time
        if 'hour' in selected_period.keys() and 'duration' in selected_period.keys():

            # Account for duration in minutes from the beginning of the hour

            delta_hours = int(selected_period['duration']/ 60)
            delta_minutes = selected_period['duration'] - delta_hours * 60

            count_interval_sdf = counts_sdf.filter((counts_sdf.hour >= selected_period['hour']) & (counts_sdf.hour <= selected_period['hour'] + delta_hours) )

            if delta_minutes > 0:
                count_interval_sdf = count_interval_sdf.filter((count_interval_sdf.hour >= selected_period['hour'] + delta_hours) & (count_interval_sdf.minute <= delta_minutes))

        else:
            count_interval_sdf = counts_sdf

        # count_interval_df = count_interval_sdf.toPandas()


        # if 'hour' in selected_time.keys() and 'minute' in selected_time.keys():
        #     count_interval_sdf = counts_sdf.filter((counts_sdf.hour < selected_time['hour'] + 2) & (counts_sdf.minute == selected_time['minute']))
        # elif 'hour' in selected_time.keys():
        #     count_interval_sdf = counts_sdf.filter((counts_sdf.hour == selected_time['hour']))
        # elif 'minute' in selected_time.keys():
        #     count_interval_sdf = counts_sdf.filter((counts_sdf.minute == selected_time['minute']))

        return count_interval_sdf

    def read_pems_counts_by_period(self, filepath: 'str', selected_period: {}) -> pd.DataFrame:

        """:arg filepath: path of a txt file with pems count data every 15 minutes at the each station "
        
        
        returns:
        Aggregated counts data in the selected period by station id
        
        
        """

        count_interval_sdf = self.read_pems_counts_data(filepath, selected_period)

        count_interval_sdf =  count_interval_sdf.na.drop \
            (subset=['flow_total']).groupby('station_id').agg(
            F.sum('flow_total').alias('flow_total')
            ,F.sum('flow_total_lane_1').alias('flow_total_lane_1')
            , F.sum('flow_total_lane_2').alias('flow_total_lane_2')
            , F.sum('flow_total_lane_3').alias('flow_total_lane_3')
            , F.sum('flow_total_lane_4').alias('flow_total_lane_4')
            , F.sum('flow_total_lane_5').alias('flow_total_lane_5')
            , F.sum('flow_total_lane_6').alias('flow_total_lane_6')
            , F.sum('flow_total_lane_7').alias('flow_total_lane_7')
            , F.sum('flow_total_lane_8').alias('flow_total_lane_8')
            , F.mean('flow_total').alias('flow_total_avg')
            , F.stddev('flow_total').alias('flow_total_sd')
        )#.cache()

        # count_interval_sdf.show()

        # count_interval_sdf.count()

        # Compute average speed for each station to compare against Inrix data.
        # Note that pems is in mile units whereas inrix in kmh (1mph = 1.61 kph). Everything will be left in miles per hour

        # speed_interval_pems_sdf = counts_sdf.na.drop(
        #     subset=['speed_avg']) \
        #     .withColumn('speed_avg_kmh', F.col('speed_avg')) \
        #     .groupby('station_id').agg(
        #     F.min('speed_avg').alias('speed_min'),
        #     F.max('speed_avg').alias('speed_max'),
        #     F.mean('speed_avg').alias('speed_avg'),
        #     F.stddev('speed_avg').alias('speed_sd')).cache()
        #
        # # I found the average speeds to high. The minimum is 84km.
        # speed_interval_pems_sdf.select(F.min(F.col('speed_avg'))).show()


        # df.filter(df['ts'] >= F.lit('2018-06-27 00:00:00'))

        # # Create pandas dataframe
        # count_interval_df = count_interval_sdf.toPandas()

        self.clear_sql_cache()


        return count_interval_sdf.toPandas()
    
    def selected_period_filter(self, sdf: pyspark.sql.DataFrame, selected_period: {}):

        selected_period_sdf = sdf

        if 'year' in selected_period.keys():
            # speed_segment_sdf = speed_sdf.where(F.col("segment_id").isin([link.id]))
            # if isinstance(years,list):
            selected_period_sdf = selected_period_sdf.where(F.col("year").isin(selected_period['year']))

        if 'month' in selected_period.keys():
            selected_period_sdf = selected_period_sdf.where(F.col("month").isin(selected_period['month']))

        if 'day_month' in selected_period.keys():
            selected_period_sdf = selected_period_sdf.where(F.col("day_month").isin(selected_period['day_month']))

        if 'day_week' in selected_period.keys():
            selected_period_sdf = selected_period_sdf.where(F.col("day_week").isin(selected_period['day_week']))

        if 'hour' in selected_period.keys():
            selected_period_sdf = selected_period_sdf.where(F.col("hour").isin(selected_period['hour']))

        if 'minute' in selected_period.keys():
            selected_period_sdf = selected_period_sdf.where(F.col("minute").isin(selected_period['minute']))

        return selected_period_sdf





    def read_inrix_data(self, filepaths: [], selected_period: {} = {}) -> pyspark.sql.DataFrame:

        # small_speed_df1 = pd.read_csv(filepaths[0], nrows=1000)
        # small_speed_df2 = pd.read_csv(filepaths[1], nrows=1000)
        #
        # print(small_speed_df1.head())
        # print(small_speed_df2.head())

        #
        # old_cols = list(speed_df.keys())

        # ['Date Time', 'Segment ID', 'UTC Date Time', 'Speed(km/hour)', 'Hist Av Speed(km/hour)', 'Ref Speed(km/hour)', 'Travel Time(Minutes)', 'CValue', 'Pct Score30', 'Pct Score20', 'Pct Score10', 'Road Closure', 'Corridor/Region Name']

        custom_schema = StructType([ \
            StructField('Date Time', DateType(), True), \
            StructField('Segment ID', IntegerType(), True), \
            StructField('UTC Date Time', TimestampType(), True), \
            StructField('Speed(km/hour)', DoubleType(), True), \
            StructField('Hist Av Speed(km/hour)', DoubleType(), True), \
            StructField('Ref Speed(km/hour)', DoubleType(), True), \
            StructField('Travel Time(Minutes)', DoubleType(), True), \
            StructField('CValue', DoubleType(), True), \
            StructField('Pct Score30', DoubleType(), True), \
            StructField('Pct Score20', DoubleType(), True), \
            StructField('Pct Score10', DoubleType(), True), \
            StructField('Road Closure', StringType(), True), \
            StructField('Corridor/Region Name', StringType(), True)
        ]
        )

        # custom_schema = StructType([ \
        #     StructField('dt', DateType(), True), \
        #     StructField('segment_id', IntegerType(), True), \
        #     StructField('ts', TimestampType(), True), \
        #     StructField('speed', DoubleType(), True), \
        #     StructField('speed_hist', DoubleType(), True), \
        #     StructField('speed_ref', DoubleType(), True), \
        #     StructField('travel_time', DoubleType(), True), \
        #     StructField('confidence', DoubleType(), True), \
        #     StructField('pct30', DoubleType(), True), \
        #     StructField('pct20', DoubleType(), True), \
        #     StructField('pct10', DoubleType(), True), \
        #     StructField('road_closure', StringType(), True), \
        #     StructField('name', StringType(), True)
        # ]
        # )

        # TODO: ask/understand what Pct Score30,Pct Score20,Pct Score10 mean.

        # df = self.sqlContext.read.format("csv").option("header", "true"). \
        #     schema(custom_schema). \
        #     load(filepaths.split(','))

        # filepaths = ['/Users/pablo/google-drive/data-science/github/transportAI/input/private/Fresno/inrix/speed/Fresno_CA_2019-10-01_to_2019-11-01_15_min_part_1/data.csv', '/Users/pablo/google-drive/data-science/github/transportAI/input/private/Fresno/inrix/speed/Fresno_CA_2019-10-01_to_2019-11-01_15_min_part_2/data.csv']

        inrix_data_sdf = self.sqlContext.read.csv(filepaths, header=True, schema=custom_schema)

        new_columns = ['dt', 'segment_id', 'ts', 'speed', 'speed_hist', 'speed_ref', 'travel_time', 'confidence',
                       'pct30', 'pct20', 'pct10', 'road_closure', 'name']

        # Rename columsn
        inrix_data_sdf = inrix_data_sdf.toDF(
            *new_columns)  # .cache() Ram cannot hold the entire set as it is just to big

        # Cast road_closure attribute to numeric

        # inrix_data_sdf = inrix_data_sdf.withColumn('road_closure', F.col('road_closure').cast('integer'))

        # inrix_data_sdf = inrix_data_sdf.withColumn('road_closure', when(F.col('road_closure').cast('integer'))

        inrix_data_sdf = inrix_data_sdf.withColumn('road_closure_num',
                                                   F.when(inrix_data_sdf.road_closure == 'T', 1).otherwise(0))

        # Convert speed units from km to miles for consistency with Sean file
        km_to_miles_factor = 0.62137119223

        inrix_data_sdf = inrix_data_sdf.withColumn("speed", km_to_miles_factor * F.col("speed"))

        inrix_data_sdf = inrix_data_sdf.withColumn("speed_ref", km_to_miles_factor * F.col("speed_ref"))

        inrix_data_sdf = inrix_data_sdf.withColumn("speed_hist", km_to_miles_factor * F.col("speed_hist"))

        # Compute time data features

        # inrix_data_sdf = inrix_data_sdf.withColumn("dt_casted", F.to_timestamp(F.col("dt"),"MM/dd/yyyy HH:mm:ss" ))

        # Note that day_week == 2 is Monday not Tuesday as apparently days start from sunday
        # https://mungingdata.com/apache-spark/week-end-start-dayofweek-next-day/

        inrix_data_sdf = inrix_data_sdf \
            .withColumn('year', F.year(inrix_data_sdf.dt)) \
            .withColumn('month', F.month(inrix_data_sdf.dt)) \
            .withColumn('day_month', F.dayofmonth(inrix_data_sdf.dt)) \
            .withColumn('day_week', F.dayofweek(inrix_data_sdf.dt)) \
            .withColumn('hour', F.hour(inrix_data_sdf.ts)) \
            .withColumn('minute', F.minute(inrix_data_sdf.ts))

        # Observations by year
        # inrix_data_sdf.groupby('year').count().show()

        # selected_period = {'year': [2019], 'month': [10], 'day_month': [1], 'hour': [10, 11, 12]}

        # selected_period = {'year': [2019], 'month': [10], 'day_month': [1]}

        # selected_period = {'year': [2019]}

        # inrix_selected_period_sdf = inrix_data_sdf

        # Filtering according to the selected period
        inrix_selected_period_sdf = self.selected_period_filter(inrix_data_sdf, selected_period)

        # # Cache the dataset
        # inrix_selected_period_sdf = inrix_selected_period_sdf.collect().cache()

        # Counts by date (very expensive to compute)
        # counts_date = inrix_selected_period_sdf.groupby('dt').count().collect()
        # print(inrix_selected_period_sdf.groupby('dt').count().collect())

        # print(counts_date)
        # print(counts_date.show(truncate = False))

        # inrix_selected_period_sdf.groupby('road_closure_num').show()

        # inrix_selected_period_sdf.groupby('road_closure_num').count().show()
        # inrix_selected_period_sdf = inrix_selected_period_sdf.cache()

        # inrix_selected_period_sdf.show()

        # inrix_selected_period_sdf.groupby('dt').count().collect()

        return inrix_selected_period_sdf

    
    def generate_inrix_data_by_segment(self, filepaths: [], selected_period: {} = {}) -> pd.DataFrame:

        print('\nReading and processing INRIX data with pyspark')

        # TODO: specify attribute for time period of analysis. There should be some consistency with period of traffic count data

        # Read data from a selected perido
        inrix_selected_period_sdf = self.read_inrix_data(filepaths, selected_period)

        # Compute variance, mean and std by segment id
        inrix_data_segments_sdf = inrix_selected_period_sdf.na.drop \
            (subset=['speed']).groupby('segment_id').agg(
            F.count('speed').alias('n')

            , F.mean('speed').alias('speed_avg')
            , F.stddev('speed').alias('speed_sd')
            , F.variance('speed').alias('speed_var')
            , F.min('speed').alias('speed_min')
            , F.max('speed').alias('speed_max')

            , F.mean('speed_hist').alias('speed_hist_avg')
            , F.stddev('speed_hist').alias('speed_hist_sd')
            , F.variance('speed_hist').alias('speed_hist_var')

            , F.mean('speed_ref').alias('speed_ref_avg')
            , F.stddev('speed_ref').alias('speed_ref_sd')
            , F.variance('speed_ref').alias('speed_ref_var')

            , F.min('travel_time').alias('traveltime_min')
            , F.max('travel_time').alias('traveltime_max')
            , F.mean('travel_time').alias('traveltime_avg')
            , F.stddev('travel_time').alias('traveltime_sd')
            , F.variance('travel_time').alias('traveltime_var')

            , F.mean('road_closure_num').alias('road_closure_avg')
            , F.max('road_closure_num').alias('road_closure_any')
        )

        inrix_data_segments_df = inrix_data_segments_sdf.toPandas()

        # Add coeficient of variation for travel time and speed
        inrix_data_segments_df['traveltime_cv'] = inrix_data_segments_df['traveltime_sd'] / inrix_data_segments_df[
            'traveltime_avg']

        inrix_data_segments_df['speed_cv'] = inrix_data_segments_df['speed_sd'] / inrix_data_segments_df['speed_avg']

        return inrix_data_segments_df


    def write_partition_inrix_data(self, filepaths, output_folderpath, unit = 'day'):

        # small_speed_df1 = pd.read_csv(filepaths[0], nrows=1000)
        # small_speed_df2 = pd.read_csv(filepaths[1], nrows=1000)
        #
        # print(small_speed_df1.head())
        # print(small_speed_df2.head())


        # https://stackoverflow.com/questions/62268860/pyspark-save-dataframe-to-single-csv-by-date-using-window-function

        # https://stackoverflow.com/questions/59278835/pyspark-how-to-write-dataframe-partition-by-year-month-day-hour-sub-directory

        # custom_schema = StructType([ \
        #     StructField('dt', DateType(), True), \
        #     StructField('segment_id', IntegerType(), True), \
        #     StructField('ts', TimestampType(), True), \
        #     StructField('speed', DoubleType(), True), \
        #     StructField('speed_hist', DoubleType(), True), \
        #     StructField('speed_ref', DoubleType(), True), \
        #     StructField('travel_time', DoubleType(), True), \
        #     StructField('confidence', DoubleType(), True), \
        #     StructField('pct30', DoubleType(), True), \
        #     StructField('pct20', DoubleType(), True), \
        #     StructField('pct10', DoubleType(), True), \
        #     StructField('road_closure', StringType(), True), \
        #     StructField('name', StringType(), True)
        # ]
        # )

        custom_schema = StructType([ \
            StructField('Date Time', DateType(), True), \
            StructField('Segment ID', IntegerType(), True), \
            StructField('UTC Date Time', TimestampType(), True), \
            StructField('Speed(km/hour)', DoubleType(), True), \
            StructField('Hist Av Speed(km/hour)', DoubleType(), True), \
            StructField('Ref Speed(km/hour)', DoubleType(), True), \
            StructField('Travel Time(Minutes)', DoubleType(), True), \
            StructField('CValue', DoubleType(), True), \
            StructField('Pct Score30', DoubleType(), True), \
            StructField('Pct Score20', DoubleType(), True), \
            StructField('Pct Score10', DoubleType(), True), \
            StructField('Road Closure', StringType(), True), \
            StructField('Corridor/Region Name', StringType(), True)
        ]
        )

        inrix_data_sdf = self.sqlContext.read.csv(filepaths, header=True, schema=custom_schema)

        if unit == 'day':

            # Get unique yyyymmddhh values in the grouping column
            groups = [x[0] for x in inrix_data_sdf.select('Date Time').distinct().collect()]

            # Create a filtered DataFrame
            groups_list = [inrix_data_sdf.filter(F.col('Date Time')== x) for x in groups]

            # Save the result by yyyy/mm/dd/hh
            for filtered_data in groups_list:
                target_date = filtered_data.select('Date Time').take(1)[0].asDict()['Date Time']
                print('writing ' + str(target_date) + '.csv')
                path = output_folderpath + str(target_date) + '.csv'

                filtered_data.toPandas().to_csv(path, index=False)

        if unit == 'year':

            inrix_data_sdf = inrix_data_sdf \
                .withColumn('year', F.year(inrix_data_sdf.select('Date Time'))) \
 \
                # Get unique yyyymmddhh values in the grouping column
            groups = [x[0] for x in inrix_data_sdf.select('year').distinct().collect()]

            # Create a filtered DataFrame
            groups_list = [inrix_data_sdf.filter(F.col('year') == x) for x in groups]

            # Save the result by yyyy/mm/dd/hh
            for filtered_data in groups_list:
                target_date = filtered_data.select(inrix_data_sdf).take(1)[0].asDict()[inrix_data_sdf]
                filtered_data = filtered_data.drop('year')
                print('writing ' + str(target_date) + '.csv')
                path = output_folderpath + str(target_date) + '.csv'

                filtered_data.toPandas().to_csv(path, index=False)


    def merge_inrix_data(self, links: Links, speed_df, options: {}, config) -> None:

        print('Merging INRIX data of speeds and travel times with network links')

        for link in links:

            # Default value for speed avg is the link free flow speed (mi/h)
            link.Z_dict['speed_avg'] = link.Z_dict['ff_speed']
            link.Z_dict['speed_ref_avg'] = link.Z_dict['ff_speed']
            link.Z_dict['speed_hist_avg'] = link.Z_dict['ff_speed']
            link.Z_dict['tt_avg'] = link.Z_dict['ff_traveltime']


            # Note: length is in miles according to Sean's files

            # Default value for the standard deviation of speed is 0 (it is assumed that a segment not part of inrix, it is a segment with low congestion)
            link.Z_dict['speed_sd'] = 0
            link.Z_dict['speed_cv'] = 0
            link.Z_dict['speed_hist_sd'] = 0
            link.Z_dict['speed_ref_sd'] = 0

            link.Z_dict['tt_sd'] = 0
            link.Z_dict['tt_var'] = 0

            link.Z_dict['tt_cv'] = 0

            link.Z_dict['tt_sd_adj'] = 0

            link.Z_dict['road_closures'] = 0

            if link.inrix_id is not None:
                # speed_segment_sdf = speed_sdf.where(F.col("segment_id").isin([link.inrix_id]))
                inrix_features = speed_df.loc[speed_df.segment_id == int(link.inrix_id)]

                if len(inrix_features) > 0:

                    # print(inrix_features['speed_avg'])
                    # print(link.inrix_id)
                    link.inrix_features = {
                        'speed_avg': float(inrix_features['speed_avg'])
                        , 'speed_sd': float(inrix_features['speed_sd'])
                        , 'speed_cv': float(inrix_features['speed_cv'])

                        , 'speed_ref_avg': float(inrix_features['speed_ref_avg'])
                        , 'speed_ref_sd': float(inrix_features['speed_ref_sd'])

                        , 'speed_hist_avg': float(inrix_features['speed_hist_avg'])
                        , 'speed_hist_sd': float(inrix_features['speed_hist_sd'])

                        , 'traveltime_avg': float(inrix_features['traveltime_avg'])
                        , 'traveltime_sd': float(inrix_features['traveltime_sd'])
                        , 'traveltime_var': float(inrix_features['traveltime_var'])
                        , 'traveltime_cv': float(inrix_features['traveltime_cv'])

                        , 'road_closure_avg': float(inrix_features['road_closure_avg'])
                        , 'road_closure_any': float(inrix_features['road_closure_any'])

                    }

                # link.inrix_features = inrix_features[]

                    # For travel time, I should consider a normalization

                    link.Z_dict['speed_avg'] = link.inrix_features['speed_avg']
                    link.Z_dict['speed_sd'] = link.inrix_features['speed_sd']

                    link.Z_dict['speed_ref_avg'] = link.inrix_features['speed_ref_avg']
                    link.Z_dict['speed_ref_sd'] = link.inrix_features['speed_ref_sd']

                    link.Z_dict['speed_hist_avg'] = link.inrix_features['speed_hist_avg']
                    link.Z_dict['speed_hist_sd'] = link.inrix_features['speed_hist_sd']

                    link.Z_dict['tt_cv'] = link.inrix_features['traveltime_cv']
                    link.Z_dict['speed_cv'] = link.inrix_features['speed_cv']

                    if options['tt_units'] == 'seconds':
                        tt_factor = 60

                    if options['tt_units'] == 'minutes':
                        tt_factor = 1

                    link.Z_dict['tt_avg'] = tt_factor*link.inrix_features['traveltime_avg']
                    link.Z_dict['tt_sd'] = tt_factor*link.inrix_features['traveltime_sd']
                    link.Z_dict['tt_var'] = tt_factor**2 * link.inrix_features['traveltime_var']

                    #Road closures are an interesting features from INRIX data but they occur 0.01% of the time.
                    link.Z_dict['road_closures'] = link.inrix_features['road_closure_any']

                    if options['update_ff_tt_inrix'] is True:

                        # Weighting by 60 will leave travel time with minutes units, because speeds are originally in per hour units
                        if options['tt_units'] == 'minutes':
                            tt_factor = 60

                        if options['tt_units'] == 'seconds':
                            tt_factor = 60 * 60

                        #Multiplied by 60 so speeds are in minutes
                        link.bpr.tf = tt_factor*link.Z_dict['length']/link.Z_dict['speed_ref_avg']


        # # Filter rows from segment ids that were matched with network links
        #
        # speed_sdf.where(F.col("segment_id").isin(["CB", "CI", "CR"]))
        #
        # speed_sdf.show()
        # speed_sdf.count()
        #
        # speed_sdf.select('segment_id')



    def read_bus_stops_txt(self, filepath: str) -> pd.DataFrame:

        stops_df = pd.read_csv(filepath, header=0, delimiter=',')

        # stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station,stop_timezone,wheelchair_boarding

        return stops_df

    def read_traffic_incidents(self, filepath: str, selected_period: {}) -> pd.DataFrame:

        print('\nReading traffic incidents data')

        # traffic_incidents_head_df = pd.read_csv(filepath, nrows=10)

        # traffic_incidents_df = pd.read_csv(filepath)

        selected_columns = ['Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'County']

        # selected_columns_schema = [ \
        #     StructField('Severity', StringType(), True), \
        #     StructField('Start_Time', StringType(), True), \
        #     StructField('End_Time', StringType(), True), \
        #     StructField('Start_Lat', DoubleType(), True), \
        #     StructField('Start_Lng', DoubleType(), True), \
        #     StructField('End_Lat', DoubleType(), True), \
        #     StructField('End_Lng', DoubleType(), True), \
        #     StructField('County', StringType(), True), \
        #     ]

        # printer.blockPrint()
        incidents_sdf = self.sqlContext.read.csv(filepath, header=True).select(selected_columns)

        incidents_sdf = incidents_sdf.filter(incidents_sdf.County == 'Fresno')

        incidents_sdf = incidents_sdf\
            .withColumn('year', F.year(incidents_sdf.Start_Time)) \
            .withColumn('month', F.month(incidents_sdf.Start_Time)) \
            .withColumn('day_month', F.dayofmonth(incidents_sdf.Start_Time)) \
            .withColumn('hour', F.hour(incidents_sdf.Start_Time)) \
            .withColumn('minute', F.minute(incidents_sdf.Start_Time))

        # incidents_period_sdf = incidents_sdf

        # if 'year' in selected_period.keys():
        #     # speed_segment_sdf = speed_sdf.where(F.col("segment_id").isin([link.inrix_id]))
        #     # if isinstance(years,list):
        #     incidents_period_sdf = incidents_period_sdf.where(F.col("year").isin(selected_period['year']))

        # Filtering according to the selected period
        incidents_period_sdf = self.selected_period_filter(incidents_sdf, selected_period)

        #
        # if 'month' in selected_period.keys():
        #
        # if 'day_month' in selected_period.keys():
        #
        # if 'hour' in selected_period.keys():
        #
        # if 'minute' in selected_period.keys():


        # if 'hour' in selected_period.keys():
        #
        #     count_interval_sdf = incidents_sdf.filter(
        #         (incidents_sdf.year == selected_time['hour']) & (counts_sdf.hour <= selected_time['hour'] + duration))



        #incidents_sdf.groupBy('year').count().show()
        #incidents_sdf.count()

        incidents_df = incidents_period_sdf.toPandas()

        # printer.enablePrint()

        return incidents_df









