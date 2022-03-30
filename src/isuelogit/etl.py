""" In charge of doing data processing, merging, data cleaning, etc. It reads data but it requires more sophicated procedures to do it (e.g. setup spark context), which explains why those functions are not includes in the reader module """

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mytypes import Positions, Links, Nodes, Dict, Feature,ColumnVector, List, DataFrame
    from links import Link

from utils import Options
from printer import blockPrinting, block_output
import config
import geographer

import requests
import json
import shutil
import os
import warnings
import pandas as pd
import numpy as np
import copy

import datetime

#=============================================================================
# SPARK (environment setup)
#=============================================================================


class DataReader:
    
    def __init__(self, setup_spark = False, network_key:str = None):

        self._setup_spark = setup_spark

        if setup_spark:
            
            self.spark_reader = SparkReader()
            self._setup_spark = True

            # Set number of cores
            # https://stackoverflow.com/questions/21791723/why-is-spark-not-using-all-cores-on-local-machine
            # sc, sqlContext = self.setup_spark_context(n_cores=1, ram_memory=1)
            # # sc = self.setup_spark_context()

        self.set_default_options()

        if network_key == 'Fresno':
            self.set_default_options_fresno()

    def read_pems_counts_data(self, **kwargs):
        if self._setup_spark is False:
            self.spark_reader = SparkReader()
            self._setup_spark = True
            
        return self.spark_reader.read_pems_counts_data(**kwargs)

    def read_pems_counts_by_period(self, **kwargs):

        if self._setup_spark is False:
            self.spark_reader = SparkReader()
            self._setup_spark = True

        return self.spark_reader.read_pems_counts_by_period(**kwargs)
    
    def selected_period_filter(self, **kwargs):
        if self._setup_spark is False:
            self.spark_reader = SparkReader()
            self._setup_spark = True

        return self.spark_reader.selected_period_filter(**kwargs)

    def read_inrix_data(self, **kwargs):

        if self._setup_spark is False:
            self.spark_reader = SparkReader()
            self._setup_spark = True

        return self.spark_reader.read_inrix_data(**kwargs)

    def generate_inrix_data_by_segment(self, **kwargs):

        if self._setup_spark is False:
            self.spark_reader = SparkReader()
            self._setup_spark = True

        return self.spark_reader.generate_inrix_data_by_segment(**kwargs)

    def write_partition_inrix_data(self, **kwargs):

        if self._setup_spark is False:
            self.spark_reader = SparkReader()
            self._setup_spark = True

        self.spark_reader.write_partition_inrix_data(**kwargs)

    def read_traffic_incidents(self, **kwargs):

        if self._setup_spark is False:
            self.spark_reader = SparkReader()
            self._setup_spark = True

        return self.spark_reader.read_traffic_incidents(**kwargs)
        

    def set_default_options(self):

        self.options = Options()


    def set_default_options_fresno(self):

        options = {}

        options['selected_date'] = None
        options['selected_hour'] = None
        options['selected_year'] = None
        options['selected_month'] = None
        options['selected_day_month'] = None

        # - Matching GIS layers with inrix instead of the network links.
        options['inrix_matching'] = {'census': False, 'incidents': False, 'bus_stops': False, 'streets_intersections': False}

        # - Buffer (in feets, which is the unit of the CA crs)
        options['buffer_size'] = {'inrix': 0, 'bus_stops': 0, 'incidents': 0, 'streets_intersections': 0}
        options['tt_units'] = 'minutes'
        options['update_ff_tt_inrix'] = False

        options['data_processing'] = {'inrix_segments': False, 'inrix_data': False, 'census': False,
                                      'incidents': False, 'bus_stops': False, 'streets_intersections': False}

        options['write_inrix_daily_data'] = False
        options['read_inrix_daily_data'] = True

        # - Periods (6 periods of 15 minutes each for Fresno)
        options['od_periods'] = []

        options['selected_period_incidents'] = {}

        self.update_options(**options)

    def get_updated_options(self, **kwargs):
        return copy.deepcopy(self.options.get_updated_options(new_options=kwargs))

    def update_options(self,**kwargs):
        self.options = self.get_updated_options(**kwargs)

    def select_period(self,
                      date,
                      hour) -> None:

        self.options['selected_date'] = date

        # Period selected for data analysis
        self.options['selected_hour'] = hour

        self.options['selected_date_datetime'] = datetime.date.fromisoformat(
            self.options['selected_date'])

        # Get year, day of month, month and day of week using datetime package functionalities
        self.options['selected_year'] = self.options['selected_date_datetime'].year
        self.options['selected_day_month'] = self.options['selected_date_datetime'].day
        self.options['selected_month'] = self.options['selected_date_datetime'].month
        self.options['selected_day_week'] = int(
            self.options['selected_date_datetime'].strftime('%w')) + 1

        # Note: for consistency with spark, the weekday number from datetime was changed a bit
        # https://stackoverflow.com/questions/9847213/how-do-i-get-the-day-of-week-given-a-date

        print('\nSelected date is ' + self.options['selected_date'] + ', ' + self.options[
            'selected_date_datetime'].strftime('%A') + ' at ' + str(self.options['selected_hour']) + ':00')

        # Examples:

        # October 1, 2020 is Thursday (day_week = 5).
        # self.options['selected_year'] = 2020
        # self.options['selected_date'] = '2020-10-01'
        # print('\nSelected period is October 1, 2020, Thursday at ' + str(self.options['selected_hour']) + ':00' )

        # October 1, 2019 is Tuesday  (day_week = 3).
        # self.options['selected_year'] = 2019
        # self.options['selected_date'] = '2019-10-01'
        # print('\nSelected period is October 1, 2019, Tuesday at ' + str(self.options['selected_hour']) + ':00' )

        self.options['selected_period_inrix'] = \
            {'year': [self.options['selected_year']],
             'month': [self.options['selected_month']],
             'day_month': [self.options['selected_day_month']],
             'hour': [self. options['selected_hour'] - 1,
                      self.options['selected_hour']]}

        self.options['selected_period_inrix'] = {}

        # config.estimation_options['selected_period_inrix'] = \
        #     {'year': [config.estimation_options['selected_year']], 'month': [config.estimation_options['selected_month']]}
        # config.estimation_options['selected_period_inrix'] = \
        #     {'year': [config.estimation_options['selected_year']], 'month': [config.estimation_options['selected_month']], 'day_week': config.estimation_options['selected_day_week'], 'hour': [config.estimation_options['selected_hour']]}

    def download_pems_data(self, path_download, years):

        # Download pems data (Note that 2001 data have 0 for all rows so that info is useless)

        # TODO: add the option to download data from specific date, months and years, and district.
        # Meanwhile I manually download data from Oct 2020 and 2019 but script is working

        # isuelogit.utils.download_pems_data(path_download = os.getcwd() + '/data/public/pems/counts/raw'
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

    def merge_inrix_data(self,
                         links: Links,
                         speed_df,
                         options: {}) -> None:

        print('Merging INRIX data of speeds and travel times with network links')

        for link in links:

            ## Default value for speed avg is the link free flow speed (mi/h)
            # Default value is 0
            link.Z_dict['speed_avg'] = 0 #.Z_dict['ff_speed']
            link.Z_dict['speed_ref_avg'] = 0 #link.Z_dict['ff_speed']
            link.Z_dict['speed_hist_avg'] = 0 # link.Z_dict['ff_speed']
            link.Z_dict['tt_avg'] = 0 # link.Z_dict['ff_traveltime']


            # Note: length is in miles according to Sean's files

            # Default value for the standard deviation of speed is 0 (it is assumed that a segment not part of inrix, it is a segment with low congestion)
            link.Z_dict['speed_max'] = 0
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
                # speed_segment_sdf = speed_sdf.where(self.F.col("segment_id").isin([link.inrix_id]))
                inrix_features = speed_df.loc[speed_df.segment_id == int(link.inrix_id)]

                if len(inrix_features) > 0:

                    # print(inrix_features['speed_avg'])
                    # print(link.inrix_id)
                    link.inrix_features = {
                        'speed_max': float(inrix_features['speed_max'])
                        , 'speed_avg': float(inrix_features['speed_avg'])
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

                    link.Z_dict['speed_max'] = link.inrix_features['speed_max']
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


        # # Filter rows from segment ids that were matched with network links
        #
        # speed_sdf.where(self.F.col("segment_id").isin(["CB", "CI", "CR"]))
        #
        # speed_sdf.show()
        # speed_sdf.count()
        #
        # speed_sdf.select('segment_id')

    def read_bus_stops_txt(self, filepath: str) -> pd.DataFrame:

        stops_df = pd.read_csv(filepath, header=0, delimiter=',')

        # stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station,stop_timezone,wheelchair_boarding

        return stops_df

    def read_spatiotemporal_data_fresno(self,
                                        **kwargs):
        network = kwargs.pop('network', None)

        self.update_options(**kwargs)

        if self._setup_spark is False:
            self.spark_reader = SparkReader()
            self._setup_spark = True


        return read_spatiotemporal_data_fresno(data_analyst = self,
                                               network = network,
                                               **self.options)


class SparkReader:
    # Loading spark sub libraries

    try:
        import findspark
        import pyspark

        from pyspark.sql.types import StructType, StructField, DateType, IntegerType, TimestampType, DoubleType, \
            BooleanType, StringType

        from pyspark.sql import SQLContext
        from pyspark.sql import SparkSession

        # from pyspark.sql.functions import *
        from pyspark.sql import functions as F

    except ImportError:
        pass

    def __init__(self):

        # Setup spark context
        self._sc, self._sqlContext = self.setup_spark_context()

        # To clear cache
        self.clear_sql_cache()

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

    # @blockPrinting
    def setup_spark_context(self, n_cores=None, ram_memory=None):

        """ Make sure that the Apache spark version match the pyspark version"""

        # Notebook with clear explanations on all steps required to setup pyspark directly from a jupyter notebook run from google colab
        # https://colab.research.google.com/drive/1cIjUEHjbBmvlELFVpKdx_VcSmdjDE46x?authuser=1

        # To setup spark environment variables:
        # https://kevinvecmanis.io/python/pyspark/install/2019/05/31/Installing-Apache-Spark.html
        # https://stackoverflow.com/questions/34685905/how-to-link-pycharm-with-pyspark
        # https://medium.com/@anoop.vasant.kumar/simple-steps-to-setup-pyspark-on-macos-with-python3-87b03ad51bb2

        # Path for spark source folder. Download spark from website directly
        os.environ[
            'SPARK_HOME'] = "/Users/pablo/OneDrive/university/cmu/1-courses/2021-1/10605-ml-with-ld/software/spark-3.0.1-bin-hadoop2.7"

        # Path for JAVA installation
        os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk-13.0.1.jdk/Contents/Home"

        # Need to Explicitly point to python3 if you are using Python 3.x
        os.environ['PYSPARK_PYTHON'] = "/Applications/anaconda3/envs/isuelogit/bin/python"

        # sys.path.append(os.environ['SPARK_HOME'] + '/python')
        # sys.path.append(os.environ['PYSPARK_PYTHON'] + '/python/lib/py4j-0.9-src.zip"')

        # TODO: look a way to ignore pyspark warnings at the beginning

        def setup_spark_context_with_warnings():
            # with block_output(suppress_stdout=True, suppress_stderr=True):
            self.findspark.init()

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

            # with block_output(suppress_stdout=True, suppress_stderr=True):
            sc = (self.SparkSession.builder.appName("hw").getOrCreate())

            # sqlContext = sc.sqlContext

            # with block_output(suppress_stdout=True, suppress_stderr=True):
            sqlContext = self.SQLContext(sc)

            # Conf details

            # ram_memory = 1

            # sc.conf.get('spark.executor.cores')
            # sc.conf.get('spark.executor.memory')

            #
            return sc, sqlContext

            # return None

        # Not working to present warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings('ignore')

            return setup_spark_context_with_warnings()

    def read_pems_counts_data(self, filepath: 'str', selected_period: {}) -> (pd.DataFrame, pd.DataFrame):

        # Read pems data with pyspark and select for a certain time window.

        # Raw data has no headers but they can be learned from the data dictionary

        # When the street station is counting at a segment with less than X lanes, then an empty value is recorded for the remaining lengths. Not sure which is the maximum number of lines in the dataset.

        # Read with pandas the first line of the dataset to have a sense of how many entries there are

        if 'hour' in selected_period.keys() and 'duration' in selected_period.keys():

            assert selected_period['duration']>0, 'duration must be greater than 0'

            print('Reading pems counts starting at ' + str(selected_period['hour']) + ':00' + ' and during ' + str(
                selected_period['duration']) + ' minutes')

        else:
            print('Reading pems counts')

        counts_head_df = pd.read_csv(filepath, nrows=10, header=None)

        # view(counts_head_df)
        # speed_df.iloc[0]

        # There are 12 lane independent attributes: len(lane_independent_columns)
        lane_independent_columns = [ \
            self.StructField('ts', self.StringType(), True), \
            self.StructField('station_id', self.IntegerType(), True), \
            self.StructField('district_id', self.IntegerType(), True), \
            self.StructField('freeway_id', self.IntegerType(), True), \
            self.StructField('travel_direction', self.StringType(), True), \
            self.StructField('lane_type', self.StringType(), True), \
            self.StructField('observed_lanes_pct', self.DoubleType(), True), \
            self.StructField('station_length', self.DoubleType(), True), \
            self.StructField('samples', self.IntegerType(), True), \
            self.StructField('flow_total', self.DoubleType(), True), \
            self.StructField('occupancy_avg', self.DoubleType(), True), \
            self.StructField('speed_avg', self.DoubleType(), True), \
            ]

        # The dataset have 52 columns: len(counts_df.columns)

        # There are 5 lane dependent attributes according to the pems dictionary, and thus, each of the attributes is computed under assumption of a maximum of 8 lanes (12 + 5 x 8 = 52)

        lane_dependent_attributes = ['samples_lane', 'flow_total_lane', 'occupancy_avg_lane', 'speed_avg_lane',
                                     'observed_lane']

        # This was being done badly as the lane dependent attributes are included for one lane at a time and not for all lanes and each attribute one time.
        # lane_dependent_labels = [str(i) + '_' + str(j) for i in lane_dependent_attributes for j in np.arange(1, 9)]

        lane_dependent_labels = [str(j) + '_' + str(i) for i in np.arange(1, 9) for j in lane_dependent_attributes]

        lane_dependent_columns = [self.StructField(key, self.DoubleType(), True) for key in lane_dependent_labels]

        custom_schema = self.StructType(lane_independent_columns + lane_dependent_columns)

        counts_sdf = self.sqlContext.read.csv(filepath, header=False, schema=custom_schema)

        # counts_sdf.show(2)

        # counts_sdf selected_time

        counts_sdf = counts_sdf.withColumn("ts_casted", self.F.to_timestamp(self.F.col("ts"), "MM/dd/yyyy HH:mm:ss"))

        # https://stackoverflow.com/questions/30949202/spark-dataframe-self.TimestampType-how-to-get-year-month-day-values-from-field
        counts_sdf = counts_sdf.withColumn('year', self.F.year(counts_sdf.ts_casted)) \
            .withColumn('month', self.F.month(counts_sdf.ts_casted)) \
            .withColumn('day_month', self.F.dayofmonth(counts_sdf.ts_casted)) \
            .withColumn('hour', self.F.hour(counts_sdf.ts_casted)) \
            .withColumn('minute', self.F.minute(counts_sdf.ts_casted))

        # counts_sdf.printSchema()

        # Compute average flow per station for selected time

        # Filter observatiosn based on selected time
        if 'hour' in selected_period.keys() and 'duration' in selected_period.keys():

            # Account for duration in minutes from the beginning of the hour

            delta_hours = int(selected_period['duration'] / 60)
            delta_minutes = selected_period['duration'] - delta_hours * 60

            count_interval_sdf = counts_sdf.filter((counts_sdf.hour >= selected_period['hour']) & (
                    counts_sdf.hour <= selected_period['hour'] + delta_hours))

            if delta_minutes > 0:
                count_interval_sdf = count_interval_sdf.filter(
                    (count_interval_sdf.hour >= selected_period['hour'] + delta_hours) & (
                            count_interval_sdf.minute <= delta_minutes))

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

        count_interval_sdf = count_interval_sdf.na.drop \
            (subset=['flow_total']).groupby('station_id').agg(
            self.F.sum('flow_total').alias('flow_total')
            , self.F.sum('flow_total_lane_1').alias('flow_total_lane_1')
            , self.F.sum('flow_total_lane_2').alias('flow_total_lane_2')
            , self.F.sum('flow_total_lane_3').alias('flow_total_lane_3')
            , self.F.sum('flow_total_lane_4').alias('flow_total_lane_4')
            , self.F.sum('flow_total_lane_5').alias('flow_total_lane_5')
            , self.F.sum('flow_total_lane_6').alias('flow_total_lane_6')
            , self.F.sum('flow_total_lane_7').alias('flow_total_lane_7')
            , self.F.sum('flow_total_lane_8').alias('flow_total_lane_8')
            , self.F.mean('flow_total').alias('flow_total_avg')
            , self.F.stddev('flow_total').alias('flow_total_sd')
        )  # .cache()

        # count_interval_sdf.show()

        # count_interval_sdf.count()

        # Compute average speed for each station to compare against Inrix data.
        # Note that pems is in mile units whereas inrix in kmh (1mph = 1.61 kph). Everything will be left in miles per hour

        # speed_interval_pems_sdf = counts_sdf.na.drop(
        #     subset=['speed_avg']) \
        #     .withColumn('speed_avg_kmh', self.F.col('speed_avg')) \
        #     .groupby('station_id').agg(
        #     self.F.min('speed_avg').alias('speed_min'),
        #     self.F.max('speed_avg').alias('speed_max'),
        #     self.F.mean('speed_avg').alias('speed_avg'),
        #     self.F.stddev('speed_avg').alias('speed_sd')).cache()
        #
        # # I found the average speeds to high. The minimum is 84km.
        # speed_interval_pems_sdf.select(self.F.min(self.F.col('speed_avg'))).show()

        # df.filter(df['ts'] >= self.F.lit('2018-06-27 00:00:00'))

        # # Create pandas dataframe
        # count_interval_df = count_interval_sdf.toPandas()

        self.clear_sql_cache()

        return count_interval_sdf.toPandas()

    def selected_period_filter(self, sdf: pyspark.sql.DataFrame, selected_period: {}):

        selected_period_sdf = sdf

        if 'year' in selected_period.keys():
            # speed_segment_sdf = speed_sdf.where(self.F.col("segment_id").isin([link.id]))
            # if isinstance(years,list):
            selected_period_sdf = selected_period_sdf.where(self.F.col("year").isin(selected_period['year']))

        if 'month' in selected_period.keys():
            selected_period_sdf = selected_period_sdf.where(self.F.col("month").isin(selected_period['month']))

        if 'day_month' in selected_period.keys():
            selected_period_sdf = selected_period_sdf.where(self.F.col("day_month").isin(selected_period['day_month']))

        if 'day_week' in selected_period.keys():
            selected_period_sdf = selected_period_sdf.where(self.F.col("day_week").isin(selected_period['day_week']))

        if 'hour' in selected_period.keys():
            selected_period_sdf = selected_period_sdf.where(self.F.col("hour").isin(selected_period['hour']))

        if 'minute' in selected_period.keys():
            selected_period_sdf = selected_period_sdf.where(self.F.col("minute").isin(selected_period['minute']))

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

        custom_schema = self.StructType([ \
            self.StructField('Date Time', self.DateType(), True), \
            self.StructField('Segment ID', self.IntegerType(), True), \
            self.StructField('UTC Date Time', self.TimestampType(), True), \
            self.StructField('Speed(km/hour)', self.DoubleType(), True), \
            self.StructField('Hist Av Speed(km/hour)', self.DoubleType(), True), \
            self.StructField('Ref Speed(km/hour)', self.DoubleType(), True), \
            self.StructField('Travel Time(Minutes)', self.DoubleType(), True), \
            self.StructField('CValue', self.DoubleType(), True), \
            self.StructField('Pct Score30', self.DoubleType(), True), \
            self.StructField('Pct Score20', self.DoubleType(), True), \
            self.StructField('Pct Score10', self.DoubleType(), True), \
            self.StructField('Road Closure', self.StringType(), True), \
            self.StructField('Corridor/Region Name', self.StringType(), True)
        ]
        )

        # custom_schema = self.StructType([ \
        #     self.StructField('dt', self.DateType(), True), \
        #     self.StructField('segment_id', self.IntegerType(), True), \
        #     self.StructField('ts', self.TimestampType(), True), \
        #     self.StructField('speed', self.DoubleType(), True), \
        #     self.StructField('speed_hist', self.DoubleType(), True), \
        #     self.StructField('speed_ref', self.DoubleType(), True), \
        #     self.StructField('travel_time', self.DoubleType(), True), \
        #     self.StructField('confidence', self.DoubleType(), True), \
        #     self.StructField('pct30', self.DoubleType(), True), \
        #     self.StructField('pct20', self.DoubleType(), True), \
        #     self.StructField('pct10', self.DoubleType(), True), \
        #     self.StructField('road_closure', self.StringType(), True), \
        #     self.StructField('name', self.StringType(), True)
        # ]
        # )

        # TODO: ask/understand what Pct Score30,Pct Score20,Pct Score10 mean.

        # df = self.sqlContext.read.format("csv").option("header", "true"). \
        #     schema(custom_schema). \
        #     load(filepaths.split(','))

        # filepaths = ['/Users/pablo/google-drive/data-science/github/isuelogit/input/private/Fresno/inrix/speed/Fresno_CA_2019-10-01_to_2019-11-01_15_min_part_1/data.csv', '/Users/pablo/google-drive/data-science/github/isuelogit/input/private/Fresno/inrix/speed/Fresno_CA_2019-10-01_to_2019-11-01_15_min_part_2/data.csv']

        inrix_data_sdf = self.sqlContext.read.csv(filepaths, header=True, schema=custom_schema)

        new_columns = ['dt', 'segment_id', 'ts', 'speed', 'speed_hist', 'speed_ref', 'travel_time', 'confidence',
                       'pct30', 'pct20', 'pct10', 'road_closure', 'name']

        # Rename columsn
        inrix_data_sdf = inrix_data_sdf.toDF(
            *new_columns)  # .cache() Ram cannot hold the entire set as it is just to big

        # Cast road_closure attribute to numeric

        # inrix_data_sdf = inrix_data_sdf.withColumn('road_closure', self.F.col('road_closure').cast('integer'))

        # inrix_data_sdf = inrix_data_sdf.withColumn('road_closure', when(self.F.col('road_closure').cast('integer'))

        inrix_data_sdf = inrix_data_sdf.withColumn('road_closure_num',
                                                   self.F.when(inrix_data_sdf.road_closure == 'T', 1).otherwise(0))

        # Convert speed units from km to miles for consistency with Sean file
        km_to_miles_factor = 0.62137119223

        inrix_data_sdf = inrix_data_sdf.withColumn("speed", km_to_miles_factor * self.F.col("speed"))

        inrix_data_sdf = inrix_data_sdf.withColumn("speed_ref", km_to_miles_factor * self.F.col("speed_ref"))

        inrix_data_sdf = inrix_data_sdf.withColumn("speed_hist", km_to_miles_factor * self.F.col("speed_hist"))

        # Compute time data features

        # inrix_data_sdf = inrix_data_sdf.withColumn("dt_casted", self.F.to_timestamp(self.F.col("dt"),"MM/dd/yyyy HH:mm:ss" ))

        # Note that day_week == 2 is Monday not Tuesday as apparently days start from sunday
        # https://mungingdata.com/apache-spark/week-end-start-dayofweek-next-day/

        inrix_data_sdf = inrix_data_sdf \
            .withColumn('year', self.F.year(inrix_data_sdf.dt)) \
            .withColumn('month', self.F.month(inrix_data_sdf.dt)) \
            .withColumn('day_month', self.F.dayofmonth(inrix_data_sdf.dt)) \
            .withColumn('day_week', self.F.dayofweek(inrix_data_sdf.dt)) \
            .withColumn('hour', self.F.hour(inrix_data_sdf.ts)) \
            .withColumn('minute', self.F.minute(inrix_data_sdf.ts))

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
            self.F.count('speed').alias('n')

            , self.F.mean('speed').alias('speed_avg')
            , self.F.stddev('speed').alias('speed_sd')
            , self.F.variance('speed').alias('speed_var')
            , self.F.min('speed').alias('speed_min')
            , self.F.max('speed').alias('speed_max')

            , self.F.mean('speed_hist').alias('speed_hist_avg')
            , self.F.stddev('speed_hist').alias('speed_hist_sd')
            , self.F.variance('speed_hist').alias('speed_hist_var')

            , self.F.mean('speed_ref').alias('speed_ref_avg')
            , self.F.stddev('speed_ref').alias('speed_ref_sd')
            , self.F.variance('speed_ref').alias('speed_ref_var')

            , self.F.min('travel_time').alias('traveltime_min')
            , self.F.max('travel_time').alias('traveltime_max')
            , self.F.mean('travel_time').alias('traveltime_avg')
            , self.F.stddev('travel_time').alias('traveltime_sd')
            , self.F.variance('travel_time').alias('traveltime_var')

            , self.F.mean('road_closure_num').alias('road_closure_avg')
            , self.F.max('road_closure_num').alias('road_closure_any')
        )

        inrix_data_segments_df = inrix_data_segments_sdf.toPandas()

        # Add coeficient of variation for travel time and speed
        inrix_data_segments_df['traveltime_cv'] = inrix_data_segments_df['traveltime_sd'] / inrix_data_segments_df[
            'traveltime_avg']

        inrix_data_segments_df['speed_cv'] = inrix_data_segments_df['speed_sd'] / inrix_data_segments_df['speed_avg']

        return inrix_data_segments_df

    def write_partition_inrix_data(self, filepaths, output_folderpath, unit='day'):

        # small_speed_df1 = pd.read_csv(filepaths[0], nrows=1000)
        # small_speed_df2 = pd.read_csv(filepaths[1], nrows=1000)
        #
        # print(small_speed_df1.head())
        # print(small_speed_df2.head())

        # https://stackoverflow.com/questions/62268860/pyspark-save-dataframe-to-single-csv-by-date-using-window-function

        # https://stackoverflow.com/questions/59278835/pyspark-how-to-write-dataframe-partition-by-year-month-day-hour-sub-directory

        # custom_schema = self.StructType([ \
        #     self.StructField('dt', self.DateType(), True), \
        #     self.StructField('segment_id', self.IntegerType(), True), \
        #     self.StructField('ts', self.TimestampType(), True), \
        #     self.StructField('speed', self.DoubleType(), True), \
        #     self.StructField('speed_hist', self.DoubleType(), True), \
        #     self.StructField('speed_ref', self.DoubleType(), True), \
        #     self.StructField('travel_time', self.DoubleType(), True), \
        #     self.StructField('confidence', self.DoubleType(), True), \
        #     self.StructField('pct30', self.DoubleType(), True), \
        #     self.StructField('pct20', self.DoubleType(), True), \
        #     self.StructField('pct10', self.DoubleType(), True), \
        #     self.StructField('road_closure', self.StringType(), True), \
        #     self.StructField('name', self.StringType(), True)
        # ]
        # )

        custom_schema = self.StructType([ \
            self.StructField('Date Time', self.DateType(), True), \
            self.StructField('Segment ID', self.IntegerType(), True), \
            self.StructField('UTC Date Time', self.TimestampType(), True), \
            self.StructField('Speed(km/hour)', self.DoubleType(), True), \
            self.StructField('Hist Av Speed(km/hour)', self.DoubleType(), True), \
            self.StructField('Ref Speed(km/hour)', self.DoubleType(), True), \
            self.StructField('Travel Time(Minutes)', self.DoubleType(), True), \
            self.StructField('CValue', self.DoubleType(), True), \
            self.StructField('Pct Score30', self.DoubleType(), True), \
            self.StructField('Pct Score20', self.DoubleType(), True), \
            self.StructField('Pct Score10', self.DoubleType(), True), \
            self.StructField('Road Closure', self.StringType(), True), \
            self.StructField('Corridor/Region Name', self.StringType(), True)
        ]
        )

        inrix_data_sdf = self.sqlContext.read.csv(filepaths, header=True, schema=custom_schema)

        if unit == 'day':

            # Get unique yyyymmddhh values in the grouping column
            groups = [x[0] for x in inrix_data_sdf.select('Date Time').distinct().collect()]

            # Create a filtered DataFrame
            groups_list = [inrix_data_sdf.filter(self.F.col('Date Time') == x) for x in groups]

            # Save the result by yyyy/mm/dd/hh
            for filtered_data in groups_list:
                target_date = filtered_data.select('Date Time').take(1)[0].asDict()['Date Time']
                print('writing ' + str(target_date) + '.csv')
                path = output_folderpath + str(target_date) + '.csv'

                filtered_data.toPandas().to_csv(path, index=False)

        if unit == 'year':

            inrix_data_sdf = inrix_data_sdf \
                .withColumn('year', self.F.year(inrix_data_sdf.select('Date Time'))) \
 \
                # Get unique yyyymmddhh values in the grouping column
            groups = [x[0] for x in inrix_data_sdf.select('year').distinct().collect()]

            # Create a filtered DataFrame
            groups_list = [inrix_data_sdf.filter(self.F.col('year') == x) for x in groups]

            # Save the result by yyyy/mm/dd/hh
            for filtered_data in groups_list:
                target_date = filtered_data.select(inrix_data_sdf).take(1)[0].asDict()[inrix_data_sdf]
                filtered_data = filtered_data.drop('year')
                print('writing ' + str(target_date) + '.csv')
                path = output_folderpath + str(target_date) + '.csv'

                filtered_data.toPandas().to_csv(path, index=False)

    def read_traffic_incidents(self, filepath: str, selected_period: {}) -> pd.DataFrame:

        print('\nReading traffic incidents data')

        # traffic_incidents_head_df = pd.read_csv(filepath, nrows=10)

        # traffic_incidents_df = pd.read_csv(filepath)

        selected_columns = ['Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng',
                            'County']

        # selected_columns_schema = [ \
        #     self.StructField('Severity', self.StringType(), True), \
        #     self.StructField('Start_Time', self.StringType(), True), \
        #     self.StructField('End_Time', self.StringType(), True), \
        #     self.StructField('Start_Lat', self.DoubleType(), True), \
        #     self.StructField('Start_Lng', self.DoubleType(), True), \
        #     self.StructField('End_Lat', self.DoubleType(), True), \
        #     self.StructField('End_Lng', self.DoubleType(), True), \
        #     self.StructField('County', self.StringType(), True), \
        #     ]

        # printer.blockPrint()
        incidents_sdf = self.sqlContext.read.csv(filepath, header=True).select(selected_columns)

        incidents_sdf = incidents_sdf.filter(incidents_sdf.County == 'Fresno')

        incidents_sdf = incidents_sdf \
            .withColumn('year', self.F.year(incidents_sdf.Start_Time)) \
            .withColumn('month', self.F.month(incidents_sdf.Start_Time)) \
            .withColumn('day_month', self.F.dayofmonth(incidents_sdf.Start_Time)) \
            .withColumn('hour', self.F.hour(incidents_sdf.Start_Time)) \
            .withColumn('minute', self.F.minute(incidents_sdf.Start_Time))

        # incidents_period_sdf = incidents_sdf

        # if 'year' in selected_period.keys():
        #     # speed_segment_sdf = speed_sdf.where(self.F.col("segment_id").isin([link.inrix_id]))
        #     # if isinstance(years,list):
        #     incidents_period_sdf = incidents_period_sdf.where(self.F.col("year").isin(selected_period['year']))

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

        # incidents_sdf.groupBy('year').count().show()
        # incidents_sdf.count()

        incidents_df = incidents_period_sdf.toPandas()

        # printer.enablePrint()

        return incidents_df


def load_pems_stations_ids(network):

    filepath = config.dirs['input_folder'] + "private/Fresno/network/qgis/adjusted/Fresno_links_adj.shp"

    network_gdf = geographer.read_qgis_shp_fresno(filepath=filepath)

    geographer.manual_match_network_and_stations_fresno(
        network_gdf=network_gdf,
        links=network.get_regular_links())





def read_spatiotemporal_data_fresno(network,
                                    data_analyst,
                                    lwrlk_only = True,
                                    **kwargs
                                    ) -> pd.Dataframe:

    options = kwargs

    manual_matching = True
    build_fresno_gis_files = False
    # Fresno_network_gdf = False

    # Summary with link types
    link_types_pd = pd.DataFrame({'link_types': [link.link_type for link in network.links]})

    with pd.option_context('display.float_format', "{:.1%}".format):
        print(pd.concat([link_types_pd.value_counts(normalize=True), link_types_pd.value_counts()]
                        , axis=1, keys=('perc', 'count')).to_string())

    # Links
    if lwrlk_only:
        links = network.get_regular_links()
        print('\nMatching geospatial datasets to links with type "LWRLK" ')
    else:
        links = network.links
        print('\nMatching geospatial datasets to all links types" ')

    # Initial link attributes
    initial_attr_links = list(links[0].Z_dict.keys())

    # TODO: enable warm start for some operations as some ofthe processing with spark or geopandas can be done offline
    # (e.g census data and inrix line to line matching is the same regardless the period of analysis)

    # i) Network data

    # Rescaling coordinates to ease the matching of the x,y coordinates to real coordinates in Qqis
    geographer.adjust_fresno_nodes_coordinates(nodes=network.nodes,
                                               rescale_factor=1)

    # Set link orientation according to the real coordinates
    geographer.set_cardinal_direction_links(links=links)

    if build_fresno_gis_files:
        # a) Write line and points shapefiles for further processing (need to be done only once)

        geographer.write_node_points_shp(
            nodes=network.nodes
            , folderpath=config.dirs['output_folder'] + 'gis/Fresno/network/nodes'
            , networkname='Fresno'
        )

        geographer.write_line_segments_shp(
            links=links,
            folderpath=config.dirs['output_folder'] + 'gis/Fresno/network/links',
            networkname='Fresno'
        )

    # b) Read shapefile of layer edited in Qgis where final polish was made (additional spatial adjustments)

    # network_filepath =  "/Users/pablo/google-drive/university/cmu/2-research/datasets/private/od-fresno-sac/SR41/shapefile/fresno-gis-nad83.shp"

    # network_filepath = config.filepaths['input_folder'] + "private/Fresno/network/qgis/raw/fresno-qgis-nad83-adj.shp"

    network_filepath = config.dirs['input_folder'] + "private/Fresno/network/qgis/adjusted/Fresno_links_adj.shp"

    network_gdf = geographer.read_qgis_shp_fresno(filepath=network_filepath)

    # TODO: update node/link coordinates consisently

    # ii) PEMS stations

    if manual_matching is False:

        # Read data for PeMS stations for fresno and return a geodataframe. Then, match pems stations with network links
        path_pems_stations = 'input/public/pems/stations/raw/D06/' + 'd06_text_meta_2020_09_11.txt'

        # Original
        pems_stations_gdf = geographer.read_pems_stations_fresno(
            filepath=path_pems_stations,
            adjusted_gis_stations=True
        )

        geographer.match_network_and_stations_fresno(
            stations_gdf=pems_stations_gdf
            , network_gdf=network_gdf
            , links= links
            , folderpath=config.dirs['output_folder'] + 'gis/Fresno/pems-stations'
            , adjusted_gis_stations=True
        )

    else:

        # Assignemnt of station ids according to manual match made in qgis and recorded in the columns:
        # pems_id1, pems_id2, pems_id3, which are three candidate stations.

        # Path of shapefile with adjustment made in qgis
        # path_pems_stations = config.dirs['input_folder'] + '/public/pems/stations/gis/adjusted/fresno_stations_adj.shp'

        geographer.manual_match_network_and_stations_fresno(
            network_gdf=network_gdf,
            links = links)

    # iii) INRIX data

    # Read and match inrix data

    if options['data_processing']['inrix_segments']:

        path_inrix_shp = config.dirs[
                             'input_folder'] + 'private/Fresno/inrix/shapefiles/USA_CA_RestOfState_shapefile/USA_CA_RestOfState.shp'

        inrix_gdf = geographer.read_inrix_shp(filepath=path_inrix_shp,
                                              county='Fresno')

        # Do this only once
        # geographer.export_inrix_shp(inrix_gdf, folderpath = config.filepaths['folder_gis_data']+'Fresno/inrix')

        # inrix_gdf.plot(figsize=(5, 5), edgecolor="purple", facecolor="None")
        # plt.show()

        links_buffer_inrix_gdf = geographer.match_network_links_and_inrix_segments_fresno(
            inrix_gdf=inrix_gdf,
            network_gdf=network_gdf,
            links=links,
            buffer_size=options['buffer_size']['inrix'],
            centroids=True
        )

        # Export buffer created to merge data from inrix segments with network links
        geographer.export_buffer_shp(
            gdf=links_buffer_inrix_gdf,
            folderpath=config.dirs['output_folder'] + 'gis/Fresno/inrix/',
            filename='links_buffer_inrix'
        )

        if options['data_processing']['inrix_data']:

            # Paths were the original data is stored
            path_inrix_data_part1 = config.dirs[
                                        'input_folder'] + 'private/Fresno/inrix/speed/Fresno_CA_2019-10-01_to_2019-11-01_15_min_part_1/data.csv'

            path_inrix_data_part2 = config.dirs[
                                        'input_folder'] + 'private/Fresno/inrix/speed/Fresno_CA_2019-10-01_to_2019-11-01_15_min_part_2/data.csv'

            filepaths = [path_inrix_data_part1, path_inrix_data_part2]

            if options['write_inrix_daily_data']:
                data_analyst.write_partition_inrix_data(
                    filepaths,
                    output_folderpath=config.dirs['input_folder'] + '/private/Fresno/inrix/speed/by-day/')

            if options['read_inrix_daily_data']:

                filepaths = config.dirs['input_folder'] + '/private/Fresno/inrix/speed/by-day/' + \
                            options['selected_date'] + '.csv'

                inrix_data_df = data_analyst.generate_inrix_data_by_segment(
                    filepaths=filepaths,
                    selected_period=options['selected_period_inrix']
                )

            else:
                # Generate a pandas dataframe with the average and standard deviaetion of the speed among INRIX link segments
                inrix_data_df = data_analyst.generate_inrix_data_by_segment(
                    filepaths=filepaths,
                    selected_period= options['selected_period_inrix'])

            # Merge speed data based on the inrix_id among links
            data_analyst.merge_inrix_data(
                links=links,
                speed_df=inrix_data_df
                # , options={**config.sim_options, **config.estimation_options}
                , options = options
            )

        # inrix_speed_sdf.head()

    # iv) Census data

    # Read census data and match it with network links

    # Write shapefile with relevant block data from TIGER files (need to be done only once)
    # geographer \
    #     .write_census_blocks_data_fresno(countyname= 'Fresno'
    #                                      , filepath= '/Users/pablo/google-drive/data-science/github/isuelogit/input/public/census/ACS_2018_5YR_BG_06_CALIFORNIA.gdb')

    if options['data_processing']['census']:

        census_tract_path = config.dirs[
                                'output_folder'] + 'gis/Fresno/census/Fresno_census_shp/Fresno_census_shp.shp'
        census_tracts_data_gdf = geographer.read_census_tracts_shp_fresno(filepath=census_tract_path)

        if options['inrix_matching']['census'] and options['inrix_matching']:
            geographer.match_network_links_and_census_tracts_fresno(
                census_tracts_gdf=census_tracts_data_gdf
                , network_gdf=inrix_gdf
                , links=links
                , attrs=['median_inc', 'median_age']
                , inrix_matching=True
            )

        else:
            geographer.match_network_links_and_census_tracts_fresno(
                census_tracts_gdf=census_tracts_data_gdf
                , network_gdf=network_gdf
                , links=links
                , attrs=['median_inc', 'median_age']
                , inrix_matching=False
            )

    if options['data_processing']['incidents']:

        # v) Traffic incident data

        # Read and match traffic incident data
        incidents_path = config.dirs['input_folder'] + "public/traffic-incidents/US_Accidents_Dec20.csv"

        traffic_incidents_Fresno_df = data_analyst.read_traffic_incidents(
            filepath=incidents_path,
            selected_period= options['selected_period_incidents']
        )

        # # Export shapefile with traffic incidents locations
        # geographer.export_fresno_incidents_shp(incidents_df = traffic_incidents_Fresno, folderpath = config.filepaths['folder_gis_data']+'Fresno/incidents')

        if options['inrix_matching']['incidents'] and options['inrix_matching']:
            links_buffer_incidents_gdf \
                = geographer.match_network_links_and_fresno_incidents(
                incidents_df=traffic_incidents_Fresno_df,
                network_gdf=inrix_gdf,
                links=links,
                buffer_size= options['buffer_size'][
                    'incidents']
                , inrix_matching=True
            )

        else:

            # Count the number of incidents within a buffer of each link to have an indicator about safety

            # It allows to select month as well, so the incident information matches date of traffic counts
            links_buffer_incidents_gdf \
                = geographer.match_network_links_and_fresno_incidents(
                incidents_df=traffic_incidents_Fresno_df,
                network_gdf=network_gdf,
                links=links,
                buffer_size= options['buffer_size']['incidents'],
                inrix_matching=False
            )

            # Export buffer created to merge data from incidents with network links
            geographer.export_buffer_shp(gdf=links_buffer_incidents_gdf,
                                         folderpath=config.dirs['output_folder'] + 'gis/Fresno/incidents/',
                                         filename='links_buffer_incidents'
                                         )

    # vi) Bus stop information (originally with txt file, similar to pems stations)

    if options['data_processing']['bus_stops']:

        # Path of bus stop txt file
        input_path_bus_stops_fresno = config.dirs['input_folder'] + "public/transit/adjusted/stops.txt"

        # Read txt file and return pandas dataframe
        bus_stops_fresno_df = data_analyst.read_bus_stops_txt(filepath=input_path_bus_stops_fresno)

        # Return geodataframe and export a shapefile if required (that may be read in qgis)
        bus_stops_fresno_gdf = geographer.generate_fresno_bus_stops_gpd(bus_stops_df=bus_stops_fresno_df)

        # # Export bus stop shapefile
        # geographer.export_bus_stops_shp(bus_stops_gdf = bus_stops_fresno_gdf,
        #                                     folderpath = config.filepaths['output_folder'] + 'gis/Fresno/bus-stops'
        #                                     ,config = config
        #                                     )

        if options['inrix_matching']['bus_stops'] and options['inrix_matching']:

            links_buffer_bus_stops_gdf \
                = geographer.match_network_links_and_fresno_bus_stops(bus_stops_gdf=bus_stops_fresno_gdf
                                                                      , network_gdf=inrix_gdf
                                                                      , links=links
                                                                      , buffer_size=
                                                                      options['buffer_size'][
                                                                          'bus_stops']
                                                                      , inrix_matching=True
                                                                      )

        else:

            # Match bus stops with network links

            # - Count the number of bus stops within a buffer around each link
            links_buffer_bus_stops_gdf \
                = geographer.match_network_links_and_fresno_bus_stops(bus_stops_gdf=bus_stops_fresno_gdf
                                                                      , network_gdf=network_gdf
                                                                      , links=links
                                                                      , buffer_size=
                                                                      options['buffer_size'][
                                                                          'bus_stops']
                                                                      , inrix_matching=False
                                                                      )

            geographer.export_buffer_shp(gdf=links_buffer_bus_stops_gdf
                                         , folderpath=config.dirs['output_folder'] + 'gis/Fresno/bus-stops/'
                                         , filename='links_buffer_bus_stops'
                                         )

    # vii) Streets intersections

    if options['data_processing']['streets_intersections']:

        input_path_intersections_fresno = config.dirs[
                                              'input_folder'] + "public/traffic/streets-intersections/adjusted/intrsect.shp"

        # - Return a geodataframe to perform gis matching later
        streets_intersections_fresno_gdf = geographer.generate_fresno_streets_intersections_gpd(
            filepath=input_path_intersections_fresno)

        if options['inrix_matching']['streets_intersections'] and options[
            'inrix_matching']:

            links_buffer_streets_intersections_gdf \
                = geographer.match_network_links_and_fresno_streets_intersections(
                streets_intersections_gdf=streets_intersections_fresno_gdf
                , network_gdf=inrix_gdf
                , links=links
                , buffer_size= options['buffer_size']['streets_intersections']
                , inrix_matching=True
            )


        else:

            # - Count the number of intersections within a buffer around each link
            links_buffer_streets_intersections_gdf \
                = geographer.match_network_links_and_fresno_streets_intersections(
                streets_intersections_gdf=streets_intersections_fresno_gdf
                , network_gdf=network_gdf
                , links=links
                , buffer_size= options['buffer_size']['streets_intersections']
                , inrix_matching=False
            )

            geographer.export_buffer_shp(gdf=links_buffer_streets_intersections_gdf
                                         , folderpath=config.dirs[
                                                          'output_folder'] + 'gis/Fresno/streets-intersections'
                                         , filename='links_buffer_streets_intersections'
                                         )

    attr_links = list(network.links[0].Z_dict.keys())

    # List of new attributes
    new_attr_links = list(set(attr_links)-set(initial_attr_links))

    for key in new_attr_links:
        for link in network.get_non_regular_links():
            link.Z_dict[key] = 0

    print('\nFeatures values of link with types different than "LWRLK" were set to 0\n')

    # Create a pandas dataframe with the new link attributes
    links_df = network.Z_data[new_attr_links]

    links_df.insert(0,'link_key', network.links_keys)

    return links_df, new_attr_links



def generate_fresno_pems_counts(links: Links,
                                data: pd.DataFrame,
                                flow_attribute: str,
                                flow_factor = 1) -> Dict:

    """
    Generate masked fresno counts

    :param links:
    :param data:
    :return:
    """

    # TODO: normalized flow total by the number of lanes with data modyfing the read_pems_count method. The flow factor should
    # not required for the algorithm to work. Need to confirm if links in the Fresno network correspond to an aggregate of lanes

    print('\nMatching PEMS traffic count measurements in network links')

    xct = np.empty(len(links))
    xct[:] = np.nan
    xct_dict = {}

    n_imputations = 0
    n_perfect_matches = 0

    for link, i in zip(links, range(len(links))):
        # if link.pems_station_id == data['station_id'].filter(link.pems_station_id):

        station_rows = data[data['station_id'].isin(link.pems_stations_ids)]

        if flow_attribute == 'flow_total':

            if len(station_rows) > 0:
                xct[i] = np.mean(station_rows[flow_attribute])*flow_factor

        if flow_attribute == 'flow_total_lane':
            lane = link.Z_dict['lane']
            lane_label = flow_attribute + '_' + str(lane)

            if len(station_rows) > 0:

                lane_counts = np.nan

                if np.count_nonzero(~np.isnan(station_rows[lane_label])) > 1:

                    # The complicated cases are those were more than one station gives no nas and the counts are different.
                    lane_counts = np.nanmean(station_rows[lane_label])

                elif np.count_nonzero(~np.isnan(station_rows[lane_label])) == 1:

                    # If two stations are matched but only one gives a no nan link flows, this means that
                    # the list of pems stations ids can be safely reduced

                    link.pems_stations_ids = [link.pems_stations_ids[np.where(~np.isnan(station_rows[lane_label]))[0][0]]]

                    lane_counts = list(station_rows[lane_label])[0]

                else:

                    # Theses cases requires imputation

                    # print(station_rows[lane_label])

                    #TODO: review these cases and maybe just define them as outliers
                    pass


                # Imputation: If no counts are available for the specific lane, we may the average over the non nan values for imputation

                if np.isnan(float(lane_counts)):

                    n_imputations += 1

                    lanes_labels = [flow_attribute + '_' + str(j) for j in np.arange(1,9)]

                    lanes_flows = [np.nansum(station_rows[lanes_labels[j]]) for j in range(len(lanes_labels))]

                    lanes_non_na_count = [np.count_nonzero(~np.isnan(station_rows[lanes_labels[j]])) for j in range(len(lanes_labels))]

                    total_no_nan_lanes = np.sum(lanes_non_na_count)

                    xct[i] = np.nansum(lanes_flows)/total_no_nan_lanes

                else:
                    xct[i] = lane_counts
                    n_perfect_matches += 1

        xct_dict[link.key] = xct[i]

    print(n_perfect_matches, 'links were perfectly matched')
    print(n_imputations, 'links counts were imputed using the average traffic counts among lanes')

    return xct_dict

def adjust_counts_by_link_capacity(network,
                                   counts):

    # If the link counts are higher than the capacity of the lane, we applied an integer deflating factor until
    # the capacity of the link is not surpasses. This tries to correct for the fact that the link counts may be recording
    # more than 1 lane.

    # The numebr of lane information in Sean file to match the corresponding lane from PEMS. This should reduce the adjustments by capacity

    x_bar = np.array(list(counts.values()))[:, np.newaxis]

    link_capacities = np.array([link.bpr.k for link in network.links])

    # current_error_by_link = error_by_link(x_bar, predicted_counts, show_nan=False)

    # matrix_error = np.append(
    #     np.append(np.append(predicted_counts[idx_nonas], x_bar[idx_nonas], axis=1), current_error_by_link, axis=1), link_capacities,
    #     axis=1)

    x_bar_adj = copy.deepcopy(x_bar)

    counter = 0

    adj_factors = []

    for i in range(x_bar.shape[0]):

        adj_factors.append(np.isnan)

        if not np.isnan(x_bar.flatten()[i]):

            factor = 1

            while x_bar_adj.flatten()[i] > link_capacities[i]:

                factor += 1

                if factor == 2:
                    counter+=1

                x_bar_adj[i] = x_bar[i]/factor


            adj_factors[i] = factor

    print('A total of ' + str(counter) + ' links counts were adjusted by capacity')

    links_keys = [(key[0],key[1]) for key in list(counts.keys())]

    # Link capaciies higher than 10000 are show as inf and the entire feature as a string to reduce space when printing

    link_capacities_print = []

    for i in range(len(link_capacities)):
        if link_capacities[i] > 10000:
            link_capacities_print.append(float('inf'))
        else:
            link_capacities_print.append(link_capacities[i])


    link_adjustment_df = pd.DataFrame({
        'link_key': links_keys
        , 'capacity': link_capacities_print
        , 'old_counts': x_bar.flatten()
        ,'adj_counts': x_bar_adj.flatten()
        ,'adj_factor': adj_factors
    })

    mask = link_adjustment_df.isnull().any(axis=1)

    # print(link_adjustment_df[~mask].to_string())

    with pd.option_context('display.float_format', '{:0.1f}'.format):
        print(link_adjustment_df[~mask].to_string())

    return dict(zip(counts.keys(), x_bar_adj.flatten()))


def feature_engineering_fresno(links, network, lwrlk_only = True):

    ''' Links may be of type lwrlk only'''

    # Initial link attributes
    existing_Z_attrs = links[0].Z_dict

    # i) High and low income dummies

    # - Percentile used to segmentation income level from CENSUS blocks income data (high income are those links with income higher than pct)
    pct_income = 30

    # - Get percentile income distribution first
    if 'median_inc' in existing_Z_attrs:
        links_income_list = [link.Z_dict['median_inc'] for link in network.get_regular_links()]

        # Create dummy variable for high income areas
        link_pct_income = np.percentile(np.array(links_income_list), pct_income)

        for link in links:

            if 'median_inc' in link.Z_dict:
                link.Z_dict['high_inc'] = 0
                link.Z_dict['low_inc'] = 1

                if link.Z_dict['median_inc'] >= link_pct_income:
                    link.Z_dict['high_inc'] = 1
                    link.Z_dict['low_inc'] = 0

    # (ii) No incidents
    if 'incidents' in existing_Z_attrs:
        for link in links:
            link.Z_dict['no_incidents'] = 1

            if link.Z_dict['incidents'] > 0:
                link.Z_dict['no_incidents'] = 0

    # (iii) No bus stops

    if 'bus_stops' in existing_Z_attrs:
        for link in links:
            link.Z_dict['no_bus_stops'] = 1

            if link.Z_dict['bus_stops'] > 0:
                link.Z_dict['no_bus_stops'] = 0

    # (iv) No street intersections

    if 'intersections' in existing_Z_attrs:

        for link in links:
            link.Z_dict['no_intersections'] = 1

            if link.Z_dict['intersections'] > 0:
                link.Z_dict['no_intersections'] = 0

    # (v) Travel time variability

    # - Adjusted standard deviation of travel time

    if 'tt_cv' in existing_Z_attrs:

        for link in links:
            link.Z_dict['tt_sd_adj'] = link.bpr.tf * link.Z_dict['tt_cv']


    # - Measure of reliability as PEMS which is relationship between true and free flow travel times
    if 'speed_avg' in existing_Z_attrs and 'speed_ref_avg' in existing_Z_attrs:

        for link in links:
            # link.Z_dict['tt_reliability'] = min(1,link.Z_dict['speed_avg']/link.Z_dict['speed_ref_avg'])
            if link.Z_dict['speed_ref_avg'] !=0:
                link.Z_dict['tt_reliability'] = link.Z_dict['speed_avg'] / link.Z_dict['speed_ref_avg']
            else:
                link.Z_dict['tt_reliability'] = 0

    new_features = ['low_inc', 'high_inc','no_incidents','no_bus_stops','no_intersections','tt_sd_adj','tt_reliability']

    if lwrlk_only:
        for key in new_features:
            for link in network.get_non_regular_links():
                link.Z_dict[key] = 0

        print('Features values of links with a type different than LWRLK were set to 0')

    # attr_links = list(links[0].Z_dict.keys())

    # # List of new attributes
    # new_features = list(set(attr_links)-set(existing_Z_attrs))

    # Create a pandas dataframe with the link attributes
    links_df = network.Z_data[new_features]

    links_df.insert(0,'key', network.links_keys)

    print('New features:', new_features)

    # return links_df, new_features


def remove_outliers_fresno(network):

    # ii) Outliers

    # - Remove traffic stations where a huge errors is observed and which indeed are considered outliers

    removed_links_keys = []

    outliers_links_keys = []

    # outliers_links_keys = [ (136, 385,'0'), (179, 183,'0'), (620, 270,'0'), (203, 415,'0') , (217, 528,'0'), (236, 260,'0'), (239, 242,'0'), (243, 261,'0'), (276, 277,'0'), (277, 278,'0'), (282, 281,'0'), (283, 197,'0'), (284, 285,'0'),(285, 286,'0'), (385, 203,'0'), (587, 583,'0'), (676, 174,'0'),  (677, 149,'0')]

    # Flow is too low maybe due to problems in od matrix
    # , (1590, 1765, '0'), (1039, 125, '0')

    for key in outliers_links_keys:
        removed_links_keys.append(key)

    od_connectors_keys = [(1542, 1717, '0'), (92, 1610, '0'), (114, 1571, '0'), (1400, 1657, '0'), (1244, 1632, '0'),
                          (1459, 996, '0')
        , (1444, 781, '0'), (42, 21, '0'), (1610, 1785, '0'), (1590, 1765, '0'), (1580, 1755, '0'), (1571, 1746, '0')]

    for key in od_connectors_keys:
        removed_links_keys.append(key)

    # List link with unlimited capacity as those are the ones with the higher errors, but this is probably because I did not perform adjustments

    removed_counter = 0
    counts = network.link_data.counts
    for link_key, count in network.link_data.counts.items():
        if link_key in removed_links_keys:

            if not np.isnan(counts[link_key]):
                removed_counter += 1

                network.links_dict[link_key].observed_count = np.nan
                counts[link_key] = np.nan
                # xc_validation[link_key] = np.nan

    network.load_traffic_counts(counts)

    print("\n" + str(
        removed_counter) + " traffic counts observations were removed because belonging to OD connectors or assumed to be outliers")

    # TODO: print rows with outliers
    new_total_counts_observations = np.count_nonzero(~np.isnan(np.array(list(counts.values()))))

    print('New total of link observations: ' + str(new_total_counts_observations))


class LinkData():
    """ Data must be provided at the link level"""

    # def __init__(self,
    #              link_key: Feature,
    #              count_key: Feature,
    #              dataframe: pd.DataFrame):

    #   assert link_key in dataframe.columns, 'No link_id column has been provided'
    #
    #   assert self.isValidLinkId(dataframe[link_key].iloc[0])
    #
    #     self.dataframe = dataframe
    #     self.count_key = count_key
    #     self.link_key = link_key

    def __init__(self,
                 links: List[Link] = None):

        if links is None:
            links = []

        self.links = links
        # self.links_keys = None
        # self.dataframe = pd.DataFrame({'key': self.links_keys})
        # self._counts = None

    @property
    def Y_dict(self):
        pass

    # def set_Z_attributes_dict_network(self, links_dict: {}):
    #     '''Add the link attributes to a general dictionary indexed by the attribute names and with key the values of that attribute for every link in the network
    #
    #     :argument links: dictionary of links objects. Each link contains the attributes values so it requires that a method such that set_random_link_attributes_network is executed before
    #     '''
    #
    #     # Index the network dictionary with the attributes names of any link
    #     Z_labels = list(links_dict.values())[0].Z_dict.keys()
    #     self.Z_dict = {}
    #
    #     for attr in Z_labels:
    #         self.Z_dict[attr] = {}
    #         for i, link in links_dict.items():
    #             self.Z_dict[attr][i] = link.Z_dict[attr]

    @property
    def features_Y(self):
        return list(self.links[0].Y_dict.keys())

    @property
    def features_Z(self):
        return list(self.links[0].Z_dict.keys())

    @property
    def Z_dict(self):

        features = self.features_Z

        Z_dict = {feature: [] for feature in features}

        for link in self.links:
            for feature in features:
                Z_dict[feature].append(link.Z_dict[feature])

        return Z_dict

    @property
    def Y_dict(self):

        features = self.features_Y

        Y_dict = {feature: [] for feature in features}

        for link in self.links:
            for feature in features:
                Y_dict[feature].append(link.Y_dict[feature])

        return Y_dict

    @property
    def Z_data(self) -> DataFrame:

        return pd.DataFrame(self.Z_dict)

    @property
    def Y_data(self) -> DataFrame:

        return pd.DataFrame(self.Y_dict)

    @property
    def links_keys(self):
        return [link.key for link in self.links]

    @property
    def counts(self) -> Dict:
        return {link.key: link.count for link in self.links}

    @property
    def counts_df(self) -> pd.DataFrame:

        counts_dict = self.counts

        return pd.DataFrame({'key': counts_dict.keys(), 'counts': counts_dict.values()})

    @property
    def x(self) -> Dict:
        return {link.key: link.x for link in self.links}

        # return dict(zip(self.dataframe[self.link_key].values
        #                 , self.dataframe[self.count_key].values))

    # @counts.setter
    # def counts(self, value) -> None:

    @property
    def counts_vector(self) -> ColumnVector:

        return np.array(list(self.counts.values()))[:, np.newaxis]

    @property
    def x_vector(self) -> ColumnVector:

        return np.array(list(self.x.values()))[:, np.newaxis]

    def feature_imputation(self, feature, pcts=(1, 99), lwrlk_only = True):

        '''

        Args:
            feature:
            pct: percentile
            lwrlk_only: if True, ignore links of types PQULK, DMOLK,DMDLK

        Returns:

        '''

        assert feature in self.features_Z

        if lwrlk_only:
            links_data = self.Z_data[['link_type', feature]]
            feature_data = np.array(links_data[links_data.link_type == 'LWRLK'][feature])
        else:
            feature_data = self.Z_data[feature]

        pct_low = np.percentile(feature_data, pcts[0])
        pct_high = np.percentile(feature_data, pcts[1])

        # find the indexes of the elements out of percentiles
        idx_under_low = np.argwhere(feature_data < pct_low).ravel()
        idx_under_high = np.argwhere(feature_data <= pct_high).ravel()

        # find the number of the elements in between percentiles
        diff_num = len(idx_under_high) - len(idx_under_low)

        # find the sum difference
        diff_sum = np.sum(np.take(feature_data, idx_under_high)) - np.sum(np.take(feature_data, idx_under_low))

        # get the mean
        mean = diff_sum / diff_num

        links_keys = []

        for link in self.links:
            feature_val = link.Z_dict[feature]
            if feature_val < pct_low or feature_val > pct_high:

                if lwrlk_only:
                    if link.link_type == 'LWRLK':
                        links_keys.append(link.key)
                        link.Z_dict[feature] = mean
                else:
                    links_keys.append(link.key)
                    link.Z_dict[feature] = mean

        print('Data for feature ' + feature + ' was imputed with value ' + str(round(mean,4)) +
              ' among ' + str(len(links_keys)) + ' links')

    def isValidLinkId(self, value):

        if len(value) <= 1 or len(value) > 3:
            print("input must be a tuple with two or three elements ")
            return False

        if not all(map(lambda x: isinstance(int(x), int), value)):
            print("all values must be integers")
            return False

        return True

    # @property
    # def counts(self) -> Dict:
    #
    #     return dict(zip(self.dataframe[self.link_key].values
    #                     , self.dataframe[self.count_key].values))
    #
    #     # return self.dataframe[[self.link_key, self.count_key]]
    #
    # @property
    # def counts_vector(self) -> ColumnVector:
    #
    #     return np.array(list(self.dataframe[self.count_key].values))[:, np.newaxis]

    # def dataframe(self):
    #     ''' Return dataframe'''
    #


def masked_observed_counts(counts: ColumnVector,
                           idx: [], complement=False):
    """

    :param xct_hat:
    :param counts:
    :param idx:
    :param complement: if complement is True, then the complement set of idx for counts is set to nan

    :return:
        count vector with all entries in idx equal to nan.

    """

    xct_list = list(counts.flatten())

    if complement is False:
        for id in idx:
            xct_list[id] = np.nan

    else:
        complement_idx = list(set(list(np.arange(len(counts)))) - set(idx))

        for id in complement_idx:
            xct_list[id] = np.nan

    return np.array(xct_list)[:, np.newaxis]


def fake_observed_counts(predicted_counts: ColumnVector,
                         observed_counts: ColumnVector) -> ColumnVector:
    """

    :param predicted_counts: predicted counts
    :param observed_counts: observed counts

    :return:
        count vector with all entries corresponding to the complement of the idx set to be equal to the predicted counts entries. This way, the difference between counts and xct_hat will be zero except for the idx entries.

    """

    # Replace values in positions with nas using the predicted count vector values

    fake_xct = copy.deepcopy(observed_counts)
    xct_hat_copy = copy.deepcopy(predicted_counts.flatten())

    for link_id in range(observed_counts.size):

        if np.isnan(fake_xct[link_id]):
            fake_xct[link_id] = xct_hat_copy[link_id]

    return fake_xct


def masked_link_counts_after_path_coverage(Nt, xct: dict, print_nan: bool = False) -> dict:
    """

    Compute t

    :return:
    """

    x_bar = np.array(list(xct.values()))[:, np.newaxis]

    idx_no_pathcoverage = np.where(np.sum(Nt.D, axis=1) == 0)[0]

    xct_remasked = masked_observed_counts(counts=x_bar, idx=idx_no_pathcoverage)

    # print('dif in coverage', np.count_nonzero(~np.isnan(x_bar))-np.count_nonzero(~np.isnan( x_bar_remasked )))

    idx_nonas = np.where(~np.isnan(xct_remasked))[0]

    if print_nan:

        # print(dict(zip(list(counts.keys()), np.sum(network.D,axis = 1))))

        # print(np.sum(network.D, axis=1)[:,np.newaxis])

        pass

    else:

        no_nanas_keys = [list(xct.keys())[i] for i in idx_nonas.tolist()]

        # print(dict(zip(no_nanas_keys, np.sum(network.D,axis = 1)[idx_nonas.tolist()])))

        # print(np.sum(network.D, axis=1)[idx_nonas.tolist()][:,np.newaxis])

    rows_sums_D_nonans = np.sum(Nt.D, axis=1)[idx_nonas.tolist()]

    print('\nAverage number of paths traversing links with counts:',
          np.round(np.sum(rows_sums_D_nonans) / len(no_nanas_keys), 1))

    print('\nMinimum number of paths traversing links with counts:',
          np.min(rows_sums_D_nonans))

    return dict(zip(xct.keys(), xct_remasked.flatten()))


def generate_training_validation_samples(xct: dict, prop_validation, prop_training=None) -> (dict, dict):
    if prop_training is None:
        prop_training = 1 - prop_validation

    x_bar = np.array(list(xct.values()))[:, np.newaxis]

    # Generate a random subset of idxs depending on coverage
    idx_nonas = np.where(~np.isnan(x_bar))[0]

    idx_training = list(np.random.choice(idx_nonas, int(np.floor(prop_training * len(idx_nonas))), replace=False))
    idx_validation = list(set(idx_nonas) - set(idx_training))

    # Only a subset of observations is assumed to be known
    train_link_counts = masked_observed_counts(counts=x_bar, idx=idx_training, complement=True)
    validation_link_counts = masked_observed_counts(counts=x_bar, idx=idx_validation, complement=True)

    # Convert to dictionary

    xct_training = dict(zip(xct.keys(), train_link_counts.flatten()))
    xct_validation = dict(zip(xct.keys(), validation_link_counts.flatten()))

    # Sizes deducting nan entries
    adjusted_size_training = np.count_nonzero(~np.isnan(train_link_counts))
    adjusted_size_validation = np.count_nonzero(~np.isnan(validation_link_counts))

    print('\nTraining and validation samples of sizes:', (adjusted_size_training, adjusted_size_validation))

    return xct_training, xct_validation

def get_informative_links_fresno(learning_results, network):
    d_errors = np.array(learning_results[2]['Fresno_report']['d_error']).astype(np.float)[:, np.newaxis]

    # learning_results[3] = {'Fresno_report':learning_results[2]['Fresno_report'] }

    for iter, results in learning_results.items():

        if iter > 2 and 'Fresno_report' in results.keys():
            report = learning_results[iter]['Fresno_report']
            d_error = np.array(report['d_error']).astype(np.float)[:, np.newaxis]
            d_errors = np.append(d_errors, d_error, axis=1)

    counts_copy = copy.deepcopy(network.link_data.counts)

    nas_link_keys = report['link_key'][np.where(np.max(d_errors, axis=1) <= 1e-10)[0]]

    for key in nas_link_keys:
        counts_copy[(key[0], key[1], '0')] = np.nan

    # print(str(initial_no_nas-final_no_nas) + ' links were removed')

    print("\n" + str(len(nas_link_keys)),
          " traffic counts observations were removed due to no variation in predicted counts over iterations")

    return counts_copy, nas_link_keys


def isWorkingNetwork(network_name) -> bool:

    # # Network factory
    # sim_options['n_custom_networks'] = 4
    # sim_options['n_random_networks'] = 4
    #
    # Custom networks
    custom_networks = ['N1','N2','N3','N4', 'Yang', 'Yang2', 'LoChan', 'Wang']

    # List of TNTP networks

    # These networks all work and have less than 1000 links (< 5 minutes)
    subfolders_tntp_networks_0 = ['Braess-Example', 'SiouxFalls', 'Eastern-Massachusetts'
        , 'Berlin-Friedrichshain', 'Berlin-Mitte-Center', 'Berlin-Tiergarten', 'Berlin-Prenzlauerberg-Center']

    # Medium size networks that work (< 4000 links) (most take about 15-20 minutes) (record: 3264 links with Terrassa asymmetric)
    subfolders_tntp_networks_1 = ['Barcelona', 'Chicago-Sketch', 'Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center',
                                  'Terrassa-Asymmetric', 'Winnipeg', 'Winnipeg-Asymmetric']

    tntp_networks = subfolders_tntp_networks_0 + subfolders_tntp_networks_1

    # Barcelona requires a lot of time (90 minutes) in the logit learning part because there are many od pairs and thus, many paths.

    # # Medium large size networks (5000-10000 links) that have to be tested (they take about 30  minutes)
    # subfolders_tntp_networks = ['Hessen-Asymmetric']

    # Next challenge (or just wait)
    # subfolders_tntp_networks = ['Austin'] # Austin 20000 links, Philadelphia: 40000 links

    # # Too large networks (> 20000 links)
    # subfolders_tntp_networks = ['Berlin-Center'
    #     , 'Birmingham-England', 'chicago-regional', 'Philadelphia','Sydney']
    #
    # # Error in reading data
    # subfolders_tntp_networks = ['Anaheim', 'GoldCoast', 'SymmetricaTestCase]

    # SymmetricaTestCase: Problem because CSV files
    # Anaheim: columns in data indexes
    # Goldcoast: filename

    # Real world networks
    real_world_networks = ['Fresno', 'Colombus', 'Sacramento']

    # Working networks
    working_networks = custom_networks + tntp_networks + real_world_networks + ['default']

    assert network_name in working_networks, ' unexisting network'

