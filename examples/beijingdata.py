import pandas as pd
import numpy as np
import os

# from .import ..transportAI as tt
import transportAI.reader as reader
# from transportAI import .master
#import pylogit as pl
import transportAI as tt

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

subdirectory = "/data"
subfolder = "/beijing-subway/"
# folder = os.getcwd() + "/" + subdirectory + subfolder
# folder =

#folder_path = os.getcwd() + subdirectory + subfolder

folder_path = '/Users/pablo/GoogleDrive/Github/transportAI/examples/data/beijing-subway/'

raw_df = tt.reader.read_beijing_data(folder_path=folder_path)
#raw_df = reader.read_beijing_data()
df = raw_df.iloc[:1000]
# b = df.head()
# print(df)
print(df.head())

# =============================================================================
# Create OD Matrix
# =============================================================================

od = tt.create_od(trips = df, from_ = 'system')

print(od)

n_origins = df['ostation'].unique()
n_destinations = df['dstation'].unique()

len(n_origins), len(n_destinations)

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



