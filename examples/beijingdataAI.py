# =============================================================================
# Network modelling with transportAI
# =============================================================================


import pandas as pd
import numpy as np
import os

# from .import ..transportAI as tt
import transportAI.reader as reader
# from transportAI import .master

import transportAI as tt

#Garbage collection
import gc
gc.enable()
gc.collect()

# =============================================================================
# Data Reading
# =============================================================================

## Select a random sample of trips
# df = df_raw.sample(n = 10000)

subdirectory = "/data"
subfolder = "/beijing-subway/"
folder_path = os.getcwd() + subdirectory + subfolder
raw_df = tt.reader.read_beijing_data(folder_path=folder_path)

df = raw_df.iloc[:1000]
# b = df.head()
# print(df)
print(df.head())

# =============================================================================
# Create OD Matrix
# =============================================================================

travellers = tt.create_agents(beijing_df=df, type = 'travellers')

od = tt.create_od(trips = travellers, from_ = 'agents')

print(od)




