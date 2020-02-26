"""
Modeller can estimate models from observed data.
"""

import pandas as pd
import numpy as np

class Modeller:

    def __init__(self, name = None, date = None):
        self._name = name
        self._date = date

    def create_od_from_system(self, trips):
        od = pd.pivot_table(trips[['ostation', 'dstation']], index=['ostation'], columns=['dstation'], aggfunc=[len])
        od = od.replace(np.nan, 0)

        return od

    def create_od_from_agents(self, trips):

        df = pd.DataFrame(columns = ['o','d'])

        for i in range(len(trips)):
            trip = trips[i]
            df = df.append({'o':trip.origin, 'd':trip.destination}, ignore_index = True)

        od = pd.pivot_table(df[['o', 'd']], index=['o'], columns=['d'], aggfunc=[len])
        od = od.replace(np.nan, 0)

        np.sum(od.to_numpy())  # Checking the total number of trips is consistent with the original dataset

        return od
    # def build_od(self, trips):
    #     """:argument trips: list of trips"""


