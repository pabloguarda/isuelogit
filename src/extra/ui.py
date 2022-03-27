""" User interface of commands"""

### External packages
import pandas as pd

# from .arquitect import Arquitect
# from .infrastructure import Infrastructure

#Compatibility with imports from Python 2.x
#from __future__ import absolute_import

# import .architect
# from transportAI.arquitect import Arquitect

# from . import modeller as md
from transportAI.factory import Modeller
from extra.agents import Traveller

#from . import architect
from extra.infrastructure import Infrastructure
from extra.system import System

def create_infrastructure(ids: list, type = None, positions = None):
    # architect = Arquitect()
    infrastructure_list = [Infrastructure(id_ = id_, positions = positions) for id_ in ids]
    #print(__name__)
    return infrastructure_list
    # raise NotImplementedError


def create_network(infrastructure):

    """:argument infrastructure is a list of building units (e.g. metro or bus station)"""

    network = None #TNetwork(,
    return network

def create_agents(beijing_df, type = 'travellers'):
    """ Method only works when receiving Beijing subway data as an input"""

    df = beijing_df
    n,m = df.shape

    travellers = []

    if type == 'travellers':

        os = df['ostation']
        ds = df['dstation']
        ids = df['nid']

        for o,d,id in zip(os, ds, ids):
            travellers.append(Traveller(origin = o, destination = d, id_ = id))

        return travellers

    pass
    # return agents


def create_system(network, agents, vehicles = None,):

    #controller = Controller()
    system = System(network = network, vehicles = vehicles, agents = agents)
    return system

# def system_equilibrium(eq_type = 'static'):
#
#     return equilibrium.edges


def create_od(trips, from_):

    """:argument from_: agents or system level data"""
    """:argument trips: a list of trips, or dataframe with all trips"""

    modeller = Modeller()

    od = pd.DataFrame()

    if from_ == 'agents':
        od = modeller.create_od_from_agents(trips = trips)

    if from_ == 'system':
        od = modeller.create_od_from_system(trips = trips)

    return od


