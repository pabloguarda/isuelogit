#Compatibility with imports from Python 2.x
from __future__ import absolute_import

# TODO: import everything from init in order to debug properly

"""Top-level package for transportAI."""

__author__ = """Pablo Guarda"""
__email__ = 'pabloguarda@cmu.edu'
__version__ = '0.1.0'

# import sys
# sys.path.append("..")

# #Make relative imports work
# import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import sys

# # #Working directory
import os


# print('main directory: ')
# print(os.getcwd())

# Set working directory
# os.chdir('/Users/pablo/google-drive/data-science/github/transportAI')

# os.chdir(os.getcwd()+"/TransportAI")

# sys.path.append(os.getcwd()+'/src/transportAI')
#
# sys.path.append('/Users/pablo/google-drive/data-science/Github/transportAI/src/transportAI')

project_root = os.path.abspath(os.path.dirname(__file__))
# output_path = os.path.join(project_root, 'transportAI')
sys.path.append(project_root)
# print(project_root)

# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.append('/path/to/application/app/folder')

# a single point (.) refer  to the current folder directory
# two points (..) refer to the directory where the current folder is located
#
# #Modeller function
# import transportAI.infrastructure
# from transportAI.infrastructure import *
# # from transportAI.ui import create_infrastructure
# # from transportAI.ui import create_network
# # from transportAI.ui import create_agents
# # from transportAI.ui import create_od
#
# #Network module
# # from transportAI.networks import *
# # from transportAI import network
#
# #Numerical stability module
# import transportAI.numericalstability
# from transportAI.numericalstability import *
#
# #Equilibrium module
# import transportAI.equilibrium
# from transportAI.equilibrium import *
#
# #GIS module
# import transportAI.gis
# from transportAI.gis import *
#
# # #Network module
# # from transportAI import network
#
# # Configuration
# import transportAI.config
# from transportAI.config import *
#
# # Types
#
# import transportAI.mytypes
# from transportAI.mytypes import *
#
# #Logit module
# import transportAI.estimation
# from transportAI.estimation import *
#
# #Links module
# import transportAI.links
# from transportAI.links import *
#
# #Simulation module
# import simulation
# from transportAI.simulation import *
#
# #Visualization
# import visualization
# from transportAI.visualization import *
#
# #Reader module
# import reader
# from transportAI.reader import *
#
# #Writer module
# import writer
# from transportAI.writer import *
#
# #Modeller module
# import modeller
# from transportAI.modeller import *
#
# #Beijing module
# import beijing
# from transportAI.beijing import *


# from .beijing import *

# from . import test1

# #Modeller function
# from .master import *

# Execution of test scripts
# exec(open(os.getcwd() + '/examples/local/production/odthetaexample.py').read())

# import runpy
# runpy.run_path(os.getcwd() + '/examples/local/production/od-theta-example.py')
# # runpy.run_module(os.getcwd() + '/examples/local/production/od-theta-example.py')
# runpy.run_path(os.getcwd() + '/examples/local/production/od-theta-example.py')

# sys.path.insert(0, '/Users/pablo/google-drive/data-science/github/transportAI/examples/local/')


# sys.path.append(os.path.abspath(os.path.join('..', 'examples/local/production')))



# from transportAI.examples.local import examples.local.odthetaexamples

# import odthetaexample


# import odthetaexample.py

# os.chdir('/Users/pablo/google-drive/data-science/github/transportAI')



######### Without transport AI



#Modeller function
# import infrastructure
from . import infrastructure
# from ui import create_infrastructure
# from ui import create_network
# from ui import create_agents
# from ui import create_od

#Network module
# from networks import *
# from import network

#Numerical stability module
# import numeric
import numeric

#Equilibrium module
import equilibrium
from equilibrium import *

#GIS module
import geographer
from geographer import *

# #Network module
# from import network

# Configuration
import config
from config import *

# Types

import mytypes
from mytypes import *

#Logit module
import estimation
from estimation import *

#Links module
import links
from links import *

#Simulation module
import simulation
from simulation import *

#Visualization
import visualization
from visualization import *

#Reader module
import reader
from reader import *

#Writer module
import writer
from writer import *

#Modeller module
import modeller
from modeller import *

#Beijing module
import beijing
from beijing import *

# Analyst
import analyst
from analyst import *

# Descriptive statistics
import descriptive_statistics
from descriptive_statistics import *

