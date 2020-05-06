#Compatibility with imports from Python 2.x
#from __future__ import absolute_import


"""Top-level package for transportAI."""

__author__ = """Pablo Guarda"""
__email__ = 'pabloguarda@cmu.edu'
__version__ = '0.1.0'


# import sys
# sys.path.append("..")

# #Make relative imports work
# import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# #Working directory
# import os
# os.chdir("/Users/pablo/GoogleDrive/Github/transportAI")

# os.chdir(os.getcwd()+"/TransportAI")

# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.append('/path/to/application/app/folder')

# a single point (.) refer  to the current folder directory
# two points (..) refer to the directory where the current folder is located

#Modeller function
from transportAI import infrastructure
# from transportAI.ui import create_infrastructure
# from transportAI.ui import create_network
# from transportAI.ui import create_agents
# from transportAI.ui import create_od

#Reader
from .reader import read_beijing_data

# from . import test1

# #Modeller function
# from .master import *


