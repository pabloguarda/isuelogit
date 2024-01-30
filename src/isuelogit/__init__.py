#Compatibility with imports from Python 2.x
from __future__ import absolute_import

# TODO: import everything from init in order to debug properly

"""Top-level package for isuelogit."""

__author__ = """Pablo Guarda"""
__email__ = 'pabloguarda@cmu.edu'
__version__ = '0.1.0'

import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# print(project_root)

# Modules available for user
import isuelogit.factory
import isuelogit.equilibrium
import isuelogit.networks
import isuelogit.estimation
import isuelogit.visualization
import isuelogit.reader
import isuelogit.writer
import isuelogit.printer
import isuelogit.descriptive_statistics
import isuelogit.experiments
import isuelogit.paths
import isuelogit.utils
import isuelogit.geographer
import isuelogit.etl

import isuelogit.config
from isuelogit.config import *

# modules not available for user
# import mytypes
# import links

