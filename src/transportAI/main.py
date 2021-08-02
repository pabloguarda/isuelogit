"""Main module."""

# # Set working directory
# os.chdir('/Users/pablo/google-drive/data-science/github/transportAI')

#Make relative imports work
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# architect = Arquitect()
# node = Node(label = 1, pos = (0,0))
# edge = Edge()

# from src import transportAI as tt

# from transportAI.ai import Alternative

# import transportAI as tt

# n_origins = 58
# ids = range(1, n_origins + 1)
#
# infrastructure = tt.create_infrastructure(ids = ids, positions = None)
# network = tt.create_network(infrastructure = infrastructure)
# travellers = tt.create_agents()
#
# system = tt.create_system(network = network, vehicles = vehicles, travellers = travellers)


#tt.master.build_infrastructure(nodes = len(n_origins) )

# print('b')

import os

# import runpy
# runpy.run_module('transportAI', run_name="__main__",alter_sys=True)
# runpy.run_module(os.getcwd() + '/examples/local/production/od-theta-example.py', run_name="__main__",alter_sys=True)


# import subprocess
#
# subprocess.call(os.getcwd() + '/examples/local/production/od-theta-example.py')

# Debug calling python script

# https://stackoverflow.com/questions/3657955/how-to-execute-another-python-script-from-your-script-and-be-able-to-debug

# import transportAI
import runpy

# from examples.local.production import odthetaexample

# exec(open(os.getcwd() + '/examples/local/production/odthetaexample.py').read())

# import transportAI

# print(sys.path)

sys.path.append('/Users/pablo/google-drive/data-science/github/transportAI/examples/local/production')

# This is the way to make possible to debug
import odthetaexample

# runpy.run_path(os.getcwd() + '/examples/local/production/odthetaexample.py', run_name="__main__")
# sys.argv = saved_argv