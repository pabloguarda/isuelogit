"""Main module."""

import os

# # Set working directory
os.chdir('/Users/pablo/google-drive/data-science/github/transportAI')

#Make relative imports work
import sys
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))

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

# sys.path.append('/Users/pablo/google-drive/data-science/github/transportAI/examples/local/production')

# This is the way to make possible to debug
# import odthetaexample

# runpy.run_path(os.getcwd() + '/examples/local/production/odthetaexample.py', run_name="__main__")
# sys.argv = saved_argv