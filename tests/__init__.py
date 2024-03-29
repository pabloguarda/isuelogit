"""Unit test package for isuelogit."""
""" The aim is to maximize the code coverage by adding as many and meaningful tests as possible """

""" To run tests on code write the following in terminal: 
Run  python pytest -m pytest
https://docs.pytest.org/en/latest/usage.html
"""

""" To calculate code coverage, write the following in terminal:
coverage run -m pytest pytest
https://coverage.readthedocs.io/en/coverage-5.0.3/
"""

import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)