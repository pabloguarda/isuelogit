# isuelogit

Stochastic user equilibrium with logit assignment (`sue-logit`) gives an assignment of travelers in a transportation network where no individual believes that he can unilaterally improve his utility by choosing an alternative paths. 

A key limitation of sue-logit is that it requires to know the parameters of the traveler's utility function beforehand. The `isuelogit`package addresses this limitation by solving the inverse problem, namely, estimating the parameters of the travelers' utility function using traffic counts, which an output of `sue-logit`.  

[![Build Status](https://travis-ci.com/github/pabloguarda/isuelogit.svg?branch=master)](https://travis-ci.com/github/pabloguarda/isuelogit)

## Development setup

1. Clone the repository
2. Create virtual environment: `python3 -m venv myvenv`
3. Activate virtual environment: `source myvenv/bin/activate`
4. Install the package: `pip install -e .`
5. Install the development dependencies: `pip install -r requirements/prod.txt`
6. Run the tests: `pytest`

## Jupyter notebooks
1. Install jupyter lab: `pip install jupyterlab`
2. Install ipython kernel for jupyter: `pip install ipykernel`
3. Add your virtual environment to Jupyter:  `python3 -m ipykernel install --name=myvenv`
4. Open any of the notebooks include in folder `notebooks` and choose kernel `myvenv` to run it
