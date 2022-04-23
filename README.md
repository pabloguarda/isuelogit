# isuelogit

Stochastic user equilibrium with logit assignment (`sue-logit`) outputs an assignment of travelers in a transportation network where no individual believes that he can unilaterally improve his utility by choosing an alternative path. A key limitation of `sue-logit` is that it requires to know the parameters of the traveler's utility function beforehand. 

The `isuelogit` package addresses the aforementioned limitation by using the output of `sue-logit` to solve an inverse problem, namely, estimating the parameters of the travelers' utility function using traffic counts.

## Development setup

1. Clone the repository
2. Create virtual environment: `python3 -m venv venv-isuelogit`
3. Activate virtual environment: `source venv-isuelogit/bin/activate`
4. Install the package: `pip install -e .`
5. Install the development dependencies: `pip install -r requirements/dev.txt`
6. Run the tests: `pytest`

## Jupyter notebooks
1. Install jupyter lab: `pip install jupyter notebook`
2. Install ipython kernel for jupyter: `pip install ipykernel`
3. Add your virtual environment to Jupyter:  `python3 -m ipykernel install --user --name=venv-isuelogit`
4. Open any of the notebooks include in folder `notebooks` and choose kernel `venv-isuelogit` to run it

## Collaboration
For any questions or interest of collaborating in this project, please contact pabloguarda@cmu.edu.