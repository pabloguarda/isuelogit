# isuelogit

Stochastic user equilibrium with logit assignment (`sue-logit`) gives an assignment of travelers in a transportation network where no individual believes that he can unilaterally improve his utility by choosing an alternative paths. 

A key limitation of sue-logit is that it requires to know the parameters of the traveler's utility function beforehand. The `isuelogit`package addresses this limitation by solving the inverse problem, namely, estimating the parameters of the travelers' utility function using traffic counts, which an output of `sue-logit`.  

[![Build Status](https://travis-ci.com/github/pabloguarda/transportAI.svg?branch=master)](https://travis-ci.com/github/pabloguarda/transportAI)

## Development setup

1. Clone the repository
2. Create virtual environment: `python3 -m venv venv`
3. Activative virtual environment: `source venv/bin/activate`
4. Install the package: `pip install -e .`
5. Install the development dependencies: `pip install -r requirements/prod.txt`
6. Run the tests: `pytest`

