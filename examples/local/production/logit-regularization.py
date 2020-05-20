# =============================================================================
# Resources
# =============================================================================

# https://freakonometrics.hypotheses.org/53470
# https://towardsdatascience.com/feature-selection-using-regularisation-a3678b71e499
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html#sphx-glr-auto-examples-linear-model-plot-logistic-l1-l2-sparsity-py
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_20newsgroups.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-20newsgroups-py

# Cvxpy implementation
# https://www.cvxpy.org/examples/machine_learning/logistic_regression.html

import numpy as np
import pandas as pd
# =============================================================================
# Simulate data for logistic regression
# =============================================================================

n = 2000

# Feature simulation
x1 = np.random.normal(0, 2, size = n)
x2 = np.random.normal(1,3, size = n)
x3 = np.random.normal(0,0.2, size = n)

theta1 = -2
theta2 = 3
theta3 = 1e-10

z = 1 + theta1*x1 + theta2*x2 + theta3*x3

pr = 1/(1+np.exp(-z))

# Bernoulli Variable
c = 2
y = np.random.binomial(c-1, pr, size = n)


# =============================================================================
# Logistic regression with stats models (R)
# =============================================================================

import statsmodels.api as sm

df = pd.DataFrame({'x1':x1})
df['x2'] = x2
df['x3'] = x3
df['c'] = y

x = df[['x1','x2','x3']]
x = sm.add_constant(x)
y = df['c']
mlogit_mod = sm.MNLogit(y,x)

logit_res = mlogit_mod.fit()

print(logit_res.summary())

# Regularization
# https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit_regularized.html
# https://stackoverflow.com/questions/43375597/logistic-regression-in-statsmodels-fitting-and-regularizing-slowly

logit_reg_res = mlogit_mod.fit_regularized(method="l1", alpha=200, disp=True)
logit_noreg_res = mlogit_mod.fit_regularized(method="l1", alpha=0, disp=True)

print(logit_reg_res.summary())
print(logit_noreg_res.summary())
print(logit_res.summary())

x_reg = df[['x1','x2']]
mlogit_mod_reg = sm.MNLogit(y,x_reg)
print(mlogit_mod_reg.fit().summary())

# =============================================================================
# Logistic regression with Scikit Learn
# =============================================================================

# https://scikit-learn.org/stable/modules/linear_model.html#lasso
# https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

from sklearn.linear_model import LogisticRegression

X = np.array(df[['x1','x2','x3']].to_numpy())
y = df['c'].to_numpy()

clf = LogisticRegression(random_state=0).fit(X, y)

clf.predict(X)

# from sklearn.datasets import load_iris
# X, y = load_iris(return_X_y=True)

# Regularization
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#sphx-glr-auto-examples-linear-model-plot-logistic-path-py
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html#sphx-glr-auto-examples-linear-model-plot-logistic-l1-l2-sparsity-py
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_20newsgroups.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-20newsgroups-py

# Data should be centered before doing regularization


# =============================================================================
# Logistic regression with Keras
# =============================================================================
# import keras as keras



# =============================================================================
# Logistic regression with Pytorch
# =============================================================================
# import torch as torch

# =============================================================================
# Logistic regression with regularization using glmnet_python
# =============================================================================

#Tibshirani package
# https://web.stanford.edu/~hastie/glmnet_python/index.html
# https://github.com/bbalasub1/glmnet_python/blob/master/test/glmnet_examples.ipynb

# Examples and explanation for the R package
# https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html

# Examples in python
# https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html

# Remember to install gfortran from
# https://github.com/fxcoudert/gfortran-for-macOS/releases

# In addition, I need to copy the file libgfortran.3.dylib in the Fortran Folder from the Anaconda one

# Create a tar file in local packages as this package is full of bugs
# import glmnet_python as glmnet
from glmnet_python.glmnet import glmnet

# from glmnet_python import *

from glmnet_python.glmnetPlot import glmnetPlot
from glmnet_python.glmnetPrint import glmnetPrint
from glmnet_python.glmnetPredict import glmnetPredict
from glmnet_python.cvglmnet import cvglmnet
from glmnet_python.cvglmnetPlot import cvglmnetPlot
from glmnet_python.cvglmnetPredict import cvglmnetPredict

# External Packages
import requests
import io
import pandas as pd
import numpy as np
import warnings

# Load data for logistic
url_x = 'https://raw.githubusercontent.com/hanfang/glmnet_py/master/data/BinomialExampleX.dat'
web_x = requests.get(url=url_x).content
x_l = pd.read_csv(io.StringIO(web_x.decode('utf-8')), header=None).to_numpy(dtype = np.float64)

url_y = 'https://raw.githubusercontent.com/hanfang/glmnet_py/master/data/BinomialExampleY.dat'
web_y = requests.get(url=url_y).content
y_l = pd.read_csv(io.StringIO(web_y.decode('utf-8')), header=None).to_numpy(dtype = np.float64)


# create weights
t = np.ones((50, 1), dtype = np.float64)
wts = np.row_stack((t, 2*t))

# call glmnet
fit_l = glmnet(x = x_l, y = y_l, family = 'binomial')

glmnetPrint(fit_l)

glmnetPlot(fit_l, xvar = 'lambda', label = True)

glmnetPlot(fit_l, xvar = 'dev', label = True)

# Predictions
glmnetPredict(fit_l, newx = x[0:5,], ptype='class', s = np.array([0.05, 0.01]))

# Misclassification error for 10 fold cross validation

# warnings.filterwarnings('ignore')
cvfit_l = cvglmnet(x = x_l, y = y_l, family = 'binomial', ptype = 'class')
# warnings.filterwarnings('default')
cvglmnetPlot(cvfit_l)
cvfit_l['lambda_min']

# Logistic Regression - Multinomial Models
url_x = 'https://raw.githubusercontent.com/hanfang/glmnet_py/master/data/MultinomialExampleX.dat'
web_x = requests.get(url=url_x).content
x = pd.read_csv(io.StringIO(web_x.decode('utf-8')), header=None).to_numpy(dtype = np.float64)

url_y = 'https://raw.githubusercontent.com/hanfang/glmnet_py/master/data/MultinomialExampleY.dat'
web_y = requests.get(url=url_y).content
y = pd.read_csv(io.StringIO(web_y.decode('utf-8')), header=None).to_numpy(dtype = np.float64)

fit_mnl = glmnet(x = x.copy(), y = y.copy(), family = 'multinomial', mtype = 'grouped')

glmnetPlot(fit_mnl, xvar = 'lambda', label = True, ptype = '2norm')


# warnings.filterwarnings('ignore')
cvfit_mnl=cvglmnet(x = x.copy(), y = y.copy(), family='multinomial', mtype = 'grouped');
# warnings.filterwarnings('default')
cvglmnetPlot(cvfit_mnl)

#Prediction
cvglmnetPredict(cvfit_mnl, newx = x[0:10, :], s = 'lambda_min', ptype = 'class')

# =============================================================================
# Logistic regression with regularization using cvxpy
# =============================================================================

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# In the following code we generate data with 𝑛=50 features by randomly choosing 𝑥𝑖 and supplying a sparse 𝛽true∈𝐑𝑛.
# We then set 𝑦𝑖=𝟙[𝛽𝑇true𝑥𝑖+𝑧𝑖>0], where the 𝑧𝑖 are i.i.d. normal random variables.
# We divide the data into training and test sets with 𝑚=50 examples each.

np.random.seed(10)
n = 50
m = 50
def sigmoid(z):
  return 1/(1 + np.exp(-z))

beta_true = np.array([1, 0.5, -0.5] + [0]*(n - 3))
X = (np.random.random((m, n)) - 0.5)*10
Y = np.round(sigmoid(X @ beta_true + np.random.randn(m)*0.5))

X_test = (np.random.random((2*m, n)) - 0.5)*10
Y_test = np.round(sigmoid(X_test @ beta_true + np.random.randn(2*m)*0.5))

# We next formulate the optimization problem using CVXPY.

beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)
log_likelihood = cp.sum(
    cp.multiply(Y, X @ beta) - cp.logistic(X @ beta)
)
problem = cp.Problem(cp.Maximize(log_likelihood/n - lambd * cp.norm(beta, 1)))

# We solve the optimization problem for a range of 𝜆 to compute a trade-off curve.
# We then plot the train and test error over the trade-off curve.
# A reasonable choice of 𝜆 is the value that minimizes the test error.

def error(scores, labels):
  scores[scores > 0] = 1
  scores[scores <= 0] = 0
  return np.sum(np.abs(scores - labels)) / float(np.size(labels))

trials = 100
train_error = np.zeros(trials)
test_error = np.zeros(trials)
lambda_vals = np.logspace(-2, 0, trials)
beta_vals = []
for i in range(trials):
    lambd.value = lambda_vals[i]
    problem.solve()
    train_error[i] = error( (X @ beta).value, Y)
    test_error[i] = error( (X_test @ beta).value, Y_test)
    beta_vals.append(beta.value)

plt.plot(lambda_vals, train_error, label="Train error")
plt.plot(lambda_vals, test_error, label="Test error")
plt.xscale("log")
plt.legend(loc="upper left")
plt.xlabel(r"$\lambda$", fontsize=16)
plt.show()

# We also plot the regularization path, or the 𝛽𝑖 versus 𝜆.
# Notice that a few features remain non-zero longer for larger 𝜆 than the rest,
# which suggests that these features are the most important.


for i in range(n):
    plt.plot(lambda_vals, [wi for wi in beta_vals])
plt.xlabel(r"$\lambda$", fontsize=16)
plt.xscale("log")
plt.show()

# We plot the true 𝛽 versus reconstructed 𝛽, as chosen to minimize error on the test set.
# The non-zero coefficients are reconstructed with good accuracy.
# There are a few values in the reconstructed 𝛽 that are non-zero but should be zero.

idx = np.argmin(test_error)
plt.plot(beta_true, label=r"True $\beta$")
plt.plot(beta_vals[idx], label=r"Reconstructed $\beta$")
plt.xlabel(r"$i$", fontsize=16)
plt.ylabel(r"$\beta_i$", fontsize=16)
plt.legend(loc="upper right")
plt.show()


# =============================================================================
# Binomial Logit with regularization using cvxpy (all alternatives available)
# =============================================================================

# Replicate example for logistic regression in https://www.cvxpy.org/examples/machine_learning/logistic_regression.html
# but for the binomial logit model.

# Assumption: Aggregated choices are known at the route level

# Simulate binary logit data (look at my code in R and translate it to Python)

def binaryLogit(x1, x2, theta):
    """ Binary logit model """

    sum_exp = np.exp(theta * x1) + np.exp(theta * x2)
    p1 = np.exp(theta * x1) / sum_exp
    p2 = 1 - p1

    return np.array([p1, p2])  # (P1,P2)






# =============================================================================
# Multinomial Logit with regularization using cvxpy
# =============================================================================


# =============================================================================
# SUE Logit with regularization using cvxpy with route level data
# =============================================================================

# Assumption: Aggregated choices are known at the route level.


# =============================================================================
# SUE Logit with regularization using cvxpy with link level data
# =============================================================================

# Assumption: Aggregated choices are known at the link level.
# Then, assumption is made to generate reasonable paths and being able to write likelihood at link level