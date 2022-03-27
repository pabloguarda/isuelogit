## Packages

import numpy as np
import cvxpy as cp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
import seaborn as sns

### Number of choices and attributes

N = 1000
K = 1

### Preference parameters

theta_true = {}
theta_true['tt'] = -1
theta_true['exp'] = 0.8
# theta_true['c'] = -0.1
# theta_true = np.array(list(theta_true.values()))[:,np.newaxis]

### X
X = {}
X['A1'] = np.abs((np.random.random((N, K)) - 0.5)*10)+1
X['A2'] = np.abs((np.random.random((N, K)) - 0.5)*10)+1

## Error terms

error = {}
error['A1'] = gumbel_r.rvs(size=N)[:,np.newaxis]
error['A2'] = gumbel_r.rvs(size=N)[: ,np.newaxis]


## Choices

Y1 = np.where(X['A1']**theta_true['exp'] + error['A1'] >= X['A2']**theta_true['exp'] + error['A2'],1,0)
Y2 = 1-Y1

### Decision variables (MLE estimators)
theta_hat = {}
theta_hat['tt'] = cp.Variable(K,pos=False)
theta_hat['exp'] = cp.Variable(K,pos=True)

# theta_hat = cp.Variable(K)
# theta_hat = cp.Variable(pos=True)



# ll_chosenV = cp.hstack([i*cp.exp(theta_hat + cp.log(cp.pos(j)+1)) for i,j in zip(Y1,X['A1'])]) + cp.hstack([i*cp.exp(theta_hat + cp.log(cp.pos(j)+1)) for i,j in zip(Y2,X['A2'])])
ll_chosenV = cp.hstack([-cp.exp(theta_hat['exp'] * cp.log(cp.pos(j+1))) for i,j in zip(Y1,X['A1'])]) #+ cp.hstack([i*cp.exp(theta_hat + cp.log(cp.pos(j)+1)) for i,j in zip(Y2,X['A2'])])
# ll_chosenV = cp.hstack([-i*cp.exp(theta_hat + cp.log(cp.pos(j)+1))/cp.exp(theta_hat + cp.log(cp.pos(j)+1)) for i,j in zip(Y1,X['A1'])]) #+ cp.hstack([i*cp.exp(theta_hat + cp.log(cp.pos(j)+1)) for i,j in zip(Y2,X['A2'])])
ll_logsum = cp.log_sum_exp(cp.vstack([cp.hstack([-0.5*cp.exp(theta_hat['exp']* cp.log(cp.pos(i+1))) for i in X['A1']]), cp.hstack([-0.5*cp.exp(theta_hat['exp']* cp.log(cp.pos(i+1))) for i in X['A2']])]), axis = 0)
print(ll_logsum.is_dcp())

cp.power(cp.exp(theta_hat['exp']),-2)

cp.exp(theta_hat)
cp.power(2,theta_hat)

[i for i,j in zip(Y1,X['A1'])]
[j for i,j in zip(Y1,X['A1'])]

print(ll_chosenV.is_dcp())

print(cp.sum(ll_chosenV - ll_logsum).is_dcp())

print((ll_chosenV).is_convex())
print((-ll_logsum).is_convex())

ll = cp.sum(ll_chosenV - ll_logsum)
print((ll).is_convex())

problem = cp.Problem(cp.Maximize(ll))
problem.solve(solver = 'ECOS')
print(theta_hat['exp'].value)


ll2 = cp.hstack([cp.exp(theta_hat) for i in Y1*X['A1']])+cp.hstack([cp.exp(theta_hat) for i in Y2*X['A2']])-cp.log_sum_exp(cp.hstack([cp.hstack([cp.exp(theta_hat) for i in Y2*X['A1']]), cp.hstack([cp.exp(theta_hat) for i in Y2*X['A2']])]), axis = 0)
print(cp.sum(ll2).is_dcp())

# log_likelihood = cp.sum(
#     Y1*X['A1']*theta_hat + Y2*X['A2']*theta_hat - cp.log_sum_exp(cp.vstack([X['A1'] @ theta_hat,X['A2'] @ theta_hat]), axis = 0)
# )

# log_likelihood = cp.sum(
#     theta_hat + theta_hat - cp.log_sum_exp(cp.vstack([X['A1'] @ theta_hat,X['A2'] @ theta_hat]), axis = 0)
# )

print(log_likelihood.is_dcp())

### Maximize likelihood

problem = cp.Problem(cp.Maximize(log_likelihood))
problem.solve(solver = 'ECOS')

# theta_hat.value

monomial = cp.exp(cp.log(theta_hat+2))

a = Y1*X['A1']
a.shape



len(Y1*X['A1'])

# =============================================================================
# BOX COX WITH BIOGEME
# =============================================================================

# Generate data

### X
X = {}
X['A1'] = np.abs((np.random.random((N, K)) - 0.5)) + 0.1
X['A2'] = np.abs((np.random.random((N, K)) - 0.5)) + 0.1

## Error terms
error = {}
error['A1'] = gumbel_r.rvs(size=N)[:, np.newaxis]
error['A2'] = gumbel_r.rvs(size=N)[:, np.newaxis]

## Choices

Y1 = np.where(
    theta_true['tt'] * X['A1'] ** theta_true['exp'] + error['A1'] >= theta_true['tt'] * X['A2'] ** theta_true[
        'exp'] + error['A2'], 1, 0)
Y2 = 1 - Y1



import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.messaging as msg
from biogeme.expressions import Beta, DefineVariable


def biogeme_time_perception(X, y):
    ## Panda dataframe
    df = pd.DataFrame({'T_1': X['A1'][:, 0], 'T_2': X['A2'][:, 0], 'CHOICE':y})
    database = db.Database('timeperception', df)

    # The following statement allows you to use the names of the variable
    # as Python variable.
    globals().update(database.variables)

    # Parameters to be estimated
    B_TIME = Beta('B_TIME', 0, None, None, 0)
    LAMBDA = Beta('LAMBDA', 0.5, 0.0001, 2, 0)

    # Definition of the utility functions
    V1 = B_TIME * models.boxcox(T_1, LAMBDA)
    V2 = B_TIME * models.boxcox(T_2, LAMBDA)

    # Associate utility functions with the numbering of alternatives
    V = {1: V1,
         2: V2, }

    # Definition of the model. This is the contribution of each
    # observation to the log likelihood function.
    logprob = models.loglogit(V=V, av=None, i=CHOICE)

    # Define level of verbosity
    logger = msg.bioMessage()
    logger.setSilent()
    # logger.setWarning()
    # logger.setGeneral()
    # logger.setDetailed()

    # Create the Biogeme object
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = '08boxcox'

    # Estimate the parameters
    results = biogeme.estimate()
    pandasResults = results.getEstimatedParameters()

    # print(pandasResults)

    return {'p': pandasResults['Value']['LAMBDA'], 'theta_tt': pandasResults['Value']['B_TIME']}


# =============================================================================
# CROSS VALIDATION
# =============================================================================

import numpy as np
import cvxpy as cp
from scipy.stats import gumbel_r
from sklearn.model_selection import train_test_split

def CV_timeperception(theta_true, lambdas, N, K):
    # theta_true['c'] = -0.1
    # theta_true = np.array(list(theta_true.values()))[:,np.newaxis]

    ### X
    X = {}
    X['A1'] = np.abs((np.random.random((N, K)) - 0.5)) + 1
    X['A2'] = np.abs((np.random.random((N, K)) - 0.5)) + 1

    ## Error terms
    error = {}
    error['A1'] = gumbel_r.rvs(size=N)[:, np.newaxis]
    error['A2'] = gumbel_r.rvs(size=N)[:, np.newaxis]

    ## Choices

    Y1 = np.where(
        theta_true['tt'] * X['A1'] ** theta_true['exp'] + error['A1'] >= theta_true['tt'] * X['A2'] ** theta_true[
            'exp'] + error['A2'], 1, 0)
    Y2 = 1 - Y1

    ### Decision variables (MLE estimators)
    theta_hat = {}
    # theta_hat['tt'] = cp.Parameter(pos=False)
    # theta_hat['tt'] = cp.Parameter(nonneg=False)
    theta_hat['tt'] = cp.Variable(K, pos=False)
    # theta_hat['exp'] = cp.Variable(K,pos=True)

    theta_hat['p'] = cp.Parameter(pos=True)
    # theta_hat['p'] = cp.Variable(K,pos=True)

    ### Train and test Data
    train_X, test_X = {}, {}
    train_X['A1'], test_X['A1'] = train_test_split(X['A1'], train_size=0.66, random_state=42)
    train_X['A2'], test_X['A2'] = train_test_split(X['A2'], train_size=0.66, random_state=42)

    train_Y1, test_Y1 = train_test_split(Y1, train_size=0.66, random_state=42)
    train_Y2, test_Y2 = train_test_split(Y2, train_size=0.66, random_state=42)

    train_error, test_error = {}, {}
    train_error['A1'], test_error['A1'] = train_test_split(error['A1'], train_size=0.66, random_state=42)
    train_error['A2'], test_error['A2'] = train_test_split(error['A2'], train_size=0.66, random_state=42)

    # Log-likelihood
    ll_chosenV = cp.hstack([i * theta_hat['tt'] * cp.exp(theta_hat['p'] * cp.log(cp.pos(j))) for i, j in
                            zip(train_Y1, train_X['A1'])]) + cp.hstack(
        [i * theta_hat['tt'] * cp.exp(theta_hat['p'] * cp.log(cp.pos(j))) for i, j in zip(train_Y2, train_X['A2'])])
    ll_logsum = cp.log_sum_exp(cp.vstack(
        [cp.hstack([theta_hat['tt'] * cp.exp(theta_hat['p'] * cp.log(cp.pos(i))) for i in train_X['A1']]),
         cp.hstack([theta_hat['tt'] * cp.exp(theta_hat['p'] * cp.log(cp.pos(i))) for i in train_X['A2']])]), axis=0)

    ll = cp.sum(ll_chosenV - ll_logsum)

    theta_hat['p'].value = theta_true['exp']
    problem = cp.Problem(cp.Maximize(ll))
    max_ll = problem.solve(solver='ECOS')

    # Optimization of hyperparameter
    cp_problem = problem
    # Solve
    results = {}
    for i, lambda_i in zip(range(len(lambdas)), lambdas):
        theta_hat['p'].value = lambda_i
        try:
            max_ll = cp_problem.solve()  # (solver = solver) # solver = 'ECOS', solver = 'SCS'

        except:
            pass  # Ignore invalid entries of lambda when the outer_optimizer fails.
            # theta_Z = {k: '' for k in cp_theta['Z'].keys()} #None
            # theta_Y = {k: '' for k in cp_theta['Y'].keys()} #None

        else:
            # theta_Z = {k: v.value for k, v in cp_theta['Z'].items()}
            # theta_Y = {k: v.value for k, v in cp_theta['Y'].items()}
            results[i] = {'p': theta_hat['p'].value, 'theta_tt': theta_hat['tt'].value, 'max_ll': max_ll}

    theta_tt_list = [results[i]['theta_tt'] for i, j in results.items()]
    max_ll_list = [results[i]['max_ll'] for i, j in results.items()]
    p_list = [results[i]['p'] for i, j in results.items()]

    ## Out of sampel cross validation
    prediction_error_test = {}
    ll_test = {}
    for i, j in results.items():
        test_Y1_hat = np.where(
            results[i]['theta_tt'] * test_X['A1'] ** results[i]['p'] + test_error['A1'] >= results[i]['theta_tt'] *
            test_X['A2'] ** results[i]['p'] + test_error['A2'], 1, 0)
        prediction_error_test[i] = len(np.where(test_Y1_hat != test_Y1)[0])

        # Log-likelihood test
        ll_chosenV_test = np.hstack([k * results[i]['theta_tt'] * np.exp(results[i]['p'] * np.log(l)) for k, l in zip(test_Y1, test_X['A1'])]) \
                          + np.hstack([k * results[i]['theta_tt'] * np.exp(results[i]['p'] * np.log(l)) for k, l in zip(test_Y2, test_X['A2'])])

        ll_logsum_test = np.log(np.exp(np.hstack([results[i]['theta_tt'] * np.exp(results[i]['p'] * np.log(k)) for k in test_X['A1']]))+
                       np.exp(np.hstack([results[i]['theta_tt']* np.exp(results[i]['p'] * np.log(k)) for k in test_X['A2']]))
                       )


        ll_test[i] = np.sum(ll_chosenV_test - ll_logsum_test)


    # print("test")
    # print(list(prediction_error.values()))
    # print(p_list[np.argmin(list(prediction_error.values()), axis=0)])
    # print(theta_tt_list[np.argmin(list(prediction_error.values()), axis=0)])

    prediction_error_train = {}
    for i, j in results.items():
        train_Y1_hat = np.where(
            results[i]['theta_tt'] * train_X['A1'] ** results[i]['p'] + train_error['A1'] >= results[i]['theta_tt'] *
            train_X['A2'] ** results[i]['p'] + train_error['A2'], 1, 0)
        prediction_error_train[i] = len(np.where(train_Y1_hat != train_Y1)[0])

    # print("train")
    # print(list(prediction_error_train.values()))
    # print(p_list[np.argmin(list(prediction_error_train.values()), axis=0)])
    # print(theta_tt_list[np.argmin(list(prediction_error_train.values()), axis=0)])

    # Using likelihood
    # print("train likelihood selection")
    # print(p_list[np.argmax(list(max_ll_list), axis=0)])

    #Biogeme box-cox
    results_biogeme = biogeme_time_perception(X= train_X, y= train_Y2[:, 0] + 1)
    print(results_biogeme)

    output = {}
    output['p_test_error'] = np.round(p_list[np.argmin(list(prediction_error_test.values()), axis=0)],1)
    output['p_test_ll'] = np.round(p_list[np.argmax(list(ll_test.values()), axis=0)], 1)
    output['p_train_error'] = np.round(p_list[np.argmin(list(prediction_error_train.values()), axis=0)],1)
    output['p_train_ll'] = np.round(p_list[np.argmax(list(max_ll_list), axis=0)],1)
    output['tt_train_ll'] = np.round(theta_tt_list[np.argmax(list(max_ll_list), axis=0)], 1)

    output['p_train_boxcox'] = np.round(results_biogeme['p'],1)
    output['tt_train_boxcox'] = np.round(results_biogeme['theta_tt'],1)

    return output

### Preference parameters

theta_true = {}
theta_true['tt'] = -3
theta_true['c'] = -0.1
theta_true['exp'] = 0.5

print(CV_timeperception(theta_true = theta_true, lambdas = np.arange(0, 2, 0.2), N = 1000, K = 1))

replicates = 100
results = {}
for i in range(replicates):
    results[i] = CV_timeperception(theta_true = theta_true, lambdas = np.arange(0, 1, 0.1), N = 100, K = 1)

print('test error')
p_test_error = [results[i]['p_test_error'] for i, j in results.items()]
print(np.mean(p_test_error))
print(np.sqrt(np.var(p_test_error)))

print('test likelihood')
p_test_ll = [results[i]['p_test_ll'] for i, j in results.items()]
print(np.mean(p_test_ll))
print(np.sqrt(np.var(p_test_ll)))


print('train error')
p_train_error = [results[i]['p_train_error'] for i, j in results.items()]
print(np.mean(p_train_error))
print(np.sqrt(np.var(p_train_error)))

print('train likelihood')
p_train_ll = [results[i]['p_train_ll'] for i, j in results.items()]
print(np.mean(p_train_ll))
print(np.sqrt(np.var(p_train_ll)))

print('boxcox')
p_train_boxcox = [results[i]['p_train_boxcox'] for i, j in results.items()]
print(np.mean(p_train_boxcox))
print(np.sqrt(np.var(p_train_boxcox)))

print('travel time parameter with likelihood')
tt_train_ll = [np.float(results[i]['tt_train_ll']) for i, j in results.items()]
print(np.mean(tt_train_ll))
print(np.sqrt(np.var(tt_train_ll)))

print('travel time parameter with boxcox')
tt_train_boxcox = [np.float(results[i]['tt_train_boxcox']) for i, j in results.items()]
print(np.mean(tt_train_boxcox))
print(np.sqrt(np.var(tt_train_boxcox)))


# =============================================================================
# Box-cox biogeme with real data on time perception experiment
# =============================================================================
import os
import pandas as pd
import numpy as np

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.messaging as msg
import biogeme.loglikelihood as ll
from biogeme.expressions import Beta, DefineVariable


# Path
subfolder = "/beijing-subway/"
folder_path = os.getcwd() + "/" + "examples/data/" + "time-perception/"

# Read the data
df1 = pd.read_csv(folder_path + "timeperceptiondb.csv")

# Replace strings to numeric variables
df1['experimentType'] = np.where(df1['experimentType'] == "animated", 0, 1)
df1['experimentalCondition'] = np.where(df1['experimentalCondition'] == "control", 0, 1)
df1['city'] = np.where(df1['city'] == "Santiago", 0, 1)
df1['participantId'] = df1['participantId'].str[1:].astype(int)

database = db.Database('timeperception', df1)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
#print(database.data.describe())

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Removing some observations can be done directly using pandas.
# remove = ((database.data.experimentType == 0) |  (database.data.city == 1))
# remove = ((database.data.experimentType == 1) | database.data.experimentalCondition == 1)
# remove = ((database.data.experimentType == 1) | database.data.experimentalCondition == 1 | (database.data.city == 0))
remove = ((database.data.experimentType == 1) | (database.data.experimentalCondition == 1) | (database.data.city == 1))
# remove = ((database.data.experimentType == 1) | database.data.experimentalCondition == 0)
# remove = ((database.data.experimentType == 1) | (database.data.city == 1) | database.data.experimentalCondition == 1)
# remove = ((database.data.experimentType == 1) | (database.data.city == 0) | database.data.experimentalCondition == 1)

database.data.drop(database.data[remove].index,inplace=True)

# Parameters to be estimated
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_WAITING = Beta('B_WAITING', 0, None, None, 0)
B_TRAVEL = Beta('B_TRAVEL', 0, None, None, 0)
LAMBDA = Beta('LAMBDA', 1.5, 0.0001, 5, 0)
LAMBDA_WAITING = Beta('LAMBDA_WAITING', 1.5, 0.0001, 5, 0)
LAMBDA_TRAVEL = Beta('LAMBDA_TRAVEL', 1.5, 0.0001, 5, 0)

# # # Definition of the utility functions
# V1 = B_WAITING * models.boxcox(w1+1, LAMBDA) + \
#      B_TRAVEL * models.boxcox(v1+1, LAMBDA)
# V2 = B_WAITING * models.boxcox(w2+1, LAMBDA) + \
#      B_TRAVEL * models.boxcox(v2+1, LAMBDA)

# # # Definition of the utility functions
# V1 = B_WAITING * w1 +  B_TRAVEL * models.boxcox(v1, LAMBDA)
# V2 = B_WAITING * w2 + B_TRAVEL * models.boxcox(v2, LAMBDA)

# # # Definition of the utility functions
V1 = B_TIME*1.2 * w1 + B_TIME*models.boxcox(v1, LAMBDA)
V2 = B_TIME*1.2 * w2 + B_TIME*models.boxcox(v2, LAMBDA)
# V1 = B_TIME * w1 + B_TIME*models.boxcox(v1, LAMBDA)
# V2 = B_TIME * w2 + B_TIME*models.boxcox(v2, LAMBDA)

# # Exponent are different led to recover the true preference but in my views this is just luck
# V1 = B_WAITING * models.boxcox(w1+1, LAMBDA_WAITING) + \
#      B_TRAVEL * models.boxcox(v1+1, LAMBDA_TRAVEL)
# V2 = B_WAITING * models.boxcox(w2+1, LAMBDA_WAITING) + \
#      B_TRAVEL * models.boxcox(v2+1, LAMBDA_TRAVEL)

# V1 = B_WAITING * w1 + B_TRAVEL * v1
# V2 = B_WAITING * w2 + B_TRAVEL * v2

# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V = V , av = None, i = choice)

# Define level of verbosity
logger = msg.bioMessage()
logger.setSilent()
#logger.setWarning()
#logger.setGeneral()
#logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = '08boxcox'

# Estimate the parameters
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(pandasResults)
ll.loglikelihood(results) #-185.6453

4.5-1/0.24

# =============================================================================
# PLOTS
# =============================================================================

import matplotlib.pyplot as plt

## Correlation between perception and preference parameter

plt.scatter(p_train_ll, tt_train_ll)
plt.show()

np.corrcoef(p_train_ll, tt_train_ll)



## histogram with estimates

np.argmax(max_ll_list)

p_list[np.argmax(max_ll_list,axis = 0)]
max_ll_list[np.argmax(max_ll_list,axis = 0)]


max_ll_list = np.array(max_ll_list)

max_ll_list

plt.hist(theta_tt_list, bins=30, density=True)  # `density=False` would make countsw
plt.ylabel('Probability')
plt.xlabel('Data')
plt.show()

# Plot the histogram.
plt.hist(np.array(theta_tt_list), density=False, alpha=0.6)
plt.show()
# Plot the PDF.
from scipy.stats import norm
xmin, xmax = plt.xlim()
x = np.linspace(-3, 2, 100)
mu, std = norm.fit(theta_tt_list)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.show()


# =============================================================================
# BIOGEME
# =============================================================================

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.messaging as msg
from biogeme.expressions import Beta, DefineVariable

# Read the data
df = pd.read_csv('https://raw.githubusercontent.com/michelbierlaire/biogeme/master/examples/notebooks/swissmetro.dat', '\t')
database = db.Database('swissmetro', df)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
#print(database.data.describe())

# The following statement allows you to use the names of the variable
# as Python variable.
globals().update(database.variables)

# Removing some observations can be done directly using pandas.
remove = (((database.data.PURPOSE != 1) &
          (database.data.PURPOSE != 3)) |
         (database.data.CHOICE == 0))
database.data.drop(database.data[remove].index,inplace=True)

# Here we use the "biogeme" way for backward compatibility
# exclude = (('PURPOSE' != 1) * ('PURPOSE' != 3) + ('CHOICE' == 0)) > 0
# exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
# database.remove(exclude)

# Parameters to be estimated
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = Beta('ASC_SM', 0, None, None, 1)
B_TIME = Beta('B_TIME', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)
LAMBDA = Beta('LAMBDA', 1.5, 0.0001, 5, 0)

# Definition of new variables
SM_COST = SM_CO * (GA == 0)
TRAIN_COST = TRAIN_CO * (GA == 0)

# Definition of new variables: adding columns to the database
CAR_AV_SP = DefineVariable('CAR_AV_SP', CAR_AV * (SP != 0), database)
TRAIN_AV_SP = DefineVariable('TRAIN_AV_SP', TRAIN_AV * (SP != 0), database)
TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0, database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100, database)
SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0, database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100, database)
CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100, database)
CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100, database)

# Definition of the utility functions
V1 = ASC_TRAIN + \
     B_TIME * models.boxcox(TRAIN_TT_SCALED, LAMBDA) + \
     B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + \
     B_TIME * models.boxcox(SM_TT_SCALED, LAMBDA) + \
     B_COST * SM_COST_SCALED
V3 = ASC_CAR + \
     B_TIME * models.boxcox(CAR_TT_SCALED, LAMBDA) + \
     B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2,
     3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP,
      2: SM_AV,
      3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = models.loglogit(V, av, CHOICE)

# Define level of verbosity
logger = msg.bioMessage()
logger.setSilent()
#logger.setWarning()
#logger.setGeneral()
#logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = '08boxcox'

# Estimate the parameters
results = biogeme.estimate()
pandasResults = results.getEstimatedParameters()
print(pandasResults)