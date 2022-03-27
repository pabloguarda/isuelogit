import pandas as pd
import numpy as np


def binaryLogit(x1, x2, theta):
    """ Binary logit model """

    sum_exp = np.exp(theta * x1) + np.exp(theta * x2)
    p1 = np.exp(theta * x1) / sum_exp
    p2 = 1 - p1

    return np.array([p1, p2])  # (P1,P2)



# def X_scaling(x,)

def compute_loss_link_level_model(theta, q, lambda_hp, D, M, C, Y, Z, x, idx_links, norm_o, norm_r):
    v = np.exp(v_normalization(v=np.sum(theta * np.hstack([np.vstack(Y), Z]),
                                        axis=1), C=C))

    V = C.dot(v)  # Denominator logit functions

    p = v / V

    # TODO: May give directly dot product of q and M to speed up as they do not change over iterations
    f = np.multiply(q.dot(M), p)

    # TODO: I may provide directly with the small matrix D
    x_hat = D[idx_links].dot(f)
    x_N = x[idx_links]

    objective_term = np.linalg.norm(x_hat - x_N, ord=norm_o)
    regularization_term = lambda_hp * np.linalg.norm(theta, ord=norm_r)

    loss = objective_term / len(idx_links) + regularization_term

    return loss


def loss_link_level_model(parameters, range_theta, range_q, end_params, q, theta, lambda_hp, D, M, C, Y, Z, x,
                          idx_links, norm_o,
                          norm_r):  # , scale = {'mean': True, 'std': False}
    '''

    :param o: o-norm of the objective function gap
    :param p: p-norm of the regularization term
    '''

    loss = {}

    # parameters = x0

    # theta0 = parameters[range_theta]
    # q0 = parameters[range_q]

    for i in M.keys():

        # n_theta = Y[i].shape[1] + Z[i].shape[1]
        if end_params['theta'] and end_params['q']:
            theta0 = parameters[range_theta]  # parameters[0:n_theta]
            q0 = parameters[range_q]  # parameters[n_theta:]

        elif end_params['theta']:
            theta0 = parameters
            q0 = q  # [i]

        elif end_params['q']:
            q0 = parameters
            theta0 = theta

        loss[i] = compute_loss_link_level_model(theta=theta0, q=q0, lambda_hp=lambda_hp, D=D[i], M=M[i], C=C[i], Y=Y[i],
                                                Z=Z[i]
                                                , x=x[i], idx_links=idx_links[i], norm_o=norm_o, norm_r=norm_r)

    return np.sum(np.array(list(loss.values())))  # objective_term/len(idx_links)


# =============================================================================
# 7) BENCHMARK REGRESSION FRESNO
# =============================================================================

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

link_flow = predicted_link_counts_over_iterations_df[
    predicted_link_counts_over_iterations_df.observed == 1].true_count.values.reshape(-1, 1)

link_traveltime = predicted_link_traveltime_over_iterations_df[
                      predicted_link_traveltime_over_iterations_df.observed == 1].iloc[:, -1].values.astype(
    'float').reshape(-1, 1)

reg = LinearRegression().fit(link_traveltime, link_flow)

X = sm.add_constant(link_traveltime)
Y = link_flow

model = sm.OLS(Y, X)

results = model.fit()

results.params

print(results.summary())

results.predict(X)

mse = np.mean((results.predict(X) - Y) ** 2)

print('MSE under OLS:', round(mse, 1))

# results.mse_model
# results.mse_total
reg.coef_
reg.intercept_

# =============================================================================
# Non-convex entropy minimization
# =============================================================================

from scipy.optimize import minimize
from scipy.stats import entropy

#Network 1


xN1 = np.concatenate((np.array(results1['t']).reshape(len(results1['t']),1),Z1),1)
qN1 = Q1
theta_true_N1 = np.append(theta_t,theta_Z1)

def logit(theta, x):
    return (np.exp(x.dot(theta))/np.sum(np.exp(x.dot(theta)))).reshape(x.shape[0])

fa = x1.value #np.round(qN1*logit(theta_true, x = xN1),2) # If rounding is at the first decimal, the theta estimates are pretty bad

fa.shape
logit(theta_true_N1, x = xN1).shape

def logit_constraint(theta, x = xN1):
    return qN1 * (logit(theta=theta, x = xN1)) - fa

def objective(theta):
    #return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]
    return qN1*entropy(logit(theta, x = xN1))

logit_constraint(theta = [-1.2,-1.2,-2])

cons_list = [logit_constraint]

constraints = [{'type': 'ineq', 'fun': cons} for cons in cons_list]

# theta_0 = [0,0,0]
theta_0 = [-0.2,-0.5,-0.35]

# theta_0 = theta_true_N1 #theta_Z1#[0.1,0.5]

#Unconstrained problem
sol = minimize(objective, theta_0, tol=1e-5)

#Constrained problem
sol = minimize(objective, theta_0, constraints=constraints, tol=1e-5)

#Constraints

#Solution
print(sol.x)
print(theta_true_N1)


## Network 2

D2, M2, Q2

xN2 = np.concatenate((np.array(results2['t']).reshape(len(results2['t']),1),Z2),1)
qN2 = Q2
theta_true_N2 = np.append(theta_t,theta_Z2)

# theta = theta_true_N2

def logit_constraint_a1(theta):
    return np.sum(Q2[0] * logit(theta=theta, x = np.array([[xN2[0]+xN2[2]],[xN2[0]+xN2[3]]]))) \
           - x2.value[0]\
           #- np.round(x2.value[0],0)

def logit_constraint_a2(theta):
    return np.sum(Q2[1] * (logit(theta=theta, x = np.array([[xN2[1]+xN2[2]],[xN2[0]+xN2[3]]])))) \
           - x2.value[1]\
           # - np.round(x2.value[1],0)

def logit_constraint_a3(theta):
    return np.sum(Q2[0] * logit(theta=theta, x = np.array([[xN2[0]+xN2[2]],[xN2[0]+xN2[3]]]))[0] \
            + Q2[1] * logit(theta=theta, x=np.array([[xN2[1] + xN2[2]], [xN2[1] + xN2[3]]]))[0] \
            + Q2[2] * logit(theta=theta, x = np.array([[xN2[2]],[xN2[3]]]))[0]) \
           - x2.value[2]

def logit_constraint_a4(theta):
    return np.sum(Q2[0] * logit(theta=theta, x = np.array([[xN2[0]+xN2[2]],[xN2[0]+xN2[3]]]))[1] \
            + Q2[1] * logit(theta=theta, x=np.array([[xN2[1] + xN2[2]], [xN2[1] + xN2[3]]]))[1] \
            + Q2[2] * logit(theta=theta, x = np.array([[xN2[2]],[xN2[3]]]))[1]) \
           - x2.value[3]
           # - np.round(x2.value[3])

def objective(theta):
    #return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]
    return np.sum(Q2[0]*entropy(logit(theta=theta, x = np.array([[xN2[0]+xN2[2]],[xN2[0]+xN2[3]]])))\
           +Q2[1]*entropy(logit(theta=theta, x = np.array([[xN2[1]+xN2[2]],[xN2[1]+xN2[3]]])))\
           +Q2[2]*entropy(logit(theta=theta, x = np.array([[xN2[2]],[xN2[3]]]))))

def penalty_logit_contraints(theta, penalty_factor):
    # penalty = logit_constraint_a1(theta) ** 2 + logit_constraint_a2(theta) ** 2 +\
    # logit_constraint_a3(theta) ** 2 + logit_constraint_a4(theta) ** 2

    #https://en.wikipedia.org/wiki/Penalty_method
    sum_penalty = penalty_factor*(np.max(logit_constraint_a3(theta),0)** 2 + np.max(logit_constraint_a4(theta),0) ** 2)

    return sum_penalty

def lassopenalty_logit_contraints(theta, penalty_factor):
    # penalty = logit_constraint_a1(theta) ** 2 + logit_constraint_a2(theta) ** 2 +\
    # logit_constraint_a3(theta) ** 2 + logit_constraint_a4(theta) ** 2

    #https://en.wikipedia.org/wiki/Penalty_method
    sum_penalty = penalty_factor*(np.abs(logit_constraint_a3(theta)) + np.abs(logit_constraint_a4(theta)))

    return sum_penalty

def objective_penalty(theta, penalty_factor = 3):
    return objective(theta) + lassopenalty_logit_contraints(theta=theta, penalty_factor=penalty_factor)
    # return objective(theta) + penalty_logit_contraints(theta = theta, penalty_factor = penalty_factor)

def objective_only_contraints_penalty(theta, penalty_factor = 1):
    return lassopenalty_logit_contraints(theta=theta, penalty_factor=penalty_factor)
    # return objective(theta) + penalty_logit_contraints(theta = theta, penalty_factor = penalty_factor)

cons_list_N2  = [logit_constraint_a1,logit_constraint_a2,logit_constraint_a3,logit_constraint_a4]
# cons_list_N2  = [logit_constraint_a1]

constraints_N2 = [{'type': 'eq', 'fun': cons} for cons in cons_list_N2 ]

# constraints_params = [{'type': 'eq', 'fun': cons} for cons in cons_list_N2 ]

# theta_0_N2 = [0.1,0.1,0.1]
theta_0_N2 = 1*theta_true_N2
# theta_0_N2 = 0.8*theta_true_N2
# theta_0_N2 = [-0.1, -0.3, -0.3]
# theta_0_N2 = [-0.2,-0.1, -0.1]
# theta_0_N2 = [-0.2,-0.1]

# theta_0_N2 = [0,0,0]
theta_0_N2 = [-0.4,-0.2,-0.1]

# theta_0_N2 = theta_true_N2 #[0.1,0.5]

bnds = ((None, 0), (theta_true_N2[1],theta_true_N2[1]),  (theta_true_N2[2],theta_true_N2[2]))

#Unconstrained problem with no penalty
sol = minimize(objective, theta_0_N2, bounds = bnds , constraints=constraints_N2)

# #Unconstrained problem with penalty
sol = minimize(objective_penalty, theta_0_N2, bounds = bnds)

#Adding bounds for parameters
# bnds = ((None, 0), (None,0), (theta_true_N2[2],theta_true_N2[2]))
bnds = ((None, 0), (theta_true_N2[1],theta_true_N2[1]),  (theta_true_N2[2],theta_true_N2[2]))

# bnds = ((0, None), (0, None), (0, None))
sol = minimize(objective_penalty, x0 = theta_0_N2, bounds = bnds #,  method = 'Nelder-Mead'
               )

sol = minimize(objective_penalty, x0 = theta_0_N2, bounds = bnds #,  method = 'Nelder-Mead'
               )

# sol = minimize(objective_penalty, x0 = theta_0_N2,  method = 'Nelder-Mead'
#                )

# #Constrained problem
# sol = minimize(objective, theta_0_N2, constraints=constraints_N2, tol=1e-5)

# sol = minimize(objective, theta_0_N2, constraints=constraints_N2, tol=1)

#Solution
print(sol.x)
print(theta_0_N2)
print(theta_true_N2)

#Objective at optimal
print(objective(sol.x))
print(objective(theta_true_N2))
print(objective(theta_0_N2))

#Objective with penalty
print(objective_penalty(sol.x))
print(objective_penalty(theta_true_N2))
print(objective_penalty(theta_0_N2))

# objective([0,0,0])


#Constraints
sol_N2 = sol.x #theta_Z2 #sol.x

print(logit_constraint_a1(sol.x))
print(logit_constraint_a2(sol.x))
print(logit_constraint_a3(sol.x))
print(logit_constraint_a4(sol.x))

print(logit_constraint_a1(theta_0_N2))
print(logit_constraint_a2(theta_0_N2))
print(logit_constraint_a3(theta_0_N2))
print(logit_constraint_a4(theta_0_N2))

print(logit_constraint_a1(theta_true_N2))
print(logit_constraint_a2(theta_true_N2))
print(logit_constraint_a3(theta_true_N2))
print(logit_constraint_a4(theta_true_N2))



## Satisfaction problem

from scipy.optimize import fsolve

objective_only_contraints_penalty()

bnds = ((None, 0), (None, 0),  (None, 0))

sol_satisfaction = minimize(objective_only_contraints_penalty, x0 = [-1,-1,-1], bounds = bnds) #,  method = 'Nelder-Mead')

from scipy.optimize import basinhopping

# To get rid of local minima use basinhopping()

# bnds = ((None, 0), (theta_true_N2[1],theta_true_N2[1]),  (theta_true_N2[2],theta_true_N2[2]))
# bnds = ((None, 0), (None, 0),  (theta_true_N2[2],theta_true_N2[2]))
# bnds = ((None, 0), (theta_true_N2[1],theta_true_N2[1]),  (None, 0))
bnds = ((theta_true_N2[0],theta_true_N2[0]), (None, 0),  (theta_true_N2[2],theta_true_N2[2]))

minimizer_kwargs = {"method": "L-BFGS-B","bounds":bnds}
guess = [100,-1,-1]
# guess = [i*j for i,j in zip(theta_true_N2,[-0.5,1,1])]

sol_satisfaction_loop = basinhopping(objective_only_contraints_penalty, guess , niter=300,minimizer_kwargs=minimizer_kwargs)


print(sol_satisfaction.x)
print(sol_satisfaction_loop.x)
print(theta_true_N2)

print(logit_constraint_a1(sol_satisfaction.x))
print(logit_constraint_a2(sol_satisfaction.x))
print(logit_constraint_a3(sol_satisfaction.x))
print(logit_constraint_a4(sol_satisfaction.x))

print(objective_only_contraints_penalty([-1,-1,-1]))
print(objective_only_contraints_penalty(theta_true_N2))
print(objective_only_contraints_penalty(sol_satisfaction.x))
print(objective_only_contraints_penalty(sol_satisfaction_loop.x))





print(sol.x)

class Modeller:

    def __init__(self, name = None, date = None):
        self._name = name
        self._date = date

    def create_od_from_system(self, trips):
        od = pd.pivot_table(trips[['ostation', 'dstation']], index=['ostation'], columns=['dstation'], aggfunc=[len])
        od = od.replace(np.nan, 0)

        return od

    def create_od_from_agents(self, trips):

        df = pd.DataFrame(columns = ['o','d'])

        for i in range(len(trips)):
            trip = trips[i]
            df = df.append({'o':trip.init, 'd':trip.destination}, ignore_index = True)

        od = pd.pivot_table(df[['o', 'd']], index=['o'], columns=['d'], aggfunc=[len])
        od = od.replace(np.nan, 0)

        np.sum(od.to_numpy())  # Checking the total number of trips is consistent with the original dataset

        return od
    # def build_od(self, trips):
    #     """:argument trips: list of trips"""


def generate_link_traveltimes_equilibrium(N, theta: np.array, k_Y: [], k_Z: [], eq_params: {}):
    N_copy = networks.clone_network(network=N, label='Clone', randomness={'Q': False, 'BPR': False, 'Z': False})[
        N.key]

    results_sue_msa = equilibrium.sue_logit_iterative(Nt=N_copy, theta=theta, features_Y=k_Y, features_Z=k_Z, params=eq_params)

    return results_sue_msa['tt_x']


# def loss_link_level_model(theta, lambda_hp, D, M, Y, Z, q, x, idx_links, norm_o = 2, norm_r=1): #, scale = {'mean': True, 'std': False}
#     '''
#
#     :param o: o-norm of the objective function gap
#     :param p: p-norm of the regularization term
#     '''
#
#     objective_term = {}
#     regularization_term = {}
#     loss = {}
#
#     # theta = x0
#
#     for i in M.keys():
#
#         theta_estimate = np.array(theta)
#
#         # theta_estimate = np.ones(np.array(theta).shape)
#
#         v = np.exp(v_normalization(v=np.sum(theta_estimate * np.hstack([np.vstack(Y[i]), Z[i]]),
#             axis=1), M=M[i]))
#
#         V = choice_set_matrix_from_M(M[i]).dot(v)  # Denominator logit functions
#
#         p = v / V
#
#         # np.multiply(q[i].dot(M[i]), p)
#         #
#         # M[i].dot(np.multiply(np.ones(q[i].shape).dot(M[i]),p))
#
#         f = np.multiply(q[i].dot(M[i]),p)
#
#         # M[i].dot(f)
#
#         x_hat = D[i][idx_links[i]].dot(f)
#         x_N = x[i][idx_links[i]]
#
#         # if scale['mean'] or scale['std']:
#         #     # Scaling by attribute
#         #     Y = preprocessing.scale(Y, with_mean=scale['mean'], with_std=scale['std'], axis=0)
#         #     Z = preprocessing.scale(Z, with_mean=scale['mean'], with_std=scale['std'], axis=1)
#
#
#         # if lambda_hp == None:
#         #     return x_hat[idx_links] -x[idx_links]
#         #
#         # else:
#         objective_term[i] = np.linalg.norm(x_hat -x_N, ord = norm_o)
#         regularization_term[i] = lambda_hp*np.linalg.norm(theta, ord=norm_r)
#
#         loss[i] = objective_term[i]/len(idx_links[i]) + regularization_term[i]
#
#     return np.sum(np.array(list(loss.values()))) #objective_term/len(idx_links)


