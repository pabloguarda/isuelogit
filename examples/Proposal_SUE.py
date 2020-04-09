import numpy as np
import cvxpy as cp
#pip install cvxopt


#Incidence Matrix

# TODO: Create function that receives arcs origin and destinations, and its index, and create the matrix automatically

# Lets consider a small network with 4 nodes: 1,2,3.4
# This network is shown in Fig 3.1, page 58 of Sheffi's book.
# There are for nodes.
# The arcs can be written as follows (1,3,1),(2,3,1), (3,4,1), (3,4,2)
# There are trips between the 4 OD pairs, unlike the Sheffi's one that only consider 2 OD pairs  (1-> 4 and 2 -> 4).

nOD = 4

# Columns of incidence matrix (one per route and rows are the [4] links)
nLinks = 4

# Routes between OD pairs
nRoutes = 6
# OD 1 -> 4: r1 = (a1,a3), r2 = (a1,a4)
# OD 2 -> 4: r1 = (a2,a3), r2 = (a2,a4)
# OD 3 -> 4: r1 = (a3), r2 = (a4)

route_cols = []

# OD 1 -> 4:
route_cols.append([1,0,1,0])
route_cols.append([1,0,0,1])
# OD 2 -> 4:
route_cols.append([0,1,1,0])
route_cols.append([0,1,0,1])
# OD 3 -> 4:
route_cols.append([0,0,1,0])
route_cols.append([0,0,0,1])

#Incidence Matrix
I = np.array(route_cols).T
print(I)

#Indexes from route vector for a given OD pair
ind_f = dict()
ind_f[(1,4)] = [0,1]
ind_f[(2,4)] = [2,3]
ind_f[(3,4)] = [4,5]

#OD Matrix
OD = np.zeros([nOD,nOD])
#Setting particular values

OD[1-1,4-1] = 10#10
OD[2-1,4-1] = 20#20
OD[3-1,4-1] = 30

# Finding the optimal solution for this simple example

# # Create decision variables
# for i in range(0, len(self.rows)):
#     for j in range(0, len(self.columns)):
#         self.x[(i, j)] = self.cp.Variable(9)  # (i,j,k)

# Decision variables for route flows
f = {}
# for i in range(0, nRoutes):
#     f[i] = cp.Variable(1)

f = cp.Variable([nRoutes])

# Decision variables for link flows
# x = {}
# for i in range(0, nLinks):
#     x[i] = cp.Variable(1)
x = cp.Variable([nLinks])

#Constraints
constraints = []
constraints += [f*I.T == x] # Arc-link flows
# x*I.T

#Routes O-D flows
constraints += [cp.sum(f[(ind_f[(1,4)])]) == OD[0,3]] # routes-od 1-4 flow
# constraints += [cp.sum(f) == np.array([0,0,0,1])] # routes-od 1-4 flow
# constraints += [f[0] == OD[0,3]] # routes-od 1-4 flow
# constraints += [f[0]+f[1] == 100] # routes-od 1-4 flow
# constraints += [f[0] == 2] # routes-od 1-4 flow

constraints += [cp.sum(f[(ind_f[(2,4)])]) == OD[1,3]] # routes-od 2-4 flow
constraints += [cp.sum(f[(ind_f[(3,4)])]) == OD[2,3]] # routes-od 3-4 flow

#Non negativity flows
constraints += [f >= 0]  # (i,j,k)
# constraints += [x >= 0]  # (i,j,k)

theta = 40E-1
print(theta)

# cp_objective = cp.Minimize(cp.sum(x**2) + 1 / theta * cp.sum(f @ cp.log(f).T))
# cp_objective = cp.Minimize(cp.sum(x**2) + 1 / theta * f[0,2] * cp.log(f[0]))

# cp_objective = cp.Minimize(cp.sum(x**2) + cp.sum(f @ cp.log(f+).T))

# Note respect to DCP: http://cvxr.com/cvx/doc/dcp.html

#Beckman integral for cost in arcs
Z11 = 0.01*x[0]+0.01*x[0]**2
Z12 = 0.02*x[1]+0.02*x[1]**2
Z13 = 0.03*x[2]+0.03*x[2]**2
Z14 = 0.06*x[3]+0.06*x[3]**2

diff_z11 = 0.01*(1+ 2*x[0])
diff_z12 = 0.02*(1+ 2*x[1])
diff_z13 = 0.03*(1+ 2*x[2])
diff_z14 = 0.06*(1+ 2*x[3])

x.value

diff_z11.value

x.value

Z13.value
Z13.value
Z14.value

Z1 = Z11 + Z12 + Z13 + Z14

cp_objective = cp.Minimize(Z1 -1/theta*cp.sum(cp.entr(f)))

cp_objective.is_dcp()

#Problem
cp_problem = cp.Problem(cp_objective, constraints)

cp_problem.is_dcp()

# cp_problem.variables()
# cp_problem.constraints
#Optimize
# objective_value = cp_problem.solve(verbose = True)
objective_value = cp_problem.solve()

# objective_value = cp_problem.solve(solver = 'SCS')

objective_value

results = []

for variable in cp_problem.variables():
    results.append(np.round(variable.value))
    print("Variable %s: value %s" % (variable.name(), np.round(variable.value)))



# constraints[0].value()
# constraints[5].value()

# f.value()
# self.cp_problem = cp.


# =============================================================================
# Recovery of logit model parameters using a discrete choice modelling framework
# =============================================================================

import scipy.optimize

def binaryLogit(x1,x2,theta):
    sum_exp = np.exp(theta * x1)+np.exp(theta * x2)
    p1 = np.exp(theta * x1)/ sum_exp
    p2 = 1-p1
    return np.array([p1,p2])  # (P1,P2)
    # delta_x2x1 = x2-x1
    # return 1/(1+np.exp(theta*delta_x)) #P1

# binaryLogit(x1 = Z13.value,x2 = Z14.value, theta = theta)*(np.sum(OD))
# Z12.value
# Z11.value
binaryLogit(x1 = diff_z13.value,x2 = diff_z14.value, theta = -theta)*(30)

OD

# binaryLogit(delta_x = 5,theta = -1)
# binaryLogit(delta_x = 10,theta = -1)
# binaryLogit(x = 10,theta = -0.5)

def F(z):
    return [binaryLogit(x1 = diff_z13.value,x2 = diff_z14.value, theta = -z[0])[0] - results[0][2] / (results[0][2] + results[0][3])]
    # return [1/(1+np.exp(x[0]*(Z13.value-Z14.value)))-results[0][2]/(results[0][2]+results[0][3])]

# F([1])
# x
sol = scipy.optimize.root(F,0)

z1 = np.round(sol.x,4)
print(z1)
theta

F(z = z1)


########### Flows in route between OD pairs are known  ##########################

#Optimization variable
theta_logit = cp.Variable(1)

# binaryLogit(x1 = diff_z13.value,x2 = diff_z14.value, theta = -theta)*(30)


#Route 1-4
CE14A = (diff_z11.value+diff_z13.value)*theta_logit - cp.log_sum_exp(np.array([(diff_z11.value+diff_z13.value),(diff_z11.value+diff_z14.value)]) * theta_logit)
CE14B = (diff_z11.value+diff_z14.value)*theta_logit - cp.log_sum_exp(np.array([(diff_z11.value+diff_z13.value),(diff_z11.value+diff_z14.value)]) * theta_logit)

#Route 2-4
CE24A = (diff_z12.value+diff_z13.value)*theta_logit - cp.log_sum_exp(np.array([(diff_z12.value+diff_z13.value),(diff_z12.value+diff_z14.value)]) * theta_logit)
CE24B = (diff_z12.value+diff_z14.value)*theta_logit - cp.log_sum_exp(np.array([(diff_z12.value+diff_z13.value),(diff_z12.value+diff_z14.value)]) * theta_logit)

# CE24A = (diff_z13.value)*theta_logit - cp.log_sum_exp(np.array([(diff_z13.value),(diff_z14.value)]) * theta_logit)
# CE24B = (diff_z14.value)*theta_logit - cp.log_sum_exp(np.array([(diff_z13.value),(diff_z14.value)]) * theta_logit)

#Route 3-4
CE34A = (diff_z13.value)*theta_logit - cp.log_sum_exp(np.array([(diff_z13.value),(diff_z14.value)]) * theta_logit)
CE34B = (diff_z14.value)*theta_logit - cp.log_sum_exp(np.array([(diff_z13.value),(diff_z14.value)]) * theta_logit)

CE34A.value
CE34B.value



#Objective
# cp_objective_logit = cp.Maximize(10*(CE14A+CE14B)+20*(CE24A+CE24B)+30*(CE34A+CE34B))

# cp_objective_logit = cp.Maximize(f.value * np.array([CE14A,CE14B,CE24A,CE24B,CE34A,CE34B]))

cp_objective_logit = cp.Maximize(f.value[0] * CE14A + f.value[1] * CE14B + f.value[2] * CE24A + f.value[3] *CE24B + f.value[4] * CE34A + f.value[5]*CE34B)

# cp_objective_logit = cp.Maximize(f.value[4] * CE24A + f.value[5] *CE24B)

cp_objective_logit.is_dcp()

cp.Maximize(f.value[4] * CE24A + f.value[5] *CE24B)

f.value[2]/f.value[3]

f.value[4]/f.value[5]


# cp_objective_logit = cp.Maximize(f.value[0] * CE14A + f.value[1] * CE14B + f.value[2] * CE24A)

# cp_objective_logit = cp.Maximize(f.value[4] * CE34A + f.value[5]*CE34B)

f.value

I

# cp_objective_logit = cp.Maximize(cp.multiply(f.value,np.array([CE14A,CE14B,CE24A,CE24B,CE34A,CE34B]))

#Problem
cp_problem_logit = cp.Problem(cp_objective_logit, constraints = None)

cp_problem_logit.solve()

theta_logit.value
# theta_logit.value

theta

B = np.array([[1,2,3],[3,5,6]])
C = [1,2,5]

B*C


### Route assignment given different theta values

def loglikelihood_logit_network(theta_sensitivity):
    # Route 1-4
    P14A, P14B = binaryLogit(x1=diff_z11.value + diff_z13.value, x2=diff_z11.value + diff_z14.value,
                             theta=theta_sensitivity)

    # Route 2-4s
    P24A, P24B = binaryLogit(x1=diff_z12.value + diff_z13.value, x2=diff_z12.value + diff_z14.value,
                             theta=theta_sensitivity)

    # Route 3-4
    P34A, P34B = binaryLogit(x1=diff_z13.value, x2=diff_z14.value, theta=theta_sensitivity)

    P = np.array([P14A,P14B,P24A,P24B,P34A,P34B])
    q = np.array([10, 10, 20, 20, 30, 30]).reshape(6,1)#[OD[:,3]]

    return np.append(np.sum(q * P * np.log(P)),P*q)

    # return (np.sum(q * P * np.log(P)),P)


loglikelihood_logit_network(theta_sensitivity = theta_logit.value)

loglikelihood_logit_network(theta_sensitivity = 5*theta_logit.value)

A = [loglikelihood_logit_network(theta_sensitivity = i*theta_logit.value) for i in np.arange(1,4,0.5)]
A = np.array(A)

I@A

I
[loglikelihood_logit_network(theta_sensitivity = i*theta_logit.value) for i in reversed(np.arange(0.2,1,0.2))]


loglikelihood_logit_network(theta_sensitivity = 3*theta_logit.value)

loglikelihood_logit_network(theta_sensitivity = 0.5*theta_logit.value)
loglikelihood_logit_network(theta_sensitivity = 0.2*theta_logit.value)




########### Entropy Approach ##########################


## Now we can use the cross entropy
#Examples: https://www.cvxpy.org/examples/machine_learning/logistic_regression.html
# cross_entropy = cp.sum(
#     cp.multiply(Y, X @ beta) - cp.logistic(X @ beta)
# )
#
# #The maximization of softmax function can be rewritten as x-log(x)
#
# cp_objective_logit = cp.Minimize(Z1 -1/theta*cp.sum(cp.entr(f)))



#Route 1-4
CE14A = cp.entr(cp.exp((diff_z11.value+diff_z13.value) * theta_logit)/(cp.exp((diff_z11.value+diff_z13.value) * theta_logit)+cp.exp((diff_z11.value+diff_z14.value) * theta_logit)))
CE14B = cp.entr(cp.exp((diff_z11.value+diff_z14.value) * theta_logit)/(cp.exp((diff_z11.value+diff_z13.value) * theta_logit)+cp.exp((diff_z11.value+diff_z14.value) * theta_logit)))

#Route 2-4
CE24A = cp.entr(cp.exp((diff_z12.value+diff_z13.value) * theta_logit)/(cp.exp((diff_z12.value+diff_z13.value) * theta_logit)+cp.exp((diff_z11.value+diff_z14.value) * theta_logit)))
CE24B = cp.entr(cp.exp((diff_z12.value+diff_z14.value) * theta_logit)/(cp.exp((diff_z12.value+diff_z13.value) * theta_logit)+cp.exp((diff_z11.value+diff_z14.value) * theta_logit)))

#Route 3-4
CE34A = cp.entr(cp.exp((diff_z13.value) * theta_logit)/(cp.exp((diff_z13.value) * theta_logit)+cp.exp((diff_z14.value) * theta_logit)))
CE34B = cp.entr(cp.exp((diff_z14.value) * theta_logit)/(cp.exp((diff_z13.value) * theta_logit)+cp.exp((diff_z14.value) * theta_logit)))

#Objective
cp_objective_logit = cp.Minimize(10*(CE14A+CE14B)+20*(CE24A+CE24B)+30*(CE34A+CE34B))


# cp_objective_logit = cp.Maximize(cp.exp((diff_z11.value+diff_z13.value) * theta_logit)/(cp.exp((diff_z11.value+diff_z13.value) * theta_logit)+cp.exp((diff_z11.value+diff_z14.value) * theta_logit)))


#cp_objective_logit = cp.Minimize(30*(CE34A+CE34B))

# cp_objective_logit = cp.Maximize(cp.multiply(cp.logistic(theta_logit), theta_logit)-cp.logistic(theta_logit))

cp.entr(-cp.logistic(theta_logit)).curvature

cp.sqrt(cp.sqrt(theta_logit)).curvature

cp.logistic(cp.logistic(theta_logit)).curvature

fun_test = cp.multiply(cp.log(theta_logit),theta_logit)
fun_test = cp.log(theta_logit)*theta_logit
fun_test.curvature
cp.entr(cp.log_sum_exp(theta_logit)).curvature

cp.entr(cp.log(theta_logit)).curvature
cp.log(theta_logit).curvature

cp_objective_logit.is_dqcp()

#Problem
cp_problem_logit = cp.Problem(cp_objective_logit, constraints = None)

cp_problem_logit.solve()

import dccp

print("problem is DCCP:", dccp.is_dccp(cp_problem_logit))  # true


# =============================================================================
# Sheffis network page 329
# =============================================================================

# There are two nodes.
# The arcs can be written as follows (1,2,1),(1,2,2)
# There are trips between 1 OD pair
nOD = 2

# Columns of incidence matrix (one per route and rows are the [4] links)
nLinks = 2

# Routes between OD pairs
nRoutes = 2
# OD 1 -> 2: r1 = (a1), r2 = (a2)

route_cols = []

# OD 1 -> 4:
route_cols.append([1,0])
route_cols.append([0,1])

#Incidence Matrix
I = np.array(route_cols).T
print(I)

#Indexes from route vector for a given OD pair
ind_f = dict()
ind_f[(1,2)] = [0,1]

#OD Matrix
OD = np.zeros([nOD,nOD])
#Setting particular values

OD[1-1,2-1] = 40

# Finding the optimal solution for this simple example

# Decision variables for route flows
f = cp.Variable([nRoutes])

# Decision variables for link flows
x = cp.Variable([nLinks])

#Constraints
constraints = []
constraints += [f*I.T == x] # Arc-link flows

#Routes O-D flows
# constraints += [cp.sum(f[(ind_f[(1,2)])]) == OD[1-1,2-1]] # routes-od 1-4 flow
# constraints += [cp.sum(f) == 3000] # routes-od 1-4 flow
# constraints += [cp.sum(f[(ind_f[(1,2)])]) == OD[1-1,2-1]] # routes-od 1-4 flow
# constraints += [cp.sum(x) ==  OD[1-1,2-1]] # routes-od 1-4 flow
constraints += [cp.sum(f) == OD[1-1,2-1]] # routes-od 1-4 flow
constraints += [cp.sum(x) == OD[1-1,2-1]] # routes-od 1-4 flow

#Non negativity flows
constraints += [f >= 0]  # (i,j,k)
constraints += [x >= 0]  # (i,j,k)

theta = 0.5

# cp_objective = cp.Minimize(cp.sum(x**2) -1/theta*cp.sum(cp.entr(f)))

# Z1 = 1.25*x[0]+(1/800)**4*x[0]**5/5 + 2.5*x[1]+(1/1200)**4*x[1]**5/5
# Z1 = 1.25*x[0]+x[0]**5/5 + 2.5*x[1]+(1/1200)**4*x[1]**5/5

c1 = 1.25/5*(1/8)**4
c2 = 2.5/5*(1/12)**4

Z1 = 1.25*x[0]+c1*x[0]**5 + 2.5*x[1]+ c2*x[1]**5
# Z1 = x[0]**2+x[1]**2

cp_objective = cp.Minimize(Z1-1/theta*cp.sum(cp.entr(f)))

#Problem
cp_problem = cp.Problem(cp_objective, constraints)

#Optimize (SCS works pretty well)
objective_value = cp_problem.solve(solver='SCS',verbose = True)
# objective_value = cp_problem.solve(verbose = True)

objective_value

for variable in cp_problem.variables():
    print("Variable %s: value %s" % (variable.name(), np.round(variable.value,2)))
# value(constraints[0])

# cp_problem.constraints[0].value()

# cp.constraints+1

# =============================================================================
# Route Generation
# =============================================================================


# =============================================================================
# Generation of incidence matrix
# =============================================================================



# =============================================================================
# Optimization of network problem using SUE with logit assignment
# =============================================================================

import cvxpy as cp

cp.variable(nLinks)


# =============================================================================
# Matrix approach to optimization of network problem using SUE with logit assignment
# =============================================================================



# =============================================================================
# Non-convex problem
# =============================================================================


import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy

#Examples of minimiziatino with multiple constraints using scipy
#https://www.programcreek.com/python/example/57330/scipy.optimize.minimize

#x = xN1

xN1 = np.array([[1,2],[3,1],[4,0]])
qN1 = 100
theta_true = np.array([-1,-2]).reshape(2,1)

def logit(theta, x):
    return (np.exp(x.dot(theta))/np.sum(np.exp(x.dot(theta)))).reshape(x.shape[0])

fa = np.round(qN1*logit(theta_true, x = xN1),1)

fa.shape
logit(theta_true, x = xN1).shape

def logit_constraint(theta, x = xN1):
    return qN1 * (logit(theta=theta, x = xN1)) - fa

def objective(theta):
    #return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]
    return qN1*entropy(logit(theta, x = xN1))

logit_constraint(theta = [-1.2,-1.2])
#logit_constraint(theta = [-1.2,-1.2]).shape

cons_list = [logit_constraint]
# cons_list = [cons1]

# testing minimize
constraints = [{'type': 'ineq', 'fun': cons} for cons in cons_list]

theta_0 = [-1.2,-1.2]

#Unconstrained problem
sol = minimize(objective, theta_0, tol=1e-5)

#Constrained problem
sol = minimize(objective, theta_0, constraints=constraints, tol=1e-5)

#Solution
sol_theta = sol.x

sol_theta

objective(sol_theta)
np.round(logit(sol_theta)*qN1,1)


x0 = np.array([2, 0])
cons_list = [fun, cons1, cons2]






