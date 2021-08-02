#Clear everything from console
#globals().clear()

import numpy as np
import cvxpy as cp
import pandas as pd

# =============================================================================
# Network functions **
# =============================================================================

def BPR_function(x, k, tf, alpha, beta):
    """ BPR function that maps link flows into travel times

    :arg alpha: shape parameter
    :arg beta: shape parameter
    :arg t0: free flow travel time
    :arg k: capacity [in flow units]
    :arg x: link flow
    """

    traveltime = tf * (1 + alpha * (x / k) ** beta)

    return traveltime

def BPR_integral(x, k, tf, alpha, beta):
    """Integral of the BPR function

    :arg alpha, beta: shape parameters
    :arg t0: free flow travel time
    :arg k: capacity [in flow units]
    :arg x: link flow
    """

    integral = tf * (1 + alpha * k ** (-beta) * x ** (beta + 1) / (beta + 1))

    return integral

def SUE_Logit(q, M, D, links, bpr, Z, theta_t, theta_Z):

    """Computation of Stochastic User Equilibrium with Logit Assignment
    :arg q: demand between OD pairs
    :arg M: OD pair - link incidence matrix
    :arg D: path-link incidence matrix
    :arg links: dictionary containing the links in the network
    :arg bpr: dictionary containing the BPR parameters of each arc in order to compute travel times
    :arg Z: matrix with attributes values at each link that does not depend on the link flows
    :arg theta: parameter measuring individual preferences for different route attributes
    """
    # Dimensions
    nRoutes = D.shape[1]
    nLinks = D.shape[0]

    # Parameters
    cp_theta_t = cp.Parameter(nonpos=True) #Parameter for travel time that is dependent on link flow
    cp_theta_Z = cp.Parameter(2,nonpos=True) #Vector of parameters for parameters not dependent on link flows

    # Decision variables
    f = cp.Variable([nRoutes])
    x = cp.Variable([nLinks])

    # Constraints
    constraints = [f * D.T == x]
    constraints += [f * M.T == q]

    # Objective function

    # Component for travel time (dependent on link flow)
    bpr_integrals = [BPR_integral(x=x[links.index(link)]
                                  , k=bpr[link]['k'], tf=bpr[link]['tf']
                                  , alpha=bpr[link]['alpha'], beta=bpr[link]['beta'])
                     for link in links]

    traveltime_utility_integral = cp_theta_t * cp.sum(cp.sum(bpr_integrals))

    #Component for attributes (independent on link flow)
    Z_utility_integral = cp.sum(Z*cp_theta_Z*x)

    # cp.sum(bpr_integrals).is_dcp()
    # (-1/theta_t*cp.sum(cp.entr(f))).is_dcp()

    # Objective function for single problem (travel time)
    # cp_objective = cp.Minimize(cp.sum(cp.sum(bpr_integrals)) + -1 / theta_var * (-1) * cp.sum(cp.entr(f)))
    # cp_objective = cp.Minimize(cp.sum(cp.sum(bpr_integrals)) + 1/theta_var * cp.sum(cp.entr(f)))

    #Objective function for multiattribute problem
    utility_integral = traveltime_utility_integral + Z_utility_integral
    entropy = cp.sum(cp.entr(f))
    cp_objective = cp.Maximize(utility_integral + entropy)
    # cp_objective.is_dcp()

    # Problem
    cp_problem = cp.Problem(cp_objective, constraints)

    # cp_problem.is_dcp()

    # Solver
    cp_theta_t.value = theta_t
    cp_theta_Z.value = theta_Z # [-10, -30] #[-40, 0] #
    objective_value = cp_problem.solve()
    objective_value



    # Results

    #- Travel times
    traveltimes = [BPR_function(x=x.value[links.index(link)]
                                  , k=bpr[link]['k'], tf=bpr[link]['tf']
                                  , alpha=bpr[link]['alpha'], beta=bpr[link]['beta'])
                     for link in links]

    results = dict({'f': f.value, 'x': x.value, 't': traveltimes})



    return results, f, x, cp_theta_t.value, cp_theta_Z.value

def dictToMatrixZAttributes(Z: dict):

    listZ = []

    for i in Z.keys():
        listZ.append([float(x) for x in Z[i].values()])

    return np.asarray(listZ)

BPR_function(x=0, k=1, tf=1, alpha=1, beta=1)
BPR_integral(x=0, k=1, tf=1, alpha=1, beta=1)

def create_network_data(n_nodes, n_sample_paths: int, theta_logit: dict, attributes_thresholds: dict = None):

    nodes_G = n_nodes

    # Adjacency matrix
    A = np.random.randint(0, 2, [nodes_G, nodes_G])

    # Create networkX graph
    G = create_network(A)

    # Node attributes

    # - Coordinates - selected at random and using factor = 1000 so they are in 'metres' (number between 0 and 1000)
    G = set_random_nodes_coordinates(G, attribute_label = 'pos', factor=1000)
    nx.get_node_attributes(G, name='pos')

    # Edges attributes (distance, cost, travel time)
    G = set_random_edge_attributes(G)

    # Utility at edges
    utility_edges = compute_edge_utility(G, theta=theta_logit)
    nx.set_edge_attributes(G, utility_edges, name='utility')

    # Edge weight equals utility
    weight_edges = {key: -val for key, val in nx.get_edge_attributes(G, 'utility').items()}
    nx.set_edge_attributes(G, values=weight_edges, name='weight')

    # Simulate observed paths - by sampling from the set of all shortest path in the network
    observed_paths = path_generator(G = G, n_paths =n_sample_paths, attribute='utility')

    return G, observed_paths

# =============================================================================
# Behavioral functions
# =============================================================================

def widetolong(wide_matrix):
    """Wide to long format
    The new matrix has one rows per route
    """

    if wide_matrix.ndim == 1:
        wide_matrix = wide_matrix.reshape(1, wide_matrix.shape[0])

    long_matrix = np.repeat(wide_matrix, repeats=np.sum(wide_matrix, axis=1), axis=0)

    return long_matrix

def nozeromatrixtolist(matrix):
    list = []
    for row_index in range(matrix.shape[0]):
        row = matrix[row_index, :]
        list.append(row[row != 0])

    return list

def binaryLogit(x1, x2, theta):
    """ Binary logit model """

    sum_exp = np.exp(theta * x1) + np.exp(theta * x2)
    p1 = np.exp(theta * x1) / sum_exp
    p2 = 1 - p1

    return np.array([p1, p2])  # (P1,P2)

def softmax_probabilities(X,avail, theta):
    """ Multinomial logit (or softmax) probabilities
    :arg avail

    """
    # AVAIL

    Z = X - np.amax(X,axis = 0).reshape(X.shape[0],1)

    return avail*np.exp(Z*theta) / np.sum(np.exp(Z*theta)*avail , axis=1).reshape(Z.shape[0], 1)

def Logit_From_Path_SUE(x,f,D,M,links, Z, theta_t, theta_Z, constraints_Z = None):
    """Logit model fitted from output from SUE

    """
    # Optimization variables
    cp_theta_t = cp.Variable(1)
    cp_theta_Z = cp.Variable(theta_Z.shape[0])

    # Input
    flow_routes = f.value

    # Attributes by route

    # - Travel time
    traveltime_links = [BPR_function(x=x.value[links.index(link)]
                                     , k=bpr[link]['k'], tf=bpr[link]['tf']
                                     , alpha=bpr[link]['alpha'], beta=bpr[link]['beta'])
                        for link in links]

    traveltime_routes = traveltime_links @ D

    # - Z attributes
    Z_routes = (Z.T @ D).T

    traveltime_long = nozeromatrixtolist(widetolong(M) * traveltime_routes)
    z1_long = nozeromatrixtolist(widetolong(M) * Z_routes[:, 0])
    z2_long = nozeromatrixtolist(widetolong(M) * Z_routes[:, 1])

    #TODO: Reduce overflow log_sum_exp
    # X = traveltime_routes - np.amax(temp,axis = 1)#.reshape(len(temp),1)
    # Z = nozeromatrixtolist(widetolong(M) * traveltime_routes)- np.amax(temp,axis = 1)

    # Z1 = [traveltime_routes[i] * theta_logit - cp.log_sum_exp(traveltime_long * theta_logit) for i in
    #       range(len(temp))]  # axis = 1 is for rows

    Z1 = [traveltime_routes[i] * cp_theta_t + Z_routes[i,:]*cp_theta_Z
          - cp.log_sum_exp(traveltime_long[i] * cp_theta_t
                           + z1_long[i] * cp_theta_Z[0]+ z2_long[i] * cp_theta_Z[1]
                           ) for i in
          range(len(flow_routes))]  # axis = 1 is for rows

    Z2 = [flow_routes[i] for i in range(len(flow_routes))]  # axis = 1 is for rows

    Z3 = [Z1[i]*Z2[i] for i in range(len(flow_routes))]

    # Objective
    # cp_objective_logit = cp.Maximize(cp.sum(cp.multiply(Z1,Z2)))
    cp_objective_logit = cp.Maximize(cp.sum(Z3))
    # cp_objective_logit.is_dcp()

    #Contraints

    # Define which attributes ignore from the estimates. If constraints_Z = [0,0], both parameters are ignored from the estimation

    cp_constraints = []

    if constraints_Z[0] is not np.nan:
        cp_constraints += [cp_theta_Z[0] == constraints_Z[0]]

    if constraints_Z[1] is not np.nan:
        cp_constraints += [cp_theta_Z[1] == constraints_Z[1]]

    # if constraints_Z is not None:
    #     # cp_constraints = [cp_theta_Z == constraints_Z]
    # else:
    #     cp_constraints = None

    # Problem
    # cp_problem_logit = cp.Problem(cp_objective_logit, constraints=None)
    cp_problem_logit = cp.Problem(cp_objective_logit, constraints = cp_constraints) #Excluding extra attributes

    # cp_problem_logit.solve(verbose=True)
    cp_problem_logit.solve()

    # #Compute probabilities
    # nRoutes = len(f.value)
    # avail = widetolong(M) #Available alternatives in choice set
    # P = np.round(softmax_probabilities(X = avail*traveltime_routes, avail = avail, theta = theta_logit.value),4)

    #
    # x.value

    # Results

    # - Difference between estimated theta and

    results = dict({'theta_t_hat': cp_theta_t.value, 'theta_t':theta_t
                    ,'theta_Z_hat': cp_theta_Z.value, 'theta_Z': theta_Z
                       , 'diff_theta': np.round((cp_theta_t.value-theta_t)/theta_t,2)
                       , 'f': f.value, 'x': x.value})
                       # ,'P': P,'f': f.value, 'x': x.value})

    # print(theta_logit.value)
    # print(theta_t.value)

    return results

#=============================================================================
#Simulation functions
#=============================================================================

def SUE_Logit_Simulation_Recovery(Q, M, D, L, bpr , Z, theta_t, theta_Z,constraints_Z):

    results, f, x, theta_t, theta_Z = SUE_Logit(q=Q, M=M, D=D, links=L, bpr=bpr
                                                     , Z=Z, theta_t=theta_t, theta_Z=theta_Z)

    results_Logit = Logit_From_Path_SUE(x=x, f=f, M=M, D=D, links= L
                                    , Z=Z, theta_t=theta_t, theta_Z=theta_Z
                                    , constraints_Z= constraints_Z
                                    )

    return results_Logit

# =============================================================================
# Network parameters
# =============================================================================

# A) NETWORK STRUCTURE

# M: Incidence OD pair-path matrix
# D: Incidence link-path matrix

# Network 1
M1 = np.array([1, 1, 1])
D1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
Q1 = np.array([10])  # 1-2
L1 = ['a1', 'a2', 'a3']

# Network 2
M2 = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]])
D2 = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
Q2 = np.array([10, 20, 30])  # 1-4, 2-4, 3-4
L2 = ['a1', 'a2', 'a3', 'a4']

# Network 3
M3 = np.array([[1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1]])
D3 = np.array([[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0],
               [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]]
              )
Q3 = np.array([10, 20])  # 1-3, 2-3
L3 = ['a1', 'a2', 'a3', 'a4', 'a5']

print(M1); print(M2); print(M3)  # [print(i) for i in [M1,M2,M3]]

# B) LINK ATTRIBUTES

# i) Travel time

# BPR parameters for 5 arcs
bpr = dict()
bpr['a1'] = {'alpha': 3E-0, 'beta': 1, 'tf': 10E-2, 'k': 1E-0}
bpr['a2'] = {'alpha': 2E-0, 'beta': 1, 'tf': 40E-2, 'k': 1E-0}
bpr['a3'] = {'alpha': 5E-0, 'beta': 1, 'tf': 33E-2, 'k': 1E-0}
bpr['a4'] = {'alpha': 5E-0, 'beta': 2, 'tf': 6E-2, 'k': 1E-0}
bpr['a5'] = {'alpha': 2E-0, 'beta': 2, 'tf': 10E-2, 'k': 1E-0}

bpr1 = dict((k, bpr[k]) for k in L1)
bpr2 = dict((k, bpr[k]) for k in L2)
bpr3 = dict((k, bpr[k]) for k in L3)

# ii) Waiting time (minutes)
# waitingtime = np.array([1,2,3,4,5]).reshape(1,5)
waitingtime = dict({'a1': 30, 'a2': 20,'a3': 15, 'a4': 10, 'a5': 5 })

# iii) Cost in USD (e.g. toll)
# cost = np.array([10,10,0,40,50]).reshape(1,5)
cost = dict({'a1': 1, 'a2': 2,'a3': 1.5, 'a4': 1, 'a5': 3})

# Common dictionary for additional attributes
Z_lb = ['z1','z2'] #Labels
Z = dict({Z_lb[0]: waitingtime, Z_lb[1]: cost})
# Z = np.vstack((waitingtime,cost))

#Matrix of additional attributes for each network
Z1 = dictToMatrixZAttributes(Z = Z)[:,[L1.index(i) for i in L1]].T
Z2 = dictToMatrixZAttributes(Z = Z)[:,[L2.index(i) for i in L2]].T
Z3 = dictToMatrixZAttributes(Z = Z)[:,[L3.index(i) for i in L3]].T

# =============================================================================
# Behavioral parameters
# =============================================================================
theta_t = -0.1

theta_w = -0.3
theta_c = -0.2

theta_Z = [theta_w,theta_c]
vot = 60*theta_t/theta_c

# =============================================================================
# Network optimization
# =============================================================================

# Network 1
results1, f1, x1, theta_t1, theta_Z1 = SUE_Logit(q = Q1, M = M1, D = D1, links = L1, bpr = bpr1, Z = Z1, theta_t = theta_t, theta_Z = theta_Z)
print(results1)

# Metwork 2
results2, f2, x2, theta_t2, theta_Z2 = SUE_Logit(q = Q2, M = M2, D = D2, links = L2, bpr = bpr2, Z = Z2, theta_t = theta_t, theta_Z = theta_Z)
print(results2)

# Network 3
results3, f3, x3, theta_t3, theta_Z3 = SUE_Logit(q = Q3, M = M3, D = D3, links = L3, bpr = bpr3, Z = Z3, theta_t = theta_t, theta_Z = theta_Z)
print(results3)

# =============================================================================
# Behavioral fitting
# =============================================================================

# Network 1

# constraints_Z1 = theta_Z1
constraints_Z1 = [np.nan,theta_Z1[1]]
constraints_Z1 = [np.nan,np.nan]

results_Logit1 = Logit_From_Path_SUE(x = x1, f = f1, M = M1, D = D1, links = L1
                                , Z = Z1, theta_t = theta_t1, theta_Z = theta_Z1
                                , constraints_Z = constraints_Z1
                                # ,constraints_Z = None
                                )
print(results_Logit1)

# x = x1; f = f1; M = M1; D = D1; links = L1; Z = Z1; theta_t = theta_t1; theta_Z = theta_Z1; constraints_Z = constraints_Z1

# x = x1; f = f1; M = M1; D = D1; links = L1; Z = Z1; theta_t = theta_t1; theta_Z = theta_Z1

# Network 2
constraints_Z2 = [np.nan,theta_Z2[1]]
constraints_Z2 = [np.nan,np.nan]

results_Logit2 = Logit_From_Path_SUE(x = x2, f = f2, M = M2, D = D2, links = L2
                                , Z = Z2, theta_t = theta_t2, theta_Z = theta_Z2
                                , constraints_Z = constraints_Z2
                                )
print(results_Logit2)

# x = x2; f = f2; M = M2; D = D2; links = L2; Z = Z2; theta_t = theta_t2; theta_Z = theta_Z2; constraints_Z = constraints_Z2

# Network 3

# constraints_Z3 = [np.nan,theta_Z3[1]]

constraints_Z3 = [np.nan,np.nan]

results_Logit3 = Logit_From_Path_SUE(x = x3, f = f3, M = M3, D = D3, links = L3
                                , Z = Z3, theta_t = theta_t3, theta_Z = theta_Z3
                                ,constraints_Z = constraints_Z3
                                )
results_Logit3

attributes_toprint = ['theta_t_hat','diff_theta']

print([results_Logit1.get(k) for k in attributes_toprint])
print([results_Logit2.get(k) for k in attributes_toprint])
print([results_Logit3.get(k) for k in attributes_toprint])



# =============================================================================
# Tables
# =============================================================================



# =============================================================================
# Plots
# =============================================================================

#A) Functions:

import matplotlib.pyplot as plt

def PlotEstimatedVSRealTheta(maintitle, theta_t_range, Q, M, D, L, bpr , Z, theta_Z,constraints_Z):

    theta_t_list = []
    theta_t_hat_list = []
    theta_Z_hat_list = []

    for theta_ti in theta_t_range:

        resultsN = SUE_Logit_Simulation_Recovery(Q=Q, M=M, D=D, L=L, bpr=bpr
                                                  , Z=Z, theta_t=theta_ti, theta_Z=theta_Z,
                                                  constraints_Z=constraints_Z)

        theta_t_list.append(resultsN['theta_t'])
        theta_Z_hat_list.append(resultsN['theta_Z_hat'])
        theta_t_hat_list.append(resultsN['theta_t_hat'])

    learned_parameter = np.array(theta_t_hat_list)
    true_parameter = np.array(theta_t_list)

    fig, ax = plt.subplots()
    ax.scatter(learned_parameter, true_parameter)
    ax.plot([true_parameter.min(), true_parameter.max()], [true_parameter.min(), true_parameter.max()], 'k--', lw=4)
    ax.set_xlabel('True theta')
    ax.set_ylabel('Learned theta')
    ax.set_title(maintitle + '\n Travel time parameter')
    plt.show()

    if theta_Z[1] != 0:
        #Value of time
        learned_parameter = 60*np.array(theta_t_hat_list)[:,0]/np.array(theta_Z_hat_list)[:,1]
        true_parameter = 60*np.array(theta_t_list)/theta_Z[1] #Multiply by 60 to convert to USD per hour

        fig, ax = plt.subplots()
        ax.scatter(learned_parameter, true_parameter)
        ax.plot([true_parameter.min(), true_parameter.max()], [true_parameter.min(), true_parameter.max()], 'k--', lw=4)
        ax.set_xlabel('True value of time')
        ax.set_ylabel('Learned value of time')
        ax.set_title(maintitle + '\n Value of Time')
        plt.show()

def PlotFlowsVSTheta(maintitle, theta_t_range, Q, M, D, L, bpr , Z, theta_Z):

    theta_t_list = []
    x_list = []
    f_list = []

    for theta_ti in theta_t_range:
        # Network
        theta = [0, 0]
        constraints = [0, 0]

        results, f, x, theta_t, theta_Z = SUE_Logit(q=Q, M=M, D=D, links=L, bpr=bpr
                                                    , Z=Z, theta_t=theta_ti, theta_Z=theta_Z)

        theta_t_list.append(np.round(theta_ti,4))
        x_list.append(results['x'])
        f_list.append(results['f'])

    theta_t = np.array(theta_t_list)
    x_N = np.array(x_list)
    f_N =  np.array(f_list)


    #Standard deviation of flow assignment
    sdX_vs_theta_N = [np.sqrt(np.var(row)) for row in x_N]
    sdF_vs_theta_N = [np.sqrt(np.var(row)) for row in f_N]

    #Plot
    # plt.plot(sdX_vs_theta_N, label='link flows')
    plt.plot(sdF_vs_theta_N, label='route flows')
    plt.title(maintitle + '\n Standard deviation of route flows')
    plt.show()

    # var_vs_theta_N2 = [np.sqrt(np.var(row)) for row in x_N2]
    # var_vs_theta_N3 = [np.sqrt(np.var(row)) for row in x_N3]

    # return np.array(theta_t_list),  np.array(x_list),  np.array(f_list)

    # predicted = np.array(theta_t_hat_list)
    # y = np.array(theta_t_list)
    #
    # fig, ax = plt.subplots()
    # ax.scatter(y, predicted)
    # ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # plt.show()

#B) Single attribute case:

# i) Estimate of theta versus real theta

# theta_t_range = theta_t * np.arange(1, 30,1)

theta_t_range = theta_t * np.arange(1, 5,0.1)

PlotEstimatedVSRealTheta(maintitle = 'Single attribute route choice. Network 1',
                         theta_t_range = theta_t_range, Q = Q1, M = M1, D = D1, L = L1, bpr = bpr1 , Z = Z1, theta_Z = [0,0], constraints_Z = [0,0])

PlotEstimatedVSRealTheta(maintitle = 'Single attribute route choice. Network 2',
    theta_t_range = theta_t_range, Q = Q2, M = M2, D = D2, L = L2, bpr = bpr2 , Z = Z2, theta_Z = [0,0], constraints_Z = [0,0])

PlotEstimatedVSRealTheta(maintitle = 'Single attribute route choice. Network 3',
    theta_t_range = theta_t_range, Q = Q3, M = M3, D = D3, L = L3, bpr = bpr3 , Z = Z3, theta_Z = [0,0], constraints_Z = [0,0])

# ii) Variance of optimal flows versus theta (as higher is theta, flow are more dispersed)

PlotFlowsVSTheta(maintitle = 'Single attribute route choice. Network 1',
                 theta_t_range = theta_t_range, Q = Q1, M = M1, D = D1, L = L1, bpr = bpr1 , Z = Z1, theta_Z = [0,0])

PlotFlowsVSTheta(maintitle = 'Single attribute route choice. Network 2',
                 theta_t_range = theta_t_range, Q = Q2, M = M2, D = D2, L = L2, bpr = bpr2 , Z = Z2, theta_Z = [0,0])

PlotFlowsVSTheta(maintitle = 'Single attribute route choice. Network 3',
                 theta_t_range = theta_t_range, Q = Q3, M = M3, D = D3, L = L3, bpr = bpr3 , Z = Z3, theta_Z = [0,0])

# iii) Sensitivity respect to beta of BPR function


#B) Two attributes case:

# i) Estimate of theta versus real theta
theta_t_range = theta_t * np.arange(1, 5,0.1)

constraints_Z1 = [theta_Z1[1],np.nan]
# constraints_Z1 = [np.nan,np.nan]
PlotEstimatedVSRealTheta(maintitle = 'Two attribute route choice. Network 1',
    theta_t_range = theta_t_range, Q = Q1, M = M1, D = D1, L = L1, bpr = bpr1 , Z = Z1, theta_Z = theta_Z1, constraints_Z = constraints_Z1)
# theta_t_range = theta_t_range; Q = Q1; M = M1; D = D1; L = L1; bpr = bpr1 ; Z = Z1; theta_Z = theta_Z1; constraints_Z = constraints_Z1

constraints_Z2 = [theta_Z2[1],np.nan]
PlotEstimatedVSRealTheta(maintitle = 'Two attribute route choice. Network 2',
    theta_t_range = theta_t_range, Q = Q2, M = M2, D = D2, L = L2, bpr = bpr2 , Z = Z2, theta_Z = theta_Z2, constraints_Z = constraints_Z2)
# theta_t_range = theta_t_range; Q = Q2; M = M2; D = D2; L = L2; bpr = bpr2 ; Z = Z2; theta_Z = theta_Z2; constraints_Z = constraints_Z2

constraints_Z3 = [theta_Z3[1],np.nan]
PlotEstimatedVSRealTheta(maintitle = 'Two attribute route choice. Network 3',
    theta_t_range = theta_t_range, Q = Q3, M = M3, D = D3, L = L3, bpr = bpr3 , Z = Z3, theta_Z = theta_Z3, constraints_Z = constraints_Z3)
# theta_t_range = theta_t_range; Q = Q3; M = M3; D = D3; L = L3; bpr = bpr3 ; Z = Z3; theta_Z = theta_Z3; constraints_Z = constraints_Z3
#c) Three-attribute case:

# i) Estimate of theta versus real theta

theta_t_range = theta_t * np.arange(1, 5,0.1)

constraints_Z1 = [np.nan,np.nan]
PlotEstimatedVSRealTheta(maintitle = 'Three attribute route choice. Network 1',
    theta_t_range = theta_t_range, Q = Q1, M = M1, D = D1, L = L1, bpr = bpr1 , Z = Z1, theta_Z = theta_Z1, constraints_Z = constraints_Z1)

constraints_Z2 = [np.nan,np.nan]
PlotEstimatedVSRealTheta(maintitle = 'Three attribute route choice. Network 2',
    theta_t_range = theta_t_range, Q = Q2, M = M2, D = D2, L = L2, bpr = bpr2 , Z = Z2, theta_Z = theta_Z2, constraints_Z = constraints_Z2)

constraints_Z3 = [np.nan,np.nan]
PlotEstimatedVSRealTheta(maintitle = 'Three attribute route choice. Network 1',
    theta_t_range = theta_t_range, Q = Q3, M = M3, D = D3, L = L3, bpr = bpr3 , Z = Z3, theta_Z = theta_Z3, constraints_Z = constraints_Z3)


#Multiattribute decisions (bias in theta is only a single attribute is considered)


# =============================================================================
# Constraint satisfaction problem
# =============================================================================

# If only link level data is available, the preference parameters can be found by using the link-path satistisfaction
# constraints when replacing by the path flows with the logit probabilities.







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




