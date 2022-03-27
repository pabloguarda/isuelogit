import numpy as np
import factory

def multiday_network(N, n_days, label,remove_zeros_Q, q_range, R_labels, cutoff_paths, od_paths, Z_attrs_classes, bpr_classes, fixed_effects, randomness):

    N_multiday = {}

    for day in range(0,n_days):
        N_multiday[day] = modeller.setup_networks(N={day:N}, label= label, R_labels=R_labels
                                         , randomness= randomness
                                         , q_range= q_range
                                         , remove_zeros_Q=remove_zeros_Q
                                         , Z_attrs_classes=Z_attrs_classes
                                         , bpr_classes=bpr_classes, cutoff_paths=cutoff_paths, n_paths= od_paths
                                         , fixed_effects = fixed_effects).get(day)


    return N_multiday

 def sue_logit_fisk(self,
                       D: Matrix,
                       M: Matrix,
                       q: ColumnVector,
                       links: Links,
                       paths: Paths,
                       theta: Parameters,
                       Z_dict: {},
                       k_Z: LogitFeatures,
                       k_Y: LogitFeatures = ['tt'],
                       network=None,
                       cp_solver='ECOS',
                       feastol=1e-24) -> Dict[str, Dict]:
        """Computation of Stochastic User Equilibrium with Logit Assignment
        :arg q: Long vector of demand obtained from demand matrix between OD pairs
        :arg M: OD pair - link incidence matrix
        :arg D: path-link incidence matrix
        :arg links: dictionary containing the links in the network
        :arg Z: matrix with exogenous attributes values at each link that does not depend on the link flows
        :arg theta: vector with parameters measuring individual preferences for different route attributes
        :arg features: subset of attributes from X chosen to perform assignment
        """

        # i = 'SiouxFalls'

        # q = tai.network.denseQ(Q=N['train'][i].Q, remove_zeros=remove_zeros_Q); M = N['train'][i].M; D = N['train'][i].D; links = N['train'][i].links_dict; paths = N['train'][i].paths; Z_dict = N['train'][i].Z_dict; features = []; theta = theta_true[i]; cp_solver = 'ECOS'; feastol = 1e-24; features_Y = ['tt']

        if network is None:
            network = self.network

        # Subset of attributes
        if len(k_Z) == 0:
            k_Z = [k for k in theta.keys() if k not in k_Y]

        # #Solving issue with negative sign of parameters
        # for k in features:
        #     if theta[k] > 0:
        #         theta[k] = -theta[k]
        #         Z_dict[k] = dict(zip(Z_dict[k].keys(), list(-1 * np.array(list(Z_dict[k].values())))))

        # Get Z matrix from Z dict
        Z_subdict = {k: Z_dict.get(k) for k in k_Z}
        Z = get_matrix_from_dict_attrs_values(W_dict=Z_subdict)

        # Decision variables
        # cp_f = {i:cp.Variable(nonneg=True) for i in range(D.shape[1])} # cp.Variable([nRoutes]) #TODO: Transform this in dictionary as with x
        # cp_x ={i:cp.Variable() for i, l_i in links.items()}  # cp.Variable([nLinks])

        cp_f = cp.Variable(D.shape[1], nonneg=True)
        # cp_f.value = np.ones(D.shape[1])*10000

        # cp_f.value[0] = q[0]
        # cp_f.value[2] = q[1]
        # cp_f.value[4] = q[2]
        # cp_f.value[0] = q[0:1]
        # np.sum(M[0:3,:]*cp_f.value,axis=1)
        # q[0:3]

        # # Set equality for link flows instead of additional constraints
        # cp_x = dict(zip(list(links.keys()), list(D * cp.hstack(list(cp_f.values())))))
        cp_x = D * cp_f

        # q

        # np.sum(cp_f.value)

        # np.sum(M[0:3, :],axis = 1)

        # Constraints (* is equivalent to numpy dot product)
        cp_constraints = []
        # cp_constraints += [M*cp.hstack(list(cp_f.values())) == q]
        cp_constraints += [M * cp.vstack(cp_f) == np.vstack(q)]
        # cp_constraints += [M * cp.vstack(cp_f) >= np.vstack(q)]
        # np.sum(M,axis = 0)

        # cp_constraints += [cp.sum(M[0:3,:] * cp.hstack(cp_f), axis=1) == q[0:3]]
        # cp_constraints += [M[0:3, :] * cp_f == q[0:3]]
        # cp_constraints += [M[0:1, :] * cp_f == q[0:1]]
        # cp_constraints += [D*cp.hstack(list(cp_f.values())) == cp.hstack(list(cp_x.values()))] #This constraint might be replaced directly into other constraints and objective function
        #
        # warm_start = np.zeros(D.shape[1])
        # warm_start[0] =
        # A = M[0:3, :]

        # type(q[0])

        # Check constraints
        # cp_constraints[0].violation()

        # [0].value

        # Parameters for cvxpy
        cp_theta = {}

        # Parameters that are dependent on link flow (only 'tt' so far)
        cp_theta['Y'] = {'tt': cp.Parameter(nonpos=True)}  # cp.Parameter(nonpos=True)
        # cp_theta['Y'] = {'tt': cp.Variable(pos=False)}  # cp.Parameter(nonpos=True)

        # Parameters for attributes not dependent on link flow (all except for 'tt't)
        cp_theta['Z'] = {k: cp.Parameter(nonpos=True) for k in k_Z if
                         k != 'tt'}  # nonpos=True is required to find unique equilibrium

        # Objective function

        # Component for endogeonous attributes dependent on link flow
        # bpr_integrals = [link.bpr.bpr_integral_x(x=cp_x[link.label]) for i,link in links.items()]
        bpr_integrals = [link.bpr.bpr_integral_x(x=cp_x[i]) for i, link in zip(range(0, len(list(links))), links.values())]
        tt_utility_integral = cp_theta['Y']['tt'] * cp.sum(cp.sum(bpr_integrals))

        # Component for attributes (independent on link flow)
        # Z_utility_integral = cp.sum(cp.multiply(Z*cp.hstack(list(cp_theta['Z'].values())),cp.hstack(list(cp_x.values()))))
        Z_utility_integral = cp.sum(
            cp.multiply(Z * cp.hstack(list(cp_theta['Z'].values())), cp.hstack(cp_x)))

        # Objective function for multiattribute problem
        utility_integral = tt_utility_integral + Z_utility_integral

        # entropy = cp.sum(cp.entr(cp.hstack(list(cp_f.values()))))
        entropy = cp.sum(cp.entr(cp_f))

        simple_obj = cp.Parameter()
        cp_objective = cp.Maximize(utility_integral + entropy)
        # cp_objective = cp.Maximize(entropy)

        # Problem
        # cp_problem = cp.Problem(cp_objective)
        cp_problem = cp.Problem(cp_objective, cp_constraints)

        # cp_problem.is_dcp()

        # Assign parameters values in objective function
        cp_theta['Y']['tt'].value = theta['tt']
        for k in k_Z:
            cp_theta['Z'][k].value = theta[k]

        # feastol = 100
        # Solve

        objective_value = None

        try:
            # objective_value = cp_problem.solve(solver=cp.ECOS, feastol=feastol)
            # cp_theta['Y']['tt'] = 0 #no congestion
            objective_value = cp_problem.solve(solver=cp.ECOS)

        except:  # SCS failed less often because of numerical problems
            objective_value = cp_problem.solve(solver=cp.SCS)
            # cp_theta['Y']['tt'] = -10
            # objective_value = cp_problem.solve(solver=cp.SCS, verbose = True, warm_start = True)

        # if cp_solver == 'ECOS':
        #     # objective_value = cp_problem.solve(solver = cp.ECOS, feastol = feastol)
        #     objective_value = cp_problem.solve(solver=cp.ECOS, feastol=feastol, verbose = True)
        #     # objective_value = cp_problem.solve(verbose=True)
        #
        # else:
        #     objective_value = cp_problem.solve(solver=cp_solver, feastol=feastol,
        #                                        verbose=True)  # #(solver = solver) #'ECOS' solver crashes with some problems

        # Results

        tt = {}
        # - Travel time by link
        # tt['x'] = {i:link.bpr.bpr_function_x(x=cp_x[link.label].value) for i,link in links.items()}
        tt['x'] = {j: link.bpr.bpr_function_x(x=cp_x.value[i]) for i, j, link in
                   zip(range(0, len(list(links))), links.keys(), links.values())}

        # Link flows
        # x = {k:v.value for k,v in cp_x.values()}
        x = {k: v for k, v in zip(links.keys(), cp_x.value)}

        # cp_x.value

        # np.sum(np.array(list(x.values())))
        # np.sum(M* np.hstack(cp_f.value),axis = 0) == q[0]
        # q[-1]

        # [cons.violation() for cons in cp_constraints]

        # cp_constraints[0].violation()

        # Path flows
        f = {k: v for k, v in zip(range(len(cp_f.value)), cp_f.value)}
        # f = {k: v.value for k, v in cp_f.items()}

        # np.sum(np.array(list(cp_f.value)))
        #
        # np.sum(cp_f.value)
        # np.sum(q)

        # np.sum(M[0, :] * np.hstack(np.array(list(f.values())))) == q[0]

        # np.sum(list(x.values()))

        # # Todo: Flow by route. Require class path
        # f = {k:v.value for k,v in cp_x.items()}

        # results = dict({'f': cp_f.value, 'x': x, 'tt': tt})

        return {'x': x, 'f': f, 'tt_x': tt['x']}

def multiday_estimation(N, end_params, theta0, q0, theta_true, remove_zeros_Q, n_days, randomness_multiday, k_Y, k_Z,
                        R_labels, Z_attrs_classes, bpr_classes, cutoff_paths, od_paths, fixed_effects, q_range,
                        N_multiday_old=None):
    N_multiday = {}
    N_multiday_new = {}
    results_multiday = {}
    # network_label = 'N1'
    for network_label in N.keys():

        results_multiday[network_label] = {}

        N_multiday_new[network_label] = networks.multiday_network(N=N[network_label], n_days=n_days, label=network_label
                                                                  , R_labels=R_labels
                                                                  , randomness=randomness_multiday
                                                                  , q_range=q_range, remove_zeros_Q=remove_zeros_Q
                                                                  , Z_attrs_classes=Z_attrs_classes,
                                                                  bpr_classes=bpr_classes
                                                                  , cutoff_paths=cutoff_paths, od_paths=od_paths
                                                                  , fixed_effects=fixed_effects)

        results_sue_multiday = {}

        valid_network = None

        # N_i = N_multiday_new[network_label][0]
        while valid_network is None:
            try:
                results_sue_multiday[network_label] = {i: equilibrium.sue_logit_fisk(
                    q=networks.denseQ(Q=N_i.Q, remove_zeros=remove_zeros_Q)
                    , M=N_i.M, D=N_i.D
                    , links=N_i.links_dict
                    , paths=N_i.paths
                    , Z_dict=N_i.Z_dict
                    , k_Z=[]
                    , theta=theta_true
                    , cp_solver='ECOS'  # 'ECOS': it is faster and crashes sometimes. 'SCS' is slow
                )
                    for i, N_i in N_multiday_new[network_label].items()}

            except:
                print('error' + str(network_label))
                # for i in N['validation'].keys():
                # exceptions['SUE']['validation'][i] += 1

                N_multiday_new[network_label] = networks.multiday_network(N=N[network_label], n_days=n_days,
                                                                          label=network_label
                                                                          , R_labels=R_labels
                                                                          , randomness=randomness_multiday
                                                                          , q_range=q_range,
                                                                          remove_zeros_Q=remove_zeros_Q
                                                                          , Z_attrs_classes=Z_attrs_classes,
                                                                          bpr_classes=bpr_classes
                                                                          , cutoff_paths=cutoff_paths,
                                                                          od_paths=od_paths,
                                                                          fixed_effects=fixed_effects)

                pass
            else:
                valid_network = True
                # Store travel time, link and path flows in Network objects
                for i, N_i in N_multiday_new[network_label].items():
                    N_i.set_Y_attr_links(y=results_sue_multiday[network_label][i]['tt_x'], feature='tt')
                    N_i.x_dict = results_sue_multiday[network_label][i]['x']
                    N_i.f_dict = results_sue_multiday[network_label][i]['f']

        if N_multiday_old[network_label] is not None:
            N_multiday_list = [*list(N_multiday_old[network_label].values()),
                               *list(N_multiday_new[network_label].values())]
            N_multiday[network_label] = dict(zip(np.arange(0, len(N_multiday_list)), N_multiday_list, ))

        else:
            N_multiday = N_multiday_new
        theta_estimates_multiday = {}
        results_logit_sue_links = {}

        range_theta = range(0, len(theta0))
        range_params_q = range(len(theta0), len(theta0) + len(q0[network_label]))

        if end_params['q'] and not end_params['theta']:
            range_theta = range(0, 0)
            range_params_q = range(0, len(q0[network_label]))

        if end_params['theta'] and not end_params['q']:
            range_params_theta = range(0, len(theta0))
            range_params_q = range(0, 0)

        # Constraints
        constraints_q0 = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[range_params_q]) - np.sum(
                networks.denseQ(Q=N_multiday[network_label][0].Q, remove_zeros=remove_zeros_Q))}
            , {'type': 'ineq', 'fun': lambda x: x[range_params_q]}
        ]
        # constraints_q0  = [{'type':'ineq', 'fun': lambda x: x[range_q]}]
        # constraints_q0=[]

        results_logit_sue_links[network_label] = tai.estimation.solve_link_level_model(end_params=end_params,
                                                                                   Mt={i: N_i.M for i, N_i
                                                                                       in N_multiday[
                                                                                           network_label].items()},
                                                                                   Ct={
                                                                                       i: N_i.generate_C(
                                                                                           N_i.M) for i, N_i
                                                                                       in N_multiday[
                                                                                           network_label].items()},
                                                                                   Dt={i: N_i.D for i, N_i
                                                                                       in N_multiday[
                                                                                           network_label].items()},
                                                                                   k_Y=k_Y,
                                                                                   Yt={i: N_i.Y_dict for
                                                                                       i, N_i in N_multiday[
                                                                                           network_label].items()},
                                                                                   k_Z=k_Z,
                                                                                   Zt={i: N_i.Z_dict for
                                                                                       i, N_i in N_multiday[
                                                                                           network_label].items()},
                                                                                   xt={i: N_i.x for i, N_i
                                                                                       in N_multiday[
                                                                                           network_label].items()},
                                                                                   idx_links={
                                                                                       i: range(len(N_i.x))
                                                                                       for i, N_i in
                                                                                       N_multiday[
                                                                                           network_label].items()},
                                                                                   scale={'mean': False,
                                                                                          'std': False},
                                                                                   q0=q0[network_label],
                                                                                   theta0=theta0,
                                                                                   lambda_hp=0,
                                                                                   constraints_q0=constraints_q0,
                                                                                   norm_o=2, norm_r=1)
        if end_params['q']:
            results_multiday[network_label]['q'] = np.round(results_logit_sue_links[network_label]['q'], 1)

        if end_params['theta']:
            results_multiday[network_label]['theta'] = results_logit_sue_links[network_label]['theta']

            if results_logit_sue_links[network_label]['theta']['c'] == 0:
                results_multiday[network_label]['vot'] = np.nan
            else:
                results_multiday[network_label]['vot'] = results_logit_sue_links[network_label]['theta']['tt'] / \
                                                         results_logit_sue_links[network_label]['theta']['c']

        # print(np.round(results_logit_sue_links[network_label]['q'],1))
        # print(tai.network.denseQ(N_multiday[network_label][0].Q, remove_zeros=remove_zeros_Q))

        # print(results_logit_sue_links[network_label]['theta'])
        # print(theta_true)

        # print(results_logit_sue_links[network_label]['theta']['tt']/results_logit_sue_links[network_label]['theta']['c'])
        # print(theta_true['tt']/theta_true['c'])

    return results_multiday, N_multiday


def hessian_check(theta):
    # Autograd https://rlhick.people.wm.edu/posts/mle-autograd.html
    # http: // www.cs.toronto.edu / ~rgrosse / courses / csc321_2017 / tutorials / tut4.pdf

    # Source: https://towardsdatascience.com/debugging-your-neural-nets-and-checking-your-gradients-f4d7f55da167
    # theta = np.array(list(theta0.values()))

    # Numeric diff
    # https: // v8doc.sas.com / sashtml / ormp / chap5 / sect28.htm

    # grads_theta = []
    #
    # epsilon = 1e-7
    #
    # for i in np.arange(theta.shape[0]):
    #     epsilon_v = np.zeros(len(theta))[:, np.newaxis]
    #     epsilon_v[i] = epsilon
    #
    #     fun = lambda theta_x: gradient_objective_function(theta=theta_x, Ix=Ix, YZ_x=YZ_x)
    #
    #     grad_fun = nd.Derivative(fun)
    #     grad_fun_theta = grad_fun(theta)
    #     grad_fun_theta/np.linalg.norm(grad_fun_theta)
    #
    #     hessian_fun = nd.Derivative(fun,2)
    #     hessian_fun_theta = hessian_fun(theta)
    #     hessian_fun_theta / np.linalg.norm(hessian_fun_theta)
    #
    #     b = hessian_l2norm(theta, YZ_x, Ix)
    #     b/ np.linalg.norm(b)
    #
    #     J, pf =jacobian_response_function(theta, YZ_x, Ix)
    #     c = J.T.dot(J)
    #
    #     np.diag(c)
    #
    #     np.diag(c)/np.linalg.norm(np.diag(c))
    #
    #     b = numeric_gradient_l2norm(theta)
    #
    #     numeric_gradient_l2norm(theta)/np.linalg.norm(b)
    #
    #     epsilon_v = np.zeros(len(theta))[:, np.newaxis]
    #     epsilon_v[i] = epsilon
    #
    #     H = np.mean(objective_function(theta=theta - epsilon_v, Ix=Ix, YZ_x=YZ_x))
    #     F = np.mean(objective_function(theta=theta + epsilon_v, Ix=Ix, YZ_x=YZ_x))
    #
    #     grads_theta.append((F - H) / (2 * epsilon))
    #
    #
    #     gradient_check(theta)
    #
    #
    #     H = np.mean(gradient_objective_function(theta=theta - epsilon_v, Ix=Ix, YZ_x=YZ_x))
    #     F = np.mean(gradient_objective_function(theta=theta + epsilon_v, Ix=Ix, YZ_x=YZ_x))
    #
    #     grads_theta.append((F - H) / (2 * epsilon))
    # #
    # # print(np.array(grads_theta))
    # #
    # # print(gradient_objective_function(theta, YZ_x, Ix))
    #
    # gap = np.linalg.norm(np.array(grads_theta)[:, np.newaxis] - hessian_l2norm(theta, YZ_x, Ix))
    #
    #
    # gap = np.linalg.norm(np.array(H)[:,np.newaxis]-hessian_l2norm(theta, YZ_x, Ix))

    raise NotImplementedError

def sue_logit_OD_estimation(Nt, D, M, q_obs, tt_obs, links: Links, paths: Paths, theta: Parameters, Z_dict: {}, x_obs: np.array, k_Z=[], cp_solver='ECOS') -> Dict[str, Dict]:
    """Computation of Stochastic User Equilibrium with Logit Assignment
    :arg q: Long vector of demand obtained from demand matrix between OD pairs
    :arg M: OD pair - link incidence matrix
    :arg D: path-link incidence matrix
    :arg links: dictionary containing the links in the network
    :arg Z: matrix with exogenous attributes values at each link that does not depend on the link flows
    :arg theta: vector with parameters measuring individual preferences for different route attributes
    :arg features: subset of attributes from X chosen to perform assignment
    """
    # i = 'N6'
    # q_obs = tai.network.denseQ(Q = N['train'][i].Q, remove_zeros = remove_zeros_Q)
    # x_obs = np.array(list(results_sue['train'][i]['x'].values()))
    # tt_obs = np.array(list(results_sue['train'][i]['tt_x'].values()))
    # np.sum(x_obs)
    # np.sum(q_obs)
    # D, M, links, paths, Z_dict, theta, cp_solver, K_Z = N['train'][i].D,N['train'][i].M, N['train'][i].links_dict,N['train'][i].paths, N['train'][i].Z_dict,theta_true,'ECOS', []

    # Subset of attributes
    if len(k_Z) == 0:
        k_Z = [k for k in theta.keys() if k != 'tt']

    # Get Z matrix from Z dict
    Z_subdict = {k: Z_dict.get(k) for k in k_Z}
    Z = get_matrix_from_dict_attrs_values(W_dict=Z_subdict)

    # Decision variables
    cp_x = {i: cp.Variable() for i, l_i in links.items()}  # cp.Variable([nLinks])
    cp_f = {i: cp.Variable() for i in
            range(D.shape[1])}  # cp.Variable([nRoutes]) #TODO: Transform this in dictionary as with x
    # q  values
    cp_q = {i: cp.Variable(pos=True) for i in range(q_obs.shape[0])}  # cp.Variable(

    # Constraints
    cp_constraints = [M * cp.hstack(list(cp_f.values())) == cp.hstack(list(cp_q.values()))]
    # cp_constraints = [M * cp.hstack(list(cp_f.values())) == q]
    cp_constraints += [D * cp.hstack(list(cp_f.values())) == cp.hstack(list(cp_x.values()))]
    cp_constraints += [cp.sum(cp.hstack(list(cp_q.values()))) == np.sum(q_obs)]

    # Parameters for cvxpy
    cp_theta = {}
    cp_scale = cp.Variable(nonpos=False)
    # Parameters that are dependent on link flow (only 'tt' so far)
    cp_theta['Y'] = {'tt': cp.Parameter(nonpos=True)}  # cp.Parameter(nonpos=True)
    # cp_theta['Y'] = {'tt': cp.Variable(pos=False)}  # cp.Parameter(nonpos=True)

    # Parameters for attributes not dependent on link flow (all except for 'tt't)
    cp_theta['Z'] = {k: cp.Parameter(nonpos=True) for k in k_Z if
                     k != 'tt'}  # nonpos=True is required to find unique equilibrium
    # cp_theta['Z']['c'] = cp.Variable(nonpos=True)

    # Objective function

    # Component for endogeonous attributes dependent on link flow
    bpr_integrals = [link.bpr.bpr_integral_x(x=cp_x[link.key]) for i, link in links.items()]
    tt_utility_integral = cp_theta['Y']['tt'] * cp.sum(cp.sum(bpr_integrals))

    # Component for attributes (independent on link flow)
    Z_utility_integral = cp.sum(
        cp.multiply(Z * cp.hstack(list(cp_theta['Z'].values())), cp.hstack(list(cp_x.values()))))

    # x_obs = np.array(list(results_sue['train'][i]['x'].values()))
    # q_obs = q
    dq = 1 / 2 * cp.sum((cp.hstack(list(cp_q.values())) - q_obs) ** 2)
    cp_tt = [link.bpr.bpr_function_x(cp_x[link.key]) for i, link in links.items()]
    dt = cp.sum(cp.hstack(np.array(cp_tt) - tt_obs))  # -cp.sum(cp.multiply(np.array(cp_tt),np.array(list(q.values()))))
    dx = 1 / 2 * cp.sum((cp.hstack(list(cp_x.values())) - x_obs) ** 2)

    OD_term = 100 * dq + dx + 0 * dt

    # Objective function for multiattribute problem
    utility_integral = tt_utility_integral + Z_utility_integral

    entropy = cp.sum(cp.entr(cp.hstack(list(cp_f.values()))))
    cp_objective = cp.Maximize(utility_integral + entropy - OD_term)

    # Problem
    # cp_problem = cp.Problem(cp_objective)
    cp_problem = cp.Problem(cp_objective, cp_constraints)

    # cp_problem.is_dcp()

    # Assign parameters values in objective function
    cp_theta['Y']['tt'].value = theta['tt']
    for k in k_Z:
        cp_theta['Z'][k].value = theta[k]

    cp_theta['Z']['c'].value

    # Solve
    objective_value = cp_problem.solve(solver=cp_solver,
                                       verbose=True)  # (solver = solver) #'ECOS' solver crashes with some problems

    # Results

    tt = {}
    # - Travel time by link
    tt['x'] = {i: link.bpr.bpr_function_x(x=cp_x[link.key].value) for i, link in links.items()}

    # Link flows
    x = {k: v.value for k, v in cp_x.items()}

    # Path flows
    f = {k: v.value for k, v in cp_f.items()}

    # OD terms
    q = {k: v.value for k, v in cp_q.items()}

    # print(np.sum(q_obs))
    # print(np.sum(np.array(list(q.values()))))
    # print(np.sum(np.array(list(f.values()))))
    # print(np.sum(np.array(list(x.values()))))
    # print(np.sum(x_obs))
    # tt['x']
    # tt_obs
    # tt['x']
    # np.round(np.array(list(q.values())),1)
    # q_obs

    # # Todo: Flow by route. Require class path
    # f = {k:v.value for k,v in cp_x.items()}

    # results = dict({'f': cp_f.value, 'x': x, 'tt': tt})

    return {'x': x, 'f': f, 'tt_x': tt['x']}



def scaling_routes_attributes_onebyone(Y, Z, M, scale):
    # Scale attributes: TODO: Speed up this method and encapsulate it in its own function

    row = 0
    sum_acc = 0

    wide_M = widetolong(M)
    nozero_M = non_zero_matrix(M)

    Y_c, Z_c = Y, Z

    Y_C = attribute_matrix_tolist(wide_M * Y, remove_zeros=True)
    Z_C = [attribute_matrix_tolist(wide_M * Z_c[:, i], remove_zeros=True) for i in range(Z_c.shape[1])]

    for i in range(len(Y_C)):
        # Scaling of matrix with attribtue values of choice set
        Y_C[i] = preprocessing.scale(Y_C[i], with_mean=scale['mean'], with_std=scale['std'])
        # Scaled fravel time in chosen route obtained from tt_long
        Y_c[i] = Y_C[i][i - sum_acc]

        if (i + 1) == len(Y_C) or nozero_M[row, i + 1] == 0:
            sum_acc += sum(nozero_M[row, :])
            row += 1

    Z_c = {}
    for Z_i in range(len(Z_C)):
        row = 0
        sum_acc = 0
        Z_c[Z_i] = {}
        for route_j in range(len(Z_C[Z_i])):

            Z_C[Z_i][route_j] = preprocessing.scale(Z_C[Z_i][route_j], with_mean=scale['mean'],
                                                    with_std=scale['std'])
            Z_c[route_j][Z_i] = Z_C[Z_i][route_j][route_j - sum_acc]
            # Z_routes[i][j] = Z_routes[i][j]
            if (route_j + 1) == len(Z_C[Z_i]) or nozero_M[row, route_j + 1] == 0:
                sum_acc += sum(nozero_M[row, :])
                row += 1

    return Y_C, Y_c, Z_C, Z_c


def prediction_error_logit_path(f, M, V_c, V_C, cp_theta, theta):
    cp_theta_all = {**cp_theta['Y'], **cp_theta['Z']}

    # Set parameters values to modify utility values and then make predictions
    for k, v in cp_theta_all.items():
        if v is not np.nan:
            cp_theta_all[k].value = theta[k]

    P = []
    pred_F = []
    q_long = f.dot(M.T) @ M  # Demand of the OD pair associated to each route

    for i in range(len(f)):
        # Stable softmax
        max_v = max(V_C[i].value)
        P.append(np.exp(V_c[i].value - max_v) / np.sum(np.exp(V_C[i].value - max_v)))
        # P.append(np.exp(V_c[i].value) / np.sum(np.exp(V_C[i].value)))
        pred_F.append(P[i] * q_long[i])

    error = np.linalg.norm(np.array(pred_F) - f, ord=2)

    return error


def likelihood_path_level_logit(f: np.array, M: np.array, D: np.array
                                , k_Y: list, Y: np.array, k_Z: list,
                                Z: {}, scale={'mean': False, 'std': False}):
    """Logit model fitted from output from SUE

    Arguments
    ----------
    :argument f: vector with path flows
    :argument M: Path/O-D demand incidence matrix Random
    :argument D: Path/link incidence matrix
    :argument Y: Matrix with attributes values of chosen routes or links that are (endogenous) flow dependent  (n_routes X n_attributes)
    :argument features_Y: list of labels of attributes in T that are used to fit the discrete choice model
    :argument Z: Dictionary with attributes values of chosen routes or links that are (exogenous) not flow dependent  (n_routes X n_attributes)
    :argument features: list of labels of attributes in Z that are used to fit the discrete choice model

    Returns
    -------
    ll: likelihood function obtained from cvxpy
    cp_theta: dictionary with lists of cvxpy.Variable. Keys: 'Y' for flow dependent and 'Z' for non flow dependent variables
    V_c_z: List with utility function for chosen alternative
    V_C_z: List with utility functions for alternatives in choice set

    """

    # Matrix of values using subset of variables features_Y and features selected for estimation
    Y = (get_matrix_from_dict_attrs_values({k_y: Y[k_y] for k_y in k_Y}).T @ D).T
    Z = (get_matrix_from_dict_attrs_values({k_z: Z[k_z] for k_z in k_Z}).T @ D).T

    # - Optimization variables
    # cp_theta = {k: cp.Parameter(nonpos=True) for k in Z_dict.keys()}
    cp_theta = {}
    cp_theta['Y'] = {k: cp.Variable(nonpos=True) for k in k_Y}
    cp_theta['Z'] = {k: cp.Variable(nonpos=True) for k in k_Z if k not in k_Y}

    # - Input
    flow_routes = numeric.round_almost_zero_flows(f)

    # Wide M matrix that serves as a choice set matrix (C)
    C = choice_set_matrix_from_M(M)

    # Attributes of chosen alternatives
    Y_c, Z_c = Y, Z

    # TODO: the scaling needs to be done by choice set to give the same solution of logit model.
    if scale['mean'] or scale['std']:
        # Scaling by attribute
        Y_c = preprocessing.scale(Y, with_mean=scale['mean'], with_std=scale['std'], axis=0)
        Z_c = preprocessing.scale(Z, with_mean=scale['mean'], with_std=scale['std'], axis=1)

    # Attributes in alternatives within choice sets
    Y_C = [get_attribute_list_by_choiceset(C=C, z=Y_c[:, i]) for i in range(Y_c.shape[1])]
    Z_C = [get_attribute_list_by_choiceset(C=C, z=Z_c[:, i]) for i in range(Z_c.shape[1])]

    # Loglikelihood function obtained from iterating across OD pairs
    ll = []

    # Utilities for chosen alternative and choice set
    V_c = []
    V_C = []

    for i in range(len(flow_routes)):

        # i = 0
        # List storing the contribution from each choice (expansion) set to the likelihood
        ll_i = []

        # if len(tt_long[i]) > 1:  # If there is a single path, then no information is added for estimation.
        # Z = Z[i]

        V_c_i = [Y_c[i] * cp_theta['Y']['tt'] + Z[i, :] * cp.hstack(list(cp_theta['Z'].values()))]

        V_C_i = []
        for j in range(Y.shape[1]):
            V_C_i.append(Y_C[j][i] * list(cp_theta['Y'].values())[j])

        for j in range(Z.shape[1]):
            V_C_i.append(Z_C[j][i] * list(cp_theta['Z'].values())[j])

        V_c.append(cp.sum(V_c_i))  # Utility of chosen alternative
        V_C.append(cp.sum(V_C_i))  # Utility vector for alternatives in choice set

        ll_i.append((cp.sum(V_c_i) - cp.log_sum_exp(cp.sum(V_C_i))) * flow_routes[i])

        ll.append(cp.sum(ll_i))

    return {'cp_ll': ll, 'cp_theta': cp_theta, 'V_c': V_c, 'V_C': V_C}


def cp_regularizer(beta, p):
    if p == 0:
        return 0
    else:
        return cp.pnorm(beta, p=p) ** p


def solve_path_level_logit(cp_ll, cp_theta, constraints_theta: list, lambdas=np.array([0]), r=1, cp_solver=cp_solver):
    """
    Arguments
    ----------
    :argument cp_ll: likelihood function obtained from cvxpy
    :argument constraints_theta: list that defines which attributes ignore and includes in the maximization of the likelihood
    :argument r: degree of regularizer (p = 1 is lasso regularization, and p = 2 is Ridge regression
    :argument lambdas: array with the range of values for lambda. If no lambda is provided, the array has the element 0.
    :argument cp_theta: list of cvxpy.Variable

    Returns
    -------
    results: TODO: use Results structure (see pylogit and other packages)

    """

    # - Constraints (for T and Z attributes)
    cp_constraints_theta = [cp_theta['Z'][k] == v for k, v in constraints_theta['Z'].items() if v is not np.nan]
    cp_constraints_theta.extend([cp_theta['Y'][k] == v for k, v in constraints_theta['Y'].items() if v is not np.nan])

    # Regularization term
    cp_lambda = cp.Parameter(nonneg=True)

    # TODO: implement softmax trick to avoid overflow and exceptions

    # Objective
    n = 1  # TODO: Define proper normaliztion constant, e.g. len(cp_ll) #Number of choice scenarios to normalize the likelihood
    cp_objective = cp.Maximize(
        cp.sum(cp_ll) / n - cp_lambda * cp.norm(cp.hstack({**cp_theta['Y'], **cp_theta['Z']}.values()), r))
    # cp_objective = cp.Maximize(cp.sum(cp_ll))
    # Problem
    cp_problem = cp.Problem(cp_objective, constraints=cp_constraints_theta)  # Excluding extra attributes

    # Fitting hyperparameter of regularization
    results = {}
    for i, lambda_i in zip(range(len(lambdas)), lambdas):
        cp_lambda.value = lambda_i
        try:
            cp_problem.solve(cp_solver)  # (solver = solver) # solver = 'ECOS', solver = 'SCS'

        except:
            pass  # Ignore invalid entries of lambda when the outer_optimizer fails.
            # theta_Z = {k: '' for k in cp_theta['Z'].keys()} #None
            # theta_Y = {k: '' for k in cp_theta['Y'].keys()} #None

        else:
            theta_Z = {k: v.value for k, v in cp_theta['Z'].items()}
            theta_Y = {k: v.value for k, v in cp_theta['Y'].items()}
            results[i] = {'lambda': cp_lambda.value, 'theta_Y': theta_Y, 'theta_Z': theta_Z}

    return results


def prediction_error_logit_regularization(theta_estimates: {}, lambda_vals: {}, likelihood: {}, f: {}, M: {}):
    errors_logit = {}
    theta_vals = {}
    lambda_valid_vals = {}

    for N_i in lambda_vals.keys():
        errors_logit[N_i] = {}
        theta_vals[N_i] = []
        lambda_valid_vals[N_i] = []
        n_paths = np.sum(M[N_i])

        for iter, val in lambda_vals[N_i].items():
            # theta_values = {**val['theta_Y'],**val['theta_Z']}  # From training

            try:
                raw_error = prediction_error_logit_path(
                    f=f[N_i]
                    , M=M[N_i]
                    , V_c=likelihood[N_i]['V_c']
                    , V_C=likelihood[N_i]['V_C']
                    , cp_theta=likelihood[N_i]['cp_theta']
                    , theta=theta_estimates[N_i][iter])
            except:
                pass
                # errors_logit['train'][i][iter] = ''

            else:
                theta_vals[N_i].append(theta_estimates[N_i])
                lambda_valid_vals[N_i].append(lambda_vals[N_i][iter])
                errors_logit[N_i][iter] = np.round(raw_error, 4) / n_paths

    return errors_logit, lambda_valid_vals


def sue_logit_simulation_recovery(N, theta, constraints_theta, k_Z, remove_zeros, scale_features):

    result_sue = equilibrium.sue_logit_fisk(M=N.M
                                                        , D=N.D
                                                        , q = networks.denseQ(Q=N.Q, remove_zeros=remove_zeros)
                                                        , links=N.links_dict
                                                        , paths=N.paths
                                                        , Z_dict = N.Z_dict
                                                        , k_Z = k_Z
                                                        , theta=theta
                                                        )

    Y_links = np.hstack(list(result_sue['tt_x'].values()))
    Y_routes = Y_links @ N.D

    # # + Exogenous attributes from link to path level (rows)
    Z_links = estimation.get_matrix_from_dict_attrs_values(N.Z_dict)
    Z_routes= (Z_links.T @ N.D).T

    likelihood_logit = estimation.likelihood_path_level_logit(f=np.array(list(result_sue['f'].values()))
                                                                          , M=N.M
                                                                          , k_Z= list(N.Z_dict.keys())
                                                                          , Z=Z_routes
                                                                          , k_Y=['tt']
                                                                          , Y= Y_routes
                                                                          , scale=scale_features
                                                                          )

    # Maximize likelihood to obtain solution
    result_logit = estimation.solve_path_level_logit(cp_ll=likelihood_logit['cp_ll']
                                                                 , cp_theta=likelihood_logit['cp_theta']
                                                                 , constraints_theta=constraints_theta
                                                                 )

    return result_logit



def solve_link_level_model(end_params: {}, Mt: {}, Ct: {}, Dt: {}, k_Y: list, Yt: {}, k_Z: list, Zt: {}, xt: {},
                           idx_links: {}, scale: {}, q0: np.array, theta0: {}
                           , lambda_hp: float, constraints_q0=[], norm_o=2, norm_r=1):
    '''

        Parameters are dictionaries where each key contains the value of a matrix or dictionary associated a specific instance of a network (e.g. day)

        :param theta0: avoid setting it to zero because it gives advantage to not regularized model as it starts from the base that the sparse parameters are zero.
        :param Mt:
        :param Dt:
        :param k_Y:
        :param Yt:
        :param k_Z:
        :param Zt:
        :param q:
        :param xt:
        :param idx_links:
        :param scale:
        :param lambda_hp:

        :return:
        :argument

        '''

    # if not isinstance(M, dict):
    #     D, M, Y, Z, q, x, idx_links = {1: D}, {1: M}, {1: Y}, {1: Z}, {1:q}, {1:x}, {1:idx_links}
    # i = 'N6'

    days = Mt.keys()
    n_days = len(list(days))

    for i in days:

        # Keep this here for efficiency instead of putting it in gap function
        Yt[i] = (get_matrix_from_dict_attrs_values({k_y: Yt[i][k_y] for k_y in k_Y}).T @ Dt[i]).T
        Zt[i] = (get_matrix_from_dict_attrs_values({k_z: Zt[i][k_z] for k_z in k_Z}).T @ Dt[i]).T

        if scale['mean'] or scale['std']:
            # Scaling by attribute
            Yt[i] = preprocessing.scale(Yt[i], with_mean=scale['mean'], with_std=scale['std'], axis=0)
            Zt[i] = preprocessing.scale(Zt[i], with_mean=scale['mean'], with_std=scale['std'], axis=1)

    # Starting values (q0, theta0)
    q = q0  # np.zeros(M[0].shape[0])
    theta = np.array(
        list(list(theta0.values())))  # np.array(list(list(theta0.values())))  # dict.fromkeys(theta_true, 1)

    # TODO: provide directly Subset of matrices with idx_links

    if end_params['theta'] and end_params['q']:

        # q = q0  # np.zeros(M[0].shape[0])

        # Constraints:

        range_theta = range(0, len(theta))
        range_q = range(len(theta), len(theta) + len(q))

        # Estimation

        # TODO: use lmfit or other non-linear least square outer_optimizer to obtain t-statistic and more robust and faster performance
        # strt using method "Levenberg-Marquardt". See https://lmfit.github.io/lmfit-py/fitting.html#choosing-different-fitting-methods
        estimation_results = minimize(loss_link_level_model
                                      , x0=np.hstack([theta, q])
                                      , args=(
                range_theta, range_q, end_params, q, theta, lambda_hp, Dt, Mt, Ct, Yt, Zt, xt, idx_links, norm_o,
                norm_r)
                                      , constraints=constraints_q0
                                      # , method='BFGS', options={'gtol': 1e-6, 'disp': True}
                                      )

        return {'theta': dict(zip(theta0.keys(), np.round(estimation_results['x'][range_theta], 4))),
                'q': np.round(estimation_results['x'][range_q], 1),
                'gap': np.round(estimation_results['fun'] / n_days, 4)}


    elif end_params['theta']:

        range_theta = range(0, len(theta))
        range_q = range(len(theta), len(theta))

        estimation_results = minimize(loss_link_level_model
                                      , x0=theta
                                      , args=(
                range_theta, range_q, end_params, q, theta, lambda_hp, Dt, Mt, Ct, Yt, Zt, xt, idx_links, norm_o,
                norm_r)
                                      # , constraints=constraints_q0
                                      # , method='BFGS', options={'gtol': 1e-6, 'disp': True}
                                      )

        return {'theta': dict(zip(theta0.keys(), np.round(estimation_results['x'], 4))),
                'gap': np.round(estimation_results['fun'] / n_days, 4)}

    elif end_params['q']:

        range_theta = range(0, 0)
        range_q = range(0, len(q))

        estimation_results = minimize(loss_link_level_model
                                      , x0=q
                                      , args=(
                range_theta, range_q, end_params, q, theta, lambda_hp, Dt, Mt, Ct, Yt, Zt, xt, idx_links, norm_o,
                norm_r)
                                      , constraints=constraints_q0
                                      # , method='BFGS', options={'gtol': 1e-6, 'disp': True}

                                      )

        return {'q': np.round(estimation_results['x'], 1),
                'gap': np.round(estimation_results['fun'] / n_days, 4)}


def loss_SUE(o, x_obs, D, M, q, links: {}, paths: [], theta: np.array, Z_dict: {}, cp_solver='ECOS', k_Z=[]):
    # if theta['tt'] < 0:
    #     theta['tt'] = -theta['tt']

    # TODO: use an iterative method that does not require a non-positive theta
    for k in theta.keys():
        if theta[k] > 0:
            theta[k] = 0
            # theta[k] = -theta[k]
            # Z_dict[k] = dict(zip(Z_dict[k].keys(), list(-1*np.array(list(Z_dict[k].values())))))
    try:
        results_sue = equilibrium.sue_logit_fisk(q=q
                                                 , M=M
                                                 , D=D
                                                 , links=links
                                                 , paths=paths
                                                 , Z_dict=Z_dict
                                                 , k_Z=[]
                                                 , theta=theta
                                                 , cp_solver='ECOS'
                                                 # 'ECOS': it is faster and crashes sometimes. 'SCS' is slow
                                                 )

    except:
        return np.nan
    else:
        return np.linalg.norm(np.array(list(results_sue['x'].values())) - x_obs, o)

# def all_paths_from_OD(G,Q):
#
#     G.nodes()
#     G.edges()
#
#
#     paths = list(nx.all_simple_paths(G, source = 0, target = 1))
#
#     for path in map(nx.utils.pairwise, paths):
#         print(list(path))
#
#     list(test_all_paths(G))
#
#     return paths



class BenchmarkOptimizersExperiment(NetworkExperiment):

    ''' Remains to be updated '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self,
            config,
            theta_true,
            network,
            network_label,
            k_Y=['tt']):

        print('\nPerforming optimization benchmark experiment')

        config.experiment_options['optimization_methods'] = ['gd', 'ngd', 'adagrad', 'adam', 'gauss-newton', 'lm']

        config.make_dirs(foldername=config.experiment_options['current_network'].lower())

        replicates = config.experiment_options['replicates']

        # config.experiment_options['features'] = config.estimation_options['features']
        #
        # config.experiment_options['theta_0_grid'] = np.linspace(*config.experiment_options['theta_0_range'], replicates)

        # config.experiment_options['alpha'] = 0.05

        # # Learning rate for first order optimization
        # config.estimation_options['eta_norefined'] = 1e-0
        #
        # # Bilevel iters
        # config.estimation_options['bilevel_iters_norefined'] = 10  # 10

        for replicate in range(1,replicates+1):

            config.make_dirs(config.experiment_options['current_network'].lower())

            config.experiment_options['theta_0'] = dict.fromkeys(k_Y + config.experiment_options['features'],
                                                                 config.experiment_options['theta_0_grid'][replicate])

            config.experiment_options['theta_true'] = theta_true[network_label]

            # printer.enablePrint()

            printer.printProgressBar(replicate, replicates, prefix='\nProgress:', suffix='', length=20)

            print('\nReplicate:' + str(replicate + 1))

            # printer.blockPrint()

            # Generate synthetic traffic counts
            counts, _ = self.linkdata_generator.simulate_counts(
                network=self.network,
                equilibrator=self.equilibrator,
                utility_function=self.utility_function)

            self.network.load_traffic_counts(counts=counts)

            t0 = time.time()

            theta_values = {}
            q_values = {}
            objective_values = {}
            result_eq_values = {}
            results = {}

            results_experiment = pd.DataFrame(columns=['iter', 'loss', 'vot', 'method'])

            for method in config.experiment_options['optimization_methods']:
                bilevel_estimation_norefined = estimation.Learner(config.theta_0)

                learning_results_refined, inference_results_refined, best_iter_refined = \
                    self.learner_refined.statistical_inference(bilevel_iters=bilevel_iters, alpha=alpha)

                q_values[method], theta_values[method], objective_values[method], result_eq_values[method], results[
                    method] \
                    = bilevel_estimation_norefined.bilevel_optimization(
                    # network= modeller.clone_network(N['train'][i], label = N['train'][i].label),
                    Nt=network,
                    k_Y=k_Y, k_Z=config.experiment_options['features'],
                    Zt={1: network.Z_dict},
                    q0=network.q,
                    xct={1: np.array(list(xc.values()))},
                    theta0=config.experiment_options['theta_0'],
                    # If change to positive number, a higher number of iterations is required but it works well
                    # theta0 = theta_true[i],
                    outeropt_params={
                        # 'method': 'adagrad',
                        # 'method': 'adam',
                        # 'method': 'gd',
                        # 'method': 'ngd',
                        'od_estimation': False,
                        'method': method,
                        'iters_scaling': int(0e0),
                        'iters': config.experiment_options['iters'],  # 10
                        'eta_scaling': 1e-1,
                        'eta': config.experiment_options['eta'],  # works well for simulated networks
                        # 'eta': 1e-4, # works well for Fresno real network
                        'gamma': config.experiment_options['gamma'],
                        'v_lm': 10, 'lambda_lm': 1,
                        'beta_1': 0.8, 'beta_2': 0.8,
                        'batch_size': 0,
                        'paths_batch_size': config.estimation_options['paths_batch_size']
                    },

                    inneropt_params={'iters': config.experiment_options['max_sue_iters'],
                                     'accuracy_eq': config.experiment_options['accuracy_eq'], 'method': 'line_search',
                                     'iters_ls': config.estimation_options['iters_ls_fw'],
                                     'uncongested_mode': config.experiment_options['uncongested_mode']},

                    bilevelopt_params={'iters': config.experiment_options['bilevel_iters']},  # {'iters': 10},
                    # plot_options = {'y': 'objective'}
                )

                # Create pandas dataframe

                results_experiment = results_experiment.append(pd.DataFrame({'iter': results[method].keys()
                                                                                ,
                                                                             'loss': [val['objective'] for key, val in
                                                                                      results[method].items()]
                                                                                , 'vot': [
                        val['theta']['tt'] / val['theta']['c'] for key, val in
                        results[method].items()]
                                                                                , 'method': [method for key, val in
                                                                                             results[method].items()]}))

                # results_experiment.append(pd.DataFrame({'iter': results[method].keys()
                #                                            , 'theta': [val['theta'] for key, val in results[method].items()]
                #                                            , 'method': [method]}))

            config.experiment_results['theta_estimates'] = theta_values
            config.experiment_results['losses'] = objective_values

            writer.write_experiment_report(filename='summary_report'
                                           , config=config
                                           , decimals=3
                                           # , float_format = 2
                                           )

            # fig_loss = plt.figure()
            plot1 = visualization.Artist(folder_plots=config.log_file['experimentpath'], dim_subplots=(2, 2))
            fig_loss, fig_vot, fig_legend = plot1.benchmark_optimization_methods(
                methods=config.experiment_options['optimization_methods'], results_experiment=results_experiment
                , colors=['blue', 'green', 'red', 'magenta', 'brown', 'gray', 'purple']
            )

            # for key, grp in results_experiment.groupby(['method']):
            #     ax.plot(grp['iter'], grp['loss'], label=key)
            # results_experiment.set_index('iter')
            # results_experiment.groupby('method')['loss'].plot(legend=True)
            # plt.show()

            writer.write_figure_experiment_to_log_folder(config.log_file, 'loss_optimization_methods.pdf', fig_loss)

            # Save legend
            writer.write_figure_experiment_to_log_folder(config.log_file, 'optimization_methods_legend.pdf',
                                                         fig_legend)

            # fig_vot = plt.figure()

            # ax.plot([min(np.array(list(vot_estimate[i].keys()))), max(np.array(list(vot_estimate[i].keys())))]
            #                   , [theta_true['tt'] / theta_true['c'], theta_true['tt'] / theta_true['c']],
            #                   label=r'$\theta_t/\theta_c$', color='black')

            # ax.legend()
            # plt.show()

            # results_experiment.set_index('iter')
            # results_experiment.groupby('method')['vot'].plot(legend=True)
            # plt.show()

            writer.write_figure_experiment_to_log_folder(config.log_file, 'vot_optimization_methods.pdf', fig_vot)

            time_ngd = time.time() - t0

            writer.write_experiment_report(filename='summary_report'
                                           , config=config
                                           , decimals=3
                                           # , float_format = 2
                                           )

        # artist = visualization.Artist(folder_plots=config.log_file['experimentpath'])

        # /Users/pablo/google-drive/data-science/github/isuelogit/output/log/experiments

        # artist.inference_experiments(results_experiment=results_experiment
        #                                        , theta_true=theta_true[current_network], subfoldername='')

        # Write report in log file

        options_df = pd.DataFrame({'group': [], 'option': [], 'value': []})

        # Replace the value of theta_0 for the reference value because otherwise it is the one corresponding the last replicate
        config.experiment_options['theta_0'] = dict.fromkeys(k_Y + config.experiment_options['features'], 0)

        for key, value in config.experiment_options.items():
            options_df = options_df.append({'group': 'experiment_options', 'option': key, 'value': value},
                                           ignore_index=True)

        writer.write_csv_to_log_folder(df=options_df,
                                       filename='experiment_options'
                                       , log_file=config.log_file
                                       , float_format='%.1f'
                                       )