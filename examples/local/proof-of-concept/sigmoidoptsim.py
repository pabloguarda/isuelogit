
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time
import os

import seaborn as sns
from matplotlib.transforms import BlendedGenericTransform
from matplotlib.ticker import FormatStrFormatter


# https://stackoverflow.com/questions/42142144/displaying-first-decimal-digit-in-scientific-notation-in-matplotlib
from matplotlib.ticker import ScalarFormatter
class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here



def sigmoid(theta, deltatt):
    return 1/(1+np.exp(-theta*deltatt))

theta_true = -1#-0.5
theta_0 = -1.2 #4
n = 50
m = 20
alpha_bh = 4 # For bordered hessian
gamma = 1 #0.5 #1: no momentum
eta = 3e-5 # 1e-5
eta_n = 3e-1 #0.1
batch_size = int(n * 0.5) #Batch equal to

iters = 50
interval_deltatt = 10
interval_q = 40
noise_factor = 5

folder_plots = os.getcwd() + '/examples' + '/plots'

# reps = 100
# simulation_sigmoids()

# https://www.wolframalpha.com/input/?i=sigmoid%28x%29+%2B+sigmoid%28-4*x%29%2B+sigmoid%28-3*x%29%2B+sigmoid%288*x%29+-1.9985+for+x+in+-5+to+5

# #Non-convex function
# deltat = [[1,-4,-3,8]]
# q = [[1,1,1,1]]

# deltat = np.random.randint(-10, 10, n)
# q = np.random.randint(0, 20, n)



#
# deltat.append([2,3,4,0])
# q.append([1,-1,3,0])

def simulate_sigmoid_system(theta_true, noise_factor, n , m):

    deltatt, q, linkflow = [], [], []

    for i in np.random.randint(-1*interval_deltatt, 1*interval_deltatt, [n,m]).tolist():
    # for i in np.random.randint(0, 10, [n,m]).tolist():
        deltatt.append(i)

    for i in np.random.randint(0, 1*interval_q, [n, m]).tolist():
        q.append(i)

    noise = noise_factor * np.random.normal(0, 1, n)
    # noise = 2*np.abs(np.random.uniform(-1,1,n))

    for i in range(0, len(deltatt)):
        constraint = np.sum(
            [q[i][j] * sigmoid(theta=theta_true, deltatt=deltatt[i][j]) for j in range(0, len(deltatt[i]))])

        # Noise
        constraint += noise[i]
        # constraint += constraint*0#np.abs(np.random.uniform(-0.000,0.000,1))

        linkflow.append(constraint)

    deltatt = np.array(deltatt)
    q = np.array(q)
    linkflow = np.array(linkflow)

    return deltatt, q, linkflow


deltatt, q, linkflow = simulate_sigmoid_system(theta_true = theta_true, noise_factor = noise_factor, n = n , m = m)

# deltat,q,linkflow  = deltat[0,:], q[0,:], linkflow[0]

def gradient_sigmoid(theta, deltatt):
    return sigmoid(theta, deltatt)*(1-sigmoid(theta, deltatt))*deltatt

def hessian_sigmoid(theta, deltatt):
    # return (gradient_sigmoid(theta, deltatt) * (1 - sigmoid(theta, deltatt)) + -gradient_sigmoid(theta, deltatt)*sigmoid(theta, deltatt)) * deltatt
    return gradient_sigmoid(theta, deltatt)*(1-2*sigmoid(theta, deltatt))*deltatt

def diff3_sigmoid(theta, deltatt):
    return (hessian_sigmoid(theta, deltatt)*(1-2*sigmoid(theta, deltatt)) - 2*gradient_sigmoid(theta, deltatt)**2)*deltatt

def bordered_hessian_sigmoid(theta, deltatt,alpha_bh):
    return hessian_sigmoid(theta, deltatt) + alpha_bh*gradient_sigmoid(theta, deltatt)**2

def objective_function_sigmoids_system(theta, q,deltatt):
    ''' Return the value of each term associated to each link in the objective function'''

    sigmoids = []

    if len(deltatt.shape) > 1:
        for i in range(0, deltatt.shape[0]):
            sigmoids.append([q[i,j] * sigmoid(theta=theta, deltatt=deltatt[i,j]) for j in range(0, deltatt.shape[1])])

    else:
        sigmoids.append([q[j] * sigmoid(theta=theta, deltatt=deltatt[j]) for j in range(0, deltatt.shape[0])])

    return sigmoids

def gradient_sigmoids_system(theta, q, deltatt):
    return np.sum(q*gradient_sigmoid(theta = theta, deltatt = deltatt),axis = 1)

def sse(theta, q, deltatt, linkflow):
    return np.sum((np.sum(objective_function_sigmoids_system(theta, q = q,deltatt = deltatt), axis=1) - linkflow.T)**2)

def confint_theta(theta, q, deltatt, linkflow, n, p =1, alpha = 0.05):
    var_error = sse(theta = theta, q = q, deltatt = deltatt, linkflow = linkflow) / (n - p)

    # # Unidimensional inverse function is just the reciprocal
    F = gradient_sigmoids_system(theta=theta, q =  q, deltatt = deltatt)
    cov_theta = var_error  * (F.dot(F))**-1
    # cov_theta = var_error  * 1/gradient_sigmoids_system(theta=theta, q = q, deltatt = deltatt) ** 2

    # T value (two-tailed)
    critical_tvalue = stats.t.ppf(1-alpha/2, df = n-p)

    width_confint = critical_tvalue*np.sqrt(cov_theta)

    return "[" + str(theta - width_confint) + ", " + str(theta + width_confint) + "]", width_confint

def ttest_theta(theta, theta_h0, q, deltatt, linkflow,n, p =1, alpha = 0.05):
    var_error = sse(theta = theta, q = q, deltatt = deltatt, linkflow = linkflow) / (n - p)

    # # Unidimensional inverse function is just the reciprocal
    F = gradient_sigmoids_system(theta=theta, q=q, deltatt=deltatt)
    cov_theta = var_error * (F.dot(F)) ** -1
    # cov_theta = var_error  * 1/gradient_sigmoids_system(theta=theta, q = q, deltatt = deltatt) ** 2

    # T value
    critical_tvalue = stats.t.ppf(1-alpha/2, df = n-p)
    ttest = np.abs(theta- theta_h0)/np.sqrt(cov_theta)
    # width_int = two_tailed_ttest*np.sqrt(cov_theta)
    # pvalue =  (1 - stats.t.cdf(ttest,df=n-p))
    # https://stackoverflow.com/questions/23879049/finding-two-tailed-p-value-from-t-distribution-and-degrees-of-freedom-in-python
    pvalue = stats.t.sf(ttest, df=n-p) * 2
    # return "[" + str(theta - width_int) + "," + str(theta + width_int) + "]"
    return ttest, critical_tvalue, pvalue

def objective_function(theta, linkflow, q, deltatt):
    ''' Return the value of each term associated to each link in the objective function'''

    sigmoids = objective_function_sigmoids_system(theta = theta, q = q , deltatt = deltatt )

    return (np.sum(np.array(sigmoids),axis = 1)-linkflow.T)**2

def gradients_l2norm(theta, deltatt, q, linkflow):

    sigmoids = []
    gradients_sigmoids = []

    if len(deltatt.shape) > 1 :

        for i in range(0, deltatt.shape[0]):
            sigmoids.append([q[i,j] * sigmoid(theta=theta, deltatt=deltatt[i,j]) for j in range(0, deltatt.shape[1])])

        for i in range(0, deltatt.shape[0]):
            gradients_sigmoids.append([q[i,j] * gradient_sigmoid(theta=theta, deltatt=deltatt[i,j]) for j in range(0, deltatt.shape[1])])



    else:
        sigmoids.append([q[j] * sigmoid(theta=theta, deltatt=deltatt[j]) for j in range(0, deltatt.shape[0])])
        gradients_sigmoids.append(
            [q[j] * gradient_sigmoid(theta=theta, deltatt=deltatt[j]) for j in range(0, deltatt.shape[0])])

    return 2 * (np.sum(np.array(sigmoids), axis=1) - np.array(linkflow).T) * np.sum(
        np.array(gradients_sigmoids), axis=1)

def hessian_l2norm(theta, deltatt, q, linkflow):

    # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts

    # return 2*np.mean(gradients_l2norm(theta, deltatt, q, linkflow)**2+np.sum(q*hessian_sigmoid(theta = theta, deltatt = deltatt),axis = 1)*(np.sum(objective_function_sigmoids_system(theta,q,deltatt),axis=1)-linkflow.T))
    # J = gradients_l2norm(theta, deltatt, q, linkflow)
    J = np.sum(q*gradient_sigmoid(theta = theta, deltatt = deltatt),axis = 1)
    R = np.sum(objective_function_sigmoids_system(theta, q, deltatt), axis=1) - linkflow.T
    H = np.sum(q * hessian_sigmoid(theta=theta, deltatt=deltatt), axis=1)
# np.sum(J**2)
    return 2*(J*J + H*R)
#     return (R * H)
#     return J.dot(J)

    # return 2 * (np.sum(
    #     q * hessian_sigmoid(theta=theta, deltatt=deltatt), axis=1) * (
    #                         np.sum(objective_function_sigmoids_system(theta, q, deltatt), axis=1) - linkflow.T))


# a =gradients_l2norm(theta, deltatt, q, linkflow)**2
# b = np.array(objective_function_sigmoids_system(theta,q,deltatt))
# b.shape
# a.shape
# theta = -1

# hessian_l2norm(-1, deltatt, q, linkflow)

def gradient_descent_update(theta, grad, eta):

    return theta - grad*eta

def gradient_descent(iters, eta, gamma, theta_0, deltatt = deltatt, q = q, linkflow = linkflow):

    theta = theta_0
    grad = 0

    # over iterations
    thetas = [theta]
    grads = [theta]
    times = [0]
    acc_t = 0

    for iter in range(0,iters):

        t0 = time.time()
        grad_new = np.mean(gradients_l2norm(theta = theta, deltatt = deltatt, q = q, linkflow = linkflow))

        grad = (1-gamma)*grad + gamma*grad_new
        theta = gradient_descent_update(theta = theta, eta = eta, grad = grad)

        delta_t = time.time() - t0
        acc_t += delta_t

        grads.append(grad)
        thetas.append(theta)
        times.append(acc_t)

    theta = thetas[-1]
    grad = grads[-1]

    return np.array(thetas), np.array(grads), np.array(times)

def stochastic_gradient_descent(iters, batch_size, eta, gamma, theta_0, deltatt = deltatt, q = q, linkflow = linkflow):




    theta = theta_0
    grad_old = 0

    # over iterations
    thetas = [theta]
    grads = [theta]
    times = [0]
    acc_t = 0

    for iter in range(0, iters):

        t0 = time.time()

        # TODO: this should be updated with no replacement
        idx = np.random.choice(np.arange(0, q.shape[0]), batch_size) # replace = False

        grad_new = np.mean(gradients_l2norm(theta=theta, deltatt=deltatt[idx,:], q=q[idx,:], linkflow = linkflow[idx]))
        grad = (1-gamma)*grad_old + gamma* grad_new

        theta = gradient_descent_update(theta=theta, eta=eta, grad=grad)

        delta_t = time.time() - t0
        acc_t += delta_t

        times.append(acc_t)
        grads.append(grad)
        thetas.append(theta)
    return np.array(thetas), np.array(grads), np.array(times)

def normalized_gradient_descent(iters, eta, gamma, theta_0, deltatt = deltatt, q = q, linkflow = linkflow):
    theta = theta_0
    grad = 0
    times = [0]
    acc_t = 0

    # over iterations
    thetas = [theta]
    grads = [theta]

    for iter in range(0,iters):

        t0 = time.time()

        if iter < iters - 1:
            grad_new = np.mean(gradients_l2norm(theta = theta, deltatt = deltatt, q = q, linkflow = linkflow))

            grad = (1-gamma)*grad + gamma*grad_new

            grad_normalized = grad/(np.linalg.norm(grad)+1e-7)
            # print(grad)

            theta = gradient_descent_update(theta = theta, eta = eta, grad = grad_normalized)

        else:
            obj_theta = [np.mean(objective_function(theta_i, linkflow = linkflow,q = q,deltatt = deltatt)) for theta_i in thetas]
            theta = thetas[np.argmin(np.array(obj_theta))]
            # print('a ' + str(theta))

        # print(theta)
        delta_t = time.time() - t0
        acc_t += delta_t

        times.append(acc_t)
        grads.append(grad_normalized)
        thetas.append(theta)

    return np.array(thetas), np.array(grads), np.array(times)

def normalized_stochastic_gradient_descent(iters, batch_size, eta, gamma, theta_0, deltatt = deltatt, q = q, linkflow = linkflow):



    theta = theta_0
    grad_old = 0

    # over iterations
    thetas = [theta]
    grads = [theta]
    times = [0]
    acc_t = 0

    for iter in range(0, iters):
        t0 = time.time()

        idx = np.random.choice(np.arange(0, q.shape[0]), batch_size)

        if iter < iters-1:


            grad_new = np.mean(gradients_l2norm(theta=theta, deltatt=deltatt[idx,:], q=q[idx,:], linkflow = linkflow[idx]))
            grad = (1-gamma)*grad_old + gamma* grad_new

            grad_normalized = grad / (np.linalg.norm(grad) + 1e-7)
            # grad_normalized = np.sign(grad)

            theta = gradient_descent_update(theta=theta, eta=eta, grad=grad_normalized)
        else:

            # Note that in Hazan and Levy paper, the function evaluation is done only over the minibatch but it could be done over the entire objective
            obj_theta = [np.mean(objective_function(theta_i, linkflow = linkflow[idx],q = q[idx,:],deltatt = deltatt[idx,:])) for theta_i in thetas]
            theta = thetas[np.argmin(np.array(obj_theta))]

        delta_t = time.time() - t0
        acc_t += delta_t

        times.append(acc_t)
        grads.append(grad_normalized)
        thetas.append(theta)

    return np.array(thetas), np.array(grads), np.array(times)

def ngd_gd(iters, eta_ngd, eta_gd, gamma, theta_0, deltatt = deltatt, q = q, linkflow = linkflow):


    iters_ngd = int(iters/2)
    iters_gd = iters - iters_ngd

    thetas_ngd, grads_ngd, times_ngd = normalized_gradient_descent(iters = iters_ngd, eta = eta_ngd, gamma = gamma, theta_0 = theta_0, deltatt = deltatt, q = q, linkflow = linkflow)
    thetas_gd, grads_gd, times_gd = gradient_descent(iters = iters_gd, eta = eta_gd, gamma = gamma, theta_0 = thetas_ngd[-1], deltatt = deltatt, q = q, linkflow = linkflow)


    return np.concatenate([thetas_ngd, thetas_gd[1:]]), np.concatenate([grads_ngd, grads_gd[1:]]), np.concatenate([times_ngd, times_ngd[-1] + times_gd[1:]])

def nsgd_gd(iters, batch_size, eta_nsgd, eta_gd, gamma, theta_0, deltatt = deltatt, q = q, linkflow = linkflow):

    iters_ngsd = int(iters/2)
    iters_gd = iters - iters_ngsd

    thetas_nsgd, grads_nsgd, times_nsgd = normalized_stochastic_gradient_descent(iters = iters_ngsd, batch_size = batch_size, eta = eta_nsgd, gamma = gamma, theta_0 = theta_0, deltatt = deltatt, q = q, linkflow = linkflow)
    thetas_gd, grads_gd, times_gd = gradient_descent(iters = iters_gd, eta = eta_gd, gamma = gamma, theta_0 = thetas_nsgd[-1], deltatt = deltatt, q = q, linkflow = linkflow)

    return np.concatenate([thetas_nsgd, thetas_gd[1:]]), np.concatenate([grads_nsgd, grads_gd[1:]]), np.concatenate([times_nsgd, times_nsgd[-1] + times_gd[1:]])

def theta_estimation(theta_0, iters, eta_gd, eta_n, deltatt, q, linkflow, batch_size, gamma = 1):

    thetas, grads, times = {}, {}, {}

    thetas['gd'], grads['gd'], times['gd'] = gradient_descent(iters=iters, eta=eta_gd, gamma=gamma, theta_0=theta_0,
                                                              deltatt=deltatt, q=q, linkflow=linkflow)
    thetas['sgd'], grads['sgd'], times['sgd'] = stochastic_gradient_descent(iters=iters, batch_size=batch_size,
                                                                            eta=eta_gd, gamma=gamma, theta_0=theta_0,
                                                                            deltatt=deltatt, q=q, linkflow=linkflow)
    thetas['ngd'], grads['ngd'], times['ngd'] = normalized_gradient_descent(iters=iters, eta=eta_n, gamma=gamma,
                                                                            theta_0=theta_0, deltatt=deltatt, q=q,
                                                                            linkflow=linkflow)
    thetas['nsgd'], grads['nsgd'], times['nsgd'] = normalized_stochastic_gradient_descent(iters=iters,
                                                                                          batch_size=batch_size,
                                                                                          eta=eta_n, gamma=gamma,
                                                                                          theta_0=theta_0,
                                                                                          deltatt=deltatt, q=q,
                                                                                          linkflow=linkflow)
    thetas['ngd_gd'], grads['ngd_gd'], times['ngd_gd'] = ngd_gd(iters=iters, eta_gd=eta, eta_ngd=eta_n, gamma=gamma,
                                                                theta_0=theta_0, deltatt=deltatt, q=q,
                                                                linkflow=linkflow)
    thetas['nsgd_gd'], grads['nsgd_gd'], times['nsgd_gd'] = nsgd_gd(iters=iters, batch_size=batch_size, eta_gd=eta,
                                                                    eta_nsgd=eta_n, gamma=gamma, theta_0=theta_0,
                                                                    deltatt=deltatt, q=q, linkflow=linkflow)

    return thetas, grads, times

# First order gradient based methods

thetas, grads, times = theta_estimation(theta_0 = theta_0, iters = iters, eta_gd = eta, eta_n = eta_n, batch_size = batch_size,  linkflow = linkflow, q = q, deltatt = deltatt, gamma = 1)

#Grid search
thetas_gs = np.arange(-2,2,0.01)
function_grid_vals = [np.mean(objective_function(theta = x_val, q = q, deltatt = deltatt, linkflow = linkflow)) for x_val in thetas_gs]
theta_gs_generalization = thetas_gs[np.argmin(function_grid_vals)]
theta_gs = thetas_gs[np.argmin(function_grid_vals)]

print("theta_true: " + str(theta_true))
print("theta_gridsearch: " + str(theta_gs))
print("generalization_gridsearch: " + str(np.sum(np.abs(np.sum(objective_function_sigmoids_system(theta = theta_gs, q = q, deltatt = deltatt),axis = 1)-linkflow.T))))
print("generalization_gridsearch_truetheta: " + str(np.sum(np.abs(np.sum(objective_function_sigmoids_system(theta = theta_gs_generalization, q = q, deltatt = deltatt),axis = 1)-linkflow.T))))

algorithms = ['gd','sgd','ngd','nsgd','ngd_gd','nsgd_gd']

for alg in algorithms:

    print(alg)
    print("theta_"+ str(alg)+": " + str(thetas[alg][-1]))
    print("grad_gd_"+ str(alg)+": " + str(grads[alg][-1]))
    print("generalization__"+ str(alg)+": " + str(
        np.mean((objective_function(theta=thetas[alg][-1], q=q, deltatt=deltatt, linkflow=linkflow)))))
    print("time_"+ str(alg)+": "+ str(times[alg][-1]))

### Inference and inductive bias
p = 1
print("SSE: " + str(sse(theta = thetas['ngd'][-1], linkflow = linkflow, q = q, deltatt = deltatt)))
print("Sigma_error: " +  str(np.sqrt(sse(theta = thetas['ngd'][-1], linkflow = linkflow, q = q, deltatt = deltatt)/(n-p))))
print("Avg. grad: " + str(np.mean(gradient_sigmoids_system(theta = thetas['ngd'][-1], q = q, deltatt = deltatt))))
print("CI: " + str(confint_theta(theta = thetas['ngd'][-1], n =n, q = q, deltatt = deltatt, linkflow = linkflow)))
print("T-test: " + str(ttest_theta(theta = thetas['ngd'][-1],theta_h0 = 0, n =n, q = q, deltatt = deltatt, linkflow = linkflow)))


#### PLOTS ###

# Range of values for theta for plots
# x_vals = np.arange(-2*max(abs(theta_true), abs(theta_0)),2*max(abs(theta_true), abs(theta_0)),0.01)


def plot_quasiconvexity(filename, subfolder, x_range, theta_true, deltatt, linkflow,q, alpha_bh):
    fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(10,5))
    fig.suptitle("Analysis of  (strict) quasiconvexity"
                 "\n(theta_true = " + str(theta_true) + ")")

    # Plot objective function over an interval
    # ax[(0, 0)].set_title("\n\nObj. function (L2)")
    ax[(0, 0)].set_title("\n\n")
    y_vals = [np.mean(objective_function(theta = x_val, q = q, deltatt = deltatt, linkflow = linkflow)) for x_val in x_range]
    ax[(0,0)].plot(x_range, y_vals,color = 'red')
    ax[(0,0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0,0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0, 0)].set_ylabel(r"$n^{-1} \  ||x(\hat{\theta})-\bar{x}||_2^2$")
    ax[(0, 0)].set_xticklabels([])

    # r"$\hat{\theta}$"

    # ax[(0, 1)].set_title("Gradient L2-norm")
    # y_vals = [np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
    y_vals = [np.mean(gradients_l2norm(theta = x_val, deltatt = deltatt, q=q, linkflow=linkflow)) for x_val in x_range]
    ax[(0,1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0,1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0,1)].plot(x_range, y_vals,color='red')
    ax[(0, 1)].set_ylabel(r"$n^{-1} \ \nabla_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2)$")
    ax[(0, 1)].set_xticklabels([])

    # ax[(0, 2)].set_title("Sign Gradient L2-norm")
    # y_vals = [np.sign(np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1))) for x_val in x_range]
    y_vals = [np.mean(gradients_l2norm(theta=x_val, deltatt=deltatt, q=q, linkflow=linkflow)) for x_val in x_range]
    y_vals = np.sign(y_vals)
    ax[(0,2)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0,2)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0,2)].plot(x_range, y_vals,color='red')
    ax[(0, 2)].set_ylabel(r"$n^{-1} \  sign (\nabla_{\theta} ||x(\hat{\theta})-\bar{x}||_2^2 )$")
    ax[(0, 2)].set_xticklabels([])

    # Hessian L2-norm
    # ax[(1, 3)].set_title("Hessian L2 norm")

    # J = gradients_l2norm(theta, deltatt, q, linkflow)
    # H = np.sum(q * hessian_sigmoid(theta=theta, deltatt=deltatt), axis=1)
    # R = np.sum(objective_function_sigmoids_system(theta, q, deltatt), axis=1) - linkflow.T

    # [np.sum(q * hessian_sigmoid(theta=x_val, deltatt=deltatt), axis=1) for x_val in x_range]

    y_vals = [np.mean(hessian_l2norm(theta = x_val, deltatt = deltatt, q=q, linkflow=linkflow)) for x_val in x_range]
    # y_vals = np.sign(y_vals)
    ax[(0,3)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0,3)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0,3)].plot(x_range, y_vals,color='red',)
    ax[(0,3)].set_ylabel(r"$n^{-1} \ \nabla^2_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2) $")

    # ax[(0, 2)].set_title("Hessian L2-norm")
    # y_vals = [np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
    # ax[(0,1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(0,1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(0,1)].plot(x_range, y_vals,color='red')

    # #Plot sigmoid system
    # ax[(1,0)].set_title("L1 norm")
    # y_vals = [np.mean(np.abs(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)) for x_val in x_range]
    # ax[(1,0)].plot(x_range, y_vals,color = 'red')
    # ax[(1,0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(1,0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(1,0)].set_title("Sigmoid system")

    # Sigmoid system
    y_vals = [np.mean(np.abs(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow)) for x_val in x_range]
    ax[(1,0)].plot(x_range, y_vals,color = 'red')
    ax[(1,0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(1,0)].set_ylabel(r"$||x(\hat{\theta})-\bar{x}||_1$")
    ax[(1, 0)].set_ylabel(r"$ n^{-1} \ ||(x(\hat{\theta})-\bar{x}||_1 $")
    # plt.show()



    # Plot gradients
    # ax[(1, 1)].set_title("Gradient sig-system")
    y_vals = [np.mean(np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
    ax[(1,1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,1)].plot(x_range, y_vals,color='red')
    ax[(1, 1)].set_ylabel(r"$n^{-1} \ \nabla_{\theta} \ x(\hat{\theta}) $")
    # plt.show()

    # ax[(1, 2)].set_title("Sign gradient sig-system")
    y_vals = [np.sign(np.mean(np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1))) for x_val in x_range]
    ax[(1,2)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,2)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,2)].plot(x_range, y_vals,color='red')
    ax[(1, 2)].set_ylabel(r"$ n^{-1} \ sign (\nabla_{\theta} \ x(\hat{\theta})) $")

    # Hessian sigmoid system
    # ax[(1, 3)].set_title("Hessian sigmoids")
    y_vals = [np.mean(q*hessian_sigmoid(theta = x_val, deltatt = deltatt)) for x_val in x_range]
    # y_vals = np.sign(y_vals)
    ax[(1,3)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,3)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,3)].plot(x_range, y_vals,color='red',)
    ax[(1,3)].set_ylabel(r"$ n^{-1} \ \nabla^2_{\theta} \ x(\hat{\theta}) $")


    # # Plot third derivative
    # x_range = np.arange(-2,2,0.01)
    # y_vals = [np.sum(diff3_sigmoid(theta = x_val, deltatt = deltatt)) for x_val in x_range]
    # plt.title("third derivative")
    # plt.plot(x_range, y_vals,color='red',)
    # plt.show()

    # Plot Bordered hessian
    # ax[(1, 1)].set_title("Bordered Hessian")
    # y_vals = [np.mean(bordered_hessian_sigmoid(theta = x_val, deltatt = deltatt, alpha_bh = alpha_bh)) for x_val in x_range]
    # ax[(1,1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(1,1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(1,1)].plot(x_range, y_vals,color='red',)

    # plt.close()

    # set labels
    plt.setp(ax[-1, :], xlabel=r"$\hat{\theta}$")
    # plt.setp(ax[:, 0], ylabel=r"$\theta_i$")

    lines, labels = [], []
    for axi in fig.get_axes():
        # axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0, 0))
        axi.yaxis.set_major_formatter(yfmt)
        linei, labeli = axi.get_legend_handles_labels()
        lines = linei + lines
        labels = labeli + labels

    # plt.legend(lines, labels, loc='upper center', ncol=3
    #                    , bbox_to_anchor=[0.52, -0.45]
    #                    , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))
    #
    # plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()
    fig.savefig(folder_plots + '/' + subfolder + '/' + filename  + ".pdf", pad_inches=0.1, bbox_inches="tight")

plot_quasiconvexity(filename  = 'quasiconvexity', subfolder = "quasiconvexity", deltatt = deltatt, linkflow = linkflow ,q = q, alpha_bh = 1*alpha_bh, theta_true = theta_true, x_range = np.arange(-2, 2, 0.1))


def plot_quasiconvexity_l2norm(filename, subfolder, x_range, theta_true, deltatt, linkflow, q, alpha_bh):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle("Analysis of  (strict) quasiconvexity of L2 norm"
                 "\n(theta_true = " + str(theta_true) + ")")

    # Plot objective function over an interval
    # ax[(0, 0)].set_title("\n\nObj. function (L2)")
    ax[(0, 0)].set_title("\n\n")
    y_vals = [np.mean(objective_function(theta=x_val, q=q, deltatt=deltatt, linkflow=linkflow)) for x_val in x_range]
    ax[(0, 0)].plot(x_range, y_vals, color='red')
    ax[(0, 0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0, 0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0, 0)].set_ylabel(r"$n^{-1} \  ||x(\hat{\theta})-\bar{x}||_2^2$")
    ax[(0, 0)].set_xticklabels([])

    # r"$\hat{\theta}$"

    # ax[(0, 1)].set_title("Gradient L2-norm")
    # y_vals = [np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
    y_vals = [np.mean(gradients_l2norm(theta=x_val, deltatt=deltatt, q=q, linkflow=linkflow)) for x_val in x_range]
    ax[(0, 1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0, 1)].plot(x_range, y_vals, color='red')
    ax[(0, 1)].set_ylabel(r"$n^{-1} \ \nabla_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2)$")
    ax[(0, 1)].set_xticklabels([])

    # ax[(0, 2)].set_title("Sign Gradient L2-norm")
    # y_vals = [np.sign(np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1))) for x_val in x_range]
    y_vals = [np.mean(gradients_l2norm(theta=x_val, deltatt=deltatt, q=q, linkflow=linkflow)) for x_val in x_range]
    y_vals = np.sign(y_vals)
    ax[(1, 0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1, 0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1, 0)].plot(x_range, y_vals, color='red')
    ax[(1, 0)].set_ylabel(r"$n^{-1} \  sign (\nabla_{\theta} ||x(\hat{\theta})-\bar{x}||_2^2 )$")
    # ax[(1, 0)].set_xticklabels([])

    # Hessian L2-norm
    # ax[(1, 3)].set_title("Hessian L2 norm")

    # J = gradients_l2norm(theta, deltatt, q, linkflow)
    # H = np.sum(q * hessian_sigmoid(theta=theta, deltatt=deltatt), axis=1)
    # R = np.sum(objective_function_sigmoids_system(theta, q, deltatt), axis=1) - linkflow.T

    # [np.sum(q * hessian_sigmoid(theta=x_val, deltatt=deltatt), axis=1) for x_val in x_range]

    y_vals = [np.mean(hessian_l2norm(theta=x_val, deltatt=deltatt, q=q, linkflow=linkflow)) for x_val in x_range]
    # y_vals = np.sign(y_vals)
    ax[(1, 1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1, 1)].plot(x_range, y_vals, color='red', )
    ax[(1, 1)].set_ylabel(r"$n^{-1} \ \nabla^2_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2) $")

    # ax[(0, 2)].set_title("Hessian L2-norm")
    # y_vals = [np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
    # ax[(0,1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(0,1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(0,1)].plot(x_range, y_vals,color='red')

    # #Plot sigmoid system
    # ax[(1,0)].set_title("L1 norm")
    # y_vals = [np.mean(np.abs(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)) for x_val in x_range]
    # ax[(1,0)].plot(x_range, y_vals,color = 'red')
    # ax[(1,0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(1,0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(1,0)].set_title("Sigmoid system")
    # ax[(0, 0)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax[(0, 1)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax[(1, 0)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax[(1, 1)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    lines, labels = [], []
    for axi in fig.get_axes():
        # axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0, 0))
        axi.yaxis.set_major_formatter(yfmt)
        linei, labeli = axi.get_legend_handles_labels()
        lines = linei + lines
        labels = labeli + labels

    # set labels
    plt.setp(ax[-1, :], xlabel=r"$\hat{\theta}$")
    # plt.setp(ax[:, 0], ylabel=r"$\theta_i$")

    plt.show()
    fig.savefig(folder_plots + '/' + subfolder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

plot_quasiconvexity_l2norm(filename = 'quasiconvexity_l2norm', subfolder = "quasiconvexity", deltatt = deltatt, linkflow = linkflow ,q = q, alpha_bh = 1*alpha_bh, theta_true = theta_true, x_range = np.arange(-2, 2, 0.1))

#### Analysis of optimization algorithms ###

def plot_optimization_algorithms(filename, subfolder, theta_true, theta_0, thetas, times, iters, batch_size  = batch_size,  n = n):

    fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(6,7))
    fig.suptitle("Analysis of first order algs for (strict) quasiconvex opt"
                 "\n (theta_true = " + str(theta_true) + ", theta_0 = " + str(theta_0)
                 + ", n = " + str(n) + ", b = " + str(batch_size)
                 + ")")
    x_vals = np.arange(0, iters+1)

    ax[(0,0)].set_title("GD vs SGD")
    ax[(0,0)].axhline(y=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0,0)].plot(times['gd'], thetas['gd'], color='red', label = "GD", linestyle='solid')
    ax[(0,0)].plot(times['sgd'], thetas['sgd'],color='blue', label = "SGD", linestyle='solid')
    plt.setp(ax[(0,0)], xlabel= "Computation time [s]")

    ax[(0, 1)].set_title("\n\n GD vs SGD")
    ax[(0, 1)].axhline(y=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0, 1)].plot(x_vals, thetas['gd'], color='red', label="GD", linestyle='solid')
    ax[(0, 1)].plot(x_vals, thetas['sgd'], color='blue', label="SGD", linestyle='solid')
    plt.setp(ax[(0, 1)], xlabel="Iterations")

    ax[(1,0)].set_title("Normalized gradients")
    ax[(1,0)].axhline(y=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,0)].plot(x_vals, thetas['ngd'],color='red', label = "NGD", linestyle='dashed')
    ax[(1,0)].plot(x_vals, thetas['nsgd'], color='blue', label = "NSGD", linestyle='dashed')
    plt.setp(ax[(1, 0)], xlabel="Iterations")

    ax[(1,1)].set_title("Normalized gradients + GD")
    ax[(1,1)].axhline(y=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,1)].plot(x_vals, thetas['ngd_gd'],color='red', label = "NGD-GD", linestyle='dotted')
    ax[(1,1)].plot(x_vals, thetas['nsgd_gd'],color='blue', label = "NSGD-GD", linestyle='dotted')
    plt.setp(ax[(1, 1)], xlabel="Iterations")

    lines1, labels1 = ax[(0,0)].get_legend_handles_labels()
    # lines2, labels2 = ax[(0,1)].get_legend_handles_labels()
    lines3, labels3 = ax[(1,0)].get_legend_handles_labels()
    lines4, labels4 = ax[(1,1)].get_legend_handles_labels()
    lines, labels = lines1 + lines3 + lines4, labels1 + labels3 + labels4

    # set y labels
    # plt.setp(ax[:, 0], ylabel=r"$\theta_i$")
    plt.setp(ax[0 :], ylabel=r"$\hat{\theta}$")
    plt.setp(ax[-1, :], ylabel=r"$\hat{\theta}$")

    # fig.subplots_adjust(bottom=0.8)

    fig.tight_layout()

    plt.legend(lines, labels, loc='upper center', ncol=3
                       , bbox_to_anchor=[0.3, -0.55, 0.5, 0.2]
                       , mode="expand"
                       # , borderaxespad=0
                       , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

    # fig.subplots_adjust(right=0.01)
    # plt.legend(lines, labels, loc=4, title=r'$\eta$')
    # plt.subplots_adjust(bottom = 0.7, wspace=0)
    # fig.subplots_adjust(top=0.9, left=0.1, right=0.2, bottom=0.6, wspace=0, hspace=0)  # create some space below the plots by increasing the bottom-value
    # fig.tight_layout()
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # create some space below the plots by increasing the bottom-value
    # ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)

    # fig.savefig(folder_plots + '/quasiconvexity.png')

    plt.show()
    fig.savefig(folder_plots + '/' + subfolder + '/' +  filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

# - Under initial theta lower than true value
thetas_a, grads_a, times_a = theta_estimation(theta_0 = theta_true - 1, iters = iters, eta_gd = eta, eta_n = eta_n, batch_size = batch_size,  linkflow = linkflow, q = q, deltatt = deltatt, gamma = 1)
# - Under initial theta higher than true value
thetas_b, grads_b, times_b = theta_estimation(theta_0 = theta_true + 1, iters = iters, eta_gd = eta, eta_n = eta_n, batch_size = batch_size,  linkflow = linkflow, q = q, deltatt = deltatt, gamma = 1)

# Plotting
plot_optimization_algorithms(theta_true = theta_true, theta_0 =theta_true - 1, times = times_a, thetas = thetas_a, iters = iters, filename = "optimization_theta0a", subfolder =  '/first-order-opt-performance')
plot_optimization_algorithms(theta_true = theta_true, theta_0 = theta_true + 1, times = times_b, thetas = thetas_b, iters = iters, filename = "optimization_theta0b", subfolder =  '/first-order-opt-performance')


def plot_optimization_performance(filename, subfolder, theta_true, theta_0, thetas, times, iters, batch_size  = batch_size,  n = n):

    fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(6,7))
    fig.suptitle("Analysis of first order algs for (strict) quasiconvex opt"
                 "\n (theta_true = " + str(theta_true) + ", theta_0 = " + str(theta_0)
                 + ", n = " + str(n) + ", b = " + str(batch_size)
                 + ")")
    x_vals = np.arange(0, iters+1)

    # Case a (theta_0 = -2)

    # times
    times_aggregated = np.array([np.array(list(times['0a'].values())).flatten(), np.array(list(times['0b'].values())).flatten()]).flatten()

    min_time, max_time = np.min(times_aggregated), np.max(times_aggregated)
    time_ticks = np.arange(min_time,max_time+0.19,0.2)

    ax[(0,0)].set_title("\n\n\n" + r"$\theta_{0} = -2$")
    ax[(0,0)].axhline(y=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0,0)].plot(times['0a']['gd'], thetas['0a']['gd'], color='red', label = "GD", linestyle='solid')
    ax[(0,0)].plot(times['0a']['sgd'], thetas['0a']['sgd'],color='blue', label = "SGD", linestyle='solid')
    ax[(0,0)].plot(times['0a']['ngd'], thetas['0a']['ngd'],color='red', label = "NGD", linestyle='dashed')
    ax[(0,0)].plot(times['0a']['nsgd'], thetas['0a']['nsgd'], color='blue', label = "NSGD", linestyle='dashed')
    ax[(0, 0)].set_xticks(time_ticks)
    # plt.setp(ax[(0,0)], xlabel= "Computation time [s]")

    # ax[(0, 1)].set_title("\n\n GD vs SGD")
    ax[(1, 0)].axhline(y=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1, 0)].plot(times['0a']['gd'], thetas['0a']['gd'], color='red', label="GD", linestyle='solid')
    ax[(1, 0)].plot(times['0a']['sgd'], thetas['0a']['sgd'], color='blue', label="SGD", linestyle='solid')
    ax[(1, 0)].plot(times['0a']['ngd_gd'], thetas['0a']['ngd_gd'], color='red', label="NGD-GD", linestyle='dotted')
    ax[(1, 0)].plot(times['0a']['nsgd_gd'], thetas['0a']['nsgd_gd'], color='blue', label="NSGD-GD", linestyle='dotted')
    ax[(1, 0)].set_xticks(time_ticks)
    plt.setp(ax[(1,0)], xlabel= "Computation time [s]")
    # plt.setp(ax[(1, 0)], xlabel="Computation time [s]"

    # Case b (theta_0 = 0) 
    
    # ax[(0,1)].set_title("\n\n")
    ax[(0, 1)].set_title("\n\n\n" + r"$\theta_{0} = 0$")
    ax[(0,1)].axhline(y=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(0,1)].plot(times['0b']['gd'], thetas['0b']['gd'], color='red', label = "GD", linestyle='solid')
    ax[(0,1)].plot(times['0b']['sgd'], thetas['0b']['sgd'],color='blue', label = "SGD", linestyle='solid')
    ax[(0,1)].plot(times['0b']['ngd'], thetas['0b']['ngd'],color='red', label = "NGD", linestyle='dashed')
    ax[(0,1)].plot(times['0b']['nsgd'], thetas['0b']['nsgd'], color='blue', label = "NSGD", linestyle='dashed')
    ax[(0, 1)].set_xticks(time_ticks)

    # ax[(0, 1)].set_title("\n\n GD vs SGD")
    ax[(1,1)].axhline(y=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,1)].plot(times['0b']['gd'], thetas['0b']['gd'], color='red', label="GD", linestyle='solid')
    ax[(1,1)].plot(times['0b']['sgd'], thetas['0b']['sgd'], color='blue', label="SGD", linestyle='solid')
    ax[(1,1)].plot(times['0b']['ngd_gd'], thetas['0b']['ngd_gd'], color='red', label="NGD-GD", linestyle='dotted')
    ax[(1,1)].plot(times['0b']['nsgd_gd'], thetas['0b']['nsgd_gd'], color='blue', label="NSGD-GD", linestyle='dotted')
    ax[(1, 1)].set_xticks(time_ticks)
    plt.setp(ax[(1, 1)], xlabel="Computation time [s]")
    # plt.setp(ax[(1,1)], xlabel="Iterations")
    
    
    # set y labels
    # plt.setp(ax[:, 0], ylabel=r"$\theta_i$")
    plt.setp(ax[0:], ylabel=r"$\hat{\theta}$")
    plt.setp(ax[-1, :], ylabel=r"$\hat{\theta}$")

    # Legend
    # lines, labels = ax[(0,0)].get_legend_handles_labels()
    # lines, labels = ax[(0, 0)].get_legend_handles_labels()

    lines, labels = [], []
    for axi in fig.get_axes():
        # axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        linei, labeli = axi.get_legend_handles_labels()
        lines = linei + lines
        labels = labeli + labels

    unique = [(h, l) for i, (h, l) in enumerate(zip(lines, labels)) if l not in labels[:i]]
    lines,labels = zip(*unique)

    sorted_idx =  []
    for i in ['GD','SGD', 'NGD', 'NSGD', 'NGD-GD', 'NSGD-GD']:
        sorted_idx.append(labels.index(i))

    lines = [lines[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]

    # fig.subplots_adjust(bottom=0.8)

    fig.tight_layout()

    plt.legend(lines, labels, loc='upper center', ncol=3
                       , bbox_to_anchor=[0.3, -0.55, 0.5, 0.2]
                       , mode="expand"
                       # , borderaxespad=0
                       , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

    # fig.subplots_adjust(right=0.01)
    # plt.legend(lines, labels, loc=4, title=r'$\eta$')
    # plt.subplots_adjust(bottom = 0.7, wspace=0)
    # fig.subplots_adjust(top=0.9, left=0.1, right=0.2, bottom=0.6, wspace=0, hspace=0)  # create some space below the plots by increasing the bottom-value
    # fig.tight_layout()
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # create some space below the plots by increasing the bottom-value
    # ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)

    # fig.savefig(folder_plots + '/quasiconvexity.png')

    plt.show()
    fig.savefig(folder_plots + '/' + subfolder + '/' +  filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

# Plotting
plot_optimization_performance(theta_true = theta_true, theta_0 =theta_true - 1, thetas = {'0a': thetas_a, '0b': thetas_b}, times = {'0a': times_a, '0b': times_b}, iters = iters, filename = "optimization_performance", subfolder =  '/first-order-opt-performance')


### INFERENCE PLOTS ###

def inference_sigmoid(theta_true, theta_0, theta_h0,  reps, iters, noise_factor, n, m, eta_gd, eta_n, batch_size, alpha, gamma = 1):

    results_reps = {}
    algorithms = ['gd','sgd','ngd','nsgd','ngd_gd','nsgd_gd']

    # thetas_df = pd.DataFrame()
    # thetas_df['alg'] = sum([['gd', 'sgd', 'ngd', 'nsgd', 'ngd_gd', 'nsgd_gd'] for rep in np.arange(0, reps)], [])
    # thetas_df['rep'] = sum([list(np.arange(0,reps)) for algorithm in algorithms],[])

    thetas_df = pd.DataFrame(columns=['rep', 'alg', 'theta_true','theta_h0','theta_0','theta', 'time', 'ttest','criticalttest', 'pvalue', 'width_confint', 'bias','fn', 'fp', 'fnfp', 'grad_obj', 'grad_m'])
    # thetas_df.loc[3] = [0,2,4]

    count = 0
    for rep in np.arange(0,reps):
        
        deltatt, q, linkflow = simulate_sigmoid_system(theta_true = theta_true, noise_factor = noise_factor, n = n, m = m)

        thetas, grads, times = theta_estimation(theta_0 = theta_0, iters = iters, eta_gd = eta_gd, eta_n = eta_n, batch_size = batch_size,  linkflow = linkflow, q = q, deltatt = deltatt,  gamma = 1)

        for alg in algorithms:

            ttest, critical_ttest, pvalue = ttest_theta(theta=thetas[alg][-1], theta_h0=theta_h0, linkflow = linkflow, q=q, deltatt=deltatt, n=n, p=1, alpha=alpha)
            confint, width_confint = confint_theta(theta=thetas[alg][-1], n=n, q=q, deltatt=deltatt, linkflow=linkflow)
            bias = thetas[alg][-1]-theta_true

            F = gradient_sigmoids_system(theta=thetas[alg][-1], q =  q, deltatt = deltatt)
            grad_m = np.mean(F)
            grad_obj = np.mean(gradients_l2norm(theta = thetas[alg][-1], deltatt = deltatt, q = q, linkflow = linkflow))
            # grads[alg][-1]

            # False positives or negatives:
            fn, fp = 0, 0
            # fnfp = 0

            if theta_true == theta_h0:

                if pvalue < alpha:
                    fn = 1

            else:
                if pvalue > alpha:
                    fp = 1

            fnfp = fn + fp

            thetas_df.loc[count] = [rep, alg, theta_true, theta_h0, theta_0, thetas[alg][-1], times[alg][-1],ttest, critical_ttest, pvalue, width_confint, bias, fn, fp, fnfp, grad_obj, grad_m] #,thetas['sgd'][-1], thetas['ngd'][-1], thetas['ngd'][-1], thetas['nsgd'][-1], thetas['ngd_gd'][-1], thetas['ngsd_gd'][-1]
            count += 1

        results_reps[rep] = thetas.copy()

    # print(results_reps[0]['gd'])

    return thetas_df

# Set theta_true = 0 to analyze false negatives.

reps = 10
iters = 60
theta_true = -1#-1 #-0.5 If it is different than 0, then the false positive are accounted for. Otherwise, false negatives are shown in plot below.
theta_0 = -3
theta_h0 = 0 #
alpha = 0.05
n = 50
m = 4
eta_gd = 3e-5
eta_n = 3e-1
batch_size = int(n * 0.5)
noise_factor = 10

def plot_inference(subfolder, filename, inference_pd, noise_factor = noise_factor, batch_size  = batch_size,  n = n):
    # fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(10,5))
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("Analysis of inductive bias and inference"
                 "\n (theta_true = " + str(int(np.mean(inference_pd['theta_true']))) + ", theta_0 = " + str(int(np.mean(inference_pd['theta_0']))) + ", theta_{h_0} = " + str(int(np.mean(inference_pd['theta_h0'])))
                 + ", n = " + str(n) + ", b = " + str(batch_size) + ", sigma = " + str(noise_factor)
                 + ")")

    ax[(0,0)].set_title("\n\n")
    sns.barplot(x="alg", y="time", data=inference_pd
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax = ax[(0,0)])
    ax[(0, 0)].set_ylabel('comp. time [s]')
    ax[(0, 0)].set_xticklabels([])


    # # ax[(0,1)].set_title("Estimate")
    # ax[(0,1)] = sns.boxplot(x="alg", y="theta", data=inference_pd
    #                         # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
    #             , ax = ax[(0,1)], showfliers=False )
    # ax[(0,1)].axhline(y=theta_true, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(0, 1)].set_ylabel(r"$\hat{\theta}$")
    # ax[(0, 1)].set_xticklabels([])
    
    # ax[(0,3)].set_title("Bias")
    sns.boxplot(x="alg", y="bias", data=inference_pd
                            # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax = ax[(0,1)], showfliers=False )
    ax[(0, 1)].set_ylabel("Bias")
    ax[(0, 1)].set_xticklabels([])
    ax[(0, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)

    # ax[(0,3)].set_title("Width CI")
    sns.boxplot(x="alg", y="width_confint", data=inference_pd
                            # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax = ax[(0,2)], showfliers=False )
    ax[(0, 2)].set_ylabel("CI width")
    ax[(0, 2)].set_xticklabels([])
    ax[(0, 2)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)

    # ax[(0,1)].set_title("critical t-values")
    # sns.barplot(x="alg", y="criticalttest", data=inference_plotdata,
    #                  linewidth=2.5, facecolor=(1, 1, 1, 0),
    #                  errcolor=".2", edgecolor=".2", ax = ax[(0,1)] )

    # ax[(1,0)].set_title("t-test values")
    sns.boxplot(x="alg", y="ttest", data=inference_pd
                            # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax = ax[(1,0)], showfliers=False )
    ax[(1,0)].axhline(y=np.mean(inference_pd['criticalttest']), color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,0)].set_ylabel("t test")
    # ax[(1,0)].set_yticklabels([])

    # When the null hypothesis is true, it is straighforward that the distirbution of the p-value is uniform [0,1] and thus, the mean should be at 0.5
    # https://stats.stackexchange.com/questions/10613/why-are-p-values-uniformly-distributed-under-the-null-hypothesis#:~:text=The%20p%2Dvalue%20is%20uniformly,all%20other%20assumptions%20are%20met.&text=We%20want%20the%20probability%20of,comes%20from%20a%20uniform%20distribution.

    # ax[(1,1)].set_title("pvalues")
    sns.boxplot(x="alg", y="pvalue", data=inference_pd
                            # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax = ax[(1,1)])
    ax[(1,1)].axhline(y=0.05, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1, 1)].set_ylabel("p value")
    # ax[(1, 1)].set_yticklabels([])

    # if theta_true == 0:
    #     ax[(1,2)].set_title("False negatives") #Depending if theta_true is 0 or is not 0
    # else:
    #     ax[(1,2)].set_title("False positives")

    sns.barplot(x="alg", y="fnfp", data=inference_pd
                            # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax = ax[(1,2)])#, showfliers=False )

    if np.mean(inference_pd['theta_true']) == 0:
        ax[(1,2)].set_ylabel("false negatives") #Depending if theta_true is 0 or is not 0
    else:
        ax[(1,2)].set_ylabel("false positives")

    # ax[(1, 3)].set_axis_off()

    # ax[(1,2)].axhline(y=0.05, color='black', linestyle='dashed', linewidth=0.5)


    # ax[(1,0)].set_title("Gradient")
    # ax[(1,0)] = sns.barplot(x="alg", y="grads", data=inference_plotdata,
    #                  linewidth=2.5, facecolor=(1, 1, 1, 0),
    #                  errcolor=".2", edgecolor=".2")

    plt.setp(ax[-1, :], xlabel="algorithm")
    plt.setp(ax[0, :], xlabel="")

    for axi in fig.get_axes():
        plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)
        # axi.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.show()

    fig.savefig(folder_plots + '/' + subfolder + '/' +  filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

inference_pd_0a = inference_sigmoid(reps = reps, iters = iters, noise_factor = noise_factor, n = n, m = m, eta_gd = eta_gd, eta_n = eta_n
                                 , theta_true = theta_true , theta_0 = theta_true-1, theta_h0 = theta_h0
                                 , batch_size = batch_size, alpha = alpha)

inference_pd_0b = inference_sigmoid(reps = reps, iters = iters, noise_factor = noise_factor, n = n, m = m, eta_gd = eta_gd, eta_n = eta_n
                                 , theta_true = theta_true , theta_0 = theta_true+1, theta_h0 = theta_h0
                                 , batch_size = batch_size, alpha = alpha)

inference_pd_1a = inference_sigmoid(reps = reps, iters = iters, noise_factor = noise_factor, n = n, m = m, eta_gd = eta_gd, eta_n = eta_n
                                 , theta_true = 0*theta_true , theta_0 = 0*theta_true-1, theta_h0 = theta_h0
                                 , batch_size = batch_size, alpha = alpha)

inference_pd_1b = inference_sigmoid(reps = reps, iters = iters, noise_factor = noise_factor, n = n, m = m, eta_gd = eta_gd, eta_n = eta_n
                                 , theta_true = 0*theta_true , theta_0 = 0*theta_true+1, theta_h0 = theta_h0
                                 , batch_size = batch_size, alpha = alpha)



plot_inference(subfolder = "inference", filename = "inference_a", inference_pd = inference_pd_0a)
plot_inference(subfolder = "inference", filename = "inference_b", inference_pd = inference_pd_0b)
plot_inference(subfolder = "inference", filename = "inference_c", inference_pd = inference_pd_1a)
plot_inference(subfolder = "inference", filename = "inference_d", inference_pd = inference_pd_1b)

# Plot with false negative and positives in each case 

def plot_inference_false_posneg_and_pvals(subfolder, filename, inference_pds, noise_factor=noise_factor,
                   batch_size=batch_size, n=n):
    fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(10,5))
    # fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("Analysis of false negatives and positives \n"
                 + "(n = " + str(n) + ", b = " + str(batch_size) + ", sigma = " + str(noise_factor)
                 + ", h_0 = " + str(int(np.mean(inference_pds['0a']['theta_h0']))) + ")")


    #P values
    ax[(0, 0)].set_title("\n\n" + r"$\theta = -1, \theta_{0} = -2$")
    sns.boxplot(x="alg", y="pvalue", data=inference_pds['0a']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(0, 0)], showfliers=False)
    ax[(0, 0)].set_ylabel('p value')
    ax[(0, 0)].set_xticklabels([])
    ax[(0, 0)].axhline(y=0.05, color='black', linestyle='dashed', linewidth=0.5)

    ax[(0, 1)].set_title(r"$\theta = -1, \theta_{0} = 0$")
    sns.boxplot(x="alg", y="pvalue", data=inference_pds['0b']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(0, 1)], showfliers=False)
    ax[(0, 1)].set_ylabel("")
    ax[(0, 1)].set_xticklabels([])
    ax[(0, 1)].axhline(y=0.05, color='black', linestyle='dashed', linewidth=0.5)

    ax[(0, 2)].set_title(r"$\theta = 0, \theta_{0} = -1$")
    sns.boxplot(x="alg", y="pvalue", data=inference_pds['1a']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(0, 2)], showfliers=False)
    ax[(0, 2)].set_ylabel("")
    ax[(0, 2)].set_xticklabels([])
    ax[(0, 2)].axhline(y=0.05, color='black', linestyle='dashed', linewidth=0.5)

    ax[(0, 3)].set_title(r"$\theta = 0, \theta_{0} = 1$")
    sns.boxplot(x="alg", y="pvalue", data=inference_pds['1b']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(0, 3)], showfliers=False)
    ax[(0, 3)].set_ylabel("")
    # ax[(1, 3)].set_ylabel("False negatives")
    ax[(0, 3)].set_xticklabels([])
    ax[(0, 3)].axhline(y=0.05, color='black', linestyle='dashed', linewidth=0.5)

    # False positives and negatives

    sns.barplot(x="alg", y="fnfp", data=inference_pds['0a']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(1, 0)])
    ax[(1, 0)].set_ylabel('false positives')
    # ax[(1, 0)].set_xticklabels([])

    sns.barplot(x="alg", y="fnfp", data=inference_pds['0b']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(1, 1)])
    ax[(1, 1)].set_ylabel("false positives")
    # ax[(1, 1)].set_xticklabels([])
    # ax[(1, 1)].axhline(y=1, color='black', linestyle='dashed', linewidth=0.5)

    sns.barplot(x="alg", y="fnfp", data=inference_pds['1a']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(1, 2)])
    ax[(1, 2)].set_ylabel("false negatives")
    # ax[(1, 2)].set_xticklabels([])
    # ax[(1, 2)].axhline(y=1, color='black', linestyle='dashed', linewidth=0.5)

    sns.barplot(x="alg", y="fnfp", data=inference_pds['1b']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(1, 3)])
    ax[(1, 3)].set_ylabel("false negatives")
    # ax[(1, 3)].set_xticklabels([])
    # ax[(1, 3)].axhline(y=1, color='black', linestyle='dashed', linewidth=0.5)

    # if theta_true == 0:
    #     ax[(1, 2)].set_ylabel("false negatives")  # Depending if theta_true is 0 or is not 0
    # else:
    #     ax[(1, 2)].set_ylabel("false positives")    #                  errcolor=".2", edgecolor=".2")

    plt.setp(ax[-1, :], xlabel="algorithm")
    plt.setp(ax[0, :], xlabel="")

    for axi in ax[1, :]:
        axi.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axi.set_ylim([0, 1.05])
        plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)

    for axi in ax[0, :]:
        axi.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.show()

    fig.savefig(folder_plots + '/' + subfolder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")


inference_pds = {'0a': inference_pd_0a, '0b': inference_pd_0b, '1a': inference_pd_1a, '1b': inference_pd_1b}

plot_inference_false_posneg_and_pvals(subfolder = "inference", filename = "inference_falseposneg_pvals", inference_pds = inference_pds)

def plot_inference_estimate_and_bias(subfolder, filename, inference_pds, noise_factor=noise_factor,
                                         batch_size=batch_size, n=n):
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
    # fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.suptitle("Analysis of parameter estimates and bias \n"
                 + "(n = " + str(n) + ", b = " + str(batch_size) + ", sigma = " + str(noise_factor)
                 + ")")

    # Estimates
    ax[(0, 0)].set_title("\n\n" + r"$\theta = -1, \theta_{0} = -2$")
    # ax[(0, 0)].set_ylabel('estimate')
    sns.boxplot(x="alg", y="theta", data=inference_pds['0a']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(0, 0)], showfliers=False)
    ax[(0, 0)].axhline(y=np.mean(inference_pds['0a']['theta_true']), color='black', linestyle='dashed', linewidth=0.5)
    ax[(0, 0)].set_xticklabels([])
    ax[(0, 0)].set_ylabel(r"$\hat{\theta}$")

    ax[(0, 1)].set_title(r"$\theta = -1, \theta_{0} = 0$")
    sns.boxplot(x="alg", y="theta", data=inference_pds['0b']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(0, 1)], showfliers=False)
    # ax[(1, 1)].set_ylabel("false positives")
    ax[(0, 1)].axhline(y=np.mean(inference_pds['0b']['theta_true']), color='black', linestyle='dashed', linewidth=0.5)
    ax[(0, 1)].set_xticklabels([])
    ax[(0, 1)].set_ylabel("")
    # ax[(1, 1)].axhline(y=1, color='black', linestyle='dashed', linewidth=0.5)

    ax[(0, 2)].set_title(r"$\theta = 0, \theta_{0} = -1$")
    sns.boxplot(x="alg", y="theta", data=inference_pds['1a']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(0, 2)], showfliers=False)
    # ax[(1, 2)].set_ylabel("false negatives")
    ax[(0, 2)].axhline(y=np.mean(inference_pds['1a']['theta_true']), color='black', linestyle='dashed', linewidth=0.5)
    ax[(0, 2)].set_xticklabels([])
    ax[(0, 2)].set_ylabel("")
    # ax[(1, 2)].axhline(y=1, color='black', linestyle='dashed', linewidth=0.5)

    ax[(0, 3)].set_title(r"$\theta = 0, \theta_{0} = 1$")
    sns.boxplot(x="alg", y="theta", data=inference_pds['1b']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(0, 3)], showfliers=False)
    # ax[(1, 3)].set_ylabel("false negatives")
    ax[(0, 3)].axhline(y=np.mean(inference_pds['1b']['theta_true']), color='black', linestyle='dashed', linewidth=0.5)
    ax[(0, 3)].set_ylabel("")
    ax[(0, 3)].set_xticklabels([])
    # ax[(1, 3)].axhline(y=1, color='black', linestyle='dashed', linewidth=0.5)


    # Bias
    sns.boxplot(x="alg", y="bias", data=inference_pds['0a']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(1, 0)], showfliers=False)
    ax[(1, 0)].set_ylabel('bias '+ r"$(\hat{\theta} - \theta)$")
    # ax[(1, 0)].set_xticklabels([])
    ax[(1, 0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)


    sns.boxplot(x="alg", y="bias", data=inference_pds['0b']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(1, 1)], showfliers=False)
    ax[(1, 1)].set_ylabel("")
    # ax[(1, 1)].set_xticklabels([])
    ax[(1, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)


    sns.boxplot(x="alg", y="bias", data=inference_pds['1a']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(1, 2)], showfliers=False)
    ax[(1, 2)].set_ylabel("")
    # ax[(1, 2)].set_xticklabels([])
    ax[(1, 2)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)

    sns.boxplot(x="alg", y="bias", data=inference_pds['1b']
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(1, 3)], showfliers=False)
    ax[(1, 3)].set_ylabel("")
    # ax[(1, 3)].set_ylabel("False negatives")
    # ax[(1, 3)].set_xticklabels([])
    ax[(1, 3)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)


    plt.setp(ax[-1, :], xlabel="algorithm")
    plt.setp(ax[0, :], xlabel="")

    for axi in ax[0, :]:
        axi.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for axi in ax[1, :]:
        axi.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axi.axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        # axi.set_ylim([0, 1.05])
        plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)

    plt.show()

    fig.savefig(folder_plots + '/' + subfolder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

inference_pds = {'0a': inference_pd_0a, '0b': inference_pd_0b, '1a': inference_pd_1a, '1b': inference_pd_1b}

plot_inference_estimate_and_bias(subfolder="inference", filename="inference_estimates_bias",
                                     inference_pds=inference_pds)

def plot_summary_performance_bias_ciwidth(subfolder, filename, inference_pds):

    algorithms = inference_pds['0a'].alg.unique()
    inference_grouped = {} # dict(zip(algorithms,np.zeros(len(algorithms))))

    inference_grouped = pd.DataFrame(columns=inference_pds['0a'].columns)

    for key in inference_pds.keys():
        inference_grouped = inference_grouped.append(inference_pds[key])

    # for alg in algorithms:
    #     inference_grouped[alg] = pd.DataFrame(columns=inference_pds['0a'].columns)
    #     for key in inference_pds.keys():
    #         inference_grouped[alg] = inference_grouped[alg].append(inference_pds[key][inference_pds[key]['alg'] == alg])

    fig, ax = plt.subplots(nrows=2, ncols=2)

    sns.barplot(x="alg", y="time", data=inference_grouped
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax = ax[(0,0)])
    ax[(0,0)].set_ylabel('comp. time [s]')
    ax[(0,0)].set_xticklabels([])

    sns.barplot(x="alg", y="width_confint", data=inference_grouped
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(0,1)])
    ax[(0,1)].set_ylabel('CI width')
    ax[(0,1)].set_xticklabels([])
    ax[(0,1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    # ax[(0, 2)].set_xlabel("")

    sns.boxplot(x="alg", y="bias", data=inference_grouped
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(1,0)])
    ax[(1, 0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,0)].set_ylabel('Bias')

    sns.barplot(x="alg", y="fnfp", data=inference_grouped
                # ,linewidth=2.5, facecolor=(1, 1, 1, 0),errcolor=".2", edgecolor=".2"
                , ax=ax[(1,1)])
    ax[(1, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
    ax[(1,1)].set_ylabel('false positives/negatives')

    # ax[(2)].set_xticklabels([])

    for axi in fig.get_axes():
        axi.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for axi in fig.get_axes():
        plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)

    plt.setp(ax[-1, :], xlabel="algorithm")
    plt.setp(ax[0, :], xlabel="")

    fig.tight_layout()

    plt.show()

    fig.savefig(folder_plots + '/' + subfolder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")


plot_summary_performance_bias_ciwidth(subfolder="inference", filename="inference_performance_summary",
                                     inference_pds=inference_pds)









# TODO: Implement bisection method by solving a sequence of convex feasibility problems.
#  Look  at Convex Feasibility slides with VIs and Boyd paper on DQCP

#TODO: Interesing that p value is not uniform for NGD without GD. It seems that it is protected against noise.
#TODO: define termination condition for algorthsm. epsilon optimality, i.e. difference between iterations, especially for the normalized case.
# remember that at the last iteration the best optimal solution found so far is used but this is very costly as it is almost a grid search.
#TODO: Implement generalized non linear least squares