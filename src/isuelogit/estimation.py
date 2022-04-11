from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Any

if TYPE_CHECKING:
    from mytypes import Union, Options, Optional, Features, ParametersDict, Dict, List, DataFrame, Feature
    from equilibrium import LUE_Equilibrator

from printer import block_output, printProgressBar, enablePrint, blockPrint, printIterationBar

from sklearn import preprocessing

from abc import ABC, abstractmethod

import numdifftools as nd
import numdifftools.nd_algopy as nda

# import autograd.numpy as np
import numpy as np

from scipy import stats

import copy
import time
import pandas as pd

from networks import TNetwork
from etl import fake_observed_counts, masked_observed_counts, get_informative_links
from descriptive_statistics import summary_links_report, rmse, nrmse

from mytypes import ColumnVector, ParametersDict, Matrix, Proportion
from utils import get_design_matrix, v_normalization, is_pos_def, is_pos_semidef, almost_zero, Options

# https://stackoverflow.com/questions/9777783/suppress-scientific-notation-in-numpy-when-creating-array-from-nested-list
np.set_printoptions(suppress=True, precision=4)


# https://stackoverflow.com/questions/22222818/how-to-printing-numpy-array-with-3-decimal-places

class Parameter:
    def __init__(self,
                 key: str,
                 type: str = None,
                 sign: str = None,
                 fixed: bool = False,
                 initial_value: float = None,
                 true_value: float = None,
                 ):
        self._key = key
        self._type = type
        self._sign = sign
        self._fixed = fixed
        self._initial_value = initial_value
        self._true_value = true_value
        self._value = None

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        self._key = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def sign(self):
        return self._sign

    @sign.setter
    def sign(self, value):
        self._sign = value

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, value):
        self._fixed = value

    @property
    def initial_value(self):
        return self._initial_value

    @initial_value.setter
    def initial_value(self, value):
        self._initial_value = value

    @property
    def true_value(self):
        return self._true_value

    @true_value.setter
    def true_value(self, value):
        self._true_value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class UtilityFunction:

    # def __init__(self, parameters: Parameters):
    # 
    #     self.parameters = parameters
    # 
    #     # Set initial values to zero if no value has been provided
    #     self.default_zero_initializer(features=parameters.features)

    def __init__(self,
                 features_Y: Features,
                 features_Z: Features = None,
                 signs: Optional[ParametersDict] = None,
                 fixed: Optional[ParametersDict] = None,
                 initial_values: Optional[ParametersDict] = None,
                 true_values: Optional[ParametersDict] = None
                 ):
        """ Inits LUE_parameters class

        Args:
            features_Y: list of names of endogenous features
            features_Z: list of names of exogenous features

        """

        self._features_Y = features_Y
        self._features_Z = features_Z
        self._initial_values = initial_values
        self._true_values = true_values
        self._signs = signs
        self._fixed = fixed

        # Dict[Feature,Parameter]
        self._parameters = {}

        # Create parameters
        self.add_features(features_Y=self._features_Y,
                          features_Z=self._features_Z,
                          fixed=fixed,
                          signs=signs,
                          initial_values=initial_values,
                          true_values=true_values)

        # self._true_values = dict.fromkeys(self.features_Y + self.features_Z)

        # if true_values is not None:
        #     self._true_values = true_values

        # Dictionary with initial values.
        # self._initial_values = dict.fromkeys(features_Y + features_Z)
        # self.initial_values = initial_values

        # # Dictionary with current values. Default values are set to None
        # self._values = dict.fromkeys(features_Y + features_Z)

    def add_features(self,
                     features_Y: Optional[List[str]] = None,
                     features_Z: Optional[List[str]] = None,
                     signs: Optional[ParametersDict] = None,
                     fixed: Optional[ParametersDict] = None,
                     initial_values: Optional[ParametersDict] = None,
                     true_values: Optional[ParametersDict] = None
                     ):

        if features_Y is None:
            features_Y = []

        if features_Z is None:
            features_Z = []

        # self.features_Z.extend(features_Z)
        # self.features_Y.extend(features_Y)

        # The values of the dictionary with initial and true values, and signs of parameters are set to None

        if signs is None:
            signs = dict.fromkeys(features_Y + features_Z)

        if fixed is None:
            fixed = dict.fromkeys(features_Y + features_Z, False)

        if initial_values is None:
            initial_values = dict.fromkeys(features_Y + features_Z)

        if true_values is None:
            true_values = dict.fromkeys(features_Y + features_Z)

        for feature in features_Y:
            self.parameters[feature] = Parameter(key=feature,
                                                 type='Y',
                                                 sign=signs.get(feature),
                                                 fixed=fixed.get(feature, False),
                                                 initial_value=initial_values.get(feature),
                                                 true_value=true_values.get(feature))

        for feature in features_Z:
            self.parameters[feature] = Parameter(key=feature,
                                                 type='Z',
                                                 sign=signs.get(feature),
                                                 fixed=fixed.get(feature, False),
                                                 initial_value=initial_values.get(feature),
                                                 true_value=true_values.get(feature))

        # Initial and values are set to 0
        self.default_zero_initializer(features=features_Z + features_Y)

        # # Initialize true values and current values with None
        # self.true_values = dict.fromkeys(Z + Y)

    def add_sparse_features(self,
                            Z: Optional[List[str]] = [],
                            Y: Optional[List[str]] = []):

        self.add_features(features_Z=Z, features_Y=Y)
        self.true_values = dict.fromkeys(Z + Y, 0)

    @property
    def features_Z(self):
        return [key for key, parameter in self.parameters.items() if parameter.type == 'Z']
        # return self._features_Z

    @features_Z.setter
    def features_Z(self, value):
        if value is None:
            self._features_Z = []

    @property
    def features_Y(self):
        return [key for key, parameter in self.parameters.items() if parameter.type == 'Y']
        # return self._features_Y

    @features_Y.setter
    def features_Y(self, value):
        if value is None:
            self._features_Y = []

    @property
    def features(self):
        return self.features_Y + self.features_Z

    @property
    def initial_values(self) -> Dict[str, float]:
        return {key: parameter.initial_value for key, parameter in self.parameters.items()}
        # return self.parameters.initial_values

    @initial_values.setter
    def initial_values(self, values: Dict[Feature, float]):
        if values is not None:
            for feature, value in values.items():
                if feature in self.parameters.keys():
                    self.parameters[feature].initial_value = value

    @property
    def values(self):
        return {key: parameter.value for key, parameter in self.parameters.items()}
        # return self.parameters.values

    @values.setter
    def values(self, values: Dict[Feature, float]):
        if values is not None:
            for feature, value in values.items():
                if feature in self.parameters.keys():
                    self.parameters[feature].value = values[feature]

        # self.parameters.values = parameters_values

    @property
    def true_values(self):
        return {key: parameter.true_value for key, parameter in self.parameters.items()}
        # return self._true_values

    @true_values.setter
    def true_values(self, values: Dict[Feature, float]):
        if values is not None:
            for feature, value in values.items():
                if feature in self.parameters.keys():
                    self.parameters[feature].true_value = values[feature]
        # self.parameters.true_values = parameters_values

    @property
    def signs(self):
        return self._signs

    @signs.setter
    def signs(self, values: Dict[Feature, str]):

        if values is not None:
            for feature, value in values.items():
                value = values[feature]
                assert value in ['-', '+']
                self.parameters[feature].sign = values[feature]

    @property
    def fixed(self):
        return {key: parameter.fixed for key, parameter in self.parameters.items()}

    @fixed.setter
    def fixed(self, values: Dict[Feature, str]):

        if values is not None:
            for feature, value in values.items():
                value = values[feature]
                assert value is bool
                self.parameters[feature].fixed = values[feature]

    @property
    def parameters(self):
        return self._parameters

    # @parameters.setter
    # def parameters(self, value):
    #     self._parameters = value

    def random_initializer(self, range):
        '''Randomly initialize values of the utility parameters based on true values and range'''

        assert isinstance(range, tuple), 'range must be a tuple'

        initial_theta_values = []

        for feature in self.features:
            initial_theta_values.append(float(self.true_values[feature]))

        random_utility_values = np.array(initial_theta_values) + np.random.uniform(
            *range, len(self.features))

        self.initial_values = dict(zip(self.features, random_utility_values))

        # print('New initial values', self.utility_function.initial_values)

        return self.initial_values

    def constant_initializer(self, value):
        '''Randomly initialize values of the utility parameters based on true values and range'''

        self.initial_values = dict.fromkeys(self.features, value)

        # print('New initial values', self.utility_function.initial_values)

        #return self.initial_values

    def zero_initializer(self):
        '''Randomly initialize values of the utility parameters based on true values and range'''

        self.constant_initializer(0)

        #return self.initial_values

    def default_zero_initializer(self, features):
        '''Set the initial value to zero for all parameters that have initial value None'''

        if features is None:
            features = self.features

        for feature in features:
            if self.initial_values.get(feature) is None:
                self.initial_values = {**self.initial_values, **{feature: 0}}

            if self.values.get(feature) is None:
                self.values = {**self.values, **{feature: 0}}


class OuterMethod(ABC):

    def __init__(self,
                 key: str = None,
                 iters: int = 0):
        self.type = None
        self.key = key
        self.iters = iters

        self.options = Options()

        # Momentum
        self.options['gamma'] = 0

    @abstractmethod
    def update_parameters(self, **kwargs):
        pass


class FirstOrderMethod(OuterMethod):

    def __init__(self, eta, **kwargs):
        super().__init__(**kwargs)

        # Learning rate in first order optimization methods
        self.eta = eta

        # Past gradient for momentum
        self.grad_old = None

        # Parameters for adagrad and adam
        self.beta_1 = 0.9
        self.beta_2 = 0.99

        self.type = 'first-order'

    def update_parameters(self, theta, gradient, features_idxs=None):
        raise NotImplementedError

    def reset_gradients(self):
        # Accumulator of gradients for adagrad
        self.acc_grads = np.zeros(len(self.theta_0))[:, np.newaxis]

        # First (m) and second moments (v) accumulators for adam
        self.acc_m = np.zeros(len(self.theta_0))[:, np.newaxis]
        self.acc_v = np.zeros(len(self.theta_0))[:, np.newaxis]


class NormalizedGradientDescent(FirstOrderMethod):
    def __init__(self, **kwargs):
        super().__init__(key='ngd', **kwargs)

    def update_parameters(self, theta, gradient, features_idxs=None):

        epsilon = 1e-12

        # grad_adj =  numeric.almost_zero_gradient(grad_adj)

        if features_idxs is not None:
            previous_theta = theta.copy()

        if len(theta) == 1:
            theta = theta - np.sign(gradient) * self.eta
        else:
            theta = theta - (gradient) / np.linalg.norm(
                gradient + epsilon) * self.eta  # epsilon to avoid problem when gradient is 0

        if features_idxs is not None:
            for feature_idx in np.arange(theta.size):
                if feature_idx not in features_idxs:
                    theta[feature_idx] = previous_theta[feature_idx]

        return theta


class GradientDescent(FirstOrderMethod):
    def __init__(self, **kwargs):
        super().__init__(key='gd', **kwargs)

    def update_parameters(self, theta, gradient, features_idxs=None):
        # theta_update_new = gamma * (theta_update_old) + eta * grad_new

        if features_idxs is not None:
            previous_theta = theta.copy()

        # Gradient update (with momentum)
        theta = theta - gradient * self.eta

        if features_idxs is not None:
            for feature_idx in np.arange(theta.size):
                if feature_idx not in features_idxs:
                    theta[feature_idx] = previous_theta[feature_idx]

        # theta_update_old = theta_update_new

        return theta


class StochasticGradientDescent(GradientDescent):
    def __init__(self, **kwargs):
        super().__init__(key='gd', **kwargs)

    def update_parameters(self, theta, gradient, features_idxs=None):
        pass


class Adagrad(FirstOrderMethod):
    def __init__(self, **kwargs):
        super().__init__(key='adagrad', **kwargs)

    def update_parameters(self, theta, gradient, features_idxs=None):
        # TODO: Fix adagrad using diagonal matrix for accumulated gradients and use acumulated gradients from past bilevel iterations. The same applies for Adam and momentum updates

        # TODO: Review gradients update Adagrad
        # self.reset_gradients()

        epsilon = 1e-12

        # self.acc_grads = 0

        self.acc_grads += gradient ** 2

        theta = theta - self.eta / (np.sqrt(self.acc_grads) + epsilon) * gradient

        return theta


class Adam(FirstOrderMethod):
    def __init__(self, **kwargs):
        super().__init__(key='adam', **kwargs)

        # Adam hyperparameters beta_1, beta_2 (when set to 0, we get adagrad)
        self.beta_1 = kwargs.get('beta_1', 0.5)  # 0.9
        self.beta_2 = kwargs.get('beta_2', 0.5)  # 0.999

    def update_parameters(self, theta, gradient, features_idxs=None):
        epsilon = 1e-7

        # Compute and update first moments (m) and second moments (v)
        self.acc_m = self.beta_1 * self.acc_m + (1 - self.beta_1) * gradient
        self.acc_v = self.beta_2 * self.acc_v + (1 - self.beta_2) * gradient ** 2

        # Adjusted first and second moments
        adj_m = self.acc_m / (1 - self.beta_1)
        adj_v = self.acc_v / (1 - self.beta_2)

        # Parameter update
        theta = theta - self.eta * (adj_m / (np.sqrt(adj_v) + epsilon))

        return theta


class SecondOrderMethod(OuterMethod):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = 'second-order'

    def update_parameters(self):
        raise NotImplementedError


class Newton(SecondOrderMethod):

    def __init__(self, **kwargs):
        super().__init__(key='newton', **kwargs)

    def update_parameters(self, theta, hessian, gradient, features_idxs):

        if features_idxs is not None:
            previous_theta = theta.copy()

        theta = theta - np.linalg.pinv(hessian).dot(gradient)

        if features_idxs is not None:
            for feature_idx in np.arange(theta.size):
                if feature_idx not in features_idxs:
                    theta[feature_idx] = previous_theta[feature_idx]

        # # Stable version
        # theta = theta + np.linalg.lstsq(hessian, -g, rcond=None)[0]

        return theta


class LevenbergMarquardt(SecondOrderMethod):
    def __init__(self, **kwargs):

        # Adaptive learning rate in L-M
        self.lambda_lm = kwargs.pop('lambda_lm', 1e-2)
        self.vdown_lm = kwargs.pop('vdown_lm', 2)
        self.vup_lm = kwargs.pop('vup_lm', 3)

        super().__init__(key='lm', **kwargs)

        # https://mljs.github.io/levenberg-marquardt/
        # Source: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

    def update_parameters(self, theta, lambda_lm, jacobian, delta_y, features_idxs=None):
        J = jacobian

        J_T_J = J.T.dot(J)

        if features_idxs is not None:
            previous_theta = theta.copy()

        theta = theta + np.linalg.lstsq(J_T_J + lambda_lm * np.eye(J_T_J.shape[0]), J.T.dot(delta_y), rcond=None)[0]

        if features_idxs is not None:
            for feature_idx in np.arange(theta.size):
                if feature_idx not in features_idxs:
                    theta[feature_idx] = previous_theta[feature_idx]

        return theta


class LevenbergMarquardtRevised(LevenbergMarquardt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key = 'lm-rev'
        # SecondOrderMethod.__init__(key='lm-revised', **kwargs)

    def update_parameters(self, theta, lambda_lm, jacobian, delta_y, features_idxs=None):

        J = jacobian

        J_T_J = J.T.dot(J)

        if features_idxs is not None:
            previous_theta = theta.copy()

        theta = theta + np.linalg.lstsq(J_T_J + lambda_lm * np.eye(J_T_J.shape[0]), J.T.dot(delta_y), rcond=None)[0]

        if features_idxs is not None:
            for feature_idx in np.arange(theta.size):
                if feature_idx not in features_idxs:
                    theta[feature_idx] = previous_theta[feature_idx]

        return theta


class GaussNewton(LevenbergMarquardt):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key = 'gn'

    def update_parameters(self, **kwargs):
        kwargs.update({'lambda_lm': 0})
        return super().update_parameters(**kwargs)

    # def update_parameters(self, theta, jacobian, delta_y):
    #
    #     J = jacobian
    #
    #     # # Package solution from scipy
    #     # theta = theta + la.lstsq(J, -delta_y )[0]
    #
    #     # # Update
    #     # theta = theta +  np.linalg.inv(J.T.dot(J)).dot(J.T).dot(delta_y)
    #     # theta = theta + np.linalg.pinv(J.T.dot(J)).dot(J.T).dot(delta_y)
    #
    #     # #Stable solution but works only if matrix is full rank
    #     # theta = theta +  np.linalg.solve(J.T.dot(J), J.dot(delta_y))
    #
    #     # Lstsq is used to obtain an approximate solution for the system
    #     # https://nmayorov.wordpress.com/2015/06/18/basic-asdlgorithms-for-nonlinear-least-squares/
    #     theta = theta + np.linalg.lstsq(J.T.dot(J), J.T.dot(delta_y), rcond=None)[0]
    #
    #     # TODO: Understand why we ignore the J^T term when doing this, maybe this means just to multiply
    #     #  for a pseudo inverse. This is certainly more numerically stable
    #     # theta = theta + np.linalg.lstsq(J, -delta_y, rcond=None)[0]
    #
    #     return theta


def od_estimation(network,
                  paths_probabilities,
                  q_0,
                  q_bar
                  ):
    D = network.D
    M = network.M
    q = network.q

    counts = network.observed_counts_vector

    # Optimization of OD matrix (using plain gradient descent meantime)
    jacobian_q_x = D.dot(paths_probabilities.dot(np.ones([paths_probabilities.size, 1]).T)).dot(M.T)

    # Prediction of link flow
    pred_x = compute_response_function(D, M, q, paths_probabilities)

    # Hyperparam
    # lambda_q = 1e2 # This works very well for Yang network
    lambda_q = 1e2  # This works very well for Sioux network

    # This is a sort of RELU in neural networks terms
    grad_new_q = 1 / counts.size * (lambda_q * 2 * (q_0 - q_bar) + 2 * jacobian_q_x.T.dot(pred_x - counts))

    # Projected Gradient descent update
    # https://www.cs.ubc.ca/~schmidtm/Courses/5XX-S20/S5.pdf
    # eta_q = 1e-2 # This works very well for Yang network and GD
    #
    eta_q = 1e-4  # This works very well for Sioux network and GD

    q = np.maximum(np.zeros([q_0.size, 1]), q_0 - grad_new_q * eta_q)

    # # Projected adagrad
    #
    # # eta_q = 1e-7  # This works very well for Yang with adagrad
    # eta_q = 1e-11 # This works very well for Sioux with adagrad
    # #
    # acc_grads_q += grad_new_q ** 2
    # #
    # epsilon = 1e-8
    # Gt = np.diag(acc_grads_q.flatten())
    # q = np.maximum(np.zeros([q0.size,1]), q0 - (eta_q / np.sqrt(Gt+ epsilon)).dot(grad_new_q))

    print(repr(np.round(q.T, 1)))


class OuterOptimizer:
    ''' Outer level outer_optimizer'''

    def __init__(self,
                 method: str,
                 network: Optional[TNetwork] = None,
                 utility_function: UtilityFunction = None,
                 **kwargs):

        self.set_default_options()

        self.update_options(**kwargs)
        # self.options = self.options.get_updated_options(new_options=kwargs)

        self.network = network

        # if 'network' in kwargs.keys():
        #     self.network = kwargs['network']

        # self.data = data

        self.utility_function = utility_function

        # Create optimizer object
        kwargs.pop('parameters_constraints', None)
        self.method = self.generate_outer_method(method, **kwargs)

    def generate_outer_method(self, method, **kwargs) -> OuterMethod:

        assert method in ['gd', 'ngd', 'nsgd', 'adagrad', 'adam', 'newton', 'gn', 'lm', 'lm-revised'], \
            'optimization method is not supported'

        if method == 'gd':
            self.method = GradientDescent(**kwargs)

        if method == 'ngd':
            self.method = NormalizedGradientDescent(**kwargs)

        if method == 'nsgd':
            raise NotImplementedError

        if method == 'adagrad':
            self.method = Adagrad(**kwargs)
            raise NotImplementedError

        if method == 'adam':
            self.method = Adam(**kwargs)
            raise NotImplementedError

        if method == 'newton':
            self.method = Newton(**kwargs)

        if method == 'gn':
            self.method = GaussNewton(**kwargs)

        if method == 'lm':
            self.method = LevenbergMarquardt(**kwargs)

        if method == 'lm-revised':
            self.method = LevenbergMarquardtRevised(**kwargs)

        return self.method

    def update_options(self, **kwargs):
        self.options = self.options.get_updated_options(new_options=kwargs)

    def set_default_options(self):

        # # Feature selected for estimation
        # self.options['features'] = self.utility_function.features
        # self.options['features_Y'] = self.utility_function.features_Y

        # Copy options from equilibrator
        # self.options['equilibrator'] = self.equilibrator.options

        self.options = Options()

        # Initial theta search
        self.options['theta_search'] = None
        self.options['q_random_search'] = False
        self.options['n_draws_random_search'] = 0
        self.options['scaling_Q'] = False

        # Scaling
        self.options['standardization'] = {'mean': False, 'sd': False}

        # OD estimation
        self.options['od_estimation'] = False

        # Constrained optimization
        self.options['parameters_constraints'] = {'sign': False, 'fixed': False}

        # # Correction using path size logit
        # self.options['paths_specific_utility'] = 0

        # Batch size for links
        self.options['batch_size'] = 0

        # Number of iterations
        self.options['iters'] = 1

        # Parameters for automatic scaling
        self.options['eta_scaling'] = 1e-1
        self.options['iters_scaling'] = 0

        self.grad_old = None

    def compute_objective_function(self,
                                   theta,
                                   network,
                                   **kwargs):

        return np.sum(Learner.compute_objective_function(theta=theta,
                                                         D=network.D,
                                                         M=network.M,
                                                         C=network.C,
                                                         q=network.q,
                                                         counts=network.observed_counts_vector,
                                                         **kwargs
                                                         ))

    def compute_gradient_objective_function(self,
                                            network,
                                            **kwargs):

        gradient = gradient_objective_function(
            D=network.D,
            M=network.M,
            C=network.C,
            q=network.q,
            counts=network.observed_counts_vector,
            **kwargs)

        return gradient

    def compute_paths_probabilities(self,
                                    theta,
                                    network,
                                    **kwargs
                                    ):

        return compute_paths_probabilities(theta=theta,
                                           D=network.D,
                                           C=network.C,
                                           **kwargs)

    def compute_jacobian_response_function(self,
                                           theta,
                                           network,
                                           **kwargs):

        jacobian, _, _, _ = jacobian_response_function(
            theta=theta,
            D=network.D,
            M=network.M,
            C=network.C,
            q=network.q,
            counts=network.observed_counts_vector,
            **kwargs)

        return jacobian

    def compute_response_function(self,
                                  network,
                                  paths_probabilities):

        return compute_response_function(D=network.D,
                                         M=network.M,
                                         q=network.q,
                                         paths_probabilities=paths_probabilities)

    def compute_hessian_objective_function(self,
                                           theta,
                                           network,
                                           **kwargs):

        return numeric_hessian_objective_function(theta=theta,
                                                  counts=network.observed_counts_vector,
                                                  D=network.D,
                                                  M=network.M,
                                                  C=network.C,
                                                  q=network.q,
                                                  **kwargs)

    def project_parameters_constraints(self, theta):
        counter = 0
        signs = self.utility_function.signs
        for feature in self.utility_function.features:
            if feature in signs.keys():
                sign = signs[feature]
                if sign == '+':
                    theta[counter] = max(0, theta[counter])
                elif sign == '-':
                    theta[counter] = min(0, theta[counter])

            counter += 1

        return theta

    def solve_outer_problem(self,
                            Y: DataFrame,
                            Z: DataFrame,
                            theta_0: ColumnVector = None,
                            paths_specific_utility: ColumnVector = None,
                            paths_probabilities: Optional[ColumnVector] = None,
                            q_0: ColumnVector = None,
                            q_bar: ColumnVector = None,
                            **kwargs):

        """ Address uncongested case first only and with data from a given day only.
        TODO: congested case, addressing multiday data, stochastic gradient descent

        Arguments
        ----------
        :argument f: vector with path predicted_counts
        :argument  opt_params={'method': None, 'iters_scaling': 1, 'iters_gd': 0, 'gamma': 0, 'eta_scaling': 1,'eta': 0, 'batch_size': int(0)}
        :argument

        """

        if kwargs.get('parameters_constraints') is None:
            kwargs['parameters_constraints'] = self.options['parameters_constraints']

        options = self.options.get_updated_options(new_options=kwargs)

        iters = options['iters']
        batch_size = options['batch_size']
        iters_scaling = options['iters_scaling']
        eta_scaling = options['eta_scaling']

        if isinstance(self.method, FirstOrderMethod):
            print('\nEstimating parameters via ' + self.method.key + ' (' + str(int(iters))
                  + ' iters, eta = ' + "{0:.1E}".format(self.method.eta) + ')\n')

        if self.method.type == 'second-order':
            print('\nEstimating parameters via ' + self.method.key + ' (' + str(int(iters)) + ' iters)\n')

        if batch_size > 0:
            print('batch size for observed link counts = ' + str(batch_size))

        design_matrix = get_design_matrix(Y=Y,
                                          Z=Z,
                                          features_Z=self.utility_function.features_Z,
                                          features_Y=self.utility_function.features_Y)

        if theta_0 is None:
            print('No initial values of the utility function parameters have been provided')
            theta_0 = copy.deepcopy(self.utility_function.initial_values)

            # Check that utility function parameters have no none initial values, otherwise it sets them to 0
            for feature, initial_value in theta_0.items():
                if initial_value is None:
                    theta_0[feature] = 0
                # self.utility_function.initial_values[feature] = 0

        features_idxs = None

        if options['parameters_constraints']['fixed']:
            if self.utility_function.fixed is not None:
                features_idxs = list(np.where(np.array(list(self.utility_function.fixed.values())).astype(int) == 0)[0])

        counts = self.network.observed_counts_vector

        # OD
        q = q_0

        if q is None:
            q = self.network.q

        total_no_nans = np.count_nonzero(~np.isnan(counts))

        assert batch_size < total_no_nans, 'Batch size larger than size of observed counts vector'

        theta = np.array(list(theta_0.values()))[:, np.newaxis]

        # Path probabilities
        if paths_probabilities is None:
            paths_probabilities = self.compute_paths_probabilities(theta=theta,
                                                                   design_matrix=design_matrix,
                                                                   network=self.network,
                                                                   paths_specific_utility=paths_specific_utility
                                                                   )

        # List to store infromation over iterations
        thetas = [theta]
        grads = []
        times = [0]
        acc_t = 0

        t0 = time.time()

        # i) Scaling method will set all parameters to be equal in order to achieve quickly a convex region of the problem

        for iter in range(0, iters_scaling):

            if iter == 0:
                theta = np.array([1])[:, np.newaxis]

            grad_current = self.compute_gradient_objective_function(
                theta=theta,
                design_matrix=design_matrix.dot(thetas[0].dot(float(theta))),
                network=self.network,
                features_idxs=features_idxs,
                paths_probabilities=paths_probabilities,
                paths_specific_utility=paths_specific_utility)

            # For scaling this makes more sense using a sign function as it operates as a grid search and reduce numerical errors
            theta = theta - np.sign(grad_current) * eta_scaling

            # Record theta with the new scaling
            thetas.append(thetas[0] * theta)

            if iter == iters_scaling - 1:
                initial_objective = self.compute_objective_function(theta=thetas[0],
                                                                    design_matrix=design_matrix,
                                                                    network=self.network,
                                                                    paths_specific_utility=paths_specific_utility)
                print('initial objective: ' + str(initial_objective))
                final_objective = self.compute_objective_function(theta=thetas[-1],
                                                                  design_matrix=design_matrix,
                                                                  network=self.network,
                                                                  paths_specific_utility=paths_specific_utility
                                                                  )
                print('objective in iter ' + str(iter + 1) + ': ' + str(final_objective))
                print('scaling factor: ' + str(theta))
                print('theta after scaling: ' + str(thetas[-1].T))
                print('objective improvement due scaling: ' + str(1 - final_objective / initial_objective))

        ## ii) Gradient descent or newton-gauss for fine scale optimization

        theta = thetas[-1]
        theta_0 = thetas[-1]

        objective_values = []
        lambdas_lm = []
        mode_lm = 'lambda'
        best_theta_lm = None

        # paths_probabilities = paths_probabilities.copy()

        for iter in range(0, iters):

            # printProgressBar(iter, iters-1, prefix='Progress:', suffix='', length=20)

            # It is very expensive to compute p_f so adequating the code is important
            if iter > 0:
                paths_probabilities = self.compute_paths_probabilities(
                    theta=theta,
                    design_matrix=design_matrix,
                    network=self.network,
                    paths_specific_utility=paths_specific_utility
                )

            # Stochastic gradient/hessian
            if batch_size > 0:
                # Generate a random subset of idxs depending on batch_size
                missing_sample_size = total_no_nans - batch_size

                # Sample over the cells of the counts vector with no nan
                idx_nonas = np.where(~np.isnan(counts))[0]

                # Set nan in the sample outside of batch that has no nan
                idx_nobatch = list(np.random.choice(idx_nonas, missing_sample_size, replace=False))

                # counts_masked = masked_observed_counts(counts=counts, idx=idx_nobatch)

            # i) First order optimization methods

            if isinstance(self.method, FirstOrderMethod):

                if iters > 1:
                    objective_value = self.compute_objective_function(theta=thetas[iter],
                                                                      design_matrix=design_matrix,
                                                                      network=self.network,
                                                                      # p_f = p_f,
                                                                      paths_specific_utility=paths_specific_utility
                                                                      )

                    objective_values.append(objective_value)

                # t0 = time.time()

                grad_current = self.compute_gradient_objective_function(
                    theta=theta,
                    design_matrix=design_matrix,
                    network=self.network,
                    features_idxs=features_idxs,
                    # numeric = True,
                    paths_probabilities=paths_probabilities,
                    paths_specific_utility=paths_specific_utility)  # /total_no_nans

                # # TODO: debug momentum
                # #  formula (https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d)
                # # grad_old = (1 - gamma) * grad_old + gamma * grad_new
                # if self.grad_old is None:
                #     # So we avoid that first graident is 0
                #     grad_adj = grad_current
                #
                # else:
                #     grad_adj = gamma * self.grad_old + (1 - gamma) * grad_current

                # self.grad_old = grad_adj

                grad_adj = grad_current

                theta = self.method.update_parameters(theta=theta, gradient=grad_current, features_idxs=features_idxs)

                if options['parameters_constraints']['sign']:
                    theta = self.project_parameters_constraints(theta)

                # print('gradient_diff : ' + str(gradient_check(theta=theta)))

                grads.append(grad_adj)
                thetas.append(theta)

            # ii) Second order optimization methods

            if isinstance(self.method, SecondOrderMethod):

                # Gauss-newthon exploit when it is not in a convex region

                if isinstance(self.method, Newton):
                    # TODO: this can be speed up by avoid recomputing some terms two times for both the gradient and Hessian

                    g = self.compute_gradient_objective_function(
                        theta=theta,
                        design_matrix=design_matrix,
                        network=self.network,
                        features_idxs=features_idxs,
                        paths_probabilities=paths_probabilities,
                        paths_specific_utility=paths_specific_utility)

                    # TODO: Check that entire Hessian is used for Newton.
                    H = self.compute_hessian_objective_function(theta=theta,
                                                                design_matrix=design_matrix,
                                                                network=self.network,
                                                                paths_probabilities=paths_probabilities,
                                                                paths_specific_utility=paths_specific_utility)

                    if features_idxs is not None:
                        previous_theta = theta.copy()

                    theta = self.method.update_parameters(theta=theta, hessian=H, gradient=g,
                                                          features_idxs=features_idxs)

                    for feature_idx in features_idxs:
                        theta[feature_idx] = previous_theta[feature_idx]

                if isinstance(self.method, (GaussNewton, LevenbergMarquardt, LevenbergMarquardtRevised)):

                    if iter == 0:
                        lambda_lm = self.method.lambda_lm
                        vup_lm = self.method.vup_lm
                        vdown_lm = self.method.vdown_lm
                        theta = theta_0

                        if iters > 1 and isinstance(self.method, (LevenbergMarquardt, LevenbergMarquardtRevised)):
                            previous_objective = self.compute_objective_function(
                                theta=thetas[iter],
                                design_matrix=design_matrix,
                                network=self.network,
                                # p_f = p_f,
                                paths_specific_utility=paths_specific_utility
                            )

                            objective_values.append(previous_objective)

                    lambdas_lm.append(lambda_lm)

                    J = self.compute_jacobian_response_function(
                        theta=theta,
                        design_matrix=design_matrix,
                        network=self.network,
                        paths_probabilities=paths_probabilities,
                        paths_specific_utility=paths_specific_utility,
                        features_idxs=features_idxs
                    )

                    predicted_counts = self.compute_response_function(network=self.network,
                                                                      paths_probabilities=paths_probabilities)

                    delta_y = fake_observed_counts(predicted_counts=predicted_counts,
                                                   observed_counts=counts) - predicted_counts

                    idxs_nan = np.where(np.isnan(counts))[0]
                    delta_y = np.delete(delta_y, idxs_nan, axis=0)
                    J = np.delete(J, idxs_nan, axis=0)

                    theta = self.method.update_parameters(theta=theta, lambda_lm=lambda_lm, jacobian=J,
                                                          delta_y=delta_y, features_idxs=features_idxs)

                    if options['parameters_constraints']['sign']:
                        theta = self.project_parameters_constraints(theta)

                    objective_value = self.compute_objective_function(theta=theta,
                                                                      design_matrix=design_matrix,
                                                                      network=self.network,
                                                                      paths_specific_utility=paths_specific_utility
                                                                      )

                    objective_values.append(objective_value)

                    if isinstance(self.method, (LevenbergMarquardt, LevenbergMarquardtRevised)):

                        # Choice of damping factor for lm

                        # Marquardt recommended starting with a value \lambda _{0} and a factor \nu > 1. Initially setting \lambda =\lambda _{0} and computing the residual sum of squares  after one step from the starting point with the damping factor of \lambda =\lambda _{0} and secondly with \lambda _{0}/\nu . If both of these are worse than the initial point, then the damping is increased by successive multiplication by \nu  until a better point is found with a new damping factor of {\lambda _{0}\nu^{k} for some k.

                        if mode_lm == 'lambda' or mode_lm == 'up':
                            objective_lambda = objective_value
                            theta_lambda = theta

                        if iter == 0:
                            lambda_lm = lambda_lm / vdown_lm
                            objective_lambda = objective_value
                            mode_lm = 'down'
                            best_theta_lm = theta

                        elif mode_lm == 'down':
                            objective_lambda_vdown = objective_value
                            theta_vdown = theta

                        if len(objective_values) >= 3:

                            if objective_lambda_vdown < previous_objective and objective_lambda_vdown < objective_lambda:
                                # If use of the damping factor  \lambda /\nu results in a reduction in squared residual, then this is taken as the new value of \lambda  (and the new optimum location is taken as that obtained with this damping factor) and the process continues

                                best_theta_lm = theta_vdown

                                lambda_lm = lambda_lm / vdown_lm
                                mode_lm = 'down'

                                objective_lambda = objective_value

                                # objective_lambda_vdown = objective_value

                                # continue


                            elif objective_lambda < objective_lambda_vdown and objective_lambda < previous_objective:

                                # lambda_lm = lambda_lm * self.options['vdown_lm']

                                # if using \lambda /\nu resulted in a worse residual, but using \lambda  resulted in a better residual, then \lambda  is left unchanged and the new optimum is taken as the value obtained with \lambda  as damping factor

                                # if mode_lm == 'down':
                                #     lambda_lm = lambda_lm * self.options['vdown_lm']
                                #
                                # else:
                                #     best_theta_lm = theta

                                mode_lm = 'lambda'

                                best_theta_lm = theta_lambda

                                # previous_objective = objective_lambda
                                # mode_lm = 'lambda'
                                # objective_lambda = previous_objective


                            else:
                                # If both of these are worse than the initial point, then the damping is increased by successive multiplication by {\displaystyle \nu }\nu  until a better point is found with a new damping factor of \lambda _{0}\nu ^{k} for some k.

                                mode_lm = 'up'

                                lambda_lm = lambda_lm * vup_lm

                        theta = best_theta_lm

                delta_t = time.time() - t0
                acc_t += delta_t

                thetas.append(theta)
                times.append(acc_t)

                if mode_lm == 'lambda' and isinstance(self.method, (LevenbergMarquardt, LevenbergMarquardtRevised)):
                    break

        if isinstance(self.method, (LevenbergMarquardt, LevenbergMarquardtRevised)):
            print('Damping factors: ' + str(["{0:.1E}".format(lambda_lm) for lambda_lm in lambdas_lm]))
            # print('theta: ' + str({key: "{0:.1E}".format(val) for key, val in theta_dict.items()}))

        q = q_0

        if options['od_estimation']:
            od_estimation(network=self.network,
                          paths_probabilities=paths_probabilities,
                          q_0=q_0,
                          q_bar=q_bar
                          )

        best_objective = float('inf')

        if isinstance(self.method, FirstOrderMethod) and iters > 1:
            for iter in range(iters):
                if objective_values[iter] < best_objective:
                    best_objective = objective_value
                    best_theta = thetas[iter]

        else:
            best_theta = thetas[-1]

        # Do not provide p_f in this part, because it does not correspond to the p_f in the last iteration but the one before the last
        # Thus, if there is only one iteration p_f and p_f_0 are equal, and the value of the objective function is wrongly assumed to be the same
        final_objective = self.compute_objective_function(
            theta=best_theta,
            design_matrix=design_matrix,
            network=self.network,
            paths_specific_utility=paths_specific_utility
        )

        # grad_old = grads[-1]

        theta = best_theta  # thetas[-1]

        # Conver theta into a dictionary to return it then
        theta_dict = dict(zip(self.utility_function.features, theta.flatten()))

        print('theta: ' + str({key: "{0:.1E}".format(val) for key, val in theta_dict.items()}))

        # print("Gradient:", str({key: "{0:.1E}".format(val) for key, val in zip(theta_dict.keys(),grads[-1].flatten().tolist())}))

        if 'c' in theta_dict.keys() and theta_dict['c'] != 0:
            print('Current ratio theta: ' + str(round(theta_dict['tt'] / theta_dict['c'], 4)))

        print('time: ' + str(np.round(time.time() - t0, 1)) + '[s]')

        return q, theta_dict, grads, final_objective


class Learner:
    def __init__(self,
                 equilibrator: LUE_Equilibrator,
                 utility_function,
                 outer_optimizer: OuterOptimizer,
                 network=None,
                 name: str = '',
                 **kwargs):

        # I may check that the utility function of the inner and outer optimzers are the same

        self.equilibrator = equilibrator
        self.outer_optimizer = outer_optimizer

        self.name = name

        # Initial theta for estimation

        self._utility_function = utility_function

        self.outer_optimizer.utility_function = self.utility_function

        # self.data = self.outer_optimizer.data

        if network is not None:
            self.load_network(network=network)

        self.set_default_options()

        self.update_options(**kwargs)

        # O

    @property
    def utility_function(self):
        return self._utility_function

    @utility_function.setter
    def utility_function(self, value):
        self._utility_function = value

        # self.options['features_Y'] = self.utility_function.features_Y
        # self.options['features_Z'] = self.utility_function.features_Z

        self.outer_optimizer.utility_function = value
        self.equilibrator.utility_function = value

        self.outer_optimizer.grad_old = None

    def load_network(self, network):
        self.network = network
        self.equilibrator.network = self.network
        self.outer_optimizer.network = self.network
        self.counts = self.network.link_data.counts
        self.observed_counts_vector = self.network.observed_counts_vector

    def update_options(self, **kwargs):
        self.options = self.options.get_updated_options(new_options=kwargs)

    def get_updated_options(self, **kwargs):
        return self.options.get_updated_options(new_options=kwargs)

    def set_default_options(self):

        # TODO: identify all default arguments from config object and store them automatically with self attribute value (see args** example). After, the method for estimation should see if the argument was provided, if it was, then it should not use the default. Otherwise, use the default

        self.options = Options()

        # Bilevel iteraitons
        self.options['bilevel_iters'] = None

        self.options['normalization'] = True,

        # Statistical inference
        self.options['numeric_hessian'] = False
        self.options['alpha'] = 0.05

        # Feature selection based on t-test from no refined step and ignoring fixed effects (I must NOT do post-selection inference as it violates basic assumptions)
        # self.options['ttest_selection'] = False

        # Computation of t-test with top proportion of observations in terms of SSE
        self.options['pct_lowest_sse'] = 100

        # Relax the critical value to remove features that are highly "no significant". A simil of regularization
        # self.options['alpha_selection'] = 3
        # self.options['paths_specific_utility'] = 0

        # Initial theta search
        # self.options['theta_search'] = None
        # self.options['q_random_search'] = False
        # self.options['n_draws_random_search'] = 0
        # self.options['scaling_Q'] = False

        # Copy options from equilibrator
        # self.options['equilibrator'] = self.equilibrator.options

        # Scaling
        # self.options['scale_features'] = {'mean': False, 'std': False}
        # self.options['standardization'] = {'mean': False, 'std': False}

        # Out of sample prediction mode
        # self.options['outofsample_prediction_mode'] = False

        # # Bounds for random search on theta and Q
        # self.bounds_theta_0 = {key: (-1, 1) for key in self.utility_function.initial_values.keys()}
        #
        # self.bounds_q = None  # (0, 2)

    def bilevel_optimization(self,
                             network: TNetwork = None,
                             q0: np.ndarray = None,
                             q_bar: ColumnVector = None,
                             link_report=False,
                             iteration_report=False,
                             convergence_report=False,
                             **kwargs):

        """

        Arguments
        ----------
        :argument f: vector with path predicted_counts
        :argument M: Path/O-D demand incidence
        :argument D: Path/link incidence matrix
        :argument counts: link counts
        """

        print('\nBilevel optimization for ' + str(self.network.key) + ' network \n')

        options = self.options.get_updated_options(new_options=kwargs)

        iters = options.get('bilevel_iters')

        # Returns
        results = {}
        best_iter = None
        if iters is None:
            return results, best_iter

        if network is None:
            network = self.network

        features_Y = self.utility_function.features_Y
        features_Z = self.utility_function.features_Z
        normalization = options['normalization']

        if q0 is None:
            q0 = network.q

        # Update od
        q_current = q0 # copy.deepcopy(q0)

        # Initialization
        theta_current = {k: v for k, v in self.utility_function.initial_values.items() if
                         k in [*features_Y, *features_Z]}

        counts = network.observed_counts_vector

        with block_output(show_stdout=iteration_report, show_stderr=iteration_report):
            print('Iteration : ' + str(1) + '/' + str(int(iters)) + '\n')

        print('Initial theta: ' + str({key: "{0:.1E}".format(val) for key, val in theta_current.items()}))
        # print('Initial q: ', repr(np.round(q_current.T,1)))
        if 'c' in theta_current.keys() and theta_current['c'] != 0:
            print('Initial ratio theta: ' + str(round(theta_current['tt'] / theta_current['c'], 4)))

        results_eq = {}

        with block_output(show_stdout=iteration_report, show_stderr=iteration_report):
            # Initial objective function
            results_eq[1] = self.equilibrator.path_based_suelogit_equilibrium(
                theta=theta_current,
                q=q_current,
                features_Z=features_Z,
                column_generation={'n_paths': None, 'paths_selection': None},
                **options)

        # Update travel times and link predicted_counts in network
        network.load_linkflows(results_eq[1]['x'])
        network.load_traveltimes(results_eq[1]['tt_x'])

        predicted_counts = network.link_data.predicted_counts_vector

        initial_objective = loss_function(observed_counts=counts,
                                          predicted_counts=predicted_counts)

        print('Initial objective: ' + '{:,}'.format(round(initial_objective)))
        print('Initial RMSE:', np.round(rmse(counts, predicted_counts), 1))
        print('Initial Normalized RMSE:', np.round(nrmse(counts, predicted_counts), 3))

        theta_array = np.array(list(theta_current.values()))

        design_matrix = get_design_matrix(
            Y=network.Y_data[features_Y],
            Z=network.Z_data[features_Z])

        results[1] = {**{'theta': theta_current,
                         'q': q_current,
                         'objective': initial_objective,
                         }, **results_eq[1]}

        p_f_0 = results_eq[1]['p_f']

        # Get path specific utility:
        paths_specific_utility = network.get_paths_specific_utility()

        initial_error_by_link = error_by_link(observed_counts=counts,
                                              predicted_counts=predicted_counts,
                                              show_nan=False)

        if convergence_report:

            outer_gradient = gradient_objective_function(
                theta=theta_array[:, np.newaxis],
                design_matrix=design_matrix,
                counts=counts,
                q=network.q,
                D=network.D,
                M=network.M,
                C=network.C,
                paths_probabilities=p_f_0,
                paths_specific_utility=paths_specific_utility
            )

            print('Initial Gradient (computed analytically):',
                  dict(zip(theta_current.keys(), outer_gradient[-1].flatten().tolist())))

            print('Initial Hessian (computed numerically):')

            H = diagonal_hessian_objective_function(theta=theta_array,
                                                    design_matrix=design_matrix,
                                                    q=network.q,
                                                    D=network.D,
                                                    M=network.M,
                                                    C=network.C,
                                                    numeric=True,
                                                    normalization=normalization,
                                                    paths_specific_utility=paths_specific_utility
                                                    )
            print(H)

            if is_pos_def(H):
                print('Hessian is positive definite')
            else:
                print('Hessian is not positive definite')

        # if network.key == 'Fresno':
        summary_table = summary_links_report(network=network)
        summary_table['error'] = initial_error_by_link.flatten()
        errors_by_link = [initial_error_by_link]

        if link_report:
            with pd.option_context('display.float_format', '{:0.1f}'.format):
                print('\n' + summary_table.to_string())

        best_objective = initial_objective
        best_predicted_counts = copy.deepcopy(predicted_counts)
        best_iter = 1
        # best_q = copy.deepcopy(q_current)
        # best_outer_gradient = None
        best_theta = copy.deepcopy(theta_current)

        objective_values = [initial_objective]

        t0_global = time.time()

        # print('\n')

        if iteration_report is False:

            if isinstance(self.outer_optimizer.method, FirstOrderMethod):
                print('\nEstimating parameters via ' + self.outer_optimizer.method.key + ' (' + str(
                    int(self.outer_optimizer.options['iters']))
                      + ' iters, eta = ' + "{0:.1E}".format(self.outer_optimizer.method.eta) + ')\n')

            if self.outer_optimizer.method.type == 'second-order':
                print('\nEstimating parameters via ' + self.outer_optimizer.method.key + ' (' + str(
                    int(self.outer_optimizer.options['iters'])) + ' iters)\n')

        for iter in np.arange(2, iters + 1, 1):

            if not iteration_report:
                if iter == 2:
                    print('')
                printIterationBar(iter, iters, prefix='Iterations:', length=20)

            else:
                print('\nIteration : ' + str(iter) + '/' + str(int(iters)))

            t0 = time.time()

            # Outer problem (* it takes more time than inner problem)

            with block_output(show_stdout=iteration_report, show_stderr=iteration_report):
                q_new, theta_new, grad_new, _ \
                    = self.outer_optimizer.solve_outer_problem(
                    Y=network.Y_data[features_Y],
                    Z=network.Z_data[features_Z],
                    theta_0=theta_current,
                    paths_probabilities=p_f_0,
                    paths_specific_utility=paths_specific_utility,
                    q_0=q_current,
                    q_bar=q_bar,
                    parameters_constraints=options.get('parameters_constraints')
                )

            # q_current = q_new.copy()
            theta_current = theta_new.copy()

            with block_output(show_stdout=iteration_report, show_stderr=iteration_report):

                options_copy = options

                if iter == iters:
                    options_copy = copy.deepcopy(
                        {**options, **{'column_generation': {'n_paths': None, 'paths_selection': None}}})

                # Inner problem
                results_eq[iter] \
                    = self.equilibrator.path_based_suelogit_equilibrium(
                    theta=theta_current,
                    q=q_new,
                    features_Z=features_Z,
                    **options_copy
                )

            # Update travel times and link predicted_counts in network
            network.load_linkflows(results_eq[iter]['x'])
            network.load_traveltimes(results_eq[iter]['tt_x'])

            # Compute new value of link predicted_counts and objective function at equilibrium
            predicted_counts = network.link_data.predicted_counts_vector

            # New objective function after computing network equilibria with updated travel times
            objective_value = loss_function(observed_counts=counts,
                                            predicted_counts=predicted_counts)

            objective_values.append(objective_value)

            results[iter] = {**{'theta': theta_current,
                                'q': q_current,
                                'objective': objective_value,
                                }, **results_eq[iter]}

            if objective_value < best_objective:
                best_objective = objective_value
                best_predicted_counts = copy.deepcopy(predicted_counts)
                best_theta = copy.deepcopy(theta_current)
                best_iter = iter

            p_f_0 = results_eq[iter]['p_f']

            # Get path specific utility:
            paths_specific_utility = network.get_paths_specific_utility()

            if iteration_report:

                print('\nTime current iteration: ' + str(np.round(time.time() - t0, 1)) + ' [s]')
                print('Current objective_value: ' + '{:,}'.format(round(objective_value)))
                print('Current objective improvement: ' + "{:.2%}".format(
                    np.round(1 - best_objective / initial_objective, 4)))
                print('Current RMSE:', np.round(rmse(counts, predicted_counts), 1))
                print('Current Normalized RMSE:', np.round(nrmse(counts, predicted_counts), 3))

                if len(objective_values) >= 2:

                    if objective_values[-2] != 0:
                        print('Marginal objective improvement: ' + "{:.2%}".format(
                            np.round(1 - objective_values[-1] / objective_values[-2], 4)))

                    print('Marginal objective improvement value: ' + '{:,}'.format(
                        np.round(objective_values[-2] - objective_values[-1], 1)))
                    print('')

            # if network.key == 'Fresno':

            current_error_by_link = error_by_link(counts, predicted_counts, show_nan=False)

            summary_table = summary_links_report(network=network)

            d_error = current_error_by_link - errors_by_link[-1]
            d_error_print = ["{0:.1E}".format(d_error_i[0]) for d_error_i in list(d_error)]

            summary_table['prev_error'] = errors_by_link[-1].flatten()
            summary_table['error'] = current_error_by_link.flatten()
            summary_table['d_error'] = d_error_print

            results[iter]['link_report'] = summary_table

            d_errors = d_error

            counts_copy = copy.deepcopy(network.link_data.counts)

            if iter == 2:
                initial_no_nas = len(np.where(~np.isnan(np.array(list(counts_copy.values()))))[0])

            if iter > 2:
                for i in range(iter - 1, 1, -1):
                    report = results[i]['link_report']
                    d_error = np.array(report['d_error']).astype(np.float)[:, np.newaxis]
                    d_errors = np.append(d_errors, d_error, axis=1)

            nas_link_keys = results[iter]['link_report']['link_key'][
                np.where(abs(np.max(d_errors, axis=1)) <= 1e-10)[0]]

            for key in nas_link_keys:
                counts_copy[(key[0], key[1], '0')] = np.nan

            final_no_nas = len(np.where(~np.isnan(np.array(list(counts_copy.values()))))[0])

            no_diff_error = initial_no_nas - final_no_nas

            errors_by_link.append(current_error_by_link)

            if link_report:
                print('Proportion of links with no difference in errors between iterations:',
                      "{:.1%}".format(no_diff_error / len(d_error)))
                with pd.option_context('display.float_format', '{:0.1f}'.format):
                    print('\n' + summary_table.to_string())

            # print('\nKey, Prediction, observed count, error and capacities by link\n', s)

            # print('\nImprovement by link', abs(initial_error_by_link)-abs(error_by_link(counts, predicted_counts, show_nan=False))/abs(initial_error_by_link))

        print('Summary results of bilevel optimization')

        #Reset counter for counter of od sampling
        self.equilibrator.options['column_generation']['iter_ods_sampling'] = 0

        best_theta_array = np.array(list(best_theta.values()))

        print('best iter: ' + str(best_iter))

        print('best theta: ' + str({key: "{0:.1E}".format(val) for key, val in best_theta.items()}))

        if 'c' in theta_current.keys() and best_theta['c'] != 0:
            print('best ratio theta: ' + str(round(best_theta['tt'] / best_theta['c'], 4)))

        if convergence_report:

            best_outer_gradient = gradient_objective_function(
                theta=best_theta_array[:, np.newaxis],
                design_matrix=design_matrix,
                counts=counts,
                q=network.q,
                D=network.D,
                M=network.M,
                C=network.C,
                paths_specific_utility=paths_specific_utility
            )

            # results[iter]['outer_gradient'] = dict(zip(theta_current.keys(), grad_new[-1].flatten().tolist()))

            print("Best gradient:", str({key: "{0:.1E}".format(val) for key, val in
                                         dict(zip(theta_current.keys(), best_outer_gradient.flatten())).items()}))

            print('Best Hessian (computed numerically):')

            H = diagonal_hessian_objective_function(theta=best_theta_array,
                                                    design_matrix=design_matrix,
                                                    counts=network.observed_counts_vector,
                                                    q=network.q,
                                                    D=network.D,
                                                    M=network.M,
                                                    C=network.C,
                                                    numeric=True,
                                                    normalization=normalization,
                                                    paths_specific_utility=paths_specific_utility
                                                    )

            print(H)

            if is_pos_def(H):
                print('Hessian is positive definite')
            else:
                print('Hessian is not positive definite')

            print("Final best Hessian:",
                  str({key: "{0:.1E}".format(val) for key, val in results[best_iter]['outer_gradient'].items()}))

        # print('best q: ' + repr(np.round(best_q.T,1)))

        print('best objective_value: ' + '{:,}'.format(round(best_objective)))

        print(
            'Final best objective improvement: ' + "{:.2%}".format(np.round(1 - best_objective / initial_objective, 4)))

        print('Final best objective improvement value: ' + '{:,}'.format(
            np.round(initial_objective - best_objective, 1)))

        print('Best RMSE:', np.round(rmse(counts, best_predicted_counts), 1))
        print('Best Normalized RMSE:', np.round(nrmse(counts, best_predicted_counts), 3))
        print('Total time: ' + str(np.round(time.time() - t0_global, 1)) + ' [s]')

        # print('Loss by link', error_by_link(counts, predicted_counts,show_nan = False))

        return results, best_iter

    @staticmethod
    def compute_objective_function(theta,
                                   design_matrix: Matrix,
                                   counts: ColumnVector,
                                   q: ColumnVector,
                                   D: Matrix,
                                   M: Matrix,
                                   C: Matrix,
                                   p_f: ColumnVector = None,
                                   normalization=True,
                                   paths_specific_utility=0
                                   ):
        '''
        SSE: Sum of squared errors


        :param theta:
        :param design_matrix:
        :param counts:
        :param q:
        :param D:
        :param M:
        :param C:
        :return:
        '''

        # Path probabilities (TODO: speed up this operation by avoiding elementwise division)

        if p_f is None:
            p_f = compute_paths_probabilities(theta=theta,
                                              design_matrix=design_matrix,
                                              D=D,
                                              C=C,
                                              normalization=normalization,
                                              paths_specific_utility=paths_specific_utility)

        # Response function
        m = compute_response_function(D, M, q, p_f)

        # Objective function

        # Account for vector positions with NA values

        counts_copy = fake_observed_counts(predicted_counts=m, observed_counts=counts)

        s = (m - counts_copy) ** 2

        return s

    @staticmethod
    def compute_sse(*args, **kwargs):
        return Learner.compute_objective_function(*args, **kwargs)

    @staticmethod
    def ttest_theta(network: TNetwork,
                    h0,
                    theta: {},
                    Y,
                    Z,
                    features_Y,
                    features_Z,
                    xc,
                    pct_lowest_sse=100,
                    alpha=0.05,
                    silent_mode: bool = False,
                    numeric_hessian=True,
                    **kwargs):
        # TODO: take advantage of kwargs argument to simplify function signature. A network object should be required only

        t0 = time.time()
        print('\nPerforming hypothesis testing (H0: theta = ' + str(h0) + ')')

        q = network.q
        D = network.D
        M = network.M
        C = network.C

        design_matrix = get_design_matrix(Y=Y, Z=Z, features_Z=features_Z)

        theta_array = np.array(list(theta.values()))[:, np.newaxis]

        p = theta_array.shape[0]
        n = np.count_nonzero(~np.isnan(xc))
        # q = q[:, np.newaxis]

        # SSE per observation
        sses = Learner.compute_sse(theta_array, design_matrix, xc, q, D, M, C)
        top_sse = sses

        if pct_lowest_sse < 100:
            idxs_nonan = np.where(~np.isnan(xc))[0]

            n_top_obs = len(sses.flatten()[idxs_nonan])
            top_n = int(n_top_obs * pct_lowest_sse / 100)
            top_sse = np.sort(sses.flatten()[idxs_nonan])[0:top_n]

            n = top_n

        epsilon = 1e-7

        sum_sse = np.sum(top_sse) + epsilon

        var_error = sum_sse / (n - p)

        if numeric_hessian is True:
            # objective_function(theta_array, design_matrix, counts, q, D, M, C)

            print('Hessian is computed numerically')

            H = nda.Hessian(objective_function_numeric_hessian)(list(theta_array.flatten()), design_matrix, xc, q, D, M,
                                                                C)

        else:
            print('Hessian is approximated as the Jacobian by its transpose')

            # # Unidimensional inverse function is just the reciprocal but this is multidimensional so inverse is required
            F, pf = jacobian_response_function(theta_array,
                                               design_matrix,
                                               q,
                                               D,
                                               M,
                                               C)

            # #Robust approximation of covariance matrix (almost no difference with previous methods)
            # cov_theta = np.linalg.lstsq(var_error*F.T.dot(F), np.eye(F.shape[1]), rcond=None)[0]
            H = F.T.dot(F)

        cov_theta = np.linalg.pinv(H)

        # Read Nonlinear regression from gallant (1979)
        # T value
        critical_tvalue = stats.t.ppf(1 - alpha / 2, df=n - p)
        ttest = (theta_array - h0) / np.sqrt(var_error * np.diag(cov_theta)[:, np.newaxis])
        # width_int = two_tailed_ttest*np.sqrt(cov_theta)
        # pvalue =  (1 - stats.t.cdf(ttest,df=n-p))
        # https://stackoverflow.com/questions/23879049/finding-two-tailed-p-value-from-t-distribution-and-degrees-of-freedom-in-python
        # pvalue = (1 - stats.t.cdf(ttest,df=n-p))#
        pvalue = 2 * stats.t.sf(np.abs(ttest), df=n - p)  # * 2

        # stats.t.sf(np.round(ttest,2), df=n - p)

        # pvalue = 2 * stats.t.sf(1.9, df=n - p)  # * 2
        # return "[" + str(theta - width_int) + "," + str(theta + width_int) + "]"

        if not silent_mode:
            print('Point estimate: ' + str({key: "{0:.0E}".format(val) for key, val in theta.items()}))
            print('T-tests: ' + str(ttest.flatten()))
            print('P-values: ' + str(pvalue.flatten()))

            print('Sample size:', n)
            print(str(round(pct_lowest_sse)) + '% of the total observations with lowest SSE were used')

        print('Time: ' + str(np.round(time.time() - t0, 1)) + '[s]')

        return ttest, critical_tvalue, pvalue

    def confint_theta(self,
                      theta: {},
                      design_matrix,
                      xc,
                      q,
                      D,
                      M,
                      C,
                      alpha=0.05,
                      pct_lowest_sse=100,
                      silent_mode: bool = False,
                      numeric_hessian=True):

        t0 = time.time()

        print('\nComputing confidence intervals (alpha = ' + str(alpha) + ')')

        theta_array = np.array(list(theta.values()))[:, np.newaxis]

        p = theta_array.shape[0]
        n = np.count_nonzero(~np.isnan(xc))  # counts.shape[0]
        q = q[:, np.newaxis]

        # SSE per observation
        sses = Learner.compute_sse(theta=theta_array,
                                   design_matrix=design_matrix,
                                   counts=xc,
                                   q=q,
                                   D=D,
                                   M=M,
                                   C=C)

        if pct_lowest_sse < 100:
            idxs_nonan = np.where(~np.isnan(xc))[0]

            n_top_obs = len(sses.flatten()[idxs_nonan])
            top_n = int(n_top_obs * pct_lowest_sse / 100)
            top_sse = np.sort(sses.flatten()[idxs_nonan])[0:top_n]

            n = top_n

        epsilon = 1e-7

        sum_sse = np.sum(top_sse) + epsilon

        var_error = sum_sse / (n - p)

        if numeric_hessian is True:

            print('Hessian is computed numerically')

            H = nd.Hessian(objective_function_numeric_hessian)(list(theta_array.flatten()), design_matrix, xc, q, D, M,
                                                               C)

        else:
            print('Hessian is approximated as the Jacobian by its transpose')

            # # Unidimensional inverse function is just the reciprocal but this is multidimensional so inverse is required

            F, pf = jacobian_response_function(theta_array, design_matrix, q, D, M, C)

            H = F.T.dot(F)

        cov_theta = np.linalg.pinv(H)

        critical_tvalue = stats.t.ppf(1 - alpha / 2, df=n - p)

        width_confint = critical_tvalue * np.sqrt(var_error * np.diag(cov_theta)[:, np.newaxis])

        confint_list = ["[" + str(round(float(i - j), 4)) + ", " + str(round(float(i + j), 4)) + "]" for i, j in
                        zip(theta_array, width_confint)]

        if not silent_mode:
            print('Point estimate: ' + str({key: "{0:.0E}".format(val) for key, val in theta.items()}))
            print('Confidence intervals :' + str(confint_list))
            print('Time: ' + str(np.round(time.time() - t0, 1)) + '[s]')

        return confint_list, width_confint

    @staticmethod
    def ftest(design_matrix,
              counts,
              q,
              D,
              M,
              C,
              theta_m1: dict = None,
              theta_m2: dict = None,
              alpha=0.05,
              pct_lowest_sse=100,
              silent_mode: bool = False) -> pd.DataFrame:

        print('\nComputing F-test')

        t0 = time.time()

        # model 1 is 'nested' within model 2 (full)

        n = np.count_nonzero(~np.isnan(counts))

        # Full model
        theta_m2_array = np.array(list(theta_m2.values()))[:, np.newaxis]
        p_2 = theta_m2_array.shape[0]

        # Restricted model
        if theta_m1 is None:
            theta_m1 = dict.fromkeys(theta_m2.keys(), 0)
            theta_m1_array = np.array(list(theta_m1.values()))[:, np.newaxis]
            p_1 = 0

        else:
            theta_m1_array = np.array(list(theta_m1.values()))[:, np.newaxis]
            p_1 = theta_m1_array.shape[0]

        # SSE for each observation of the first model
        sses_1 = Learner.compute_sse(theta=theta_m1_array,
                                     design_matrix=design_matrix,
                                     counts=counts,
                                     q=q,
                                     D=D,
                                     M=M,
                                     C=C)
        top_sse_1 = sses_1

        # SSE  for each observation of the second model (full model)
        sses_2 = Learner.compute_sse(theta=theta_m2_array,
                                     design_matrix=design_matrix,
                                     counts=counts,
                                     q=q,
                                     D=D,
                                     M=M,
                                     C=C)
        top_sse_2 = sses_2

        if pct_lowest_sse < 100:
            idxs_nonan = np.where(~np.isnan(counts))[0]

            n_top_obs = len(sses_2.flatten()[idxs_nonan])
            top_n = int(n_top_obs * pct_lowest_sse / 100)
            n = top_n

            # Subset of SSE models 1 and 2
            top_sse_1 = np.sort(sses_1.flatten()[idxs_nonan])[0:top_n]
            top_sse_2 = np.sort(sses_2.flatten()[idxs_nonan])[0:top_n]

        # # SSE per observation
        # sses = sse(theta_array, design_matrix, counts, q, D, M, C)
        # top_sse = sses

        # Source: https://en.wikipedia.org/wiki/F-test

        numerator_ftest = (np.sum(top_sse_1) - np.sum(top_sse_2)) / (p_2 - p_1)
        denominator_ftest = np.sum(top_sse_2) / (n - p_2)

        ftest_value = numerator_ftest / denominator_ftest

        # source: https://stackoverflow.com/questions/39813470/f-test-with-python-finding-the-critical-value

        critical_fvalue = stats.f.ppf(1 - alpha, dfn=p_2 - p_1, dfd=n - p_2)

        pvalue = stats.f.sf(ftest_value, dfn=p_2 - p_1, dfd=n - p_2)  # * 2

        if not silent_mode:
            # print('Point estimate: ' + str({key: "{0:.0E}".format(val) for key, val in theta.items()}))
            print('F-test: ' + str(round(ftest_value, 4)))
            print('P-value: ' + str(round(pvalue, 4)))
            print('Critical f-value: ' + str(round(critical_fvalue, 4)))
            print('n:', n)
            print(str(round(pct_lowest_sse)) + '% of the total observations with lowest SSE were used')

            print('Time: ' + str(np.round(time.time() - t0, 1)) + '[s]')

        summary_inference_model = {'F': ftest_value,
                                   'critical-F': critical_fvalue,
                                   'p': pvalue,
                                   'dof_m1': p_1,
                                   'dof_m2': p_2,
                                   'sse_m1': np.sum(top_sse_1),
                                   'sse_m2': np.sum(top_sse_2),
                                   'n': n,
                                   }

        return pd.DataFrame([summary_inference_model])

    def statistical_inference(self,
                              **kwargs
                              ):

        ''' Perform both learning and inference'''

        # true_traveltimes_dict = {link.key: link.true_traveltime for link in self.network.links}
        # self.network.set_Y_attr_links(Y={'tt': true_traveltimes_dict})

        learning_results, best_iter = self.bilevel_optimization(**kwargs)

        inference_results = {}

        if best_iter is None:
            return learning_results, inference_results, best_iter

        best_results = learning_results[best_iter]
        theta = best_results['theta']

        # Update travel times and link predicted_counts in network
        self.network.load_linkflows(best_results['x'])
        self.network.load_traveltimes(best_results['tt_x'])

        counts = self.network.observed_counts_vector

        if kwargs.get('link_selection', False) and len(learning_results.keys())>2:
            counts, _ = get_informative_links(learning_results=learning_results, network=self.network)
            # self.network.load_traffic_counts(new_counts)
            counts = np.array(list(counts.values()))[:, np.newaxis]

        parameter_inference_table, model_inference_table = hypothesis_tests(
            network=self.network,
            theta=theta,
            design_matrix=get_design_matrix(
                Y=self.network.Y_data,
                Z=self.network.Z_data[self.utility_function.features_Z]),
            counts=counts,
            predicted_counts=self.network.link_data.predicted_counts_vector,
            **kwargs)

        inference_results['model'] = model_inference_table
        inference_results['parameters'] = parameter_inference_table

        return learning_results, inference_results, best_iter


def feature_selection(utility_function: UtilityFunction,
                      theta: Dict[str, float],
                      criterion='sign'):
    assert criterion in ['ttest', 'sign']

    features_Z = []
    features_Y = []

    if criterion == 'sign':
        for feature, value in theta.items():
            if feature in utility_function.parameters:
                expected_sign = utility_function.parameters[feature].sign
                parameter_type = utility_function.parameters[feature].type

                if value > 0 and expected_sign == '+' or value < 0 and expected_sign == '-':
                    if parameter_type == 'Z':
                        features_Z.append(feature)
                    elif parameter_type == 'Y':
                        features_Y.append(feature)

    return features_Y, features_Z


def scaling_Q(counts: ColumnVector,
              network: TNetwork,
              equilibrator: LUE_Equilibrator,
              utility_function: UtilityFunction,
              grid,
              n_paths=None,
              silent_mode=False,
              uncongested_mode=True):
    print("\nScaling Q matrix with grid: ", str(["{0:.2}".format(val) for val in grid]), '\n')

    # Generate new paths if required

    if n_paths is not None:
        # Save previous paths and paths per od lists
        original_paths, original_paths_od = network.paths, network.paths_od
        original_M, original_D, original_C = network.M, network.D, network.C

        paths, paths_od = equilibrator.paths_generator.k_shortest_paths(
            k=n_paths,
            network=network,
            theta=utility_function.values())

        network.load_paths(paths_od=paths_od)

    # Noise or scale difference in Q matrix
    Q_original = network.Q.copy()

    loss_scale_dict = {}

    for scale_factor, iter in zip(grid, range(len(grid))):

        printProgressBar(iter, len(grid), prefix='Progress:', suffix='',
                         length=20)

        assert scale_factor > 0, 'scale factor cannot be 0'

        Q_scaled = scale_factor * Q_original

        # Update Q matrix and dense q vector temporarily
        network.load_OD(Q_scaled)

        if uncongested_mode is True:
            loss_after_scale, _ = loss_counts_uncongested_network(
                counts=counts,
                network=network,
                utility_function=utility_function,
                equilibrator=equilibrator)
        else:
            loss_after_scale = loss_predicted_counts_congested_network(
                counts=counts,
                network=network,
                utility_function=utility_function,
                equilibrator=equilibrator)

        loss_scale_dict[scale_factor] = loss_after_scale

        with block_output(show_stdout=silent_mode, show_stderr=silent_mode):
            print('current scale', scale_factor)
            print('current loss', loss_after_scale, '\n')

    # Revert Q matrix and q dense vector to original form
    network.load_OD(Q_original)

    # Revert original paths and incidence matrices
    if n_paths is not None:
        network.load_paths(paths=original_paths, paths_od=original_paths_od)
        network.M, network.D, network.C = original_M, original_D, original_C

    return loss_scale_dict


def hypothesis_tests(theta: ParametersDict,
                     design_matrix,
                     counts: ColumnVector,
                     network: TNetwork,
                     alpha=0.05,
                     h0=0,
                     predicted_counts: Optional[ColumnVector] = None,
                     **kwargs
                     ):
    t0 = time.time()

    pct_lowest_sse = kwargs.get('pct_lowest_sse', 100)
    numeric_hessian = kwargs.get('numeric_hessian', False)
    numeric_jacobian = kwargs.get('numeric_jacobian', False)
    normalization = kwargs.get('normalization', True)
    paths_specific_utility = kwargs.get('paths_specific_utility', 0)
    p_f = kwargs.get('p_f', None)

    q = network.q
    D = network.D
    M = network.M
    C = network.C
    design_matrix = design_matrix

    theta_array = np.array(list(theta.values()))[:, np.newaxis]

    p = theta_array.shape[0]
    n = np.count_nonzero(~np.isnan(counts))

    print('\nHypothesis testing (H0: theta = ' + str(h0) + ', alpha = ' + str(alpha) + ', n = ' + str(n) + ')')

    assert q.shape[1] == 1, 'q is not a column vector'

    # SSE per observation
    if predicted_counts is not None:

        # Account for Nas when sensor coverage is lower than 1
        sses = (predicted_counts - fake_observed_counts(predicted_counts=predicted_counts, observed_counts=counts)) ** 2

    else:

        sses = Learner.compute_sse(theta=theta_array,
                                   design_matrix=design_matrix,
                                   counts=counts,
                                   q=q,
                                   D=D,
                                   M=M,
                                   C=C)

    top_sse = sses

    if pct_lowest_sse < 100:
        idxs_nonan = np.where(~np.isnan(counts))[0]

        n_top_obs = len(sses.flatten()[idxs_nonan])
        top_n = int(n_top_obs * pct_lowest_sse / 100)
        top_sse = np.sort(sses.flatten()[idxs_nonan])[0:top_n]

        n = top_n

    sum_sse = np.sum(top_sse)  # + epsilon

    assert p < n, 'number of observations must be greater than the degrees of freedom'

    var_error = sum_sse / (n - p)

    features_idxs = np.arange(0, theta_array.shape[0])

    if numeric_hessian is True:

        H = diagonal_hessian_objective_function(theta=theta_array,
                                                design_matrix=design_matrix,
                                                counts=counts,
                                                q=q,
                                                D=D,
                                                M=M,
                                                C=C,
                                                paths_probabilities=p_f,
                                                numeric=numeric_hessian,
                                                normalization=normalization,
                                                paths_specific_utility=paths_specific_utility)

    else:

        # # Check if parameters are close to zero. When true, it removes those parameters from the computation of Jacobian
        # features_idxs = []
        # zero_features_idxs = []
        # for feature_idx in range(theta_array.shape[0]):
        #     if np.allclose(theta_array[feature_idx], 0):
        #         zero_features_idxs.append(feature_idx)
        #     else:
        #         features_idxs.append(feature_idx)

        # # Unidimensional inverse function is just the reciprocal but this is multidimensional so inverse is required
        F, pf = jacobian_response_function(theta_array,
                                           design_matrix=design_matrix,
                                           q=q,
                                           D=D,
                                           M=M,
                                           C=C,
                                           # features_idxs = features_idxs,
                                           paths_probabilities=p_f,
                                           paths_specific_utility=paths_specific_utility,
                                           numeric=numeric_jacobian,
                                           normalization=normalization)

        # Remove elements of Jacobian associated to link with no traffic counts
        idxs_nan = np.where(np.isnan(counts))[0]
        F = np.delete(F, idxs_nan, axis=0)

        # # Remove columns of feature that is almost zero
        # F = np.delete(F, zero_features_idxs, axis=1)

        print('\nHessian approximated as J^T J')
        H = F.T.dot(F)

    # # Replaced nan values in Hessian
    # if np.any(np.isnan(H)):
    #     print('Cells of the Hessian matrix have NA values')
    #     H[np.isnan(H)] = epsilon

    # #Robust approximation of covariance matrix (almost no difference with previous method)
    # cov_theta = np.linalg.lstsq(H, np.eye(F.shape[1]), rcond=None)[0]
    cov_theta = np.linalg.pinv(H)
    # cov_theta = np.linalg.inv(H)

    diag_cov_theta = np.diag(cov_theta)

    # diag_cov_theta = np.zeros_like(theta_array)
    # for feature_idx in features_idxs:
    #     if not np.allclose(theta_array[feature_idx], 0):
    #         diag_cov_theta[feature_idx] = float(cov_theta[(feature_idx,feature_idx)])

    # i) T-tests
    critical_tvalue = stats.t.ppf(1 - alpha / 2, df=n - p)

    ttest = np.zeros_like(theta_array, dtype=np.float64)

    for feature_idx in features_idxs:
        if not np.allclose(theta_array[feature_idx] - h0, 0):
            ttest[feature_idx] = (theta_array[feature_idx] - h0) / np.sqrt(var_error * diag_cov_theta[feature_idx])

    pvalues = 2 * stats.t.sf(np.abs(ttest), df=n - p)

    # ii) Confidence intervals
    width_confint = critical_tvalue * np.sqrt(var_error * diag_cov_theta)

    confint_list = ["[" + str(round(float(i - j), 3)) + ", " + str(round(float(i + j), 3)) + "]" for i, j in
                    zip(theta_array, width_confint)]

    summary_inference_parameters = pd.DataFrame(
        {'parameter': theta.keys(), 'est': theta.values(), 'CI': confint_list,
         'width_CI': width_confint.flatten(), 't-test': ttest.flatten(), 'p-value': pvalues.flatten()})

    # summary_inference_parameters = pd.DataFrame(
    #     {'parameter': theta.keys(), 't-test': theta.values(), 'critical-t-value': list(criticalval_t * np.ones(len(theta))), 'p-value': pvals.flatten(), 'CI': confint_list, 'null_f_test': theta_m1.values()})

    with pd.option_context('display.float_format', '{:0.3f}'.format):
        print('\n', summary_inference_parameters.to_string(index=False))

    # (iii) F-test assuming that restricted model set all parameters equal to 0
    summary_inference_model = Learner.ftest(
        theta_m2=theta,
        design_matrix=design_matrix,
        counts=counts,
        q=q,
        D=D,
        M=M,
        C=C,
        pct_lowest_sse=pct_lowest_sse,
        alpha=alpha,
        silent_mode=True)

    # summary_inference_model = pd.DataFrame([summary_inference_model])

    with pd.option_context('display.float_format', '{:0.3f}'.format):
        print(summary_inference_model.to_string(index=False))

    if pct_lowest_sse < 100:
        print(str(round(pct_lowest_sse)) + '% of the total observations with lowest SSE were used')

    print('Time: ' + str(np.round(time.time() - t0, 1)) + '[s]')

    # return a pandas dataframe with summary of inference
    return summary_inference_parameters, summary_inference_model


def grid_search_theta(network: TNetwork,
                      utility_function: UtilityFunction,
                      equilibrator: LUE_Equilibrator,
                      counts: ColumnVector,
                      grid,
                      feature):
    # TODO: theta_0 may be not necessary

    # Grid and random search are performed under the assumption of an uncongested network to speed up the search

    theta_attr_grid, f_vals, grad_f_vals, hessian_f_vals \
        = grid_search_optimization(network=network,
                                   equilibrator=equilibrator,
                                   counts=counts,
                                   theta_attr_grid=grid,
                                   feature=feature,
                                   utility_function=utility_function,
                                   gradients=False, hessians=False)

    # print('grid for theta_t', theta_attr_grid)
    # print('losses ', f_vals)
    # print('gradients ', grad_f_vals)

    # # Plot
    # plot1 = visualization.Artist(folder_plots=config.plots_options['folder_plots'], dim_subplots=(2, 2))
    #
    # plot1.pseudoconvexity_loss_function(
    #     filename='quasiconvexity_l2norm_' + config.sim_options['current_network']
    #     , folder="experiments/quasiconvexity"
    #     , f_vals=f_vals, grad_f_vals=grad_f_vals, hessian_f_vals=hessian_f_vals
    #     , x_range=theta_attr_grid  # np.arange(-3,3, 0.5)
    #     , theta_true=theta_true[current_network]['tt'])

    # Initial point to perform the scaling using the best value for grid search of the travel time parameter
    min_loss = float('inf')
    best_theta_gs = np.array(list(utility_function.values.values()))

    for theta_gs, loss in zip(theta_attr_grid, f_vals):
        if loss < min_loss:
            min_loss = loss
            best_theta_gs = theta_gs

    # print('best theta is ', min_theta_gs)
    # print('best theta: ', str("{0:.0E}".format(min_theta_gs)))
    print('best theta is ', str({key: round(val, 3)
                                 for key, val in
                                 zip(utility_function.values.keys(), best_theta_gs.flatten().tolist())}))

    return best_theta_gs


def random_search_theta(q_bounds=(1, 1),
                        **kwargs):
    thetas_rs, q_scales_rs, f_vals = random_search_optimization(q_bounds=q_bounds, **kwargs)

    min_loss = float('inf')
    min_theta_rs = 0
    min_q_scale_rs = 0

    for theta_rs, q_scale_rs, loss in zip(thetas_rs, q_scales_rs, f_vals):
        if loss < min_loss:
            min_loss = loss
            min_theta_rs = theta_rs
            min_q_scale_rs = q_scale_rs

    # print('best theta is: ', str({key: "{0:.1E}".format(val) for key, val in min_theta_rs.items()}))
    print('best theta is ', str({key: round(val, 3) for key, val in min_theta_rs.items()}))

    print('best q scale is: ', str({key: "{0:.2E}".format(val) for key, val in min_q_scale_rs.items()}))
    # print('best q scale is ', str({key: round(val, 3) for key, val in min_q_scale_rs.items()}))

    return min_theta_rs, min_q_scale_rs


def grid_search_theta_ttest(network: TNetwork,
                            equilibrator: LUE_Equilibrator,
                            utility_function: UtilityFunction,
                            grid: np.ndarray,
                            counts: ColumnVector,
                            feature: str
                            ):
    features_Y = utility_function.features_Y
    features_Z = utility_function.features_Z
    theta = utility_function.values

    theta_current = copy.deepcopy(theta)
    ttests = []

    for iter, theta_attr_val in zip(range(len(grid)), grid):
        printProgressBar(iter, len(grid), prefix='Progress:', suffix='', length=20)

        theta_current[feature] = theta_attr_val

        results_eq = equilibrator.path_based_suelogit_equilibrium(network=network,
                                                                  theta=theta_current,
                                                                  features_Y=features_Y,
                                                                  features_Z=features_Z,
                                                                  silent_mode=True
                                                                  )
        predicted_counts = np.array(list(results_eq['x'].values()))[:, np.newaxis]

        print('current theta: ', "{0:.3}".format(float(theta_attr_val)))

        design_matrix = network.design_matrix(features_Y, features_Z)

        summary_inference_parameters, summary_inference_model = hypothesis_tests(
            theta=theta_current,
            design_matrix=design_matrix,
            counts=counts,
            network=network,
            h0=0,
            predicted_counts=predicted_counts)

        # float(summary_inference_parameters[summary_inference_parameters.parameter == 'tt']['t-test'].values)

        ttest = summary_inference_parameters[['parameter', 't-test']]

        ttests.append({'theta': theta_current,
                       't-tests': dict(zip(ttest['parameter'].values, np.round(ttest['t-test'].values, 2)))})

    return ttests


def grid_search_Q_ttest(network: TNetwork,
                        equilibrator: LUE_Equilibrator,
                        utility_function: UtilityFunction,
                        scales: np.ndarray,
                        counts: ColumnVector
                        ):
    features_Y = utility_function.features_Y
    features_Z = utility_function.features_Z
    theta = utility_function.values

    theta_current = copy.deepcopy(theta)
    ttests = []

    for iter, scale in zip(range(len(scales)), scales):
        printProgressBar(iter, len(scales), prefix='Progress:', suffix='', length=20)

        network.scale_OD(scale)

        results_eq = equilibrator.path_based_suelogit_equilibrium(network=network,
                                                                  theta=theta_current,
                                                                  features_Y=features_Y,
                                                                  features_Z=features_Z,
                                                                  silent_mode=True
                                                                  )
        predicted_counts = np.array(list(results_eq['x'].values()))[:, np.newaxis]

        print('current scale: ', "{0:.1}".format(float(scale)))

        design_matrix = network.design_matrix(features_Y, features_Z)

        summary_inference_parameters, summary_inference_model = hypothesis_tests(
            theta=theta_current,
            design_matrix=design_matrix,
            counts=counts,
            network=network,
            p_f=results_eq['p_f'],
            h0=0,
            predicted_counts=predicted_counts)

        # ttest = float(np.array(summary_inference_parameters['t-test']).flatten())

        ttest = summary_inference_parameters[['parameter', 't-test']]

        ttests.append({'scale_Q': scale, 't-tests': dict(zip(ttest['parameter'].values, ttest['t-test'].values))})

    # Reset OD with original scale
    network.scale_OD(scale=1)

    return ttests


def grid_search_optimization(network: TNetwork,
                             equilibrator: LUE_Equilibrator,
                             counts: ColumnVector,
                             feature: str,
                             utility_function: UtilityFunction,
                             theta_attr_grid: Union[List, np.ndarray],
                             gradients: bool = False,
                             hessians: bool = False,
                             q=None):
    """
    Perform grid search optimization by computing equilibrium for a grid of values of the logit parameter
    associated to one of the attributes
    """

    print('\nPerforming grid search for ' + str(feature) + '\n')

    loss_function_vals = []
    grad_vals = []
    hessian_vals = []

    features_Y = utility_function.features_Y
    features_Z = utility_function.features_Z
    theta = utility_function.values

    if q is None:
        q = network.Q

    theta_current = copy.deepcopy(theta)

    for iter, theta_attr_val in zip(range(len(theta_attr_grid)), theta_attr_grid):

        printProgressBar(iter, len(theta_attr_grid) - 1, prefix='Progress:', suffix='', length=20)

        theta_current[feature] = theta_attr_val

        results_eq = equilibrator.path_based_suelogit_equilibrium(network=network,
                                                                  theta=theta_current,
                                                                  features_Y=features_Y,
                                                                  features_Z=features_Z,
                                                                  silent_mode=True
                                                                  )

        network.load_traveltimes(results_eq['tt_x'])

        predicted_counts = np.array(list(results_eq['x'].values()))[:, np.newaxis]

        p_f = results_eq['p_f']

        loss_function_vals.append(loss_function(counts, predicted_counts))

        # print('current loss: ', loss_function_vals[-1])
        # print('current theta: ', "{0:.3}".format(float(theta_attr_val)))

        design_matrix = network.design_matrix(features_Y, features_Z)

        index_feature = list(theta_current.keys()).index(feature)

        theta_current_vector = np.array(list(theta_current.values()))[:, np.newaxis]

        if gradients:
            grad_vals.append(
                float(gradient_objective_function(
                    features_idxs=[index_feature],
                    theta=theta_current_vector,
                    design_matrix=design_matrix,
                    counts=counts,
                    q=q,
                    D=network.D,
                    M=network.M,
                    C=network.C,
                    paths_probabilities=p_f)[index_feature]
                      ))

        if hessians:
            second_derivatives = diagonal_hessian_objective_function(theta=theta_current_vector,
                                                                     counts=counts,
                                                                     design_matrix=design_matrix,
                                                                     q=q,
                                                                     D=network.D,
                                                                     M=network.M,
                                                                     C=network.C,
                                                                     paths_probabilities=p_f,
                                                                     numeric=False)

            hessian_vals.append(second_derivatives[index_feature])

    return theta_attr_grid, loss_function_vals, grad_vals, hessian_vals


def random_search_optimization(network: TNetwork,
                               equilibrator: LUE_Equilibrator,
                               utility_function: UtilityFunction,
                               counts: ColumnVector,
                               n_draws: int,
                               theta_bounds: tuple,
                               q_bounds: tuple = None,
                               silent_mode=False,
                               uncongested_mode=True):
    """
    Perform grid search optimization by computing equilibrium for a grid of values of the logit parameter associated to one of the attributes
    """

    print('\nPerforming random search with ' + str(n_draws) + ' draws\n')

    loss_function_vals = []

    thetas = []
    q_scales = []

    features_Y = utility_function.features_Y
    features_Z = utility_function.features_Z
    theta = utility_function.values

    # Bound for values of the theta vector entries
    theta_0 = {key: theta_bounds for key, val in theta.items()}

    for draw in range(n_draws):

        printProgressBar(draw, n_draws, prefix='Progress:', suffix='', length=20)

        # Get a random theta vector according to the dictionary of bounds tuples
        theta_current = {key: 0 for key, val in theta_0.items()}

        for attribute, bounds in theta_0.items():
            theta_current[attribute] = float(np.random.uniform(*bounds, 1))

        if q_bounds is not None:
            q_scale = float(np.random.uniform(*q_bounds, 1))

            if silent_mode is True:
                blockPrint()

            loss_dict = scaling_Q(counts=counts,
                                  network=network,
                                  utility_function=utility_function,
                                  grid=[q_scale],
                                  n_paths=None,
                                  equilibrator=equilibrator,
                                  silent_mode=True,
                                  uncongested_mode=uncongested_mode)

            q_scales.append({'q_scale': q_scale})

            loss_function_vals.append(loss_dict[q_scale])

            if silent_mode is False:
                print('current q scale: ', str("{0:.1E}".format(q_scale)))

        else:

            # Do not generate new paths via column generation to save computation
            results_eq = equilibrator.path_based_suelogit_equilibrium(
                Nt=network,
                theta=theta_current,
                features_Y=features_Y,
                features_Z=features_Z,
                silent_mode=True)

            # predicted_counts = np.array(list(results_eq_initial['x'].values()))
            x_eq = np.array(list(results_eq['x'].values()))[:, np.newaxis]

            loss_function_vals.append(loss_function(counts, x_eq))

        if silent_mode is False:
            # print('current theta: ',str({key: "{0:.1E}".format(val) for key, val in theta_current.items()}))
            print('current theta: ', str({key: round(val, 3) for key, val in theta_current.items()}))

            print('current loss: ', '{:,}'.format(loss_function_vals[-1]), '\n')

        thetas.append(theta_current)

    if silent_mode is True:
        enablePrint()

    return thetas, q_scales, loss_function_vals


# @blockPrinting
def loss_predicted_counts_congested_network(equilibrator: LUE_Equilibrator,
                                            counts: ColumnVector,
                                            network: TNetwork,
                                            utility_function):
    """ Compute the l2 norm with naive prediction assuming a congested network"""

    features_Y = utility_function.features_Y
    features_Z = utility_function.features_Z
    theta = utility_function.values

    results_congested_eq = equilibrator.path_based_suelogit_equilibrium(network=network,
                                                                        theta=theta,
                                                                        features_Y=features_Y,
                                                                        features_Z=features_Z)

    predicted_counts = np.array(list(results_congested_eq['x'].values()))[:, np.newaxis]

    return loss_function(observed_counts=counts, predicted_counts=predicted_counts)


# @blockPrinting
def loss_counts_uncongested_network(equilibrator: LUE_Equilibrator,
                                    network: TNetwork,
                                    counts: ColumnVector,
                                    utility_function: UtilityFunction):
    """ Compute objective function with naive prediction assuming an uncongested network"""

    features_Y = utility_function.features_Y
    features_Z = utility_function.features_Z
    theta = utility_function.values

    with block_output(show_stdout=False, show_stderr=False):
        results_uncongested_eq = equilibrator.path_based_suelogit_equilibrium(
            theta=theta,
            network=network,
            features_Y=features_Y,
            features_Z=features_Z,
            column_generation={'n_paths': None, 'paths_selection': None},
            path_size_correction=0,
            uncongested_mode=True)

    predicted_counts = np.array(list(results_uncongested_eq['x'].values()))[:, np.newaxis]

    return loss_function(observed_counts=counts, predicted_counts=predicted_counts), predicted_counts


def loss_counts_equilikely_choices(**kwargs):
    utility_function = copy.deepcopy(kwargs.pop('utility_function'))
    utility_function.values = dict.fromkeys(utility_function.features, 0)

    return loss_counts_uncongested_network(utility_function=utility_function, **kwargs)


def monotonocity_traffic_count_functions(network: TNetwork,
                                         equilibrator: LUE_Equilibrator,
                                         utility_function: UtilityFunction,
                                         feature: str,
                                         theta_attr_grid: np.ndarray):
    """ Analyze the monotonicity of the traffic counts functions and it is analyzed if the range of the function includes the traffic counts measurements (vertical var)"""

    features_Y = utility_function.features_Y
    features_Z = utility_function.features_Z
    theta = utility_function.values

    theta_current = copy.deepcopy(theta)

    x_eq_vals = []
    x_ids = []
    thetas_list = []

    for iter, theta_attr_val in zip(range(len(theta_attr_grid)), theta_attr_grid):

        printProgressBar(iter, len(theta_attr_grid) - 1, prefix='Progress:', suffix='', length=20)

        theta_current[feature] = theta_attr_val

        results_eq = equilibrator.path_based_suelogit_equilibrium(network=network,
                                                                  theta=theta_current,
                                                                  features_Y=features_Y,
                                                                  features_Z=features_Z,
                                                                  silent_mode=True
                                                                  )
        x_eq = list(results_eq['x'].values())

        thetas_list.extend([theta_attr_val] * len(x_eq))

        # Add 1 to ids so they match the visualization

        x_ids_list = []
        for key in list(results_eq['x'].keys()):
            x_ids_list.append((key[0] + 1, key[1] + 1, key[2]))

        x_ids.extend(x_ids_list)
        x_eq_vals.extend(x_eq)

    # # Create dictionary of values by link
    # n_links = len(x_eq_vals[0])
    # traffic_count_links_dict = {}
    # for theta_grid in range(len(x_eq_vals)):
    #     for link_id in range(len(x_eq_vals[theta_grid])):
    #         traffic_count_links_dict[link_id] = x_eq_vals[theta_grid][link_id]

    # Create pandas dataframe
    traffic_count_links_df = pd.DataFrame({'link': x_ids, 'theta': thetas_list, 'count': x_eq_vals})

    return traffic_count_links_df


def compute_vot(parameters: Dict,
                numerator_feature='tt',
                denominator_feature='c'):
    if denominator_feature in parameters:

        if parameters['c'] != 0:
            return parameters[numerator_feature] / parameters[denominator_feature]
        else:
            return float('nan')

    else:
        return float('nan')


def jacobian_response_function(theta,
                               design_matrix: Matrix,
                               q: ColumnVector,
                               D: Matrix,
                               M: Matrix,
                               C: Matrix,
                               paths_specific_utility: ColumnVector,
                               paths_probabilities: ColumnVector = None,
                               counts: ColumnVector = None,
                               features_idxs: List = None,
                               numeric=False,
                               normalization=True):
    '''

        :param theta:
        :param design_matrix: data at link level
        :param q: dense vector of OD demand
        :param paths_probabilities:  Path choice probabilities

        :return:
        '''

    paths_idxs = []
    path_reduction = False

    if counts is not None:

        # The path reduction will not change results but it is worth to do in cases with low coverage
        if path_reduction:
            # idx_links_nas = np.where(np.isnan(counts))[0]
            idx_links_nonas = np.where(~np.isnan(counts))[0]

            # Identify indices where paths traverse some link with traffic counts.
            paths_idxs = list(np.where(np.sum(D[idx_links_nonas, :], axis=0) == 1)[0])

            print('Path reduction found ' + str(len(paths_idxs)) + ' seemingly irrelevant paths')

        # Subsampling of paths

    # Path probabilities (TODO: I may speed up this operation by avoiding elementwise division)

    if paths_probabilities is None:
        paths_probabilities = compute_paths_probabilities(theta,
                                                          design_matrix,
                                                          D,
                                                          C=C,
                                                          attr_types=None,
                                                          normalization=normalization,
                                                          paths_specific_utility=paths_specific_utility
                                                          )

    else:
        pass
        # print('The computation of the Jacobian may be innacurate when path choice probabilities are provided')

    if numeric:

        # print('Jacobian is computed numerically')

        J = nd.Jacobian(response_function_numeric_jacobian, method="central")(
            theta,
            design_matrix=design_matrix,
            C=C,
            D=D,
            M=M,
            q=q,
            normalization=normalization,
            paths_specific_utility=paths_specific_utility
        )

        # # Jacobian with automatic differentiation (nan max is not compatible)
        # F = jacobian(objective_function_numeric_jacobian)(theta_array.flatten(), design_matrix, counts,q, D, M, C, normalization)

        return J, paths_probabilities

    else:
        pass
        # print('Jacobian is computed analytically')

    # TODO: perform the gradient operation for each attribute using autograd
    # J = []

    # Jacobian/gradient of response function

    grad_m_terms = {}

    grad_m_terms[0] = M.T.dot(q)
    grad_m_terms[2] = paths_probabilities.dot(paths_probabilities.T)

    counter = 0
    n_features = theta.shape[0]
    jacobian = np.zeros((D.shape[0], n_features))

    if features_idxs is None:
        features_idxs = np.arange(theta.shape[0])

    # This operation is performed for each attribute k
    for k in range(n_features):

        if k in features_idxs:

            # printProgressBar(counter, n_features-1, prefix='Progress:', suffix='', length=20)

            # Attributes vector at link and path level
            Zk_x = design_matrix[:, k][:, np.newaxis]
            Zk_f = D.T.dot(Zk_x)

            # TODO: These operations are computationally expensive and may be vectorized further to make the complexity
            #  to not depend on the dimension of the utility vector

            grad_m_terms[3] = -(np.ones(Zk_f.shape).dot(Zk_f.T) - Zk_f.dot(np.ones(Zk_f.shape).T))

            # grad_m = D.dot(np.multiply(grad_m_terms[0],
            #                             np.multiply(grad_m_terms[2], grad_m_terms[3]))).dot(
            #     np.ones(Zk_f.shape))
            grad_m = D.dot(np.multiply(grad_m_terms[0],
                                       np.multiply(C, np.multiply(grad_m_terms[2],grad_m_terms[3])))).dot(np.ones(Zk_f.shape))

        else:
            grad_m = np.zeros((jacobian.shape[0],1))

        # Gradient of objective function

        # if counter == 0:
        #     jacobian = grad_m

        # if k > 0:
        jacobian[:,counter] = grad_m.flatten() #np.column_stack((J, grad_m))

        counter += 1

    # counter = 0
    # for feature_idx in features_idxs:
    #     jacobian[:, feature_idx] = J[:, counter]
    #     counter += 1

    if counts is not None:
        return jacobian, paths_probabilities, D, M
    else:
        return jacobian, paths_probabilities


def compute_links_utilities(theta,
                            design_matrix: Matrix):
    # Linkutilities
    v_x = np.dot(design_matrix, theta)

    return v_x


def compute_paths_utilities(theta,
                            design_matrix: Matrix,
                            D: Matrix,
                            C: Matrix,
                            paths_specific_utility=0,
                            normalization=True):
    # link utilities
    v_x = compute_links_utilities(theta, design_matrix)

    # path utilities
    v_f = np.dot(D.T, v_x)

    v_f = v_f + paths_specific_utility

    assert v_f.shape[1] == 1, 'vector of path utilities is not a column vector'

    if normalization is True:
        # softmax trick (TODO: this operation is computationally expensive and does not allow to use automatic differentiation)
        v_f = v_normalization(v_f, C)

    assert v_f.shape[1] == 1, 'vector of link utilities is not a column vector'

    return v_f


def compute_paths_utilities_by_attribute(theta,
                                         design_matrix: Matrix,
                                         D: Matrix,
                                         C: Matrix,
                                         normalization=True,
                                         attr_types=None,
                                         paths_specific_utility=0
                                         ):
    # If the attribute type is absolute, the computation of link utilities can be done efficiently,

    counter_idx = 0

    absolute_attr_type_idx = []

    if attr_types is not None:
        for attr, type in attr_types.items():

            if attr == 'absolute':
                absolute_attr_type_idx.append(counter_idx)

            counter_idx += 1

    # link utilities
    v_x = compute_links_utilities(theta, design_matrix)

    # path utilities
    v_f = D.T.dot(v_x)

    if paths_specific_utility > 0:
        v_f = v_f + paths_specific_utility

    if normalization is True:
        # softmax trick (TODO: this operation is computationally expensive)
        v_f = v_normalization(v_f.reshape(v_f.shape[0]), C)[:, np.newaxis]

    return v_f


def compute_paths_probabilities(theta,
                                design_matrix: Matrix,
                                D: Matrix,
                                C: Matrix,
                                attr_types=None,
                                normalization=True,
                                paths_specific_utility: ColumnVector = 0
                                ):
    # TODO: the effect of incidents seems to be additive so normalizing by mean will not necessarily help

    epsilon = 1e-12

    if attr_types is None:

        v_f = compute_paths_utilities(theta=theta,
                                      design_matrix=design_matrix,
                                      D=D,
                                      C=C,
                                      normalization=normalization,
                                      paths_specific_utility=paths_specific_utility)

    else:
        v_f = compute_paths_utilities_by_attribute(theta=theta,
                                                   design_matrix=design_matrix,
                                                   D=D,
                                                   C=C,
                                                   normalization=normalization,
                                                   attr_types=attr_types,
                                                   paths_specific_utility=paths_specific_utility)

    # if correlation_factors is None:
    #     correlation_factors = np.ones(v_f.size)[:, np.newaxis]

    # Path probabilities (TODO: speed up this operation by avoiding elementwise division)
    p_f = np.divide(np.exp(v_f), np.dot(C, np.exp(v_f)) + epsilon)

    return p_f


def prediction_x(theta,
                 design_matrix: Matrix,
                 D: Matrix,
                 C: Matrix,
                 M: Matrix,
                 q: ColumnVector,
                 paths_probabilities: ColumnVector = None):
    # Link and path utilities
    # v_x = design_matrix.dot(theta)

    # A = M.T.dot(M)

    # Path probabilities
    if paths_probabilities is None:
        paths_probabilities = compute_paths_probabilities(theta, design_matrix, D, C)

    # f = np.multiply(M.T.dot(q), p_f)
    x_pred = D.dot(np.multiply(M.T.dot(q), paths_probabilities))  # D.dot(f)

    return x_pred


def compute_response_function(D: Matrix,
                              M: Matrix,
                              q: ColumnVector,
                              paths_probabilities):
    # Response function
    m = np.dot(D, np.multiply(M.T.dot(q), paths_probabilities))

    return m


def error_by_link(observed_counts: ColumnVector,
                  predicted_counts: ColumnVector,
                  show_nan=True):
    """ Difference between observed counts and counts computed at equilibrium.

    """

    # assert counts.shape[0] == predicted_counts.shape[0], ' shape of vectors is different'
    assert observed_counts.shape[1] == 1 and predicted_counts.shape[1] == 1, ' no column vectors'

    # Store the number of elements different than nan
    adjusted_n = np.count_nonzero(~np.isnan(observed_counts))
    #
    # counts = fake_observed_counts(xct_hat = predicted_counts, counts= counts)

    # list(np.sort(-((predicted_counts - counts) ** 2).flatten())*-1)

    if show_nan:
        return (predicted_counts - observed_counts)

    else:

        idx_nonas = np.where(~np.isnan(observed_counts))[0]

        return (predicted_counts[idx_nonas] - observed_counts[idx_nonas])


def loss_function_by_link(observed_counts: ColumnVector,
                          predicted_counts: ColumnVector):
    """ Difference between observed counts and counts computed at equilibrium.

    """

    # assert counts.shape[0] == predicted_counts.shape[0], ' shape of vectors is different'
    assert observed_counts.shape[1] == 1 and predicted_counts.shape[1] == 1, ' no column vectors'

    # Store the number of elements different than nan
    adjusted_n = np.count_nonzero(~np.isnan(observed_counts))
    #
    # counts = fake_observed_counts(xct_hat = predicted_counts, counts= counts)

    # list(np.sort(-((predicted_counts - counts) ** 2).flatten())*-1)

    return (predicted_counts - observed_counts) ** 2 / adjusted_n


def loss_function(observed_counts: ColumnVector,
                  predicted_counts: ColumnVector):
    """ Difference between observed counts and counts computed at equilibrium
        It takes into account that there may not full link coverage.

    """

    # assert counts.shape[0] == predicted_counts.shape[0], ' shape of vectors is different'
    assert observed_counts.shape[1] == 1 and predicted_counts.shape[1] == 1, ' no column vectors'

    # Store the number of elements different than nan
    adjusted_n = np.count_nonzero(~np.isnan(observed_counts))

    observed_counts = fake_observed_counts(predicted_counts=predicted_counts, observed_counts=observed_counts)

    # list(np.sort(-((predicted_counts - counts) ** 2).flatten())*-1)

    return np.sum((predicted_counts - observed_counts) ** 2)  # /adjusted_n


def lasso_soft_thresholding_operator(lambda_hp, theta_estimate):
    # TODO: theta needs to be refitted each time, the soft-thresholding only
    #  gives us a direction to perform gradient updates. A coordinate descent is used for
    # multiple predictors

    regularized_theta = copy.deepcopy(theta_estimate)

    for attr, theta_val in theta_estimate.items():

        if theta_val > lambda_hp:
            regularized_theta[attr] = theta_val - lambda_hp

        if theta_val < -lambda_hp:
            regularized_theta[attr] = theta_val + lambda_hp

        if abs(theta_val) <= lambda_hp:
            regularized_theta[attr] = 0

    return regularized_theta


def lasso_regularization(network,
                         grid_lambda,
                         theta_estimate,
                         features_Y,
                         features_Z: [],
                         equilibrator,
                         counts: ColumnVector,
                         standardization: dict):
    print('\nPerforming Lasso regularization with lambda grid:', grid_lambda)

    # Write soft-thresholding operator

    # ref: https://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-15.pdf

    regularized_thetas = {}
    losses = {}

    for lambda_hp in grid_lambda:
        regularized_thetas[lambda_hp] = lasso_soft_thresholding_operator(lambda_hp, theta_estimate)

        # Run stochastic user equilibrium
        results_eq = equilibrator.path_based_suelogit_equilibrium(Nt=network,
                                                                  theta=regularized_thetas[lambda_hp],
                                                                  features_Y=features_Y,
                                                                  features_Z=features_Z,
                                                                  params=equilibrator,
                                                                  silent_mode=True,
                                                                  standardization=standardization)

        x_eq = np.array(list(results_eq['x'].values()))[:, np.newaxis]

        losses[lambda_hp] = loss_function(observed_counts=counts, predicted_counts=x_eq)

        # print('theta:', regularized_thetas[lambda_hp] )
        print('\nlambda: ', "{0:.3}".format(float(lambda_hp)))
        print('current theta: ', str({key: round(val, 3) for key, val in regularized_thetas[lambda_hp].items()}))
        print('loss:', losses[lambda_hp])

    return regularized_thetas, losses[lambda_hp]


def mean_count_prediction(counts: ColumnVector,
                          mean_x=None) -> Tuple[Any, Any]:
    """
    Benchmark prediction with naive model that predicts the mean value count.
    If a mean value is provided, then the mean of the training sample is not computed
    If none, it computes the mean of the observed counts
    """

    if mean_x is None:
        mean_x = np.nanmean(counts)
        x_benchmark = mean_x * np.ones(counts.shape)

    else:
        x_benchmark = mean_x * np.ones(counts.shape)

    return loss_function(observed_counts=counts, predicted_counts=x_benchmark), mean_x


def response_function_numeric_jacobian(theta, **args):
    """ Wrapper function to compute the Jacobian numerically.
    """

    # return np.sum(objective_function(np.array(theta)[:, np.newaxis], design_matrix = args[0], counts= args[1], q= args[2], D= args[3], M = args[4], C = args[5]))

    D = args['D']
    M = args['M']
    q = args['q']

    p_f = compute_paths_probabilities(theta,
                                      design_matrix=args['design_matrix'],
                                      D=D,
                                      C=args['C'],
                                      normalization=True
                                      )

    return np.dot(D, np.multiply(M.T.dot(q), p_f))


def objective_function_numeric_jacobian(theta, **args):
    """ Wrapper function to compute the Jacobian numerically.
    """

    return Learner.compute_objective_function(
        np.array(theta)[:, np.newaxis],
        design_matrix=args['design_matrix'],
        counts=args['counts'],
        q=args['q'],
        D=args['D'],
        M=args['M'],
        C=args['C'],
        paths_specific_utility=args.get('paths_specific_utility', 0),
        normalization=args.get('normalization')
    )


def objective_function_numeric_hessian(theta, **args):
    """ Wrapper function to compute the Hessian numerically. This type of computation is unreliable and unstable. The hessian is not guaranteed to be PSD which generates issues with its inversion to obtain the covariance matrix
    """

    # return np.sum(objective_function(np.array(theta)[:, np.newaxis], design_matrix = args[0], counts= args[1], q= args[2], D= args[3], M = args[4], C = args[5]))

    return np.sum(
        Learner.compute_objective_function(
            np.array(theta)[:, np.newaxis],
            design_matrix=args['design_matrix'],
            counts=args['counts'],
            q=args['q'],
            D=args['D'],
            M=args['M'],
            C=args['C'],
            paths_specific_utility=args['paths_specific_utility'],
            normalization=args['normalization']))


def gradient_objective_function(theta: ColumnVector,
                                design_matrix: Matrix,
                                counts: ColumnVector,
                                q: ColumnVector,
                                D: Matrix,
                                M: Matrix,
                                C: Matrix,
                                numeric=False,
                                paths_probabilities: ColumnVector = None,
                                features_idxs: List = None,
                                standardization: dict = None,
                                paths_specific_utility=0
                                ) -> ColumnVector:
    path_reduction = False

    # The path reduction will change results by eliminating those not traversing observed paths will change results, because in the logit model the alternatives probability are dependent of the utilities of the other in the same path set.

    if path_reduction:
        # idx_links_nas = np.where(np.isnan(counts))[0]
        idx_links_nonas = np.where(~np.isnan(counts))[0]

        # Identify indices where paths traverse some link with traffic counts.
        paths_idxs = list(np.where(np.sum(D[idx_links_nonas, :], axis=0) == 1)[0])

        print('Path reduction found ' + str(len(paths_idxs)) + ' seemingly irrelevant paths')

    # Subsampling of paths

    # Path probabilities (TODO: I may speed up this operation by avoiding elementwise division)

    if paths_probabilities is None:
        paths_probabilities = compute_paths_probabilities(theta,
                                                          design_matrix,
                                                          D,
                                                          C,
                                                          paths_specific_utility=paths_specific_utility)
    else:
        pass
        # print('The computation of the gradient may be innacurate when path choice probabilities are provided')

    if numeric:
        t0 = time.time()

        numeric_grad = np.sum(nd.Gradient(numeric_gradient_objective_function)(
            theta.T,
            design_matrix=design_matrix,
            counts=counts,
            q=q,
            D=D,
            M=M,
            C=C,
            paths_specific_utility=paths_specific_utility
        ), axis=0).reshape(theta.shape)

        print(time.time() - t0)

        return numeric_grad

    # TODO: perform the gradient operation for each attribute using a tensor

    # Jacobian/gradient of response function

    grad_m_terms = {}

    grad_m_terms[0] = M.T.dot(q)

    # This is the availability matrix and it is very expensive to compute when using matrix operation M.T.dot(M) but not when calling function choice_set_matrix_from_M
    grad_m_terms[1] = paths_probabilities.dot(paths_probabilities.T)

    # This operation is performed for each attribute k. Then, the compl

    # Objective function
    m = compute_response_function(D, M, q, paths_probabilities)

    # Store the number of elements different than nan
    # adjusted_n = np.count_nonzero(~np.isnan(counts))

    # To account for missing link counts
    counts = fake_observed_counts(predicted_counts=m, observed_counts=counts)

    counter = 0
    n_features = theta.shape[0]
    gradient = np.zeros_like(theta, dtype=np.float64)

    if features_idxs is None:
        features_idxs = np.arange(theta.shape[0])

    for k in range(n_features):

        if k in features_idxs:

            # printProgressBar(counter, n_features - 1, prefix='Progress:', suffix='', length=20)

            # Attributes vector at link and path levels
            Zk_x = design_matrix[:, k][:, np.newaxis]
            Zk_f = D.T.dot(Zk_x)

            if standardization is not None:
                Zk_f = preprocessing.scale(Zk_f,
                                           with_mean=standardization['mean'],
                                           with_std=standardization['sd'],
                                           axis=0)

            grad_m_terms[2] = Zk_f.dot(np.ones(Zk_f.shape).T)
            # grad_m_terms[3] = -(np.ones(Zk_f.shape).dot(Zk_f.T) - Zk_f.dot(np.ones(Zk_f.shape).T))

            grad_m = D.dot(
                np.multiply(grad_m_terms[0],
                            np.multiply(C,
                                        np.multiply(grad_m_terms[1],
                                                    grad_m_terms[2]-grad_m_terms[2].T)))).dot(np.ones(Zk_f.shape))

            gradient[counter] = float(2 * grad_m.T.dot(m - counts))

        else:
            gradient[counter] = 0

        counter += 1

    return gradient


def numeric_gradient_objective_function(theta: ColumnVector,
                                        **kwargs):
    # When p_f is provided, the gradient becomes apparently zero because the objective function does not vary explicitly
    # on theta

    return Learner.compute_objective_function(
        np.array(theta)[:, np.newaxis],
        design_matrix=kwargs['design_matrix'],
        counts=kwargs['counts'],
        p_f=kwargs.get('p_f', None),
        q=kwargs['q'],
        D=kwargs['D'],
        M=kwargs['M'],
        C=kwargs['C'],
        paths_specific_utility=kwargs.get('paths_specific_utility', 0),
        normalization=kwargs.get('normalization', True)
    )


def numeric_hessian_objective_function(theta: ColumnVector,
                                       counts: ColumnVector,
                                       design_matrix: Matrix,
                                       q: ColumnVector,
                                       D: Matrix,
                                       M: Matrix,
                                       C: Matrix,
                                       paths_probabilities: ColumnVector = None,
                                       normalization=True,
                                       paths_specific_utility=0):
    '''

    Numeric Hessian

    Args:
        theta:
        counts:
        design_matrix:
        q:
        D:
        M:
        C:
        paths_probabilities:
        normalization:
        paths_specific_utility:

    Returns:

    '''

    # http://math.gmu.edu/~igriva/book/Appendix%20D.pdf
    # https: // www.eecs189.org / static / notes / n12.pdf

    if paths_probabilities is None:
        paths_probabilities = compute_paths_probabilities(theta, design_matrix, D, C)

    # print('Hessian is being computed numerically')

    # With normalization equals False, there is a 20X speed up in the numeric computation of the Hessian but inference is worsen
    H = nd.Hessian(objective_function_numeric_hessian)(list(theta.flatten()),
                                                       design_matrix=design_matrix,
                                                       counts=counts,
                                                       q=q,
                                                       D=D,
                                                       M=M,
                                                       C=C,
                                                       p_f=paths_probabilities,
                                                       normalization=normalization,
                                                       paths_specific_utility=paths_specific_utility)

    dimension = int(H.size ** 0.5)
    H = H.reshape(dimension, dimension)

    # # Automatic differentiation must be used with normalization = False because no gradient for np.nanmax is registered in autograd
    # It generates NA after inversion of Hessian in many instances. Also, it does not work with only a subset of the traffic counts is available

    # print('Hessian is being computed with automatic differentiation')
    # df = egrad(objective_function_numeric_hessian)
    # H = jacobian(egrad(df))
    # H = H(theta_array.flatten(), design_matrix, counts,q, D, M, C, normalization)

    return H


def diagonal_hessian_objective_function(theta: ColumnVector,
                                        counts: ColumnVector,
                                        design_matrix: Matrix,
                                        q: ColumnVector,
                                        D: Matrix,
                                        M: Matrix,
                                        C: Matrix,
                                        paths_probabilities: ColumnVector = None,
                                        numeric=False,
                                        normalization=True,
                                        paths_specific_utility=0):
    # TODO: Enable estimation of Hessian using batch size as it is done for gradient computation

    # Hesisan was debugged. Something mandatory is that at the optima, the hessian is positive as it is a minimizer

    # http://math.gmu.edu/~igriva/book/Appendix%20D.pdf
    # https: // www.eecs189.org / static / notes / n12.pdf

    if paths_probabilities is None:
        paths_probabilities = compute_paths_probabilities(theta, design_matrix, D, C)

    hessian_l2norm = []

    if numeric:

        # print('Hessian is being computed numerically')
        theta_array = theta

        # With normalization equals False, there is a 20X speed up in the numeric computation of the Hessian but inference is worsen
        H = numeric_hessian_objective_function(theta=theta_array,
                                               design_matrix=design_matrix,
                                               counts=counts,
                                               q=q,
                                               D=D,
                                               M=M,
                                               C=C,
                                               paths_probabilities=paths_probabilities,
                                               normalization=normalization,
                                               paths_specific_utility=paths_specific_utility)
        H = np.diag(H)

        # # Automatic differentiation must be used with normalization = False because no gradient for np.nanmax is registered in autograd
        # It generates NA after inversion of Hessian in many instances. Also, it does not work with only a subset of the traffic counts is available

        # print('Hessian is being computed with automatic differentiation')
        # df = egrad(objective_function_numeric_hessian)
        # H = jacobian(egrad(df))
        # H = H(theta_array.flatten(), design_matrix, counts,q, D, M, C, normalization)

    else:

        # print('Hessian is being computed analytically')

        grad_p_f_terms = {}
        grad_p_f_terms[1] = paths_probabilities.dot(paths_probabilities.T)

        jac_m_k_terms = {}
        jac_m_k_terms[0] = M.T.dot(q)

        # This operation is performed for each attribute k
        for k in np.arange(theta.shape[0]):  # np.arange(len([*features_Y,*features])):
            # k = 0

            # Attributes vector at link and path level
            Zk_x = design_matrix[:, k][:, np.newaxis]
            Zk_f = D.T.dot(Zk_x)

            # Gradient for path probabilities

              # grad_m_terms[1] # grad_m_terms[2]
            grad_p_f_terms[2] = (np.ones(Zk_f.shape).dot(Zk_f.T) - Zk_f.dot(np.ones(Zk_f.shape).T))  # grad_m_terms[3]
            grad_p_f = np.multiply(C, np.multiply(grad_p_f_terms[1], grad_p_f_terms[2])).dot(
                np.ones(Zk_f.shape))

            # Gradient of objective function
            jac_m_k_terms[1] = C  # computing M.T.dot(M) is too slow
            jac_m_k_terms[2] = grad_p_f_terms[1]
            jac_m_k_terms[3] = (np.ones(Zk_f.shape).dot(Zk_f.T) - Zk_f.dot(np.ones(Zk_f.shape).T))

            jac_m_k = D.dot(np.multiply(jac_m_k_terms[0], np.multiply(jac_m_k_terms[1], np.multiply(jac_m_k_terms[2],
                                                                                                    jac_m_k_terms[
                                                                                                        3])))).dot(
                np.ones(Zk_f.shape))

            # jac_m = D.dot(np.multiply(jac_m_k_terms[0], np.multiply(jac_m_k_terms[1], np.multiply(jac_m_k_terms[2],jac_m_k_terms[3]))))

            hessian_m_terms = {}
            hessian_m_terms[0] = jac_m_k_terms[0]  # jac_m_k_terms[0]
            hessian_m_terms[1] = jac_m_k_terms[1]  # jac_m_k_terms[1]
            hessian_m_terms[2] = grad_p_f.dot(paths_probabilities.T) + paths_probabilities.dot(
                grad_p_f.T)  # This is the key term
            hessian_m_terms[3] = jac_m_k_terms[3]

            hessian_m_k = D.dot(np.multiply(hessian_m_terms[0], np.multiply(hessian_m_terms[1],
                                                                            np.multiply(hessian_m_terms[2],
                                                                                        hessian_m_terms[3]))).dot(
                np.ones(Zk_f.shape)))

            # hessian_m_k = D.dot(np.multiply(hessian_m_terms[0], np.multiply(hessian_m_terms[1], np.multiply(hessian_m_terms[2],hessian_m_terms[3]))))

            m = compute_response_function(D, M, q, paths_probabilities)

            # Hessian of objective function

            # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts
            # hessian_l2norm_k = float(2 * jac_m_k.T.dot(jac_m_k) + hessian_m_k.T.dot(counts - m))

            # To account for missing link counts
            counts = fake_observed_counts(predicted_counts=m, observed_counts=counts)

            hessian_l2norm_k = -2 * float(hessian_m_k.T.dot(counts - m) - jac_m_k.T.dot(jac_m_k))

            # Store hessian for particular attribute k
            hessian_l2norm.append(hessian_l2norm_k)

        H = np.array(hessian_l2norm)

    return H
