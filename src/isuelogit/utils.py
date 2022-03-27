from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from mytypes import Links, Options, Optional, DataFrame, Features, ParametersDict, Dict, List, Matrix, Feature, Union, Option


import inspect
import sys
import os
import numpy as np
import time
import copy
import networkx as nx

class Options(Dict):

    def __init__(self, **kwargs):

        self.recursive_update(options = self, new_options = kwargs)

        pass

    # def __call__(self):
    #     print('calling')
    #     return self._options

    # def __str__(self):
    #     return "member of Test"

    # def __repr__(self):
    #     return self._options

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, value):
        self._options = value

    def recursive_update(self, options: dict, new_options: Union[Option, dict]):

        for key, val in new_options.items():

            if isinstance(val, dict):
                options[key] = self.recursive_update(options.get(key, {}), val)

            else:
                options[key] = val

        return options


    def get_updated_options(self, **kwargs):
        ''' Wrapper function '''

        if 'options' not in kwargs.keys():
            options = copy.deepcopy(self)

        else:
            options = copy.deepcopy(kwargs['options'])

        return self.recursive_update(options = options,
                                     new_options =  kwargs['new_options'])


def get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs



def get_matrix_from_dict_attrs_values(W_dict: dict):
    # Return Matrix Y or Z using Y or Z_dict
    listW = []
    for i in W_dict.keys():
        listW.append([float(x) for x in W_dict[i]])

    return np.asarray(listW).T


def get_design_matrix(Y: Union[DataFrame,Dict[Feature, float]],
                      Z: Union[DataFrame,Dict[Feature, float]],
                      features_Y: Optional[List[Feature]] = None,
                      features_Z: Optional[List[Feature]] = None
                      ):

    '''

    Matrix of endogenous (|Y|x|K_Y|) and exogenous link attributes (|Z|x|K_Z|)

    Args:
        Y:
        Z:
        features_Y:
        features_Z:

    Returns:

    '''

    if features_Y is None:
        features_Y = list(Y.keys())

    if features_Z is None:
        features_Z = list(Z.keys())

    if len(features_Z)>0:
        Y_x = get_matrix_from_dict_attrs_values({k_y: Y[k_y] for k_y in features_Y})
        Z_x = get_matrix_from_dict_attrs_values({k_z: Z[k_z] for k_z in features_Z})
        YZ_x = np.column_stack([Y_x, Z_x])

    else:
        Y_x = get_matrix_from_dict_attrs_values({k_y: Y[k_y] for k_y in features_Y})
        YZ_x = np.column_stack([Y_x])

    return YZ_x




def v_normalization(v, C):
    '''
    :param v: this has to be an unidimensional array otherwise it returns a matrix
    :param C:
    :return: column vector with normalization
    '''

    # TODO: this function is not compatible with automatic differentiation

    # flattened = False
    if len(v.shape)>1:
        v = v.flatten()
        # flattened = True

    C = C.astype('float')
    C[C == 0] = float('nan') #np.nan
    v_max = np.nanmax(v * C, axis=1)
    # vC = v * C
    # vC[np.isnan(vC)] = float('-inf')


    # Without using npnanmax
    # C = C.astype('float')
    # C[C == 0] = np.nan
    # # v_max = np.nanmax(v * C, axis=1)
    # vC = v * C
    # vC[np.isnan(vC)] = -np.inf
    #
    # v_max = np.amax(vC, axis=1)

    return (v - v_max)[:,np.newaxis]

def softmax_probabilities(X, theta, avail=1):
    """ Multinomial logit (or softmax) probabilities
    :arg avail

    """
    # AVAIL

    # Z = X - np.amax(X,axis = 0).reshape(X.shape[0],1)
    Z = X

    return avail * np.exp(Z * theta) / np.sum(np.exp(Z * theta) * avail, axis=1).reshape(Z.shape[0], 1)



def stable_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator

    return softmax


def attribute_matrix_tolist(matrix, remove_zeros=True):
    '''This function assumed that the elements with 0 entries are the
    non-available alternatives
    '''
    list = []
    for row_index in range(matrix.shape[0]):
        row = matrix[row_index, :]
        if remove_zeros:
            list.append(row[row != 0])
    return list



def get_attribute_list_by_choiceset(C, z):
    ''' Return a list with attribute values (vector, i.e. Matrix 1D) for each alternative in the choice set (only with entries different than 0)
    :arg: C: Choice set matrix
    :arg: z: single attribute values
    '''
    z_avail = []

    for i in range(C.shape[0]):
        z_avail.append(z[np.where(C[i, :] == 1)[0]])

    return z_avail



def widetolong(wide_matrix):
    """Wide to long format
    The new matrix has one rows per route
    """

    if wide_matrix.ndim == 1:
        wide_matrix = wide_matrix.reshape(1, wide_matrix.shape[0])

    long_matrix = np.repeat(wide_matrix, repeats=np.sum(wide_matrix, axis=1), axis=0)

    return long_matrix


def non_zero_matrix(M):
    # # Remove null rows
    M = M[~np.all(M == 0, axis=1)]
    # Remove null columns
    mask = (M == 0).all(0)
    column_indices = np.where(mask)[0]
    M = M[:, ~mask]

    return M




def get_euclidean_distances_links(G, nodes_coordinate_label = 'pos'):

    pos_nodes = nx.get_node_attributes(G,nodes_coordinate_label)
    len_edges = {}
    for edge in G.edges():
        len_edges[edge] = np.linalg.norm(np.array(pos_nodes[edge[0]]) - np.array(pos_nodes[edge[1]]))

    return len_edges

def is_pos_def(x: Matrix):

    if np.isnan(x).any():
        return None

    return np.all(np.linalg.eigvals(x) > 0)

def is_pos_semidef(x: Matrix):

    if np.isnan(x).any():
        return None


    return np.all(np.linalg.eigvals(x) >= 0)

def round_almost_zero_flows(x, tol = 1e-5):
    """ numerical stability """
    for i in range(len(x)):
        if abs(x[i]) < tol:
            x[i] = 0

    return x

def softmaxOverflowTrick(X):
    raise NotImplementedError

def almost_zero(array: np.array, tol = 1e-5):
    array[np.abs(array) < tol] = 0

    return array

# def isAlmostZero(array: np.array):
#
#     np.all(array)
#     epsilon = 1e-10
#     array[np.abs(array) < epsilon] = 0
#
#     return array

def no_zeros(array: np.array,
             tol= 1e-10):

    array[np.abs(array) < tol] = tol

    return array

def almost_zero_gradient(gradient: np.array):
    epsilon = 1e-5
    gradient[np.abs(gradient) < epsilon] = 0

    return gradient

