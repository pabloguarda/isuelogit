""" Numeric operations, approximations and tranformations """

def round_almost_zero_flows(x, tol = 1e-5):
    """ numerical stability """
    for i in range(len(x)):
        if abs(x[i]) < tol:
            x[i] = 0

    return x

def softmaxOverflowTrick(X):
    raise NotImplementedError