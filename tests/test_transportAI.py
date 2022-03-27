#!/usr/bin/env python

"""Tests for `isuelogit` package."""

import pytest

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_package_import():
    import transportAI as tt
    assert tt.__version__

# Gradient and hessian checks
# a = hessian_l2norm(theta, YZ_x, Ix, idx_links = np.arange(0, n))
# print('hessian_diff : ' + str(hessian_check(theta=np.array(list(theta0.values()))[:, np.newaxis])))
# assert gradient_check(theta = np.array(list(theta0.values()))[:,np.newaxis])<1e-6, 'unreliable gradients'
# print('gradient_diff : ' + str(gradient_objective_function_check(theta =np.array(list(theta0.values()))[:, np.newaxis], YZ_x = YZ_x, q = q, Ix = Ix, Iq = Iq, C = C, x_bar = x_bar)))
# gradient_check(theta = 10*np.ones(len(np.array(list(theta0.values()))))[:,np.newaxis])


# print('gradient_diff : ' + str(gradient_check(theta=theta)))

# print('Jacobian is computed numerically')
# numeric_jacobian = nd.Jacobian(response_function_numeric_jacobian, method="central")(
#     np.array(list(theta.flatten()))[:, np.newaxis],
#     design_matrix = design_matrix,
#     C = C,
#     D = D,
#     M = M,
#     q = q)

# np.allclose(J,numeric_jacobian)