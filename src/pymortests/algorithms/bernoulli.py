# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
from scipy.stats import ortho_group

from pymor.algorithms.bernoulli import solve_bernoulli

import pytest


n_list = [10, 50, 100]


@pytest.mark.parametrize('n', n_list)
@pytest.mark.parametrize('with_E', [False, True])
@pytest.mark.parametrize('trans', [False, True])
def test_dense(n, with_E, trans):
    np.random.seed(0)
    E = -ortho_group.rvs(dim=n)
    A = np.diag(np.concatenate((np.arange(-n + 4, 0), np.arange(1, 5)))) @ E
    B = np.random.randn(n, 1)

    if not trans:
        B = B.T

    Yp = solve_bernoulli(A, E, B, trans=trans)
    X = Yp @ Yp.T

    if not trans:
        assert spla.norm(A @ X @ E.T + E @ X @ A.T - E @ X @ B.T @ B @ X @ E.T) / spla.norm(X) < 1e-9
    else:
        assert spla.norm(A.T @ X @ E + E.T @ X @ A - E.T @ X @ B @ B.T @ X @ E) / spla.norm(X) < 1e-9
