# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla


def solve_bernoulli(A, E, B, trans=False, maxiter=100, after_maxiter=3, tol=1e-8):
    """Compute an approximate solution of a Bernoulli equation.

    Returns a matrix :math:`Y` such that :math:`Y Y^T`is an approximate solution
    of a (generalized) algebraic Bernoulli equation:

    - if trans is `True`

      .. math::
          A^T X E + E^T X A
          - E^T X B B^T X E = 0.

    - if trans is `False`

      .. math::
          A X E^T + E X A^T
          - E X B^T B X E^T = 0.

    This function is based on https://link.springer.com/content/pdf/10.1007/s11075-007-9143-x.pdf.

    Parameters
    ----------
    A
        The operator A as a 2D |NumPy array|.
    E
        The operator E as a 2D |NumPy array| or `None`.
    B
        The operator B as a 2D |NumPy array|.
    trans
        Wether to solve transposed or standard Bernoulli equation.
    maxiter
        The maximum amount of iterations.
    after_maxiter
        The number of iterations which are to be performed after tolerance is reached.
    tol
        Tolerance for stopping criterion based on relative change of iterates.

    Returns
    -------
    Yp
        Approximate low-rank solution factor of Bernoulli equation
    """

    n = len(A)
    after_iter = 0

    assert n != 0

    if E is None:
        E = np.eye(n)

    if not trans:
        E = E.T
        A = A.T
        B = B.T

    for i in range(maxiter):
        Aprev = A
        lu, piv = spla.lu_factor(A.conj().T)
        detE = spla.det(E)
        if detE.real > 0.:
            c = np.abs(np.prod(np.diag(lu)) / detE)**(1. / n)  # |det(A_k)/det(E)|^(1/n)
        else:
            c = 1
        AinvTET = spla.lu_solve((lu, piv), E.conj().T)
        A = 0.5 * ((1 / c) * A + c * AinvTET.conj().T @ E)
        BT = (1 / np.sqrt(2 * c)) * np.vstack((B.conj().T, c * B.conj().T @ AinvTET))
        Q, R, perm = spla.qr(BT, mode='economic', pivoting=True)
        B = np.eye(n)[perm].T @ R.conj().T
        if after_iter > after_maxiter:
            break
        rnorm = spla.norm(A - Aprev) / spla.norm(A)
        if rnorm <= tol:
            after_iter += 1

    Q, R, _ = spla.qr(E.conj() - A.conj(), pivoting=True)
    nsp_rk = 0
    for r in R:
        if np.allclose(r, np.zeros(r.shape)):
            nsp_rk = nsp_rk + 1
    Q = Q[:, n-nsp_rk:].conj()
    _, R = spla.qr(B.conj().T @ Q, mode='economic')
    Yp = spla.solve_triangular(R, np.sqrt(2) * Q.conj().T, trans='C')

    return Yp.conj().T
