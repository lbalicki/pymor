# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.eigs import eigs
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator


def solve_bernoulli(A, E, B, trans=False, maxiter=100, after_maxiter=3, tol=1e-8):
    """Compute an approximate solution of a Bernoulli equation.

    Returns a matrix :math:`Y` such that :math:`Y Y^T`is an approximate solution
    of a (generalized) algebraic Bernoulli equation:

    - if trans is `True`

      .. math::
          A^H X E + E^H X A
          - E^H X B B^H X E = 0.

    - if trans is `False`

      .. math::
          A X E^H + E X A^H
          - E X B^H B X E^H = 0.

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

    logger = getLogger('pymor.algorithms.bernoulli.solve_bernoulli')

    n = len(A)
    after_iter = 0

    assert n != 0

    if E is None:
        E = np.eye(n)

    if not trans:
        E = E.conj().T
        A = A.conj().T
        B = B.conj().T

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
        logger.info(f'Relative change of iterates at step {i}: {rnorm:.5e}')
        if rnorm <= tol:
            after_iter += 1

    # Q = spla.null_space(E.conj().T - A.conj().T)
    Q, R, _ = spla.qr(E.conj() - A.conj(), pivoting=True)
    nsp_rk = 0
    for r in R:
        if np.allclose(r, np.zeros(r.shape)):
            nsp_rk = nsp_rk + 1
    Q = Q[:, n-nsp_rk:].conj()
    _, R = spla.qr(B.conj().T @ Q, mode='economic')
    Yp = spla.solve_triangular(R, np.sqrt(2) * Q.conj().T, trans='C')

    return Yp.conj().T


def bernoulli_stabilize(A, E, B, trans=False, num_eig=50):
    """"Compute Bernoulli stabilizing feedback."""
    logger = getLogger('pymor.algorithms.bernoulli.bernoulli_stabilize')
    assert num_eig < A.source.dim

    if E is None:
        E = IdentityOperator(A.source)

    ew, rev = eigs(A, E=E, k=num_eig, sigma=0)
    unst_idx = np.where(ew.real > 0.)
    unst_ews = ew[unst_idx]

    logger.info(f'Detected {len(unst_ews)} antistable eigenvalues in Bernoulli stabilization.')

    if len(unst_ews) == 0:
        return A.source.zeros(len(B))

    unst_levs = A.source.empty(reserve=len(unst_ews))
    for ue in unst_ews:
        _, lev = eigs(A, E=E, k=1, l=2, sigma=ue, right_EVP=False)
        unst_levs.append(lev)

    unst_revs = rev[unst_idx[0]]

    Mt = E.apply2(unst_levs, unst_revs)
    At = A.apply2(unst_levs, unst_revs)

    if trans:
        Bt = unst_levs.dot(B)
    else:
        Bt = B.dot(unst_revs)

    Yz = solve_bernoulli(At, Mt, Bt, trans=trans)
    Xz = Yz @ Yz.conj().T

    if trans:
        K = E.apply_adjoint(unst_levs.conj()).lincomb(B.dot(unst_levs) @ Xz)
    else:
        K = E.apply_adjoint(unst_revs.conj()).lincomb(B.dot(unst_revs) @ Xz)

    return K.real


def bernoulli_stabilize_dense(A, E, B, trans=False):
    """"Compute Bernoulli stabilizing feedback for dense systems."""
    if E is None:
        E = IdentityOperator(A.source)

    A = to_matrix(A, format='dense')
    E = to_matrix(E, format='dense')
    B = B.to_numpy().T

    ew, lev, rev = spla.eig(A, E, True)
    unst_idx = np.where(ew.real > 0.)
    unst_ews = ew[unst_idx]

    if len(unst_ews) == 0:
        return A.source.zeros(len(B))

    unst_levs = lev[:, unst_idx][:, 0, :]
    unst_revs = rev[:, unst_idx][:, 0, :]

    Mt = unst_levs.conj().T @ E @ unst_revs
    At = unst_levs.conj().T @ A @ unst_revs

    if trans:
        Bt = unst_levs.conj().T @ B
    else:
        Bt = B.conj().T @ unst_revs

    Yz = solve_bernoulli(At, Mt, Bt, trans=trans)
    Xz = Yz @ Yz.conj().T

    if trans:
        K = E.conj().T @ unst_levs.conj() @ Xz @ unst_levs.conj().T @ B
    else:
        K = E.conj().T @ unst_revs.conj() @ Xz @ unst_revs.conj().T @ B

    return K.real
