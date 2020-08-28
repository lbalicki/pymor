# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.lradi import projection_shifts_init, projection_shifts
from pymor.algorithms.bernoulli import bernoulli_stabilize
from pymor.algorithms.genericsolvers import _parse_options
from pymor.core.defaults import defaults
from pymor.vectorarrays.constructions import cat_arrays
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator
from pymor.operators.constructions import LowRankOperator


@defaults('ricc_tol', 'lyap_tol', 'newton_maxiter', 'lradi_maxiter', 'initial_feedback',
          'lradi_shifts', 'projection_shifts_init_maxiter', 'projection_shifts_init_seed')
def ricc_lrcf_solver_options(ricc_tol=1e-9,
                             lyap_tol=1e-11,
                             newton_maxiter=20,
                             lradi_maxiter=500,
                             initial_feedback=None,
                             lradi_shifts='projection_shifts',
                             projection_shifts_init_maxiter=20,
                             projection_shifts_init_seed=None):
    """Return available Lyapunov solvers with default options.

    Parameters
    ----------
    lradi_tol
        See :func:`solve_lyap_lrcf`.
    lradi_maxiter
        See :func:`solve_lyap_lrcf`.
    lradi_shifts
        See :func:`solve_lyap_lrcf`.
    projection_shifts_init_maxiter
        See :func:`projection_shifts_init`.
    projection_shifts_init_seed
        See :func:`projection_shifts_init`.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'lrnadi': {'type': 'lrnadi',
                       'ricc_tol': ricc_tol,
                       'lyap_tol': lyap_tol,
                       'newton_maxiter': newton_maxiter,
                       'lradi_maxiter': lradi_maxiter,
                       'initial_feedback': initial_feedback,
                       'shifts': lradi_shifts,
                       'shift_options':
                       {'projection_shifts': {'type': 'projection_shifts',
                                              'init_maxiter': projection_shifts_init_maxiter,
                                              'init_seed': projection_shifts_init_seed}}}}


def solve_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None, return_K=False):
    """Solve Riccati equation via Newton iteration."""

    options = _parse_options(options, ricc_lrcf_solver_options(), 'lrnadi', None, False)
    logger = getLogger('pymor.algorithms.stokes_newton_lradi.solve_ricc_lrcf')

    shift_options = options['shift_options'][options['shifts']]
    use_ricc_res = False

    if E is None:
        E = IdentityOperator(A.source)

    if S is not None:
        raise NotImplementedError

    if R is not None:
        raise NotImplementedError

    if not trans:
        B, C = C, B

    K = options['initial_feedback']

    if K is None:
        K = bernoulli_stabilize(A, E, B, trans=trans)

    for k in range(options['newton_maxiter']):
        W = cat_arrays([C, K])
        Z = A.source.empty(reserve=len(W) * options['lradi_maxiter'])
        K_newton = A.source.zeros(len(K))
        WTW = W.gramian()
        if k == 0:
            CTC = WTW[:len(C), :len(C)]
            ricc_init_res = ricc_res = np.linalg.norm(CTC, ord=2)
            Ctol = ricc_init_res * options['ricc_tol']
            Csqrttol = ricc_init_res * np.sqrt(options['ricc_tol'])
        lyap_res = lyap_init_res = np.linalg.norm(WTW, ord=2)
        Wtol = lyap_init_res * options['lyap_tol']
        if trans:
            shifts = projection_shifts_init(A - LowRankOperator(B, np.eye(len(B)), K), E, C, shift_options)
        else:
            shifts = projection_shifts_init(A - LowRankOperator(K, np.eye(len(K)), B), E, C, shift_options)
        j = 0
        j_shift = 0

        while lyap_res > Wtol and j < options['lradi_maxiter']:
            s = shifts[j_shift]
            if s.imag == 0.:
                j = j + 1
                j_shift = j_shift + 1
                gs = -2 * s.real
                if trans:
                    V = (A + s.real * E - LowRankOperator(B, np.eye(len(B)), K)).apply_inverse_adjoint(W)
                    Vp = E.apply_adjoint(V) * gs
                else:
                    V = (A + s.real * E - LowRankOperator(K, np.eye(len(K)), B)).apply_inverse(W)
                    Vp = E.apply(V) * gs
                W = W + Vp
                Z.append(V * np.sqrt(gs))
                K_newton = K_newton + Vp.lincomb(V.dot(B).T)
            else:
                j = j + 2
                j_shift = j_shift + 2
                gs = -4 * s.real
                d = -s.real / s.imag
                if trans:
                    V = (A + s * E - LowRankOperator(B, np.eye(len(B)), K)).apply_inverse_adjoint(W)
                    W = W + E.apply_adjoint(V.real + V.imag * d) * gs
                else:
                    V = (A + s * E - LowRankOperator(K, np.eye(len(K)), B)).apply_inverse(W)
                    W = W + E.apply(V.real + V.imag * d) * gs
                Zp = cat_arrays([V.real + V.imag * d, V.imag * np.sqrt(d**2 + 1)]) * np.sqrt(gs)
                Z.append(Zp)
                if trans:
                    K_newton = K_newton + E.apply_adjoint(Zp).lincomb(Zp.dot(B).T)
                else:
                    K_newton = K_newton + E.apply(Zp).lincomb(Zp.dot(B).T)

            if j_shift >= len(shifts):
                j_shift = 0
                if trans:
                    shifts = projection_shifts(A - LowRankOperator(B, np.eye(len(B)), K), E, V[:len(C)], shifts)
                else:
                    shifts = projection_shifts(A - LowRankOperator(K, np.eye(len(K)), B), E, V[:len(C)], shifts)

            WTW = W.gramian()
            lyap_res = np.linalg.norm(WTW, ord=2)
            if use_ricc_res:
                ricc_res = _compute_ricc_res(W, WTW, K, K_newton)
                logger.info(f'Relative Lyapunov / Riccati residual at inner step {j}:'
                            f'{lyap_res/lyap_init_res:.5e} / {ricc_res/ricc_init_res:.5e}')
                if ricc_res < Ctol:
                    break
            else:
                logger.info(f'Relative Lyapunov residual at inner step {j}: {lyap_res/lyap_init_res:.5e}')

        if not use_ricc_res:
            ricc_res = _compute_ricc_res(W, WTW, K, K_newton)
            if ricc_res < Csqrttol:
                use_ricc_res = True

        logger.info(f'Relative Riccati residual at outer step {k+1}: {ricc_res/ricc_init_res:.5e}')
        if ricc_res < Ctol:
            break

        K = K_newton

    if return_K:
        K = K_newton
        return Z, K
    else:
        return Z


def _compute_ricc_res(W, WTW, K, K_newton):
    """Compute Riccati residual."""
    K_diff = K - K_newton
    RR = np.bmat([
        [WTW, -W.dot(K_diff)],
        [K_diff.dot(W), - K_diff.gramian()]
    ])
    return np.linalg.norm(RR, ord=2)
