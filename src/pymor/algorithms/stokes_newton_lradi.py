# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import scipy.linalg as spla
import scipy.sparse as sps
import numpy as np

from pymor.algorithms.genericsolvers import _parse_options
from pymor.vectorarrays.constructions import cat_arrays
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator
from pymor.vectorarrays.block import BlockVectorArray
from pymor.operators.block import BlockOperator
from pymor.operators.constructions import AdjointOperator, ZeroOperator, LowRankOperator

from pymor.algorithms.bernoulli import solve_bernoulli
from pymor.algorithms.lradi import projection_shifts_init, projection_shifts, lyap_lrcf_solver_options
from pymor.algorithms.to_matrix import to_matrix


def solve_stokes_riccati(A, E, G, B, C, Kz=None, trans=False, options=None):
    """Implicitly solve Riccati equation for structured DAE system on hidden manifold."""

    options = _parse_options(options, lyap_lrcf_solver_options(), 'lradi', None, False)
    logger = getLogger('pymor.algorithms.stokes_newton_lradi.solve_stokes_riccati')

    shift_options = options['shift_options'][options['shifts']]
    n_newton = 15
    n_adi = 250
    tol_adi = 1e-7
    tol_newton = 1e-8

    if E is None:
        E = IdentityOperator(A.source)

    if not trans:
        A = AdjointOperator(A)
        E = AdjointOperator(E)
        B, C = C, B

    if Kz is None:
        Kznp = _nse_ricc_initial_feedback(A, E, G, B)
        K = A.source.from_numpy(Kznp)

    CBK = cat_arrays([C, B, K])
    RHS = BlockVectorArray([CBK, G.source.zeros(len(CBK))])
    LHS = BlockOperator([
        [E, G],
        [AdjointOperator(G), ZeroOperator(G.source, G.source)]
    ])
    Q = LHS.apply_inverse(RHS).block(0)
    RHSo = E.apply(Q)
    Co, Bo, Ko = RHSo[:len(C)], RHSo[len(C):len(C)+len(B)], RHSo[-len(K):]

    for k in range(n_newton):
        Wo = cat_arrays([Co, Ko])
        Z = A.source.empty(reserve=len(Wo) * n_adi)
        Ko_newton = A.source.zeros(len(Ko))
        res = init_res = np.linalg.norm(Wo.gramian(), ord=2)
        Wtol = init_res * tol_adi
        # why does this work better without applying stabilizing feedback?
        shifts = projection_shifts_init(A, E, Co, shift_options)
        j = 0
        j_shift = 0

        while res > Wtol and j < n_adi:
            s = shifts[j_shift]
            LHS = BlockOperator([
                [A + s * E, G],
                [AdjointOperator(G), ZeroOperator(G.source, G.source)]
            ])
            KoBVA = BlockVectorArray([Ko, G.source.zeros(len(Ko))])
            BoBVA = BlockVectorArray([Bo, G.source.zeros(len(Bo))])
            LR = LowRankOperator(BoBVA, np.eye(len(Bo)), KoBVA)
            RHS = BlockVectorArray([Wo, G.source.zeros(len(Wo))])
            V = (LHS - LR).apply_inverse_adjoint(RHS).block(0)
            if s.imag == 0.:
                j = j + 1
                j_shift = j_shift + 1
                gs = -2 * s.real
                Vp = E.apply_adjoint(V) * gs
                Wo = Wo + Vp
                Z.append(V * np.sqrt(gs))
                Ko_newton = Ko_newton + Vp.lincomb(V.dot(Bo).T)
            else:
                j = j + 2
                j_shift = j_shift + 2
                gs = -4 * s.real
                d = -s.real / s.imag
                Wo = Wo + E.apply_adjoint(V.real + V.imag * d) * gs
                Zp = cat_arrays([V.real + V.imag * d, V.imag * np.sqrt(d**2 + 1)]) * np.sqrt(gs)
                Z.append(Zp)
                Ko_newton = Ko_newton + E.apply_adjoint(Zp).lincomb(Zp.dot(Bo).T)

            if j_shift >= len(shifts):
                j_shift = 0
                shifts = projection_shifts(A - LowRankOperator(Bo, np.eye(len(Bo)), Ko), E, V[:len(Co)], shifts)

            res = np.linalg.norm(Wo.gramian(), ord=2)
            logger.info(f'Relative ADI residual at inner step {j}: {res/init_res:.5e}')

        rel_change_newton = spla.norm((Ko - Ko_newton).norm() / Ko_newton.norm())
        logger.info(f'Relative change at outer step {k}: {rel_change_newton}')
        if rel_change_newton < tol_newton:
            break
        Ko = Ko_newton

    return Z


def _nse_ricc_initial_feedback(A, E, G, B, num_eig=100):
    """"Compute initial stabilizing feedback."""

    Anp = to_matrix(A)
    Enp = to_matrix(E)
    Gnp = to_matrix(G)
    Bnp = B.to_numpy().T
    Eb = sps.bmat([
        [Enp, sps.csc_matrix(Gnp.shape)],
        [sps.csc_matrix(Gnp.T.shape), None]
    ])
    Ab = sps.bmat([
        [Anp, Gnp],
        [Gnp.T, None]
    ])
    Bb = sps.bmat([
        [Bnp],
        [sps.csc_matrix((Gnp.shape[1], Bnp.shape[1]))]
    ])

    ew, rev = sps.linalg.eigs(Ab, M=Eb, k=num_eig, sigma=0)
    unst_idx = np.where(ew.real > 0.)
    unst_ews = ew[unst_idx]

    if len(unst_ews) == 0:
        return np.zeros(Bnp.shape).T

    unst_revs = rev[:, unst_idx][:, 0, :]

    # compute left eigenvectors w.r.t. unstable eigenvalues
    unst_levs = np.empty((Ab.shape[0], 0))
    for ue in unst_ews:
        _, lev = sps.linalg.eigs(Ab.T, M=Eb.T, k=1, sigma=ue)
        unst_levs = np.c_[unst_levs, lev]

    Mt = unst_levs.conj().T @ Eb @ unst_revs
    At = unst_levs.conj().T @ Ab @ unst_revs
    Bt = unst_levs.conj().T @ Bb

    Yz = solve_bernoulli(At, Mt, Bt, trans=True)
    Xz = Yz @ Yz.conj().T

    Kz = Bb.T @ unst_levs @ Xz @ unst_levs.conj().T @ Eb
    Kz = Kz.real

    return Kz[:, :Bnp.shape[0]]
