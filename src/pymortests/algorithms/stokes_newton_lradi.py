# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.algorithms.stokes_newton_lradi import solve_stokes_riccati
from pymor.algorithms.to_matrix import to_matrix
from pymor.vectorarrays.constructions import cat_arrays

from pymor.operators.block import BlockOperator
from pymor.vectorarrays.block import BlockVectorArray
from pymor.operators.constructions import ZeroOperator, AdjointOperator


def test_stokes_newton_lradi():
    A = NumpyMatrixOperator.from_file('./lin_nse/A_9356.mtx')
    C = NumpyMatrixOperator.from_file('./lin_nse/C_9356.mtx')
    C = A.source.from_numpy(to_matrix(C))
    B = NumpyMatrixOperator.from_file('./lin_nse/B_9356.mtx')
    B = A.source.from_numpy(to_matrix(B).T)
    G = NumpyMatrixOperator.from_file('./lin_nse/G_9356.mtx')
    M = NumpyMatrixOperator.from_file('./lin_nse/M_9356.mtx')

    G = AdjointOperator(G)

    CB = cat_arrays([C, B])
    RHS = BlockVectorArray([CB, G.source.zeros(len(CB))])
    LHS = BlockOperator([
        [M, G],
        [AdjointOperator(G), ZeroOperator(G.source, G.source)]
    ])
    Q = LHS.apply_inverse_adjoint(RHS).block(0)
    RHSo = M.apply(Q)
    Co, Bo = RHSo[:len(C)], RHSo[-len(B):]

    Conp = Co.to_numpy()
    Bonp = Bo.to_numpy().T

    Anp = to_matrix(A)
    Mnp = to_matrix(M)
    Gnp = to_matrix(G)

    MnpINV = spsla.spsolve(Mnp, sps.eye(Mnp.shape[0])).todense()
    INV = spsla.spsolve(Gnp.T @ MnpINV @ Gnp, Gnp.T).todense()
    GNPINV = Gnp @ INV

    def apply_projection(MAT):
        return MAT - (GNPINV @ MnpINV) @ MAT

    def apply_projection_T(MAT):
        return MAT - MAT @ (MnpINV @ GNPINV)

    Anp = apply_projection(apply_projection_T(Anp))
    Mnp = apply_projection(apply_projection_T(Mnp))

    for tra in [True, False]:
        Z = solve_stokes_riccati(A, M, G, B, C, trans=tra)
        Znp = Z.to_numpy().T
        if tra:
            AnpTZnp = Anp.T @ Znp
            MnpTZnp = Mnp.T @ Znp
            MnpTZnpZnpTBonp = MnpTZnp @ Znp.T @ Bonp
            AnpTZnpMnpTZnpT = AnpTZnp @ MnpTZnp.T
            MnpTZnpZnpTBonpMnpTZnpZnpTBonpT = MnpTZnpZnpTBonp @ MnpTZnpZnpTBonp.T
            ConpTConp = Conp.T @ Conp
            res = spla.norm(AnpTZnpMnpTZnpT + AnpTZnpMnpTZnpT.T - MnpTZnpZnpTBonpMnpTZnpZnpTBonpT + ConpTConp)
        else:
            AnpZnp = Anp @ Znp
            MnpZnp = Mnp @ Znp
            AnpZnpMnpTZnpT = AnpZnp @ MnpZnp.T
            MnpZnpTConpT = MnpZnp @ Znp.T @ Conp.T
            MnpZnpTConpTMnpZnpTConpTT = MnpZnpTConpT @ MnpZnpTConpT.T
            BonpBonpT = Bonp @ Bonp.T
            res = spla.norm(AnpZnpMnpTZnpT + AnpZnpMnpTZnpT.T - MnpZnpTConpTMnpZnpTConpTT + BonpBonpT)

        assert res < 1e-6
