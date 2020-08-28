# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pickle
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.io as spio

from pymor.algorithms.bernoulli import bernoulli_stabilize_dense
from pymor.models.iosys import LTIModel
from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.constructions import LerayProjectedOperator, LowRankOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.algorithms.eigs import eigs
from pymor.algorithms.newton_lradi import solve_ricc_lrcf
from pymor.vectorarrays.constructions import cat_arrays
from pymor.algorithms.lradi import solve_lyap_lrcf

np.random.seed(0)
with open('fom_data', 'rb') as fom_file:
    fom_dict = pickle.load(fom_file)

fom = fom_dict['fom']
'''
A = np.random.rand(10, 10)
E = np.random.rand(10, 10)
B = np.random.rand(10, 2)
C = np.random.rand(8, 10)
G = np.random.rand(10, 5)

Aop = NumpyMatrixOperator(A)
Eop = NumpyMatrixOperator(E)
Bop = NumpyMatrixOperator(B)
Cop = NumpyMatrixOperator(C)
Gop = NumpyMatrixOperator(G)

Pi = np.eye(A.shape[0]) - G @ spla.inv(G.T @ spla.inv(E) @ G) @ G.T @ spla.inv(E)
PiT = np.eye(A.shape[0]) - spla.inv(E).T @ G @ spla.inv(G.T @ spla.inv(E) @ G).T @ G.T

Aproj = LerayProjectedOperator(Aop, Gop, Eop)
Bproj = LerayProjectedOperator(Bop, Gop, Eop, projection_space='range')
Cproj = LerayProjectedOperator(Cop, Gop, Eop, projection_space='source')
Eproj = LerayProjectedOperator(Eop, Gop, Eop)

from pymor.models.iosys import LTIModel
m = LTIModel.from_matrices(A, B, C, None, E)
print('L2-NORM', m.l2_norm())

U = Bproj.source.random(2)
V = Aproj.range.random(3)
W = Aproj.range.random(3)
Anp = to_matrix(Aproj)
print('cat(-V, W) - [-V, W]', spla.norm(cat_arrays([-V, W]).to_numpy().T - np.concatenate((-V.to_numpy().T, W.to_numpy().T), axis=1)))
print('')
BprojU = Bproj.apply(U)
print('B U = Pi B U', spla.norm(BprojU.to_numpy().T - Pi @ Bproj.as_range_array().to_numpy().T @ U.to_numpy().T))
print('to_matrix error', spla.norm(Anp - Pi @ A @ PiT))
AprojV = Aproj.apply_inverse(V)
print('apply_inverse', spla.norm(Pi @ A @ PiT @ AprojV.to_numpy().T - V.to_numpy().T))
print('A^-1 V = Pi^T A^-1 V', spla.norm(PiT @ AprojV.to_numpy().T - AprojV.to_numpy().T))
AprojW = Aproj.apply_inverse_adjoint(W)
print('apply_inverse_adjoint', spla.norm((Pi @ A @ PiT).T @ AprojW.to_numpy().T - W.to_numpy().T))
print('A^-T W^T =  Pi^T A^-T W^T', spla.norm(PiT @ AprojW.to_numpy().T - AprojW.to_numpy().T))
print('W = Pi W', spla.norm(W.to_numpy().T - Pi @ W.to_numpy().T))
print('V = Pi V', spla.norm(V.to_numpy().T - Pi @ V.to_numpy().T))
from pymor.operators.constructions import LowRankOperator
ApVW = Aproj + LowRankOperator(V, np.eye(len(V)), W)
ApVWaiW = ApVW.apply_inverse(W)
print('LRO', spla.norm(Pi @ (Anp + V.to_numpy().T @ W.to_numpy()) @ Pi.T @ ApVWaiW.to_numpy().T - W.to_numpy().T))
print('B = Pi B', spla.norm(Bproj.as_range_array().to_numpy().T - Pi @ Bproj.as_range_array().to_numpy().T))
print('Pi C^T = C^T', spla.norm(Pi @ Cproj.as_source_array().to_numpy().T - Cproj.as_source_array().to_numpy().T))
print('to_matrix:', spla.norm(Anp - Pi @ A @ PiT))
print('apply2 after apply_inverse:', spla.norm(AprojW.to_numpy() @ Anp @ AprojV.to_numpy().T - Aproj.apply2(AprojW, AprojV)))

'''


E = 5 * np.eye(5)
A = -np.eye(5)
A[0, 0] = 1
B = np.ones((5, 1))
C = np.ones((1, 5))
fom = LTIModel.from_matrices(A, B, C, E=E)

E = 3 * np.eye(3)
A = -np.eye(3)
A[0, 0] = 1
B = np.ones((3, 1))
C = np.ones((1, 3))
rom1 = LTIModel.from_matrices(A, B, C, E=E)

rom1_err = fom - rom1
K = bernoulli_stabilize_dense(rom1_err.A, rom1_err.E, rom1_err.C.as_source_array(), trans=True)
K = rom1_err.A.source.from_numpy(K.T)
KC = LowRankOperator(K, np.eye(len(K)), rom1_err.C.as_source_array())
rom1_err_stab = LTIModel(rom1_err.A - KC, rom1_err.B, rom1_err.C, None, rom1_err.E)

E = 4 * np.eye(4)
A = -np.eye(4)
A[0, 0] = 3
B = np.ones((4, 1))
C = np.ones((1, 4))
rom2 = LTIModel.from_matrices(A, B, C, E=E)

rom2_err = fom - rom2
K = bernoulli_stabilize_dense(rom2_err.A, rom2_err.E, rom2_err.B.as_range_array(), trans=False)
K = rom2_err.A.source.from_numpy(K.T)
BK = LowRankOperator(rom2_err.B.as_range_array(), np.eye(len(K)), K)
rom2_err_stab = LTIModel(rom2_err.A - BK, rom2_err.B, rom2_err.C, None, rom2_err.E)


print('rom2_err_stab.h2_norm()', rom2_err_stab.h2_norm())
print('rom1_err_stab.h2_norm()', rom1_err_stab.h2_norm())
print('rom1_err.l2_norm()', rom1_err.l2_norm())
print('rom2_err.l2_norm()', rom2_err.l2_norm())


'''
Aproj = LerayProjectedOperator(Aop, Gop, Eop)
Bproj = LerayProjectedOperator(Bop, Gop, Eop, projection_space='range')
Cproj = LerayProjectedOperator(Cop, Gop, Eop, projection_space='source')
Eproj = LerayProjectedOperator(Eop, Gop, Eop)

from pymor.models.iosys import StokesDescriptorModel
from pymor.reductors.bt import StabilizingBTReductor, LQGBTReductor
from pymor.reductors.h2 import GapIRKAReductor, IRKAReductor
from pymor.algorithms.newton_lradi import ricc_lrcf_solver_options

# fom = StokesDescriptorModel(Aop, Gop, Bop, Cop, None, Eop)
# reductor = StabilizingBTReductor(fom)
# rom = reductor.reduce(10, projection='biorth')
# print('ERROR:', (fom - rom).l2_norm())

fom = StokesDescriptorModel(Aop, Gop, Bop, Cop, None, Eop)

_, Kc = solve_ricc_lrcf(fom.A, fom.E, fom.B.as_range_array(), fom.C.as_source_array(), trans=True, return_K=True)
_, Ko = solve_ricc_lrcf(fom.A, fom.E, fom.B.as_range_array(), fom.C.as_source_array(), trans=False, return_K=True)

Acl = fom.E - 1e-6 * (fom.A - LowRankOperator(fom.B.as_range_array(), np.eye(len(Kc)), Kc) \
                            - LowRankOperator(Ko, np.eye(len(Ko)), fom.C.as_source_array()))

print('Test apply inverse')
V = fom.E.source.zeros(3)
Acl.apply_inverse(V)
Acl.apply_inverse(V)



ew, ev = eigs(fom.A, fom.E, k=25, sigma=0)
print(np.sort(ew))

Z = solve_ricc_lrcf(fom.A, fom.E, Bra, Cva, trans=False)
Znp = Z.to_numpy().T
X = Znp @ Znp.T

K = fom.E.apply(Z).lincomb(Z.dot(Cva).T)

KC = LowRankOperator(K, np.eye(len(K)), Cva)
mKB = cat_arrays([-K, Bra]).to_numpy().T
mKBop = NumpyMatrixOperator(mKB)

mKBop_proj = LerayProjectedOperator(mKBop, fom.A.source.G, fom.A.source.E, projection_space='range')

cl_fom = LTIModel(fom.A - KC, mKBop_proj, fom.C, None, fom.E)

reductor = LQGBTReductor(fom, solver_options=ricc_lrcf_solver_options()['lrnadi'])

rom = reductor.reduce(40, projection='biorth')
rom2 = reductor.reduce(35, projection='biorth')

A = to_matrix(rom.A, format='dense')
B = to_matrix(rom.B, format='dense')
C = to_matrix(rom.C, format='dense')
from pymor.operators.constructions import IdentityOperator

if isinstance(rom.E, IdentityOperator):
    P = spla.solve_continuous_are(A.T, C.T, B.dot(B.T), np.eye(len(C)), balanced=False)
    F = P @ C.T
else:
    E = to_matrix(rom.E, format='dense')
    P = spla.solve_continuous_are(A.T, C.T, B.dot(B.T), np.eye(len(C)), e=E.T, balanced=False)
    F = E @ P @ C.T

AF1 = A - F @ C
mFB1 = np.concatenate((-F, B), axis=1)
C1 = C
D1 = LTIModel.from_matrices(AF1, -F, C, np.eye(len(C)))
D1inv = LTIModel.from_matrices(A, F, C, np.eye(len(C)))
N1 = LTIModel.from_matrices(AF1, B, C)
gap_rom = LTIModel.from_matrices(AF1, mFB1, C, E=None if isinstance(rom.E, IdentityOperator) else E)

A = to_matrix(rom2.A, format='dense')
B = to_matrix(rom2.B, format='dense')
C = to_matrix(rom2.C, format='dense')

if isinstance(rom2.E, IdentityOperator):
    P = spla.solve_continuous_are(A.T, C.T, B.dot(B.T), np.eye(len(C)), balanced=False)
    F = P @ C.T
else:
    E = to_matrix(rom2.E, format='dense')
    P = spla.solve_continuous_are(A.T, C.T, B.dot(B.T), np.eye(len(C)), e=E.T, balanced=False)
    F = E @ P @ C.T

AF2 = A - F @ C
mFB2 = np.concatenate((-F, B), axis=1)
C2 = C
D2 = LTIModel.from_matrices(AF2, -F, C, np.eye(len(C)))
D2inv = LTIModel.from_matrices(A, F, C, np.eye(len(C)))
N2 = LTIModel.from_matrices(AF2, B, C)
gap_rom2 = LTIModel.from_matrices(AF2, mFB2, C, E=None if isinstance(rom.E, IdentityOperator) else E)

gap_err = cl_fom - gap_rom
print((gap_rom - gap_rom2).h2_norm())
print((rom - rom2).l2_norm())
print((D2inv * ((D2 - D1) * rom + (N1 - N2))).l2_norm())
print((D2inv * (D2 - D1) * rom).l2_norm())
print((D2inv.linf_norm() * (D2 - D1).l2_norm() * rom.linf_norm()))
print((D2 - D1).l2_norm(), (D2 - D1).linf_norm(), rom.linf_norm(), D2inv.linf_norm())

Znp = solve_lyap_lrcf(gap_err.A, gap_err.E, gap_err.C.as_source_array(),  trans=True).to_numpy().T[:100]
print(Znp.shape)

Anp = to_matrix(gap_err.A, format='dense')[:100, :100]
Enp = to_matrix(gap_err.E, format='dense')[:100, :100]
Cnp = gap_err.C.as_source_array().to_numpy().T[:100]

norm_pi_part = spla.norm(Pi @ Anp.T @ Znp @ Znp.T @ Enp @ PiT + Pi @ Enp.T @ Znp @ Znp.T @ Anp @ PiT + Pi @ Cnp @ Cnp.T @ PiT)

Znp = solve_lyap_lrcf(gap_err.A, gap_err.E, gap_err.C.as_source_array(),  trans=True).to_numpy().T[:-10]
print(Znp.shape)

Anp = to_matrix(gap_err.A, format='dense')[:-10, :-10]
Enp = to_matrix(gap_err.E, format='dense')[:-10, :-10]
Cnp = gap_err.C.as_source_array().to_numpy().T[:-10]

other_part = spla.norm(Anp.T @ Znp @ Znp.T @ Enp + Enp.T @ Znp @ Znp.T @ Anp + Cnp @ Cnp.T)

print(norm_pi_part, other_part)


errs = []
for r in [10]:
    rom = reductor.reduce(r, tol=1e-9, maxit=20, projection='Eorth', compute_errors=True, closed_loop_fom=cl_fom)
    A = to_matrix(rom.A, format='dense')
    B = to_matrix(rom.B, format='dense')
    C = to_matrix(rom.C, format='dense')
    from pymor.operators.constructions import IdentityOperator

    if isinstance(rom.E, IdentityOperator):
        P = spla.solve_continuous_are(A.T, C.T, B.dot(B.T), np.eye(len(C)), balanced=False)
        F = P @ C.T
    else:
        E = to_matrix(rom.E, format='dense')
        P = spla.solve_continuous_are(A.T, C.T, B.dot(B.T), np.eye(len(C)), e=E.T, balanced=False)
        F = E @ P @ C.T

    AF = A - F @ C
    mFB = np.concatenate((-F, B), axis=1)
    gap_rom = LTIModel.from_matrices(AF, mFB, C, E=None if isinstance(rom.E, IdentityOperator) else E)

    print('GAP-ERROR:', (cl_fom - gap_rom).h2_norm())
    errs.append((cl_fom - gap_rom).h2_norm())

    for s in range(len(reductor.sigma_list)):
        print(reductor.errors[s])
        # print(reductor.sigma_list[s])
        # print('')

for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    if i == 4:
        continue
    print('EXAMPLE', i)
    mat_file = spio.loadmat('stokes_mat/stokes_' + str(i) + '.mat')
    A = mat_file['A']
    G = mat_file['G']
    B = mat_file['B'].todense()
    C = mat_file['C'].todense()

    print('G Rank, shape', np.linalg.matrix_rank(G.todense()), G.shape)
    print('B Rank, shape', np.linalg.matrix_rank(B), B.shape)
    print('C Rank, shape', np.linalg.matrix_rank(C), C.shape)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)
    Gop = NumpyMatrixOperator(G)

    fom = StokesDescriptorModel(Aop, Gop, Bop, Cop, None, None)
    ew, ev = eigs(fom.A, fom.E, k=15, sigma=0)
    print(ew)

    Bra = fom.B.as_range_array()
    Cva = fom.C.as_source_array()

    Z = solve_ricc_lrcf(fom.A, fom.E, Bra, Cva, trans=False)
    K = fom.E.apply(Z).lincomb(Z.dot(Cva).T)

    KC = LowRankOperator(K, np.eye(len(K)), Cva)
    mKB = cat_arrays([-K, Bra]).to_numpy().T
    mKBop = NumpyMatrixOperator(mKB)

    mKBop_proj = LerayProjectedOperator(mKBop, fom.A.source.G, fom.A.source.E, projection_space='range')

    cl_fom = LTIModel(fom.A - KC, mKBop_proj, fom.C, None, fom.E)

    reductor = GapIRKAReductor(fom)

    rom = reductor.reduce(10, tol=1e-9, maxit=20, projection='Eorth', compute_errors=True, closed_loop_fom=cl_fom)

    for e in reductor.errors:
        print(e)


    input("Press Enter to continue...")
'''
# fom = StokesDescriptorModel(Aop, Gop, Bop, Cop, None, Eop)
# reductor = GapIRKAReductor(fom)
# rom = reductor.reduce(10, projection='Eorth')
# print('ERROR:', (fom - rom).l2_norm())

# ew, _ = spla.eig(to_matrix(Aproj), to_matrix(Eproj))
# print('to_matrix ew', np.sort(ew))

# ew, _ = spla.eig(Pi @ A @ PiT, Pi @ E @ PiT)
# print('Pi projected ew', np.sort(ew))

# from pymor.algorithms.newton_lradi import solve_ricc_lrcf
# Zva = solve_ricc_lrcf(Aproj, Eproj, Bproj.as_range_array(), Cproj.as_source_array(), trans=True)
# Z = Zva.to_numpy().T
# X = Z @ Z.T
# print(spla.norm(Pi @ A.T @ PiT @ X @ Pi @ E @ PiT + Pi @ E.T @ PiT @ X @ Pi @ A @ PiT
#                -(Pi @ E.T @ PiT) @ X @ (Pi @ B) @ (B.T @ PiT) @ X @ (Pi @ E @ PiT) + Pi @ C.T @ C @ PiT) / spla.norm(X))

# Zva = solve_ricc_lrcf(Aproj, Eproj, Bproj.as_range_array(), Cproj.as_source_array(), trans=False)
# Z = Zva.to_numpy().T
# X = Z @ Z.T
# print(spla.norm(Pi @ A @ PiT @ X @ Pi @ E.T @ PiT + Pi @ E @ PiT @ X @ Pi @ A.T @ PiT
#                 -(Pi @ E @ PiT) @ X @ (Pi @ C.T) @ (C @ PiT) @ X @ (Pi @ E.T @ PiT) + Pi @ B @ B.T @ PiT) / spla.norm(X))
