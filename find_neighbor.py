#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np


# function [NN,NL]=find_neighbor(N,L,pbc,rc,r) % slow for large systems
# % N: number of atoms in the system
# % L(1,3): L(d) is the box length in the d-th direction
# % pbc(1,3): pbc(d)=1(0) means periodic (free) in the d-th direction
# % rc: cutoff distance
# % r(N,3): r(i,d) is the position of the i-th atom in the d-th direction
# % NN(N,1): NN(i) is the number of neighbors of atom i
# % NL(N,:): NL(i,k) is the index of the k-th neighbor of atom i
# NN=zeros(N,1);
# NL=zeros(N,N-1);
# L_times_pbc=L.*pbc;
# rc_square=rc*rc;
# for n1=1:N-1
#     for n2=n1+1:N
#         r12=r(n2,:)-r(n1,:);
#         r12=r12-round(r12./L).*L_times_pbc; %minimum image convention
#         d12_square=sum(r12.*r12);
#         if d12_square<rc_square
#             NN(n1)=NN(n1)+1;NL(n1,NN(n1))=n2;
#             NN(n2)=NN(n2)+1;NL(n2,NN(n2))=n1;%not used now but useful later
#         end
#     end
# end
# NL=NL(:,1:max(NN)); %may save so=me memory
# def find_neighbor(N, L, pbc, rc, r):
#     # % N: number of atoms in the system
#     # % L(1,3): L(d) is the box length in the d-th direction
#     # % pbc(1,3): pbc(d)=1(0) means periodic (free) in the d-th direction
#     # % rc: cutoff distance
#     # % r(N,3): r(i,d) is the position of the i-th atom in the d-th direction
#     # % NN(N,1): NN(i) is the number of neighbors of atom i
#     # % NL(N,:): NL(i,k) is the index of the k-th neighbor of atom i
#     NN = np.zeros((N, 1), dtype=float)
#     NL = np.zeros((N, N-1), dtype=float)
#     L_times_pbc = L * pbc
#     rc_square = rc * rc
#     for n1 in range(0, N-1):
#         # n1 1:4
#         for n2 in range(n1+1, N):
#             # n2 2:5
#             # r shape(5:3)
#             r12 = r[n2] - r[n1]
#             r12 = r12 - np.round(r12 / L) * L_times_pbc
#             print(r12 * r12)
#             d12_square = sum(r12 * r12)
#             if d12_square < rc_square:
#                 NN[n1] = NN[n1] + 1
#                 NL[n1, NN[n1]] = n2
#                 NN[n2] = NN[n2] + 1
#                 NL[n2, NN[n2]] = n1
#     NL = NL[:,1:max(NN)]
#     return NN, NL
#
#
# N = 12
# L = 8.8000001907
# pbc = 2
# rc = 5
# # load源数组
# arr = np.loadtxt('POSCAR(1_Original_in)', skiprows=8, encoding='utf-8')
# r = arr[:, 1]
# NN, NL = find_neighbor(N, L, pbc, rc, r)
# print(NN, NL)
