#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 改进矩阵平移计算公式

import numpy as np
import random


# load单位单元
arr_s = np.array([(8.8000001907, 0.0000000000, 0.0000000000),
              (0.0000000000, 8.8000001907, 0.0000000000),
              (0.0000000000, 0.0000000000, 12.6850004196)],
             dtype=float)

# 扩展倍数、原子比率、截断距离
nx = 2
ny = 2
nz = 2
V_r = 0.2
Br_r = 0.4
r = 5

#  扩展后单位单元
N = np.array([nx, ny, nz])
Lattice = N*arr_s

# load源数组
arr = np.loadtxt('POSCAR(1_Original_in)', skiprows=8, encoding='utf-8')


def array_move(arr, x, y, z):
    """
    计算位点数组arr沿xyz轴平移后的位点坐标
    :param arr: 数组rr
    :param x: 沿x轴平移量
    :param y: 沿y轴平移量
    :param z: 沿z轴平移量
    :return: 平移后的位点数组
    """
    n = arr.shape[0]
    arr = np.hstack([arr, np.ones((n, 1))])
    arr_m = np.array([(1,0,0,0),(0,1,0,0),(0,0,1,0),(x,y,z,1)])
    arr_move = np.dot(arr, arr_m)[:,:3]
    return arr_move


def arr_all(arr, x, y, z, nx=2, ny=2, nz=2):
    """
    计算位点数组按单元格（x,y,z）沿xyz轴分别扩展（nx,ny,nz）后的位点集合
    :param arr:
    :param x: 三位单元格x轴长度
    :param y: 三位单元格y轴长度
    :param z: 三位单元格z轴长度
    :param nx: 在x轴扩展nx倍
    :param ny: 在y轴扩展ny倍
    :param nz: 在z轴扩展nz倍
    :return: 扩展后的所有位点坐标
    """
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if i == 0 and j == 0 and k == 0:
                    arr_all = array_move(arr, i*x, j*y, k*z)
                else:
                    arr_r = array_move(arr, i*x, j*y, k*z)
                    arr_all = np.vstack([arr_all, arr_r])
    return arr_all


def pbc_dist(dot1, dot2, arr):
    dot_1 = np.zeros([3])
    # 计算两点间最近距离
    if dot1[0] - dot2[0] < -arr[0][0]/2:
        dot_1[0] = dot1[0] + arr[0][0]
    elif dot1[0] - dot2[0] > arr[0][0]/2:
        dot_1[0] = dot1[0] - arr[0][0]
    else:
        dot_1[0] = dot1[0]

    if dot1[1] - dot2[1] < -arr[1][1]/2:
        dot_1[1] = dot1[1] + arr[1][1]
    elif dot1[1] - dot2[1] > arr[1][1]/2:
        dot_1[1] = dot1[1] - arr[1][1]
    else:
        dot_1[1] = dot1[1]

    if dot1[2] - dot2[2] < -arr[2][2]/2:
        dot_1[2] = dot1[2] + arr[2][2]
    elif dot1[2] - dot2[2] > arr[2][2]/2:
        dot_1[2] = dot1[2] - arr[2][2]
    else:
        dot_1[2] = dot1[2]
    dist = np.linalg.norm(dot_1 - dot2)
    return dist



def round_filter(arr, rd, r, arr_s):
    """
    返回数组arr内，与位点rd距离r以内的所有位点
    :param arr: 带标签列的位点数组 n行4列  第一列标签列，第2-4列 位点坐标
    :param rd: 位点坐标，圆心
    :param r: 半径
    :return: 位点列表
    """
    list = []
    for j in range(arr.shape[0]):
        dist = pbc_dist(arr[j][2:5], rd, arr_s)
        if dist <= r and dist > 0:
            list.append(arr[j][0])
    return list


def arr_line(arr, arr_s, r=5, V_r=0.2):
    """
    :param arr: array n行4列  第一列标签列，第2-4列 位点坐标
    :param r: 半径
    :return: list_a 半径内刚好有8个位点的位点坐标v
    :return: list_b 半径内位点数不等于8的位点坐标
    :return: list_all V原子及其半径内8个位点数组的集合
    """
    list_neighbor = []
    for i in range(int(arr.shape[0]*V_r)):
        list_filter = round_filter(arr, arr[i][2:5], r, arr_s)
        # print(i, len(list_filter))
        if len(list_filter) == 8:
            list_neighbor.append((arr[i][0], list_filter))
    return list_neighbor


def add_category(arr, V_r=0.2, Br_r=0.4):
    # 生成标签列
    vn = int(arr.shape[0]*V_r)
    brn = int(arr.shape[0]*Br_r)
    ln = arr.shape[0] - vn -brn
    a = np.ones(vn).reshape(vn,1)
    b = (np.ones(brn)*2).reshape(brn,1)
    c = (np.ones(ln)*3).reshape(ln,1)
    arr_one = np.vstack([a, b, c])
    # 随机打乱后,加入标签列输出
    np.random.shuffle(arr)
    arr_output = np.hstack([arr_one, arr])
    return arr_output


def add_index(arr, n):
    # 添加索引
    index = np.ones(n).reshape(n, 1)
    for i in range(n):
        index[i] = i + 1
    arr_output = np.hstack([index, arr])
    return arr_output


def atomic_change(arr, i, j):
    value_i = arr[arr[:,0]==i][:,1]
    value_j = arr[arr[:,0]==j][:,1]
    arr[arr[:, 0] == j][:,1] = value_j
    arr[arr[:, 0] == i][:,1] = value_i
    # arr_cs = arr[arr[:, 1].argsort()]
    return arr


def atomic_change1(arr, a, b):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            if arr[i][0] == a and arr[j][0] == b:
                value_i = arr[i][1]
                value_j = arr[j][1]
                arr[i][1] = value_j
                arr[j][1] = value_i
    return arr


def atomic_iterative(arr_all_index, Lattice, n):
    """

    :param arr: 打乱后添加index、类别后的位点数组
    :param n: 迭代次数
    :return: arr，list
    """
    for i in range(n):
        list_change = []
        list_neighbor = arr_line(arr_all_index, Lattice, r=5)
        for j in range(len(list_neighbor)):
            a = int(list_neighbor[j][0])
            b = int(random.choice(list_neighbor[j][1]))
            arr_all_index = atomic_change1(arr_all_index, a, b)
            list_change.append([a, b])
        arr_all_index = arr_all_index[arr_all_index[:, 1].argsort()]
    return arr_all_index, list_neighbor


# 输出扩展后的初始位点
arr_all = arr_all(arr, arr_s[0, 0], arr_s[1, 1], arr_s[2, 2], nx, ny, nz)
# 添加类别列
arr_all_shuffle = add_category(arr_all)
# 添加索引列
arr_all_index = add_index(arr_all_shuffle, arr_all_shuffle.shape[0])
# 迭代
arr_all_index, list_neighbor = atomic_iterative(arr_all_index, Lattice, 3)

vn = int(arr_all.shape[0]*V_r)
brn = int(arr_all.shape[0]*Br_r)
ln = arr_all.shape[0] - vn -brn

# 输出所有位点
np.savetxt("POSCAR2", arr_all,
           fmt="%.9f",
           header='POSCAR\n' +'1.0\n'
                  + str("%.9f" % Lattice[0,0])+'    '+ str("%.9f" % Lattice[0,1])+'    '+ str("%.9f" % Lattice[0,2]) +'\n'
                  + str("%.9f" % Lattice[1,0])+'    '+ str("%.9f" % Lattice[1,1])+'    '+ str("%.9f" % Lattice[1,2]) +'\n'
                  + str("%.9f" % Lattice[2,0])+'    '+ str("%.9f" % Lattice[2,1])+'    '+ str("%.9f" % Lattice[2,2]) +'\n'
                  '    I'+'\n' + '    '+ str(np.size(arr_all, 0)) + '\n' + 'Cartesian', comments='', encoding='UTF-8')

# 输出迭代后的所有位点
np.savetxt("POSCAR3", arr_all_index[:, 2:5],
           fmt="%.9f",
           header='Perovskite\n' +'1.0\n'
                  + str("%.9f" % Lattice[0,0])+'    '+ str("%.9f" % Lattice[0,1])+'    '+ str("%.9f" % Lattice[0,2]) +'\n'
                  + str("%.9f" % Lattice[1,0])+'    '+ str("%.9f" % Lattice[1,1])+'    '+ str("%.9f" % Lattice[1,2]) +'\n'
                  + str("%.9f" % Lattice[2,0])+'    '+ str("%.9f" % Lattice[2,1])+'    '+ str("%.9f" % Lattice[2,2]) +'\n'
                  '    V    Br    I'+'\n' + '    '+str(vn)+ '    '
                  +str(brn)+ '    '+str(ln)
                  + '\n' + 'Cartesian',
           comments='',
           encoding='UTF-8')

with open('POSCAR4', 'w+') as f:
    f.writelines("Perovskite" + "\n")
    f.writelines("1.0" + "\n")
    for i in range(Lattice.shape[0]):
        for j in range(3):
            f.writelines(str("%.9f" % Lattice[i, j])+'    ')
        f.writelines("\n")
    f.writelines('    '+"V" +'    '+"Br" +'    '+"I" +'    '+ "V*" + "\n")
    f.writelines('    '+str(vn)+ '    '+str(brn)+ '    '+str(ln) + '    '+ str(len(list_neighbor))+"\n")
    for j in range(len(list_neighbor)):
        f.writelines("V*" + "    "+str(j+1) +"\n")
        for k in range(3):
            f.writelines(str("%.9f" %arr_all_index[arr_all_index[:, 0] == list_neighbor[j][0]][:,2:5].tolist()[0][k]) + "   ")
        f.writelines("\n")
        f.writelines("Cartesian" + "\n")
        f.writelines('    ' + "V" + '    ' + "Br" + '    ' + "I" + "\n")
        list_num = []
        for m in range(8):
            list_num.append(int(arr_all_index[arr_all_index[:, 0] == list_neighbor[j][1][m]][:, 1].tolist()[0]))
        f.writelines('    ' + str(list_num.count(1)) + '    ' + str(list_num.count(2)) + '    ' + str(list_num.count(3)) + "\n")
        for m in range(8):
            if int(arr_all_index[arr_all_index[:, 0] == list_neighbor[j][1][m]][:, 1].tolist()[0]) == 1:
                f.writelines("V" + "    ")
                for n in range(3):
                    f.writelines(str("%.9f" %arr_all_index[arr_all_index[:, 0] == list_neighbor[j][1][m]][:,2:5].tolist()[0][n]) + "    ")
                f.writelines("\n")
            elif int(arr_all_index[arr_all_index[:, 0] == list_neighbor[j][1][m]][:, 1].tolist()[0]) == 2:
                f.writelines("Br" + "    ")
                for n in range(3):
                    f.writelines(str("%.9f" %arr_all_index[arr_all_index[:, 0] == list_neighbor[j][1][m]][:,2:5].tolist()[0][n]) + "    ")
                f.writelines("\n")
            else:
                f.writelines("I" + "    ")
                for n in range(3):
                    f.writelines(str("%.9f" %arr_all_index[arr_all_index[:, 0] == list_neighbor[j][1][m]][:,2:5].tolist()[0][n]) + "    ")
                f.writelines("\n")
        f.writelines("------------------"+ "\n")
f.close()
