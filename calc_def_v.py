#!/usr/bin/python
# -*- coding: UTF-8 -*-
# mayavi画图版本
# 改进矩阵平移计算公式
#

import numpy as np
from mayavi import mlab


# load单位单元
arr_s = np.array([(8.8000001907, 0.0000000000, 0.0000000000),
              (0.0000000000, 8.8000001907, 0.0000000000),
              (0.0000000000, 0.0000000000, 12.6850004196)],
             dtype=float)

# 扩展倍数
nx = 2
ny = 2
nz = 2
arr_n = np.array([nx, ny, nz])

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


def arr_i(arr, x, y, z, nx=2, ny=2, nz=2):
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
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            for k in range(1, nz+1):
                if i == 1 and j == 1 and k == 1:
                    arr_in = array_move(arr, i*x, j*y, k*z)
                else:
                    arr_r = array_move(arr, i*x, j*y, k*z)
                    arr_in = np.vstack([arr_in, arr_r])
    return arr_in


def round_filter(arr, rd, r):
    """
    返回数组arr内，与位点rd距离r以内的所有位点
    :param arr: 带标签列的位点数组 n行4列  第一列标签列，第2-4列 位点坐标
    :param rd: 位点坐标，圆心
    :param r: 半径
    :return: 位点列表
    """
    list = []
    for j in range(arr.shape[0]):
        dist = np.linalg.norm(arr[j][1:4] - rd)
        if dist <= r and dist > 0:
            list.append(arr[j])
    return list


def arr_line(arr_a, arr_b, r=5):
    """
    :param arr: array n行4列  第一列标签列，第2-4列 位点坐标
    :param r: 半径
    :return: list_a 半径内刚好有8个位点的位点坐标v
    :return: list_b 半径内位点数不等于8的位点坐标
    :return: list_all V原子及其半径内8个位点数组的集合
    """
    list_a = []
    list_b = []
    list_all = []

    for i in range(int(arr_a.shape[0]*0.2)):
        list_filter = round_filter(arr_b, arr_a[i][1:4], r)
        if len(list_filter) == 8:
            list_all.append((arr_a[i], np.array(list_filter)))
    return list_all


def numpy_diff(x, y):
    '''
    计算2个数组的差集，即从y中去掉x中的位点
    :param x: list
    :param y: list
    :return:
    '''
    list1 = []
    for i in range(len(y)):
        if y[i] not in x:
            list1.append(y[i])
    return list1


def arr_ouput(arr, x, y, z, nx=2, ny=2, nz=2):
    # 计算l*m*n的数组，就必须计算(l+2)*(n+2)*(m+2)
    arr_total = arr_all(arr, x, y, z, nx+2, ny+2, nz+2)
    arr_in = arr_i(arr, x, y, z, nx, ny, nz)
    arr_in_last = array_move(arr_in, -x, -y, -z)
    arr_out = np.array(numpy_diff(arr_in.tolist(), arr_total.tolist()))
    arr_out_last = array_move(arr_out, -x, -y, -z)
    return arr_in_last, arr_out_last


def add_index(arr, x=0.2, y=0.4):
    # 生成标签列
    vn = int(arr.shape[0]*x)
    brn = int(arr.shape[0]*y)
    ln = arr.shape[0] - vn -brn
    a = np.ones(vn).reshape(vn,1)
    b = (np.ones(brn)*2).reshape(brn,1)
    c = (np.ones(ln)*3).reshape(ln,1)
    arr_one = np.vstack([a, b, c])
    # 随机打乱后,加入标签列输出
    np.random.shuffle(arr)
    arr_output = np.hstack([arr_one, arr])
    return arr_output


arr_in, arr_out = arr_ouput(arr, arr_s[0, 0], arr_s[1, 1], arr_s[2, 2], nx, ny, nz)
arr_in = add_index(arr_in, 0.2, 0.4)
arr_out = add_index(arr_out, 0.2, 0.4)
arr_allslot = np.vstack([arr_in, arr_out])
vn = int(arr_in.shape[0]*0.2)
brn = int(arr_in.shape[0]*0.4)
ln = arr_in.shape[0] - vn -brn

list_all = arr_line(arr_in, arr_allslot, 5)

for i in range(len(list_all)):
    for j in range(8):
        if list_all[i][1][j][1] < 0:
            list_all[i][1][j][1] = list_all[i][1][j][1] + arr_s[0, 0]
        elif list_all[i][1][j][1] > Lattice[0, 0]:
            list_all[i][1][j][1] = list_all[i][1][j][1] - arr_s[0, 0]
for i in range(len(list_all)):
    for j in range(8):
        if list_all[i][1][j][2] < 0:
            list_all[i][1][j][2] = list_all[i][1][j][2] + arr_s[1, 1]
        elif list_all[i][1][j][2] > Lattice[1, 1]:
            list_all[i][1][j][2] = list_all[i][1][j][2] - arr_s[1, 1]
for i in range(len(list_all)):
    for j in range(8):
        if list_all[i][1][j][3] < 0:
            list_all[i][1][j][3] = list_all[i][1][j][3] + arr_s[2, 2]
        elif list_all[i][1][j][3] > Lattice[2, 2]:
            list_all[i][1][j][3] = list_all[i][1][j][3] - arr_s[2, 2]

for i in range(len(list_all)):
    for j in range(len(list_all[i][1])):
        line_plot = np.vstack([list_all[i][0], list_all[i][1][j]])
        mlab.plot3d(line_plot[:, 1], line_plot[:, 2], line_plot[:, 3])

#
#
# # V:红色； Br:蓝色； I:绿色
vn = int(arr_in.shape[0]*0.2)
brn = int(arr_in.shape[0]*0.4)
ln = arr_in.shape[0] - vn -brn
mlab.points3d(arr_in[:vn, 1], arr_in[:vn, 2], arr_in[:vn, 3], scale_factor=1, mode='sphere', color=(1, 0, 0))
mlab.points3d(arr_in[vn:brn, 1], arr_in[vn:brn, 2], arr_in[vn:brn, 3], scale_factor=1, mode='sphere', color=(0, 0, 1))
mlab.points3d(arr_in[brn:, 1], arr_in[brn:, 2], arr_in[brn:, 3], scale_factor=1, mode='sphere', color=(0, 1, 0))
mlab.points3d(Lattice[:, 0], Lattice[:, 1], Lattice[:, 2], scale_factor=1, mode='sphere', color=(0, 0, 0))
# mlab.points3d(arr_out[brn:, 1], arr_out[brn:, 2], arr_out[brn:, 3], scale_factor=1, mode='sphere', color=(1, 1, 1))
mlab.axes()
mlab.outline()
mlab.show()







