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


def arr_line(arr, r=5):
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

    for i in range(int(arr.shape[0]*0.2)):
        list_filter = round_filter(arr, arr[i][1:4], r)
        if len(list_filter) == 8:
            list_all.append((arr[i], np.array(list_filter)))
            list_a.append(arr[i])
        else:
            list_b.append(arr[i])
    for i in range(int(arr.shape[0]*0.2), arr.shape[0]):
        list_b.append(arr[i])
    return list_a, list_b, list_all


arr_all = arr_all(arr, arr_s[0, 0], arr_s[1, 1], arr_s[2, 2], 2, 2, 2)

# 生成标签列
vn = int(arr_all.shape[0]*0.2)
brn = int(arr_all.shape[0]*0.4)
ln = arr_all.shape[0] - vn -brn
a = np.ones(vn).reshape(vn,1)
b = (np.ones(brn)*2).reshape(brn,1)
c = (np.ones(ln)*3).reshape(ln,1)
d = np.vstack([a, b, c])

# 随机打乱后,加入标签列输出
np.random.shuffle(arr_all)
arr_all = np.hstack([d, arr_all])

list_a, list_b, list_all = arr_line(arr_all)

for i in range(len(list_all)):
    for j in range(len(list_all[i][1])):
        line_plot = np.vstack([list_all[i][0], list_all[i][1][j]])
        mlab.plot3d(line_plot[:, 1], line_plot[:, 2], line_plot[:, 3])

arr1 = np.array(list_a, dtype=float)

# V:红色； Br:蓝色； I:绿色
mlab.points3d(arr_all[:vn, 1], arr_all[:vn, 2], arr_all[:vn, 3], scale_factor=1, mode='sphere', color=(1, 0, 0))
mlab.points3d(arr_all[vn:brn, 1], arr_all[vn:brn, 2], arr_all[vn:brn, 3], scale_factor=1, mode='sphere', color=(0, 0, 1))
mlab.points3d(arr_all[brn:, 1], arr_all[brn:, 2], arr_all[brn:, 3], scale_factor=1, mode='sphere', color=(0, 1, 0))
mlab.axes()
mlab.show()







