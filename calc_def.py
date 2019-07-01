#!/usr/bin/python
# -*- coding: UTF-8 -*-
# mayavi画图版本

import numpy as np
from mayavi import mlab


# load单位单元
arr_s = np.array([(8.8000001907, 0.0000000000,0.0000000000),
              (0.0000000000, 8.8000001907,0.0000000000),
              (0.0000000000,0.0000000000,12.6850004196)],
             dtype=float)

# load源数组
arr = np.loadtxt('POSCAR(1_Original_in)', skiprows=8, encoding='utf-8')


def array_extend(arr, arr_s, x=2, y=2, z=2):
    """
    :param arr: arr_s内的位点数组
    :param arr_s: 标准三维单元
    :param x: 沿x扩展倍数
    :param y: 沿y扩展倍数
    :param z: 沿z扩展倍数
    :return: 将arr以单元arr_s沿xyz分别扩展，返回扩展后的位点数组
    """
    xyz = np.array([x, y, z])
    arr_s1 = arr_s * xyz

    for i in range(2, x + 1):
        b = (i - 1) * np.array([(1, 0, 0)])
        c = arr + b * arr_s[0]
        if i == 2:
            xd = np.vstack([arr, c])
        else:
            xd = np.vstack([xd, c])

    for j in range(2, y + 1):
        b = (j - 1) * np.array([(0, 1, 0)])
        c = xd + b * arr_s[1]
        if j == 2:
            yd = np.vstack([xd, c])
        else:
            yd = np.vstack([yd, c])

    for k in range(2, z + 1):
        b = (k - 1) * np.array([(0, 0, 1)])
        c = yd + b * arr_s[2]
        if k == 2:
            zd = np.vstack([yd, c])
        else:
            zd = np.vstack([zd, c])
    return zd, arr_s1


def round_filter(arr, rd, r):
    """
    :param arr: array
    :param rd: 位点坐标，圆心
    :param r: 半径
    :return: 返回数组arr内，与位点rd距离r以内的所有位点
    """
    list = []
    for j in range(np.size(arr, 0)):
        dist = np.linalg.norm(arr[j] - rd)
        if dist <= r and dist > 0:
            list.append(arr[j])
    return list


def arr_line(arr, r=5):
    """
    :param arr: array
    :param r: 半径
    :return: list_a
    :return: list_b
    :return: list_all
    """
    list_a = []
    list_b = []
    list_all = []
    print(arr.shape[0])
    for i in range(arr.shape[0]):
        list_filter = round_filter(arr, arr[i], r)
        if len(list_filter) == 8:
            list_all.append((arr[i], np.array(list_filter)))
            list_a.append(arr[i])
        else:
            list_b.append(arr[i])
    return list_a, list_b, list_all


arr, arr_s1 = array_extend(arr, arr_s, 2, 2, 2)
list_a, list_b, list_all = arr_line(arr)

for i in range(len(list_all)):
    for j in range(len(list_all[i][1])):
        line_plot = np.vstack([list_all[i][0], list_all[i][1][j]])
        mlab.plot3d(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2])
arr1 = np.array(list_a, dtype=float)
arr2 = np.array(list_b, dtype=float)
mlab.points3d(arr1[:, 0], arr1[:, 1], arr1[:, 2], scale_factor=1, mode='sphere', color=(1, 0, 0))
mlab.points3d(arr2[:, 0], arr2[:, 1], arr2[:, 2], scale_factor=1, mode='sphere', color=(0, 0.5, 0.5))
mlab.axes()
mlab.show()







