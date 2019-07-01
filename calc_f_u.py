#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 改进矩阵平移计算公式

import numpy as np


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
    arr_m = np.array([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (x, y, z, 1)])
    arr_move = np.dot(arr, arr_m)[:, :3]
    return arr_move


def arr_all(arr, x, y, z, nx=2, ny=2, nz=2):
    """
    计算位点数组按单元格（x,y,z）沿xyz轴分别扩展（nx,ny,nz）后的位点集合
    :param arr:
    :param x: 三位单元格x轴长度
    :param y: 三位单元格y轴长度
    :param z: 三位单元格z轴长度
    :param nx: 在x轴扩展nx倍,默认2
    :param ny: 在y轴扩展ny倍,默认2
    :param nz: 在z轴扩展nz倍,默认2
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
    :param arr: array
    :param rd: 位点坐标，圆心
    :param r: 半径
    :return: 位点列表
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
    :return: list_a 半径内刚好有8个位点的位点坐标v
    :return: list_b 半径内位点数不等于8的位点坐标
    :return: list_all V原子及其半径内8个位点数组的集合
    """
    list_a = []
    list_b = []
    list_all = []
    for i in range(arr.shape[0]):
        list_filter = round_filter(arr, arr[i], r)
        if len(list_filter) == 8:
            list_all.append((arr[i], np.array(list_filter)))
            list_a.append(arr[i])
        else:
            list_b.append(arr[i])
    return list_a, list_b, list_all


# 赋值扩展后的所有位点
arr_all = arr_all(arr, arr_s[0,0], arr_s[1,1],arr_s[2,2],3,3,3)

# 输出所有位点
np.savetxt("POSCAR2", arr_all,
           fmt="%.9f",
           header='POSCAR\n' +'1.0\n'
                  + str("%.9f" % arr_s[0,0])+'    '+ str("%.9f" % arr_s[0,1])+'    '+ str("%.9f" % arr_s[0,2]) +'\n'
                  + str("%.9f" % arr_s[1,0])+'    '+ str("%.9f" % arr_s[1,1])+'    '+ str("%.9f" % arr_s[1,2]) +'\n'
                  + str("%.9f" % arr_s[2,0])+'    '+ str("%.9f" % arr_s[2,1])+'    '+ str("%.9f" % arr_s[2,2]) +'\n'
                  '    I'+'\n' + '    '+ str(np.size(arr_all, 0)) + '\n' + 'Cartesian', comments='', encoding='UTF-8')

# 随机打乱后输出
np.random.shuffle(arr_all)
num = arr_all.shape[0]
ir, brr = 0.4, 0.4
np.savetxt("POSCAR3", arr_all,
           fmt="%.9f",
           header='Perovskite\n' +'1.0\n'
                  + str("%.9f" % arr_s[0,0])+'    '+ str("%.9f" % arr_s[0,1])+'    '+ str("%.9f" % arr_s[0,2]) +'\n'
                  + str("%.9f" % arr_s[1,0])+'    '+ str("%.9f" % arr_s[1,1])+'    '+ str("%.9f" % arr_s[1,2]) +'\n'
                  + str("%.9f" % arr_s[2,0])+'    '+ str("%.9f" % arr_s[2,1])+'    '+ str("%.9f" % arr_s[2,2]) +'\n'
                  '    I    Br    V'+'\n' + '    '+str(int(num*ir))+ '    '
                  +str(int(num*brr))+ '    '+str(num-int(num*ir)-int(num*brr))
                  + '\n' + 'Cartesian',
           comments='',
           encoding='UTF-8')

# 以所有位点为圆心，默认半径5，自过滤
list_a, list_b, list_all = arr_line(arr_all, 5)

with open('POSCAR4', 'w+') as f:
    f.writelines("Perovskite" + "\n")
    f.writelines("1.0" + "\n")
    for i in range(arr_s.shape[0]):
        for j in range(3):
            f.writelines(str("%.9f" % arr_s[i, j])+'    ')
        f.writelines("\n")
    f.writelines('    '+"I" +'    '+"Br" +'    '+"v" + "\n")
    f.writelines('    '+"TBD"+ '    '+"TBD"+ '    '+str(len(list_all))+"\n")
    for j in range(len(list_all)):
        f.writelines("V" + "    "+str(j+1) +"\n")
        for k in range(3):
            f.writelines(str("%.9f" %list_all[j][0][k])+ "    ")
        f.writelines("\n")
        f.writelines("Cartesian" + "\n")
        for k in range(list_all[j][1].shape[0]):
            for l in range(3):
                f.writelines(str("%.9f" %list_all[j][1][k][l]) +'    ')
            f.writelines("\n")
        f.writelines("------------------"+ "\n")
f.close()






