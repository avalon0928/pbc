#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 改进矩阵平移计算公式
#

import numpy as np


# load单位单元
arr_s = np.array([(8.8000001907, 0.0000000000, 0.0000000000),
              (0.0000000000, 8.8000001907, 0.0000000000),
              (0.0000000000, 0.0000000000, 12.6850004196)],
             dtype=float)

# 扩展倍数
nx = 2
ny = 2
nz = 2
arr_n = np.array([(nx, 0.0000000000, 0.0000000000),
              (0.0000000000, ny, 0.0000000000),
              (0.0000000000, 0.0000000000, nz)],
             dtype=float)

#  扩展后单位单元
arr_r = np.dot(arr_s, arr_n)

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
            list_a.append(arr_a[i])
        else:
            list_b.append(arr_a[i])
    for i in range(int(arr.shape[0]*0.2), arr.shape[0]):
        list_b.append(arr[i])
    return list_a, list_b, list_all


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


def add_index(arr, vn, brn, ln):
    # 生成标签列
    vn = int(arr.shape[0]*0.2)
    brn = int(arr.shape[0]*0.4)
    ln = arr.shape[0] - vn -brn
    a = np.ones(vn).reshape(vn,1)
    b = (np.ones(brn)*2).reshape(brn,1)
    c = (np.ones(ln)*3).reshape(ln,1)
    arr_one = np.vstack([a, b, c])
    # 随机打乱后,加入标签列输出
    np.random.shuffle(arr)
    arr_output = np.hstack([arr_one, arr])
    return arr_output


arr_in, arr_out = arr_ouput(arr, arr_s[0, 0], arr_s[1, 1], arr_s[2, 2], 3, 3, 3)

N = np.array([3, 3, 3])
Lattice = N*arr_s

arr_in_index = add_index(arr_in, 0.2, 0.4, 0.4)
arr_out_index = add_index(arr_out, 0.2, 0.4, 0.4)
arr_alldot = np.vstack([arr_in_index, arr_out_index])

vn = int(arr_in.shape[0]*0.2)
brn = int(arr_in.shape[0]*0.4)
ln = arr_in.shape[0] - vn -brn

# 以所有位点为圆心，默认半径5，自过滤
list_a, list_b, list_all = arr_line(arr_in_index, arr_alldot, 5)

# 输出所有位点
np.savetxt("POSCAR2", arr_in,
           fmt="%.9f",
           header='POSCAR\n' +'1.0\n'
                  + str("%.9f" % Lattice[0,0])+'    '+ str("%.9f" % Lattice[0,1])+'    '+ str("%.9f" % Lattice[0,2]) +'\n'
                  + str("%.9f" % Lattice[1,0])+'    '+ str("%.9f" % Lattice[1,1])+'    '+ str("%.9f" % Lattice[1,2]) +'\n'
                  + str("%.9f" % Lattice[2,0])+'    '+ str("%.9f" % Lattice[2,1])+'    '+ str("%.9f" % Lattice[2,2]) +'\n'
                  '    I'+'\n' + '    '+ str(np.size(arr_in, 0)) + '\n' + 'Cartesian', comments='', encoding='UTF-8')
np.savetxt("POSCAR3", arr_in_index[:, 1:4],
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
    for i in range(arr_s.shape[0]):
        for j in range(3):
            f.writelines(str("%.9f" % Lattice[i, j])+'    ')
        f.writelines("\n")
    f.writelines('    '+"V" +'    '+"Br" +'    '+"I" +'    '+ "V*" + "\n")
    f.writelines('    '+str(vn)+ '    '+str(brn)+ '    '+str(ln) + '    '+ str(len(list_all))+"\n")
    for j in range(len(list_all)):
        f.writelines("V*" + "    "+str(j+1) +"\n")
        for k in range(1,4):
            f.writelines(str("%.9f" %list_all[j][0][k])+ "    ")
        f.writelines("\n")
        f.writelines("Cartesian" + "\n")
        list_index = list_all[j][1][:,0].tolist()
        vn = list_index.count(1)
        brn = list_index.count(2)
        ln = list_index.count(3)
        f.writelines('    ' + "V" + '    ' + "Br" + '    ' + "I" + "\n")
        f.writelines('    ' + str(vn) + '    ' + str(brn) + '    ' + str(ln) + "\n")
        for k in range(list_all[j][1].shape[0]):
            for l in range(1, 4):
                f.writelines(str("%.9f" %list_all[j][1][k][l]) +'    ')
            f.writelines("\n")
        f.writelines("------------------"+ "\n")
f.close()






