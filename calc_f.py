#!/usr/bin/python
# -*- coding: UTF-8 -*-
# txt输出版
import numpy as np


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
    for i in range(arr.shape[0]):
        list_filter = round_filter(arr, arr[i], r)
        if len(list_filter) == 8:
            list_all.append((arr[i], np.array(list_filter)))
            list_a.append(arr[i])
        else:
            list_b.append(arr[i])
    return list_a, list_b, list_all


arr, arr_s = array_extend(arr, arr_s, 2, 2, 2)

np.savetxt("POSCAR2", arr, fmt="%.9f",
           header='\n' +'\n' +'\n' +'\n' +'\n' +
                  '    I'+'\n' + '    '+ str(np.size(arr, 0)) + '\n' + 'Cartesian', comments='', encoding='UTF-8')

# 随机打乱
np.random.shuffle(arr)
num = arr.shape[0]
ir, brr = 0.4, 0.4
np.savetxt("POSCAR3", arr,
           fmt="%.9f",
           header='\n' +'\n' +'\n' +'\n' +'\n' +
                  '    I    Br    V'+'\n' + '    '+str(int(num*ir))+ '    '
                  +str(int(num*brr))+ '    '+str(num-int(num*ir)-int(num*brr))
                  + '\n' + 'Cartesian',
           comments='',
           encoding='UTF-8')

list_a, list_b, list_all = arr_line(arr)
arr_all = np.array(list_all)
with open('POSCAR5', 'w+') as f:
    f.writelines("Perovskite" + "\n")
    f.writelines("1.0" + "\n")
    for i in range(arr_s.shape[0]):
        f.writelines(str("%.9f" % arr_s[i,1])+'    '+ str("%.9f" % arr_s[i,1])+'    '+ str("%.9f" % arr_s[i,2]) +'\n')
    f.writelines('    '+"I" +'    '+"Br" +'    '+"v" + "\n")
    f.writelines('    '+str(int(num*ir))+ '    '+str(int(num*brr))+ '    '+str(num-int(num*ir)-int(num*brr))+"\n")
    for j in range(len(list_all)):
        f.writelines("V" + "    "+str(j+1) +"\n")
        for k in range(3):
            f.writelines(str(list_all[j][0][k])+ "    ")
        f.writelines("\n")
        f.writelines("Cartesian" + "\n")
        for k in range(list_all[j][1].shape[0]):
            f.writelines(str(list_all[j][1][k]) + "\n")
        f.writelines("------------------"+ "\n")
f.close()






