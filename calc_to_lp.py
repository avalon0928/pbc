#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 给李培的版本

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_circles
from mpl_toolkits import mplot3d


#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_circles
from mpl_toolkits import mplot3d


# 扩大倍数 必须>1
x = 5
y = 5
z = 5
xyz = np.array([x, y, z])

# 原子分配比例
ir = 0.4
brr = 0.4
vr = 0.2

# 定义圆
rd = np.array([10, 10, 10])
r = 10

# load单位单元
s = np.array([(8.8000001907, 0.0000000000,0.0000000000),
              (0.0000000000, 8.8000001907,0.0000000000),
              (0.0000000000,0.0000000000,12.6850004196)],
             dtype=float)

# load源数组
a = np.loadtxt('POSCAR(1_Original_in)', skiprows=8, encoding='utf-8')

for i in range(2, x+1):
    b = (i-1) * np.array([(1, 0, 0)])
    c = a + b * s[0]
    if i == 2:
        xd = np.vstack([a, c])
    else:
        xd = np.vstack([xd,c])

for j in range(2, y+1):
    b = (j-1) * np.array([(0, 1, 0)])
    c = xd + b * s[1]
    if j == 2:
        yd = np.vstack([xd, c])
    else:
        yd = np.vstack([yd,c])

for k in range(2, z+1):
    b = (k-1) * np.array([(0, 0, 1)])
    c = yd + b * s[2]
    if k == 2:
        zd = np.vstack([yd, c])
    else:
        zd = np.vstack([zd,c])

# 输出未随机打乱的位点数组
np.savetxt("POSCAR2", zd, fmt="%.9f",
           header='\n' +'\n' +'\n' +'\n' +'\n' +
                  '    I'+'\n' + '    '+ str(np.size(zd, 0)) + '\n' + 'Cartesian', comments='', encoding='UTF-8')

# 随机打乱
np.random.shuffle(zd)

# 位点总数
num = np.size(zd, 0)
xyz_1 = s * xyz

# 在位点数组zd上添加原子标签列
# 第一列为原子标签，1:原子I，2:原子Br，3:原子V
# 目的：避免后续按条件过滤后，原子混乱
list_atom = []
for i in range(int(num*ir)):
    list_atom.append(1)
for i in range(int(num*ir),int(num * (brr + ir))):
    list_atom.append(2)
for i in range(int(num * (brr + ir)), num):
    list_atom.append(3)

atom_a = np.array(list_atom).reshape(num,1)
# 添加原子标签
zd_atom = np.hstack([atom_a, zd])

# 输出
np.savetxt("POSCAR3", zd,
           fmt="%.9f",
           header='Perovskite\n' +'1.0\n'
                  + str("%.9f" % xyz_1[0,1])+'    '+ str("%.9f" % xyz_1[0,1])+'    '+ str("%.9f" % xyz_1[0,2]) +'\n'
                  + str("%.9f" % xyz_1[1,0])+'    '+ str("%.9f" % xyz_1[1,1])+'    '+ str("%.9f" % xyz_1[1,2]) +'\n'
                  + str("%.9f" % xyz_1[2,0])+'    '+ str("%.9f" % xyz_1[2,1])+'    '+ str("%.9f" % xyz_1[2,2]) +'\n' 
                  '    I    Br    V'+'\n' + '    '+str(int(num*ir))+ '    '
                  +str(int(num*brr))+ '    '+str(num-int(num*ir)-int(num*brr))
                  + '\n' + 'Cartesian',
           comments='',
           encoding='UTF-8')

# 将符合条件的位点添加到list
list = []
for i in range(np.size(zd,0)):
    dist = np.linalg.norm(zd[i] - rd[0])
    if dist <= r:
        list.append(zd_atom[i])

# 生成含随机原子标签的过滤数组
ud = np.array(list, dtype=float)

# 统计过滤数组中各原子数量
list_b = ud[:,0].tolist()
icount = list_b.count(1)
brcount = list_b.count(2)
vcount = list_b.count(3)

# 输出过滤后的位点数组 隐藏原子标签列备用
np.savetxt("POSCAR4", ud[:, 1:4],
           fmt="%.9f",
           header='Perovskite\n' +'1.0\n'
                  + str("%.9f" % xyz_1[0,1])+'    '+ str("%.9f" % xyz_1[0,1])+'    '+ str("%.9f" % xyz_1[0,2]) +'\n'
                  + str("%.9f" % xyz_1[1,0])+'    '+ str("%.9f" % xyz_1[1,1])+'    '+ str("%.9f" % xyz_1[1,2]) +'\n'
                  + str("%.9f" % xyz_1[2,0])+'    '+ str("%.9f" % xyz_1[2,1])+'    '+ str("%.9f" % xyz_1[2,2]) +'\n'
                  '    I    Br    V'+'\n' + '    '+str(icount)+ '    '
                  +str(brcount)+ '    '+str(vcount)
                  + '\n' + 'Cartesian',
           comments='',
           encoding='UTF-8')


