# pbc
POSCAR(1_Original_in)   原始数据    单位立方体框架与内部12点
pbc_v1  mayavi画图版本
pbc_v2  文本输出结果版本

1、计算单位立方体中所有位点 按框架分别沿XYZ扩展后的 所有位点坐标；
2、随机打乱以上所有位点，以部分位点为中心，计算与其最小镜像距离小于指定截断距离的位点，并记录（画图版为连线2点）
3、迭代：重复第2步