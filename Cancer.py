import math

import numpy as np
import pandas as pd

data = pd.read_excel("./data.xlsx")


# 构建数据集
def Dataset(Data, rate):
    # 第一种切行方法
    # Train = data.loc[data['diagnosis'] == 'M', :]
    # print(Train)
    # 第二种切行方法
    M = Data[Data['diagnosis'] == 'M']
    Train_M = M.sample(frac=rate, random_state=888, replace=False, axis=0)
    # 删除指定行列
    Test_M = M.drop(Train_M.index, axis=0)

    B = Data[Data['diagnosis'] == 'B']
    Train_B = B.sample(frac=rate, random_state=888, replace=False, axis=0)
    Test_B = B.drop(Train_B.index, axis=0)

    # 拼接训练集测试集
    Train = pd.concat([Train_B, Train_M])
    Test = pd.concat([Test_B, Test_M])

    return Train, Test


# 计算先验概率
def Priori(Train):
    S = len(Train)
    M = len(Train[Train['diagnosis'] == 'M'])
    B = len(Train[Train['diagnosis'] == 'B'])
    PM = M / S
    PB = B / S

    return PM, PB


# 计算均值和方差
def Mean_Var(Train):
    Mean = Train.iloc[:, 2:].mean()
    Var = Train.iloc[:, 2:].var()

    return Mean, Var


# 计算恶性高斯概率密度函数
def Gauss_M(Train, Test):
    gauss_m = []
    Train = Train[Train['diagnosis'] == 'M']
    mean, var = Mean_Var(Train)
    for i in range(len(Test)):
        for j in range(len(mean)):
            gauss_m.append((1 / math.sqrt(2 * math.pi * var[j])) * math.exp(-(Test.iloc[i, j + 2] - mean[j]) *
                                                                            (Test.iloc[i, j + 2] - mean[j]) /
                                                                            (2 * var[j])))
    return gauss_m


# 计算良性高斯概率密度函数
def Gauss_B(Train, Test):
    gauss_b = []
    Train = Train[Train['diagnosis'] == 'B']
    mean, var = Mean_Var(Train)
    for i in range(len(Test)):
        for j in range(len(mean)):
            gauss_b.append((1 / (math.sqrt(2 * math.pi * var[j]))) * math.exp(-(Test.iloc[i, j + 2] - mean[j]) *
                                                                              (Test.iloc[i, j + 2] - mean[j]) /
                                                                              (2 * var[j])))
    return gauss_b


# 基于训练集得到的均值、方差以及先验概率对测试集进行分类
def Class(Train, Test):
    p_m = []
    p_b = []
    p = []
    m = 0
    pm, pb = Priori(Train)  # 得到先验概率
    pm_c, pb_c = pm, pb  # 固定先验概率，便于后续使用
    gauss_m = Gauss_M(Train, Test)  # 计算测试集各独立特征的恶性似然概率
    gauss_b = Gauss_B(Train, Test)  # 计算测试集各独立特征的良性似然概率

    # 计算测试集各独立特征的恶性后验概率
    for i in range(0, len(gauss_m), 30):
        for j in range(i, i + 30, 1):
            pm = pm * gauss_m[j]
        p_m.append(pm)
        pm = pm_c

    # 计算测试集各独立特征的良性后验概率
    for i in range(0, len(gauss_b), 30):
        for j in range(i, i + 30, 1):
            pb = pb * gauss_b[j]
        p_b.append(pb)
        pb = pb_c

    # 根据恶性、良性后验概率分类
    for i in range(len(p_m)):
        if p_m[i] > p_b[i]:
            p.append("M")
        elif p_m[i] < p_b[i]:
            p.append("B")
        else:
            p.append("NO")

    # 获得测试集标签
    label = np.array(Test['diagnosis']).tolist()
    # 对比测试集标签与后验概率分类，相同则计数
    for i in range(len(p)):
        if label[i] == p[i]:
            m = m + 1
    # 计算测试集分类准确率
    print("测试集准确率：{} %".format(m / len(p) * 100))


if __name__ == '__main__':
    train, test = Dataset(data, 0.7)
    Class(train, test)
