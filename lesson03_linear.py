# -*- coding:utf-8 -*-
"""
@author: xinquan
@file: lesson03_linear.py
@time: 2021/6/22 10:26 上午
@desc: 
"""
import os
import warnings
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings('ignore')  # 忽略一些警告,可以删除
root_path = os.path.split(os.path.realpath(__file__))[0]  # 获取该脚本的地址,有效避免Linux和Windows文件路径格式不一致等问题,可以删除

torch.manual_seed(10)
Ir = 0.05

# 创建训练数据源
x = torch.rand(20, 1) * 10
y = 2 * x + (5 + torch.rand(20, 1))


# 构建线性参数
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(1000):
    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算
    loss = (0.5*(y-y_pred)**2).mean()

    # 反向传播求导
    loss.backward()

    # 更新参数
    b.data.sub_(Ir * b.grad)
    w.data.sub_(Ir * w.grad)

    if iteration % 20 == 0:
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, "Loss=%.4f" % loss.data.numpy(), fontdict={"size": 20, "color": "red"})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)
        print(iteration)
        if loss.data.numpy() < 1:
            break

    plt.show()


