# -*- coding:utf-8 -*-
"""
@author: xinquan
@file: lesson-gpu-use.py
@time: 2021/7/26 23:17
@desc: 
"""
import os

import warnings

warnings.filterwarnings('ignore')  # 忽略一些警告,可以删除
root_path = os.path.split(os.path.realpath(__file__))[0]  # 获取该脚本的地址,有效避免Linux和Windows文件路径格式不一致等问题,可以删除


import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 获取显卡数量
print(torch.cuda.device_count())

# 获取显卡名称
print(torch.cuda.get_device_name())


def get_gpu_memory():
    import platform
    if 'Windows' != platform.system():
        import os
        os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
        memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
        os.system('rm tmp.txt')
    else:
        memory_gpu = False
        print("显存计算功能暂不支持windows操作系统")
    return memory_gpu


class FooNet(nn.Module):
    def __init__(self, neural_num, layers=3):
        super(FooNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])

    def forward(self, x):

        print("\nbatch size in forward: {}".format(x.size()[0]))

        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)
        return x


if __name__ == "__main__":

    batch_size = 16

    # data
    inputs = torch.randn(batch_size, 3)
    labels = torch.randn(batch_size, 3)

    inputs, labels = inputs.to(device), labels.to(device)

    # model
    net = FooNet(neural_num=3, layers=3)
    net = nn.DataParallel(net)
    net.to(device)

    # training
    for epoch in range(1):

        outputs = net(inputs)

        print("model outputs.size: {}".format(outputs.size()))

    print("CUDA_VISIBLE_DEVICES :{}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("device_count :{}".format(torch.cuda.device_count()))


if __name__ == '__main__':
    get_gpu_memory()
