# -*- coding:utf-8 -*-
"""
@author: xinquan
@file: gpu_use.py
@time: 2021/8/3 10:00 下午
@desc: 
"""
import os
import warnings

warnings.filterwarnings('ignore')  # 忽略一些警告,可以删除
root_path = os.path.split(os.path.realpath(__file__))[0]  # 获取该脚本的地址,有效避免Linux和Windows文件路径格式不一致等问题,可以删除


# get_gpu_memory
def get_gpu_memory():
    import os
    os.system('nvidid-smi -q-d Memory|grep -A4 GPU|gprep Free > tnp.txt')
    memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
    os.system('rm tmp.txt')
    return memory_gpu


# use GPU
def get_information_gpu():
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_info = torch.cuda.get_device_name()
    print(device_info)


if __name__ == '__main__':
    pass
