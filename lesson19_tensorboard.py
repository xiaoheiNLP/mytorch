# -*- coding:utf-8 -*-
"""
@author: xinquan
@file: lesson19_tensorboard.py
@time: 2021/7/2 11:48 上午
@desc: 
"""
import os
import warnings

warnings.filterwarnings('ignore')  # 忽略一些警告,可以删除
root_path = os.path.split(os.path.realpath(__file__))[0]  # 获取该脚本的地址,有效避免Linux和Windows文件路径格式不一致等问题,可以删除
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='test_tensorboard')

for x in range(100):
    print(x)
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow(2, x)', 2 ** x, x)

    writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                             "xcosx": x * np.cos(x),
                                             "arctanx": np.arctan(x)}, x)
writer.close()